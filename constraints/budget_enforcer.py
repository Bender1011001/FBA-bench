from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Callable
from copy import deepcopy

from event_bus import EventBus
from events import TickEvent, BudgetWarning, BudgetExceeded
from .token_counter import TokenCounter


class BudgetEnforcer:
    """
    Compute Budget Enforcer with per-agent and per-tool metering.

    Configuration schema (dict):
      limits:
        total_tokens_per_tick: int = 200000
        total_tokens_per_run: int = 5000000
        total_cost_cents_per_tick: int = 1000
        total_cost_cents_per_run: int = 25000
      tool_limits: dict = {
        tool_name: {
          calls_per_tick: int,
          calls_per_run: int,
          tokens_per_tick: int,
          tokens_per_run: int,
          cost_cents_per_tick: int,
          cost_cents_per_run: int
        },
        ...
      }
      warning_threshold_pct: float = 0.8
      allow_soft_overage: bool = False

    Internal state (per agent):
      usage[agent_id] = {
        'tick':  { 'tokens': int, 'cost_cents': int, 'calls': int, 'per_tool': {tool: {...}} },
        'run':   { 'tokens': int, 'cost_cents': int, 'calls': int, 'per_tool': {tool: {...}} },
      }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, event_bus: Optional[EventBus] = None, metrics_tracker: Any = None) -> None:
        """
        New metering constructor (dict config), backward-compatible with legacy signature:
          BudgetEnforcer(ConstraintConfig, event_bus=None, metrics_tracker=None)
        """
        # Backward compatibility: detect legacy ConstraintConfig-like objects
        legacy_cfg = {}
        if config is not None and not isinstance(config, dict):
            # Duck-typing expected fields from legacy ConstraintConfig
            for key in [
                "max_tokens_per_action", "max_total_tokens", "token_cost_per_1k",
                "violation_penalty_weight", "grace_period_percentage",
                "hard_fail_on_violation", "inject_budget_status", "track_token_efficiency"
            ]:
                if hasattr(config, key):
                    legacy_cfg[key] = getattr(config, key)
        cfg = config if isinstance(config, dict) else {}

        limits = (cfg.get("limits", {}) or {})
        self._meter_cfg: Dict[str, Any] = {
            "limits": {
                "total_tokens_per_tick": int(limits.get("total_tokens_per_tick", 200_000)),
                "total_tokens_per_run": int(limits.get("total_tokens_per_run", 5_000_000)),
                "total_cost_cents_per_tick": int(limits.get("total_cost_cents_per_tick", 1_000)),
                "total_cost_cents_per_run": int(limits.get("total_cost_cents_per_run", 25_000)),
            },
            "tool_limits": dict(cfg.get("tool_limits", {}) or {}),
            "warning_threshold_pct": float(cfg.get("warning_threshold_pct", 0.8)),
            "allow_soft_overage": bool(cfg.get("allow_soft_overage", False)),
        }
        # Expose legacy-style config attribute expected by older modules (e.g., AgentGateway)
        # If the provided config was a dict, keep self.config as the metering dict.
        # If it was a legacy object, keep that object on self.config for attribute access.
        self.config = self._meter_cfg if isinstance(config, dict) else config

        # usage structure described above
        self.usage: Dict[str, Dict[str, Any]] = {}

        # event bus reference set in start() for new system; keep legacy ref for backward compat
        self._event_bus: Optional[EventBus] = None
        self.event_bus: Optional[EventBus] = event_bus  # legacy path uses this directly

        # legacy-compatible state
        self.token_counter: TokenCounter = TokenCounter()
        self.metrics_tracker = metrics_tracker
        self.violation_triggered: bool = False
        self.current_tick_tokens_used: int = 0
        self.total_simulation_tokens_used: int = 0

        # Store legacy config values if provided to support legacy methods
        self._legacy: Dict[str, Any] = {
            "max_tokens_per_action": int(legacy_cfg.get("max_tokens_per_action", 100)),
            "max_total_tokens": int(legacy_cfg.get("max_total_tokens", 500)),
            "token_cost_per_1k": float(legacy_cfg.get("token_cost_per_1k", 0.01)),
            "violation_penalty_weight": float(legacy_cfg.get("violation_penalty_weight", 10.0)),
            "grace_period_percentage": float(legacy_cfg.get("grace_period_percentage", 10.0)),
            "hard_fail_on_violation": bool(legacy_cfg.get("hard_fail_on_violation", True)),
            "inject_budget_status": bool(legacy_cfg.get("inject_budget_status", False)),
        }

    # ---------------------------
    # Lifecycle
    # ---------------------------
    async def start(self, event_bus: EventBus) -> None:
        """
        Bind to EventBus and subscribe to TickEvent for per-tick resets.
        """
        self._event_bus = event_bus

        async def _on_tick(event: TickEvent) -> None:
            # Reset only per-tick counters; retain run totals
            for agent_id in list(self.usage.keys()):
                self._reset_tick(agent_id)

        await event_bus.subscribe(TickEvent, _on_tick)

    # ---------------------------
    # Public API
    # ---------------------------
    async def meter_api_call(
        self,
        agent_id: str,
        tool_name: str,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        cost_cents: int = 0,
    ) -> Dict[str, Any]:
        """
        Meter a single API/tool call. Updates per-agent and per-tool counters (tick and run),
        checks tool-specific and overall constraints, and emits warnings/exceeded events.
        """
        self._ensure_agent(agent_id)
        tokens = int(tokens_prompt) + int(tokens_completion)
        cost_cents = int(cost_cents)

        # Update counters
        self._update_counters(agent_id, tool_name, tokens, cost_cents)

        # Evaluate constraints in order: per-tool tick/run, then overall tick/run
        # Return first exceeded encountered (deterministic order)
        checks: Tuple[Tuple[str, Callable[[], Optional[Dict[str, Any]]]], ...] = (
            ("tool_calls_tick", lambda: self._check_tool_limit(agent_id, tool_name, "calls_per_tick", window="tick")),
            ("tool_calls_run", lambda: self._check_tool_limit(agent_id, tool_name, "calls_per_run", window="run")),
            ("tool_tokens_tick", lambda: self._check_tool_limit(agent_id, tool_name, "tokens_per_tick", window="tick")),
            ("tool_tokens_run", lambda: self._check_tool_limit(agent_id, tool_name, "tokens_per_run", window="run")),
            ("tool_cost_tick", lambda: self._check_tool_limit(agent_id, tool_name, "cost_cents_per_tick", window="tick")),
            ("tool_cost_run", lambda: self._check_tool_limit(agent_id, tool_name, "cost_cents_per_run", window="run")),
            ("overall_tokens_tick", lambda: self._check_overall_limit(agent_id, "total_tokens_per_tick", window="tick", field="tokens")),
            ("overall_tokens_run", lambda: self._check_overall_limit(agent_id, "total_tokens_per_run", window="run", field="tokens")),
            ("overall_cost_tick", lambda: self._check_overall_limit(agent_id, "total_cost_cents_per_tick", window="tick", field="cost_cents")),
            ("overall_cost_run", lambda: self._check_overall_limit(agent_id, "total_cost_cents_per_run", window="run", field="cost_cents")),
        )

        # Process checks and return on first exceeded
        for budget_type, fn in checks:
            result = fn()
            if result and result.get("exceeded"):
                # Publish BudgetExceeded event before returning
                event_obj = result.pop("event", None)
                if event_obj is not None:
                    await self._publish(event_obj)
                return result

        # If no exceeds, possibly emit warnings for nearing thresholds (best-effort)
        await self._emit_threshold_warnings(agent_id, tool_name)

        return {"exceeded": False, "usage_snapshot": self.get_usage_snapshot(agent_id)}

    def get_usage_snapshot(self, agent_id: str) -> Dict[str, Any]:
        """
        Returns deep copy snapshot of counters for the agent.
        """
        self._ensure_agent(agent_id)
        return deepcopy(self.usage[agent_id])

    def reset_run(self, agent_id: Optional[str] = None) -> None:
        """
        Reset cumulative run counters for one agent or all agents.
        """
        if agent_id is not None:
            self._ensure_agent(agent_id)
            self._reset_run(agent_id)
        else:
            for aid in list(self.usage.keys()):
                self._reset_run(aid)

    # ---------------------------
    # Internal helpers - usage bookkeeping
    # ---------------------------
    def _ensure_agent(self, agent_id: str) -> None:
        if agent_id not in self.usage:
            self.usage[agent_id] = {
                "tick": {"tokens": 0, "cost_cents": 0, "calls": 0, "per_tool": {}},
                "run": {"tokens": 0, "cost_cents": 0, "calls": 0, "per_tool": {}},
            }

    def _ensure_tool(self, agent_id: str, tool_name: str) -> None:
        for window in ("tick", "run"):
            if tool_name not in self.usage[agent_id][window]["per_tool"]:
                self.usage[agent_id][window]["per_tool"][tool_name] = {
                    "tokens": 0,
                    "cost_cents": 0,
                    "calls": 0,
                }

    def _update_counters(self, agent_id: str, tool_name: str, tokens: int, cost_cents: int) -> None:
        self._ensure_agent(agent_id)
        self._ensure_tool(agent_id, tool_name)

        # increment calls (1 per API call)
        for window in ("tick", "run"):
            self.usage[agent_id][window]["calls"] += 1
            self.usage[agent_id][window]["tokens"] += tokens
            self.usage[agent_id][window]["cost_cents"] += cost_cents

            tool_bucket = self.usage[agent_id][window]["per_tool"][tool_name]
            tool_bucket["calls"] += 1
            tool_bucket["tokens"] += tokens
            tool_bucket["cost_cents"] += cost_cents

    def _reset_tick(self, agent_id: str) -> None:
        self._ensure_agent(agent_id)
        self.usage[agent_id]["tick"] = {"tokens": 0, "cost_cents": 0, "calls": 0, "per_tool": {}}

    def _reset_run(self, agent_id: str) -> None:
        self._ensure_agent(agent_id)
        self.usage[agent_id]["run"] = {"tokens": 0, "cost_cents": 0, "calls": 0, "per_tool": {}}

    # ---------------------------
    # Internal helpers - limits evaluation
    # ---------------------------
    def _get_tool_limit(self, tool_name: str, key: str) -> Optional[int]:
        tool_cfg: Dict[str, Any] = (self.config.get("tool_limits") or {}).get(tool_name, {}) or {}
        value = tool_cfg.get(key, None)
        if value is None:
            return None
        return int(value)

    def _build_event(self, cls, agent_id: str, budget_type: str, current_usage: int, limit: int, reason: str, **extra) -> Any:
        # events require event_id and timestamp
        base_kwargs = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow(),
            "agent_id": agent_id,
            "budget_type": budget_type,
            "current_usage": int(current_usage),
            "limit": int(limit),
            "reason": reason,
        }
        base_kwargs.update(extra)
        return cls(**base_kwargs)

    async def _publish(self, event_obj: Any) -> None:
        if self._event_bus is not None:
            await self._event_bus.publish(event_obj)

    def _threshold_crossed(self, usage: int, limit: Optional[int]) -> bool:
        if limit is None:
            return False
        if limit <= 0:
            # Any usage crosses threshold effectively
            return usage >= 1
        # Compare using integer arithmetic when possible
        # usage/limit >= warning_threshold_pct
        pct = self.config["warning_threshold_pct"]
        # Avoid floats by cross-multiplying, but pct is float; single multiplication is fine (no persistence)
        return (usage / float(limit)) >= float(pct)

    def _exceeded(self, usage: int, limit: Optional[int]) -> bool:
        if limit is None:
            return False
        return usage > int(limit)

    def _limit_tuple(self, agent_id: str, window: str, tool_name: Optional[str], metric_key: str) -> Tuple[int, Optional[int], str]:
        """
        Returns (current_usage, limit, budget_type_string)
        """
        if tool_name is not None:
            # per-tool
            self._ensure_tool(agent_id, tool_name)
            usage_val = self.usage[agent_id][window]["per_tool"][tool_name][metric_key]
            # map metric key to config key
            cfg_key = {
                "calls": f"calls_per_{window}",
                "tokens": f"tokens_per_{window}",
                "cost_cents": f"cost_cents_per_{window}",
            }[metric_key]
            limit_val = self._get_tool_limit(tool_name, cfg_key)
            budget_type = f"tool:{tool_name}.{cfg_key}"
        else:
            # overall
            usage_val = self.usage[agent_id][window][metric_key]
            # map to overall limits config
            cfg_key = {
                ("tick", "tokens"): "total_tokens_per_tick",
                ("run", "tokens"): "total_tokens_per_run",
                ("tick", "cost_cents"): "total_cost_cents_per_tick",
                ("run", "cost_cents"): "total_cost_cents_per_run",
            }[(window, metric_key)]
            limit_val = int(self.config["limits"][cfg_key])
            budget_type = cfg_key

        return int(usage_val), limit_val, budget_type

    async def _emit_threshold_warnings(self, agent_id: str, tool_name: str) -> None:
        """
        Emit BudgetWarning events when usage crosses threshold of any configured limit.
        Non-deduplicated by design; callers should avoid spamming meter calls if undesired.
        """
        # Tool-level warnings (calls, tokens, cost) for both windows
        for window in ("tick", "run"):
            for metric in ("calls", "tokens", "cost_cents"):
                usage, limit, budget_type = self._limit_tuple(agent_id, window, tool_name, metric)
                if self._threshold_crossed(usage, limit) and limit is not None:
                    reason = f"Usage reached {usage} of {limit} for {budget_type}"
                    await self._publish(self._build_event(BudgetWarning, agent_id, budget_type, usage, limit, reason))

        # Overall warnings (tokens, cost) for both windows
        for window, metric in (("tick", "tokens"), ("run", "tokens"), ("tick", "cost_cents"), ("run", "cost_cents")):
            usage, limit, budget_type = self._limit_tuple(agent_id, window, None, metric)
            if self._threshold_crossed(usage, limit):
                reason = f"Usage reached {usage} of {limit} for {budget_type}"
                await self._publish(self._build_event(BudgetWarning, agent_id, budget_type, usage, int(limit), reason))

    def _build_exceeded_response(
        self, agent_id: str, budget_type: str, usage: int, limit: int
    ) -> Dict[str, Any]:
        severity = "soft" if self.config["allow_soft_overage"] else "hard_fail"
        reason = f"Exceeded {budget_type}: usage {usage} > limit {limit}"
        event = self._build_event(
            BudgetExceeded, agent_id, budget_type, usage, limit, reason, severity=severity
        )
        # Publish asynchronously if bus is available (caller is async in meter_api_call)
        return {
            "exceeded": True,
            "agent_id": agent_id,
            "budget_type": budget_type,
            "limit": int(limit),
            "usage": int(usage),
            "severity": severity,
            "event": event,  # returned for context; actual publish performed by caller
        }

    def _check_tool_limit(self, agent_id: str, tool_name: str, cfg_key: str, window: str) -> Optional[Dict[str, Any]]:
        # Determine metric from key
        metric = "calls" if "calls" in cfg_key else ("tokens" if "tokens" in cfg_key else "cost_cents")
        usage, limit, budget_type = self._limit_tuple(agent_id, window, tool_name, metric)
        if limit is None:
            return None

        # Hard exceed check
        if self._exceeded(usage, limit):
            resp = self._build_exceeded_response(agent_id, budget_type, usage, int(limit))
            # Publish inside meter_api_call to ensure async context and ordering
            return resp
        return None

    def _check_overall_limit(self, agent_id: str, limit_key: str, window: str, field: str) -> Optional[Dict[str, Any]]:
        usage = int(self.usage[agent_id][window][field])
        limit = int(self.config["limits"][limit_key])
        if self._exceeded(usage, limit):
            return self._build_exceeded_response(agent_id, limit_key, usage, limit)
        return None

    # ---------------------------
    # Backward-compatibility methods (legacy API used by existing modules/tests)
    # ---------------------------
    def record_token_usage(self, tokens_used: int, action_type: str = "general") -> None:
        """
        Legacy: Records token usage for the current tick and overall simulation.
        Also forwards to metrics tracker if available.
        """
        tokens_used = int(tokens_used)
        self.current_tick_tokens_used += tokens_used
        self.total_simulation_tokens_used += tokens_used
        if self.metrics_tracker:
            # maintain legacy behavior
            try:
                self.metrics_tracker.record_token_usage(tokens_used, action_type)
            except Exception:
                pass

    def reset_for_new_tick(self) -> None:
        """
        Legacy: Reset per-tick counters only. Run totals remain.
        """
        self.current_tick_tokens_used = 0

    def _legacy_publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        """
        Helper to match legacy tests that expect EventBus.publish(event_name, payload) calls.
        Uses async publish if available; otherwise, ignores.
        """
        if self.event_bus is None:
            return
        # If publish is async (AsyncMock or real), schedule it. In tests it's an AsyncMock.
        pub = getattr(self.event_bus, "publish", None)
        if pub is None:
            return
        try:
            # Schedule without awaiting to keep legacy sync method signature
            # Tests use AsyncMock and only check call arguments.
            pub(event_type, payload)  # type: ignore
        except Exception:
            pass

    def check_per_tick_limit(self) -> Tuple[bool, str]:
        """
        Legacy: Checks if the current tick's token usage exceeds the per-action limit with grace.
        Publishes BudgetWarning or BudgetExceeded (legacy string+dict format) accordingly.
        """
        limit = self._legacy["max_tokens_per_action"]
        exceeded = self.current_tick_tokens_used > limit
        msg = ""

        grace_limit = int(limit * (1 + self._legacy["grace_period_percentage"] / 100))
        hard_exceeded = self.current_tick_tokens_used > grace_limit

        if exceeded:
            if hard_exceeded and self._legacy["hard_fail_on_violation"]:
                msg = (
                    f"HARD VIOLATION: Per-tick token limit exceeded by "
                    f"{self.current_tick_tokens_used - limit} tokens (Grace Period Exceeded)."
                )
                self._trigger_hard_fail(msg)
                return False, msg
            else:
                msg = (
                    f"WARNING: Per-tick token limit nearing/exceeded by "
                    f"{self.current_tick_tokens_used - limit} tokens (within Grace Period)."
                )
                self._legacy_publish("BudgetWarning", {"type": "per_tick", "message": msg})
                return True, msg
        return True, ""

    def check_total_simulation_limit(self) -> Tuple[bool, str]:
        """
        Legacy: Checks if the total simulation token usage exceeds the total limit with grace.
        Publishes BudgetWarning or BudgetExceeded (legacy string+dict format) accordingly.
        """
        limit = self._legacy["max_total_tokens"]
        exceeded = self.total_simulation_tokens_used > limit
        msg = ""

        grace_limit = int(limit * (1 + self._legacy["grace_period_percentage"] / 100))
        hard_exceeded = self.total_simulation_tokens_used > grace_limit

        if exceeded:
            if hard_exceeded and self._legacy["hard_fail_on_violation"]:
                msg = (
                    f"HARD VIOLATION: Total simulation token budget exceeded by "
                    f"{self.total_simulation_tokens_used - limit} tokens (Grace Period Exceeded)."
                )
                self._trigger_hard_fail(msg)
                return False, msg
            else:
                msg = (
                    f"WARNING: Total simulation token budget nearing/exceeded by "
                    f"{self.total_simulation_tokens_used - limit} tokens (within Grace Period)."
                )
                self._legacy_publish("BudgetWarning", {"type": "total_sim", "message": msg})
                return True, msg
        return True, ""

    def _trigger_hard_fail(self, reason: str):
        """
        Legacy: Emits BudgetExceeded and raises SystemExit to simulate enforced stop.
        """
        if not self.violation_triggered:
            self.violation_triggered = True
            if self.metrics_tracker:
                try:
                    self.metrics_tracker.apply_penalty("budget_violation", self._legacy["violation_penalty_weight"])
                except Exception:
                    pass
            # publish legacy style event for tests
            self._legacy_publish("BudgetExceeded", {"reason": reason, "severity": "hard_fail"})
            raise SystemExit(reason)

    def get_remaining_budget_info(self) -> Dict[str, Any]:
        """
        Legacy: Provides a dictionary of current budget status using legacy config.
        """
        max_per_turn = self._legacy["max_tokens_per_action"]
        max_total = self._legacy["max_total_tokens"]
        remaining_current_tick = max(0, max_per_turn - self.current_tick_tokens_used)
        remaining_total_sim = max(0, max_total - self.total_simulation_tokens_used)

        estimated_cost_current_tick = self.token_counter.calculate_cost(
            self.current_tick_tokens_used, self._legacy["token_cost_per_1k"]
        )
        estimated_cost_total_sim = self.token_counter.calculate_cost(
            self.total_simulation_tokens_used, self._legacy["token_cost_per_1k"]
        )

        overall_budget_health = "HEALTHY"
        if max_per_turn and (self.current_tick_tokens_used / max_per_turn) > 0.8 or \
           max_total and (self.total_simulation_tokens_used / max_total) > 0.8:
            overall_budget_health = "WARNING"
        if self.current_tick_tokens_used > max_per_turn or self.total_simulation_tokens_used > max_total:
            overall_budget_health = "CRITICAL"

        return {
            "tokens_used_this_turn": self.current_tick_tokens_used,
            "max_tokens_per_turn": max_per_turn,
            "total_simulation_tokens_used": self.total_simulation_tokens_used,
            "max_total_simulation_tokens": max_total,
            "remaining_prompt_tokens": remaining_current_tick,
            "remaining_total_tokens": remaining_total_sim,
            "estimated_cost_this_turn_usd": estimated_cost_current_tick,
            "estimated_cost_total_sim_usd": estimated_cost_total_sim,
            "budget_health": overall_budget_health,
        }

    def format_budget_status_for_prompt(self) -> str:
        """
        Legacy: Generates a formatted string for injection into agent prompts.
        """
        budget_info = self.get_remaining_budget_info()

        tokens_used_this_turn = budget_info["tokens_used_this_turn"]
        max_tokens_per_turn = budget_info["max_tokens_per_turn"]
        total_simulation_tokens_used = budget_info["total_simulation_tokens_used"]
        max_total_simulation_tokens = budget_info["max_total_simulation_tokens"]
        estimated_cost_total_sim_usd = budget_info["estimated_cost_total_sim_usd"]
        budget_health = budget_info["budget_health"]

        percent_this_turn = (tokens_used_this_turn / max_tokens_per_turn) * 100 if max_tokens_per_turn else 0
        percent_total_sim = (total_simulation_tokens_used / max_total_simulation_tokens) * 100 if max_total_simulation_tokens else 0

        status_string = (
            "BUDGET STATUS:\n"
            f"- Tokens used this turn: {tokens_used_this_turn:,} / {max_tokens_per_turn:,} ({percent_this_turn:.1f}%)\n"
            f"- Total simulation tokens: {total_simulation_tokens_used:,} / {max_total_simulation_tokens:,} ({percent_total_sim:.1f}%)\n"
            f"- Remaining budget: ${estimated_cost_total_sim_usd:.2f} (at current token costs)\n"
            f"- Budget health: {budget_health}"
        )
        return status_string
