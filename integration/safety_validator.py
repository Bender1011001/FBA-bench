from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class SafetyConstraint:
    """
    Generic safety constraint used by tests.

    Fields observed in tests:
      - name: str
      - constraint_type: str  (e.g., "percentage_change", "absolute_value", "rate_limit", "absolute_minimum")
      - max_value: float
      - time_window_minutes: int
    """
    name: str
    constraint_type: str
    max_value: float
    time_window_minutes: int


@dataclass
class _ValidationResult:
    # Compatibility shape accessed by tests:
    #  - is_safe: bool
    #  - violations: list[str]
    is_safe: bool = True
    violations: List[str] = field(default_factory=list)


class SafetyValidator:
    """
    Deterministic safety validator with minimal state to satisfy tests:
      - add_constraint(constraint)
      - validate_action(action) -> result with is_safe and violations
      - get_circuit_breaker_status() -> dict(triggered: bool, ...)
      - emergency_stop(reason: str) -> bool
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._constraints: List[SafetyConstraint] = []

        # Simple counters for "rate_limit" constraints and circuit breaker
        self._window_start = monotonic()
        self._window_seconds_default = 60.0  # convert minutes to window as needed
        self._call_counts: Dict[str, int] = {}
        self._violation_count: int = 0
        self._circuit_breaker_triggered: bool = False
        self._emergency_stopped: bool = False

    async def add_constraint(self, constraint: SafetyConstraint) -> None:
        with self._lock:
            self._constraints.append(constraint)

    async def validate_action(self, action: Dict[str, Any]) -> _ValidationResult:
        """
        Validate an action dict. Tests pass actions like:
          - {"action": "set_price", "asin": "...", "old_price": 1000, "new_price": 1200}
          - {"action": "place_order", "asin": "...", "quantity": 5000, "unit_cost": 15}
          - {"action": "set_price", "asin": "...", "new_price": -10}
        """
        res = _ValidationResult(is_safe=True, violations=[])

        if self._emergency_stopped:
            res.is_safe = False
            res.violations.append("emergency_stop_active")
            await self._record_violation("emergency_stop_active")
            return res

        action_type = str(action.get("action", "")).lower().strip()
        now = monotonic()
        with self._lock:
            # reset window counters every minute for simplicity
            if now - self._window_start >= 60.0:
                self._window_start = now
                self._call_counts.clear()

        # Evaluate constraints
        for c in list(self._constraints):
            ctype = c.constraint_type
            try:
                if ctype == "percentage_change" and action_type == "set_price":
                    old_price = float(action.get("old_price", 0) or 0)
                    new_price = float(action.get("new_price", 0) or 0)
                    # If old_price missing, treat change as 0 (no violation)
                    if old_price > 0:
                        pct = abs((new_price - old_price) / old_price)
                        if pct > float(c.max_value):
                            res.is_safe = False
                            res.violations.append(c.name)

                elif ctype == "absolute_value":
                    # Used for order_value_limit in tests
                    if action_type == "place_order":
                        unit_cost = float(action.get("unit_cost", 0) or 0)
                        quantity = int(action.get("quantity", 0) or 0)
                        order_value = unit_cost * quantity
                        if order_value > float(c.max_value):
                            res.is_safe = False
                            res.violations.append(c.name)

                elif ctype == "absolute_minimum":
                    # Used for negative price block
                    if action_type == "set_price":
                        new_price = float(action.get("new_price", 0) or 0)
                        if new_price < float(c.max_value):
                            res.is_safe = False
                            res.violations.append(c.name)

                elif ctype == "rate_limit":
                    # Simple count within the 1-minute rolling window
                    key = f"rate_limit::{c.name}"
                    with self._lock:
                        self._call_counts[key] = self._call_counts.get(key, 0) + 1
                        if self._call_counts[key] > int(c.max_value):
                            res.is_safe = False
                            res.violations.append(c.name)

            except Exception as e:
                logger.warning("SafetyValidator constraint check error for %s: %s", c.name, e)

        # Update circuit breaker on violations
        if not res.is_safe:
            await self._record_violation(",".join(res.violations))

        return res

    async def validate_action_safety(self, action: Dict[str, Any]) -> bool:
        """
        Some tests may call real_world_adapter.validate_action_safety(action) and expect
        a falsy result for unsafe actions. Provide a compatible helper here for reuse.
        """
        result = await self.validate_action(action)
        return result.is_safe

    async def get_circuit_breaker_status(self) -> Dict[str, Any]:
        with self._lock:
            return {"triggered": self._circuit_breaker_triggered, "violations": self._violation_count}

    async def emergency_stop(self, reason: str) -> bool:
        with self._lock:
            self._emergency_stopped = True
        logger.error("Emergency stop activated: %s", reason)
        return True

    async def reset(self) -> None:
        with self._lock:
            self._violation_count = 0
            self._circuit_breaker_triggered = False
            self._emergency_stopped = False
            self._window_start = monotonic()
            self._call_counts.clear()

    # ---------------- internal ----------------
    async def _record_violation(self, tag: str) -> None:
        with self._lock:
            self._violation_count += 1
            # Trigger breaker if many violations within short period (heuristic)
            if self._violation_count >= 5:
                self._circuit_breaker_triggered = True


__all__ = [
    "SafetyValidator",
    "SafetyConstraint",
    "RiskLevel",
]