from .constraint_config import ConstraintConfig, get_tier_config_path
from .token_counter import TokenCounter
from typing import Dict, Any, Tuple
import os

class BudgetEnforcer:
    def __init__(self, config: ConstraintConfig, event_bus=None, metrics_tracker=None):
        self.config = config
        self.token_counter = TokenCounter()
        self.current_tick_tokens_used = 0
        self.total_simulation_tokens_used = 0
        self.event_bus = event_bus
        self.metrics_tracker = metrics_tracker
        self.violation_triggered = False

    @classmethod
    def from_tier_config(cls, tier: str, event_bus=None, metrics_tracker=None):
        config_path = get_tier_config_path(tier)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file for tier {tier} not found at {config_path}")
        config = ConstraintConfig.from_yaml(config_path)
        return cls(config, event_bus, metrics_tracker)

    def reset_for_new_tick(self):
        self.current_tick_tokens_used = 0

    def record_token_usage(self, tokens_used: int, action_type: str = "general"):
        """Records token usage for the current tick and overall simulation."""
        self.current_tick_tokens_used += tokens_used
        self.total_simulation_tokens_used += tokens_used

        # Notify metrics system if available
        if self.metrics_tracker:
            self.metrics_tracker.record_token_usage(tokens_used, action_type)

    def check_per_tick_limit(self) -> Tuple[bool, str]:
        """Checks if the current tick's token usage exceeds the limit."""
        exceeded = self.current_tick_tokens_used > self.config.max_tokens_per_action
        msg = ""

        # Apply grace period for soft violations on a per-tick basis
        grace_limit = self.config.max_tokens_per_action * (1 + self.config.grace_period_percentage / 100)
        hard_exceeded = self.current_tick_tokens_used > grace_limit

        if exceeded:
            if hard_exceeded and self.config.hard_fail_on_violation:
                msg = f"HARD VIOLATION: Per-tick token limit exceeded by {self.current_tick_tokens_used - self.config.max_tokens_per_action} tokens (Grace Period Exceeded)."
                self._trigger_hard_fail(msg)
                return False, msg # Return False as simulation will terminate
            else:
                msg = f"WARNING: Per-tick token limit nearing/exceeded by {self.current_tick_tokens_used - self.config.max_tokens_per_action} tokens (within Grace Period)."
                if self.event_bus:
                    self.event_bus.publish("BudgetWarning", {"type": "per_tick", "message": msg})
                # Soft violation, allow to continue but raise warning
                return True, msg # Return True as simulation can continue, but with warning
        return True, ""

    def check_total_simulation_limit(self) -> Tuple[bool, str]:
        """Checks if the total simulation token usage exceeds the limit."""
        exceeded = self.total_simulation_tokens_used > self.config.max_total_tokens
        msg = ""

        # Apply grace period for soft violations on a total simulation basis
        grace_limit = self.config.max_total_tokens * (1 + self.config.grace_period_percentage / 100)
        hard_exceeded = self.total_simulation_tokens_used > grace_limit

        if exceeded:
            if hard_exceeded and self.config.hard_fail_on_violation:
                msg = f"HARD VIOLATION: Total simulation token budget exceeded by {self.total_simulation_tokens_used - self.config.max_total_tokens} tokens (Grace Period Exceeded)."
                self._trigger_hard_fail(msg)
                return False, msg # Return False as simulation will terminate
            else:
                msg = f"WARNING: Total simulation token budget nearing/exceeded by {self.total_simulation_tokens_used - self.config.max_total_tokens} tokens (within Grace Period)."
                if self.event_bus:
                    self.event_bus.publish("BudgetWarning", {"type": "total_sim", "message": msg})
                # Soft violation, allow to continue but raise warning
                return True, msg # Return True as simulation can continue, but with warning
        return True, ""

    def _trigger_hard_fail(self, reason: str):
        if not self.violation_triggered:
            self.violation_triggered = True
            print(f"Simulation Hard Fail Triggered: {reason}")
            if self.metrics_tracker:
                self.metrics_tracker.apply_penalty("budget_violation", self.config.violation_penalty_weight)
            if self.event_bus:
                self.event_bus.publish("BudgetExceeded", {"reason": reason, "severity": "hard_fail"})
            # In a real simulation, this would typically raise an exception or signal termination
            raise SystemExit(reason) # Example: Terminate the simulation

    def get_remaining_budget_info(self) -> Dict[str, Any]:
        """Provides a dictionary of current budget status."""
        remaining_current_tick = max(0, self.config.max_tokens_per_action - self.current_tick_tokens_used)
        remaining_total_sim = max(0, self.config.max_total_tokens - self.total_simulation_tokens_used)
        
        estimated_cost_current_tick = self.token_counter.calculate_cost(
            self.current_tick_tokens_used, self.config.token_cost_per_1k
        )
        estimated_cost_total_sim = self.token_counter.calculate_cost(
            self.total_simulation_tokens_used, self.config.token_cost_per_1k
        )

        overall_budget_health = "HEALTHY"
        if self.current_tick_tokens_used / self.config.max_tokens_per_action > 0.8 or \
           self.total_simulation_tokens_used / self.config.max_total_tokens > 0.8:
            overall_budget_health = "WARNING"
        if self.current_tick_tokens_used > self.config.max_tokens_per_action or \
           self.total_simulation_tokens_used > self.config.max_total_tokens:
            overall_budget_health = "CRITICAL" # Even with grace period, visually show critical

        return {
            "tokens_used_this_turn": self.current_tick_tokens_used,
            "max_tokens_per_turn": self.config.max_tokens_per_action,
            "total_simulation_tokens_used": self.total_simulation_tokens_used,
            "max_total_simulation_tokens": self.config.max_total_tokens,
            "remaining_prompt_tokens": remaining_current_tick, # This is an internal calc, mainly for agent to reason
            "remaining_total_tokens": remaining_total_sim,
            "estimated_cost_this_turn_usd": estimated_cost_current_tick,
            "estimated_cost_total_sim_usd": estimated_cost_total_sim,
            "budget_health": overall_budget_health,
        }

    def format_budget_status_for_prompt(self) -> str:
        """Generates a formatted string for injection into agent prompts."""
        budget_info = self.get_remaining_budget_info()

        tokens_used_this_turn = budget_info["tokens_used_this_turn"]
        max_tokens_per_turn = budget_info["max_tokens_per_turn"]
        total_simulation_tokens_used = budget_info["total_simulation_tokens_used"]
        max_total_simulation_tokens = budget_info["max_total_simulation_tokens"]
        estimated_cost_total_sim_usd = budget_info["estimated_cost_total_sim_usd"]
        budget_health = budget_info["budget_health"]

        # Calculate percentages safely to avoid division by zero
        percent_this_turn = (tokens_used_this_turn / max_tokens_per_turn) * 100 if max_tokens_per_turn else 0
        percent_total_sim = (total_simulation_tokens_used / max_total_simulation_tokens) * 100 if max_total_simulation_tokens else 0
        
        status_string = (
            "BUDGET STATUS:\n"
            f"- Tokens used this turn: {tokens_used_this_turn:,} / {max_tokens_per_turn:,} ({percent_this_turn:.1f}%)\n"
            f"- Total simulation tokens: {total_simulation_tokens_used:,} / {max_total_simulation_tokens:,} ({percent_total_sim:.1f}%)\n"
            f"- Remaining budget: ${estimated_cost_total_sim_usd:.2f} (at current token costs)\n" # This should ideally be 'remaining cost budget'
            f"- Budget health: {budget_health}"
        )
        return status_string

