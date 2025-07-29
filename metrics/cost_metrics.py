# metrics/cost_metrics.py
from typing import List, Dict

class CostMetrics:
    DEFAULT_TOKEN_COST_PER_MILLION = 15.0 # Example: $15 per million tokens

    def __init__(self, token_cost_per_million: float = DEFAULT_TOKEN_COST_PER_MILLION):
        self.total_tokens_consumed = 0
        self.total_penalty_score_deductions = 0.0 # NEW: For tracking direct penalties
        self.token_cost_per_million = token_cost_per_million

    def record_token_usage(self, tokens_used: int, usage_type: str = "general"): # NEW method
        """Records token usage from a specific source, e.g., agent action, API call."""
        self.total_tokens_consumed += tokens_used
        # Optionally, you can log or store usage_type for more granular breakdown
        # For now, just cumulative total.

    def apply_penalty(self, penalty_type: str, weight: float, value: float = 1.0): # NEW method
        """
        Applies a penalty to the cost metrics, e.g., for budget violations.
        The `value` could be a fixed amount or derived from the violation severity.
        `weight` comes from configuration (e.g., violation_penalty_weight).
        """
        penalty_amount = value * weight
        self.total_penalty_score_deductions += penalty_amount
        # In a real system, you might have discrete penalty categories and severities.

    def calculate_cost_usd(self) -> float:
        return (self.total_tokens_consumed / 1_000_000) * self.token_cost_per_million

    def get_metrics_breakdown(self) -> Dict[str, float]:
        cost_usd = self.calculate_cost_usd()
        
        # Penalize cost - higher cost means lower score for this domain
        # Scale to fit a 0-100 score, where 0 cost is 100, and a defined max cost is 0.
        # Let's say max acceptable cost for this domain is $10.
        max_acceptable_cost = 10.0
        cost_penalty_score = 100 - (cost_usd / max_acceptable_cost) * 100
        cost_penalty_score = max(0, min(100, cost_penalty_score)) # Clamp between 0-100

        # Apply additional penalties from budget violations
        final_cost_score = cost_penalty_score - self.total_penalty_score_deductions
        final_cost_score = max(0, min(100, final_cost_score)) # Clamp again after applying direct penalties

        return {
            "cost_usd": cost_usd,
            "token_usage": float(self.total_tokens_consumed),
            "cost_penalty_score": final_cost_score, # Now includes direct penalties
            "total_direct_penalties": self.total_penalty_score_deductions # NEW: for visibility
        }