import yaml
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class ConstraintConfig:
    max_tokens_per_action: int
    max_total_tokens: int
    token_cost_per_1k: float
    violation_penalty_weight: float
    grace_period_percentage: float
    hard_fail_on_violation: bool
    inject_budget_status: bool
    track_token_efficiency: bool
    # Add other flexible settings as needed, e.g., for different LLM providers

    @classmethod
    def from_yaml(cls, filepath: str):
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        budget_constraints = config_dict.get('budget_constraints', {})
        enforcement = config_dict.get('enforcement', {})

        return cls(
            max_tokens_per_action=budget_constraints.get('max_tokens_per_action'),
            max_total_tokens=budget_constraints.get('max_total_tokens'),
            token_cost_per_1k=budget_constraints.get('token_cost_per_1k'),
            violation_penalty_weight=budget_constraints.get('violation_penalty_weight'),
            grace_period_percentage=budget_constraints.get('grace_period_percentage'),
            hard_fail_on_violation=enforcement.get('hard_fail_on_violation'),
            inject_budget_status=enforcement.get('inject_budget_status'),
            track_token_efficiency=enforcement.get('track_token_efficiency')
        )

# Default configurations for different tiers
def get_tier_config_path(tier: str) -> str:
    return f"constraints/tier_configs/{tier.lower()}_config.yaml"
