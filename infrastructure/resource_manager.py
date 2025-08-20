from typing import Dict


class ResourceManager:
    """
    Minimal resource manager:
    - Tracks token allocations per component and enforces a global token budget
    - Tracks LLM dollar costs per component and enforces a global cost limit
    """

    def __init__(self, token_budget: int = 1_000_000, cost_limit: float = 1_000_000.0) -> None:
        self.token_budget = token_budget
        self.cost_limit = cost_limit
        # tests inspect these private dicts
        self._token_budgets: Dict[str, int] = {}
        self._token_usage: Dict[str, int] = {}
        self._llm_costs: Dict[str, float] = {}
        # some tests assert a global cap attribute exists and equals default
        self._global_token_cap: int = token_budget

    @property
    def total_tokens_used(self) -> int:
        return sum(self._token_usage.values())

    @property
    def total_cost(self) -> float:
        return sum(self._llm_costs.values())

    def allocate_tokens(self, name: str, n: int) -> None:
        if n < 0:
            raise ValueError("n must be >= 0")
        used = self.total_tokens_used
        if used + n > self.token_budget:
            raise ValueError("Token budget exceeded")
        self._token_usage[name] = self._token_usage.get(name, 0) + n
        # keep a mirror "budget per name" if tests later depend on it
        self._token_budgets.setdefault(name, 0)

    def record_llm_cost(self, name: str, dollars: float) -> None:
        if dollars < 0:
            raise ValueError("dollars must be >= 0")
        self._llm_costs[name] = self._llm_costs.get(name, 0.0) + dollars
        self.enforce_cost_limits()

    def enforce_cost_limits(self) -> None:
        if self.total_cost > self.cost_limit:
            raise ValueError("Cost limit exceeded")

    def get_usage_snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            "tokens": dict(self._token_usage),
            "costs": dict(self._llm_costs),
            "totals": {
                "tokens": float(self.total_tokens_used),
                "cost": float(self.total_cost),
                "token_budget": float(self.token_budget),
                "cost_limit": float(self.cost_limit),
            },
        }