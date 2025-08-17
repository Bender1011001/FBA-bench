from __future__ import annotations

from typing import Any, Dict

from event_bus import EventBus
from constraints.budget_enforcer import BudgetEnforcer


class AdvancedAgent:
    """
    AdvancedAgent provides a standard hook for metering API/tool calls through BudgetEnforcer.
    This class is intentionally minimal and non-invasive to existing runners.
    """

    def __init__(self, agent_id: str, budget_enforcer: BudgetEnforcer, event_bus: EventBus):
        self.agent_id = agent_id
        self.budget_enforcer = budget_enforcer
        self.event_bus = event_bus

    async def meter_api_call(
        self,
        tool_name: str,
        tokens_prompt: int = 0,
        tokens_completion: int = 0,
        cost_cents: int = 0,
    ) -> Dict[str, Any]:
        """
        Delegate metering to BudgetEnforcer. If a hard limit is exceeded and soft overage is not allowed,
        raise RuntimeError to simulate enforced stop behavior at the agent level.
        """
        result = await self.budget_enforcer.meter_api_call(
            agent_id=self.agent_id,
            tool_name=tool_name,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            cost_cents=cost_cents,
        )
        if result.get("exceeded"):
            allow_soft = bool(self.budget_enforcer.config.get("allow_soft_overage", False)) if isinstance(self.budget_enforcer.config, dict) else False
            if not allow_soft:
                # Hard fail behavior for agents
                raise RuntimeError(
                    f"Budget exceeded for {result.get('budget_type')}: usage {result.get('usage')} > limit {result.get('limit')}"
                )
        return result