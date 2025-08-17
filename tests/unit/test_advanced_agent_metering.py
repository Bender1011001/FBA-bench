import asyncio
import pytest

from event_bus import EventBus
from constraints.budget_enforcer import BudgetEnforcer
from agents.advanced_agent import AdvancedAgent


@pytest.mark.asyncio
async def test_advanced_agent_forwarding_and_usage():
    bus = EventBus()
    await bus.start()
    bus.start_recording()

    enforcer = BudgetEnforcer({
        "limits": {"total_tokens_per_tick": 1000},
        "allow_soft_overage": False
    })
    await enforcer.start(bus)

    agent = AdvancedAgent(agent_id="agent-42", budget_enforcer=enforcer, event_bus=bus)
    res = await agent.meter_api_call("toolx", tokens_prompt=10, tokens_completion=5, cost_cents=12)
    assert res["exceeded"] is False

    await asyncio.sleep(0.05)
    snap = enforcer.get_usage_snapshot("agent-42")
    assert snap["tick"]["tokens"] == 15
    assert snap["run"]["tokens"] == 15
    assert snap["tick"]["per_tool"]["toolx"]["calls"] == 1


@pytest.mark.asyncio
async def test_advanced_agent_hard_exceed_raises_and_records():
    bus = EventBus()
    await bus.start()
    bus.start_recording()

    enforcer = BudgetEnforcer({
        "limits": {"total_tokens_per_tick": 20},
        "allow_soft_overage": False
    })
    await enforcer.start(bus)

    agent = AdvancedAgent(agent_id="agent-99", budget_enforcer=enforcer, event_bus=bus)

    with pytest.raises(RuntimeError):
        await agent.meter_api_call("heavy_tool", tokens_prompt=30, tokens_completion=0, cost_cents=0)

    # Ensure BudgetExceeded event recorded by enforcer
    await asyncio.sleep(0.05)
    rec = bus.get_recorded_events()
    exceeded = [e for e in rec if e.get("event_type") == "BudgetExceeded"]
    assert len(exceeded) >= 1
    assert exceeded[-1]["data"]["severity"] == "hard_fail"