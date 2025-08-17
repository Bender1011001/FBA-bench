import asyncio
from datetime import datetime
import pytest

from event_bus import EventBus
from events import TickEvent
from constraints.budget_enforcer import BudgetEnforcer


@pytest.mark.asyncio
async def test_warning_threshold_tokens_per_tick():
    # Setup EventBus with in-memory backend
    bus = EventBus()
    await bus.start()
    bus.start_recording()

    # Configure small per-tick token limit and 50% warning threshold
    enforcer = BudgetEnforcer({
        "limits": {"total_tokens_per_tick": 100},
        "warning_threshold_pct": 0.5,
        "allow_soft_overage": False
    })
    await enforcer.start(bus)

    agent = "agent-a"
    # 30 + 25 = 55 tokens, crosses 50% threshold
    res = await enforcer.meter_api_call(agent, "research", tokens_prompt=30, tokens_completion=25)
    assert res["exceeded"] is False

    # Allow event loop to process published events
    await asyncio.sleep(0.05)
    rec = bus.get_recorded_events()
    warnings = [e for e in rec if e.get("event_type") == "BudgetWarning"]
    assert len(warnings) >= 1

    # Verify usage snapshot matches counters
    snap = enforcer.get_usage_snapshot(agent)
    assert snap["tick"]["tokens"] == 55
    assert snap["run"]["tokens"] == 55
    assert snap["tick"]["calls"] == 1
    assert "research" in snap["tick"]["per_tool"]

@pytest.mark.asyncio
async def test_hard_exceed_total_tokens_per_tick():
    bus = EventBus()
    await bus.start()
    bus.start_recording()

    enforcer = BudgetEnforcer({
        "limits": {"total_tokens_per_tick": 60},
        "allow_soft_overage": False
    })
    await enforcer.start(bus)

    agent = "agent-b"
    res = await enforcer.meter_api_call(agent, "planner", tokens_prompt=61, tokens_completion=0)
    assert res["exceeded"] is True
    assert res["severity"] == "hard_fail"
    assert res["limit"] == 60
    assert res["usage"] == 61

    await asyncio.sleep(0.05)
    rec = bus.get_recorded_events()
    exceeded = [e for e in rec if e.get("event_type") == "BudgetExceeded"]
    assert len(exceeded) >= 1
    data = exceeded[-1]["data"]
    assert data["agent_id"] == agent
    assert data["severity"] == "hard_fail"
    assert data["limit"] == 60
    assert data["current_usage"] == 61

@pytest.mark.asyncio
async def test_per_tool_calls_limit_tick():
    bus = EventBus()
    await bus.start()
    bus.start_recording()

    enforcer = BudgetEnforcer({
        "tool_limits": {"search": {"calls_per_tick": 2}},
        "allow_soft_overage": False
    })
    await enforcer.start(bus)

    agent = "agent-c"
    r1 = await enforcer.meter_api_call(agent, "search")
    assert not r1["exceeded"]
    r2 = await enforcer.meter_api_call(agent, "search")
    assert not r2["exceeded"]
    r3 = await enforcer.meter_api_call(agent, "search")
    assert r3["exceeded"] is True

    await asyncio.sleep(0.05)
    rec = bus.get_recorded_events()
    exceeded = [e for e in rec if e.get("event_type") == "BudgetExceeded"]
    assert len(exceeded) >= 1
    # Ensure it is marked as a tool-level budget
    assert "tool:search.calls_per_tick" in exceeded[-1]["data"]["budget_type"]

@pytest.mark.asyncio
async def test_tick_reset_preserves_run_totals():
    bus = EventBus()
    await bus.start()
    bus.start_recording()

    enforcer = BudgetEnforcer({
        "limits": {"total_tokens_per_tick": 1_000},
    })
    await enforcer.start(bus)

    agent = "agent-d"
    await enforcer.meter_api_call(agent, "tool-x", tokens_prompt=10, tokens_completion=5, cost_cents=12)

    # Publish TickEvent to reset per-tick counters
    tick_event = TickEvent(
        event_id="tick-1",
        timestamp=datetime.utcnow(),
        tick_number=1,
        simulation_time=datetime.utcnow(),
        metadata={}
    )
    await bus.publish(tick_event)
    await asyncio.sleep(0.05)

    snap = enforcer.get_usage_snapshot(agent)
    assert snap["tick"]["tokens"] == 0
    assert snap["run"]["tokens"] == 15