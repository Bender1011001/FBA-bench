import math
import pytest

from agents.advanced_agent import AdvancedAgent  # noqa: F401
from agent_runners.base_runner import SimulationState, ToolCall


@pytest.mark.asyncio
async def test_decide_basic_returns_toolcall():
    cfg = {
        "id": "agent-1",
        "parameters": {
            "agent_type": "advanced",
            "target_asin": "B0TESTASIN",
            "min_margin": 0.12,
            "undercut": 0.01,
            "max_change_pct": 0.15,
            "price_sensitivity": 0.10,
            "reaction_speed": 1.0,
            "inventory_low_threshold": 10,
            "inventory_target": 100,
        },
    }
    agent = AdvancedAgent(cfg)
    # initialize explicitly to exercise the path; agent also self-inits on first decide()
    agent.initialize()

    state = SimulationState(
        tick=1,
        products=[{
            "asin": "B0TESTASIN",
            "price": 20.0,
            "cost": 10.0,
            "competitors": [{"price": 21.0}, {"price": 22.0}],
            "inventory": 100,
            "demand": 5.0
        }],
        recent_events=[
            {"asin": "B0TESTASIN", "units_sold": 2},
            {"asin": "B0TESTASIN", "units_sold": 1},
        ],
    )

    actions = await agent.decide(state)
    assert isinstance(actions, list)
    assert len(actions) == 1
    call = actions[0]
    assert isinstance(call, ToolCall)
    assert call.tool_name == "set_price"
    assert isinstance(call.parameters, dict)
    assert call.parameters.get("asin") == "B0TESTASIN"
    assert isinstance(call.parameters.get("price"), (float, int))
    assert 0.5 <= float(call.confidence) <= 0.99


@pytest.mark.asyncio
async def test_respects_cost_floor_even_with_smoothing():
    # Setup a case where competitor anchor is below cost floor to ensure floor enforcement post-smoothing
    cfg = {
        "id": "agent-2",
        "parameters": {
            "agent_type": "advanced",
            "target_asin": "B0FLOOR",
            "min_margin": 0.20,   # 20% margin
            "undercut": 0.01,
            "max_change_pct": 0.15,
            "price_sensitivity": 0.0,  # neutral
            "reaction_speed": 1.0,
            "inventory_target": 100,
        },
    }
    agent = AdvancedAgent(cfg)
    agent.initialize()

    # Current price is low; floor is high to trigger smoothing, then floor re-application
    state = SimulationState(
        tick=1,
        products=[{
            "asin": "B0FLOOR",
            "price": 10.0,         # current price
            "cost": 25.0,          # cost leads to floor 30.0
            "competitors": [{"price": 20.0}],
            "inventory": 100,
        }],
        recent_events=[],
    )

    actions = await agent.decide(state)
    call = actions[0]
    floor_price = 25.0 * 1.20
    assert call.parameters["price"] >= floor_price - 1e-6


@pytest.mark.asyncio
async def test_undercuts_competitor_when_safe():
    cfg = {
        "id": "agent-3",
        "parameters": {
            "agent_type": "advanced",
            "target_asin": "B0UNDER",
            "min_margin": 0.05,
            "undercut": 0.01,      # 1% undercut
            "max_change_pct": 0.50,  # allow larger move to avoid smoothing interference
            "price_sensitivity": 0.0,
            "reaction_speed": 1.0,
            "inventory_target": 100,
        },
    }
    agent = AdvancedAgent(cfg)
    agent.initialize()

    competitor_price = 20.00
    state = SimulationState(
        tick=1,
        products=[{
            "asin": "B0UNDER",
            "price": 19.50,
            "cost": 10.0,
            "competitors": [{"price": competitor_price}],
            "inventory": 100,
        }],
        recent_events=[],
    )

    actions = await agent.decide(state)
    call = actions[0]
    # Expected anchor undercut target = 20.0 * (1 - 0.01) = 19.8 (subject to floor and adjustments)
    expected_anchor = competitor_price * (1.0 - 0.01)
    # Because sensitivity and inventory adjustments are neutral, final target should be close to anchor or smoothed.
    assert call.parameters["price"] <= expected_anchor * 1.02 + 1e-6


@pytest.mark.asyncio
async def test_smoothing_limits_per_tick_change():
    cfg = {
        "id": "agent-4",
        "parameters": {
            "agent_type": "advanced",
            "target_asin": "B0SMOOTH",
            "min_margin": 0.05,
            "undercut": 0.01,
            "max_change_pct": 0.15,   # 15% cap
            "price_sensitivity": 0.0, # neutral to isolate smoothing
            "reaction_speed": 1.0,
            "inventory_target": 100,
        },
    }
    agent = AdvancedAgent(cfg)
    agent.initialize()

    current_price = 10.0
    competitor_price = 50.0  # if unconstrained, target would be very high
    state = SimulationState(
        tick=1,
        products=[{
            "asin": "B0SMOOTH",
            "price": current_price,
            "cost": 5.0,
            "competitors": [{"price": competitor_price}],
            "inventory": 100,
        }],
        recent_events=[],
    )

    actions = await agent.decide(state)
    call = actions[0]
    # Smoothing should cap increase to 15%: 10.0 * 1.15 = 11.5
    max_up = current_price * (1.0 + 0.15)
    assert call.parameters["price"] <= max_up + 1e-6


@pytest.mark.asyncio
async def test_inventory_pressure_adjusts_price_down_when_high_inventory():
    cfg = {
        "id": "agent-5",
        "parameters": {
            "agent_type": "advanced",
            "target_asin": "B0INV",
            "min_margin": 0.05,
            "undercut": 0.00,        # isolate inventory effect
            "max_change_pct": 0.50,
            "price_sensitivity": 0.5, # allow visible adjustment
            "reaction_speed": 1.0,
            "inventory_target": 100,
        },
    }
    agent = AdvancedAgent(cfg)
    agent.initialize()

    # High inventory should nudge price downward vs anchor
    state_high_inv = SimulationState(
        tick=1,
        products=[{
            "asin": "B0INV",
            "price": 20.0,
            "cost": 10.0,
            "competitors": [{"price": 20.0}],
            "inventory": 150,  # 1.5x target
        }],
        recent_events=[],
    )
    actions_high = await agent.decide(state_high_inv)
    price_high = actions_high[0].parameters["price"]

    # Normal inventory should produce a higher/equal price vs high inventory case (since high inventory pushes down)
    state_normal_inv = SimulationState(
        tick=2,
        products=[{
            "asin": "B0INV",
            "price": 20.0,
            "cost": 10.0,
            "competitors": [{"price": 20.0}],
            "inventory": 100,  # target
        }],
        recent_events=[],
    )
    actions_normal = await agent.decide(state_normal_inv)
    price_normal = actions_normal[0].parameters["price"]

    assert price_high <= price_normal + 1e-9


@pytest.mark.asyncio
async def test_demand_signal_adjusts_price_up_when_strong():
    cfg = {
        "id": "agent-6",
        "parameters": {
            "agent_type": "advanced",
            "target_asin": "B0DEM",
            "min_margin": 0.05,
            "undercut": 0.00,
            "max_change_pct": 0.50,
            "price_sensitivity": 0.5,
            "reaction_speed": 1.0,
            "inventory_target": 100,
        },
    }
    agent = AdvancedAgent(cfg)
    agent.initialize()

    base_state = SimulationState(
        tick=1,
        products=[{
            "asin": "B0DEM",
            "price": 20.0,
            "cost": 10.0,
            "competitors": [{"price": 20.0}],
            "inventory": 100,
        }],
        recent_events=[],  # avg demand baseline remains ~0 => factor 0
    )
    actions_base = await agent.decide(base_state)
    price_base = actions_base[0].parameters["price"]

    # Now provide strong recent demand to push price upward
    strong_demand_state = SimulationState(
        tick=2,
        products=[{
            "asin": "B0DEM",
            "price": 20.0,
            "cost": 10.0,
            "competitors": [{"price": 20.0}],
            "inventory": 100,
            "demand": 20.0
        }],
        recent_events=[
            {"asin": "B0DEM", "units_sold": 10},
            {"asin": "B0DEM", "units_sold": 12},
        ],
    )
    actions_strong = await agent.decide(strong_demand_state)
    price_strong = actions_strong[0].parameters["price"]

    assert price_strong >= price_base - 1e-9