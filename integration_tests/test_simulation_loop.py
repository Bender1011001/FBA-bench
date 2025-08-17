import asyncio
from datetime import datetime, timezone

import pytest
from unittest.mock import AsyncMock, patch

from event_bus import EventBus, set_event_bus
from services.world_store import WorldStore
from services.market_simulator import MarketSimulationService
from services.supply_chain_service import SupplyChainService
from services.marketing_service import MarketingService
from agents.multi_domain_controller import MultiDomainController
from agents.skill_coordinator import SkillCoordinator
from agents.skill_modules.base_skill import SkillAction
from money import Money


class DummyRunnerToolCall:
    """Lightweight ToolCall-like object to reuse AgentManager path if needed."""
    def __init__(self, tool_name: str, parameters: dict, confidence: float = 0.9, priority: int = 80, reasoning: str = ""):
        self.tool_name = tool_name
        self.parameters = parameters
        self.confidence = confidence
        self.priority = priority
        self.reasoning = reasoning


@pytest.mark.asyncio
async def test_multi_skill_arbitration_single_action_approved_under_tight_budget():
    """
    Multi-skill arbitration integration test (CEO-level):
      - Create competing actions: 'place_order' (supply) and 'run_marketing_campaign' (marketing).
      - Impose a very tight budget in the MultiDomainController resource plan.
      - Spy on MultiDomainController.arbitrate_actions to ensure it is invoked with both actions.
      - Assert only one action is approved due to budget constraint.
      - Execute the approved action through services and validate resulting world state effect.
    """
    # Setup EventBus
    bus = EventBus()
    await bus.start()
    bus.start_recording()
    set_event_bus(bus)

    # Setup WorldStore
    world_store = WorldStore(event_bus=bus)
    await world_store.start()

    # Core services
    market = MarketSimulationService(world_store=world_store, event_bus=bus, base_demand=80, demand_elasticity=1.1)
    await market.start()
    supply_chain = SupplyChainService(world_store=world_store, event_bus=bus, base_lead_time=1)
    await supply_chain.start()
    marketing = MarketingService(world_store=world_store, event_bus=bus, alpha_per_dollar=0.005, retention=0.6)
    await marketing.start()

    # Initialize a product in world store
    asin = "ARBITRATE-ASIN-001"
    initial_price = Money.from_dollars("20.00")
    world_store.initialize_product(asin, initial_price, initial_inventory=10)  # low inventory to make supply action valuable

    # Build a MultiDomainController with very tight budget
    coordinator = SkillCoordinator(agent_id="agent-1", event_bus=bus, config={})
    controller = MultiDomainController(agent_id="agent-1", skill_coordinator=coordinator, config={
        "total_budget_cents": 5000  # $50 total
    })
    # Overwrite domain allocations to be even tighter and explicit
    controller.resource_plan.total_budget = Money(5000)  # $50
    controller.resource_plan.allocations.update({
        "inventory_management": Money(2000),  # $20
        "marketing": Money(2000),             # $20
        "strategic_reserve": Money(1000),     # $10
    })
    # Keep multipliers neutral to simplify
    controller._update_priority_multipliers()

    # Compose two competing actions exceeding per-domain thresholds when combined
    place_order_action = SkillAction(
        action_type="place_order",
        parameters={
            "supplier_id": "SUP-1",
            "asin": asin,
            "quantity": 25,
            "max_price": str(Money.from_dollars("1.00"))  # $1/unit => $25 intent cost
        },
        confidence=0.9,
        reasoning="Replenish low inventory to avoid stockouts",
        priority=0.8,
        resource_requirements={"budget_cents": 2500},
        expected_outcome={"inventory_increase": 25},
        skill_source="SupplyManager"
    )

    run_campaign_action = SkillAction(
        action_type="run_marketing_campaign",
        parameters={
            "asin": asin,
            "campaign_type": "display_ads",
            "budget": str(Money.from_dollars("25.00")),
            "duration_days": 3
        },
        confidence=0.85,
        reasoning="Increase visibility to grow demand",
        priority=0.7,
        resource_requirements={"budget_cents": 2500},
        expected_outcome={"visibility_boost": 0.2},
        skill_source="MarketingManager"
    )

    competing = [place_order_action, run_campaign_action]

    # Spy on arbitrate_actions
    with patch.object(MultiDomainController, "arbitrate_actions", wraps=controller.arbitrate_actions) as spy_arbitrate:
        approved = await controller.arbitrate_actions(competing)

        # Ensure arbitration was called with both actions
        spy_arbitrate.assert_called_once()
        called_args, called_kwargs = spy_arbitrate.call_args
        assert len(called_args[0]) == 2, "Expected both competing actions passed to arbitrate_actions"

    # Under the tight budget, only one should be approved
    assert len(approved) == 1, f"Expected a single approved action due to budget constraint, got {len(approved)}"
    approved_action = approved[0]

    # Execute the approved action by publishing the corresponding command through services
    if approved_action.action_type == "place_order":
        # Publish PlaceOrderCommand via event bus for SupplyChainService
        from fba_events.supplier import PlaceOrderCommand as PO
        po = PO(
            event_id="test-po-1",
            timestamp=datetime.now(),
            supplier_id=approved_action.parameters["supplier_id"],
            asin=approved_action.parameters["asin"],
            quantity=int(approved_action.parameters["quantity"]),
            max_price=Money(approved_action.parameters["max_price"]),
            reason=approved_action.reasoning,
        )
        # attach agent id for traceability (optional)
        setattr(po, "agent_id", "agent-1")
        await bus.publish(po)

        # Tick once for arrival (base_lead_time=1)
        await bus.publish(type("TickEvent", (), {"event_type": "TickEvent", "tick_number": 1})())
        await asyncio.sleep(0.05)
        await supply_chain.process_tick()
        await asyncio.sleep(0.05)

        inv_qty = world_store.get_product_inventory_quantity(asin)
        assert inv_qty >= 10 + 1, "Inventory should have increased after approved place_order is executed"

        vis = world_store.get_marketing_visibility(asin)
        assert vis == 1.0 or vis is None, "Marketing visibility should remain unchanged when supply action is approved"

    elif approved_action.action_type == "run_marketing_campaign":
        # Publish RunMarketingCampaignCommand via event bus for MarketingService
        from fba_events.marketing import RunMarketingCampaignCommand as MK
        mk = MK(
            event_id="test-mk-1",
            timestamp=datetime.now(),
            campaign_type=approved_action.parameters["campaign_type"],
            budget=Money(approved_action.parameters["budget"]),
            duration_days=int(approved_action.parameters["duration_days"]),
            reason=approved_action.reasoning,
        )
        # dynamic asin attribute used by MarketingService
        setattr(mk, "asin", approved_action.parameters["asin"])
        setattr(mk, "agent_id", "agent-1")
        await bus.publish(mk)

        # Advance ticks to allow spend to apply
        for t in range(1, 3):
            await bus.publish(type("TickEvent", (), {"event_type": "TickEvent", "tick_number": t})())
            await asyncio.sleep(0.05)
            await marketing.process_tick()
            await asyncio.sleep(0.05)

        vis = world_store.get_marketing_visibility(asin)
        assert vis and vis > 1.0, "Marketing visibility should increase after approved marketing campaign"

        inv_qty = world_store.get_product_inventory_quantity(asin)
        assert inv_qty == 10, "Inventory should be unchanged when marketing action is approved"

    else:
        raise AssertionError(f"Unexpected approved action type: {approved_action.action_type}")

    # Light sanity: market still processes without errors
    await market.process_for_asin(asin)
import asyncio
from datetime import datetime, timezone

import pytest

from event_bus import EventBus, set_event_bus
from services.world_store import WorldStore
from services.market_simulator import MarketSimulationService
from services.supply_chain_service import SupplyChainService  # Not strictly required for this test but ensures imports are valid
from benchmarking.scenarios.price_optimization import PriceOptimizationScenario
from benchmarking.scenarios.base import ScenarioConfig
from money import Money


class DummyPricingAgent:
    """
    Minimal async agent compatible with PriceOptimizationScenario.
    Always sets price to $19.99.
    """
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    async def decide(self, agent_input: dict) -> dict:
        return {"new_price": "19.99"}


@pytest.mark.asyncio
async def test_core_simulation_loop_price_scenario():
    """
    End-to-end integration:
      Agent -> SetPriceCommand -> WorldStore -> MarketSimulationService -> World updates

    Asserts:
      - SetPriceCommand is published
      - WorldStore reflects new price after tick
      - MarketSimulationService publishes SaleOccurred
      - WorldStore inventory decreases after sale
      - AgentRunResult contains valid metrics
    """
    # Setup EventBus
    bus = EventBus()
    await bus.start()
    bus.start_recording()
    set_event_bus(bus)

    # Setup WorldStore
    world_store = WorldStore(event_bus=bus)
    await world_store.start()

    # Setup MarketSimulationService
    market = MarketSimulationService(world_store=world_store, event_bus=bus, base_demand=100, demand_elasticity=1.2)
    await market.start()

    # Optionally start SupplyChainService (not used in this test directly)
    supply_chain = SupplyChainService(world_store=world_store, event_bus=bus)
    await supply_chain.start()

    # Create scenario and initialize product
    scenario_cfg = ScenarioConfig(
        name="core-price-optimization-test",
        description="Core loop validation",
        parameters={
            "initial_product_price": 25.00,
            "simulation_duration_ticks": 5,
            "product_asin": "TEST-ASIN-001",
            "demand_elasticity": 1.2,
        },
    )
    scenario = PriceOptimizationScenario(scenario_cfg)

    # Initialize scenario with world_store so it creates the product
    await scenario.setup(world_store=world_store)

    # Prepare agent
    agent = DummyPricingAgent(agent_id="agent-core-test")

    # Track initial inventory and price
    product_asin = scenario.product_asin
    world_state = world_store.get_product_state(product_asin)
    if not world_state:
        # Initialize product defensively if setup didn't
        world_store.initialize_product(product_asin, Money.from_dollars("25.00"), initial_inventory=1000)
        world_state = world_store.get_product_state(product_asin)
    initial_inventory = world_state.inventory_quantity

    # Run 3 ticks
    sale_seen = False
    set_price_seen = False
    last_result = None

    for tick in range(3):
        # Execute scenario tick
        last_result = await scenario.run(agent, run_number=tick, world_store=world_store, event_bus=bus, market_simulator=market)
        # Allow async subscribers to process
        await asyncio.sleep(0.05)

        # Trigger market processing explicitly to ensure deterministic ordering (already done in scenario, but safe)
        await market.process_for_asin(product_asin)
        await asyncio.sleep(0.05)

        # Check recorded events so far
        recorded = bus.get_recorded_events()

        if any(e.get("event_type") == "SetPriceCommand" and e.get("data", {}).get("asin") == product_asin for e in recorded):
            set_price_seen = True
        if any(e.get("event_type") == "SaleOccurred" and e.get("data", {}).get("asin") == product_asin for e in recorded):
            sale_seen = True

    # Assertions

    # 1) SetPriceCommand was published
    assert set_price_seen, "Expected SetPriceCommand to be published by scenario/agent."

    # 2) WorldStore reflects new price
    updated_state = world_store.get_product_state(product_asin)
    assert updated_state is not None, "Product state should exist in WorldStore."
    assert updated_state.price.cents == Money.from_dollars("19.99").cents, "WorldStore price should reflect agent's price."

    # 3) MarketSimulationService published a SaleOccurred event (very likely given base demand and inventory)
    assert sale_seen, "Expected at least one SaleOccurred event from MarketSimulationService."

    # 4) Inventory decreased
    current_inventory = world_store.get_product_inventory_quantity(product_asin)
    assert current_inventory < initial_inventory, f"Inventory should decrease after sales; start={initial_inventory} end={current_inventory}"

    # 5) AgentRunResult contains valid metrics
    assert last_result is not None, "Scenario.run should return an AgentRunResult."
    assert last_result.success is True
    assert "current_price" in last_result.metrics
    assert "units_sold_this_tick" in last_result.metrics
    # Units sold could be zero on an unlucky tick, but across 3 ticks we saw SaleOccurred, so last_result may still be > 0
    assert isinstance(last_result.metrics["units_sold_this_tick"], int)