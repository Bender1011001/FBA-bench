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