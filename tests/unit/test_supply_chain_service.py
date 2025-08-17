import asyncio
from datetime import datetime

import pytest

from event_bus import EventBus, set_event_bus
from services.world_store import WorldStore
from services.supply_chain_service import SupplyChainService
from fba_events.supplier import PlaceOrderCommand
from events import TickEvent
from money import Money


@pytest.mark.asyncio
async def test_supply_chain_schedules_and_delivers_inventory():
    # Arrange
    bus = EventBus()
    await bus.start()
    bus.start_recording()
    set_event_bus(bus)

    world_store = WorldStore(event_bus=bus)
    await world_store.start()

    # Base lead time = 2 ticks
    supply_chain = SupplyChainService(world_store=world_store, event_bus=bus, base_lead_time=2)
    await supply_chain.start()

    # Init product
    asin = "UT-SUP-ASIN-001"
    initial_qty = 10
    world_store.initialize_product(asin, initial_price=Money.from_dollars("10.00"), initial_inventory=initial_qty)

    # Act: place order for 50 units
    order_qty = 50
    cmd = PlaceOrderCommand(
        event_id="po-1",
        timestamp=datetime.now(),
        agent_id="agent-supply-ut",
        supplier_id="supplier-1",
        asin=asin,
        quantity=order_qty,
        max_price=Money.from_dollars("3.00"),
        reason="unit-test-order"
    )
    await bus.publish(cmd)
    await asyncio.sleep(0.02)

    # Tick progression: deliveries arrive at current_tick + base_lead_time
    # Tick 0
    await bus.publish(TickEvent(event_id="t0", timestamp=datetime.now(), tick_number=0))
    await asyncio.sleep(0.02)
    assert world_store.get_product_inventory_quantity(asin) == initial_qty

    # Tick 1
    await bus.publish(TickEvent(event_id="t1", timestamp=datetime.now(), tick_number=1))
    await asyncio.sleep(0.02)
    assert world_store.get_product_inventory_quantity(asin) == initial_qty

    # Tick 2 -> arrival should process
    await bus.publish(TickEvent(event_id="t2", timestamp=datetime.now(), tick_number=2))
    await asyncio.sleep(0.05)

    # Assert inventory increased by delivered amount (full, since no disruption)
    inv_after = world_store.get_product_inventory_quantity(asin)
    assert inv_after == initial_qty + order_qty, f"Expected inventory {initial_qty + order_qty}, got {inv_after}"

    # Assert InventoryUpdate event recorded
    recorded = bus.get_recorded_events()
    inv_updates = [e for e in recorded if e.get("event_type") == "InventoryUpdate" and e.get("data", {}).get("asin") == asin]
    assert inv_updates, "Expected InventoryUpdate event for delivery"


@pytest.mark.asyncio
async def test_supply_chain_disruption_delays_and_reduces_fulfillment():
    # Arrange
    bus = EventBus()
    await bus.start()
    bus.start_recording()
    set_event_bus(bus)

    world_store = WorldStore(event_bus=bus)
    await world_store.start()

    # Base lead time = 2 ticks; disruption adds +1 lead time and 50% fulfillment
    supply_chain = SupplyChainService(world_store=world_store, event_bus=bus, base_lead_time=2)
    supply_chain.set_disruption(active=True, lead_time_increase=1, fulfillment_rate=0.5)
    await supply_chain.start()

    asin = "UT-SUP-ASIN-002"
    initial_qty = 0
    world_store.initialize_product(asin, initial_price=Money.from_dollars("12.00"), initial_inventory=initial_qty)

    order_qty = 40
    cmd = PlaceOrderCommand(
        event_id="po-2",
        timestamp=datetime.now(),
        agent_id="agent-supply-ut",
        supplier_id="supplier-1",
        asin=asin,
        quantity=order_qty,
        max_price=Money.from_dollars("4.00"),
        reason="unit-test-disruption"
    )
    await bus.publish(cmd)
    await asyncio.sleep(0.02)

    # With disruption, arrival at tick 0 + 2 + 1 = 3 for first partial lot
    # Tick 0, 1, 2: no arrival
    for t in (0, 1, 2):
        await bus.publish(TickEvent(event_id=f"td{t}", timestamp=datetime.now(), tick_number=t))
        await asyncio.sleep(0.02)
    assert world_store.get_product_inventory_quantity(asin) == 0

    # Tick 3 -> first partial delivery (50% of 40 = 20)
    await bus.publish(TickEvent(event_id="td3", timestamp=datetime.now(), tick_number=3))
    await asyncio.sleep(0.05)
    inv_after_first = world_store.get_product_inventory_quantity(asin)
    assert inv_after_first == 20, f"Expected partial delivery of 20 units, got {inv_after_first}"

    # Remaining 20 should be scheduled for next tick (re-queued)
    await bus.publish(TickEvent(event_id="td4", timestamp=datetime.now(), tick_number=4))
    await asyncio.sleep(0.05)
    inv_after_second = world_store.get_product_inventory_quantity(asin)
    assert inv_after_second == 40, f"Expected second partial delivery bringing total to 40, got {inv_after_second}"

    # Validate we saw at least two InventoryUpdate events for the ASIN
    recorded = bus.get_recorded_events()
    inv_updates = [e for e in recorded if e.get("event_type") == "InventoryUpdate" and e.get("data", {}).get("asin") == asin]
    assert len(inv_updates) >= 2, "Expected multiple InventoryUpdate events during disrupted deliveries"