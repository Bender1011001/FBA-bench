import asyncio
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from event_bus import EventBus, AsyncioQueueBackend
from events import WorldStateSnapshotEvent
from money import Money
from services.toolbox_api_service import ToolboxAPIService
from services.toolbox_schemas import (
    ObserveRequest,
    SetPriceRequest,
    LaunchProductRequest,
)


@pytest_asyncio.fixture
async def event_bus():
    bus = EventBus(AsyncioQueueBackend())
    await bus.start()
    # enable recording so we can assert on published events
    bus.start_recording()
    yield bus
    await bus.stop()


@pytest_asyncio.fixture
async def toolbox_service(event_bus: EventBus):
    svc = ToolboxAPIService()
    await svc.start(event_bus)
    return svc


@pytest.mark.asyncio
async def test_observe_returns_found_false_when_no_cache(toolbox_service: ToolboxAPIService):
    asin = "B00TEST01"
    resp = toolbox_service.observe(ObserveRequest(asin=asin))
    assert resp.asin == asin
    assert resp.found is False
    assert resp.price is None
    assert resp.inventory is None
    assert resp.bsr is None
    assert resp.conversion_rate is None
    assert isinstance(resp.timestamp, datetime)


@pytest.mark.asyncio
async def test_world_snapshot_populates_cache_and_observe_returns_data(event_bus: EventBus, toolbox_service: ToolboxAPIService):
    asin = "B00TEST02"
    snapshot = WorldStateSnapshotEvent(
        event_id="snap-001",
        timestamp=datetime.now(timezone.utc),
        snapshot_id="snapshot-001",
        tick_number=0,
        product_count=1,
        summary_metrics={
            "products": {
                asin: {
                    "price_cents": 1299,
                    "inventory": 50,
                    "bsr": 1200,
                    "conversion_rate": 0.12,
                }
            }
        },
    )
    await event_bus.publish(snapshot)
    await asyncio.sleep(0.05)

    resp = toolbox_service.observe(ObserveRequest(asin=asin))
    assert resp.found is True
    assert isinstance(resp.price, Money)
    assert resp.price.cents == 1299
    assert resp.inventory == 50
    assert resp.bsr == 1200
    assert pytest.approx(resp.conversion_rate, rel=1e-6) == 0.12
    assert isinstance(resp.timestamp, datetime)


@pytest.mark.asyncio
async def test_set_price_publishes_setpricecommand(event_bus: EventBus, toolbox_service: ToolboxAPIService):
    asin = "B00TEST03"
    # prime minimal cache so observe works later if needed
    await event_bus.publish(
        WorldStateSnapshotEvent(
            event_id="snap-002",
            timestamp=datetime.now(timezone.utc),
            snapshot_id="snapshot-002",
            tick_number=0,
            product_count=1,
            summary_metrics={"products": {asin: {"price_cents": 1200}}},
        )
    )
    await asyncio.sleep(0.02)

    req = SetPriceRequest(
        agent_id="agent-1", asin=asin, new_price=Money.from_dollars("12.99"), reason="unit-test"
    )
    rsp = toolbox_service.set_price(req)
    assert rsp.accepted is True
    assert rsp.command_id
    assert rsp.asin == asin
    assert isinstance(rsp.new_price, Money)
    await asyncio.sleep(0.05)

    recorded = event_bus.get_recorded_events()
    assert any(e.get("event_type") == "SetPriceCommand" and e.get("data", {}).get("asin") == asin for e in recorded)


@pytest.mark.asyncio
async def test_launch_product_registers_and_observe_reflects(event_bus: EventBus, toolbox_service: ToolboxAPIService):
    asin = "B00NEW001"
    req = LaunchProductRequest(
        asin=asin,
        initial_price=Money.from_dollars("15.75"),
        initial_inventory=25,
        category="TestCat",
        dimensions_inches=[1, 2, 3],
        weight_oz=0,
    )
    rsp = toolbox_service.launch_product(req)
    assert rsp.accepted is True
    assert rsp.asin == asin

    reg = toolbox_service.get_registered(asin)
    assert reg is not None
    assert isinstance(reg["initial_price"], Money)
    assert reg["initial_price"].cents == 1575
    assert reg["initial_inventory"] == 25

    obs = toolbox_service.observe(ObserveRequest(asin=asin))
    assert obs.found is True
    assert obs.price.cents == 1575
    assert obs.inventory == 25