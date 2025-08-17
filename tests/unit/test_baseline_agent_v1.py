import asyncio
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from event_bus import EventBus, AsyncioQueueBackend
from events import WorldStateSnapshotEvent
from money import Money
from agents.baseline.baseline_agent_v1 import BaselineAgentV1
from services.toolbox_api_service import ToolboxAPIService
from services.toolbox_schemas import ObserveRequest


@pytest_asyncio.fixture
async def event_bus():
    bus = EventBus(AsyncioQueueBackend())
    await bus.start()
    bus.start_recording()
    yield bus
    await bus.stop()


@pytest_asyncio.fixture
async def toolbox(event_bus: EventBus):
    svc = ToolboxAPIService()
    await svc.start(event_bus)
    return svc


async def seed_snapshot(event_bus: EventBus, asin: str, price_cents: int, conversion_rate: float):
    snapshot = WorldStateSnapshotEvent(
        event_id="snap-baseline",
        timestamp=datetime.now(timezone.utc),
        snapshot_id="snapshot-baseline",
        tick_number=0,
        product_count=1,
        summary_metrics={
            "products": {
                asin: {
                    "price_cents": price_cents,
                    "inventory": 100,
                    "bsr": 1000,
                    "conversion_rate": conversion_rate,
                }
            }
        },
    )
    await event_bus.publish(snapshot)
    # allow async bus to process
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_low_conversion_decreases_price(event_bus: EventBus, toolbox: ToolboxAPIService):
    asin = "B00BASL01"
    await seed_snapshot(event_bus, asin, 2000, 0.03)  # $20.00, low CR

    agent = BaselineAgentV1(agent_id="agent-1", toolbox=toolbox)
    resp = agent.decide(asin)
    assert resp is not None
    await asyncio.sleep(0.05)

    # Verify a SetPriceCommand was published with a lower price
    recorded = event_bus.get_recorded_events()
    cmd = next(e for e in recorded if e.get("event_type") == "SetPriceCommand" and e["data"]["asin"] == asin)
    # 5% decrease from 2000 -> 1900 cents
    assert cmd["data"]["new_price"] == str(Money(1900))
    # Also ensure toolbox observe reflects the starting state (price unchanged until WorldStore updates)
    obs = toolbox.observe(ObserveRequest(asin=asin))
    assert obs.found is True
    assert obs.price.cents == 2000  # cache remains at snapshot until ProductPriceUpdated


@pytest.mark.asyncio
async def test_high_conversion_increases_price(event_bus: EventBus, toolbox: ToolboxAPIService):
    asin = "B00BASL02"
    await seed_snapshot(event_bus, asin, 2000, 0.25)  # $20.00, high CR

    agent = BaselineAgentV1(agent_id="agent-1", toolbox=toolbox)
    resp = agent.decide(asin)
    assert resp is not None
    await asyncio.sleep(0.05)

    recorded = event_bus.get_recorded_events()
    cmd = next(e for e in recorded if e.get("event_type") == "SetPriceCommand" and e["data"]["asin"] == asin)
    # 5% increase from 2000 -> 2100 cents
    assert cmd["data"]["new_price"] == str(Money(2100))


@pytest.mark.asyncio
async def test_no_data_returns_none(toolbox: ToolboxAPIService):
    agent = BaselineAgentV1(agent_id="agent-1", toolbox=toolbox)
    assert agent.decide("B00UNKNOWN") is None