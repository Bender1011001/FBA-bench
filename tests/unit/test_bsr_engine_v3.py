import asyncio
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

import pytest

from event_bus import EventBus
from events import SaleOccurred, CompetitorPricesUpdated, CompetitorState
from money import Money
from services.bsr_engine_v3 import BsrEngineV3Service


Q6 = Decimal("0.000001")


def dq(x: Decimal) -> Decimal:
    """Quantize to 6 decimals for deterministic assertions."""
    return x.quantize(Q6, rounding=ROUND_HALF_UP)


def make_sale_event(asin: str, units_sold: int, units_demanded: int, price_dollars: Decimal = Decimal("10.00")) -> SaleOccurred:
    unit_price = Money.from_dollars(price_dollars)
    total_revenue = unit_price * units_sold
    total_fees = Money.zero()
    cost_basis = Money.zero()
    total_profit = total_revenue - total_fees - cost_basis
    return SaleOccurred(
        event_id=f"sale_{asin}_{units_sold}_{units_demanded}",
        timestamp=datetime.now(),
        asin=asin,
        units_sold=units_sold,
        units_demanded=units_demanded,
        unit_price=unit_price,
        total_revenue=total_revenue,
        total_fees=total_fees,
        total_profit=total_profit,
        cost_basis=cost_basis,
        trust_score_at_sale=0.9,
        bsr_at_sale=1000,
        conversion_rate=0.0,  # will be computed from units if demanded > 0
        fee_breakdown={},
        market_conditions={},
        customer_segment=None,
    )


def make_competitor_update(tick: int, competitors: list[CompetitorState]) -> CompetitorPricesUpdated:
    return CompetitorPricesUpdated(
        event_id=f"comp_update_{tick}",
        timestamp=datetime.now(),
        tick_number=tick,
        competitors=competitors,
        market_summary={"competitor_count": len(competitors)},
    )


def comp_state(asin: str, price: Money, bsr: int, velocity: float) -> CompetitorState:
    return CompetitorState(asin=asin, price=price, bsr=bsr, sales_velocity=velocity)


@pytest.mark.asyncio
async def test_ema_updates_single_asin_over_3_sales():
    # Config: alpha 0.5 for both to simplify expected math
    svc = BsrEngineV3Service(config={"ema_alpha_velocity": Decimal("0.5"), "ema_alpha_conversion": Decimal("0.5")})
    bus = EventBus()
    await bus.start()
    try:
        await svc.start(bus)

        asin = "ASIN-TEST-1"
        # Three sales with constant conversion 0.5 (sold/demanded = 1/2)
        sales = [
            make_sale_event(asin, 2, 4),
            make_sale_event(asin, 4, 8),
            make_sale_event(asin, 6, 12),
        ]
        for ev in sales:
            await bus.publish(ev)

        # Allow async processing
        await asyncio.sleep(0.05)

        metrics = svc.get_product_metrics(asin)
        assert metrics["updates"] == 3

        # Expected EMA with alpha=0.5:
        # velocity: v1=2, v2=(4+2)/2=3, v3=(6+3)/2=4.5
        expected_v = dq(Decimal("4.5"))
        expected_c = dq(Decimal("0.5"))  # constant 0.5 remains 0.5 under EMA

        assert dq(metrics["ema_velocity"]) == expected_v
        assert dq(metrics["ema_conversion"]) == expected_c
    finally:
        await svc.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_market_ema_updates_via_competitor_prices_updated():
    svc = BsrEngineV3Service(config={"ema_alpha_velocity": Decimal("0.5"), "ema_alpha_conversion": Decimal("0.5")})
    bus = EventBus()
    await bus.start()
    try:
        await svc.start(bus)

        # Snapshot 1
        comps1 = [
            comp_state("C1", Money.from_dollars("20.00"), 1000, 10.0),
            comp_state("C2", Money.from_dollars("25.00"), 2000, 20.0),
        ]
        await bus.publish(make_competitor_update(1, comps1))
        await asyncio.sleep(0.02)

        # Snapshot 2
        comps2 = [
            comp_state("C1", Money.from_dollars("21.00"), 1100, 30.0),
            comp_state("C2", Money.from_dollars("24.00"), 2100, 50.0),
        ]
        await bus.publish(make_competitor_update(2, comps2))
        await asyncio.sleep(0.05)

        market = svc.get_market_metrics()
        # Expected velocity EMA:
        # avg1 = (10+20)/2 = 15, m1 = 15
        # avg2 = (30+50)/2 = 40, m2 = 0.5*40 + 0.5*15 = 27.5
        expected_m_v = dq(Decimal("27.5"))

        # Expected conversion proxy EMA:
        # conv(v)=v/(v+1)
        def conv_avg(vs: list[Decimal]) -> Decimal:
            terms = [v / (v + Decimal("1")) for v in vs]
            return sum(terms, Decimal("0")) / Decimal(len(terms))

        avg_c1 = conv_avg([Decimal("10"), Decimal("20")])
        avg_c2 = conv_avg([Decimal("30"), Decimal("50")])
        expected_m_c = dq((Decimal("0.5") * avg_c2) + (Decimal("0.5") * avg_c1))

        assert dq(market["market_ema_velocity"]) == expected_m_v
        assert dq(market["market_ema_conversion"]) == expected_m_c
        assert market["competitor_count"] == 2
        assert market["market_ema_conversion"] >= Decimal("0")
    finally:
        await svc.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_relative_indices_after_min_samples():
    cfg = {
        "ema_alpha_velocity": Decimal("0.5"),
        "ema_alpha_conversion": Decimal("0.5"),
        "min_samples_to_index": 3,
        "index_floor": Decimal("0.01"),
        "index_ceiling": Decimal("100.0"),
        "smoothing_eps": Decimal("1e-9"),
    }
    svc = BsrEngineV3Service(config=cfg)
    bus = EventBus()
    await bus.start()
    try:
        await svc.start(bus)

        # One competitor snapshot to initialize market EMAs
        comps = [
            comp_state("C1", Money.from_dollars("20.00"), 1000, 10.0),
            comp_state("C2", Money.from_dollars("25.00"), 2000, 20.0),
        ]
        await bus.publish(make_competitor_update(1, comps))
        await asyncio.sleep(0.02)

        asin = "ASIN-IX"
        # Three sales to reach min_samples
        for ev in [make_sale_event(asin, 2, 4), make_sale_event(asin, 4, 8), make_sale_event(asin, 6, 12)]:
            await bus.publish(ev)
        await asyncio.sleep(0.05)

        # Expected product EMAs
        p_v = Decimal("4.5")
        p_c = Decimal("0.5")

        # Market EMAs from comps snapshot 1 (alpha applies, prev=None => equal to averages)
        m_v = (Decimal("10") + Decimal("20")) / Decimal("2")  # 15
        m_c = ((Decimal("10") / (Decimal("11"))) + (Decimal("20") / (Decimal("21")))) / Decimal("2")

        eps = cfg["smoothing_eps"]
        floor = cfg["index_floor"]
        ceil = cfg["index_ceiling"]

        denom_v = max(m_v, floor)
        denom_c = max(m_c, floor)

        v_index = (p_v + eps) / denom_v
        c_index = (p_c + eps) / denom_c

        # clamp
        v_index = min(max(v_index, floor), ceil)
        c_index = min(max(c_index, floor), ceil)
        comp_index = (v_index * c_index) ** Decimal("0.5")

        expected = {
            "velocity_index": dq(v_index),
            "conversion_index": dq(c_index),
            "composite_index": dq(min(max(comp_index, floor), ceil)),
        }

        actual = svc.get_product_indices(asin)
        assert actual["velocity_index"] == expected["velocity_index"]
        assert actual["conversion_index"] == expected["conversion_index"]
        assert actual["composite_index"] == expected["composite_index"]
    finally:
        await svc.stop()
        await bus.stop()


@pytest.mark.asyncio
async def test_guardrails_none_and_clamping():
    # Part A: Indices are None when insufficient data or missing market EMA
    svc = BsrEngineV3Service(config={"min_samples_to_index": 3})
    bus = EventBus()
    await bus.start()
    try:
        await svc.start(bus)
        asin = "ASIN-GR"
        await bus.publish(make_sale_event(asin, 1, 2))
        await bus.publish(make_sale_event(asin, 1, 2))
        await asyncio.sleep(0.05)

        idx = svc.get_product_indices(asin)
        assert idx["velocity_index"] is None
        assert idx["conversion_index"] is None
        assert idx["composite_index"] is None
    finally:
        await svc.stop()
        await bus.stop()

    # Part B: Division guards and ceiling clamp
    cfg = {
        "min_samples_to_index": 3,
        "index_floor": Decimal("0.01"),
        "index_ceiling": Decimal("100.0"),
        "ema_alpha_velocity": Decimal("0.5"),
        "ema_alpha_conversion": Decimal("0.5"),
    }
    svc2 = BsrEngineV3Service(config=cfg)
    bus2 = EventBus()
    await bus2.start()
    try:
        await svc2.start(bus2)
        # Market snapshot with near-zero velocity -> market EMAs effectively 0 (will be floored by index_floor)
        comps_zero = [
            comp_state("Z1", Money.from_dollars("10.00"), 100, 0.0),
            comp_state("Z2", Money.from_dollars("10.00"), 100, 0.0),
        ]
        await bus2.publish(make_competitor_update(1, comps_zero))
        await asyncio.sleep(0.02)

        asin = "ASIN-GR2"
        for ev in [make_sale_event(asin, 2, 4), make_sale_event(asin, 4, 8), make_sale_event(asin, 6, 12)]:
            await bus2.publish(ev)
        await asyncio.sleep(0.05)

        idx2 = svc2.get_product_indices(asin)
        assert idx2["velocity_index"] <= cfg["index_ceiling"]
        # Given floor=0.01 and p_v≈4.5, velocity_index should clamp to 100.0
        assert dq(idx2["velocity_index"]) == dq(cfg["index_ceiling"])
        assert idx2["conversion_index"] is not None
        assert idx2["conversion_index"] <= cfg["index_ceiling"]
        assert idx2["composite_index"] is not None
        assert idx2["composite_index"] <= cfg["index_ceiling"]
    finally:
        await svc2.stop()
        await bus2.stop()


@pytest.mark.asyncio
async def test_snapshot_structure():
    svc = BsrEngineV3Service()
    bus = EventBus()
    await bus.start()
    try:
        await svc.start(bus)
        # Initialize market and product
        comps = [
            comp_state("C1", Money.from_dollars("20.00"), 1000, 5.0),
            comp_state("C2", Money.from_dollars("25.00"), 2000, 7.0),
        ]
        await bus.publish(make_competitor_update(1, comps))
        asin = "ASIN-SNAP"
        await bus.publish(make_sale_event(asin, 2, 4))
        await bus.publish(make_sale_event(asin, 4, 8))
        await bus.publish(make_sale_event(asin, 6, 12))
        await asyncio.sleep(0.05)

        snap = svc.get_snapshot()
        assert "market" in snap
        assert "products" in snap
        assert "competitors_latest_count" in snap
        assert asin in snap["products"]
        prod_entry = snap["products"][asin]
        assert "ema_velocity" in prod_entry
        assert "ema_conversion" in prod_entry
        assert "updates" in prod_entry
        assert "indices" in prod_entry
    finally:
        await svc.stop()
        await bus.stop()