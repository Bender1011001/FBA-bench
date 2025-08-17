import asyncio
from datetime import datetime, timezone

import pytest
import httpx

from event_bus import EventBus
from money import Money
from events import CompetitorState, CompetitorPricesUpdated, SaleOccurred
from financial_audit import FinancialAuditService
from services.double_entry_ledger_service import DoubleEntryLedgerService
from services.bsr_engine_v3 import BsrEngineV3Service
from services.dashboard_api_service import DashboardAPIService, FeeMetricsAggregatorService


@pytest.mark.asyncio
async def test_metrics_endpoints_end_to_end():
    # Setup EventBus and services
    event_bus = EventBus()
    await event_bus.start()

    audit = FinancialAuditService(config={"halt_on_violation": False})
    ledger = DoubleEntryLedgerService(config={})
    bsr = BsrEngineV3Service()
    fee_agg = FeeMetricsAggregatorService()

    # Wire audit to ledger for position sync
    audit.ledger_service = ledger

    # Start services (subscription wiring)
    await audit.start(event_bus)
    await ledger.start(event_bus)
    await bsr.start(event_bus)
    await fee_agg.start(event_bus)

    # Publish competitor update to seed BSR market EMA
    asin_a = "B00METRICS1"
    asin_b = "B00METRICS2"
    comp_event = CompetitorPricesUpdated(
        event_id="cmp-1",
        timestamp=datetime.now(timezone.utc),
        tick_number=1,
        competitors=[
            CompetitorState(asin=asin_a, price=Money.from_dollars(24.99), bsr=1200, sales_velocity=2.5),
            CompetitorState(asin=asin_b, price=Money.from_dollars(29.99), bsr=800, sales_velocity=3.0),
        ],
        market_summary={"note": "seed snapshot"},
    )
    await event_bus.publish(comp_event)

    # Publish a SaleOccurred with fee breakdown to seed fee aggregator and audit
    sale_asin = "B00METRICSX"
    unit_price = Money.from_dollars(25.00)
    units_sold = 2
    total_revenue = unit_price * units_sold  # $50.00
    referral_fee = Money.from_dollars(3.00)
    fba_fee = Money.from_dollars(4.00)
    fee_breakdown = {"referral": referral_fee, "fba": fba_fee}
    total_fees = referral_fee + fba_fee  # $7.00
    cost_basis = Money.from_dollars(20.00)  # $20.00
    total_profit = total_revenue - total_fees - cost_basis  # $23.00

    sale_event = SaleOccurred(
        event_id="sale-1",
        timestamp=datetime.now(timezone.utc),
        asin=sale_asin,
        units_sold=units_sold,
        units_demanded=3,
        unit_price=unit_price,
        total_revenue=total_revenue,
        total_fees=total_fees,
        total_profit=total_profit,
        cost_basis=cost_basis,
        trust_score_at_sale=0.92,
        bsr_at_sale=900,
        conversion_rate=2 / 3,
        fee_breakdown=fee_breakdown,
    )
    await event_bus.publish(sale_event)

    # Allow async handlers to process
    await asyncio.sleep(0.1)

    # Build ASGI app from extended DashboardAPIService
    dashboard = DashboardAPIService(
        event_bus=event_bus,
        audit_service=audit,
        ledger_service=ledger,
        bsr_service=bsr,
        fee_aggregator=fee_agg,
    )
    app = dashboard.build_app()

    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        # Audit endpoint
        r = await client.get("/api/metrics/audit")
        assert r.status_code == 200
        audit_json = r.json()
        for key in [
            "processed_transactions",
            "total_violations",
            "critical_violations",
            "total_revenue_audited",
            "total_fees_audited",
            "total_profit_audited",
            "current_position",
            "audit_enabled",
            "halt_on_violation",
            "tolerance_cents",
        ]:
            assert key in audit_json
        assert isinstance(audit_json["current_position"]["accounting_identity_valid"], bool)
        # Money fields are strings with "$" prefix
        assert isinstance(audit_json["total_revenue_audited"], str) and audit_json["total_revenue_audited"].startswith("$")
        assert isinstance(audit_json["total_fees_audited"], str) and audit_json["total_fees_audited"].startswith("$")
        assert isinstance(audit_json["total_profit_audited"], str) and audit_json["total_profit_audited"].startswith("$")

        # Ledger endpoint
        r = await client.get("/api/metrics/ledger")
        assert r.status_code == 200
        ledger_json = r.json()
        for key in [
            "cash",
            "inventory_value",
            "accounts_receivable",
            "accounts_payable",
            "accrued_liabilities",
            "total_assets",
            "total_liabilities",
            "total_equity",
            "accounting_identity_valid",
            "identity_difference",
            "timestamp",
        ]:
            assert key in ledger_json
        assert isinstance(ledger_json["cash"], str) and ledger_json["cash"].startswith("$")
        # timestamp is ISO
        try:
            datetime.fromisoformat(ledger_json["timestamp"].replace("Z", "+00:00"))
        except Exception:
            pytest.fail("Ledger timestamp is not ISO formatted")

        # BSR endpoint
        r = await client.get("/api/metrics/bsr")
        assert r.status_code == 200
        bsr_json = r.json()
        assert "products" in bsr_json and isinstance(bsr_json["products"], list)
        # Indices are numbers or null
        for p in bsr_json["products"]:
            for idx_key in ["velocity_index", "conversion_index", "composite_index"]:
                v = p.get(idx_key)
                assert v is None or isinstance(v, (int, float))
        # Market EMA fields (if present) serialized as strings
        if "market_ema_velocity" in bsr_json:
            assert isinstance(bsr_json["market_ema_velocity"], str)
        if "market_ema_conversion" in bsr_json:
            assert isinstance(bsr_json["market_ema_conversion"], str)
        if "competitor_count" in bsr_json:
            assert isinstance(bsr_json["competitor_count"], int)

        # Fees endpoint
        r = await client.get("/api/metrics/fees")
        assert r.status_code == 200
        fees_json = r.json()
        # Should include the fee types we published
        assert "referral" in fees_json and "fba" in fees_json
        for fee_type in ["referral", "fba"]:
            entry = fees_json[fee_type]
            assert isinstance(entry["count"], int) and entry["count"] >= 1
            assert isinstance(entry["total_amount"], str) and entry["total_amount"].startswith("$")
            assert isinstance(entry["average_amount"], str) and entry["average_amount"].startswith("$")

    # Cleanup
    await bsr.stop()
    await audit.stop()
    await event_bus.stop()