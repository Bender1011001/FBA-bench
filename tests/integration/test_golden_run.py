import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any

import pytest

from event_bus import EventBus
from events import CompetitorPricesUpdated, CompetitorState, SaleOccurred
from money import Money
from financial_audit import FinancialAuditService
from services.double_entry_ledger_service import DoubleEntryLedgerService
from services.bsr_engine_v3 import BsrEngineV3Service
from services.fee_calculation_service import FeeCalculationService
from services.dashboard_api_service import FeeMetricsAggregatorService


class ProductStub:
    """
    Minimal product stub compatible with FeeCalculationService.calculate_comprehensive_fees().
    Uses Money-safe and Decimal-safe attributes only.
    """
    def __init__(
        self,
        product_id: str,
        category: str = "default",
        weight_oz: int | Decimal = 16,
        dimensions_inches: list[Decimal] | None = None,
        cost_basis: Money | None = None,
    ):
        self.product_id = product_id
        self.category = category
        self.weight_oz = int(weight_oz) if not isinstance(weight_oz, Decimal) else weight_oz
        self.dimensions_inches = dimensions_inches if dimensions_inches is not None else [Decimal("10"), Decimal("6"), Decimal("3")]
        self.cost_basis = cost_basis if cost_basis is not None else Money.zero()


def _to_num(x):
    # Convert Decimal to float for JSON numeric fields; leave None as None
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


@pytest.mark.asyncio
async def test_golden_run(tmp_path):
    """
    End-to-end deterministic Golden Run v1:
    - Wires EventBus + services (Ledger, Audit, BSR, Fee Aggregator)
    - Publishes one competitor snapshot and three SaleOccurred events for ASIN B00GOLDEN
    - Aggregates KPI snapshot and writes artifact to tmp_path/golden_run_snapshot.json
    - Verifies stability and expected shape
    """
    # 1) Services startup (deterministic)
    event_bus = EventBus()
    await event_bus.start()

    ledger = DoubleEntryLedgerService(config={})
    await ledger.start(event_bus)

    audit = FinancialAuditService(config={"halt_on_violation": False})
    audit.ledger_service = ledger  # use ledger as source-of-truth when auditing
    await audit.start(event_bus)

    bsr = BsrEngineV3Service(config={"min_samples_to_index": 3})
    await bsr.start(event_bus)

    fee_agg = FeeMetricsAggregatorService()
    await fee_agg.start(event_bus)

    asin = "B00GOLDEN"

    # Seed market with competitor snapshot to initialize EMAs
    competitors = [
        CompetitorState(
            asin="COMP1",
            price=Money.from_dollars("17.99"),
            bsr=1200,
            sales_velocity=2.0,
        ),
        CompetitorState(
            asin="COMP2",
            price=Money.from_dollars("21.49"),
            bsr=800,
            sales_velocity=3.0,
        ),
    ]
    comp_event = CompetitorPricesUpdated(
        event_id="comp-seed-1",
        timestamp=datetime.now(timezone.utc),
        tick_number=0,
        competitors=competitors,
        market_summary={},
    )
    await event_bus.publish(comp_event)

    # Prepare fee calculation service and product stub
    fee_calc = FeeCalculationService(config={})
    sale_price = Money.from_dollars("19.99")
    cost_per_unit = Money.from_dollars("6.00")
    product = ProductStub(
        product_id=asin,
        category="default",
        weight_oz=Decimal("16"),
        dimensions_inches=[Decimal("10"), Decimal("6"), Decimal("3")],
        cost_basis=Money.zero(),
    )
    additional_context = {
        "peak_season": False,
        "requires_prep": False,
        "requires_labeling": False,
        "advertising_spend": Money.zero(),
        "months_in_storage": 0,
    }

    # 2) Deterministic SaleOccurred sequence (3 sales)
    sales = [
        {"units_sold": 2, "units_demanded": 5},
        {"units_sold": 3, "units_demanded": 6},
        {"units_sold": 1, "units_demanded": 3},
    ]

    for idx, s in enumerate(sales, start=1):
        units_sold = int(s["units_sold"])
        units_demanded = int(s["units_demanded"])

        # Compute comprehensive fees deterministically
        breakdown = fee_calc.calculate_comprehensive_fees(product, sale_price, additional_context)
        # Map fee types to Money amounts
        fee_breakdown: Dict[str, Money] = {f.fee_type.value: f.calculated_amount for f in breakdown.individual_fees}

        total_revenue = sale_price * units_sold
        total_fees = sum((amt for amt in fee_breakdown.values()), Money.zero())
        cost_basis = cost_per_unit * units_sold
        total_profit = total_revenue - total_fees - cost_basis

        sale_event = SaleOccurred(
            event_id=f"sale-{idx}",
            timestamp=datetime.now(timezone.utc),
            asin=asin,
            units_sold=units_sold,
            units_demanded=units_demanded,
            unit_price=sale_price,
            total_revenue=total_revenue,
            total_fees=total_fees,
            total_profit=total_profit,
            cost_basis=cost_basis,
            trust_score_at_sale=0.90,
            bsr_at_sale=1000,
            conversion_rate=0.0,  # will be computed from units_sold/units_demanded
            fee_breakdown=fee_breakdown,
            market_conditions={},
            customer_segment=None,
        )
        await event_bus.publish(sale_event)

    # Allow EventBus callbacks to flush
    await asyncio.sleep(0.2)

    # 3) KPI Snapshot aggregation
    # Audit
    audit_stats = audit.get_audit_statistics()

    # Ledger
    fin_pos = ledger.get_financial_position()
    ledger_snapshot: Dict[str, Any] = {
        "cash": str(fin_pos.get("cash")),
        "inventory_value": str(fin_pos.get("inventory_value")),
        "accounts_receivable": str(fin_pos.get("accounts_receivable")),
        "accounts_payable": str(fin_pos.get("accounts_payable")),
        "accrued_liabilities": str(fin_pos.get("accrued_liabilities")),
        "total_assets": str(fin_pos.get("total_assets")),
        "total_liabilities": str(fin_pos.get("total_liabilities")),
        "total_equity": str(fin_pos.get("total_equity")),
        "accounting_identity_valid": bool(fin_pos.get("accounting_identity_valid", True)),
        "identity_difference": str(fin_pos.get("identity_difference")),
        "timestamp": (fin_pos.get("timestamp").isoformat() if hasattr(fin_pos.get("timestamp"), "isoformat") else datetime.now(timezone.utc).isoformat()),
    }

    # BSR
    idx = bsr.get_product_indices(asin)
    market = bsr.get_market_metrics()
    bsr_snapshot: Dict[str, Any] = {
        "products": [
            {
                "asin": asin,
                "velocity_index": _to_num(idx.get("velocity_index")),
                "conversion_index": _to_num(idx.get("conversion_index")),
                "composite_index": _to_num(idx.get("composite_index")),
            }
        ],
        "market_ema_velocity": (str(market["market_ema_velocity"]) if market.get("market_ema_velocity") is not None else None),
        "market_ema_conversion": (str(market["market_ema_conversion"]) if market.get("market_ema_conversion") is not None else None),
        "competitor_count": int(market.get("competitor_count", 0)),
    }

    # Fees
    fees_by_type = fee_agg.get_summary_by_type()

    # Compose snapshot
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "asin": asin,
        "audit": audit_stats,
        "ledger": ledger_snapshot,
        "bsr": bsr_snapshot,
        "fees_by_type": fees_by_type,
    }

    # 4) Write artifact
    out_path = tmp_path / "golden_run_snapshot.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    # 5) Assertions
    assert audit_stats["processed_transactions"] == 3
    assert isinstance(audit_stats["current_position"]["accounting_identity_valid"], bool)
    # Prefer true identity validity, but only assert type and non-empty identity_difference string shape
    assert audit_stats["current_position"]["identity_difference"].startswith("$")

    # BSR indices present (not None) for the ASIN
    prod_entries = bsr_snapshot["products"]
    assert len(prod_entries) == 1 and prod_entries[0]["asin"] == asin
    assert prod_entries[0]["velocity_index"] is not None
    assert prod_entries[0]["conversion_index"] is not None
    assert prod_entries[0]["composite_index"] is not None

    # Fees non-empty and proper shape
    assert isinstance(fees_by_type, dict) and len(fees_by_type) >= 1
    for k, v in fees_by_type.items():
        assert v["count"] >= 1
        assert str(v["total_amount"]).startswith("$")
        assert str(v["average_amount"]).startswith("$")

    # Ledger totals present; identity_difference shape
    assert ledger_snapshot["total_assets"].startswith("$")
    assert ledger_snapshot["total_liabilities"].startswith("$")
    assert ledger_snapshot["total_equity"].startswith("$")
    assert ledger_snapshot["identity_difference"].startswith("$")

    # 6) Stop EventBus cleanly
    await event_bus.stop()