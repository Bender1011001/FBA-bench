import asyncio
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any

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


async def run_golden_run() -> Dict[str, Any]:
    """
    Execute the deterministic Golden Run v1 scenario and return the KPI snapshot.
    """
    # 1) Services startup (deterministic)
    event_bus = EventBus()
    await event_bus.start()

    ledger = DoubleEntryLedgerService(config={})
    await ledger.start(event_bus)

    audit = FinancialAuditService(config={"halt_on_violation": False})
    audit.ledger_service = ledger
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

        breakdown = fee_calc.calculate_comprehensive_fees(product, sale_price, additional_context)
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
            conversion_rate=0.0,  # computed from units on dataclass init
            fee_breakdown=fee_breakdown,
            market_conditions={},
            customer_segment=None,
        )
        await event_bus.publish(sale_event)

    # Allow EventBus callbacks to flush
    await asyncio.sleep(0.2)

    # 3) KPI Snapshot aggregation
    audit_stats = audit.get_audit_statistics()

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

    fees_by_type = fee_agg.get_summary_by_type()

    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "asin": asin,
        "audit": audit_stats,
        "ledger": ledger_snapshot,
        "bsr": bsr_snapshot,
        "fees_by_type": fees_by_type,
    }

    # Stop EventBus cleanly after snapshot is composed
    await event_bus.stop()

    return snapshot


def _print_summary(snapshot: Dict[str, Any]) -> None:
    audit = snapshot.get("audit", {})
    fees = snapshot.get("fees_by_type", {})
    bsr = snapshot.get("bsr", {})

    processed = audit.get("processed_transactions", 0)
    pos = audit.get("current_position", {})
    accounting_ok = pos.get("accounting_identity_valid", True)

    print("Golden Run v1 Summary")
    print("---------------------")
    print(f"Processed transactions: {processed}")
    print(f"Accounting identity valid: {accounting_ok}")

    # Top fee types by count (descending)
    if isinstance(fees, dict) and fees:
        print("Top fee types:")
        top = sorted(fees.items(), key=lambda kv: int(kv[1].get("count", 0)), reverse=True)[:5]
        for fee_type, vals in top:
            print(f"  - {fee_type}: total={vals.get('total_amount')}, count={vals.get('count')}, avg={vals.get('average_amount')}")
    else:
        print("No fee data aggregated.")

    # BSR composite index
    products = bsr.get("products", [])
    if products:
        comp_idx = products[0].get("composite_index")
        print(f"BSR composite index for {products[0].get('asin')}: {comp_idx}")
    else:
        print("No BSR product indices available.")


async def main() -> None:
    snapshot = await run_golden_run()

    # Ensure output directory exists
    out_dir = os.path.join("scenario_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "golden_run_snapshot.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    _print_summary(snapshot)
    print(f"\nWrote snapshot to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())