import math
import pytest
from decimal import Decimal, ROUND_UP, ROUND_HALF_UP

from services.fee_calculation_service import FeeCalculationService, FeeType
from money import Money


class ProductStub:
    def __init__(
        self,
        product_id: str = "P-001",
        category: str = "unknown",
        weight_oz: int = 16,
        dimensions_inches=None,
        cost_basis: Money = None,
    ):
        self.product_id = product_id
        self.category = category
        self.weight_oz = weight_oz
        self.dimensions_inches = dimensions_inches if dimensions_inches is not None else [12, 8, 1]
        self.cost_basis = cost_basis if cost_basis is not None else Money.zero()


def _find_fee(breakdown, fee_type: FeeType):
    return next((f for f in breakdown.individual_fees if f.fee_type == fee_type), None)


def _sum_money(items):
    return sum(items, Money.zero())


def test_money_math_referral_minimum_enforced_decimal_safe():
    service = FeeCalculationService(config={})
    sale_price = Money.from_dollars(2.00)
    product = ProductStub(category="unknown")  # uses referral_base_rate 0.15

    breakdown = service.calculate_comprehensive_fees(product, sale_price, {})

    referral_fee = _find_fee(breakdown, FeeType.REFERRAL)
    assert referral_fee is not None
    # 2.00 * 0.15 = 0.30, enforced minimum $0.30
    assert referral_fee.calculated_amount.cents == 30


def test_fee_percentage_and_profit_margin_percent_do_not_divide_money():
    service = FeeCalculationService(config={})
    sale_price = Money.from_dollars(10.00)
    product = ProductStub(cost_basis=Money.from_dollars(5.00))

    # Should compute without Money/Money division errors
    estimate = service.estimate_fees_for_price_point(product, sale_price, {})
    assert isinstance(estimate["fee_percentage"], float)

    breakdown = service.calculate_comprehensive_fees(product, sale_price, {})
    assert isinstance(breakdown.profit_margin_percent, float)
    # Sanity: profit margin percentage matches cents-based computation
    profit = breakdown.net_proceeds - product.cost_basis
    expected_pct = (profit.cents / product.cost_basis.cents) * 100.0 if product.cost_basis.cents else 0.0
    assert math.isclose(breakdown.profit_margin_percent, expected_pct, rel_tol=1e-9, abs_tol=1e-9)


def test_billable_weight_large_standard_uses_dim_weight_and_additional_pounds():
    # Dimensions at the max of large_standard to force dim weight >> actual
    dims = [18, 14, 8]  # in inches
    weight_oz = 40  # 2.5 lb actual
    product = ProductStub(category="unknown", weight_oz=weight_oz, dimensions_inches=dims)
    service = FeeCalculationService(config={"dimensional_weight_divisor": 139})

    breakdown = service.calculate_comprehensive_fees(product, Money.from_dollars(20.00), {})
    fba_fee = _find_fee(breakdown, FeeType.FBA)
    assert fba_fee is not None

    # Expected billable weight
    L, W, H = [Decimal(x) for x in dims]
    dim_weight = (L * W * H) / Decimal(139)
    # ceil to nearest 0.1
    billable = dim_weight.quantize(Decimal("0.1"), rounding=ROUND_UP)
    # If actual > dim, use actual; here dim should be greater
    actual_lb = (Decimal(weight_oz) / Decimal(16)).quantize(Decimal("0.1"))
    billable = billable if billable > actual_lb else actual_lb

    # Additional weight over 1.0 lb
    add_weight = billable - Decimal("1.0")
    if add_weight < 0:
        add_weight = Decimal("0")

    base_fee_cents = 409  # default 'fba_large_standard_1lb'
    add_rate_cents = 42   # default 'fba_large_standard_additional_lb'
    add_fee_cents = int((add_weight * Decimal(add_rate_cents)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    expected_fba_cents = base_fee_cents + add_fee_cents

    assert fba_fee.calculated_amount.cents == expected_fba_cents


def test_storage_fee_decimal_daily_prorating_uses_30_days_and_returns_money():
    # Make rates same in peak and non-peak to avoid system month dependency
    storage_rates = {
        "standard_jan_sep": Money.from_dollars(0.90),
        "standard_oct_dec": Money.from_dollars(0.90),
        "oversize_jan_sep": Money.from_dollars(0.90),
        "oversize_oct_dec": Money.from_dollars(0.90),
    }
    service = FeeCalculationService(config={"storage_rates": storage_rates})
    # 1 cubic foot box: 12x12x12 inches
    product = ProductStub(dimensions_inches=[12, 12, 12])
    sale_price = Money.from_dollars(10.00)

    breakdown = service.calculate_comprehensive_fees(product, sale_price, {"storage_duration_days": 15})
    storage_fee = _find_fee(breakdown, FeeType.STORAGE)
    assert storage_fee is not None
    # 0.90 / 30 = 0.03 per day (HALF_UP at daily step), 15 days = 0.45
    assert storage_fee.calculated_amount.cents == 45


def test_surcharges_peak_remote_hazmat():
    surcharges = {
        "peak_season_pct_on_fba": Decimal("0.10"),
        "fuel_pct_on_fba": Decimal("0.05"),
        "remote_area_flat": Money.from_dollars(1.00),
        "hazardous_flat": Money.from_dollars(2.00),
    }
    service = FeeCalculationService(config={"surcharges": surcharges})
    product = ProductStub()
    sale_price = Money.from_dollars(10.00)

    breakdown = service.calculate_comprehensive_fees(
        product,
        sale_price,
        {"peak_season": True, "is_remote_area": True, "is_hazmat": True},
    )
    sur_fees = [f for f in breakdown.individual_fees if f.fee_type == FeeType.SURCHARGE]
    assert len(sur_fees) == 4  # peak pct, fuel pct, remote flat, hazmat flat

    # Base FBA for small_standard is $3.22 by default
    fba_fee = _find_fee(breakdown, FeeType.FBA)
    assert fba_fee is not None
    fba_cents = fba_fee.calculated_amount.cents

    peak = int((Decimal(fba_cents) * Decimal("0.10")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    fuel = int((Decimal(fba_cents) * Decimal("0.05")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    total_expected = Money(peak + fuel + 100 + 200)

    total_surcharges = _sum_money([f.calculated_amount for f in sur_fees])
    assert total_surcharges.cents == total_expected.cents


def test_penalties_ltsf_and_low_price_penalty():
    penalties = {
        "ltsf_monthly_per_cu_ft": Money.from_dollars(1.00),
        "low_price_threshold": Money.from_dollars(10.00),
        "low_price_fee_flat": Money.from_dollars(1.23),
    }
    service = FeeCalculationService(config={"penalties": penalties})
    # 1 cubic foot box
    product = ProductStub(dimensions_inches=[12, 12, 12])
    sale_price = Money.from_dollars(9.99)

    breakdown = service.calculate_comprehensive_fees(
        product,
        sale_price,
        {"months_in_storage": 12},
    )
    pen_fees = [f for f in breakdown.individual_fees if f.fee_type == FeeType.PENALTY]
    # Expect both LTSF and low-price penalties
    assert len(pen_fees) >= 2

    # Validate LTSF = $1.00 * 12 months * 1 cu ft = $12.00
    ltsf = next((f for f in pen_fees if "Long-term storage penalty" in f.description), None)
    assert ltsf is not None
    assert ltsf.calculated_amount.cents == 1200

    # Validate low-price penalty flat
    low_price = next((f for f in pen_fees if "Low-price penalty" in f.description), None)
    assert low_price is not None
    assert low_price.calculated_amount.cents == 123


def test_summary_by_type_totals_and_averages():
    surcharges = {
        "peak_season_pct_on_fba": Decimal("0.10"),
    }
    service = FeeCalculationService(config={"surcharges": surcharges})

    prod1 = ProductStub()
    prod2 = ProductStub(product_id="P-002")

    sale_price = Money.from_dollars(10.00)
    # Force peak season via context override for determinism
    b1 = service.calculate_comprehensive_fees(prod1, sale_price, {"peak_season": True})
    b2 = service.calculate_comprehensive_fees(prod2, sale_price, {"peak_season": True})

    summary = service.get_fee_summary_by_type([b1, b2])

    assert "surcharge" in summary
    total = summary["surcharge"]["total_amount"]
    count = summary["surcharge"]["count"]
    avg = summary["surcharge"]["average_amount"]

    # Compute expected from individual breakdowns
    def sur_total(bd):
        return _sum_money([f.calculated_amount for f in bd.individual_fees if f.fee_type == FeeType.SURCHARGE])

    expected_total = sur_total(b1) + sur_total(b2)
    assert total == expected_total
    assert count == 2
    assert avg.cents == expected_total.cents // count