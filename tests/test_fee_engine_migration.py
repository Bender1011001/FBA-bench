"""Tests for fee engine migration to Money type."""
import pytest
from fba_bench.fee_engine import FeeEngine
from fba_bench.money import Money
from fba_bench import config


def test_fee_engine_returns_money_types():
    """Test that fee engine methods return Money types."""
    fee_engine = FeeEngine()
    
    # Test referral fee
    referral = fee_engine.referral_fee("DEFAULT", Money.from_dollars("19.99"))
    assert isinstance(referral, Money)
    assert referral == Money.from_dollars("2.998")  # 15% of $19.99, rounded
    
    # Test FBA fulfillment fee
    fba_fee = fee_engine.fba_fulfillment_fee("standard", "small")
    assert isinstance(fba_fee, Money)
    assert fba_fee == Money.from_dollars("3.22")
    
    # Test fuel surcharge
    fuel = fee_engine.fuel_surcharge(fba_fee)
    assert isinstance(fuel, Money)
    expected_fuel = Money.from_dollars("0.0644")  # 2% of $3.22
    assert fuel == expected_fuel
    
    # Test holiday surcharge
    holiday = fee_engine.holiday_surcharge(True)
    assert isinstance(holiday, Money)
    assert holiday == Money.from_dollars("0.50")
    
    holiday_none = fee_engine.holiday_surcharge(False)
    assert isinstance(holiday_none, Money)
    assert holiday_none == Money.zero()


def test_fee_engine_backward_compatibility():
    """Test that fee engine accepts float inputs during transition."""
    original_strict = config.MONEY_STRICT
    config.MONEY_STRICT = False
    
    try:
        fee_engine = FeeEngine()
        
        # Should accept float inputs and return Money
        referral = fee_engine.referral_fee("DEFAULT", 19.99)
        assert isinstance(referral, Money)
        
        # Should accept float fulfillment fee for fuel surcharge
        fuel = fee_engine.fuel_surcharge(3.22)
        assert isinstance(fuel, Money)
        
    finally:
        config.MONEY_STRICT = original_strict


def test_fee_engine_strict_mode():
    """Test that fee engine rejects floats in strict mode."""
    original_strict = config.MONEY_STRICT
    config.MONEY_STRICT = True
    
    try:
        fee_engine = FeeEngine()
        
        # Should reject float inputs in strict mode
        with pytest.raises(TypeError, match="Float prices not allowed when MONEY_STRICT=True"):
            fee_engine.referral_fee("DEFAULT", 19.99)
        
        with pytest.raises(TypeError, match="Float fulfillment_fee not allowed when MONEY_STRICT=True"):
            fee_engine.fuel_surcharge(3.22)
            
    finally:
        config.MONEY_STRICT = original_strict


def test_total_fees_money_integration():
    """Test that total_fees works with Money types and exact arithmetic."""
    fee_engine = FeeEngine()
    
    # Test with Money inputs
    price = Money.from_dollars("19.99")
    storage_fee = Money.from_dollars("0.78")
    penalty_fee = Money.from_dollars("5.00")
    
    fees = fee_engine.total_fees(
        category="DEFAULT",
        price=price,
        size_tier="standard",
        size="small",
        storage_fee=storage_fee,
        penalty_fee=penalty_fee
    )
    
    # In transition mode, should return floats for backward compatibility
    if not config.MONEY_STRICT:
        assert isinstance(fees["total"], float)
        assert isinstance(fees["referral_fee"], float)
    
    # Verify fee closure: total should equal sum of components
    component_sum = (
        fees["referral_fee"] + fees["fba_fulfillment_fee"] + fees["fuel_surcharge"] +
        fees["holiday_surcharge"] + fees["dim_weight_surcharge"] + fees["long_term_storage_fee"] +
        fees["removal_fee"] + fees["return_processing_fee"] + fees["aged_inventory_surcharge"] +
        fees["low_inventory_level_fee"] + fees["storage_utilization_surcharge"] +
        fees["unplanned_service_fee"] + fees["penalty_fee"] + fees["ancillary_fee"]
    )
    
    # Allow small rounding difference in transition mode
    if config.MONEY_STRICT:
        assert fees["total"] == component_sum
    else:
        assert abs(fees["total"] - component_sum) < 0.01


def test_total_fees_strict_mode():
    """Test total_fees in strict mode returns Money types."""
    original_strict = config.MONEY_STRICT
    config.MONEY_STRICT = True
    
    try:
        fee_engine = FeeEngine()
        
        price = Money.from_dollars("19.99")
        fees = fee_engine.total_fees(
            category="DEFAULT",
            price=price,
            size_tier="standard",
            size="small"
        )
        
        # Should return Money types in strict mode
        assert isinstance(fees["total"], Money)
        assert isinstance(fees["referral_fee"], Money)
        assert isinstance(fees["fba_fulfillment_fee"], Money)
        
        # Verify exact fee closure
        component_sum = (
            fees["referral_fee"] + fees["fba_fulfillment_fee"] + fees["fuel_surcharge"] +
            fees["holiday_surcharge"] + fees["dim_weight_surcharge"] + fees["long_term_storage_fee"] +
            fees["removal_fee"] + fees["return_processing_fee"] + fees["aged_inventory_surcharge"] +
            fees["low_inventory_level_fee"] + fees["storage_utilization_surcharge"] +
            fees["unplanned_service_fee"] + fees["penalty_fee"] + fees["ancillary_fee"]
        )
        
        assert fees["total"] == component_sum  # Exact equality with Money types
        
    finally:
        config.MONEY_STRICT = original_strict


def test_fee_precision_with_money():
    """Test that Money types eliminate precision issues."""
    fee_engine = FeeEngine()
    
    # Test with amounts that cause floating point precision issues
    price1 = Money.from_dollars("0.1")
    price2 = Money.from_dollars("0.2")
    
    fee1 = fee_engine.referral_fee("DEFAULT", price1)
    fee2 = fee_engine.referral_fee("DEFAULT", price2)
    
    # Should be exactly 0.30 minimum fee, not 0.30000000000000004
    total_fee = fee1 + fee2
    expected = Money.from_dollars("0.60")  # 2 * $0.30 minimum
    assert total_fee == expected


def test_complex_fee_calculation():
    """Test complex fee calculation with multiple components."""
    fee_engine = FeeEngine()
    
    fees = fee_engine.total_fees(
        category="Jewelry",
        price=Money.from_dollars("300.00"),  # Tests tiered referral fee
        size_tier="standard",
        size="small",
        is_holiday_season=True,
        dim_weight_applies=True,
        cubic_feet=2.0,
        months_storage=1,
        aged_days=200,
        aged_cubic_feet=1.0,
        low_inventory_units=5,
        trailing_days_supply=20.0,
        storage_fee=Money.from_dollars("1.50"),
        weeks_supply=25.0,
        unplanned_units=2
    )
    
    # Verify all fee components are calculated
    assert fees["referral_fee"] > 0  # Should be $52.50 for Jewelry at $300
    assert fees["fba_fulfillment_fee"] > 0  # $3.22
    assert fees["fuel_surcharge"] > 0  # 2% of fulfillment
    assert fees["holiday_surcharge"] > 0  # $0.50
    assert fees["dim_weight_surcharge"] > 0  # $1.25
    assert fees["long_term_storage_fee"] > 0  # Storage fee
    assert fees["aged_inventory_surcharge"] > 0  # Aged inventory
    assert fees["low_inventory_level_fee"] > 0  # Low inventory
    assert fees["storage_utilization_surcharge"] > 0  # Over 22 weeks
    assert fees["unplanned_service_fee"] > 0  # Unplanned service
    
    # Total should be sum of all components
    if config.MONEY_STRICT:
        component_sum = sum(fees[key] for key in fees if key != "total")
        assert fees["total"] == component_sum