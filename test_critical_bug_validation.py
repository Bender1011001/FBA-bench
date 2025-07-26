#!/usr/bin/env python3
"""
Critical Bug Validation Test

This test validates the specific bugs identified in the user's analysis:
1. Double-multiplication bug in fee_engine.py total_fees method
2. Flat dimensional weight instead of tiered schedule
3. Price floor validation working correctly
4. Type compatibility issues
"""

import sys
sys.path.append('.')

from fba_bench.fee_engine import FeeEngine
from fba_bench.money import Money
from fba_bench.config_loader import load_config
from fba_bench.market_dynamics import calculate_demand
from decimal import Decimal
import warnings

def test_double_multiplication_bug():
    """Test the double-multiplication bug in total_fees method."""
    print("=== TESTING DOUBLE-MULTIPLICATION BUG ===")
    
    config = load_config()
    fee_engine = FeeEngine()
    
    # Test with storage_fee parameter - this should NOT be double-counted
    fees = fee_engine.total_fees(
        category="DEFAULT",
        price=19.99,
        size_tier="standard",
        size="small",
        storage_fee=5.0,  # This gets added as storage_money
        months_storage=1,  # This calculates long_term_storage_fee
        cubic_feet=1.0
    )
    
    print(f"Total fees: ${fees['total']:.2f}")
    print(f"Long-term storage fee: ${fees['long_term_storage_fee']:.2f}")
    
    # The bug: storage_fee parameter (5.0) gets added PLUS long_term_storage_fee calculation
    # This should be fixed to avoid double-counting
    
    # Calculate what long_term_storage_fee should be
    expected_storage = fee_engine.long_term_storage_fee(1.0, 1)
    print(f"Expected storage fee: ${expected_storage.to_float():.2f}")
    
    # If there's double-counting, total will include both storage_fee (5.0) and calculated storage
    if fees['total'] > 20.0:  # Rough threshold indicating double-counting
        print("✗ DOUBLE-MULTIPLICATION BUG DETECTED: Storage fee appears to be double-counted")
        return False
    else:
        print("✓ Storage fee calculation appears correct")
        return True

def test_dimensional_weight_tiers():
    """Test that dimensional weight uses tiered schedule from config."""
    print("\n=== TESTING DIMENSIONAL WEIGHT TIERS ===")
    
    config = load_config()
    fee_engine = FeeEngine()
    
    # Test different weights to see if tiered schedule is used
    weights_to_test = [0.5, 1.5, 3.0, 5.0]
    
    for weight in weights_to_test:
        surcharge = fee_engine.dim_weight_surcharge(True, weight)
        print(f"Weight {weight}lbs: ${surcharge.to_float():.2f}")
    
    # Check if we get different values for different weights (indicating tiers)
    surcharges = [fee_engine.dim_weight_surcharge(True, w).to_float() for w in weights_to_test]
    
    if len(set(surcharges)) == 1:
        print("✗ FLAT DIMENSIONAL WEIGHT BUG: All weights return same surcharge")
        print(f"All surcharges: {surcharges}")
        return False
    else:
        print("✓ Tiered dimensional weight schedule working")
        return True

def test_price_floor_validation():
    """Test that price floor validation prevents elasticity explosion."""
    print("\n=== TESTING PRICE FLOOR VALIDATION ===")
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Test with very low price that should trigger floor
        demand_low = calculate_demand(
            base_demand=100,
            price=Money.from_dollars(0.01),  # Very low price
            elasticity=2.0,
            seasonality_multiplier=1.0
        )
        
        # Test with normal price
        demand_normal = calculate_demand(
            base_demand=100,
            price=Money.from_dollars(19.99),
            elasticity=2.0,
            seasonality_multiplier=1.0
        )
        
        print(f"Demand at $0.01: {demand_low}")
        print(f"Demand at $19.99: {demand_normal}")
        
        # Check if warning was issued
        floor_warning_issued = any("below minimum floor" in str(warning.message) for warning in w)
        
        if floor_warning_issued:
            print("✓ Price floor validation working - warning issued for low price")
            return True
        else:
            print("✗ PRICE FLOOR BUG: No warning issued for extremely low price")
            return False

def test_type_compatibility_issues():
    """Test Money type compatibility issues."""
    print("\n=== TESTING TYPE COMPATIBILITY ISSUES ===")
    
    issues_found = []
    
    # Test 1: Float * Decimal multiplication
    try:
        trust_factor = max(0.1, 1.0)  # Returns float
        review_multiplier = Decimal('0.8') * Decimal('1.0')  # Returns Decimal
        result = trust_factor * review_multiplier  # This should fail
        print(f"✗ Float * Decimal should fail but didn't: {result}")
        issues_found.append("Float * Decimal multiplication allowed")
    except TypeError as e:
        print(f"✓ Float * Decimal properly rejected: {e}")
    
    # Test 2: Money + float issue
    try:
        inventory_value = 0.0  # float
        cost = Money.from_dollars(5.0)  # Money
        qty = 10
        inventory_value += qty * cost  # This should fail
        print(f"✗ Money + float should fail but didn't: {inventory_value}")
        issues_found.append("Money + float addition allowed")
    except TypeError as e:
        print(f"✓ Money + float properly rejected: {e}")
    
    # Test 3: Sum of Money objects starting with int
    try:
        prices = [Money.from_dollars(20.0), Money.from_dollars(18.0)]
        total = sum(prices)  # This should fail - starts with int(0)
        print(f"✗ sum(Money) should fail but didn't: {total}")
        issues_found.append("sum() of Money objects allowed")
    except TypeError as e:
        print(f"✓ sum(Money) properly rejected: {e}")
    
    # Test 4: Proper Money sum should work
    try:
        prices = [Money.from_dollars(20.0), Money.from_dollars(18.0)]
        total = Money.zero()
        for price in prices:
            total += price
        avg = total / len(prices)
        print(f"✓ Proper Money arithmetic works: total={total}, avg={avg}")
    except Exception as e:
        print(f"✗ Proper Money arithmetic failed: {e}")
        issues_found.append("Proper Money arithmetic broken")
    
    return len(issues_found) == 0

def main():
    """Run all critical bug validation tests."""
    print("CRITICAL BUG VALIDATION TEST")
    print("=" * 50)
    
    results = []
    
    results.append(("Double-multiplication bug", test_double_multiplication_bug()))
    results.append(("Dimensional weight tiers", test_dimensional_weight_tiers()))
    results.append(("Price floor validation", test_price_floor_validation()))
    results.append(("Type compatibility", test_type_compatibility_issues()))
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All critical bug tests PASSED!")
    else:
        print("\n❌ Some critical bugs still present - fixes needed!")
    
    return all_passed

if __name__ == "__main__":
    main()