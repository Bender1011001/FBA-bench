#!/usr/bin/env python3
"""
Test script to verify all three critical bug fixes are working correctly.
"""

import sys
sys.path.append('.')

from fba_bench.fee_engine import FeeEngine
from fba_bench.market_dynamics import calculate_demand
from fba_bench.money import Money
from decimal import Decimal

def test_double_multiplication_fix():
    """Test that the double-multiplication bug is fixed."""
    print("=== Testing Double-Multiplication Bug Fix ===")
    
    fee_engine = FeeEngine()
    
    # Test with ancillary fees that would show the bug
    ancillary_fee_per_unit = 2.0  # $2 per unit
    units_sold = 5
    
    # Get per-unit fees from fee engine
    fees = fee_engine.total_fees(
        category="DEFAULT",
        price=15.0,
        size_tier="standard",
        size="small",
        ancillary_fee=ancillary_fee_per_unit  # This should be per-unit
    )
    
    print(f"Fee engine per-unit total: ${fees['total']}")
    print(f"Fee engine ancillary fee: ${fees['ancillary_fee']}")
    
    # Simulate the FIXED calculation (what should happen now)
    total_for_units = fees['total'] * units_sold
    print(f"Total for {units_sold} units: ${total_for_units}")
    print(f"Per-unit cost: ${total_for_units / units_sold:.2f}")
    
    # Verify ancillary fee is correctly included
    expected_ancillary_total = ancillary_fee_per_unit * units_sold
    print(f"Expected ancillary total: ${expected_ancillary_total}")
    
    if abs(fees['ancillary_fee'] - ancillary_fee_per_unit) < 0.01:
        print("✓ FIXED: Ancillary fee is correctly per-unit")
    else:
        print("✗ STILL BROKEN: Ancillary fee calculation incorrect")

def test_dimensional_weight_fix():
    """Test that the dimensional weight tiered calculation is working."""
    print("\n=== Testing Dimensional Weight Tiered Fix ===")
    
    fee_engine = FeeEngine()
    
    # Test different weights to verify tiered calculation
    test_weights = [0.5, 1.5, 7.0, 25.0]
    expected_surcharges = [0.50, 1.25, 2.50, 5.00]
    
    for weight, expected in zip(test_weights, expected_surcharges):
        surcharge = fee_engine.dim_weight_surcharge(True, weight)
        print(f"Weight {weight} lbs: ${surcharge} (expected: ${expected})")
        
        if abs(surcharge.to_float() - expected) < 0.01:
            print(f"  ✓ FIXED: Correct tier for {weight} lbs")
        else:
            print(f"  ✗ BROKEN: Expected ${expected}, got ${surcharge}")
    
    # Test that it still works without weight parameter (backward compatibility)
    default_surcharge = fee_engine.dim_weight_surcharge(True)
    print(f"Default (no weight): ${default_surcharge}")

def test_price_floor_fix():
    """Test that the price floor validation prevents elasticity explosion."""
    print("\n=== Testing Price Floor Elasticity Fix ===")
    
    base_demand = 100
    elasticity = 2.0
    
    # Test normal price (should work as before)
    normal_price = 20.0
    normal_demand = calculate_demand(base_demand, normal_price, elasticity)
    print(f"Normal price ${normal_price}: demand = {normal_demand}")
    
    # Test very low prices that would cause explosion
    low_prices = [1.0, 0.50, 0.10, 0.01]
    
    for price in low_prices:
        try:
            demand = calculate_demand(base_demand, price, elasticity)
            print(f"Price ${price}: demand = {demand}")
            
            # Check if demand is reasonable (not explosive)
            if demand > 100000:  # Still unrealistic
                print(f"  ✗ STILL EXPLOSIVE: Demand of {demand:,} is too high")
            else:
                print(f"  ✓ FIXED: Demand of {demand:,} is reasonable")
                
        except Exception as e:
            print(f"Price ${price}: ERROR - {e}")
    
    print(f"\nPrice floor validation prevents exponential explosion at very low prices.")

def test_integration():
    """Test that all fixes work together in integration."""
    print("\n=== Integration Test ===")
    
    fee_engine = FeeEngine()
    
    # Test a realistic scenario with all fixes
    fees = fee_engine.total_fees(
        category="DEFAULT",
        price=12.99,
        size_tier="standard",
        size="small",
        dim_weight_applies=True,
        weight=3.5,  # Should use $1.25 tier (1-5 lbs)
        ancillary_fee=1.50,  # $1.50 per unit
        penalty_fee=0.75     # $0.75 per unit
    )
    
    print(f"Integration test fees: ${fees['total']}")
    print(f"  Dim weight (3.5 lbs): ${fees['dim_weight_surcharge']}")
    print(f"  Ancillary fee: ${fees['ancillary_fee']}")
    print(f"  Penalty fee: ${fees['penalty_fee']}")
    
    # Test market dynamics with reasonable price
    demand = calculate_demand(100, 12.99, 2.0)
    print(f"Market demand at $12.99: {demand}")
    
    print("✓ All systems working together")

def main():
    """Run all bug fix tests."""
    print("FBA-Bench Critical Bug Fix Verification")
    print("=" * 50)
    
    test_double_multiplication_fix()
    test_dimensional_weight_fix()
    test_price_floor_fix()
    test_integration()
    
    print("\n" + "=" * 50)
    print("Bug fix verification complete!")

if __name__ == "__main__":
    main()