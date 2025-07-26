#!/usr/bin/env python3
"""
Detailed test to expose the double-multiplication bug more clearly.
"""

import sys
sys.path.append('.')

from fba_bench.fee_engine import FeeEngine
from fba_bench.money import Money
from decimal import Decimal

def test_double_multiplication_with_ancillary():
    """Test the double-multiplication bug specifically with ancillary fees."""
    print("=== Detailed Double-Multiplication Bug Test ===")
    
    fee_engine = FeeEngine()
    
    # Test with a clear ancillary fee that should be per-unit
    ancillary_fee_per_unit = 2.50  # $2.50 per unit
    units_sold = 3
    
    # Get fees for 1 unit (this should return per-unit fees)
    fees = fee_engine.total_fees(
        category="DEFAULT",
        price=15.0,
        size_tier="standard", 
        size="small",
        ancillary_fee=ancillary_fee_per_unit  # This is per-unit
    )
    
    print(f"Fee engine returns (per-unit): ${fees['total']}")
    print(f"Ancillary fee in result: ${fees['ancillary_fee']}")
    
    # Current BUGGY implementation in sales_processor.py line 319:
    # total_fees = (Money.from_dollars(fees["total"]) * units_sold * trust_fee_multiplier) + selling_plan_fee
    trust_fee_multiplier = 1.0
    buggy_total = Money.from_dollars(fees["total"]) * units_sold * Decimal(str(trust_fee_multiplier))
    
    print(f"\nBUGGY calculation for {units_sold} units:")
    print(f"  fees['total'] * {units_sold} = ${fees['total']} * {units_sold} = ${buggy_total}")
    print(f"  Per-unit cost: ${buggy_total.to_float() / units_sold:.2f}")
    
    # CORRECT implementation should be:
    # The fee_engine.total_fees() already returns per-unit fees
    # So we should multiply by units_sold only ONCE, not twice
    correct_per_unit = Money.from_dollars(fees["total"]) * Decimal(str(trust_fee_multiplier))
    correct_total = correct_per_unit * units_sold
    
    print(f"\nCORRECT calculation for {units_sold} units:")
    print(f"  Per-unit fee: ${correct_per_unit}")
    print(f"  Total for {units_sold} units: ${correct_total}")
    
    # Show the problem
    if abs(buggy_total.to_float() - correct_total.to_float()) < 0.01:
        print(f"\n⚠️  Bug not visible with current test case")
        print(f"   This suggests the fee_engine.total_fees() might already be handling multiplication correctly")
        print(f"   OR the bug is in a different part of the calculation")
    else:
        overcharge = buggy_total.to_float() - correct_total.to_float()
        print(f"\n🐛 BUG CONFIRMED: Overcharging by ${overcharge:.2f}")

def test_fee_engine_behavior():
    """Test whether fee_engine.total_fees() is per-unit or total."""
    print("\n=== Testing Fee Engine Behavior ===")
    
    fee_engine = FeeEngine()
    
    # Test with different quantities to see if fee_engine multiplies internally
    base_price = 10.0
    ancillary_per_unit = 1.0
    
    # The fee_engine.total_fees() method doesn't take a quantity parameter
    # This suggests it returns per-unit fees
    fees = fee_engine.total_fees(
        category="DEFAULT",
        price=base_price,
        size_tier="standard",
        size="small", 
        ancillary_fee=ancillary_per_unit
    )
    
    print(f"Fee engine total_fees() method signature analysis:")
    print(f"  - No 'quantity' or 'units_sold' parameter")
    print(f"  - Returns: ${fees['total']} (this should be per-unit)")
    print(f"  - Ancillary fee: ${fees['ancillary_fee']} (should equal input ${ancillary_per_unit})")
    
    if abs(fees['ancillary_fee'] - ancillary_per_unit) < 0.01:
        print(f"  ✓ Ancillary fee matches input - confirms per-unit calculation")
    else:
        print(f"  ✗ Ancillary fee doesn't match - unexpected behavior")

if __name__ == "__main__":
    test_double_multiplication_with_ancillary()
    test_fee_engine_behavior()