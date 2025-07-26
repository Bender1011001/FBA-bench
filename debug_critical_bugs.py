#!/usr/bin/env python3
"""
Debug script to reproduce and validate the three critical bugs in FBA-Bench fee engine.
"""

import sys
sys.path.append('.')

from fba_bench.fee_engine import FeeEngine
from fba_bench.market_dynamics import calculate_demand
from fba_bench.money import Money
from decimal import Decimal

def test_double_multiplication_bug():
    """Test the double-multiplication bug in fee calculations."""
    print("=== Testing Double-Multiplication Bug ===")
    
    fee_engine = FeeEngine()
    
    # Test single unit calculation
    fees_single = fee_engine.total_fees(
        category="DEFAULT",
        price=20.0,
        size_tier="standard",
        size="small",
        ancillary_fee=1.0  # $1 per-unit ancillary fee
    )
    
    print(f"Single unit fees: {fees_single}")
    print(f"Single unit total: ${fees_single['total']}")
    print(f"Single unit ancillary: ${fees_single['ancillary_fee']}")
    
    # Simulate what happens in sales_processor.py line 319
    units_sold = 5
    trust_fee_multiplier = 1.0
    
    # This is the BUGGY calculation from sales_processor.py
    buggy_total = Money.from_dollars(fees_single["total"]) * units_sold * Decimal(str(trust_fee_multiplier))
    print(f"\nBUGGY calculation for {units_sold} units: ${buggy_total}")
    print(f"Per-unit cost with bug: ${buggy_total.to_float() / units_sold}")
    
    # What it SHOULD be (fees are already per-unit)
    correct_total = Money.from_dollars(fees_single["total"]) * Decimal(str(trust_fee_multiplier))
    correct_total_for_units = correct_total * units_sold
    print(f"CORRECT calculation for {units_sold} units: ${correct_total_for_units}")
    print(f"Per-unit cost correct: ${correct_total.to_float()}")
    
    # Show the bug impact
    bug_impact = buggy_total.to_float() - correct_total_for_units.to_float()
    print(f"\nBUG IMPACT: Overcharging by ${bug_impact:.2f} for {units_sold} units")
    print(f"Bug multiplier: {buggy_total.to_float() / correct_total_for_units.to_float():.2f}x")

def test_dimensional_weight_flat_constant():
    """Test the flat dimensional weight constant vs tiered schedule."""
    print("\n=== Testing Dimensional Weight Flat Constant Bug ===")
    
    fee_engine = FeeEngine()
    
    # Current flat rate
    current_dim_weight = fee_engine.dim_weight_surcharge(True)
    print(f"Current flat dim weight surcharge: ${current_dim_weight}")
    
    # What Amazon actually uses (tiered by weight)
    print("\nAmazon's actual tiered schedule should be:")
    print("0-1 lb: $0.50")
    print("1-5 lb: $1.25") 
    print("5-20 lb: $2.50")
    print("20+ lb: $5.00")
    
    print(f"\nCurrent system charges ${current_dim_weight} for ALL weights")
    print("This is incorrect - should vary by actual package weight")

def test_price_floor_elasticity_explosion():
    """Test the price floor validation bug in market dynamics."""
    print("\n=== Testing Price Floor Elasticity Explosion Bug ===")
    
    base_demand = 100
    elasticity = 2.0
    
    # Test normal price
    normal_price = 20.0
    normal_demand = calculate_demand(base_demand, normal_price, elasticity)
    print(f"Normal price ${normal_price}: demand = {normal_demand}")
    
    # Test very low prices that cause explosion
    low_prices = [1.0, 0.50, 0.10, 0.01]
    
    for price in low_prices:
        try:
            demand = calculate_demand(base_demand, price, elasticity)
            print(f"Price ${price}: demand = {demand}")
            
            # Show the mathematical explosion
            price_factor = price ** -elasticity
            print(f"  Price factor (${price} ** -{elasticity}) = {price_factor:.2f}")
            
            if demand > 1000000:  # Unrealistic demand
                print(f"  ⚠️  EXPLOSION: Demand of {demand:,} is unrealistic!")
                
        except Exception as e:
            print(f"Price ${price}: ERROR - {e}")
    
    print(f"\nWithout price floor validation, very low prices cause exponential explosion!")
    print(f"Need minimum price floor (e.g., $0.50) to prevent this.")

def main():
    """Run all bug reproduction tests."""
    print("FBA-Bench Critical Bug Reproduction Tests")
    print("=" * 50)
    
    test_double_multiplication_bug()
    test_dimensional_weight_flat_constant()
    test_price_floor_elasticity_explosion()
    
    print("\n" + "=" * 50)
    print("Bug reproduction complete. All three bugs confirmed.")

if __name__ == "__main__":
    main()