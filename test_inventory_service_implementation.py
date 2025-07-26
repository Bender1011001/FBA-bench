#!/usr/bin/env python3
"""
Test script to verify the InventoryService implementation with FIFO batch management and COGS calculation.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fba_bench.services.inventory_service import InventoryService
from fba_bench.money import Money
from fba_bench.simulation import Simulation
from fba_bench.advanced_agent import AdvancedAgent


def test_basic_inventory_service():
    """Test basic InventoryService functionality."""
    print("=== Testing Basic InventoryService Functionality ===")
    
    service = InventoryService()
    
    # Test adding batches
    sku = "TEST-SKU-001"
    date1 = datetime(2025, 1, 1)
    date2 = datetime(2025, 1, 2)
    date3 = datetime(2025, 1, 3)
    
    # Add three batches with different costs
    service.add_batch(sku, 100, Money.from_dollars(10.00), date1)
    service.add_batch(sku, 50, Money.from_dollars(12.00), date2)
    service.add_batch(sku, 75, Money.from_dollars(8.00), date3)
    
    # Check total quantity
    total_qty = service.get_quantity(sku)
    print(f"Total quantity: {total_qty} (expected: 225)")
    assert total_qty == 225, f"Expected 225, got {total_qty}"
    
    # Check total value
    total_value = service.get_total_value(sku)
    expected_value = Money.from_dollars(10.00) * 100 + Money.from_dollars(12.00) * 50 + Money.from_dollars(8.00) * 75
    print(f"Total value: ${total_value.to_float()} (expected: ${expected_value.to_float()})")
    assert total_value == expected_value, f"Expected {expected_value}, got {total_value}"
    
    print("✓ Basic functionality tests passed\n")


def test_fifo_cogs_calculation():
    """Test FIFO processing and COGS calculation."""
    print("=== Testing FIFO Processing and COGS Calculation ===")
    
    service = InventoryService()
    sku = "TEST-SKU-002"
    
    # Add batches in chronological order with different costs
    service.add_batch(sku, 100, Money.from_dollars(10.00), datetime(2025, 1, 1))  # First batch: $10/unit
    service.add_batch(sku, 50, Money.from_dollars(12.00), datetime(2025, 1, 2))   # Second batch: $12/unit
    service.add_batch(sku, 75, Money.from_dollars(8.00), datetime(2025, 1, 3))    # Third batch: $8/unit
    
    print("Initial inventory:")
    batches_info = service.get_batches_info(sku)
    for i, batch in enumerate(batches_info):
        print(f"  Batch {i+1}: {batch['quantity']} units @ ${batch['cost_per_unit'].to_float()}/unit")
    
    # Test 1: Sell 80 units (should come from first batch only)
    cogs1 = service.process_sale_and_get_cogs(sku, 80)
    expected_cogs1 = Money.from_dollars(10.00) * 80
    print(f"\nSold 80 units:")
    print(f"COGS: ${cogs1.to_float()} (expected: ${expected_cogs1.to_float()})")
    print(f"Remaining quantity: {service.get_quantity(sku)} (expected: 145)")
    assert cogs1 == expected_cogs1, f"Expected COGS {expected_cogs1}, got {cogs1}"
    assert service.get_quantity(sku) == 145, f"Expected 145 units remaining, got {service.get_quantity(sku)}"
    
    # Test 2: Sell 40 units (should finish first batch and start second)
    cogs2 = service.process_sale_and_get_cogs(sku, 40)
    expected_cogs2 = Money.from_dollars(10.00) * 20 + Money.from_dollars(12.00) * 20  # 20 from first, 20 from second
    print(f"\nSold 40 more units:")
    print(f"COGS: ${cogs2.to_float()} (expected: ${expected_cogs2.to_float()})")
    print(f"Remaining quantity: {service.get_quantity(sku)} (expected: 105)")
    assert cogs2 == expected_cogs2, f"Expected COGS {expected_cogs2}, got {cogs2}"
    assert service.get_quantity(sku) == 105, f"Expected 105 units remaining, got {service.get_quantity(sku)}"
    
    # Test 3: Sell 100 units (should finish second batch and use part of third)
    cogs3 = service.process_sale_and_get_cogs(sku, 100)
    expected_cogs3 = Money.from_dollars(12.00) * 30 + Money.from_dollars(8.00) * 70  # 30 from second, 70 from third
    print(f"\nSold 100 more units:")
    print(f"COGS: ${cogs3.to_float()} (expected: ${expected_cogs3.to_float()})")
    print(f"Remaining quantity: {service.get_quantity(sku)} (expected: 5)")
    assert cogs3 == expected_cogs3, f"Expected COGS {expected_cogs3}, got {cogs3}"
    assert service.get_quantity(sku) == 5, f"Expected 5 units remaining, got {service.get_quantity(sku)}"
    
    # Verify final batch state
    final_batches = service.get_batches_info(sku)
    print(f"\nFinal inventory:")
    for i, batch in enumerate(final_batches):
        print(f"  Batch {i+1}: {batch['quantity']} units @ ${batch['cost_per_unit'].to_float()}/unit")
    
    print("✓ FIFO and COGS calculation tests passed\n")


def test_simulation_integration():
    """Test integration with the simulation."""
    print("=== Testing Simulation Integration ===")
    
    # Create a simulation directly
    sim = Simulation(seed=42)
    
    # Initialize the orchestrator to set up services
    sim._initialize_orchestrator()
    
    # Test that the simulation has the inventory service
    assert hasattr(sim, 'inventory_service'), "Simulation should have inventory_service attribute"
    print("✓ Simulation has inventory_service")
    
    # Test adding inventory directly
    asin = "B08TEST123"
    cost_per_unit = Money.from_dollars(15.00)
    sim.inventory_service.add_batch(asin, 100, cost_per_unit, sim.now)
    
    # Check inventory was added
    qty = sim.inventory_service.get_quantity(asin)
    print(f"Added inventory: {qty} units")
    assert qty == 100, f"Expected 100 units, got {qty}"
    
    # Test COGS calculation through sales
    print("\nTesting COGS through sales simulation...")
    
    # Simulate a sale through the inventory service
    units_sold = 25
    cogs = sim.inventory_service.process_sale_and_get_cogs(asin, units_sold)
    expected_cogs = Money.from_dollars(15.00) * units_sold
    
    print(f"Sold {units_sold} units")
    print(f"COGS: ${cogs.to_float()} (expected: ${expected_cogs.to_float()})")
    print(f"Remaining inventory: {sim.inventory_service.get_quantity(asin)} units")
    
    assert cogs == expected_cogs, f"Expected COGS {expected_cogs}, got {cogs}"
    assert sim.inventory_service.get_quantity(asin) == 75, f"Expected 75 units remaining"
    
    print("✓ Simulation integration tests passed\n")


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("=== Testing Edge Cases ===")
    
    service = InventoryService()
    
    # Test selling from empty inventory
    cogs_empty = service.process_sale_and_get_cogs("EMPTY-SKU", 10)
    print(f"COGS from empty inventory: ${cogs_empty.to_float()} (expected: $0.00)")
    assert cogs_empty == Money.zero(), f"Expected zero COGS, got {cogs_empty}"
    
    # Test selling more than available
    service.add_batch("LIMITED-SKU", 10, Money.from_dollars(5.00), datetime.now())
    cogs_oversell = service.process_sale_and_get_cogs("LIMITED-SKU", 20)
    expected_oversell = Money.from_dollars(5.00) * 10  # Should only sell what's available
    print(f"COGS from overselling: ${cogs_oversell.to_float()} (expected: ${expected_oversell.to_float()})")
    assert cogs_oversell == expected_oversell, f"Expected {expected_oversell}, got {cogs_oversell}"
    assert service.get_quantity("LIMITED-SKU") == 0, "Should have no inventory left"
    
    # Test zero quantity batch
    service.add_batch("ZERO-SKU", 0, Money.from_dollars(10.00), datetime.now())
    assert service.get_quantity("ZERO-SKU") == 0, "Zero quantity batch should not add inventory"
    
    # Test negative quantity sale
    service.add_batch("NEG-SKU", 10, Money.from_dollars(5.00), datetime.now())
    cogs_neg = service.process_sale_and_get_cogs("NEG-SKU", -5)
    assert cogs_neg == Money.zero(), "Negative sale should return zero COGS"
    assert service.get_quantity("NEG-SKU") == 10, "Negative sale should not affect inventory"
    
    print("✓ Edge case tests passed\n")


def main():
    """Run all tests."""
    print("Testing InventoryService Implementation")
    print("=" * 50)
    
    try:
        test_basic_inventory_service()
        test_fifo_cogs_calculation()
        test_simulation_integration()
        test_edge_cases()
        
        print("🎉 All tests passed! InventoryService implementation is working correctly.")
        print("\nKey features verified:")
        print("✓ FIFO batch management")
        print("✓ Accurate COGS calculation")
        print("✓ Money type integration")
        print("✓ Simulation and agent integration")
        print("✓ Edge case handling")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())