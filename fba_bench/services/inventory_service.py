"""
InventoryService for FBA-Bench: handles inventory batch management with FIFO and COGS calculation.
"""

from typing import List, Dict, Optional
from datetime import datetime
from fba_bench.money import Money
from fba_bench.inventory import InventoryBatch


class InventoryService:
    """
    Service for managing inventory batches with FIFO processing and COGS calculation.
    
    This service is the sole authority for all inventory operations, replacing the
    old InventoryManager with proper batch management and cost tracking.
    """

    def __init__(self):
        """Initialize the InventoryService with empty batch storage."""
        self._batches: Dict[str, List[InventoryBatch]] = {}

    def add_batch(self, sku: str, qty: int, cost_per_unit: Money, received_date: datetime) -> None:
        """
        Add a new batch of inventory for a given SKU.

        Args:
            sku: Stock Keeping Unit identifier.
            qty: Quantity of units to add.
            cost_per_unit: Cost per unit for this batch (Money object).
            received_date: Date and time the batch was received.
        """
        if qty <= 0:
            return
            
        # Convert Money to float for InventoryBatch compatibility
        cost_float = cost_per_unit.to_float()
        
        self._batches.setdefault(sku, []).append(
            InventoryBatch(quantity=qty, cost_per_unit=cost_float, received=received_date)
        )

    def process_sale_and_get_cogs(self, sku: str, units_sold: int) -> Money:
        """
        Remove inventory based on FIFO and return the exact COGS for the units sold.
        
        Args:
            sku: Stock Keeping Unit identifier.
            units_sold: Number of units sold to remove from inventory.
            
        Returns:
            Money: The total cost of goods sold for the units removed.
        """
        if sku not in self._batches or units_sold <= 0:
            return Money.zero()

        cogs = Money.zero()
        remaining_to_sell = units_sold
        batches = self._batches[sku]

        while remaining_to_sell > 0 and batches:
            batch = batches[0]
            take = min(batch.quantity, remaining_to_sell)

            # Calculate COGS for this portion using Money for precision
            batch_cost_per_unit = Money.from_dollars(batch.cost_per_unit)
            cogs += batch_cost_per_unit * take
            
            # Update batch and remaining quantities
            batch.quantity -= take
            remaining_to_sell -= take

            # Remove empty batches
            if batch.quantity == 0:
                batches.pop(0)

        return cogs

    def get_quantity(self, sku: str) -> int:
        """
        Get the total quantity available for a given SKU.

        Args:
            sku: Stock Keeping Unit identifier.

        Returns:
            int: Total quantity available across all batches.
        """
        return sum(batch.quantity for batch in self._batches.get(sku, []))

    def get_total_value(self, sku: str) -> Money:
        """
        Get the total value of inventory for a given SKU.

        Args:
            sku: Stock Keeping Unit identifier.

        Returns:
            Money: Total value of inventory (quantity * cost_per_unit) across all batches.
        """
        batches = self._batches.get(sku, [])
        total_value = Money.zero()
        
        for batch in batches:
            batch_cost = Money.from_dollars(batch.cost_per_unit)
            total_value += batch_cost * batch.quantity
            
        return total_value

    def get_average_cost(self, sku: str) -> Money:
        """
        Get the weighted average cost per unit for a given SKU.

        Args:
            sku: Stock Keeping Unit identifier.

        Returns:
            Money: Weighted average cost per unit across all batches.
        """
        total_quantity = self.get_quantity(sku)
        if total_quantity == 0:
            return Money.zero()
            
        total_value = self.get_total_value(sku)
        return Money(total_value.cents // total_quantity)

    def has_sufficient_inventory(self, sku: str, required_qty: int) -> bool:
        """
        Check if there is sufficient inventory for a given SKU.

        Args:
            sku: Stock Keeping Unit identifier.
            required_qty: Required quantity to check against.

        Returns:
            bool: True if sufficient inventory is available.
        """
        return self.get_quantity(sku) >= required_qty

    def get_batches_info(self, sku: str) -> List[Dict]:
        """
        Get detailed information about all batches for a given SKU.
        
        Args:
            sku: Stock Keeping Unit identifier.
            
        Returns:
            List[Dict]: List of batch information dictionaries.
        """
        batches = self._batches.get(sku, [])
        return [
            {
                "quantity": batch.quantity,
                "cost_per_unit": Money.from_dollars(batch.cost_per_unit),
                "received": batch.received,
                "total_value": Money.from_dollars(batch.cost_per_unit) * batch.quantity
            }
            for batch in batches
        ]

    def update_inventory(self, product, units_sold: int) -> Money:
        """
        Legacy method for compatibility with existing code.
        Updates inventory and returns COGS.
        
        Args:
            product: The product object to update.
            units_sold: Number of units sold to remove from inventory.
            
        Returns:
            Money: Cost of goods sold for the units removed.
        """
        # Handle legacy product objects that might have a simple qty attribute
        if hasattr(product, "qty"):
            product.qty = max(0, product.qty - units_sold)
            
        # If product has an ASIN or SKU, use batch-based inventory
        sku = getattr(product, "asin", getattr(product, "sku", None))
        if sku:
            return self.process_sale_and_get_cogs(sku, units_sold)
        
        return Money.zero()

    def clear_all_inventory(self) -> None:
        """Clear all inventory batches. Useful for testing."""
        self._batches.clear()

    def get_all_skus(self) -> List[str]:
        """Get a list of all SKUs that have inventory."""
        return list(self._batches.keys())