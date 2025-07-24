from dataclasses import dataclass
from typing import Dict
from datetime import datetime

@dataclass
class InventoryBatch:
    """
    Represents a batch of inventory received at a specific time.

    Attributes:
        quantity (int): Number of units in the batch.
        cost_per_unit (float): Cost per unit for this batch.
        received (datetime): Date and time the batch was received.
    """
    quantity: int
    cost_per_unit: float
    received: datetime

class InventoryManager:
    """
    Manages inventory batches for multiple SKUs.

    Attributes:
        _batches (Dict[str, list[InventoryBatch]]): Mapping of SKU to list of inventory batches.
    """
    def __init__(self):
        """
        Initialize the InventoryManager with empty batches.
        """
        self._batches: Dict[str, list[InventoryBatch]] = {}

    def add(self, sku: str, qty: int, cost: float, received: datetime):
        """
        Add a new batch of inventory for a given SKU.

        Args:
            sku (str): Stock Keeping Unit identifier.
            qty (int): Quantity of units to add.
            cost (float): Cost per unit.
            received (datetime): Date and time the batch was received.
        """
        self._batches.setdefault(sku, []).append(InventoryBatch(qty, cost, received))

    def remove(self, sku: str, qty: int):
        """
        Remove units from inventory for a given SKU, using FIFO order.

        Args:
            sku (str): Stock Keeping Unit identifier.
            qty (int): Quantity of units to remove.

        Returns:
            int: Number of units actually removed (may be less if not enough inventory).
        """
        batches = self._batches.get(sku, [])
        removed = 0
        while qty > 0 and batches:
            batch = batches[0]
            take = min(batch.quantity, qty)
            batch.quantity -= take
            qty -= take
            removed += take
            if batch.quantity == 0:
                batches.pop(0)
        self._batches[sku] = batches
        return removed

    def quantity(self, sku: str) -> int:
        """
        Get the total quantity available for a given SKU.

        Args:
            sku (str): Stock Keeping Unit identifier.

        Returns:
            int: Total quantity available across all batches.
        """
        batches = self._batches.get(sku, [])
        return sum(batch.quantity for batch in batches)

    def total_value(self, sku: str) -> float:
        """
        Get the total value of inventory for a given SKU.

        Args:
            sku (str): Stock Keeping Unit identifier.

        Returns:
            float: Total value of inventory (quantity * cost_per_unit) across all batches.
        """
        batches = self._batches.get(sku, [])
        return sum(batch.quantity * batch.cost_per_unit for batch in batches)