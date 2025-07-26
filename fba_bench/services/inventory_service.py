"""
InventoryService for FBA-Bench: handles inventory removal and reconciliation.
"""

from typing import Any

class InventoryService:
    """
    Service for updating and reconciling product inventory after sales.
    """

    def update_inventory(self, product: Any, units_sold: int) -> None:
        """
        Remove sold units from inventory and reconcile state.

        Args:
            product: The product object to update.
            units_sold: Number of units sold to remove from inventory.
        """
        if hasattr(product, "qty"):
            product.qty = max(0, product.qty - units_sold)
        # If there is a more complex inventory system (e.g., batches), add logic here.