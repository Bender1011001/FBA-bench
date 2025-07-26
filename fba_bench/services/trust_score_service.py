"""
TrustScoreService for FBA-Bench: encapsulates trust score update logic.
"""

from typing import List, Dict, Any

class TrustScoreService:
    """
    Service for updating agent/product trust scores based on customer events and order history.
    """

    def update_score(
        self,
        current_score: float,
        customer_events: List[Dict[str, Any]],
        total_orders: int
    ) -> float:
        """
        Update trust score based on customer event history and current score.

        Args:
            current_score: The current trust score (float)
            customer_events: List of customer events (dicts)
            total_orders: Total number of orders for normalization

        Returns:
            New trust score (float, bounded [0.0, 1.0])
        """
        if not customer_events or total_orders == 0:
            return 1.0  # Default high trust for new products

        # Count negative events
        negative_events = [
            e for e in customer_events
            if e.get("type") in ["negative_review", "a_to_z_claim", "return_request"]
        ]
        negative_rate = len(negative_events) / total_orders

        # Trust score decreases with negative event rate
        trust_score = max(0.1, 1.0 - (negative_rate * 2.0))  # Cap at 0.1 minimum

        # Optionally, blend with current_score for smoother updates
        blended_score = (current_score * 0.7) + (trust_score * 0.3)

        return min(1.0, max(0.0, blended_score))