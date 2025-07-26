"""
ListingManagerService for FBA-Bench: handles listing suppression and reinstatement logic.
"""

from typing import Any, Dict, Optional, List

class ListingManagerService:
    """
    Service for managing listing suppression and reinstatement based on trust score.
    """

    def update_listings(
        self,
        trust_score: float,
        product: Any,
        event_log: Optional[List[str]],
        current_date: Any
    ) -> Dict[str, Any]:
        """
        Update listing suppression status and apply effects.

        Args:
            trust_score: The current trust score for the product/agent.
            product: The product object to update.
            event_log: List to append log messages to.
            current_date: The current simulation date.

        Returns:
            suppression_info: Dict with suppression status and effects.
        """
        # Suppression logic (moved from event_management_service/sales_processor)
        if trust_score >= 0.7:
            suppression_info = {
                "suppressed": False,
                "level": "none",
                "demand_multiplier": 1.0,
                "search_penalty": 0.0
            }
        elif trust_score >= 0.5:
            suppression_info = {
                "suppressed": True,
                "level": "warning",
                "demand_multiplier": 0.8,
                "search_penalty": 0.1
            }
        elif trust_score >= 0.3:
            suppression_info = {
                "suppressed": True,
                "level": "moderate",
                "demand_multiplier": 0.5,
                "search_penalty": 0.3
            }
        elif trust_score >= 0.1:
            suppression_info = {
                "suppressed": True,
                "level": "severe",
                "demand_multiplier": 0.2,
                "search_penalty": 0.6
            }
        else:
            suppression_info = {
                "suppressed": True,
                "level": "critical",
                "demand_multiplier": 0.05,
                "search_penalty": 0.9
            }

        # Apply effects to product (if needed)
        if suppression_info["suppressed"]:
            # Demand and BSR effects should be applied by the orchestrator using this info
            if event_log is not None:
                level = suppression_info["level"]
                multiplier = suppression_info["demand_multiplier"]
                event_log.append(
                    f"Day {getattr(current_date, 'day', '?')}: Listing suppressed ({level} level) - "
                    f"trust score {trust_score:.2f}, demand reduced to {multiplier*100:.0f}%"
                )
        return suppression_info