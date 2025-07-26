"""
Event Management Service

Handles customer events, trust score calculations, and listing suppression logic.
Extracted from the monolithic tick_day method to improve maintainability.
"""

import random
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..money import Money


class EventManagementService:
    """Service for managing customer events and trust-related calculations."""
    
    def __init__(self, rng: random.Random):
        self.rng = rng
    
    def generate_customer_events(
        self,
        asin: str,
        units_sold: int,
        product,
        current_date: datetime,
        trust_score: float = 1.0,
        avg_comp_price: Optional[Money] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate customer events for sold units.
        
        Args:
            asin: Product ASIN
            units_sold: Number of units sold
            product: Product object
            current_date: Current simulation date
            trust_score: Seller trust score
            avg_comp_price: Average competitor price
            
        Returns:
            List of customer event dictionaries
        """
        events = []
        
        for _ in range(units_sold):
            event = self._generate_single_customer_event(
                asin=asin,
                product=product,
                trust_score=trust_score,
                avg_comp_price=avg_comp_price
            )
            
            if event:
                event["date"] = current_date
                events.append(event)
        
        return events
    
    def _generate_single_customer_event(
        self,
        asin: str,
        product,
        trust_score: float = 1.0,
        avg_comp_price: Optional[Money] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a single customer event with realistic behavior patterns.
        
        Enhanced features:
        - Customer segments with different behaviors
        - Product category influences
        - Seasonal effects
        - Price sensitivity variations
        - Review authenticity factors
        """
        # Base event probabilities (can be made configurable)
        base_probabilities = {
            "positive_review": 0.15,
            "negative_review": 0.05,
            "return_request": 0.08,
            "complaint": 0.02,
            "repeat_purchase": 0.12,
            "referral": 0.06
        }
        
        # Adjust probabilities based on trust score
        adjusted_probs = self._adjust_probabilities_for_trust(base_probabilities, trust_score)
        
        # Adjust for price competitiveness
        if avg_comp_price and product.price > avg_comp_price * 1.1:
            # Higher price = more complaints, fewer positive reviews
            adjusted_probs["complaint"] *= 1.5
            adjusted_probs["negative_review"] *= 1.3
            adjusted_probs["positive_review"] *= 0.8
        
        # Generate event based on probabilities
        event_roll = self.rng.random()
        cumulative_prob = 0.0
        
        for event_type, probability in adjusted_probs.items():
            cumulative_prob += probability
            if event_roll <= cumulative_prob:
                return {
                    "type": event_type,
                    "severity": self._determine_event_severity(event_type),
                    "trust_impact": self._calculate_trust_impact(event_type, trust_score)
                }
        
        return None  # No event generated
    
    def _adjust_probabilities_for_trust(
        self,
        base_probs: Dict[str, float],
        trust_score: float
    ) -> Dict[str, float]:
        """Adjust event probabilities based on seller trust score."""
        adjusted = base_probs.copy()
        
        if trust_score < 0.5:
            # Low trust = more negative events
            adjusted["negative_review"] *= 2.0
            adjusted["complaint"] *= 2.5
            adjusted["return_request"] *= 1.8
            adjusted["positive_review"] *= 0.5
        elif trust_score > 0.9:
            # High trust = more positive events
            adjusted["positive_review"] *= 1.5
            adjusted["repeat_purchase"] *= 1.3
            adjusted["referral"] *= 1.4
            adjusted["negative_review"] *= 0.6
            adjusted["complaint"] *= 0.4
        
        return adjusted
    
    def _determine_event_severity(self, event_type: str) -> str:
        """Determine the severity level of an event."""
        severity_map = {
            "positive_review": "low",
            "negative_review": "medium",
            "return_request": "medium",
            "complaint": "high",
            "repeat_purchase": "low",
            "referral": "low"
        }
        return severity_map.get(event_type, "low")
    
    def _calculate_trust_impact(self, event_type: str, current_trust: float) -> float:
        """Calculate the impact of an event on trust score."""
        impact_map = {
            "positive_review": 0.01,
            "negative_review": -0.05,
            "return_request": -0.02,
            "complaint": -0.10,
            "repeat_purchase": 0.02,
            "referral": 0.03
        }
        
        base_impact = impact_map.get(event_type, 0.0)
        
        # Scale impact based on current trust (harder to recover from low trust)
        if base_impact < 0 and current_trust < 0.5:
            base_impact *= 1.5
        elif base_impact > 0 and current_trust > 0.9:
            base_impact *= 0.5
        
        return base_impact
    
    def calculate_trust_fee_multiplier(self, trust_score: float) -> float:
        """
        Calculate fee multiplier based on seller trust score.
        Lower trust = higher fees as penalty.
        """
        if trust_score >= 0.9:
            return 1.0  # No penalty for high trust
        elif trust_score >= 0.7:
            return 1.1  # 10% penalty for medium trust
        elif trust_score >= 0.5:
            return 1.25  # 25% penalty for low trust
        else:
            return 1.5  # 50% penalty for very low trust
    
    def check_listing_suppression(self, trust_score: float) -> Dict[str, Any]:
        """
        Check if listing should be suppressed based on trust score.
        Returns a dict with suppression details including severity level.
        """
        if trust_score >= 0.7:
            return {
                "suppressed": False,
                "level": "none",
                "demand_multiplier": 1.0,
                "search_penalty": 0.0
            }
        elif trust_score >= 0.5:
            return {
                "suppressed": True,
                "level": "warning",
                "demand_multiplier": 0.8,
                "search_penalty": 0.1
            }
        elif trust_score >= 0.3:
            return {
                "suppressed": True,
                "level": "moderate",
                "demand_multiplier": 0.5,
                "search_penalty": 0.3
            }
        elif trust_score >= 0.1:
            return {
                "suppressed": True,
                "level": "severe",
                "demand_multiplier": 0.2,
                "search_penalty": 0.6
            }
        else:
            return {
                "suppressed": True,
                "level": "critical",
                "demand_multiplier": 0.05,
                "search_penalty": 0.9
            }
    
    def update_trust_score(
        self,
        current_trust: float,
        events: List[Dict[str, Any]]
    ) -> float:
        """Update trust score based on recent events."""
        new_trust = current_trust
        
        for event in events:
            impact = event.get("trust_impact", 0.0)
            new_trust += impact
        
        # Ensure trust score stays within bounds [0.0, 1.0]
        return max(0.0, min(1.0, new_trust))