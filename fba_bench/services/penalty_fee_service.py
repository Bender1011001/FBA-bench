"""
Penalty and Ancillary Fee Service.

Handles calculation of penalty fees and ancillary charges based on
performance metrics, policy violations, and operational issues.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import random
from fba_bench.money import Money
from fba_bench import market_dynamics


class PenaltyFeeService:
    """
    Service responsible for calculating penalty and ancillary fees.
    
    This service encapsulates the complex fee calculation logic for penalties
    and ancillary charges that was previously embedded in the monolithic tick_day() method.
    """
    
    def __init__(self, rng: random.Random, config: Dict[str, Any]):
        """
        Initialize penalty fee service.
        
        Args:
            rng: Seeded random number generator for deterministic behavior
            config: Configuration dictionary containing fee rates and thresholds
        """
        self.rng = rng
        self.config = config
        
        # Load fee rates from config
        self.return_processing_fee_pct = config.get('return_processing_fee_pct', 0.15)
        self.unplanned_service_fee_per_unit = config.get('unplanned_service_fee_per_unit', 3.00)
        
        # High-prep categories that require more processing
        self.high_prep_categories = ["Electronics", "Toys", "Health", "Beauty"]
        
        # High-risk categories with stricter penalties
        self.high_risk_categories = ["Health", "Beauty", "Baby", "Electronics"]
    
    def calculate_ancillary_fees(
        self,
        asin: str,
        product: Any,
        units_sold: int,
        customer_events: Dict[str, List[Dict[str, Any]]],
        current_date: datetime
    ) -> Money:
        """
        Calculate ancillary fees based on product state and recent events.
        
        Args:
            asin: Product ASIN
            product: Product object
            units_sold: Number of units sold
            customer_events: Dictionary of customer events by ASIN
            current_date: Current simulation date
            
        Returns:
            Total ancillary fees as Money object
        """
        ancillary_fee = Money.zero()
        
        # 1. Return processing fees
        ancillary_fee += self._calculate_return_processing_fees(
            asin, product, units_sold, customer_events, current_date
        )
        
        # 2. Prep service fees
        ancillary_fee += self._calculate_prep_service_fees(
            product, units_sold
        )
        
        # 3. Labeling and packaging fees
        ancillary_fee += self._calculate_labeling_fees(units_sold)
        
        # 4. Disposal fees
        ancillary_fee += self._calculate_disposal_fees(
            asin, units_sold, customer_events
        )
        
        # 5. Repackaging fees
        ancillary_fee += self._calculate_repackaging_fees(
            asin, units_sold, customer_events
        )
        
        # 6. Content and photography fees
        ancillary_fee += self._calculate_content_fees()
        
        return ancillary_fee
    
    def calculate_penalty_fees(
        self,
        asin: str,
        product: Any,
        units_sold: int,
        customer_events: Dict[str, List[Dict[str, Any]]],
        inventory: Any,
        competitors: List[Any],
        event_log: List[str],
        current_date: datetime
    ) -> Money:
        """
        Calculate penalty fees based on policy violations and performance issues.
        
        Args:
            asin: Product ASIN
            product: Product object
            units_sold: Number of units sold
            customer_events: Dictionary of customer events by ASIN
            inventory: Inventory management object
            competitors: List of competitor objects
            event_log: Simulation event log
            current_date: Current simulation date
            
        Returns:
            Total penalty fees as Money object
        """
        penalty_fee = Money.zero()
        
        # 1. Performance-based penalties
        penalty_fee += self._calculate_performance_penalties(
            asin, customer_events, current_date
        )
        
        # 2. Inventory performance penalties
        penalty_fee += self._calculate_inventory_penalties(
            asin, inventory, units_sold
        )
        
        # 3. Trust score penalties
        penalty_fee += self._calculate_trust_score_penalties(
            asin, customer_events, units_sold
        )
        
        # 4. Policy violation penalties
        penalty_fee += self._calculate_policy_violation_penalties(
            asin, event_log, current_date
        )
        
        # 5. Category-specific penalty adjustments
        penalty_fee = self._apply_category_penalty_adjustments(
            penalty_fee, product.category
        )
        
        # 6. Pricing policy penalties
        penalty_fee += self._calculate_pricing_penalties(
            product, competitors
        )
        
        # 7. Late shipment penalties
        penalty_fee += self._calculate_shipment_penalties(units_sold)
        
        # 8. Account health penalties
        penalty_fee += self._calculate_account_health_penalties(
            event_log, current_date
        )
        
        return penalty_fee
    
    def _calculate_return_processing_fees(
        self,
        asin: str,
        product: Any,
        units_sold: int,
        customer_events: Dict[str, List[Dict[str, Any]]],
        current_date: datetime
    ) -> Money:
        """Calculate return processing fees based on actual returns."""
        if asin not in customer_events:
            return Money.zero()
        
        # Get recent events (last 30 days)
        recent_events = [
            e for e in customer_events[asin]
            if (current_date - e["date"]).days <= 30
        ]
        
        # Count returns and refunds
        returns = [
            e for e in recent_events 
            if e["type"] in ["a_to_z_claim", "negative_review"]
        ]
        
        if not returns:
            return Money.zero()
        
        # Calculate return rate and fees
        estimated_return_rate = min(0.3, len(returns) / max(1, units_sold * 4))
        returned_units = int(units_sold * estimated_return_rate)
        
        return product.price * returned_units * Money.from_dollars(self.return_processing_fee_pct)
    
    def _calculate_prep_service_fees(self, product: Any, units_sold: int) -> Money:
        """Calculate prep service fees based on product characteristics."""
        prep_fee_probability = 0.02  # Base 2% chance
        
        # Higher prep fees for certain categories
        if product.category in self.high_prep_categories:
            prep_fee_probability = 0.08  # 8% for high-prep categories
        
        # Size-based prep fee adjustments
        if hasattr(product, 'weight') and product.weight > 5.0:
            prep_fee_probability *= 1.5
        
        if self.rng.random() < prep_fee_probability:
            prep_units = max(1, int(units_sold * 0.1))  # 10% of units need prep
            return Money.from_dollars(self.unplanned_service_fee_per_unit) * prep_units
        
        return Money.zero()
    
    def _calculate_labeling_fees(self, units_sold: int) -> Money:
        """Calculate labeling and packaging fees."""
        if self.rng.random() < 0.03:  # 3% chance of labeling issues
            return Money.from_dollars(0.55) * units_sold
        return Money.zero()
    
    def _calculate_disposal_fees(
        self,
        asin: str,
        units_sold: int,
        customer_events: Dict[str, List[Dict[str, Any]]]
    ) -> Money:
        """Calculate disposal fees for damaged/unsellable inventory."""
        if asin not in customer_events:
            return Money.zero()
        
        damage_events = [
            e for e in customer_events[asin]
            if (e.get("type") == "negative_review" and 
                "damaged" in e.get("text", "").lower())
        ]
        
        if not damage_events:
            return Money.zero()
        
        disposal_rate = min(0.05, len(damage_events) / max(1, units_sold * 10))
        disposal_units = int(units_sold * disposal_rate)
        
        return Money.from_dollars(0.15) * disposal_units
    
    def _calculate_repackaging_fees(
        self,
        asin: str,
        units_sold: int,
        customer_events: Dict[str, List[Dict[str, Any]]]
    ) -> Money:
        """Calculate repackaging fees for customer returns."""
        if asin not in customer_events:
            return Money.zero()
        
        return_events = [
            e for e in customer_events[asin]
            if e.get("type") == "a_to_z_claim"
        ]
        
        if not return_events:
            return Money.zero()
        
        repackaging_rate = min(0.15, len(return_events) / max(1, units_sold * 5))
        repackaging_units = int(units_sold * repackaging_rate)
        
        return Money.from_dollars(1.00) * repackaging_units
    
    def _calculate_content_fees(self) -> Money:
        """Calculate photography and content fees (occasional)."""
        if self.rng.random() < 0.01:  # 1% chance per day
            return Money.from_dollars(50.0)
        return Money.zero()
    
    def _calculate_performance_penalties(
        self,
        asin: str,
        customer_events: Dict[str, List[Dict[str, Any]]],
        current_date: datetime
    ) -> Money:
        """Calculate performance-based penalties."""
        if asin not in customer_events:
            return Money.zero()
        
        recent_events = [
            e for e in customer_events[asin]
            if (current_date - e["date"]).days <= 30
        ]
        
        if not recent_events:
            return Money.zero()
        
        negative_events = [
            e for e in recent_events 
            if e["type"] in ["negative_review", "a_to_z_claim"]
        ]
        
        negative_rate = len(negative_events) / len(recent_events)
        
        # Graduated penalty system
        if negative_rate > 0.3:
            return Money.from_dollars(200.0)  # Severe performance penalty
        elif negative_rate > 0.2:
            return Money.from_dollars(100.0)  # Moderate performance penalty
        elif negative_rate > 0.1:
            return Money.from_dollars(50.0)   # Warning penalty
        
        return Money.zero()
    
    def _calculate_inventory_penalties(
        self, asin: str, inventory: Any, units_sold: int
    ) -> Money:
        """Calculate inventory performance penalties."""
        penalty_fee = Money.zero()
        
        # Get current inventory
        batches = inventory._batches.get(asin, [])
        current_inventory = sum(getattr(batch, "quantity", 0) for batch in batches)
        
        # Stockout penalty
        if current_inventory == 0 and units_sold > 0:
            stockout_penalty = min(100.0, units_sold * 2.0)
            penalty_fee += Money.from_dollars(stockout_penalty)
        
        # Excess inventory penalty
        if current_inventory > units_sold * 10:
            excess_units = current_inventory - units_sold * 10
            excess_penalty = min(50.0, excess_units * 0.10)
            penalty_fee += Money.from_dollars(excess_penalty)
        
        return penalty_fee
    
    def _calculate_trust_score_penalties(
        self,
        asin: str,
        customer_events: Dict[str, List[Dict[str, Any]]],
        units_sold: int
    ) -> Money:
        """Calculate trust score-based penalties."""
        if asin not in customer_events:
            return Money.zero()
        
        events = customer_events[asin]
        cancellations = sum(1 for e in events if e.get("type") == "a_to_z_claim")
        negative_reviews = sum(1 for e in events if e.get("type") == "negative_review")
        customer_issues = sum(
            1 for e in events 
            if e.get("type") in ["negative_review", "negative_feedback"]
        )
        
        trust_score = market_dynamics.calculate_trust_score(
            cancellations=cancellations,
            negative_reviews=negative_reviews,
            customer_issues=customer_issues,
            total_orders=max(1, units_sold * 10)
        )
        
        if trust_score < 0.5:
            trust_penalty = (0.5 - trust_score) * 200.0
            return Money.from_dollars(trust_penalty)
        
        return Money.zero()
    
    def _calculate_policy_violation_penalties(
        self,
        asin: str,
        event_log: List[str],
        current_date: datetime
    ) -> Money:
        """Calculate policy violation penalties."""
        # Find recent violations (last 30 days)
        recent_violations = [
            e for e in event_log
            if ("policy" in e.lower() and asin in e and
                f"Day {max(1, current_date.day - 30)}" <= e <= f"Day {current_date.day}")
        ]
        
        penalty_fee = Money.zero()
        
        # Escalating penalties for repeated violations
        for i, violation in enumerate(recent_violations):
            base_penalty = Money.from_dollars(150.0)
            escalation_multiplier = 1.0 + (i * 0.5)
            penalty_fee += base_penalty * escalation_multiplier
        
        return penalty_fee
    
    def _apply_category_penalty_adjustments(
        self, penalty_fee: Money, category: str
    ) -> Money:
        """Apply category-specific penalty adjustments."""
        if category in self.high_risk_categories and penalty_fee > Money.zero():
            return penalty_fee * 1.3  # 30% increase
        return penalty_fee
    
    def _calculate_pricing_penalties(
        self, product: Any, competitors: List[Any]
    ) -> Money:
        """Calculate pricing policy penalties."""
        if not competitors:
            return Money.zero()
        
        # Calculate average competitor price
        total_comp_price = Money.zero()
        for competitor in competitors:
            total_comp_price += competitor.price
        avg_competitor_price = total_comp_price / len(competitors)
        
        # Penalty for excessive pricing (potential price gouging)
        if product.price > avg_competitor_price * 2:
            price_diff = product.price.to_decimal() - avg_competitor_price.to_decimal()
            pricing_penalty = min(75.0, float(price_diff) * 0.1)
            return Money.from_dollars(pricing_penalty)
        
        return Money.zero()
    
    def _calculate_shipment_penalties(self, units_sold: int) -> Money:
        """Calculate late shipment penalties (simulated)."""
        if self.rng.random() < 0.02:  # 2% chance of late shipment issues
            late_shipment_penalty = min(200.0, units_sold * 25.0)
            return Money.from_dollars(late_shipment_penalty)
        return Money.zero()
    
    def _calculate_account_health_penalties(
        self, event_log: List[str], current_date: datetime
    ) -> Money:
        """Calculate account health penalties (cumulative effect)."""
        # Count recent penalties (last 90 days)
        total_recent_penalties = sum(
            1 for e in event_log
            if ("penalty" in e.lower() and
                f"Day {max(1, current_date.day - 90)}" <= e <= f"Day {current_date.day}")
        )
        
        if total_recent_penalties > 5:
            excess_penalties = total_recent_penalties - 5
            account_health_penalty = min(300.0, excess_penalties * 50.0)
            return Money.from_dollars(account_health_penalty)
        
        return Money.zero()