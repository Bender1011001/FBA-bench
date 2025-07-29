"""Unified TrustScoreService with Money type integration and enhanced analytics."""

import asyncio
import logging
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from money import Money
from models.product import Product
from events import SaleOccurred
from event_bus import EventBus


logger = logging.getLogger(__name__)


class TrustEvent(Enum):
    """Types of trust-affecting events."""
    SALE = "sale"
    RETURN = "return"
    COMPLAINT = "complaint"
    POSITIVE_REVIEW = "positive_review"
    NEGATIVE_REVIEW = "negative_review"
    PRICE_CHANGE = "price_change"
    STOCKOUT = "stockout"
    RESTOCK = "restock"


@dataclass
class TrustEventRecord:
    """Record of a trust-affecting event with financial impact."""
    event_type: TrustEvent
    timestamp: datetime
    impact_score: float
    financial_impact: Money
    product_id: str
    details: Dict
    confidence: float = 1.0


class TrustScoreService:
    """
    Unified trust score service with Money type integration.
    
    Combines the clean architecture from fba_bench_good_sim with
    enhanced analytics and financial accuracy from FBA-bench-main.
    
    Now event-driven: subscribes to SaleOccurred events to automatically
    update trust scores based on sales performance.
    """
    
    def __init__(self, config: Dict):
        """Initialize trust score service with configuration."""
        self.config = config
        self.trust_scores: Dict[str, float] = {}  # product_id -> trust_score
        self.event_history: List[TrustEventRecord] = []
        self.event_bus: Optional[EventBus] = None
        
        # Trust score parameters
        self.base_trust_score = config.get('base_trust_score', 0.7)
        self.max_trust_score = config.get('max_trust_score', 1.0)
        self.min_trust_score = config.get('min_trust_score', 0.1)
        
        # Event impact weights
        self.event_weights = config.get('trust_event_weights', {
            TrustEvent.SALE.value: 0.02,
            TrustEvent.RETURN.value: -0.05,
            TrustEvent.COMPLAINT.value: -0.10,
            TrustEvent.POSITIVE_REVIEW.value: 0.08,
            TrustEvent.NEGATIVE_REVIEW.value: -0.12,
            TrustEvent.PRICE_CHANGE.value: -0.01,
            TrustEvent.STOCKOUT.value: -0.03,
            TrustEvent.RESTOCK.value: 0.01
        })
        # Trust event value thresholds
        self.low_value_threshold = Money.from_dollars(config.get('low_value_threshold_dollars', 10.0))
        self.high_value_threshold = Money.from_dollars(config.get('high_value_threshold_dollars', 100.0))
        
        # Decay parameters
        self.trust_decay_rate = config.get('trust_decay_rate', 0.001)  # Daily decay
        self.event_memory_days = config.get('event_memory_days', 30)
        
        # Financial impact thresholds
        self.high_value_threshold = Money.from_dollars(config.get('high_value_threshold_dollars', 100.0))
        
        logger.info("TrustScoreService initialized")
    
    async def start(self, event_bus: EventBus) -> None:
        """Start the trust score service and subscribe to events."""
        self.event_bus = event_bus
        await self.event_bus.subscribe(SaleOccurred, self._handle_sale_occurred)
        logger.info("TrustScoreService started and subscribed to SaleOccurred events")
    
    async def stop(self) -> None:
        """Stop the trust score service."""
        logger.info("TrustScoreService stopped")
    
    async def _handle_sale_occurred(self, event: SaleOccurred) -> None:
        """Handle SaleOccurred events to update trust scores."""
        try:
            # Process the sale event to update trust score
            await self._process_sale_event(event)
            
        except Exception as e:
            logger.error(f"Error handling SaleOccurred event {event.event_id}: {e}")
    
    async def _process_sale_event(self, sale_event: SaleOccurred) -> None:
        """Process a sale event to update trust scores and related metrics."""
        product_id = sale_event.asin
        
        # Record sale event
        if sale_event.units_sold > 0:
            # Positive sale event
            self.record_event(
                product_id=product_id,
                event_type=TrustEvent.SALE,
                financial_impact=sale_event.total_revenue,
                details={
                    'units_sold': sale_event.units_sold,
                    'conversion_rate': sale_event.conversion_rate,
                    'profit_margin': sale_event.get_profit_margin_percentage(),
                    'sale_event_id': sale_event.event_id
                },
                confidence=1.0
            )
            
            # If conversion rate is very low, record as potential quality issue
            if sale_event.conversion_rate < 0.05:  # Less than 5% conversion
                self.record_event(
                    product_id=product_id,
                    event_type=TrustEvent.COMPLAINT,
                    financial_impact=Money.zero(),
                    details={
                        'reason': 'low_conversion_rate',
                        'conversion_rate': sale_event.conversion_rate,
                        'units_demanded': sale_event.units_demanded
                    },
                    confidence=0.3  # Lower confidence since it's inferred
                )
        
        # Check for potential stockout if units demanded > units sold
        if sale_event.units_demanded > sale_event.units_sold:
            stockout_impact = sale_event.units_demanded - sale_event.units_sold
            self.record_event(
                product_id=product_id,
                event_type=TrustEvent.STOCKOUT,
                financial_impact=sale_event.unit_price * stockout_impact,
                details={
                    'missed_sales': stockout_impact,
                    'demand_vs_inventory': {
                        'demanded': sale_event.units_demanded,
                        'sold': sale_event.units_sold
                    }
                },
                confidence=0.8
            )
        
        # Analyze profit margins for pricing trust impact
        profit_margin_pct = sale_event.get_profit_margin_percentage()
        if profit_margin_pct < 5:  # Very low margin might indicate pricing pressure
            self.record_event(
                product_id=product_id,
                event_type=TrustEvent.PRICE_CHANGE,
                financial_impact=sale_event.total_revenue,
                details={
                    'profit_margin_percent': profit_margin_pct,
                    'reason': 'low_margin_pressure'
                },
                confidence=0.5
            )
        
        logger.debug(f"Processed sale event for product {product_id}: {sale_event.units_sold} units sold")
    
    def _calculate_sale_trust_impact(self, sale_event: SaleOccurred) -> float:
        """Calculate trust impact score for a sale event."""
        base_impact = 0.02  # Base positive impact per sale
        
        # Adjust based on conversion rate
        conversion_bonus = (sale_event.conversion_rate - 0.1) * 0.1  # Bonus for good conversion
        
        # Adjust based on profit margin
        profit_margin = sale_event.get_profit_margin_percentage()
        margin_factor = min(1.5, max(0.5, profit_margin / 20.0))  # Scale based on 20% target margin
        
        # Adjust based on units sold
        volume_factor = min(2.0, 1.0 + (sale_event.units_sold - 1) * 0.1)  # Bonus for multiple units
        
        return base_impact * (1.0 + conversion_bonus) * margin_factor * volume_factor
        self.low_value_threshold = Money.from_dollars(config.get('low_value_threshold_dollars', 10.0))
        
        logger.info("TrustScoreService initialized")
    
    async def start(self, event_bus: EventBus) -> None:
        """Start the trust score service and subscribe to events."""
        self.event_bus = event_bus
        await self.event_bus.subscribe(SaleOccurred, self._handle_sale_occurred)
        logger.info("TrustScoreService started and subscribed to SaleOccurred events")
    
    async def stop(self) -> None:
        """Stop the trust score service."""
        logger.info("TrustScoreService stopped")
    
    async def _handle_sale_occurred(self, event: SaleOccurred) -> None:
        """Handle SaleOccurred events to update trust scores."""
        try:
            # Process the sale event to update trust score
            await self._process_sale_event(event)
            
        except Exception as e:
            logger.error(f"Error handling SaleOccurred event {event.event_id}: {e}")
    
    async def _process_sale_event(self, sale_event: SaleOccurred) -> None:
        """Process a sale event to update trust scores and related metrics."""
        product_id = sale_event.asin
        
        # Record sale event
        if sale_event.units_sold > 0:
            # Positive sale event
            sale_impact = self._calculate_sale_trust_impact(sale_event)
            self.record_event(
                product_id=product_id,
                event_type=TrustEvent.SALE,
                financial_impact=sale_event.total_revenue,
                details={
                    'units_sold': sale_event.units_sold,
                    'conversion_rate': sale_event.conversion_rate,
                    'profit_margin': sale_event.get_profit_margin_percentage(),
                    'sale_event_id': sale_event.event_id
                },
                confidence=1.0
            )
            
            # If conversion rate is very low, record as potential quality issue
            if sale_event.conversion_rate < 0.05:  # Less than 5% conversion
                self.record_event(
                    product_id=product_id,
                    event_type=TrustEvent.COMPLAINT,
                    financial_impact=Money.zero(),
                    details={
                        'reason': 'low_conversion_rate',
                        'conversion_rate': sale_event.conversion_rate,
                        'units_demanded': sale_event.units_demanded
                    },
                    confidence=0.3  # Lower confidence since it's inferred
                )
        
        # Check for potential stockout if units demanded > units sold
        if sale_event.units_demanded > sale_event.units_sold:
            stockout_impact = sale_event.units_demanded - sale_event.units_sold
            self.record_event(
                product_id=product_id,
                event_type=TrustEvent.STOCKOUT,
                financial_impact=sale_event.unit_price * stockout_impact,
                details={
                    'missed_sales': stockout_impact,
                    'demand_vs_inventory': {
                        'demanded': sale_event.units_demanded,
                        'sold': sale_event.units_sold
                    }
                },
                confidence=0.8
            )
        
        # Analyze profit margins for pricing trust impact
        profit_margin_pct = sale_event.get_profit_margin_percentage()
        if profit_margin_pct < 5:  # Very low margin might indicate pricing pressure
            self.record_event(
                product_id=product_id,
                event_type=TrustEvent.PRICE_CHANGE,
                financial_impact=sale_event.total_revenue,
                details={
                    'profit_margin_percent': profit_margin_pct,
                    'reason': 'low_margin_pressure'
                },
                confidence=0.5
            )
        
        logger.debug(f"Processed sale event for product {product_id}: {sale_event.units_sold} units sold")
    
    def _calculate_sale_trust_impact(self, sale_event: SaleOccurred) -> float:
        """Calculate trust impact score for a sale event."""
        base_impact = 0.02  # Base positive impact per sale
        
        # Adjust based on conversion rate
        conversion_bonus = (sale_event.conversion_rate - 0.1) * 0.1  # Bonus for good conversion
        
        # Adjust based on profit margin
        profit_margin = sale_event.get_profit_margin_percentage()
        margin_factor = min(1.5, max(0.5, profit_margin / 20.0))  # Scale based on 20% target margin
        
        # Adjust based on units sold
        volume_factor = min(2.0, 1.0 + (sale_event.units_sold - 1) * 0.1)  # Bonus for multiple units
        
        return base_impact * (1.0 + conversion_bonus) * margin_factor * volume_factor
        
    def get_trust_score(self, product_id: str) -> float:
        """Get current trust score for a product."""
        if product_id not in self.trust_scores:
            self.trust_scores[product_id] = self.base_trust_score
        
        # Apply time-based decay
        self._apply_trust_decay(product_id)
        
        return self.trust_scores[product_id]
        
    def record_event(
        self, 
        product_id: str, 
        event_type: TrustEvent, 
        financial_impact: Money = None,
        details: Dict = None,
        confidence: float = 1.0
    ) -> float:
        """
        Record a trust-affecting event and update trust score.
        
        Args:
            product_id: Product identifier
            event_type: Type of trust event
            financial_impact: Financial impact of the event
            details: Additional event details
            confidence: Confidence in the event (0.0 to 1.0)
            
        Returns:
            Updated trust score
        """
        if financial_impact is None:
            financial_impact = Money.zero()
        if details is None:
            details = {}
            
        # Calculate impact score based on event type and financial impact
        impact_score = self._calculate_impact_score(event_type, financial_impact, details)
        
        # Create event record
        event_record = TrustEventRecord(
            event_type=event_type,
            timestamp=datetime.now(),
            impact_score=impact_score,
            financial_impact=financial_impact,
            product_id=product_id,
            details=details,
            confidence=confidence
        )
        
        # Add to history
        self.event_history.append(event_record)
        
        # Update trust score
        current_score = self.get_trust_score(product_id)
        adjusted_impact = impact_score * confidence
        new_score = current_score + adjusted_impact
        
        # Apply bounds
        new_score = max(self.min_trust_score, min(self.max_trust_score, new_score))
        self.trust_scores[product_id] = new_score
        
        # Clean old events
        self._clean_old_events()
        
        return new_score
        
    def _calculate_impact_score(
        self, 
        event_type: TrustEvent, 
        financial_impact: Money,
        details: Dict
    ) -> float:
        """Calculate the impact score for an event."""
        base_impact = self.event_weights.get(event_type.value, 0.0)
        
        # Adjust impact based on financial magnitude
        financial_multiplier = self._get_financial_multiplier(financial_impact)
        
        # Adjust impact based on event-specific details
        detail_multiplier = self._get_detail_multiplier(event_type, details)
        
        return base_impact * financial_multiplier * detail_multiplier
        
    def _get_financial_multiplier(self, financial_impact: Money) -> float:
        """Get multiplier based on financial impact magnitude."""
        if financial_impact.cents == 0:
            return 1.0
            
        abs_impact = abs(financial_impact)
        
        if abs_impact >= self.high_value_threshold:
            return 2.0  # High-value events have double impact
        elif abs_impact <= self.low_value_threshold:
            return 0.5  # Low-value events have half impact
        else:
            # Linear interpolation between thresholds
            range_size = self.high_value_threshold - self.low_value_threshold
            position_cents = abs_impact.cents - self.low_value_threshold.cents
            range_cents = range_size.cents
            position = position_cents / range_cents if range_cents > 0 else 0
            return 0.5 + (1.5 * position)  # Scale from 0.5 to 2.0
            
    def _get_detail_multiplier(self, event_type: TrustEvent, details: Dict) -> float:
        """Get multiplier based on event-specific details."""
        multiplier = 1.0
        
        if event_type == TrustEvent.PRICE_CHANGE:
            # Larger price changes have more impact
            price_change_percent = details.get('price_change_percent', 0.0)
            multiplier = 1.0 + abs(price_change_percent) * 2.0
            
        elif event_type == TrustEvent.RETURN:
            # Return reason affects impact
            reason = details.get('reason', 'unknown')
            if reason in ['defective', 'not_as_described']:
                multiplier = 2.0
            elif reason in ['changed_mind', 'found_better_price']:
                multiplier = 0.8
                
        elif event_type in [TrustEvent.POSITIVE_REVIEW, TrustEvent.NEGATIVE_REVIEW]:
            # Review rating affects impact
            rating = details.get('rating', 3)
            if event_type == TrustEvent.POSITIVE_REVIEW:
                multiplier = (rating - 3) / 2.0  # 4-5 star reviews
            else:
                multiplier = (3 - rating) / 2.0  # 1-2 star reviews
                
        elif event_type == TrustEvent.STOCKOUT:
            # Duration of stockout affects impact
            duration_hours = details.get('duration_hours', 24)
            multiplier = 1.0 + (duration_hours / 24.0) * 0.5  # Increase impact for longer stockouts
            
        return max(0.1, min(3.0, multiplier))  # Bound between 0.1 and 3.0
        
    def _apply_trust_decay(self, product_id: str) -> None:
        """Apply time-based decay to trust score."""
        if product_id not in self.trust_scores:
            return
            
        current_score = self.trust_scores[product_id]
        
        # Calculate days since last update (simplified - in real implementation,
        # you'd track last update time per product)
        days_since_update = 1.0  # Assume 1 day for simplicity
        
        # Apply exponential decay towards base score
        target_score = self.base_trust_score
        decay_factor = math.exp(-self.trust_decay_rate * days_since_update)
        
        new_score = current_score * decay_factor + target_score * (1 - decay_factor)
        self.trust_scores[product_id] = new_score
        
    def _clean_old_events(self) -> None:
        """Remove events older than the memory window."""
        cutoff_date = datetime.now() - timedelta(days=self.event_memory_days)
        self.event_history = [
            event for event in self.event_history 
            if event.timestamp >= cutoff_date
        ]
        
    def get_trust_trend(self, product_id: str, days: int = 7) -> Dict:
        """Get trust score trend over the specified period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get relevant events
        relevant_events = [
            event for event in self.event_history
            if event.product_id == product_id and event.timestamp >= cutoff_date
        ]
        
        if not relevant_events:
            return {
                'trend': 'stable',
                'change': 0.0,
                'event_count': 0,
                'net_financial_impact': Money.zero()
            }
            
        # Calculate trend
        total_impact = sum(event.impact_score * event.confidence for event in relevant_events)
        net_financial_impact = sum((event.financial_impact for event in relevant_events), Money.zero())
        
        # Determine trend direction
        if total_impact > 0.05:
            trend = 'improving'
        elif total_impact < -0.05:
            trend = 'declining'
        else:
            trend = 'stable'
            
        return {
            'trend': trend,
            'change': total_impact,
            'event_count': len(relevant_events),
            'net_financial_impact': net_financial_impact
        }
        
    def get_trust_factors(self, product_id: str) -> Dict:
        """Get detailed breakdown of trust factors for a product."""
        recent_events = [
            event for event in self.event_history
            if event.product_id == product_id and 
            event.timestamp >= datetime.now() - timedelta(days=self.event_memory_days)
        ]
        
        # Group events by type
        event_groups = {}
        for event in recent_events:
            event_type = event.event_type.value
            if event_type not in event_groups:
                event_groups[event_type] = {
                    'count': 0,
                    'total_impact': 0.0,
                    'total_financial_impact': Money.zero()
                }
            
            group = event_groups[event_type]
            group['count'] += 1
            group['total_impact'] += event.impact_score * event.confidence
            group['total_financial_impact'] += event.financial_impact
            
        # Calculate factor contributions
        factors = {}
        for event_type, group in event_groups.items():
            factors[event_type] = {
                'event_count': group['count'],
                'trust_contribution': group['total_impact'],
                'financial_impact': group['total_financial_impact'],
                'average_impact': group['total_impact'] / group['count'] if group['count'] > 0 else 0.0
            }
            
        return {
            'current_trust_score': self.get_trust_score(product_id),
            'base_trust_score': self.base_trust_score,
            'factors': factors,
            'total_events': len(recent_events)
        }
        
    def simulate_event_impact(
        self, 
        product_id: str, 
        event_type: TrustEvent,
        financial_impact: Money = None,
        details: Dict = None
    ) -> Dict:
        """Simulate the impact of an event without recording it."""
        if financial_impact is None:
            financial_impact = Money.zero()
        if details is None:
            details = {}
            
        current_score = self.get_trust_score(product_id)
        impact_score = self._calculate_impact_score(event_type, financial_impact, details)
        
        projected_score = current_score + impact_score
        projected_score = max(self.min_trust_score, min(self.max_trust_score, projected_score))
        
        return {
            'current_trust_score': current_score,
            'projected_trust_score': projected_score,
            'impact_score': impact_score,
            'score_change': projected_score - current_score,
            'event_type': event_type.value,
            'financial_impact': financial_impact
        }
        
    def get_product_rankings(self, product_ids: List[str]) -> List[Tuple[str, float]]:
        """Get products ranked by trust score."""
        rankings = [(pid, self.get_trust_score(pid)) for pid in product_ids]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
        
    def get_statistics(self) -> Dict:
        """Get trust score service statistics."""
        if not self.trust_scores:
            return {
                'total_products': 0,
                'average_trust_score': 0.0,
                'total_events': 0,
                'event_type_distribution': {}
            }
            
        # Calculate statistics
        scores = list(self.trust_scores.values())
        avg_score = sum(scores) / len(scores)
        
        # Event type distribution
        event_distribution = {}
        for event in self.event_history:
            event_type = event.event_type.value
            event_distribution[event_type] = event_distribution.get(event_type, 0) + 1
            
        return {
            'total_products': len(self.trust_scores),
            'average_trust_score': avg_score,
            'highest_trust_score': max(scores),
            'lowest_trust_score': min(scores),
            'total_events': len(self.event_history),
            'event_type_distribution': event_distribution
        }
        
    def reset_trust_score(self, product_id: str) -> float:
        """Reset a product's trust score to base level."""
        self.trust_scores[product_id] = self.base_trust_score
        return self.base_trust_score
        
    def bulk_update_trust_scores(self, score_updates: Dict[str, float]) -> Dict[str, float]:
        """Update multiple trust scores at once."""
        updated_scores = {}
        
        for product_id, new_score in score_updates.items():
            # Apply bounds
            bounded_score = max(self.min_trust_score, min(self.max_trust_score, new_score))
            self.trust_scores[product_id] = bounded_score
            updated_scores[product_id] = bounded_score
            
        return updated_scores