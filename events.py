"""Event schema definitions for FBA-Bench v3 event-driven architecture."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod

from money import Money


@dataclass
class CompetitorState:
    """
    Snapshot of a competitor's key performance metrics.
    
    Contains all essential data needed by downstream services
    (primarily SalesService) to perform market analysis and
    demand calculations without requiring additional queries.
    
    Attributes:
        asin: Amazon Standard Identification Number for the competitor
        price: Current price of the competitor's product
        bsr: Best Seller Rank at time of snapshot
        sales_velocity: Current sales velocity (units per time period)
    """
    asin: str
    price: Money
    bsr: int
    sales_velocity: float
    
    def __post_init__(self):
        """Validate competitor state data."""
        # Validate ASIN
        if not self.asin or not isinstance(self.asin, str):
            raise ValueError("Competitor ASIN must be a non-empty string")
        
        # Validate Money type
        if not isinstance(self.price, Money):
            raise TypeError(f"Competitor price must be Money type, got {type(self.price)}")
        
        # Validate BSR
        if self.bsr < 1:
            raise ValueError(f"Competitor BSR must be >= 1, got {self.bsr}")
        
        # Validate sales velocity
        if self.sales_velocity < 0:
            raise ValueError(f"Competitor sales velocity must be >= 0, got {self.sales_velocity}")


@dataclass
class BaseEvent(ABC):
    """Base class for all events in the FBA-Bench simulation."""
    event_id: str
    timestamp: datetime
    
    def __post_init__(self):
        """Validate event data after initialization."""
        if not self.event_id:
            raise ValueError("Event ID cannot be empty")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("Timestamp must be a datetime object")


@dataclass
class TickEvent(BaseEvent):
    """
    Time advancement event published by SimulationOrchestrator.
    
    This is the heartbeat of the simulation - published at each time step
    to trigger all time-based processing across services.
    
    Attributes:
        event_id: Unique identifier for this tick event
        timestamp: When this tick occurred
        tick_number: Sequential tick counter starting from 0
        simulation_time: Logical simulation time (may differ from real time)
        metadata: Additional context for this tick
    """
    tick_number: int
    simulation_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        if self.tick_number < 0:
            raise ValueError("Tick number must be >= 0")
        if not isinstance(self.simulation_time, datetime):
            raise TypeError("Simulation time must be a datetime object")


@dataclass
class SaleOccurred(BaseEvent):
    """
    Sales transaction completion event published by SalesService.
    
    Contains comprehensive transaction details including financial data,
    product information, and market conditions at time of sale.
    
    Attributes:
        event_id: Unique identifier for this sale event
        timestamp: When the sale was processed
        asin: Product ASIN that was sold
        units_sold: Number of units sold in this transaction
        units_demanded: Number of units that were demanded (for conversion rate)
        unit_price: Price per unit at time of sale
        total_revenue: Total revenue from this sale (units_sold * unit_price)
        total_fees: Total fees deducted from revenue
        total_profit: Net profit after fees (revenue - fees - costs)
        cost_basis: Total cost basis for units sold
        trust_score_at_sale: Trust score of product at time of sale
        bsr_at_sale: Best Seller Rank at time of sale
        conversion_rate: Demand to sales conversion rate for this transaction
        fee_breakdown: Detailed breakdown of all fees applied
        market_conditions: Market state at time of sale
        customer_segment: Customer segment that made the purchase (if known)
    """
    asin: str
    units_sold: int
    units_demanded: int
    unit_price: Money
    total_revenue: Money
    total_fees: Money
    total_profit: Money
    cost_basis: Money
    trust_score_at_sale: float
    bsr_at_sale: int
    conversion_rate: float
    fee_breakdown: Dict[str, Money] = field(default_factory=dict)
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    customer_segment: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate ASIN
        if not self.asin or not isinstance(self.asin, str):
            raise ValueError("ASIN must be a non-empty string")
        
        # Validate units
        if self.units_sold < 0:
            raise ValueError("Units sold must be >= 0")
        if self.units_demanded < 0:
            raise ValueError("Units demanded must be >= 0")
        if self.units_sold > self.units_demanded:
            raise ValueError("Units sold cannot exceed units demanded")
        
        # Validate Money types
        money_fields = ['unit_price', 'total_revenue', 'total_fees', 'total_profit', 'cost_basis']
        for field_name in money_fields:
            value = getattr(self, field_name)
            if not isinstance(value, Money):
                raise TypeError(f"{field_name} must be a Money instance, got {type(value)}")
        
        # Validate fee breakdown contains Money types
        for fee_type, amount in self.fee_breakdown.items():
            if not isinstance(amount, Money):
                raise TypeError(f"Fee breakdown '{fee_type}' must be Money type, got {type(amount)}")
        
        # Validate trust score
        if not 0.0 <= self.trust_score_at_sale <= 1.0:
            raise ValueError("Trust score must be between 0.0 and 1.0")
        
        # Validate BSR
        if self.bsr_at_sale < 1:
            raise ValueError("BSR must be >= 1")
        
        # Validate conversion rate
        if not 0.0 <= self.conversion_rate <= 1.0:
            raise ValueError("Conversion rate must be between 0.0 and 1.0")
        
        # Financial consistency checks
        if self.units_sold > 0:
            expected_revenue = self.unit_price * self.units_sold
            if abs(self.total_revenue.cents - expected_revenue.cents) > 1:  # Allow 1 cent rounding
                raise ValueError(f"Revenue mismatch: expected {expected_revenue}, got {self.total_revenue}")
        
        # Calculate conversion rate if not provided and units_demanded > 0
        if self.conversion_rate == 0.0 and self.units_demanded > 0:
            object.__setattr__(self, 'conversion_rate', self.units_sold / self.units_demanded)
    
    def get_profit_margin_percentage(self) -> float:
        """Calculate profit margin as percentage of revenue."""
        if self.total_revenue.cents == 0:
            return 0.0
        return (self.total_profit.cents / self.total_revenue.cents) * 100
    
    def get_fee_percentage(self) -> float:
        """Calculate total fees as percentage of revenue."""
        if self.total_revenue.cents == 0:
            return 0.0
        return (self.total_fees.cents / self.total_revenue.cents) * 100
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'asin': self.asin,
            'units_sold': self.units_sold,
            'units_demanded': self.units_demanded,
            'conversion_rate': round(self.conversion_rate, 3),
            'unit_price': str(self.unit_price),
            'total_revenue': str(self.total_revenue),
            'total_fees': str(self.total_fees),
            'total_profit': str(self.total_profit),
            'profit_margin_pct': round(self.get_profit_margin_percentage(), 2),
            'fee_percentage': round(self.get_fee_percentage(), 2),
            'trust_score_at_sale': round(self.trust_score_at_sale, 3),
            'bsr_at_sale': self.bsr_at_sale,
            'customer_segment': self.customer_segment
        }


@dataclass
class SetPriceCommand(BaseEvent):
    """
    Agent command to change product price.
    
    Agents publish this command to express their intent to change a product's price.
    The WorldStore will arbitrate conflicts and apply valid changes.
    
    Attributes:
        event_id: Unique identifier for this command
        timestamp: When the command was issued
        agent_id: Unique identifier of the agent issuing the command
        asin: Product ASIN to update
        new_price: Requested new price for the product
        reason: Optional reason for the price change (for auditing)
    """
    agent_id: str
    asin: str
    new_price: Money
    reason: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate agent_id
        if not self.agent_id or not isinstance(self.agent_id, str):
            raise ValueError("Agent ID must be a non-empty string")
        
        # Validate ASIN
        if not self.asin or not isinstance(self.asin, str):
            raise ValueError("ASIN must be a non-empty string")
        
        # Validate Money type
        if not isinstance(self.new_price, Money):
            raise TypeError(f"New price must be Money type, got {type(self.new_price)}")
        
        # Validate price is positive
        if self.new_price.cents <= 0:
            raise ValueError(f"New price must be positive, got {self.new_price}")


@dataclass
class ProductPriceUpdated(BaseEvent):
    """
    Canonical product price update event published by WorldStore.
    
    Published after WorldStore processes SetPriceCommand and updates canonical state.
    All services should use this as the source of truth for product prices.
    
    Attributes:
        event_id: Unique identifier for this price update event
        timestamp: When the price was officially updated
        asin: Product ASIN that was updated
        new_price: The canonical new price (post-arbitration)
        previous_price: The previous canonical price
        agent_id: Agent that triggered this price change (if any)
        command_id: The command event ID that caused this update (if any)
        arbitration_notes: Optional notes about conflict resolution
    """
    asin: str
    new_price: Money
    previous_price: Money
    agent_id: Optional[str] = None
    command_id: Optional[str] = None
    arbitration_notes: Optional[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate ASIN
        if not self.asin or not isinstance(self.asin, str):
            raise ValueError("ASIN must be a non-empty string")
        
        # Validate Money types
        if not isinstance(self.new_price, Money):
            raise TypeError(f"New price must be Money type, got {type(self.new_price)}")
        if not isinstance(self.previous_price, Money):
            raise TypeError(f"Previous price must be Money type, got {type(self.previous_price)}")
        
        # Validate prices are positive
        if self.new_price.cents <= 0:
            raise ValueError(f"New price must be positive, got {self.new_price}")
        if self.previous_price.cents <= 0:
            raise ValueError(f"Previous price must be positive, got {self.previous_price}")
    
    def get_price_change_percentage(self) -> float:
        """Calculate price change as percentage."""
        if self.previous_price.cents == 0:
            return 0.0
        return ((self.new_price.cents - self.previous_price.cents) / self.previous_price.cents) * 100
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'asin': self.asin,
            'new_price': str(self.new_price),
            'previous_price': str(self.previous_price),
            'price_change_pct': round(self.get_price_change_percentage(), 2),
            'agent_id': self.agent_id,
            'command_id': self.command_id,
            'arbitration_notes': self.arbitration_notes
        }


@dataclass
class CompetitorPricesUpdated(BaseEvent):
    """
    Market competitive landscape update event published by CompetitorManager.
    
    Published after each TickEvent when competitors update their pricing and metrics.
    Contains complete competitive state snapshot for demand calculations.
    
    Attributes:
        event_id: Unique identifier for this competitor update event
        timestamp: When competitor updates were processed
        tick_number: Associated tick number that triggered this update
        competitors: List of current competitor states (price, BSR, sales velocity)
        market_summary: Optional aggregated market metrics for quick analysis
    """
    tick_number: int
    competitors: List[CompetitorState] = field(default_factory=list)
    market_summary: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate tick number
        if self.tick_number < 0:
            raise ValueError("Tick number must be >= 0")
        
        # Validate competitors list contains CompetitorState objects
        for i, competitor in enumerate(self.competitors):
            if not isinstance(competitor, CompetitorState):
                raise TypeError(f"competitors[{i}] must be CompetitorState instance, got {type(competitor)}")
        
        # Validate no duplicate ASINs
        asins = [comp.asin for comp in self.competitors]
        if len(asins) != len(set(asins)):
            raise ValueError("Duplicate competitor ASINs found in competitors list")
    
    def get_competitor_by_asin(self, asin: str) -> Optional[CompetitorState]:
        """Get competitor state by ASIN."""
        for competitor in self.competitors:
            if competitor.asin == asin:
                return competitor
        return None
    
    def get_average_competitor_price(self) -> Optional[Money]:
        """Calculate average competitor price."""
        if not self.competitors:
            return None
        total_cents = sum(comp.price.cents for comp in self.competitors)
        return Money(total_cents // len(self.competitors))
    
    def get_price_range(self) -> Optional[Tuple[Money, Money]]:
        """Get min and max competitor prices."""
        if not self.competitors:
            return None
        prices = [comp.price for comp in self.competitors]
        return min(prices), max(prices)
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'tick_number': self.tick_number,
            'competitor_count': len(self.competitors),
            'competitors': [
                {
                    'asin': comp.asin,
                    'price': str(comp.price),
                    'bsr': comp.bsr,
                    'sales_velocity': comp.sales_velocity
                }
                for comp in self.competitors
            ],
            'average_price': str(self.get_average_competitor_price()) if self.get_average_competitor_price() else None,
            'price_range': [str(p) for p in self.get_price_range()] if self.get_price_range() else None,
            'market_summary': self.market_summary
        }


# Event type registry for serialization/deserialization
EVENT_TYPES = {
    'TickEvent': TickEvent,
    'SaleOccurred': SaleOccurred,
    'CompetitorPricesUpdated': CompetitorPricesUpdated,
    'SetPriceCommand': SetPriceCommand,
    'ProductPriceUpdated': ProductPriceUpdated,
}


def get_event_type(event_type_name: str) -> type:
    """Get event class by name."""
    if event_type_name not in EVENT_TYPES:
        raise ValueError(f"Unknown event type: {event_type_name}")
    return EVENT_TYPES[event_type_name]