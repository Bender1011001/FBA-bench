"""Event schema definitions for FBA-Bench v3 event-driven architecture."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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

    @abstractmethod
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to a summary dictionary for logging or external use."""
        raise NotImplementedError("Subclasses must implement to_summary_dict")


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

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'tick_number': self.tick_number,
            'simulation_time': self.simulation_time.isoformat(),
            'metadata': self.metadata.copy()
        }


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
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'asin': self.asin,
            'new_price': str(self.new_price),
            'reason': self.reason
        }


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


@dataclass
class BudgetWarning(BaseEvent):
    """
    Event indicating a budget is nearing or has softly exceeded its limit.
    """
    agent_id: Optional[str]
    budget_type: str # e.g., "per_tick", "total_simulation"
    current_usage: int
    limit: int
    reason: str
    
    def __post_init__(self):
        super().__post_init__()
        if not self.budget_type:
            raise ValueError("Budget type cannot be empty")
        if self.current_usage < 0 or self.limit < 0:
            raise ValueError("Usage and limit must be non-negative")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'budget_type': self.budget_type,
            'current_usage': self.current_usage,
            'limit': self.limit,
            'reason': self.reason
        }

@dataclass
class BudgetExceeded(BaseEvent):
    """
    Event indicating a budget has been hard-exceeded, leading to termination.
    """
    agent_id: Optional[str]
    budget_type: str # e.g., "per_tick", "total_simulation"
    current_usage: int
    limit: int
    reason: str
    severity: str # "soft", "hard_fail"
    
    def __post_init__(self):
        super().__post_init__()
        if not self.budget_type:
            raise ValueError("Budget type cannot be empty")
        if self.current_usage < 0 or self.limit < 0:
            raise ValueError(f"Current usage ({self.current_usage}) and limit ({self.limit}) must be non-negative")
        if self.severity not in ["soft", "hard_fail"]:
            raise ValueError(f"Severity must be 'soft' or 'hard_fail', got {self.severity}")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'budget_type': self.budget_type,
            'current_usage': self.current_usage,
            'limit': self.limit,
            'reason': self.reason,
            'severity': self.severity
        }

@dataclass
class ConstraintViolation(BaseEvent):
    """
    General event for any constraint violation, could be budget or other types.
    """
    agent_id: Optional[str]
    constraint_type: str # e.g., "budget", "time", "action_rate"
    violation_details: Dict[str, Any]
    is_critical: bool # True for hard fails, False for warnings
    
    def __post_init__(self):
        super().__post_init__()
        if not self.constraint_type:
            raise ValueError("Constraint type cannot be empty")
        if not isinstance(self.violation_details, dict):
            raise TypeError("Violation details must be a dictionary")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'constraint_type': self.constraint_type,
            'violation_details': self.violation_details,
            'is_critical': self.is_critical
        }


@dataclass
class AdversarialEvent(BaseEvent):
    """
    Base class for all adversarial/red-team events in the simulation.
    
    These events test agent resistance to various exploit scenarios including
    phishing, social engineering, market manipulation, and compliance traps.
    
    Attributes:
        event_id: Unique identifier for this adversarial event
        timestamp: When this adversarial event was triggered
        exploit_type: Type of exploit (phishing, market_manipulation, etc.)
        difficulty_level: Difficulty rating from 1-5
        target_action: What the agent should NOT do if resistant
        deception_vector: How the attack attempts to deceive
        realistic_context: Supporting context to make attack believable
        success_criteria: Conditions that indicate agent fell for exploit
        time_window: How long the attack remains active (hours)
        financial_impact_limit: Maximum financial damage if agent falls for exploit
    """
    exploit_type: str
    difficulty_level: int
    target_action: str
    deception_vector: str
    realistic_context: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    time_window: int = 24  # hours
    financial_impact_limit: Optional[Money] = None
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate exploit type
        valid_exploit_types = [
            "phishing", "social_engineering", "market_manipulation",
            "compliance_trap", "financial_exploit", "information_warfare"
        ]
        if self.exploit_type not in valid_exploit_types:
            raise ValueError(f"Exploit type must be one of {valid_exploit_types}, got {self.exploit_type}")
        
        # Validate difficulty level
        if not 1 <= self.difficulty_level <= 5:
            raise ValueError("Difficulty level must be between 1 and 5")
        
        # Validate target action
        if not self.target_action or not isinstance(self.target_action, str):
            raise ValueError("Target action must be a non-empty string")
        
        # Validate deception vector
        if not self.deception_vector or not isinstance(self.deception_vector, str):
            raise ValueError("Deception vector must be a non-empty string")
        
        # Validate time window
        if self.time_window <= 0:
            raise ValueError("Time window must be positive")
        
        # Validate financial impact limit if provided
        if self.financial_impact_limit is not None and not isinstance(self.financial_impact_limit, Money):
            raise TypeError("Financial impact limit must be Money type if provided")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'exploit_type': self.exploit_type,
            'difficulty_level': self.difficulty_level,
            'target_action': self.target_action,
            'deception_vector': self.deception_vector,
            'time_window': self.time_window,
            'financial_impact_limit': str(self.financial_impact_limit) if self.financial_impact_limit else None
        }


@dataclass
class PhishingEvent(AdversarialEvent):
    """
    Phishing/social engineering adversarial event.
    
    Tests agent susceptibility to fraudulent communications that attempt
    to trick them into revealing sensitive information or taking harmful actions.
    
    Attributes:
        sender_email: Fake email address of the phishing attempt
        message_content: The deceptive message content
        requested_action: What the phishing email is asking the agent to do
        spoofed_domain: Domain being impersonated (if any)
        urgency_indicators: Phrases/elements designed to create urgency
    """
    sender_email: str = ""
    message_content: str = ""
    requested_action: str = ""
    spoofed_domain: Optional[str] = None
    urgency_indicators: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        # Set exploit_type for phishing events
        object.__setattr__(self, 'exploit_type', 'phishing')
        
        # Validate required fields
        if not self.sender_email or '@' not in self.sender_email:
            raise ValueError("Sender email must be a valid email format")
        if not self.message_content:
            raise ValueError("Message content cannot be empty")
        if not self.requested_action:
            raise ValueError("Requested action cannot be empty")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            **super().to_summary_dict(), # Include base adversarial event details
            'sender_email': self.sender_email,
            'requested_action': self.requested_action,
            'spoofed_domain': self.spoofed_domain
        }


@dataclass
class MarketManipulationEvent(AdversarialEvent):
    """
    Market manipulation adversarial event.
    
    Tests agent resistance to false market information designed to
    influence pricing, inventory, or strategic decisions.
    
    Attributes:
        false_data_type: Type of false data (pricing, sales, reviews, etc.)
        manipulated_values: The false values being presented
        source_credibility: How credible the source appears (1-5)
        market_impact_claim: Claimed impact on market conditions
    """
    false_data_type: str = ""
    manipulated_values: Dict[str, Any] = field(default_factory=dict)
    source_credibility: int = 3
    market_impact_claim: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        # Set exploit_type for market manipulation events
        object.__setattr__(self, 'exploit_type', 'market_manipulation')
        
        # Validate fields
        valid_data_types = ["pricing", "sales_data", "reviews", "competitor_intel", "demand_forecast"]
        if self.false_data_type not in valid_data_types:
            raise ValueError(f"False data type must be one of {valid_data_types}")
        
        if not 1 <= self.source_credibility <= 5:
            raise ValueError("Source credibility must be between 1 and 5")
        
        if not self.market_impact_claim:
            raise ValueError("Market impact claim cannot be empty")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            **super().to_summary_dict(), # Include base adversarial event details
            'false_data_type': self.false_data_type,
            'source_credibility': self.source_credibility
        }


@dataclass
class ComplianceTrapEvent(AdversarialEvent):
    """
    Compliance trap adversarial event.
    
    Tests agent susceptibility to fake policy changes or compliance
    requirements that look legitimate but are actually harmful.
    
    Attributes:
        fake_policy_name: Name of the fake policy/requirement
        policy_description: Description of the fake policy
        compliance_deadline: Fake deadline for compliance
        penalty_claim: Claimed penalties for non-compliance
        official_appearance: How official the communication appears (1-5)
    """
    fake_policy_name: str = ""
    policy_description: str = ""
    compliance_deadline: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=30))
    penalty_claim: str = ""
    official_appearance: int = 3
    
    def __post_init__(self):
        super().__post_init__()
        # Set exploit_type for compliance trap events
        object.__setattr__(self, 'exploit_type', 'compliance_trap')
        
        # Validate fields
        if not self.fake_policy_name:
            raise ValueError("Fake policy name cannot be empty")
        if not self.policy_description:
            raise ValueError("Policy description cannot be empty")
        if not isinstance(self.compliance_deadline, datetime):
            raise TypeError("Compliance deadline must be a datetime object")
        if not self.penalty_claim:
            raise ValueError("Penalty claim cannot be empty")
        if not 1 <= self.official_appearance <= 5:
            raise ValueError("Official appearance must be between 1 and 5")

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            **super().to_summary_dict(), # Include base adversarial event details
            'fake_policy_name': self.fake_policy_name,
            'compliance_deadline': self.compliance_deadline.isoformat(),
            'penalty_claim': self.penalty_claim
        }


@dataclass
class AdversarialResponse(BaseEvent):
    """
    Agent response to an adversarial event, used for ARS calculation.
    
    Tracks how an agent responded to an adversarial attack, including
    whether they fell for the exploit, detected it, or took protective action.
    
    Attributes:
        event_id: Unique identifier for this response event
        timestamp: When the response was recorded
        adversarial_event_id: ID of the adversarial event this responds to
        agent_id: ID of the agent that responded
        fell_for_exploit: Whether the agent fell for the adversarial attack
        detected_attack: Whether the agent detected the attack
        reported_attack: Whether the agent reported the attack
        protective_action_taken: What protective action was taken (if any)
        response_time_seconds: How long it took to respond
        financial_damage: Actual financial damage incurred (if any)
        exploit_difficulty: Difficulty level of the exploit (copied for analysis)
    """
    adversarial_event_id: str
    agent_id: str
    fell_for_exploit: bool
    detected_attack: bool
    reported_attack: bool
    protective_action_taken: Optional[str] = None
    response_time_seconds: float = 0.0
    financial_damage: Optional[Money] = None
    exploit_difficulty: int = 1
    
    def __post_init__(self):
        super().__post_init__()
        
        # Validate required fields
        if not self.adversarial_event_id:
            raise ValueError("Adversarial event ID cannot be empty")
        if not self.agent_id:
            raise ValueError("Agent ID cannot be empty")
        
        # Validate response time
        if self.response_time_seconds < 0:
            raise ValueError("Response time must be non-negative")
        
        # Validate exploit difficulty
        if not 1 <= self.exploit_difficulty <= 5:
            raise ValueError("Exploit difficulty must be between 1 and 5")
        
        # Validate financial damage if provided
        if self.financial_damage is not None and not isinstance(self.financial_damage, Money):
            raise TypeError("Financial damage must be Money type if provided")
        
        # Logic validation: can't both fall for exploit and detect it
        if self.fell_for_exploit and self.detected_attack:
            raise ValueError("Agent cannot both fall for exploit and detect it simultaneously")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'adversarial_event_id': self.adversarial_event_id,
            'agent_id': self.agent_id,
            'fell_for_exploit': self.fell_for_exploit,
            'detected_attack': self.detected_attack,
            'reported_attack': self.reported_attack,
            'protective_action_taken': self.protective_action_taken,
            'response_time_seconds': round(self.response_time_seconds, 2),
            'financial_damage': str(self.financial_damage) if self.financial_damage else None,
            'exploit_difficulty': self.exploit_difficulty
        }


@dataclass
class InventoryUpdate(BaseEvent):
    """
    Event indicating a change in product inventory quantity.

    This event is published by the WorldStore when the canonical inventory
    quantity of a product is updated, usually as a result of sales,
    returns, or inbound shipments.

    Attributes:
        event_id: Unique identifier for this inventory update event
        timestamp: When the inventory was officially updated
        asin: Product ASIN that was updated
        new_quantity: The canonical new inventory quantity
        previous_quantity: The previous canonical inventory quantity
        change_reason: Reason for the inventory change (e.g., "sale", "return", "shipment")
        agent_id: Agent that triggered this inventory change (if any)
        command_id: The command event ID that caused this update
    """
    asin: str
    new_quantity: int
    previous_quantity: int
    change_reason: str
    agent_id: Optional[str] = None
    command_id: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()

        # Validate ASIN
        if not self.asin or not isinstance(self.asin, str):
            raise ValueError("ASIN must be a non-empty string")

        # Validate quantities
        if self.new_quantity < 0 or self.previous_quantity < 0:
            raise ValueError("New and previous quantities must be non-negative")

        # Validate change reason
        if not self.change_reason or not isinstance(self.change_reason, str):
            raise ValueError("Change reason must be a non-empty string")

    def get_quantity_change(self) -> int:
        """Calculate the net change in quantity."""
        return self.new_quantity - self.previous_quantity

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'asin': self.asin,
            'new_quantity': self.new_quantity,
            'previous_quantity': self.previous_quantity,
            'quantity_change': self.get_quantity_change(),
            'change_reason': self.change_reason,
            'agent_id': self.agent_id,
            'command_id': self.command_id,
        }


@dataclass
class AgentDecisionEvent(BaseEvent):
    """
    Event representing a decision made by an agent.
    """
    agent_id: str
    turn: int
    tool_calls: List[Dict[str, Any]]
    simulation_time: datetime
    reasoning: str
    llm_usage: Dict[str, Any]
    prompt_metadata: Dict[str, Any]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'turn': self.turn,
            'tool_calls': self.tool_calls,
            'simulation_time': self.simulation_time.isoformat(),
            'reasoning': self.reasoning,
            'llm_usage': self.llm_usage,
            'prompt_metadata': self.prompt_metadata,
        }


# Skill Coordination Events

@dataclass
class SkillActivated(BaseEvent):
    """
    Event indicating a skill has been activated to handle an event.
    
    Attributes:
        skill_name: Name of the activated skill
        event_trigger: Type of event that triggered the skill
        priority_score: Priority score assigned to this activation
        agent_id: ID of the agent whose skill was activated
        activation_reason: Reason for skill activation
    """
    skill_name: str
    event_trigger: str
    priority_score: float
    agent_id: str
    activation_reason: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if not self.skill_name:
            raise ValueError("Skill name cannot be empty")
        if not 0.0 <= self.priority_score <= 1.0:
            raise ValueError("Priority score must be between 0.0 and 1.0")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'skill_name': self.skill_name,
            'event_trigger': self.event_trigger,
            'priority_score': round(self.priority_score, 3),
            'agent_id': self.agent_id,
            'activation_reason': self.activation_reason
        }


@dataclass
class SkillActionGenerated(BaseEvent):
    """
    Event indicating a skill has generated an action.
    
    Attributes:
        skill_name: Name of the skill that generated the action
        action_type: Type of action generated
        confidence_score: Confidence score for the action
        agent_id: ID of the agent whose skill generated the action
        action_parameters: Parameters of the generated action
        expected_outcome: Expected outcome of the action
        resource_requirements: Resource requirements for the action
    """
    skill_name: str
    action_type: str
    confidence_score: float
    agent_id: str
    action_parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        if not self.skill_name:
            raise ValueError("Skill name cannot be empty")
        if not self.action_type:
            raise ValueError("Action type cannot be empty")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'skill_name': self.skill_name,
            'action_type': self.action_type,
            'confidence_score': round(self.confidence_score, 3),
            'agent_id': self.agent_id,
            'action_parameters': self.action_parameters,
            'expected_outcome': self.expected_outcome,
            'resource_requirements': self.resource_requirements
        }


@dataclass
class SkillConflictDetected(BaseEvent):
    """
    Event indicating a conflict between multiple skills.
    
    Attributes:
        competing_skills: List of skills involved in the conflict
        conflict_type: Type of conflict (resource, action, priority, etc.)
        resolution: How the conflict was resolved
        agent_id: ID of the agent where conflict occurred
        conflict_details: Additional details about the conflict
        resolution_strategy: Strategy used to resolve the conflict
    """
    competing_skills: List[str]
    conflict_type: str
    resolution: str
    agent_id: str
    conflict_details: Dict[str, Any] = field(default_factory=dict)
    resolution_strategy: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        if len(self.competing_skills) < 2:
            raise ValueError("Must have at least 2 competing skills")
        if not self.conflict_type:
            raise ValueError("Conflict type cannot be empty")
        if not self.resolution:
            raise ValueError("Resolution cannot be empty")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'competing_skills': self.competing_skills,
            'conflict_type': self.conflict_type,
            'resolution': self.resolution,
            'agent_id': self.agent_id,
            'conflict_details': self.conflict_details,
            'resolution_strategy': self.resolution_strategy
        }


@dataclass
class MultiDomainDecisionMade(BaseEvent):
    """
    Event indicating a multi-domain decision has been made by the controller.
    
    Attributes:
        coordinated_actions: List of actions that were coordinated and approved
        reasoning: Strategic reasoning for the decision
        resource_allocation: Resource allocation decisions
        agent_id: ID of the agent that made the decision
        business_priority: Current business priority at time of decision
        strategic_alignment_score: Overall strategic alignment score
        rejected_actions_count: Number of actions that were rejected
    """
    coordinated_actions: List[Dict[str, Any]]
    reasoning: str
    resource_allocation: Dict[str, Any]
    agent_id: str
    business_priority: str = ""
    strategic_alignment_score: float = 0.0
    rejected_actions_count: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        if not self.reasoning:
            raise ValueError("Reasoning cannot be empty")
        if not isinstance(self.coordinated_actions, list):
            raise TypeError("Coordinated actions must be a list")
        if not isinstance(self.resource_allocation, dict):
            raise TypeError("Resource allocation must be a dictionary")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'coordinated_actions_count': len(self.coordinated_actions),
            'reasoning': self.reasoning,
            'resource_allocation': self.resource_allocation,
            'agent_id': self.agent_id,
            'business_priority': self.business_priority,
            'strategic_alignment_score': round(self.strategic_alignment_score, 3),
            'rejected_actions_count': self.rejected_actions_count
        }


# Domain-Specific Events for Skill Triggers

@dataclass
class LowInventoryEvent(BaseEvent):
    """
    Event indicating low inventory levels requiring attention.
    
    Attributes:
        asin: Product ASIN with low inventory
        current_level: Current inventory level
        reorder_point: Reorder point threshold
        days_remaining: Estimated days until stockout
        urgency_level: Urgency level (low, medium, high, critical)
        recommended_order_quantity: Recommended quantity to order
    """
    asin: str
    current_level: int
    reorder_point: int
    days_remaining: float
    urgency_level: str
    recommended_order_quantity: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        if not self.asin:
            raise ValueError("ASIN cannot be empty")
        if self.current_level < 0 or self.reorder_point < 0:
            raise ValueError("Inventory levels must be non-negative")
        if self.urgency_level not in ["low", "medium", "high", "critical"]:
            raise ValueError("Urgency level must be low, medium, high, or critical")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'asin': self.asin,
            'current_level': self.current_level,
            'reorder_point': self.reorder_point,
            'days_remaining': round(self.days_remaining, 1),
            'urgency_level': self.urgency_level,
            'recommended_order_quantity': self.recommended_order_quantity
        }


@dataclass
class SupplierResponseEvent(BaseEvent):
    """
    Event representing a response from a supplier.
    
    Attributes:
        supplier_id: ID of the responding supplier
        response_type: Type of response (quote, delivery_update, quality_report, etc.)
        content: Response content
        order_id: Related order ID if applicable
        delivery_date: Promised delivery date if applicable
        quoted_price: Quoted price if applicable
        response_time_hours: Time taken to respond in hours
    """
    supplier_id: str
    response_type: str
    content: str
    order_id: Optional[str] = None
    delivery_date: Optional[datetime] = None
    quoted_price: Optional[Money] = None
    response_time_hours: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        if not self.supplier_id:
            raise ValueError("Supplier ID cannot be empty")
        if not self.response_type:
            raise ValueError("Response type cannot be empty")
        if not self.content:
            raise ValueError("Response content cannot be empty")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'supplier_id': self.supplier_id,
            'response_type': self.response_type,
            'content': self.content[:100] + "..." if len(self.content) > 100 else self.content,
            'order_id': self.order_id,
            'delivery_date': self.delivery_date.isoformat() if self.delivery_date else None,
            'quoted_price': str(self.quoted_price) if self.quoted_price else None,
            'response_time_hours': round(self.response_time_hours, 2)
        }


@dataclass
class MarketTrendEvent(BaseEvent):
    """
    Event indicating a significant market trend or change.
    
    Attributes:
        trend_type: Type of trend (price, demand, competitor, seasonal, etc.)
        trend_direction: Direction of trend (up, down, stable)
        magnitude: Magnitude of the trend (0.0 to 1.0)
        affected_products: List of ASINs affected by the trend
        confidence_level: Confidence in the trend analysis
        duration_estimate: Estimated duration of the trend
        market_data: Supporting market data
    """
    trend_type: str
    trend_direction: str
    magnitude: float
    affected_products: List[str] = field(default_factory=list)
    confidence_level: float = 0.5
    duration_estimate: str = "unknown"
    market_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        if not self.trend_type:
            raise ValueError("Trend type cannot be empty")
        if self.trend_direction not in ["up", "down", "stable"]:
            raise ValueError("Trend direction must be up, down, or stable")
        if not 0.0 <= self.magnitude <= 1.0:
            raise ValueError("Magnitude must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence_level <= 1.0:
            raise ValueError("Confidence level must be between 0.0 and 1.0")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'trend_type': self.trend_type,
            'trend_direction': self.trend_direction,
            'magnitude': round(self.magnitude, 3),
            'affected_products': self.affected_products,
            'confidence_level': round(self.confidence_level, 3),
            'duration_estimate': self.duration_estimate,
            'market_data': self.market_data
        }


@dataclass
class CustomerMessageReceived(BaseEvent):
    """
    Event indicating a customer message has been received.
    
    Attributes:
        customer_id: ID of the customer
        message_type: Type of message (inquiry, complaint, return_request, etc.)
        content: Message content
        sentiment_score: Sentiment analysis score (-1.0 to 1.0)
        priority_level: Priority level for response
        related_asin: Product ASIN if message is product-related
        response_required: Whether a response is required
        escalation_needed: Whether escalation is needed
    """
    customer_id: str
    message_type: str
    content: str
    sentiment_score: float = 0.0
    priority_level: str = "medium"
    related_asin: Optional[str] = None
    response_required: bool = True
    escalation_needed: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if not self.customer_id:
            raise ValueError("Customer ID cannot be empty")
        if not self.message_type:
            raise ValueError("Message type cannot be empty")
        if not self.content:
            raise ValueError("Message content cannot be empty")
        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError("Sentiment score must be between -1.0 and 1.0")
        if self.priority_level not in ["low", "medium", "high", "urgent"]:
            raise ValueError("Priority level must be low, medium, high, or urgent")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'customer_id': self.customer_id,
            'message_type': self.message_type,
            'content_length': len(self.content),
            'sentiment_score': round(self.sentiment_score, 3),
            'priority_level': self.priority_level,
            'related_asin': self.related_asin,
            'response_required': self.response_required,
            'escalation_needed': self.escalation_needed
        }


@dataclass
class NegativeReviewEvent(BaseEvent):
    """
    Event indicating a negative review has been posted.
    
    Attributes:
        review_id: ID of the review
        customer_id: ID of the customer who posted the review
        asin: Product ASIN being reviewed
        rating: Star rating (1-5)
        review_content: Review text content
        sentiment_score: Sentiment analysis score
        impact_score: Potential impact on sales/reputation
        response_needed: Whether a response is needed
        escalation_required: Whether escalation is required
    """
    review_id: str
    customer_id: str
    asin: str
    rating: int
    review_content: str
    sentiment_score: float = -0.5
    impact_score: float = 0.5
    response_needed: bool = True
    escalation_required: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if not self.review_id:
            raise ValueError("Review ID cannot be empty")
        if not self.customer_id:
            raise ValueError("Customer ID cannot be empty")
        if not self.asin:
            raise ValueError("ASIN cannot be empty")
        if not 1 <= self.rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        if not self.review_content:
            raise ValueError("Review content cannot be empty")
        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError("Sentiment score must be between -1.0 and 1.0")
        if not 0.0 <= self.impact_score <= 1.0:
            raise ValueError("Impact score must be between 0.0 and 1.0")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'review_id': self.review_id,
            'customer_id': self.customer_id,
            'asin': self.asin,
            'rating': self.rating,
            'review_length': len(self.review_content),
            'sentiment_score': round(self.sentiment_score, 3),
            'impact_score': round(self.impact_score, 3),
            'response_needed': self.response_needed,
            'escalation_required': self.escalation_required
        }


@dataclass
class ComplaintEvent(BaseEvent):
    """
    Event indicating a formal customer complaint.
    
    Attributes:
        complaint_id: ID of the complaint
        customer_id: ID of the complaining customer
        complaint_type: Type of complaint
        severity: Severity level of the complaint
        description: Complaint description
        related_asin: Product ASIN if complaint is product-related
        related_order_id: Order ID if complaint is order-related
        resolution_deadline: Deadline for resolution
        escalation_level: Current escalation level
    """
    complaint_id: str
    customer_id: str
    complaint_type: str
    severity: str
    description: str
    related_asin: Optional[str] = None
    related_order_id: Optional[str] = None
    resolution_deadline: Optional[datetime] = None
    escalation_level: int = 1
    
    def __post_init__(self):
        super().__post_init__()
        if not self.complaint_id:
            raise ValueError("Complaint ID cannot be empty")
        if not self.customer_id:
            raise ValueError("Customer ID cannot be empty")
        if not self.complaint_type:
            raise ValueError("Complaint type cannot be empty")
        if self.severity not in ["low", "medium", "high", "critical"]:
            raise ValueError("Severity must be low, medium, high, or critical")
        if not self.description:
            raise ValueError("Description cannot be empty")
        if self.escalation_level < 1:
            raise ValueError("Escalation level must be at least 1")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'complaint_id': self.complaint_id,
            'customer_id': self.customer_id,
            'complaint_type': self.complaint_type,
            'severity': self.severity,
            'description_length': len(self.description),
            'related_asin': self.related_asin,
            'related_order_id': self.related_order_id,
            'resolution_deadline': self.resolution_deadline.isoformat() if self.resolution_deadline else None,
            'escalation_level': self.escalation_level
        }


@dataclass
class ProfitReport(BaseEvent):
    """
    Event containing profit/loss reporting information.
    
    Attributes:
        reporting_period: Period covered by the report
        total_revenue: Total revenue for the period
        total_expenses: Total expenses for the period
        net_profit: Net profit (revenue - expenses)
        profit_margin: Profit margin percentage
        product_breakdown: Profit breakdown by product
        expense_breakdown: Expense breakdown by category
        performance_vs_target: Performance compared to targets
    """
    reporting_period: str
    total_revenue: Money
    total_expenses: Money
    net_profit: Money
    profit_margin: float
    product_breakdown: Dict[str, Money] = field(default_factory=dict)
    expense_breakdown: Dict[str, Money] = field(default_factory=dict)
    performance_vs_target: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        super().__post_init__()
        if not self.reporting_period:
            raise ValueError("Reporting period cannot be empty")
        if not isinstance(self.total_revenue, Money):
            raise TypeError("Total revenue must be Money type")
        if not isinstance(self.total_expenses, Money):
            raise TypeError("Total expenses must be Money type")
        if not isinstance(self.net_profit, Money):
            raise TypeError("Net profit must be Money type")
        
        # Validate profit calculation
        calculated_profit = Money(self.total_revenue.cents - self.total_expenses.cents)
        if abs(self.net_profit.cents - calculated_profit.cents) > 1:  # Allow 1 cent rounding
            raise ValueError(f"Net profit calculation error: expected {calculated_profit}, got {self.net_profit}")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'reporting_period': self.reporting_period,
            'total_revenue': str(self.total_revenue),
            'total_expenses': str(self.total_expenses),
            'net_profit': str(self.net_profit),
            'profit_margin': round(self.profit_margin, 4),
            'product_count': len(self.product_breakdown),
            'expense_categories': len(self.expense_breakdown),
            'performance_metrics': self.performance_vs_target
        }


@dataclass
class LossEvent(BaseEvent):
    """
    Event indicating a significant financial loss or negative event.
    
    Attributes:
        loss_type: Type of loss (operational, one_time, writeoff, etc.)
        amount: Amount of the loss
        cause: Cause of the loss
        affected_area: Business area affected
        recovery_plan: Plan for recovery if any
        prevention_measures: Measures to prevent recurrence
        severity: Severity of the loss impact
        insurance_claim: Whether insurance claim is applicable
    """
    loss_type: str
    amount: Money
    cause: str
    affected_area: str
    recovery_plan: str = ""
    prevention_measures: List[str] = field(default_factory=list)
    severity: str = "medium"
    insurance_claim: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if not self.loss_type:
            raise ValueError("Loss type cannot be empty")
        if not isinstance(self.amount, Money):
            raise TypeError("Amount must be Money type")
        if self.amount.cents <= 0:
            raise ValueError("Loss amount must be positive")
        if not self.cause:
            raise ValueError("Cause cannot be empty")
        if not self.affected_area:
            raise ValueError("Affected area cannot be empty")
        if self.severity not in ["low", "medium", "high", "critical"]:
            raise ValueError("Severity must be low, medium, high, or critical")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'loss_type': self.loss_type,
            'amount': str(self.amount),
            'cause': self.cause,
            'affected_area': self.affected_area,
            'recovery_plan_provided': bool(self.recovery_plan),
            'prevention_measures_count': len(self.prevention_measures),
            'severity': self.severity,
            'insurance_claim': self.insurance_claim
        }

# Event type registry for serialization/deserialization
EVENT_TYPES = {
    'AgentDecisionEvent': AgentDecisionEvent,
    'TickEvent': TickEvent,
    'SaleOccurred': SaleOccurred,
    'CompetitorPricesUpdated': CompetitorPricesUpdated,
    'SetPriceCommand': SetPriceCommand,
    'ProductPriceUpdated': ProductPriceUpdated,
    'InventoryUpdate': InventoryUpdate,
    'BudgetWarning': BudgetWarning,
    'BudgetExceeded': BudgetExceeded,
    'ConstraintViolation': ConstraintViolation,
    # Adversarial/Red-team events
    'AdversarialEvent': AdversarialEvent,
    'PhishingEvent': PhishingEvent,
    'MarketManipulationEvent': MarketManipulationEvent,
    'ComplianceTrapEvent': ComplianceTrapEvent,
    'AdversarialResponse': AdversarialResponse,
    # Skill coordination events
    'SkillActivated': SkillActivated,
    'SkillActionGenerated': SkillActionGenerated,
    'SkillConflictDetected': SkillConflictDetected,
    'MultiDomainDecisionMade': MultiDomainDecisionMade,
    # Domain-specific events for skill triggers
    'LowInventoryEvent': LowInventoryEvent,
    'SupplierResponseEvent': SupplierResponseEvent,
    'MarketTrendEvent': MarketTrendEvent,
    'CustomerMessageReceived': CustomerMessageReceived,
    'NegativeReviewEvent': NegativeReviewEvent,
    'ComplaintEvent': ComplaintEvent,
    'ProfitReport': ProfitReport,
    'LossEvent': LossEvent,
}


def get_event_type(event_type_name: str) -> type:
    """Get event class by name."""
    if event_type_name not in EVENT_TYPES:
        raise ValueError(f"Unknown event type: {event_type_name}")
    return EVENT_TYPES[event_type_name]