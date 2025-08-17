"""
Central registry and lookup utilities for all event types in the FBA-Bench simulation.

This module is responsible for importing all defined `BaseEvent` subclasses
from their respective modules within the `fba_events` package and registering
them in a centralized `EVENT_TYPES` dictionary. This dictionary allows for
dynamic lookup of event classes by their string names, which is crucial for
event serialization, deserialization, and dynamic event handling across the system.

The `__all__` variable is also defined here to control what symbols are
exported when `fba_events` is imported using `from fba_events import *`.
"""
from __future__ import annotations
from .base import BaseEvent
from .time_events import TickEvent
from .competitor import CompetitorState, CompetitorPricesUpdated
from .sales import SaleOccurred
from .pricing import SetPriceCommand, ProductPriceUpdated
from .inventory import InventoryUpdate, LowInventoryEvent, WorldStateSnapshotEvent
from .budget import BudgetWarning, BudgetExceeded, ConstraintViolation
from .adversarial import AdversarialEvent, PhishingEvent, MarketManipulationEvent, ComplianceTrapEvent, AdversarialResponse
from .skills import SkillActivated, SkillActionGenerated, SkillConflictDetected, MultiDomainDecisionMade
from .agent import AgentDecisionEvent
from .customer import CustomerMessageReceived, NegativeReviewEvent, ComplaintEvent, RespondToCustomerMessageCommand, CustomerReviewEvent, RespondToReviewCommand
from .supplier import SupplierResponseEvent, PlaceOrderCommand
from .marketing import MarketTrendEvent, RunMarketingCampaignCommand, AdSpendEvent
from .reporting import ProfitReport, LossEvent

# Define __all__ to control what's exposed when doing 'from fba_events import *'
__all__ = [
    "BaseEvent",
    "TickEvent",
    "CompetitorState",
    "CompetitorPricesUpdated",
    "SaleOccurred",
    "SetPriceCommand",
    "ProductPriceUpdated",
    "InventoryUpdate",
    "LowInventoryEvent",
    "WorldStateSnapshotEvent",
    "BudgetWarning",
    "BudgetExceeded",
    "ConstraintViolation",
    "AdversarialEvent",
    "PhishingEvent",
    "MarketManipulationEvent",
    "ComplianceTrapEvent",
    "AdversarialResponse",
    "SkillActivated",
    "SkillActionGenerated",
    "SkillConflictDetected",
    "MultiDomainDecisionMade",
    "AgentDecisionEvent",
    "CustomerMessageReceived",
    "NegativeReviewEvent",
    "ComplaintEvent",
    "RespondToCustomerMessageCommand",
    "CustomerReviewEvent",
    "RespondToReviewCommand",
    "SupplierResponseEvent",
    "PlaceOrderCommand",
    "MarketTrendEvent",
    "RunMarketingCampaignCommand",
    "AdSpendEvent",
    "ProfitReport",
    "LossEvent",
    'EVENT_TYPES',      # Explicitly export the registry itself
    'get_event_type',   # Explicitly export the lookup function
]

# Central registry mapping event names (strings) to their corresponding dataclass types.
# This dictionary is used throughout the FBA-Bench system for dynamic event instantiation
# based on type names received from queues, databases, or configuration files.
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
    # Agent Commands
    'PlaceOrderCommand': PlaceOrderCommand,
    'RespondToCustomerMessageCommand': RespondToCustomerMessageCommand,
    'CustomerReviewEvent': CustomerReviewEvent,
    'RespondToReviewCommand': RespondToReviewCommand,
    'RunMarketingCampaignCommand': RunMarketingCampaignCommand,
    'AdSpendEvent': AdSpendEvent,
    # Specific Domain Events
    'WorldStateSnapshotEvent': WorldStateSnapshotEvent,
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
    """
    Retrieves an event class (dataclass type) from the central registry by its string name.
    
    This function is essential for dynamically creating event instances
    when only the event's string identifier is known (e.g., from a message queue).
    
    Args:
        event_type_name (str): The string name of the event class to retrieve
                               (e.g., "TickEvent", "SaleOccurred").
                               This name must exist as a key in the `EVENT_TYPES` dictionary.
                               
    Returns:
        type: The actual dataclass type corresponding to the `event_type_name`.
        
    Raises:
        ValueError: If `event_type_name` is not found in the `EVENT_TYPES` registry.
    """
    if event_type_name not in EVENT_TYPES:
        raise ValueError(f"Unknown event type: {event_type_name}. Please check the fba_events.EVENT_TYPES registry.")
    return EVENT_TYPES[event_type_name]
