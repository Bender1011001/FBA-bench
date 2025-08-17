from __future__ import annotations

"""
Centralized shared types for FBA-Bench.

This module provides a single import location for common data structures and
typing interfaces used across agent runners, unified agents, and the event bus.
It is intentionally light to avoid circular dependencies and heavy import-time
side effects.

Exports:
- SimulationState, ToolCall: canonical simulation types used by runners
- AgentObservation: observation type used by unified agents
- TickEvent, SetPriceCommand: structural protocols for event types to break cycles

Design:
- Runtime classes are re-exported from their source modules where possible.
- Event types are provided as Protocols capturing attributes used by dependents,
  allowing code to type-check without importing full event implementations.

Usage:
  from fba_bench.core.types import SimulationState, ToolCall, AgentObservation, TickEvent, SetPriceCommand
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, TYPE_CHECKING

# Prefer importing concrete classes from their home modules at runtime.
# If those modules are unavailable in a specific environment, fall back to
# minimal structural definitions to ensure imports don't fail. These fallbacks
# are only used when the source modules cannot be imported.

# Re-export SimulationState and ToolCall from agent_runners.base_runner if available
try:
    from agent_runners.base_runner import SimulationState as _BR_SimulationState  # type: ignore
    from agent_runners.base_runner import ToolCall as _BR_ToolCall  # type: ignore

    SimulationState = _BR_SimulationState
    ToolCall = _BR_ToolCall
except Exception:
    @dataclass
    class SimulationState:  # Fallback structural type
        tick: int = 0
        simulation_time: Optional[datetime] = None
        products: List[Any] = None  # type: ignore
        recent_events: List[Dict[str, Any]] = None  # type: ignore
        financial_position: Dict[str, Any] = None  # type: ignore
        market_conditions: Dict[str, Any] = None  # type: ignore
        agent_state: Dict[str, Any] = None  # type: ignore

        def __post_init__(self):
            if self.products is None:
                self.products = []
            if self.recent_events is None:
                self.recent_events = []
            if self.financial_position is None:
                self.financial_position = {}
            if self.market_conditions is None:
                self.market_conditions = {}
            if self.agent_state is None:
                self.agent_state = {}

    @dataclass
    class ToolCall:  # Fallback structural type
        tool_name: str
        parameters: Dict[str, Any]
        confidence: float = 1.0
        reasoning: Optional[str] = None
        priority: int = 0


# Re-export AgentObservation from unified agent if available
try:
    from benchmarking.agents.unified_agent import AgentObservation as _UA_AgentObservation  # type: ignore

    AgentObservation = _UA_AgentObservation
except Exception:
    @dataclass
    class AgentObservation:  # Fallback structural type
        observation_type: str
        data: Dict[str, Any]
        timestamp: datetime = datetime.now()
        source: Optional[str] = None


# Structural protocols for events to avoid importing the full fba_events registry.
@runtime_checkable
class TickEvent(Protocol):
    tick_number: int


@runtime_checkable
class SetPriceCommand(Protocol):
    event_id: str
    agent_id: str
    asin: str
    price: float


@runtime_checkable
class PlaceOrderCommand(Protocol):
    """Structural protocol for supply chain order events."""
    event_id: str
    agent_id: str
    supplier_id: str
    asin: str
    quantity: int


@runtime_checkable
class RunMarketingCampaignCommand(Protocol):
    """Structural protocol for marketing campaign command."""
    event_id: str
    timestamp: datetime
    campaign_type: str
    budget: Any  # Money
    duration_days: int


@runtime_checkable
class AdSpendEvent(Protocol):
    """Structural protocol for ad spend event."""
    event_id: str
    timestamp: datetime
    asin: str
    campaign_id: str
    spend: Any  # Money
    clicks: int
    impressions: int

@runtime_checkable
class CustomerReviewEvent(Protocol):
    """Structural protocol for a customer product review event."""
    event_id: str
    timestamp: datetime
    asin: str
    rating: int
    comment: str

@runtime_checkable
class RespondToReviewCommand(Protocol):
    """Structural protocol for a respond-to-review command."""
    event_id: str
    timestamp: datetime
    review_id: str
    asin: str
    response_content: str


__all__ = [
    "SimulationState",
    "ToolCall",
    "AgentObservation",
    "TickEvent",
    "SetPriceCommand",
    "PlaceOrderCommand",
    "RunMarketingCampaignCommand",
    "AdSpendEvent",
    "CustomerReviewEvent",
    "RespondToReviewCommand",
]