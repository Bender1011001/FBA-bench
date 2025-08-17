"""
Base event and core event-related definitions for the FBA-Bench simulation.

This module provides the `BaseEvent` abstract class, which all concrete
event types throughout the simulation must inherit from. It establishes
fundamental properties and methods common to all events, ensuring a
consistent structure for event handling, traceability, and serialization.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod

@dataclass
class BaseEvent(ABC):
    """
    Abstract base class for all events within the FBA-Bench simulation.
    
    All concrete event types must inherit from this class, ensuring they have
    a unique `event_id` and a `timestamp` for traceability, and implement
    the `to_summary_dict` method for consistent data representation.

    Attributes:
        event_id (str): A unique identifier for this specific event instance.
        timestamp (datetime): The UTC datetime when the event occurred or was generated.
    """
    event_id: str
    timestamp: datetime
    
    def __post_init__(self):
        """
        Performs basic validation on common event attributes after initialization.
        Ensures `event_id` is not empty and `timestamp` is a `datetime` object.
        """
        # Validate event_id: Must be a non-empty string.
        if not self.event_id:
            raise ValueError("Event ID cannot be empty")
        # Validate timestamp: Must be a datetime object.
        if not isinstance(self.timestamp, datetime):
            raise TypeError("Timestamp must be a datetime object")

    @abstractmethod
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Converts the event instance into a summary dictionary.
        
        This method is crucial for logging, debugging, serialization, and
        external systems that need a standardized, concise representation
        of the event's key data. Implementations should convert complex
        objects (like Money or datetime) to string representations,
        and ensure that the output is JSON-serializable.

        Returns:
            Dict[str, Any]: A dictionary containing key attributes of the event.
        """
        raise NotImplementedError("Subclasses must implement to_summary_dict")
