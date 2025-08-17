"""
Event definitions related to supplier interactions in the FBA-Bench simulation.

This module defines `SupplierResponseEvent`, which captures responses from
simulated suppliers to agent queries or orders, and `PlaceOrderCommand`,
an agent-issued command to initiate a purchase from a supplier. These
events are critical for enabling agents to manage their supply chain
and ensure timely inventory replenishment.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from .base import BaseEvent
from money import Money # External dependency for precise financial calculations

@dataclass
class SupplierResponseEvent(BaseEvent):
    """
    Represents a response received from a simulated supplier.
    
    This event is published by a `SupplierService` or equivalent component
    in response to agent actions (e.g., `PlaceOrderCommand`, `RequestQuoteCommand`).
    It carries information regarding quotes, delivery updates, quality reports, etc.
    
    Attributes:
        event_id (str): Unique identifier for this supplier response event. Inherited from `BaseEvent`.
        timestamp (datetime): When the supplier response was received or processed. Inherited from `BaseEvent`.
        supplier_id (str): The unique identifier of the responding supplier.
        response_type (str): The category of the response, e.g., "quote", "delivery_update",
                             "quality_report", "order_confirmation", "out_of_stock".
        content (str): The free-form message content of the supplier's response (e.g., quote details, status message).
        order_id (Optional[str]): The ID of the related order, if this response pertains to a specific order.
        delivery_date (Optional[datetime]): The promised or estimated delivery date, if applicable.
        quoted_price (Optional[Money]): The quoted price for an item or order, if this is a quote response.
                                        Represented using the `Money` class.
        response_time_hours (float): The time taken for the supplier to respond in hours (simulated time).
    """
    supplier_id: str
    response_type: str
    content: str
    order_id: Optional[str] = None
    delivery_date: Optional[datetime] = None
    quoted_price: Optional[Money] = None
    response_time_hours: float = 0.0
    
    def __post_init__(self):
        """
        Validates the attributes of the `SupplierResponseEvent` upon initialization.
        Ensures supplier ID, response type, and content are provided.
        """
        super().__post_init__() # Call base class validation
        
        # Validate supplier_id: Must be a non-empty string.
        if not self.supplier_id:
            raise ValueError("Supplier ID cannot be empty for SupplierResponseEvent.")
        
        # Validate response_type: Must be a non-empty string.
        if not self.response_type:
            raise ValueError("Response type cannot be empty for SupplierResponseEvent.")
        
        # Validate content: Must be a non-empty string.
        if not self.content:
            raise ValueError("Response content cannot be empty for SupplierResponseEvent.")
        
        # Validate quoted_price: If provided, must be a Money instance.
        if self.quoted_price is not None and not isinstance(self.quoted_price, Money):
            raise TypeError(f"Quoted price must be a Money type if provided, but got {type(self.quoted_price)}.")
        
        # Validate response_time_hours: Must be non-negative.
        if self.response_time_hours < 0:
            raise ValueError(f"Response time (hours) must be non-negative, but got {self.response_time_hours}.")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Converts the `SupplierResponseEvent` into a concise summary dictionary.
        
        Content is truncated for brevity, and `Money`/`datetime` objects
        are converted to string representations for logging and serialization.
        """
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'supplier_id': self.supplier_id,
            'response_type': self.response_type,
            'content': self.content[:100] + "..." if len(self.content) > 100 else self.content, # Truncate long content
            'order_id': self.order_id,
            'delivery_date': self.delivery_date.isoformat() if self.delivery_date else None,
            'quoted_price': str(self.quoted_price) if self.quoted_price else None,
            'response_time_hours': round(self.response_time_hours, 2)
        }

@dataclass
class PlaceOrderCommand(BaseEvent):
    """
    Represents an agent's command to place an order with a supplier.
    
    This command is issued by agents (e.g., an `InventoryManagerSkill`) when
    they determine a need to replenish stock. It specifies the product, quantity,
    and a maximum acceptable price. This command can trigger `SupplierResponseEvent`s.
    
    Attributes:
        event_id (str): Unique identifier for this order command. Inherited from `BaseEvent`.
        timestamp (datetime): When the order command was issued by the agent. Inherited from `BaseEvent`.
        supplier_id (str): The unique identifier of the supplier to place the order with.
        asin (str): The Amazon Standard Identification Number of the product to order.
        quantity (int): The number of units to order. Must be a positive integer.
        max_price (Money): The maximum price per unit the agent is willing to pay for this order.
                           Represented using the `Money` class.
        reason (Optional[str]): An optional, human-readable reason or justification for placing the order.
                                Useful for auditing and understanding agent behavior.
    """
    supplier_id: str
    asin: str
    quantity: int
    max_price: Money
    reason: Optional[str] = None

    def __post_init__(self):
        """
        Validates the attributes of the `PlaceOrderCommand` upon initialization.
        Ensures supplier ID, ASIN, and quantity are provided, and max price is a valid `Money` object.
        """
        super().__post_init__() # Call base class validation
        
        # Validate supplier_id: Must be a non-empty string.
        if not self.supplier_id:
            raise ValueError("Supplier ID cannot be empty for PlaceOrderCommand.")
        
        # Validate ASIN: Must be a non-empty string.
        if not self.asin:
            raise ValueError("ASIN cannot be empty for PlaceOrderCommand.")
        
        # Validate quantity: Must be a positive integer.
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, but got {self.quantity}.")
        
        # Validate max_price: Must be a Money object.
        if not isinstance(self.max_price, Money):
            raise TypeError(f"Max price must be a Money object, but got {type(self.max_price)}.")

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Converts the `PlaceOrderCommand` into a concise summary dictionary.
        `Money` objects are converted to string representation.
        """
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': getattr(self, 'agent_id', 'N/A'), # Assuming agent_id might be set by publisher or agent itself
            'supplier_id': self.supplier_id,
            'asin': self.asin,
            'quantity': self.quantity,
            'max_price': str(self.max_price),
            'reason': self.reason
        }
