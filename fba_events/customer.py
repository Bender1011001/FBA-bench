"""
Event definitions related to customer interactions in the FBA-Bench simulation.

This module defines events that track various customer-initiated
communications and feedback, including general messages, negative
reviews, and formal complaints. These events are crucial for enabling
agents to provide customer service, manage reputation, and respond
to critical issues. It also includes `RespondToCustomerMessageCommand`,
an agent-issued command to reply to customer inquiries.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from .base import BaseEvent

@dataclass
class CustomerMessageReceived(BaseEvent):
    """
    Signals that a customer message has been received in the simulation.
    
    This event is generated when a simulated customer sends an inquiry,
    provides feedback, or initiates any form of communication. It serves
    as a trigger for customer service skills within agents.
    
    Attributes:
        event_id (str): Unique identifier for this customer message event. Inherited from `BaseEvent`.
        timestamp (datetime): When the customer message was received. Inherited from `BaseEvent`.
        customer_id (str): The unique identifier of the customer who sent the message.
        message_type (str): The category of the message, e.g., "inquiry", "complaint",
                            "return_request", "feedback", "support_ticket".
        content (str): The full textual content of the customer's message.
                       Can be analyzed for sentiment and keywords.
        sentiment_score (float): An analyzed sentiment score for the message content,
                                 typically ranging from -1.0 (very negative) to 1.0 (very positive).
                                 Defaults to 0.0 (neutral).
        priority_level (str): The assessed priority for responding to or addressing the message,
                              e.g., "low", "medium", "high", "urgent". Defaults to "medium".
        related_asin (Optional[str]): The product ASIN this message pertains to, if it's product-related.
        response_required (bool): `True` if an agent response is expected or required for this message.
        escalation_needed (bool): `True` if the message warrants immediate escalation to a higher-level
                                  agent or a specialized skill due to severity or complexity.
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
        """
        Validates the attributes of the `CustomerMessageReceived` event upon initialization.
        Ensures customer ID, message type, and content are provided,
        and sentiment/priority values are within valid ranges/sets.
        """
        super().__post_init__() # Call base class validation
        
        # Validate customer_id: Must be a non-empty string.
        if not self.customer_id:
            raise ValueError("Customer ID cannot be empty for CustomerMessageReceived event.")
        
        # Validate message_type: Must be a non-empty string.
        if not self.message_type:
            raise ValueError("Message type cannot be empty for CustomerMessageReceived event.")
        
        # Validate content: Must be a non-empty string.
        if not self.content:
            raise ValueError("Message content cannot be empty for CustomerMessageReceived event.")
        
        # Validate sentiment_score: Must be within the range [-1.0, 1.0].
        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError(f"Sentiment score must be between -1.0 and 1.0, but got {self.sentiment_score} for CustomerMessageReceived event.")
        
        # Validate priority_level: Must be one of the predefined categories.
        if self.priority_level not in ["low", "medium", "high", "urgent"]:
            raise ValueError(f"Priority level must be 'low', 'medium', 'high', or 'urgent', but got '{self.priority_level}' for CustomerMessageReceived event.")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Converts the `CustomerMessageReceived` event into a concise summary dictionary.
        Content length, sentiment, and priority are included for quick overview.
        """
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'customer_id': self.customer_id,
            'message_type': self.message_type,
            'content_length': len(self.content), # Summarize content by its length
            'sentiment_score': round(self.sentiment_score, 3),
            'priority_level': self.priority_level,
            'related_asin': self.related_asin,
            'response_required': self.response_required,
            'escalation_needed': self.escalation_needed
        }

@dataclass
class NegativeReviewEvent(BaseEvent):
    """
    Signals that a negative customer review has been posted for a product.
    
    This event allows agents to react to negative feedback, protect brand
    reputation, and address product or service issues. It's often consumed
    by skills like `CustomerServiceSkill` or `BrandManagerSkill`.
    
    Attributes:
        event_id (str): Unique identifier for this review event. Inherited from `BaseEvent`.
        timestamp (datetime): When the negative review was posted. Inherited from `BaseEvent`.
        review_id (str): Unique identifier for the specific review.
        customer_id (str): The ID of the customer who submitted the review.
        asin (str): The product ASIN that received the negative review.
        rating (int): The star rating given (1-5), typically 1, 2, or 3 for negative reviews.
        review_content (str): The full text content of the review.
        sentiment_score (float): An analyzed sentiment score (-1.0 to 1.0) of the review text.
                                 Defaults to -0.5 (slightly negative).
        impact_score (float): A calculated score (0.0 to 1.0) representing the potential negative
                               impact on sales or brand reputation based on the review's content and rating.
                               Defaults to 0.5 (medium impact).
        response_needed (bool): `True` if a response from an agent is recommended or required for this review.
                                Defaults to `True`.
        escalation_required (bool): `True` if the review warrants immediate escalation to a higher-level
                                    agent or specialized department (e.g., legal, product development)
                                    due to a critical issue. Defaults to `False`.
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
        """
        Validates the attributes of the `NegativeReviewEvent` upon initialization.
        Ensures IDs, ASIN, content are provided, and scores/rating are within valid ranges.
        """
        super().__post_init__() # Call base class validation
        
        # Validate review_id: Must be a non-empty string.
        if not self.review_id:
            raise ValueError("Review ID cannot be empty for NegativeReviewEvent.")
        
        # Validate customer_id: Must be a non-empty string.
        if not self.customer_id:
            raise ValueError("Customer ID cannot be empty for NegativeReviewEvent.")
        
        # Validate asin: Must be a non-empty string.
        if not self.asin:
            raise ValueError("ASIN cannot be empty for NegativeReviewEvent.")
        
        # Validate rating: Must be an integer between 1 and 5.
        if not 1 <= self.rating <= 5:
            raise ValueError(f"Rating must be between 1 and 5, but got {self.rating} for NegativeReviewEvent.")
        
        # Validate review_content: Must be a non-empty string.
        if not self.review_content:
            raise ValueError("Review content cannot be empty for NegativeReviewEvent.")
        
        # Validate sentiment_score: Must be within the range [-1.0, 1.0].
        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError(f"Sentiment score must be between -1.0 and 1.0, but got {self.sentiment_score} for NegativeReviewEvent.")
        
        # Validate impact_score: Must be within the range [0.0, 1.0].
        if not 0.0 <= self.impact_score <= 1.0:
            raise ValueError(f"Impact score must be between 0.0 and 1.0, but got {self.impact_score} for NegativeReviewEvent.")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Converts the `NegativeReviewEvent` into a concise summary dictionary.
        Review length, sentiment, and impact scores are included.
        """
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'review_id': self.review_id,
            'customer_id': self.customer_id,
            'asin': self.asin,
            'rating': self.rating,
            'review_length': len(self.review_content), # Summarize content by its length
            'sentiment_score': round(self.sentiment_score, 3),
            'impact_score': round(self.impact_score, 3),
            'response_needed': self.response_needed,
            'escalation_required': self.escalation_required
        }

@dataclass
class ComplaintEvent(BaseEvent):
    """
    Signals the occurrence of a formal customer complaint.
    
    This event is used for more serious customer issues that require structured
    tracking and resolution. It often triggers dedicated complaint handling
    workflows within agent operations.
    
    Attributes:
        event_id (str): Unique identifier for this complaint event. Inherited from `BaseEvent`.
        timestamp (datetime): When the complaint was received/recorded. Inherited from `BaseEvent`.
        complaint_id (str): Unique identifier for this specific complaint.
        customer_id (str): The ID of the complaining customer.
        complaint_type (str): The category of the complaint, e.g., "product_defect", "shipping_issue",
                              "billing_error", "customer_service_dispute".
        severity (str): The severity level of the complaint,
                        e.g., "low", "medium", "high", "critical".
        description (str): A detailed textual description of the complaint.
        related_asin (Optional[str]): The product ASIN if the complaint is product-related.
        related_order_id (Optional[str]): The order ID if the complaint is order-related.
        resolution_deadline (Optional[datetime]): A target date/time by which the complaint should be resolved.
                                                Can be set by a `CustomerServiceManagerSkill`.
        escalation_level (int): The current escalation level of the complaint (e.g., 1=initial, 3=executive review).
                                Must be at least 1.
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
        """
        Validates the attributes of the `ComplaintEvent` upon initialization.
        Ensures complaint ID, customer ID, type, severity, and description are provided,
        and escalation level is valid.
        """
        super().__post_init__() # Call base class validation
        
        # Validate complaint_id: Must be a non-empty string.
        if not self.complaint_id:
            raise ValueError("Complaint ID cannot be empty for ComplaintEvent.")
        
        # Validate customer_id: Must be a non-empty string.
        if not self.customer_id:
            raise ValueError("Customer ID cannot be empty for ComplaintEvent.")
        
        # Validate complaint_type: Must be a non-empty string.
        if not self.complaint_type:
            raise ValueError("Complaint type cannot be empty for ComplaintEvent.")
        
        # Validate severity: Must be one of the predefined categories.
        if self.severity not in ["low", "medium", "high", "critical"]:
            raise ValueError(f"Severity must be 'low', 'medium', 'high', or 'critical', but got '{self.severity}' for ComplaintEvent.")
        
        # Validate description: Must be a non-empty string.
        if not self.description:
            raise ValueError("Description cannot be empty for ComplaintEvent.")
        
        # Validate escalation_level: Must be at least 1.
        if self.escalation_level < 1:
            raise ValueError(f"Escalation level must be at least 1, but got {self.escalation_level} for ComplaintEvent.")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Converts the `ComplaintEvent` into a concise summary dictionary.
        Description length and key identifiers are included for quick overview.
        """
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'complaint_id': self.complaint_id,
            'customer_id': self.customer_id,
            'complaint_type': self.complaint_type,
            'severity': self.severity,
            'description_length': len(self.description), # Summarize description by its length
            'related_asin': self.related_asin,
            'related_order_id': self.related_order_id,
            'resolution_deadline': self.resolution_deadline.isoformat() if self.resolution_deadline else None,
            'escalation_level': self.escalation_level
        }

@dataclass
class RespondToCustomerMessageCommand(BaseEvent):
    """
    Represents an agent's command to send a response to a customer message.
    
    This command is issued by agents (e.g., a `CustomerServiceSkill`) who
    are tasked with engaging directly with customers. It specifies the message
    to respond to and the content of the agent's reply.
    
    Attributes:
        event_id (str): Unique identifier for this response command. Inherited from `BaseEvent`.
        timestamp (datetime): When the response command was issued by the agent. Inherited from `BaseEvent`.
        message_id (str): The ID of the original customer message to which this is a response.
        response_content (str): The full textual content of the agent's response to the customer.
        reason (Optional[str]): An optional, human-readable reason or justification for the response.
                                Useful for auditing and understanding agent behavior.
    """
    message_id: str
    response_content: str
    reason: Optional[str] = None

    def __post_init__(self):
        """
        Validates the attributes of the `RespondToCustomerMessageCommand` upon initialization.
        Ensures message ID and response content are provided.
        """
        super().__post_init__() # Call base class validation
        
        # Validate message_id: Must be a non-empty string.
        if not self.message_id:
            raise ValueError("Message ID cannot be empty for RespondToCustomerMessageCommand.")
        
        # Validate response_content: Must be a non-empty string.
        if not self.response_content:
            raise ValueError("Response content cannot be empty for RespondToCustomerMessageCommand.")

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Converts the `RespondToCustomerMessageCommand` into a concise summary dictionary.
        """
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': getattr(self, 'agent_id', 'N/A'), # Assuming agent_id might be set by publisher or agent itself
            'message_id': self.message_id,
            'response_content': self.response_content,
            'reason': self.reason
        }

@dataclass
class CustomerReviewEvent(BaseEvent):
    """
    Event representing a customer review posted for a product.

    Attributes:
        event_id (str): Unique identifier for the review event.
        timestamp (datetime): When the review was created.
        asin (str): The product ASIN being reviewed.
        rating (int): Star rating 1-5.
        comment (str): Free-text review comment.
    """
    asin: str
    rating: int
    comment: str

    def __post_init__(self):
        super().__post_init__()
        if not self.asin or not isinstance(self.asin, str):
            raise ValueError("CustomerReviewEvent.asin must be a non-empty string")
        if not isinstance(self.rating, int) or not (1 <= self.rating <= 5):
            raise ValueError("CustomerReviewEvent.rating must be an integer in [1,5]")
        if self.comment is None:
            self.comment = ""

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'asin': self.asin,
            'rating': self.rating,
            'comment_len': len(self.comment),
        }

@dataclass
class RespondToReviewCommand(BaseEvent):
    """
    Command issued by an agent to respond to a specific customer review.

    Attributes:
        event_id (str): Unique identifier for the command.
        timestamp (datetime): When the command was issued.
        review_id (str): Identifier of the review to respond to.
        asin (str): The ASIN of the product the review refers to.
        response_content (str): The response content authored by the agent.
    """
    review_id: str
    asin: str
    response_content: str

    def __post_init__(self):
        super().__post_init__()
        if not self.review_id:
            raise ValueError("RespondToReviewCommand.review_id cannot be empty")
        if not self.asin:
            raise ValueError("RespondToReviewCommand.asin cannot be empty")
        if not self.response_content:
            raise ValueError("RespondToReviewCommand.response_content cannot be empty")

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': getattr(self, 'agent_id', 'N/A'),
            'review_id': self.review_id,
            'asin': self.asin,
            'response_content': self.response_content,
        }
