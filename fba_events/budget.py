"""
Event definitions related to budget and constraint management in the FBA-Bench simulation.

This module defines events that signal budget-related statuses, such as
warnings for approaching limits (`BudgetWarning`), critical breaches
leading to termination (`BudgetExceeded`), and general constraint violations.
These events are crucial for monitoring simulation resource usage and for
enforcing operational boundaries on agent behaviors.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from .base import BaseEvent

@dataclass
class BudgetWarning(BaseEvent):
    """
    Signals that a predefined budget is nearing its limit or has been softly exceeded.
    
    This event acts as an early warning system, informing agents or monitoring
    systems that a budget is about to be exhausted or has temporarily gone
    over a 'soft' threshold, prompting a review of resource consumption.
    
    Attributes:
        event_id (str): Unique identifier for this budget warning event. Inherited from `BaseEvent`.
        timestamp (datetime): When the budget warning was triggered. Inherited from `BaseEvent`.
        agent_id (Optional[str]): The ID of the agent associated with the budget, if applicable.
                                  `None` if it's a global simulation budget.
        budget_type (str): The type of budget being monitored, e.g., "per_tick_llm_cost",
                           "total_simulation_time", "action_rate".
        current_usage (int): The current reported usage count or amount for the budget.
        limit (int): The predefined limit for this budget.
        reason (str): A descriptive reason for the warning, explaining what triggered it.
    """
    agent_id: Optional[str]
    budget_type: str 
    current_usage: int
    limit: int
    reason: str
    
    def __post_init__(self):
        """
        Validates the attributes of the `BudgetWarning` event upon initialization.
        Ensures `budget_type` is not empty and usage/limit are non-negative.
        """
        super().__post_init__() # Call base class validation
        
        # Validate budget_type: Must be a non-empty string.
        if not self.budget_type:
            raise ValueError("Budget type cannot be empty for BudgetWarning.")
        
        # Validate current_usage and limit: Must be non-negative.
        if self.current_usage < 0 or self.limit < 0:
            raise ValueError(f"Current usage ({self.current_usage}) and limit ({self.limit}) must be non-negative for BudgetWarning.")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Converts the `BudgetWarning` event into a concise summary dictionary.
        """
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
    Signals that a predefined budget has been hard-exceeded, potentially leading to termination.
    
    This event indicates a critical breach of a resource or operational budget.
    Depending on configuration, such an event might trigger an immediate halt
    of the simulation or particular agent's operation to prevent uncontrolled costs or behavior.
    
    Attributes:
        event_id (str): Unique identifier for this budget exceeded event. Inherited from `BaseEvent`.
        timestamp (datetime): When the budget was critically exceeded. Inherited from `BaseEvent`.
        agent_id (Optional[str]): The ID of the agent associated with the budget, if applicable.
                                  `None` if it's a global simulation budget.
        budget_type (str): The type of budget that was exceeded, e.g., "per_tick_llm_cost",
                           "total_simulation_time", "action_rate".
        current_usage (int): The current reported usage amount for the budget, which is now above the limit.
        limit (int): The predefined hard limit for this budget.
        reason (str): A descriptive reason for why the budget was exceeded.
        severity (str): Indicates the severity of the budget breach: "soft" (for warnings)
                        or "hard_fail" (for critical, potentially terminating breaches).
    """
    agent_id: Optional[str]
    budget_type: str
    current_usage: int
    limit: int
    reason: str
    severity: str 
    
    def __post_init__(self):
        """
        Validates the attributes of the `BudgetExceeded` event upon initialization.
        Ensures `budget_type` is not empty, usage/limit are non-negative,
        and `severity` is one of the allowed values.
        """
        super().__post_init__() # Call base class validation
        
        # Validate budget_type: Must be a non-empty string.
        if not self.budget_type:
            raise ValueError("Budget type cannot be empty for BudgetExceeded.")
        
        # Validate current_usage and limit: Must be non-negative.
        if self.current_usage < 0 or self.limit < 0:
            raise ValueError(f"Current usage ({self.current_usage}) and limit ({self.limit}) must be non-negative for BudgetExceeded.")
        
        # Validate severity: Must be "soft" or "hard_fail".
        if self.severity not in ["soft", "hard_fail"]:
            raise ValueError(f"Severity must be 'soft' or 'hard_fail', but got {self.severity}.")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Converts the `BudgetExceeded` event into a concise summary dictionary.
        """
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
    A general-purpose event for any activated constraint violation within the simulation.
    
    This event can be used to signal breaches of various types of constraints,
    not just budgets. It provides flexibility to report issues like rate limits,
    time limits, or other operational boundaries defined in the simulation's
    constraint system.
    
    Attributes:
        event_id (str): Unique identifier for this constraint violation event. Inherited from `BaseEvent`.
        timestamp (datetime): When the constraint violation occurred. Inherited from `BaseEvent`.
        agent_id (Optional[str]): The ID of the agent that triggered or is affected by the violation.
                                  `None` if the violation is global or not agent-specific.
        constraint_type (str): The specific type of constraint that was violated,
                               e.g., "rate_limit", "time_limit", "llm_token_limit", "invalid_action".
        violation_details (Dict[str, Any]): A dictionary containing specific details about the violation,
                                           e.g., the exact value, the threshold, additional context.
        is_critical (bool): A boolean flag indicating the severity:
                            `True` for critical violations (hard fails), `False` for warnings.
    """
    agent_id: Optional[str]
    constraint_type: str
    violation_details: Dict[str, Any]
    is_critical: bool 
    
    def __post_init__(self):
        """
        Validates the attributes of the `ConstraintViolation` event upon initialization.
        Ensures `constraint_type` is not empty and `violation_details` is a dictionary.
        """
        super().__post_init__() # Call base class validation
        
        # Validate constraint_type: Must be a non-empty string.
        if not self.constraint_type:
            raise ValueError("Constraint type cannot be empty for ConstraintViolation.")
        
        # Validate violation_details: Must be a dictionary.
        if not isinstance(self.violation_details, dict):
            raise TypeError("Violation details must be a dictionary for ConstraintViolation.")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Converts the `ConstraintViolation` event into a concise summary dictionary.
        """
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'constraint_type': self.constraint_type,
            'violation_details': self.violation_details,
            'is_critical': self.is_critical
        }
