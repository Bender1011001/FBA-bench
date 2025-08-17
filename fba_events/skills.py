"""
Event definitions related to agent skill coordination and multi-domain decision-making
in the FBA-Bench simulation.

This module formalizes the communication around how agents activate, generate actions,
resolve conflicts among their skills, and make overarching strategic decisions.
It includes:
- `SkillActivated`: Signals when an agent's specific skill is chosen to handle an event.
- `SkillActionGenerated`: Reports actions proposed by individual skills.
- `SkillConflictDetected`: Notifies when multiple skills propose conflicting actions requiring arbitration.
- `MultiDomainDecisionMade`: Documents the final coordinated decision made by an agent's
  central `MultiDomainController` or equivalent, often after resolving skill conflicts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from .base import BaseEvent

@dataclass
class SkillActivated(BaseEvent):
    """
    Signals that a specific skill within an agent has been activated to handle an event.
    
    This event is published when a skill's pre-defined trigger conditions are met,
    indicating that the skill is now actively processing an incoming event and
    will potentially generate actions.
    
    Attributes:
        event_id (str): Unique identifier for this skill activation event. Inherited from `BaseEvent`.
        timestamp (datetime): When the skill was activated. Inherited from `BaseEvent`.
        skill_name (str): The programmatic name of the skill that was activated (e.g., 'FinancialAnalystSkill').
        event_trigger (str): The type or name of the `BaseEvent` that triggered this skill's activation.
        priority_score (float): A normalized score (0.0 to 1.0) indicating the priority of this
                                activation relative to other potential skill activations.
        agent_id (str): The unique identifier of the agent whose skill was activated.
        activation_reason (str): A brief, human-readable explanation for why the skill was activated.
    """
    skill_name: str
    event_trigger: str
    priority_score: float
    agent_id: str
    activation_reason: str = ""
    
    def __post_init__(self):
        """
        Validates the attributes of the `SkillActivated` event upon initialization.
        Ensures skill name is provided and priority score is within range.
        """
        super().__post_init__() # Call base class validation
        
        # Validate skill_name: Must be a non-empty string.
        if not self.skill_name:
            raise ValueError("Skill name cannot be empty for SkillActivated event.")
        
        # Validate priority_score: Must be within [0.0, 1.0].
        if not 0.0 <= self.priority_score <= 1.0:
            raise ValueError(f"Priority score must be between 0.0 and 1.0, but got {self.priority_score} for SkillActivated event.")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Converts the `SkillActivated` event into a concise summary dictionary.
        Priority score is rounded for readability.
        """
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
    Signals that a specific skill within an agent has generated a proposed action.
    
    After a skill is activated, it may decide to propose one or more actions
    to the `SkillCoordinator` or `MultiDomainController`. This event captures
    the details of such a proposed action, along with the skill's confidence
    in its recommendation.
    
    Attributes:
        event_id (str): Unique identifier for this skill action event. Inherited from `BaseEvent`.
        timestamp (datetime): When the action was generated. Inherited from `BaseEvent`.
        skill_name (str): The programmatic name of the skill that generated the action.
        action_type (str): The type of action proposed (e.g., 'set_price', 'place_order', 'respond_to_customer').
        confidence_score (float): A normalized score (0.0 to 1.0) indicating the skill's confidence
                                  in the validity and effectiveness of its proposed action.
        agent_id (str): The unique identifier of the agent whose skill generated the action.
        action_parameters (Dict[str, Any]): A dictionary containing the parameters required to execute the action.
                                             This typically matches the input schema of a `ToolboxAPI` function.
        expected_outcome (Dict[str, Any]): Optional; a description of the expected result if this action is executed.
                                          Useful for comparing against actual outcomes for learning/evaluation.
        resource_requirements (Dict[str, Any]): Optional; a dictionary outlining estimated resources (e.g., LLM tokens,
                                                compute cycles) required to execute this action.
    """
    skill_name: str
    action_type: str
    confidence_score: float
    agent_id: str
    action_parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """
        Validates the attributes of the `SkillActionGenerated` event upon initialization.
        Ensures skill name and action type are provided, and confidence score is within range.
        """
        super().__post_init__() # Call base class validation
        
        # Validate skill_name: Must be a non-empty string.
        if not self.skill_name:
            raise ValueError("Skill name cannot be empty for SkillActionGenerated event.")
        
        # Validate action_type: Must be a non-empty string.
        if not self.action_type:
            raise ValueError("Action type cannot be empty for SkillActionGenerated event.")
        
        # Validate confidence_score: Must be within [0.0, 1.0].
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"Confidence score must be between 0.0 and 1.0, but got {self.confidence_score} for SkillActionGenerated event.")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Converts the `SkillActionGenerated` event into a concise summary dictionary.
        Confidence score is rounded for readability.
        """
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'skill_name': self.skill_name,
            'action_type': self.action_type,
            'confidence_score': round(self.confidence_score, 3),
            'agent_id': self.agent_id,
            'action_parameters': self.action_parameters, # Include full parameters as they are often small and critical for understanding
            'expected_outcome': self.expected_outcome,
            'resource_requirements': self.resource_requirements
        }

@dataclass
class SkillConflictDetected(BaseEvent):
    """
    Signals that a conflict has been detected between multiple skills within an agent.
    
    This event is typically published by the `SkillCoordinator` or `MultiDomainController`
    when activated skills propose conflicting actions, resource requests, or strategic
    recommendations that cannot be simultaneously executed. It prompts a conflict
    resolution process.
    
    Attributes:
        event_id (str): Unique identifier for this conflict event. Inherited from `BaseEvent`.
        timestamp (datetime): When the conflict was detected. Inherited from `BaseEvent`.
        competing_skills (List[str]): A list of programmatic names of the skills involved in the conflict.
                                      Must include at least two skills.
        conflict_type (str): The category of conflict, e.g., "resource_contention", "contradictory_actions",
                             "priority_mismatch", "domain_overlap".
        resolution (str): A description of how the conflict was ultimately resolved.
                          (e.g., "prioritized_financial_skill", "agent_override", "negotiated_compromise").
        agent_id (str): The unique identifier of the agent where the conflict occurred.
        conflict_details (Dict[str, Any]): A dictionary providing additional structured details about the conflict,
                                          e.g., specific resources contended, conflicting action parameters, etc.
        resolution_strategy (str): The named strategy or algorithm used by the agent to resolve the conflict.
    """
    competing_skills: List[str]
    conflict_type: str
    resolution: str
    agent_id: str
    conflict_details: Dict[str, Any] = field(default_factory=dict)
    resolution_strategy: str = ""
    
    def __post_init__(self):
        """
        Validates the attributes of the `SkillConflictDetected` event upon initialization.
        Ensures at least two competing skills are listed, and critical strings are non-empty.
        """
        super().__post_init__() # Call base class validation
        
        # Validate competing_skills: Must contain at least two skill names.
        if len(self.competing_skills) < 2:
            raise ValueError("`competing_skills` must include at least 2 skill names for SkillConflictDetected event.")
        
        # Validate conflict_type: Must be a non-empty string.
        if not self.conflict_type:
            raise ValueError("Conflict type cannot be empty for SkillConflictDetected event.")
        
        # Validate resolution: Must be a non-empty string.
        if not self.resolution:
            raise ValueError("Resolution cannot be empty for SkillConflictDetected event.")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Converts the `SkillConflictDetected` event into a concise summary dictionary.
        """
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
    Signals that a high-level, multi-domain strategic decision has been finalized by an agent's controller.
    
    This event represents the outcome of the toughest internal coordination tasks
    within a sophisticated agent, where various skill inputs and conflicts have
    been arbitrated and a unified strategic action plan is adopted. This is the
    result of the `MultiDomainController`'s work.
    
    Attributes:
        event_id (str): Unique identifier for this multi-domain decision event. Inherited from `BaseEvent`.
        timestamp (datetime): When the comprehensive decision was finalized. Inherited from `BaseEvent`.
        coordinated_actions (List[Dict[str, Any]]): A list of the final, approved, and coordinated actions
                                                    that will be executed as a result of this decision.
                                                    Each item is typically a dictionary representing a tool call.
        reasoning (str): The strategic reasoning or overall rationale behind the multi-domain decision.
                         This is usually a high-level summary of the complex thought process.
        resource_allocation (Dict[str, Any]): A dictionary detailing how resources (e.g., budget, time, LLM tokens)
                                             were allocated or re-allocated as part of this decision.
        agent_id (str): The unique identifier of the agent that made this strategic decision.
        business_priority (str): The top business priority or goal influencing this decision at the time.
                                 (e.g., "maximize_profit", "increase_market_share", "improve_customer_satisfaction").
        strategic_alignment_score (float): A normalized score (0.0 to 1.0) indicating how well this decision
                                           aligns with the agent's long-term strategy or overall objectives.
        rejected_actions_count (int): The number of proposed actions from individual skills that were
                                      rejected or overridden during the conflict resolution and coordination phase.
    """
    coordinated_actions: List[Dict[str, Any]]
    reasoning: str
    resource_allocation: Dict[str, Any]
    agent_id: str
    business_priority: str = ""
    strategic_alignment_score: float = 0.0
    rejected_actions_count: int = 0
    
    def __post_init__(self):
        """
        Validates the attributes of the `MultiDomainDecisionMade` event upon initialization.
        Ensures `reasoning` and `agent_id` are provided, and required fields are of correct types.
        """
        super().__post_init__() # Call base class validation
        
        # Validate reasoning: Must be a non-empty string.
        if not self.reasoning:
            raise ValueError("Reasoning cannot be empty for MultiDomainDecisionMade event.")
        
        # Validate coordinated_actions: Must be a list.
        if not isinstance(self.coordinated_actions, list):
            raise TypeError("Coordinated actions must be a list for MultiDomainDecisionMade event.")
        
        # Validate resource_allocation: Must be a dictionary.
        if not isinstance(self.resource_allocation, dict):
            raise TypeError("Resource allocation must be a dictionary for MultiDomainDecisionMade event.")
        
        # Validate agent_id: Must be a non-empty string.
        if not self.agent_id:
            raise ValueError("Agent ID cannot be empty for MultiDomainDecisionMade event.")
        
        # Validate strategic_alignment_score: Must be within [0.0, 1.0].
        if not 0.0 <= self.strategic_alignment_score <= 1.0:
            raise ValueError(f"Strategic alignment score must be between 0.0 and 1.0, but got {self.strategic_alignment_score}.")
        
        # Validate rejected_actions_count: Must be non-negative.
        if self.rejected_actions_count < 0:
            raise ValueError("Rejected actions count must be non-negative.")
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Converts the `MultiDomainDecisionMade` event into a concise summary dictionary.
        
        Provides high-level details about the coordinated actions, the
        underlying strategic reasoning (truncated for brevity), and key metrics.
        """
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'coordinated_actions_count': len(self.coordinated_actions),
            'reasoning_summary': self.reasoning[:200] + '...' if len(self.reasoning) > 200 else self.reasoning, # Truncate long reasoning
            'resource_allocation': self.resource_allocation,
            'agent_id': self.agent_id,
            'business_priority': self.business_priority,
            'strategic_alignment_score': round(self.strategic_alignment_score, 3),
            'rejected_actions_count': self.rejected_actions_count
        }
