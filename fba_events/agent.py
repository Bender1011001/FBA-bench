"""
Event definitions related to agent decisions and actions in the FBA-Bench simulation.

This module defines `AgentDecisionEvent`, which captures the comprehensive
details of an agent's reasoning process and the resulting tool calls.
It is crucial for analyzing agent behavior, debugging, and evaluating
strategic performance within the simulation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from .base import BaseEvent # Assuming BaseEvent is in a sibling module

@dataclass
class AgentDecisionEvent(BaseEvent):
    """
    Represents a comprehensive record of a decision made by an autonomous agent.
    
    This event is published whenever an agent completes a decision-making cycle,
    encapsulating its reasoning, the tools it attempted to use, and metrics
    related to its Large Language Model (LLM) interaction. It's a critical
    event for understanding and replaying agent behavior.
    
    Attributes:
        event_id (str): Unique identifier for this agent decision event. Inherited from `BaseEvent`.
        timestamp (datetime): The UTC datetime when the decision was made. Inherited from `BaseEvent`.
        agent_id (str): Unique identifier of the agent that made the decision.
        turn (int): The sequential turn number within the simulation tick this decision belongs to.
        tool_calls (List[Dict[str, Any]]): A list of dictionaries, each representing a tool call
                                           attempted or executed by the agent as part of this decision.
                                           Each dictionary details the tool name and its parameters.
        simulation_time (datetime): The logical simulation time at which this decision was made.
        reasoning (str): The explicit reasoning, thought process, or rationale provided by the agent
                         for its decision, typically from the LLM's output.
        llm_usage (Dict[str, Any]): A dictionary detailing the LLM tokens and costs incurred
                                    for this particular decision. E.g., {'prompt_tokens': 100, 'completion_tokens': 50}.
        prompt_metadata (Dict[str, Any]): Additional metadata about the LLM prompt used to generate
                                          this decision, such as model name, temperature, etc.
    """
    agent_id: str
    turn: int
    tool_calls: List[Dict[str, Any]]
    simulation_time: datetime
    reasoning: str
    llm_usage: Dict[str, Any]
    prompt_metadata: Dict[str, Any]

    def __post_init__(self):
        """
        Validates the attributes of the `AgentDecisionEvent` upon initialization.
        Ensures required fields are present and types are correct.
        """
        super().__post_init__() # Call base class validation
        
        # Validate agent_id: Must be a non-empty string.
        if not self.agent_id:
            raise ValueError("Agent ID cannot be empty for AgentDecisionEvent.")
        
        # Validate turn: Must be non-negative.
        if self.turn < 0:
            raise ValueError("Turn number must be >= 0 for AgentDecisionEvent.")
        
        # Validate tool_calls: Must be a list.
        if not isinstance(self.tool_calls, list):
            raise TypeError("Tool calls must be a list for AgentDecisionEvent.")
        
        # Validate simulation_time: Must be a datetime object.
        if not isinstance(self.simulation_time, datetime):
            raise TypeError("Simulation time must be a datetime object for AgentDecisionEvent.")
        
        # Validate reasoning: Must be a non-empty string.
        if not self.reasoning:
            raise ValueError("Reasoning cannot be empty for AgentDecisionEvent.")
        
        # Validate llm_usage and prompt_metadata: Must be dictionaries.
        if not isinstance(self.llm_usage, dict):
            raise TypeError("LLM usage must be a dictionary for AgentDecisionEvent.")
        if not isinstance(self.prompt_metadata, dict):
            raise TypeError("Prompt metadata must be a dictionary for AgentDecisionEvent.")
            

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        Converts the `AgentDecisionEvent` into a concise summary dictionary.
        
        This method is particularly useful for logging and analytical purposes,
        providing a high-level overview of the agent's decision without
        excessive detail from the raw tool calls or prompt metadata.
        """
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_id': self.agent_id,
            'turn': self.turn,
            'tool_calls_count': len(self.tool_calls), # Provide count instead of full list for summary
            'simulation_time': self.simulation_time.isoformat(),
            'reasoning_summary': self.reasoning[:200] + '...' if len(self.reasoning) > 200 else self.reasoning, # Truncate long reasoning
            'llm_total_tokens': self.llm_usage.get('total_tokens', 0), # Sum tokens for summary
            'llm_cost_usd': self.llm_usage.get('total_cost_usd', 0.0), # Sum cost for summary
            'prompt_model': self.prompt_metadata.get('model', 'N/A') #  Add model info for context
        }
