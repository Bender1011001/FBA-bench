"""
Agents module for the FBA-Bench benchmarking framework.

This module provides the core components for defining and managing agents
that participate in benchmark scenarios. It includes base classes, configurations,
and a registry for agent implementations.
"""

from .registry import AgentRegistry, agent_registry
from .base import BaseAgent, AgentConfig
from .unified_agent import (
    # Types and enums
    AgentState,
    AgentCapability,
    AgentMessage,
    AgentObservation,
    AgentAction,
    AgentContext,
    
    # Base classes
    BaseUnifiedAgent,
    NativeFBAAdapter,
    UnifiedAgentRunner,
    
    # Factory
    AgentFactory,
    
    # Global instance
    agent_factory
)

from llm_interface.config import LLMConfig

__all__ = [
    # Legacy agents
    "AgentRegistry",
    "agent_registry",
    "BaseAgent",
    "AgentConfig",
    "LLMConfig",
    
    # Unified agent framework
    "AgentState",
    "AgentCapability",
    "AgentMessage",
    "AgentObservation",
    "AgentAction",
    "AgentContext",
    "BaseUnifiedAgent",
    "NativeFBAAdapter",
    "UnifiedAgentRunner",
    "AgentFactory",
    "agent_factory"
]

# You can add logic here to automatically register built-in agents if desired
def _register_builtin_agents():
    """Register built-in agents on module import (placeholder)."""
    # Example:
    # from .llm_agent import LLMAgent
    # agent_registry.register("llm_agent", LLMAgent)
    pass

_register_builtin_agents()