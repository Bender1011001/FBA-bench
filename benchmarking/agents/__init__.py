"""
Agent components for the benchmarking framework.

This module provides agent-related functionality for the benchmarking framework,
including agent registry, agent adapters, and agent management.
"""

from .registry import AgentRegistry, AgentRegistration, agent_registry

__all__ = [
    "AgentRegistry",
    "AgentRegistration", 
    "agent_registry"
]