"""
Agent plugins module for FBA-Bench.

This module provides base classes and utilities for developing
agent plugins that extend the functionality of FBA-Bench.
"""

# Make key classes available at the package level
from .base_agent_plugin import AgentPlugin

__all__ = ['AgentPlugin']