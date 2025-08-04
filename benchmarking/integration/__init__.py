"""
Integration module for connecting benchmarking framework with existing systems.

This module provides integration points between the new benchmarking framework
and existing systems like agent_runners, metrics, and infrastructure components.
"""

from .manager import (
    IntegrationStatus,
    IntegrationConfig,
    IntegrationManager,
    SimpleEventBus,
    integration_manager
)

__all__ = [
    "IntegrationStatus",
    "IntegrationConfig",
    "IntegrationManager",
    "SimpleEventBus",
    "integration_manager"
]