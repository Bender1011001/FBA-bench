"""
Integration module for FBA-Bench.

This module provides integration capabilities with external systems,
including real-world adapters and marketplace APIs.
"""

# Make key classes available at the package level
from .real_world_adapter import RealWorldAdapter
from .integration_validator import IntegrationValidator

__all__ = ['RealWorldAdapter', 'IntegrationValidator']