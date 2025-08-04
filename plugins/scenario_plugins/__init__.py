"""
Scenario plugins module for FBA-Bench.

This module provides base classes and utilities for developing
scenario plugins that extend the functionality of FBA-Bench.
"""

# Make key classes available at the package level
from .base_scenario_plugin import ScenarioPlugin

__all__ = ['ScenarioPlugin']