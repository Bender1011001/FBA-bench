"""
Plugins module for FBA-Bench.

This module provides a plugin framework for extending the functionality
of FBA-Bench with custom agents and scenarios.
"""

# Make key classes available at the package level
from .plugin_framework import PluginManager

__all__ = ['PluginManager']