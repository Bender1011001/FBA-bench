"""
FBA-Bench services package.

This package contains service classes that encapsulate business logic
and provide focused, testable functionality for the FBA-Bench simulation.
"""

from .competitor_manager import CompetitorManager
from .sales_processor import SalesProcessor

__all__ = ['CompetitorManager', 'SalesProcessor']