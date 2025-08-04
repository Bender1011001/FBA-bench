"""
Marketplace APIs module for FBA-Bench.

This module provides integration with various marketplace APIs,
including Amazon Seller Central and other e-commerce platforms.
"""

# Make key classes available at the package level
from .marketplace_factory import MarketplaceFactory

__all__ = ['MarketplaceFactory']