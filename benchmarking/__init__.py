"""
Benchmarking package initialization.

This __init__ keeps imports lightweight and side-effect free to ensure
subpackage imports like `benchmarking.metrics.base` succeed during test collection.
Heavy components (engines, registries, services) should be imported explicitly by
consumers to avoid import-time failures.
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Expose minimal public API without importing heavy modules at package import time.
# Users/tests should import submodules directly, e.g.:
#   from benchmarking.metrics.base import BaseMetric
#   from benchmarking.core.engine import BenchmarkEngine

__all__ = []