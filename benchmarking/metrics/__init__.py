"""
Metrics package initialization kept lightweight.

This package's __init__ intentionally avoids importing submodules to prevent
import-time side effects during test collection. Import modules directly, e.g.:
- from benchmarking.metrics.base import BaseMetric, MetricConfig
- from benchmarking.metrics.registry import metrics_registry
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Public API is intentionally empty to avoid eager imports.
__all__: list[str] = []