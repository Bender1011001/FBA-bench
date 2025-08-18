"""
Core package initialization kept intentionally lightweight.

Avoid importing heavy submodules (like engine) at package import time to
prevent cascading side effects during test collection and simple imports
(e.g., importing benchmarking.metrics.base).

Consumers should import submodules explicitly:
- from benchmarking.core.engine import BenchmarkEngine
- from benchmarking.core.results import BenchmarkResult
- from benchmarking.core.config import BenchmarkConfig
"""

__all__: list[str] = []