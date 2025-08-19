"""
Compatibility shim for API dependencies within the fba_bench_api package.

This module re-exports objects from the top-level `api.dependencies` to provide a
stable import path: `from fba_bench_api.api.dependencies import connection_manager`.
"""

from __future__ import annotations

# Re-export public symbols from the legacy module to maintain backwards compatibility
try:
    from api.dependencies import (  # type: ignore F401
        ConnectionManager,
        SimulationManager,
        connection_manager,
        simulation_manager,
        active_simulations,
        active_experiments,
    )
except Exception as exc:  # pragma: no cover - defensive fallback
    # Provide a clear error to help diagnose import path issues during runtime.
    raise ImportError(
        "Failed to import API dependencies from 'api.dependencies'. Ensure the 'api' "
        "package is on PYTHONPATH and contains dependencies.py."
    ) from exc