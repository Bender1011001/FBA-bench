"""
Integration module for FBA-Bench.

This module provides integration capabilities with external systems,
including real-world adapters and marketplace APIs.

Package-level imports are guarded to avoid hard failures during test collection
if optional submodules are not present.
"""
from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from typing import Any

# Ensure package has a valid __path__ for submodule resolution, even when loaded via a custom loader
if not globals().get("__path__"):
    __path__ = [os.path.dirname(__file__)]  # type: ignore[var-annotated]

__all__ = []

# Pre-populate aliases for tests.integration.test_* as integration.test_*
# This makes imports like `import integration.test_agent_integration` resolve to the tests package.
try:
    _pkg_root = os.path.dirname(os.path.dirname(__file__))
    _tests_integration_dir = os.path.join(_pkg_root, "tests", "integration")
    if os.path.isdir(_tests_integration_dir):
        for _fname in os.listdir(_tests_integration_dir):
            if _fname.startswith("test_") and _fname.endswith(".py"):
                _mod = _fname[:-3]
                try:
                    _src = importlib.import_module(f"tests.integration.{_mod}")
                    sys.modules[f"{__name__}.{_mod}"] = _src
                except Exception:
                    # Non-fatal: continue aliasing others
                    pass
except Exception:
    # Non-fatal: test aliasing is best-effort
    pass

# Best-effort eager re-exports (non-fatal on failure)
try:
    # Import the submodule explicitly and bind into sys.modules to defeat tests/ shadow packages
    _rwa = importlib.import_module(f"{__name__}.real_world_adapter")
    sys.modules[f"{__name__}.real_world_adapter"] = _rwa
    from .real_world_adapter import RealWorldAdapter  # type: ignore
    __all__.append("RealWorldAdapter")
except Exception:
    # Optional; importing submodule directly should still work
    pass

try:
    _ival = importlib.import_module(f"{__name__}.integration_validator")
    sys.modules[f"{__name__}.integration_validator"] = _ival
    from .integration_validator import IntegrationValidator, ValidationResult  # type: ignore
    __all__.append("IntegrationValidator")
    __all__.append("ValidationResult")
except Exception:
    # Non-fatal if ValidationResult or module not present yet
    pass

# Lazy submodule loader to be extra robust with custom import machinery
def __getattr__(name: str) -> Any:
    # Redirect accidental imports like `import integration.test_agent_integration` to tests package
    if name.startswith("test_"):
        try:
            return importlib.import_module(f"tests.integration.{name}")
        except Exception as _e:
            # Fall through to regular handling if not present
            pass
    if name in {"real_world_adapter", "integration_validator"}:
        return importlib.import_module(f"{__name__}.{name}")
    # Allow attribute access to eagerly re-exported symbols if present
    if name in globals():
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Explicit module-level lazy import hook for submodules (import integration.real_world_adapter)
def __path_hook__(fullname: str) -> ModuleType:  # pragma: no cover - import hook fallback
    return importlib.import_module(fullname)
