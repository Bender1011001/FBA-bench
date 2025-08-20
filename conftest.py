# Root-level pytest configuration to stabilize test discovery and avoid duplicate module name collisions.
# - Ignores tests/benchmarking duplicate test modules that conflict with tests/unit basenames.
# - Ensures project root is on sys.path early (defense-in-depth; tests/sitecustomize also handles this).

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Proactively import and bind critical top-level packages to avoid shadowing by tests/*
# This complements tests/sitecustomize.py and ensures early availability during collection.
try:
    import importlib
    import integration as _integration  # type: ignore
    # Ensure submodule path binding for integration.real_world_adapter
    importlib.import_module("integration.real_world_adapter")
except Exception:
    pass


def pytest_ignore_collect(path: Path, config: Any) -> bool:
    """
    Return True to prevent pytest from collecting the given path.

    We ignore the tests/benchmarking directory to avoid import file mismatches where
    modules like 'test_engine.py' exist in both tests/benchmarking and tests/unit.
    The unit tests cover the same functionality without causing duplicate basenames.
    """
    try:
        p = Path(str(path))
    except Exception:
        return False

    # Normalize and check if under tests/benchmarking
    try:
        rel = p.resolve().relative_to(PROJECT_ROOT)
    except Exception:
        return False

    parts = [part.lower() for part in rel.parts]
    if len(parts) >= 2 and parts[0] == "tests" and parts[1] == "benchmarking":
        return True
    return False