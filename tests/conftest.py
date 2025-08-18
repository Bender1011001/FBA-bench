"""
Pytest configuration for FBA-Bench tests.

Ensures the repository root is on sys.path so absolute imports like
'benchmarking.*' resolve correctly when pytest sets rootdir to tests/.
Also can host shared fixtures if needed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Compute repository root as the parent of the tests directory
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent

# Prepend project root to sys.path if not already present
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# Emit a trace so we can confirm conftest executed and path was set when pytest starts
print(f"[conftest] PROJECT_ROOT added to sys.path: {project_root_str in sys.path} -> {project_root_str}")

# Optionally expose an environment flag for tests
os.environ.setdefault("FBA_TESTING", "1")

# --- Early import diagnostic for benchmarking.metrics.base ---
try:
    import importlib, traceback
    mod = importlib.import_module("benchmarking.metrics.base")
    print("[conftest] SUCCESS importing benchmarking.metrics.base ->", getattr(mod, "__file__", None))
except Exception as e:
    import traceback as _tb
    print("[conftest] FAILURE importing benchmarking.metrics.base:", repr(e))
    _tb.print_exc()