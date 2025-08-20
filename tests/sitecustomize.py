"""
Ensure the project root is on sys.path when pytest sets rootdir to tests/, and
force the 'benchmarking' package to resolve to the real project package instead of
the shadowing tests/benchmarking/ package.

Python automatically imports 'sitecustomize' at startup if it is importable on sys.path.
Placing this file in tests/ guarantees it will be imported first during test collection.
"""
from __future__ import annotations

import sys
from pathlib import Path
import importlib.util as _util

# Compute repository root as the parent of the tests directory
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
REAL_BENCHMARKING_DIR = PROJECT_ROOT / "benchmarking"

# Prepend project root to sys.path if not already present
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# If the real benchmarking package exists, load and bind it explicitly into sys.modules
# to prevent Python from importing the shadow package at tests/benchmarking.
try:
    real_init = REAL_BENCHMARKING_DIR / "__init__.py"
    if real_init.exists():
        spec = _util.spec_from_file_location(
            "benchmarking",
            str(real_init),
            submodule_search_locations=[str(REAL_BENCHMARKING_DIR)],
        )
        if spec and spec.loader:
            mod = _util.module_from_spec(spec)
            # Bind before exec to ensure submodule imports find the correct parent
            sys.modules["benchmarking"] = mod
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            # Ensure correct package path for submodules like benchmarking.metrics
            if not hasattr(mod, "__path__") or not mod.__path__:
                mod.__path__ = [str(REAL_BENCHMARKING_DIR)]  # type: ignore[attr-defined]
except Exception:
    # If anything goes wrong, leave sys.path fix in place; tests will surface issues.
    pass

# Ensure 'integration' resolves to project integration package (avoid collision with tests.integration)
try:
    import importlib.util as _util2
    from pathlib import Path as _Path
    ROOT = _Path(__file__).resolve().parents[1]
    REAL_INTEGRATION_DIR = ROOT / "integration"
    real_init = REAL_INTEGRATION_DIR / "__init__.py"
    if real_init.exists():
        spec2 = _util2.spec_from_file_location(
            "integration",
            str(real_init),
            submodule_search_locations=[str(REAL_INTEGRATION_DIR)],
        )
        if spec2 and spec2.loader:
            imod = _util2.module_from_spec(spec2)
            import sys as _sys
            _sys.modules["integration"] = imod
            spec2.loader.exec_module(imod)  # type: ignore[attr-defined]
            if not hasattr(imod, "__path__") or not imod.__path__:
                imod.__path__ = [str(REAL_INTEGRATION_DIR)]  # type: ignore[attr-defined]
except Exception:
    pass