from __future__ import annotations

# Robust local import of the project's money.py that avoids shadowing by the third-party
# "money" package installed in site-packages. We load by file path explicitly.
import importlib.util
import os
import sys
from types import ModuleType

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_MONEY_PATH = os.path.join(_THIS_DIR, "money.py")


def _load_local_money_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fba_local_money", _LOCAL_MONEY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load local money module from {_LOCAL_MONEY_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Ensure dependencies of money.py can resolve repository-local imports
    # by prioritizing repo root in sys.path for the duration of load
    repo_root = _THIS_DIR
    added = False
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
        added = True
    try:
        spec.loader.exec_module(module)
    finally:
        # Keep repo_root in sys.path (harmless and useful), but do not duplicate entries
        if added:
            try:
                sys.path.remove(repo_root)
            except ValueError:
                pass
            # Re-insert at front once to keep a single copy
            sys.path.insert(0, repo_root)
    return module


# Load local money symbols
_local_money = _load_local_money_module()

# Re-export the canonical API used across the codebase
Money = getattr(_local_money, "Money")
max_money = getattr(_local_money, "max_money")
min_money = getattr(_local_money, "min_money")
sum_money = getattr(_local_money, "sum_money")

__all__ = ["Money", "max_money", "min_money", "sum_money"]