"""
Framework-agnostic agent runner abstraction layer for FBA-Bench.

This package intentionally uses lazy exports to prevent heavy import-time side
effects and circular imports with benchmarking.* modules.

Key Concepts (resolved lazily on attribute access):
- AgentRunner API and errors
- SimulationState and ToolCall data structures
- RunnerFactory for framework registration/creation
- AgentManager/AgentRegistry integration layer
- Dependency management utilities
- Typed configuration helpers
"""

from __future__ import annotations

import importlib
from typing import Any, List

# Public API surface (names will be resolved lazily)
__all__: List[str] = [
    # Core interfaces and data
    "AgentRunner",
    "AgentRunnerStatus",
    "AgentRunnerError",
    "AgentRunnerInitializationError",
    "AgentRunnerDecisionError",
    "AgentRunnerCleanupError",
    "AgentRunnerTimeoutError",
    "SimulationState",
    "ToolCall",

    # Factory system
    "RunnerFactory",

    # Integration
    "AgentManager",
    "AgentRegistry",

    # Dependency management
    "DependencyManager",
    "dependency_manager",
    "check_framework_availability",
    "get_available_frameworks",
    "install_framework",

    # Utilities
    "get_framework_status",
]

# Candidate submodules to search when resolving attributes lazily.
# Order matters: light/leaf modules should come first to minimize side effects.
_CANDIDATE_MODULES = [
    "agent_runners.base_runner",
    "agent_runners.runner_factory",
    "agent_runners.dependency_manager",
    "agent_runners.agent_manager",  # kept last to reduce circular import risk
]


def __getattr__(name: str) -> Any:
    """
    Lazily resolve attributes from submodules to avoid import-time cycles.
    """
    for module_name in _CANDIDATE_MODULES:
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            # Optional dependency or module may not resolve in this environment
            continue
        if hasattr(mod, name):
            attr = getattr(mod, name)
            globals()[name] = attr  # cache for subsequent lookups
            return attr
    raise AttributeError(f"module 'agent_runners' has no attribute '{name}'")


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + __all__)


def get_framework_status() -> dict:
    """
    Best-effort status inspection for supported frameworks without hard imports.
    Returns:
      {
        'available_frameworks': [...],
        'all_frameworks': [...],
        'framework_info': {...}
      }
    """
    result = {
        "available_frameworks": [],
        "all_frameworks": [],
        "framework_info": {},
    }
    try:
        dep_mod = importlib.import_module("agent_runners.dependency_manager")
        rf_mod = importlib.import_module("agent_runners.runner_factory")
        get_avail = getattr(dep_mod, "get_available_frameworks", lambda: [])
        dep_mgr = getattr(dep_mod, "dependency_manager", None)
        get_all = getattr(rf_mod.RunnerFactory, "get_all_frameworks", lambda: [])

        result["available_frameworks"] = list(get_avail() or [])
        result["all_frameworks"] = list(get_all() or [])
        if dep_mgr is not None and hasattr(dep_mgr, "get_all_framework_info"):
            try:
                result["framework_info"] = dep_mgr.get_all_framework_info()
            except Exception:
                result["framework_info"] = {}
    except Exception:
        # Keep a minimal, non-failing status if anything isn't available
        pass
    return result