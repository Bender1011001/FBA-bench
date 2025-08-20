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

    # Registry helper
    "create_runner",
    "supported_runners",
    "RunnerFactory",
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

    # Builder/config helpers (back-compat for tests)
    "create_agent_builder",
    "DIYConfig",
    "validate_config",
    "AgentRunnerConfig",
]

# Candidate submodules to search when resolving attributes lazily.
# Order matters: light/leaf modules should come first to minimize side effects.
_CANDIDATE_MODULES = [
    "agent_runners.base_runner",
    "agent_runners.registry",
    "agent_runners.dependency_manager",
    "agent_runners.agent_manager",  # kept last to reduce circular import risk
]


def __getattr__(name: str) -> Any:
    """
    Lazily resolve attributes from submodules to avoid import-time cycles.
    """
    # Special-case alias: some tests import RunnerFactory from agent_runners
    if name == "RunnerFactory":
        try:
            mod = importlib.import_module("agent_runners.runner_factory")
            if hasattr(mod, "RunnerFactory"):
                attr = getattr(mod, "RunnerFactory")
                globals()[name] = attr
                return attr
        except Exception:
            # Fallback alias to create_runner function if class is unavailable
            def _runner_factory(*args, **kwargs):
                creator = __getattr__("create_runner")
                return creator(*args, **kwargs)
            globals()[name] = _runner_factory
            return _runner_factory

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


# ------------------------------
# Back-compat helpers for tests
# ------------------------------
from dataclasses import dataclass, field as _field
from typing import Dict as _Dict, Any as _Any

@dataclass
class _AgentConfigPayload:
    agent_type: str = "advanced"
    target_asin: str = "B0DEFAULT"
    parameters: _Dict[str, _Any] = _field(default_factory=dict)

@dataclass
class AgentRunnerConfig:
    agent_id: str
    framework: str
    agent_config: _AgentConfigPayload

def validate_config(config_dict: _Dict[str, _Any]) -> AgentRunnerConfig:
    """Validate minimal runner config and return a normalized dataclass."""
    if not isinstance(config_dict, dict):
        raise ValueError("config must be a dict")
    framework = str(config_dict.get("framework") or "").lower()
    if framework not in {"diy", "mock", "crewai", "langchain"}:
        raise ValueError(f"Unsupported framework: {framework}")
    agent_id = str(config_dict.get("agent_id") or config_dict.get("id") or "agent")
    ac = config_dict.get("agent_config") or {}
    payload = _AgentConfigPayload(
        agent_type=str(ac.get("agent_type") or "advanced"),
        target_asin=str(ac.get("target_asin") or "B0DEFAULT"),
        parameters=dict(ac.get("parameters") or {})
    )
    return AgentRunnerConfig(agent_id=agent_id, framework=framework, agent_config=payload)

class DIYConfig:
    """Convenience factory for DIY framework configs expected by tests."""
    @staticmethod
    def advanced_agent(agent_id: str, target_asin: str) -> AgentRunnerConfig:
        return validate_config({
            "agent_id": agent_id,
            "framework": "diy",
            "agent_config": {"agent_type": "advanced", "target_asin": target_asin}
        })

    @staticmethod
    def baseline_greedy(agent_id: str) -> AgentRunnerConfig:
        return validate_config({
            "agent_id": agent_id,
            "framework": "diy",
            "agent_config": {"agent_type": "baseline", "target_asin": "B0DEFAULT"}
        })

def check_framework_availability(name: str) -> bool:
    try:
        mod = importlib.import_module("agent_runners.registry")
        fn = getattr(mod, "is_framework_available", None)
        if callable(fn):
            return bool(fn(name))
    except Exception:
        pass
    return False

class _AgentBuilder:
    def __init__(self, framework: str, agent_id: str):
        self._framework = framework
        self._agent_id = agent_id
        self._config: _Dict[str, _Any] = {}

    def with_config(self, **kwargs) -> "_AgentBuilder":
        self._config.update(kwargs)
        return self

    def build(self):
        rf = importlib.import_module("agent_runners.runner_factory")
        creator = getattr(rf, "RunnerFactory").create_runner
        return creator(self._framework, self._agent_id, self._config)

    async def build_and_initialize(self):
        runner = self.build()
        init = getattr(runner, "initialize", None)
        if callable(init):
            await init(self._config)
        return runner

def create_agent_builder(framework: str, agent_id: str) -> _AgentBuilder:
    return _AgentBuilder(framework, agent_id)


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
        reg_mod = importlib.import_module("agent_runners.registry")
        get_avail = getattr(dep_mod, "get_available_frameworks", lambda: [])
        dep_mgr = getattr(dep_mod, "dependency_manager", None)
        get_all = getattr(reg_mod, "supported_runners", lambda: [])

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