from __future__ import annotations

"""
Typed registry and explicit helper for agent runner creation.

- Provides a small, testable mapping from runner_key -> concrete runner class
- Validates config using the target runner's Pydantic v2 model when available
- Defers optional third-party imports to runner initialization (soft deps preserved)
"""

import logging
from typing import Any, Dict, List, Tuple, Type, Optional

# Core base API
from .base_runner import AgentRunner

# Import concrete runners and their config models (safe: these modules only soft-import 3rd parties at runtime)
from .crewai_runner import CrewAIRunner, CrewAIRunnerConfig
from .langchain_runner import LangChainRunner, LangChainRunnerConfig

# DIY is always available and does not have a dedicated Pydantic config model
try:
    from .diy_runner import DIYRunner
except Exception:
    DIYRunner = None  # type: ignore

logger = logging.getLogger(__name__)

# Internal registry holds (RunnerClass, Optional[ConfigModelClass])
# Keep imports soft: config models do not import third-party frameworks; runners soft-import at _do_initialize
RUNNER_REGISTRY: Dict[str, Tuple[Type[AgentRunner], Optional[Type[Any]]]] = {}

if DIYRunner is not None:
    RUNNER_REGISTRY["diy"] = (DIYRunner, None)

RUNNER_REGISTRY.update({
    "crewai": (CrewAIRunner, CrewAIRunnerConfig),
    "langchain": (LangChainRunner, LangChainRunnerConfig),
})


def supported_runners() -> List[str]:
    """Return supported runner keys."""
    return sorted(RUNNER_REGISTRY.keys())


def _validate_config(key: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize config for the target runner when a Pydantic model is available."""
    runner_entry = RUNNER_REGISTRY.get(key)
    if not runner_entry:
        return cfg
    _, cfg_model = runner_entry
    if cfg_model is None:
        return cfg or {}
    try:
        # Pydantic v2: model_validate
        validated = cfg_model.model_validate(cfg or {})
        # Use model_dump to pass a normalized dict to the runner constructor
        return validated.model_dump(exclude_none=True)
    except Exception as e:
        # Clear error message surfaced to callers
        raise ValueError(f"Invalid config for runner '{key}': {e}") from e


def create_runner(key: str, config: Dict[str, Any]) -> AgentRunner:
    """
    Create a runner instance with validation.

    Args:
        key: runner key (e.g., 'crewai', 'langchain', 'diy')
        config: configuration dict for the runner

    Returns:
        AgentRunner instance

    Raises:
        ValueError: when key is unknown or config validation fails
        AgentRunnerInitializationError: if optional dependency is missing during initialization
    """
    if not isinstance(key, str):
        raise ValueError("Runner key must be a string")
    norm_key = key.strip().lower()

    entry = RUNNER_REGISTRY.get(norm_key)
    if entry is None:
        msg = (
            f"Unknown runner key: '{key}'. "
            f"Supported keys: {', '.join(supported_runners()) or '(none)'}"
        )
        logger.error(msg)
        raise ValueError(msg)

    runner_cls, _ = entry

    # Validate config using the specific model if available
    normalized_cfg = _validate_config(norm_key, config or {})

    # Instantiate explicitly; AgentRunner __init__ may trigger initialization and optional imports
    # This preserves soft-dependency behavior: import errors happen at instantiation time
    try:
        return runner_cls(normalized_cfg.get("agent_id") or normalized_cfg.get("name") or normalized_cfg.get("agent_name") or "agent", normalized_cfg)
    except Exception as e:
        # Let caller see runner-specific exceptions (e.g., AgentRunnerInitializationError)
        raise


__all__ = ["RUNNER_REGISTRY", "create_runner", "supported_runners"]