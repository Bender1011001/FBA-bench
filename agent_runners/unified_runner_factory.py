"""
Deprecated unified runner factory.

Agent creation is now centralized in benchmarking.agents.unified_agent.AgentFactory
and consumed exclusively via AgentManager. This module remains as a soft-compat shim
to avoid import-time failures in legacy code and tests.
"""

import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


class UnifiedRunnerFactory:
    """
    Deprecated shim. Do not use.
    """

    def __init__(self) -> None:
        logger.warning(
            "agent_runners.unified_runner_factory.UnifiedRunnerFactory is deprecated. "
            "Use AgentManager with unified agents via benchmarking.agents.unified_agent.AgentFactory."
        )

    def create_runner(self, agent_type: str, agent_id: str, config: Union[Dict[str, Any], Any, Any]) -> Any:
        raise ImportError(
            "UnifiedRunnerFactory is deprecated. Agent creation is handled by AgentManager "
            "with unified agents (benchmarking.agents.unified_agent.AgentFactory)."
        )

    def get_available_agent_types(self) -> List[str]:
        return []

    def get_available_adapter_types(self) -> List[str]:
        return []

    def is_agent_type_registered(self, agent_type: str) -> bool:
        return False

    def is_adapter_type_registered(self, adapter_type: str) -> bool:
        return False


# Convenience function retained for legacy imports; raises with clear guidance
def create_unified_runner(agent_type: str, agent_id: str, config: Union[Dict[str, Any], Any, Any]) -> Any:
    raise ImportError(
        "create_unified_runner is deprecated. Use AgentManager with unified agents "
        "(benchmarking.agents.unified_agent.AgentFactory) instead."
    )