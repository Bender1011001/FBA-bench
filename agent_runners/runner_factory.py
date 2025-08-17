# DEPRECATED: RunnerFactory has been removed in favor of unified AgentFactory usage.
# This module remains importable for legacy code paths but raises on use.

import logging
from typing import Any, Dict, List, Type

logger = logging.getLogger(__name__)


class AgentRunner:  # minimal type for legacy annotations; not to be instantiated here
    pass


class RunnerFactory:  # legacy shim
    _runners: Dict[str, Type[AgentRunner]] = {}

    @classmethod
    def register_runner(cls, framework_name: str, runner_cls: Type[AgentRunner]) -> None:
        raise ImportError(
            "RunnerFactory.register_runner is deprecated. "
            "Agent creation is handled by AgentManager with unified agents."
        )

    @classmethod
    def create_runner(cls, framework: str, agent_id: str, config: Dict[str, Any]) -> AgentRunner:
        raise ImportError(
            "RunnerFactory.create_runner is deprecated. "
            "Agent creation is handled by AgentManager with unified agents."
        )

    @classmethod
    def list_runners(cls) -> List[str]:
        # Return empty list for compatibility with code that enumerates frameworks
        logger.warning("RunnerFactory.list_runners is deprecated. Returning empty list.")
        return []

    @classmethod
    async def create_and_initialize_runner(cls, framework: str, agent_id: str, config: Dict[str, Any]) -> AgentRunner:
        raise ImportError(
            "RunnerFactory.create_and_initialize_runner is deprecated. "
            "Agent creation is handled by AgentManager with unified agents."
        )

    @classmethod
    def get_all_frameworks(cls) -> List[str]:
        return []

