"""
Agent Runner Factory - Central registry and factory for creating agent runners.

This factory enables framework-agnostic agent creation and provides a central
registry system for adding new agent framework support.
"""

import logging
import asyncio
import importlib
from typing import Dict, Type, Any, Optional, List
from .base_runner import AgentRunner, AgentRunnerError, AgentRunnerInitializationError, AgentRunnerDecisionError, AgentRunnerCleanupError
# from .base_runner import AgentRegistration # AgentRegistration is not directly used here, but in AgentManager via AgentRegistry

logger = logging.getLogger(__name__)


class RunnerFactory:
    """
    Factory for creating and managing agent runners based on their framework.
    
    This centralizes the creation logic and allows for easy extension with
    new agent frameworks.
    """
    _runners: Dict[str, Type[AgentRunner]] = {}

    @classmethod
    def register_runner(cls, framework_name: str, runner_cls: Type[AgentRunner]):
        """Register a new agent runner type."""
        if not issubclass(runner_cls, AgentRunner):
            raise ValueError(f"Class {runner_cls.__name__} must inherit from AgentRunner")
        cls._runners[framework_name.lower()] = runner_cls
        logger.info(f"Registered agent runner: {framework_name} -> {runner_cls.__name__}")

    @classmethod
    def create_runner(cls, framework: str, agent_id: str, config: Dict[str, Any]) -> AgentRunner:
        """Create an instance of an agent runner."""
        framework_lower = framework.lower()
        runner_cls = cls._runners.get(framework_lower)
        if not runner_cls:
            raise ValueError(f"Unsupported agent framework: {framework}. Available: {list(cls._runners.keys())}")
        try:
            return runner_cls(agent_id=agent_id, config=config)
        except Exception as e:
            raise AgentRunnerInitializationError(
                f"Failed to create runner for framework '{framework}': {e}",
                agent_id=agent_id,
                framework=framework
            ) from e

    @classmethod
    def _attempt_register(cls, module_path: str, class_name: str, framework_name: str) -> None:
        """
        Best-effort import and registration for optional frameworks.
        Failures are non-fatal; they are logged at INFO/DEBUG and skipped.
        """
        try:
            module = importlib.import_module(module_path)
            runner_cls = getattr(module, class_name, None)
            if runner_cls and issubclass(runner_cls, AgentRunner):
                cls.register_runner(framework_name, runner_cls)
            else:
                logger.info(f"Framework '{framework_name}' found but class '{class_name}' missing or invalid")
        except Exception as e:
            # Keep logs concise to avoid noisy imports in environments without deps
            logger.info(f"Framework '{framework_name}' not available: {e}")

    @classmethod
    def _safe_register_defaults(cls) -> None:
        """
        Attempt to register built-in frameworks without hard import failures.
        """
        # DIY is intended to be available, but still guard import to avoid cascading issues
        cls._attempt_register("agent_runners.diy_runner", "DIYRunner", "diy")
        cls._attempt_register("agent_runners.crewai_runner", "CrewAIRunner", "crewai")
        cls._attempt_register("agent_runners.langchain_runner", "LangChainRunner", "langchain")

    @classmethod
    def list_runners(cls) -> List[str]:
        """Alias used by integration manager to enumerate frameworks."""
        return cls.get_all_frameworks()

    @classmethod
    async def create_and_initialize_runner(cls, framework: str, agent_id: str, config: Dict[str, Any]) -> AgentRunner:
        """
        Convenience API expected by integration manager to create and initialize a runner.
        """
        runner = cls.create_runner(framework, agent_id, config)
        init = getattr(runner, "initialize", None)
        if callable(init):
            maybe_coro = init(config)
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
        return runner

    @classmethod
    def get_all_frameworks(cls) -> List[str]:
        """Get a list of all registered framework names."""
        return list(cls._runners.keys())

# Register default runners (best-effort; frameworks are optional)
RunnerFactory._safe_register_defaults()

