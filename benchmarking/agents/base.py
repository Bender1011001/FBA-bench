"""
Base classes for all agent types in the FBA-Bench benchmarking framework.

This module defines the abstract base `BaseAgent` class that all specific
agent implementations (e.g., rule-based, LLM-driven) must inherit from.
It also includes the `AgentConfig` dataclass for configuring agents.
"""

import abc
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from llm_interface.config import LLMConfig

@dataclass
class AgentConfig:
    """Configuration for a benchmark agent."""
    id: str
    name: str = "Unnamed Agent"
    description: str = ""
    enabled: bool = True
    type: str = "default"  # e.g., "llm", "rule_based", "human"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Specific configuration for LLM-based agents
    llm_config: Optional[LLMConfig] = None
    
    # Custom parameters specific to an agent's implementation
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.llm_config and not isinstance(self.llm_config, LLMConfig):
            # Attempt to convert dict to LLMConfig if necessary
            self.llm_config = LLMConfig(**self.llm_config)


class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents participating in benchmarks.

    This class defines the interface that all agents must implement,
    including methods for initialization, interaction, and state management.
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the agent.

        Args:
            config: Agent configuration.
        """
        self.config = config
        self.agent_id = config.id

    @abc.abstractmethod
    async def initialize(self, *args, **kwargs) -> None:
        """
        Asynchronously initialize the agent's resources (e.g., load models, connect to APIs).
        """
        pass

    @abc.abstractmethod
    async def reset(self, *args, **kwargs) -> None:
        """
        Asynchronously reset the agent's internal state for a new run or scenario.
        """
        pass

    @abc.abstractmethod
    async def process_stimulus(self, stimulus: Any, *args, **kwargs) -> Any:
        """
        Asynchronously process a stimulus (e.g., a prompt, an observation) and
        return a response.

        Args:
            stimulus: The input data or event for the agent to process.
            *args: Positional arguments for processing.
            **kwargs: Keyword arguments for processing.

        Returns:
            The agent's response to the stimulus.
        """
        pass

    @abc.abstractmethod
    async def get_state(self) -> Dict[str, Any]:
        """
        Asynchronously get the current internal state of the agent.

        Returns:
            A dictionary representing the agent's current state.
        """
        pass

    @abc.abstractmethod
    async def shutdown(self, *args, **kwargs) -> None:
        """
        Asynchronously shut down and clean up agent resources.
        """
        pass