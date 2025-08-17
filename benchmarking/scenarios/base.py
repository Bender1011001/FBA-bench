"""
Base classes for all scenario types in the FBA-Bench benchmarking framework.

This module defines the abstract base `BaseScenario` class that all specific
scenario implementations must inherit from. It also includes the `ScenarioConfig`
dataclass for configuring scenarios.
"""

import abc
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from ..agents.base import BaseAgent # Scenarios interact with agents
from ..core.results import AgentRunResult # Scenarios produce AgentRunResults

@dataclass
class ScenarioConfig:
    """Configuration for a benchmark scenario."""
    id: str
    name: str = "Unnamed Scenario"
    description: str = ""
    enabled: bool = True
    priority: int = 1
    # Custom parameters specific to a scenario's implementation
    parameters: Dict[str, Any] = field(default_factory=dict)

class BaseScenario(abc.ABC):
    """
    Abstract base class for all benchmark scenarios.

    This class defines the interface that all scenarios must implement,
    including methods for setup, execution with agents, and result collection.
    """

    def __init__(self, config: ScenarioConfig):
        """
        Initialize the scenario.

        Args:
            config: Scenario configuration.
        """
        self.config = config
        self.scenario_id = config.id

    @abc.abstractmethod
    async def setup(self, *args, **kwargs) -> None:
        """
        Asynchronously set up the scenario, loading any necessary data or initial states.
        """
        pass

    @abc.abstractmethod
    async def run(self, agent: BaseAgent, run_number: int, *args, **kwargs) -> AgentRunResult:
        """
        Asynchronously run a single iteration of the scenario with a given agent.

        Args:
            agent: The BaseAgent instance to run the scenario against.
            run_number: The current run number for this scenario (e.g., 1st, 2nd run).
            *args: Positional arguments for scenario execution.
            **kwargs: Keyword arguments for scenario execution.

        Returns:
            An AgentRunResult object containing the outcome and metrics of this run.
        """
        pass

    @abc.abstractmethod
    async def teardown(self, *args, **kwargs) -> None:
        """
        Asynchronously clean up resources after the scenario runs (e.g., close connections).
        """
        pass

    @abc.abstractmethod
    async def get_progress(self) -> Dict[str, Any]:
        """
        Asynchronously get the current progress or state of the scenario.

        Returns:
            A dictionary containing progress information.
        """
        pass