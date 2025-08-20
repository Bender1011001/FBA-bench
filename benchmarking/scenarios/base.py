"""
Base classes for all scenario types in the FBA-Bench benchmarking framework.

This module defines the abstract base `BaseScenario` class that all specific
scenario implementations must inherit from. It also includes the `ScenarioConfig`
dataclass for configuring scenarios.
"""

import abc
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import uuid

from ..agents.base import BaseAgent  # Scenarios interact with agents
from ..core.results import AgentRunResult  # Scenarios produce AgentRunResults

@dataclass
class ScenarioConfig:
    """Configuration for a benchmark scenario (aligned with tests)."""
    # Required by tests
    name: str
    description: str
    domain: str
    duration_ticks: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    # Optional/legacy fields used by parts of the engine/UI
    id: Optional[str] = None
    priority: int = 1

    def ensure_id(self) -> str:
        """Return a stable id, generating one if absent."""
        if not self.id or not isinstance(self.id, str) or not self.id.strip():
            # Prefer a deterministic id from name when available
            base = (self.name or "scenario").strip().lower().replace(" ", "_")
            self.id = f"{base}-{uuid.uuid4().hex[:8]}"
        return self.id

@dataclass
class ScenarioResult:
    """Minimal ScenarioResult structure for tests expecting import."""
    scenario_name: str
    success: bool = True
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None


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
        # Ensure we always have a usable scenario identifier
        self.scenario_id = config.ensure_id() if hasattr(config, "ensure_id") else (getattr(config, "id", None) or getattr(config, "name", "scenario"))

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


class ScenarioTemplate(BaseScenario):
    """
    Backwards-compat alias base used by scenario templates.

    Concrete templates may override additional validation hooks. This class does not
    add new abstract methods beyond BaseScenario to keep requirements minimal.
    """
    pass


__all__ = [
    "ScenarioConfig",
    "ScenarioResult",
    "BaseScenario",
    "ScenarioTemplate",
]