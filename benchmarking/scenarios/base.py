"""
Base classes for the extensible scenario framework.

This module provides the foundation for all benchmarking scenarios, including
abstract base classes and configuration structures.
"""

import abc
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Type, Union

from agent_runners.base_runner import SimulationState

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """Configuration for a scenario."""
    name: str
    description: str
    domain: str
    duration_ticks: int
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Result of a scenario execution."""
    scenario_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    success: bool = True
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tick_results: List[Dict[str, Any]] = field(default_factory=list)


class BaseScenario(abc.ABC):
    """
    Abstract base class for all scenarios.
    
    This class defines the interface that all scenarios must implement,
    including methods for initialization, execution, and validation.
    """
    
    def __init__(self, config: ScenarioConfig):
        """
        Initialize the scenario.
        
        Args:
            config: Scenario configuration
        """
        self.config = config
        self.name = config.name
        self.description = config.description
        self.domain = config.domain
        self.duration_ticks = config.duration_ticks
        self.parameters = config.parameters
        self.metadata = config.metadata
        
        # Scenario state
        self.current_tick = 0
        self.is_initialized = False
        self.is_setup = False
        self.results: List[Dict[str, Any]] = []
        
        # Validation
        self.validation_errors: List[str] = []
        
        logger.info(f"Initialized scenario: {self.name}")
    
    @property
    def is_valid(self) -> bool:
        """Check if the scenario configuration is valid."""
        return len(self.validation_errors) == 0
    
    def validate(self) -> List[str]:
        """
        Validate the scenario configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate required fields
        if not self.config.name:
            errors.append("Scenario name cannot be empty")
        
        if not self.config.description:
            errors.append("Scenario description cannot be empty")
        
        if not self.config.domain:
            errors.append("Scenario domain cannot be empty")
        
        if self.config.duration_ticks <= 0:
            errors.append("Duration ticks must be positive")
        
        # Validate difficulty
        if self.config.difficulty not in ["easy", "medium", "hard"]:
            errors.append("Difficulty must be one of: easy, medium, hard")
        
        # Validate domain-specific parameters
        domain_errors = self._validate_domain_parameters()
        errors.extend(domain_errors)
        
        self.validation_errors = errors
        return errors
    
    @abc.abstractmethod
    def _validate_domain_parameters(self) -> List[str]:
        """
        Validate domain-specific parameters.
        
        Returns:
            List of validation errors
        """
        pass
    
    @abc.abstractmethod
    async def initialize(self, parameters: Dict[str, Any]) -> None:
        """
        Initialize the scenario with parameters.
        
        Args:
            parameters: Scenario-specific parameters
        """
        pass
    
    @abc.abstractmethod
    async def setup_for_agent(self, agent_id: str) -> None:
        """
        Set up the scenario for a specific agent.
        
        Args:
            agent_id: ID of the agent
        """
        pass
    
    @abc.abstractmethod
    async def update_tick(self, tick: int, state: SimulationState) -> None:
        """
        Update the scenario for a specific tick.
        
        Args:
            tick: Current tick number
            state: Current simulation state
        """
        pass
    
    @abc.abstractmethod
    async def get_scenario_state(self) -> Dict[str, Any]:
        """
        Get the current state of the scenario.
        
        Returns:
            Dictionary representing the scenario state
        """
        pass
    
    @abc.abstractmethod
    async def evaluate_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """
        Evaluate the performance of an agent in this scenario.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with performance metrics
        """
        pass
    
    async def start(self) -> None:
        """Start the scenario."""
        if not self.is_valid:
            raise ValueError(f"Scenario configuration is invalid: {'; '.join(self.validation_errors)}")
        
        logger.info(f"Starting scenario: {self.name}")
        self.current_tick = 0
        self.results.clear()
        
        # Initialize if not already done
        if not self.is_initialized:
            await self.initialize(self.parameters)
            self.is_initialized = True
    
    async def stop(self) -> None:
        """Stop the scenario."""
        logger.info(f"Stopping scenario: {self.name}")
        self.is_setup = False
    
    async def reset(self) -> None:
        """Reset the scenario to its initial state."""
        logger.info(f"Resetting scenario: {self.name}")
        self.current_tick = 0
        self.results.clear()
        self.is_setup = False
        
        # Re-initialize
        await self.initialize(self.parameters)
    
    async def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the scenario.
        
        Returns:
            Dictionary with scenario summary
        """
        return {
            "name": self.name,
            "description": self.description,
            "domain": self.domain,
            "duration_ticks": self.duration_ticks,
            "current_tick": self.current_tick,
            "is_valid": self.is_valid,
            "is_initialized": self.is_initialized,
            "is_setup": self.is_setup,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "validation_errors": self.validation_errors
        }
    
    def record_tick_result(self, tick: int, result: Dict[str, Any]) -> None:
        """
        Record the result of a tick.
        
        Args:
            tick: Tick number
            result: Tick result data
        """
        result["tick"] = tick
        result["timestamp"] = datetime.now().isoformat()
        self.results.append(result)
    
    def get_tick_results(self, tick: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get tick results.
        
        Args:
            tick: Specific tick (None for all ticks)
            
        Returns:
            List of tick results
        """
        if tick is None:
            return self.results.copy()
        
        return [r for r in self.results if r.get("tick") == tick]
    
    def get_latest_tick_result(self) -> Optional[Dict[str, Any]]:
        """Get the most recent tick result."""
        return self.results[-1] if self.results else None
    
    def calculate_scenario_metrics(self) -> Dict[str, Any]:
        """
        Calculate scenario-level metrics.
        
        Returns:
            Dictionary with scenario metrics
        """
        if not self.results:
            return {"error": "No results available"}
        
        # Basic metrics
        total_ticks = len(self.results)
        completed_ticks = len([r for r in self.results if r.get("completed", False)])
        success_rate = completed_ticks / total_ticks if total_ticks > 0 else 0.0
        
        # Calculate average scores if available
        scores = [r.get("score", 0.0) for r in self.results if "score" in r]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "total_ticks": total_ticks,
            "completed_ticks": completed_ticks,
            "success_rate": success_rate,
            "average_score": avg_score,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0
        }


class ScenarioTemplate(BaseScenario):
    """
    Template class for creating new scenarios.
    
    This class provides a convenient base for implementing new scenarios
    with common functionality already implemented.
    """
    
    def __init__(self, config: ScenarioConfig):
        """
        Initialize the scenario template.
        
        Args:
            config: Scenario configuration
        """
        super().__init__(config)
        
        # Template-specific state
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.global_state: Dict[str, Any] = {}
    
    def _validate_domain_parameters(self) -> List[str]:
        """
        Validate domain-specific parameters.
        
        Returns:
            List of validation errors
        """
        # Override in subclasses
        return []
    
    async def initialize(self, parameters: Dict[str, Any]) -> None:
        """
        Initialize the scenario with parameters.
        
        Args:
            parameters: Scenario-specific parameters
        """
        # Store parameters
        self.parameters = parameters
        
        # Initialize global state
        self.global_state = {
            "start_time": datetime.now(),
            "total_agents": 0,
            "active_agents": 0,
            "completed_agents": 0
        }
        
        logger.info(f"Initialized scenario template: {self.name}")
    
    async def setup_for_agent(self, agent_id: str) -> None:
        """
        Set up the scenario for a specific agent.
        
        Args:
            agent_id: ID of the agent
        """
        # Initialize agent state
        self.agent_states[agent_id] = {
            "start_time": datetime.now(),
            "current_tick": 0,
            "score": 0.0,
            "completed": False,
            "errors": []
        }
        
        # Update global state
        self.global_state["total_agents"] += 1
        self.global_state["active_agents"] += 1
        
        self.is_setup = True
        logger.debug(f"Set up scenario for agent: {agent_id}")
    
    async def update_tick(self, tick: int, state: SimulationState) -> None:
        """
        Update the scenario for a specific tick.
        
        Args:
            tick: Current tick number
            state: Current simulation state
        """
        self.current_tick = tick
        
        # Update agent states
        for agent_id, agent_state in self.agent_states.items():
            agent_state["current_tick"] = tick
        
        # Record tick result
        tick_result = {
            "tick": tick,
            "timestamp": datetime.now().isoformat(),
            "global_state": self.global_state.copy(),
            "agent_states": {aid: state.copy() for aid, state in self.agent_states.items()}
        }
        
        self.record_tick_result(tick, tick_result)
        
        logger.debug(f"Updated scenario tick: {tick}")
    
    async def get_scenario_state(self) -> Dict[str, Any]:
        """
        Get the current state of the scenario.
        
        Returns:
            Dictionary representing the scenario state
        """
        return {
            "name": self.name,
            "current_tick": self.current_tick,
            "global_state": self.global_state.copy(),
            "agent_states": {aid: state.copy() for aid, state in self.agent_states.items()},
            "parameters": self.parameters.copy(),
            "metadata": self.metadata.copy()
        }
    
    async def evaluate_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """
        Evaluate the performance of an agent in this scenario.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with performance metrics
        """
        if agent_id not in self.agent_states:
            return {"error": f"Agent {agent_id} not found in scenario"}
        
        agent_state = self.agent_states[agent_id]
        
        # Calculate basic metrics
        completion_rate = 1.0 if agent_state["completed"] else 0.0
        tick_progress = self.current_tick / self.duration_ticks if self.duration_ticks > 0 else 0.0
        
        return {
            "agent_id": agent_id,
            "scenario_name": self.name,
            "score": agent_state["score"],
            "completed": agent_state["completed"],
            "completion_rate": completion_rate,
            "tick_progress": tick_progress,
            "current_tick": agent_state["current_tick"],
            "errors": agent_state["errors"],
            "start_time": agent_state["start_time"].isoformat(),
            "evaluation_time": datetime.now().isoformat()
        }
    
    def update_agent_score(self, agent_id: str, score_delta: float) -> None:
        """
        Update an agent's score.
        
        Args:
            agent_id: ID of the agent
            score_delta: Change in score
        """
        if agent_id in self.agent_states:
            self.agent_states[agent_id]["score"] += score_delta
    
    def mark_agent_completed(self, agent_id: str) -> None:
        """
        Mark an agent as completed.
        
        Args:
            agent_id: ID of the agent
        """
        if agent_id in self.agent_states:
            self.agent_states[agent_id]["completed"] = True
            self.global_state["completed_agents"] += 1
            self.global_state["active_agents"] -= 1
    
    def add_agent_error(self, agent_id: str, error: str) -> None:
        """
        Add an error for an agent.
        
        Args:
            agent_id: ID of the agent
            error: Error message
        """
        if agent_id in self.agent_states:
            self.agent_states[agent_id]["errors"].append(error)