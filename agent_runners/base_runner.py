"""
Base AgentRunner interface and data structures for framework-agnostic agent abstraction.

This module defines the core interface that all agent frameworks must implement,
enabling seamless swapping between DIY, CrewAI, LangChain, and future frameworks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from money import Money
from models.product import Product


@dataclass
class SimulationState:
    """
    Unified simulation state containing all information an agent needs to make decisions.
    
    This standardized state format ensures all agent frameworks receive the same
    comprehensive view of the simulation, regardless of their internal implementation.
    """
    # Core simulation metadata
    tick: int
    simulation_time: datetime
    
    # Product portfolio with full market context
    products: List[Product]
    
    # Recent events for context-aware decision making
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Financial position and constraints
    financial_position: Dict[str, Any] = field(default_factory=dict)
    
    # Market conditions and competitor intelligence
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Agent-specific state (budget, tokens, etc.)
    agent_state: Dict[str, Any] = field(default_factory=dict)
    
    def get_product(self, asin: str) -> Optional[Product]:
        """Get a specific product by ASIN."""
        for product in self.products:
            if product.asin == asin:
                return product
        return None
    
    def get_recent_events_since_tick(self, since_tick: int) -> List[Dict[str, Any]]:
        """Get recent events since a specific tick."""
        return [event for event in self.recent_events 
                if event.get('tick', 0) >= since_tick]


@dataclass
class ToolCall:
    """
    Standardized tool call representation for agent actions.
    
    All agent frameworks must return their intended actions in this format,
    ensuring consistent interpretation by the simulation engine.
    """
    # Tool identification
    tool_name: str
    
    # Tool parameters as key-value pairs
    parameters: Dict[str, Any]
    
    # Agent's confidence in this decision (0.0 to 1.0)
    confidence: float = 1.0
    
    # Optional reasoning for debugging/analysis
    reasoning: str = ""
    
    # Priority for execution ordering (higher = more urgent)
    priority: int = 0
    
    def __post_init__(self):
        """Validate tool call data."""
        if not self.tool_name:
            raise ValueError("tool_name cannot be empty")
        
        if not isinstance(self.parameters, dict):
            raise TypeError("parameters must be a dictionary")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


class AgentRunner(ABC):
    """
    Abstract base class for all agent framework runners.
    
    This interface provides a unified way for the simulation to interact with
    agents regardless of their underlying framework (DIY, CrewAI, LangChain, etc.).
    
    Key design principles:
    - Framework agnostic: Core simulation doesn't know which framework is running
    - Async by default: Supports both local and remote agent execution
    - Stateless interface: All state passed in SimulationState, no hidden dependencies
    - Tool-based actions: All agent intentions expressed as standardized ToolCalls
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initialize the agent runner.
        
        Args:
            agent_id: Unique identifier for this agent instance
            config: Framework-specific configuration dictionary
        """
        self.agent_id = agent_id
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def decide(self, state: SimulationState) -> List[ToolCall]:
        """
        Core decision-making interface that all agent frameworks must implement.
        
        This is the single point of interaction between the simulation and the agent.
        The agent receives complete simulation state and returns a list of tool calls
        representing its intended actions.
        
        Args:
            state: Current simulation state with all relevant information
            
        Returns:
            List of tool calls the agent wants to execute this tick
            
        Raises:
            AgentRunnerError: If the agent encounters an error during decision making
        """
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the agent with configuration.
        
        Called once before the first decide() call. Use this to set up
        framework-specific resources, load models, establish connections, etc.
        
        Args:
            config: Framework-specific configuration dictionary
            
        Raises:
            AgentRunnerError: If initialization fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup resources when simulation ends.
        
        Called once when the simulation is shutting down. Use this to
        close connections, save state, release resources, etc.
        
        Raises:
            AgentRunnerError: If cleanup fails (non-critical)
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Optional health check for monitoring agent status.
        
        Returns:
            Dictionary with health status information
        """
        return {
            "agent_id": self.agent_id,
            "status": "healthy" if self._initialized else "not_initialized",
            "framework": self.__class__.__name__,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_framework_info(self) -> Dict[str, Any]:
        """
        Get information about the agent framework.
        
        Returns:
            Dictionary with framework metadata
        """
        return {
            "framework_name": self.__class__.__name__,
            "agent_id": self.agent_id,
            "config_keys": list(self.config.keys()) if self.config else []
        }


class AgentRunnerError(Exception):
    """Base exception for agent runner errors."""
    
    def __init__(self, message: str, agent_id: str = None, framework: str = None):
        super().__init__(message)
        self.agent_id = agent_id
        self.framework = framework
        self.timestamp = datetime.utcnow()
    
    def __str__(self):
        parts = [self.args[0]]
        if self.agent_id:
            parts.append(f"agent_id={self.agent_id}")
        if self.framework:
            parts.append(f"framework={self.framework}")
        return f"{' '.join(parts)} at {self.timestamp.isoformat()}"


class AgentRunnerInitializationError(AgentRunnerError):
    """Raised when agent runner initialization fails."""
    pass


class AgentRunnerDecisionError(AgentRunnerError):
    """Raised when agent runner decision making fails."""
    pass


class AgentRunnerCleanupError(AgentRunnerError):
    """Raised when agent runner cleanup fails."""
    pass