"""
Agent Registry for the FBA-Bench benchmarking framework.

This module provides a centralized registry to manage and access different
types of agents used in benchmark scenarios.
"""

from typing import Dict, Type, Any, List

class AgentRegistry:
    """
    A registry for managing benchmark agents.

    Allows for dynamic registration and retrieval of agent implementations.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
            cls._instance._agents: Dict[str, Type[Any]] = {} # Store agent classes
        return cls._instance

    def register(self, agent_id: str, agent_class: Type[Any]):
        """
        Register an agent class with a unique ID.

        Args:
            agent_id: A unique string identifier for the agent.
            agent_class: The class of the agent to register.
        
        Raises:
            ValueError: If an agent with the same ID is already registered.
        """
        if agent_id in self._agents:
            raise ValueError(f"Agent with ID '{agent_id}' already registered.")
        self._agents[agent_id] = agent_class
        # print(f"Registered agent: {agent_id}") # For debugging

    def get(self, agent_id: str) -> Type[Any]:
        """
        Retrieve an agent class by its ID.

        Args:
            agent_id: The unique string identifier for the agent.

        Returns:
            The registered agent class.
        
        Raises:
            KeyError: If no agent with the given ID is registered.
        """
        try:
            return self._agents[agent_id]
        except KeyError:
            raise KeyError(f"No agent registered with ID '{agent_id}'.")

    def list_agents(self) -> List[str]:
        """
        List all registered agent IDs.

        Returns:
            A list of strings, where each string is an agent ID.
        """
        return list(self._agents.keys())

    def clear(self):
        """
        Clears all registered agents.
        This method is primarily for testing or reinitialization purposes.
        """
        self._agents.clear()

# Global instance of the AgentRegistry for easy access
agent_registry = AgentRegistry()