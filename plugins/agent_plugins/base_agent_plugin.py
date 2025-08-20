import logging
from typing import Dict, Any, List, Protocol

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AgentPlugin(Protocol):
    """
    Base class for custom agent plugins. Community contributors can extend this
    to implement new agent decision-making strategies and integrate them seamlessly
    with the FBA-Bench infrastructure.
    """

    __is_fba_plugin__ = True # Marker for the PluginManager to identify FBA-Bench plugins

    plugin_id: str = "base_agent_plugin"
    version: str = "0.1.0"
    name: str = "Base Agent Plugin"
    description: str = "A base class for creating FBA-Bench custom agents."
    agent_type: str = "general" # e.g., "rule_based", "learning_agent", "llm_agent"

    def initialize(self, config: Dict[str, Any]):
        """
        Initializes the agent plugin with specific configuration.
        This method should be overridden by concrete plugins.
        """
        logging.info(f"Initializing AgentPlugin: {self.name} with config: {config}")

    async def decide_action(self, current_state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines the next action(s) for the agent based on the current simulation state.
        This is the core decision-making logic of the agent.
        
        :param current_state: A dictionary representing the current state of the simulation relevant to the agent.
        :param context: Additional contextual information (e.g., historical data, specific events).
        :return: A dictionary representing the action(s) to be taken (e.g., price adjustment, inventory order).
        """
        logging.info(f"Agent {self.name} making decision based on state keys: {current_state.keys()}")
        # Example decision logic: a simple rule-based agent
        current_price = current_state.get("price", 10.0)
        current_inventory = current_state.get("inventory", 100)
        
        if current_inventory < 20:
            logging.info(f"Agent {self.name}: Low inventory ({current_inventory}), ordering more.")
            return {"type": "adjust_inventory", "value": 50}
        elif current_state.get("demand", 0) > 80 and current_price < 20.0:
            logging.info(f"Agent {self.name}: High demand, increasing price.")
            return {"type": "set_price", "value": current_price * 1.05}
        else:
            return {"type": "no_action"} # Default action

    async def learn_from_experience(self, episode_experience: Dict[str, Any]):
        """
        Allows the agent to update its internal strategy based on past outcomes.
        This method is crucial for 'Framework integration' with learning systems.
        
        :param episode_experience: Data from a past episode (e.g., states, actions, rewards, final outcomes).
        """
        logging.info(f"Agent {self.name} learning from episode experience: {episode_experience.keys()}")
        # In a real learning agent (e.g., ML-based), this would involve model training.
        # Placeholder for demonstration:
        outcomes = episode_experience.get("outcomes", {})
        if outcomes.get("profit") and outcomes["profit"] < 0:
            logging.warning(f"Agent {self.name}: Learned from negative profit episode. Adjusting strategy.")
            # self.internal_model.adjust_weights() # Example
        else:
            logging.info(f"Agent {self.name}: Episode was neutral or positive. Reinforcing strategy.")

    def get_plugin_info(self) -> Dict[str, Any]:
        """Returns metadata about the plugin."""
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "agent_type": self.agent_type,
            "supported_actions": ["set_price", "adjust_inventory"] # Example
        }

    def get_performance_benchmarks(self) -> Dict[str, Any]:
        """
        Provides metrics for comparing this agent's performance against baselines.
        This method would typically be called after a series of evaluation runs.
        """
        logging.info(f"Retrieving performance benchmarks for AgentPlugin: {self.name}")
        # Placeholder metrics. In practice, these would be computed from evaluations.
        return {
            "average_profit_per_episode": 1500.0,
            "inventory_turnover_rate": 3.5,
            "decision_latency_ms": 50,
            "evaluation_episodes_run": 100
        }


class BaseAgentPlugin:
    """
    Backwards-compat concrete base class expected by tests:
    `from plugins.agent_plugins.base_agent_plugin import BaseAgentPlugin`
    """
    __is_fba_plugin__ = True
    plugin_id: str = "base_agent_plugin"
    version: str = "0.1.0"
    name: str = "Base Agent Plugin"
    description: str = "A base class for creating FBA-Bench custom agents."
    agent_type: str = "general"

    def initialize(self, config: Dict[str, Any]):
        logging.info(f"Initializing AgentPlugin: {self.name} with config: {config}")

    async def decide_action(self, current_state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        current_price = current_state.get("price", 10.0)
        current_inventory = current_state.get("inventory", 100)
        if current_inventory < 20:
            return {"type": "adjust_inventory", "value": 50}
        if current_state.get("demand", 0) > 80 and current_price < 20.0:
            return {"type": "set_price", "value": current_price * 1.05}
        return {"type": "no_action"}

    async def learn_from_experience(self, episode_experience: Dict[str, Any]):
        logging.info(f"{self.name} learning from experience keys: {list(episode_experience.keys())}")

    def get_plugin_info(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "agent_type": self.agent_type,
            "supported_actions": ["set_price", "adjust_inventory"],
        }

    def get_performance_benchmarks(self) -> Dict[str, Any]:
        return {
            "average_profit_per_episode": 1500.0,
            "inventory_turnover_rate": 3.5,
            "decision_latency_ms": 50,
            "evaluation_episodes_run": 100,
        }

    def get_sharing_mechanism_info(self) -> Dict[str, Any]:
        """
        Provides information on how this agent's strategy can be shared or distributed.
        """
        logging.info(f"Retrieving sharing mechanism info for AgentPlugin: {self.name}")
        return {
            "export_format": "JSON_config", # Or "Python_script", "Serialized_Model"
            "dependencies": ["numpy", "tensorflow"], # If it's an ML agent
            "licensing": "MIT",
            "contribution_guidelines": "See FBA-Bench docs on agent contribution."
        }

# -----------------------------------------------------------------------------
# Plugin Agent Registration Interface (Documentation)
# -----------------------------------------------------------------------------
# Plugins can optionally provide a `register_agents(registry)` method that the
# PluginManager will invoke after plugin initialization. Use this to register
# your agents with the core AgentRegistry.
#
# Example:
#
# from benchmarking.agents.registry import AgentDescriptor
#
# class MyAgentPlugin(AgentPlugin):
#     __is_fba_plugin__ = True
#     plugin_id = "my_agent_plugin"
#     version = "1.0.0"
#     name = "My Agent Plugin"
#
#     def register_agents(self, registry):
#         class MyAgentClass:
#             def __init__(self, config=None):
#                 self.config = config
#
#         registry.register_agent(AgentDescriptor(
#             slug="my_agent",
#             display_name="My Agent",
#             constructor=MyAgentClass,
#             version="1.0.0",
#             provenance="plugin",
#             supported_capabilities=["decision_making"],
#             tags=["example", "plugin"],
#             help="Example plugin-registered agent",
#         ))
