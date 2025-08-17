"""
Unified Agent Runner Factory for FBA-Bench.

This module provides a simplified factory for creating unified agent runners,
minimizing wrapping layers and enabling direct integration of native FBA agents.
"""

import logging
from typing import Dict, Type, Any, Optional, List, Union
from .base_runner import AgentRunner, AgentRunnerError, AgentRunnerInitializationError
from ..benchmarking.agents.unified_agent import UnifiedAgentInterface, UnifiedAgentConfig, UnifiedAgentAdapter
from ..benchmarking.agents.base import BaseAgent, AgentConfig

logger = logging.getLogger(__name__)


class UnifiedRunnerFactory:
    """
    Simplified factory for creating unified agent runners.
    
    This factory eliminates the need for complex wrapping layers by providing
    direct integration paths for different agent types.
    """
    
    def __init__(self):
        """Initialize the unified runner factory."""
        self._agent_types: Dict[str, Type[Union[UnifiedAgentInterface, BaseAgent]]] = {}
        self._adapter_types: Dict[str, Type[UnifiedAgentAdapter]] = {}
        
        # Register built-in agent types
        self._register_builtin_agents()
    
    def _register_builtin_agents(self) -> None:
        """Register built-in agent types."""
        # Note: Built-in agents would be registered here
        logger.info("No built-in agents registered in unified factory")
    
    def register_agent_type(
        self, 
        name: str, 
        agent_class: Type[Union[UnifiedAgentInterface, BaseAgent]],
        is_unified: bool = False
    ) -> None:
        """
        Register a new agent type.
        
        Args:
            name: Name of the agent type
            agent_class: Class implementing the agent
            is_unified: Whether the agent implements UnifiedAgentInterface
        """
        if not (issubclass(agent_class, UnifiedAgentInterface) or issubclass(agent_class, BaseAgent)):
            raise ValueError(f"Agent class {agent_class.__name__} must inherit from UnifiedAgentInterface or BaseAgent")
        
        self._agent_types[name.lower()] = agent_class
        logger.info(f"Registered agent type: {name} -> {agent_class.__name__} (unified: {is_unified})")
    
    def register_adapter_type(
        self, 
        name: str, 
        adapter_class: Type[UnifiedAgentAdapter]
    ) -> None:
        """
        Register a new adapter type.
        
        Args:
            name: Name of the adapter type
            adapter_class: Class implementing the adapter
        """
        if not issubclass(adapter_class, UnifiedAgentAdapter):
            raise ValueError(f"Adapter class {adapter_class.__name__} must inherit from UnifiedAgentAdapter")
        
        self._adapter_types[name.lower()] = adapter_class
        logger.info(f"Registered adapter type: {name} -> {adapter_class.__name__}")
    
    def create_runner(
        self, 
        agent_type: str, 
        agent_id: str, 
        config: Union[Dict[str, Any], AgentConfig, UnifiedAgentConfig]
    ) -> AgentRunner:
        """
        Create a unified agent runner.
        
        Args:
            agent_type: Type of agent to create
            agent_id: Unique identifier for the agent
            config: Agent configuration
            
        Returns:
            AgentRunner instance
        """
        agent_type_lower = agent_type.lower()
        
        # Convert config to UnifiedAgentConfig if needed
        if not isinstance(config, UnifiedAgentConfig):
            unified_config = self._convert_to_unified_config(agent_id, config)
        else:
            unified_config = config
        
        # Try to create unified agent directly
        if agent_type_lower in self._agent_types:
            agent_class = self._agent_types[agent_type_lower]
            
            try:
                # Check if it's a unified agent
                if issubclass(agent_class, UnifiedAgentInterface):
                    return agent_class(unified_config)
                else:
                    # Create adapter for BaseAgent
                    base_agent = agent_class(unified_config)
                    return UnifiedAgentAdapter(base_agent, unified_config)
                    
            except Exception as e:
                raise AgentRunnerInitializationError(
                    f"Failed to create agent of type '{agent_type}': {e}",
                    agent_id=agent_id,
                    framework=agent_type
                ) from e
        
        # Try to create using adapter
        if agent_type_lower in self._adapter_types:
            adapter_class = self._adapter_types[agent_type_lower]
            
            try:
                # Create base agent instance
                base_agent_class = self._get_base_agent_class(agent_type)
                base_agent = base_agent_class(unified_config)
                
                # Create adapter
                return adapter_class(base_agent, unified_config)
                
            except Exception as e:
                raise AgentRunnerInitializationError(
                    f"Failed to create adapter for agent type '{agent_type}': {e}",
                    agent_id=agent_id,
                    framework=agent_type
                ) from e
        
        # If no direct registration, try to create using legacy approach
        try:
            return self._create_legacy_runner(agent_type, agent_id, unified_config)
        except Exception as e:
            raise AgentRunnerInitializationError(
                f"Failed to create runner for agent type '{agent_type}': {e}",
                agent_id=agent_id,
                framework=agent_type
            ) from e
    
    def _convert_to_unified_config(
        self, 
        agent_id: str, 
        config: Union[Dict[str, Any], AgentConfig]
    ) -> UnifiedAgentConfig:
        """
        Convert configuration to UnifiedAgentConfig.
        
        Args:
            agent_id: Agent ID
            config: Original configuration
            
        Returns:
            UnifiedAgentConfig instance
        """
        if isinstance(config, AgentConfig):
            # Convert from AgentConfig
            return UnifiedAgentConfig(
                id=config.id,
                name=config.name,
                description=config.description,
                enabled=config.enabled,
                type=config.type,
                metadata=config.metadata,
                llm_config=config.llm_config,
                parameters=config.parameters,
                framework_type="direct",
                direct_integration=True
            )
        elif isinstance(config, dict):
            # Convert from dictionary
            return UnifiedAgentConfig(
                id=agent_id,
                name=config.get("name", f"Agent {agent_id}"),
                description=config.get("description", ""),
                enabled=config.get("enabled", True),
                type=config.get("type", "default"),
                metadata=config.get("metadata", {}),
                llm_config=config.get("llm_config"),
                parameters=config.get("parameters", {}),
                framework_type=config.get("framework_type", "direct"),
                direct_integration=config.get("direct_integration", True),
                framework_config=config.get("framework_config", {}),
                capabilities=config.get("capabilities", []),
                performance_profile=config.get("performance_profile", {})
            )
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")
    
    def _get_base_agent_class(self, agent_type: str) -> Type[BaseAgent]:
        """
        Get base agent class for legacy creation.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            BaseAgent class
        """
        agent_type_lower = agent_type.lower()
        
        # Import and return the appropriate agent class based on agent type
        try:
            if agent_type_lower in ["diy", "baseline"]:
                from baseline_bots.greedy_script_bot import GreedyScriptBot
                return GreedyScriptBot
            elif agent_type_lower in ["llm", "claude", "sonnet"]:
                from baseline_bots.claude_sonnet_bot import ClaudeSonnetBot
                return ClaudeSonnetBot
            elif agent_type_lower in ["advanced", "adaptive"]:
                from agents.advanced_agent import AdvancedAgent
                return AdvancedAgent
            elif agent_type_lower in ["crewai", "crew"]:
                from benchmarking.agents.unified_agent import CrewAIAdapter
                return CrewAIAdapter
            elif agent_type_lower in ["langchain", "lc"]:
                from benchmarking.agents.unified_agent import LangChainAdapter
                return LangChainAdapter
            elif agent_type_lower in ["native", "fba"]:
                from benchmarking.agents.unified_agent import NativeFBAAdapter
                return NativeFBAAdapter
            else:
                # Try to dynamically import based on agent type
                module_path = f"agents.{agent_type_lower}_agent"
                class_name = f"{agent_type.title()}Agent"
                
                try:
                    module = __import__(module_path, fromlist=[class_name])
                    agent_class = getattr(module, class_name, None)
                    if agent_class and issubclass(agent_class, BaseAgent):
                        return agent_class
                except ImportError:
                    pass
                
                # If no specific agent found, raise an error
                raise ValueError(f"Unknown agent type: {agent_type}")
                
        except ImportError as e:
            logger.error(f"Failed to import agent class for type '{agent_type}': {e}")
            raise AgentRunnerInitializationError(
                f"Agent type '{agent_type}' is not available. Please ensure the required packages are installed.",
                agent_id="unknown",
                framework=agent_type
            ) from e
    
    def _create_legacy_runner(
        self,
        agent_type: str,
        agent_id: str,
        config: UnifiedAgentConfig
    ) -> AgentRunner:
        """
        Create runner using legacy approach.
        
        This method provides support for older agent configurations that may not
        follow the new unified configuration format. It attempts to map legacy
        configurations to the appropriate runner type.
        
        Args:
            agent_type: Type of agent
            agent_id: Agent ID
            config: Unified configuration
            
        Returns:
            AgentRunner instance
        """
        try:
            logger.info(f"Creating legacy runner for agent type: {agent_type}")
            
            # Normalize agent type
            normalized_type = agent_type.lower().replace('-', '_').replace(' ', '_')
            
            # Map legacy agent types to new runner types
            legacy_type_mapping = {
                'crewai': 'crewai',
                'langchain': 'langchain',
                'diy': 'diy',
                'custom': 'diy',  # Map custom agents to DIY runner
                'legacy': 'diy',   # Map legacy agents to DIY runner
                'basic': 'diy',    # Map basic agents to DIY runner
                'advanced': 'diy', # Map advanced agents to DIY runner
            }
            
            # Get the mapped type or default to DIY
            mapped_type = legacy_type_mapping.get(normalized_type, 'diy')
            
            # Create the appropriate runner based on the mapped type
            if mapped_type == 'crewai':
                from .crewai_runner import CrewAIRunner
                return CrewAIRunner(agent_id, config)
            elif mapped_type == 'langchain':
                from .langchain_runner import LangchainRunner
                return LangchainRunner(agent_id, config)
            elif mapped_type == 'diy':
                from .diy_runner import DIYRunner
                return DIYRunner(agent_id, config)
            else:
                # Default to DIY runner for unknown types
                logger.warning(f"Unknown legacy agent type: {agent_type}, defaulting to DIY runner")
                from .diy_runner import DIYRunner
                return DIYRunner(agent_id, config)
                
        except ImportError as e:
            raise AgentRunnerInitializationError(
                f"Failed to import required module for legacy agent type {agent_type}: {e}",
                agent_id=agent_id,
                framework="Legacy"
            ) from e
        except Exception as e:
            raise AgentRunnerInitializationError(
                f"Failed to create legacy runner for agent type {agent_type}: {e}",
                agent_id=agent_id,
                framework="Legacy"
            ) from e
    
    def get_available_agent_types(self) -> List[str]:
        """
        Get list of available agent types.
        
        Returns:
            List of agent type names
        """
        return list(self._agent_types.keys())
    
    def get_available_adapter_types(self) -> List[str]:
        """
        Get list of available adapter types.
        
        Returns:
            List of adapter type names
        """
        return list(self._adapter_types.keys())
    
    def is_agent_type_registered(self, agent_type: str) -> bool:
        """
        Check if an agent type is registered.
        
        Args:
            agent_type: Type of agent
            
        Returns:
            True if registered
        """
        return agent_type.lower() in self._agent_types
    
    def is_adapter_type_registered(self, adapter_type: str) -> bool:
        """
        Check if an adapter type is registered.
        
        Args:
            adapter_type: Type of adapter
            
        Returns:
            True if registered
        """
        return adapter_type.lower() in self._adapter_types


# Global unified runner factory instance
unified_runner_factory = UnifiedRunnerFactory()


def create_unified_runner(
    agent_type: str, 
    agent_id: str, 
    config: Union[Dict[str, Any], AgentConfig, UnifiedAgentConfig]
) -> AgentRunner:
    """
    Convenience function to create a unified agent runner.
    
    Args:
        agent_type: Type of agent to create
        agent_id: Unique identifier for the agent
        config: Agent configuration
        
    Returns:
        AgentRunner instance
    """
    return unified_runner_factory.create_runner(agent_type, agent_id, config)