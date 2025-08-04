"""
Agent registry for managing available agents.

This module provides a centralized registry for all available agents,
allowing for dynamic registration, discovery, and instantiation of agents.
"""

import logging
from typing import Dict, Any, List, Optional, Type, Union
from dataclasses import dataclass

from ..core.config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentRegistration:
    """Information about a registered agent."""
    name: str
    description: str
    framework: str
    agent_class: Type
    default_config: AgentConfig
    tags: List[str] = None
    enabled: bool = True
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class AgentRegistry:
    """
    Registry for managing available agents.
    
    This class provides a centralized way to register, discover, and instantiate
    agents. It supports dynamic registration and categorization of agents.
    """
    
    def __init__(self):
        """Initialize the agent registry."""
        self._agents: Dict[str, AgentRegistration] = {}
        self._frameworks: Dict[str, List[str]] = {}
        self._tags: Dict[str, List[str]] = {}
        
        # Register built-in agents
        self._register_builtin_agents()
    
    def _register_builtin_agents(self) -> None:
        """Register all built-in agents."""
        # Note: Built-in agents would be registered here
        # For now, this is a placeholder for future agent implementations
        logger.info("No built-in agents registered")
    
    def register_agent(
        self,
        name: str,
        description: str,
        framework: str,
        agent_class: Type,
        default_config: AgentConfig,
        tags: List[str] = None,
        enabled: bool = True
    ) -> None:
        """
        Register a new agent.
        
        Args:
            name: Unique name for the agent
            description: Description of the agent
            framework: Framework of the agent
            agent_class: Class implementing the agent
            default_config: Default configuration for the agent
            tags: List of tags for categorization
            enabled: Whether the agent is enabled by default
        """
        if name in self._agents:
            logger.warning(f"Agent '{name}' already registered, overwriting")
        
        registration = AgentRegistration(
            name=name,
            description=description,
            framework=framework,
            agent_class=agent_class,
            default_config=default_config,
            tags=tags or [],
            enabled=enabled
        )
        
        self._agents[name] = registration
        
        # Update framework index
        if framework not in self._frameworks:
            self._frameworks[framework] = []
        if name not in self._frameworks[framework]:
            self._frameworks[framework].append(name)
        
        # Update tag index
        for tag in registration.tags:
            if tag not in self._tags:
                self._tags[tag] = []
            if name not in self._tags[tag]:
                self._tags[tag].append(name)
        
        logger.info(f"Registered agent: {name} ({framework})")
    
    def unregister_agent(self, name: str) -> bool:
        """
        Unregister an agent.
        
        Args:
            name: Name of the agent to unregister
            
        Returns:
            True if successful, False if agent not found
        """
        if name not in self._agents:
            logger.warning(f"Agent '{name}' not found for unregistration")
            return False
        
        registration = self._agents[name]
        
        # Remove from framework index
        framework = registration.framework
        if framework in self._frameworks and name in self._frameworks[framework]:
            self._frameworks[framework].remove(name)
            if not self._frameworks[framework]:
                del self._frameworks[framework]
        
        # Remove from tag index
        for tag in registration.tags:
            if tag in self._tags and name in self._tags[tag]:
                self._tags[tag].remove(name)
                if not self._tags[tag]:
                    del self._tags[tag]
        
        # Remove from agents
        del self._agents[name]
        
        logger.info(f"Unregistered agent: {name}")
        return True
    
    def get_agent(self, name: str) -> Optional[AgentRegistration]:
        """
        Get an agent registration by name.
        
        Args:
            name: Name of the agent
            
        Returns:
            Agent registration or None if not found
        """
        return self._agents.get(name)
    
    def list_agents(self, framework: Optional[str] = None, enabled_only: bool = True) -> List[str]:
        """
        List available agents.
        
        Args:
            framework: Filter by framework (optional)
            enabled_only: Only return enabled agents
            
        Returns:
            List of agent names
        """
        agents = self._agents
        
        # Filter by framework
        if framework is not None:
            if framework in self._frameworks:
                agent_names = self._frameworks[framework]
                agents = {name: self._agents[name] for name in agent_names}
            else:
                return []
        
        # Filter by enabled status
        if enabled_only:
            agents = {name: reg for name, reg in agents.items() if reg.enabled}
        
        return list(agents.keys())
    
    def get_agents_by_framework(self) -> Dict[str, List[str]]:
        """
        Get agents grouped by framework.
        
        Returns:
            Dictionary mapping frameworks to agent names
        """
        return {framework: names.copy() for framework, names in self._frameworks.items()}
    
    def get_agents_by_tag(self, tag: str) -> List[str]:
        """
        Get agents by tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of agent names with the specified tag
        """
        return self._tags.get(tag, []).copy()
    
    def create_agent(
        self, 
        name: str, 
        config: Optional[AgentConfig] = None
    ) -> Optional[Any]:
        """
        Create an agent instance.
        
        Args:
            name: Name of the agent
            config: Agent configuration (optional)
            
        Returns:
            Agent instance or None if not found
        """
        registration = self.get_agent(name)
        if registration is None:
            logger.error(f"Agent '{name}' not found")
            return None
        
        if not registration.enabled:
            logger.warning(f"Agent '{name}' is disabled")
            return None
        
        # Use provided config or default
        agent_config = config or registration.default_config
        
        try:
            agent = registration.agent_class(agent_config)
            logger.info(f"Created agent instance: {name}")
            return agent
        except Exception as e:
            logger.error(f"Failed to create agent '{name}': {e}")
            return None
    
    def enable_agent(self, name: str) -> bool:
        """
        Enable an agent.
        
        Args:
            name: Name of the agent
            
        Returns:
            True if successful, False if agent not found
        """
        if name not in self._agents:
            return False
        
        self._agents[name].enabled = True
        logger.info(f"Enabled agent: {name}")
        return True
    
    def disable_agent(self, name: str) -> bool:
        """
        Disable an agent.
        
        Args:
            name: Name of the agent
            
        Returns:
            True if successful, False if agent not found
        """
        if name not in self._agents:
            return False
        
        self._agents[name].enabled = False
        logger.info(f"Disabled agent: {name}")
        return True
    
    def get_agent_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about an agent.
        
        Args:
            name: Name of the agent
            
        Returns:
            Dictionary with agent information
        """
        registration = self.get_agent(name)
        if registration is None:
            return {"error": f"Agent '{name}' not found"}
        
        return {
            "name": registration.name,
            "description": registration.description,
            "framework": registration.framework,
            "class": registration.agent_class.__name__,
            "module": registration.agent_class.__module__,
            "default_config": {
                "agent_id": registration.default_config.agent_id,
                "framework": registration.default_config.framework,
                "config": registration.default_config.config,
                "enabled": registration.default_config.enabled
            },
            "tags": registration.tags,
            "enabled": registration.enabled
        }
    
    def create_agent_suite(
        self, 
        agent_names: List[str], 
        configs: Dict[str, AgentConfig] = None
    ) -> Dict[str, Any]:
        """
        Create a suite of agents.
        
        Args:
            agent_names: List of agent names to include
            configs: Custom configurations for agents (optional)
            
        Returns:
            Dictionary of agent instances
        """
        suite = {}
        configs = configs or {}
        
        for name in agent_names:
            config = configs.get(name)
            agent = self.create_agent(name, config)
            if agent is not None:
                suite[name] = agent
        
        return suite
    
    def validate_agent_config(self, name: str, config: AgentConfig) -> List[str]:
        """
        Validate an agent configuration.
        
        Args:
            name: Name of the agent
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        registration = self.get_agent(name)
        if registration is None:
            errors.append(f"Unknown agent: {name}")
            return errors
        
        # Validate required fields
        if not config.agent_id:
            errors.append("Agent ID cannot be empty")
        
        if not config.framework:
            errors.append("Agent framework cannot be empty")
        
        return errors
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the registry.
        
        Returns:
            Dictionary with registry summary
        """
        enabled_count = sum(1 for r in self._agents.values() if r.enabled)
        disabled_count = len(self._agents) - enabled_count
        
        return {
            "total_agents": len(self._agents),
            "enabled_agents": enabled_count,
            "disabled_agents": disabled_count,
            "frameworks": {
                framework: len(agents)
                for framework, agents in self._frameworks.items()
            },
            "tags": {
                tag: len(agents)
                for tag, agents in self._tags.items()
            }
        }


# Global registry instance
agent_registry = AgentRegistry()