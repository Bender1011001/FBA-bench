"""
Agent Runner Factory - Central registry and factory for creating agent runners.

This factory enables framework-agnostic agent creation and provides a central
registry system for adding new agent framework support.
"""

import logging
from typing import Dict, Type, Any, Optional, List
from .base_runner import AgentRunner, AgentRunnerError

logger = logging.getLogger(__name__)


class RunnerFactory:
    """
    Factory for creating agent runners across different frameworks.
    
    Provides a centralized registry system that allows:
    - Framework-agnostic agent creation
    - Runtime framework availability checking
    - Dynamic registration of new frameworks
    - Configuration validation and defaults
    """
    
    # Registry of available runners
    _runners: Dict[str, Type[AgentRunner]] = {}
    _framework_availability: Dict[str, bool] = {}
    _default_configs: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register_runner(cls, name: str, runner_class: Type[AgentRunner], 
                       default_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a new agent runner framework.
        
        Args:
            name: Framework name (e.g., 'diy', 'crewai', 'langchain')
            runner_class: AgentRunner implementation class
            default_config: Default configuration for this framework
        """
        if not issubclass(runner_class, AgentRunner):
            raise ValueError(f"Runner class must inherit from AgentRunner, got {runner_class}")
        
        cls._runners[name] = runner_class
        cls._default_configs[name] = default_config or {}
        
        # Check framework availability
        cls._framework_availability[name] = cls._check_framework_availability(name, runner_class)
        
        logger.info(f"Registered agent runner: {name} (available: {cls._framework_availability[name]})")
    
    @classmethod
    def _check_framework_availability(cls, name: str, runner_class: Type[AgentRunner]) -> bool:
        """Check if a framework's dependencies are available."""
        try:
            # Try to create a dummy instance to check for import errors
            if name == 'crewai':
                try:
                    import crewai
                    return True
                except ImportError:
                    return False
            elif name == 'langchain':
                try:
                    import langchain
                    return True
                except ImportError:
                    return False
            elif name == 'diy':
                # DIY runner should always be available
                return True
            else:
                # For custom runners, assume available unless we can check
                return True
        except Exception as e:
            logger.warning(f"Framework availability check failed for {name}: {e}")
            return False
    
    @classmethod
    def create_runner(cls, framework: str, agent_id: str, config: Dict[str, Any]) -> AgentRunner:
        """
        Create an agent runner for the specified framework.
        
        Args:
            framework: Framework name (e.g., 'diy', 'crewai', 'langchain')
            agent_id: Unique identifier for the agent
            config: Configuration dictionary for the runner
            
        Returns:
            Initialized AgentRunner instance
            
        Raises:
            AgentRunnerError: If framework is unknown or unavailable
        """
        if framework not in cls._runners:
            available_frameworks = list(cls._runners.keys())
            raise AgentRunnerError(
                f"Unknown framework: {framework}. Available: {available_frameworks}",
                agent_id=agent_id,
                framework=framework
            )
        
        if not cls._framework_availability.get(framework, False):
            raise AgentRunnerError(
                f"Framework {framework} is not available. Check dependencies.",
                agent_id=agent_id,
                framework=framework
            )
        
        # Merge default config with provided config
        merged_config = cls._default_configs[framework].copy()
        merged_config.update(config)
        
        try:
            runner_class = cls._runners[framework]
            runner = runner_class(agent_id, merged_config)
            logger.info(f"Created {framework} runner for agent {agent_id}")
            return runner
        except Exception as e:
            raise AgentRunnerError(
                f"Failed to create {framework} runner: {str(e)}",
                agent_id=agent_id,
                framework=framework
            ) from e
    
    @classmethod
    async def create_and_initialize_runner(cls, framework: str, agent_id: str, 
                                         config: Dict[str, Any]) -> AgentRunner:
        """
        Create and initialize an agent runner in one step.
        
        Args:
            framework: Framework name
            agent_id: Unique identifier for the agent
            config: Configuration dictionary
            
        Returns:
            Initialized and ready-to-use AgentRunner instance
        """
        runner = cls.create_runner(framework, agent_id, config)
        await runner.initialize(config)
        return runner
    
    @classmethod
    def get_available_frameworks(cls) -> List[str]:
        """Get list of available framework names."""
        return [name for name, available in cls._framework_availability.items() if available]
    
    @classmethod
    def get_all_frameworks(cls) -> List[str]:
        """Get list of all registered framework names."""
        return list(cls._runners.keys())
    
    @classmethod
    def is_framework_available(cls, framework: str) -> bool:
        """Check if a specific framework is available."""
        return cls._framework_availability.get(framework, False)
    
    @classmethod
    def get_framework_info(cls, framework: str) -> Dict[str, Any]:
        """Get information about a specific framework."""
        if framework not in cls._runners:
            raise ValueError(f"Unknown framework: {framework}")
        
        runner_class = cls._runners[framework]
        return {
            "name": framework,
            "class": runner_class.__name__,
            "module": runner_class.__module__,
            "available": cls._framework_availability.get(framework, False),
            "default_config": cls._default_configs.get(framework, {}),
            "description": runner_class.__doc__.split('\n')[0] if runner_class.__doc__ else "No description"
        }
    
    @classmethod
    def get_all_framework_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered frameworks."""
        return {name: cls.get_framework_info(name) for name in cls._runners.keys()}
    
    @classmethod
    def validate_config(cls, framework: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize configuration for a framework.
        
        Args:
            framework: Framework name
            config: Configuration to validate
            
        Returns:
            Validated and normalized configuration
            
        Raises:
            AgentRunnerError: If configuration is invalid
        """
        if framework not in cls._runners:
            raise AgentRunnerError(f"Unknown framework: {framework}")
        
        # Start with defaults
        validated_config = cls._default_configs[framework].copy()
        validated_config.update(config)
        
        # Framework-specific validation
        if framework == 'diy':
            cls._validate_diy_config(validated_config)
        elif framework == 'crewai':
            cls._validate_crewai_config(validated_config)
        elif framework == 'langchain':
            cls._validate_langchain_config(validated_config)
        
        return validated_config
    
    @classmethod
    def _validate_diy_config(cls, config: Dict[str, Any]) -> None:
        """Validate DIY runner configuration."""
        agent_type = config.get('agent_type', 'advanced')
        valid_types = ['advanced', 'baseline', 'llm']
        
        if agent_type not in valid_types:
            raise AgentRunnerError(f"Invalid DIY agent_type: {agent_type}. Valid types: {valid_types}")
        
        if agent_type == 'baseline':
            bot_type = config.get('bot_type', 'greedy')
            if bot_type not in ['greedy']:  # Add more as implemented
                raise AgentRunnerError(f"Invalid baseline bot_type: {bot_type}")
        
        elif agent_type == 'llm':
            llm_type = config.get('llm_type', 'claude')
            if llm_type not in ['claude', 'gpt']:  # Add more as implemented
                raise AgentRunnerError(f"Invalid LLM llm_type: {llm_type}")
    
    @classmethod
    def _validate_crewai_config(cls, config: Dict[str, Any]) -> None:
        """Validate CrewAI runner configuration."""
        llm_config = config.get('llm_config', {})
        if not llm_config.get('provider'):
            config.setdefault('llm_config', {})['provider'] = 'openai'
        
        agent_config = config.get('agent_config', {})
        crew_config = config.get('crew_config', {})
        
        # Set defaults
        if 'allow_delegation' not in agent_config:
            agent_config['allow_delegation'] = True
        if 'process' not in crew_config:
            crew_config['process'] = 'sequential'
    
    @classmethod
    def _validate_langchain_config(cls, config: Dict[str, Any]) -> None:
        """Validate LangChain runner configuration."""
        llm_config = config.get('llm_config', {})
        if not llm_config.get('model'):
            config.setdefault('llm_config', {})['model'] = 'gpt-4'
        
        memory_config = config.get('memory_config', {})
        if memory_config.get('type') not in ['buffer', 'summary', None]:
            raise AgentRunnerError(f"Invalid memory type: {memory_config.get('type')}")
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear the runner registry (mainly for testing)."""
        cls._runners.clear()
        cls._framework_availability.clear()
        cls._default_configs.clear()


# Auto-register built-in runners
def _register_builtin_runners():
    """Register the built-in agent runners."""
    try:
        # Register DIY runner
        from .diy_runner import DIYRunner
        RunnerFactory.register_runner('diy', DIYRunner, {
            'agent_type': 'advanced',
            'strategy': 'profit_maximizer',
            'price_sensitivity': 0.1,
            'reaction_speed': 1,
            'min_price_cents': 500,
            'max_price_cents': 5000
        })
    except ImportError as e:
        logger.warning(f"Failed to register DIY runner: {e}")
    
    try:
        # Register CrewAI runner
        from .crewai_runner import CrewAIRunner
        RunnerFactory.register_runner('crewai', CrewAIRunner, {
            'llm_config': {
                'provider': 'openai',
                'model': 'gpt-4',
                'temperature': 0.1
            },
            'agent_config': {
                'allow_delegation': True
            },
            'crew_config': {
                'process': 'sequential'
            },
            'verbose': False
        })
    except ImportError as e:
        logger.warning(f"Failed to register CrewAI runner: {e}")
    
    try:
        # Register LangChain runner
        from .langchain_runner import LangChainRunner
        RunnerFactory.register_runner('langchain', LangChainRunner, {
            'llm_config': {
                'model': 'gpt-4',
                'temperature': 0.1
            },
            'memory_config': {
                'type': 'buffer',
                'window_size': 10
            },
            'max_iterations': 5,
            'verbose': False
        })
    except ImportError as e:
        logger.warning(f"Failed to register LangChain runner: {e}")


# Register built-in runners on module import
_register_builtin_runners()


class AgentRunnerBuilder:
    """
    Builder pattern for creating agent runners with fluent configuration.
    
    Provides a more convenient way to create and configure agent runners
    with method chaining and validation.
    """
    
    def __init__(self, framework: str, agent_id: str):
        self.framework = framework
        self.agent_id = agent_id
        self.config: Dict[str, Any] = {}
    
    def with_llm(self, provider: str, model: str, temperature: float = 0.1, 
                 api_key: Optional[str] = None) -> 'AgentRunnerBuilder':
        """Configure LLM settings."""
        self.config.setdefault('llm_config', {}).update({
            'provider': provider,
            'model': model,
            'temperature': temperature
        })
        if api_key:
            self.config['llm_config']['api_key'] = api_key
        return self
    
    def with_memory(self, memory_type: str, **kwargs) -> 'AgentRunnerBuilder':
        """Configure memory settings."""
        self.config.setdefault('memory_config', {}).update({
            'type': memory_type,
            **kwargs
        })
        return self
    
    def with_tools(self, tools: List[Any]) -> 'AgentRunnerBuilder':
        """Configure custom tools."""
        self.config['custom_tools'] = tools
        return self
    
    def with_agent_type(self, agent_type: str) -> 'AgentRunnerBuilder':
        """Configure agent type (for DIY runner)."""
        self.config['agent_type'] = agent_type
        return self
    
    def with_config(self, **kwargs) -> 'AgentRunnerBuilder':
        """Add arbitrary configuration options."""
        self.config.update(kwargs)
        return self
    
    def build(self) -> AgentRunner:
        """Build the agent runner."""
        return RunnerFactory.create_runner(self.framework, self.agent_id, self.config)
    
    async def build_and_initialize(self) -> AgentRunner:
        """Build and initialize the agent runner."""
        return await RunnerFactory.create_and_initialize_runner(
            self.framework, self.agent_id, self.config
        )


def create_agent_builder(framework: str, agent_id: str) -> AgentRunnerBuilder:
    """Convenience function to create an agent runner builder."""
    return AgentRunnerBuilder(framework, agent_id)