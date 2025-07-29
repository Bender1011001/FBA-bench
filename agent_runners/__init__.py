"""
Framework-agnostic agent runner abstraction layer for FBA-Bench.

This module provides a unified interface for different agent frameworks
(DIY, CrewAI, LangChain, etc.) while keeping the core simulation framework-agnostic.

Key Components:
- AgentRunner: Base interface for all agent frameworks
- RunnerFactory: Factory for creating and managing agent runners
- AgentManager: Integration layer with simulation orchestrator
- Dependency management: Handles optional framework dependencies
- Configuration system: Structured configs for all frameworks
"""

from .base_runner import AgentRunner, SimulationState, ToolCall, AgentRunnerError
from .runner_factory import RunnerFactory, AgentRunnerBuilder, create_agent_builder
from .agent_manager import AgentManager
from .dependency_manager import (
    DependencyManager,
    dependency_manager,
    check_framework_availability,
    get_available_frameworks,
    install_framework
)
from .configs import (
    AgentRunnerConfig,
    DIYConfig,
    CrewAIConfig,
    LangChainConfig,
    validate_config,
    load_config_from_file
)

__all__ = [
    # Core interfaces
    'AgentRunner',
    'SimulationState',
    'ToolCall',
    'AgentRunnerError',
    
    # Factory system
    'RunnerFactory',
    'AgentRunnerBuilder',
    'create_agent_builder',
    
    # Integration
    'AgentManager',
    
    # Dependency management
    'DependencyManager',
    'dependency_manager',
    'check_framework_availability',
    'get_available_frameworks',
    'install_framework',
    
    # Configuration
    'AgentRunnerConfig',
    'DIYConfig',
    'CrewAIConfig',
    'LangChainConfig',
    'validate_config',
    'load_config_from_file'
]

# Framework availability info
def get_framework_status():
    """Get status of all supported frameworks."""
    return {
        'available_frameworks': get_available_frameworks(),
        'all_frameworks': list(RunnerFactory.get_all_frameworks()),
        'framework_info': dependency_manager.get_all_framework_info()
    }