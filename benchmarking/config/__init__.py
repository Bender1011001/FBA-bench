"""
Configuration management for benchmarking.

This module provides comprehensive configuration management for FBA-Bench,
including schema validation, environment-specific settings, and configuration templates.

DEPRECATED: The legacy schema-based configuration system is deprecated and will be removed in a future version.
Please use the Pydantic-based configuration models (e.g., PydanticBenchmarkConfig) instead.
"""

import warnings

# Issue a deprecation warning when importing from the legacy schema system
warnings.warn(
    "The legacy schema-based configuration system in 'benchmarking.config' is deprecated and will be removed in a future version. "
    "Please use the Pydantic-based configuration models (e.g., PydanticBenchmarkConfig) instead.",
    DeprecationWarning,
    stacklevel=2
)



from .manager import (
    ConfigurationProfile,
    ConfigurationManager,
    config_manager
)
from .pydantic_config import (
    # Enums
    EnvironmentType,
    LogLevel,
    FrameworkType,
    LLMProvider,
    MetricType,
    ScenarioType,
    
    # Configuration models
    BaseConfig as PydanticBaseConfig,
    LLMConfig as PydanticLLMConfig,
    AgentCapability,
    AgentConfig as PydanticAgentConfig,
    MemoryConfig as PydanticMemoryConfig,
    CrewConfig as PydanticCrewConfig,
    ExecutionConfig as PydanticExecutionConfig,
    MetricsCollectionConfig as PydanticMetricsConfig,
    ScenarioConfig as PydanticScenarioConfig,
    BenchmarkConfig as PydanticBenchmarkConfig,
    EnvironmentConfig,
    ConfigTemplate,
    ConfigProfile as PydanticConfigProfile,
    UnifiedAgentRunnerConfig as PydanticUnifiedAgentRunnerConfig,
    
    # Builders and managers
    ConfigBuilder,
    ConfigurationManager as PydanticConfigurationManager,
    
    # Global instance
    config_manager as pydantic_config_manager
)

__all__ = [
    # Primary Pydantic configuration (canonical)
    "EnvironmentType",
    "LogLevel",
    "FrameworkType",
    "LLMProvider",
    "MetricType",
    "ScenarioType",
    "PydanticBaseConfig",
    "PydanticLLMConfig",
    "AgentCapability",
    "PydanticAgentConfig",
    "PydanticMemoryConfig",
    "PydanticCrewConfig",
    "PydanticExecutionConfig",
    "PydanticMetricsConfig",
    "PydanticScenarioConfig",
    "PydanticBenchmarkConfig",
    "EnvironmentConfig",
    "ConfigTemplate",
    "PydanticConfigProfile",
    "PydanticUnifiedAgentRunnerConfig",
    "ConfigBuilder",
    "PydanticConfigurationManager",
    "pydantic_config_manager",

    # Manager interfaces
    "ConfigurationProfile",
    "ConfigurationManager",
    "config_manager",
]