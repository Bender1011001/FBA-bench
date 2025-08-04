"""
Configuration management for benchmarking.

This module provides comprehensive configuration management for FBA-Bench,
including schema validation, environment-specific settings, and configuration templates.
"""

from .schema import (
    SchemaValidationError,
    ValidationResult,
    ConfigurationSchema,
    BenchmarkConfigurationSchema,
    ScenarioConfigurationSchema,
    AgentConfigurationSchema,
    SchemaRegistry,
    schema_registry
)

from .manager import (
    ConfigurationProfile,
    ConfigurationManager,
    config_manager
)

__all__ = [
    "SchemaValidationError",
    "ValidationResult",
    "ConfigurationSchema",
    "BenchmarkConfigurationSchema",
    "ScenarioConfigurationSchema",
    "AgentConfigurationSchema",
    "SchemaRegistry",
    "schema_registry",
    "ConfigurationProfile",
    "ConfigurationManager",
    "config_manager"
]