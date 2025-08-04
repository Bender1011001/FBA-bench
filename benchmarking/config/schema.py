"""
Configuration schema validation for benchmarking.

This module provides schema validation for benchmarking configurations,
ensuring that all configuration files are properly structured and validated.
"""

import logging
import yaml
import json
from typing import Dict, Any, List, Optional, Union, Type
from dataclasses import dataclass, field
from pathlib import Path
import jsonschema

logger = logging.getLogger(__name__)


@dataclass
class SchemaValidationError:
    """Schema validation error."""
    field: str
    message: str
    value: Any = None
    expected_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "message": self.message,
            "value": self.value,
            "expected_type": self.expected_type
        }


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    errors: List[SchemaValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, field: str, message: str, value: Any = None, expected_type: Optional[str] = None) -> None:
        """Add a validation error."""
        self.errors.append(SchemaValidationError(
            field=field,
            message=message,
            value=value,
            expected_type=expected_type
        ))
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": [error.to_dict() for error in self.errors],
            "warnings": self.warnings
        }


class ConfigurationSchema:
    """Base class for configuration schemas."""
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize the configuration schema.
        
        Args:
            schema: JSON schema definition
        """
        self.schema = schema
        self.validator = jsonschema.Draft7Validator(schema)
    
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate a configuration against the schema.
        
        Args:
            config: Configuration to validate
            
        Returns:
            ValidationResult with validation results
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Validate against JSON schema
            self.validator.validate(config)
            
        except jsonschema.ValidationError as e:
            # Add validation error
            result.add_error(
                field=".".join(str(p) for p in e.path),
                message=e.message,
                value=e.instance,
                expected_type=getattr(e.validator, "type", None)
            )
        
        except jsonschema.SchemaError as e:
            # Schema error
            result.add_error(
                field="schema",
                message=f"Schema error: {e.message}",
                expected_type="valid_schema"
            )
        
        # Perform custom validation
        self._custom_validate(config, result)
        
        return result
    
    def _custom_validate(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """
        Perform custom validation beyond JSON schema.
        
        Args:
            config: Configuration to validate
            result: ValidationResult to update
        """
        # Override in subclasses
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema definition."""
        return self.schema


class BenchmarkConfigurationSchema(ConfigurationSchema):
    """Schema for benchmark configuration."""
    
    def __init__(self):
        """Initialize the benchmark configuration schema."""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Benchmark Configuration",
            "description": "Configuration schema for FBA-Bench benchmarking",
            "type": "object",
            "required": ["benchmark_id", "scenarios", "agents", "metrics"],
            "properties": {
                "benchmark_id": {
                    "type": "string",
                    "description": "Unique identifier for the benchmark"
                },
                "name": {
                    "type": "string",
                    "description": "Human-readable name for the benchmark"
                },
                "description": {
                    "type": "string",
                    "description": "Description of the benchmark"
                },
                "version": {
                    "type": "string",
                    "description": "Version of the benchmark configuration"
                },
                "environment": {
                    "type": "object",
                    "description": "Environment configuration",
                    "properties": {
                        "deterministic": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to use deterministic execution"
                        },
                        "random_seed": {
                            "type": "integer",
                            "default": 42,
                            "description": "Random seed for deterministic execution"
                        },
                        "parallel_execution": {
                            "type": "boolean",
                            "default": False,
                            "description": "Whether to enable parallel execution"
                        },
                        "max_workers": {
                            "type": "integer",
                            "default": 1,
                            "description": "Maximum number of parallel workers"
                        }
                    }
                },
                "scenarios": {
                    "type": "array",
                    "description": "List of scenarios to run",
                    "items": {
                        "type": "object",
                        "required": ["id", "type", "config"],
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique identifier for the scenario"
                            },
                            "name": {
                                "type": "string",
                                "description": "Human-readable name for the scenario"
                            },
                            "type": {
                                "type": "string",
                                "description": "Type of scenario (e.g., 'ecommerce', 'healthcare')"
                            },
                            "config": {
                                "type": "object",
                                "description": "Scenario-specific configuration"
                            },
                            "enabled": {
                                "type": "boolean",
                                "default": True,
                                "description": "Whether this scenario is enabled"
                            },
                            "priority": {
                                "type": "integer",
                                "default": 0,
                                "description": "Priority for execution ordering"
                            }
                        }
                    }
                },
                "agents": {
                    "type": "array",
                    "description": "List of agents to benchmark",
                    "items": {
                        "type": "object",
                        "required": ["id", "type", "config"],
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique identifier for the agent"
                            },
                            "name": {
                                "type": "string",
                                "description": "Human-readable name for the agent"
                            },
                            "type": {
                                "type": "string",
                                "description": "Type of agent (e.g., 'diy', 'crewai', 'langchain')"
                            },
                            "config": {
                                "type": "object",
                                "description": "Agent-specific configuration"
                            },
                            "enabled": {
                                "type": "boolean",
                                "default": True,
                                "description": "Whether this agent is enabled"
                            }
                        }
                    }
                },
                "metrics": {
                    "type": "object",
                    "description": "Metrics configuration",
                    "required": ["categories"],
                    "properties": {
                        "categories": {
                            "type": "array",
                            "description": "List of metric categories to collect",
                            "items": {
                                "type": "string",
                                "enum": ["cognitive", "business", "technical", "ethical"]
                            }
                        },
                        "custom_metrics": {
                            "type": "array",
                            "description": "Custom metric configurations",
                            "items": {
                                "type": "object",
                                "required": ["name", "type"],
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Name of the custom metric"
                                    },
                                    "type": {
                                        "type": "string",
                                        "description": "Type of the custom metric"
                                    },
                                    "config": {
                                        "type": "object",
                                        "description": "Metric-specific configuration"
                                    }
                                }
                            }
                        }
                    }
                },
                "execution": {
                    "type": "object",
                    "description": "Execution configuration",
                    "properties": {
                        "runs_per_scenario": {
                            "type": "integer",
                            "default": 1,
                            "description": "Number of runs per scenario"
                        },
                        "max_duration": {
                            "type": "integer",
                            "description": "Maximum duration in seconds (0 for unlimited)"
                        },
                        "timeout": {
                            "type": "integer",
                            "default": 300,
                            "description": "Timeout in seconds for individual runs"
                        },
                        "retry_on_failure": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to retry on failure"
                        },
                        "max_retries": {
                            "type": "integer",
                            "default": 3,
                            "description": "Maximum number of retries"
                        }
                    }
                },
                "output": {
                    "type": "object",
                    "description": "Output configuration",
                    "properties": {
                        "format": {
                            "type": "string",
                            "default": "json",
                            "enum": ["json", "csv", "yaml"],
                            "description": "Output format"
                        },
                        "path": {
                            "type": "string",
                            "description": "Output directory path"
                        },
                        "include_detailed_logs": {
                            "type": "boolean",
                            "default": False,
                            "description": "Whether to include detailed logs"
                        },
                        "include_audit_trail": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to include audit trail"
                        }
                    }
                },
                "validation": {
                    "type": "object",
                    "description": "Validation configuration",
                    "properties": {
                        "enabled": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to enable validation"
                        },
                        "statistical_significance": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to check statistical significance"
                        },
                        "confidence_level": {
                            "type": "number",
                            "default": 0.95,
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Confidence level for statistical tests"
                        },
                        "reproducibility_check": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to check reproducibility"
                        }
                    }
                }
            }
        }
        
        super().__init__(schema)
    
    def _custom_validate(self, config: Dict[str, Any], result: ValidationResult) -> None:
        """Perform custom validation for benchmark configuration."""
        # Validate scenario IDs are unique
        scenario_ids = [s.get("id") for s in config.get("scenarios", [])]
        if len(scenario_ids) != len(set(scenario_ids)):
            result.add_error(
                field="scenarios",
                message="Scenario IDs must be unique"
            )
        
        # Validate agent IDs are unique
        agent_ids = [a.get("id") for a in config.get("agents", [])]
        if len(agent_ids) != len(set(agent_ids)):
            result.add_error(
                field="agents",
                message="Agent IDs must be unique"
            )
        
        # Validate at least one scenario is enabled
        enabled_scenarios = [s for s in config.get("scenarios", []) if s.get("enabled", True)]
        if not enabled_scenarios:
            result.add_warning("No scenarios are enabled")
        
        # Validate at least one agent is enabled
        enabled_agents = [a for a in config.get("agents", []) if a.get("enabled", True)]
        if not enabled_agents:
            result.add_warning("No agents are enabled")
        
        # Validate parallel execution settings
        if config.get("environment", {}).get("parallel_execution", False):
            max_workers = config.get("environment", {}).get("max_workers", 1)
            if max_workers < 1:
                result.add_error(
                    field="environment.max_workers",
                    message="max_workers must be at least 1",
                    value=max_workers
                )
        
        # Validate execution settings
        runs_per_scenario = config.get("execution", {}).get("runs_per_scenario", 1)
        if runs_per_scenario < 1:
            result.add_error(
                field="execution.runs_per_scenario",
                message="runs_per_scenario must be at least 1",
                value=runs_per_scenario
            )
        
        max_retries = config.get("execution", {}).get("max_retries", 3)
        if max_retries < 0:
            result.add_error(
                field="execution.max_retries",
                message="max_retries must be non-negative",
                value=max_retries
            )


class ScenarioConfigurationSchema(ConfigurationSchema):
    """Schema for scenario configuration."""
    
    def __init__(self):
        """Initialize the scenario configuration schema."""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Scenario Configuration",
            "description": "Configuration schema for benchmark scenarios",
            "type": "object",
            "required": ["id", "type", "parameters"],
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Unique identifier for the scenario"
                },
                "name": {
                    "type": "string",
                    "description": "Human-readable name for the scenario"
                },
                "type": {
                    "type": "string",
                    "description": "Type of scenario"
                },
                "description": {
                    "type": "string",
                    "description": "Description of the scenario"
                },
                "parameters": {
                    "type": "object",
                    "description": "Scenario parameters",
                    "properties": {
                        "duration": {
                            "type": "integer",
                            "description": "Duration of the scenario in ticks"
                        },
                        "complexity": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Complexity level of the scenario"
                        },
                        "domain": {
                            "type": "string",
                            "description": "Domain of the scenario"
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard", "expert"],
                            "description": "Difficulty level of the scenario"
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata for the scenario",
                    "properties": {
                        "author": {
                            "type": "string",
                            "description": "Author of the scenario"
                        },
                        "version": {
                            "type": "string",
                            "description": "Version of the scenario"
                        },
                        "tags": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Tags for the scenario"
                        }
                    }
                }
            }
        }
        
        super().__init__(schema)


class AgentConfigurationSchema(ConfigurationSchema):
    """Schema for agent configuration."""
    
    def __init__(self):
        """Initialize the agent configuration schema."""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Agent Configuration",
            "description": "Configuration schema for benchmark agents",
            "type": "object",
            "required": ["id", "type", "framework_config"],
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Unique identifier for the agent"
                },
                "name": {
                    "type": "string",
                    "description": "Human-readable name for the agent"
                },
                "type": {
                    "type": "string",
                    "description": "Type of agent"
                },
                "description": {
                    "type": "string",
                    "description": "Description of the agent"
                },
                "framework_config": {
                    "type": "object",
                    "description": "Framework-specific configuration",
                    "properties": {
                        "framework": {
                            "type": "string",
                            "description": "Name of the framework"
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use"
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Framework parameters"
                        }
                    }
                },
                "capabilities": {
                    "type": "array",
                    "description": "List of agent capabilities",
                    "items": {
                        "type": "string"
                    }
                },
                "constraints": {
                    "type": "object",
                    "description": "Agent constraints",
                    "properties": {
                        "max_memory": {
                            "type": "integer",
                            "description": "Maximum memory usage in MB"
                        },
                        "max_cpu": {
                            "type": "number",
                            "description": "Maximum CPU usage percentage"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds"
                        }
                    }
                }
            }
        }
        
        super().__init__(schema)


class SchemaRegistry:
    """Registry for configuration schemas."""
    
    def __init__(self):
        """Initialize the schema registry."""
        self._schemas: Dict[str, ConfigurationSchema] = {}
        
        # Register built-in schemas
        self.register_schema("benchmark", BenchmarkConfigurationSchema())
        self.register_schema("scenario", ScenarioConfigurationSchema())
        self.register_schema("agent", AgentConfigurationSchema())
    
    def register_schema(self, name: str, schema: ConfigurationSchema) -> None:
        """
        Register a configuration schema.
        
        Args:
            name: Name of the schema
            schema: Configuration schema instance
        """
        self._schemas[name] = schema
        logger.info(f"Registered schema: {name}")
    
    def get_schema(self, name: str) -> Optional[ConfigurationSchema]:
        """
        Get a configuration schema by name.
        
        Args:
            name: Name of the schema
            
        Returns:
            Configuration schema or None if not found
        """
        return self._schemas.get(name)
    
    def validate(self, config_type: str, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate a configuration using the appropriate schema.
        
        Args:
            config_type: Type of configuration
            config: Configuration to validate
            
        Returns:
            ValidationResult with validation results
        """
        schema = self.get_schema(config_type)
        if schema is None:
            result = ValidationResult(is_valid=False)
            result.add_error(
                field="config_type",
                message=f"Unknown configuration type: {config_type}"
            )
            return result
        
        return schema.validate(config)
    
    def list_schemas(self) -> List[str]:
        """
        List all registered schema names.
        
        Returns:
            List of schema names
        """
        return list(self._schemas.keys())


# Global schema registry instance
schema_registry = SchemaRegistry()