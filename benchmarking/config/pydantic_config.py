"""
Enhanced Configuration Management with Pydantic for FBA-Bench.

This module provides centralized, user-friendly configuration schemas using Pydantic,
programmatic configuration builders, and secure handling of sensitive information.
"""

import os
import yaml
import json
from typing import Dict, Any, List, Optional, Union, Type, TypeVar, Generic
from pathlib import Path
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, SecretStr, ConfigDict

# Type variables for generic models
T = TypeVar('T')

class EnvironmentType(str, Enum):
    """Environment types for configuration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Log levels for configuration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class FrameworkType(str, Enum):
    """Supported agent framework types."""
    DIRECT = "direct"
    DIY = "diy"
    CREWAI = "crewai"
    LANGCHAIN = "langchain"
    ADAPTED = "adapted"

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    OPENROUTER = "openrouter"
    LOCAL = "local"

class MetricType(str, Enum):
    """Types of metrics."""
    COGNITIVE = "cognitive"
    BUSINESS = "business"
    TECHNICAL = "technical"
    ETHICAL = "ethical"
    CUSTOM = "custom"

class ScenarioType(str, Enum):
    """Types of scenarios."""
    BUSINESS_SIMULATION = "business_simulation"
    COGNITIVE_TEST = "cognitive_test"
    MULTI_AGENT = "multi_agent"
    CUSTOM = "custom"

class Tier(str, Enum):
    """Constraint tiers for benchmark runs."""
    T0 = "T0"
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"

# Base configuration models
class BaseConfig(BaseModel):
    """Base configuration model with common fields."""
    name: str = Field(..., description="Name of the configuration")
    description: str = Field("", description="Description of the configuration")
    enabled: bool = Field(True, description="Whether this configuration is enabled")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True
    )

class LLMConfig(BaseConfig):
    """LLM configuration using Pydantic."""
    provider: LLMProvider = Field(LLMProvider.OPENAI, description="LLM provider")
    model: str = Field(..., description="Model name")
    api_key: Optional[SecretStr] = Field(None, description="API key (secure)")
    base_url: Optional[str] = Field(None, description="Base URL for API")
    max_tokens: int = Field(2048, description="Maximum tokens for response")
    temperature: float = Field(0.7, description="Temperature for generation", ge=0.0, le=2.0)
    top_p: float = Field(1.0, description="Top-p for generation", ge=0.0, le=1.0)
    timeout: int = Field(30, description="Timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    
    @field_validator('api_key', mode='before')
    def validate_api_key(cls, v):
        """Validate API key from environment if not provided."""
        if v is None:
            # Try to get from environment variable
            env_key = os.getenv('LLM_API_KEY')
            if env_key:
                return env_key
        return v

class AgentCapability(BaseModel):
    """Agent capability definition."""
    name: str = Field(..., description="Name of the capability")
    description: str = Field("", description="Description of the capability")
    version: str = Field("1.0.0", description="Version of the capability")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Capability parameters")

class AgentConfig(BaseConfig):
    """Agent configuration using Pydantic."""
    agent_id: str = Field(..., description="Unique identifier for the agent")
    type: str = Field("default", description="Type of agent")
    framework: FrameworkType = Field(FrameworkType.DIRECT, description="Framework type")
    llm_config: Optional[LLMConfig] = Field(None, description="LLM configuration")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="Agent capabilities")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific parameters")
    performance_profile: Dict[str, Any] = Field(default_factory=dict, description="Performance profile")
    
    @model_validator(mode='before')
    def validate_agent_config(cls, values):
        """Validate agent configuration."""
        framework = values.get('framework')
        llm_config = values.get('llm_config')
        
        # Validate that LLM config is provided for LLM-based agents
        if framework in [FrameworkType.DIRECT, FrameworkType.DIY] and llm_config is None:
            raise ValueError("LLM configuration is required for direct and DIY agents")
        
        return values

class MemoryConfig(BaseConfig):
    """Configuration for agent memory systems using Pydantic."""
    type: str = Field("buffer", description="Type of memory: 'buffer', 'summary', or 'none'")
    window_size: int = Field(10, description="Size of the memory window", ge=1)
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for memory summary", gt=0)
    
    @field_validator('type')
    def validate_memory_type(cls, v):
        if v not in ["buffer", "summary", "none"]:
            raise ValueError("Memory type must be 'buffer', 'summary', or 'none'")
        return v

class CrewConfig(BaseConfig):
    """Configuration for CrewAI crew setup using Pydantic."""
    process: str = Field("sequential", description="Crew process: 'sequential' or 'hierarchical'")
    crew_size: int = Field(4, description="Number of agents in the crew", ge=1)
    roles: List[str] = Field(
        default_factory=lambda: ["pricing_specialist", "inventory_manager", "market_analyst", "strategy_coordinator"],
        description="Roles for the crew members"
    )
    collaboration_mode: str = Field("sequential", description="How crew members collaborate")
    allow_delegation: bool = Field(True, description="Whether to allow task delegation")
    
    @field_validator('process')
    def validate_process(cls, v):
        if v not in ["sequential", "hierarchical"]:
            raise ValueError("Process must be 'sequential' or 'hierarchical'")
        return v

class ExecutionConfig(BaseConfig):
    """Execution configuration using Pydantic."""
    num_runs: int = Field(1, description="Number of runs to execute", ge=1)
    parallel_execution: bool = Field(True, description="Whether to execute agents in parallel")
    timeout_seconds: int = Field(300, description="Timeout in seconds", gt=0)
    output_dir: str = Field("./results", description="Output directory for results")
    save_intermediate_results: bool = Field(False, description="Whether to save intermediate results")
    
class MetricsCollectionConfig(BaseConfig):
    """Metrics collection configuration."""
    collection_interval: int = Field(10, description="Interval for metrics collection", ge=1)
    enabled_metrics: List[MetricType] = Field(
        default_factory=lambda: [MetricType.COGNITIVE, MetricType.BUSINESS],
        description="Enabled metric types"
    )
    custom_metrics: List[str] = Field(default_factory=list, description="Custom metric names")
    aggregation_method: str = Field("average", description="Method for aggregating metrics")

class ScenarioConfig(BaseConfig):
    """Scenario configuration using Pydantic."""
    scenario_type: ScenarioType = Field(ScenarioType.BUSINESS_SIMULATION, description="Type of scenario")
    duration_ticks: int = Field(100, description="Duration in ticks", ge=1)
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Scenario-specific parameters")
    version: str = Field("1.0.0", description="Scenario version")
    dependencies: List[str] = Field(default_factory=list, description="Scenario dependencies")
    
class BenchmarkConfig(BaseConfig):
    """Main benchmark configuration using Pydantic."""
    benchmark_id: str = Field(..., description="Unique identifier for the benchmark")
    environment: EnvironmentType = Field(EnvironmentType.DEVELOPMENT, description="Environment type")
    log_level: LogLevel = Field(LogLevel.INFO, description="Log level")

    # Constraints/tier configuration
    tier: Tier = Field(Tier.T0, description="Benchmark constraint tier (T0-T3)")
    # Optional budget overrides to replace tier defaults in BudgetEnforcer when provided.
    # Schema should match constraints.budget_enforcer dict format:
    # {
    #   "limits": {"total_tokens_per_tick": int, "total_tokens_per_run": int, ...},
    #   "tool_limits": { "tool": {...} },
    #   "warning_threshold_pct": float,
    #   "allow_soft_overage": bool
    # }
    budget_overrides: Optional[Dict[str, Any]] = Field(None, description="Optional budget constraints override")

    # Sub-configurations
    agents: List[AgentConfig] = Field(default_factory=list, description="Agent configurations")
    scenarios: List[ScenarioConfig] = Field(default_factory=list, description="Scenario configurations")
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig, description="Execution configuration")
    metrics: MetricsCollectionConfig = Field(default_factory=MetricsCollectionConfig, description="Metrics configuration")

    # Weighting for KPI aggregation (used by engine.metric_suite)
    metric_weights: Dict[str, float] = Field(default_factory=dict, description="Weights for KPI aggregation")

    # Additional settings
    tags: List[str] = Field(default_factory=list, description="Tags for the benchmark")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

    @model_validator(mode='before')
    def validate_benchmark_config(cls, values):
        """Validate benchmark configuration."""
        agents = values.get('agents', [])
        scenarios = values.get('scenarios', [])

        # Validate that at least one agent and one scenario are provided
        if not agents:
            raise ValueError("At least one agent configuration is required")
        if not scenarios:
            raise ValueError("At least one scenario configuration is required")

        # Validate agent IDs are unique
        agent_ids = [agent.agent_id for agent in agents]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("Agent IDs must be unique")

        # Validate budget_overrides when provided
        bo = values.get('budget_overrides')
        if bo is not None and not isinstance(bo, dict):
            raise ValueError("budget_overrides must be a dict matching BudgetEnforcer config schema")
        # Minimal structure validation if present
        if isinstance(bo, dict):
            limits = bo.get("limits")
            if limits is not None and not isinstance(limits, dict):
                raise ValueError("budget_overrides.limits must be a dict of limit fields")

        # Update updated_at timestamp
        values['updated_at'] = datetime.now()

        return values

class UnifiedAgentRunnerConfig(BaseConfig):
    """
    Unified configuration for agent runners across all frameworks using Pydantic.
    
    This provides a structured way to configure agents with validation,
    defaults, and framework-specific sections, consolidating the previous
    AgentRunnerConfig from agent_runners/configs/config_schema.py.
    """
    # Core identification
    agent_id: str = Field(..., description="Unique identifier for the agent")
    framework: FrameworkType = Field(..., description="Agent framework type")
    
    # Framework-specific configurations
    llm_config: Optional[LLMConfig] = Field(None, description="LLM configuration")
    memory_config: Optional[MemoryConfig] = Field(None, description="Memory system configuration")
    agent_config: Optional[AgentConfig] = Field(None, description="Agent-specific behavior configuration")
    crew_config: Optional[CrewConfig] = Field(None, description="CrewAI-specific configuration")
    
    # General settings
    verbose: bool = Field(False, description="Enable verbose logging")
    max_iterations: int = Field(5, description="Maximum number of iterations", ge=1)
    timeout_seconds: int = Field(30, description="Timeout in seconds", gt=0)
    
    # Custom configurations
    custom_tools: List[Dict[str, Any]] = Field(default_factory=list, description="Custom tools for the agent")
    custom_config: Dict[str, Any] = Field(default_factory=dict, description="Additional custom configuration")
    
    @model_validator(mode='before')
    def set_framework_defaults(cls, values):
        """Set framework-specific defaults and validate configuration."""
        framework = values.get('framework')
        llm_config = values.get('llm_config')
        memory_config = values.get('memory_config')
        agent_config = values.get('agent_config')
        crew_config = values.get('crew_config')
        
        if framework == FrameworkType.DIY:
            if agent_config is None:
                values['agent_config'] = AgentConfig(name=f"{values.get('agent_id')}_config", agent_id=values.get('agent_id'))
        
        elif framework == FrameworkType.CREWAI:
            if llm_config is None:
                values['llm_config'] = LLMConfig(name=f"{values.get('agent_id')}_llm", model="gpt-4")
            if crew_config is None:
                values['crew_config'] = CrewConfig(name=f"{values.get('agent_id')}_crew")
        
        elif framework == FrameworkType.LANGCHAIN:
            if llm_config is None:
                values['llm_config'] = LLMConfig(name=f"{values.get('agent_id')}_llm", model="gpt-4")
            if memory_config is None:
                values['memory_config'] = MemoryConfig(name=f"{values.get('agent_id')}_memory")
        
        return values

class EnvironmentConfig(BaseConfig):
    """Environment-specific configuration."""
    environment: EnvironmentType = Field(..., description="Environment type")
    database_url: Optional[SecretStr] = Field(None, description="Database URL")
    redis_url: Optional[SecretStr] = Field(None, description="Redis URL")
    api_keys: Dict[str, SecretStr] = Field(default_factory=dict, description="API keys for services")
    feature_flags: Dict[str, bool] = Field(default_factory=dict, description="Feature flags")
    
    @field_validator('database_url', mode='before')
    def validate_database_url(cls, v):
        """Validate database URL from environment if not provided."""
        if v is None:
            env_url = os.getenv('DATABASE_URL')
            if env_url:
                return env_url
        return v
    
    @field_validator('redis_url', mode='before')
    def validate_redis_url(cls, v):
        """Validate Redis URL from environment if not provided."""
        if v is None:
            env_url = os.getenv('REDIS_URL')
            if env_url:
                return env_url
        return v

class ConfigTemplate(BaseModel):
    """Configuration template definition."""
    name: str = Field(..., description="Template name")
    description: str = Field("", description="Template description")
    config_type: str = Field(..., description="Type of configuration this template creates")
    template: Dict[str, Any] = Field(..., description="Template configuration")
    required_fields: List[str] = Field(default_factory=list, description="Required fields")
    optional_fields: List[str] = Field(default_factory=list, description="Optional fields")
    
class ConfigProfile(BaseModel):
    """Configuration profile for different environments/use cases."""
    name: str = Field(..., description="Profile name")
    description: str = Field("", description="Profile description")
    base_config: Dict[str, Any] = Field(..., description="Base configuration")
    overrides: Dict[str, Any] = Field(default_factory=dict, description="Profile-specific overrides")
    environment: EnvironmentType = Field(EnvironmentType.DEVELOPMENT, description="Target environment")

# Configuration builders
class ConfigBuilder(Generic[T]):
    """Generic configuration builder."""
    
    def __init__(self, config_class: Type[T]):
        """Initialize the builder."""
        self.config_class = config_class
        self.config_data = {}
    
    def set_field(self, field_name: str, value: Any) -> 'ConfigBuilder[T]':
        """Set a field value."""
        self.config_data[field_name] = value
        return self
    
    def set_fields(self, **kwargs) -> 'ConfigBuilder[T]':
        """Set multiple field values."""
        self.config_data.update(kwargs)
        return self
    
    def from_template(self, template: Dict[str, Any]) -> 'ConfigBuilder[T]':
        """Load configuration from template."""
        self.config_data.update(template)
        return self
    
    def from_file(self, file_path: Union[str, Path]) -> 'ConfigBuilder[T]':
        """Load configuration from file."""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        self.config_data.update(data)
        return self
    
    def apply_environment_overrides(self, prefix: str = "FBA_") -> 'ConfigBuilder[T]':
        """Apply environment variable overrides."""
        env_vars = {k: v for k, v in os.environ.items() if k.startswith(prefix)}
        
        for env_var, env_value in env_vars:
            # Convert environment variable name to configuration path
            config_path = env_var[len(prefix):].lower().split('_')
            
            # Navigate to the target location in the configuration
            current = self.config_data
            for part in config_path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the final value with type conversion
            final_key = config_path[-1]
            if final_key in current:
                # Try to maintain the original type
                original_value = current[final_key]
                if isinstance(original_value, bool):
                    current[final_key] = env_value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(original_value, int):
                    current[final_key] = int(env_value)
                elif isinstance(original_value, float):
                    current[final_key] = float(env_value)
                else:
                    current[final_key] = env_value
            else:
                # Default to string
                current[final_key] = env_value
        
        return self
    
    def build(self) -> T:
        """Build the configuration object."""
        return self.config_class(**self.config_data)

class ConfigurationManager:
    """Centralized configuration manager."""
    
    def __init__(self, config_dir: Union[str, Path] = "./config"):
        """Initialize the configuration manager."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.config_dir / "templates").mkdir(exist_ok=True)
        (self.config_dir / "profiles").mkdir(exist_ok=True)
        (self.config_dir / "environments").mkdir(exist_ok=True)
        
        # Loaded configurations
        self._templates: Dict[str, ConfigTemplate] = {}
        self._profiles: Dict[str, ConfigProfile] = {}
        self._environments: Dict[str, EnvironmentConfig] = {}
        self._active_configs: Dict[str, Any] = {}
        
        # Load existing configurations
        self._load_configurations()
    
    def _load_configurations(self) -> None:
        """Load existing configurations from disk."""
        # Load templates
        template_dir = self.config_dir / "templates"
        for template_file in template_dir.glob("*.json"):
            try:
                with open(template_file, 'r') as f:
                    data = json.load(f)
                template = ConfigTemplate(**data)
                self._templates[template.name] = template
            except Exception as e:
                print(f"Failed to load template {template_file}: {e}")
        
        # Load profiles
        profile_dir = self.config_dir / "profiles"
        for profile_file in profile_dir.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    data = json.load(f)
                profile = ConfigProfile(**data)
                self._profiles[profile.name] = profile
            except Exception as e:
                print(f"Failed to load profile {profile_file}: {e}")
        
        # Load environments
        env_dir = self.config_dir / "environments"
        for env_file in env_dir.glob("*.yaml"):
            try:
                with open(env_file, 'r') as f:
                    data = yaml.safe_load(f)
                env_config = EnvironmentConfig(**data)
                self._environments[env_config.environment.value] = env_config
            except Exception as e:
                print(f"Failed to load environment {env_file}: {e}")
    
    def create_benchmark_config(self, **kwargs) -> ConfigBuilder[BenchmarkConfig]:
        """Create a benchmark configuration builder."""
        return ConfigBuilder(BenchmarkConfig).set_fields(**kwargs)
    
    def create_agent_config(self, **kwargs) -> ConfigBuilder[AgentConfig]:
        """Create an agent configuration builder."""
        return ConfigBuilder(AgentConfig).set_fields(**kwargs)
    
    def create_scenario_config(self, **kwargs) -> ConfigBuilder[ScenarioConfig]:
        """Create a scenario configuration builder."""
        return ConfigBuilder(ScenarioConfig).set_fields(**kwargs)
    
    def register_template(self, template: ConfigTemplate) -> None:
        """Register a configuration template."""
        self._templates[template.name] = template
        
        # Save to disk
        template_file = self.config_dir / "templates" / f"{template.name}.json"
        with open(template_file, 'w') as f:
            json.dump(template.model_dump(), f, indent=2)
    
    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """Get a configuration template."""
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """List available template names."""
        return list(self._templates.keys())
    
    def register_profile(self, profile: ConfigProfile) -> None:
        """Register a configuration profile."""
        self._profiles[profile.name] = profile
        
        # Save to disk
        profile_file = self.config_dir / "profiles" / f"{profile.name}.json"
        with open(profile_file, 'w') as f:
            json.dump(profile.model_dump(), f, indent=2)
    
    def get_profile(self, name: str) -> Optional[ConfigProfile]:
        """Get a configuration profile."""
        return self._profiles.get(name)
    
    def list_profiles(self) -> List[str]:
        """List available profile names."""
        return list(self._profiles.keys())
    
    def register_environment(self, env_config: EnvironmentConfig) -> None:
        """Register an environment configuration."""
        self._environments[env_config.environment.value] = env_config
        
        # Save to disk
        env_file = self.config_dir / "environments" / f"{env_config.environment.value}.yaml"
        with open(env_file, 'w') as f:
            yaml.dump(env_config.model_dump(), f, default_flow_style=False)
    
    def get_environment(self, environment: Union[str, EnvironmentType]) -> Optional[EnvironmentConfig]:
        """Get an environment configuration."""
        if isinstance(environment, EnvironmentType):
            environment = environment.value
        return self._environments.get(environment)
    
    def list_environments(self) -> List[str]:
        """List available environment names."""
        return list(self._environments.keys())
    
    def load_config_from_file(self, file_path: Union[str, Path], config_type: Type[T]) -> T:
        """Load configuration from file."""
        return ConfigBuilder(config_type).from_file(file_path).build()
    
    def save_config_to_file(self, config: BaseModel, file_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
            with open(file_path, 'w') as f:
                yaml.dump(config.model_dump(), f, default_flow_style=False, indent=2)
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(config.model_dump(), f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def activate_config(self, config_id: str, config: Any) -> None:
        """Activate a configuration."""
        self._active_configs[config_id] = config
    
    def get_active_config(self, config_id: str) -> Optional[Any]:
        """Get an active configuration."""
        return self._active_configs.get(config_id)
    
    def list_active_configs(self) -> List[str]:
        """List active configuration IDs."""
        return list(self._active_configs.keys())

# Global configuration manager instance
config_manager = ConfigurationManager()