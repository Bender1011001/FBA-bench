"""
FBA-Bench Comprehensive Benchmarking Framework

This module provides a comprehensive benchmarking framework for FBA-Bench,
including core benchmarking engine, multi-dimensional metrics, extensible scenarios,
reproducibility tools, configuration management, and integration with existing systems.

The framework is designed to be:
- Extensible: Easy to add new metrics, scenarios, and validation tools
- Reproducible: Deterministic execution with comprehensive audit trails
- Scalable: Parallel execution capabilities for large-scale benchmarks
- Compatible: Seamless integration with existing FBA-Bench components
- Configurable: Flexible configuration system for different environments

Example Usage:
    ```python
    from benchmarking import BenchmarkEngine, ConfigurationManager, IntegrationManager
    
    # Initialize components
    config_manager = ConfigurationManager()
    integration_manager = IntegrationManager()
    engine = BenchmarkEngine(config_manager, integration_manager)
    
    # Load configuration
    config = config_manager.load_config("benchmark_config.yaml")
    
    # Run benchmark
    result = await engine.run_benchmark(config)
    
    # Analyze results
    print(f"Benchmark completed with score: {result.overall_score}")
    ```
"""

from .core.engine import (
    BenchmarkEngine,
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkStatus,
    BenchmarkError,
    benchmark_engine,
    get_benchmark_engine
)

from .config.manager import (
    ConfigurationProfile,
    ConfigurationManager,
    config_manager
)

from .config.schema import (
    SchemaValidationError,
    ValidationResult,
    ConfigurationSchema,
    BenchmarkConfigurationSchema,
    ScenarioConfigurationSchema,
    AgentConfigurationSchema,
    SchemaRegistry,
    schema_registry
)

from .metrics.registry import (
    metrics_registry,
    register_metric,
    get_metric
)

from .metrics.base import (
    BaseMetric,
    MetricResult,
    MetricCategory,
    CognitiveMetrics,
    BusinessMetrics,
    TechnicalMetrics,
    EthicalMetrics
)

from .metrics.statistical import (
    StatisticalValidator,
    ConfidenceInterval,
    SignificanceTest,
    OutlierDetector
)

from .scenarios.registry import (
    scenario_registry,
    register_scenario,
    get_scenario
)

from .scenarios.base import (
    Scenario,
    ScenarioResult,
    ScenarioStatus,
    ScenarioConfig
)

from .validators.registry import (
    validator_registry,
    register_validator,
    get_validator
)

from .validators.deterministic import (
    DeterministicExecutor,
    RandomSeedManager
)

from .validators.version_control import (
    VersionTracker,
    ComponentVersion
)

from .validators.statistical_validator import (
    StatisticalValidator,
    ValidationResult as StatisticalValidationResult
)

from .validators.audit_trail import (
    AuditTrail,
    AuditEvent,
    AuditEventType
)

from .validators.reproducibility_validator import (
    ReproducibilityValidator,
    ReproducibilityResult
)

from .integration.manager import (
    IntegrationStatus,
    IntegrationConfig,
    IntegrationManager,
    SimpleEventBus,
    integration_manager
)

from .integration.agent_adapter import (
    AgentAdapterConfig,
    AgentExecutionResult,
    AgentAdapter,
    AgentAdapterFactory
)

from .integration.metrics_adapter import (
    MetricsAdapterConfig,
    MetricsAdapterResult,
    MetricsAdapter,
    MetricsAdapterFactory
)

# Version information
__version__ = "1.0.0"
__author__ = "FBA-Bench Team"
__email__ = "team@fba-bench.org"

# Main classes for easy import
__all__ = [
    # Core engine
    "BenchmarkEngine",
    "BenchmarkResult",
    "BenchmarkRun",
    "BenchmarkStatus",
    "BenchmarkError",
    "benchmark_engine",
    "get_benchmark_engine",
    
    # Configuration
    "ConfigurationProfile",
    "ConfigurationManager",
    "config_manager",
    "SchemaValidationError",
    "ValidationResult",
    "ConfigurationSchema",
    "BenchmarkConfigurationSchema",
    "ScenarioConfigurationSchema",
    "AgentConfigurationSchema",
    "SchemaRegistry",
    "schema_registry",
    
    # Metrics
    "metrics_registry",
    "register_metric",
    "get_metric",
    "BaseMetric",
    "MetricResult",
    "MetricCategory",
    "CognitiveMetrics",
    "BusinessMetrics",
    "TechnicalMetrics",
    "EthicalMetrics",
    "StatisticalValidator",
    "ConfidenceInterval",
    "SignificanceTest",
    "OutlierDetector",
    
    # Scenarios
    "scenario_registry",
    "register_scenario",
    "get_scenario",
    "Scenario",
    "ScenarioResult",
    "ScenarioStatus",
    "ScenarioConfig",
    
    # Validators
    "validator_registry",
    "register_validator",
    "get_validator",
    "DeterministicExecutor",
    "RandomSeedManager",
    "VersionTracker",
    "ComponentVersion",
    "StatisticalValidator",
    "StatisticalValidationResult",
    "AuditTrail",
    "AuditEvent",
    "AuditEventType",
    "ReproducibilityValidator",
    "ReproducibilityResult",
    
    # Integration
    "IntegrationStatus",
    "IntegrationConfig",
    "IntegrationManager",
    "SimpleEventBus",
    "integration_manager",
    "AgentAdapterConfig",
    "AgentExecutionResult",
    "AgentAdapter",
    "AgentAdapterFactory",
    "MetricsAdapterConfig",
    "MetricsAdapterResult",
    "MetricsAdapter",
    "MetricsAdapterFactory",
]

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Create default instances for convenience
_default_config_manager = None
_default_integration_manager = None
_default_engine = None


def get_default_config_manager() -> ConfigurationManager:
    """Get the default configuration manager instance."""
    global _default_config_manager
    if _default_config_manager is None:
        _default_config_manager = ConfigurationManager()
    return _default_config_manager


def get_default_integration_manager() -> IntegrationManager:
    """Get the default integration manager instance."""
    global _default_integration_manager
    if _default_integration_manager is None:
        _default_integration_manager = IntegrationManager()
    return _default_integration_manager


def get_default_engine() -> BenchmarkEngine:
    """Get the default benchmark engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = BenchmarkEngine(
            get_default_config_manager(),
            get_default_integration_manager()
        )
    return _default_engine


async def run_benchmark(config_path: str, **kwargs) -> BenchmarkResult:
    """
    Convenience function to run a benchmark with default components.
    
    Args:
        config_path: Path to benchmark configuration file
        **kwargs: Additional arguments for the benchmark engine
        
    Returns:
        BenchmarkResult instance
    """
    engine = get_default_engine()
    config = get_default_config_manager().load_config(config_path)
    
    # Apply any additional configuration
    for key, value in kwargs.items():
        if '.' in key:
            # Handle nested configuration
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            config[key] = value
    
    return await engine.run_benchmark(config)


def create_benchmark_config(
    benchmark_id: str,
    scenarios: List[str],
    agents: List[str],
    **kwargs
) -> dict:
    """
    Create a basic benchmark configuration.
    
    Args:
        benchmark_id: Unique identifier for the benchmark
        scenarios: List of scenario IDs to include
        agents: List of agent IDs to benchmark
        **kwargs: Additional configuration options
        
    Returns:
        Basic benchmark configuration dictionary
    """
    config = {
        "benchmark_id": benchmark_id,
        "name": f"Benchmark {benchmark_id}",
        "description": f"Auto-generated benchmark configuration for {benchmark_id}",
        "version": "1.0.0",
        "environment": {
            "deterministic": True,
            "random_seed": 42,
            "parallel_execution": False,
            "max_workers": 1
        },
        "scenarios": [
            {
                "id": scenario_id,
                "name": f"Scenario {scenario_id}",
                "enabled": True,
                "priority": 1
            }
            for scenario_id in scenarios
        ],
        "agents": [
            {
                "id": agent_id,
                "name": f"Agent {agent_id}",
                "enabled": True
            }
            for agent_id in agents
        ],
        "metrics": {
            "categories": ["cognitive", "business", "technical"],
            "custom_metrics": []
        },
        "execution": {
            "runs_per_scenario": 3,
            "max_duration": 0,
            "timeout": 300,
            "retry_on_failure": True,
            "max_retries": 3
        },
        "output": {
            "format": "json",
            "path": "./results",
            "include_detailed_logs": False,
            "include_audit_trail": True
        },
        "validation": {
            "enabled": True,
            "statistical_significance": True,
            "confidence_level": 0.95,
            "reproducibility_check": True
        },
        "metadata": {
            "author": "FBA-Bench Framework",
            "created": "auto-generated",
            "tags": ["auto-generated", "benchmark"]
        }
    }
    
    # Apply additional configuration
    for key, value in kwargs.items():
        if '.' in key:
            # Handle nested configuration
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            config[key] = value
    
    return config


# Register built-in metrics and scenarios on import
def _register_builtin_components():
    """Register built-in metrics and scenarios."""
    try:
        # Register built-in metrics
        from .metrics.cognitive import ReasoningMetric, PlanningMetric, MemoryMetric
        from .metrics.business import ROIMetric, EfficiencyMetric, StrategicAlignmentMetric
        from .metrics.technical import PerformanceMetric, ReliabilityMetric, ResourceUsageMetric
        from .metrics.ethical import BiasDetectionMetric, SafetyMetric, TransparencyMetric
        
        metrics_registry.register("reasoning", ReasoningMetric())
        metrics_registry.register("planning", PlanningMetric())
        metrics_registry.register("memory", MemoryMetric())
        metrics_registry.register("roi", ROIMetric())
        metrics_registry.register("efficiency", EfficiencyMetric())
        metrics_registry.register("strategic_alignment", StrategicAlignmentMetric())
        metrics_registry.register("performance", PerformanceMetric())
        metrics_registry.register("reliability", ReliabilityMetric())
        metrics_registry.register("resource_usage", ResourceUsageMetric())
        metrics_registry.register("bias_detection", BiasDetectionMetric())
        metrics_registry.register("safety", SafetyMetric())
        metrics_registry.register("transparency", TransparencyMetric())
        
        # Register built-in scenarios
        from .scenarios.ecommerce import EcommerceScenario
        from .scenarios.healthcare import HealthcareScenario
        from .scenarios.financial import FinancialScenario
        from .scenarios.legal import LegalScenario
        from .scenarios.scientific import ScientificScenario
        
        scenario_registry.register("ecommerce", EcommerceScenario())
        scenario_registry.register("healthcare", HealthcareScenario())
        scenario_registry.register("financial", FinancialScenario())
        scenario_registry.register("legal", LegalScenario())
        scenario_registry.register("scientific", ScientificScenario())
        
        # Register built-in validators
        from .validators.deterministic import DeterministicValidator
        from .validators.version_control import VersionControlValidator
        from .validators.statistical_validator import StatisticalValidator as StatsValidator
        from .validators.reproducibility_validator import ReproducibilityValidator as ReproValidator
        
        validator_registry.register("deterministic", DeterministicValidator())
        validator_registry.register("version_control", VersionControlValidator())
        validator_registry.register("statistical", StatsValidator())
        validator_registry.register("reproducibility", ReproValidator())
        
    except ImportError as e:
        logging.warning(f"Failed to register some built-in components: {e}")


# Register built-in components on module import
_register_builtin_components()