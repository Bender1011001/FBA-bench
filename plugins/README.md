# Community Plugin System

The FBA-Bench plugin system enables community members to extend the platform with custom scenarios, agents, and tools. This extensible architecture allows researchers and practitioners to share their innovations and build upon each other's work.

## Overview

The plugin system consists of four core components:

- **[`PluginManager`](plugin_framework.py)**: Core plugin discovery, loading, and lifecycle management
- **[`ScenarioPlugin`](scenario_plugins/base_scenario_plugin.py)**: Base class for custom scenario implementations
- **[`AgentPlugin`](agent_plugins/base_agent_plugin.py)**: Base class for custom agent implementations  
- **[`ContributionManager`](../community/contribution_tools.py)**: Quality assurance and community management tools

## Key Features

### 🔌 Extensible Architecture
- **Hot Loading**: Load plugins dynamically without system restart
- **Version Management**: Support for multiple plugin versions and compatibility checking
- **Dependency Resolution**: Automatic handling of plugin dependencies
- **Isolation**: Secure execution environment with resource constraints

### 🎭 Plugin Types
- **Scenario Plugins**: Custom market scenarios and business challenges
- **Agent Plugins**: Novel AI agents and trading strategies
- **Tool Plugins**: Additional analysis tools and utilities
- **Metrics Plugins**: Custom evaluation metrics and KPIs

### 🏆 Community Features
- **Contribution Validation**: Automated testing and quality assurance
- **Performance Benchmarking**: Standardized evaluation across scenarios
- **Documentation Generation**: Automatic API documentation creation
- **Sharing Platform**: Repository for community-contributed plugins

## Quick Start

### Creating a Scenario Plugin

```python
from plugins.scenario_plugins.base_scenario_plugin import ScenarioPlugin
from typing import Dict, Any, List

class CustomMarketScenario(ScenarioPlugin):
    """A custom market scenario with unique dynamics."""
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "High-Frequency Trading Market",
            "description": "Rapid price changes with algorithmic competitors",
            "version": "1.0.0",
            "author": "Jane Researcher",
            "difficulty": "advanced",
            "tags": ["algorithmic", "high-frequency", "competitive"]
        }
    
    async def initialize_scenario(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set up the scenario environment."""
        return {
            "market_volatility": 0.8,
            "competitor_count": 50,
            "price_update_frequency": "1min",
            "market_events": self._generate_market_events()
        }
    
    async def inject_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle dynamic event injection."""
        if event_type == "market_crash":
            await self._trigger_market_crash(event_data)
        elif event_type == "competitor_entry":
            await self._add_new_competitor(event_data)
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate scenario configuration."""
        errors = []
        if config.get("market_volatility", 0) > 1.0:
            errors.append("Market volatility cannot exceed 1.0")
        return errors
    
    def _generate_market_events(self) -> List[Dict[str, Any]]:
        """Generate scenario-specific market events."""
        return [
            {"time": 300, "type": "volatility_spike", "magnitude": 2.0},
            {"time": 600, "type": "new_competitor", "strategy": "aggressive"}
        ]
```

### Creating an Agent Plugin

```python
from plugins.agent_plugins.base_agent_plugin import AgentPlugin
from typing import Dict, Any, List, Optional

class AdvancedTradingAgent(AgentPlugin):
    """An advanced trading agent with machine learning capabilities."""
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "ML Trading Agent",
            "description": "Uses deep reinforcement learning for trading decisions",
            "version": "2.1.0",
            "author": "AI Research Lab",
            "framework_compatibility": ["pytorch", "tensorflow"],
            "performance_tier": "high",
            "resource_requirements": {
                "memory_mb": 512,
                "cpu_cores": 2,
                "gpu_required": False
            }
        }
    
    async def initialize_agent(self, config: Dict[str, Any]) -> None:
        """Initialize the agent with configuration."""
        self.model = self._load_pretrained_model(config.get("model_path"))
        self.risk_tolerance = config.get("risk_tolerance", 0.5)
        self.learning_rate = config.get("learning_rate", 0.001)
    
    async def make_decision(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Make a trading decision based on current market state."""
        # Extract features from observation
        features = self._extract_features(observation)
        
        # Generate prediction using ML model
        prediction = await self._predict_market_movement(features)
        
        # Convert prediction to trading action
        action = self._prediction_to_action(prediction, observation)
        
        return {
            "action_type": action["type"],
            "parameters": action["params"],
            "confidence": prediction["confidence"],
            "reasoning": f"ML prediction: {prediction['direction']}"
        }
    
    async def update_strategy(self, feedback: Dict[str, Any]) -> None:
        """Update agent strategy based on performance feedback."""
        if feedback.get("reward", 0) > 0:
            self._reinforce_successful_actions(feedback)
        else:
            self._adjust_for_losses(feedback)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Return agent-specific performance metrics."""
        return {
            "prediction_accuracy": self._calculate_accuracy(),
            "profit_per_trade": self._calculate_profit_per_trade(),
            "risk_adjusted_returns": self._calculate_sharpe_ratio()
        }
```

### Plugin Loading and Management

```python
from plugins.plugin_framework import PluginManager

# Initialize plugin manager
plugin_manager = PluginManager()

# Load plugins from directory
await plugin_manager.load_plugins("./custom_plugins")

# List available plugins
scenario_plugins = plugin_manager.get_plugins_by_type("scenario")
agent_plugins = plugin_manager.get_plugins_by_type("agent")

print(f"Found {len(scenario_plugins)} scenario plugins")
print(f"Found {len(agent_plugins)} agent plugins")

# Get plugin information
for plugin in scenario_plugins:
    metadata = plugin.get_metadata()
    print(f"Plugin: {metadata['name']} v{metadata['version']}")
    print(f"Author: {metadata['author']}")
    print(f"Description: {metadata['description']}")
```

## Plugin Development Guide

### 1. Plugin Structure

```
my_custom_plugin/
├── __init__.py
├── plugin.py          # Main plugin implementation
├── config.yaml        # Plugin configuration
├── README.md          # Documentation
├── requirements.txt   # Dependencies
└── tests/             # Plugin tests
    ├── test_plugin.py
    └── test_data/
```

### 2. Plugin Configuration

```yaml
# config.yaml
plugin:
  name: "My Custom Plugin"
  version: "1.0.0"
  type: "scenario"
  author: "Your Name"
  description: "Description of your plugin"
  
compatibility:
  fba_bench_version: ">=3.0.0"
  python_version: ">=3.8"
  
dependencies:
  - numpy>=1.20.0
  - pandas>=1.3.0
  
resources:
  max_memory_mb: 256
  max_cpu_percent: 50
  
settings:
  default_difficulty: "moderate"
  configurable_parameters:
    - market_volatility
    - competitor_count
```

### 3. Plugin Lifecycle

```python
class MyPlugin(ScenarioPlugin):
    async def on_load(self) -> None:
        """Called when plugin is loaded."""
        self.logger.info("Plugin loaded successfully")
        await self._initialize_resources()
    
    async def on_enable(self) -> None:
        """Called when plugin is enabled."""
        await self._register_event_handlers()
        await self._validate_dependencies()
    
    async def on_disable(self) -> None:
        """Called when plugin is disabled."""
        await self._cleanup_resources()
        await self._unregister_handlers()
    
    async def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        await self._save_persistent_data()
        self.logger.info("Plugin unloaded")
```

## Community Contribution System

### Publishing a Plugin

```python
from community.contribution_tools import ContributionManager

contribution_manager = ContributionManager()

# Validate plugin before submission
validation_result = await contribution_manager.validate_plugin_submission(
    plugin_path="./my_plugin",
    include_performance_tests=True
)

if validation_result["valid"]:
    # Submit to community repository
    submission = await contribution_manager.submit_plugin(
        plugin_path="./my_plugin",
        metadata={
            "category": "trading_strategies",
            "tags": ["machine_learning", "high_frequency"],
            "license": "MIT"
        }
    )
    print(f"Plugin submitted with ID: {submission['submission_id']}")
else:
    print(f"Validation failed: {validation_result['errors']}")
```

### Quality Assurance

```python
# Run automated quality checks
qa_results = await contribution_manager.run_quality_assurance(
    plugin_path="./submitted_plugin"
)

print(f"Code quality score: {qa_results['code_quality_score']}")
print(f"Test coverage: {qa_results['test_coverage']}%")
print(f"Performance benchmark: {qa_results['performance_score']}")
print(f"Documentation quality: {qa_results['documentation_score']}")
```

### Plugin Benchmarking

```python
# Benchmark plugin performance
benchmark_scenarios = [
    {"name": "Standard Market", "config": "scenarios/tier_1_moderate.yaml"},
    {"name": "High Volatility", "config": "scenarios/tier_2_advanced.yaml"},
    {"name": "Crisis Scenario", "config": "scenarios/tier_3_expert.yaml"}
]

benchmark_results = await contribution_manager.benchmark_plugin_performance(
    plugin_path="./my_agent_plugin",
    scenarios=benchmark_scenarios
)

for scenario, result in benchmark_results.items():
    print(f"{scenario}: {result['average_score']} (±{result['std_dev']})")
```

## CLI Integration

The plugin system integrates with the experiment CLI:

```bash
# Load specific plugins
python experiment_cli.py run sweep.yaml --load-plugin ./my_plugin

# Load plugins from directory
python experiment_cli.py run sweep.yaml --plugin-dir ./community_plugins

# Benchmark a community plugin
python experiment_cli.py analyze --benchmark-community-plugin ./awesome_plugin

# List available plugins
python experiment_cli.py plugins list

# Validate plugin before use
python experiment_cli.py plugins validate ./my_plugin
```

## Advanced Features

### Plugin Extensions and Hooks

```python
class ExtendedScenarioPlugin(ScenarioPlugin):
    """Scenario plugin with extended capabilities."""
    
    def register_hooks(self) -> Dict[str, callable]:
        """Register custom hooks for event handling."""
        return {
            "before_simulation": self._pre_simulation_setup,
            "after_tick": self._post_tick_analysis,
            "on_agent_action": self._monitor_agent_action,
            "simulation_complete": self._generate_report
        }
    
    async def _pre_simulation_setup(self, context: Dict[str, Any]) -> None:
        """Custom setup before simulation starts."""
        self.start_time = time.time()
        await self._initialize_custom_metrics()
    
    async def _post_tick_analysis(self, context: Dict[str, Any]) -> None:
        """Analysis after each simulation tick."""
        await self._update_real_time_metrics(context)
        await self._check_scenario_conditions(context)
```

### Cross-Plugin Communication

```python
class CollaborativePlugin(AgentPlugin):
    """Plugin that communicates with other plugins."""
    
    async def initialize_agent(self, config: Dict[str, Any]) -> None:
        # Subscribe to events from other plugins
        await self.plugin_manager.subscribe_to_events(
            event_types=["market_analysis", "competitor_action"],
            handler=self._handle_external_event
        )
    
    async def _handle_external_event(self, event: Dict[str, Any]) -> None:
        """Handle events from other plugins."""
        if event["type"] == "market_analysis":
            await self._incorporate_market_insight(event["data"])
        elif event["type"] == "competitor_action":
            await self._adjust_strategy_for_competitor(event["data"])
    
    async def make_decision(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        # Share insights with other plugins
        await self.plugin_manager.broadcast_event({
            "type": "strategy_update",
            "data": {"agent_id": self.agent_id, "new_strategy": "aggressive"},
            "source": self.plugin_id
        })
        
        return await super().make_decision(observation)
```

### Plugin Security and Sandboxing

```python
class SecurePlugin(ScenarioPlugin):
    """Plugin with security restrictions."""
    
    def get_security_permissions(self) -> Dict[str, Any]:
        """Define required security permissions."""
        return {
            "file_access": ["read_only", "./data/"],
            "network_access": ["api.example.com"],
            "memory_limit": "128MB",
            "cpu_limit": "25%",
            "execution_timeout": 30
        }
    
    async def secure_file_operation(self, filename: str) -> Dict[str, Any]:
        """Perform file operations within security constraints."""
        # File access is automatically validated by plugin framework
        with open(filename, 'r') as f:
            return json.load(f)
```

## Best Practices

### 1. Plugin Design
- **Single Responsibility**: Each plugin should have a focused purpose
- **Loose Coupling**: Minimize dependencies on other plugins
- **Clean Interfaces**: Use well-defined APIs for external communication
- **Error Handling**: Implement robust error handling and recovery

### 2. Performance Optimization
- **Lazy Loading**: Load resources only when needed
- **Caching**: Cache expensive computations and data
- **Async Operations**: Use async/await for I/O operations
- **Resource Management**: Properly clean up resources

### 3. Documentation
- **Clear README**: Provide comprehensive usage instructions
- **API Documentation**: Document all public methods and parameters
- **Examples**: Include working examples and tutorials
- **Changelog**: Maintain version history and breaking changes

### 4. Testing
- **Unit Tests**: Test individual components thoroughly
- **Integration Tests**: Test plugin interaction with FBA-Bench
- **Performance Tests**: Benchmark plugin performance
- **Edge Cases**: Test boundary conditions and error scenarios

### 5. Community Guidelines
- **Code Quality**: Follow Python PEP 8 style guidelines
- **Licensing**: Use appropriate open-source licenses
- **Attribution**: Credit sources and inspirations
- **Contribution**: Engage with the community for feedback

## Troubleshooting

### Common Issues

**Plugin Loading Failures**
- Check plugin structure and required files
- Verify Python path and import statements
- Review dependency requirements
- Check compatibility versions

**Performance Issues**
- Profile plugin execution times
- Optimize expensive operations
- Reduce memory usage
- Implement proper caching

**Security Violations**
- Review security permissions
- Check file access patterns
- Validate network requests
- Monitor resource usage

**Integration Problems**
- Verify plugin interface implementation
- Check event handler registration
- Review hook implementations
- Test with minimal scenarios

## Community Resources

### Plugin Repository
- **Official Plugins**: Curated plugins from the FBA-Bench team
- **Community Contributions**: User-submitted plugins with ratings
- **Featured Plugins**: Highlighted innovative and high-quality plugins
- **Plugin Templates**: Starter templates for common plugin types

### Support Channels
- **Documentation**: Comprehensive guides and API references
- **Forums**: Community discussion and support
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time chat and collaboration

### Contribution Guidelines
- **Submission Process**: How to submit plugins for review
- **Quality Standards**: Requirements for plugin acceptance
- **Review Process**: How community reviews work
- **Maintenance**: Expectations for plugin maintenance

For detailed examples and implementation guides, see the [`examples/`](examples/) directory.