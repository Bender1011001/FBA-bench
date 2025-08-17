# Agent Runners - Framework-Agnostic Agent Abstraction

This module provides a unified interface for different agent frameworks in **FBA-Bench**, enabling seamless swapping between DIY, CrewAI, LangChain, and future frameworks without changing simulation code.

## 🎯 Key Benefits

- **Framework Agnostic**: Switch between agent frameworks without code changes
- **Unified Interface**: All agents implement the same `AgentRunner` interface
- **Easy Integration**: Drop-in replacement for existing agent systems
- **Extensible**: Add new frameworks with minimal effort
- **Optional Dependencies**: Core simulation works without optional frameworks

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                 Simulation Orchestrator                │
└─────────────────────┬───────────────────────────────────┘
                       │
┌─────────────────────▼───────────────────────────────────┐
│                  Agent Manager                         │
│  • Lifecycle management                                 │
│  • State conversion                                     │
│  • Tool call execution                                  │
└─────────────────────┬───────────────────────────────────┘
                       │
┌─────────────────────▼───────────────────────────────────┐
│                 Runner Factory                         │
│  • Framework registration                               │
│  • Agent creation                                       │
│  • Configuration validation                             │
└─────────────────────┬───────────────────────────────────┘
                       │
┌─────────────────────▼───────────────────────────────────┐
│               AgentRunner Interface                    │
│  • async decide(state) -> [ToolCall]                   │
│  • async initialize(config)                            │
│  • async cleanup()                                      │
└─────────────────────┬───────────────────────────────────┘
                       │
     ┌─────────────────┼─────────────────┬─────────────────┐
     │                 │                 │                 │
┌───▼───┐        ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
│  DIY  │        │ CrewAI  │       │LangChain│       │ Future  │
│Runner │        │ Runner  │       │ Runner  │       │Framework│
└───────┘        └─────────┘       └─────────┘       └─────────┘
```

## 🚀 Quick Start

### Basic Usage

```python
from agent_runners import RunnerFactory, AgentManager
from event_bus import get_event_bus

# Create agent manager
event_bus = get_event_bus()
agent_manager = AgentManager(event_bus)
await agent_manager.initialize()

# Register an agent (DIY framework)
runner = await agent_manager.register_agent(
    agent_id="my_agent",
    framework="diy",
    config={
        "agent_type": "advanced",
        "target_asin": "B0EXAMPLE",
        "strategy": "profit_maximizer"
    }
)

# Agent automatically participates in simulation tick events
```

### Using Builder Pattern

```python
from agent_runners import create_agent_builder, RunnerFactory

# Create and configure agent
agent = (create_agent_builder("langchain", "my_langchain_agent")
         .with_llm("openai", "gpt-4", temperature=0.1, api_key="your_key")
         .with_memory("buffer", window_size=10)
         .with_config(max_iterations=5, verbose=True)
         .build())

# Initialize and use
runner = RunnerFactory.create_runner("langchain", "my_agent", agent.to_dict())
await runner.initialize(agent.to_dict())
```

### Using Pre-built Configurations

```python
from agent_runners import DIYConfig, CrewAIConfig, LangChainConfig

# DIY configurations
diy_config = DIYConfig.advanced_agent("agent_1", "B0PRODUCT")
baseline_config = DIYConfig.baseline_greedy("agent_2")
llm_config = DIYConfig.llm_claude("agent_3", api_key="your_key")

# CrewAI configurations
crew_config = CrewAIConfig.standard_crew("crew_agent", api_key="your_key")
pricing_crew = CrewAIConfig.focused_pricing_crew("pricing_agent", api_key="your_key")

# LangChain configurations
reasoning_config = LangChainConfig.reasoning_agent("lang_agent", api_key="your_key")
memory_config = LangChainConfig.memory_agent("memory_agent", api_key="your_key")
```

## 📦 Framework Support

### DIY Framework (Always Available)
- **Advanced Agent**: Event-driven agent with pricing strategies
- **Baseline Bots**: Simple rule-based agents (greedy, etc.)
- **LLM Bots**: Direct LLM integration (Claude, GPT)

### CrewAI Framework (Optional)
- **Multi-agent crews**: Collaborative decision making
- **Specialized roles**: Pricing, inventory, market analysis
- **Hierarchical coordination**: Manager-coordinated teams

### LangChain Framework (Optional)
- **Reasoning chains**: Step-by-step problem solving
- **Memory systems**: Context retention across decisions
- **Tool ecosystems**: Rich set of analysis tools

## 🔧 Installation

### Core System (DIY only)
```bash
pip install -r requirements.txt
```

### With All Frameworks
```bash
pip install -r requirements.txt -r requirements-frameworks.txt
```

### Individual Frameworks
```bash
# CrewAI
pip install crewai>=0.28.0 crewai-tools>=0.2.0

# LangChain  
pip install langchain>=0.1.0 langchain-community>=0.0.20 langchain-openai>=0.0.5

# LLM Providers
pip install openai>=1.12.0 anthropic>=0.8.0
```

## 📋 Configuration System

### YAML Configuration
```yaml
agent_id: "my_agent"
framework: "crewai"

llm_config:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.1
  api_key: "${OPENAI_API_KEY}"

crew_config:
  process: "sequential"
  crew_size: 4
  roles:
    - "pricing_specialist"
    - "inventory_manager" 
    - "market_analyst"
    - "strategy_coordinator"
  allow_delegation: true

verbose: false
max_iterations: 5
```

### Loading Configuration
```python
from agent_runners import load_config_from_file, validate_config

# From file
config = load_config_from_file("agent_config.yaml")

# Validation
validated_config = validate_config(config_dict)
```

## 🔌 Adding New Frameworks

### 1. Implement AgentRunner Interface
```python
from agent_runners import AgentRunner, SimulationState, ToolCall

class MyFrameworkRunner(AgentRunner):
    async def initialize(self, config: Dict[str, Any]) -> None:
        # Initialize your framework
        pass
    
    async def decide(self, state: SimulationState) -> List[ToolCall]:
        # Implement decision logic
        return [ToolCall(
            tool_name="set_price",
            parameters={"asin": "B0TEST", "price": 19.99},
            confidence=0.9,
            reasoning="My framework decision"
        )]
    
    async def cleanup(self) -> None:
        # Cleanup resources
        pass
```

### 2. Register Framework
```python
from agent_runners import RunnerFactory

RunnerFactory.register_runner(
    name="my_framework",
    runner_class=MyFrameworkRunner,
    default_config={"param1": "default_value"}
)
```

### 3. Use Your Framework
```python
runner = await agent_manager.register_agent(
    "my_agent", 
    "my_framework", 
    {"custom_param": "value"}
)
```

## 🧪 Testing and Validation

### Run Tests
```bash
python -m pytest tests/test_agent_runners.py -v
```

### Demo Framework Capabilities
```bash
python agent_runners/examples/framework_demo.py
```

### Check Framework Availability
```python
from agent_runners import get_available_frameworks, check_framework_availability

print("Available frameworks:", get_available_frameworks())
print("CrewAI available:", check_framework_availability("crewai"))
```

## 📊 Monitoring and Health Checks

### Agent Health Monitoring
```python
# Get health status of all agents
health = await agent_manager.health_check()

# Check specific agent
agent_info = await agent_manager.get_agent_info("my_agent")

# Framework usage statistics
usage = agent_manager.get_framework_usage()
print(f"Framework usage: {usage}")
```

### Framework Status
```python
from agent_runners import get_framework_status, dependency_manager

# Overall status
status = get_framework_status()

# Detailed framework information
info = dependency_manager.get_all_framework_info()
```

## 🔄 Migration Guide

### From Direct Agent Usage
**Before:**
```python
from agents.advanced_agent import AdvancedAgent, AgentConfig

agent = AdvancedAgent(AgentConfig(agent_id="test", target_asin="B0TEST"))
await agent.start()
```

**After:**
```python
from agent_runners import AgentManager, DIYConfig

config = DIYConfig.advanced_agent("test", "B0TEST")
runner = await agent_manager.register_agent("test", "diy", config