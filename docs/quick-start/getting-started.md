# Getting Started with FBA-Bench Enhanced

This guide will help you quickly set up and run your first enhanced agent simulation with FBA-Bench.

## Installation and Setup for Enhanced Features

To get started, ensure you have Python 3.9+ and pip installed. We recommend using a virtual environment.

```bash
# Clone the FBA-Bench repository
git clone https://github.com/your-repo/fba-bench.git
cd fba-bench

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core and enhanced dependencies
pip install -r requirements.txt
pip install -r requirements-enhanced.txt # Assuming a new requirements file for enhanced features
```

## Your First Enhanced Agent Simulation

Let's run a simple simulation with an enhanced cognitive agent.

```python
# example_simulation.py
from fba_bench.agents.advanced_agent import AdvancedAgent
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.scenarios.tier_0_baseline import tier_0_scenario

# Initialize the advanced agent (with hierarchical planning and reflection enabled by default)
agent = AdvancedAgent(name="MyEnhancedAgent")

# Load a simple Tier 0 scenario
scenario_engine = ScenarioEngine(tier_0_scenario)

# Run the simulation
results = scenario_engine.run_simulation(agent)

print("Simulation complete! Results:", results)
```

To run this example:

```bash
python example_simulation.py
```

## Basic Configuration and Customization

FBA-Bench uses YAML files for configuration. You can customize agent behavior, scenario parameters, and infrastructure settings.

For cognitive agents, modify configuration files in [`agents/configs/`](agents/configs/). For example:

```yaml
# agents/configs/cognitive_agent_config.yaml
hierarchical_planning:
  enabled: true
  depth: 3
reflection:
  enabled: true
  frequency: "end_of_quarter"
memory:
  validation: true
  consistency_checks: true
```

Load custom configurations:

```python
from fba_bench.agents.cognitive_config import CognitiveConfig
from fba_bench.agents.advanced_agent import AdvancedAgent

custom_config = CognitiveConfig.from_yaml("agents/configs/cognitive_agent_config.yaml")
agent = AdvancedAgent(name="CustomAgent", config=custom_config)
```

## Simple Examples for Each Major Feature Area

Detailed examples and tutorials for each major feature are available in the [`docs/quick-start/tutorials/`](docs/quick-start/tutorials/) directory.

- **Cognitive Agents**: See [`docs/quick-start/tutorials/tutorial-01-cognitive-agents.md`](docs/quick-start/tutorials/tutorial-01-cognitive-agents.md)
- **Multi-Skill Agents**: See [`docs/quick-start/tutorials/tutorial-02-multi-skill-agents.md`](docs/quick-start/tutorials/tutorial-02-multi-skill-agents.md)
- **Scalable Simulations**: See [`docs/quick-start/tutorials/tutorial-03-scalable-simulations.md`](docs/quick-start/tutorials/tutorial-03-scalable-simulations.md)
- **Scenario Creation**: See [`docs/quick-start/tutorials/tutorial-04-scenario-creation.md`](docs/quick-start/tutorials/tutorial-04-scenario-creation.md)
- **Observability Setup**: See [`docs/quick-start/tutorials/tutorial-05-observability-setup.md`](docs/quick-start/tutorials/tutorial-05-observability-setup.md)
- **Agent Learning**: See [`docs/quick-start/tutorials/tutorial-06-agent-learning.md`](docs/quick-start/tutorials/tutorial-06-agent-learning.md)
- **Real-World Integration**: See [`docs/quick-start/tutorials/tutorial-07-real-world-integration.md`](docs/quick-start/tutorials/tutorial-07-real-world-integration.md)