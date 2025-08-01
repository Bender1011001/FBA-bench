# Tutorial 4: Creating Custom Business Scenarios

This tutorial guides you through defining and implementing your own custom business scenarios in FBA-Bench, including multi-agent and dynamic elements.

## Understanding the Scenario Framework

FBA-Bench scenarios are defined using YAML files and Python scripts, controlling the environment, events, and goals of a simulation.

A basic scenario typically includes:
- `name`: Unique name for the scenario.
- `description`: A brief overview.
- `duration_days`: Length of the simulation.
- `initial_state`: Starting conditions (e.g., inventory, capital).
- `marketplace_events`: Scheduled or probabilistic events.
- `goals`: Objectives for the agent(s) to achieve.

## Defining a Simple Custom Scenario

Create a new YAML file for your scenario, e.g., [`scenarios/my_custom_scenario.yaml`](scenarios/my_custom_scenario.yaml):

```yaml
# scenarios/my_custom_scenario.yaml
name: "MyFirstCustomScenario"
description: "A simple scenario with a specific market demand change."
duration_days: 90 # 3 months

initial_state:
  starting_capital: 10000.00
  initial_inventory:
    product_A: 100
    product_B: 50
  marketplace:
    base_demand_product_A: 10
    base_demand_product_B: 5
    competition_level: "low"

marketplace_events:
  - day: 15
    type: "demand_spike"
    product_id: "product_A"
    multiplier: 2.5
    duration_days: 7
    description: "Unexpected surge in demand for Product A due to viral trend."
  - day: 45
    type: "competitor_launch"
    competitor_name: "NewKidOnTheBlock"
    product_id: "product_B"
    impact: "price_drop_10%"
    description: "New competitor enters the market, impacting Product B's pricing."

goals:
  - name: "MaximizeProfit"
    metric: "net_profit"
    target: 5000.00
    type: "minimum_threshold"
  - name: "MaintainInventory"
    metric: "average_inventory_product_A"
    target: 75
    type: "minimum_threshold"
```

## Running Your Custom Scenario

```python
# tutorial_custom_scenario.py
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.agents.advanced_agent import AdvancedAgent

# For this example, assuming 'MyCustomAgent' is a basic agent, you would replace
# this with an actual agent loaded from your system or a default FBA-Bench agent.
# Example: from fba_bench.baseline_bots.gpt_4o_mini_bot import GPT4OMiniBot
# agent = GPT4OMiniBot(name="MyAgent")

# Load your custom scenario
scenario_path = "scenarios/my_custom_scenario.yaml"
scenario_engine = ScenarioEngine.from_yaml(scenario_path)

# Initialize a basic agent (or your own custom agent)
class SimpleAgent: # Placeholder for a real agent implementation
    def __init__(self, name): self.name = name
    def act(self, current_state, marketplace_data): return [] # No actions for simplicity
    def receive_feedback(self, feedback): pass

placeholder_agent = SimpleAgent("CustomScenarioRunnerAgent")

print(f"Running custom scenario: {scenario_engine.scenario_name}")
results = scenario_engine.run_simulation(placeholder_agent) # Use your actual agent here
print("Custom scenario simulation complete! Results:", results)
```

## Multi-Agent Scenarios and Dynamic Generation

FBA-Bench supports scenarios with multiple interacting agents (cooperative or competitive) and can dynamically generate scenario parameters.
Refer to [`docs/scenarios/curriculum-design.md`](docs/scenarios/curriculum-design.md) for curriculum validation and [`docs/scenarios/dynamic-generation.md`](docs/scenarios/dynamic-generation.md) for advanced dynamic scenario capabilities.