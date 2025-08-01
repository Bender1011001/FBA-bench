# Scenario Framework and Architecture

FBA-Bench's scenario system is a powerful framework for defining, customizing, and executing diverse business simulation environments. This document outlines the architecture of the scenario system and how scenarios are designed to challenge and evaluate agents.

## Core Concepts

-   **Scenario Definition**: A YAML configuration file that specifies the initial state of the marketplace, scheduled events, market dynamics, and simulation goals.
-   **Scenario Engine (`scenarios/scenario_engine.py`)**: The runtime component that interprets a scenario definition, initializes the simulation environment, injects events, and tracks progress.
-   **Marketplace Models**: Underlying models that simulate market behavior (demand, competition, supply dynamics) based on scenario parameters.
-   **Events**: Discrete occurrences that impact the simulation, ranging from simple price changes to complex supply chain disruptions or competitor actions. Events can be scheduled or dynamically triggered.
-   **Goals/Metrics**: Defined objectives against which agent performance is evaluated (e.g., maximize profit, maintain inventory levels, achieve market share).

## Architecture

The scenario system is designed for flexibility and reusability:

-   **Declarative Scenarios**: Scenarios are primarily defined using declarative YAML files (e.g., [`scenarios/tier_0_baseline.yaml`](scenarios/tier_0_baseline.yaml), [`scenarios/business_types/international_expansion.yaml`](scenarios/business_types/international_expansion.yaml)). This makes them easy to create, read, and share.
-   **Python Extensibility**: While YAML defines the core, complex event logic or dynamic scenario generation can be implemented in Python modules (e.g., [`scenarios/dynamic_generator.py`](scenarios/dynamic_generator.py)) that interact with the scenario engine.
-   **Modular Event Handling**: The `ScenarioEngine` processes events by dispatching them through a central `EventBus` (`event_bus.py`), allowing various components (agents, marketplace models, monitoring) to react appropriately.
-   **Curriculum Integration**: Scenarios can be organized into curricula of increasing difficulty, enabling progressive agent training and evaluation.

## Scenario Definition Structure

A typical scenario YAML file includes:

```yaml
name: "UniqueScenarioName"
description: "A brief description of what this scenario simulates."
duration_days: 365 # Total duration of the simulation in days

initial_state:
  starting_capital: 100000.00
  initial_inventory:
    product_A: 500
  marketplace:
    base_demand_product_A: 100
    competition_level: "moderate"
    economic_conditions: "stable"

marketplace_events:
  # Scheduled events
  - day: 30
    type: "demand_surge"
    product_id: "product_A"
    multiplier: 1.5
    duration_days: 14
    description: "Unexpected increase in demand for Product A."
  - day: 90
    type: "supply_chain_disruption"
    product_id: "raw_material_X"
    impact: "cost_increase_20%"
    duration_days: 30
    description: "Supplier issue causing raw material cost hike."

  # Probabilistic events (example structure, requires engine support)
  # - type: "competitor_price_war_start"
  #   probability: 0.1 # 10% chance per quarter
  #   trigger_frequency: "quarterly"

goals:
  - name: "MaximizeNetProfit"
    metric: "net_profit"
    target: 20000.00
    type: "minimum_threshold" # "minimum_threshold", "target_value", "relative_improvement"

  - name: "MaintainGoodwill"
    metric: "customer_satisfaction_score"
    target: 4.0
    type: "minimum_threshold"
```

## Running a Scenario

```python
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.agents.advanced_agent import AdvancedAgent # Or your chosen agent

# Load a scenario from a YAML file
scenario_config_path = "scenarios/tier_1_moderate.yaml" # Or your custom scenario
scenario_engine = ScenarioEngine.from_yaml(scenario_config_path)

# Initialize your agent
agent = AdvancedAgent(name="ScenarioAgent")

print(f"Starting simulation for scenario: {scenario_engine.scenario_name}")
results = scenario_engine.run_simulation(agent)

print("Simulation complete. Final Metrics:")
for metric, value in results.get("metrics", {}).items():
    print(f"- {metric}: {value}")

print("Goals Achieved:")
for goal, status in results.get("goal_status", {}).items():
    print(f"- {goal}: {'Achieved' if status else 'Not Achieved'}")
```

For information on designing progressive curricula, multi-agent interactions, and dynamic scenario generation, refer to:
- [`Curriculum Design`](curriculum-design.md)
- [`Multi-Agent Scenarios`](multi-agent-scenarios.md)
- [`Dynamic Generation`](dynamic-generation.md)
- [`Scenario Validation`](scenario-validation.md)
- [`Scenario Configuration Guide`](../configuration/scenario-config.md)