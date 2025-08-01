# Scenario and Curriculum Configuration

This document provides a detailed reference for configuring FBA-Bench's simulation scenarios and curricula. Proper scenario configuration is crucial for designing meaningful experiments and systematically evaluating agent performance across different levels of complexity.

## Configuration File Locations

Scenario configurations are typically defined in YAML files located in the `scenarios/` directory. The primary configuration schema is represented by the `ScenarioConfig` class in [`scenarios/scenario_config.py`](scenarios/scenario_config.py). Tier-specific configurations are found in [`constraints/tier_configs/`](constraints/tier_configs/).

## Root-Level Parameters

```yaml
# Example scenario.yaml
name: "MyCustomBusinessScenario"
description: "A scenario demonstrating market recovery after a minor shock."
duration_days: 120 # Total length of the simulation in days

initial_state: # Initial conditions for the marketplace and agent
  starting_capital: 75000.00
  initial_inventory:
    product_A: 300
    product_B: 150
  marketplace:
    base_demand_product_A: 10
    base_demand_product_B: 5
    competition_level: "medium"
    economic_conditions: "stable"

marketplace_events: # Defines scheduled or probabilistic events
  # ... (list of events) ...

goals: # Defines objectives for agent evaluation
  # ... (list of goals) ...

agent_profiles: # Optional: Define profiles for agents participating in multi-agent scenarios
  # ... (list of agent profiles) ...

metrics_to_track: # Optional: Custom metrics to record for this scenario
  - "net_profit"
  - "customer_satisfaction_score"
```

-   **`name`**: (`str`, required) A unique identifier for the scenario.
-   **`description`**: (`str`, required) A brief, human-readable summary of the scenario.
-   **`duration_days`**: (`integer`, required) The total number of simulation days the scenario will run.
-   **`initial_state`**: (`dict`, required) Configures the starting conditions of the simulation at Day 0.
    -   `starting_capital`: (`float`) Initial financial capital for the agent.
    -   `initial_inventory`: (`dict`) Initial stock levels for each product (e.g., `product_id: quantity`).
    -   `marketplace`: (`dict`) Initial market conditions, including `base_demand` for products, `competition_level` (e.g., "low", "medium", "high"), and `economic_conditions` (e.g., "stable", "recession", "boom").
    -   Other initial states can be added as custom fields.

-   **`marketplace_events`**: (`list[dict]`, optional) A list of events that will occur during the simulation. Each event dictionary can include:
    -   `day`: (`integer`, required for scheduled events) The simulation day the event occurs.
    -   `type`: (`str`, required) The type of event (e.g., "demand_spike", "supply_chain_disruption", "competitor_price_cut").
    -   `product_id`: (`str`, optional) The product affected by the event.
    -   `impact`: (`str`, optional) A description of the event's effect (e.g., "price_increase_20%", "demand_multiplier_1.5").
    -   `duration_days`: (`integer`, optional) How long the event's impact lasts.
    -   `probability`: (`float`, optional) For probabilistic events (requires `DynamicGenerator` or custom event injection logic).
    -   `trigger_frequency`: (`str`, optional) For recurring probabilistic events (e.g., "quarterly").

-   **`goals`**: (`list[dict]`, required) A list of objectives for the agent(s) to achieve within the scenario. Each goal dictionary defines:
    -   `name`: (`str`, required) Unique name for the goal.
    -   `metric`: (`str`, required) The KPI to track (e.g., "net_profit", "customer_satisfaction_score", "sales_product_A").
    -   `target`: (`float`/`int`, required) The target value for the metric.
    -   `type`: (`str`, required) How the target is evaluated. Valid options: `"minimum_threshold"`, `"maximum_threshold"`, `"target_value"`, `"relative_improvement"`.
    -   `product_id`: (`str`, optional) If the metric is product-specific.

-   **`agent_profiles`**: (`list[dict]`, optional) Used in multi-agent scenarios to define the roles and initial configurations of multiple agents that will participate. Each profile can include:
    -   `name`: (`str`, unique identifier for this agent instance).
    -   `type`: (`str`, e.g., "seller", "competitor", "supplier").
    -   `config_profiles`: (`dict`, e.g., `{"cognitive": "planning_heavy", "skill": "finance_focused"}`) references to specific agent configuration profiles to load.

-   **`metrics_to_track`**: (`list[str]`, optional) A list of specific metrics to track and aggregate for this scenario, overriding global defaults.

## Tier-Specific Constraints (`constraints/tier_configs/`)

Scenario difficulty tiers (T0-T3) are largely defined by constraint YAMLs, which the `CurriculumValidator` uses. These files (e.g., [`constraints/tier_configs/t1_config.yaml`](constraints/tier_configs/t1_config.yaml)) don't define scenarios directly, but rather the *rules* a scenario must adhere to for that tier.

Example: `t1_config.yaml` might specify:
```yaml
tier_requirements:
  max_num_events: 5
  max_market_volatility: 0.10 # 10% daily price swing
  allow_multi_agent: false
  llm_token_limits:
    max_agent_prompt_tokens: 2048
```

## Dynamic Scenario Generation Configuration

While scenarios can be generated dynamically using the `DynamicGenerator`, the rules and parameters for such generation are often configured in separate Python scripts or implied by the `params` passed to the generator. Example settings might include:

```python
# In a Python script using DynamicGenerator
generation_params = {
    "num_products": 3,
    "event_frequency": "high",
    "economic_trend": "recession_start",
    "force_competitive_agents": True
}
# generator.generate_scenario(generation_params)
```

## Example Usage

To load and use a specific scenario:

```python
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.scenarios.scenario_config import ScenarioConfig # For programmatic creation

# Load from YAML
scenario_engine = ScenarioEngine.from_yaml("scenarios/your_custom_scenario.yaml")

# Or programmatically create and then use ScenarioEngine
# my_scenario_dict = {
#     "name": "QuickTest",
#     "duration_days": 1,
#     "initial_state": {"starting_capital": 1000},
#     "goals": [{"name": "Profit", "metric": "net_profit", "target": 10, "type": "minimum_threshold"}]
# }
# scenario_config_obj = ScenarioConfig(**my_scenario_dict) # Validate against schema
# scenario_engine = ScenarioEngine(scenario_config_obj.dict()) # Convert back to dict for engine

# Then run simulation with your agent
# results = scenario_engine.run_simulation(my_agent)
```

For more on creating scenarios, multi-agent setups, and curriculum design, see:
- [`Scenario System Overview`](../scenarios/scenario-system.md)
- [`Curriculum Design`](../scenarios/curriculum-design.md)
- [`Multi-Agent Scenarios`](../scenarios/multi-agent-scenarios.md)
- [`Dynamic Generation`](../scenarios/dynamic-generation.md)