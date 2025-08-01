# Procedural Scenario Generation

FBA-Bench's dynamic generation capabilities allow for programmatic creation of diverse and novel simulation scenarios, moving beyond static YAML definitions. This is crucial for comprehensive agent training, robust evaluation, and exploring a vast space of possible market conditions.

## Why Dynamic Generation?

-   **Increased Diversity**: Generate an infinite variety of scenarios that might be difficult or time-consuming to define manually.
-   **Automated Curriculum Building**: Programmatically create sequences of increasingly complex scenarios for systematic agent training.
-   **Robustness Testing**: Introduce unexpected combinations of events and conditions to test agent resilience.
-   **Parameter Sweeps**: Easily run simulations across a range of initial states or event parameters.

## Core Component: Dynamic Generator

The `Dynamic Generator` (implemented in [`scenarios/dynamic_generator.py`](scenarios/dynamic_generator.py)) is the primary module for procedural scenario creation. It allows you to define templates or rules that guide the generation process, rather than hard-coding every detail.

### Key features of the `DynamicGenerator`:

-   **Template-Based Generation**: Use base scenario YAMLs as templates and programmatically override or inject specific elements.
-   **Randomization**: Introduce randomness for market fluctuations, event timing, or initial conditions within defined bounds.
-   **Rule-Based Events**: Define rules (e.g., "if market demand drops by X%, then trigger a competitor price-cut event with Y% probability").
-   **Dependency Injection**: Dynamically insert agents, products, or marketplace models based on generation parameters.

## Example: Generating a Simple Dynamic Scenario

```python
# dynamic_scenario_generator_tutorial.py
from fba_bench.scenarios.dynamic_generator import DynamicGenerator
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.agents.advanced_agent import AdvancedAgent

generator = DynamicGenerator()

# Define parameters for a dynamically generated scenario
scenario_params = {
    "name": "DynamicDemandFluctuation",
    "duration_days": 120,
    "initial_capital": 75000,
    "products_to_simulate": ["product_C", "product_D"],
    "event_density": "high", # Custom parameter for event generation logic
    "max_price_volatility_percent": 0.15 # Max 15% daily price change allowed
}

# Generate the scenario configuration
# The generate_scenario method uses internal rules to create a full scenario dict
dynamic_scenario_config_dict = generator.generate_scenario(scenario_params)

# You can then save this generated config to a YAML file for reproducibility
# import yaml
# with open("generated_scenario.yaml", "w") as f:
#     yaml.dump(dynamic_scenario_config_dict, f, sort_keys=False)

# Or directly load it into a ScenarioEngine for immediate use
scenario_engine = ScenarioEngine(dynamic_scenario_config_dict)

agent = AdvancedAgent(name="DynamicScenarioAgent")

print(f"Starting dynamically generated scenario: {scenario_engine.scenario_name}")
results = scenario_engine.run_simulation(agent)
print("Dynamically generated simulation complete. Results:", results)
```

## Advanced Dynamic Generation

### Integrating with Curriculum Validation
Dynamic generation can be combined with [`Curriculum Design`](curriculum-design.md) and [`Scenario Validation`](scenario-validation.md) to build and test progressive learning curricula automatically. The `CurriculumValidator` can verify if a dynamically generated scenario matches the criteria for a specified tier.

### Reinforcement Learning (RL) Environments
Dynamic generation is particularly useful for creating varied episodes for Reinforcement Learning agents. Each episode can start with slightly different conditions, forcing the RL agent to learn more generalizeable policies. Refer to [`RL Environment Integration`](../learning-system/reinforcement-learning.md).

### Adversarial Scenario Generation (Red Teaming)
The `DynamicGenerator` can be extended to create challenging or adversarial scenarios. This can involve generating events that exploit agent weaknesses, introduce false information, or simulate sophisticated competitor tactics. This is often used in conjunction with FBA-Bench's Red Teaming capabilities (see [`red_team_framework.md`](../docs/red_team_framework.md)).

## Extending the Dynamic Generator

You can extend `DynamicGenerator` to implement custom generation logic by:
-   Adding new methods for specific types of event generation (e.g., `generate_economic_crisis_event`).
-   Integrating external data sources or probabilistic models.
-   Defining more complex interdependencies between scenario elements.

This enables a highly flexible approach to scenario exploration and agent evaluation.