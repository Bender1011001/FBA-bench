# Designing Progressive Difficulty Curricula

FBA-Bench supports the design and validation of agent curricula, allowing you to gradually increase the complexity and difficulty of scenarios. This approach is invaluable for training robust agents and performing systematic benchmarking of their capabilities.

## Curriculum Levels (Tiers)

FBA-Bench defines a tiered system to categorize scenario difficulty. These tiers correspond to increasing levels of environmental complexity, agent challenges, and required capabilities:

-   **Tier 0 (T0 - Basic)**: Simple, predictable environments with clear goals and minimal disruptions. Ideal for initial agent validation and basic skill development. Examples: stable market, single product, no unexpected events.
-   **Tier 1 (T1 - Moderate)**: Introduces some variability, minor market fluctuations, or predictable external events. Requires agents to adapt to simple changes. Examples: seasonal demand, predictable competitor actions.
-   **Tier 2 (T2 - Advanced)**: Features significant market volatility, multiple interacting challenges, and requires more sophisticated planning and adaptability. Examples: supply chain crises, rapid demand shifts, complex competitor strategies.
-   **Tier 3 (T3 - Expert/Adversarial)**: Highly complex, unpredictable, and potentially adversarial environments. May involve "black swan" events, deceptive market signals, or multi-agent competition. Designed to push agents to their limits. Examples: targeted market manipulation, highly volatile economic cycles, multiple simultaneous disruptions.

Each tier typically corresponds to a set of pre-defined configuration files (e.g., [`constraints/tier_configs/t0_config.yaml`](constraints/tier_configs/t0_config.yaml) up to `t3_config.yaml`) that control various parameters, such as the allowed token limits, observation granularity, or action space complexity.

## Principles of Curriculum Design

When designing a curriculum, consider the following principles:

1.  **Gradual Increase in Complexity**:
    -   Start with basic scenarios (T0) to confirm fundamental agent function.
    -   Introduce one new challenge or variable at a time as you move up tiers.
    -   Combine challenges only in higher tiers.
2.  **Diverse Challenges**: Ensure the curriculum covers a wide range of business functions (marketing, finance, operations, customer service) and market conditions.
3.  **Measurable Objectives**: Each tier should have clear, measurable goals and success criteria for agents.
4.  **Reproducibility**: Use FBA-Bench's reproducibility features (LLM caching, seed management) to ensure that tier benchmarks are consistent.
5.  **Benchmarking**: Leverage the curriculum to systematically benchmark different agent architectures, LLM models, or cognitive configurations.

## Implementing a Curriculum

A curriculum is typically an ordered sequence of scenarios. You can define this sequence programmatically or implicitly through a folder structure.

```python
# tutorial_curriculum.py
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.scenarios.curriculum_validator import CurriculumValidator
from fba_bench.agents.advanced_agent import AdvancedAgent

# Define a list of scenarios in increasing order of difficulty
scenario_paths = [
    "scenarios/tier_0_baseline.yaml",
    "scenarios/tier_1_moderate.yaml",
    "scenarios/business_types/seasonal_demand.yaml", # Example custom T1 scenario
    "scenarios/tier_2_advanced.yaml",
    "scenarios/business_types/supply_chain_crisis.yaml", # Example custom T2 scenario
]

agent_to_evaluate = AdvancedAgent(name="CurriculumAgent")
curriculum_results = {}

validator = CurriculumValidator()

for i, path in enumerate(scenario_paths):
    print(f"\n--- Running Scenario Tier {i} ({path}) ---")
    scenario_engine = ScenarioEngine.from_yaml(path)

    # Optional: Validate scenario against predefined tier constraints
    is_valid_for_tier = validator.validate_scenario(scenario_engine.get_scenario_config(), f"T{i}")
    if not is_valid_for_tier:
        print(f"WARNING: Scenario {path} does not fully conform to Tier {i} guidelines. Proceeding anyway.")

    results = scenario_engine.run_simulation(agent_to_evaluate)
    curriculum_results[path] = results

    # Logic to adapt agent or provide feedback based on results for the next tier
    # E.g., agent_to_evaluate.learn_from_experience(results)

print("\nCurriculum Evaluation Complete!")
# Summarize overall curriculum performance
```

## Curriculum Validation

The `CurriculumValidator` (implemented in [`scenarios/curriculum_validator.py`](scenarios/curriculum_validator.py)) helps ensure that scenarios adhere to the characteristics of their intended difficulty tier. It can check:

-   Number of events
-   Volatility of market parameters
-   Presence of competitive agents
-   Complexity of goals
-   Enforcement of resource constraints (e.g., token limits)

This helps maintain consistency and comparability across benchmarks.
For creating specific scenario types, see [`Multi-Agent Scenarios`](multi-agent-scenarios.md) and [`Dynamic Generation`](dynamic-generation.md).