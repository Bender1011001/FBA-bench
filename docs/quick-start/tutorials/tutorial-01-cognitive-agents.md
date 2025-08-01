# Tutorial 1: Building Agents with Reflection and Planning

This tutorial guides you through creating and configuring cognitive agents in FBA-Bench, focusing on hierarchical planning and structured reflection.

## Hierarchical Strategic Planning

FBA-Bench's enhanced agents can perform hierarchical planning, breaking down long-term goals into actionable sub-tasks.

```python
# tutorial_cognitive_agent.py
from fba_bench.agents.advanced_agent import AdvancedAgent
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.scenarios.business_types.international_expansion import international_expansion_scenario
from fba_bench.agents.cognitive_config import CognitiveConfig

# Define a custom cognitive configuration for planning
planning_config = CognitiveConfig(
    hierarchical_planning={
        "enabled": True,
        "depth": 2, # Plan across 2 levels of hierarchy (e.g., Annual, Quarterly)
        "goal_horizon": "annual" # Plan for annual goals
    },
    reflection={"enabled": False} # Disable reflection for this example
)

agent = AdvancedAgent(name="StrategicPlannerAgent", config=planning_config)
scenario_engine = ScenarioEngine(international_expansion_scenario)

print("Running simulation with hierarchical planning...")
results = scenario_engine.run_simulation(agent)
print("Simulation complete! Check agent logs for planning details.")
```

## Structured Reflection Loops

Agents can reflect on their past actions and simulation outcomes to generate insights and improve future performance.

```python
# tutorial_reflective_agent.py
from fba_bench.agents.advanced_agent import AdvancedAgent
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.scenarios.tier_1_moderate import tier_1_scenario
from fba_bench.agents.cognitive_config import CognitiveConfig

# Define a cognitive configuration for reflection
reflection_config = CognitiveConfig(
    hierarchical_planning={"enabled": False}, # Disable planning for this example
    reflection={
        "enabled": True,
        "frequency": "end_of_quarter", # Reflect at the end of each quarter
        "insight_generation": True # Enable automated insight generation
    },
    memory={"validation": True} # Ensure memory consistency during reflection
)

agent = AdvancedAgent(name="ReflectiveAgent", config=reflection_config)
scenario_engine = ScenarioEngine(tier_1_scenario)

print("Running simulation with structured reflection...")
results = scenario_engine.run_simulation(agent)
print("Simulation complete! Insights generated during reflection are in the trace.")
```

For more advanced configuration options, refer to the [`docs/cognitive-architecture/configuration-guide.md`](docs/cognitive-architecture/configuration-guide.md).