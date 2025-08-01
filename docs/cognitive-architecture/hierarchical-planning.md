# Hierarchical Planning

FBA-Bench's hierarchical planning system enables agents to develop and execute multi-level strategies, moving from high-level annual goals to daily tactical decisions. This allows for more coherent and effective long-term agent behavior.

## Core Concepts

-   **Goal Horizon**: Defines the timeframe for the highest-level plan (e.g., annual, quarterly, monthly).
-   **Planning Depth**: The number of nested planning levels. A depth of 2 could mean annual goals broken into quarterly plans, which are then translated into daily actions.
-   **Strategic Goals**: Long-term, overarching objectives.
-   **Tactical Plans**: Mid-term steps to achieve strategic goals.
-   **Operational Actions**: Daily, concrete tasks derived from tactical plans.

## How it Works

The Hierarchical Planner (implemented in [`agents/hierarchical_planner.py`](agents/hierarchical_planner.py)) works by:

1.  **Defining Strategic Goals**: At the beginning of a simulation, or at predefined intervals (e.g., annually), the agent formulates its primary objectives.
2.  **Decomposition**: The planner breaks down strategic goals into more manageable tactical plans for a shorter period (e.g.,
    quarters or months).
3.  **Real-time Adaptation**: As the simulation progresses, the agent's daily actions are guided by the current tactical plan. The planner can re-evaluate and adjust plans based on new information or unexpected events through its integration with the Reflection Module.
4. **Goal Management**: The system tracks progress against both strategic and tactical goals, providing feedback to the agent.

## Configuration

Hierarchical planning is configured within the agent's cognitive settings, typically via `cognitive_config.yaml`.

```yaml
# Example cognitive_config.yaml snippet
hierarchical_planning:
  enabled: true
  depth: 3 # Number of planning levels (e.g., Annual, Quarterly, Monthly/Daily)
  goal_horizon: "annual" # Top-level planning horizon
  replan_frequency: "quarterly" # How often to re-evaluate and generate new top-level plans
  lookahead_days: 90 # How many simulation days the tactical plan looks ahead
```

## Example Usage

While planning is typically integrated directly into the [`AdvancedAgent`](agents/advanced_agent.py) or [`MultiDomainController`](agents/multi_domain_controller.py), you can manually inspect the planning process or configure it for specific experiments.

```python
from fba_bench.agents.hierarchical_planner import HierarchicalPlanner
from fba_bench.agents.cognitive_config import CognitiveConfig

# Create a sample config
planning_config = CognitiveConfig(
    hierarchical_planning={
        "enabled": True,
        "depth": 2,
        "goal_horizon": "annual",
        "replan_frequency": "quarterly"
    }
).hierarchical_planning_config # Access the specific planning config

planner = HierarchicalPlanner(agent_name="MyStrategicAgent", config=planning_config)

# Simulate initial strategic planning
initial_goals = ["Increase Market Share by 10%", "Achieve 20% Profit Margin"]
planner.set_strategic_goals(initial_goals)

print("Strategic Goals:", planner.get_strategic_goals())

# Simulate quarterly tactical planning based on current state and strategic goals
current_state = {"current_quarter": 1, "market_share": 0.05}
tactical_plan = planner.generate_tactical_plan(current_state)
print("Generated Tactical Plan for Q1:", tactical_plan)

# The agent would then execute actions based on this tactical plan.
```

For more details on how planning integrates with reflection and memory, refer to the [`Cognitive Architecture Overview`](cognitive-overview.md).