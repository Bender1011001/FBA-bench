# Cognitive System Configuration Guide

FBA-Bench's cognitive architecture is highly configurable, allowing users and researchers to fine-tune agent intelligence, planning, reflection, and memory parameters. This guide explains the structure and key options for configuring the cognitive systems.

## Configuration File Structure

Cognitive system configurations are primarily managed through YAML files, typically located in the `agents/configs/` directory (e.g., [`agents/cognitive_config.py`](agents/cognitive_config.py) defines the schema and default values).

A typical cognitive configuration YAML might look like this:

```yaml
# agents/configs/default_cognitive_config.yaml
cognitive_system:
  enabled: true # Master switch to enable/disable cognitive features

  hierarchical_planning:
    enabled: true
    depth: 2 # Number of planning levels (e.g., Annual, Quarterly, Monthly/Daily)
    goal_horizon: "annual" # Top-level planning horizon: "annual", "quarterly", "monthly"
    replan_frequency: "quarterly" # How often to re-evaluate and generate new top-level plans
    lookahead_days: 90 # How many simulation days the tactical plan looks ahead

  reflection:
    enabled: true
    frequency: "end_of_quarter" # "end_of_day", "end_of_quarter", "after_critical_event"
    insight_generation: true # Enable LLM to generate insights from reflection
    feedback_loop_enabled: true # Whether insights feed back into planning/decision-making
    max_reflection_tokens: 2500 # Limit for LLM reflection prompt/response token usage
    min_progress_for_reflection: 0.05 # Minimum change in KPIs to trigger reflection (e.g., 5% change)

  memory:
    enabled: true
    validation: true # Enable active memory validation
    consistency_checks: true # Enable consistency checks on memory updates
    forgetting_strategy: "least_recent_access" # "least_recent_access", "fixed_size_window", "none"
    max_memory_size: 1000 # Max entries in short-term memory (0 for unlimited if storage permits)
    long_term_storage_backend: "json_file" # "json_file", "database", "vector_db" (requires specific setup)
    memory_path: "./memory_data/" # Directory for memory persistence
```

## Key Configuration Options

### `cognitive_system.enabled`
-   **Type**: `boolean`
-   **Description**: A global toggle for all advanced cognitive features. If `false`, the agent will revert to basic decision-making without hierarchical planning, reflection, or advanced memory.
-   **Default**: `true`

### `hierarchical_planning`
-   **`enabled`**: `boolean` (Default: `true`) - Activates/deactivates hierarchical planning.
-   **`depth`**: `integer` (Default: `2`) - Defines how many levels deep the planning hierarchy goes. Higher values allow for more granular planning.
-   **`goal_horizon`**: `string` (Default: `"annual"`) - The primary time horizon for long-term goals. Affects how strategic plans are formulated.
-   **`replan_frequency`**: `string` (Default: `"quarterly"`) - How often the strategic goals and high-level plans are re-evaluated.
-   **`lookahead_days`**: `integer` (Default: `90`) - The number of simulation days for which tactical plans are generated.

### `reflection`
-   **`enabled`**: `boolean` (Default: `true`) - Activates/deactivates the reflection mechanism.
-   **`frequency`**: `string` (Default: `"end_of_quarter"`) - When reflection occurs. Options include "end_of_day", "end_of_quarter", "after_critical_event".
-   **`insight_generation`**: `boolean` (Default: `true`) - If `true`, the LLM attempts to extract actionable insights from reflection.
-   **`feedback_loop_enabled`**: `boolean` (Default: `true`) - If `true`, generated insights are used to influence future planning and decision-making.
-   **`max_reflection_tokens`**: `integer` (Default: `2500`) - Limits the token usage for LLM calls during reflection to control costs and latency.
-   **`min_progress_for_reflection`**: `float` (Default: `0.05`) - Defines a threshold for significant change in key performance indicators (KPIs) to trigger an "after_critical_event" reflection.

### `memory`
-   **`enabled`**: `boolean` (Default: `true`) - Activates/deactivates advanced memory features.
-   **`validation`**: `boolean` (Default: `true`) - Enables active validation of memory entries for consistency and correctness.
-   **`consistency_checks`**: `boolean` (Default: `true`) - Enables checks to prevent contradictory information from being stored.
-   **`forgetting_strategy`**: `string` (Default: `"least_recent_access"`) - Policy for pruning old or less relevant memories. "none" disables forgetting.
-   **`max_memory_size`**: `integer` (Default: `1000`) - Maximum number of entries in short-term memory before pruning occurs. Set to 0 for no limit.
-   **`long_term_storage_backend`**: `string` (Default: `"json_file"`) - Specifies how long-term memories are persisted. "database" or "vector_db" options exist but require additional setup (see [`docs/infrastructure/deployment.md`](docs/infrastructure/deployment.md)).
-   **`memory_path`**: `string` (Default: `"./memory_data/"`) - File path for storing persistent memory data.

## Customizing Configuration

You can create multiple YAML configuration files to define different cognitive agent types or experiment with various settings. To use a custom configuration:

```python
from fba_bench.agents.cognitive_config import CognitiveConfig
from fba_bench.agents.advanced_agent import AdvancedAgent

# Load your custom configuration file
custom_config_path = "path/to/your/custom_cognitive_config.yaml"
custom_config = CognitiveConfig.from_yaml(custom_config_path)

# Initialize your agent with the custom configuration
agent = AdvancedAgent(name="MyCustomAgent", config=custom_config)

# Then run your simulation with this agent
# scenario_engine.run_simulation(agent)
```

Refer to the [`API Reference Documentation`](../api-reference/cognitive-api.md) for programmatic configuration options, and the [`Best Practices`](../best-practices/agent-design.md) for recommendations on effective cognitive agent design.