# Multi-Skill Agent Configuration

This document provides a detailed reference for configuring FBA-Bench's multi-skill agent system. This includes enabling specific skill modules, defining their parameters, and setting up their coordination strategies.

## Configuration File Location

Multi-skill agent configurations are typically loaded from YAML files. The primary configuration schema is defined by the `SkillConfig` class in [`agents/skill_config.py`](agents/skill_config.py).

## Root-Level Parameters

```yaml
multi_skill_system:
  enabled: true # Master switch to enable/disable all multi-skill features

  # Core settings
  # enabled_skills: [...]
  # coordination: {...}

  # Skill-specific configurations
  # skills: {...}
```

-   **`enabled`**: (`boolean`, default: `true`)
    -   If `false`, the agent will not utilize its multi-skill capabilities and will fall back to its primary `act` method, bypassing skill module invocation and coordination.

-   **`llm`**: (Optional, `dict`)
    -   Global LLM settings for skill modules unless overridden by specific skill configurations.
    -   `model_name`: (`str`, e.g., `"gpt-3.5-turbo"`, `"claude-3-haiku"`) The default LLM model for skills.
    -   `temperature`: (`float`, default: `0.5`) Controls the creativity/randomness of LLM responses within skills.
    -   `max_tokens`: (`integer`, default: `1024`) Maximum tokens for LLM responses by skills.
    -   `retry_attempts`: (`integer`, default: `2`) Number of times to retry failed LLM calls for skills.

## `enabled_skills`

A list of strings specifying which skill modules are active for the agent. Only skills listed here will be initialized and considered by the `MultiDomainController`.

```yaml
enabled_skills:
  - supply_manager
  - marketing_manager
  - financial_analyst
  #- customer_service # Commented out to disable this skill
```

-   **Type**: `list[str]`
-   **Description**: Each string in the list must correspond to the `skill_name` attribute of an implemented skill module (e.g., from files in [`agents/skill_modules/`](agents/skill_modules/)).

## `coordination` Parameters

Controls how proposed actions from different skill modules are reconciled and prioritized. See [`Skill Coordination`](../multi-skill-agents/skill-coordination.md) for more details.

```yaml
coordination:
  enabled: true
  conflict_resolution_strategy: "prioritize_financial_impact"
  llm_for_resolution:
    enabled: true
    model_name: "gpt-4o-mini"
    temperature: 0.3
  skill_weights:
    financial_analyst: 1.0
    marketing_manager: 0.8
```

-   **`enabled`**: (`boolean`, default: `true`) Activates/deactivates the skill coordination mechanism. If `false`, agents might execute conflicting actions if multiple skills propose them.
-   **`conflict_resolution_strategy`**: (`string`, default: `"priority_list"`) Defines how conflicts between proposed actions are resolved.
    -   Valid options: `"prioritize_financial_impact"`, `"prioritize_market_share"`, `"llm_mediated"`, `"last_one_wins"` (simple, less robust), `"first_one_wins"`.
-   **`llm_for_resolution`**: (`dict`, optional) Configuration for using an LLM to resolve complex conflicts.
    -   `enabled`: (`boolean`, default: `false`) Enable LLM for conflict resolution.
    -   `model_name`: (`str`) Specific LLM model for this task.
    -   `temperature`: (`float`) LLM temperature for resolution.
-   **`skill_weights`**: (`dict`, optional) A dictionary mapping `skill_name` to a float weight. Used in strategies like "weighted_averaging" or for prioritization Tie-breaking.

## `skills` (Skill-Specific Configurations)

Allows for defining parameters unique to each skill module or overriding global LLM settings for a specific skill.

```yaml
skills:
  supply_manager:
    llm_model: "claude-3-haiku" # Use a faster model for supply chain
    inventory_safety_margin: 0.20 # 20% buffer for inventory
    preferred_supplier_id: "SUPP001"
  marketing_manager:
    llm_model: "gpt-4o" # Use a more capable model for complex marketing decisions
    min_ad_spend_budget: 1000.00
    target_roi_percent: 1.5
  my_custom_skill: # Configuration for a custom skill
    custom_threshold: 0.75
    api_endpoint: "https://my-external-service.com/api"
```

-   **Type**: `dict`
-   **Description**: Each key in this dictionary should be the `skill_name` of an enabled skill. The value is a dictionary of parameters specific to that skill.
    -   These parameters are passed directly to the skill module's `__init__` method.
    -   Can include LLM overrides for a particular skill (e.g., `llm_model`, `temperature`).

## Example Usage

To load and use a custom skill configuration:

```python
from fba_bench.agents.skill_config import SkillConfig
from fba_bench.agents.multi_domain_controller import MultiDomainController
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.scenarios.tier_0_baseline import tier_0_scenario

# Load your custom skill configuration file
custom_config_path = "path/to/your/custom_skill_config.yaml"
skill_config = SkillConfig.from_yaml(custom_config_path)

# Initialize your multi-skill agent with the custom configuration
agent = MultiDomainController(name="MyCustomMultiSkillAgent", skill_config=skill_config)

# Then run your simulation
# scenario_engine = ScenarioEngine(tier_0_scenario)
# results = scenario_engine.run_simulation(agent)
```

For more details on developing custom skill modules, refer to [`Custom Skills`](../multi-skill-agents/custom-skills.md).