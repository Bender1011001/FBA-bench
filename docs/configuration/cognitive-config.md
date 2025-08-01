# Cognitive System Configuration

This document provides a detailed reference for configuring the cognitive architecture of FBA-Bench agents, including settings for hierarchical planning, structured reflection, and enhanced memory integration. These configurations allow fine-grained control over an agent's intelligence and adaptive capabilities.

## Configuration File Location

Cognitive configurations are typically loaded from YAML files found in the [`agents/configs/`](agents/configs/) directory. The primary configuration schema is defined by the `CognitiveConfig` class in [`agents/cognitive_config.py`](agents/cognitive_config.py).

## Root-Level Parameters

```yaml
cognitive_system:
  enabled: true # Master switch to enable/disable all advanced cognitive features

  # Nested configurations for sub-systems
  # hierarchical_planning: {...}
  # reflection: {...}
  # memory: {...}
  # llm: {...} # General LLM settings for cognitive activities (can be overridden per sub-system)
```

-   **`enabled`**: (`boolean`, default: `true`)
    -   If `false`, the agent will bypass all advanced cognitive features and rely on its basic decision loop, significantly reducing LLM calls and computational overhead. Useful for baseline comparisons.

-   **`llm`**: (Optional, `dict`)
    -   Global LLM settings for cognitive components unless overridden by specific sub-sections.
    -   `model_name`: (`str`, e.g., `"gpt-4o"`, `"claude-3-haiku"`) The default LLM model to use.
    -   `temperature`: (`float`, default: `0.7`) Controls the creativity/randomness of LLM responses.
    -   `max_tokens`: (`integer`, default: `4096`) Maximum tokens for LLM responses.
    -   `retry_attempts`: (`integer`, default: `3`) Number of times to retry failed LLM calls.

## `hierarchical_planning` Parameters

Controls the agent's ability to create and execute multi-level plans.

```yaml
hierarchical_planning:
  enabled: true
  depth: 2
  goal_horizon: "annual"
  replan_frequency: "quarterly"
  lookahead_days: 90
  llm_model: "gpt-4o" # Optional override for planning-specific LLM model
  prompt_template: "planning_prompt_v1" # Reference to a template in llm_interface/prompt_templates.py
```

-   **`enabled`**: (`boolean`, default: `true`) Activates/deactivates hierarchical planning.
-   **`depth`**: (`integer`, default: `2`) Number of nested planning levels (e.g., `depth: 2` means Annual -> Quarterly; `depth: 3` means Annual -> Quarterly -> Monthly).
-   **`goal_horizon`**: (`string`, default: `"annual"`) The top-level time horizon for strategic goals. Valid options: `"annual"`, `"quarterly"`, `"monthly"`.
-   **`replan_frequency`**: (`string`, default: `"quarterly"`) How often the agent re-evaluates and generates new strategic/tactical plans. Valid options: `"daily"`, `"weekly"`, `"monthly"`, `"quarterly"`, `"annual"`, `"on_event"`.
-   **`lookahead_days`**: (`integer`, default: `90`) The number of simulation days the tactical plan typically covers.
-   **`llm_model`**: (`str`, optional) Overrides the global `llm.model_name` for planning-specific LLM calls.
-   **`prompt_template`**: (`str`, optional) Specifies which prompt template (from [`llm_interface/prompt_templates.py`](llm_interface/prompt_templates.py)) to use for planning requests.

## `reflection` Parameters

Manages the agent's structured reflection cycles for learning and insight generation.

```yaml
reflection:
  enabled: true
  frequency: "end_of_quarter"
  insight_generation: true
  feedback_loop_enabled: true
  max_reflection_tokens: 2500
  min_progress_for_reflection: 0.05
  llm_model: "claude-3-opus" # Optional override for reflection-specific LLM model
  prompt_template: "reflection_prompt_v1"
```

-   **`enabled`**: (`boolean`, default: `true`) Activates/deactivates the reflection mechanism.
-   **`frequency`**: (`string`, default: `"end_of_quarter"`) When reflection occurs. Valid options: `"end_of_day"`, `"end_of_week"`, `"end_of_quarter"`, `"on_event"`. "on_event" requires specific event triggers.
-   **`insight_generation`**: (`boolean`, default: `true`) If `true`, the LLM attempts to extract actionable insights from the reflection process.
-   **`feedback_loop_enabled`**: (`boolean`, default: `true`) If `true`, generated insights are integrated into the agent's memory and can influence future planning or decision parameters.
-   **`max_reflection_tokens`**: (`integer`, default: `2500`) Limits the input/output token usage for LLM calls during reflection to manage costs and response times.
-   **`min_progress_for_reflection`**: (`float`, default: `0.05`) For `frequency: "on_event"`, this threshold (e.g., 0.05 for 5% change) defines a significant change in key performance indicators (KPIs) to trigger an ad-hoc reflection.
-   **`llm_model`**: (`str`, optional) Overrides the global `llm.model_name` for reflection-specific LLM calls.
-   **`prompt_template`**: (`str`, optional) Specifies which prompt template to use for reflection requests.

## `memory` Parameters

Configures the agent's knowledge storage, validation, and retention mechanisms.

```yaml
memory:
  enabled: true
  validation: true
  consistency_checks: true
  forgetting_strategy: "least_recent_access"
  max_memory_size: 1000
  long_term_storage_backend: "json_file"
  memory_path: "./memory_data/"
  llm_model: "gpt-3.5-turbo" # Optional override for memory-specific LLM model (e.g., for summarization)
```

-   **`enabled`**: (`boolean`, default: `true`) Activates/deactivates advanced memory features.
-   **`validation`**: (`boolean`, default: `true`) Enables active validation of memory entries for consistency and correctness. Performed by `MemoryValidator`.
-   **`consistency_checks`**: (`boolean`, default: `true`) Enables checks to prevent contradictory or redundant information from being stored.
-   **`forgetting_strategy`**: (`string`, default: `"least_recent_access"`) Policy for pruning old or less relevant short-term memories. Valid options: `"least_recent_access"`, `"fixed_size_window"`, `"none"` (disables forgetting).
-   **`max_memory_size`**: (`integer`, default: `1000`) Maximum number of entries in short-term memory before the `forgetting_strategy` is applied. Set to `0` for no limit.
-   **`long_term_storage_backend`**: (`string`, default: `"json_file"`) Specifies how long-term memories are persisted. Valid options: `"json_file"`, `"database"` (requires a configured database connection), `"vector_db"` (requires a configured vector database).
-   **`memory_path`**: (`str`, default: `"./memory_data/"`) File system path for storing persistent memory data for file-based backends.
-   **`llm_model`**: (`str`, optional) Overrides the global `llm.model_name` for memory-specific LLM calls (e.g., for memory summarization or re-encoding).

For a high-level overview of the cognitive architecture, refer to the [`Cognitive Architecture Overview`](cognitive-overview.md).