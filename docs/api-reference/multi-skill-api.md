# Multi-Skill Agent System API Reference

This document provides a comprehensive API reference for interacting with FBA-Bench's multi-skill agent system, including the `MultiDomainController`, the `SkillCoordinator`, and individual `BaseSkill` modules.

## 1. `MultiDomainController`

The orchestrator for agents capable of leveraging multiple specialized skills.

-   **Module**: [`agents/multi_domain_controller.py`](agents/multi_domain_controller.py)
-   **Class**: `MultiDomainController`

### Constructor

`__init__(self, name: str, skill_config: SkillConfig = None, llm_client: LLMClient = None)`

-   **`name`**: (`str`, required) The name of the multi-skill agent.
-   **`skill_config`**: (`SkillConfig`, optional) An instance of `SkillConfig` defining enabled skills and coordination strategy. If `None`, defaults are loaded.
-   **`llm_client`**: (`LLMClient`, optional) An instance of `LLMClient` for LLM interactions within skills. If `None`, a default client is initialized.

### Key Methods

#### `act(self, current_state: dict, marketplace_data: dict) -> list[Event]`
Orchestrates the decision-making by invoking relevant skill modules and coordinating their proposed actions.

-   **`current_state`**: (`dict`, required) The agent's internal state.
-   **`marketplace_data`**: (`dict`, required) Current observed data from the simulation marketplace.
-   **Returns**: `list[Event]` - A list of consolidated actions proposed by the agent.
-   **Raises**: `SkillExecutionError`, `CoordinationError`

#### `register_skill_module(self, skill_instance: BaseSkill)`
Registers a custom skill module with the controller. (Typically done internally during initialization based on `skill_config`).

-   **`skill_instance`**: (`BaseSkill`, required) An initialized instance of a class inheriting from `BaseSkill`.

#### `get_active_skills(self) -> list[BaseSkill]`
Returns a list of currently active and initialized skill modules.

## 2. `SkillCoordinator`

Responsible for resolving conflicts and synthesizing actions from multiple skill modules.

-   **Module**: [`agents/skill_coordinator.py`](agents/skill_coordinator.py)
-   **Class**: `SkillCoordinator`

### Constructor

`__init__(self, config: dict, llm_client: LLMClient = None)`

-   **`config`**: (`dict`, required) Configuration for skill coordination (e.g., `conflict_resolution_strategy`).
-   **`llm_client`**: (`LLMClient`, optional) LLM client instance, used for LLM-mediated conflict resolution.

### Key Methods

#### `coordinate_actions(self, proposed_actions: list[dict]) -> list[Event]`
Analyzes a list of proposed actions from different skills, resolves conflicts based on configured strategy, and returns a single list of final actions.

-   **`proposed_actions`**: (`list[dict]`, required) A list of dictionaries, where each dictionary represents an action proposed by a skill (typically including a `source_skill` key).
-   **Returns**: `list[Event]` - A list of `Event` objects representing the final, coordinated actions.
-   **Raises**: `CoordinationError`

## 3. `BaseSkill` (Abstract Base Class)

The foundational class from which all specific skill modules must inherit.

-   **Module**: [`agents/skill_modules/base_skill.py`](agents/skill_modules/base_skill.py)
-   **Class**: `BaseSkill`

### Constructor

`__init__(self, agent_name: str, config: dict)`

-   **`agent_name`**: (`str`, required) The name of the agent this skill belongs to.
-   **`config`**: (`dict`, required) Configuration specific to this skill module.

### Abstract Methods (Must be implemented by subclasses)

#### `execute(self, current_state: dict, marketplace_data: dict) -> list[Event]`
Contains the core logic of the skill. Analyzes the current simulation state and proposes actions.

-   **`current_state`**: (`dict`, required) The agent's internal state.
-   **`marketplace_data`**: (`dict`, required) Current observed data from the simulation marketplace.
-   **Returns**: `list[Event]` - A list of `Event` objects representing actions proposed by this skill.

### Optional Methods (Can be overridden by subclasses)

#### `observe(self, events: list[Event])`
Allows the skill to react to global events published on the event bus, even if they don't directly trigger `execute`.

-   **`events`**: (`list[Event]`, required) A list of recent simulation events.

#### `get_tools(self) -> list[Callable[..., Any]]`
If the skill exposes specific tools that can be invoked by an LLM within its context, this method returns a list of callable functions (typically decorated with metadata for LLM tool calling).

-   **Returns**: `list[Callable[..., Any]]` - A list of callable tool functions.

## 4. `SkillConfig`

The configuration schema for the multi-skill agent system.

-   **Module**: [`agents/skill_config.py`](agents/skill_config.py)
-   **Class**: `SkillConfig`

### Key Properties

-   **`enabled_skills`**: (`list[str]`) List of skill names ('marketing_manager', 'supply_manager', etc.) that are active for the agent.
-   **`coordination`**: (`dict`) Configuration for the `SkillCoordinator`, including `conflict_resolution_strategy` and LLM settings for resolution.
-   **`skills`**: (`dict`) A nested dictionary allowing for skill-specific configurations (e.g., `skills.marketing_manager.llm_model`).

### Factory Method

#### `from_yaml(cls, file_path: str) -> SkillConfig`
Loads `SkillConfig` from a YAML file.