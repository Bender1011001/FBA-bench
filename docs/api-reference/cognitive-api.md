# Cognitive System API Reference

This document provides a comprehensive API reference for interacting with FBA-Bench's cognitive systems, including hierarchical planning, structured reflection, and memory management. Developers can use these interfaces to extend or integrate with the cognitive architecture programmatically.

## 1. `AdvancedAgent`

The primary entry point for agents with cognitive capabilities.

-   **Module**: [`agents/advanced_agent.py`](agents/advanced_agent.py)
-   **Class**: `AdvancedAgent`

### Constructor

`__init__(self, name: str, config: CognitiveConfig = None, llm_client: LLMClient = None)`

-   **`name`**: (`str`, required) The name of the agent.
-   **`config`**: (`CognitiveConfig`, optional) An instance of `CognitiveConfig` defining the agent's cognitive parameters. If `None`, defaults are loaded.
-   **`llm_client`**: (`LLMClient`, optional) An instance of `LLMClient` for LLM interactions. If `None`, a default client is initialized.

### Key Methods

#### `act(self, current_state: dict, marketplace_data: dict) -> list[Event]`
Orchestrates the agent's decision-making process, including planning, tool use, and potentially invoking reflection. Returns a list of actions (as `Event` objects) for the simulation.

-   **`current_state`**: (`dict`, required) The agent's internal state.
-   **`marketplace_data`**: (`dict`, required) Current observed data from the simulation marketplace.
-   **Returns**: `list[Event]` - A list of proposed actions.
-   **Raises**: `AgentDecisionError`, `PlanningError`, `LLMInteractionError`

#### `reflect(self, simulation_results: dict)`
Triggers the agent's structured reflection process based on recent simulation outcomes. (Usually called internally by the simulation loop at configured intervals).

-   **`simulation_results`**: (`dict`, required) A summary of recent simulation outcomes relevant for reflection.
-   **Returns**: `list[str]` - A list of generated insights (strings).
-   **Raises**: `ReflectionError`, `LLMInteractionError`

#### `update_memory(self, new_information: Any, source: str)`
Allows external components to inject new information into the agent's memory system.

-   **`new_information`**: (`Any`, required) The data to be stored.
-   **`source`**: (`str`, required) The source of the information (e.g., "observation", "reflection_insight", "event").
-   **Returns**: `bool` - True if memory was updated successfully, False otherwise.
-   **Raises**: `MemoryValidationError`, `MemoryIntegrityError`

## 2. `HierarchicalPlanner`

Manages the agent's multi-level strategic and tactical planning.

-   **Module**: [`agents/hierarchical_planner.py`](agents/hierarchical_planner.py)
-   **Class**: `HierarchicalPlanner`

### Constructor

`__init__(self, agent_name: str, config: dict, llm_client: LLMClient = None)`

-   **`agent_name`**: (`str`, required) The name of the agent using this planner.
-   **`config`**: (`dict`, required) Configuration dictionary for planning (e.g., `depth`, `goal_horizon`).
-   **`llm_client`**: (`LLMClient`, optional) LLM client instance.

### Key Methods

#### `set_strategic_goals(self, goals: list[str])`
Sets the long-term, high-level strategic goals for the agent.

-   **`goals`**: (`list[str]`, required) A list of strategic goal descriptions.

#### `generate_tactical_plan(self, current_state: dict) -> dict`
Generates a detailed tactical plan based on current state and strategic goals.

-   **`current_state`**: (`dict`, required) The current state of the agent and simulation.
-   **Returns**: `dict` - A dictionary representing the tactical plan.

## 3. `ReflectionModule`

Handles the structured reflection process.

-   **Module**: [`memory_experiments/reflection_module.py`](memory_experiments/reflection_module.py)
-   **Class**: `ReflectionModule`

### Constructor

`__init__(self, agent_name: str, config: dict, agent_state: Any)`

-   **`agent_name`**: (`str`, required) The name of the agent.
-   **`config`**: (`dict`, required) Configuration for reflection (e.g., `frequency`, `insight_generation`).
-   **`agent_state`**: (`Any`, required) An object providing access to agent's memory, LLM client, and simulation context.

### Key Methods

#### `perform_reflection(self) -> list[str]`
Executes the reflection process, reviewing past actions/outcomes and generating insights.

-   **Returns**: `list[str]` - A list of generated insights.

## 4. Memory Managers (`DualMemoryManager`, `MemoryValidator`, `MemoryEnforcer`)

These modules collectively manage the agent's memory.

-   **Modules**: [`memory_experiments/dual_memory_manager.py`](memory_experiments/dual_memory_manager.py), [`memory_experiments/memory_validator.py`](memory_experiments/memory_validator.py), [`memory_experiments/memory_enforcer.py`](memory_experiments/memory_enforcer.py)

### `DualMemoryManager` (Main Memory Interface)

#### Constructor
`__init__(self, agent_name: str, config: dict)`

-   **`agent_name`**: (`str`, required)
-   **`config`**: (`dict`, required) Memory configuration.

#### Key Methods
-   `add_to_short_term(self, information: Any, type: str = "general")`: Adds information to short-term memory.
-   `add_to_long_term(self, information: Any, type: str = "general")`: Persists information to long-term memory.
-   `retrieve_relevant_memories(self, query: str, limit: int = 5) -> list[str]`: Retrieves contextually relevant memories.

### `MemoryValidator`

#### Constructor
`__init__(self, memory_manager: DualMemoryManager, config: dict = None)`

#### Key Methods
-   `validate_entry(self, entry: Any) -> bool`: Validates a single memory entry.
-   `validate_all_memory(self) -> ValidationReport`: Performs a comprehensive validation of all memories.

### `MemoryEnforcer`

(Typically operates internally within `DualMemoryManager` during `add` operations)

#### Key Methods
-   `enforce_constraints(self, new_entry: Any, existing_memory: list[Any]) -> Any`: Applies rules to ensure new entries maintain integrity.