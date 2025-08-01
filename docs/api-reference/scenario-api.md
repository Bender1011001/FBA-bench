# Scenario Management API Reference

This document provides a comprehensive API reference for interacting with FBA-Bench's scenario management system, including the `ScenarioEngine`, `CurriculumValidator`, and `DynamicGenerator`. These APIs allow for programmatic creation, loading, and validation of simulation environments.

## 1. `ScenarioEngine`

The core component for loading, interpreting, and running simulation scenarios.

-   **Module**: [`scenarios/scenario_engine.py`](scenarios/scenario_engine.py)
-   **Class**: `ScenarioEngine`

### Constructor

`__init__(self, scenario_config: dict)`

-   **`scenario_config`**: (`dict`, required) A dictionary representing the parsed scenario configuration (either loaded from YAML or dynamically generated).

### Class Methods

#### `from_yaml(cls, file_path: str) -> ScenarioEngine`
Factory method to load a scenario from a YAML file.

-   **`file_path`**: (`str`, required) The path to the scenario YAML file.
-   **Returns**: `ScenarioEngine` - An instance of the ScenarioEngine initialized with the loaded configuration.
-   **Raises**: `IOError`, `YAMLParseError`, `ValidationError`

### Key Methods

#### `run_simulation(self, agent: Any) -> dict`
Executes the simulation from the defined scenario with a given agent.

-   **`agent`**: (`Any`, required) An instance of an FBA-Bench agent (e.g., `AdvancedAgent`, `MultiDomainController`) that will interact with the simulation.
-   **Returns**: `dict` - A dictionary containing simulation results, metrics, and goal completion status.
-   **Raises**: `SimulationRuntimeError`

#### `get_scenario_config(self) -> dict`
Retrieves the full configuration dictionary used to initialize the scenario.

-   **Returns**: `dict` - The scenario configuration.

#### `get_initial_state(self) -> dict`
Returns the initial state of the marketplace as defined in the scenario.

-   **Returns**: `dict` - The initial simulation state.

#### `schedule_event(self, event: Event, delay_days: int = 0)`
Programmatically schedules an event to occur at a future point in the simulation.

-   **`event`**: (`Event`, required) An instance of an FBA-Bench `Event` to be scheduled.
-   **`delay_days`**: (`int`, optional) Number of simulation days from the current time to delay the event. Default is `0` (immediate).

## 2. `CurriculumValidator`

A tool for validating scenarios against predefined criteria for difficulty tiers.

-   **Module**: [`scenarios/curriculum_validator.py`](scenarios/curriculum_validator.py)
-   **Class**: `CurriculumValidator`

### Constructor

`__init__(self, tier_configs_path: str = "constraints/tier_configs/")`

-   **`tier_configs_path`**: (`str`, optional) Path to the directory containing tier configuration YAMLs (e.g., `t0_config.yaml`).

### Key Methods

#### `validate_scenario(self, scenario_data: dict, target_tier: str) -> ValidationReport`
Validates a given scenario configuration against the rules defined for a `target_tier`.

-   **`scenario_data`**: (`dict`, required) The scenario configuration to validate.
-   **`target_tier`**: (`str`, required) The name of the target difficulty tier (e.g., "T0", "T1", "T3").
-   **Returns**: `ValidationReport` - An object detailing the validity status and any detected issues.

#### `compare_to_golden_master(self, scenario_data: dict, golden_master_path: str) -> ComparisonResult`
Compares a scenario's expected behavior or structure against a previously recorded "golden master" benchmark.

-   **`scenario_data`**: (`dict`, required) The scenario configuration to compare.
-   **`golden_master_path`**: (`str`, required) Path to the golden master JSON file.
-   **Returns**: `ComparisonResult` - An object indicating consistency and any discrepancies.

## 3. `DynamicGenerator`

Enables procedural and programmatic generation of simulation scenarios.

-   **Module**: [`scenarios/dynamic_generator.py`](scenarios/dynamic_generator.py)
-   **Class**: `DynamicGenerator`

### Constructor

`__init__(self, base_scenario_template_path: str = None)`

-   **`base_scenario_template_path`**: (`str`, optional) Path to a base YAML file to use as a template for generation.

### Key Methods

#### `generate_scenario(self, params: dict) -> dict`
Creates a new scenario configuration dictionary based on provided parameters and internal generation rules.

-   **`params`**: (`dict`, required) A dictionary of parameters that guide the scenario generation (e.g., `duration_days`, `event_density`, `num_competitors`).
-   **Returns**: `dict` - A newly generated scenario configuration.

#### `inject_dynamic_events(self, base_scenario: dict, event_rules: list[dict]) -> dict`
Injects dynamically created events into an existing scenario configuration.

-   **`base_scenario`**: (`dict`, required) The existing scenario dictionary.
-   **`event_rules`**: (`list[dict]`, required) Rules defining how new events should be generated (e.g., type, frequency, impact).
-   **Returns**: `dict` - The modified scenario configuration with new events.