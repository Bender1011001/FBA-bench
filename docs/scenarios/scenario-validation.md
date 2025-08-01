# Testing and Validating Scenarios

Ensuring the quality and fidelity of FBA-Bench scenarios is paramount for reliable agent benchmarking and meaningful experimental results. This guide details the processes and tools available for testing and validating your custom or modified scenarios.

## 1. Static Validation of Scenario Definitions

Before running a simulation, it's essential to validate the YAML structure and content of your scenario files against the expected schema.

-   **Schema Validation**: The `ScenarioConfig` class (defined in [`scenarios/scenario_config.py`](scenarios/scenario_config.py)) uses Pydantic or similar mechanisms to ensure that all required fields are present and that data types and ranges are correct within your scenario YAML.
-   **Syntax Check**: Basic YAML syntax checking.

```python
# scenario_validation_script.py
from fba_bench.scenarios.scenario_config import ScenarioConfig
from fba_bench.scenarios.curriculum_validator import CurriculumValidator # For higher-level validation

def validate_scenario_file(file_path: str):
    try:
        # This will raise a ValidationError if the YAML doesn't match the schema
        scenario_data = ScenarioConfig.from_yaml(file_path)
        print(f"Scenario '{file_path}' successfully validated against schema.")
        return True
    except Exception as e:
        print(f"Error validating scenario '{file_path}': {e}")
        return False

# Example usage:
validate_scenario_file("scenarios/tier_0_baseline.yaml")
validate_scenario_file("scenarios/my_invalid_scenario_template.yaml") # Assuming this file exists and is invalid
```

## 2. Dynamic Validation during Simulation Runs

Beyond static checks, validating scenario behavior *during* live simulations is critical.

### Curriculum Validator
The `CurriculumValidator` (implemented in [`scenarios/curriculum_validator.py`](scenarios/curriculum_validator.py)) performs checks on scenarios within the context of a curriculum (see [`Curriculum Design`](curriculum-design.md)). It can assess if a scenario meets the criteria for its intended difficulty tier (e.g., number of events, market volatility, complexity thresholds).

```python
from fba_bench.scenarios.curriculum_validator import CurriculumValidator
from fba_bench.scenarios.scenario_engine import ScenarioEngine

validator = CurriculumValidator()
tier_1_scenario_engine = ScenarioEngine.from_yaml("scenarios/tier_1_moderate.yaml")

# Check if the scenario adheres to Tier 1 complexity rules
validation_report = validator.validate_scenario(tier_1_scenario_engine.get_scenario_config(), target_tier="T1")

if validation_report.is_valid:
    print(f"Scenario '{tier_1_scenario_engine.scenario_name}' is valid for Tier 1.")
else:
    print(f"Scenario '{tier_1_scenario_engine.scenario_name}' has validation issues for Tier 1:")
    for issue in validation_report.issues:
        print(f"- {issue}")

# The validator can also compare a scenario against a known "golden master"
# to detect unintended changes.
```

### Golden Master Testing
For critical scenarios, FBA-Bench supports Golden Master (or snapshot) testing. A successful run of a scenario with a known-good agent is recorded as a "golden master" trace. Subsequent runs are compared against this master to detect any deviations, ensuring reproducibility and preventing regressions.

-   **Capture Golden Master**:
    ```python
    from fba_bench.reproducibility.golden_master import GoldenMasterRecorder
    recorder = GoldenMasterRecorder("golden_masters/baseline_T0.json")
    recorder.start_recording()
    # Run your simulation
    # scenario_engine.run_simulation(agent)
    recorder.stop_recording()
    ```
-   **Verify against Golden Master**:
    ```python
    from fba_bench.reproducibility.golden_master import GoldenMasterVerifier
    verifier = GoldenMasterVerifier("golden_masters/baseline_T0.json")
    # Run your new simulation
    # new_results = scenario_engine.run_simulation(agent)
    # is_consistent = verifier.verify(new_results)
    # print(f"New simulation is consistent with golden master: {is_consistent}")
    ```
    See [`reproducibility/golden_master.py`](reproducibility/golden_master.py) for details.

### Automated Testing (`tests/integration/`)
The `tests/integration/` directory contains various test suites designed to validate scenario behavior, agent-scenario interactions, and overall system integration.
-   `test_cross_system_integration.py`: Validates how different FBA-Bench components interact within a scenario.
-   `test_end_to_end_workflow.py`: Runs full simulation workflows to ensure the entire pipeline is functional.
-   `test_scientific_reproducibility.py`: Specifically targets reproducibility aspects across scenarios with various configurations.

By combining static schema validation, dynamic curriculum checks, and robust golden master testing, you can ensure that your FBA-Bench scenarios are high-quality, reliable, and serve their intended purpose for agent evaluation.