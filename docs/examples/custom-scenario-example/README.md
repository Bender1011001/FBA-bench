# Custom Scenario Example

This directory provides a working example of creating and running a custom business scenario in FBA-Bench. It demonstrates how to define initial marketplace conditions, schedule dynamic events, and set specific goals for agent evaluation.

## Features Demonstrated

-   **Modular Scenario Definition**: Using YAML files to define scenario parameters.
-   **Marketplace Events**: Scheduling various events to challenge agent adaptability.
-   **Goal Definition**: Setting clear, measurable objectives for agent performance.
-   **Integration with Agents**: Running a custom scenario with an FBA-Bench agent.

## Directory Structure

-   `run_custom_scenario_example.py`: The main script to run this custom scenario.
-   `my_custom_scenario.yaml`: The definition of the custom scenario.

## How to Run the Example

1.  **Navigate to the Example Directory**:
    ```bash
    cd docs/examples/custom-scenario-example
    ```

2.  **Ensure FBA-Bench Dependencies are Installed**: If you haven't already, install the core FBA-Bench requirements:
    ```bash
    pip install -r ../../../requirements.txt
    ```
    (Adjust path as necessary based on your current working directory relative to the FBA-Bench root)

3.  **Configure LLM Access**: Ensure your LLM API keys are set as environment variables (e.g., `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`) if your agent uses LLMs.

4.  **Run the Simulation**:
    ```bash
    python run_custom_scenario_example.py
    ```

## Analyzing the Output

-   **Console Output**: The script will print high-level simulation progress, event occurrences, and final metrics/goal status.
-   **Trace Files**: A detailed trace of the simulation (including scenario events) will be saved to the `artifacts/` or `simulation_traces/` directory. Use the FBA-Bench frontend's `TraceViewer` or the `TraceAnalyzer` API ([`docs/observability/trace-analysis.md`](docs/observability/trace-analysis.md)) for deep inspection of how the scenario unfolded.

## Customization

-   **`my_custom_scenario.yaml`**: This is the core file to modify. Experiment with:
    -   Changing `initial_state` parameters (starting capital, inventory, base demand).
    -   Adding, removing, or modifying `marketplace_events` (change types, timing, impact, products affected).
    -   Defining new `goals` or adjusting existing targets.
    -   Refer to the [`Scenario and Curriculum Configuration`](../../configuration/scenario-config.md) guide for all available parameters and event types.
-   **`run_custom_scenario_example.py`**: Modify the script to integrate different agents or conduct multiple runs with varied scenario parameters.