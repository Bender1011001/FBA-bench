# Cognitive Agent Example

This directory contains a complete, working example of an FBA-Bench agent configured with advanced cognitive capabilities: hierarchical planning, structured reflection, and enhanced memory integration. Use this example to understand how these features work together and as a starting point for developing your own intelligent agents.

## Features Demonstrated

-   **Hierarchical Planning**: Agent sets long-term strategic goals and breaks them into quarterly tactical plans.
-   **Structured Reflection**: Agent periodically reviews its performance and generates insights.
-   **Memory Integration**: Agent stores and retrieves information, with validation and consistency checks.
-   **LLM Interaction**: Demonstrates how an agent interacts with LLMs for complex reasoning tasks.

## Directory Structure

-   `run_cognitive_example.py`: The main script to run this cognitive agent simulation.
-   `cognitive_agent_config.yaml`: Configuration file for the agent's cognitive architecture.
-   `sample_scenario.yaml`: A simple market scenario for the agent to operate in.

## How to Run the Example

1.  **Navigate to the Example Directory**:
    ```bash
    cd docs/examples/cognitive-agent-example
    ```

2.  **Ensure FBA-Bench Dependencies are Installed**: If you haven't already, install the core FBA-Bench requirements:
    ```bash
    pip install -r ../../../requirements.txt
    ```
    (Adjust path as necessary based on your current working directory relative to the FBA-Bench root)

3.  **Configure LLM Access**: Ensure your LLM API keys are set as environment variables (e.g., `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`). The default LLM is configured in `cognitive_agent_config.yaml`.

4.  **Run the Simulation**:
    ```bash
    python run_cognitive_example.py
    ```

## Analyzing the Output

-   **Console Output**: The script will print high-level simulation progress and final metrics.
-   **Trace Files**: A detailed trace of the agent's internal thoughts, LLM calls, and decisions will be saved to the `artifacts/` or `simulation_traces/` directory (configured in `observability_config.yaml`). Use the FBA-Bench frontend's `TraceViewer` or the `TraceAnalyzer` API ([`docs/observability/trace-analysis.md`](docs/observability/trace-analysis.md)) for deep inspection.
-   **Memory Persistence**: If configured, the agent's memory will be persisted in the directory specified by `memory_path` in `cognitive_agent_config.yaml`.

### Inspecting the Trace for Cognitive Activities

After running, load the generated trace file and look for:
-   **Planning Events**: You'll see records of when `HierarchicalPlanner` creates strategic goals and tactical plans.
-   **Reflection Events**: Look for `ReflectionModule` activities, including the context reviewed and insights generated.
-   **Memory Interactions**: Observe how information is added to and retrieved from the agent's memory.

## Customization

-   **`cognitive_agent_config.yaml`**: Modify this file to experiment with different `depth` for planning, `frequency` for reflection, or `validation` options for memory. Refer to the [`Cognitive System Configuration`](../../configuration/cognitive-config.md) guide for all available parameters.
-   **`sample_scenario.yaml`**: Adjust the market conditions, events, or goals to challenge the cognitive agent in different ways. See the [`Scenario and Curriculum Configuration`](../../configuration/scenario-config.md) guide.
-   **`run_cognitive_example.py`**: Modify the script to integrate your custom FBA-Bench components or orchestrate iterative experiments.