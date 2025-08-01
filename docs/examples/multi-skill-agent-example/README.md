# Multi-Skill Agent Example

This directory contains a complete, working example of an FBA-Bench agent configured with multi-skill capabilities. It demonstrates how an agent can leverage specialized skill modules (e.g., Supply Manager, Marketing Manager, Financial Analyst) and coordinate their actions to operate in a complex business scenario.

## Features Demonstrated

-   **Multi-Domain Expertise**: Agent utilizes multiple specialized skills.
-   **Skill Coordination**: Demonstrates how conflicting or interdependent actions from different skills are resolved.
-   **Event-Driven Triggers**: Skills react to specific events in the simulation.
-   **Configurable Skill Set**: Easily enable/disable skills for different experiments.

## Directory Structure

-   `run_multi_skill_example.py`: The main script to run this multi-skill agent simulation.
-   `multi_skill_agent_config.yaml`: Configuration file for the agent's multi-skill architecture.
-   `complex_marketplace_scenario.yaml`: A scenario designed to require diverse skill sets.

## How to Run the Example

1.  **Navigate to the Example Directory**:
    ```bash
    cd docs/examples/multi-skill-agent-example
    ```

2.  **Ensure FBA-Bench Dependencies are Installed**: If you haven't already, install the core FBA-Bench requirements:
    ```bash
    pip install -r ../../../requirements.txt
    ```
    (Adjust path as necessary based on your current working directory relative to the FBA-Bench root)

3.  **Configure LLM Access**: Ensure your LLM API keys are set as environment variables (e.g., `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`). The LLM models used by the skills are configured in `multi_skill_agent_config.yaml`.

4.  **Run the Simulation**:
    ```bash
    python run_multi_skill_example.py
    ```

## Analyzing the Output

-   **Console Output**: The script will print high-level simulation progress and final metrics.
-   **Trace Files**: A detailed trace of the agent's internal thoughts, skill invocations, LLM calls, and coordinated decisions will be saved to the `artifacts/` or `simulation_traces/` directory. Use the FBA-Bench frontend's `TraceViewer` or the `TraceAnalyzer` API ([`docs/observability/trace-analysis.md`](docs/observability/trace-analysis.md)) for deep inspection.
-   **Skill-Specific Logs**: Individual skill modules might also generate their own logs, visible in the overall simulation output.

### Inspecting the Trace for Multi-Skill Activities

After running, load the generated trace file and look for:
-   **Skill Invocations**: Events showing which skills were activated and when.
-   **Proposed Actions**: The actions recommended by each individual skill module.
-   **Coordination Decisions**: How the `SkillCoordinator` resolved conflicts and synthesized final actions.

## Customization

-   **`multi_skill_agent_config.yaml`**: Modify this file to experiment with different sets of `enabled_skills`, change `conflict_resolution_strategy`, or fine-tune individual skill parameters. Refer to the [`Multi-Skill Agent Configuration`](../../configuration/skill-config.md) guide for all available parameters.
-   **`complex_marketplace_scenario.yaml`**: Adjust market conditions, events, or goals to emphasize specific business challenges requiring different skill sets. See the [`Scenario and Curriculum Configuration`](../../configuration/scenario-config.md) guide.
-   **`run_multi_skill_example.py`**: Modify the script to integrate your custom FBA-Bench components or orchestrate iterative experiments.