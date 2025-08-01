# Learning Agent Example

This directory provides a comprehensive example of an FBA-Bench agent configured for episodic learning. It demonstrates how an agent can retain insights and adapt its behavior across multiple simulation runs, simulating continuous improvement.

## Features Demonstrated

-   **Episodic Learning**: Agent's state and learned parameters persist and evolve over multiple simulation episodes.
-   **State Persistence**: Saving and loading of agent memory, planning weights, or skill parameters.
-   **Performance Tracking**: Monitoring how agent performance changes across episodes.
-   **RL Environment Setup**: (Conceptual) How FBA-Bench can be adapted as an OpenAI Gym-like environment for RL training.

## Directory Structure

-   `run_episodic_learning_example.py`: The main script to run the episodic learning simulation.
-   `learning_agent_config.yaml`: Configuration file for the agent's learning system.
-   `simple_learning_scenario.yaml`: A foundational scenario suitable for learning experiments.

## How to Run the Example

1.  **Navigate to the Example Directory**:
    ```bash
    cd docs/examples/learning-agent-example
    ```

2.  **Ensure FBA-Bench Dependencies are Installed**: If you haven't already, install the core FBA-Bench requirements:
    ```bash
    pip install -r ../../../requirements.txt
    ```
    (Adjust path as necessary based on your current working directory relative to the FBA-Bench root)

3.  **Configure LLM Access**: Ensure your LLM API keys are set as environment variables (e.g., `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`) if your agent uses LLMs for learning processes (e.g., reflection to generate insights).

4.  **Run the Learning Simulation**:
    ```bash
    python run_episodic_learning_example.py
    ```

## Analyzing the Output

-   **Console Output**: The script will print progress for each episode, including initial performance and any changes due to learning.
-   **Learning Data Directory**: Check the directory specified by `learning.storage_path` in `learning_agent_config.yaml` (default: `./learning_data/`). This is where the agent's persistent state and episodic history are saved.
-   **Trace Files**: Detailed traces for each episode might be saved to the `artifacts/` or `simulation_traces/` directory. Use the FBA-Bench frontend's `TraceViewer` or the `TraceAnalyzer` API ([`docs/observability/trace-analysis.md`](docs/observability/trace-analysis.md)) for deep inspection.

### Inspecting Learning Progress

-   Review the console output for "Episode X complete" messages and observe how metrics change over time.
-   Examine the saved learning data directly or use custom analysis scripts to visualize the agent's performance trajectory.

## Customization

-   **`learning_agent_config.yaml`**: Modify this file to experiment with different `learning_rate`, control `insight_persistence`, or specify which `agent_state_keys` are part of the learning process. Refer to the [`Agent Learning System Configuration`](../../configuration/learning-config.md) guide for all available parameters.
-   **`simple_learning_scenario.yaml`**: Adjust the scenario. To test learning robustness, consider introducing slight variations per episode if using a custom scenario. See the [`Scenario and Curriculum Configuration`](../../configuration/scenario-config.md) guide.
-   **`run_episodic_learning_example.py`**: Modify the script to change the number of episodes, integrate a different agent, or introduce more complex learning loops (e.g., integrating with an external RL framework using `FBABenchGymEnv`).