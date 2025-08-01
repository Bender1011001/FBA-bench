# Plugin Development Example

This directory provides a basic, runnable example demonstrating how to create and integrate a custom plugin into FBA-Bench. This example focuses on an "Agent Plugin" that introduces a new, simple custom skill module.

## Features Demonstrated

-   **Plugin Structure**: How to organize a standalone FBA-Bench plugin.
-   **Dynamic Skill Injection**: How a plugin can register new skill modules with the FBA-Bench core.
-   **Plugin Activation**: How to enable a custom plugin via agent configuration.

## Directory Structure

-   `plugin_example_agent_config.yaml`: The agent configuration that enables the custom skill from the plugin.
-   `plugin_example_scenario.yaml`: A simple scenario for testing the agent with the plugin.
-   `run_plugin_example.py`: The main script to run this simulation.
-   `my_simple_agent_plugin/`:
    -   `__init__.py`
    -   `my_simple_skill.py`: The custom skill module implemented by the plugin.
    -   `my_simple_agent_plugin.py`: The main plugin class that registers the skill.

## How to Run the Example

1.  **Navigate to the Example Directory**:
    ```bash
    cd docs/examples/plugin-development-example
    ```

2.  **Ensure FBA-Bench Dependencies are Installed**: If you haven't already, install the core FBA-Bench requirements:
    ```bash
    pip install -r ../../../requirements.txt
    ```
    (Adjust path as necessary based on your current working directory relative to the FBA-Bench root)

3.  **Place the Plugin for Discovery**:
    Ensure the `my_simple_agent_plugin/` directory is within a path discoverable by FBA-Bench's plugin system. By default, FBA-Bench often scans its `plugins/` top-level directory. For demonstration, this example assumes it's correctly placed within `docs/examples/plugin-development-example/`.

4.  **Configure LLM Access**: Ensure your LLM API keys are set as environment variables if your agent or custom skill uses LLMs.

5.  **Run the Simulation**:
    ```bash
    python run_plugin_example.py
    ```

## Analyzing the Output

-   **Console Output**: You should see log messages indicating that `MySimpleAgentPlugin` and `MySimpleSkill` are initialized and active. Observe when "MySimpleSkill for MyPluginAgent executing..." messages appear.
-   **Trace Files**: Detailed traces will show the invocation of `MySimpleSkill` within the agent's decision loop. Use the `TraceViewer` or `TraceAnalyzer` API ([`docs/observability/trace-analysis.md`](docs/observability/trace-analysis.md)).

## Customization

-   **`my_simple_agent_plugin/my_simple_skill.py`**: Modify the `execute` method to implement more complex behavior for `MySimpleSkill`.
-   **`my_simple_agent_plugin/my_simple_agent_plugin.py`**: Extend this class to register other components (e.g., new scenario generators) or implement `activate`/`deactivate` logic.
-   **`plugin_example_agent_config.yaml`**: Adjust agent parameters, including LLM settings for `my_simple_skill` if it uses an LLM.
-   **`plugin_example_scenario.yaml`**: Modify the scenario to create conditions that specifically trigger your new skill's logic.
-   Refer to the [`Plugin Development`](../../development/plugin-development.md) guide for comprehensive details on creating various plugin types.