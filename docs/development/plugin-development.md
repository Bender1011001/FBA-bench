# Creating Community Plugins

FBA-Bench supports a flexible plugin system that allows the community to extend core functionality, add new agent behaviors, and integrate with external systems without modifying the main codebase. This guide details how to create and integrate your own FBA-Bench plugins.

## Plugin Types

FBA-Bench currently supports the following primary plugin types:

-   **Agent Plugins (`plugins/agent_plugins/`)**: Extend the capabilities of existing agents or introduce entirely new agent types. Can inject new cognitive modules, skill modules, or tool definitions. All agent plugins must inherit from [`plugins/agent_plugins/base_agent_plugin.py`](plugins/agent_plugins/base_agent_plugin.py).
-   **Scenario Plugins (`plugins/scenario_plugins/`)**: Add new scenario templates, dynamic event generators, or custom marketplace models. All scenario plugins must inherit from [`plugins/scenario_plugins/base_scenario_plugin.py`](plugins/scenario_plugins/base_scenario_plugin.py).
-   **Tool Plugins (Conceptual)**: Integrate new external tools or data sources that agents can use. These are often manifested as new skill modules within agent plugins.

## Plugin Architecture

The plugin system is managed by the `Plugin Framework` (implemented in [`plugins/plugin_framework.py`](plugins/plugin_framework.py)). It works by:

1.  **Discovery**: Scans predefined plugin directories (e.g., `plugins/`) for plugin modules.
2.  **Loading**: Dynamically loads valid plugin classes.
3.  **Registration**: Allows plugins to register their components (e.g., new skill modules, new scenario types) with the core FBA-Bench system.
4.  **Activation**: Activates plugins based on configuration settings.

## Step-by-Step: Creating an Agent Plugin

This example demonstrates creating a simple agent plugin that adds a new "Health Monitor" skill to agents.

### Step 1: Create Your Plugin Directory and Files

Create a new directory for your plugin (e.g., `plugins/my_health_plugin/`). Inside, create an `__init__.py` (to make it a Python package) and your main plugin file.

```
plugins/
└── my_health_plugin/
    ├── __init__.py
    └── health_monitor_skill.py
    └── my_health_plugin.py # Main plugin registration file
```

### Step 2: Implement Your Custom Skill (Example)

Create `plugins/my_health_plugin/health_monitor_skill.py`:

```python
# plugins/my_health_plugin/health_monitor_skill.py
from fba_bench.agents.skill_modules.base_skill import BaseSkill
from fba_bench.events import Event as FBAEvent # Using alias to avoid conflict if `Event` is common

class HealthMonitorSkill(BaseSkill):
    def __init__(self, agent_name: str, config: dict):
        super().__init__(agent_name, config)
        self.skill_name = "health_monitor"
        self.critical_health_threshold = config.get("critical_health_threshold", 0.2)

    def execute(self, current_state: dict, marketplace_data: dict) -> list[FBAEvent]:
        # Example: Monitor agent's capital and propose actions if low
        current_capital = current_state.get("financials", {}).get("current_capital", 0.0)
        proposed_events = []

        if current_capital < self.critical_health_threshold * 100000: # Assuming 100k initial capital
            self.logger.warning(f"Agent {self.agent_name} capital is critically low: {current_capital}")
            # In a real scenario, you'd propose a financial action event
            # E.g., proposed_events.append(CapitalInjectionRequestEvent(amount=10000))
        return proposed_events

    def observe(self, events: list[FBAEvent]):
        for event in events:
            if event.event_type == "ERROR_EVENT":
                self.logger.error(f"HealthMonitor observed an error: {event.message}")
                # Potentially trigger recovery actions
```

### Step 3: Create Your Main Plugin Class

Create `plugins/my_health_plugin/my_health_plugin.py`:

```python
# plugins/my_health_plugin/my_health_plugin.py
from fba_bench.plugins.agent_plugins.base_agent_plugin import BaseAgentPlugin
from .health_monitor_skill import HealthMonitorSkill # Import your custom skill

class MyHealthPlugin(BaseAgentPlugin):
    PLUGIN_NAME = "MyHealthMonitorPlugin"
    VERSION = "1.0.0"
    DESCRIPTION = "Adds a health monitoring skill to FBA-Bench agents."

    def register(self, core_system):
        """
        Register plugin components with the core FBA-Bench system.
        'core_system' might be a registry for skills, scenarios, etc.
        """
        self.core_system = core_system
        print(f"[{self.PLUGIN_NAME}] Registering HealthMonitorSkill...")
        core_system.register_skill_class("health_monitor", HealthMonitorSkill)
        self.logger.info(f"Plugin '{self.PLUGIN_NAME}' registered successfully.")

    def activate(self):
        """
        Called when the plugin is activated. Perform any setup or startup tasks.
        """
        self.logger.info(f"Plugin '{self.PLUGIN_NAME}' activated.")

    def deactivate(self):
        """
        Called when the plugin is deactivated or FBA-Bench shuts down. Clean up resources.
        """
        self.logger.info(f"Plugin '{self.PLUGIN_NAME}' deactivated.")

# Example from template_agent_plugin.py - this would be typically discovered
# by the plugin framework.
# If you need to make it discoverable by `npx @anaisbetts/mcp-installer` as a local MCP server,
# you would enable it via that process.
```

### Step 4: Enable Your Plugin

Configure your FBA-Bench instance to discover and enable your new plugin. This is typically done in a main configuration file or via command-line arguments that point to your plugin directory.

1.  **Ensure plugin directory is discoverable**: The `Plugin Framework` scans a default `plugins/` directory. If your plugin is elsewhere, you might need to specify its path.

2.  **Enable in a configuration file**: (Conceptual, depending on FBA-Bench's plugin loading mechanism)

    ```yaml
    # global_config.yaml
    plugins:
      enabled_plugins:
        - MyHealthMonitorPlugin
      plugin_paths:
        - "./plugins/my_health_plugin" # Or wherever your plugin lives
    ```

3.  **Activate in Agent Configuration**: Now, enable the `health_monitor` skill in your agent's `skill_config.yaml`:

    ```yaml
    # agents/skill_config.yaml
    multi_skill_system:
      enabled_skills:
        - supply_manager
        - marketing_manager
        - health_monitor # Your new skill!
      skills:
        health_monitor:
          critical_health_threshold: 0.15 # Customize skill parameter
    ```

## Testing and Validation Requirements

-   **Unit Tests**: Write comprehensive unit tests for your custom skill logic.
-   **Integration Tests**: Test how your plugin interacts with FBA-Bench's core components and other skills.
-   **Validation**: Ensure your plugin correctly registers and de-registers its components.

## Distribution and Sharing Guidelines

If you plan to share your plugin with the community:
-   **GitHub Repository**: Host your plugin code on GitHub.
-   **README.md**: Provide clear installation, usage, and configuration instructions.
-   **Licensing**: Choose an open-source license.
-   **Versioning**: Use semantic versioning (e.g., `1.0.0`).

By following these guidelines, you can contribute powerful and useful extensions to the FBA-Bench ecosystem.