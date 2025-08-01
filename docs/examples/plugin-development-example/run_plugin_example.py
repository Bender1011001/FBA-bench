import os
import sys
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.agents.multi_domain_controller import MultiDomainController # Agent supporting skills
from fba_bench.agents.skill_config import SkillConfig
from fba_bench.plugins.plugin_framework import PluginFramework
from fba_bench.observability.observability_config import ObservabilityConfig
from fba_bench.instrumentation.tracer import Tracer
from fba_bench.llm_interface.openrouter_client import OpenRouterClient # Example LLM client

# This script demonstrates running an FBA-Bench simulation
# with a dynamically loaded custom plugin that provides a new skill.

def run_plugin_example():
    print("--- Starting Plugin Development Example Simulation ---")

    current_dir = os.path.dirname(__file__)
    agent_config_path = os.path.join(current_dir, "plugin_example_agent_config.yaml")
    scenario_config_path = os.path.join(current_dir, "plugin_example_scenario.yaml")
    plugin_dir = os.path.join(current_dir, "my_simple_agent_plugin")

    # 1. Initialize Plugin Framework and Discover Plugins
    print(f"\nInitializing Plugin Framework and scanning for plugins in: {plugin_dir}")
    plugin_framework = PluginFramework()
    # Add the example plugin directory to be discoverable
    plugin_framework.add_plugin_path(plugin_dir)
    plugin_framework.discover_plugins()

    # If your plugin has a specific activation name (like "MySimpleAgentPlugin"),
    # you might enable it here or via a global config
    # For this example, we expect it to be discovered and its components registered implicitly.

    # 2. Load Agent Configuration that enables the new skill
    try:
        skill_config = SkillConfig.from_yaml(agent_config_path)
    except Exception as e:
        print(f"Error loading agent config: {e}")
        print("Please ensure plugin_example_agent_config.yaml is valid and exists.")
        return

    # 3. Load Scenario
    try:
        scenario_engine = ScenarioEngine.from_yaml(scenario_config_path)
    except Exception as e:
        print(f"Error loading scenario config: {e}")
        print("Please ensure plugin_example_scenario.yaml is valid and exists.")
        return

    # Load Observability Configuration for tracing
    observability_config = ObservabilityConfig.load_default()
    tracer = Tracer(config=observability_config.tracing)

    # 4. Initialize LLM Client (if the custom skill uses LLMs)
    llm_api_key = os.getenv("OPENROUTER_API_KEY")
    if not llm_api_key:
        print("Warning: OPENROUTER_API_KEY environment variable not set. LLM-powered skills may not function.")
        llm_client = None
    else:
        llm_client = OpenRouterClient(api_key=llm_api_key)

    # 5. Initialize the Multi-Domain Controller with the skill config (which includes the plugin's skill)
    print("\nInitializing Multi-Domain Controller with skills (including plugin-provided skill)...")
    agent = MultiDomainController(
        name="MyPluginAgent",
        skill_config=skill_config,
        llm_client=llm_client
    )

    # 6. Run the Simulation
    print(f"\nRunning simulation for scenario: {scenario_engine.scenario_name}")
    print(f"Simulation duration: {scenario_engine.duration} days")

    results = scenario_engine.run_simulation(agent)

    # 7. Process and Display Results
    print("\n--- Simulation Complete ---")
    print("\nFinal Metrics:")
    for metric, value in results.get("metrics", {}).items():
        print(f"- {metric}: {value:.2f}")

    print("\nGoal Status:")
    for goal, status in results.get("goal_status", {}).items():
        print(f"- {goal}: {'Achieved' if status else 'Not Achieved'}")

    print(f"\nDetailed simulation trace saved to: {observability_config.tracing.trace_storage_path} (check config)")
    print("\nCheck the console output and trace for messages from 'MySimpleSkill'.")

if __name__ == "__main__":
    # Add the current directory to Python path to allow direct import of `my_simple_agent_plugin`
    sys.path.append(os.path.dirname(__file__))
    # Manually activate any discovered plugins that might need it (depends on framework details)
    # The PluginFramework's register method might be called automatically if a central runner exists.
    # For this example, we assume skills are discoverable once the path is added and agent uses skill_config.
    
    run_plugin_example()