import os
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.agents.advanced_agent import AdvancedAgent # Or your preferred agent type
from fba_bench.observability.observability_config import ObservabilityConfig
from fba_bench.instrumentation.tracer import Tracer
from fba_bench.llm_interface.openrouter_client import OpenRouterClient # Example LLM client

def run_custom_scenario_example():
    print("--- Starting Custom Scenario Example Simulation ---")

    # 1. Define Paths to Configuration Files
    current_dir = os.path.dirname(__file__)
    scenario_config_path = os.path.join(current_dir, "my_custom_scenario.yaml")

    # 2. Load the Custom Scenario
    try:
        scenario_engine = ScenarioEngine.from_yaml(scenario_config_path)
    except Exception as e:
        print(f"Error loading custom scenario config: {e}")
        print("Please ensure my_custom_scenario.yaml is valid and exists.")
        return

    # Load Observability Configuration for tracing
    observability_config = ObservabilityConfig.load_default()
    tracer = Tracer(config=observability_config.tracing)

    # 3. Initialize LLM Client (if your agent uses LLMs)
    llm_api_key = os.getenv("OPENROUTER_API_KEY")
    if not llm_api_key:
        print("Warning: OPENROUTER_API_KEY environment variable not set. Agent may not function optimally.")
        llm_client = None # Or a mock client if basic actions are sufficient
    else:
        llm_client = OpenRouterClient(api_key=llm_api_key)

    # 4. Initialize Your Agent
    # For this example, we use a simple AdvancedAgent.
    # You can replace this with any agent type (e.g., MultiDomainController)
    # and configure it as needed.
    print(f"\nInitializing agent for scenario: {scenario_engine.scenario_name}")
    agent = AdvancedAgent(name="CustomScenarioAgent", llm_client=llm_client)

    # 5. Run the Simulation
    print(f"\nRunning simulation for scenario: {scenario_engine.scenario_name}")
    print(f"Scenario Description: {scenario_engine.description}")
    print(f"Simulation duration: {scenario_engine.duration} days")

    results = scenario_engine.run_simulation(agent)

    # 6. Process and Display Results
    print("\n--- Simulation Complete ---")
    print("\nFinal Metrics:")
    for metric, value in results.get("metrics", {}).items():
        print(f"- {metric}: {value:.2f}")

    print("\nGoal Status:")
    for goal, status in results.get("goal_status", {}).items():
        print(f"- {goal}: {'Achieved' if status else 'Not Achieved'}")

    print(f"\nDetailed simulation trace saved to: {observability_config.tracing.trace_storage_path} (check config)")
    print("\nTo analyze traces, use the FBA-Bench frontend or TraceAnalyzer API.")

if __name__ == "__main__":
    run_custom_scenario_example()