import os
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.agents.advanced_agent import AdvancedAgent # Agent that might propose real-world actions
from fba_bench.integration.integration_config import IntegrationConfig
from fba_bench.integration.real_world_adapter import RealWorldAdapter
from fba_bench.integration.integration_validator import IntegrationValidator
from fba_bench.observability.observability_config import ObservabilityConfig
from fba_bench.instrumentation.tracer import Tracer
from fba_bench.llm_interface.openrouter_client import OpenRouterClient # Example LLM client
from fba_bench.events import PriceUpdateEvent # Example event type that implies real-world action

def run_real_world_integration_example():
    print("--- Starting Real-World Integration Example Simulation ---")

    # 1. Load Configurations
    current_dir = os.path.dirname(__file__)
    integration_config_path = os.path.join(current_dir, "integration_config.yaml")
    scenario_config_path = os.path.join(current_dir, "live_marketplace_scenario.yaml")

    try:
        integration_config = IntegrationConfig.from_yaml(integration_config_path)
    except Exception as e:
        print(f"Error loading integration config: {e}")
        print("Please ensure integration_config.yaml is valid and exists.")
        return

    try:
        scenario_engine = ScenarioEngine.from_yaml(scenario_config_path)
    except Exception as e:
        print(f"Error loading scenario config: {e}")
        print("Please ensure live_marketplace_scenario.yaml is valid and exists.")
        return

    observability_config = ObservabilityConfig.load_default()
    tracer = Tracer(config=observability_config.tracing)

    # 2. Perform Pre-Flight Validation
    validator = IntegrationValidator(config=integration_config)
    print("\nPerforming pre-flight validation of integration setup...")
    if not validator.validate_setup():
        print("Pre-flight validation FAILED. Please check your integration_config.yaml and environment variables.")
        print("Simulation will proceed in dry-run mode if configured, but live integration is not ready.")
        # Depending on criticality, you might exit here
    else:
        print("Pre-flight validation PASSED.")
    
    # 3. Initialize LLM Client
    llm_api_key = os.getenv("OPENROUTER_API_KEY") # Recommended to use env var for API keys
    llm_client = None
    if llm_api_key:
        llm_client = OpenRouterClient(api_key=llm_api_key)
    else:
        print("Warning: OPENROUTER_API_KEY environment variable not set. Agent may not function optimally.")

    # 4. Initialize Agent and RealWorldAdapter
    # The RealWorldAdapter intercepts agent actions and applies safety constraints
    real_world_adapter = RealWorldAdapter(config=integration_config)

    # Initialize your agent. This agent's actions will be passed through the adapter.
    # We are using AdvancedAgent here as an example that could propose PriceUpdateEvent etc.
    agent = AdvancedAgent(name="LiveAgent", llm_client=llm_client)

    # Crucial step: Register the adapter as an action handler for the scenario engine
    # So that agent actions are routed through the adapter before affecting the marketplace.
    # In a full FBA-Bench run, this is handled by the main runner.
    scenario_engine.register_action_handler(real_world_adapter.handle_agent_action)

    print(f"\nIntegration mode: {'DRY RUN (NO LIVE CALLS)' if integration_config.safety_constraints.dry_run_mode else 'LIVE'}")
    if integration_config.safety_constraints.dry_run_mode:
        print("All agent actions will be simulated and checked against safety constraints, but NOT sent to real APIs.")
    else:
        print("WARNING: Agent actions MAY be sent to live marketplace APIs. Ensure you understand the risks!")


    # 5. Run the Simulation
    print(f"\nRunning simulation for scenario: {scenario_engine.scenario_name}")
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
    print("\nReview logs for any safety constraint violations or API interaction details.")
    print("Remember to disable 'dry_run_mode' in integration_config.yaml ONLY when you are ready for live deployment.")


if __name__ == "__main__":
    run_real_world_integration_example()