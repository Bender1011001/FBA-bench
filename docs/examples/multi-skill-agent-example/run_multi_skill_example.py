import os
from fba_bench.agents.multi_domain_controller import MultiDomainController
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.agents.skill_config import SkillConfig
from fba_bench.observability.observability_config import ObservabilityConfig
from fba_bench.instrumentation.tracer import Tracer
from fba_bench.llm_interface.openrouter_client import OpenRouterClient # Example LLM client

def run_multi_skill_agent_example():
    print("--- Starting Multi-Skill Agent Example Simulation ---")

    # 1. Load Configurations
    current_dir = os.path.dirname(__file__)
    multi_skill_config_path = os.path.join(current_dir, "multi_skill_agent_config.yaml")
    scenario_config_path = os.path.join(current_dir, "complex_marketplace_scenario.yaml")

    try:
        skill_config = SkillConfig.from_yaml(multi_skill_config_path)
    except Exception as e:
        print(f"Error loading multi-skill config: {e}")
        print("Please ensure multi_skill_agent_config.yaml is valid and exists.")
        return

    try:
        scenario_engine = ScenarioEngine.from_yaml(scenario_config_path)
    except Exception as e:
        print(f"Error loading scenario config: {e}")
        print("Please ensure complex_marketplace_scenario.yaml is valid and exists.")
        return

    observability_config = ObservabilityConfig.load_default()
    tracer = Tracer(config=observability_config.tracing)

    # 2. Initialize LLM Client
    llm_api_key = os.getenv("OPENROUTER_API_KEY")
    if not llm_api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        print("Please set your OpenRouter API key to run LLM-powered agents.")
        return
    llm_client = OpenRouterClient(api_key=llm_api_key)

    # 3. Initialize the Multi-Domain Controller Agent
    print(f"\nInitializing MultiDomainController with configured skills...")
    agent = MultiDomainController(
        name="MultiSkilledAgent-01", # You can name your agent here
        skill_config=skill_config,
        llm_client=llm_client
    )

    # 4. Run the Simulation
    print(f"\nRunning simulation for scenario: {scenario_engine.scenario_name}")
    print(f"Simulation duration: {scenario_engine.duration} days")

    results = scenario_engine.run_simulation(agent)

    # 5. Process and Display Results
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
    run_multi_skill_agent_example()