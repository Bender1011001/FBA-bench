import os
from fba_bench.agents.advanced_agent import AdvancedAgent
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.agents.cognitive_config import CognitiveConfig
from fba_bench.instrumentation.tracer import Tracer
from fba_bench.observability.observability_config import ObservabilityConfig
from fba_bench.llm_interface.openrouter_client import OpenRouterClient # Example LLM client

def run_cognitive_agent_example():
    print("--- Starting Cognitive Agent Example Simulation ---")

    # 1. Load Configurations
    # Assumes these config files are in the same directory as this script.
    # In a real setup, you might have a central config loading mechanism.
    current_dir = os.path.dirname(__file__)
    cognitive_config_path = os.path.join(current_dir, "cognitive_agent_config.yaml")
    scenario_config_path = os.path.join(current_dir, "sample_scenario.yaml")

    # Load Cognitive Configuration
    try:
        cognitive_config = CognitiveConfig.from_yaml(cognitive_config_path)
    except Exception as e:
        print(f"Error loading cognitive config: {e}")
        print("Please ensure cognitive_agent_config.yaml is valid and exists.")
        return

    # Load Scenario Configuration
    try:
        scenario_engine = ScenarioEngine.from_yaml(scenario_config_path)
    except Exception as e:
        print(f"Error loading scenario config: {e}")
        print("Please ensure sample_scenario.yaml is valid and exists.")
        return

    # Load Observability Configuration (for tracing etc.)
    # In a larger system, this might be loaded globally
    observability_config = ObservabilityConfig.load_default()
    tracer = Tracer(config=observability_config.tracing)

    # 2. Initialize LLM Client
    # Ensure OPENROUTER_API_KEY environment variable is set
    llm_api_key = os.getenv("OPENROUTER_API_KEY")
    if not llm_api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        print("Please set your OpenRouter API key to run LLM-powered agents.")
        return
    llm_client = OpenRouterClient(api_key=llm_api_key)

    # 3. Initialize the Advanced Agent
    print(f"\nInitializing AdvancedAgent '{cognitive_config.agent_name}' with cognitive features...")
    agent = AdvancedAgent(
        name=cognitive_config.agent_name,
        config=cognitive_config,
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

    # Accessing trace data (optional, for deeper analysis)
    # trace_data = tracer.get_trace()
    # print(f"\nTrace recorded with {len(trace_data['events'])} events.")
    # from fba_bench.observability.trace_analyzer import TraceAnalyzer
    # analyzer = TraceAnalyzer(trace_data)
    # llm_calls = analyzer.get_llm_calls_for_agent(cognitive_config.agent_name)
    # print(f"Total LLM calls by agent: {len(llm_calls)}")

    print(f"\nDetailed simulation trace saved to: {observability_config.tracing.trace_storage_path} (check config)")
    print("\nTo analyze traces, use the FBA-Bench frontend or TraceAnalyzer API.")

if __name__ == "__main__":
    run_cognitive_agent_example()