import os
import subprocess
import time
from fba_bench.scenarios.scenario_engine import ScenarioEngine
from fba_bench.infrastructure.scalability_config import ScalabilityConfig
from fba_bench.agents.advanced_agent import AdvancedAgent # Using AdvancedAgent as the example agent type
from fba_bench.observability.observability_config import ObservabilityConfig
from fba_bench.instrumentation.tracer import Tracer

# Note: This script is for demonstration.
# For production-grade distributed simulations, manage Docker Compose/Kubernetes manually
# or use a more robust orchestration framework.

def run_local_distributed_example():
    print("--- Starting Local Distributed Simulation Example ---")

    current_dir = os.path.dirname(__file__)
    infrastructure_config_path = os.path.join(current_dir, "infrastructure_config.yaml")
    scenario_config_path = os.path.join(current_dir, "simple_scenario.yaml")

    # 1. Load Infrastructure Configuration
    try:
        infra_config = ScalabilityConfig.from_yaml(infrastructure_config_path)
    except Exception as e:
        print(f"Error loading infrastructure config: {e}")
        print("Please ensure infrastructure_config.yaml is valid and exists.")
        return

    # Load Observability Configuration (for tracing and output paths)
    observability_config = ObservabilityConfig.load_default()
    tracer = Tracer(config=observability_config.tracing)

    # Assume Docker Compose file is in the main infrastructure/deployment directory
    docker_compose_dir = os.path.abspath(os.path.join(current_dir, "../../infrastructure/deployment"))
    docker_compose_file = os.path.join(docker_compose_dir, "docker-compose.yml")

    if not os.path.exists(docker_compose_file):
        print(f"Error: Docker Compose file not found at {docker_compose_file}")
        print("Please ensure docker-compose.yml exists in infrastructure/deployment.")
        return

    # Set environment variables for Docker Compose containers (e.g., LLM API Key)
    # This is crucial for agents inside containers to access LLMs.
    os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")
    if not os.environ["OPENROUTER_API_KEY"]:
        print("Warning: OPENROUTER_API_KEY not set. LLM-powered agents will likely fail.")

    num_agent_runners = infra_config.distributed_simulation.agent_runner_concurrency
    print(f"\nStarting Docker Compose services with {num_agent_runners} agent runners...")
    # Use -d for detached mode, so this script can continue.
    # --build only if you've made code changes that need to be re-built into images.
    try:
        subprocess.run(
            ["docker-compose", "-f", docker_compose_file, "up", "--build", "--scale", f"agent_runner={num_agent_runners}", "-d"],
            check=True,
            cwd=docker_compose_dir
        )
        print("Docker Compose services started successfully. Waiting for services to become ready...")
        time.sleep(10) # Give services time to spin up
    except subprocess.CalledProcessError as e:
        print(f"Error starting Docker Compose services: {e}")
        print(f"Stderr: {e.stderr.decode()}")
        print("Please check Docker is running and docker-compose.yml is valid.")
        return

    # 2. Prepare Scenario
    try:
        scenario_engine = ScenarioEngine.from_yaml(scenario_config_path)
    except Exception as e:
        print(f"Error loading scenario config: {e}")
        print("Please ensure simple_scenario.yaml is valid and exists.")
        # Attempt to shut down Docker services before exiting
        subprocess.run(["docker-compose", "-f", docker_compose_file, "down"], cwd=docker_compose_dir)
        return

    # 3. Submit Agents to Distributed Coordinator (simplified)
    # In a real distributed setup, we would submit agents to a central coordinator
    # which then assigns them to available agent runners.
    # For this example, we'll simulate running a single agent for clarity,
    # knowing that the Docker setup *is* capable of running multiple in parallel.

    # Normally, you would use an AgentManager connected to the DistributedCoordinator
    # from fba_bench.agent_runners.agent_manager import AgentManager
    # from fba_bench.infrastructure.distributed_coordinator import DistributedCoordinator
    # coordinator = DistributedCoordinator()
    # agent_manager = AgentManager(coordinator=coordinator, ...)
    # agent_manager.submit_agent_for_simulation(agent_instance, scenario_config)

    print(f"\nSimulating submission of an agent to the distributed system for: {scenario_engine.scenario_name}")
    print("In a full distributed setup, the DistributedCoordinator would manage many agents.")

    # Since we can't fully simulate parallel runs easily in a single script without complex
    # multiprocessing, we'll demonstrate one agent running locally to show the concept.
    # The true distributed execution happens WITHIN the Docker containers.

    # This part would typically be replaced by interaction with the DistributedCoordinator API
    # For now, we'll just show the environment setup.
    print("\nTo see true distributed behavior, monitor the Docker container logs during a manual execution of a multi-agent scenario.")
    print("\n--- Distributed Setup Complete. Manual simulation or external orchestration needed now. ---")

    input("\nPress Enter to stop Docker Compose services and clean up...")

    # 6. Stop and Clean Up Docker Services
    print("\nStopping Docker Compose services...")
    try:
        subprocess.run(["docker-compose", "-f", docker_compose_file, "down"], check=True, cwd=docker_compose_dir)
        print("Docker Compose services stopped and cleaned up successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error stopping Docker Compose services: {e}")
        print("You may need to manually run 'docker-compose down' in your infrastructure/deployment directory.")

if __name__ == "__main__":
    run_local_distributed_example()