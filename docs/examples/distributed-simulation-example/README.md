# Distributed Simulation Example

This directory contains an example demonstrating how to set up and run FBA-Bench simulations in a distributed manner, utilizing Docker Compose for local orchestration. This allows for concurrent execution of multiple agents, significantly accelerating large-scale experiments.

## Features Demonstrated

-   **Parallel Agent Execution**: Run multiple agents simultaneously across different processes.
-   **Dockerized Components**: All core FBA-Bench services run as Docker containers.
-   **LLM Batching**: Optimization for LLM API calls that occurs automatically when using the distributed setup.
-   **Centralized Event Bus**: Communication between distributed components via Redis.

## Directory Structure

-   `run_local_distributed_example.py`: A script to kick off a distributed simulation.
-   `infrastructure_config.yaml`: Configuration for the distributed infrastructure.
-   `simple_scenario.yaml`: A basic scenario for the distributed agents to interact with.

## How to Run the Example

Before running, ensure you have Docker Desktop installed and running on your machine.

1.  **Navigate to the Example Directory**:
    ```bash
    cd docs/examples/distributed-simulation-example
    ```

2.  **Ensure FBA-Bench Dependencies are Installed**: If you haven't already, install the core FBA-Bench requirements:
    ```bash
    pip install -r ../../../requirements.txt
    ```
    (Adjust path as necessary based on your current working directory relative to the FBA-Bench root)

3.  **Configure LLM Access**: Ensure your LLM API keys are set as environment variables (e.g., `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`). The `agent_runner` containers will inherit these.

4.  **Start the Distributed Services (Docker Compose)**:
    This example uses a simplified `docker-compose.yml` which might be located in `infrastructure/deployment/`. For this example to work directly, you might need to adapt the `docker-compose.yml` or ensure this script explicitly starts the correct services.

    Traditionally:
    ```bash
    # Go to the main deployment directory first
    cd ../../../infrastructure/deployment
    # Start services -- ensure this points to the FBA-Bench root Dockerfile locations
    docker-compose up --build --scale agent_runner=3 -d # Starts 3 agent runners in detached mode
    ```
    Alternatively, for simplicity, `run_local_distributed_example.py` will attempt to use subprocess to manage Docker, but it's often more robust to run Docker Compose manually.

5.  **Run the Simulation Orchestrator**:
    Go back to the example directory if you moved.
    ```bash
    cd docs/examples/distributed-simulation-example
    python run_local_distributed_example.py
    ```

6.  **Stop Distributed Services**:
    Once the simulation is complete or you wish to stop, go back to the `infrastructure/deployment` directory:
    ```bash
    cd ../../../infrastructure/deployment
    docker-compose down # Stops and removes containers
    ```

## Analyzing the Output

-   **Combined Logs**: Docker Compose will show combined logs from all running services. Pay attention to `agent_runner` logs to see individual agent progress.
-   **Trace Files**: Detailed traces for each agent will be saved to the configured `trace_storage_path` (check `infrastructure_config.yaml` or `observability_config.yaml`). Use the FBA-Bench frontend's `TraceViewer` or the `TraceAnalyzer` API ([`docs/observability/trace-analysis.md`](docs/observability/trace-analysis.md)) for deep inspection.
-   **Performance Metrics**: Review the metrics output by the orchestrator and individual runners to understand LLM usage, call batching efficiency, and simulation throughput.

## Customization

-   **`infrastructure_config.yaml`**: Modify this file to experiment with different `batch_size`, `batch_interval_seconds`, `event_bus_type`, or number of `agent_runner_concurrency`. Refer to the [`Infrastructure System Configuration`](../../configuration/infrastructure-config.md) guide for all available parameters.
-   **`simple_scenario.yaml`**: Adjust the scenario complexity or event types. See the [`Scenario and Curriculum Configuration`](../../configuration/scenario-config.md) guide.
-   **`run_local_distributed_example.py`**: Adapt the script to orchestrate more complex distributed experiments or simulations with a larger number of agents. For Kubernetes deployments, refer to the [`Deployment Guide`](../../infrastructure/deployment-guide.md).