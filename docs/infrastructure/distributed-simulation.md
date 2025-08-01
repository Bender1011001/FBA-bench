# Setting Up and Running Distributed Simulations

FBA-Bench's distributed simulation capabilities allow you to scale your experiments by running multiple agents and simulation components concurrently across different processes or machines. This guide provides instructions for setting up and utilizing the distributed environment.

## Architecture for Distribution

The distributed architecture relies on:
-   **Central Process/Orchestrator**: Manages the overall simulation flow, distributes tasks, and aggregates results.
-   **Agent Runners**: Dedicated processes or containers that host and execute individual agent simulations in parallel.
-   **Distributed Event Bus**: A messaging system (e.g., Redis, Kafka) that enables decoupled communication between the central orchestrator, agent runners, and other services.
-   **Shared Storage (Optional)**: For persistent memory, trace data, or results.

## Local Distributed Setup with Docker Compose

For rapid local development and testing of distributed simulations, FBA-Bench provides a `docker-compose.yml` file.

1.  **Ensure Docker is Running**: Make sure Docker Desktop (or your Docker daemon) is active on your system.
2.  **Navigate to Deployment Directory**:
    ```bash
    cd infrastructure/deployment
    ```
3.  **Start Services**: Use `docker-compose up` to build and start the necessary services. The `--scale` flag allows you to specify the number of agent runner instances.

    ```bash
    docker-compose up --build --scale agent_runner=5 # Starts 5 concurrent agent runners
    ```
    -   `--build`: Rebuilds service images (useful after code changes).
    -   `--scale agent_runner=X`: Specifies `X` instances of the `agent_runner` service. Adjust `X` based on your system's resources.

4.  **Monitor Output**: Docker Compose will output logs from all services. You'll see messages indicating agents being assigned to runners, simulation progress, and results being aggregated.
5.  **Stop Services**: When done, press `Ctrl+C` in the terminal where Docker Compose is running, then optionally prune unused resources:

    ```bash
    docker-compose down --volumes --rmi all
    ```

## Cloud Deployment with Kubernetes

For production-grade, highly scalable, and resilient distributed simulations, FBA-Bench can be deployed on Kubernetes.

1.  **Prerequisites**:
    -   A running Kubernetes cluster (e.g., MiniKube for local testing, or a cloud-managed cluster like GKE, EKS, AKS).
    -   `kubectl` configured to connect to your cluster.
    -   Docker images of FBA-Bench components pushed to a container registry (e.g., Docker Hub, GCR).

2.  **Review Kubernetes Manifests**:
    Examine the Kubernetes YAML files in [`infrastructure/deployment/kubernetes.yaml`](infrastructure/deployment/kubernetes.yaml). These define:
    -   `Deployments` for the central orchestrator, agent runners, and any shared services (e.g., Redis for event bus).
    -   `Services` for exposing necessary endpoints.
    -   `ConfigMaps` for externalizing configuration.
    -   `PersistentVolumeClaims` if persistent storage is required for memory or results.

3.  **Deploy to Kubernetes**:
    Apply the manifest files to your cluster:

    ```bash
    kubectl apply -f infrastructure/deployment/kubernetes.yaml
    ```

4.  **Scale Agent Runners**: Scale the agent runner deployment as needed:

    ```bash
    kubectl scale deployment/fba-bench-agent-runner --replicas=20
    ```

5.  **Monitor**: Use `kubectl get pods`, `kubectl logs`, and K8s dashboard/monitoring tools to observe your distributed simulation.

## Cost Mitigation Strategies

Distributed simulations can incur significant cloud costs. Implement the following:
-   **LLM Batching**: Always enable and tune LLM batching (covered in [`Performance Optimization`](performance-optimization.md)).
-   **Spot Instances/Preemptible VMs**: Utilize cheaper, interruptible compute instances for agent runners if your simulations can tolerate interruptions.
-   **Auto-scaling**: Configure Kubernetes HPA (Horizontal Pod Autoscaler) to scale agent runners up or down based on queue depth or CPU utilization.
-   **Fast-Forwarding**: Use the fast-forward engine to reduce overall simulation time (see [`Fast-Forward Simulation` in `Quick Start`](../quick-start/getting-started.md)).

For more in-depth details on infrastructure components, refer to the [`Infrastructure Scalability Overview`](scalability-overview.md).