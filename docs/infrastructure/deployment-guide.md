# Docker and Kubernetes Deployment

This guide provides instructions and best practices for deploying FBA-Bench in containerized environments using Docker and orchestrating them with Kubernetes. Proper deployment ensures scalability, reliability, and efficient resource utilization for large-scale simulations.

## 1. Docker Deployment

Docker is recommended for packaging FBA-Bench components into portable, self-contained units.

### Building Docker Images

FBA-Bench includes Dockerfiles for its core services (e.g., `agent_runner`, `api_server`, shared event bus like Redis). You'll typically find individual Dockerfiles in the root of each major service directory or a central `docker/` directory.

To build a Docker image for a component (e.g., the main FBA-Bench application):

```bash
docker build -t fba-bench-app:latest .
# Or for a specific service, e.g., the agent runner:
docker build -t fba-bench-agent-runner:latest -f infrastructure/deployment/Dockerfile.agent-runner .
```
Replace `fba-bench-app` and `fba-bench-agent-runner` with appropriate service names as defined in your Dockerfiles.

### Local Deployment with Docker Compose

For orchestrating multiple FBA-Bench services locally, `docker-compose.yml` is the primary tool. It defines the services, their dependencies, networking, and scaling.

The main Docker Compose file is located at [`infrastructure/deployment/docker-compose.yml`](infrastructure/deployment/docker-compose.yml).

```yaml
# Simplified example of docker-compose.yml
version: '3.8'
services:
  event_bus:
    image: "redis:6-alpine"
    ports:
      - "6379:6379"
  api_server:
    build:
      context: ../..
      dockerfile: Dockerfile.api-server # Path relative to service context
    ports:
      - "8000:8000"
    depends_on:
      - event_bus
  agent_runner:
    build:
      context: ../..
      dockerfile: Dockerfile.agent-runner
    environment:
      - REDIS_HOST=event_bus
    depends_on:
      - event_bus
      - api_server # If agent runners need to talk to API server
```

To run your multi-service FBA-Bench application locally:

```bash
cd infrastructure/deployment
docker-compose up -d --build --scale agent_runner=5 # -d for detached mode, --build to rebuild images
```
This command will:
-   Build (if `--build` is used) and start an `event_bus` (Redis).
-   Build and start the `api_server`.
-   Build and start 5 instances of the `agent_runner`.

## 2. Kubernetes Deployment

For production deployments, Kubernetes provides advanced orchestration features like self-healing, load balancing, and auto-scaling.

### Kubernetes Manifests

FBA-Bench's Kubernetes deployment manifests are defined in [`infrastructure/deployment/kubernetes.yaml`](infrastructure/deployment/kubernetes.yaml). This file (or a set of files) defines:

-   **Deployments**: For stateless services like `api_server` and `agent_runner`.
    -   Example: `fba-bench-agent-runner` Deployment, scaled using `replicas`.
-   **StatefulSets**: For stateful services like a message queue or database, if they are not external services. (Currently Redis is often used externally or as a simple container in Docker Compose).
-   **Services**: To expose internal services within the cluster or externally (e.g., `NodePort`, `LoadBalancer`).
-   **ConfigMaps**: For injecting configuration data (e.g., LLM API keys, simulation parameters) into pods.
-   **[Optional] Secrets**: For sensitive data like API keys (though often managed by cloud provider secrets managers).
-   **[Optional] Horizontal Pod Autoscalers (HPAs)**: To automatically scale `agent_runner` pods based on CPU utilization or custom metrics.

### Deploying to a Kubernetes Cluster

1.  **Container Registry**: Ensure your Docker images are pushed to a public or private container registry (e.g., Docker Hub, Google Container Registry, Amazon ECR). Update `image` fields in `kubernetes.yaml` accordingly.
2.  **Apply Manifests**: Use `kubectl` to deploy the application:

    ```bash
    kubectl apply -f infrastructure/deployment/kubernetes.yaml
    ```

3.  **Scaling**: Scale the `agent_runner` deployment independently:

    ```bash
    kubectl scale deployment/fba-bench-agent-runner --replicas=20
    ```

4.  **Monitoring and Access**:
    -   `kubectl get pods`: Check the status of your pods.
    -   `kubectl logs <pod-name>`: View logs for a specific pod.
    -   Access the Frontend/API: Depending on your Service type (`NodePort`, `LoadBalancer`), access your FBA-Bench frontend or API from outside the cluster.

## Best Practices for Production Deployment

-   **Resource Limits**: Define CPU and memory limits for your containers to prevent resource exhaustion and ensure stable performance.
-   **Probes**: Implement liveness and readiness probes in your Kubernetes deployments for health checks and graceful restarts.
-   **Centralized Logging and Monitoring**: Integrate with external logging (e.g., ELK stack, Datadog) and monitoring (e.g., Prometheus/Grafana) solutions for production visibility.
-   **CI/CD Integration**: Automate your build, test, and deployment process using CI/CD pipelines.
-   **Security**: Implement network policies, role-based access control (RBAC), and secure handling of secrets.

For more details on `Performance Optimization` and `Monitoring and Alerts`, refer to their dedicated documentation sections.