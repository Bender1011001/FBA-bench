# Infrastructure Scalability Overview

FBA-Bench's enhanced infrastructure is designed for robust scalability, enabling the execution of large-scale simulations with many concurrent agents and long durations without prohibitive costs or performance bottlenecks. This overview describes the architectural components and design principles that ensure highthroughput and cost-efficiency.

## Key Scalability Features

-   **Distributed Simulation**: Ability to run multiple agents and simulation components across various processes, machines, or cloud instances.
-   **LLM Request Batching**: Optimizes interactions with Large Language Models by grouping multiple API calls, significantly reducing latency and cost.
-   **Fast-Forward Simulation**: Accelerates long-duration scenarios by intelligently skipping less critical simulation steps.
-   **Performance Monitoring and Optimization**: Integrated tools for observing system health, identifying bottlenecks, and fine-tuning resource allocation.

## Architectural Principles

### Microservices-Oriented
FBA-Bench's components are loosely coupled microservices (e.g., agent runners, scenario engine, event bus, LLM clients). Each component can be scaled independently based on demand.

### Asynchronous Communication
Components communicate primarily via an event bus (e.g., [`event_bus.py`](event_bus.py) and [`infrastructure/distributed_event_bus.py`](infrastructure/distributed_event_bus.py)), which supports asynchronous messaging, allowing for non-blocking operations and high concurrency.

### Statelessness (where possible)
Many components are designed to be largely stateless, simplifying scaling and fault tolerance. State management is centralized or handled by persistent storage solutions where necessary.

### Caching
Aggressive caching strategies are employed, particularly for LLM responses (see [`reproducibility/llm_cache.py`](reproducibility/llm_cache.py)), to minimize redundant computations and external API calls.

### Configuration-Driven
Scalability parameters (e.g., number of concurrent agents, batch sizes, fast-forward rules) are externally configurable, allowing users to adapt the system to their specific hardware and simulation needs.

## Core Infrastructure Components

-   **Distributed Coordinator (`infrastructure/distributed_coordinator.py`)**: Manages the orchestration of distributed simulation runs, including agent assignment to runners and overall progress tracking.
-   **LLM Batcher (`infrastructure/llm_batcher.py`)**: Aggregates LLM requests before sending them to the LLM client, reducing the number of individual API calls.
-   **Fast-Forward Engine (`infrastructure/fast_forward_engine.py`)**: Implements rules for accelerating simulation time by skipping or compressing periods.
-   **Performance Monitor (`infrastructure/performance_monitor.py`)**: Collects and reports metrics on resource utilization, LLM token usage, and simulation throughput.
-   **Resource Manager (`infrastructure/resource_manager.py`)**: Handles the allocation and deallocation of computational resources for distributed components.
-   **Scalability Configuration (`infrastructure/scalability_config.py`)**: Defines the schema and manages parameters for tuning infrastructure performance.

## Deployment Options

FBA-Bench can be deployed using standard containerization and orchestration technologies:
-   **Docker Compose**: For local multi-service deployments (see [`infrastructure/deployment/docker-compose.yml`](infrastructure/deployment/docker-compose.yml)).
-   **Kubernetes**: For production-grade, highly scalable deployments in cloud environments (see [`infrastructure/deployment/kubernetes.yaml`](infrastructure/deployment/kubernetes.yaml)).

For more detailed information on configuring and deploying scalable simulations, refer to:
- [`Distributed Simulation`](distributed-simulation.md)
- [`Performance Optimization`](performance-optimization.md) (focusing on LLM batching and cost)
- [`Monitoring and Alerts`](monitoring-and-alerts.md)
- [`Deployment Guide`](deployment-guide.md)
- [`Infrastructure Configuration Guide`](../configuration/infrastructure-config.md)

## SPA Deployment Behind Nginx (High-Level Notes)

- Build the frontend SPA and deploy the generated assets to a static root such as /usr/share/nginx/html (or your configured path).
- Enable long-term caching for hashed static assets (e.g., *.css, *.js) with Cache-Control: public, max-age=31536000, immutable.
- Serve index.html with no-store/no-cache headers to ensure clients always fetch the latest application shell.
- Proxy API requests to the FastAPI backend and support WebSocket upgrades. See nginx.conf in the repository root for a reference configuration, including:
  - location /api/ proxy settings
  - proxy_set_header Upgrade and Connection for WebSocket routes
  - appropriate timeouts and buffering for streaming endpoints