# Infrastructure System Configuration

This document provides a detailed reference for configuring FBA-Bench's infrastructure components, including settings for distributed simulation, LLM batching, and performance monitoring. These configurations are crucial for scaling your simulations and managing operational costs.

## Configuration File Location

Infrastructure configurations are primarily loaded from YAML files. The main configuration schema is defined by the `ScalabilityConfig` class in [`infrastructure/scalability_config.py`](infrastructure/scalability_config.py).

## Root-Level Parameters

```yaml
infrastructure_system:
  enabled: true # Master switch for advanced infrastructure features

  # Nested configurations for sub-systems
  # distributed_simulation: {...}
  # llm_orchestration: {...}
  # resource_management: {...}
  # performance_monitoring: {...}
```

-   **`enabled`**: (`boolean`, default: `true`)
    -   If `false`, global infrastructure enhancements (like distributed coordination or LLM batching) will be bypassed, reverting to local, single-process execution where applicable.

## `distributed_simulation` Parameters

Controls how simulations are distributed across multiple processes or machines. See [`Distributed Simulation`](../infrastructure/distributed-simulation.md) for more details.

```yaml
distributed_simulation:
  enabled: true
  event_bus_type: "redis" # "local_memory", "redis", "kafka"
  redis_host: "localhost"
  redis_port: 6379
  agent_runner_concurrency: 5 # Number of agents an individual runner can process concurrently
  orchestrator_mode: "auto" # "auto", "manual"
```

-   **`enabled`**: (`boolean`, default: `true`) Activates/deactivates distributed simulation.
-   **`event_bus_type`**: (`string`, default: `"redis"`) The type of message bus to use for inter-component communication.
    -   Valid options: `"local_memory"` (for single-process, debugging), `"redis"`, `"kafka"`.
-   **`redis_host`**: (`str`, default: `"localhost"`) Host for Redis event bus.
-   **`redis_port`**: (`int`, default: `6379`) Port for Redis event bus.
-   **`agent_runner_concurrency`**: (`integer`, default: `5`) The number of simulation instances (agents) that a *single* agent runner process can execute in parallel. This is distinct from the total number of agent runner processes.
-   **`orchestrator_mode`**: (`string`, default: `"auto"`) Determines how the distributed coordination is managed.
    -   `"auto"`: The `DistributedCoordinator` automatically manages agent assignment and resource allocation.
    -   `"manual"`: Requires explicit API calls to the `DistributedCoordinator` to assign agents.

## `llm_orchestration` Parameters

Manages LLM interaction for cost and performance optimization. See [`Performance Optimization`](../infrastructure/performance-optimization.md) for more details.

```yaml
llm_orchestration:
  batching:
    enabled: true
    batch_size: 50
    batch_interval_seconds: 0.1
    max_concurrent_batches: 5
  caching:
    enabled: true
    cache_backend: "sqlite" # "sqlite", "redis", "memory"
    cache_path: "./llm_cache.db"
    cache_ttl_seconds: 86400 # 24 hours
```

-   **`batching.enabled`**: (`boolean`, default: `true`) Activates/deactivates LLM request batching.
-   **`batching.batch_size`**: (`integer`, default: `50`) Number of LLM requests to group into a single batch.
-   **`batching.batch_interval_seconds`**: (`float`, default: `0.1`) Maximum time (in seconds) to wait before sending a batch.
-   **`batching.max_concurrent_batches`**: (`integer`, default: `5`) Maximum number of batches that can be sent concurrently.
-   **`caching.enabled`**: (`boolean`, default: `true`) Activates/deactivates LLM response caching.
-   **`caching.cache_backend`**: (`string`, default: `"sqlite"`) The storage backend for the LLM cache. Valid options: `"sqlite"`, `"redis"`, `"memory"`.
-   **`caching.cache_path`**: (`str`, default: `"./llm_cache.db"`) File path for SQLite cache.
-   **`caching.cache_ttl_seconds`**: (`integer`, default: `86400`) Time-to-live for cache entries in seconds.

## `resource_management` Parameters

Controls how simulation resources are managed and allocated.

```yaml
resource_management:
  auto_scaling_enabled: true
  min_agent_runners: 1
  max_agent_runners: 10
  cpu_threshold_percent: 80 # Trigger scale-up if average CPU across runners exceeds this
  memory_threshold_percent: 75
```

-   **`auto_scaling_enabled`**: (`boolean`, default: `true`) Enables automatic scaling of agent runners (requires a compatible orchestrator like Kubernetes HPA).
-   **`min_agent_runners`**: (`integer`, default: `1`) Minimum number of agent runner instances to maintain.
-   **`max_agent_runners`**: (`integer`, default: `10`) Maximum number of agent runner instances allowed.
-   **`cpu_threshold_percent`**: (`integer`, default: `80`) CPU utilization percentage that triggers a scale-up event.
-   **`memory_threshold_percent`**: (`integer`, default: `75`) Memory utilization percentage that triggers a scale-up event.

## `performance_monitoring` Parameters

Configures the collection and reporting of performance metrics. See [`Monitoring and Alerts`](../infrastructure/monitoring-and-alerts.md) for more details.

```yaml
performance_monitoring:
  enabled: true
  collection_interval_seconds: 5 # How often to collect metrics
  export_formats: ["json", "csv"] # To what formats to export reports
  export_path: "./performance_logs/"
  integrations: # External monitoring system integrations
    grafana_prometheus:
      enabled: false
      endpoint: "http://localhost:9090"
```

-   **`enabled`**: (`boolean`, default: `true`) Activates/deactivates performance data collection.
-   **`collection_interval_seconds`**: (`integer`, default: `5`) Frequency (in seconds) at which performance metrics are sampled.
-   **`export_formats`**: (`list[str]`, default: `["json"]`) Formats for exporting collected performance data. Valid options: `"json"`, `"csv"`, `"none"`.
-   **`export_path`**: (`str`, default: `"./performance_logs/"`) Directory for saving performance logs.
-   **`integrations`**: (`dict`, optional) Configuration for external monitoring system integrations.

## Example Usage

To load and use a custom infrastructure configuration:

```python
from fba_bench.infrastructure.scalability_config import ScalabilityConfig
from fba_bench.infrastructure.distributed_coordinator import DistributedCoordinator

# Load your custom infrastructure configuration file
custom_config_path = "path/to/your/custom_infrastructure_config.yaml"
infrastructure_config = ScalabilityConfig.from_yaml(custom_config_path)

# Initialize components with the custom configuration
coordinator = DistributedCoordinator(config=infrastructure_config.distributed_simulation)
# llm_batcher = LLMBatcher(my_llm_client, **infrastructure_config.llm_orchestration.batching)
# performance_monitor = PerformanceMonitor(infrastructure_config.performance_monitoring)

# Then run your distributed simulation
# coordinator.start_simulation(...)