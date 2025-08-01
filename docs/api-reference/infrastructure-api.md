# Infrastructure API Reference

This document provides an API reference for FBA-Bench's infrastructure components, including distributed simulation, LLM batching, and performance monitoring. These APIs allow for programmatic control and interaction with the core simulation environment.

## 1. `LLMBatcher`

Manages batching of LLM requests to optimize cost and throughput.

-   **Module**: [`infrastructure/llm_batcher.py`](infrastructure/llm_batcher.py)
-   **Class**: `LLMBatcher`

### Constructor

`__init__(self, llm_client: LLMClient, batch_size: int = 20, batch_interval_seconds: float = 0.05, max_concurrent_batches: int = 5)`

-   **`llm_client`**: (`LLMClient`, required) The underlying LLM client (e.g., `OpenRouterClient`) to which batched requests are sent.
-   **`batch_size`**: (`int`, optional) The maximum number of requests to include in a single batch. Default is `20`.
-   **`batch_interval_seconds`**: (`float`, optional) The maximum time to wait (in seconds) before sending a batch, even if `batch_size` is not met. Default is `0.05`.
-   **`max_concurrent_batches`**: (`int`, optional) The maximum number of batches that can be sent concurrently. Default is `5`.

### Key Methods

#### `add_request(self, prompt_messages: list[dict], callback: Callable)`
Adds an LLM request to the batching queue.

-   **`prompt_messages`**: (`list[dict]`, required) The LLM prompt in the format expected by the underlying `llm_client`.
-   **`callback`**: (`Callable`, required) A callback function `(response: dict)` that will be invoked with the LLM response once the batch is processed.

#### `process_batch(self)`
Manually triggers the processing of the current batch (usually run by an internal timer).

## 2. `DistributedCoordinator`

Orchestrates distributed simulation runs across multiple agent runners.

-   **Module**: [`infrastructure/distributed_coordinator.py`](infrastructure/distributed_coordinator.py)
-   **Class**: `DistributedCoordinator`

### Constructor

`__init__(self, config: dict = None, event_bus: EventBus = None)`

-   **`config`**: (`dict`, optional) Configuration for the coordinator (e.g., resource management, scaling policies).
-   **`event_bus`**: (`EventBus`, optional) The distributed event bus instance for communication.

### Key Methods

#### `start_simulation(self, scenario_config: dict, agent_configs: list[dict])`
Initiates a distributed simulation run.

-   **`scenario_config`**: (`dict`, required) The configuration of the scenario to run.
-   **`agent_configs`**: (`list[dict]`, required) A list of configurations for each agent to be simulated.

#### `register_agent_runner(self, runner_id: str, capabilities: dict)`
Registers a new agent runner instance with the coordinator. (Typically called by the runner itself).

#### `monitor_progress(self) -> dict`
Provides real-time monitoring of distributed simulation progress, resource utilization, and agent status.

## 3. `FastForwardEngine`

Manages rules for accelerating simulation time.

-   **Module**: [`infrastructure/fast_forward_engine.py`](infrastructure/fast_forward_engine.py)
-   **Class**: `FastForwardEngine`

### Constructor

`__init__(self, config: dict = None)`

-   **`config`**: (`dict`, optional) Configuration for fast-forward rules.

### Key Methods

#### `add_rule(self, event_type: Any, condition: Callable, action: Callable)`
Adds a fast-forward rule. When `condition` is met for `event_type`, `action` is executed.

-   **`event_type`**: (`Any`, required) The type of event to monitor (e.g., `SimulationEvent.DAY_START`).
-   **`condition`**: (`Callable`, required) A callable that takes relevant event data and returns `True` if the rule should activate.
-   **`action`**: (`Callable`, required) A callable that describes the fast-forward action (e.g., `self.advance_simulation_time(days)`).

#### `evaluate_and_apply_rules(self, current_event: Event)`
Called by the simulation engine to check if any fast-forward rules apply to the current event.

## 4. `PerformanceMonitor`

Collects and reports key performance metrics.

-   **Module**: [`infrastructure/performance_monitor.py`](infrastructure/performance_monitor.py)
-   **Class**: `PerformanceMonitor`

### Constructor

`__init__(self, config: dict = None)`

-   **`config`**: (`dict`, optional) Configuration for metrics collection frequency, etc.

### Key Methods

#### `record_metric(self, metric_name: str, value: float, tags: dict = None)`
Records a specific performance metric.

-   **`metric_name`**: (`str`, required) Name of the metric (e.g., "llm_token_usage", "agent_cpu_utilization").
-   **`value`**: (`float`, required) The value of the metric.
-   **`tags`**: (`dict`, optional) Key-value pairs for additional context (e.g., `{"agent_id": "Agent1"}`).

#### `get_current_metrics() -> dict`
Retrieves a snapshot of current performance metrics.

#### `generate_report() -> dict`
Generates a summary report of collected performance data.