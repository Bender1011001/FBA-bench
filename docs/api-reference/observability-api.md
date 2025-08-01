# Observability API Reference

This document provides a comprehensive API reference for FBA-Bench's observability components, including `TraceAnalyzer`, `Error Handling Utilities`, and `Alert System`. These APIs enable programmatic access to simulation traces, error context, and monitoring capabilities.

## 1. `TraceAnalyzer`

Provides utilities for parsing, querying, and extracting insights from simulation trace data.

-   **Module**: [`observability/trace_analyzer.py`](observability/trace_analyzer.py)
-   **Class**: `TraceAnalyzer`

### Constructor

`__init__(self, trace_data: dict)`

-   **`trace_data`**: (`dict`, required) A dictionary containing the full simulation trace data (typically loaded from a JSON file).

### Key Methods

#### `get_agent_steps(self, agent_name: str = None) -> list[dict]`
Retrieves all recorded steps for a specific agent. Each step includes observations, decisions, and outcomes. If `agent_name` is `None`, returns steps for all agents.

-   **`agent_name`**: (`str`, optional) The name of the agent.
-   **Returns**: `list[dict]` - A list of agent step dictionaries.

#### `get_llm_calls_for_agent(self, agent_name: str) -> list[dict]`
Extracts all LLM interaction records for a given agent.

-   **`agent_name`**: (`str`, required) The name of the agent.
-   **Returns**: `list[dict]` - A list of LLM call dictionaries, including prompt, response, tokens, and latency.

#### `get_tool_invocations(self, agent_name: str = None, tool_name: str = None) -> list[dict]`
Retrieves records of tool invocations. Can be filtered by agent or specific tool name.

-   **`agent_name`**: (`str`, optional)
-   **`tool_name`**: (`str`, optional)
-   **Returns**: `list[dict]` - A list of tool invocation dictionaries.

#### `get_events_by_type(self, event_type: str) -> list[Event]`
Filters and returns simulation events of a specific type.

-   **`event_type`**: (`str`, required) The type of event (e.g., "DEMAND_SPIKE", "PRICE_UPDATE").
-   **Returns**: `list[Event]` - A list of `Event` objects.

#### `find_decision_points(self, agent_name: str = None) -> list[dict]`
Identifies and returns key decision points in the simulation trace.

-   **`agent_name`**: (`str`, optional)
-   **Returns**: `list[dict]` - A list of dictionaries representing decision points.

#### `generate_summary_report(self) -> dict`
Generates a high-level summary report of the simulation trace, including overall metrics and statistics.

-   **Returns**: `dict` - A summary report.

## 2. Error Handling Utilities

FBA-Bench's error handling provides structured error messages for easier debugging and comprehension.

-   **Module**: Frontend (`frontend/src/utils/errorHandler.ts`) and Backend Event Definitions (`events.py`, specifically `ErrorEvent`).

### Backend Error Event (`events.py`)

-   **Class**: `ErrorEvent` (inherits from `Event`)

#### `__init__(self, message: str, context: dict = None, suggestion: str = None, severity: str = "error")`

-   **`message`**: (`str`, required) A concise summary of the error.
-   **`context`**: (`dict`, optional) A dictionary providing additional contextual information (e.g., component, relevant data).
-   **`suggestion`**: (`str`, optional) A human-readable suggestion for how to resolve the error.
-   **`severity`**: (`str`, optional) The severity level ("info", "warning", "error", "critical"). Default is "error".

### Frontend `errorHandler.ts`

This module centralizes error reporting and display logic in the frontend.

#### `handleError(error: any, context: string = "")`
Processes an error, logs it, and potentially displays it to the user.

-   **`error`**: (`any`, required) The error object.
-   **`context`**: (`string`, optional) Additional context message.

#### `displayNotification(message: string, type: 'success' | 'info' | 'warning' | 'error')`
Shows a user-facing notification.

## 3. `AlertSystem`

Manages configured alerts based on metrics and events.

-   **Module**: [`observability/alert_system.py`](observability/alert_system.py)
-   **Class**: `AlertSystem`

### Constructor

`__init__(self, config: dict, event_bus: EventBus)`

-   **`config`**: (`dict`, required) Alert rules and notification channels configuration.
-   **`event_bus`**: (`EventBus`, required) The event bus to subscribe to events for alert evaluation.

### Key Methods

#### `register_alert_rule(self, rule_name: str, metric_or_event: str, threshold: float, operator: str, severity: str, notification_interval_minutes: int, additional_params: dict = None)`
Dynamically registers an alert rule.

-   **`rule_name`**: (`str`, required) Unique name for the alert rule.
-   **`metric_or_event`**: (`str`, required) The name of the metric or event type to monitor.
-   **`threshold`**: (`float`, required) The value that triggers the alert.
-   **`operator`**: (`str`, required) Comparison operator (e.g., "greater_than", "less_than", "equals").
-   **`severity`**: (`str`, required) "info", "warning", "error", "critical".
-   **`notification_interval_minutes`**: (`int`, required) Minimum time between notifications for the same alert.
-   **`additional_params`**: (`dict`, optional) Extra parameters for specific rule types (e.g., `frequency_threshold`, `time_window_minutes` for event-based alerts).

#### `evaluate_metrics(self, current_metrics: dict)`
Evaluates current metrics against defined alert rules. (Called periodically by `PerformanceMonitor`).

#### `process_event(self, event: Event)`
Processes an incoming event against event-based alert rules. (Subscribes to `EventBus`).

#### `send_notification(self, alert_message: str, channel: str, severity: str)`
Sends out notifications through configured channels. (Internal utility).