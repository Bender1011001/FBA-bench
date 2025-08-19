# Observability Overview: Tool Interfaces and Monitoring Capabilities

FBA-Bench's enhanced observability suite provides comprehensive tools and interfaces for monitoring, debugging, and analyzing the behavior of agents and the simulation environment. This is crucial for understanding `why` agents make certain decisions, identifying performance bottlenecks, and validating experimental results.

## Key Observability Features

-   **LLM-Friendly APIs**: Structured API responses designed to be easily consumed and interpreted by LLM agents, enhancing their ability to interact with tools and understand simulation data.
-   **Robust Error Handling**: Provides detailed error messages with contextual information and explicit suggestions for troubleshooting, significantly accelerating debugging.
-   **Advanced Trace Analysis**: Captures a rich, granular history of agent thoughts, actions, and system events, enabling post-simulation introspection and root cause analysis.
-   **Real-time Observability Dashboards**: A web-based frontend for visualizing key performance indicators (KPIs), agent states, event logs, and system health in real-time.

## Design Principles

### Comprehensive Instrumentation
Nearly every significant operation within FBA-Bench (agent decision, LLM call, event publication, metric update) is instrumented to produce observable data.

### Granular Traceability
The system generates highly detailed traces, allowing users to drill down from high-level simulation outcomes to individual agent thoughts and tool invocations.

### Actionable Feedback
Errors and warnings are designed to provide direct, actionable guidance, helping users and developers quickly resolve issues.

### User-Friendly Interfaces
While robust under the hood, the observability tools prioritize user experience, presenting complex data in intuitive dashboards and human-readable formats.

### Extensibility
The event-driven architecture and modular design allow for easy integration with external monitoring, logging, and analytics platforms.

## Core Observability Components

-   **Agent Tracer (`instrumentation/agent_tracer.py`)**: Records detailed internal states, thoughts, and decisions of individual agents.
-   **Simulation Tracer (`instrumentation/simulation_tracer.py`)**: Captures global simulation events, marketplace changes, and interactions between components.
-   **Trace Analyzer (`observability/trace_analyzer.py`)**: Provides programmatic utilities to parse and extract insights from recorded traces.
-   **Error Handler (`frontend/src/utils/errorHandler.ts`, and backend components)**: Standardizes error reporting, adds context, and suggests fixes. Also visible in [`observability/error_handling.md`](error-handling.md).
-   **Alert System (`observability/alert_system.py`)**: Configurable system for defining and triggering alerts based on monitored metrics or events (see [`monitoring-and-alerts.md`](../infrastructure/monitoring-and-alerts.md)).
-   **API Server (`api_server.py`)**: Exposes simulation data and control endpoints for the frontend and external tools.
-   **Frontend Dashboard Components (`frontend/src/components/observability/`)**:
    -   [`AgentInsightsDashboard.tsx`](frontend/src/components/observability/AgentInsightsDashboard.tsx): Visualizes agent-specific performance and behavior.
    -   [`SimulationHealthMonitor.tsx`](frontend/src/components/observability/SimulationHealthMonitor.tsx): Displays overall simulation health and resource usage.
    -   [`ToolUsageAnalyzer.tsx`](frontend/src/components/observability/ToolUsageAnalyzer.tsx): Tracks how agents utilize their available tools.

## Integration with External Tools

FBA-Bench's observability data can be integrated with popular external tools for advanced analysis and visualization:
-   **Logging Platforms**: ELK Stack (Elasticsearch, Logstash, Kibana), Splunk, Datadog.
-   **Monitoring Systems**: Prometheus + Grafana.
-   **Tracing Systems**: OpenTelemetry-compatible platforms.
-   **Custom Data Warehouses**: Using `instrumentation/export_utils.py` to push data.

For detailed instructions on setting up dashboards, performing trace analysis, handling errors, and configuring alerts, refer to:
-   [`Trace Analysis`](trace-analysis.md)
-   [`Error Handling`](error-handling.md)
-   [`Dashboard Setup`](dashboard-setup.md)
-   [`Alert Configuration`](alert-configuration.md)
-   [`OpenTelemetry Setup`](opentelemetry-setup.md)

## Tracing Status Endpoint Behavior

The tracing status endpoint follows these policies:
- Exposure is governed by the OTEL_ENABLED environment variable. When OTEL_ENABLED is not enabled, tracing is effectively disabled and the status endpoint reports tracing as inactive.
- collectorConnected reflects best-effort connectivity: it may be true when a collector is reachable, false when not, and unknown if connectivity cannot be determined reliably at runtime.
- The configured OTLP endpoint is redacted by default for safety. It is only shown in full when DEBUG is true; otherwise a masked/redacted value is returned.