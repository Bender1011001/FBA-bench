# System Monitoring and Performance Tracking

Effective monitoring is crucial for understanding the health, performance, and cost of FBA-Bench simulations, especially in distributed environments. This guide covers the built-in monitoring mechanisms and how to set up alerts for critical events.

## 1. Performance Monitoring

FBA-Bench integrates a `Performance Monitor` (implemented in [`infrastructure/performance_monitor.py`](infrastructure/performance_monitor.py)) that collects key metrics during simulation runs. This includes:

-   **Resource Utilization**: CPU, memory, and network usage per simulation component or agent runner.
-   **LLM Metrics**: Token usage (input/output), API call latency, and cost attributed to LLM interactions.
-   **Simulation Throughput**: Actions per second, simulation days processed per real-time second.
-   **Agent-Specific KPIs**: Tracking of business metrics (e.g., net profit, inventory turnover, customer satisfaction) at granular levels.

These metrics are typically exposed via:
-   **Logger Output**: Console logs provide real-time updates.
-   **Frontend Dashboards**: The FBA-Bench frontend (`frontend/`) visualizes many of these metrics in real-time (see [`frontend/src/components/SystemHealthMonitor.tsx`](frontend/src/components/SystemHealthMonitor.tsx), [`frontend/src/components/KPIDashboard.tsx`](frontend/src/components/KPIDashboard.tsx)).
-   **Exported Reports**: Metrics can be exported to CSV, JSON, or integrated with external monitoring systems.

## 2. Event-Driven Observability

FBA-Bench utilizes an event bus (`event_bus.py`, `distributed_event_bus.py`) which publishes various events, making the system highly observable:
-   **Simulation Events**: Day start/end, quarter end, scenario complete.
-   **Agent Actions**: Price changes, inventory orders, marketing campaigns.
-   **LLM Events**: Request sent, response received, token count.
-   **Error Events**: Detailed error messages with context and suggestions (see [`error_handling.md`](../observability/error-handling.md)).
-   **Performance Events**: Threshold warnings (e.g., high LLM cost, slow response times).

These events can be consumed by internal monitoring components or external systems for real-time analysis and alerting.

## 3. Alerts Configuration

The `Alert System` (implemented in [`observability/alert_system.py`](observability/alert_system.py)) allows you to define thresholds for various metrics or event patterns that, when crossed, trigger notifications.

### Typical Alert Conditions:
-   **High LLM Cost**: Alert if daily LLM spend exceeds a budget.
-   **Low Agent Profitability**: Notify if an agent's net profit drops below a certain threshold.
-   **Excessive API Errors**: Alert if the rate of LLM API errors (e.g., 429 Too Many Requests) surpasses a limit.
-   **Simulation Deadlock/Stall**: Detect if the simulation is not progressing for an unusual period.
-   **Memory Inconsistency**: Alert if memory validation (see [`docs/cognitive-architecture/memory-integration.md`](docs/cognitive-architecture/memory-integration.md)) detects critical issues.

### Configuration

Alerts are configured in `observability_config.yaml`:

```yaml
# Example observability_config.yaml snippet
monitoring:
  enabled: true
  metrics_collection_interval: 10 # seconds

alerts:
  enabled: true
  channels: # Where to send alerts
    - "console"
    - "email" # Requires email server config
    - "slack" # Requires Slack webhook config

  rules:
    - name: "HighLLMCostAlert"
      metric: "llm_daily_cost"
      threshold: 10.00 # USD
      operator: "greater_than"
      severity: "critical"
      notification_interval_minutes: 60

    - name: "AgentProfitDropWarning"
      metric: "agent_net_profit_change_percent"
      threshold: -0.15 # 15% drop
      operator: "less_than"
      severity: "warning"
      notification_interval_minutes: 240

    - name: "APIRateLimitExceeded"
      event_type: "LLMRateLimitExceeded"
      frequency_threshold: 5 # 5 times in 5 minutes
      time_window_minutes: 5
      severity: "major"
      notification_interval_minutes: 15
```

## Integrating with External Dashboards

For advanced visualization and long-term data storage, FBA-Bench can export metrics and traces to external systems like Prometheus, Grafana, ELK Stack (Elasticsearch, Logstash, Kibana), or a custom data warehouse. You would typically use `export_utils.py` ([`instrumentation/export_utils.py`](instrumentation/export_utils.py)) or implement custom data sinks.

For setting up observability dashboards, refer to [`Setting Up and Customizing Dashboards`](../observability/dashboard-setup.md).