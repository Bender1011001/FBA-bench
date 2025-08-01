# Tutorial 5: Setting Up Monitoring and Analysis

This tutorial guides you through setting up FBA-Bench's observability features, including real-time dashboards, trace analysis, and error handling for enhanced monitoring.

## Real-time Observability Dashboards

FBA-Bench provides a frontend application to visualize simulation progress, agent metrics, and system health in real-time.

### Running the Frontend Dashboard

Navigate to the `frontend/` directory and install dependencies if you haven't already:

```bash
cd frontend
npm install
npm run dev
```

This will start the development server for the frontend, typically accessible at `http://localhost:5173`. Ensure your `api_server.py` is also running for data to be displayed.

## Advanced Trace Analysis

FBA-Bench captures detailed traces of agent actions, decisions, and system events. The `TraceViewer` and `AgentInsightsDashboard` components in the frontend, along with backend [`instrumentation/trace_analyzer.py`](instrumentation/trace_analyzer.py) utilities, provide deep insights.

To analyze a trace programmatically:

```python
# tutorial_trace_analysis.py
from fba_bench.instrumentation.trace_analyzer import TraceAnalyzer
from fba_bench.instrumentation.export_utils import load_trace_from_file

# Assuming a trace file (e.g., from a completed simulation) is saved in artifacts/
trace_file_path = "artifacts/simulation_trace_example.json"
trace_data = load_trace_from_file(trace_file_path)

analyzer = TraceAnalyzer(trace_data)

# Example: Get all LLM calls made by a specific agent
llm_calls = analyzer.get_llm_calls_for_agent("MyEnhancedAgent")
print(f"Total LLM calls for MyEnhancedAgent: {len(llm_calls)}")

# Example: Identify critical decision points
decision_points = analyzer.find_decision_points()
print(f"Identified {len(decision_points)} critical decision points.")

# Example: Generate a summary report
summary_report = analyzer.generate_summary_report()
print("\nSimulation Summary:")
for key, value in summary_report.items():
    print(f"- {key}: {value}")

# Further analysis can be done using the TraceAnalyzer methods.
```

## Robust Error Handling with Educational Feedback

FBA-Bench's error handling provides detailed context and corrective suggestions, accelerating debugging and understanding. This is visible both in the console output and in the observability dashboards.

Log and error events are captured and can be viewed via the `EventLog` in the frontend or programmatically:

```python
# Assuming you have an EventLog instance configured to capture events
from fba_bench.events import EventBus, ErrorEvent, LogEvent

event_bus = EventBus()

def error_listener(event):
    if isinstance(event, ErrorEvent):
        print(f"ERROR: {event.message} - Context: {event.context}")
        print(f"Suggested fix: {event.suggestion}")

event_bus.subscribe(ErrorEvent, error_listener)

# Simulate an error
# event_bus.publish(ErrorEvent(
#     message="Invalid API key provided for LLM client.",
#     context={"component": "LLMClient", "api_call": "/v1/chat/completions"},
#     suggestion="Verify your LLM_API_KEY environment variable."
# ))
```

For more detailed information on observability features, dashboard customization, and alert configuration, refer to the [`docs/observability/`](docs/observability/) documentation.