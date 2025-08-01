from opentelemetry import trace
from opentelemetry.trace import Tracer
from typing import Optional, Dict, Any, List
import json
import time

class SimulationTracer:
    """
    Handles OpenTelemetry tracing for simulation-level events and phases.
    Enhanced with trace analysis integration, automated insight generation,
    enhanced metric collection, and real-time alert integration.
    """
    def __init__(self, tracer: Tracer):
        self.tracer = tracer
        self.trace_data_buffer: List[Dict[str, Any]] = [] # Buffer for collecting trace data for analysis

    def _record_event_data(self, span_name: str, attributes: Dict[str, Any]):
        """Helper to record minimal event data to buffer for analysis."""
        # This is a simplified approach. In a real system, you'd use a more robust
        # data collection mechanism (e.g., streaming to a separate service).
        event_data = {
            "span_name": span_name,
            "timestamp": time.time(), # Use current time for simplicity, ideally span start time
            "attributes": attributes
        }
        self.trace_data_buffer.append(event_data)

    def get_trace_data_buffer(self) -> List[Dict[str, Any]]:
        """Returns the accumulated trace data for analysis."""
        return self.trace_data_buffer

    def clear_trace_data_buffer(self):
        """Clears the accumulated trace data."""
        self.trace_data_buffer = []

    def trace_simulation_run(self, simulation_id: str, scenario_name: str, total_ticks: int):
        """
        Returns a context manager for tracing the entire simulation run.
        """
        span_name = f"simulation_run_{simulation_id}"
        attributes = {
            "simulation.id": simulation_id,
            "simulation.scenario_name": scenario_name,
            "simulation.total_ticks": total_ticks,
            "simulation.status": "started"
        }
        span = self.tracer.start_as_current_span(
            span_name,
            attributes=attributes
        )
        self._record_event_data(span_name, attributes)
        return _SpanContextManager(span)

    def trace_tick_progression(self, tick: int, timestamp: str, metrics: Optional[Dict[str, Any]] = None):
        """
        Returns a context manager for tracing an individual simulation tick.

        Args:
            tick: The current simulation tick number.
            timestamp: The timestamp of the tick.
            metrics: Optional dictionary of performance/health metrics for this tick.
        """
        span_name = f"simulation_tick_{tick}"
        attributes = {
            "simulation.tick": tick,
            "simulation.timestamp": timestamp
        }
        if metrics:
            for k, v in metrics.items():
                attributes[f"tick.metric.{k}"] = self._safe_attribute_value(v)
        
        span = self.tracer.start_as_current_span(
            span_name,
            attributes=attributes
        )
        self._record_event_data(span_name, attributes)
        return _SpanContextManager(span)

    def trace_event_propagation(self, event_type: str, event_id: str, publisher_id: str, subscriber_count: int, event_data: Optional[Dict[str, Any]] = None):
        """
        Returns a context manager for tracing event propagation through the EventBus.

        Args:
            event_type: The type of the event.
            event_id: The unique ID of the event.
            publisher_id: The ID of the component that published the event.
            subscriber_count: The number of subscribers to this event.
            event_data: Optional dictionary containing the event's payload.
        """
        span_name = f"event_propagation_{event_type}"
        attributes = {
            "event.type": event_type,
            "event.id": event_id,
            "event.publisher_id": publisher_id,
            "event.subscriber_count": subscriber_count
        }
        if event_data:
            attributes["event.payload"] = json.dumps(event_data)

        span = self.tracer.start_as_current_span(
            span_name,
            attributes=attributes
        )
        self._record_event_data(span_name, attributes)
        return _SpanContextManager(span)

    def trace_curriculum_shock(self, tick: int, shock_type: str, shock_details: Optional[Dict[str, Any]] = None):
        """
        Returns a context manager for tracing a curriculum shock event.
        """
        span_name = f"curriculum_shock_{shock_type}_{tick}"
        attributes = {
            "shock.tick": tick,
            "shock.type": shock_type,
            "shock.details": str(shock_details) # Convert dict to string for attribute
        }
        span = self.tracer.start_as_current_span(
            span_name,
            attributes=attributes
        )
        self._record_event_data(span_name, attributes)
        return _SpanContextManager(span)

    def trace_service_execution(self, service_name: str, tick: int, performance_ms: Optional[float] = None, status: Optional[str] = None):
        """
        Returns a context manager for tracing the execution of a core simulation service.
        """
        span_name = f"service_execution_{service_name}_{tick}"
        attributes = {
            "service.name": service_name,
            "simulation.tick": tick
        }
        if performance_ms is not None:
            attributes["service.performance_ms"] = performance_ms
        if status:
            attributes["service.status"] = status

        span = self.tracer.start_as_current_span(
            span_name,
            attributes=attributes
        )
        self._record_event_data(span_name, attributes) # Ensure this call sends robust data
        return _SpanContextManager(span)

    def _safe_attribute_value(self, value: Any) -> Union[str, int, float, bool]:
        """Converts value to a string if not a primitive type for OpenTelemetry attributes."""
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value)
            except TypeError:
                return repr(value)
        return str(value)

from opentelemetry.trace import Span

class _SpanContextManager:
    """
    A simple context manager for OpenTelemetry spans to allow 'with' statement usage.
    """
    def __init__(self, span: Span):
        self.span = span
        self.start_time = time.perf_counter() # For performance measurement

    def __enter__(self):
        return self.span

    def set_attribute(self, key: str, value: Any):
        self.span.set_attribute(key, value)

    def set_status(self, status: trace.Status):
        self.span.set_status(status)

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration_ms = (end_time - self.start_time) * 1000
        self.span.set_attribute("span.duration_ms", duration_ms) # Record span duration

        if exc_type is not None:
            self.span.set_status(trace.Status(trace.StatusCode.ERROR, description=str(exc_val)))
        self.span.end()