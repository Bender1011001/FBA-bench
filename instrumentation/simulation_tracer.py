from opentelemetry import trace
from opentelemetry.trace import Tracer
from typing import Optional, Dict, Any

class SimulationTracer:
    """
    Handles OpenTelemetry tracing for simulation-level events and phases.
    """
    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    def trace_simulation_run(self, simulation_id: str, scenario_name: str, total_ticks: int):
        """
        Returns a context manager for tracing the entire simulation run.
        """
        span = self.tracer.start_as_current_span(
            f"simulation_run_{simulation_id}",
            attributes={
                "simulation.id": simulation_id,
                "simulation.scenario_name": scenario_name,
                "simulation.total_ticks": total_ticks,
                "simulation.status": "started"
            }
        )
        return _SpanContextManager(span)

    def trace_tick_progression(self, tick: int, timestamp: str):
        """
        Returns a context manager for tracing an individual simulation tick.
        """
        span = self.tracer.start_as_current_span(
            f"simulation_tick_{tick}",
            attributes={
                "simulation.tick": tick,
                "simulation.timestamp": timestamp
            }
        )
        return _SpanContextManager(span)

    def trace_event_propagation(self, event_type: str, event_id: str, publisher_id: str, subscriber_count: int):
        """
        Returns a context manager for tracing event propagation through the EventBus.
        """
        span = self.tracer.start_as_current_span(
            f"event_propagation_{event_type}",
            attributes={
                "event.type": event_type,
                "event.id": event_id,
                "event.publisher_id": publisher_id,
                "event.subscriber_count": subscriber_count
            }
        )
        return _SpanContextManager(span)

    def trace_curriculum_shock(self, tick: int, shock_type: str, shock_details: Optional[Dict[str, Any]] = None):
        """
        Returns a context manager for tracing a curriculum shock event.
        """
        span = self.tracer.start_as_current_span(
            f"curriculum_shock_{shock_type}_{tick}",
            attributes={
                "shock.tick": tick,
                "shock.type": shock_type,
                "shock.details": str(shock_details) # Convert dict to string for attribute
            }
        )
        return _SpanContextManager(span)

    def trace_service_execution(self, service_name: str, tick: int):
        """
        Returns a context manager for tracing the execution of a core simulation service.
        """
        span = self.tracer.start_as_current_span(
            f"service_execution_{service_name}_{tick}",
            attributes={
                "service.name": service_name,
                "simulation.tick": tick
            }
        )
        return _SpanContextManager(span)

from opentelemetry.trace import Span

class _SpanContextManager:
    """
    A simple context manager for OpenTelemetry spans to allow 'with' statement usage.
    """
    def __init__(self, span: Span):
        self.span = span

    def __enter__(self):
        return self.span

    def set_attribute(self, key: str, value: Any):
        self.span.set_attribute(key, value)

    def set_status(self, status: trace.Status):
        self.span.set_status(status)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.span.set_status(trace.Status(trace.StatusCode.ERROR, description=str(exc_val)))
        self.span.end()