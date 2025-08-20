import json
import os
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import format_span_id, format_trace_id, SpanKind
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.attributes import BoundedAttributes

class ChromeTracingExporter(SpanExporter):
    """
    OpenTelemetry SpanExporter that exports spans to Chrome DevTools trace format.
    """

    def __init__(self):
        self.trace_events = []

    def export(self, spans) -> SpanExportResult:
        for span in spans:
            # Chrome tracing expects timestamps in microseconds
            start_time_us = span.start_time // 1000
            end_time_us = span.end_time // 1000
            duration_us = end_time_us - start_time_us

            # Process ID (pid) and Thread ID (tid)
            # Use OS pid and span-based stable tid
            pid = os.getpid()
            tid = int(format_span_id(span.context.span_id), 16) % 1000000  # stable per-span thread id surrogate

            event = {
                "name": span.name,
                "cat": self._get_category(span),
                "ph": "X",  # Complete event (ph: phase, X: complete event)
                "ts": start_time_us,
                "dur": duration_us,
                "pid": pid,
                "tid": tid,
                "args": self._get_attributes(span.attributes)
            }
            self.trace_events.append(event)
        return SpanExportResult.SUCCESS

    def _get_category(self, span) -> str:
        # Categorize spans based on name or attributes
        if span.name.startswith("agent_turn"):
            return "agent"
        elif span.name.startswith("simulation_tick"):
            return "simulation"
        elif span.name.startswith("observe"):
            return "agent.observe"
        elif span.name.startswith("think"):
            return "agent.think"
        elif "tool_call" in span.name:
            return "agent.tool_call"
        elif "event_propagation" in span.name or "service_execution" in span.name:
            return "system"
        return "default"

    def _get_attributes(self, attributes: BoundedAttributes) -> dict:
        # Convert span attributes to a dictionary suitable for Chrome tracing args
        args = {}
        if attributes:
            for key, value in attributes.items():
                # Simple conversion for now, flatten complex objects if necessary
                args[key] = str(value) if not isinstance(value, (str, int, float, bool)) else value
        return args

    def get_chrome_trace_format(self) -> dict:
        return {"traceEvents": self.trace_events, "displayTimeUnit": "ns"} # "ns", "ms", "us"

    def shutdown(self):
        self.trace_events = [] # Clear events on shutdown

def export_spans_to_chrome_json(spans) -> str:
    """
    Utility function to convert a list of OpenTelemetry spans directly to Chrome DevTools JSON.
    This is useful for ad-hoc exports without setting up a full exporter pipeline.
    """
    exporter = ChromeTracingExporter()
    exporter.export(spans) # This will add spans to exporter.trace_events
    return json.dumps(exporter.get_chrome_trace_format(), indent=2)

# --- Integration Example for setup_tracing in tracer.py ---

class ExportUtils:
    """
    Thin utility wrapper expected by some tests. Provides static helpers
    for exporting spans to Chrome trace JSON format.
    """
    @staticmethod
    def to_chrome_trace(spans) -> str:
        return export_spans_to_chrome_json(spans)
# To use this in setup_tracing, you'd add:
# from instrumentation.export_utils import ChromeTracingExporter
# chrome_exporter = ChromeTracingExporter()
# provider.add_span_processor(BatchSpanProcessor(chrome_exporter))
# You'd then need a way to retrieve the trace_events from the chrome_exporter instance.
# This often involves a custom global exporter or a way to get the exporter from the provider.
# For simplicity, for now, we'll provide a direct utility function.