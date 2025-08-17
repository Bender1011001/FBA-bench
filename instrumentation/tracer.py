from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, sampling
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter, SpanExportResult
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class FallbackSpanExporter(SpanExporter):
    """
    A span exporter that falls back to console logging when OTLP exporter fails.
    """
    
    def __init__(self, otlp_endpoint=None, fallback_to_console=True):
        self.otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        self.fallback_to_console = fallback_to_console
        self.otlp_exporter = None
        self.console_exporter = ConsoleSpanExporter()
        self.otlp_failed = False
        
        # Try to initialize OTLP exporter
        try:
            self.otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
            logger.info(f"OTLP exporter initialized with endpoint: {self.otlp_endpoint}")
        except Exception as e:
            logger.warning(f"Failed to initialize OTLP exporter: {e}")
            self.otlp_failed = True
    
    def export(self, spans):
        """
        Export spans to OTLP collector with fallback to console.
        """
        if not self.otlp_failed and self.otlp_exporter:
            try:
                return self.otlp_exporter.export(spans)
            except Exception as e:
                logger.warning(f"OTLP export failed: {e}")
                self.otlp_failed = True
        
        # Fallback to console if OTLP failed
        if self.fallback_to_console:
            return self.console_exporter.export(spans)
        
        return SpanExportResult.SUCCESS
    
    def shutdown(self):
        """
        Shutdown the exporter.
        """
        if self.otlp_exporter:
            try:
                self.otlp_exporter.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down OTLP exporter: {e}")
        
        if self.console_exporter:
            try:
                self.console_exporter.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down console exporter: {e}")

def setup_tracing(service_name="fba-bench", enable_tracing=None, otlp_endpoint=None, fallback_to_console=True):
    """
    Sets up OpenTelemetry tracing with OTLP exporter and graceful fallback.
    
    Args:
        service_name: Name of the service being traced
        enable_tracing: Override to enable/disable tracing. If None, uses OTEL_ENABLED env var
        otlp_endpoint: OTLP endpoint URL. If None, uses OTEL_EXPORTER_OTLP_ENDPOINT env var
        fallback_to_console: Whether to fall back to console exporter if OTLP fails
    
    Returns:
        OpenTelemetry tracer instance or None if tracing is disabled
    """
    # Check if tracing is enabled
    if enable_tracing is None:
        enable_tracing = os.getenv("OTEL_ENABLED", "true").lower() == "true"
    
    if not enable_tracing:
        logger.info("OpenTelemetry tracing is disabled")
        return None
    
    # Set up resource attributes
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.SERVICE_NAMESPACE: "fba-bench",
        ResourceAttributes.SERVICE_INSTANCE_ID: os.getenv("FBA_INSTANCE_ID", "default"),
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.version": os.getenv("OTEL_PYTHON_SDK_VERSION", "1.x.x")
    })

    # Check if a TracerProvider is already set
    if trace.get_tracer_provider() is not None:
        logger.info("TracerProvider already exists, using existing one")
        return trace.get_tracer(__name__)
    
    # Set up a TracerProvider with a basic sampler
    provider = TracerProvider(resource=resource, sampler=sampling.ALWAYS_ON)
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(__name__)
    
    # Get OTLP endpoint from parameter or environment variable
    if otlp_endpoint is None:
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    
    # Set up the fallback exporter
    fallback_exporter = FallbackSpanExporter(
        otlp_endpoint=otlp_endpoint,
        fallback_to_console=fallback_to_console
    )
    span_processor = BatchSpanProcessor(fallback_exporter)
    provider.add_span_processor(span_processor)
    logger.info(f"OpenTelemetry tracing configured with OTLP endpoint: {otlp_endpoint}")
    
    return tracer