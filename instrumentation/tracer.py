from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, sampling
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import os

def setup_tracing(service_name="fba-bench"):
    """
    Sets up OpenTelemetry tracing with OTLP exporter.
    """
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.SERVICE_NAMESPACE: "fba-bench",
        ResourceAttributes.SERVICE_INSTANCE_ID: os.getenv("FBA_INSTANCE_ID", "default"),
        "telemetry.sdk.language": "python",
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.version": os.getenv("OTEL_PYTHON_SDK_VERSION", "1.x.x") # Populate with actual version
    })

    # Set up a TracerProvider with a basic sampler
    provider = TracerProvider(resource=resource, sampler=sampling.ALWAYS_ON)
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(__name__)

    # Configure OTLP Exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"), # Default OTLP gRPC endpoint
    )
    span_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(span_processor)

    return tracer