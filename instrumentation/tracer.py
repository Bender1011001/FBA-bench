from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, sampling
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.semconv.resource import ResourceAttributes
import os

def setup_tracing(service_name="fba-bench"):
    """
    Sets up OpenTelemetry tracing with Jaeger exporter.
    """
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: service_name,
        ResourceAttributes.SERVICE_NAMESPACE: "fba-bench",
        ResourceAttributes.SERVICE_INSTANCE_ID: os.getenv("FBA_INSTANCE_ID", "default")
    })

    # Set up a TracerProvider with a basic sampler
    provider = TracerProvider(resource=resource, sampler=sampling.ALWAYS_ON)
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(__name__)

    # Configure Jaeger Exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
        agent_port=int(os.getenv("JAEGER_AGENT_PORT", 6831))
    )
    span_processor = BatchSpanProcessor(jaeger_exporter)
    provider.add_span_processor(span_processor)

    return tracer