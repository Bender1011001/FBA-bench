# OpenTelemetry Setup Guide for FBA-Bench

This guide explains how to set up OpenTelemetry tracing for the FBA-Bench Research Toolkit.

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. Start the observability stack:
   ```bash
   docker-compose -f docker-compose-otel-collector.yml up -d
   ```

2. Set environment variables for your FBA-Bench application:
   ```bash
   export OTEL_ENABLED=true
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
   ```

3. Run your FBA-Bench application:
   ```bash
   python api_server.py
   ```

4. Access the Jaeger UI at http://localhost:16686 to view traces.

### Option 2: Console Fallback (Development)

If you don't want to set up a collector, you can use the console fallback:

1. Set environment variables:
   ```bash
   export OTEL_ENABLED=true
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317  # This will fail and trigger fallback
   ```

2. Run your application:
   ```bash
   python api_server.py
   ```

3. Traces will be printed to the console instead of being sent to a collector.

### Option 3: Disable Tracing

To disable tracing completely:

1. Set environment variable:
   ```bash
   export OTEL_ENABLED=false
   ```

2. Run your application:
   ```bash
   python api_server.py
   ```

## Dashboard

The FBA-Bench dashboard includes a Tracing Dashboard that shows the current status of OpenTelemetry tracing:

- **Enabled/Disabled**: Shows whether tracing is active
- **Collector Status**: Shows connection to the OTLP collector
- **Fallback Mode**: Indicates if traces are being logged to console
- **Trace Count**: Shows the number of traces collected

The dashboard provides clear status messages and action buttons to help you configure tracing properly.

## Accessing Observability Tools

When using the Docker Compose setup, you can access the following tools:

- **Jaeger UI**: http://localhost:16686
  - View and analyze traces
  - Search by service, operation, or tags
  - Visualize trace timelines

- **Prometheus**: http://localhost:9090
  - Query metrics
  - Create dashboards
  - Set up alerts

- **Grafana**: http://localhost:3000 (admin/admin)
  - Create comprehensive dashboards
  - Visualize metrics and traces
  - Set up alerts and notifications

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OTEL_ENABLED` | Enable or disable tracing | `true` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | `http://localhost:4317` |
| `FBA_INSTANCE_ID` | Instance ID for tracing | `default` |

### Advanced Configuration

For more advanced configuration, you can modify the `setup_tracing` function call in your code:

```python
from instrumentation.tracer import setup_tracing

# Custom configuration
tracer = setup_tracing(
    service_name="my-service",
    enable_tracing=True,
    otlp_endpoint="http://my-collector:4317",
    fallback_to_console=True
)
```

## Troubleshooting

### Common Issues

1. **Connection Refused Errors**
   - Ensure the OTLP collector is running
   - Check the endpoint URL
   - Verify network connectivity

2. **No Traces Appearing**
   - Check if tracing is enabled
   - Verify the collector configuration
   - Check application logs for errors

3. **Dashboard Shows Disconnected**
   - Verify the collector is running
   - Check environment variables
   - Restart the application after changing configuration

### Debug Mode

To enable debug logging for OpenTelemetry:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will provide detailed information about the tracing configuration and any issues.

## Architecture

The tracing system consists of the following components:

1. **Application**: FBA-Bench with OpenTelemetry instrumentation
2. **OTLP Exporter**: Sends traces to the collector
3. **Jaeger Collector**: Receives and stores traces
4. **Jaeger Query**: Provides UI for viewing traces
5. **Dashboard**: Shows tracing status in the FBA-Bench UI

When the collector is unavailable, the system automatically falls back to console logging to ensure trace data is not lost.

## Production Considerations

For production deployments, consider:

1. **Persistent Storage**: Configure Jaeger with persistent storage
2. **Security**: Secure your observability endpoints
3. **Sampling**: Adjust sampling rates for high-traffic environments
4. **Monitoring**: Monitor your observability stack itself

## Customization

You can customize the tracing setup by:

1. Modifying the `setup_tracing` function in `instrumentation/tracer.py`
2. Adding custom exporters
3. Configuring sampling strategies
4. Adding custom attributes to spans

## Support

For issues or questions about OpenTelemetry tracing in FBA-Bench:

1. Check the logs for error messages
2. Verify your configuration
3. Consult the OpenTelemetry documentation
4. Refer to the FBA-Bench documentation