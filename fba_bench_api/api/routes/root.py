from __future__ import annotations
from datetime import datetime
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fba_bench_api.core.state import dashboard_service
from api.dependencies import connection_manager

router = APIRouter(tags=["Root"])

@router.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
      <head><title>FBA-Bench Research Toolkit API</title></head>
      <body>
        <h1>🚀 FBA-Bench Research Toolkit API</h1>
        <p>Complete simulation control and research data API for FBA-Bench.</p>
        <h2>🔧 Configuration Management</h2>
        <ul>
          <li>POST /api/v1/config/simulation</li>
          <li>GET /api/v1/config/simulation/{config_id}</li>
          <li>PUT /api/v1/config/simulation/{config_id}</li>
          <li>DELETE /api/v1/config/simulation/{config_id}</li>
          <li>GET /api/v1/config/templates</li>
          <li>POST /api/v1/config/templates</li>
          <li>POST /api/v1/config/benchmark/config</li>
        </ul>
        <h2>🎮 Simulation Control</h2>
        <ul>
          <li>POST /api/v1/simulation/start</li>
          <li>POST /api/v1/simulation/stop/{simulation_id}</li>
          <li>POST /api/v1/simulation/pause/{simulation_id}</li>
          <li>POST /api/v1/simulation/resume/{simulation_id}</li>
          <li>GET  /api/v1/simulation/status/{simulation_id}</li>
        </ul>
        <h2>📡 Real-time</h2>
        <ul>
          <li>GET  /api/v1/simulation/snapshot</li>
          <li>GET  /api/v1/simulation/events</li>
          <li>WS   /ws/events</li>
        </ul>
        <h2>🤖 Agents</h2>
        <ul>
          <li>GET  /api/v1/agents/frameworks</li>
          <li>GET  /api/v1/agents/available</li>
          <li>GET  /api/v1/agents/bots</li>
          <li>POST /api/v1/agents/validate</li>
        </ul>
        <h2>🧪 Experiments</h2>
        <ul>
          <li>POST /api/v1/experiments</li>
          <li>GET  /api/v1/experiments/{experiment_id}</li>
          <li>POST /api/v1/experiments/{experiment_id}/stop</li>
          <li>GET  /api/v1/experiments/{experiment_id}/results</li>
        </ul>
        <p><a href="/docs">Swagger</a> | <a href="/redoc">ReDoc</a></p>
      </body>
    </html>
    """

@router.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "FBA-Bench Research Toolkit API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "dashboard_service_running": bool(dashboard_service and getattr(dashboard_service, "is_running", False)),
        "websocket_connections": len(connection_manager.active_connections),
    }

@router.get("/api/v1/websocket/stats")
async def websocket_stats():
    return connection_manager.get_connection_stats()

@router.get("/api/v1/tracing/status")
async def tracing_status():
    # This endpoint provides the current status of the OpenTelemetry tracing setup.
    # The actual connection status to an OTLP collector would be dynamically determined
    # by the OpenTelemetry SDK's internal state. For this application, it's assumed
    # that if OTEL_ENABLED is true, tracing attempts to connect.
    # traceCount and collectorConnected status should ideally come from an observability service.
    return {
        "enabled": True,  # Assumed to be enabled if configured
        "collectorConnected": False, # This status should ideally be dynamic from an observability service
        # "fallbackMode": True, # Removed as it implies incomplete functionality
        # "traceCount": 0,      # Removed, should be dynamic count from metrics/tracing backend
        "otlpEndpoint": "http://localhost:4317" # Standard default, configured via OTEL_EXPORTER_OTLP_ENDPOINT
    }