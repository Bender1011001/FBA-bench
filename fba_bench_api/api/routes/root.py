from __future__ import annotations
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from fba_bench_api.core.state import dashboard_service
from fba_bench_api.api.dependencies import connection_manager  # fixed import
from fba_bench import __version__, get_build_metadata  # centralized version

router = APIRouter(tags=["Root"])

# Module-level start time for uptime calculation
_START_TIME = time.time()


@router.get("/", response_class=HTMLResponse)
async def root() -> str:
    # Minimal, stable root page with links to documentation UIs
    return """
    <html>
      <head><title>FBA-Bench Research Toolkit API</title></head>
      <body>
        <h1>🚀 FBA-Bench Research Toolkit API</h1>
        <p>Backend for simulations, benchmarking, and research data.</p>
        <p><a href="/docs">Swagger</a> | <a href="/redoc">ReDoc</a></p>
      </body>
    </html>
    """


# Removed: /api/v1/health is now provided as an alias in the main app and delegates
# to the primary /health handler to ensure identical payload and status codes.


@router.get("/api/v1/websocket/stats")
async def websocket_stats() -> Dict[str, Any]:
    return connection_manager.get_connection_stats()


@router.get("/api/v1/tracing/status")
async def tracing_status() -> Dict[str, Any]:
    # Config from env
    otel_enabled = os.getenv("OTEL_ENABLED", "false").lower() in ("1", "true", "yes")
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "") or ""
    debug = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

    if not otel_enabled:
        return {"enabled": False}

    collector_connected = False
    try:
        # Best-effort: if an SDK provider is configured with any span processors, assume connected
        from opentelemetry import trace  # type: ignore

        provider = trace.get_tracer_provider()
        # Heuristic: non-default provider and has active processors/exporters
        if provider is not None and provider.__class__.__name__ != "ProxyTracerProvider":
            # Some providers expose _active_span_processor or similar
            proc = getattr(provider, "active_span_processor", None) or getattr(
                provider, "_active_span_processor", None
            )
            collector_connected = proc is not None
    except Exception:
        collector_connected = False

    return {
        "enabled": True,
        "collectorConnected": bool(collector_connected),
        "endpoint": endpoint if debug else "redacted",
    }