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


@router.get("/api/v1/health")
async def health_check() -> Dict[str, Any]:
    # Base metadata
    ts = datetime.now(tz=timezone.utc).isoformat()
    uptime = max(0.0, time.time() - _START_TIME) if _START_TIME else 0.0
    build = get_build_metadata()
    env_name = os.getenv("ENV_NAME", "unknown")

    # Optional dependency checks gated by env flags
    redis_ok = None
    db_ok = None

    try:
        if os.getenv("CHECK_REDIS", "0") == "1":
            redis_url = (
                os.getenv("REDIS_URL")
                or os.getenv("FBA_BENCH_REDIS_URL")
                or os.getenv("FBA_REDIS_URL")
            )
            if redis_url:
                try:
                    import redis  # type: ignore

                    client = redis.from_url(redis_url)
                    client.ping()
                    redis_ok = True
                except Exception:
                    redis_ok = False
    except Exception:
        # Never raise from health endpoint
        redis_ok = False

    try:
        if os.getenv("CHECK_DB", "0") == "1":
            db_url = os.getenv("DATABASE_URL") or os.getenv("FBA_BENCH_DB_URL")
            if db_url:
                try:
                    from sqlalchemy import create_engine  # type: ignore

                    engine = create_engine(db_url, pool_pre_ping=True)
                    with engine.connect() as conn:
                        conn.execute("SELECT 1")
                    db_ok = True
                except Exception:
                    db_ok = False
    except Exception:
        db_ok = False

    payload: Dict[str, Any] = {
        "status": "healthy",
        "service": "FBA-Bench Research Toolkit API",
        "version": __version__,
        "timestamp": ts,
        "environment": env_name,
        "build_time": build.get("build_time", "unknown"),
        "git_sha": build.get("git_sha", "unknown"),
        "uptime_s": round(uptime, 3),
        "dashboard_service_running": bool(
            dashboard_service and getattr(dashboard_service, "is_running", False)
        ),
        "websocket_connections": len(getattr(connection_manager, "active_connections", [])),
    }

    if redis_ok is not None:
        payload["redis_ok"] = redis_ok
    if db_ok is not None:
        payload["db_ok"] = db_ok

    return payload


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