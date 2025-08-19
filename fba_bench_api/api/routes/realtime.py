from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from typing import Optional, Set, Dict, Any
import json
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query, Header

from fba_bench_api.core.state import dashboard_service
from fba_bench_api.api.dependencies import connection_manager  # fixed import
from fba_bench_api.core.redis_client import get_pubsub, get_redis

logger = logging.getLogger(__name__)

router = APIRouter(tags=["simulation"])

# Protocol documentation:
# Client -> Server JSON frames:
#   {"type":"subscribe","topic":"topic-name"}
#   {"type":"unsubscribe","topic":"topic-name"}
#   {"type":"publish","topic":"topic-name","data":{...}}
#   {"type":"ping"}
# Server -> Client JSON frames:
#   {"type":"event","topic":"topic-name","data":{...},"ts":"2025-08-18T16:01:00Z"}
#   {"type":"pong","ts":"2025-08-18T16:01:01Z"}
#   {"type":"error","error":"message"}


def _now_iso() -> str:
    """UTC ISO-8601 timestamp."""
    return datetime.now(tz=timezone.utc).isoformat()


def _current_status() -> str:
    """Best-effort simulation status derived from available services."""
    try:
        return "running" if (dashboard_service and getattr(dashboard_service, "is_running", False)) else "idle"
    except Exception:
        return "idle"


def _default_snapshot() -> dict:
    """Stable, minimal snapshot shape when no engine is available."""
    return {
        "status": "idle",
        "tick": 0,
        "kpis": {
            "revenue": 0.0,
            "profit": 0.0,
            "units_sold": 0,
        },
        "agents": [],
        "timestamp": _now_iso(),
    }


def _map_dashboard_snapshot() -> dict:
    """
    Map dashboard service snapshot (if available) to the canonical shape.

    Expected output:
    {
      "status": "idle" | "running" | "stopped",
      "tick": int,
      "kpis": {"revenue": float, "profit": float, "units_sold": int},
      "agents": [{"slug": str, "display_name": str, "state": str}],
      "timestamp": ISO-8601 string
    }
    """
    try:
        raw = dashboard_service.get_simulation_snapshot() if dashboard_service else None
        if not isinstance(raw, dict):
            return _default_snapshot()

        status = _current_status()
        tick = int(raw.get("current_tick", 0))
        fin = raw.get("financial_summary", {}) or {}
        kpis = {
            "revenue": float(fin.get("total_revenue", 0.0) or 0.0),
            "profit": float(fin.get("total_profit", 0.0) or 0.0),
            "units_sold": int(fin.get("total_units_sold", 0) or 0),
        }

        agents_raw = raw.get("agents", {}) or {}
        agents_list = []
        # agents may be dict keyed by slug or list; normalize to list
        if isinstance(agents_raw, dict):
            for slug, meta in agents_raw.items():
                agents_list.append({
                    "slug": str(slug),
                    "display_name": str(meta.get("display_name", slug)) if isinstance(meta, dict) else str(slug),
                    "state": str(meta.get("state", "unknown")) if isinstance(meta, dict) else "unknown",
                })
        elif isinstance(agents_raw, list):
            for a in agents_raw:
                agents_list.append({
                    "slug": str(a.get("slug", "agent")) if isinstance(a, dict) else "agent",
                    "display_name": str(a.get("display_name", a.get("slug", "agent"))) if isinstance(a, dict) else "agent",
                    "state": str(a.get("state", "unknown")) if isinstance(a, dict) else "unknown",
                })

        return {
            "status": status,
            "tick": tick,
            "kpis": kpis,
            "agents": agents_list,
            "timestamp": _now_iso(),
        }
    except Exception:
        # On any error, fall back to a deterministic idle snapshot
        return _default_snapshot()


@router.get("/api/v1/simulation/snapshot", tags=["simulation"])
async def get_simulation_snapshot():
    """
    Return a canonical simulation snapshot.

    Payload:
      - status: "idle" | "running" | "stopped"
      - tick: int
      - kpis: { revenue: float, profit: float, units_sold: int }
      - agents: [{ slug, display_name, state }]
      - timestamp: ISO-8601
    """
    try:
        # Prefer mapped dashboard snapshot when available; otherwise idle default
        return _map_dashboard_snapshot() if dashboard_service else _default_snapshot()
    except Exception as e:
        # Structured 500
        raise HTTPException(status_code=500, detail=f"Failed to fetch snapshot: {e}")


@router.get("/api/v1/simulation/events")
async def get_recent_events(
    event_type: Optional[str] = Query(None, description="sales|commands"),
    limit: int = Query(20, ge=1, le=100),
    since_tick: Optional[int] = Query(None),
):
    if not dashboard_service:
        raise HTTPException(503, "Dashboard service not available")
    events = dashboard_service.get_recent_events(event_type=event_type, limit=limit, since_tick=since_tick)
    resp = {
        "events": events,
        "event_type": event_type,
        "limit": limit,
        "total_returned": len(events),
        "timestamp": _now_iso(),
        "filtered": since_tick is not None,
    }
    if since_tick is not None:
        resp["since_tick"] = since_tick
    return resp


@router.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket, origin: Optional[str] = Header(None)):
    """
    Topic-based realtime WebSocket over Redis pub/sub.

    - Supports multiple topics per connection
    - JSON protocol:
        subscribe:   {"type":"subscribe","topic":"X"}
        unsubscribe: {"type":"unsubscribe","topic":"X"}
        publish:     {"type":"publish","topic":"X","data":{...}}
        ping:        {"type":"ping"}
    - Server event: {"type":"event","topic":"X","data":{...},"ts": "..."}
    - Error:        {"type":"error","error":"..."}
    """
    # Accept connection up-front
    await websocket.accept()

    # Prepare per-connection state
    subscribed_topics: Set[str] = set()
    stop_event = asyncio.Event()
    pubsub = None

    async def _send_safe(payload: Dict[str, Any]) -> None:
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception as exc:
            # Client likely disconnected or backpressure failure; trigger shutdown
            logger.warning("WebSocket send failed, closing connection: %s", exc)
            stop_event.set()
            raise

    async def _listener_loop():
        # Background task that forwards Redis pubsub messages to this websocket
        try:
            while not stop_event.is_set():
                try:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                except Exception as exc:
                    logger.error("Redis pubsub get_message error: %s", exc)
                    await _send_safe({"type": "error", "error": "redis_error"})
                    stop_event.set()
                    break
                if not message:
                    continue
                # redis-py returns dict with keys: type, channel, data
                msg_type = message.get("type")
                if msg_type != "message":
                    continue
                topic = message.get("channel")
                raw = message.get("data")
                try:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    data = {"raw": raw}
                await _send_safe({"type": "event", "topic": topic, "data": data, "ts": _now_iso()})
        except asyncio.CancelledError:
            pass
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error("Listener loop error: %s", exc)
            try:
                await _send_safe({"type": "error", "error": "listener_error"})
            except Exception:
                pass
        finally:
            try:
                await pubsub.close()
            except Exception:
                pass

    # Initialize Redis pubsub
    try:
        pubsub = await get_pubsub()
    except Exception:
        await _send_safe({"type": "error", "error": "redis_unavailable"})
        await websocket.close()
        return

    listener_task = asyncio.create_task(_listener_loop())

    # Acknowledge connection
    await _send_safe(
        {
            "type": "connection_established",
            "message": "Realtime WebSocket connection established",
            "ts": _now_iso(),
            "origin": origin,
        }
    )

    malformed_count = 0
    try:
        while not stop_event.is_set():
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception as exc:
                logger.warning("WebSocket receive error: %s", exc)
                break

            try:
                msg = json.loads(raw)
            except Exception:
                malformed_count += 1
                await _send_safe({"type": "error", "error": "invalid_json"})
                if malformed_count >= 3:
                    await websocket.close()
                    break
                continue

            if not isinstance(msg, dict):
                await _send_safe({"type": "error", "error": "invalid_message"})
                continue

            mtype = msg.get("type")
            topic = msg.get("topic")
            data = msg.get("data")

            if mtype == "ping":
                await _send_safe({"type": "pong", "ts": _now_iso()})
                continue

            if mtype == "subscribe":
                if not topic:
                    await _send_safe({"type": "error", "error": "missing_topic"})
                    continue
                if topic in subscribed_topics:
                    # idempotent
                    await _send_safe({"type": "subscribed", "topic": topic, "ts": _now_iso()})
                    continue
                try:
                    await pubsub.subscribe(topic)
                    subscribed_topics.add(topic)
                    logger.info("WS subscribed topic=%s (origin=%s)", topic, origin)
                    await _send_safe({"type": "subscribed", "topic": topic, "ts": _now_iso()})
                except Exception as exc:
                    logger.error("Subscribe failed topic=%s: %s", topic, exc)
                    await _send_safe({"type": "error", "error": "redis_error"})
                continue

            if mtype == "unsubscribe":
                if not topic:
                    await _send_safe({"type": "error", "error": "missing_topic"})
                    continue
                if topic not in subscribed_topics:
                    # no-op
                    await _send_safe({"type": "unsubscribed", "topic": topic, "ts": _now_iso()})
                    continue
                try:
                    await pubsub.unsubscribe(topic)
                    subscribed_topics.discard(topic)
                    logger.info("WS unsubscribed topic=%s (origin=%s)", topic, origin)
                    await _send_safe({"type": "unsubscribed", "topic": topic, "ts": _now_iso()})
                except Exception as exc:
                    logger.error("Unsubscribe failed topic=%s: %s", topic, exc)
                    await _send_safe({"type": "error", "error": "redis_error"})
                continue

            if mtype == "publish":
                if not topic or data is None:
                    await _send_safe({"type": "error", "error": "missing_topic_or_data"})
                    continue
                try:
                    r = await get_redis()
                    await r.publish(topic, json.dumps(data))
                except Exception as exc:
                    logger.error("Publish failed topic=%s: %s", topic, exc)
                    await _send_safe({"type": "error", "error": "redis_error"})
                continue

            # Unknown type
            await _send_safe({"type": "error", "error": "unknown_type"})

    finally:
        stop_event.set()
        try:
            listener_task.cancel()
        except Exception:
            pass
        try:
            await pubsub.close()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
    """
    WebSocket stream for real-time events.

    Behavior:
      - Accepts connections and emits JSON messages.
      - If event stream exists, forwards events (future integration).
      - Else, sends a heartbeat every 2 seconds:
        { "type": "heartbeat", "status": "idle|running|stopped", "timestamp": ISO-8601 }
    """
    client_id = await connection_manager.connect(websocket, origin)
    if not client_id:
        return

    try:
        # Acknowledge connection
        await connection_manager.send_to_connection(
            websocket,
            {
                "type": "connection_established",
                "message": "Events WebSocket connection established",
                "timestamp": _now_iso(),
                "client_id": client_id,
                "origin": origin,
            },
        )

        # Initial snapshot (canonical shape)
        try:
            initial = _map_dashboard_snapshot() if dashboard_service else _default_snapshot()
            await connection_manager.send_to_connection(
                websocket,
                {"type": "snapshot", "data": initial, "timestamp": _now_iso()},
            )
        except Exception:
            await connection_manager.send_to_connection(
                websocket,
                {
                    "type": "error",
                    "message": "Failed to load initial snapshot",
                    "timestamp": _now_iso(),
                },
            )

        # Heartbeat loop (2s)
        while True:
            try:
                # Non-blocking receive to detect disconnects/keepalive pings
                await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
                # We don't process commands yet; send immediate heartbeat response
                await connection_manager.send_to_connection(
                    websocket,
                    {
                        "type": "heartbeat",
                        "status": _current_status(),
                        "timestamp": _now_iso(),
                    },
                )
            except asyncio.TimeoutError:
                # Periodic heartbeat
                await connection_manager.send_to_connection(
                    websocket,
                    {
                        "type": "heartbeat",
                        "status": _current_status(),
                        "timestamp": _now_iso(),
                    },
                )
            except WebSocketDisconnect:
                break
    finally:
        await connection_manager.disconnect(websocket)
# Back-compat alias: keep /ws/events endpoint working by delegating to /ws/realtime
@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket, origin: Optional[str] = Header(None)):
    await websocket_realtime(websocket, origin)