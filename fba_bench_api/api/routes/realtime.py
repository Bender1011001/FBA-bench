from __future__ import annotations
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query, Header

from fba_bench_api.core.state import dashboard_service
from api.dependencies import connection_manager

router = APIRouter(tags=["Realtime"])

@router.get("/api/v1/simulation/snapshot")
async def get_simulation_snapshot():
    if not dashboard_service:
        raise HTTPException(503, "Dashboard service not available")
    return dashboard_service.get_simulation_snapshot()

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
        "timestamp": datetime.now().isoformat(),
        "filtered": since_tick is not None,
    }
    if since_tick is not None:
        resp["since_tick"] = since_tick
    return resp

@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket, origin: Optional[str] = Header(None)):
    client_id = await connection_manager.connect(websocket, origin)
    if not client_id:
        return
    try:
        await connection_manager.send_to_connection(websocket, {
            "type": "connection_established",
            "message": "Events WebSocket connection established",
            "timestamp": datetime.now().isoformat(),
            "client_id": client_id,
            "origin": origin
        })
        if dashboard_service:
            try:
                snap = dashboard_service.get_simulation_snapshot()
                await connection_manager.send_to_connection(websocket, {
                    "type": "snapshot", "data": snap, "timestamp": datetime.now().isoformat()
                })
            except Exception:
                await connection_manager.send_to_connection(websocket, {
                    "type": "error", "message": "Failed to load initial snapshot",
                    "timestamp": datetime.now().isoformat()
                })

        while True:
            try:
                data = await websocket.receive_text()
                # heartbeat + simple command handling
                await connection_manager.send_to_connection(websocket, {"type": "heartbeat", "timestamp": datetime.now().isoformat()})
            except WebSocketDisconnect:
                break
    finally:
        await connection_manager.disconnect(websocket)