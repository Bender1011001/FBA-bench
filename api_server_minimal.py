"""
Minimal FBA-Bench API Server - bypasses problematic imports for now
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
import uuid
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Basic models for API
class SimulationSnapshot(BaseModel):
    """Pydantic model for simulation snapshot response."""
    current_tick: Optional[int] = 0
    simulation_time: Optional[str] = None
    last_update: Optional[str] = None
    uptime_seconds: Optional[int] = 0
    products: Optional[Dict[str, Any]] = {}
    competitors: Optional[Dict[str, Any]] = {}
    market_summary: Optional[Dict[str, Any]] = {}
    financial_summary: Optional[Dict[str, Any]] = {}
    agents: Optional[Dict[str, Any]] = {}
    command_stats: Optional[Dict[str, Any]] = {}
    event_stats: Optional[Dict[str, Any]] = {}
    metadata: Optional[Dict[str, Any]] = {}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock dashboard service
class MockDashboardService:
    def __init__(self):
        self.is_running = True
    
    def get_simulation_snapshot(self) -> Dict[str, Any]:
        return {
            'current_tick': 42,
            'simulation_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat(),
            'uptime_seconds': 120,
            'products': {
                'B08N5WRWNW': {
                    'price': '$19.99',
                    'last_updated': datetime.now().isoformat(),
                    'update_count': 5
                }
            },
            'competitors': {},
            'market_summary': {'total_competitors': 3},
            'financial_summary': {
                'total_revenue': 199900,
                'total_profit': 59970,
                'total_units_sold': 10,
                'total_transactions': 10
            },
            'agents': {},
            'command_stats': {
                'total_commands': 15,
                'accepted_commands': 12,
                'rejected_commands': 3
            },
            'event_stats': {
                'events_processed': 100,
                'events_per_second': 2.5
            },
            'metadata': {
                'service_version': '1.0.0-minimal',
                'snapshot_generation': 1
            }
        }
    
    def get_recent_events(self, event_type=None, limit=20, since_tick=None):
        return [
            {
                'event_id': 'mock_event_1',
                'timestamp': datetime.now().isoformat(),
                'tick_number': 42,
                'type': 'sale',
                'data': {'asin': 'B08N5WRWNW', 'units_sold': 1}
            }
        ]

dashboard_service = MockDashboardService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage API server lifecycle."""
    logger.info("Starting FBA-Bench Minimal API Server...")
    yield
    logger.info("API Server stopped")

# Create FastAPI app
app = FastAPI(
    title="FBA-Bench Minimal API",
    description="Minimal API server for testing the frontend",
    version="1.0.0-minimal",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint."""
    return """
    <html>
        <head><title>FBA-Bench Minimal API</title></head>
        <body>
            <h1>🚀 FBA-Bench Minimal API Server</h1>
            <p>This is a minimal API server for testing the frontend.</p>
            <ul>
                <li><a href="/api/v1/simulation/snapshot">📸 Simulation Snapshot</a></li>
                <li><a href="/api/v1/health">❤️ Health Check</a></li>
                <li><a href="/docs">📖 API Documentation</a></li>
            </ul>
        </body>
    </html>
    """

@app.get("/api/v1/simulation/snapshot", response_model=SimulationSnapshot)
async def get_simulation_snapshot():
    """Get simulation snapshot."""
    snapshot = dashboard_service.get_simulation_snapshot()
    return SimulationSnapshot(**snapshot)

@app.get("/api/v1/simulation/events")
async def get_recent_events(
    event_type: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    since_tick: Optional[int] = Query(None)
):
    """Get recent events."""
    events = dashboard_service.get_recent_events(
        event_type=event_type,
        limit=limit,
        since_tick=since_tick
    )
    
    return {
        "events": events,
        "event_type": event_type,
        "limit": limit,
        "total_returned": len(events),
        "timestamp": datetime.now().isoformat(),
        "filtered": since_tick is not None,
        "since_tick": since_tick if since_tick is not None else None
    }

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "FBA-Bench Minimal API",
        "version": "1.0.0-minimal",
        "timestamp": datetime.now().isoformat(),
        "dashboard_service_running": dashboard_service.is_running
    }

# WebSocket endpoint
@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """WebSocket endpoint for real-time events."""
    await websocket.accept()
    logger.info("WebSocket connected")
    
    try:
        # Send initial snapshot
        snapshot = dashboard_service.get_simulation_snapshot()
        await websocket.send_text(json.dumps({
            "type": "snapshot",
            "data": snapshot,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting FBA-Bench Minimal API Server...")
    print("📊 API: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "api_server_minimal:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )