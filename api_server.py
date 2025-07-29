"""
FBA-Bench v3 Research Toolkit API Server

FastAPI-based REST API and WebSocket server that provides real-time access
to simulation state for research tools and dashboards.

Core Endpoints:
- GET /api/v1/simulation/snapshot - Complete simulation state snapshot
- GET /api/v1/simulation/events - Recent events with filtering
- WebSocket /ws/events - Real-time event streaming

The API is read-only and cannot influence the simulation.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from event_bus import EventBus
from services.dashboard_api_service import DashboardAPIService


class SimulationSnapshot(BaseModel):
    """Pydantic model for simulation snapshot response."""
    current_tick: int
    simulation_time: Optional[str]
    last_update: Optional[str]
    uptime_seconds: int
    products: Dict[str, Any]
    competitors: Dict[str, Any]
    market_summary: Dict[str, Any]
    financial_summary: Dict[str, Any]
    agents: Dict[str, Any]
    command_stats: Dict[str, Any]
    event_stats: Dict[str, Any]
    metadata: Dict[str, Any]


class EventFilter(BaseModel):
    """Pydantic model for event filtering parameters."""
    event_type: Optional[str] = None
    limit: int = 20
    since_tick: Optional[int] = None


class ConnectionManager:
    """Manages WebSocket connections for real-time event streaming."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"📡 WebSocket connected. Active connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"📡 WebSocket disconnected. Active connections: {len(self.active_connections)}")
    
    async def broadcast_event(self, event_data: Dict[str, Any]):
        """Broadcast event to all connected WebSocket clients."""
        if not self.active_connections:
            return
            
        message = json.dumps(event_data)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


# Global state
dashboard_service: Optional[DashboardAPIService] = None
connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage API server lifecycle."""
    global dashboard_service
    
    print("Starting FBA-Bench Research Toolkit API Server...")
    
    # Initialize EventBus and DashboardAPIService
    event_bus = EventBus()
    dashboard_service = DashboardAPIService(event_bus)
    
    # Start services (but don't connect to actual simulation yet)
    # In production, this would connect to the running simulation
    print("API Server ready (dashboard service initialized)")
    
    yield
    
    # Cleanup
    if dashboard_service:
        await dashboard_service.stop()
    print("API Server stopped")


# Create FastAPI app
app = FastAPI(
    title="FBA-Bench Research Toolkit API",
    description="Real-time simulation data API for research and analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative React dev server
        "http://127.0.0.1:5173",  # Alternative localhost format
        "http://127.0.0.1:3000",  # Alternative localhost format
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# REST API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation."""
    return """
    <html>
        <head>
            <title>FBA-Bench Research Toolkit API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .method { color: #007bff; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>🚀 FBA-Bench Research Toolkit API</h1>
            <p>Real-time simulation data API for research and analysis.</p>
            
            <h2>📡 API Endpoints</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/api/v1/simulation/snapshot</code><br>
                Get complete simulation state snapshot
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/api/v1/simulation/events</code><br>
                Get recent events with optional filtering
            </div>
            
            <div class="endpoint">
                <span class="method">WebSocket</span> <code>/ws/events</code><br>
                Real-time event streaming
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/docs</code><br>
                Interactive API documentation (Swagger UI)
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/dashboard</code><br>
                Real-time simulation dashboard
            </div>
            
            <h2>🔗 Quick Links</h2>
            <ul>
                <li><a href="/docs">📖 API Documentation</a></li>
                <li><a href="/dashboard">📊 Live Dashboard</a></li>
                <li><a href="/api/v1/simulation/snapshot">📸 Simulation Snapshot</a></li>
            </ul>
        </body>
    </html>
    """


@app.get("/api/v1/simulation/snapshot", response_model=SimulationSnapshot)
async def get_simulation_snapshot():
    """
    Get complete simulation state snapshot.
    
    Returns comprehensive real-time simulation state including:
    - Current tick and timing information
    - Product prices and inventory
    - Competitor market landscape  
    - Sales and financial metrics
    - Agent activity and command history
    - System performance stats
    """
    if not dashboard_service:
        raise HTTPException(status_code=503, detail="Dashboard service not available")
    
    snapshot = dashboard_service.get_simulation_snapshot()
    return SimulationSnapshot(**snapshot)


@app.get("/api/v1/simulation/events")
async def get_recent_events(
    event_type: Optional[str] = Query(None, description="Filter by event type (sales, commands)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of events to return"),
    since_tick: Optional[int] = Query(None, description="Only return events since this tick")
):
    """
    Get recent events with optional filtering.
    
    Parameters:
    - event_type: Filter by 'sales' or 'commands'
    - limit: Maximum events to return (1-100)
    - since_tick: Only events after this tick
    """
    if not dashboard_service:
        raise HTTPException(status_code=503, detail="Dashboard service not available")
    
    events = dashboard_service.get_recent_events(event_type=event_type, limit=limit)
    
    # Filter by tick if specified
    if since_tick is not None:
        # This would need tick information in events - simplified for now
        pass
    
    return {
        "events": events,
        "event_type": event_type,
        "limit": limit,
        "total_returned": len(events),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "FBA-Bench Research Toolkit API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "dashboard_service_running": dashboard_service is not None and dashboard_service.is_running
    }


# WebSocket Endpoints

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """
    WebSocket endpoint for real-time event streaming.
    
    Clients can connect to receive real-time updates about:
    - Sales transactions
    - Price changes  
    - Agent commands
    - Market updates
    - System events
    """
    await connection_manager.connect(websocket)
    
    try:
        # Send initial snapshot
        if dashboard_service:
            snapshot = dashboard_service.get_simulation_snapshot()
            await websocket.send_text(json.dumps({
                "type": "snapshot",
                "data": snapshot,
                "timestamp": datetime.now().isoformat()
            }))
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client message (ping, filter updates, etc.)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle client commands
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                
            except asyncio.TimeoutError:
                # Send periodic heartbeat
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


# Dashboard Endpoints

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the real-time dashboard."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FBA-Bench Real-Time Dashboard</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: #f5f5f5; 
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 20px; 
                border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
            }
            .header { 
                border-bottom: 1px solid #eee; 
                padding-bottom: 20px; 
                margin-bottom: 20px; 
            }
            .metrics { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 20px; 
                margin-bottom: 30px; 
            }
            .metric-card { 
                background: #f8f9fa; 
                padding: 15px; 
                border-radius: 6px; 
                border-left: 4px solid #007bff; 
            }
            .metric-title { 
                font-size: 12px; 
                color: #666; 
                text-transform: uppercase; 
                letter-spacing: 0.5px; 
            }
            .metric-value { 
                font-size: 24px; 
                font-weight: bold; 
                color: #333; 
                margin-top: 5px; 
            }
            .status { 
                display: inline-block; 
                padding: 4px 8px; 
                border-radius: 12px; 
                font-size: 12px; 
                font-weight: bold; 
            }
            .status.connected { 
                background: #d4edda; 
                color: #155724; 
            }
            .status.disconnected { 
                background: #f8d7da; 
                color: #721c24; 
            }
            .log { 
                background: #f8f9fa; 
                border: 1px solid #dee2e6; 
                border-radius: 4px; 
                padding: 15px; 
                height: 200px; 
                overflow-y: auto; 
                font-family: 'Courier New', monospace; 
                font-size: 12px; 
            }
            .log-entry { 
                margin-bottom: 5px; 
                word-wrap: break-word; 
            }
            .timestamp { 
                color: #666; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 FBA-Bench Real-Time Dashboard</h1>
                <p>Live simulation monitoring and analytics</p>
                <span id="connectionStatus" class="status disconnected">Disconnected</span>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-title">Current Tick</div>
                    <div class="metric-value" id="currentTick">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Total Revenue</div>
                    <div class="metric-value" id="totalRevenue">$0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Total Profit</div>
                    <div class="metric-value" id="totalProfit">$0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Units Sold</div>
                    <div class="metric-value" id="unitsSold">0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Active Agents</div>
                    <div class="metric-value" id="activeAgents">0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Events/sec</div>
                    <div class="metric-value" id="eventsPerSec">0</div>
                </div>
            </div>
            
            <h3>📡 Real-Time Event Stream</h3>
            <div id="eventLog" class="log">
                <div class="log-entry">Connecting to simulation...</div>
            </div>
        </div>
        
        <script>
            let websocket = null;
            let reconnectAttempts = 0;
            const maxReconnectAttempts = 10;
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = protocol + '//' + window.location.host + '/ws/events';
                
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = function() {
                    console.log('WebSocket connected');
                    document.getElementById('connectionStatus').textContent = 'Connected';
                    document.getElementById('connectionStatus').className = 'status connected';
                    reconnectAttempts = 0;
                    addLogEntry('✅ Connected to simulation', 'success');
                };
                
                websocket.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    handleWebSocketMessage(message);
                };
                
                websocket.onclose = function() {
                    console.log('WebSocket disconnected');
                    document.getElementById('connectionStatus').textContent = 'Disconnected';
                    document.getElementById('connectionStatus').className = 'status disconnected';
                    addLogEntry('❌ Disconnected from simulation', 'error');
                    
                    // Attempt to reconnect
                    if (reconnectAttempts < maxReconnectAttempts) {
                        reconnectAttempts++;
                        setTimeout(connectWebSocket, 2000 * reconnectAttempts);
                        addLogEntry(`🔄 Reconnecting... (attempt ${reconnectAttempts})`, 'info');
                    }
                };
                
                websocket.onerror = function(error) {
                    console.error('WebSocket error:', error);
                    addLogEntry('⚠️ Connection error', 'error');
                };
            }
            
            function handleWebSocketMessage(message) {
                if (message.type === 'snapshot') {
                    updateDashboard(message.data);
                    addLogEntry('📸 Snapshot received', 'info');
                } else if (message.type === 'event') {
                    addLogEntry(`📝 ${message.event_type}: ${JSON.stringify(message.data)}`, 'event');
                } else if (message.type === 'heartbeat') {
                    // Silent heartbeat
                } else {
                    addLogEntry(`📨 ${message.type}`, 'info');
                }
            }
            
            function updateDashboard(data) {
                document.getElementById('currentTick').textContent = data.current_tick || 0;
                
                const financial = data.financial_summary || {};
                document.getElementById('totalRevenue').textContent = 
                    '$' + (financial.total_revenue / 100 || 0).toFixed(2);
                document.getElementById('totalProfit').textContent = 
                    '$' + (financial.total_profit / 100 || 0).toFixed(2);
                document.getElementById('unitsSold').textContent = 
                    financial.total_units_sold || 0;
                
                const agentCount = Object.keys(data.agents || {}).length;
                document.getElementById('activeAgents').textContent = agentCount;
                
                const eventStats = data.event_stats || {};
                document.getElementById('eventsPerSec').textContent = 
                    eventStats.events_per_second || 0;
            }
            
            function addLogEntry(message, type = 'info') {
                const log = document.getElementById('eventLog');
                const timestamp = new Date().toLocaleTimeString();
                const entry = document.createElement('div');
                entry.className = 'log-entry';
                entry.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
                log.appendChild(entry);
                log.scrollTop = log.scrollHeight;
                
                // Keep only last 100 entries
                while (log.children.length > 100) {
                    log.removeChild(log.firstChild);
                }
            }
            
            // Initial connection
            connectWebSocket();
            
            // Periodic ping to keep connection alive
            setInterval(() => {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({type: 'ping'}));
                }
            }, 30000);
        </script>
    </body>
    </html>
    """


# Utility function to broadcast events (called by external services)
async def broadcast_event_to_clients(event_type: str, event_data: Dict[str, Any]):
    """Broadcast event to all connected WebSocket clients."""
    await connection_manager.broadcast_event({
        "type": "event",
        "event_type": event_type,
        "data": event_data,
        "timestamp": datetime.now().isoformat()
    })


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting FBA-Bench Research Toolkit API Server...")
    print("📊 Dashboard: http://localhost:8000/dashboard")
    print("📖 API Docs: http://localhost:8000/docs")
    print("📡 WebSocket: ws://localhost:8000/ws/events")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )