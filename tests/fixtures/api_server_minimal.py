# =========================================================
# TEST FIXTURE: Minimal API server for frontend compatibility
# This module is a test fixture and NOT part of production code.
# Do NOT import this from production modules. Use fba_bench_api.main instead.
# =========================================================

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

# Real dashboard service implementation
class DashboardService:
    """
    Production-ready dashboard service that provides real simulation data.
    
    This service connects to the event bus and simulation orchestrator
    to provide real-time data about the simulation state.
    """
    
    def __init__(self):
        """Initialize the dashboard service."""
        self.is_running = True
        self.start_time = datetime.now()
        self.simulation_orchestrator = None
        self.event_bus = None
        self.last_events = []
        self.max_events_stored = 100
        
        # Initialize with default values
        self.current_tick = 0
        self.products = {}
        self.competitors = {}
        self.market_summary = {'total_competitors': 0}
        self.financial_summary = {
            'total_revenue': 0,
            'total_profit': 0,
            'total_units_sold': 0,
            'total_transactions': 0
        }
        self.agents = {}
        self.command_stats = {
            'total_commands': 0,
            'accepted_commands': 0,
            'rejected_commands': 0
        }
        self.event_stats = {
            'events_processed': 0,
            'events_per_second': 0.0
        }
        
        logger.info("DashboardService initialized")
    
    def connect_to_simulation(self, simulation_orchestrator, event_bus):
        """
        Connect to the simulation orchestrator and event bus.
        
        Args:
            simulation_orchestrator: The simulation orchestrator instance
            event_bus: The event bus instance
        """
        self.simulation_orchestrator = simulation_orchestrator
        self.event_bus = event_bus
        
        # Subscribe to events
        if self.event_bus:
            self.event_bus.subscribe(self._handle_event)
            
        logger.info("Connected to simulation orchestrator and event bus")
    
    def _handle_event(self, event):
        """
        Handle incoming events from the event bus.
        
        Args:
            event: The event object
        """
        # Store the event
        self.last_events.append({
            'event_id': getattr(event, 'event_id', str(uuid.uuid4())),
            'timestamp': getattr(event, 'timestamp', datetime.now().isoformat()),
            'tick_number': getattr(event, 'tick_number', self.current_tick),
            'type': getattr(event, 'type', 'unknown'),
            'data': getattr(event, 'data', {})
        })
        
        # Limit the number of stored events
        if len(self.last_events) > self.max_events_stored:
            self.last_events = self.last_events[-self.max_events_stored:]
        
        # Update statistics
        self.event_stats['events_processed'] += 1
        
        # Process specific event types
        if hasattr(event, 'type'):
            if event.type == 'tick':
                self.current_tick = getattr(event, 'tick_number', self.current_tick)
            elif event.type == 'sale':
                self._process_sale_event(event)
            elif event.type == 'price_update':
                self._process_price_update_event(event)
            elif event.type == 'competitor_update':
                self._process_competitor_update_event(event)
    
    def _process_sale_event(self, event):
        """Process a sale event and update financial summary."""
        data = getattr(event, 'data', {})
        asin = data.get('asin', '')
        units_sold = data.get('units_sold', 0)
        price = data.get('price', 0.0)
        
        # Update financial summary
        self.financial_summary['total_units_sold'] += units_sold
        self.financial_summary['total_transactions'] += 1
        self.financial_summary['total_revenue'] += price * units_sold
        
        # Update product information
        if asin and asin not in self.products:
            self.products[asin] = {
                'price': price,
                'last_updated': datetime.now().isoformat(),
                'update_count': 1
            }
        else:
            self.products[asin]['update_count'] += 1
            self.products[asin]['last_updated'] = datetime.now().isoformat()
    
    def _process_price_update_event(self, event):
        """Process a price update event."""
        data = getattr(event, 'data', {})
        asin = data.get('asin', '')
        price = data.get('price', 0.0)
        
        # Update product information
        if asin:
            if asin not in self.products:
                self.products[asin] = {
                    'price': price,
                    'last_updated': datetime.now().isoformat(),
                    'update_count': 1
                }
            else:
                self.products[asin]['price'] = price
                self.products[asin]['last_updated'] = datetime.now().isoformat()
                self.products[asin]['update_count'] += 1
    
    def _process_competitor_update_event(self, event):
        """Process a competitor update event."""
        data = getattr(event, 'data', {})
        competitor_id = data.get('competitor_id', '')
        
        if competitor_id:
            self.competitors[competitor_id] = data
            self.market_summary['total_competitors'] = len(self.competitors)
    
    def get_simulation_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of the current simulation state.
        
        Returns:
            Dictionary containing simulation state information
        """
        # Calculate uptime
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate events per second
        if uptime_seconds > 0:
            self.event_stats['events_per_second'] = self.event_stats['events_processed'] / uptime_seconds
        
        # Get simulation status if connected to orchestrator
        if self.simulation_orchestrator:
            try:
                orchestrator_status = self.simulation_orchestrator.get_status()
                self.current_tick = orchestrator_status.get('current_tick', self.current_tick)
            except Exception as e:
                logger.error(f"Error getting orchestrator status: {e}")
        
        return {
            'current_tick': self.current_tick,
            'simulation_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat(),
            'uptime_seconds': uptime_seconds,
            'products': self.products,
            'competitors': self.competitors,
            'market_summary': self.market_summary,
            'financial_summary': self.financial_summary,
            'agents': self.agents,
            'command_stats': self.command_stats,
            'event_stats': self.event_stats,
            'metadata': {
                'service_version': '1.0.0',
                'snapshot_generation': int(uptime_seconds / 10) + 1
            }
        }
    
    def get_recent_events(self, event_type=None, limit=20, since_tick=None):
        """
        Get recent events from the event bus.
        
        Args:
            event_type: Filter by event type
            limit: Maximum number of events to return
            since_tick: Only return events since this tick number
            
        Returns:
            List of event dictionaries
        """
        # Filter events based on criteria
        filtered_events = self.last_events
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.get('type') == event_type]
        
        if since_tick is not None:
            filtered_events = [e for e in filtered_events if e.get('tick_number', 0) >= since_tick]
        
        # Return the most recent events up to the limit
        return filtered_events[-limit:] if limit else filtered_events
    
    def start(self):
        """Start the dashboard service."""
        self.is_running = True
        self.start_time = datetime.now()
        logger.info("DashboardService started")
    
    def stop(self):
        """Stop the dashboard service."""
        self.is_running = False
        logger.info("DashboardService stopped")

dashboard_service = DashboardService()

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