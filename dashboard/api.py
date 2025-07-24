"""
FastAPI backend for FBA-Bench Dashboard.
Provides REST endpoints with automatic validation and real-time WebSocket support.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .models import (
    DashboardState, ExecutiveSummary, FinancialDeepDive, ProductMarketAnalysis,
    SupplyChainOperations, AgentCognition, WebSocketEvent, KPIMetrics
)
from .data_exporter import DashboardDataExporter


class InMemoryCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, default_ttl: float = 1.0):
        """
        Initialize cache with default TTL.
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if time.time() > entry['expires_at']:
            del self._cache[key]
            return None
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache with TTL."""
        ttl = ttl or self.default_ttl
        self._cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific WebSocket."""
        try:
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected WebSockets."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)


class DashboardAPI:
    """Main dashboard API class."""
    
    def __init__(self):
        self.cache = InMemoryCache(default_ttl=1.0)  # 1-second TTL
        self.websocket_manager = WebSocketManager()
        self.data_exporter: Optional[DashboardDataExporter] = None
        self.simulation = None
        self.agent = None
        self._background_task_running = False
    
    def set_simulation_components(self, simulation, agent=None):
        """Set simulation and agent components for data export."""
        self.simulation = simulation
        self.agent = agent
        self.data_exporter = DashboardDataExporter(simulation, agent)
    
    async def start_background_updates(self):
        """Start background task for real-time updates."""
        if not self._background_task_running:
            self._background_task_running = True
            asyncio.create_task(self._background_update_loop())
    
    async def stop_background_updates(self):
        """Stop background updates."""
        self._background_task_running = False
    
    async def _background_update_loop(self):
        """Background loop for real-time data updates."""
        while self._background_task_running:
            try:
                if self.data_exporter and self.websocket_manager.active_connections:
                    # Get fresh KPI data for real-time updates
                    kpis = self.data_exporter.extract_kpi_metrics()
                    
                    # Broadcast KPI updates
                    event = WebSocketEvent(
                        event_type="kpi_update",
                        data=kpis.dict(),
                        timestamp=datetime.now()
                    )
                    
                    await self.websocket_manager.broadcast(event.dict())
                
                # Wait 1 second before next update
                await asyncio.sleep(1.0)
                
            except Exception as e:
                print(f"Background update error: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error


# Global API instance
dashboard_api = DashboardAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    await dashboard_api.start_background_updates()
    yield
    # Shutdown
    await dashboard_api.stop_background_updates()


# Create FastAPI app
app = FastAPI(
    title="FBA-Bench Dashboard API",
    description="Real-time dashboard API for FBA-Bench simulation analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utility functions

def get_cached_or_compute(cache_key: str, compute_func, ttl: Optional[float] = None):
    """Get data from cache or compute and cache it."""
    cached_value = dashboard_api.cache.get(cache_key)
    if cached_value is not None:
        return cached_value
    
    if not dashboard_api.data_exporter:
        raise HTTPException(status_code=503, detail="Simulation not connected")
    
    computed_value = compute_func()
    dashboard_api.cache.set(cache_key, computed_value, ttl)
    return computed_value


# REST API Endpoints

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "FBA-Bench Dashboard API",
        "version": "1.0.0",
        "status": "running",
        "simulation_connected": dashboard_api.data_exporter is not None,
        "active_websockets": len(dashboard_api.websocket_manager.active_connections),
        "cache_size": dashboard_api.cache.size()
    }


@app.get("/api/health", response_model=dict)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "simulation_connected": dashboard_api.data_exporter is not None
    }


@app.get("/api/dashboard/complete", response_model=DashboardState)
async def get_complete_dashboard_state():
    """Get complete dashboard state for all tabs."""
    return get_cached_or_compute(
        "complete_dashboard_state",
        lambda: dashboard_api.data_exporter.extract_complete_dashboard_state()
    )


@app.get("/api/dashboard/executive-summary", response_model=ExecutiveSummary)
async def get_executive_summary():
    """Get executive summary data for Tab 1."""
    return get_cached_or_compute(
        "executive_summary",
        lambda: dashboard_api.data_exporter.extract_executive_summary()
    )


@app.get("/api/dashboard/financial", response_model=FinancialDeepDive)
async def get_financial_deep_dive():
    """Get financial deep dive data for Tab 2."""
    return get_cached_or_compute(
        "financial_deep_dive",
        lambda: dashboard_api.data_exporter.extract_financial_deep_dive()
    )


@app.get("/api/dashboard/product-market", response_model=ProductMarketAnalysis)
async def get_product_market_analysis():
    """Get product and market analysis data for Tab 3."""
    return get_cached_or_compute(
        "product_market_analysis",
        lambda: dashboard_api.data_exporter.extract_product_market_analysis()
    )


@app.get("/api/dashboard/supply-chain", response_model=SupplyChainOperations)
async def get_supply_chain_operations():
    """Get supply chain operations data for Tab 4."""
    return get_cached_or_compute(
        "supply_chain_operations",
        lambda: dashboard_api.data_exporter.extract_supply_chain_operations()
    )


@app.get("/api/dashboard/agent-cognition", response_model=AgentCognition)
async def get_agent_cognition():
    """Get agent cognition and strategy data for Tab 5."""
    return get_cached_or_compute(
        "agent_cognition",
        lambda: dashboard_api.data_exporter.extract_agent_cognition()
    )


@app.get("/api/kpis", response_model=KPIMetrics)
async def get_kpi_metrics():
    """Get current KPI metrics (frequently updated)."""
    return get_cached_or_compute(
        "kpi_metrics",
        lambda: dashboard_api.data_exporter.extract_kpi_metrics(),
        ttl=0.5  # Shorter TTL for KPIs
    )


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear all cached data."""
    dashboard_api.cache.clear()
    return {"message": "Cache cleared successfully"}


@app.get("/api/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    return {
        "size": dashboard_api.cache.size(),
        "default_ttl": dashboard_api.cache.default_ttl
    }


# WebSocket endpoint for real-time updates

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    await dashboard_api.websocket_manager.connect(websocket)
    
    try:
        # Send initial connection confirmation
        await dashboard_api.websocket_manager.send_personal_message({
            "event_type": "connection_established",
            "data": {"message": "Connected to FBA-Bench Dashboard"},
            "timestamp": datetime.now().isoformat()
        }, websocket)
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_json()
                
                # Handle client requests
                if data.get("type") == "request_update":
                    # Send immediate update
                    if dashboard_api.data_exporter:
                        kpis = dashboard_api.data_exporter.extract_kpi_metrics()
                        event = WebSocketEvent(
                            event_type="kpi_update",
                            data=kpis.dict(),
                            timestamp=datetime.now()
                        )
                        await dashboard_api.websocket_manager.send_personal_message(
                            event.dict(), websocket
                        )
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {e}")
                break
    
    finally:
        dashboard_api.websocket_manager.disconnect(websocket)


# Simulation integration endpoints

@app.post("/api/simulation/connect")
async def connect_simulation(simulation_data: dict):
    """Connect simulation components to the dashboard."""
    # This would be called by the simulation to register itself
    # For now, we'll just acknowledge the connection
    return {
        "message": "Simulation connection acknowledged",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/simulation/update")
async def simulation_update(update_data: dict):
    """Receive simulation updates and broadcast to WebSocket clients."""
    # Clear relevant cache entries
    dashboard_api.cache.clear()
    
    # Broadcast update to WebSocket clients
    event = WebSocketEvent(
        event_type="simulation_update",
        data=update_data,
        timestamp=datetime.now()
    )
    
    await dashboard_api.websocket_manager.broadcast(event.dict())
    
    return {"message": "Update processed successfully"}


# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# Utility function to run the server
def run_dashboard_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    simulation=None,
    agent=None
):
    """
    Run the dashboard server with optional simulation components.
    
    Args:
        host: Server host
        port: Server port
        simulation: FBA-Bench simulation instance
        agent: AdvancedAgent instance
    """
    if simulation:
        dashboard_api.set_simulation_components(simulation, agent)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    run_dashboard_server()