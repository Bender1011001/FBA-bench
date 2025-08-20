from services.dashboard_api_service import DashboardAPIService
from fba_events.bus import InMemoryEventBus as EventBus
import asyncio
import json
import logging
import uuid
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Set

from fastapi import WebSocket, HTTPException

from simulation_orchestrator import SimulationOrchestrator, SimulationConfig # Assuming SimulationConfig will be needed

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time event streaming."""
    
    def __init__(self, max_connections=200):
        self.active_connections: List[WebSocket] = []
        self.connection_subscriptions: Dict[WebSocket, Set[str]] = {}  # Track what each connection subscribes to
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}  # Additional metadata for each connection
        self.max_connections = max_connections
        self._heartbeat_task = None
    
    async def start(self):
        """Starts any background tasks for the ConnectionManager, e.g., heartbeats."""
        logger.info("ConnectionManager: Starting background tasks.")
        # Example: Start a periodic heartbeat task
        self._heartbeat_task = asyncio.create_task(self._periodic_heartbeat())

    async def stop(self):
        """Stops all background tasks and gracefully closes connections."""
        logger.info("ConnectionManager: Stopping background tasks and closing connections.")
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                logger.info("ConnectionManager: Heartbeat task cancelled.")
        
        # Close all active websockets gracefully
        for connection in list(self.active_connections):
            try:
                await connection.close(code=1000, reason="Server shutting down")
            except Exception as e:
                logger.error(f"Error closing WebSocket during shutdown: {e}")
        self.active_connections.clear()
        self.connection_subscriptions.clear()
        self.connection_metadata.clear()
        logger.info("ConnectionManager: All connections closed.")

    async def _periodic_heartbeat(self):
        """Sends periodic heartbeats to all connected websockets."""
        while True:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            # logger.debug("ConnectionManager: Sending heartbeat to all connections.")
            # This heartbeat is mostly for keeping the connection alive at the client side.
            # The server-side ping/pong is handled by FastAPI/uvicorn.
            # Clients can also send a 'ping' message and expect a 'pong'.
    
    async def connect(self, websocket: WebSocket, origin: Optional[str] = None):
        """Accept new WebSocket connection."""
        # Check if we've reached the maximum number of connections
        if len(self.active_connections) >= self.max_connections:
            logger.warning(f"Connection rejected for Origin: {origin} - max connections reached.")
            await websocket.close(code=1008, reason="Maximum connections reached")
            return None # Explicitly return None on rejection
            
        await websocket.accept()
        client_id = str(uuid.uuid4())
        self.active_connections.append(websocket)
        self.connection_subscriptions[websocket] = set()
        self.connection_metadata[websocket] = {
            "connected_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "client_id": client_id,
            "origin": origin # Store origin for debugging
        }
        logger.info(f"📡 WebSocket connected (Client ID: {client_id}, Origin: {origin}). Active connections: {len(self.active_connections)}")
        return client_id # Return the generated client_id
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_subscriptions:
            del self.connection_subscriptions[websocket]
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        logger.info(f"📡 WebSocket disconnected. Active connections: {len(self.active_connections)}")
    
    async def send_to_connection(self, websocket: WebSocket, event_data: Dict[str, Any]):
        """Send event to a specific WebSocket connection."""
        try:
            # Check if the connection is still active
            if websocket not in self.active_connections:
                logger.warning("Attempted to send to a closed or inactive WebSocket connection")
                return False
                
            message = json.dumps(event_data)
            await websocket.send_text(message)
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["last_activity"] = datetime.now().isoformat()
            return True
        except Exception as e: # Catch all exceptions including WebSocketDisconnect cleanly
            logger.error(f"Error sending data to WebSocket client: {e}")
            self.disconnect(websocket)
            return False
    
    async def broadcast_event(self, event_data: Dict[str, Any], event_type: Optional[str] = None):
        """Broadcast event to all connected WebSocket clients.
        
        Args:
            event_data: The event data to broadcast
            event_type: Optional event type for subscription filtering
        """
        if not self.active_connections:
            return
            
        message = json.dumps(event_data)
        disconnected = []
        
        for connection in list(self.active_connections): # Iterate over a copy to allow modification during loop
            try:
                # Check if connection is subscribed to this event type
                if event_type and event_type in self.connection_subscriptions.get(connection, set()):
                    await connection.send_text(message)
                    self.connection_metadata[connection]["last_activity"] = datetime.now().isoformat()
                elif not event_type:  # Broadcast to all if no specific type
                    await connection.send_text(message)
                    self.connection_metadata[connection]["last_activity"] = datetime.now().isoformat()
            except Exception as e: # Catch all exceptions including WebSocketDisconnect cleanly
                logger.error(f"Error broadcasting to WebSocket client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_to_subscribers(self, event_type: str, event_data: Dict[str, Any]):
        """Broadcast event only to clients subscribed to a specific event type."""
        await self.broadcast_event(event_data, event_type)
    
    def add_subscription(self, websocket: WebSocket, event_type: str):
        """Add a subscription for a specific event type to a WebSocket connection."""
        if websocket in self.connection_subscriptions:
            self.connection_subscriptions[websocket].add(event_type)
    
    def remove_subscription(self, websocket: WebSocket, event_type: str):
        """Remove a subscription for a specific event type from a WebSocket connection."""
        if websocket in self.connection_subscriptions:
            self.connection_subscriptions[websocket].discard(event_type)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current WebSocket connections."""
        return {
            "total_connections": len(self.active_connections),
            "subscriptions": {
                event_type: sum(1 for subs in self.connection_subscriptions.values() if event_type in subs)
                for event_type in ["simulation_status", "agent_status", "financial_metrics", "system_health"]
            },
            "connections": [
                {
                    "client_id": metadata.get("client_id"),
                    "connected_at": metadata.get("connected_at"),
                    "last_activity": metadata.get("last_activity"),
                    "subscriptions": list(self.connection_subscriptions.get(conn, set()))
                }
                for conn, metadata in self.connection_metadata.items()
            ]
        }

class SimulationManager:
    """Manages the lifecycle and state of simulation orchestrator instances."""
    def __init__(self):
        self.orchestrators: Dict[str, SimulationOrchestrator] = {}
        self.orchestrator_lock = asyncio.Lock() # For thread safety

    async def get_orchestrator(self, sim_id: str) -> SimulationOrchestrator:
        async with self.orchestrator_lock:
            orchestrator = self.orchestrators.get(sim_id)
            if not orchestrator:
                raise HTTPException(status_code=404, detail=f"Simulation with ID '{sim_id}' not found.")
            return orchestrator

    async def add_orchestrator(self, sim_id: str, orchestrator: SimulationOrchestrator):
        async with self.orchestrator_lock:
            if sim_id in self.orchestrators:
                logger.warning(f"Simulation with ID '{sim_id}' already exists. Overwriting.")
            self.orchestrators[sim_id] = orchestrator
            logger.info(f"Added simulation orchestrator with ID: {sim_id}")

    async def remove_orchestrator(self, sim_id: str):
        async with self.orchestrator_lock:
            if sim_id in self.orchestrators:
                del self.orchestrators[sim_id]
                logger.info(f"Removed simulation orchestrator with ID: {sim_id}")

    def get_simulation_status(self, sim_id: str) -> Optional[Dict[str, Any]]:
        orchestrator = self.orchestrators.get(sim_id)
        if orchestrator:
            return orchestrator.get_status()
        return None

    def get_all_simulation_ids(self) -> List[str]:
        return list(self.orchestrators.keys())

# Global instances
connection_manager = ConnectionManager(max_connections=100)
simulation_manager = SimulationManager()

# Global state for simulation orchestrator instances, kept in api_server for simplicity now
active_simulations: Dict[str, SimulationOrchestrator] = {} # {simulation_id: orchestrator_instance}
active_experiments: Dict[str, Any] = {} # {experiment_id: experiment_manager_instance}
