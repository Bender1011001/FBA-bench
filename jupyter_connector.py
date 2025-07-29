"""
FBA-Bench v3 Jupyter Connector - Observer Mode

This module provides a secure, read-only interface for connecting Jupyter notebooks
to live FBA-Bench simulations. The connector operates in strict observer mode,
ensuring no notebook can alter simulation state.

Core Architecture:
- Leverages existing DashboardAPIService via FastAPI endpoints
- Maintains real-time sync with simulation via WebSocket
- Exposes Pandas-friendly data structures for analysis
- Zero write capabilities - pure observer pattern
"""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
import pandas as pd
import requests
import websockets
from websockets.exceptions import ConnectionClosed


logger = logging.getLogger(__name__)


class JupyterConnector:
    """
    Secure, read-only connector for FBA-Bench simulation analysis.
    
    This class connects to a running api_server.py instance and provides
    real-time access to simulation data via Pandas DataFrames and event streams.
    
    Features:
    - Automatic WebSocket reconnection
    - Thread-safe data access
    - Real-time event buffering
    - Multiple data export formats (DataFrame, JSON, dict)
    - Zero simulation write capabilities (observer mode)
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000", 
                 websocket_url: str = "ws://localhost:8000/ws/events",
                 event_buffer_size: int = 1000):
        """
        Initialize Jupyter connector to FBA-Bench simulation.
        
        Args:
            api_base_url: Base URL of the FastAPI server
            websocket_url: WebSocket endpoint for real-time events
            event_buffer_size: Maximum events to keep in memory buffer
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.websocket_url = websocket_url
        self.event_buffer_size = event_buffer_size
        
        # Thread-safe data storage
        self._lock = threading.RLock()
        self._current_snapshot: Optional[Dict[str, Any]] = None
        self._event_buffer: deque = deque(maxlen=event_buffer_size)
        self._last_update: Optional[datetime] = None
        
        # WebSocket connection management
        self._websocket_task: Optional[asyncio.Task] = None
        self._websocket_loop: Optional[asyncio.AbstractEventLoop] = None
        self._websocket_thread: Optional[threading.Thread] = None
        self._is_connected = False
        self._connection_callbacks: List[Callable] = []
        
        # Start WebSocket connection in background
        self._start_websocket_thread()
        
        logger.info(f"JupyterConnector initialized for {api_base_url}")
    
    def _start_websocket_thread(self):
        """Start WebSocket connection in a background thread."""
        def run_websocket():
            self._websocket_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._websocket_loop)
            self._websocket_loop.run_until_complete(self._websocket_handler())
        
        self._websocket_thread = threading.Thread(target=run_websocket, daemon=True)
        self._websocket_thread.start()
    
    async def _websocket_handler(self):
        """Handle WebSocket connection with automatic reconnection."""
        while True:
            try:
                logger.info(f"Connecting to WebSocket: {self.websocket_url}")
                async with websockets.connect(self.websocket_url) as websocket:
                    self._is_connected = True
                    logger.info("WebSocket connected successfully")
                    
                    # Notify connection callbacks
                    for callback in self._connection_callbacks:
                        callback(True)
                    
                    # Listen for events
                    async for message in websocket:
                        try:
                            event_data = json.loads(message)
                            with self._lock:
                                self._event_buffer.append({
                                    'timestamp': datetime.now(),
                                    'data': event_data
                                })
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse WebSocket message: {e}")
                            
            except ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                self._is_connected = False
                await asyncio.sleep(5)  # Wait before reconnecting
            except Exception as e:
                logger.error(f"WebSocket error: {e}, reconnecting...")
                self._is_connected = False
                await asyncio.sleep(5)
    
    def refresh_snapshot(self) -> bool:
        """
        Fetch the latest simulation snapshot from the API.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = requests.get(f"{self.api_base_url}/api/v1/simulation/snapshot", 
                                  timeout=10)
            response.raise_for_status()
            
            with self._lock:
                self._current_snapshot = response.json()
                self._last_update = datetime.now()
            
            logger.debug("Snapshot refreshed successfully")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to refresh snapshot: {e}")
            return False
    
    def get_snapshot_dict(self) -> Optional[Dict[str, Any]]:
        """
        Get the current simulation snapshot as a dictionary.
        
        Returns:
            Dict containing simulation state or None if not available
        """
        with self._lock:
            return self._current_snapshot.copy() if self._current_snapshot else None
    
    def get_snapshot_df(self) -> pd.DataFrame:
        """
        Get the current simulation snapshot as a Pandas DataFrame.
        
        Returns:
            DataFrame with key simulation metrics flattened for analysis
        """
        snapshot = self.get_snapshot_dict()
        if not snapshot:
            return pd.DataFrame()
        
        # Flatten key metrics into a single-row DataFrame
        flattened = {}
        
        # Basic metrics
        if 'tick' in snapshot:
            flattened['tick'] = snapshot['tick']
        if 'timestamp' in snapshot:
            flattened['timestamp'] = snapshot['timestamp']
        
        # Financial metrics
        if 'metrics' in snapshot:
            metrics = snapshot['metrics']
            for key, value in metrics.items():
                if isinstance(value, dict) and 'amount' in value:
                    # Handle Money type
                    flattened[f'metrics_{key}'] = float(value['amount'])
                else:
                    flattened[f'metrics_{key}'] = value
        
        # Agent data
        if 'agents' in snapshot:
            agents = snapshot['agents']
            for agent_id, agent_data in agents.items():
                for key, value in agent_data.items():
                    if isinstance(value, dict) and 'amount' in value:
                        # Handle Money type
                        flattened[f'agent_{agent_id}_{key}'] = float(value['amount'])
                    else:
                        flattened[f'agent_{agent_id}_{key}'] = value
        
        # Competitor data
        if 'competitors' in snapshot:
            competitors = snapshot['competitors']
            for comp_name, comp_data in competitors.items():
                for key, value in comp_data.items():
                    if isinstance(value, dict) and 'amount' in value:
                        # Handle Money type
                        flattened[f'competitor_{comp_name}_{key}'] = float(value['amount'])
                    else:
                        flattened[f'competitor_{comp_name}_{key}'] = value
        
        return pd.DataFrame([flattened])
    
    def get_events_df(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get recent events as a Pandas DataFrame.
        
        Args:
            limit: Maximum number of events to return (default: all buffered)
            
        Returns:
            DataFrame with event data and timestamps
        """
        with self._lock:
            events = list(self._event_buffer)
        
        if limit:
            events = events[-limit:]
        
        if not events:
            return pd.DataFrame()
        
        # Convert events to DataFrame
        rows = []
        for event in events:
            row = {
                'timestamp': event['timestamp'],
                'event_type': event['data'].get('event_type', 'Unknown'),
            }
            
            # Flatten event data
            event_data = event['data'].get('data', {})
            for key, value in event_data.items():
                if isinstance(value, dict) and 'amount' in value:
                    # Handle Money type
                    row[key] = float(value['amount'])
                else:
                    row[key] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_event_stream(self, event_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Get recent events from the live stream.
        
        Args:
            event_types: Filter events by type (default: all events)
            
        Returns:
            List of event dictionaries
        """
        with self._lock:
            events = list(self._event_buffer)
        
        if event_types:
            events = [e for e in events 
                     if e['data'].get('event_type') in event_types]
        
        return events
    
    def get_financial_history_df(self) -> pd.DataFrame:
        """
        Extract financial transaction history from events.
        
        Returns:
            DataFrame with sales and price change history
        """
        events = self.get_event_stream(['SaleOccurred', 'ProductPriceUpdated'])
        
        if not events:
            return pd.DataFrame()
        
        rows = []
        for event in events:
            event_data = event['data']
            row = {
                'timestamp': event['timestamp'],
                'event_type': event_data.get('event_type'),
            }
            
            # Extract event-specific data
            data = event_data.get('data', {})
            for key, value in data.items():
                if isinstance(value, dict) and 'amount' in value:
                    row[key] = float(value['amount'])
                else:
                    row[key] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def is_connected(self) -> bool:
        """Check if WebSocket connection is active."""
        return self._is_connected
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get detailed connection status information.
        
        Returns:
            Dictionary with connection details
        """
        with self._lock:
            return {
                'websocket_connected': self._is_connected,
                'last_snapshot_update': self._last_update,
                'events_buffered': len(self._event_buffer),
                'has_snapshot_data': self._current_snapshot is not None,
                'api_base_url': self.api_base_url,
                'websocket_url': self.websocket_url
            }
    
    def add_connection_callback(self, callback: Callable[[bool], None]):
        """
        Add callback to be called when connection status changes.
        
        Args:
            callback: Function that takes connection status (bool) as parameter
        """
        self._connection_callbacks.append(callback)
    
    def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """
        Wait for WebSocket connection to be established.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if connected within timeout, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_connected():
                return True
            time.sleep(0.1)
        return False
    
    def close(self):
        """Clean up resources and close connections."""
        if self._websocket_loop and not self._websocket_loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._websocket_loop.shutdown_asyncgens(), 
                self._websocket_loop
            )
        
        logger.info("JupyterConnector closed")


# Convenience function for quick notebook setup
def connect_to_simulation(api_url: str = "http://localhost:8000") -> JupyterConnector:
    """
    Quick connection setup for Jupyter notebooks.
    
    Args:
        api_url: Base URL of the running FBA-Bench API server
        
    Returns:
        Connected JupyterConnector instance
    """
    connector = JupyterConnector(api_base_url=api_url)
    
    # Wait for connection and refresh snapshot
    if connector.wait_for_connection(timeout=10):
        connector.refresh_snapshot()
        print(f"✅ Connected to FBA-Bench simulation at {api_url}")
        status = connector.get_connection_status()
        print(f"📊 Events buffered: {status['events_buffered']}")
        print(f"📸 Snapshot available: {status['has_snapshot_data']}")
    else:
        print(f"⚠️ Connection timeout - check that API server is running at {api_url}")
    
    return connector