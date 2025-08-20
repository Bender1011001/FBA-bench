"""
Frontend-facing WebSocket service used by integration tests.

Implements an async client over the `websockets` package with helpers:
- connect()
- subscribe(benchmark_id)
- get_status(benchmark_id)
- ping()
- listen_for_updates(benchmark_id) -> async generator of updates
- disconnect()

The tests monkeypatch `websockets.connect` to return an async mock with
`send` and `recv` coroutine methods, so we must:
- Call `websockets.connect(self.url)` on connect()
- Use `await self.websocket.send(json.dumps(msg))`
- Use `await self.websocket.recv()` to receive messages
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Optional

import websockets  # type: ignore[import-not-found]


class WebSocketService:
    """Async WebSocket client with convenience methods for the test protocol."""

    def __init__(self, url: str, connect_timeout: float = 10.0) -> None:
        self.url = url
        self.connect_timeout = float(connect_timeout)
        self.websocket: Optional[Any] = None
        self.connected: bool = False

    async def connect(self) -> None:
        """Establish the WebSocket connection."""
        if self.connected:
            return
        # tests patch websockets.connect to return an AsyncMock
        self.websocket = await websockets.connect(self.url)
        self.connected = True

    async def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self.websocket:
            try:
                # AsyncMock in tests may not implement .close(); guard it.
                close = getattr(self.websocket, "close", None)
                if callable(close):
                    await close()
            finally:
                self.websocket = None
                self.connected = False

    async def subscribe(self, benchmark_id: str) -> dict:
        """Send a subscription message and await a single response."""
        await self._ensure_connected()
        message = {"type": "subscribe", "benchmark_id": benchmark_id}
        await self.websocket.send(json.dumps(message))
        raw = await self.websocket.recv()
        return self._parse_json(raw)

    async def get_status(self, benchmark_id: str) -> dict:
        """Request a status update for a given benchmark id."""
        await self._ensure_connected()
        message = {"type": "get_status", "benchmark_id": benchmark_id}
        await self.websocket.send(json.dumps(message))
        raw = await self.websocket.recv()
        return self._parse_json(raw)

    async def ping(self) -> dict:
        """Send a ping and wait for a pong."""
        await self._ensure_connected()
        message = {"type": "ping"}
        await self.websocket.send(json.dumps(message))
        raw = await self.websocket.recv()
        return self._parse_json(raw)

    async def listen_for_updates(self, benchmark_id: str) -> AsyncGenerator[dict, None]:
        """
        Subscribe and then yield updates until the caller stops iteration.
        Protocol: send subscribe, then stream recv() results as parsed JSON.
        """
        await self._ensure_connected()
        # Ensure subscription first
        sub_msg = {"type": "subscribe", "benchmark_id": benchmark_id}
        await self.websocket.send(json.dumps(sub_msg))
        # Consume the subscription confirmation (or first update)
        raw = await self.websocket.recv()
        first = self._parse_json(raw)
        yield first
        # Stream subsequent messages
        while True:
            raw = await self.websocket.recv()
            yield self._parse_json(raw)

    async def _ensure_connected(self) -> None:
        if not self.connected or self.websocket is None:
            await self.connect()

    @staticmethod
    def _parse_json(raw: Any) -> dict:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except Exception:
                data = {"type": "error", "message": "invalid_json", "raw": raw}
            if isinstance(data, dict):
                return data
            return {"type": "message", "data": data}
        # If tests yield already parsed dict
        if isinstance(raw, dict):
            return raw
        return {"type": "message", "data": raw}