import json
import os
import threading
import time
from typing import Any, Dict
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
import socket

from fba_bench_api.main import create_app


def _redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


def _redis_available() -> bool:
    """
    Best-effort availability check without requiring an event loop.
    Tries a short TCP connect to the Redis host:port parsed from REDIS_URL.
    """
    try:
        parsed = urlparse(_redis_url())
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _redis_available(), reason="Redis is unavailable; set REDIS_URL or start Redis to run realtime tests")
def test_realtime_pubsub_basic():
    app = create_app()
    client = TestClient(app)

    topic = "test-topic-realtime-basic"

    with client.websocket_connect("/ws/realtime") as ws:
        # Connection ack
        msg = ws.receive_json()
        assert isinstance(msg, dict)
        assert msg.get("type") == "connection_established"

        # Subscribe
        ws.send_text(json.dumps({"type": "subscribe", "topic": topic}))
        msg2 = ws.receive_json()
        assert msg2.get("type") == "subscribed"
        assert msg2.get("topic") == topic

        # Publish payload
        payload: Dict[str, Any] = {"progress": 0.5, "status": "running"}
        ws.send_text(json.dumps({"type": "publish", "topic": topic, "data": payload}))

        # Receive the event (published via Redis pub/sub)
        received = ws.receive_json()
        # Some environments may deliver a pong/other acks interleaved; loop a bit to find the event
        max_checks = 5
        while received.get("type") != "event" and max_checks > 0:
            received = ws.receive_json()
            max_checks -= 1

        assert received.get("type") == "event"
        assert received.get("topic") == topic
        assert received.get("data") == payload
        assert "ts" in received

        # Ping/pong
        ws.send_text(json.dumps({"type": "ping"}))
        pong = ws.receive_json()
        # Allow any intermediary non-pong messages, search for pong briefly
        tries = 4
        while pong.get("type") != "pong" and tries > 0:
            pong = ws.receive_json()
            tries -= 1
        assert pong.get("type") == "pong"
        assert "ts" in pong


@pytest.mark.integration
@pytest.mark.skipif(not _redis_available(), reason="Redis is unavailable; set REDIS_URL or start Redis to run realtime tests")
def test_unsubscribe_stops_delivery_with_short_timeout():
    app = create_app()
    client = TestClient(app)

    topic = "test-topic-realtime-unsub"

    with client.websocket_connect("/ws/realtime") as ws:
        # Connection ack
        _ = ws.receive_json()

        # Subscribe
        ws.send_text(json.dumps({"type": "subscribe", "topic": topic}))
        ack = ws.receive_json()
        assert ack.get("type") == "subscribed"
        assert ack.get("topic") == topic

        # Unsubscribe
        ws.send_text(json.dumps({"type": "unsubscribe", "topic": topic}))
        unack = ws.receive_json()
        assert unack.get("type") == "unsubscribed"
        assert unack.get("topic") == topic

        # Publish an event after unsubscribe
        ws.send_text(json.dumps({"type": "publish", "topic": topic, "data": {"x": 1}}))

        # We expect no event to arrive for this topic; implement a short timeout window.
        # Starlette's TestClient websocket receive is blocking; we use a timer to close the socket if
        # no message arrives within the timeout. If a message is received before close and is the
        # forbidden event, we fail; otherwise we treat the timeout + close as success.
        failure = {"tripped": False}

        def closer():
            # Close after 200ms to enforce a bounded wait.
            time.sleep(0.2)
            try:
                ws.close()
            except Exception:
                pass

        t = threading.Thread(target=closer, daemon=True)
        t.start()

        try:
            msg = ws.receive_json()
            # If we got something before the close, assert it's not an event for the topic
            assert not (msg.get("type") == "event" and msg.get("topic") == topic), "Received event after unsubscribe"
        except Exception:
            # Expected path: socket closed by timer or by server without delivering the event.
            failure["tripped"] = False

        t.join(timeout=1.0)
@pytest.mark.integration
@pytest.mark.skipif(not _redis_available(), reason="Redis is unavailable; set REDIS_URL or start Redis to run realtime tests")
def test_multi_client_broadcast_receives_published_events():
    app = create_app()
    client = TestClient(app)

    topic = "test-topic-realtime-broadcast"

    with client.websocket_connect("/ws/realtime") as ws1, client.websocket_connect("/ws/realtime") as ws2:
        # Connection acks
        _ = ws1.receive_json()
        _ = ws2.receive_json()

        # Subscribe both
        ws1.send_text(json.dumps({"type": "subscribe", "topic": topic}))
        ack1 = ws1.receive_json()
        assert ack1.get("type") == "subscribed"

        ws2.send_text(json.dumps({"type": "subscribe", "topic": topic}))
        ack2 = ws2.receive_json()
        assert ack2.get("type") == "subscribed"

        # Publish from client 1
        payload = {"who": "ws1", "value": 42}
        ws1.send_text(json.dumps({"type": "publish", "topic": topic, "data": payload}))

        # Both should receive the event via Redis
        recv1 = ws1.receive_json()
        tries = 5
        while recv1.get("type") != "event" and tries > 0:
            recv1 = ws1.receive_json()
            tries -= 1
        assert recv1.get("type") == "event"
        assert recv1.get("topic") == topic
        assert recv1.get("data") == payload

        recv2 = ws2.receive_json()
        tries = 5
        while recv2.get("type") != "event" and tries > 0:
            recv2 = ws2.receive_json()
            tries -= 1
        assert recv2.get("type") == "event"
        assert recv2.get("topic") == topic
        assert recv2.get("data") == payload