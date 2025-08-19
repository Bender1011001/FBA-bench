import os
import asyncio
from typing import Any, Dict

import pytest

from fba_events.bus import InMemoryEventBus


class _ManyEvent:
    """Simple event without explicit summary, exercises generic jsonify path."""
    def __init__(self, i: int) -> None:
        self.i = i


class _SensitiveEvent:
    """Event that exposes a summary dict with sensitive fields for redaction test."""
    def __init__(self) -> None:
        self.payload = {"api_key": "abc", "nested": {"token": "xyz"}, "ok": "value"}

    def to_summary_dict(self) -> Dict[str, Any]:
        return {"api_key": "abc", "nested": {"token": "xyz"}, "ok": "value"}


@pytest.mark.asyncio
async def test_recording_cap_and_truncation(tmp_path, monkeypatch):
    # Set low cap for fast test
    monkeypatch.setenv("EVENT_RECORDING_MAX", "10")

    bus = InMemoryEventBus()
    await bus.start()
    await bus.start_recording()

    # Publish more than cap
    N = 25
    for i in range(N):
        await bus.publish(_ManyEvent(i))

    # Allow runner to drain queue
    await asyncio.sleep(0.05)

    recorded = await bus.get_recorded_events()
    stats = bus.get_recording_stats()

    assert len(recorded) == 10, "Recorded count must be capped at EVENT_RECORDING_MAX"
    assert stats["count"] == 10
    assert stats["truncated"] is True
    assert stats["enabled"] is True
    assert stats["max"] == 10

    await bus.stop()


@pytest.mark.asyncio
async def test_redaction_policy_for_sensitive_fields(monkeypatch):
    # Ensure deterministic cap not interfering
    monkeypatch.setenv("EVENT_RECORDING_MAX", "100")

    bus = InMemoryEventBus()
    await bus.start()
    await bus.start_recording()

    evt = _SensitiveEvent()
    await bus.publish(evt)

    await asyncio.sleep(0.02)

    rec = await bus.get_recorded_events()
    assert len(rec) >= 1

    # Find the sensitive event by type
    redacted = None
    for r in rec[::-1]:
        if r.get("event_type") == "_SensitiveEvent":
            redacted = r
            break

    assert redacted is not None, "Sensitive event record not found"
    data = redacted["data"]

    # Verify redaction
    assert data["api_key"] == "[redacted]"
    assert isinstance(data["nested"], dict)
    assert data["nested"]["token"] == "[redacted]"
    # Non-sensitive fields preserved
    assert data["ok"] == "value"

    # Original event must remain unmodified
    assert evt.payload["api_key"] == "abc"
    assert evt.payload["nested"]["token"] == "xyz"
    assert evt.payload["ok"] == "value"

    await bus.stop()