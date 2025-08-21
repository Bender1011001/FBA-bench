import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

import pytest

from fba_events.bus import InMemoryEventBus as EventBus
try:
    # Prefer compat to be resilient to package layout
    from fba_events.compat import TickEvent  # type: ignore
except Exception:  # pragma: no cover
    from fba_events.time_events import TickEvent  # type: ignore


def _make_tick_event(n: int, meta: Dict[str, Any] | None = None) -> Any:
    meta = meta or {}
    # Include sensitive-looking keys to verify redaction
    sensitive_meta = {
        "api_key": "super-secret",
        "nested": {"token": "very-secret"},
        "ok": "value",
    }
    sensitive_meta.update(meta)
    return TickEvent(
        event_id=f"tick_{n}",
        timestamp=datetime.now(),
        tick_number=n,
        simulation_time=datetime.now(),
        metadata=sensitive_meta,
    )


@pytest.mark.asyncio
async def test_log_event_emits_info(caplog, monkeypatch):
    # Ensure logging is enabled
    monkeypatch.setenv("EVENT_LOGGING_ENABLED", "1")

    bus = EventBus()
    await bus.start()

    with caplog.at_level(logging.INFO, logger="fba_events.bus"):
        evt = _make_tick_event(1)
        await bus.publish(evt)
        # log_event is called synchronously before queueing, so no sleep required

    # Look for our structured log record
    records = [
        r for r in caplog.records
        if r.name == "fba_events.bus" and getattr(r, "msg", "") == "Event published"
    ]
    assert records, "Expected 'Event published' log record"

    # Validate structured extras are present
    found = False
    for r in records:
        if getattr(r, "event_type", "") == "TickEvent":
            # 'event' extra should contain JSON-serializable summary
            logged_event = getattr(r, "event", {})
            assert isinstance(logged_event, dict)
            assert logged_event.get("tick_number") in (1, "1")
            found = True
            break

    assert found, "Expected a record with event_type=TickEvent"

    await bus.stop()


@pytest.mark.asyncio
async def test_log_event_can_be_disabled(caplog, monkeypatch):
    # Disable logging via env
    monkeypatch.setenv("EVENT_LOGGING_ENABLED", "0")

    bus = EventBus()
    await bus.start()

    with caplog.at_level(logging.INFO, logger="fba_events.bus"):
        evt = _make_tick_event(2)
        await bus.publish(evt)

    # Should not emit our "Event published" logs when disabled
    records = [
        r for r in caplog.records
        if r.name == "fba_events.bus" and getattr(r, "msg", "") == "Event published"
    ]
    assert not records, "Event logging should be disabled but a log record was found"

    await bus.stop()


@pytest.mark.asyncio
async def test_log_event_redacts_sensitive_fields(caplog, monkeypatch):
    monkeypatch.setenv("EVENT_LOGGING_ENABLED", "1")

    bus = EventBus()
    await bus.start()

    with caplog.at_level(logging.INFO, logger="fba_events.bus"):
        evt = _make_tick_event(3)
        await bus.publish(evt)

    # Find the most recent "Event published" record
    target = None
    for r in reversed(caplog.records):
        if r.name == "fba_events.bus" and getattr(r, "msg", "") == "Event published":
            target = r
            break

    assert target is not None, "Expected 'Event published' log record"
    logged_event = getattr(target, "event", {})
    assert isinstance(logged_event, dict)

    # Redaction occurs over the summary; metadata should be redacted
    # Depending on summary path, metadata could be nested under 'metadata'
    meta = logged_event.get("metadata") if isinstance(logged_event.get("metadata"), dict) else None
    assert meta is not None, "Expected metadata in logged event summary"
    assert meta.get("api_key") == "[redacted]", "api_key should be redacted"
    assert isinstance(meta.get("nested"), dict)
    assert meta["nested"].get("token") == "[redacted]", "token should be redacted"
    assert meta.get("ok") == "value", "Non-sensitive fields should be preserved"

    await bus.stop()