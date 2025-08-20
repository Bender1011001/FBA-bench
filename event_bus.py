from __future__ import annotations

"""
Legacy event_bus compatibility shim.

This module preserves the historical import path:
  - from event_bus import EventBus, get_event_bus, set_event_bus
  - from event_bus import AsyncioQueueBackend, DistributedBackend

It bridges to the concrete implementations in fba_events.bus to avoid broad refactors.
"""

from typing import Optional

from fba_events.bus import EventBus as _BaseEventBus
from fba_events.bus import InMemoryEventBus as _InMemoryEventBus

# Try to expose BaseEvent symbol for backwards compatibility
try:
    from fba_events.base import BaseEvent as _BaseEvent  # type: ignore
except Exception:
    class _BaseEvent:  # minimal fallback to satisfy imports
        def __init__(self, *args, **kwargs):
            setattr(self, "event_id", kwargs.get("event_id", None))
            setattr(self, "timestamp", kwargs.get("timestamp", None))

# Public aliases to maintain backwards compatibility in code/tests
EventBus = _BaseEventBus
InMemoryEventBus = _InMemoryEventBus
BaseEvent = _BaseEvent

# Historical backends used in tests; mapped to the in-memory implementation
AsyncioQueueBackend = _InMemoryEventBus
DistributedBackend = _InMemoryEventBus

# Singleton holder used by legacy code paths
_bus_singleton: Optional[_BaseEventBus] = None


def get_event_bus() -> _BaseEventBus:
    """
    Return a process-local singleton EventBus instance.

    Historically this returned a configured backend. To keep behavior stable,
    we provide an InMemoryEventBus when no bus has been set explicitly.
    """
    global _bus_singleton
    if _bus_singleton is None:
        _bus_singleton = _InMemoryEventBus()
    return _bus_singleton


def set_event_bus(bus: _BaseEventBus) -> None:
    """
    Explicitly set the process-local EventBus singleton.
    Useful for tests or embedding in other runtimes.
    """
    global _bus_singleton
    _bus_singleton = bus