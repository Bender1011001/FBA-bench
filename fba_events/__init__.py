"""fba_events: split modules for FBA-Bench v3 event schema."""
from __future__ import annotations
from .registry import *  # noqa: F401,F403
# Ergonomic re-export for canonical in-memory event bus
from .bus import InMemoryEventBus as InMemoryEventBus  # noqa: F401
