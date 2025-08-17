from __future__ import annotations

"""
agent_runners.compat - Temporary event compatibility layer.

This module provides thin passthroughs to event types used by some code paths and tests,
without importing deprecated monolithic modules. If the canonical fba_events package is
available, its definitions are re-exported. Otherwise, structural Protocol fallbacks are
declared to satisfy type-checking and runtime attribute access.

Exports:
- TickEvent
- SetPriceCommand
"""

from typing import Any, Protocol, runtime_checkable

# Prefer canonical-first via fba_events.compat (emits DeprecationWarning) to maintain legacy paths temporarily.
try:
    from fba_events.compat import TickEvent as TickEvent  # type: ignore
    from fba_events.compat import SetPriceCommand as SetPriceCommand  # type: ignore
except Exception:
    # Fallback to top-level canonical exports
    try:
        from fba_events import TickEvent as TickEvent  # type: ignore
        from fba_events import SetPriceCommand as SetPriceCommand  # type: ignore
    except Exception:
        # Fall back to specific submodules if import-all is constrained
        try:
            from fba_events.time_events import TickEvent as TickEvent  # type: ignore
        except Exception:
            @runtime_checkable
            class TickEvent(Protocol):
                """Minimal structural protocol for a simulation tick event."""
                tick_number: int

                # Optional fields sometimes accessed by callers
                # timestamp: datetime
                # type: str

        try:
            from fba_events.pricing import SetPriceCommand as SetPriceCommand  # type: ignore
        except Exception:
            @runtime_checkable
            class SetPriceCommand(Protocol):
                """Minimal structural protocol for a SetPriceCommand used in tests."""
                event_id: str
                agent_id: str
                asin: str
                new_price: Any  # Can be Money or float depending on caller context

                # Optional fields sometimes present
                # timestamp: datetime
                # reason: str


__all__ = ["TickEvent", "SetPriceCommand"]