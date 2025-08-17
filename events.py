"""
Compatibility shim for legacy imports.

Some modules import from 'events' (legacy) while the new event system
lives under 'fba_events'. This module re-exports all public symbols from
fba_events to preserve backward compatibility.

Usage:
    from events import TickEvent, CompetitorPricesUpdated, ...

Implementation detail:
    fba_events/__init__.py already exports the full public API via its registry.
"""
from fba_events import *  # re-export complete public event API