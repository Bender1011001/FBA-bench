"""
Deprecated module: agent_runners.runner_factory

This module has been removed. Import from agent_runners.registry instead.

Migration:
- from agent_runners.registry import create_runner, supported_runners

Example:
- runner = create_runner("crewai", {"model": "gpt-4o-mini"})
"""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

try:
    registry = importlib.import_module("agent_runners.registry")
    supported = getattr(registry, "supported_runners", lambda: [])()
except Exception:
    supported = []

msg = (
    "runner_factory is deprecated and has been removed.\n"
    "Use agent_runners.registry:create_runner instead.\n"
    f"Supported keys: {', '.join(supported) if supported else '(unknown)'}"
)

raise RuntimeError(msg)

