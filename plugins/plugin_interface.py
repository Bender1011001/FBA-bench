from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

__all__ = [
    "IPlugin",
    "ISkillPlugin",
    "IAnalysisPlugin",
    "IIntegrationPlugin",
]

logger = logging.getLogger(__name__)


@runtime_checkable
class IPlugin(Protocol):
    """
    Base plugin interface used by community and regression tests.

    Required methods inferred from tests/regression/regression_tests.py:
    - initialize
    - activate
    - deactivate
    - cleanup
    - get_metadata
    """

    async def initialize(self, context: Dict[str, Any]) -> bool:
        """
        Initialize the plugin with a context.
        Returns True when initialization succeeds.
        """
        ...

    async def activate(self) -> bool:
        """Activate the plugin and return True if successful."""
        ...

    async def deactivate(self) -> bool:
        """Deactivate the plugin and return True if successful."""
        ...

    def cleanup(self) -> Any:
        """Cleanup resources. May return an optional status/value."""
        ...

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return plugin metadata.
        Tests expect at least: name, version, (optionally author/description).
        """
        ...


@runtime_checkable
class ISkillPlugin(IPlugin, Protocol):
    """
    Skill plugin interface.

    Regression tests additionally expect:
    - handle_event
    - get_supported_event_types
    """

    async def handle_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process an event and optionally return a result."""
        ...

    def get_supported_event_types(self) -> List[str]:
        """Return a list of supported event type identifiers."""
        ...


@runtime_checkable
class IAnalysisPlugin(IPlugin, Protocol):
    """
    Analysis plugin interface.

    Not strictly exercised by the current tests beyond import/type hints,
    but provided for completeness.
    """

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run an analysis on the provided data and return structured results."""
        ...


@runtime_checkable
class IIntegrationPlugin(IPlugin, Protocol):
    """
    Integration plugin interface.

    Provided to satisfy imports in community/extensibility tests.
    """

    async def integrate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform an integration action and return results."""
        ...