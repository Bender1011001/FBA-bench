from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class APIVersion:
    """
    Lightweight semantic version wrapper used by tests.
    Accepts strings like "1.0.0" and performs simple tuple comparison.
    """
    raw: str

    def __post_init__(self) -> None:
        # Validate basic semver shape (major.minor[.patch])
        parts = self.raw.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid version '{self.raw}' - expected at least major.minor")
        # Ensure numeric-ish, but keep original string for display
        for p in parts:
            if not p.isdigit():
                # Allow cases like "1.0.0-beta" by stripping suffix after '-'
                core = p.split("-")[0]
                if not core.isdigit():
                    raise ValueError(f"Invalid version component '{p}' in '{self.raw}'")

    @property
    def tuple(self) -> Tuple[int, int, int]:
        parts = self.raw.split(".")
        # Normalize to 3-length tuple
        major = int(parts[0].split("-")[0])
        minor = int(parts[1].split("-")[0]) if len(parts) > 1 else 0
        patch = int(parts[2].split("-")[0]) if len(parts) > 2 else 0
        return (major, minor, patch)

    def __str__(self) -> str:
        return self.raw


@dataclass
class Endpoint:
    path: str
    version: APIVersion
    methods: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EndpointRegistry:
    """
    In-memory endpoint registry used by community tests.
    Supports async registration and simple lookup.
    """

    def __init__(self) -> None:
        self._endpoints: Dict[Tuple[str, Tuple[int, int, int]], Endpoint] = {}

    async def register_endpoint(self, path: str, version: APIVersion, methods: List[str]) -> bool:
        if not path.startswith("/"):
            logger.error("Endpoint path must start with '/': %s", path)
            return False
        methods_norm = [m.upper() for m in methods]
        key = (path, version.tuple)
        self._endpoints[key] = Endpoint(path=path, version=version, methods=methods_norm)
        logger.info("Registered endpoint %s %s with methods %s", path, version.raw, methods_norm)
        return True

    def get_endpoint(self, path: str, version: Optional[APIVersion] = None) -> Optional[Endpoint]:
        if version is not None:
            return self._endpoints.get((path, version.tuple))
        # Return highest version if not specified
        candidates = [(ver, ep) for (p, ver), ep in self._endpoints.items() if p == path]
        if not candidates:
            return None
        _, ep = max(candidates, key=lambda it: it[0])
        return ep

    def list_endpoints(self) -> List[Endpoint]:
        return [ep for _, ep in sorted(self._endpoints.items(), key=lambda it: (it[0][0], it[0][1]))]


class APIGateway:
    """
    Facade for endpoint registration and (future) routing.
    Community tests only require that the registry operations succeed.
    """

    def __init__(self, registry: Optional[EndpointRegistry] = None) -> None:
        self._registry = registry or EndpointRegistry()

    async def register(self, path: str, version: str, methods: List[str]) -> bool:
        """Helper to register using string version."""
        try:
            return await self._registry.register_endpoint(path, APIVersion(version), methods)
        except Exception as e:
            logger.error("Failed to register endpoint %s@%s: %s", path, version, e)
            return False

    @property
    def registry(self) -> EndpointRegistry:
        return self._registry