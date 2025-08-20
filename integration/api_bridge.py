from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class APIEndpoint:
    """
    Minimal endpoint descriptor used by tests.

    Fields provided:
    - name: logical name (e.g., "set_price", "get_inventory")
    - path: string path (for completeness)
    - method: HTTP-like method ("GET","POST", etc.), not strictly enforced here
    - version: semantic version string
    - rate_limit_per_minute: integer throttle per minute (enforced by RateLimiter)
    """
    name: str
    path: str
    method: str
    version: str = "1.0.0"
    rate_limit_per_minute: int = 60
    metadata: Dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """
    Simple token bucket style limiter per key with 1-minute windows.

    Methods:
    - allow(key: str, limit_per_minute: int) -> bool
    - reset() -> None
    """
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._window_start: float = monotonic()
        self._counts: Dict[str, int] = {}

    def allow(self, key: str, limit_per_minute: int) -> bool:
        now = monotonic()
        with self._lock:
            # Reset counts each minute
            if now - self._window_start >= 60.0:
                self._window_start = now
                self._counts.clear()
            self._counts[key] = self._counts.get(key, 0) + 1
            return self._counts[key] <= max(1, int(limit_per_minute))

    def reset(self) -> None:
        with self._lock:
            self._window_start = monotonic()
            self._counts.clear()


class APIBridge:
    """
    Lightweight API bridge registry and invoker used in integration tests.

    Capabilities:
    - register_endpoint(endpoint: APIEndpoint) -> None
    - get_endpoint(name: str) -> Optional[APIEndpoint]
    - async request(name: str, **params) -> dict
      Enforces endpoint-specific rate limits and returns a normalized dict.

    NOTE: The bridge does not perform real HTTP; it standardizes and throttles calls.
    """
    def __init__(self, rate_limiter: Optional[RateLimiter] = None) -> None:
        self._endpoints: Dict[str, APIEndpoint] = {}
        self._rate_limiter = rate_limiter or RateLimiter()

    def register_endpoint(self, endpoint: APIEndpoint) -> None:
        self._endpoints[endpoint.name] = endpoint
        logger.info("Registered API endpoint: %s %s %s", endpoint.method, endpoint.path, endpoint.version)

    def get_endpoint(self, name: str) -> Optional[APIEndpoint]:
        return self._endpoints.get(name)

    async def request(self, name: str, **params: Any) -> Dict[str, Any]:
        """
        Perform a logical request to an endpoint. Applies rate limiting and returns a simple response dict.
        """
        ep = self._endpoints.get(name)
        if not ep:
            raise ValueError(f"Unknown endpoint: {name}")

        key = f"{ep.name}:{ep.version}"
        if not self._rate_limiter.allow(key, ep.rate_limit_per_minute):
            # Simulate rate limit exceeded
            logger.warning("Rate limit exceeded for endpoint %s", name)
            return {"success": False, "error": "rate_limit_exceeded", "endpoint": name}

        # For test friendliness, echo back normalized structure
        response = {
            "success": True,
            "endpoint": name,
            "path": ep.path,
            "method": ep.method.upper(),
            "version": ep.version,
            "request": params,
        }
        return response

    async def configure_rate_limit(self, name: str, per_minute: int) -> bool:
        ep = self._endpoints.get(name)
        if not ep:
            return False
        # dataclasses are frozen? Our APIEndpoint is frozen via dataclass default? It is not frozen, so can mutate.
        self._endpoints[name] = APIEndpoint(
            name=ep.name,
            path=ep.path,
            method=ep.method,
            version=ep.version,
            rate_limit_per_minute=int(per_minute),
            metadata=dict(ep.metadata),
        )
        return True

    def reset_limits(self) -> None:
        self._rate_limiter.reset()


__all__ = [
    "APIBridge",
    "APIEndpoint",
    "RateLimiter",
]