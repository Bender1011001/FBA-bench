from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Iterable
from uuid import UUID

try:
    # Preferred project logger
    from fba_bench.core.logging import setup_logging  # noqa: F401
    logger = logging.getLogger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

# External, already in your repo (lifespan populates these)
from services.dashboard_api_service import DashboardAPIService
from fba_events.bus import EventBus  # type: ignore
from fba_bench_api.core.redis_client import get_redis

# --------------------------------------------------------------------------------------
# Backward-compat in-memory dicts (DO NOT REMOVE without refactoring callers)
#   These remain for modules that still directly import the dicts. New code should use
#   StateManager instead of mutating these globals.
# --------------------------------------------------------------------------------------
experiment_configs_db: Dict[str, Dict[str, Any]] = {}
simulation_configs_db: Dict[str, Dict[str, Any]] = {}
templates_db: Dict[str, Dict[str, Any]] = {}

# Long-lived singletons initialized in lifespan
dashboard_service: Optional[DashboardAPIService] = None
active_event_bus: Optional[EventBus] = None


# --------------------------------------------------------------------------------------
# Redis-backed State Manager
# --------------------------------------------------------------------------------------
class StateError(Exception):
    """State layer error (HTTP-agnostic)."""


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _to_json_compatible(value: Any) -> Any:
    """Coerce common non-JSON types to JSON-compatible representations."""
    try:
        # Fast path for already JSON-native types
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [_to_json_compatible(v) for v in value]
        if isinstance(value, dict):
            # Ensure keys are strings; coerce values
            out: Dict[str, Any] = {}
            for k, v in value.items():
                out[str(k)] = _to_json_compatible(v)
            return out
        if isinstance(value, datetime):
            # Always store as UTC ISO string
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()
        if isinstance(value, UUID):
            return str(value)
        # Fallback to object-provided JSON conversion
        if hasattr(value, "to_json") and callable(getattr(value, "to_json")):
            return _to_json_compatible(value.to_json())
    except Exception as exc:
        logger.debug("JSON compatibility conversion failed quick path: %s", exc)

    # Last resort
    try:
        return str(value)
    except Exception:
        # As absolute fallback, return repr
        return repr(value)


def _dumps(value: Any) -> str:
    try:
        return json.dumps(_to_json_compatible(value), ensure_ascii=False, separators=(",", ":"))
    except Exception as exc:
        logger.error("State serialization error: %s", exc, exc_info=True)
        raise StateError(f"Failed to serialize value to JSON: {exc}") from exc


def _loads(text: Optional[str]) -> Any:
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception as exc:
        # If corrupted or not JSON, return None-like behavior to callers via their defaults
        logger.error("State deserialization error: %s; raw=%r", exc, text[:256] if isinstance(text, str) else text)
        raise StateError(f"Failed to deserialize JSON: {exc}") from exc


class StateManager:
    """
    Redis-backed namespaced state access.

    - Keys are stored as: state:{namespace}:{key}
    - JSON serialization with coercion of common Python types
    - Optional TTL per set/incr (TTL set on first creation for counters)
    - Simple pub/sub notifications via publish to state:{namespace}:events
    """

    def __init__(self, namespace: str, redis=None) -> None:
        self.namespace = namespace
        self._redis = redis  # Optional injected client (useful in tests)

    @property
    def channel(self) -> str:
        return f"state:{self.namespace}:events"

    def _k(self, key: str) -> str:
        if not key or not isinstance(key, str):
            raise StateError("Key must be a non-empty string")
        return f"state:{self.namespace}:{key}"

    async def _client(self):
        if self._redis is not None:
            return self._redis
        # Lazily obtain the singleton client
        return await get_redis()

    # ------------- CRUD -------------

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        k = self._k(key)
        payload = _dumps(value)
        try:
            r = await self._client()
            if ttl_seconds is not None and ttl_seconds > 0:
                await r.set(k, payload, ex=ttl_seconds)
            else:
                await r.set(k, payload)
        except Exception as exc:
            logger.error("Redis set failed key=%s ns=%s: %s", key, self.namespace, exc, exc_info=True)
            raise StateError(f"Failed to set key '{key}': {exc}") from exc

    async def get(self, key: str, default: Any = None) -> Any:
        k = self._k(key)
        try:
            r = await self._client()
            raw = await r.get(k)
            if raw is None:
                return default
            return _loads(raw)
        except StateError:
            # Deserialize error -> return default
            return default
        except Exception as exc:
            logger.error("Redis get failed key=%s ns=%s: %s", key, self.namespace, exc, exc_info=True)
            return default

    async def delete(self, key: str) -> bool:
        k = self._k(key)
        try:
            r = await self._client()
            res = await r.delete(k)
            return bool(res == 1)
        except Exception as exc:
            logger.error("Redis delete failed key=%s ns=%s: %s", key, self.namespace, exc, exc_info=True)
            raise StateError(f"Failed to delete key '{key}': {exc}") from exc

    async def keys(self, pattern: str = "*") -> list[str]:
        """List keys within namespace, returning unprefixed logical keys."""
        ns_prefix = f"state:{self.namespace}:"
        match = f"{ns_prefix}{pattern}"
        results: list[str] = []
        try:
            r = await self._client()
            # Use scan to avoid blocking Redis
            cursor = 0
            while True:
                cursor, batch = await r.scan(cursor=cursor, match=match, count=200)
                for full in batch:
                    # redis-py with decode_responses=True returns str
                    if isinstance(full, bytes):
                        full = full.decode("utf-8", errors="ignore")
                    if full.startswith(ns_prefix):
                        results.append(full[len(ns_prefix):])
                if cursor == 0:
                    break
            return sorted(results)
        except Exception as exc:
            logger.error("Redis keys scan failed ns=%s pattern=%s: %s", self.namespace, pattern, exc, exc_info=True)
            raise StateError(f"Failed to scan keys for pattern '{pattern}': {exc}") from exc

    async def exists(self, key: str) -> bool:
        try:
            r = await self._client()
            return bool(await r.exists(self._k(key)))
        except Exception as exc:
            logger.error("Redis exists failed key=%s ns=%s: %s", key, self.namespace, exc, exc_info=True)
            raise StateError(f"Failed to check existence for '{key}': {exc}") from exc

    async def clear(self) -> int:
        """Delete all keys under this namespace. Returns number deleted."""
        total_deleted = 0
        try:
            r = await self._client()
            ns_prefix = f"state:{self.namespace}:"
            cursor = 0
            to_delete: list[str] = []
            while True:
                cursor, batch = await r.scan(cursor=cursor, match=f"{ns_prefix}*", count=500)
                to_delete.extend(batch)
                # Delete in chunks to avoid large payloads
                if len(to_delete) >= 500 or (cursor == 0 and to_delete):
                    # redis DEL variadic
                    deleted = await r.delete(*to_delete)
                    total_deleted += int(deleted or 0)
                    to_delete.clear()
                if cursor == 0:
                    break
            return total_deleted
        except Exception as exc:
            logger.error("Redis clear namespace failed ns=%s: %s", self.namespace, exc, exc_info=True)
            raise StateError(f"Failed to clear namespace '{self.namespace}': {exc}") from exc

    # ------------- Counters -------------

    async def incr(self, key: str, amount: int = 1, ttl_seconds: Optional[int] = None) -> int:
        """Atomically increment an integer counter. If newly created and ttl provided, set expire."""
        k = self._k(key)
        try:
            r = await self._client()
            val = await r.incrby(k, amount)
            # If this was the first creation and ttl provided, set expire if not already set
            if ttl_seconds is not None and ttl_seconds > 0:
                # Only set expire if key has no ttl (-1 means no expire; -2 means missing)
                ttl = await r.ttl(k)
                if ttl == -1:
                    await r.expire(k, ttl_seconds)
            return int(val)
        except Exception as exc:
            logger.error("Redis incr failed key=%s ns=%s: %s", key, self.namespace, exc, exc_info=True)
            raise StateError(f"Failed to increment key '{key}': {exc}") from exc

    # ------------- Notifications -------------

    async def notify(self, event: str, payload: Dict[str, Any]) -> int:
        """Publish a change notification to the namespace event channel."""
        if not event or not isinstance(event, str):
            raise StateError("Event must be a non-empty string")
        message = {
            "event": event,
            "payload": _to_json_compatible(payload or {}),
            "ts": _now_iso(),
        }
        try:
            r = await self._client()
            count = await r.publish(self.channel, _dumps(message))
            return int(count or 0)
        except Exception as exc:
            logger.error("Redis notify failed ns=%s event=%s: %s", self.namespace, event, exc, exc_info=True)
            # Notifications are optional; do not raise unless caller depends on it
            return 0


# --------------------------------------------------------------------------------------
# Utility: bulk set helper (optional)
# --------------------------------------------------------------------------------------
async def set_many(sm: StateManager, items: Iterable[tuple[str, Any]], ttl_seconds: Optional[int] = None) -> None:
    """Best-effort bulk setter using pipelining."""
    try:
        r = await sm._client()
        async with r.pipeline(transaction=False) as pipe:
            for k, v in items:
                full = sm._k(k)
                payload = _dumps(v)
                if ttl_seconds is not None and ttl_seconds > 0:
                    pipe.set(full, payload, ex=ttl_seconds)
                else:
                    pipe.set(full, payload)
            await pipe.execute()
    except Exception as exc:
        logger.error("Bulk set failed ns=%s: %s", sm.namespace, exc, exc_info=True)
        raise StateError(f"Bulk set failed: {exc}") from exc


# --------------------------------------------------------------------------------------
# Simple self-check to avoid noisy un-awaited tasks on import in sync contexts
# --------------------------------------------------------------------------------------
def _ensure_event_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None