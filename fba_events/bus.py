from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple, Union, Set

logger = logging.getLogger(__name__)

Handler = Callable[[Any], Awaitable[None]]
Selector = Union[type, str]
SubscriptionHandle = Tuple[Selector, Handler]


class EventBus:
    """
    Async-first event bus contract.

    This base class defines the canonical async API. Implementations should be fully async and
    safe for concurrent usage from multiple tasks.

    Methods:
    - publish(event): enqueue and dispatch to matching subscribers
    - subscribe(event_selector, handler): register handler for an event class or string name
    - unsubscribe(handle): remove a previously registered handler
    - start(): start background dispatch loop(s) if required
    - stop(): stop background dispatch loop(s) gracefully
    - start_recording(): enable in-memory event recording
    - get_recorded_events(): retrieve a stable list of summarized recorded events

    Selectors:
    - event class/type: dispatches via isinstance(event, selector)
    - string event name: dispatches when the event's canonical type string matches

    Handlers:
    - async callable(event) -> None (preferred)
    - sync callable(event) -> None (will be wrapped in an async shim)

    Examples:
    - Subscribe to a class:
        async def on_tick(evt): ...
        handle = await bus.subscribe(TickEvent, on_tick)

    - Publish an event:
        await bus.publish(TickEvent(...))

    - Recording:
        await bus.start_recording()
        recorded = await bus.get_recorded_events()
    """

    async def publish(self, event: Any) -> None:
        raise NotImplementedError

    async def subscribe(self, event_selector: Any, handler: Any) -> Any:
        raise NotImplementedError

    async def unsubscribe(self, handle: Any) -> None:
        raise NotImplementedError

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def start_recording(self) -> None:
        raise NotImplementedError

    async def get_recorded_events(self) -> List[Dict[str, Any]]:
        raise NotImplementedError


class InMemoryEventBus(EventBus):
    """
    In-memory asyncio-based EventBus implementation.

    Features:
    - asyncio.Queue staging of events
    - Background runner task to drain the queue
    - Concurrent handler execution per event via asyncio.create_task
    - Resilient to handler errors (logged, do not stop the bus)
    - Optional in-memory recording of summarized events for observability

    Recording schema (stable):
    { "event_type": str, "timestamp": str, "data": dict }
    """

    def __init__(self) -> None:
        # Subscribers keyed by either event class or string name
        self._subscribers: MutableMapping[Selector, List[Handler]] = {}
        self._queue: "asyncio.Queue[Tuple[Any, str, str]]" = asyncio.Queue()
        self._runner_task: Optional[asyncio.Task] = None

        # Recording controls (defaults hardened for production-safety)
        self._recording_enabled: bool = False  # Default OFF
        self._recorded: List[Dict[str, Any]] = []
        # Cap for recorded events; default 5000, configurable via env
        self._recording_max: int = self._read_recording_max()
        self._recording_truncated: bool = False

        # Pre-compiled redaction configuration
        self._redact_key_patterns: List[re.Pattern] = [
            re.compile(pat, re.IGNORECASE)
            for pat in ["password", "api_key", "token", "secret", "authorization", "cookie"]
        ]

        self._started: bool = False

    async def start(self) -> None:
        if self._started:
            return
        self._runner_task = asyncio.create_task(self._runner(), name="InMemoryEventBusRunner")
        self._started = True
        logger.debug("InMemoryEventBus started")

    async def stop(self) -> None:
        if not self._started:
            return
        if self._runner_task:
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.exception("Error while stopping InMemoryEventBus runner: %s", e)
        self._runner_task = None
        self._started = False
        logger.debug("InMemoryEventBus stopped")

    async def publish(self, event: Any) -> None:
        """
        Enqueue an event for dispatch with resolved type string and ISO-8601 timestamp.
        """
        event_type = self._event_type_name(event)
        ts = datetime.now(timezone.utc).isoformat()
        await self._queue.put((event, event_type, ts))

    async def subscribe(self, event_selector: Any, handler: Any) -> SubscriptionHandle:
        """
        Register a handler for a selector (class or string). Returns a handle for unsubscribe.
        If a sync handler is provided, it is wrapped in an async shim.
        """
        if not callable(handler):
            raise TypeError("handler must be callable")

        async_handler: Handler = self._wrap_handler(handler)
        sel: Selector = self._normalize_selector(event_selector)
        self._subscribers.setdefault(sel, []).append(async_handler)
        return (sel, async_handler)

    async def unsubscribe(self, handle: SubscriptionHandle) -> None:
        """
        Remove a previously registered handler. If the handle is unknown, no-op.
        """
        sel, async_handler = handle
        handlers = self._subscribers.get(sel)
        if not handlers:
            return
        try:
            handlers.remove(async_handler)
            if not handlers:
                # Drop empty lists to keep mapping tidy
                self._subscribers.pop(sel, None)
        except ValueError:
            # Already removed or never registered; ignore
            pass

    async def start_recording(self) -> None:
        # Enable recording; do not clear existing buffer to avoid losing prior diagnostics
        self._recording_enabled = True

    async def get_recorded_events(self) -> List[Dict[str, Any]]:
        # Return a shallow copy to ensure stability for callers
        # Non-blocking, respects cap (we stop appending once cap reached)
        return list(self._recorded)

    # -------------------------
    # Internal implementation
    # -------------------------

    async def _runner(self) -> None:
        try:
            while True:
                event, event_type, ts = await self._queue.get()
                # Collect handlers for class selectors (isinstance) and string selectors
                handlers = self._matching_handlers(event, event_type)
                # Record before dispatch to ensure full audit trail even if handlers fail
                if self._recording_enabled:
                    try:
                        if len(self._recorded) < self._recording_max:
                            # Build summary and redact sensitive fields
                            summary = self._event_to_summary(event)
                            safe_summary = self._redact_sensitive(summary)
                            self._recorded.append({
                                "event_type": event_type,
                                "timestamp": ts,
                                "data": safe_summary,
                            })
                        else:
                            # Cap reached; mark truncated and stop appending
                            if not self._recording_truncated:
                                self._recording_truncated = True
                    except Exception as rec_e:
                        # Defensive: never crash the bus due to recording failure
                        logger.warning("Failed to record event %s: %s", event_type, rec_e)
                        try:
                            # Append minimal error record if capacity allows
                            if len(self._recorded) < self._recording_max:
                                self._recorded.append({
                                    "event_type": event_type,
                                    "timestamp": ts,
                                    "data": {"_error": "recording_failed"},
                                })
                            else:
                                if not self._recording_truncated:
                                    self._recording_truncated = True
                        except Exception:
                            # As a last resort, swallow errors silently to keep dispatching
                            pass

                # Dispatch concurrently
                for h in handlers:
                    asyncio.create_task(self._safe_invoke(h, event), name=f"EventHandler[{event_type}]")
        except asyncio.CancelledError:
            # Graceful exit
            return
        except Exception as e:
            logger.exception("Unhandled exception in InMemoryEventBus runner: %s", e)
            # Continue loop even on unexpected errors
            await asyncio.sleep(0.01)
            asyncio.create_task(self._runner(), name="InMemoryEventBusRunner-Restarted")

    def _matching_handlers(self, event: Any, event_type: str) -> List[Handler]:
        matched: List[Handler] = []

        # String keys
        str_handlers = self._subscribers.get(event_type)
        if str_handlers:
            matched.extend(str_handlers)

        # Class keys: iterate and isinstance
        for key, handlers in self._subscribers.items():
            if isinstance(key, str):
                continue
            try:
                if isinstance(event, key):  # type: ignore[arg-type]
                    matched.extend(handlers)
            except Exception:
                # Non-type keys or unexpected selector; ignore
                continue

        return matched

    async def _safe_invoke(self, handler: Handler, event: Any) -> None:
        try:
            await handler(event)
        except Exception as e:
            logger.error("Event handler error for %s: %s", self._event_type_name(event), e, exc_info=True)

    def _wrap_handler(self, handler: Any) -> Handler:
        if asyncio.iscoroutinefunction(handler):
            return handler  # type: ignore[return-value]

        async def _shim(evt: Any) -> None:
            loop = asyncio.get_running_loop()
            # Execute sync handler in default executor to avoid blocking
            await loop.run_in_executor(None, handler, evt)

        return _shim

    def _event_type_name(self, event: Any) -> str:
        """
        Resolve canonical event type string.
        Preference: event.__class__.__name__, else getattr(event, 'event_type', str)
        """
        try:
            return event.__class__.__name__
        except Exception:
            pass
        try:
            et = getattr(event, "event_type", None)
            if isinstance(et, str):
                return et
        except Exception:
            pass
        return str(type(event))

    def _event_to_summary(self, event: Any) -> Dict[str, Any]:
        """
        Best-effort conversion to a JSON-serializable summary dict.
        Priority:
        - event.to_summary_dict() if available
        - dataclasses.asdict if dataclass
        - vars(event) filtered/converted to JSON-serializable primitives
        """
        # 1) Explicit summary if provided
        try:
            to_sum = getattr(event, "to_summary_dict", None)
            if callable(to_sum):
                data = to_sum()
                if isinstance(data, dict):
                    return self._jsonify_dict(data)
        except Exception:
            pass

        # 2) Dataclass fallback
        try:
            if is_dataclass(event):
                return self._jsonify_dict(asdict(event))
        except Exception:
            pass

        # 3) Generic object __dict__ fallback
        try:
            d = vars(event)
            if isinstance(d, dict):
                return self._jsonify_dict(d)
        except Exception:
            pass

        # 4) Last resort: string representation
        return {"repr": repr(event)}

    def _jsonify_dict(self, d: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in d.items():
            out[k] = self._to_jsonable(v)
        return out

    def _to_jsonable(self, v: Any) -> Any:
        # Primitives
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        # Datetime -> ISO
        if isinstance(v, datetime):
            # Preserve timezone if present; assume UTC otherwise
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc).isoformat()
            return v.isoformat()
        # Dict
        if isinstance(v, dict):
            return {str(self._to_jsonable(k)): self._to_jsonable(val) for k, val in v.items()}
        # List/Tuple
        if isinstance(v, (list, tuple)):
            return [self._to_jsonable(i) for i in v]
        # Dataclasses
        if is_dataclass(v):
            try:
                return self._jsonify_dict(asdict(v))
            except Exception:
                return str(v)
        # Money or other custom types -> str()
        try:
            return str(v)
        except Exception:
            return repr(v)

    # -------------------------
    # Recording configuration & helpers
    # -------------------------
    def _read_recording_max(self) -> int:
        default_max = 5000
        try:
            val = os.getenv("EVENT_RECORDING_MAX", str(default_max)).strip()
            if not val:
                return default_max
            parsed = int(val)
            return parsed if parsed > 0 else default_max
        except Exception:
            return default_max

    def get_recording_stats(self) -> Dict[str, Any]:
        """
        Read-only recording stats.
        Returns: {"enabled": bool, "count": int, "truncated": bool, "max": int}
        """
        return {
            "enabled": self._recording_enabled,
            "count": len(self._recorded),
            "truncated": self._recording_truncated,
            "max": self._recording_max,
        }

    def _redact_sensitive(self, data: Any, *, max_depth: int = 20, _depth: int = 0, _seen: Optional[Set[int]] = None) -> Any:
        """
        Return a deep-copied, redacted version of data.
        Redacts values for keys matching common sensitive names (case-insensitive).
        Handles dicts/lists/tuples safely, avoids cycles by tracking visited ids.
        """
        if _seen is None:
            _seen = set()

        # Depth guard
        if _depth > max_depth:
            return data  # Stop traversal; return as-is (already jsonified primitives expected)

        # Prevent cycles
        obj_id = id(data)
        if obj_id in _seen:
            return data
        _seen.add(obj_id)

        # Primitives remain as-is
        if data is None or isinstance(data, (bool, int, float, str)):
            return data

        # Dict: redact keys
        if isinstance(data, dict):
            redacted: Dict[Any, Any] = {}
            for k, v in data.items():
                key_str = str(k)
                value = v
                if any(pat.search(key_str) for pat in self._redact_key_patterns):
                    redacted[key_str] = "[redacted]"
                else:
                    redacted[key_str] = self._redact_sensitive(v, max_depth=max_depth, _depth=_depth + 1, _seen=_seen)
            return redacted

        # List/Tuple
        if isinstance(data, (list, tuple)):
            return [self._redact_sensitive(i, max_depth=max_depth, _depth=_depth + 1, _seen=_seen) for i in data]

        # Fallback: keep string representation (should be rare after _to_jsonable)
        try:
            return str(data)
        except Exception:
            return repr(data)