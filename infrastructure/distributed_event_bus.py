from collections import defaultdict, deque
from typing import Callable, Dict, List, Any, Deque, Optional
import threading


__all__ = ["DistributedEventBus", "MockRedisBroker"]


class DistributedEventBus:
    """
    Minimal in-memory pub/sub bus used by infrastructure tests.
    - subscribe(topic, handler)
    - publish(topic, payload)
    Handlers are invoked synchronously in subscription order.
    """

    def __init__(self, broker: Optional[Any] = None) -> None:
        self._broker = broker
        self._subscriptions: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._subscriptions[topic].append(handler)
        if self._broker and hasattr(self._broker, "subscribe"):
            try:
                self._broker.subscribe(topic)
            except TypeError:
                self._broker.subscribe(topic, handler)

    def unsubscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        handlers = self._subscriptions.get(topic)
        if not handlers:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            pass

    def publish(self, topic: str, payload: Any) -> None:
        # Iterate over a snapshot to allow handlers to modify subscriptions safely
        for h in list(self._subscriptions.get(topic, ())):
            h(payload)
        if self._broker and hasattr(self._broker, "publish"):
            self._broker.publish(topic, payload)


class MockRedisBroker:
    """
    Lightweight stand-in for a Redis-based broker used in tests.
    Provides:
      - publish(channel, message)
      - subscribe(channel, handler)
      - get_queue(channel) for test introspection
    Internally uses deques per channel and synchronous dispatch to handlers.
    """

    def __init__(self) -> None:
        self._channels: Dict[str, Deque[Any]] = defaultdict(deque)
        self._handlers: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, channel: str, handler: Callable[[Any], None]) -> None:
        if not callable(handler):
            raise TypeError("handler must be callable")
        with self._lock:
            self._handlers[channel].append(handler)

    def unsubscribe(self, channel: str, handler: Callable[[Any], None]) -> None:
        with self._lock:
            handlers = self._handlers.get(channel)
            if not handlers:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    def publish(self, channel: str, message: Any) -> None:
        with self._lock:
            self._channels[channel].append(message)
            handlers = list(self._handlers.get(channel, ()))
        # Invoke handlers outside lock
        for h in handlers:
            h(message)

    def get_queue(self, channel: str) -> Deque[Any]:
        # For tests to inspect enqueued messages
        return self._channels[channel]

    def pop(self, channel: str) -> Optional[Any]:
        with self._lock:
            if self._channels[channel]:
                return self._channels[channel].popleft()
            return None