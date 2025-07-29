"""EventBus implementation for FBA-Bench v3 event-driven architecture."""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional, Type, Union
from datetime import datetime
from abc import ABC, abstractmethod

from events import BaseEvent, EVENT_TYPES


logger = logging.getLogger(__name__)


class EventBusBackend(ABC):
    """Abstract base class for EventBus backends."""
    
    @abstractmethod
    async def publish(self, event: BaseEvent) -> None:
        """Publish an event to the bus."""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: str, callback: Callable[[BaseEvent], None]) -> None:
        """Subscribe to events of a specific type."""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the backend."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the backend."""
        pass


class AsyncioQueueBackend(EventBusBackend):
    """AsyncIO Queue-based EventBus backend for in-memory event processing."""
    
    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize the asyncio queue backend.
        
        Args:
            max_queue_size: Maximum number of events in queue before blocking
        """
        self.max_queue_size = max_queue_size
        self._queue: Optional[asyncio.Queue] = None
        self._subscribers: Dict[str, List[Callable[[BaseEvent], None]]] = {}
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0,
            'subscribers_count': 0
        }
    
    async def start(self) -> None:
        """Start the event processing loop."""
        if self._running:
            logger.warning("EventBus backend already running")
            return
        
        self._queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("AsyncioQueueBackend started")
    
    async def stop(self) -> None:
        """Stop the event processing loop."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel the processor task
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        # Clear the queue
        if self._queue:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        logger.info("AsyncioQueueBackend stopped")
    
    async def publish(self, event: BaseEvent) -> None:
        """Publish an event to the queue."""
        if not self._running or not self._queue:
            raise RuntimeError("EventBus backend not started")
        
        try:
            await self._queue.put(event)
            self._stats['events_published'] += 1
            logger.debug(f"Published event: {event.event_id} ({type(event).__name__})")
        except asyncio.QueueFull:
            logger.error(f"Event queue full, dropping event: {event.event_id}")
            raise RuntimeError("Event queue is full")
    
    async def subscribe(self, event_type: str, callback: Callable[[BaseEvent], None]) -> None:
        """Subscribe to events of a specific type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(callback)
        self._stats['subscribers_count'] = sum(len(callbacks) for callbacks in self._subscribers.values())
        logger.info(f"Subscribed to {event_type} events. Total subscribers: {self._stats['subscribers_count']}")
    
    async def _process_events(self) -> None:
        """Process events from the queue and route to subscribers."""
        logger.info("Event processor started")
        
        while self._running:
            try:
                # Wait for an event with timeout to allow graceful shutdown
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._route_event(event)
                self._stats['events_processed'] += 1
                
            except asyncio.TimeoutError:
                # Timeout is normal, continue processing
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
                self._stats['events_failed'] += 1
        
        logger.info("Event processor stopped")
    
    async def _route_event(self, event: BaseEvent) -> None:
        """Route an event to all registered subscribers."""
        event_type = type(event).__name__
        
        if event_type not in self._subscribers:
            logger.debug(f"No subscribers for event type: {event_type}")
            return
        
        callbacks = self._subscribers[event_type]
        logger.debug(f"Routing {event_type} to {len(callbacks)} subscriber(s)")
        
        # Execute all callbacks concurrently
        tasks = []
        for callback in callbacks:
            task = asyncio.create_task(self._execute_callback(callback, event))
            tasks.append(task)
        
        # Wait for all callbacks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_callback(self, callback: Callable[[BaseEvent], None], event: BaseEvent) -> None:
        """Execute a subscriber callback safely."""
        try:
            # Check if callback is async
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                # Run sync callback in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, callback, event)
        except Exception as e:
            logger.error(f"Error executing callback for {type(event).__name__}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backend statistics."""
        return self._stats.copy()


class EventBus:
    """
    Central event bus for FBA-Bench v3 event-driven architecture.
    
    Provides a clean interface for publishing and subscribing to events.
    Designed to be easily swappable between different backends (asyncio.Queue, RabbitMQ, etc.).
    """
    
    def __init__(self, backend: Optional[EventBusBackend] = None):
        """
        Initialize the EventBus.
        
        Args:
            backend: Backend implementation (defaults to AsyncioQueueBackend)
        """
        self.backend = backend or AsyncioQueueBackend()
        self._started = False
    
    async def start(self) -> None:
        """Start the event bus."""
        if self._started:
            logger.warning("EventBus already started")
            return
        
        await self.backend.start()
        self._started = True
        logger.info("EventBus started")
    
    async def stop(self) -> None:
        """Stop the event bus."""
        if not self._started:
            return
        
        await self.backend.stop()
        self._started = False
        logger.info("EventBus stopped")
    
    async def publish(self, event: BaseEvent) -> None:
        """
        Publish an event to the bus.
        
        Args:
            event: Event instance to publish
            
        Raises:
            RuntimeError: If EventBus is not started
            TypeError: If event is not a BaseEvent instance
        """
        if not self._started:
            raise RuntimeError("EventBus not started. Call start() first.")
        
        if not isinstance(event, BaseEvent):
            raise TypeError(f"Event must be a BaseEvent instance, got {type(event)}")
        
        await self.backend.publish(event)
    
    async def subscribe(
        self, 
        event_type: Union[str, Type[BaseEvent]], 
        callback: Callable[[BaseEvent], None]
    ) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Event type name (string) or event class
            callback: Callback function to execute when event is received.
                     Can be sync or async function.
                     
        Raises:
            RuntimeError: If EventBus is not started
            ValueError: If event_type is not valid
        """
        if not self._started:
            raise RuntimeError("EventBus not started. Call start() first.")
        
        # Convert class to string if needed
        if isinstance(event_type, type):
            if not issubclass(event_type, BaseEvent):
                raise ValueError(f"Event type must be a BaseEvent subclass, got {event_type}")
            event_type_str = event_type.__name__
        elif isinstance(event_type, str):
            if event_type not in EVENT_TYPES:
                raise ValueError(f"Unknown event type: {event_type}")
            event_type_str = event_type
        else:
            raise ValueError(f"event_type must be string or BaseEvent class, got {type(event_type)}")
        
        await self.backend.subscribe(event_type_str, callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        backend_stats = self.backend.get_stats()
        return {
            'started': self._started,
            'backend_type': type(self.backend).__name__,
            **backend_stats
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Singleton instance for global access
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def set_event_bus(event_bus: EventBus) -> None:
    """Set the global event bus instance."""
    global _global_event_bus
    _global_event_bus = event_bus