"""EventBus implementation for FBA-Bench v3 event-driven architecture."""

import asyncio
import logging
import inspect
import uuid
from typing import Dict, List, Callable, Any, Optional, Type, Union
from datetime import datetime
from abc import ABC, abstractmethod

from events import BaseEvent, EVENT_TYPES # Ensure BaseEvent is imported for type hinting

# OpenTelemetry Imports
from opentelemetry import trace
from instrumentation.tracer import setup_tracing

logger = logging.getLogger(__name__)

# Initialize tracer for EventBus module
event_bus_tracer = setup_tracing(service_name="fba-bench-eventbus")

class EventBusBackend(ABC):
    """Abstract base class for EventBus backends."""
    
    @abstractmethod
    async def publish(self, event: BaseEvent) -> None:
        """Publish an event to the bus."""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: Any, callback: Callable) -> None:
        """Subscribe to events of a specific type (class or string name)."""
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
    """
    AsyncIO Queue-based EventBus backend for in-memory event processing.
    Includes event recording capabilities for golden snapshots.
    """
    
    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize the asyncio queue backend.
        
        Args:
            max_queue_size: Maximum number of events in queue before blocking
        """
        self.max_queue_size = max_queue_size
        self._queue: Optional[asyncio.Queue] = None
        self._subscribers: Dict[str, List[Callable]] = {}
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0,
            'subscribers_count': 0
        }
        
        # Event recording for reproducibility
        self._is_recording = False
        self._recorded_events: List[Dict[str, Any]] = [] # Stores serialized events


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
    
    def start_recording(self) -> None:
        """Starts recording all events that pass through the bus."""
        self._is_recording = True
        self._recorded_events.clear()
        logger.info("Event recording started.")

    def stop_recording(self) -> None:
        """Stops recording events."""
        self._is_recording = False
        logger.info("Event recording stopped.")

    def get_recorded_events(self) -> List[Dict[str, Any]]:
        """Returns a copy of the recorded events."""
        return self._recorded_events[:]
        
    def clear_recorded_events(self) -> None:
        """Clears the recorded event list."""
        self._recorded_events.clear()
        logger.info("Recorded events cleared.")

    async def publish(self, event: BaseEvent) -> None: # Takes only the event object
        """
        Publishes an event to the queue and optionally records it.
        
        Parameters:
            event: The event object to publish.
        """
        with event_bus_tracer.start_as_current_span(
            f"event_bus.publish.{type(event).__name__}",
            attributes={
                "event.id": event.event_id if hasattr(event, 'event_id') else "unknown", # Use unique ID
                "event.type": type(event).__name__,
                "event.timestamp": event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.now().isoformat()
            }
        ):
            if not self._running or not self._queue:
                logger.error("EventBus backend not started, cannot publish event.")
                self._stats['dropped_events'] += 1
                return
            
            # Record event if recording is enabled
            if self._is_recording:
                # Convert BaseEvent to a serializable dictionary for recording
                event_data = {
                    "event_id": event.event_id if hasattr(event, 'event_id') else str(uuid.uuid4()),
                    "timestamp": event.timestamp.isoformat() if hasattr(event, 'timestamp') else datetime.now().isoformat(),
                    "event_type": type(event).__name__,
                    "data": event.to_summary_dict() if hasattr(event, 'to_summary_dict') and callable(event.to_summary_dict) else {k: str(v) for k,v in event.__dict__.items()} # Fallback to dict of string values
                }
                self._recorded_events.append(event_data)

            try:
                await self._queue.put(event)
                self._stats['events_published'] += 1
                logger.debug(f"Published event: {event.event_id if hasattr(event, 'event_id') else 'unknown'} ({type(event).__name__})")
            except asyncio.QueueFull:
                logger.error(f"Event queue full, dropping event: {event.event_id if hasattr(event, 'event_id') else 'unknown'}")
                raise RuntimeError("Event queue is full")
    
    async def subscribe(self, event_type: Any, callback: Callable) -> None: # Changed event_type to Any/Type for class support
        """Subscribe to events of a specific type (class or string name)."""
        if inspect.isclass(event_type) and issubclass(event_type, BaseEvent):
            event_name = event_type.__name__
        else:
            event_name = str(event_type) # Handle string names
        with event_bus_tracer.start_as_current_span(f"event_bus.subscribe.{event_name}"):
            if event_name not in self._subscribers:
                self._subscribers[event_name] = []
            
            self._subscribers[event_name].append(callback)
            self._stats['subscribers_count'] = sum(len(callbacks) for callbacks in self._subscribers.values())
            logger.info(f"Subscribed to {event_name} events. Total subscribers: {self._stats['subscribers_count']}")
    
    async def _process_events(self) -> None:
        """Process events from the queue and route to subscribers."""
        logger.info("Event processor started")
        
        while self._running:
            with event_bus_tracer.start_as_current_span("event_bus.process_events"):
                try:
                    # Wait for an event with timeout to allow graceful shutdown
                    event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                    event_type = type(event).__name__ # Get event type from instance
                    
                    self._stats['last_processed_event_type'] = event_type
                    self._stats['last_processed_timestamp'] = datetime.now().isoformat()
                    
                    await self._route_event(event) # This will create child spans for routing and callbacks
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
        
        with event_bus_tracer.start_as_current_span(
            f"event_bus.route_event.{event_type}",
            attributes={"event.id": event.event_id if hasattr(event, 'event_id') else "unknown", "event.type": event_type}
        ):
            if event_type not in self._subscribers:
                logger.debug(f"No subscribers for event type: {event_type}")
                return
            
            callbacks = self._subscribers[event_type]
            logger.debug(f"Routing {event_type} to {len(callbacks)} subscriber(s)")
            
            # Execute all callbacks concurrently
            tasks = []
            for callback in callbacks:
                # Check if callback is a coroutine function (async) or a regular function
                if asyncio.iscoroutinefunction(callback):
                    task = asyncio.create_task(callback(event)) # Await the callback
                else:
                    task = asyncio.create_task(run_in_executor(callback, event)) # Run sync callback in thread pool
                tasks.append(task)
            
            # Wait for all callbacks to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True) # Gather with return_exceptions allows other tasks to continue even if one fails
    
    async def _execute_callback(self, callback: Callable, event: BaseEvent) -> None:
        """Wrapper to execute a callback with error handling."""
        try:
            # Check if event is expected type by callback (if type-hinted)
            # This is more robust than simple hasattr as it respects type hints
            callback_sig = inspect.signature(callback)
            if 'event' in callback_sig.parameters and callback_sig.parameters['event'].annotation != inspect.Parameter.empty:
                expected_type = callback_sig.parameters['event'].annotation
                if not isinstance(event, expected_type):
                    logger.warning(f"Callback {callback.__name__} for {type(event).__name__} received unexpected type: {type(event).__name__}. Expected: {expected_type.__name__}")
            
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event) # Execute sync callbacks directly if not expected to be async

        except Exception as e:
            logger.error(f"Error executing callback {callback.__name__} for event {type(event).__name__} (ID: {event.event_id if hasattr(event, 'event_id') else 'unknown'}): {e}", exc_info=True)


# Global instance management for EventBus
_event_bus_instance: Optional["EventBus"] = None  # Forward reference to EventBus


class EventBus:
    """High-level interface for the event bus, wrapping a backend."""

    def __init__(self, backend: Optional[EventBusBackend] = None):
        """Initialize with a specific backend, or default to AsyncioQueueBackend."""
        self._backend = backend or AsyncioQueueBackend()

    async def publish(self, event: BaseEvent) -> None:
        """Publish an event."""
        await self._backend.publish(event)

    async def subscribe(self, event_type: Any, callback: Callable) -> None:
        """Subscribe to an event type."""
        await self._backend.subscribe(event_type, callback)

    async def start(self) -> None:
        """Start the event bus backend."""
        await self._backend.start()

    async def stop(self) -> None:
        """Stop the event bus backend."""
        await self._backend.stop()

    def start_recording(self) -> None:
        """Start recording events."""
        if hasattr(self._backend, "start_recording"):
            self._backend.start_recording()

    def stop_recording(self) -> List[Dict[str, Any]]:
        """Stop recording and return recorded events."""
        if hasattr(self._backend, "stop_recording"):
            return self._backend.stop_recording()
        return []

    def get_recorded_events(self) -> List[Dict[str, Any]]:
        """Get recorded events without stopping."""
        if hasattr(self._backend, "get_recorded_events"):
            return self._backend.get_recorded_events()
        return []


def get_event_bus() -> "EventBus":
    """Get the global EventBus instance."""
    global _event_bus_instance
    if _event_bus_instance is None:
        _event_bus_instance = EventBus()
    return _event_bus_instance


def set_event_bus(event_bus: "EventBus"):
    """Set the global EventBus instance."""
    global _event_bus_instance
    _event_bus_instance = event_bus


def run_in_executor(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Helper to run synchronous functions in a thread pool executor."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))