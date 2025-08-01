"""EventBus implementation for FBA-Bench v3 event-driven architecture."""

import asyncio
import logging
import inspect
import uuid
from typing import Dict, List, Callable, Any, Optional, Type, Union
from datetime import datetime
from abc import ABC, abstractmethod
import zlib # For compression

from events import BaseEvent, EVENT_TYPES # Ensure BaseEvent is imported for type hinting

# OpenTelemetry Imports
from opentelemetry import trace
from instrumentation.tracer import setup_tracing

logger = logging.getLogger(__name__)

# Initialize tracer for EventBus module
event_bus_tracer = setup_tracing(service_name="fba-bench-eventbus")

# Type hinting for DistributedEventBus to avoid circular import at runtime
if TYPE_CHECKING:
    from infrastructure.distributed_event_bus import DistributedEventBus

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
            logger.warning("AsyncioQueueBackend already running")
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


class DistributedBackend(EventBusBackend):
    """
    Distributed EventBus backend using a message broker (e.g., Redis Pub/Sub, Kafka).
    Enables event sharing across processes and persistence.
    """
    def __init__(self, distributed_bus: "DistributedEventBus"): # Type hint for DistributedEventBus
        self.distributed_bus = distributed_bus
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._running = False
        logger.info("DistributedBackend initialized.")

    async def start(self) -> None:
        """Connects to the distributed message broker."""
        if self._running:
            logger.warning("DistributedBackend already running.")
            return
        await self.distributed_bus.start() # Start the underlying distributed bus
        self._running = True
        logger.info("DistributedBackend started.")

    async def stop(self) -> None:
        """Disconnects from the distributed message broker."""
        if not self._running:
            return
        await self.distributed_bus.stop() # Stop the underlying distributed bus
        self._running = False
        logger.info("DistributedBackend stopped.")

    async def publish(self, event: BaseEvent) -> None:
        """
        Publishes an event to the distributed bus.
        Serializes events and sends them via the distributed_bus.
        """
        event_dict = event.to_summary_dict() if hasattr(event, 'to_summary_dict') and callable(event.to_summary_dict) else {k: str(v) for k,v in event.__dict__.items()}
        await self.distributed_bus.publish_event(type(event).__name__, event_dict)
        logger.debug(f"Published event {type(event).__name__} to distributed bus.")

    async def subscribe(self, event_type: Any, callback: Callable) -> None:
        """
        Subscribes to events on the distributed bus.
        Callbacks will be invoked when relevant events are received from the distributed_bus.
        """
        if inspect.isclass(event_type) and issubclass(event_type, BaseEvent):
            event_name = event_type.__name__
        else:
            event_name = str(event_type)
        
        # Register a wrapper handler with the underlying distributed bus
        # This handler will deserialize the event and then call original callbacks
        async def distributed_event_wrapper(message: Dict):
            if message.get("event_type") == event_name:
                # Reconstruct BaseEvent from message. For simplicity, just pass message data
                # In a real system, you'd deserialize to the actual BaseEvent subtype
                # event_obj = EVENT_TYPES.get(event_name)(**message.get("event_data", {})) # Example
                # await self._execute_callback(callback, event_obj)
                
                # For now, just pass the dict directly if the event cannot be re-instantiated easily
                logger.debug(f"DistributedBackend: Received event {event_name} from distributed bus. Invoking local subscribers.")
                # Execute all local callbacks for this event type
                for cb in self._subscribers[event_name]:
                    if asyncio.iscoroutinefunction(cb):
                        asyncio.create_task(cb(message)) # Pass the raw message for simplicity
                    else:
                        asyncio.create_task(run_in_executor(cb, message))

        await self.distributed_bus.subscribe_to_event(event_name, distributed_event_wrapper)
        self._subscribers[event_name].append(callback)
        logger.info(f"Subscribed to {event_name} events on distributed backend.")

    async def compress_old_events(self, age_threshold_seconds: int = 3600, storage_backend: Any = None):
        """
        Compresses and potentially offloads old events to reduce memory usage.
        This would typically involve reading from a primary event store, compressing,
        and writing to an archival store.
        """
        logger.info(f"Compressing events older than {age_threshold_seconds} seconds (conceptual).")
        # Example: iterate through a hypothetical in-memory event log
        # if hasattr(self.distributed_bus, '_event_log'): # if distributed bus has a log
        #     events_to_compress = [
        #         e for e in self.distributed_bus._event_log if (datetime.now() - e.timestamp).total_seconds() > age_threshold_seconds
        #     ]
        #     for event in events_to_compress:
        #         event_data = json.dumps(event.to_summary_dict()).encode('utf-8')
        #         compressed_data = zlib.compress(event_data)
        #         # Persist compressed_data to storage_backend
        #         logger.debug(f"Compressed event {event.event_id}, original size {len(event_data)} bytes, compressed {len(compressed_data)} bytes.")
        #         # Remove original event from active memory
        # else:
        logger.warning("Event compression is conceptual and requires a persistent event store to operate.")

    async def persist_events_to_storage(self, storage_backend: Any):
        """
        Saves ongoing events to an external storage backend for durability.
        This would usually be integrated into the publish/process pipeline.
        """
        logger.info(f"Persisting events to storage backend (conceptual).")
        # This is primarily handled by the distributed message broker if it's persistent (like Kafka).
        # If using Redis or a transient broker, you'd need a separate service consuming
        # events from the bus and writing them to PostgreSQL or S3.
        # Example: event_stream = await self.distributed_bus.get_event_stream("all_events")
        # async for event in event_stream:
        #    storage_backend.save_event(event)
        logger.warning("Event persistence is mainly delegated to the distributed message broker's features.")


    async def load_events_from_storage(self, time_range: Tuple[datetime, datetime]) -> List[Dict[str, Any]]:
        """
        Retrieves historical events from external storage for replay or analysis.
        """
        logger.info(f"Loading historical events from storage for time range {time_range} (conceptual).")
        # This would read from the persistent storage (e.g., PostgreSQL, S3 archive).
        # For a full replay, events should be loaded in chronological order.
        return [] # Return empty list for conceptual implementation.

    async def enable_distributed_mode(self, coordinator: "DistributedCoordinator"):
        """
        Activates distributed processing mode, integrating with the coordinator.
        The event bus will now route relevant events via the distributed coordinator.
        """
        # This method is primarily for enabling on the main EventBus facade,
        # which would then switch its backend to DistributedBackend.
        logger.info("Distributed mode activated (functionality handled by EventBus switching to this backend).")

# Global instance management for EventBus
# Existing _event_bus_instance and get_event_bus, set_event_bus remain the same.
# We'll modify the EventBus class to allow swapping backends.

# Add to the existing EventBus class
# No, this needs to be a new backend that EventBus can use.


# Global instance management for EventBus
_event_bus_instance: Optional["EventBus"] = None  # Forward reference to EventBus


class EventBus:
    """High-level interface for the event bus, wrapping a backend."""

    def __init__(self, backend: Optional[EventBusBackend] = None):
        """Initialize with a specific backend, or default to AsyncioQueueBackend."""
        self._backend = backend or AsyncioQueueBackend()
        self._distributed_mode_enabled = False
        self._distributed_coordinator: Optional[Any] = None # Will hold DistributedCoordinator instance

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
            # Check if stop_recording returns a value, as AsyncioQueueBackend does
            result = self._backend.stop_recording()
            if result is not None:
                return result
        return []

    def get_recorded_events(self) -> List[Dict[str, Any]]:
        """Get recorded events without stopping."""
        if hasattr(self._backend, "get_recorded_events"):
            return self._backend.get_recorded_events()
        return []
    
    # New methods for Enhanced Event Processing
    async def compress_old_events(self, age_threshold_seconds: int = 3600, storage_backend: Any = None):
        """Facade for backend's event compression."""
        if hasattr(self._backend, "compress_old_events"):
            await self._backend.compress_old_events(age_threshold_seconds, storage_backend)
        else:
            logger.warning("Current EventBus backend does not support event compression.")

    async def persist_events_to_storage(self, storage_backend: Any):
        """Facade for backend's event persistence."""
        if hasattr(self._backend, "persist_events_to_storage"):
            await self._backend.persist_events_to_storage(storage_backend)
        else:
            logger.warning("Current EventBus backend does not support event persistence.")

    async def load_events_from_storage(self, time_range: Tuple[datetime, datetime]) -> List[Dict[str, Any]]:
        """Facade for backend's historical event loading."""
        if hasattr(self._backend, "load_events_from_storage"):
            return await self._backend.load_events_from_storage(time_range)
        else:
            logger.warning("Current EventBus backend does not support loading historical events.")
            return []

    def enable_distributed_mode(self, coordinator: "DistributedCoordinator"):
        """
        Configures the EventBus to use a DistributedBackend.
        This effectively swaps the backend if not already set.
        """
        # Ensure that DistributedEventBus is used here, not MockRedisBroker directly
        from infrastructure.distributed_event_bus import DistributedEventBus, MockRedisBroker
        if isinstance(self._backend, AsyncioQueueBackend): # Only swap if currently using in-memory
            self._distributed_coordinator = coordinator
            # Pass the internal broker used by the coordinator, if it's a mock or real one
            # OR create a new one but ensure it connects to the same underlying system.
            # Simplified: assuming coordinator's bus is the one to use internally
            self._backend = DistributedBackend(self._distributed_coordinator.distributed_event_bus)
            self._distributed_mode_enabled = True
            logger.info("EventBus switched to DistributedBackend.")
        else:
            logger.info("EventBus is already using a non-AsyncioQueueBackend (or distributed mode is already active).")


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