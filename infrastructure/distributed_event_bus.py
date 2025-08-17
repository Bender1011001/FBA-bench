import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict
from abc import ABC, abstractmethod

# Assuming BaseEvent is available from common event definitions
# from events import BaseEvent # Uncomment if actual event objects are used

logger = logging.getLogger(__name__)

class AbstractDistributedBroker(ABC):
    """Abstract base class for a distributed message broker."""
    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def disconnect(self):
        pass

    @abstractmethod
    async def publish(self, topic: str, message: Dict):
        pass

    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable):
        pass

    @abstractmethod
    async def create_topic(self, topic: str):
        pass

    @abstractmethod
    async def delete_topic(self, topic: str):
        pass

class RedisBroker(AbstractDistributedBroker):
    """
    Production-ready Redis broker implementation using aioredis.
    Supports pub/sub messaging with proper connection management and error handling.
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379",
                 connection_pool_size: int = 10,
                 retry_attempts: int = 3,
                 retry_delay: float = 1.0):
        """
        Initialize Redis broker with connection configuration.
        
        Args:
            redis_url: Redis connection URL
            connection_pool_size: Size of the connection pool
            retry_attempts: Number of connection retry attempts
            retry_delay: Delay between retry attempts in seconds
        """
        self.redis_url = redis_url
        self.connection_pool_size = connection_pool_size
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        self._redis_client = None
        self._pubsub = None
        self._connection_lock = asyncio.Lock()
        self._subscriber_tasks: Dict[str, asyncio.Task] = {}
        self._is_connected = False
        
        logger.info(f"RedisBroker initialized with URL: {redis_url}")

    async def connect(self):
        """Establish connection to Redis server."""
        async with self._connection_lock:
            if self._is_connected:
                logger.warning("Redis connection already established")
                return
                
            for attempt in range(self.retry_attempts):
                try:
                    import aioredis
                    
                    # Create Redis client with connection pooling
                    self._redis_client = aioredis.from_url(
                        self.redis_url,
                        max_connections=self.connection_pool_size,
                        retry_on_timeout=True,
                        health_check_interval=30,
                        encoding="utf-8",
                        decode_responses=True
                    )
                    
                    # Test connection
                    await self._redis_client.ping()
                    
                    # Initialize pubsub
                    self._pubsub = self._redis_client.pubsub()
                    
                    self._is_connected = True
                    logger.info(f"Successfully connected to Redis at {self.redis_url}")
                    return
                    
                except Exception as e:
                    logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(self.retry_delay)
                    else:
                        raise ConnectionError(f"Failed to connect to Redis after {self.retry_attempts} attempts: {e}")

    async def disconnect(self):
        """Close connection to Redis server."""
        async with self._connection_lock:
            if not self._is_connected:
                logger.warning("Redis connection not established")
                return
                
            try:
                # Cancel all subscriber tasks
                for task in self._subscriber_tasks.values():
                    task.cancel()
                
                # Wait for tasks to complete
                if self._subscriber_tasks:
                    await asyncio.gather(*self._subscriber_tasks.values(), return_exceptions=True)
                
                # Close pubsub connection
                if self._pubsub:
                    await self._pubsub.close()
                
                # Close Redis client
                if self._redis_client:
                    await self._redis_client.close()
                
                self._is_connected = False
                self._subscriber_tasks.clear()
                logger.info("Successfully disconnected from Redis")
                
            except Exception as e:
                logger.error(f"Error during Redis disconnect: {e}")
                raise

    async def publish(self, topic: str, message: Dict):
        """
        Publish a message to a Redis channel.
        
        Args:
            topic: Channel/topic to publish to
            message: Message data to publish
        """
        if not self._is_connected:
            raise ConnectionError("Redis connection not established")
            
        try:
            # Serialize message to JSON
            message_json = json.dumps(message)
            
            # Publish to Redis channel
            subscribers = await self._redis_client.publish(topic, message_json)
            
            logger.debug(f"Published to {topic}: {message} (delivered to {subscribers} subscribers)")
            
        except Exception as e:
            logger.error(f"Failed to publish message to {topic}: {e}")
            raise

    async def subscribe(self, topic: str, handler: Callable):
        """
        Subscribe to a Redis channel with a handler function.
        
        Args:
            topic: Channel/topic to subscribe to
            handler: Async function to handle incoming messages
        """
        if not self._is_connected:
            raise ConnectionError("Redis connection not established")
            
        try:
            # Subscribe to the channel
            await self._pubsub.subscribe(topic)
            
            # Create a task to listen for messages on this channel
            task = asyncio.create_task(self._listen_for_messages(topic, handler))
            self._subscriber_tasks[topic] = task
            
            logger.info(f"Subscribed to {topic}")
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {topic}: {e}")
            raise

    async def create_topic(self, topic: str):
        """
        Create a new topic (Redis channel).
        In Redis, channels are created automatically on first use,
        but we'll validate the topic name format.
        
        Args:
            topic: Name of the topic to create
        """
        if not topic or not isinstance(topic, str):
            raise ValueError("Topic name must be a non-empty string")
            
        # Redis channels are created automatically, so we just validate
        logger.info(f"Topic {topic} is ready for use")

    async def delete_topic(self, topic: str):
        """
        Delete a topic (Redis channel).
        In Redis, we can't directly delete channels, but we can
        unsubscribe all listeners and clear any related data.
        
        Args:
            topic: Name of the topic to delete
        """
        try:
            # Cancel subscriber task for this topic
            if topic in self._subscriber_tasks:
                self._subscriber_tasks[topic].cancel()
                del self._subscriber_tasks[topic]
            
            # Unsubscribe from the channel
            if self._pubsub:
                await self._pubsub.unsubscribe(topic)
            
            logger.info(f"Cleaned up topic: {topic}")
            
        except Exception as e:
            logger.error(f"Failed to clean up topic {topic}: {e}")
            raise

    async def _listen_for_messages(self, topic: str, handler: Callable):
        """
        Internal method to listen for messages on a specific channel.
        
        Args:
            topic: Channel to listen on
            handler: Message handler function
        """
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    try:
                        # Parse message data
                        message_data = json.loads(message["data"])
                        
                        # Call the handler with the message data
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message_data)
                        else:
                            # Wrap sync handlers in async
                            await asyncio.get_event_loop().run_in_executor(
                                None, handler, message_data
                            )
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse message from {topic}: {e}")
                    except Exception as e:
                        logger.error(f"Error in message handler for {topic}: {e}")
                        
        except asyncio.CancelledError:
            logger.info(f"Listener for {topic} cancelled")
        except Exception as e:
            logger.error(f"Listener for {topic} failed: {e}")
            # Attempt to reconnect
            if self._is_connected:
                logger.info(f"Attempting to restart listener for {topic}")
                await asyncio.sleep(1)  # Brief delay before retry
                self._subscriber_tasks[topic] = asyncio.create_task(
                    self._listen_for_messages(topic, handler)
                )


class DistributedEventBus:
    """
    Enables event sharing and coordination across multiple processes/nodes.

    - Multi-process coordination: Enables event sharing across processes.
    - Partition management: Distributes events based on agent IDs or topics.
    - Load balancing: Distributes computational load across workers.
    - Fault tolerance: Handles worker failures gracefully.
    """
    
    # Define system topics
    COORDINATOR_TOPIC = "coordinator_events"
    WORKER_REGISTRY_TOPIC = "worker_registry"
    
    def __init__(self, broker: Optional[AbstractDistributedBroker] = None):
        self._broker = broker or RedisBroker()
        self._partitions: Dict[str, List[str]] = {} # partition_id -> list of agent_ids
        self._workers: Dict[str, Dict[str, Any]] = {} # worker_id -> capabilities, last_heartbeat, etc.
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list) # topic -> list of handlers
        self._running = False
        self._sub_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        logger.info("DistributedEventBus initialized.")

    async def start(self):
        """Connects to the distributed broker and starts listening for events."""
        if self._running:
            logger.warning("DistributedEventBus already running.")
            return
        await self._broker.connect()
        self._running = True
        
        # Subscribe to all relevant topics
        await self._broker.subscribe(self.COORDINATOR_TOPIC, self._handle_coordinator_event)
        await self._broker.subscribe(self.WORKER_REGISTRY_TOPIC, self._handle_worker_registry_event)
        
        logger.info("DistributedEventBus started and subscribed to core topics.")

    async def stop(self):
        """Disconnects from the distributed broker and cleans up."""
        if not self._running:
            return
        self._running = False
        if self._sub_task:
            self._sub_task.cancel()
            try:
                await self._sub_task
            except asyncio.CancelledError:
                pass
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        await self._broker.disconnect()
        logger.info("DistributedEventBus stopped.")

    async def publish_event(self, event_type: str, event_data: Dict, target_partition: Optional[str] = None):
        """
        Publishes an event to the distributed bus.
        If target_partition is specified, routes to that partition's topic.
        Otherwise, it's a general event on the coordinator topic or based on event_type.
        """
        full_event = {
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": datetime.now().isoformat() # Use ISO format timestamp
        }

        if target_partition:
            topic = f"partition_{target_partition}_events"
            await self._broker.publish(topic, full_event)
            logger.debug(f"Published event {event_type} to partition {target_partition}")
        else:
            # General events can go to a topic derived from their type or a global topic
            topic = f"general_{event_type}_events" if event_type else self.COORDINATOR_TOPIC
            await self._broker.publish(topic, full_event)
            logger.debug(f"Published general event {event_type} to topic {topic}")

    async def subscribe_to_event(self, event_type: str, handler: Callable):
        """Subscribes a handler to a specific event type globally (via general topic)."""
        topic = f"general_{event_type}_events"
        await self._broker.subscribe(topic, lambda msg: asyncio.create_task(handler(msg))) # Wrap in task
        self._event_handlers[topic].append(handler)
        logger.info(f"Subscribed handler to general event type: {event_type}")

    async def create_partition(self, partition_id: str, agents: List[str]):
        """
        Creates an isolated simulation partition and associated topics.
        
        Args:
            partition_id: Unique ID for the partition.
            agents: List of agent IDs assigned to this partition.
        """
        if partition_id in self._partitions:
            logger.warning(f"Partition {partition_id} already exists.")
            return

        self._partitions[partition_id] = agents
        
        # Create a dedicated topic for this partition's events
        partition_topic = f"partition_{partition_id}_events"
        await self._broker.create_topic(partition_topic)
        
        logger.info(f"Created partition {partition_id} with agents: {agents}")

    async def route_event(self, event: Dict, target_partition: Optional[str] = None):
        """
        Sends events to the correct partition or general topic.
        This acts as a high-level router for internal components.
        """
        event_type = event.get("event_type")
        if not event_type:
            logger.error("Event missing 'event_type' field, cannot route.")
            return

        await self.publish_event(event_type, event.get("event_data", {}), target_partition)
    
    async def register_worker(self, worker_id: str, capabilities: Dict[str, Any]):
        """
        Registers a computational worker with the distributed bus.
        The worker sends its capabilities and a heartbeat.
        """
        message = {
            "action": "register",
            "worker_id": worker_id,
            "capabilities": capabilities,
            "timestamp": time.time(),
        }
        await self._broker.publish(self.WORKER_REGISTRY_TOPIC, message)
        self._workers[worker_id] = {"capabilities": capabilities, "last_heartbeat": time.time()}
        logger.info(f"Worker {worker_id} registered with capabilities: {capabilities}")

    async def balance_load(self):
        """
        Redistributes work based on current load and worker capabilities.
        (Conceptual - actual implementation would be complex, involving tracking worker queues/CPU).
        """
        logger.info("Attempting to balance load across workers...")
        # This is a placeholder for complex load balancing logic.
        # It would query worker statuses, partition loads, and potentially
        # send commands to workers to reassign agent processing responsibilities.
        total_agents = sum(len(agents) for agents in self._partitions.values())
        active_workers = {wid: data for wid, data in self._workers.items() if time.time() - data.get("last_heartbeat", 0) < 30} # Active in last 30s
        
        if not active_workers:
            logger.warning("No active workers to balance load across.")
            return

        agents_per_worker = total_agents / len(active_workers)
        logger.info(f"Targeting approx. {agents_per_worker:.2f} agents per active worker.")

        # Simplified re-partitioning logic for demonstration:
        # In a real system, this would involve careful state transfer and coordination.
        # Here, we just log a theoretical re-distribution.
        # await self._send_repartitioning_commands(active_workers, self._partitions)
        logger.info("Load balancing logic needs full implementation based on actual workload metrics.")

    async def handle_worker_failure(self, worker_id: str):
        """
        Recovers from worker failures.
        (Conceptual - would involve reassigning failed worker's partitions/agents).
        """
        logger.warning(f"Worker {worker_id} detected as failed. Initiating recovery...")
        failed_worker_info = self._workers.pop(worker_id, None)
        if failed_worker_info:
            logger.info(f"Removed failed worker {worker_id} from registry.")
            # Logic to reassign agents/partitions of the failed worker
            for partition_id, agents in self._partitions.items():
                # This is an oversimplification; actual partition assignment is more granular
                # For demo, just log that reallocation would happen
                logger.info(f"Partition {partition_id} (agents: {agents}) needs to be reallocated.")
            await self.balance_load() # Try to rebalance after failure
        else:
            logger.info(f"Worker {worker_id} not found in registry (already failed or never registered).")
        
    def _handle_coordinator_event(self, message: Dict):
        """Handles events from the coordinator topic (e.g., control signals)."""
        event_type = message.get("event_type", "unknown")
        event_data = message.get("event_data", {})
        logger.info(f"Received coordinator event: {event_type} - {event_data}")
        # Add specific handling logic for coordinator events, e.g.,
        # if event_type == "SHUTDOWN":
        #    self.stop()
        pass

    async def _handle_worker_registry_event(self, message: Dict):
        """Handles worker registration/heartbeat events."""
        action = message.get("action")
        worker_id = message.get("worker_id")
        timestamp = message.get("timestamp")

        if action == "register" or action == "heartbeat":
            self._workers[worker_id] = {
                "capabilities": message.get("capabilities", {}),
                "last_heartbeat": timestamp,
                "status": "active"
            }
            logger.debug(f"Worker {worker_id} heartbeat/registration. Active workers: {len(self._workers)}")
        elif action == "deregister":
            self._workers.pop(worker_id, None)
            logger.info(f"Worker {worker_id} deregistered. Active workers: {len(self._workers)}")
            
        # Potentially trigger load balancing or failure handling
        await self.monitor_worker_health() # Check all workers after a registry update

    async def monitor_worker_health(self, heartbeat_timeout: int = 60):
        """Monitors health of registered workers and triggers failure handling."""
        current_time = time.time()
        failed_workers = []
        for worker_id, data in self._workers.items():
            if (current_time - data.get("last_heartbeat", 0)) > heartbeat_timeout:
                failed_workers.append(worker_id)
        
        for worker_id in failed_workers:
            logger.warning(f"Worker {worker_id} timed out. Last heartbeat: {datetime.fromtimestamp(self._workers[worker_id]['last_heartbeat'])}")
            await self.handle_worker_failure(worker_id)


class MockRedisBroker(AbstractDistributedBroker):
    """Mock Redis broker for testing purposes."""
    
    def __init__(self, *args, **kwargs):
        self.connected = True
        self.messages = {}
        self.subscriptions = {}
        
    async def connect(self):
        """Mock connect method."""
        self.connected = True
        
    async def disconnect(self):
        """Mock disconnect method."""
        self.connected = False
        
    async def publish(self, topic: str, message: Dict):
        """Mock publish method."""
        if topic not in self.messages:
            self.messages[topic] = []
        self.messages[topic].append(message)
        
    async def subscribe(self, topic: str, handler: Callable):
        """Mock subscribe method."""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        self.subscriptions[topic].append(handler)
        
    async def create_topic(self, topic: str):
        """Mock create_topic method."""
        if topic not in self.messages:
            self.messages[topic] = []
        
    async def delete_topic(self, topic: str):
        """Mock delete_topic method."""
        if topic in self.messages:
            del self.messages[topic]
        if topic in self.subscriptions:
            del self.subscriptions[topic]