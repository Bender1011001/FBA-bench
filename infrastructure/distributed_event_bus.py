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

class MockRedisBroker(AbstractDistributedBroker):
    """
    A mock Redis-like broker for demonstration purposes.
    In a real system, this would use `aioredis` or similar.
    """
    def __init__(self):
        self._channels: Dict[str, List[Callable]] = defaultdict(list)
        logger.info("MockRedisBroker initialized.")

    async def connect(self):
        logger.info("MockRedisBroker connected.")

    async def disconnect(self):
        logger.info("MockRedisBroker disconnected.")

    async def publish(self, topic: str, message: Dict):
        logger.debug(f"MockRedisBroker: Publishing to {topic}: {message}")
        # Simulate async message delivery
        if topic in self._channels:
            for handler in self._channels[topic]:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message) # Await async handler
                else:
                    handler(message) # Call sync handler directly

    async def subscribe(self, topic: str, handler: Callable):
        self._channels[topic].append(handler)
        logger.info(f"MockRedisBroker: Subscribed to {topic}")

    async def create_topic(self, topic: str):
        logger.info(f"MockRedisBroker: Topic {topic} created (mock implementation).")

    async def delete_topic(self, topic: str):
        self._channels.pop(topic, None)
        logger.info(f"MockRedisBroker: Topic {topic} deleted (mock implementation).")


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
        self._broker = broker or MockRedisBroker()
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
            "timestamp": asyncio.Semaphore() # Prevent race conditions when updating the timestamp
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