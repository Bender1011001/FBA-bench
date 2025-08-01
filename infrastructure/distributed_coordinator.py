import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
from datetime import datetime, timedelta

# Assuming DistributedEventBus and other necessary infrastructure components
# from infrastructure.distributed_event_bus import DistributedEventBus
# from events import BaseEvent, TickEvent, WorkerHeartbeatEvent # and other relevant events

logger = logging.getLogger(__name__)

class DistributedCoordinator:
    """
    Coordinates and manages distributed simulation workers across multiple processes/nodes.

    - Worker management: Spawns and manages simulation worker processes.
    - Cross-partition communication: Enables agent interactions across partitions.
    - Synchronization: Coordinates tick progression across distributed workers.
    - Result aggregation: Collects and merges results from all partitions.
    """

    def __init__(self, distributed_event_bus: Any, simulation_config: Any): # Use Any to avoid circular deps
        self.distributed_event_bus = distributed_event_bus
        self.simulation_config = simulation_config # This would be an instance of ScalabilityConfig
        
        self._workers: Dict[str, Dict[str, Any]] = {} # worker_id -> info (e.g., process, partitions, capabilities, last_heartbeat)
        self._active_partitions: Dict[str, List[str]] = {} # partition_id -> list of agent_ids
        self._tick_acknowledgements: Dict[int, Set[str]] = defaultdict(set) # tick_number -> set of worker_ids that acknowledged
        self._global_current_tick: int = 0
        self._running = False
        self._worker_monitor_task: Optional[asyncio.Task] = None
        self._synchronization_task: Optional[asyncio.Task] = None
        
        # Buffers for results
        self._aggregated_results: List[Any] = [] # Stores results from workers
        self._cross_partition_event_buffer: asyncio.Queue = asyncio.Queue()

        logger.info("DistributedCoordinator initialized.")

    async def start(self):
        """Starts the coordinator's background tasks and connects to the event bus."""
        if self._running:
            logger.warning("DistributedCoordinator already running.")
            return

        self._running = True
        await self.distributed_event_bus.start() # Ensure the underlying bus is started
        
        # Subscribe to worker heartbeats and messages relevant to coordination
        # Example: await self.distributed_event_bus.subscribe_to_event("WorkerHeartbeat", self._handle_worker_heartbeat)
        # Example: await self.distributed_event_bus.subscribe_to_event("TickAckEvent", self._handle_tick_acknowledgement)
        # Example: await self.distributed_event_bus.subscribe_to_event("CrossPartitionEvent", self.handle_cross_partition_event)
        
        self._worker_monitor_task = asyncio.create_task(self.monitor_worker_health())
        self._synchronization_task = asyncio.create_task(self.coordinate_tick_progression())
        
        logger.info("DistributedCoordinator started.")

    async def stop(self):
        """Stops all coordinator tasks and disconnects from the event bus."""
        if not self._running:
            return
        self._running = False
        
        if self._worker_monitor_task:
            self._worker_monitor_task.cancel()
            try: await self._worker_monitor_task
            except asyncio.CancelledError: pass
        
        if self._synchronization_task:
            self._synchronization_task.cancel()
            try: await self._synchronization_task
            except asyncio.CancelledError: pass

        # Signal workers to shut down (conceptual)
        await self._signal_workers_to_shutdown()

        await self.distributed_event_bus.stop()
        logger.info("DistributedCoordinator stopped.")

    async def spawn_worker(self, partition_config: Dict[str, Any]) -> str:
        """
        Spawns a new simulation worker process and registers it.
        
        Args:
            partition_config: Configuration for the partition(s) this worker will handle
                              (e.g., list of agent IDs, initial state slice).
        Returns:
            The unique ID of the spawned worker.
        """
        worker_id = f"worker_{uuid.uuid4().hex[:8]}"
        
        # Ideally, this would use `multiprocessing.Process` or `asyncio.create_subprocess_exec`
        # to start a new Python process that runs a "worker" script.
        # For this conceptual implementation, we'll just register it internally.
        
        capabilities = {"cpu_cores": 1, "memory_gb": 4, "can_run_llm": True} # Example capabilities
        
        self._workers[worker_id] = {
            "id": worker_id,
            "partition_config": partition_config,
            "capabilities": capabilities,
            "last_heartbeat": time.time(),
            "status": "starting",
            "current_tick": 0 # Track individual worker's tick progress
        }
        
        # Inform the distributed event bus about the new worker
        await self.distributed_event_bus.register_worker(worker_id, capabilities)
        
        # Create partition(s) on the event bus for this worker
        for partition_id, agents in partition_config.items():
            await self.distributed_event_bus.create_partition(partition_id, agents)
            self._active_partitions[partition_id] = agents

        logger.info(f"Spawned and registered worker {worker_id} for partitions: {list(partition_config.keys())}")
        return worker_id

    async def _signal_workers_to_shutdown(self):
        """Sends a shutdown signal to all active workers."""
        logger.info("Sending shutdown signal to workers...")
        # This would publish a control event to a specific worker management topic
        # which workers are subscribed to.
        for worker_id in list(self._workers.keys()): # Copy keys for safe iteration
            await self.distributed_event_bus.publish_event(
                "CoordinatorControlEvent", 
                {"action": "shutdown", "target_worker_id": worker_id},
                target_partition=None # Global control message
            )
            # await self.distributed_event_bus.handle_worker_failure(worker_id) # Deregister immediately
        logger.info("Shutdown signals sent.")


    async def coordinate_tick_progression(self):
        """
        Synchronizes tick progression across all distributed workers.
        This is a consensus mechanism where all workers must acknowledge a tick
        before the global tick advances.
        """
        logger.info("Tick synchronization loop started.")
        while self._running:
            # Wait for all active workers to acknowledge the current tick
            required_acks = set(self._workers.keys()) # All current active workers
            
            if not required_acks:
                logger.debug("No active workers to synchronize. Waiting...")
                await asyncio.sleep(1.0) # Wait if no workers
                continue

            # This loop waits for acknowledgements for the _global_current_tick
            # before advancing it.
            try:
                # print(f"Current tick {self._global_current_tick}. Waiting for {len(required_acks)} acks. Received: {self._tick_acknowledgements.get(self._global_current_tick, set())}")
                if self._tick_acknowledgements[self._global_current_tick] == required_acks:
                    logger.info(f"All {len(required_acks)} workers acknowledged tick {self._global_current_tick}.")
                    self._global_current_tick += 1
                    # Publish a global "AdvanceTick" event, or orchestrator proceeds based on this
                    await self.distributed_event_bus.publish_event(
                        "GlobalTickAdvanceEvent", 
                        {"new_tick_number": self._global_current_tick},
                        target_partition=None
                    )
                    # Clear acknowledgements for the new tick
                    self._tick_acknowledgements[self._global_current_tick].clear()
                    # Clean up old acknowledgements
                    if self._global_current_tick > 1:
                        self._tick_acknowledgements.pop(self._global_current_tick - 1, None)
                else:
                    missing_acks = required_acks - self._tick_acknowledgements[self._global_current_tick]
                    logger.debug(f"Tick {self._global_current_tick} waiting for acks from: {missing_acks} (Total workers: {len(required_acks)})")
                
                await asyncio.sleep(self.simulation_config.tick_interval_seconds if self.simulation_config else 1.0) # Adapt to config

            except Exception as e:
                logger.error(f"Error in tick coordination loop: {e}", exc_info=True)
                await asyncio.sleep(1.0) # Avoid tight loop on error
        logger.info("Tick synchronization loop stopped.")


    async def handle_cross_partition_event(self, event: Dict, source_partition: str, target_partition: str):
        """
        Routes events that an agent in one partition needs to send to an agent in another.
        
        Args:
            event: The event data.
            source_partition: The partition from which the event originated.
            target_partition: The partition to which the event is directed.
        """
        logger.info(f"Routing cross-partition event from {source_partition} to {target_partition}: {event.get('event_type')}")
        # Add to an internal buffer or directly publish to target partition's topic
        await self._cross_partition_event_buffer.put((event, target_partition))

        # Re-publish the event to the target partition's specific topic
        await self.distributed_event_bus.publish_event(
            event.get("event_type", "CrossPartitionMessage"),
            event.get("event_data", {}),
            target_partition=target_partition
        )
        logger.debug(f"Event routed to partition {target_partition}.")

    async def aggregate_simulation_results(self) -> List[Any]:
        """
        Collects and merges results from all worker processes.
        (Conceptual: In a real system, workers would publish results to a shared data store
        or a dedicated results topic.)
        """
        logger.info("Aggregating simulation results...")
        # This function would typically query a centralized result store (e.g., PostgreSQL)
        # or consume from dedicated result topics on the distributed event bus.
        
        # For now, it just returns a conceptual aggregation.
        # It could process the _aggregated_results buffer.
        
        # Example: await self.distributed_event_bus.subscribe_to_event("SimulationResultEvent", self._handle_worker_result)
        # This example assumes _handle_worker_result adds to _aggregated_results list

        # Placeholder for actual data retrieval and merging logic
        if not self._aggregated_results:
            logger.warning("No results to aggregate yet.")
            return []
        
        final_results = self._aggregated_results[:] # Return a copy
        self._aggregated_results.clear() # Clear after aggregation
        logger.info(f"Aggregated {len(final_results)} results.")
        return final_results

    async def monitor_worker_health(self):
        """
        Continuously monitors worker health using heartbeats and triggers failure handling.
        This task runs in the background.
        """
        heartbeat_interval = 5 # seconds
        timeout_multiplier = 3 # consider worker failed after 3 missing heartbeats

        logger.info("Worker health monitor started.")
        while self._running:
            for worker_id in list(self._workers.keys()): # Iterate over copy
                worker_info = self._workers.get(worker_id)
                if worker_info:
                    last_heartbeat = worker_info.get("last_heartbeat", 0)
                    if (time.time() - last_heartbeat) > (heartbeat_interval * timeout_multiplier):
                        logger.error(f"Worker {worker_id} timed out! Last heartbeat: {datetime.fromtimestamp(last_heartbeat).isoformat()}.")
                        await self.distributed_event_bus.handle_worker_failure(worker_id)
                        self._workers.pop(worker_id, None) # Remove from local registry
            
            await asyncio.sleep(heartbeat_interval)
        logger.info("Worker health monitor stopped.")

    async def _handle_worker_heartbeat(self, message: Dict):
        """Processes incoming worker heartbeat events."""
        worker_id = message.get("worker_id")
        timestamp = message.get("timestamp")
        
        if worker_id and worker_id in self._workers:
            self._workers[worker_id]["last_heartbeat"] = timestamp
            self._workers[worker_id]["status"] = "active"
            self._workers[worker_id]["current_tick"] = message.get("current_tick", 0) # Update worker's tick progress
            logger.debug(f"Received heartbeat from worker {worker_id} at tick {self._workers[worker_id]['current_tick']}")
        else:
            logger.warning(f"Received heartbeat from unknown or unregistered worker: {worker_id}. Message: {message}")

    async def _handle_tick_acknowledgement(self, message: Dict):
        """Processes worker acknowledgements for a given tick."""
        worker_id = message.get("worker_id")
        tick_number = message.get("tick_number")
        if worker_id and tick_number is not None:
             self._tick_acknowledgements[tick_number].add(worker_id)
             logger.debug(f"Worker {worker_id} acknowledged tick {tick_number}.")
        else:
            logger.warning(f"Invalid tick acknowledgement message: {message}")

    @property
    def current_global_tick(self) -> int:
        """Returns the current globally synchronized tick."""
        return self._global_current_tick

    def get_workers_status(self) -> Dict[str, Any]:
        """Returns the status of all registered workers."""
        return {wid: {k: v for k, v in data.items() if k not in ['_process_obj']} for wid, data in self._workers.items()}
    
    def get_status(self) -> Dict[str, Any]:
        """Provides overall status of the DistributedCoordinator."""
        return {
            "running": self._running,
            "global_current_tick": self._global_current_tick,
            "num_active_workers": len(self._workers),
            "num_active_partitions": len(self._active_partitions),
            "pending_cross_partition_events": self._cross_partition_event_buffer.qsize(),
            "workers_status": self.get_workers_status()
        }