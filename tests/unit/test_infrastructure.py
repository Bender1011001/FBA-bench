import unittest
import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from infrastructure.distributed_event_bus import DistributedEventBus, MockRedisBroker
from infrastructure.distributed_coordinator import DistributedCoordinator
from infrastructure.fast_forward_engine import FastForwardEngine
from infrastructure.llm_batcher import LLMBatcher, LLMRequest
from infrastructure.performance_monitor import PerformanceMonitor
from infrastructure.resource_manager import ResourceManager
from infrastructure.deployment import DeploymentManager, DeploymentConfig, DeploymentEnvironment, DeploymentType


class TestDistributedEventBus(unittest.TestCase):
    """Test suite for the DistributedEventBus class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.broker = MockRedisBroker()
        self.event_bus = DistributedEventBus(self.broker)
    
    async def async_setUp(self):
        """Async set up for tests that need it."""
        await self.event_bus.start()
    
    async def async_tearDown(self):
        """Async tear down for tests that need it."""
        await self.event_bus.stop()
    
    def test_event_bus_initialization(self):
        """Test that the event bus initializes correctly."""
        self.assertIsNotNone(self.event_bus)
        self.assertEqual(self.event_bus._broker, self.broker)
        self.assertEqual(len(self.event_bus._subscriptions), 0)
    
    @pytest.mark.asyncio
    async def test_event_bus_start_stop(self):
        """Test starting and stopping the event bus."""
        await self.event_bus.start()
        self.assertTrue(self.event_bus._running)
        
        await self.event_bus.stop()
        self.assertFalse(self.event_bus._running)
    
    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self):
        """Test publishing and subscribing to events."""
        await self.event_bus.start()
        
        # Set up a mock handler
        received_events = []
        async def mock_handler(event_data):
            received_events.append(event_data)
        
        # Subscribe to an event type
        await self.event_bus.subscribe_to_event("test_event", mock_handler)
        
        # Publish an event
        test_data = {"message": "test"}
        await self.event_bus.publish_event("test_event", test_data)
        
        # Give a moment for async processing
        await asyncio.sleep(0.1)
        
        # Verify the handler was called
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0], test_data)
        
        await self.event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_create_partition(self):
        """Test creating partitions."""
        await self.event_bus.start()
        
        # Create a partition
        await self.event_bus.create_partition("test_partition", ["agent1", "agent2"])
        
        # Verify the partition was created
        self.assertIn("test_partition", self.event_bus._partitions)
        self.assertEqual(self.event_bus._partitions["test_partition"], ["agent1", "agent2"])
        
        await self.event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_register_worker(self):
        """Test registering workers."""
        await self.event_bus.start()
        
        # Register a worker
        capabilities = {"model": "gpt-4", "max_tokens": 1000}
        await self.event_bus.register_worker("worker1", capabilities)
        
        # Verify the worker was registered
        self.assertIn("worker1", self.event_bus._workers)
        self.assertEqual(self.event_bus._workers["worker1"], capabilities)
        
        await self.event_bus.stop()


class TestDistributedCoordinator(unittest.TestCase):
    """Test suite for the DistributedCoordinator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.event_bus = Mock()
        self.simulation_config = Mock()
        self.coordinator = DistributedCoordinator(self.event_bus, self.simulation_config)
    
    @pytest.mark.asyncio
    async def test_coordinator_initialization(self):
        """Test that the coordinator initializes correctly."""
        self.assertIsNotNone(self.coordinator)
        self.assertEqual(self.coordinator._distributed_event_bus, self.event_bus)
        self.assertEqual(self.coordinator._simulation_config, self.simulation_config)
        self.assertEqual(len(self.coordinator._workers), 0)
    
    @pytest.mark.asyncio
    async def test_coordinator_start_stop(self):
        """Test starting and stopping the coordinator."""
        # Mock the event bus methods
        self.event_bus.subscribe_to_event = AsyncMock()
        self.event_bus.publish_event = AsyncMock()
        
        await self.coordinator.start()
        self.assertTrue(self.coordinator._running)
        
        await self.coordinator.stop()
        self.assertFalse(self.coordinator._running)
    
    @pytest.mark.asyncio
    async def test_spawn_worker(self):
        """Test spawning workers."""
        # Mock the event bus methods
        self.event_bus.subscribe_to_event = AsyncMock()
        self.event_bus.publish_event = AsyncMock()
        
        await self.coordinator.start()
        
        # Spawn a worker
        partition_config = {"partition_id": "test_partition", "agents": ["agent1"]}
        worker_id = await self.coordinator.spawn_worker(partition_config)
        
        # Verify the worker was spawned
        self.assertIsNotNone(worker_id)
        self.assertIn(worker_id, self.coordinator._workers)
        
        await self.coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_coordinate_tick_progression(self):
        """Test coordinating tick progression."""
        # Mock the event bus methods
        self.event_bus.subscribe_to_event = AsyncMock()
        self.event_bus.publish_event = AsyncMock()
        
        await self.coordinator.start()
        
        # Add a mock worker
        self.coordinator._workers["worker1"] = Mock()
        self.coordinator._workers["worker1"].send_tick = AsyncMock()
        self.coordinator._workers["worker1"].send_tick.return_value = {"status": "ack"}