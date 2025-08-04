"""
Unit tests for the DualMemoryManager class.

This module tests the functionality of the DualMemoryManager, including
the new configurability features for memory stores.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from memory_experiments.dual_memory_manager import DualMemoryManager, MemoryEvent, InMemoryStore
from memory_experiments.memory_config import MemoryConfig, MemoryMode, MemoryStoreType, DecayFunction
from constraints.constraint_config import ConstraintConfig


class TestDualMemoryManager:
    """Test cases for the DualMemoryManager class."""

    @pytest.fixture
    def base_config(self):
        """Create a base constraint config for testing."""
        return ConstraintConfig(
            max_tokens_per_action=1000,
            max_total_tokens=10000,
            token_cost_per_1k=0.002,
            violation_penalty_weight=1.0,
            grace_period_percentage=0.1,
            hard_fail_on_violation=False,
            inject_budget_status=True,
            track_token_efficiency=True
        )

    @pytest.fixture
    def memory_config(self, base_config):
        """Create a memory config for testing."""
        return MemoryConfig(
            base_config=base_config,
            memory_mode=MemoryMode.REFLECTION_ENABLED,
            short_term_capacity=100,
            short_term_retention_days=1,
            short_term_decay_function=DecayFunction.LINEAR,
            long_term_capacity=1000,
            long_term_retention_days=None,
            long_term_decay_function=DecayFunction.NONE,
            reflection_enabled=True,
            reflection_frequency_hours=24,
            short_term_store_type=MemoryStoreType.IN_MEMORY,
            long_term_store_type=MemoryStoreType.IN_MEMORY
        )

    @pytest.fixture
    def agent_id(self):
        """Return a test agent ID."""
        return "test_agent"

    @pytest.fixture
    def memory_manager(self, memory_config, agent_id):
        """Create a DualMemoryManager instance for testing."""
        return DualMemoryManager(memory_config, agent_id)

    @pytest.mark.asyncio
    async def test_initialization_with_default_config(self, memory_config, agent_id):
        """Test that DualMemoryManager initializes correctly with default config."""
        memory_manager = DualMemoryManager(memory_config, agent_id)
        
        assert memory_manager.config == memory_config
        assert memory_manager.agent_id == agent_id
        assert memory_manager.current_tick == 0
        assert memory_manager.last_reflection_time is None
        assert isinstance(memory_manager.short_term_store, InMemoryStore)
        assert isinstance(memory_manager.long_term_store, InMemoryStore)
        assert memory_manager.retrieval_stats == {}
        assert memory_manager.consolidation_stats == {}

    @pytest.mark.asyncio
    async def test_initialization_with_custom_store_types(self, base_config, agent_id):
        """Test that DualMemoryManager initializes correctly with custom store types."""
        memory_config = MemoryConfig(
            base_config=base_config,
            short_term_store_type=MemoryStoreType.IN_MEMORY,
            long_term_store_type=MemoryStoreType.IN_MEMORY
        )
        
        memory_manager = DualMemoryManager(memory_config, agent_id)
        
        assert isinstance(memory_manager.short_term_store, InMemoryStore)
        assert isinstance(memory_manager.long_term_store, InMemoryStore)

    @pytest.mark.asyncio
    async def test_create_memory_store_with_in_memory_type(self, memory_manager):
        """Test that _create_memory_store returns InMemoryStore for IN_MEMORY type."""
        store = memory_manager._create_memory_store(MemoryStoreType.IN_MEMORY)
        assert isinstance(store, InMemoryStore)

    @pytest.mark.asyncio
    async def test_create_memory_store_with_unsupported_type(self, memory_manager):
        """Test that _create_memory_store falls back to InMemoryStore for unsupported types."""
        # Create a mock unsupported store type
        unsupported_type = Mock()
        unsupported_type.value = "unsupported_store"
        
        with patch('memory_experiments.dual_memory_manager.logger') as mock_logger:
            store = memory_manager._create_memory_store(unsupported_type)
            
            # Should fall back to InMemoryStore
            assert isinstance(store, InMemoryStore)
            
            # Should log a warning
            mock_logger.warning.assert_called_once_with(
                f"Unsupported memory store type: {unsupported_type}. Falling back to InMemoryStore."
            )

    @pytest.mark.asyncio
    async def test_store_event_with_memory_enabled(self, memory_manager):
        """Test storing an event when memory is enabled."""
        # Create a mock event
        mock_event = Mock()
        mock_event.event_type = "TestEvent"
        mock_event.timestamp = datetime.now()
        
        # Mock the _calculate_importance_score method
        with patch.object(memory_manager, '_calculate_importance_score', return_value=0.5) as mock_calculate:
            result = await memory_manager.store_event(mock_event)
            
            # Should return True
            assert result is True
            
            # Should calculate importance score
            mock_calculate.assert_called_once()
            
            # Should store in short-term store by default
            assert await memory_manager.short_term_store.size() == 1
            assert await memory_manager.long_term_store.size() == 0

    @pytest.mark.asyncio
    async def test_store_event_with_memory_disabled(self, base_config, agent_id):
        """Test storing an event when memory is disabled."""
        memory_config = MemoryConfig(
            base_config=base_config,
            memory_mode=MemoryMode.MEMORY_FREE
        )
        
        memory_manager = DualMemoryManager(memory_config, agent_id)
        
        # Create a mock event
        mock_event = Mock()
        mock_event.event_type = "TestEvent"
        mock_event.timestamp = datetime.now()
        
        result = await memory_manager.store_event(mock_event)
        
        # Should return True (no-op)
        assert result is True
        
        # Should not store anything
        assert await memory_manager.short_term_store.size() == 0
        assert await memory_manager.long_term_store.size() == 0

    @pytest.mark.asyncio
    async def test_retrieve_memories_with_memory_enabled(self, memory_manager):
        """Test retrieving memories when memory is enabled."""
        # Create a mock event
        mock_event = Mock()
        mock_event.event_type = "TestEvent"
        mock_event.timestamp = datetime.now()
        
        # Store the event
        await memory_manager.store_event(mock_event)
        
        # Retrieve memories
        memories = await memory_manager.retrieve_memories("test")
        
        # Should return a list
        assert isinstance(memories, list)
        
        # Should contain the stored memory
        assert len(memories) == 1
        assert memories[0].event_type == "TestEvent"

    @pytest.mark.asyncio
    async def test_retrieve_memories_with_memory_disabled(self, base_config, agent_id):
        """Test retrieving memories when memory is disabled."""
        memory_config = MemoryConfig(
            base_config=base_config,
            memory_mode=MemoryMode.MEMORY_FREE
        )
        
        memory_manager = DualMemoryManager(memory_config, agent_id)
        
        # Retrieve memories
        memories = await memory_manager.retrieve_memories("test")
        
        # Should return an empty list
        assert memories == []

    @pytest.mark.asyncio
    async def test_get_memory_summary(self, memory_manager):
        """Test getting memory summary."""
        # Create a mock event
        mock_event = Mock()
        mock_event.event_type = "TestEvent"
        mock_event.timestamp = datetime.now()
        
        # Store the event
        await memory_manager.store_event(mock_event)
        
        # Get summary
        summary = await memory_manager.get_memory_summary()
        
        # Should return a dictionary with expected keys
        assert isinstance(summary, dict)
        assert "short_term_memory_count" in summary
        assert "long_term_memory_count" in summary
        assert "total_memory_count" in summary
        assert "memory_mode" in summary
        assert "last_reflection" in summary
        assert "retrieval_stats" in summary
        assert "consolidation_stats" in summary
        
        # Should have correct counts
        assert summary["short_term_memory_count"] == 1
        assert summary["long_term_memory_count"] == 0
        assert summary["total_memory_count"] == 1

    @pytest.mark.asyncio
    async def test_clear_memories(self, memory_manager):
        """Test clearing memories."""
        # Create a mock event
        mock_event = Mock()
        mock_event.event_type = "TestEvent"
        mock_event.timestamp = datetime.now()
        
        # Store the event
        await memory_manager.store_event(mock_event)
        
        # Verify storage
        assert await memory_manager.short_term_store.size() == 1
        
        # Clear short-term memories
        await memory_manager.clear_memories("short_term")
        
        # Should be cleared
        assert await memory_manager.short_term_store.size() == 0
        
        # Store again
        await memory_manager.store_event(mock_event)
        
        # Clear all memories
        await memory_manager.clear_memories("all")
        
        # Should all be cleared
        assert await memory_manager.short_term_store.size() == 0
        assert await memory_manager.long_term_store.size() == 0
        assert memory_manager.retrieval_stats == {}
        assert memory_manager.consolidation_stats == {}
        assert memory_manager.last_reflection_time is None

    @pytest.mark.asyncio
    async def test_update_tick(self, memory_manager):
        """Test updating the current tick."""
        assert memory_manager.current_tick == 0
        
        memory_manager.update_tick(5)
        
        assert memory_manager.current_tick == 5

    @pytest.mark.asyncio
    async def test_should_reflect(self, memory_manager):
        """Test reflection triggering logic."""
        current_time = datetime.now()
        
        # Should reflect if last_reflection_time is None
        assert await memory_manager.should_reflect(current_time) is True
        
        # Set last reflection time
        memory_manager.last_reflection_time = current_time
        
        # Should not reflect if not enough time has passed
        assert await memory_manager.should_reflect(current_time) is False
        
        # Should reflect if enough time has passed
        future_time = datetime.fromtimestamp(current_time.timestamp() + (25 * 3600))  # 25 hours later
        assert await memory_manager.should_reflect(future_time) is True

    @pytest.mark.asyncio
    async def test_get_memories_for_promotion(self, memory_manager):
        """Test getting memories for promotion."""
        # Create a mock event
        mock_event = Mock()
        mock_event.event_type = "TestEvent"
        mock_event.timestamp = datetime.now()
        
        # Store the event
        await memory_manager.store_event(mock_event)
        
        # Get memories for promotion
        candidates = await memory_manager.get_memories_for_promotion()
        
        # Should return a list
        assert isinstance(candidates, list)
        
        # Should be empty because the memory is too recent
        assert len(candidates) == 0

    @pytest.mark.asyncio
    async def test_promote_memories(self, memory_manager):
        """Test promoting memories from short-term to long-term storage."""
        # Create a mock event
        mock_event = Mock()
        mock_event.event_type = "TestEvent"
        mock_event.timestamp = datetime.now()
        
        # Store the event
        await memory_manager.store_event(mock_event)
        
        # Get the memory event
        memories = await memory_manager.short_term_store.get_all()
        assert len(memories) == 1
        
        memory_to_promote = memories[0]
        
        # Promote the memory
        result = await memory_manager.promote_memories([memory_to_promote])
        
        # Should return True
        assert result is True
        
        # Should be in long-term store
        assert await memory_manager.short_term_store.size() == 0
        assert await memory_manager.long_term_store.size() == 1
        
        # Should be marked as promoted
        promoted_memories = await memory_manager.long_term_store.get_all()
        assert promoted_memories[0].promoted_to_long_term is True
        assert promoted_memories[0].promotion_timestamp is not None