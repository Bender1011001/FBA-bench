import unittest
import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from memory_experiments.dual_memory_manager import DualMemoryManager
from memory_experiments.experiment_protocols import ExperimentProtocols
from memory_experiments.memory_metrics import MemoryMetrics


class TestDualMemoryManager(unittest.TestCase):
    """Test suite for the DualMemoryManager class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.dual_memory_manager = DualMemoryManager()
    
    def test_dual_memory_manager_initialization(self):
        """Test that the dual memory manager initializes correctly."""
        self.assertIsNotNone(self.dual_memory_manager)
        self.assertIsNotNone(self.dual_memory_manager._working_memory)
        self.assertIsNotNone(self.dual_memory_manager._long_term_memory)
        self.assertEqual(len(self.dual_memory_manager._working_memory), 0)
        self.assertEqual(len(self.dual_memory_manager._long_term_memory), 0)
    
    def test_store_working_memory(self):
        """Test storing data in working memory."""
        memory_data = {
            "memory_id": "mem1",
            "content": {"type": "observation", "data": {"price": 10.0}},
            "timestamp": datetime.now(),
            "importance": 0.8,
            "tags": ["pricing", "observation"]
        }
        
        memory_id = self.dual_memory_manager.store_working_memory(memory_data)
        
        self.assertEqual(memory_id, "mem1")
        self.assertIn("mem1", self.dual_memory_manager._working_memory)
        self.assertEqual(self.dual_memory_manager._working_memory["mem1"]["content"]["data"]["price"], 10.0)
        self.assertEqual(self.dual_memory_manager._working_memory["mem1"]["importance"], 0.8)
    
    def test_retrieve_working_memory(self):
        """Test retrieving data from working memory."""
        memory_data = {
            "memory_id": "mem1",
            "content": {"type": "observation", "data": {"price": 10.0}},
            "timestamp": datetime.now(),
            "importance": 0.8,
            "tags": ["pricing", "observation"]
        }
        
        self.dual_memory_manager.store_working_memory(memory_data)
        
        memory = self.dual_memory_manager.retrieve_working_memory("mem1")
        
        self.assertIsNotNone(memory)
        self.assertEqual(memory["memory_id"], "mem1")
        self.assertEqual(memory["content"]["data"]["price"], 10.0)
        self.assertEqual(memory["importance"], 0.8)
    
    def test_store_long_term_memory(self):
        """Test storing data in long-term memory."""
        memory_data = {
            "memory_id": "mem1",
            "content": {"type": "knowledge", "data": {"rule": "price_elasticity"}},
            "timestamp": datetime.now(),
            "importance": 0.9,
            "tags": ["pricing", "knowledge"]
        }
        
        memory_id = self.dual_memory_manager.store_long_term_memory(memory_data)
        
        self.assertEqual(memory_id, "mem1")
        self.assertIn("mem1", self.dual_memory_manager._long_term_memory)
        self.assertEqual(self.dual_memory_manager._long_term_memory["mem1"]["content"]["data"]["rule"], "price_elasticity")
        self.assertEqual(self.dual_memory_manager._long_term_memory["mem1"]["importance"], 0.9)
    
    def test_retrieve_long_term_memory(self):
        """Test retrieving data from long-term memory."""
        memory_data = {
            "memory_id": "mem1",
            "content": {"type": "knowledge", "data": {"rule": "price_elasticity"}},
            "timestamp": datetime.now(),
            "importance": 0.9,
            "tags": ["pricing", "knowledge"]
        }
        
        self.dual_memory_manager.store_long_term_memory(memory_data)
        
        memory = self.dual_memory_manager.retrieve_long_term_memory("mem1")
        
        self.assertIsNotNone(memory)
        self.assertEqual(memory["memory_id"], "mem1")
        self.assertEqual(memory["content"]["data"]["rule"], "price_elasticity")
        self.assertEqual(memory["importance"], 0.9)
    
    def test_search_working_memory(self):
        """Test searching working memory."""
        # Store multiple memories
        mem1_data = {
            "memory_id": "mem1",
            "content": {"type": "observation", "data": {"price": 10.0}},
            "timestamp": datetime.now(),
            "importance": 0.8,
            "tags": ["pricing", "observation"]
        }
        
        mem2_data = {
            "memory_id": "mem2",
            "content": {"type": "observation", "data": {"quantity": 100}},
            "timestamp": datetime.now(),
            "importance": 0.7,
            "tags": ["inventory", "observation"]
        }
        
        self.dual_memory_manager.store_working_memory(mem1_data)
        self.dual_memory_manager.store_working_memory(mem2_data)
        
        # Search by tag
        results = self.dual_memory_manager.search_working_memory({"tags": ["pricing"]})
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["memory_id"], "mem1")
        self.assertEqual(results[0]["content"]["data"]["price"], 10.0)
    
    def test_search_long_term_memory(self):
        """Test searching long-term memory."""
        # Store multiple memories
        mem1_data = {
            "memory_id": "mem1",
            "content": {"type": "knowledge", "data": {"rule": "price_elasticity"}},
            "timestamp": datetime.now(),
            "importance": 0.9,
            "tags": ["pricing", "knowledge"]
        }
        
        mem2_data = {
            "memory_id": "mem2",
            "content": {"type": "knowledge", "data": {"rule": "inventory_turnover"}},
            "timestamp": datetime.now(),
            "importance": 0.8,
            "tags": ["inventory", "knowledge"]
        }
        
        self.dual_memory_manager.store_long_term_memory(mem1_data)
        self.dual_memory_manager.store_long_term_memory(mem2_data)
        
        # Search by tag
        results = self.dual_memory_manager.search_long_term_memory({"tags": ["pricing"]})
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["memory_id"], "mem1")
        self.assertEqual(results[0]["content"]["data"]["rule"], "price_elasticity")
    
    def test_consolidate_memory(self):
        """Test consolidating working memory to long-term memory."""
        # Store a memory in working memory
        mem_data = {
            "memory_id": "mem1",
            "content": {"type": "observation", "data": {"price": 10.0}},
            "timestamp": datetime.now(),
            "importance": 0.9,  # High importance for consolidation
            "tags": ["pricing", "observation"]
        }
        
        self.dual_memory_manager.store_working_memory(mem_data)
        
        # Consolidate memory
        consolidated_ids = self.dual_memory_manager.consolidate_memory(importance_threshold=0.8)
        
        self.assertIn("mem1", consolidated_ids)
        self.assertNotIn("mem1", self.dual_memory_manager._working_memory)
        self.assertIn("mem1", self.dual_memory_manager._long_term_memory)
    
    def test_forget_working_memory(self):
        """Test forgetting working memory."""
        # Store a memory in working memory
        mem_data = {
            "memory_id": "mem1",
            "content": {"type": "observation", "data": {"price": 10.0}},
            "timestamp": datetime.now() - timedelta(hours=2),  # Old memory
            "importance": 0.5,
            "tags": ["pricing", "observation"]
        }
        
        self.dual_memory_manager.store_working_memory(mem_data)
        
        # Forget old memories
        forgotten_ids = self.dual_memory_manager.forget_working_memory(age_threshold=timedelta(hours=1))
        
        self.assertIn("mem1", forgotten_ids)
        self.assertNotIn("mem1", self.dual_memory_manager._working_memory)
    
    def test_get_memory_statistics(self):
        """Test getting memory statistics."""
        # Store memories in both working and long-term memory
        working_mem_data = {
            "memory_id": "mem1",
            "content": {"type": "observation", "data": {"price": 10.0}},
            "timestamp": datetime.now(),
            "importance": 0.8,
            "tags": ["pricing", "observation"]
        }
        
        long_term_mem_data = {
            "memory_id": "mem2",
            "content": {"type": "knowledge", "data": {"rule": "price_elasticity"}},
            "timestamp": datetime.now(),
            "importance": 0.9,
            "tags": ["pricing", "knowledge"]
        }
        
        self.dual_memory_manager.store_working_memory(working_mem_data)
        self.dual_memory_manager.store_long_term_memory(long_term_mem_data)
        
        # Get statistics
        stats = self.dual_memory_manager.get_memory_statistics()
        
        self.assertIsNotNone(stats)
        self.assertEqual(stats["working_memory_count"], 1)
        self.assertEqual(stats["long_term_memory_count"], 1)
        self.assertEqual(stats["total_memory_count"], 2)
        self.assertIn("average_importance", stats)
        self.assertIn("tag_distribution", stats)


class TestExperimentProtocols(unittest.TestCase):
    """Test suite for the ExperimentProtocols class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.experiment_protocols = ExperimentProtocols()
    
    def test_experiment_protocols_initialization(self):
        """Test that the experiment protocols initialize correctly."""
        self.assertIsNotNone(self.experiment_protocols)
        self.assertEqual(len(self.experiment_protocols._protocols), 0)
        self.assertEqual(len(self.experiment_protocols._experiments), 0)
    
    def test_define_protocol(self):
        """Test defining an experiment protocol."""
        protocol_data = {
            "protocol_id": "protocol1",
            "name": "Memory Retention Test",
            "description": "Test the retention of memories over time",
            "memory_types": ["working_memory", "long_term_memory"],
            "experiment_duration": timedelta(hours=24),
            "test_intervals": [timedelta(hours=1), timedelta(hours=6), timedelta(hours=24)],
            "metrics": ["retention_rate", "recall_accuracy", "retrieval_time"],
            "parameters": {
                "memory_count": 100,
                "importance_distribution": "uniform",
                "tag_distribution": "uniform"
            }
        }
        
        protocol_id = self.experiment_protocols.define_protocol(protocol_data)
        
        self.assertEqual(protocol_id, "protocol1")
        self.assertIn("protocol1", self.experiment_protocols._protocols)
        self.assertEqual(self.experiment_protocols._protocols["protocol1"]["name"], "Memory Retention Test")
        self.assertEqual(len(self.experiment_protocols._protocols["protocol1"]["test_intervals"]), 3)
    
    def test_get_protocol(self):
        """Test getting an experiment protocol."""
        protocol_data = {
            "protocol_id": "protocol1",
            "name": "Memory Retention Test",
            "description": "Test the retention of memories over time",
            "memory_types": ["working_memory", "long_term_memory"],
            "experiment_duration": timedelta(hours=24),
            "test_intervals": [timedelta(hours=1), timedelta(hours=6), timedelta(hours=24)],
            "metrics": ["retention_rate", "recall_accuracy", "retrieval_time"],
            "parameters": {
                "memory_count": 100,
                "importance_distribution": "uniform",
                "tag_distribution": "uniform"
            }
        }
        
        self.experiment_protocols.define_protocol(protocol_data)
        
        protocol = self.experiment_protocols.get_protocol("protocol1")
        
        self.assertIsNotNone(protocol)
        self.assertEqual(protocol["protocol_id"], "protocol1")
        self.assertEqual(protocol["name"], "Memory Retention Test")
        self.assertEqual(len(protocol["test_intervals"]), 3)
    
    def test_run_experiment(self):
        """Test running an experiment."""
        # Define a protocol
        protocol_data = {
            "protocol_id": "protocol1",
            "name": "Memory Retention Test",
            "description": "Test the retention of memories over time",
            "memory_types": ["working_memory", "long_term_memory"],
            "experiment_duration": timedelta(hours=24),
            "test_intervals": [timedelta(hours=1), timedelta(hours=6), timedelta(hours=24)],
            "metrics": ["retention_rate", "recall_accuracy", "retrieval_time"],
            "parameters": {
                "memory_count": 100,
                "importance_distribution": "uniform",
                "tag_distribution": "uniform"
            }
        }
        
        self.experiment_protocols.define_protocol(protocol_data)
        
        # Mock the memory manager
        memory_manager = Mock()
        memory_manager.store_working_memory = Mock(return_value="mem1")
        memory_manager.store_long_term_memory = Mock(return_value="mem2")
        memory_manager.retrieve_working_memory = Mock(return_value={
            "memory_id": "mem1",
            "content": {"type": "observation", "data": {"price": 10.0}},
            "timestamp": datetime.now(),
            "importance": 0.8,
            "tags": ["pricing", "observation"]
        })
        memory_manager.retrieve_long_term_memory = Mock(return_value={
            "memory_id": "mem2",
            "content": {"type": "knowledge", "data": {"rule": "price_elasticity"}},
            "timestamp": datetime.now(),
            "importance": 0.9,
            "tags": ["pricing", "knowledge"]
        })
        
        # Run experiment
        experiment_id = self.experiment_protocols.run_experiment("protocol1", memory_manager)
        
        self.assertIsNotNone(experiment_id)
        self.assertIn(experiment_id, self.experiment_protocols._experiments)
        self.assertEqual(self.experiment_protocols._experiments[experiment_id]["protocol_id"], "protocol1")
        self.assertIn("start_time", self.experiment_protocols._experiments[experiment_id])
    
    def test_get_experiment(self):
        """Test getting an experiment."""
        # Define a protocol
        protocol_data = {
            "protocol_id": "protocol1",
            "name": "Memory Retention Test",
            "description": "Test the retention of memories over time",
            "memory_types": ["working_memory", "long_term_memory"],
            "experiment_duration": timedelta(hours=24),
            "test_intervals": [timedelta(hours=1), timedelta(hours=6), timedelta(hours=24)],
            "metrics": ["retention_rate", "recall_accuracy", "retrieval_time"],
            "parameters": {
                "memory_count": 100,
                "importance_distribution": "uniform",
                "tag_distribution": "uniform"
            }
        }
        
        self.experiment_protocols.define_protocol(protocol_data)
        
        # Mock the memory manager
        memory_manager = Mock()
        memory_manager.store_working_memory = Mock(return_value="mem1")
        memory_manager.store_long_term_memory = Mock(return_value="mem2")
        memory_manager.retrieve_working_memory = Mock(return_value={
            "memory_id": "mem1",
            "content": {"type": "observation", "data": {"price": 10.0}},
            "timestamp": datetime.now(),
            "importance": 0.8,
            "tags": ["pricing", "observation"]
        })
        memory_manager.retrieve_long_term_memory = Mock(return_value={
            "memory_id": "mem2",
            "content": {"type": "knowledge", "data": {"rule": "price_elasticity"}},
            "timestamp": datetime.now(),
            "importance": 0.9,
            "tags": ["pricing", "knowledge"]
        })
        
        # Run experiment
        experiment_id = self.experiment_protocols.run_experiment("protocol1", memory_manager)
        
        # Get experiment
        experiment = self.experiment_protocols.get_experiment(experiment_id)
        
        self.assertIsNotNone(experiment)
        self.assertEqual(experiment["experiment_id"], experiment_id)
        self.assertEqual(experiment["protocol_id"], "protocol1")
        self.assertIn("start_time", experiment)
    
    def test_get_experiment_results(self):
        """Test getting experiment results."""
        # Define a protocol
        protocol_data = {
            "protocol_id": "protocol1",
            "name": "Memory Retention Test",
            "description": "Test the retention of memories over time",
            "memory_types": ["working_memory", "long_term_memory"],
            "experiment_duration": timedelta(hours=24),
            "test_intervals": [timedelta(hours=1), timedelta(hours=6), timedelta(hours=24)],
            "metrics": ["retention_rate", "recall_accuracy", "retrieval_time"],
            "parameters": {
                "memory_count": 100,
                "importance_distribution": "uniform",
                "tag_distribution": "uniform"
            }
        }
        
        self.experiment_protocols.define_protocol(protocol_data)
        
        # Mock the memory manager
        memory_manager = Mock()
        memory_manager.store_working_memory = Mock(return_value="mem1")
        memory_manager.store_long_term_memory = Mock(return_value="mem2")
        memory_manager.retrieve_working_memory = Mock(return_value={
            "memory_id": "mem1",
            "content": {"type": "observation", "data": {"price": 10.0}},
            "timestamp": datetime.now(),
            "importance": 0.8,
            "tags": ["pricing", "observation"]
        })
        memory_manager.retrieve_long_term_memory = Mock(return_value={
            "memory_id": "mem2",
            "content": {"type": "knowledge", "data": {"rule": "price_elasticity"}},
            "timestamp": datetime.now(),
            "importance": 0.9,
            "tags": ["pricing", "knowledge"]
        })
        
        # Run experiment
        experiment_id = self.experiment_protocols.run_experiment("protocol1", memory_manager)
        
        # Mock results
        self.experiment_protocols._experiments[experiment_id]["results"] = {
            "retention_rate": 0.85,
            "recall_accuracy": 0.90,
            "retrieval_time": 0.05
        }
        
        # Get results
        results = self.experiment_protocols.get_experiment_results(experiment_id)
        
        self.assertIsNotNone(results)
        self.assertEqual(results["retention_rate"], 0.85)
        self.assertEqual(results["recall_accuracy"], 0.90)
        self.assertEqual(results["retrieval_time"], 0.05)
    
    def test_list_protocols(self):
        """Test listing all protocols."""
        # Define multiple protocols
        protocol1_data = {
            "protocol_id": "protocol1",
            "name": "Memory Retention Test",
            "description": "Test the retention of memories over time",
            "memory_types": ["working_memory", "long_term_memory"],
            "experiment_duration": timedelta(hours=24),
            "test_intervals": [timedelta(hours=1), timedelta(hours=6), timedelta(hours=24)],
            "metrics": ["retention_rate", "recall_accuracy", "retrieval_time"],
            "parameters": {
                "memory_count": 100,
                "importance_distribution": "uniform",
                "tag_distribution": "uniform"
            }
        }
        
        protocol2_data = {
            "protocol_id": "protocol2",
            "name": "Memory Capacity Test",
            "description": "Test the capacity of memory systems",
            "memory_types": ["working_memory", "long_term_memory"],
            "experiment_duration": timedelta(hours=12),
            "test_intervals": [timedelta(hours=1), timedelta(hours=6), timedelta(hours=12)],
            "metrics": ["capacity", "utilization", "efficiency"],
            "parameters": {
                "memory_count": 200,
                "importance_distribution": "normal",
                "tag_distribution": "normal"
            }
        }
        
        self.experiment_protocols.define_protocol(protocol1_data)
        self.experiment_protocols.define_protocol(protocol2_data)
        
        # List protocols
        protocols = self.experiment_protocols.list_protocols()
        
        self.assertEqual(len(protocols), 2)
        self.assertTrue(any(p["protocol_id"] == "protocol1" for p in protocols))
        self.assertTrue(any(p["protocol_id"] == "protocol2" for p in protocols))
    
    def test_list_experiments(self):
        """Test listing all experiments."""
        # Define a protocol
        protocol_data = {
            "protocol_id": "protocol1",
            "name": "Memory Retention Test",
            "description": "Test the retention of memories over time",
            "memory_types": ["working_memory", "long_term_memory"],
            "experiment_duration": timedelta(hours=24),
            "test_intervals": [timedelta(hours=1), timedelta(hours=6), timedelta(hours=24)],
            "metrics": ["retention_rate", "recall_accuracy", "retrieval_time"],
            "parameters": {
                "memory_count": 100,
                "importance_distribution": "uniform",
                "tag_distribution": "uniform"
            }
        }
        
        self.experiment_protocols.define_protocol(protocol_data)
        
        # Mock the memory manager
        memory_manager = Mock()
        memory_manager.store_working_memory = Mock(return_value="mem1")
        memory_manager.store_long_term_memory = Mock(return_value="mem2")
        memory_manager.retrieve_working_memory = Mock(return_value={
            "memory_id": "mem1",
            "content": {"type": "observation", "data": {"price": 10.0}},
            "timestamp": datetime.now(),
            "importance": 0.8,
            "tags": ["pricing", "observation"]
        })
        memory_manager.retrieve_long_term_memory = Mock(return_value={
            "memory_id": "mem2",
            "content": {"type": "knowledge", "data": {"rule": "price_elasticity"}},
            "timestamp": datetime.now(),
            "importance": 0.9,
            "tags": ["pricing", "knowledge"]
        })
        
        # Run multiple experiments
        experiment_id1 = self.experiment_protocols.run_experiment("protocol1", memory_manager)
        experiment_id2 = self.experiment_protocols.run_experiment("protocol1", memory_manager)
        
        # List experiments
        experiments = self.experiment_protocols.list_experiments()
        
        self.assertEqual(len(experiments), 2)
        self.assertTrue(any(e["experiment_id"] == experiment_id1 for e in experiments))
        self.assertTrue(any(e["experiment_id"] == experiment_id2 for e in experiments))


class TestMemoryMetrics(unittest.TestCase):
    """Test suite for the MemoryMetrics class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.memory_metrics = MemoryMetrics()
    
    def test_memory_metrics_initialization(self):
        """Test that the memory metrics initialize correctly."""
        self.assertIsNotNone(self.memory_metrics)
        self.assertEqual(len(self.memory_metrics._metrics), 0)
    
    def test_calculate_retention_rate(self):
        """Test calculating retention rate."""
        # Mock memory data
        memory_data = {
            "stored_memories": [
                {"memory_id": "mem1", "timestamp": datetime.now() - timedelta(hours=1)},
                {"memory_id": "mem2", "timestamp": datetime.now() - timedelta(hours=2)},
                {"memory_id": "mem3", "timestamp": datetime.now() - timedelta(hours=3)}
            ],
            "retrieved_memories": [
                {"memory_id": "mem1", "timestamp": datetime.now()},
                {"memory_id": "mem2", "timestamp": datetime.now()}
            ]
        }
        
        retention_rate = self.memory_metrics.calculate_retention_rate(memory_data)
        
        self.assertEqual(retention_rate, 2/3)  # 2 out of 3 memories retrieved
    
    def test_calculate_recall_accuracy(self):
        """Test calculating recall accuracy."""
        # Mock memory data
        memory_data = {
            "query_memories": [
                {"memory_id": "mem1", "content": {"data": {"price": 10.0}}},
                {"memory_id": "mem2", "content": {"data": {"quantity": 100}}},
                {"memory_id": "mem3", "content": {"data": {"discount": 0.1}}}
            ],
            "retrieved_memories": [
                {"memory_id": "mem1", "content": {"data": {"price": 10.0}}},
                {"memory_id": "mem2", "content": {"data": {"quantity": 100}}}
            ]
        }
        
        recall_accuracy = self.memory_metrics.calculate_recall_accuracy(memory_data)
        
        self.assertEqual(recall_accuracy, 2/3)  # 2 out of 3 memories correctly recalled
    
    def test_calculate_retrieval_time(self):
        """Test calculating retrieval time."""
        # Mock memory data
        memory_data = {
            "retrieval_times": [
                0.05,  # seconds
                0.10,
                0.03,
                0.08
            ]
        }
        
        retrieval_time = self.memory_metrics.calculate_retrieval_time(memory_data)
        
        self.assertEqual(retrieval_time, 0.065)  # Average retrieval time
    
    def test_calculate_memory_capacity(self):
        """Test calculating memory capacity."""
        # Mock memory data
        memory_data = {
            "total_memory_slots": 1000,
            "used_memory_slots": 750
        }
        
        capacity = self.memory_metrics.calculate_memory_capacity(memory_data)
        
        self.assertEqual(capacity, 0.75)  # 75% capacity utilization
    
    def test_calculate_memory_efficiency(self):
        """Test calculating memory efficiency."""
        # Mock memory data
        memory_data = {
            "successful_retrievals": 80,
            "failed_retrievals": 20,
            "total_retrieval_time": 5.0  # seconds
        }
        
        efficiency = self.memory_metrics.calculate_memory_efficiency(memory_data)
        
        self.assertEqual(efficiency, 16.0)  # 80 successful retrievals / 5 seconds
    
    def test_calculate_memory_importance_distribution(self):
        """Test calculating memory importance distribution."""
        # Mock memory data
        memory_data = {
            "memories": [
                {"importance": 0.9},
                {"importance": 0.8},
                {"importance": 0.7},
                {"importance": 0.6},
                {"importance": 0.5}
            ]
        }
        
        distribution = self.memory_metrics.calculate_memory_importance_distribution(memory_data)
        
        self.assertIsNotNone(distribution)
        self.assertIn("mean", distribution)
        self.assertIn("median", distribution)
        self.assertIn("std_dev", distribution)
        self.assertIn("min", distribution)
        self.assertIn("max", distribution)
        self.assertEqual(distribution["mean"], 0.7)
        self.assertEqual(distribution["median"], 0.7)
    
    def test_calculate_memory_tag_distribution(self):
        """Test calculating memory tag distribution."""
        # Mock memory data
        memory_data = {
            "memories": [
                {"tags": ["pricing", "observation"]},
                {"tags": ["pricing", "knowledge"]},
                {"tags": ["inventory", "observation"]},
                {"tags": ["inventory", "knowledge"]},
                {"tags": ["pricing", "observation"]}
            ]
        }
        
        distribution = self.memory_metrics.calculate_memory_tag_distribution(memory_data)
        
        self.assertIsNotNone(distribution)
        self.assertEqual(distribution["pricing"], 3)
        self.assertEqual(distribution["inventory"], 2)
        self.assertEqual(distribution["observation"], 3)
        self.assertEqual(distribution["knowledge"], 2)
    
    def test_calculate_memory_age_distribution(self):
        """Test calculating memory age distribution."""
        # Mock memory data
        now = datetime.now()
        memory_data = {
            "memories": [
                {"timestamp": now - timedelta(hours=1)},
                {"timestamp": now - timedelta(hours=2)},
                {"timestamp": now - timedelta(hours=3)},
                {"timestamp": now - timedelta(hours=4)},
                {"timestamp": now - timedelta(hours=5)}
            ]
        }
        
        distribution = self.memory_metrics.calculate_memory_age_distribution(memory_data)
        
        self.assertIsNotNone(distribution)
        self.assertIn("mean_age", distribution)
        self.assertIn("median_age", distribution)
        self.assertIn("std_dev_age", distribution)
        self.assertIn("min_age", distribution)
        self.assertIn("max_age", distribution)
    
    def test_generate_memory_report(self):
        """Test generating a memory report."""
        # Mock memory data
        memory_data = {
            "stored_memories": [
                {"memory_id": "mem1", "timestamp": datetime.now() - timedelta(hours=1)},
                {"memory_id": "mem2", "timestamp": datetime.now() - timedelta(hours=2)},
                {"memory_id": "mem3", "timestamp": datetime.now() - timedelta(hours=3)}
            ],
            "retrieved_memories": [
                {"memory_id": "mem1", "timestamp": datetime.now()},
                {"memory_id": "mem2", "timestamp": datetime.now()}
            ],
            "query_memories": [
                {"memory_id": "mem1", "content": {"data": {"price": 10.0}}},
                {"memory_id": "mem2", "content": {"data": {"quantity": 100}}},
                {"memory_id": "mem3", "content": {"data": {"discount": 0.1}}}
            ],
            "retrieval_times": [0.05, 0.10, 0.03, 0.08],
            "total_memory_slots": 1000,
            "used_memory_slots": 750,
            "successful_retrievals": 80,
            "failed_retrievals": 20,
            "total_retrieval_time": 5.0,
            "memories": [
                {"importance": 0.9, "tags": ["pricing", "observation"], "timestamp": datetime.now() - timedelta(hours=1)},
                {"importance": 0.8, "tags": ["pricing", "knowledge"], "timestamp": datetime.now() - timedelta(hours=2)},
                {"importance": 0.7, "tags": ["inventory", "observation"], "timestamp": datetime.now() - timedelta(hours=3)},
                {"importance": 0.6, "tags": ["inventory", "knowledge"], "timestamp": datetime.now() - timedelta(hours=4)},
                {"importance": 0.5, "tags": ["pricing", "observation"], "timestamp": datetime.now() - timedelta(hours=5)}
            ]
        }
        
        report = self.memory_metrics.generate_memory_report(memory_data)
        
        self.assertIsNotNone(report)
        self.assertIn("retention_rate", report)
        self.assertIn("recall_accuracy", report)
        self.assertIn("retrieval_time", report)
        self.assertIn("capacity", report)
        self.assertIn("efficiency", report)
        self.assertIn("importance_distribution", report)
        self.assertIn("tag_distribution", report)
        self.assertIn("age_distribution", report)
        self.assertEqual(report["retention_rate"], 2/3)
        self.assertEqual(report["recall_accuracy"], 2/3)
        self.assertEqual(report["retrieval_time"], 0.065)
        self.assertEqual(report["capacity"], 0.75)
        self.assertEqual(report["efficiency"], 16.0)


if __name__ == '__main__':
    unittest.main()