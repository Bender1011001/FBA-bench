"""
Unit tests for the benchmarking core engine.

This module contains comprehensive unit tests for the BenchmarkEngine class
and related components in the benchmarking framework.
"""

import unittest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from benchmarking.core.engine import (
    BenchmarkEngine,
    BenchmarkResult,
    BenchmarkRun,
    BenchmarkStatus,
    BenchmarkError
)
from benchmarking.config.manager import ConfigurationManager
from benchmarking.integration.manager import IntegrationManager


class TestBenchmarkEngine(unittest.TestCase):
    """Test cases for BenchmarkEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = Mock(spec=ConfigurationManager)
        self.integration_manager = Mock(spec=IntegrationManager)
        self.engine = BenchmarkEngine(self.config_manager, self.integration_manager)
        
        # Mock configuration
        self.test_config = {
            "benchmark_id": "test_benchmark",
            "name": "Test Benchmark",
            "description": "A test benchmark configuration",
            "version": "1.0.0",
            "environment": {
                "deterministic": True,
                "random_seed": 42,
                "parallel_execution": False,
                "max_workers": 1
            },
            "scenarios": [
                {
                    "id": "test_scenario",
                    "name": "Test Scenario",
                    "type": "test",
                    "enabled": True,
                    "priority": 1,
                    "config": {
                        "duration": 100,
                        "complexity": "medium"
                    }
                }
            ],
            "agents": [
                {
                    "id": "test_agent",
                    "name": "Test Agent",
                    "framework": "test",
                    "enabled": True,
                    "config": {
                        "model": "test_model",
                        "temperature": 0.5
                    }
                }
            ],
            "metrics": {
                "categories": ["cognitive", "business"],
                "custom_metrics": []
            },
            "execution": {
                "runs_per_scenario": 2,
                "max_duration": 0,
                "timeout": 300,
                "retry_on_failure": True,
                "max_retries": 3
            },
            "output": {
                "format": "json",
                "path": "./test_results",
                "include_detailed_logs": False,
                "include_audit_trail": True
            },
            "validation": {
                "enabled": True,
                "statistical_significance": True,
                "confidence_level": 0.95,
                "reproducibility_check": True
            },
            "metadata": {
                "author": "Test Author",
                "created": "2025-01-01T00:00:00Z",
                "tags": ["test"]
            }
        }
    
    def test_init(self):
        """Test BenchmarkEngine initialization."""
        self.assertIsInstance(self.engine, BenchmarkEngine)
        self.assertEqual(self.engine.config_manager, self.config_manager)
        self.assertEqual(self.engine.integration_manager, self.integration_manager)
        self.assertEqual(len(self.engine.active_runs), 0)
        self.assertEqual(len(self.engine.completed_runs), 0)
    
    def test_validate_configuration_valid(self):
        """Test configuration validation with valid configuration."""
        # Mock config manager validation
        self.config_manager.validate_config.return_value = (True, [])
        
        is_valid, errors = self.engine._validate_configuration(self.test_config)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        self.config_manager.validate_config.assert_called_once_with(self.test_config)
    
    def test_validate_configuration_invalid(self):
        """Test configuration validation with invalid configuration."""
        # Mock config manager validation with errors
        self.config_manager.validate_config.return_value = (False, ["Invalid configuration"])
        
        is_valid, errors = self.engine._validate_configuration(self.test_config)
        
        self.assertFalse(is_valid)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0], "Invalid configuration")
    
    def test_validate_configuration_missing_required_fields(self):
        """Test configuration validation with missing required fields."""
        invalid_config = self.test_config.copy()
        del invalid_config["benchmark_id"]
        
        is_valid, errors = self.engine._validate_configuration(invalid_config)
        
        self.assertFalse(is_valid)
        self.assertTrue(any("benchmark_id" in error for error in errors))
    
    def test_create_benchmark_run(self):
        """Test creating a benchmark run."""
        run = self.engine._create_benchmark_run(self.test_config)
        
        self.assertIsInstance(run, BenchmarkRun)
        self.assertEqual(run.benchmark_id, "test_benchmark")
        self.assertEqual(run.status, BenchmarkStatus.CREATED)
        self.assertEqual(run.config, self.test_config)
        self.assertIsNotNone(run.start_time)
        self.assertIsNone(run.end_time)
    
    def test_run_benchmark_success(self):
        """Test successful benchmark execution."""
        # Mock dependencies
        self.config_manager.validate_config.return_value = (True, [])
        self.integration_manager.initialize.return_value = asyncio.Future()
        self.integration_manager.initialize.return_value.set_result(None)
        
        # Mock scenario execution
        with patch.object(self.engine, '_execute_scenario') as mock_execute:
            mock_execute.return_value = asyncio.Future()
            mock_execute.return_value.set_result({
                "scenario_id": "test_scenario",
                "status": "completed",
                "metrics": {"score": 0.85},
                "execution_time": 10.0
            })
            
            # Run benchmark
            result = asyncio.run(self.engine.run_benchmark(self.test_config))
            
            # Verify result
            self.assertIsInstance(result, BenchmarkResult)
            self.assertEqual(result.benchmark_id, "test_benchmark")
            self.assertEqual(result.status, BenchmarkStatus.COMPLETED)
            self.assertGreater(result.overall_score, 0.0)
            self.assertIsNotNone(result.start_time)
            self.assertIsNotNone(result.end_time)
    
    def test_run_benchmark_invalid_config(self):
        """Test benchmark execution with invalid configuration."""
        # Mock config manager validation with errors
        self.config_manager.validate_config.return_value = (False, ["Invalid config"])
        
        # Run benchmark and expect error
        with self.assertRaises(BenchmarkError):
            asyncio.run(self.engine.run_benchmark(self.test_config))
    
    def test_run_benchmark_already_running(self):
        """Test benchmark execution when already running."""
        # Create an active run
        run = self.engine._create_benchmark_run(self.test_config)
        run.status = BenchmarkStatus.RUNNING
        self.engine.active_runs["test_benchmark"] = run
        
        # Run benchmark and expect error
        with self.assertRaises(BenchmarkError):
            asyncio.run(self.engine.run_benchmark(self.test_config))
    
    def test_execute_scenario_success(self):
        """Test successful scenario execution."""
        scenario_config = self.test_config["scenarios"][0]
        agent_config = self.test_config["agents"][0]
        
        # Mock scenario and agent execution
        with patch.object(self.engine, '_load_scenario') as mock_load_scenario, \
             patch.object(self.engine, '_load_agent') as mock_load_agent, \
             patch.object(self.engine, '_collect_metrics') as mock_collect_metrics:
            
            # Mock scenario
            mock_scenario = Mock()
            mock_scenario.run.return_value = asyncio.Future()
            mock_scenario.run.return_value.set_result({
                "status": "completed",
                "metrics": {"score": 0.9},
                "events": [],
                "execution_time": 5.0
            })
            mock_load_scenario.return_value = mock_scenario
            
            # Mock agent
            mock_agent = Mock()
            mock_agent.execute.return_value = asyncio.Future()
            mock_agent.execute.return_value.set_result({
                "actions": ["test_action"],
                "execution_time": 2.0
            })
            mock_load_agent.return_value = mock_agent
            
            # Mock metrics
            mock_collect_metrics.return_value = {
                "cognitive": {"score": 0.85},
                "business": {"score": 0.9}
            }
            
            # Execute scenario
            result = asyncio.run(self.engine._execute_scenario(
                scenario_config, agent_config, self.test_config, 1
            ))
            
            # Verify result
            self.assertEqual(result["scenario_id"], "test_scenario")
            self.assertEqual(result["status"], "completed")
            self.assertIn("metrics", result)
            self.assertGreater(result["execution_time"], 0)
    
    def test_execute_scenario_failure(self):
        """Test scenario execution with failure."""
        scenario_config = self.test_config["scenarios"][0]
        agent_config = self.test_config["agents"][0]
        
        # Mock scenario execution with failure
        with patch.object(self.engine, '_load_scenario') as mock_load_scenario:
            mock_scenario = Mock()
            mock_scenario.run.return_value = asyncio.Future()
            mock_scenario.run.return_value.set_exception(Exception("Scenario failed"))
            mock_load_scenario.return_value = mock_scenario
            
            # Execute scenario and expect failure
            result = asyncio.run(self.engine._execute_scenario(
                scenario_config, agent_config, self.test_config, 1
            ))
            
            # Verify failure result
            self.assertEqual(result["scenario_id"], "test_scenario")
            self.assertEqual(result["status"], "failed")
            self.assertIn("error", result)
    
    def test_collect_metrics(self):
        """Test metrics collection."""
        # Mock metrics collection
        with patch.object(self.engine, '_calculate_cognitive_metrics') as mock_cognitive, \
             patch.object(self.engine, '_calculate_business_metrics') as mock_business, \
             patch.object(self.engine, '_calculate_technical_metrics') as mock_technical:
            
            mock_cognitive.return_value = {"reasoning": 0.8, "planning": 0.9}
            mock_business.return_value = {"roi": 0.85, "efficiency": 0.75}
            mock_technical.return_value = {"performance": 0.9, "reliability": 0.95}
            
            # Collect metrics
            metrics = asyncio.run(self.engine._collect_metrics(
                {"events": []}, {"agent_data": {}}, {"scenario_data": {}}
            ))
            
            # Verify metrics
            self.assertIn("cognitive", metrics)
            self.assertIn("business", metrics)
            self.assertIn("technical", metrics)
            self.assertEqual(metrics["cognitive"]["reasoning"], 0.8)
            self.assertEqual(metrics["business"]["roi"], 0.85)
            self.assertEqual(metrics["technical"]["performance"], 0.9)
    
    def test_aggregate_results(self):
        """Test result aggregation."""
        # Mock run results
        run_results = [
            {
                "scenario_id": "test_scenario",
                "agent_id": "test_agent",
                "run_number": 1,
                "metrics": {"score": 0.85},
                "execution_time": 10.0,
                "status": "completed"
            },
            {
                "scenario_id": "test_scenario",
                "agent_id": "test_agent",
                "run_number": 2,
                "metrics": {"score": 0.9},
                "execution_time": 12.0,
                "status": "completed"
            }
        ]
        
        # Aggregate results
        aggregated = self.engine._aggregate_results(run_results)
        
        # Verify aggregation
        self.assertIn("overall_score", aggregated)
        self.assertIn("average_execution_time", aggregated)
        self.assertIn("success_rate", aggregated)
        self.assertEqual(aggregated["overall_score"], 0.875)  # Average of 0.85 and 0.9
        self.assertEqual(aggregated["average_execution_time"], 11.0)  # Average of 10.0 and 12.0
        self.assertEqual(aggregated["success_rate"], 1.0)  # Both runs successful
    
    def test_get_benchmark_status(self):
        """Test getting benchmark status."""
        # Create a test run
        run = self.engine._create_benchmark_run(self.test_config)
        run.status = BenchmarkStatus.RUNNING
        self.engine.active_runs["test_benchmark"] = run
        
        # Get status
        status = self.engine.get_benchmark_status("test_benchmark")
        
        # Verify status
        self.assertEqual(status["benchmark_id"], "test_benchmark")
        self.assertEqual(status["status"], BenchmarkStatus.RUNNING)
        self.assertIsNotNone(status["start_time"])
    
    def test_get_benchmark_status_not_found(self):
        """Test getting benchmark status for non-existent benchmark."""
        status = self.engine.get_benchmark_status("non_existent")
        
        # Verify status
        self.assertEqual(status["benchmark_id"], "non_existent")
        self.assertEqual(status["status"], BenchmarkStatus.NOT_FOUND)
    
    def test_list_benchmarks(self):
        """Test listing benchmarks."""
        # Create test runs
        run1 = self.engine._create_benchmark_run(self.test_config)
        run1.status = BenchmarkStatus.COMPLETED
        self.engine.completed_runs.append(run1)
        
        run2 = self.engine._create_benchmark_run(self.test_config.copy())
        run2.benchmark_id = "test_benchmark_2"
        run2.status = BenchmarkStatus.RUNNING
        self.engine.active_runs["test_benchmark_2"] = run2
        
        # List benchmarks
        benchmarks = self.engine.list_benchmarks()
        
        # Verify list
        self.assertEqual(len(benchmarks), 2)
        self.assertTrue(any(b["benchmark_id"] == "test_benchmark" for b in benchmarks))
        self.assertTrue(any(b["benchmark_id"] == "test_benchmark_2" for b in benchmarks))
    
    def test_stop_benchmark(self):
        """Test stopping a benchmark."""
        # Create a test run
        run = self.engine._create_benchmark_run(self.test_config)
        run.status = BenchmarkStatus.RUNNING
        self.engine.active_runs["test_benchmark"] = run
        
        # Stop benchmark
        result = self.engine.stop_benchmark("test_benchmark")
        
        # Verify stop
        self.assertTrue(result)
        self.assertEqual(run.status, BenchmarkStatus.STOPPED)
        self.assertIsNotNone(run.end_time)
    
    def test_stop_benchmark_not_found(self):
        """Test stopping a non-existent benchmark."""
        result = self.engine.stop_benchmark("non_existent")
        
        # Verify stop
        self.assertFalse(result)
    
    def test_cleanup_completed_runs(self):
        """Test cleanup of completed runs."""
        # Create old completed run
        old_run = self.engine._create_benchmark_run(self.test_config)
        old_run.status = BenchmarkStatus.COMPLETED
        old_run.start_time = datetime.now().replace(year=2020)  # Old run
        self.engine.completed_runs.append(old_run)
        
        # Create recent completed run
        recent_run = self.engine._create_benchmark_run(self.test_config.copy())
        recent_run.benchmark_id = "recent_benchmark"
        recent_run.status = BenchmarkStatus.COMPLETED
        recent_run.start_time = datetime.now()
        self.engine.completed_runs.append(recent_run)
        
        # Cleanup runs older than 30 days
        self.engine.cleanup_completed_runs(max_age_days=30)
        
        # Verify cleanup
        self.assertEqual(len(self.engine.completed_runs), 1)
        self.assertEqual(self.engine.completed_runs[0].benchmark_id, "recent_benchmark")


class TestBenchmarkResult(unittest.TestCase):
    """Test cases for BenchmarkResult class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.result = BenchmarkResult(
            benchmark_id="test_benchmark",
            status=BenchmarkStatus.COMPLETED,
            overall_score=0.85,
            start_time=datetime.now(),
            end_time=datetime.now(),
            config={"test": "config"},
            results={"scenario_results": []},
            metadata={"test": "metadata"}
        )
    
    def test_init(self):
        """Test BenchmarkResult initialization."""
        self.assertEqual(self.result.benchmark_id, "test_benchmark")
        self.assertEqual(self.result.status, BenchmarkStatus.COMPLETED)
        self.assertEqual(self.result.overall_score, 0.85)
        self.assertIsNotNone(self.result.start_time)
        self.assertIsNotNone(self.result.end_time)
        self.assertEqual(self.result.config, {"test": "config"})
        self.assertEqual(self.result.results, {"scenario_results": []})
        self.assertEqual(self.result.metadata, {"test": "metadata"})
    
    def test_to_dict(self):
        """Test BenchmarkResult to_dict conversion."""
        result_dict = self.result.to_dict()
        
        self.assertEqual(result_dict["benchmark_id"], "test_benchmark")
        self.assertEqual(result_dict["status"], BenchmarkStatus.COMPLETED.value)
        self.assertEqual(result_dict["overall_score"], 0.85)
        self.assertIn("start_time", result_dict)
        self.assertIn("end_time", result_dict)
        self.assertEqual(result_dict["config"], {"test": "config"})
        self.assertEqual(result_dict["results"], {"scenario_results": []})
        self.assertEqual(result_dict["metadata"], {"test": "metadata"})


class TestBenchmarkRun(unittest.TestCase):
    """Test cases for BenchmarkRun class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.run = BenchmarkRun(
            benchmark_id="test_benchmark",
            config={"test": "config"}
        )
    
    def test_init(self):
        """Test BenchmarkRun initialization."""
        self.assertEqual(self.run.benchmark_id, "test_benchmark")
        self.assertEqual(self.run.config, {"test": "config"})
        self.assertEqual(self.run.status, BenchmarkStatus.CREATED)
        self.assertIsNotNone(self.run.start_time)
        self.assertIsNone(self.run.end_time)
        self.assertEqual(len(self.run.run_results), 0)
    
    def test_add_run_result(self):
        """Test adding run result."""
        result = {"test": "result"}
        self.run.add_run_result(result)
        
        self.assertEqual(len(self.run.run_results), 1)
        self.assertEqual(self.run.run_results[0], result)
    
    def test_to_dict(self):
        """Test BenchmarkRun to_dict conversion."""
        run_dict = self.run.to_dict()
        
        self.assertEqual(run_dict["benchmark_id"], "test_benchmark")
        self.assertEqual(run_dict["status"], BenchmarkStatus.CREATED.value)
        self.assertIn("start_time", run_dict)
        self.assertEqual(run_dict["config"], {"test": "config"})
        self.assertEqual(run_dict["run_results"], [])


if __name__ == "__main__":
    unittest.main()