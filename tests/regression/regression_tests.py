"""
Regression Testing Framework for FBA-Bench

Ensures existing functionality still works correctly and catches performance degradations
or unintended side effects introduced by new changes.
"""

import asyncio
import logging
import pytest
import time
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import sys
import os
from enum import Enum
from unittest.mock import Mock, patch, MagicMock
import hashlib

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reproducibility.golden_master import GoldenMaster, SnapshotComparison
from reproducibility.llm_cache import LLMResponseCache
from reproducibility.sim_seed import SimSeed
from infrastructure.performance_monitor import PerformanceMonitor
from agents.hierarchical_planner import StrategicPlanner, TacticalPlanner
from agents.skill_coordinator import SkillCoordinator
from memory_experiments.dual_memory_manager import DualMemoryManager
from event_bus import EventBus, get_event_bus
from events import BaseEvent, TickEvent, SaleOccurred
from services.world_store import WorldStore

logger = logging.getLogger(__name__)


class RegressionTestType(Enum):
    """Types of regression tests."""
    GOLDEN_MASTER_COMPARISON = "golden_master_comparison"
    PERFORMANCE_STABILITY = "performance_stability"
    FUNCTIONAL_INTEGRITY = "functional_integrity"
    DATA_CONSISTENCY = "data_consistency"
    CONFIGURATION_VALIDITY = "configuration_validity"
    API_CONTRACT_VALIDATION = "api_contract_validation"


@dataclass
class RegressionTestResult:
    """Results from regression testing."""
    test_name: str
    test_type: RegressionTestType
    success: bool
    regression_detected: bool
    details: Dict[str, Any]
    duration_seconds: float
    error_details: Optional[str] = None


@dataclass
class FunctionalBaseline:
    """Baseline for functional integrity tests."""
    name: str
    initial_state_hash: str
    final_output_hash: str
    expected_metrics: Dict[str, float]


@dataclass
class PerformanceBaseline:
    """Baseline for performance stability tests."""
    name: str
    average_response_time_ms: float
    max_memory_usage_mb: float
    max_cpu_usage_percent: float
    allowed_deviation_percent: float = 0.10  # 10% deviation allowed


class RegressionTestSuite:
    """
    Comprehensive regression testing suite for FBA-Bench.
    
    Ensures existing functionality still works and catches performance degradations
    or unintended side effects introduced by new changes.
    """
    
    def __init__(self):
        self.event_bus = get_event_bus()
        self.test_results: List[RegressionTestResult] = []
        self.temp_dir = None
        self.golden_master_directory = "tests/golden_masters"
        self.baseline_perf_data_file = os.path.join(self.golden_master_directory, "performance_baseline.json")
        self.baseline_func_data_file = os.path.join(self.golden_master_directory, "functional_baseline.json")
        
    async def setup_test_environment(self) -> Dict[str, Any]:
        """Setup test environment for regression testing."""
        logger.info("Setting up regression test environment")
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp(prefix="fba_bench_regression_")
        
        os.makedirs(self.golden_master_directory, exist_ok=True)
        
        environment = {
            "temp_dir": self.temp_dir,
            "agent_id": "regression_test_agent"
        }
        
        # Generate dummy golden master files if they don't exist for the first run
        await self._generate_dummy_baselines()
        
        return environment
    
    async def _generate_dummy_baselines(self):
        """Generates dummy baseline files for testing if not present."""
        if not os.path.exists(self.baseline_perf_data_file):
            dummy_perf_baseline = PerformanceBaseline(
                name="initial_baseline",
                average_response_time_ms=100.0,
                max_memory_usage_mb=500.0,
                max_cpu_usage_percent=50.0,
                allowed_deviation_percent=0.10
            )
            with open(self.baseline_perf_data_file, 'w') as f:
                json.dump(dummy_perf_baseline.__dict__, f)
            logger.info("Generated dummy performance baseline.")
        
        if not os.path.exists(self.baseline_func_data_file):
            dummy_func_baseline = FunctionalBaseline(
                name="initial_functional_baseline",
                initial_state_hash=hashlib.sha256(b"initial_state_data").hexdigest(),
                final_output_hash=hashlib.sha256(b"final_output_data").hexdigest(),
                expected_metrics={"total_sales": 1000.0, "profit": 500.0}
            )
            with open(self.baseline_func_data_file, 'w') as f:
                json.dump(dummy_func_baseline.__dict__, f)
            logger.info("Generated dummy functional baseline.")
            
    async def _run_mock_simulation(self, agent_id: str, duration_ticks: int) -> Dict[str, Any]:
        """Runs a mock simulation to generate a state for golden master."""
        event_bus = get_event_bus()
        memory_manager = DualMemoryManager(agent_id, Mock())
        strategic_planner = StrategicPlanner(agent_id, event_bus)
        skill_coordinator = SkillCoordinator(agent_id, event_bus)
        
        sim_state = {
            "tick_history": [],
            "agent_actions": [],
            "metrics": {"total_sales": 0, "profit": 0},
            "final_data_hash": ""
        }
        
        for tick in range(duration_ticks):
            # Simulate agent actions and market events
            market_event = TickEvent(event_id=f"tick_event_{tick}", timestamp=datetime.now(), tick=tick)
            await event_bus.publish(market_event)
            
            if tick % 10 == 0: # Simulate a sale
                sale_event = SaleOccurred(event_id=f"sale_event_{tick}", timestamp=datetime.now(), 
                                          asin="TEST-PRODUCT", quantity=10, unit_price=100, total_revenue=1000, fees=50)
                await event_bus.publish(sale_event)
                sim_state["metrics"]["total_sales"] += 1000
                sim_state["metrics"]["profit"] += 950
            
            # Simulate some internal agent logic resulting in actions
            plan_context = {"market_data": "some_data"}
            objectives = await strategic_planner.create_strategic_plan(plan_context, 10)
            if objectives:
                sim_state["agent_actions"].append({"tick": tick, "action_count": len(objectives)})

            sim_state["tick_history"].append(tick)

        # Generate a hash of the final state (simplified for mock)
        final_state_str = json.dumps(sim_state["metrics"], sort_keys=True) + \
                          json.dumps({"actions": len(sim_state["agent_actions"]), "ticks": len(sim_state["tick_history"])}, sort_keys=True)
        sim_state["final_data_hash"] = hashlib.sha256(final_state_str.encode()).hexdigest()

        return sim_state
        
    async def test_golden_master_regression(self) -> RegressionTestResult:
        """Compares current simulation output against a golden master baseline."""
        test_name = "golden_master_regression"
        start_time = time.time()
        
        try:
            logger.info("Running golden master regression test")
            environment = await self.setup_test_environment()
            agent_id = environment["agent_id"]
            
            # Define golden master file path
            golden_master_file = os.path.join(self.golden_master_directory, "simulation_golden_master.json")
            golden_master = GoldenMaster(golden_master_file)
            
            # Step 1: Generate or load baseline
            baseline_data = None
            if os.path.exists(golden_master_file):
                baseline_data = golden_master.load_golden_master()
                logger.info(f"Loaded existing golden master from {golden_master_file}")
            
            if not baseline_data:
                # If no golden master exists, create one from a mock run
                logger.warning("No golden master found. Creating a dummy baseline for this run.")
                simulation_result = await self._run_mock_simulation(agent_id, 50)
                golden_master.create_golden_master(simulation_result)
                baseline_data = simulation_result # For the purpose of this test, this will be our "baseline"
                logger.info("Dummy golden master created. Re-run test to compare against it.")
                
            # Step 2: Run current simulation and capture snapshot
            current_simulation_result = await self._run_mock_simulation(agent_id, 50)
            
            # Step 3: Compare current snapshot with golden master
            comparison_result = golden_master.compare_snapshots(
                baseline_data, 
                current_simulation_result
            )
            
            regression_detected = not comparison_result.match
            
            details = {
                "baseline_hash": comparison_result.baseline_hash,
                "current_hash": comparison_result.current_hash,
                "differences": comparison_result.differences
            }
            
            success = not regression_detected # If no regression, it's a success
            
            duration = time.time() - start_time
            
            return RegressionTestResult(
                test_name=test_name,
                test_type=RegressionTestType.GOLDEN_MASTER_COMPARISON,
                success=success,
                regression_detected=regression_detected,
                details=details,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Golden master regression test failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return RegressionTestResult(
                test_name=test_name,
                test_type=RegressionTestType.GOLDEN_MASTER_COMPARISON,
                success=False,
                regression_detected=True, # Indicate failure
                details={},
                duration_seconds=duration,
                error_details=str(e)
            )

    async def test_performance_stability_regression(self) -> RegressionTestResult:
        """Monitors key performance indicators for unacceptable regressions."""
        test_name = "performance_stability_regression"
        start_time = time.time()
        
        try:
            logger.info("Running performance stability regression test")
            environment = await self.setup_test_environment()
            agent_id = environment["agent_id"]
            
            performance_monitor = PerformanceMonitor()
            
            # Load performance baseline
            baseline_perf = None
            if os.path.exists(self.baseline_perf_data_file):
                with open(self.baseline_perf_data_file, 'r') as f:
                    baseline_data = json.load(f)
                    baseline_perf = PerformanceBaseline(**baseline_data)
                logger.info(f"Loaded performance baseline from {self.baseline_perf_data_file}")
            else:
                logger.warning("No performance baseline found. Creating a dummy for this run.")
                baseline_perf = PerformanceBaseline(
                    name="default_baseline",
                    average_response_time_ms=100.0,
                    max_memory_usage_mb=500.0,
                    max_cpu_usage_percent=50.0,
                    allowed_deviation_percent=0.10
                )
                with open(self.baseline_perf_data_file, 'w') as f:
                    json.dump(baseline_perf.__dict__, f)
                logger.info("Dummy performance baseline created. Re-run test to compare against it.")

            # Run a simulated workload and collect current performance metrics
            await performance_monitor.start()
            for i in range(100):
                # Simulate a complex operation
                await asyncio.sleep(0.01 + (i % 10) * 0.001) # Simulate varying response times
                memory_usage_mb = 100 + (i % 400) # Simulate memory fluctuations
                cpu_usage_percent = 20 + (i % 30) # Simulate CPU usage
                
                await performance_monitor.record_event({
                    "event_type": "complex_operation",
                    "duration_ms": 10 + (i % 50),
                    "success": True
                })
                await performance_monitor.record_system_metrics(
                    cpu_percent=cpu_usage_percent,
                    memory_percent=(memory_usage_mb / 2048) * 100, # Assuming 2GB total memory for percent
                    disk_io_mbps=10
                )
            
            await performance_monitor.stop()
            current_metrics = await performance_monitor.get_metrics()
            
            # Extract relevant current performance indicators
            current_avg_response_time = current_metrics["average_response_time_ms"]
            current_max_memory_usage = current_metrics["max_memory_usage_mb"]
            current_max_cpu_usage = current_metrics["max_cpu_percent"]
            
            regression_detected = False
            details = {
                "current_avg_response_time_ms": current_avg_response_time,
                "baseline_avg_response_time_ms": baseline_perf.average_response_time_ms,
                "current_max_memory_usage_mb": current_max_memory_usage,
                "baseline_max_memory_usage_mb": baseline_perf.max_memory_usage_mb,
                "current_max_cpu_usage_percent": current_max_cpu_usage,
                "baseline_max_cpu_usage_percent": baseline_perf.max_cpu_usage_percent
            }
            
            # Check for significant deviations
            if abs(current_avg_response_time - baseline_perf.average_response_time_ms) / baseline_perf.average_response_time_ms > baseline_perf.allowed_deviation_percent:
                regression_detected = True
                details["response_time_regression"] = True
            
            if abs(current_max_memory_usage - baseline_perf.max_memory_usage_mb) / baseline_perf.max_memory_usage_mb > baseline_perf.allowed_deviation_percent:
                regression_detected = True
                details["memory_usage_regression"] = True
                
            if abs(current_max_cpu_usage - baseline_perf.max_cpu_usage_percent) / baseline_perf.max_cpu_usage_percent > baseline_perf.allowed_deviation_percent:
                regression_detected = True
                details["cpu_usage_regression"] = True

            success = not regression_detected
            duration = time.time() - start_time
            
            return RegressionTestResult(
                test_name=test_name,
                test_type=RegressionTestType.PERFORMANCE_STABILITY,
                success=success,
                regression_detected=regression_detected,
                details=details,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Performance stability regression test failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return RegressionTestResult(
                test_name=test_name,
                test_type=RegressionTestType.PERFORMANCE_STABILITY,
                success=False,
                regression_detected=True,
                details={},
                duration_seconds=duration,
                error_details=str(e)
            )

    async def test_functional_integrity_regression(self) -> RegressionTestResult:
        """Verifies core functions and critical workflows remain intact."""
        test_name = "functional_integrity_regression"
        start_time = time.time()
        
        try:
            logger.info("Running functional integrity regression test")
            environment = await self.setup_test_environment()
            agent_id = environment["agent_id"]
            
            # Load functional baseline
            baseline_func = None
            if os.path.exists(self.baseline_func_data_file):
                with open(self.baseline_func_data_file, 'r') as f:
                    baseline_data = json.load(f)
                    baseline_func = FunctionalBaseline(**baseline_data)
                logger.info(f"Loaded functional baseline from {self.baseline_func_data_file}")
            else:
                logger.warning("No functional baseline found. Skipping this test.")
                # This should ideally not happen if _generate_dummy_baselines is called
                return RegressionTestResult(
                    test_name=test_name,
                    test_type=RegressionTestType.FUNCTIONAL_INTEGRITY,
                    success=False, # Mark as failed because baseline is missing
                    regression_detected=False,
                    details={"message": "Functional baseline not found. Cannot perform comparison."},
                    duration_seconds=0,
                    error_details="Functional baseline missing."
                )

            # Re-run critical functional flows
            event_bus = get_event_bus()
            world_store = WorldStore() # Assume WorldStore is updated by events

            # Simulate initial state (e.g., set up initial products)
            await world_store.initialize_world_state({"products": [{"id": "P001", "name": "Product A", "price": 100}]})

            # Calculate initial state hash
            initial_state_str = json.dumps(world_store.get_current_state(), sort_keys=True)
            current_initial_state_hash = hashlib.sha256(initial_state_str.encode()).hexdigest()

            # Simulate core workflow (e.g., agent setting price, sales occurring)
            # Create agent with basic skills
            skill_coordinator = SkillCoordinator(agent_id, event_bus)
            
            # Register a dummy skill to simulate interaction
            class MockPricingSkill:
                async def handle_event(self, event: BaseEvent):
                    if isinstance(event, TickEvent) and event.tick % 5 == 0:
                        # Simulate agent setting a price
                        set_price_command = SetPriceCommand(
                            event_id=f"set_price_{event.tick}",
                            timestamp=datetime.now(),
                            agent_id=agent_id,
                            asin="P001",
                            new_price=100 + (event.tick % 10) # Dynamic pricing
                        )
                        await event_bus.publish(set_price_command)
            
            mock_skill = MockPricingSkill()
            await skill_coordinator.register_skill(mock_skill, [TickEvent])

            # Simulate some ticks
            for tick in range(20):
                await event_bus.publish(TickEvent(event_id=f"tick_{tick}", timestamp=datetime.now(), tick=tick))
                # Simulate sales events
                if tick % 3 == 0:
                    await event_bus.publish(SaleOccurred(
                        event_id=f"sale_{tick}",
                        timestamp=datetime.now(),
                        asin="P001",
                        quantity=max(1, (tick % 5)),
                        unit_price=100,
                        total_revenue=100 * (tick % 5),
                        fees=5 * (tick % 5)
                    ))
                await asyncio.sleep(0.001) # Small delay to allow async operations

            # Capture final output/state hash after workflow
            final_state_after_workflow = world_store.get_current_state()
            current_final_output_str = json.dumps(final_state_after_workflow, sort_keys=True) + \
                                       json.dumps(skill_coordinator.get_skill_performance_metrics(), sort_keys=True)
            current_final_output_hash = hashlib.sha256(current_final_output_str.encode()).hexdigest()
            
            # Capture critical metrics
            current_metrics = {
                "total_sales": final_state_after_workflow.get("metrics", {}).get("total_sales", 0),
                "profit": final_state_after_workflow.get("metrics", {}).get("profit", 0)
            }

            regression_detected = False
            details = {
                "initial_state_hash_match": current_initial_state_hash == baseline_func.initial_state_hash,
                "final_output_hash_match": current_final_output_hash == baseline_func.final_output_hash,
                "metric_deviations": {}
            }

            # Check if any core functional regression
            if not details["initial_state_hash_match"]:
                regression_detected = True
                details["initial_state_regression"] = True
            if not details["final_output_hash_match"]:
                regression_detected = True
                details["final_output_regression"] = True

            # Check for metric regressions (e.g., +/- 5% deviation)
            for metric, baseline_value in baseline_func.expected_metrics.items():
                current_value = current_metrics.get(metric, 0)
                if baseline_value != 0 and abs(current_value - baseline_value) / baseline_value > 0.05:
                    regression_detected = True
                    details["metric_deviations"][metric] = {
                        "baseline": baseline_value,
                        "current": current_value,
                        "deviation_percent": (abs(current_value - baseline_value) / baseline_value) * 100
                    }
            
            success = not regression_detected
            duration = time.time() - start_time
            
            return RegressionTestResult(
                test_name=test_name,
                test_type=RegressionTestType.FUNCTIONAL_INTEGRITY,
                success=success,
                regression_detected=regression_detected,
                details=details,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Functional integrity regression test failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return RegressionTestResult(
                test_name=test_name,
                test_type=RegressionTestType.FUNCTIONAL_INTEGRITY,
                success=False,
                regression_detected=True,
                details={},
                duration_seconds=duration,
                error_details=str(e)
            )

    async def test_data_consistency_regression(self) -> RegressionTestResult:
        """Ensures data integrity and consistency across data stores."""
        test_name = "data_consistency_regression"
        start_time = time.time()
        
        try:
            logger.info("Running data consistency regression test")
            environment = await self.setup_test_environment()
            agent_id = environment["agent_id"]

            success = True
            regression_detected = False
            details = {}

            # Test 1: Memory Manager Consistency
            logger.info("Testing DualMemoryManager consistency")
            mem_config = type("MockMemoryConfig", (), {"short_term_capacity": 100, "long_term_capacity": 500})() # Mock config
            memory_manager = DualMemoryManager(agent_id, mem_config)

            # Add test memories
            for i in range(50):
                await memory_manager.store_memory(
                    event_id=f"mem_event_{i}",
                    event_type="TestEvent",
                    content=f"Memory content {i}",
                    domain="test",
                    importance_score=0.5,
                    timestamp=datetime.now()
                )
            
            # Query memories and ensure expected count
            retrieved_memories = await memory_manager.query_memories("Memory content", limit=100)
            if len(retrieved_memories) != 50:
                regression_detected = True
                details["memory_store_count_mismatch"] = True
                success = False

            # Test 2: LLM Cache Integrity
            logger.info("Testing LLM Cache integrity")
            cache_file = os.path.join(self.temp_dir, "test_llm_cache.db")
            llm_cache = LLMResponseCache(cache_file=cache_file)
            
            # Add some dummy responses
            llm_cache.cache_response("hash1", {"response": "foo"}, {"model": "test", "temp": 0})
            llm_cache.cache_response("hash2", {"response": "bar"}, {"model": "test", "temp": 0})

            is_valid, errors = llm_cache.validate_cache_integrity()
            if not is_valid or errors:
                regression_detected = True
                details["llm_cache_integrity_issue"] = errors if errors else "Cache invalid but no specific errors reported."
                success = False

            # Test 3: WorldStore Data Integrity (simplified, assumes internal consistency)
            logger.info("Testing WorldStore data integrity")
            world_store = WorldStore()
            initial_state = world_store.get_current_state()
            
            # Simulate an update
            await world_store.update_world_state({"products": [{"id": "P001", "name": "Updated Product", "price": 105}]})
            updated_state = world_store.get_current_state()
            
            if "products" not in updated_state or updated_state["products"][0]["price"] != 105:
                regression_detected = True
                details["world_store_update_failure"] = True
                success = False

            details["data_consistency_checks"] = {
                "memory_manager_consistent": success, # Reflects success of test 1
                "llm_cache_consistent": is_valid and not errors,
                "world_store_consistent_update": "world_store_update_failure" not in details
            }
            
            duration = time.time() - start_time
            
            return RegressionTestResult(
                test_name=test_name,
                test_type=RegressionTestType.DATA_CONSISTENCY,
                success=success,
                regression_detected=regression_detected,
                details=details,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Data consistency regression test failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return RegressionTestResult(
                test_name=test_name,
                test_type=RegressionTestType.DATA_CONSISTENCY,
                success=False,
                regression_detected=True,
                details={},
                duration_seconds=duration,
                error_details=str(e)
            )

    async def test_configuration_validity_regression(self) -> RegressionTestResult:
        """Tests that system configurations remain valid and loadable."""
        test_name = "configuration_validity_regression"
        start_time = time.time()
        
        try:
            logger.info("Running configuration validity regression test")
            
            # Define common configuration files to check
            config_paths = [
                "agents/cognitive_config.py",
                "agents/skill_config.py",
                "memory_experiments/memory_config.py",
                "reproducibility/reproducibility_config.py",
                "infrastructure/scalability_config.py",
                "observability/observability_config.py",
                # Add paths to actual config files (e.g., .yaml, .json) if they contain validation logic
                "baseline_bots/configs/gpt_4o_mini_config.yaml"
            ]
            
            success = True
            regression_detected = False
            details = {}

            for config_path in config_paths:
                file_path = Path(os.path.join(Path(__file__).parent.parent.parent, config_path))
                config_name = file_path.stem

                if not file_path.exists():
                    logger.warning(f"Config file not found, skipping: {config_path}")
                    details[f"{config_name}_found"] = False
                    continue
                
                try:
                    # Attempt to load and validate the config
                    # For Python files, we can attempt to import and instantiate relevant config classes
                    if file_path.suffix == '.py':
                        module_name = file_path.stem
                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            if module_name == "cognitive_config":
                                CognitiveConfig = getattr(module, "CognitiveConfig")
                                config_instance = CognitiveConfig(enable_reflection=True, reflection_interval_hours=24, enable_hierarchical_planning=True, strategic_planning_horizon_days=90)
                                details[f"{config_name}_load_success"] = True
                            elif module_name == "skill_config":
                                SkillConfig = getattr(module, "SkillConfig")
                                config_instance = SkillConfig(enable_supply_management=True, enable_marketing=True, enable_customer_service=True, enable_financial_analysis=True, coordination_strategy="priority_based", max_concurrent_skills=5)
                                details[f"{config_name}_load_success"] = True
                            elif module_name == "memory_config":
                                MemoryConfig = getattr(module, "MemoryConfig")
                                from memory_experiments.memory_config import ConsolidationAlgorithm
                                config_instance = MemoryConfig(consolidation_algorithm=ConsolidationAlgorithm.LLM_REFLECTION)
                                details[f"{config_name}_load_success"] = True
                            elif module_name == "reproducibility_config":
                                ReproducibilityConfig = getattr(module, "ReproducibilityConfig")
                                config_instance = ReproducibilityConfig(enable_llm_cache=True, enable_deterministic_sim=True)
                                details[f"{config_name}_load_success"] = True
                            elif module_name == "scalability_config":
                                ScalabilityConfig = getattr(module, "ScalabilityConfig")
                                config_instance = ScalabilityConfig(enable_batching=True, max_batch_size=10, enable_distributed_simulation=False)
                                details[f"{config_name}_load_success"] = True
                            elif module_name == "observability_config":
                                ObservabilityConfig = getattr(module, "ObservabilityConfig")
                                config_instance = ObservabilityConfig(enable_tracing=True, enable_alerts=True)
                                details[f"{config_name}_load_success"] = True
                            else:
                                logger.warning(f"No specific validation logic for {config_path}, only import test.")
                                details[f"{config_name}_load_success"] = True # Successful import is a basic pass
                        else:
                            raise ImportError("Could not load module spec.")
                    elif file_path.suffix == '.yaml':
                        import yaml # Ensure yaml is imported.
                        with open(file_path, 'r') as f:
                            yaml.safe_load(f) # Just loading as valid YAML is a first step
                            details[f"{config_name}_load_success"] = True
                    elif file_path.suffix == '.json':
                        with open(file_path, 'r') as f:
                            json.load(f) # Just loading as valid JSON is a first step
                            details[f"{config_name}_load_success"] = True
                    
                    else:
                        details[f"{config_name}_load_success"] = True # No specific validation, assume success if readable
                except Exception as e:
                    logger.error(f"Configuration validation failed for {config_path}: {e}")
                    details[f"{config_name}_load_success"] = False
                    regression_detected = True
                    success = False
            
            duration = time.time() - start_time
            
            return RegressionTestResult(
                test_name=test_name,
                test_type=RegressionTestType.CONFIGURATION_VALIDITY,
                success=success,
                regression_detected=regression_detected,
                details=details,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Configuration validity regression test failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return RegressionTestResult(
                test_name=test_name,
                test_type=RegressionTestType.CONFIGURATION_VALIDITY,
                success=False,
                regression_detected=True,
                details={},
                duration_seconds=duration,
                error_details=str(e)
            )

    async def test_api_contract_validation_regression(self) -> RegressionTestResult:
        """Verifies that internal and external API contracts are maintained."""
        test_name = "api_contract_validation_regression"
        start_time = time.time()
        
        try:
            logger.info("Running API contract validation regression test")
            
            success = True
            regression_detected = False
            details = {}

            # Test 1: LLM Interface Contract
            logger.info("Testing LLM interface contract")
            from llm_interface.contract import LLMContractValidator
            from llm_interface.openrouter_client import OpenRouterLLMClient
            from llm_interface.deterministic_client import DeterministicLLMClient
            
            # Test concrete implementations against contract
            try:
                # OpenRouter client (needs dummy args for instantiation)
                openrouter_client = OpenRouterLLMClient(api_key="sk-dummy", model="mistralai/mistral-7b-instruct") 
                contract_issues_openrouter = LLMContractValidator.validate(openrouter_client)
                details["openrouter_client_contract_valid"] = not contract_issues_openrouter
                if contract_issues_openrouter: regression_detected = True; success = False; details["openrouter_issues"] = contract_issues_openrouter
            except Exception as e:
                details["openrouter_client_contract_valid"] = False
                regression_detected = True
                success = False
                details["openrouter_error"] = str(e)
            
            try:
                # Deterministic client
                deterministic_client = DeterministicLLMClient(responses={"test_prompt": "test_response"})
                contract_issues_deterministic = LLMContractValidator.validate(deterministic_client)
                details["deterministic_client_contract_valid"] = not contract_issues_deterministic
                if contract_issues_deterministic: regression_detected = True; success = False; details["deterministic_issues"] = contract_issues_deterministic
            except Exception as e:
                details["deterministic_client_contract_valid"] = False
                regression_detected = True
                success = False
                details["deterministic_error"] = str(e)

            # Test 2: Event Bus Contract (publish/subscribe behavior)
            logger.info("Testing Event Bus contract")
            event_bus = get_event_bus()

            mock_listener_received_events = []
            async def mock_listener(event: BaseEvent):
                mock_listener_received_events.append(event)
            
            event_bus.subscribe(TickEvent, mock_listener)
            
            test_event = TickEvent(event_id="contract_test_tick", timestamp=datetime.now(), tick=1)
            await event_bus.publish(test_event)

            if not mock_listener_received_events or mock_listener_received_events[0].event_id != test_event.event_id:
                regression_detected = True
                details["event_bus_contract_violation"] = "Event not received or mismatched."
                success = False
            details["event_bus_contract_valid"] = not ("event_bus_contract_violation" in details)
            
            # Test 3: Plugin Interface Contracts (dummy check for existence of key methods)
            logger.info("Testing Plugin Interface contracts")
            from plugins.plugin_interface import IPlugin, ISkillPlugin, IAnalysisPlugin, IIntegrationPlugin

            # Define a mock plugin class for structural validation
            class MockValidSkillPlugin(ISkillPlugin):
                name = "MockValidSkillPlugin"
                version = "1.0.0"
                
                def __init__(self, agent_id: str):
                    self._agent_id = agent_id
                async def initialize(self, context: dict) -> bool: return True
                async def activate(self) -> bool: return True
                async def deactivate(self) -> bool: return True
                async def cleanup(self) -> bool: return True
                def get_metadata(self) -> dict: return {"name": self.name}
                async def handle_event(self, event: BaseEvent) -> Optional[List[Any]]: return []
                def get_supported_event_types(self) -> List[type]: return [TickEvent]
            
            class MockInvalidSkillPlugin: # Missing methods
                name = "MockInvalidSkillPlugin"
                version = "1.0.0"
                # Missing all ISkillPlugin methods

            plugin_contract_checks = {}
            for plugin_cls in [MockValidSkillPlugin, MockInvalidSkillPlugin]:
                has_all_methods = True
                required_methods = ["initialize", "activate", "deactivate", "cleanup", "get_metadata"]
                if issubclass(plugin_cls, ISkillPlugin): # specific skill plugin methods
                    required_methods.extend(["handle_event", "get_supported_event_types"])

                for method_name in required_methods:
                    if not hasattr(plugin_cls, method_name) or not callable(getattr(plugin_cls, method_name)):
                        has_all_methods = False
                        break
                
                plugin_contract_checks[plugin_cls.name] = has_all_methods
                if not has_all_methods and plugin_cls == MockInvalidSkillPlugin:
                    details["plugin_contract_violation_expected"] = True # This indicates the check works
                elif not has_all_methods:
                    regression_detected = True
                    success = False
                    details[f"{plugin_cls.name}_contract_violation"] = "Missing required methods."

            details["plugin_interface_contracts_checked"] = plugin_contract_checks
            
            duration = time.time() - start_time
            
            return RegressionTestResult(
                test_name=test_name,
                test_type=RegressionTestType.API_CONTRACT_VALIDATION,
                success=success,
                regression_detected=regression_detected,
                details=details,
                duration_seconds=duration
            )
            
        except ImportError as ie: # Catch specific import errors for modules under test
            logger.error(f"API contract validation test failed due to missing module: {ie}", exc_info=True)
            duration = time.time() - start_time
            
            return RegressionTestResult(
                test_name=test_name,
                test_type=RegressionTestType.API_CONTRACT_VALIDATION,
                success=False,
                regression_detected=True,
                details={"message": f"Required module for API contract validation not found: {ie}"},
                duration_seconds=duration,
                error_details=str(ie)
            )
        except Exception as e:
            logger.error(f"API contract validation regression test failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return RegressionTestResult(
                test_name=test_name,
                test_type=RegressionTestType.API_CONTRACT_VALIDATION,
                success=False,
                regression_detected=True,
                details={},
                duration_seconds=duration,
                error_details=str(e)
            )


    async def run_regression_test_suite(self) -> Dict[str, Any]:
        """Run complete regression testing suite."""
        logger.info("Starting comprehensive regression testing suite")
        suite_start = time.time()
        
        # Regression test methods to run
        test_methods = [
            self.test_golden_master_regression,
            self.test_performance_stability_regression,
            self.test_functional_integrity_regression,
            self.test_data_consistency_regression,
            self.test_configuration_validity_regression,
            self.test_api_contract_validation_regression
        ]
        
        results = []
        
        for test_method in test_methods:
            try:
                logger.info(f"Running {test_method.__name__}")
                result = await test_method()
                results.append(result)
                self.test_results.append(result)
                
                if result.success:
                    logger.info(f"✅ {result.test_name} passed")
                else:
                    logger.error(f"❌ {result.test_name} failed: {result.error_details if result.error_details else 'Regression detected'}")
                    
            except Exception as e:
                logger.error(f"Execution of test method {test_method.__name__} crashed: {e}", exc_info=True)
                results.append(RegressionTestResult(
                    test_name=test_method.__name__,
                    test_type=RegressionTestType.FUNCTIONAL_INTEGRITY, # Default type if crash
                    success=False,
                    regression_detected=True,
                    details={},
                    duration_seconds=0,
                    error_details=f"Test runner crashed: {str(e)}"
                ))
        
        suite_duration = time.time() - suite_start
        
        # Compile summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests
        
        total_regressions_detected = sum(1 for r in results if r.regression_detected)
        
        summary = {
            "suite_duration_seconds": suite_duration,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_regressions_detected": total_regressions_detected,
            "regression_free": total_regressions_detected == 0,
            "test_results": [result.__dict__ for result in results]
        }
        
        logger.info(f"Regression testing suite completed: {passed_tests}/{total_tests} tests passed.")
        if summary["regression_free"]:
            logger.info("🎉 No regressions detected!")
        else:
            logger.warning(f"⚠️ {total_regressions_detected} regressions detected.")

        return summary
    
    async def cleanup_test_environment(self):
        """Clean up test resources."""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info("Regression test environment cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# CLI runner for direct execution
async def main():
    """Run regression testing suite."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_suite = RegressionTestSuite()
    
    try:
        results = await test_suite.run_regression_test_suite()
        
        print("\n" + "="*80)
        print("REGRESSION TESTING RESULTS")
        print("="*80)
        print(f"Total Tests Run: {results['total_tests']}")
        print(f"Tests Passed: {results['passed_tests']}")
        print(f"Tests Failed: {results['failed_tests']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Total Regressions Detected: {results['total_regressions_detected']}")
        print(f"Suite Duration: {results['suite_duration_seconds']:.2f}s")
        
        if results['regression_free']:
            print("\n🎉 REGRESSION-FREE!")
            print("Existing functionality confirmed stable.")
        else:
            print("\n⚠️  REGRESSIONS DETECTED.")
            print("Review test results for details and address identified issues.")
        
        print("="*80)
        
    finally:
        await test_suite.cleanup_test_environment()


if __name__ == "__main__":
    asyncio.run(main())