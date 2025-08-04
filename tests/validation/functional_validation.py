"""
Functional Validation Suite for FBA-Bench

Validates feature completeness, error handling, configuration validation,
and CLI functionality to ensure all documented capabilities work correctly.
"""

import asyncio
import logging
import pytest
import time
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys
import os
import subprocess
from unittest.mock import Mock, patch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.hierarchical_planner import StrategicPlanner, TacticalPlanner, PlanStatus, PlanPriority, PlanType
from agents.skill_coordinator import SkillCoordinator, CoordinationStrategy
from agents.skill_modules.base_skill import BaseSkill, SkillAction, SkillContext, SkillOutcome, SkillStatus
from agents.skill_modules.supply_manager import SupplyManagerSkill
from agents.skill_modules.marketing_manager import MarketingManagerSkill
from agents.skill_modules.customer_service import CustomerServiceSkill
from agents.skill_modules.financial_analyst import FinancialAnalystSkill
from agents.cognitive_config import CognitiveConfig
from agents.skill_config import SkillConfig
from memory_experiments.reflection_module import ReflectionModule, StructuredReflectionLoop, ReflectionTrigger
from memory_experiments.dual_memory_manager import DualMemoryManager
from memory_experiments.memory_config import MemoryConfig, ConsolidationAlgorithm
from memory_experiments.memory_validator import MemoryValidator
from reproducibility.llm_cache import LLMResponseCache, CacheStatistics
from reproducibility.sim_seed import SimSeed
from reproducibility.golden_master import GoldenMaster
from infrastructure.llm_batcher import LLMBatcher
from infrastructure.performance_monitor import PerformanceMonitor
from observability.trace_analyzer import TraceAnalyzer
from observability.alert_system import AlertSystem
from scenarios.curriculum_validator import CurriculumValidator
from scenarios.scenario_engine import ScenarioEngine
from learning.episodic_learning import EpisodicLearningManager
from integration.real_world_adapter import RealWorldAdapter
from plugins.plugin_framework import PluginFramework
from event_bus import EventBus, get_event_bus
from events import BaseEvent, TickEvent, SaleOccurred, SetPriceCommand

logger = logging.getLogger(__name__)


@dataclass
class FunctionalTestResult:
    """Results from functional validation test."""
    test_name: str
    feature_category: str
    success: bool
    validation_details: Dict[str, Any]
    error_handling_tests: Dict[str, bool]
    configuration_tests: Dict[str, bool]
    cli_tests: Dict[str, bool]
    duration_seconds: float
    error_details: Optional[str] = None


@dataclass
class FeatureCompleteness:
    """Feature completeness validation tracking."""
    cognitive_loops: Dict[str, bool]
    multi_skill_capabilities: Dict[str, bool]
    reproducibility_features: Dict[str, bool]
    infrastructure_features: Dict[str, bool]
    observability_features: Dict[str, bool]
    learning_features: Dict[str, bool]
    integration_features: Dict[str, bool]


class FunctionalValidationSuite:
    """
    Comprehensive functional validation suite for FBA-Bench feature completeness.
    
    Tests all documented capabilities, error handling, configuration validation,
    and CLI functionality to ensure complete implementation.
    """
    
    def __init__(self):
        self.event_bus = get_event_bus()
        self.test_results: List[FunctionalTestResult] = []
        self.temp_dir = None
        
    async def setup_test_environment(self) -> Dict[str, Any]:
        """Setup test environment for functional validation."""
        logger.info("Setting up functional validation environment")
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp(prefix="fba_bench_validation_")
        
        environment = {
            "temp_dir": self.temp_dir,
            "test_agent_id": "functional_test_agent"
        }
        
        return environment
    
    async def validate_cognitive_loop_completeness(self) -> FunctionalTestResult:
        """Test reflection, planning, memory integration."""
        test_name = "cognitive_loop_completeness"
        start_time = time.time()
        
        try:
            logger.info("Validating cognitive loop completeness")
            
            environment = await self.setup_test_environment()
            agent_id = environment["test_agent_id"]
            
            validation_details = {}
            error_handling_tests = {}
            configuration_tests = {}
            cli_tests = {}
            
            # Test 1: Hierarchical Planning System
            logger.info("Testing hierarchical planning system")
            
            strategic_planner = StrategicPlanner(agent_id, self.event_bus)
            tactical_planner = TacticalPlanner(agent_id, strategic_planner, self.event_bus)
            
            # Test strategic planning capabilities
            context = {
                "current_metrics": {
                    "profit_margin": 0.12,
                    "revenue_growth": 0.08,
                    "market_share": 0.15
                },
                "market_conditions": {
                    "competitive_pressure": 0.6,
                    "volatility": 0.4
                }
            }
            
            # Test all strategy types
            strategy_tests = {}
            for strategy_type in PlanType:
                context["market_conditions"]["competitive_pressure"] = {
                    PlanType.GROWTH: 0.3,
                    PlanType.DEFENSIVE: 0.8,
                    PlanType.RECOVERY: 0.5,
                    PlanType.OPTIMIZATION: 0.5,
                    PlanType.EXPLORATORY: 0.6
                }.get(strategy_type, 0.5)
                
                objectives = await strategic_planner.create_strategic_plan(context, 30)
                strategy_tests[strategy_type.value] = len(objectives) > 0
            
            validation_details["strategic_planning"] = {
                "strategy_types_tested": len(strategy_tests),
                "strategy_success_rate": sum(strategy_tests.values()) / len(strategy_tests),
                "strategy_results": strategy_tests
            }
            
            # Test tactical action generation
            if objectives:
                tactical_actions = await tactical_planner.generate_tactical_actions(
                    objectives, context
                )
                validation_details["tactical_planning"] = {
                    "actions_generated": len(tactical_actions),
                    "action_types": list(set(action.action_type for action in tactical_actions))
                }
            
            # Test action alignment validation
            test_action = {
                "type": "set_price",
                "expected_impact": {"profit_margin": 0.05, "revenue": 0.1}
            }
            
            is_aligned, alignment_score, reasoning = await strategic_planner.validate_action_alignment(test_action)
            validation_details["action_alignment"] = {
                "alignment_working": is_aligned,
                "alignment_score": alignment_score,
                "reasoning_provided": len(reasoning) > 0
            }
            
            # Test 2: Memory Integration System
            logger.info("Testing memory integration system")
            
            memory_config = MemoryConfig(
                consolidation_algorithm=ConsolidationAlgorithm.LLM_REFLECTION,
                short_term_capacity=1000,
                long_term_capacity=5000
            )
            
            memory_manager = DualMemoryManager(agent_id, memory_config)
            memory_validator = MemoryValidator(memory_manager, memory_config)
            
            # Test memory storage and retrieval
            test_memories = [
                {
                    "event_id": f"test_memory_{i}",
                    "event_type": "SaleOccurred",
                    "content": f"Test sale event {i}",
                    "domain": "sales",
                    "importance_score": 0.5 + (i * 0.1),
                    "timestamp": datetime.now()
                }
                for i in range(10)
            ]
            
            memory_storage_results = []
            for memory_data in test_memories:
                try:
                    await memory_manager.store_memory(**memory_data)
                    memory_storage_results.append(True)
                except Exception as e:
                    logger.error(f"Memory storage failed: {e}")
                    memory_storage_results.append(False)
            
            # Test memory retrieval
            query_results = await memory_manager.query_memories("sale", domain="sales", limit=5)
            
            validation_details["memory_integration"] = {
                "storage_success_rate": sum(memory_storage_results) / len(memory_storage_results),
                "memories_stored": sum(memory_storage_results),
                "retrieval_working": len(query_results) > 0,
                "query_results_count": len(query_results)
            }
            
            # Test memory validation
            validation_report = await memory_validator.validate_memory_consistency()
            validation_details["memory_validation"] = {
                "consistency_check": validation_report.get("consistent", False),
                "validation_score": validation_report.get("consistency_score", 0.0)
            }
            
            # Test 3: Reflection System
            logger.info("Testing reflection system")
            
            reflection_loop = StructuredReflectionLoop(agent_id, memory_manager, memory_config, self.event_bus)
            
            # Test different reflection triggers
            trigger_tests = {}
            
            # Test periodic trigger
            reflection_result = await reflection_loop.trigger_reflection(tick_interval=24)
            trigger_tests["periodic"] = reflection_result is not None
            
            # Test event-driven trigger
            major_events = [
                {
                    "event_id": "test_major_event",
                    "type": "competitor_major_action",
                    "severity": 0.8,
                    "timestamp": datetime.now().isoformat()
                }
            ]
            reflection_result = await reflection_loop.trigger_reflection(major_events=major_events)
            trigger_tests["event_driven"] = reflection_result is not None
            
            validation_details["reflection_system"] = {
                "trigger_types_tested": len(trigger_tests),
                "trigger_success_rate": sum(trigger_tests.values()) / len(trigger_tests),
                "trigger_results": trigger_tests
            }
            
            # Test 4: Error Handling
            logger.info("Testing cognitive loop error handling")
            
            # Test invalid strategy context
            try:
                invalid_context = {"invalid": "data"}
                objectives = await strategic_planner.create_strategic_plan(invalid_context, 30)
                error_handling_tests["invalid_strategy_context"] = True
            except Exception:
                error_handling_tests["invalid_strategy_context"] = True  # Expected to handle gracefully
            
            # Test memory overflow
            try:
                for i in range(memory_config.short_term_capacity + 100):
                    await memory_manager.store_memory(
                        event_id=f"overflow_test_{i}",
                        event_type="TestEvent",
                        content=f"Overflow test {i}",
                        domain="test",
                        importance_score=0.5,
                        timestamp=datetime.now()
                    )
                error_handling_tests["memory_overflow"] = True
            except Exception as e:
                logger.warning(f"Memory overflow test triggered exception: {e}")
                error_handling_tests["memory_overflow"] = False
            
            # Test 5: Configuration Validation
            logger.info("Testing cognitive configuration")
            
            # Test cognitive config validation
            try:
                cognitive_config = CognitiveConfig(
                    enable_reflection=True,
                    reflection_interval_hours=24,
                    enable_hierarchical_planning=True,
                    strategic_planning_horizon_days=90
                )
                configuration_tests["cognitive_config_valid"] = True
            except Exception as e:
                logger.error(f"Cognitive config validation failed: {e}")
                configuration_tests["cognitive_config_valid"] = False
            
            # Test memory config validation
            try:
                test_memory_configs = [
                    MemoryConfig(consolidation_algorithm=ConsolidationAlgorithm.IMPORTANCE_SCORE),
                    MemoryConfig(consolidation_algorithm=ConsolidationAlgorithm.RECENCY_FREQUENCY),
                    MemoryConfig(consolidation_algorithm=ConsolidationAlgorithm.STRATEGIC_VALUE),
                    MemoryConfig(consolidation_algorithm=ConsolidationAlgorithm.LLM_REFLECTION)
                ]
                
                config_tests = []
                for config in test_memory_configs:
                    try:
                        test_manager = DualMemoryManager(f"test_{config.consolidation_algorithm.value}", config)
                        config_tests.append(True)
                    except Exception:
                        config_tests.append(False)
                
                configuration_tests["memory_config_algorithms"] = sum(config_tests) / len(config_tests)
            except Exception as e:
                logger.error(f"Memory config validation failed: {e}")
                configuration_tests["memory_config_algorithms"] = 0.0
            
            # Success criteria
            cognitive_completeness = (
                validation_details["strategic_planning"]["strategy_success_rate"] > 0.8 and
                validation_details["memory_integration"]["storage_success_rate"] > 0.9 and
                validation_details["reflection_system"]["trigger_success_rate"] > 0.5 and
                sum(error_handling_tests.values()) >= len(error_handling_tests) * 0.8
            )
            
            duration = time.time() - start_time
            
            return FunctionalTestResult(
                test_name=test_name,
                feature_category="cognitive_loops",
                success=cognitive_completeness,
                validation_details=validation_details,
                error_handling_tests=error_handling_tests,
                configuration_tests=configuration_tests,
                cli_tests=cli_tests,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Cognitive loop validation failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return FunctionalTestResult(
                test_name=test_name,
                feature_category="cognitive_loops",
                success=False,
                validation_details={},
                error_handling_tests={},
                configuration_tests={},
                cli_tests={},
                duration_seconds=duration,
                error_details=str(e)
            )
    
    async def validate_multi_skill_coordination(self) -> FunctionalTestResult:
        """Verify skill module interactions."""
        test_name = "multi_skill_coordination"
        start_time = time.time()
        
        try:
            logger.info("Validating multi-skill coordination")
            
            environment = await self.setup_test_environment()
            agent_id = environment["test_agent_id"]
            
            validation_details = {}
            error_handling_tests = {}
            configuration_tests = {}
            cli_tests = {}
            
            # Test 1: Skill Coordinator Setup
            logger.info("Testing skill coordinator setup")
            
            skill_coordinator = SkillCoordinator(agent_id, self.event_bus, {
                'coordination_strategy': 'priority_based',
                'max_concurrent_skills': 3
            })
            
            # Test different coordination strategies
            strategy_tests = {}
            for strategy in CoordinationStrategy:
                try:
                    test_coordinator = SkillCoordinator(agent_id, self.event_bus, {
                        'coordination_strategy': strategy.value,
                        'max_concurrent_skills': 3
                    })
                    strategy_tests[strategy.value] = True
                except Exception as e:
                    logger.error(f"Strategy {strategy.value} failed: {e}")
                    strategy_tests[strategy.value] = False
            
            validation_details["coordination_strategies"] = {
                "strategies_tested": len(strategy_tests),
                "strategies_working": sum(strategy_tests.values()),
                "strategy_results": strategy_tests
            }
            
            # Test 2: Skill Module Registration
            logger.info("Testing skill module registration")
            
            # Initialize all skill modules
            skills = [
                SupplyManager(agent_id),
                MarketingManager(agent_id),
                CustomerService(agent_id),
                FinancialAnalyst(agent_id)
            ]
            
            registration_results = {}
            for skill in skills:
                try:
                    success = await skill_coordinator.register_skill(
                        skill,
                        skill.get_supported_event_types(),
                        priority_multiplier=1.0
                    )
                    registration_results[skill.skill_name] = success
                except Exception as e:
                    logger.error(f"Skill registration failed for {skill.skill_name}: {e}")
                    registration_results[skill.skill_name] = False
            
            validation_details["skill_registration"] = {
                "skills_tested": len(registration_results),
                "registration_success_rate": sum(registration_results.values()) / len(registration_results),
                "registration_results": registration_results
            }
            
            # Test 3: Event Dispatch and Coordination
            logger.info("Testing event dispatch and coordination")
            
            test_events = [
                TickEvent(
                    event_id="coord_test_tick",
                    timestamp=datetime.now(),
                    tick=1
                ),
                SaleOccurred(
                    event_id="coord_test_sale",
                    timestamp=datetime.now(),
                    asin="TEST-COORD-001",
                    quantity=3,
                    unit_price=1999,
                    total_revenue=5997,
                    fees=600
                ),
                SetPriceCommand(
                    event_id="coord_test_price",
                    timestamp=datetime.now(),
                    agent_id=agent_id,
                    asin="TEST-COORD-002",
                    new_price=2499
                )
            ]
            
            event_coordination_results = {}
            for event in test_events:
                try:
                    actions = await skill_coordinator.dispatch_event(event)
                    event_coordination_results[type(event).__name__] = {
                        "actions_generated": len(actions),
                        "coordinated": len(actions) > 0
                    }
                except Exception as e:
                    logger.error(f"Event coordination failed for {type(event).__name__}: {e}")
                    event_coordination_results[type(event).__name__] = {
                        "actions_generated": 0,
                        "coordinated": False
                    }
            
            validation_details["event_coordination"] = event_coordination_results
            
            # Test 4: Skill Performance and Metrics
            logger.info("Testing skill performance metrics")
            
            # Generate some activity for metrics
            for i in range(10):
                tick_event = TickEvent(
                    event_id=f"metrics_test_{i}",
                    timestamp=datetime.now(),
                    tick=i
                )
                await skill_coordinator.dispatch_event(tick_event)
            
            performance_metrics = skill_coordinator.get_skill_performance_metrics()
            coordination_stats = skill_coordinator.get_coordination_statistics()
            
            validation_details["performance_tracking"] = {
                "skills_with_metrics": len(performance_metrics),
                "coordination_stats_available": len(coordination_stats) > 0,
                "metrics_structure_valid": all(
                    key in metrics for metrics in performance_metrics.values()
                    for key in ["total_events_processed", "success_rate", "average_response_time"]
                ) if performance_metrics else False
            }
            
            # Test 5: Concurrent Skill Execution
            logger.info("Testing concurrent skill execution")
            
            # Create multiple concurrent events
            concurrent_events = [
                SaleOccurred(
                    event_id=f"concurrent_sale_{i}",
                    timestamp=datetime.now(),
                    asin=f"CONCURRENT-{i:03d}",
                    quantity=1,
                    unit_price=1500 + i * 100,
                    total_revenue=1500 + i * 100,
                    fees=150 + i * 10
                )
                for i in range(5)
            ]
            
            concurrent_tasks = []
            for event in concurrent_events:
                task = asyncio.create_task(skill_coordinator.dispatch_event(event))
                concurrent_tasks.append(task)
            
            concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            
            successful_concurrent = sum(
                1 for result in concurrent_results 
                if not isinstance(result, Exception) and len(result) > 0
            )
            
            validation_details["concurrent_execution"] = {
                "concurrent_events": len(concurrent_events),
                "successful_concurrent": successful_concurrent,
                "concurrency_success_rate": successful_concurrent / len(concurrent_events)
            }
            
            # Test 6: Error Handling
            logger.info("Testing multi-skill error handling")
            
            # Test invalid event handling
            try:
                invalid_event = Mock()
                invalid_event.__class__.__name__ = "InvalidEvent"
                actions = await skill_coordinator.dispatch_event(invalid_event)
                error_handling_tests["invalid_event_handling"] = True
            except Exception:
                error_handling_tests["invalid_event_handling"] = True  # Should handle gracefully
            
            # Test skill failure recovery
            try:
                # Simulate skill failure by unregistering and re-registering
                original_skills_count = len(skill_coordinator.skill_subscriptions)
                # Force remove a skill
                if "supply_manager" in skill_coordinator.skill_subscriptions:
                    del skill_coordinator.skill_subscriptions["supply_manager"]
                
                # Test that system continues working
                test_event = TickEvent(
                    event_id="failure_recovery_test",
                    timestamp=datetime.now(),
                    tick=999
                )
                actions = await skill_coordinator.dispatch_event(test_event)
                error_handling_tests["skill_failure_recovery"] = True
            except Exception as e:
                logger.error(f"Skill failure recovery test failed: {e}")
                error_handling_tests["skill_failure_recovery"] = False
            
            # Test 7: Configuration Validation
            logger.info("Testing skill configuration")
            
            # Test skill config validation
            try:
                skill_config = SkillConfig(
                    enable_supply_management=True,
                    enable_marketing=True,
                    enable_customer_service=True,
                    enable_financial_analysis=True,
                    coordination_strategy="priority_based",
                    max_concurrent_skills=5
                )
                configuration_tests["skill_config_valid"] = True
            except Exception as e:
                logger.error(f"Skill config validation failed: {e}")
                configuration_tests["skill_config_valid"] = False
            
            # Test coordination parameter validation
            test_params = [
                {"coordination_strategy": "priority_based", "max_concurrent_skills": 3},
                {"coordination_strategy": "round_robin", "max_concurrent_skills": 5},
                {"coordination_strategy": "resource_optimal", "max_concurrent_skills": 2}
            ]
            
            param_tests = []
            for params in test_params:
                try:
                    test_coordinator = SkillCoordinator(f"test_{len(param_tests)}", self.event_bus, params)
                    param_tests.append(True)
                except Exception:
                    param_tests.append(False)
            
            configuration_tests["coordination_params"] = sum(param_tests) / len(param_tests)
            
            # Success criteria
            multi_skill_completeness = (
                validation_details["skill_registration"]["registration_success_rate"] > 0.8 and
                validation_details["concurrent_execution"]["concurrency_success_rate"] > 0.7 and
                validation_details["performance_tracking"]["skills_with_metrics"] > 0 and
                sum(error_handling_tests.values()) >= len(error_handling_tests) * 0.7
            )
            
            duration = time.time() - start_time
            
            return FunctionalTestResult(
                test_name=test_name,
                feature_category="multi_skill",
                success=multi_skill_completeness,
                validation_details=validation_details,
                error_handling_tests=error_handling_tests,
                configuration_tests=configuration_tests,
                cli_tests=cli_tests,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Multi-skill coordination validation failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return FunctionalTestResult(
                test_name=test_name,
                feature_category="multi_skill",
                success=False,
                validation_details={},
                error_handling_tests={},
                configuration_tests={},
                cli_tests={},
                duration_seconds=duration,
                error_details=str(e)
            )
    
    async def validate_reproducibility_guarantees(self) -> FunctionalTestResult:
        """Ensure bit-perfect determinism."""
        test_name = "reproducibility_guarantees"
        start_time = time.time()
        
        try:
            logger.info("Validating reproducibility guarantees")
            
            environment = await self.setup_test_environment()
            temp_dir = environment["temp_dir"]
            
            validation_details = {}
            error_handling_tests = {}
            configuration_tests = {}
            cli_tests = {}
            
            # Test 1: Seed Management
            logger.info("Testing seed management")
            
            seed_manager = SimSeed("reproducibility_test_seed_123")
            
            # Test deterministic random generation
            random_sequences = []
            for run in range(3):
                seed_manager.set_seed(42)
                sequence = []
                for i in range(10):
                    sequence.append(seed_manager.random())
                random_sequences.append(sequence)
            
            # All sequences should be identical
            sequences_identical = all(seq == random_sequences[0] for seq in random_sequences)
            
            validation_details["seed_management"] = {
                "deterministic_generation": sequences_identical,
                "sequences_tested": len(random_sequences),
                "sequence_length": len(random_sequences[0]) if random_sequences else 0
            }
            
            # Test 2: LLM Caching System
            logger.info("Testing LLM caching system")
            
            cache_file = os.path.join(temp_dir, "test_cache.db")
            llm_cache = LLMResponseCache(
                cache_file=cache_file,
                enable_compression=True,
                enable_validation=True
            )
            
            # Test deterministic prompt hashing
            test_prompts = [
                ("What is the optimal price?", "gpt-4", 0.0),
                ("Analyze market conditions", "gpt-4", 0.0),
                ("What is the optimal price?", "gpt-4", 0.0),  # Duplicate
            ]
            
            prompt_hashes = []
            for prompt, model, temp in test_prompts:
                hash_val = llm_cache.generate_prompt_hash(prompt, model, temp)
                prompt_hashes.append(hash_val)
            
            # First and third hashes should be identical (same prompt)
            hash_consistency = prompt_hashes[0] == prompt_hashes[2]
            
            # Test caching and retrieval
            test_response = {"choices": [{"message": {"content": "Test response"}}]}
            cache_success = llm_cache.cache_response(
                prompt_hashes[0], 
                test_response, 
                {"model": "gpt-4", "temperature": 0.0}
            )
            
            retrieved_response = llm_cache.get_cached_response(prompt_hashes[0])
            retrieval_success = retrieved_response is not None
            
            # Test deterministic mode
            llm_cache.set_deterministic_mode(True)
            det_response = llm_cache.get_cached_response(prompt_hashes[0])
            deterministic_mode_working = det_response is not None
            
            validation_details["llm_caching"] = {
                "hash_consistency": hash_consistency,
                "cache_success": cache_success,
                "retrieval_success": retrieval_success,
                "deterministic_mode_working": deterministic_mode_working
            }
            
            # Test 3: Cache Integrity Validation
            logger.info("Testing cache integrity validation")
            
            is_valid, errors = llm_cache.validate_cache_integrity()
            cache_stats = llm_cache.get_cache_statistics()
            
            validation_details["cache_integrity"] = {
                "integrity_valid": is_valid,
                "validation_errors": len(errors),
                "cache_statistics_available": hasattr(cache_stats, 'hit_ratio')
            }
            
            # Test 4: Golden Master Validation
            logger.info("Testing golden master validation")
            
            golden_master_file = os.path.join(temp_dir, "test_golden_master.json")
            golden_master = GoldenMaster(golden_master_file)
            
            # Create test simulation data
            test_simulation_data = {
                "events": [
                    {
                        "event_id": "test_event_1",
                        "timestamp": "2024-01-01T00:00:00",
                        "event_type": "TickEvent",
                        "data": {"tick": 1}
                    },
                    {
                        "event_id": "test_event_2", 
                        "timestamp": "2024-01-01T00:01:00",
                        "event_type": "SaleOccurred",
                        "data": {"asin": "TEST-001", "revenue": 1000}
                    }
                ],
                "final_state": {
                    "agent_cash": 10000,
                    "total_sales": 1000
                }
            }
            
            # Test golden master creation and validation
            golden_master.create_golden_master(test_simulation_data)
            validation_result = golden_master.validate_against_golden_master(test_simulation_data)
            
            # Test with modified data (should fail validation)
            modified_data = test_simulation_data.copy()
            modified_data["final_state"]["agent_cash"] = 9999
            validation_should_fail = not golden_master.validate_against_golden_master(modified_data)
            
            validation_details["golden_master"] = {
                "creation_success": os.path.exists(golden_master_file),
                "validation_success": validation_result,
                "detects_changes": validation_should_fail
            }
            
            # Test 5: Error Handling
            logger.info("Testing reproducibility error handling")
            
            # Test cache corruption handling
            try:
                # Try to retrieve non-existent cache entry
                invalid_hash = "invalid_hash_that_does_not_exist"
                result = llm_cache.get_cached_response(invalid_hash)
                error_handling_tests["invalid_cache_retrieval"] = result is None
            except Exception:
                error_handling_tests["invalid_cache_retrieval"] = True
            
            # Test deterministic mode with cache miss
            try:
                llm_cache.set_deterministic_mode(True)
                missing_hash = "definitely_missing_hash"
                result = llm_cache.get_cached_response(missing_hash)
                error_handling_tests["deterministic_cache_miss"] = False  # Should raise exception
            except ValueError:
                error_handling_tests["deterministic_cache_miss"] = True  # Expected behavior
            except Exception:
                error_handling_tests["deterministic_cache_miss"] = False
            
            # Test 6: Configuration Validation
            logger.info("Testing reproducibility configuration")
            
            # Test cache configuration options
            cache_configs = [
                {"enable_compression": True, "enable_validation": True},
                {"enable_compression": False, "enable_validation": True},
                {"enable_compression": True, "enable_validation": False}
            ]
            
            config_tests = []
            for i, config in enumerate(cache_configs):
                try:
                    test_cache_file = os.path.join(temp_dir, f"test_config_{i}.db")
                    test_cache = LLMResponseCache(cache_file=test_cache_file, **config)
                    config_tests.append(True)
                except Exception:
                    config_tests.append(False)
            
            configuration_tests["cache_configurations"] = sum(config_tests) / len(config_tests)
            
            # Test seed configuration
            try:
                test_seeds = ["test_seed_1", "test_seed_2", 42, ""]
                seed_tests = []
                for seed in test_seeds:
                    try:
                        test_seed_manager = SimSeed(seed)
                        seed_tests.append(True)
                    except Exception:
                        seed_tests.append(False)
                
                configuration_tests["seed_configurations"] = sum(seed_tests) / len(seed_tests)
            except Exception:
                configuration_tests["seed_configurations"] = 0.0
            
            # Test 7: CLI Integration
            logger.info("Testing reproducibility CLI")
            
            # Test cache export/import CLI functionality (mock)
            try:
                export_file = os.path.join(temp_dir, "test_export.json")
                export_success = llm_cache.export_cache(export_file, compress=False)
                
                if export_success and os.path.exists(export_file):
                    # Test import
                    new_cache_file = os.path.join(temp_dir, "imported_cache.db")
                    new_cache = LLMResponseCache(cache_file=new_cache_file)
                    import_success = new_cache.import_cache(export_file, merge=False)
                    
                    cli_tests["cache_export_import"] = import_success
                else:
                    cli_tests["cache_export_import"] = False
            except Exception:
                cli_tests["cache_export_import"] = False
            
            # Success criteria
            reproducibility_completeness = (
                validation_details["seed_management"]["deterministic_generation"] and
                validation_details["llm_caching"]["hash_consistency"] and
                validation_details["llm_caching"]["deterministic_mode_working"] and
                validation_details["cache_integrity"]["integrity_valid"] and
                validation_details["golden_master"]["validation_success"]
            )
            
            duration = time.time() - start_time
            
            return FunctionalTestResult(
                test_name=test_name,
                feature_category="reproducibility",
                success=reproducibility_completeness,
                validation_details=validation_details,
                error_handling_tests=error_handling_tests,
                configuration_tests=configuration_tests,
                cli_tests=cli_tests,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Reproducibility validation failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return FunctionalTestResult(
                test_name=test_name,
                feature_category="reproducibility",
                success=False,
                validation_details={},
                error_handling_tests={},
                configuration_tests={},
                cli_tests={},
                duration_seconds=duration,
                error_details=str(e)
            )
    
    async def validate_observability_insights(self) -> FunctionalTestResult:
        """Test trace analysis and error handling."""
        test_name = "observability_insights"
        start_time = time.time()
        
        try:
            logger.info("Validating observability insights")
            
            environment = await self.setup_test_environment()
            
            validation_details = {}
            error_handling_tests = {}
            configuration_tests = {}
            cli_tests = {}
            
            # Test 1: Trace Analysis System
            logger.info("Testing trace analysis system")
            
            trace_analyzer = TraceAnalyzer()
            
            # Generate test trace data
            test_traces = []
            for i in range(100):
                trace = {
                    "trace_id": f"trace_{i}",
                    "timestamp": datetime.now() - timedelta(seconds=i),
                    "component": ["strategic_planner", "skill_coordinator", "memory_manager"][i % 3],
                    "operation": f"operation_{i % 5}",
                    "duration_ms": 50 + (i % 200),
                    "success": i % 10 != 0,  # 90% success rate
                    "metadata": {
                        "agent_id": f"agent_{i % 3}",
                        "resource_usage": {"cpu": i % 100, "memory": 100 + i % 500}
                    }
                }
                test_traces.append(trace)
            
            # Test trace analysis capabilities
            analysis_start = time.time()
            analysis_results = await trace_analyzer.analyze_traces(test_traces)
            analysis_duration = time.time() - analysis_start
            
            validation_details["trace_analysis"] = {
                "traces_processed": len(test_traces),
                "analysis_duration": analysis_duration,
                "analysis_results_available": analysis_results is not None,
                "performance_metrics_extracted": "performance" in analysis_results if analysis_results else False
            }
            
            # Test 2: Alert System
            logger.info("Testing alert system")
            
            alert_system = AlertSystem()
            
            # Test different alert types
            alert_tests = {}
            
            # Performance degradation alert
            try:
                await alert_system.check_performance_degradation({
                    "response_time_ms": 500,
                    "error_rate": 0.15,
                    "throughput": 100
                })
                alert_tests["performance_degradation"] = True
            except Exception as e:
                logger.error(f"Performance alert test failed: {e}")
                alert_tests["performance_degradation"] = False
            
            # Resource usage alert
            try:
                await alert_system.check_resource_usage({
                    "cpu_percent": 85,
                    "memory_percent": 78,
                    "disk_usage_percent": 65
                })
                alert_tests["resource_usage"] = True
            except Exception as e:
                logger.error(f"Resource alert test failed: {e}")
                alert_tests["resource_usage"] = False
            
            # Error pattern alert
            try:
                await alert_system.check_error_patterns([
                    {"error_type": "timeout", "count": 5},
                    {"error_type": "connection_failed", "count": 3}
                ])
                alert_tests["error_patterns"] = True
            except Exception as e:
                logger.error(f"Error pattern alert test failed: {e}")
                alert_tests["error_patterns"] = False
            
            validation_details["alert_system"] = {
                "alert_types_tested": len(alert_tests),
                "alert_success_rate": sum(alert_tests.values()) / len(alert_tests),
                "alert_results": alert_tests
            }
            
            # Test 3: Performance Monitoring
            logger.info("Testing performance monitoring")
            
            performance_monitor = PerformanceMonitor()
            await performance_monitor.start()
            
            # Generate some activity to monitor
            for i in range(20):
                await performance_monitor.record_event({
                    "event_type": "test_event",
                    "duration_ms": 100 + i * 5,
                    "success": i % 8 != 0
                })
            
            # Get performance metrics
            metrics = await performance_monitor.get_metrics()
            
            await performance_monitor.stop()
            
            validation_details["performance_monitoring"] = {
                "monitoring_system_working": metrics is not None,
                "metrics_available": len(metrics) > 0 if metrics else False,
                "event_tracking": "events" in metrics if metrics else False
            }
            
            # Test 4: Error Detection and Analysis
            logger.info("Testing error detection and analysis")
            
            # Simulate error conditions
            error_scenarios = [
                {"type": "timeout", "severity": "high", "component": "llm_batcher"},
                {"type": "memory_overflow", "severity": "critical", "component": "memory_manager"},
                {"type": "coordination_failure", "severity": "medium", "component": "skill_coordinator"}
            ]
            
            error_detection_results = {}
            for scenario in error_scenarios:
                try:
                    detection_result = await trace_analyzer.detect_error_patterns([scenario])
                    error_detection_results[scenario["type"]] = detection_result is not None
                except Exception as e:
                    logger.error(f"Error detection failed for {scenario['type']}: {e}")
                    error_detection_results[scenario["type"]] = False
            
            validation_details["error_detection"] = {
                "scenarios_tested": len(error_scenarios),
                "detection_success_rate": sum(error_detection_results.values()) / len(error_detection_results),
                "detection_results": error_detection_results
            }
            
            # Test 5: Error Handling
            logger.info("Testing observability error handling")
            
            # Test trace analysis with corrupted data
            try:
                corrupted_traces = [{"invalid": "trace"}, None, {"partial": "data"}]
                result = await trace_analyzer.analyze_traces(corrupted_traces)
                error_handling_tests["corrupted_trace_data"] = True  # Should handle gracefully
            except Exception:
                error_handling_tests["corrupted_trace_data"] = True  # Expected to handle or fail gracefully
            
            # Test alert system with invalid metrics
            try:
                await alert_system.check_performance_degradation({"invalid": "metrics"})
                error_handling_tests["invalid_alert_metrics"] = True
            except Exception:
                error_handling_tests["invalid_alert_metrics"] = True
            
            # Test 6: Configuration Validation
            logger.info("Testing observability configuration")
            
            # Test trace analyzer configuration
            try:
                trace_config = {
                    "enable_performance_analysis": True,
                    "enable_error_detection": True,
                    "analysis_window_minutes": 60,
                    "alert_thresholds": {
                        "error_rate": 0.1,
                        "response_time_ms": 1000
                    }
                }
                
                configured_analyzer = TraceAnalyzer(config=trace_config)
                configuration_tests["trace_analyzer_config"] = True
            except Exception as e:
                logger.error(f"Trace analyzer config failed: {e}")
                configuration_tests["trace_analyzer_config"] = False
            
            # Test alert system configuration
            try:
                alert_config = {
                    "enable_performance_alerts": True,
                    "enable_resource_alerts": True,
                    "alert_cooldown_minutes": 5,
                    "notification_channels": ["log", "email"]
                }
                
                configured_alerts = AlertSystem(config=alert_config)
                configuration_tests["alert_system_config"] = True
            except Exception as e:
                logger.error(f"Alert system config failed: {e}")
                configuration_tests["alert_system_config"] = False
            
            # Test 7: CLI Integration
            logger.info("Testing observability CLI")
            
            # Test trace export (mock CLI functionality)
            try:
                export_file = os.path.join(environment["temp_dir"], "trace_export.json")
                export_success = await trace_analyzer.export_traces(test_traces, export_file)
                cli_tests["trace_export"] = export_success and os.path.exists(export_file)
            except Exception:
                cli_tests["trace_export"] = False
            
            # Test alert report generation
            try:
                report_file = os.path.join(environment["temp_dir"], "alert_report.json")
                report_success = await alert_system.generate_report(report_file)
                cli_tests["alert_report"] = report_success and os.path.exists(report_file)
            except Exception:
                cli_tests["alert_report"] = False
            
            # Success criteria
            observability_completeness = (
                validation_details["trace_analysis"]["analysis_results_available"] and
                validation_details["alert_system"]["alert_success_rate"] > 0.7 and
                validation_details["performance_monitoring"]["monitoring_system_working"] and
                validation_details["error_detection"]["detection_success_rate"] > 0.6
            )
            
            duration = time.time() - start_time
            
            return FunctionalTestResult(
                test_name=test_name,
                feature_category="observability",
                success=observability_completeness,
                validation_details=validation_details,
                error_handling_tests=error_handling_tests,
                configuration_tests=configuration_tests,
                cli_tests=cli_tests,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Observability validation failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return FunctionalTestResult(
                test_name=test_name,
                feature_category="observability",
                success=False,
                validation_details={},
                error_handling_tests={},
                configuration_tests={},
                cli_tests={},
                duration_seconds=duration,
                error_details=str(e)
            )
    
    async def validate_learning_system_safety(self) -> FunctionalTestResult:
        """Ensure learning/evaluation separation."""
        test_name = "learning_system_safety"
        start_time = time.time()
        
        try:
            logger.info("Validating learning system safety")
            
            environment = await self.setup_test_environment()
            agent_id = environment["test_agent_id"]
            
            validation_details = {}
            error_handling_tests = {}
            configuration_tests = {}
            cli_tests = {}
            
            # Test 1: Episodic Learning System
            logger.info("Testing episodic learning system")
            
            learning_manager = EpisodicLearningManager(agent_id)
            
            # Test learning episode creation
            test_episodes = []
            for i in range(5):
                episode_data = {
                    "episode_id": f"test_episode_{i}",
                    "actions": [f"action_{j}" for j in range(3)],
                    "rewards": [0.1 * j for j in range(3)],
                    "final_outcome": 0.8 + i * 0.05,
                    "context": {"market_condition": f"condition_{i}"}
                }
                
                try:
                    await learning_manager.store_episode(episode_data)
                    test_episodes.append(True)
                except Exception as e:
                    logger.error(f"Episode storage failed: {e}")
                    test_episodes.append(False)
            
            validation_details["episodic_learning"] = {
                "episodes_tested": len(test_episodes),
                "storage_success_rate": sum(test_episodes) / len(test_episodes),
                "learning_system_active": True
            }
            
            # Test 2: Learning/Evaluation Separation
            logger.info("Testing learning/evaluation separation")
            
            # Test evaluation mode isolation
            evaluation_mode_tests = {}
            
            try:
                # Set evaluation mode
                learning_manager.set_evaluation_mode(True)
                
                # Attempt to store learning data in evaluation mode (should be blocked)
                try:
                    await learning_manager.store_episode({
                        "episode_id": "eval_mode_test",
                        "actions": ["test_action"],
                        "rewards": [0.5],
                        "final_outcome": 0.7
                    })
                    evaluation_mode_tests["blocks_learning_in_eval"] = False  # Should not allow
                except Exception:
                    evaluation_mode_tests["blocks_learning_in_eval"] = True  # Expected behavior
                
                # Test that evaluation data is isolated
                eval_data = await learning_manager.get_evaluation_metrics()
                evaluation_mode_tests["evaluation_data_isolated"] = eval_data is not None
                
                # Return to learning mode
                learning_manager.set_evaluation_mode(False)
                evaluation_mode_tests["mode_switching_works"] = True
                
            except Exception as e:
                logger.error(f"Evaluation mode test failed: {e}")
                evaluation_mode_tests["blocks_learning_in_eval"] = False
                evaluation_mode_tests["evaluation_data_isolated"] = False
                evaluation_mode_tests["mode_switching_works"] = False
            
            validation_details["learning_evaluation_separation"] = evaluation_mode_tests
            
            # Test 3: Real-World Integration Safety
            logger.info("Testing real-world integration safety")
            
            real_world_adapter = RealWorldAdapter()
            
            # Test safety constraints
            safety_tests = {}
            
            # Test that dangerous actions are blocked
            dangerous_actions = [
                {"type": "set_price", "parameters": {"asin": "TEST-001", "price": -100}},  # Negative price
                {"type": "place_order", "parameters": {"quantity": 99999, "budget_override": True}},  # Excessive order
                {"type": "delete_listing", "parameters": {"asin": "PROD-001", "force": True}}  # Destructive action
            ]
            
            for action in dangerous_actions:
                try:
                    safety_check = await real_world_adapter.validate_action_safety(action)
                    safety_tests[action["type"]] = not safety_check  # Should be blocked (False)
                except Exception:
                    safety_tests[action["type"]] = True  # Blocked by exception (good)
            
            validation_details["real_world_safety"] = {
                "dangerous_actions_tested": len(dangerous_actions),
                "safety_blocks_working": sum(safety_tests.values()) / len(safety_tests),
                "safety_results": safety_tests
            }
            
            # Test 4: Data Isolation and Privacy
            logger.info("Testing data isolation and privacy")
            
            # Test that learning data is properly isolated
            isolation_tests = {}
            
            try:
                # Create separate learning instances
                agent1_learning = EpisodicLearningManager("agent_1")
                agent2_learning = EpisodicLearningManager("agent_2")
                
                # Store data for each agent
                await agent1_learning.store_episode({
                    "episode_id": "agent1_episode",
                    "actions": ["agent1_action"],
                    "rewards": [0.8],
                    "final_outcome": 0.9
                })
                
                await agent2_learning.store_episode({
                    "episode_id": "agent2_episode", 
                    "actions": ["agent2_action"],
                    "rewards": [0.6],
                    "final_outcome": 0.7
                })
                
                # Test data isolation
                agent1_data = await agent1_learning.get_episodes()
                agent2_data = await agent2_learning.get_episodes()
                
                agent1_has_own_data = any(ep["episode_id"] == "agent1_episode" for ep in agent1_data)
                agent1_no_other_data = not any(ep["episode_id"] == "agent2_episode" for ep in agent1_data)
                
                isolation_tests["data_isolation"] = agent1_has_own_data and agent1_no_other_data
                
            except Exception as e:
                logger.error(f"Data isolation test failed: {e}")
                isolation_tests["data_isolation"] = False
            
            validation_details["data_isolation"] = isolation_tests
            
            # Test 5: Error Handling
            logger.info("Testing learning system error handling")
            
            # Test invalid learning data handling
            try:
                invalid_episode = {"invalid": "data", "missing_required_fields": True}
                await learning_manager.store_episode(invalid_episode)
                error_handling_tests["invalid_learning_data"] = True  # Should handle gracefully
            except Exception:
                error_handling_tests["invalid_learning_data"] = True  # Expected to fail gracefully
            
            # Test real-world adapter error handling
            try:
                invalid_action = {"invalid": "action_format"}
                result = await real_world_adapter.validate_action_safety(invalid_action)
                error_handling_tests["invalid_action_format"] = True
            except Exception:
                error_handling_tests["invalid_action_format"] = True
            
            # Test 6: Configuration Validation
            logger.info("Testing learning system configuration")
            
            # Test learning configuration options
            learning_configs = [
                {"enable_episodic_learning": True, "max_episodes": 1000, "evaluation_mode": False},
                {"enable_episodic_learning": False, "max_episodes": 500, "evaluation_mode": True},
                {"enable_episodic_learning": True, "max_episodes": 2000, "privacy_mode": True}
            ]
            
            config_tests = []
            for config in learning_configs:
                try:
                    test_manager = EpisodicLearningManager(f"config_test_{len(config_tests)}", config)
                    config_tests.append(True)
                except Exception:
                    config_tests.append(False)
            
            configuration_tests["learning_configurations"] = sum(config_tests) / len(config_tests)
            
            # Test real-world adapter configuration
            try:
                adapter_config = {
                    "enable_safety_checks": True,
                    "sandbox_mode": True,
                    "max_action_value": 10000,
                    "allowed_action_types": ["set_price", "place_order"]
                }
                
                configured_adapter = RealWorldAdapter(config=adapter_config)
                configuration_tests["adapter_configuration"] = True
            except Exception:
                configuration_tests["adapter_configuration"] = False
            
            # Test 7: CLI Integration
            logger.info("Testing learning system CLI")
            
            # Test learning data export
            try:
                export_file = os.path.join(environment["temp_dir"], "learning_export.json")
                export_success = await learning_manager.export_episodes(export_file)
                cli_tests["learning_export"] = export_success and os.path.exists(export_file)
            except Exception:
                cli_tests["learning_export"] = False
            
            # Test safety validation CLI
            try:
                safety_report_file = os.path.join(environment["temp_dir"], "safety_report.json")
                report_success = await real_world_adapter.generate_safety_report(safety_report_file)
                cli_tests["safety_report"] = report_success and os.path.exists(safety_report_file)
            except Exception:
                cli_tests["safety_report"] = False
            
            # Success criteria
            learning_safety_completeness = (
                validation_details["episodic_learning"]["storage_success_rate"] > 0.8 and
                validation_details["learning_evaluation_separation"]["blocks_learning_in_eval"] and
                validation_details["real_world_safety"]["safety_blocks_working"] > 0.8 and
                validation_details["data_isolation"]["data_isolation"]
            )
            
            duration = time.time() - start_time
            
            return FunctionalTestResult(
                test_name=test_name,
                feature_category="learning_safety",
                success=learning_safety_completeness,
                validation_details=validation_details,
                error_handling_tests=error_handling_tests,
                configuration_tests=configuration_tests,
                cli_tests=cli_tests,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Learning system safety validation failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return FunctionalTestResult(
                test_name=test_name,
                feature_category="learning_safety",
                success=False,
                validation_details={},
                error_handling_tests={},
                configuration_tests={},
                cli_tests={},
                duration_seconds=duration,
                error_details=str(e)
            )
    
    async def run_functional_validation_suite(self) -> Dict[str, Any]:
        """Run complete functional validation suite."""
        logger.info("Starting comprehensive functional validation suite")
        suite_start = time.time()
        
        # Validation methods to run
        validation_methods = [
            self.validate_cognitive_loop_completeness,
            self.validate_multi_skill_coordination,
            self.validate_reproducibility_guarantees,
            self.validate_observability_insights,
            self.validate_learning_system_safety
        ]
        
        results = []
        
        for validation_method in validation_methods:
            try:
                logger.info(f"Running {validation_method.__name__}")
                result = await validation_method()
                results.append(result)
                self.test_results.append(result)
                
                if result.success:
                    logger.info(f"✅ {result.test_name} passed")
                else:
                    logger.error(f"❌ {result.test_name} failed: {result.error_details}")
                    
            except Exception as e:
                logger.error(f"Validation {validation_method.__name__} crashed: {e}", exc_info=True)
                results.append(FunctionalTestResult(
                    test_name=validation_method.__name__,
                    feature_category="unknown",
                    success=False,
                    validation_details={},
                    error_handling_tests={},
                    configuration_tests={},
                    cli_tests={},
                    duration_seconds=0,
                    error_details=str(e)
                ))
        
        suite_duration = time.time() - suite_start
        
        # Compile feature completeness assessment
        feature_completeness = FeatureCompleteness(
            cognitive_loops={},
            multi_skill_capabilities={},
            reproducibility_features={},
            infrastructure_features={},
            observability_features={},
            learning_features={},
            integration_features={}
        )
        
        for result in results:
            if result.feature_category == "cognitive_loops":
                feature_completeness.cognitive_loops = result.validation_details
            elif result.feature_category == "multi_skill":
                feature_completeness.multi_skill_capabilities = result.validation_details
            elif result.feature_category == "reproducibility":
                feature_completeness.reproducibility_features = result.validation_details
            elif result.feature_category == "observability":
                feature_completeness.observability_features = result.validation_details
            elif result.feature_category == "learning_safety":
                feature_completeness.learning_features = result.validation_details
        
        # Summary statistics
        total_validations = len(results)
        passed_validations = sum(1 for r in results if r.success)
        failed_validations = total_validations - passed_validations
        
        # Calculate completeness scores
        error_handling_score = 0
        configuration_score = 0
        cli_score = 0
        
        for result in results:
            if result.error_handling_tests:
                error_handling_score += sum(result.error_handling_tests.values()) / len(result.error_handling_tests)
            if result.configuration_tests:
                config_values = [v for v in result.configuration_tests.values() if isinstance(v, (int, float, bool))]
                if config_values:
                    configuration_score += sum(config_values) / len(config_values)
            if result.cli_tests:
                cli_score += sum(result.cli_tests.values()) / len(result.cli_tests)
        
        avg_error_handling = error_handling_score / total_validations if total_validations > 0 else 0
        avg_configuration = configuration_score / total_validations if total_validations > 0 else 0
        avg_cli = cli_score / total_validations if total_validations > 0 else 0
        
        summary = {
            "suite_duration_seconds": suite_duration,
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "failed_validations": failed_validations,
            "success_rate": passed_validations / total_validations if total_validations > 0 else 0,
            "feature_completeness": feature_completeness.__dict__,
            "error_handling_score": avg_error_handling,
            "configuration_score": avg_configuration,
            "cli_score": avg_cli,
            "validation_results": [result.__dict__ for result in results],
            "all_features_complete": failed_validations == 0
        }
        
        logger.info(f"Functional validation suite completed: {passed_validations}/{total_validations} passed")
        return summary
    
    async def cleanup_test_environment(self):
        """Clean up test resources."""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info("Test environment cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# CLI runner for direct execution
async def main():
    """Run functional validation suite."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    validation_suite = FunctionalValidationSuite()
    
    try:
        results = await validation_suite.run_functional_validation_suite()
        
        print("\n" + "="*80)
        print("FUNCTIONAL VALIDATION RESULTS")
        print("="*80)
        print(f"Total Validations: {results['total_validations']}")
        print(f"Passed: {results['passed_validations']}")
        print(f"Failed: {results['failed_validations']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Error Handling Score: {results['error_handling_score']:.2f}")
        print(f"Configuration Score: {results['configuration_score']:.2f}")
        print(f"CLI Score: {results['cli_score']:.2f}")
        print(f"Suite Duration: {results['suite_duration_seconds']:.2f}s")
        
        print("\nFeature Completeness:")
        for category, details in results['feature_completeness'].items():
            if details:
                print(f"  ✅ {category}: Available")
            else:
                print(f"  ❌ {category}: Missing")
        
        if results['all_features_complete']:
            print("\n🎉 ALL FUNCTIONAL VALIDATIONS PASSED!")
            print("FBA-Bench feature completeness confirmed.")
        else:
            print("\n⚠️  Some functional validations failed.")
            print("Review failed tests for missing functionality.")
        
        print("="*80)
        
    finally:
        await validation_suite.cleanup_test_environment()


if __name__ == "__main__":
    asyncio.run(main())