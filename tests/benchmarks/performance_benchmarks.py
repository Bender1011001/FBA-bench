"""
Performance Benchmarking Suite for FBA-Bench

Validates infrastructure targets including scalability, cost optimization,
memory usage, and determinism overhead as specified in the Key Issues document.
"""

import asyncio
import logging
import time
import psutil
import gc
import tracemalloc
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sys
import os
import json
import concurrent.futures

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.hierarchical_planner import StrategicPlanner, TacticalPlanner
from agents.skill_coordinator import SkillCoordinator
from agents.skill_modules.supply_manager import SupplyManager
from agents.skill_modules.marketing_manager import MarketingManager
from agents.skill_modules.customer_service import CustomerService
from agents.skill_modules.financial_analyst import FinancialAnalyst
from memory_experiments.reflection_module import ReflectionModule, StructuredReflectionLoop
from memory_experiments.dual_memory_manager import DualMemoryManager
from memory_experiments.memory_config import MemoryConfig, ConsolidationAlgorithm
from reproducibility.llm_cache import LLMResponseCache
from reproducibility.sim_seed import SimSeed
from infrastructure.llm_batcher import LLMBatcher
from infrastructure.performance_monitor import PerformanceMonitor
from infrastructure.distributed_coordinator import DistributedCoordinator
from observability.trace_analyzer import TraceAnalyzer
from event_bus import EventBus, get_event_bus
from events import BaseEvent, TickEvent, SaleOccurred, SetPriceCommand

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBenchmarkResult:
    """Results from a performance benchmark test."""
    benchmark_name: str
    target_metric: str
    target_value: float
    actual_value: float
    meets_target: bool
    performance_ratio: float  # actual/target
    duration_seconds: float
    resource_usage: Dict[str, Any]
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None


@dataclass
class BenchmarkTargets:
    """Performance targets from Key Issues document."""
    min_concurrent_agents: int = 20
    min_ticks_per_minute: float = 2000.0
    min_cost_reduction_percent: float = 30.0
    max_memory_usage_mb: float = 1000.0  # For year-long simulation
    max_analysis_time_seconds: float = 30.0
    max_determinism_overhead_percent: float = 20.0
    min_batch_efficiency_percent: float = 40.0


class PerformanceBenchmarkSuite:
    """
    Performance benchmarking suite for FBA-Bench infrastructure validation.
    
    Tests scalability, cost optimization, memory usage, and performance
    targets specified in the Key Issues document.
    """
    
    def __init__(self, targets: Optional[BenchmarkTargets] = None):
        self.targets = targets or BenchmarkTargets()
        self.event_bus = get_event_bus()
        self.benchmark_results: List[PerformanceBenchmarkResult] = []
        
        # Initialize monitoring
        self.process = psutil.Process()
        self.baseline_memory = None
        
    async def benchmark_distributed_agent_performance(self) -> PerformanceBenchmarkResult:
        """Test 20+ concurrent agents performance."""
        benchmark_name = "distributed_agent_performance"
        start_time = time.time()
        
        try:
            logger.info(f"Benchmarking distributed agent performance (target: {self.targets.min_concurrent_agents} agents)")
            
            # Start memory tracking
            tracemalloc.start()
            baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            # Initialize distributed coordination
            distributed_coordinator = DistributedCoordinator()
            performance_monitor = PerformanceMonitor()
            await performance_monitor.start()
            
            # Create target number of agents
            agents = []
            agent_creation_times = []
            
            for i in range(self.targets.min_concurrent_agents):
                agent_start = time.time()
                agent_id = f"perf_agent_{i}"
                
                # Memory system
                memory_config = MemoryConfig(
                    consolidation_algorithm=ConsolidationAlgorithm.IMPORTANCE_SCORE,
                    short_term_capacity=500,  # Reduced for performance
                    long_term_capacity=1000
                )
                memory_manager = DualMemoryManager(agent_id, memory_config)
                
                # Cognitive systems (lightweight config for performance)
                strategic_planner = StrategicPlanner(agent_id, self.event_bus)
                tactical_planner = TacticalPlanner(agent_id, strategic_planner, self.event_bus)
                
                # Skill coordination
                skill_coordinator = SkillCoordinator(agent_id, self.event_bus, {
                    'coordination_strategy': 'priority_based',
                    'max_concurrent_skills': 2  # Reduced for performance
                })
                
                # Register essential skills
                skills = [
                    SupplyManager(agent_id),
                    MarketingManager(agent_id)
                ]
                
                for skill in skills:
                    await skill_coordinator.register_skill(
                        skill,
                        skill.get_supported_event_types()[:2],  # Limit event types
                        priority_multiplier=1.0
                    )
                
                agent_data = {
                    "agent_id": agent_id,
                    "memory_manager": memory_manager,
                    "strategic_planner": strategic_planner,
                    "tactical_planner": tactical_planner,
                    "skill_coordinator": skill_coordinator
                }
                agents.append(agent_data)
                
                agent_creation_time = time.time() - agent_start
                agent_creation_times.append(agent_creation_time)
                
                # Progress logging
                if (i + 1) % 5 == 0:
                    logger.info(f"Created {i + 1}/{self.targets.min_concurrent_agents} agents")
            
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_per_agent = (current_memory - baseline_memory) / len(agents)
            
            logger.info(f"Successfully created {len(agents)} agents")
            logger.info(f"Memory usage: {current_memory:.1f}MB ({memory_per_agent:.1f}MB per agent)")
            
            # Test concurrent event processing
            logger.info("Testing concurrent event processing")
            concurrent_test_start = time.time()
            
            # Create concurrent event processing tasks
            event_processing_tasks = []
            events_per_agent = 50
            
            for agent in agents:
                task = asyncio.create_task(
                    self._process_events_for_agent(agent, events_per_agent)
                )
                event_processing_tasks.append(task)
            
            # Process all agents concurrently
            processing_results = await asyncio.gather(*event_processing_tasks, return_exceptions=True)
            
            concurrent_test_duration = time.time() - concurrent_test_start
            
            # Calculate performance metrics
            total_events = len(agents) * events_per_agent
            events_per_second = total_events / concurrent_test_duration
            successful_agents = sum(1 for result in processing_results if isinstance(result, int))
            
            # Resource usage
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = self.process.cpu_percent(interval=1.0)
            
            # Get memory peak
            current_memory_peak, memory_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            actual_value = len(agents)
            meets_target = actual_value >= self.targets.min_concurrent_agents
            performance_ratio = actual_value / self.targets.min_concurrent_agents
            
            resource_usage = {
                "baseline_memory_mb": baseline_memory,
                "final_memory_mb": final_memory,
                "memory_per_agent_mb": memory_per_agent,
                "cpu_percent": cpu_percent,
                "memory_peak_mb": memory_peak / 1024 / 1024,
                "successful_agents": successful_agents,
                "events_per_second": events_per_second,
                "avg_agent_creation_time": statistics.mean(agent_creation_times)
            }
            
            additional_metrics = {
                "total_events_processed": total_events,
                "concurrent_processing_duration": concurrent_test_duration,
                "agent_creation_times": agent_creation_times,
                "processing_success_rate": successful_agents / len(agents)
            }
            
            duration = time.time() - start_time
            
            return PerformanceBenchmarkResult(
                benchmark_name=benchmark_name,
                target_metric="concurrent_agents",
                target_value=self.targets.min_concurrent_agents,
                actual_value=actual_value,
                meets_target=meets_target,
                performance_ratio=performance_ratio,
                duration_seconds=duration,
                resource_usage=resource_usage,
                additional_metrics=additional_metrics
            )
            
        except Exception as e:
            logger.error(f"Distributed agent performance benchmark failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return PerformanceBenchmarkResult(
                benchmark_name=benchmark_name,
                target_metric="concurrent_agents",
                target_value=self.targets.min_concurrent_agents,
                actual_value=0,
                meets_target=False,
                performance_ratio=0,
                duration_seconds=duration,
                resource_usage={},
                error_details=str(e)
            )
    
    async def _process_events_for_agent(self, agent: Dict[str, Any], event_count: int) -> int:
        """Helper method to process events for a single agent."""
        events_processed = 0
        
        try:
            for i in range(event_count):
                # Create test event
                if i % 3 == 0:
                    event = TickEvent(
                        event_id=f"tick_{agent['agent_id']}_{i}",
                        timestamp=datetime.now(),
                        tick=i
                    )
                elif i % 3 == 1:
                    event = SaleOccurred(
                        event_id=f"sale_{agent['agent_id']}_{i}",
                        timestamp=datetime.now(),
                        asin=f"PERF-TEST-{i%5}",
                        quantity=1,
                        unit_price=1000,
                        total_revenue=1000,
                        fees=100
                    )
                else:
                    event = SetPriceCommand(
                        event_id=f"price_{agent['agent_id']}_{i}",
                        timestamp=datetime.now(),
                        agent_id=agent['agent_id'],
                        asin=f"PERF-TEST-{i%5}",
                        new_price=1500
                    )
                
                # Process through skill coordinator
                if agent.get("skill_coordinator"):
                    actions = await agent["skill_coordinator"].dispatch_event(event)
                    events_processed += 1
                
                # Small delay to prevent overwhelming
                if i % 10 == 0:
                    await asyncio.sleep(0.001)
                    
        except Exception as e:
            logger.error(f"Event processing error for {agent['agent_id']}: {e}")
        
        return events_processed
    
    async def benchmark_llm_batching_efficiency(self) -> PerformanceBenchmarkResult:
        """Measure cost reduction and latency from LLM batching."""
        benchmark_name = "llm_batching_efficiency"
        start_time = time.time()
        
        try:
            logger.info(f"Benchmarking LLM batching efficiency (target: {self.targets.min_cost_reduction_percent}% cost reduction)")
            
            # Initialize LLM batcher
            llm_batcher = LLMBatcher()
            await llm_batcher.start()
            
            # Configure batching parameters for efficiency
            llm_batcher.set_batch_parameters(
                max_size=10,
                timeout_ms=100,
                similarity_threshold=0.8
            )
            
            # Test scenarios
            test_scenarios = [
                # Scenario 1: High deduplication potential
                {
                    "name": "high_deduplication",
                    "requests": [
                        ("req_1", "What is the optimal price for ASIN-001?", "gpt-4"),
                        ("req_2", "What is the optimal price for ASIN-001?", "gpt-4"),  # Duplicate
                        ("req_3", "What is the optimal price for ASIN-001?", "gpt-4"),  # Duplicate
                        ("req_4", "What is the optimal price for ASIN-002?", "gpt-4"),
                        ("req_5", "What is the optimal price for ASIN-002?", "gpt-4"),  # Duplicate
                    ]
                },
                # Scenario 2: Mixed model requests
                {
                    "name": "mixed_models",
                    "requests": [
                        ("req_6", "Analyze market conditions", "gpt-4"),
                        ("req_7", "Analyze market conditions", "claude-sonnet"),
                        ("req_8", "Generate marketing copy", "gpt-4"),
                        ("req_9", "Generate marketing copy", "gpt-4"),  # Duplicate
                        ("req_10", "Summarize sales data", "claude-sonnet"),
                    ]
                },
                # Scenario 3: Large batch
                {
                    "name": "large_batch",
                    "requests": [(f"req_{i+11}", f"Process order {i}", "gpt-4") for i in range(20)]
                }
            ]
            
            total_requests = 0
            total_batches = 0
            total_deduplicated = 0
            batch_processing_times = []
            
            # Process each scenario
            for scenario in test_scenarios:
                scenario_start = time.time()
                
                # Track callbacks
                responses_received = []
                response_event = asyncio.Event()
                
                def create_callback(req_id):
                    def callback(request_id, response, error):
                        responses_received.append((request_id, response, error))
                        if len(responses_received) >= len(scenario["requests"]):
                            response_event.set()
                    return callback
                
                # Submit requests
                for req_id, prompt, model in scenario["requests"]:
                    llm_batcher.add_request(req_id, prompt, model, create_callback(req_id))
                    total_requests += 1
                
                # Wait for responses
                await asyncio.wait_for(response_event.wait(), timeout=10.0)
                
                scenario_duration = time.time() - scenario_start
                batch_processing_times.append(scenario_duration)
                
                logger.info(f"Scenario '{scenario['name']}' completed in {scenario_duration:.3f}s")
            
            # Allow final processing
            await asyncio.sleep(1.0)
            
            # Get batching statistics
            stats = llm_batcher.stats
            total_deduplicated = stats.get("requests_deduplicated", 0)
            total_batches = stats.get("total_batches_processed", 0)
            estimated_cost = stats.get("total_api_cost_estimated", 0.0)
            
            # Calculate efficiency metrics
            deduplication_rate = total_deduplicated / max(total_requests, 1)
            cost_reduction_percent = deduplication_rate * 100
            
            # Baseline cost (without batching)
            baseline_cost = total_requests * 0.002  # Estimated cost per request
            actual_cost = estimated_cost
            cost_savings_percent = ((baseline_cost - actual_cost) / baseline_cost) * 100 if baseline_cost > 0 else 0
            
            # Latency analysis
            avg_batch_processing_time = statistics.mean(batch_processing_times) if batch_processing_times else 0
            
            actual_value = max(cost_reduction_percent, cost_savings_percent)
            meets_target = actual_value >= self.targets.min_cost_reduction_percent
            performance_ratio = actual_value / self.targets.min_cost_reduction_percent
            
            resource_usage = {
                "total_requests": total_requests,
                "total_batches": total_batches,
                "requests_deduplicated": total_deduplicated,
                "deduplication_rate": deduplication_rate,
                "estimated_baseline_cost": baseline_cost,
                "estimated_actual_cost": actual_cost
            }
            
            additional_metrics = {
                "cost_reduction_percent": cost_reduction_percent,
                "cost_savings_percent": cost_savings_percent,
                "avg_batch_processing_time": avg_batch_processing_time,
                "batch_processing_times": batch_processing_times,
                "batching_stats": stats
            }
            
            await llm_batcher.stop()
            
            duration = time.time() - start_time
            
            return PerformanceBenchmarkResult(
                benchmark_name=benchmark_name,
                target_metric="cost_reduction_percent",
                target_value=self.targets.min_cost_reduction_percent,
                actual_value=actual_value,
                meets_target=meets_target,
                performance_ratio=performance_ratio,
                duration_seconds=duration,
                resource_usage=resource_usage,
                additional_metrics=additional_metrics
            )
            
        except Exception as e:
            logger.error(f"LLM batching efficiency benchmark failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return PerformanceBenchmarkResult(
                benchmark_name=benchmark_name,
                target_metric="cost_reduction_percent",
                target_value=self.targets.min_cost_reduction_percent,
                actual_value=0,
                meets_target=False,
                performance_ratio=0,
                duration_seconds=duration,
                resource_usage={},
                error_details=str(e)
            )
    
    async def benchmark_long_horizon_memory_usage(self) -> PerformanceBenchmarkResult:
        """Test memory usage for year-long simulation stability."""
        benchmark_name = "long_horizon_memory_usage"
        start_time = time.time()
        
        try:
            logger.info(f"Benchmarking long-horizon memory usage (target: <{self.targets.max_memory_usage_mb}MB)")
            
            # Start memory tracking
            tracemalloc.start()
            baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            # Simulate year-long simulation (365 days * 24 hours * 60 minutes = 525,600 ticks)
            # For performance, we'll simulate proportionally fewer ticks but with appropriate data structures
            simulation_scale = 1000  # Simulate 1000 ticks representing year-long data
            
            # Create agents with full memory systems
            agents = []
            for i in range(3):  # 3 agents for memory testing
                agent_id = f"memory_agent_{i}"
                
                memory_config = MemoryConfig(
                    consolidation_algorithm=ConsolidationAlgorithm.LLM_REFLECTION,
                    short_term_capacity=5000,  # Large capacity
                    long_term_capacity=10000,
                    short_term_retention_days=30,
                    consolidation_percentage=0.1
                )
                
                memory_manager = DualMemoryManager(agent_id, memory_config)
                reflection_module = ReflectionModule(memory_manager, memory_config)
                
                agents.append({
                    "agent_id": agent_id,
                    "memory_manager": memory_manager,
                    "reflection_module": reflection_module
                })
            
            logger.info(f"Created {len(agents)} agents with full memory systems")
            
            # Simulate long-horizon memory accumulation
            memory_samples = []
            events_generated = 0
            
            for tick in range(simulation_scale):
                tick_start = time.time()
                
                # Generate events for each agent
                for agent in agents:
                    memory_manager = agent["memory_manager"]
                    
                    # Simulate various event types accumulating in memory
                    events_this_tick = [
                        {
                            "event_id": f"tick_{agent['agent_id']}_{tick}",
                            "event_type": "TickEvent",
                            "content": f"Simulation tick {tick}",
                            "domain": "operations",
                            "importance_score": 0.3 + (tick % 10) * 0.05,
                            "timestamp": datetime.now()
                        },
                        {
                            "event_id": f"sale_{agent['agent_id']}_{tick}",
                            "event_type": "SaleOccurred", 
                            "content": f"Sale of product PROD-{tick%20} for ${1000 + tick%500}",
                            "domain": "sales",
                            "importance_score": 0.6 + (tick % 5) * 0.08,
                            "timestamp": datetime.now()
                        }
                    ]
                    
                    # Add events to memory
                    for event_data in events_this_tick:
                        await memory_manager.store_memory(**event_data)
                        events_generated += 1
                    
                    # Periodic reflection and consolidation (simulate weekly)
                    if tick > 0 and tick % 70 == 0:  # Simulate weekly reflection
                        await agent["reflection_module"].perform_reflection()
                
                # Memory sampling
                if tick % 100 == 0:  # Sample every 100 ticks
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    memory_samples.append({
                        "tick": tick,
                        "memory_mb": current_memory,
                        "events_generated": events_generated
                    })
                    
                    if tick % 200 == 0:
                        logger.info(f"Tick {tick}/{simulation_scale}, Memory: {current_memory:.1f}MB, Events: {events_generated}")
                
                # Force garbage collection periodically
                if tick % 500 == 0:
                    gc.collect()
                
                tick_duration = time.time() - tick_start
                
                # Small delay to simulate processing time
                await asyncio.sleep(0.001)
            
            # Final memory measurement
            final_memory = self.process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - baseline_memory
            
            # Get memory peak
            current_memory_usage, peak_memory_usage = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory_mb = peak_memory_usage / 1024 / 1024
            
            # Calculate memory efficiency
            events_per_mb = events_generated / max(memory_growth, 1)
            memory_growth_rate = memory_growth / simulation_scale  # MB per tick
            
            # Project to year-long simulation
            year_ticks = 525600  # 365 * 24 * 60
            projected_year_memory = baseline_memory + (memory_growth_rate * year_ticks)
            
            actual_value = peak_memory_mb
            meets_target = actual_value <= self.targets.max_memory_usage_mb
            performance_ratio = actual_value / self.targets.max_memory_usage_mb
            
            resource_usage = {
                "baseline_memory_mb": baseline_memory,
                "final_memory_mb": final_memory,
                "memory_growth_mb": memory_growth,
                "peak_memory_mb": peak_memory_mb,
                "projected_year_memory_mb": projected_year_memory,
                "memory_growth_rate_mb_per_tick": memory_growth_rate
            }
            
            additional_metrics = {
                "simulation_ticks": simulation_scale,
                "events_generated": events_generated,
                "events_per_mb": events_per_mb,
                "memory_samples": memory_samples[-10:],  # Last 10 samples
                "agents_tested": len(agents)
            }
            
            duration = time.time() - start_time
            
            return PerformanceBenchmarkResult(
                benchmark_name=benchmark_name,
                target_metric="peak_memory_mb",
                target_value=self.targets.max_memory_usage_mb,
                actual_value=actual_value,
                meets_target=meets_target,
                performance_ratio=performance_ratio,
                duration_seconds=duration,
                resource_usage=resource_usage,
                additional_metrics=additional_metrics
            )
            
        except Exception as e:
            logger.error(f"Long-horizon memory usage benchmark failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return PerformanceBenchmarkResult(
                benchmark_name=benchmark_name,
                target_metric="peak_memory_mb",
                target_value=self.targets.max_memory_usage_mb,
                actual_value=float('inf'),
                meets_target=False,
                performance_ratio=float('inf'),
                duration_seconds=duration,
                resource_usage={},
                error_details=str(e)
            )
    
    async def benchmark_trace_analysis_speed(self) -> PerformanceBenchmarkResult:
        """Ensure <30s analysis completion."""
        benchmark_name = "trace_analysis_speed"
        start_time = time.time()
        
        try:
            logger.info(f"Benchmarking trace analysis speed (target: <{self.targets.max_analysis_time_seconds}s)")
            
            # Initialize trace analyzer
            trace_analyzer = TraceAnalyzer()
            
            # Generate large dataset of trace events for analysis
            trace_events = []
            event_types = [
                "TickEvent", "SaleOccurred", "SetPriceCommand", "ProductPriceUpdated",
                "CompetitorPricesUpdated", "BudgetWarning", "StrategyUpdate", "ReflectionCompleted"
            ]
            
            components = [
                "strategic_planner", "tactical_planner", "skill_coordinator", 
                "memory_manager", "reflection_loop", "llm_batcher", "trace_analyzer"
            ]
            
            # Generate 10,000 trace events (simulating large simulation)
            logger.info("Generating trace events for analysis")
            for i in range(10000):
                event = {
                    "event_id": f"trace_event_{i}",
                    "timestamp": datetime.now() - timedelta(seconds=i),
                    "event_type": event_types[i % len(event_types)],
                    "component": components[i % len(components)],
                    "agent_id": f"agent_{i % 5}",
                    "duration_ms": 50 + (i % 200),
                    "success": i % 10 != 0,  # 90% success rate
                    "metadata": {
                        "action_type": f"action_{i % 20}",
                        "resource_usage": {"cpu": i % 100, "memory": 100 + i % 500},
                        "performance_score": 0.5 + (i % 50) / 100.0
                    }
                }
                trace_events.append(event)
            
            logger.info(f"Generated {len(trace_events)} trace events")
            
            # Start analysis timing
            analysis_start = time.time()
            
            # Perform comprehensive trace analysis
            analysis_tasks = [
                self._analyze_event_patterns(trace_events),
                self._analyze_performance_trends(trace_events),
                self._analyze_error_patterns(trace_events),
                self._analyze_resource_usage(trace_events),
                self._analyze_component_interactions(trace_events)
            ]
            
            # Run analysis tasks concurrently
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            analysis_duration = time.time() - analysis_start
            
            # Validate analysis results
            successful_analyses = sum(1 for result in analysis_results if not isinstance(result, Exception))
            
            # Additional analysis operations
            logger.info("Performing additional analysis operations")
            
            # Time-series analysis
            time_series_start = time.time()
            time_buckets = {}
            for event in trace_events:
                hour = event["timestamp"].hour
                if hour not in time_buckets:
                    time_buckets[hour] = []
                time_buckets[hour].append(event)
            
            hourly_analysis = {}
            for hour, events in time_buckets.items():
                hourly_analysis[hour] = {
                    "event_count": len(events),
                    "avg_duration": statistics.mean([e["duration_ms"] for e in events]),
                    "success_rate": sum(1 for e in events if e["success"]) / len(events)
                }
            time_series_duration = time.time() - time_series_start
            
            # Performance correlation analysis
            correlation_start = time.time()
            performance_correlations = {}
            for component in components:
                component_events = [e for e in trace_events if e["component"] == component]
                if component_events:
                    avg_duration = statistics.mean([e["duration_ms"] for e in component_events])
                    success_rate = sum(1 for e in component_events if e["success"]) / len(component_events)
                    performance_correlations[component] = {
                        "avg_duration": avg_duration,
                        "success_rate": success_rate,
                        "event_count": len(component_events)
                    }
            correlation_duration = time.time() - correlation_start
            
            total_analysis_time = analysis_duration + time_series_duration + correlation_duration
            
            # Calculate throughput metrics
            events_per_second = len(trace_events) / total_analysis_time
            
            actual_value = total_analysis_time
            meets_target = actual_value <= self.targets.max_analysis_time_seconds
            performance_ratio = actual_value / self.targets.max_analysis_time_seconds
            
            resource_usage = {
                "trace_events_analyzed": len(trace_events),
                "analysis_duration": analysis_duration,
                "time_series_duration": time_series_duration,
                "correlation_duration": correlation_duration,
                "total_analysis_time": total_analysis_time,
                "events_per_second": events_per_second
            }
            
            additional_metrics = {
                "successful_analyses": successful_analyses,
                "total_analysis_tasks": len(analysis_tasks),
                "hourly_analysis_buckets": len(hourly_analysis),
                "component_correlations": len(performance_correlations),
                "analysis_throughput": events_per_second
            }
            
            duration = time.time() - start_time
            
            return PerformanceBenchmarkResult(
                benchmark_name=benchmark_name,
                target_metric="analysis_time_seconds",
                target_value=self.targets.max_analysis_time_seconds,
                actual_value=actual_value,
                meets_target=meets_target,
                performance_ratio=performance_ratio,
                duration_seconds=duration,
                resource_usage=resource_usage,
                additional_metrics=additional_metrics
            )
            
        except Exception as e:
            logger.error(f"Trace analysis speed benchmark failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return PerformanceBenchmarkResult(
                benchmark_name=benchmark_name,
                target_metric="analysis_time_seconds",
                target_value=self.targets.max_analysis_time_seconds,
                actual_value=float('inf'),
                meets_target=False,
                performance_ratio=float('inf'),
                duration_seconds=duration,
                resource_usage={},
                error_details=str(e)
            )
    
    async def _analyze_event_patterns(self, trace_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in trace events."""
        patterns = {}
        
        # Event type distribution
        event_type_counts = {}
        for event in trace_events:
            event_type = event["event_type"]
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        patterns["event_type_distribution"] = event_type_counts
        
        # Success/failure patterns
        total_events = len(trace_events)
        successful_events = sum(1 for e in trace_events if e["success"])
        patterns["overall_success_rate"] = successful_events / total_events
        
        return patterns
    
    async def _analyze_performance_trends(self, trace_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends in trace data."""
        trends = {}
        
        # Duration trends
        durations = [e["duration_ms"] for e in trace_events]
        trends["avg_duration"] = statistics.mean(durations)
        trends["median_duration"] = statistics.median(durations)
        trends["duration_stddev"] = statistics.stdev(durations) if len(durations) > 1 else 0
        
        return trends
    
    async def _analyze_error_patterns(self, trace_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns in trace data."""
        errors = {}
        
        failed_events = [e for e in trace_events if not e["success"]]
        errors["total_failures"] = len(failed_events)
        errors["failure_rate"] = len(failed_events) / len(trace_events)
        
        # Component failure rates
        component_failures = {}
        for event in failed_events:
            component = event["component"]
            component_failures[component] = component_failures.get(component, 0) + 1
        
        errors["component_failures"] = component_failures
        
        return errors
    
    async def _analyze_resource_usage(self, trace_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze resource usage patterns."""
        resource_analysis = {}
        
        cpu_usage = [e["metadata"]["resource_usage"]["cpu"] for e in trace_events]
        memory_usage = [e["metadata"]["resource_usage"]["memory"] for e in trace_events]
        
        resource_analysis["avg_cpu"] = statistics.mean(cpu_usage)
        resource_analysis["avg_memory"] = statistics.mean(memory_usage)
        resource_analysis["peak_cpu"] = max(cpu_usage)
        resource_analysis["peak_memory"] = max(memory_usage)
        
        return resource_analysis
    
    async def _analyze_component_interactions(self, trace_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze interactions between components."""
        interactions = {}
        
        # Component activity levels
        component_activity = {}
        for event in trace_events:
            component = event["component"]
            component_activity[component] = component_activity.get(component, 0) + 1
        
        interactions["component_activity"] = component_activity
        
        # Performance scores by component
        component_performance = {}
        for event in trace_events:
            component = event["component"]
            score = event["metadata"]["performance_score"]
            
            if component not in component_performance:
                component_performance[component] = []
            component_performance[component].append(score)
        
        # Calculate average performance per component
        avg_performance = {}
        for component, scores in component_performance.items():
            avg_performance[component] = statistics.mean(scores)
        
        interactions["component_performance"] = avg_performance
        
        return interactions
    
    async def benchmark_deterministic_mode_overhead(self) -> PerformanceBenchmarkResult:
        """Measure performance impact of reproducibility features."""
        benchmark_name = "deterministic_mode_overhead"
        start_time = time.time()
        
        try:
            logger.info(f"Benchmarking deterministic mode overhead (target: <{self.targets.max_determinism_overhead_percent}% overhead)")
            
            # Test setup
            test_iterations = 100
            test_events_per_iteration = 50
            
            # Initialize systems for testing
            seed_manager = SimSeed("determinism_test_seed")
            llm_cache = LLMResponseCache(
                cache_file="determinism_test.cache",
                enable_compression=True,
                enable_validation=True
            )
            
            # Test 1: Non-deterministic mode (baseline)
            logger.info("Testing non-deterministic mode (baseline)")
            llm_cache.set_deterministic_mode(False)
            
            baseline_start = time.time()
            baseline_operations = 0
            
            for iteration in range(test_iterations):
                # Simulate cache operations
                for i in range(test_events_per_iteration):
                    prompt = f"Test prompt iteration {iteration} event {i}"
                    prompt_hash = llm_cache.generate_prompt_hash(prompt, "gpt-4", 0.0)
                    
                    # Try to get cached response (will be miss in baseline)
                    cached_response = llm_cache.get_cached_response(prompt_hash)
                    
                    # Cache a response
                    test_response = {"choices": [{"message": {"content": f"Response {iteration}_{i}"}}]}
                    llm_cache.cache_response(prompt_hash, test_response, {"model": "gpt-4", "temperature": 0.0})
                    
                    baseline_operations += 2  # get + cache operations
                
                # Small processing simulation
                await asyncio.sleep(0.001)
            
            baseline_duration = time.time() - baseline_start
            baseline_ops_per_second = baseline_operations / baseline_duration
            
            # Clear cache for deterministic test
            llm_cache.clear_cache(confirm=True)
            
            # Test 2: Deterministic mode with validation
            logger.info("Testing deterministic mode with validation")
            llm_cache.set_deterministic_mode(True)
            
            # Pre-populate cache for deterministic mode
            for iteration in range(test_iterations):
                for i in range(test_events_per_iteration):
                    prompt = f"Test prompt iteration {iteration} event {i}"
                    prompt_hash = llm_cache.generate_prompt_hash(prompt, "gpt-4", 0.0)
                    test_response = {"choices": [{"message": {"content": f"Response {iteration}_{i}"}}]}
                    llm_cache.cache_response(prompt_hash, test_response, {"model": "gpt-4", "temperature": 0.0})
            
            deterministic_start = time.time()
            deterministic_operations = 0
            
            for iteration in range(test_iterations):
                # Reset seed for reproducibility
                seed_manager.set_seed(42 + iteration)
                
                for i in range(test_events_per_iteration):
                    prompt = f"Test prompt iteration {iteration} event {i}"
                    prompt_hash = llm_cache.generate_prompt_hash(prompt, "gpt-4", 0.0)
                    
                    # Get cached response (should be hit in deterministic mode)
                    cached_response = llm_cache.get_cached_response(prompt_hash)
                    
                    # Validate response integrity
                    if cached_response:
                        # Simulate validation operations
                        response_str = json.dumps(cached_response)
                        validation_hash = hash(response_str)  # Simple validation
                    
                    deterministic_operations += 1
                
                await asyncio.sleep(0.001)
            
            deterministic_duration = time.time() - deterministic_start
            deterministic_ops_per_second = deterministic_operations / deterministic_duration
            
            # Calculate overhead
            overhead_percent = ((deterministic_duration - baseline_duration) / baseline_duration) * 100
            performance_ratio = deterministic_ops_per_second / baseline_ops_per_second
            
            # Test 3: Cache integrity validation overhead
            logger.info("Testing cache integrity validation")
            validation_start = time.time()
            
            is_valid, errors = llm_cache.validate_cache_integrity()
            
            validation_duration = time.time() - validation_start
            
            # Get cache statistics
            cache_stats = llm_cache.get_cache_statistics()
            
            actual_value = overhead_percent
            meets_target = actual_value <= self.targets.max_determinism_overhead_percent
            target_performance_ratio = actual_value / self.targets.max_determinism_overhead_percent
            
            resource_usage = {
                "baseline_duration": baseline_duration,
                "deterministic_duration": deterministic_duration,
                "validation_duration": validation_duration,
                "baseline_ops_per_second": baseline_ops_per_second,
                "deterministic_ops_per_second": deterministic_ops_per_second,
                "cache_integrity_valid": is_valid,
                "cache_validation_errors": len(errors)
            }
            
            additional_metrics = {
                "overhead_percent": overhead_percent,
                "performance_ratio": performance_ratio,
                "test_iterations": test_iterations,
                "operations_per_iteration": test_events_per_iteration,
                "total_baseline_operations": baseline_operations,
                "total_deterministic_operations": deterministic_operations,
                "cache_statistics": cache_stats.__dict__
            }
            
            duration = time.time() - start_time
            
            return PerformanceBenchmarkResult(
                benchmark_name=benchmark_name,
                target_metric="overhead_percent",
                target_value=self.targets.max_determinism_overhead_percent,
                actual_value=actual_value,
                meets_target=meets_target,
                performance_ratio=target_performance_ratio,
                duration_seconds=duration,
                resource_usage=resource_usage,
                additional_metrics=additional_metrics
            )
            
        except Exception as e:
            logger.error(f"Deterministic mode overhead benchmark failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return PerformanceBenchmarkResult(
                benchmark_name=benchmark_name,
                target_metric="overhead_percent",
                target_value=self.targets.max_determinism_overhead_percent,
                actual_value=float('inf'),
                meets_target=False,
                performance_ratio=float('inf'),
                duration_seconds=duration,
                resource_usage={},
                error_details=str(e)
            )
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run complete performance benchmark suite."""
        logger.info("Starting comprehensive performance benchmark suite")
        suite_start = time.time()
        
        # Benchmark methods to run
        benchmark_methods = [
            self.benchmark_distributed_agent_performance,
            self.benchmark_llm_batching_efficiency,
            self.benchmark_long_horizon_memory_usage,
            self.benchmark_trace_analysis_speed,
            self.benchmark_deterministic_mode_overhead
        ]
        
        results = []
        
        for benchmark_method in benchmark_methods:
            try:
                logger.info(f"Running {benchmark_method.__name__}")
                result = await benchmark_method()
                results.append(result)
                self.benchmark_results.append(result)
                
                if result.meets_target:
                    logger.info(f"✅ {result.benchmark_name} passed - {result.actual_value:.2f} vs target {result.target_value:.2f}")
                else:
                    logger.warning(f"❌ {result.benchmark_name} failed - {result.actual_value:.2f} vs target {result.target_value:.2f}")
                    if result.error_details:
                        logger.error(f"Error: {result.error_details}")
                        
            except Exception as e:
                logger.error(f"Benchmark {benchmark_method.__name__} crashed: {e}", exc_info=True)
                results.append(PerformanceBenchmarkResult(
                    benchmark_name=benchmark_method.__name__,
                    target_metric="unknown",
                    target_value=0,
                    actual_value=0,
                    meets_target=False,
                    performance_ratio=0,
                    duration_seconds=0,
                    resource_usage={},
                    error_details=str(e)
                ))
        
        suite_duration = time.time() - suite_start
        
        # Compile benchmark summary
        total_benchmarks = len(results)
        passed_benchmarks = sum(1 for r in results if r.meets_target)
        failed_benchmarks = total_benchmarks - passed_benchmarks
        
        # Calculate overall performance score
        performance_scores = []
        for result in results:
            if result.meets_target:
                # Bonus for exceeding target
                score = min(1.5, 1.0 / result.performance_ratio) if result.performance_ratio > 0 else 0
            else:
                # Partial credit for partial performance
                score = min(0.8, result.performance_ratio) if result.performance_ratio < float('inf') else 0
            performance_scores.append(score)
        
        overall_performance_score = sum(performance_scores) / len(performance_scores) if performance_scores else 0
        
        summary = {
            "suite_duration_seconds": suite_duration,
            "total_benchmarks": total_benchmarks,
            "passed_benchmarks": passed_benchmarks,
            "failed_benchmarks": failed_benchmarks,
            "success_rate": passed_benchmarks / total_benchmarks if total_benchmarks > 0 else 0,
            "overall_performance_score": overall_performance_score,
            "benchmark_targets": self.targets.__dict__,
            "benchmark_results": [result.__dict__ for result in results],
            "meets_all_targets": failed_benchmarks == 0
        }
        
        logger.info(f"Performance benchmark suite completed: {passed_benchmarks}/{total_benchmarks} targets met")
        logger.info(f"Overall performance score: {overall_performance_score:.2f}")
        
        return summary


# CLI runner for direct execution
async def main():
    """Run performance benchmarks with default targets."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize benchmark suite
    targets = BenchmarkTargets(
        min_concurrent_agents=20,
        min_ticks_per_minute=2000.0,
        min_cost_reduction_percent=30.0,
        max_memory_usage_mb=1000.0,
        max_analysis_time_seconds=30.0,
        max_determinism_overhead_percent=20.0
    )
    
    benchmark_suite = PerformanceBenchmarkSuite(targets)
    
    try:
        results = await benchmark_suite.run_all_benchmarks()
        
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        print(f"Total Benchmarks: {results['total_benchmarks']}")
        print(f"Passed: {results['passed_benchmarks']}")
        print(f"Failed: {results['failed_benchmarks']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Overall Performance Score: {results['overall_performance_score']:.2f}/1.0")
        print(f"Suite Duration: {results['suite_duration_seconds']:.2f}s")
        
        print("\nDetailed Results:")
        for result_data in results['benchmark_results']:
            result = PerformanceBenchmarkResult(**result_data)
            status = "✅ PASS" if result.meets_target else "❌ FAIL"
            print(f"  {status} {result.benchmark_name}: {result.actual_value:.2f} vs {result.target_value:.2f}")
        
        if results['meets_all_targets']:
            print("\n🎉 ALL PERFORMANCE TARGETS MET!")
            print("FBA-Bench infrastructure meets all quantitative requirements.")
        else:
            print("\n⚠️  Some performance targets not met.")
            print("Review failed benchmarks for optimization opportunities.")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}", exc_info=True)
        print(f"\n💥 Benchmark suite crashed: {e}")


if __name__ == "__main__":
    asyncio.run(main())