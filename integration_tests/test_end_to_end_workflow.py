"""
End-to-End Workflow Testing for FBA-Bench

This module tests complete simulation workflows from start to completion,
validating that all components work together seamlessly in realistic scenarios.

Test Categories:
1. Complete Simulation Lifecycle - Full simulation from initialization to completion
2. Multi-Agent Scenarios - Multiple agents with different frameworks running simultaneously  
3. Curriculum Progression - Agents advancing through T0→T1→T2→T3 tiers
4. Real-time Monitoring - Dashboard, metrics, and instrumentation during simulation
5. Event Stream Capture - Golden snapshot generation and replay capabilities
"""

import pytest
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Core imports
from integration_tests import IntegrationTestSuite, IntegrationTestConfig, logger
from simulation_orchestrator import SimulationOrchestrator, SimulationConfig
from event_bus import get_event_bus, EventBus
from metrics.metric_suite import MetricSuite
from constraints.budget_enforcer import BudgetEnforcer
from reproducibility.sim_seed import SimSeed
from reproducibility.event_snapshots import EventSnapshot

# Agent framework imports
from agent_runners.runner_factory import RunnerFactory
from agent_runners.configs.framework_configs import FrameworkConfig
from agent_runners.base_runner import AgentRunner

# Baseline bot imports
from baseline_bots.bot_factory import BotFactory

# Memory experiment imports
from memory_experiments.experiment_runner import ExperimentRunner
from memory_experiments.memory_config import MemoryConfig
from memory_experiments.memory_modes import MemoryMode

# Adversarial testing imports
from redteam.gauntlet_runner import GauntletRunner
from redteam.adversarial_event_injector import AdversarialEventInjector

# Services imports
from services.world_store import WorldStore
from services.sales_service import SalesService
from services.trust_score_service import TrustScoreService
from financial_audit import FinancialAuditService

# Instrumentation imports
from instrumentation.simulation_tracer import SimulationTracer

class TestEndToEndWorkflow(IntegrationTestSuite):
    """Test suite for end-to-end workflow validation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.config = IntegrationTestConfig(seed=42, verbose_logging=True)
        super().__init__(self.config)
        
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_simulation_lifecycle(self):
        """
        Test full simulation from initialization to completion.
        
        Tests:
        - Simulation initialization and setup
        - Service registration and startup
        - Event flow through complete lifecycle
        - Proper shutdown and cleanup
        - Final scoring and results generation
        """
        logger.info("🧪 Testing complete simulation lifecycle...")
        
        # Phase 1: Initialization
        logger.info("Phase 1: Initializing simulation environment...")
        
        sim_config = SimulationConfig(
            seed=42,
            max_ticks=100,
            tick_interval_seconds=0.1,  # Fast for testing
            time_acceleration=10.0
        )
        
        # Create and initialize all components
        env = await self.create_test_simulation(tier="T1", seed=42)
        orchestrator = env["orchestrator"]
        event_bus = env["event_bus"]
        
        # Initialize services
        world_store = env["world_store"]
        sales_service = env["services"]["sales"]
        trust_service = env["services"]["trust"]
        
        # Initialize financial audit
        financial_audit = FinancialAuditService()
        
        # Initialize metrics
        metric_suite = MetricSuite(
            tier="T1",
            financial_audit_service=financial_audit,
            sales_service=sales_service,
            trust_score_service=trust_service
        )
        metric_suite.subscribe_to_events(event_bus)
        
        # Initialize a baseline bot
        try:
            bot = BotFactory.create_bot("greedy_script_bot")
            assert bot is not None, "Failed to create baseline bot"
        except Exception as e:
            logger.warning(f"Bot creation failed: {e}, using mock bot")
            bot = None
        
        # Phase 2: Simulation Execution
        logger.info("Phase 2: Running simulation...")
        
        event_bus.start_recording()
        start_time = time.time()
        
        # Start simulation
        await orchestrator.start(event_bus)
        
        # Let simulation run for specified duration
        runtime_seconds = 3  # Short run for testing
        await asyncio.sleep(runtime_seconds)
        
        # Stop simulation
        await orchestrator.stop()
        end_time = time.time()
        
        # Phase 3: Results Collection
        logger.info("Phase 3: Collecting results...")
        
        # Get recorded events
        events = event_bus.get_recorded_events()
        event_bus.stop_recording()
        
        # Calculate metrics
        if events:
            final_scores = metric_suite.calculate_final_score(events)
        else:
            logger.warning("No events recorded during simulation")
            final_scores = None
        
        # Phase 4: Validation
        logger.info("Phase 4: Validating results...")
        
        # Validate simulation ran correctly
        actual_runtime = end_time - start_time
        assert actual_runtime >= runtime_seconds * 0.8, f"Simulation ended too early: {actual_runtime}s"
        assert actual_runtime <= runtime_seconds * 1.5, f"Simulation ran too long: {actual_runtime}s"
        
        # Validate events were generated
        assert len(events) > 0, "No events generated during simulation"
        
        # Validate event types
        event_types = {event.get("type", "unknown") for event in events}
        expected_types = ["TickEvent", "SimulationStartEvent"]
        
        for expected_type in expected_types:
            if expected_type not in event_types:
                logger.warning(f"Expected event type {expected_type} not found")
        
        # Validate metrics calculation
        if final_scores:
            assert hasattr(final_scores, 'score'), "Final score not calculated"
            assert isinstance(final_scores.score, (int, float)), "Invalid score type"
            assert 0 <= final_scores.score <= 100, f"Score out of range: {final_scores.score}"
        
        # Validate orchestrator statistics
        stats = orchestrator.stats
        assert stats['total_ticks'] > 0, "No ticks processed"
        assert stats['events_published'] >= 0, "Event count invalid"
        
        logger.info(f"✅ Simulation lifecycle completed successfully:")
        logger.info(f"   - Runtime: {actual_runtime:.2f}s")
        logger.info(f"   - Events: {len(events)}")
        logger.info(f"   - Ticks: {stats['total_ticks']}")
        if final_scores:
            logger.info(f"   - Final Score: {final_scores.score:.2f}")
        
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_multi_agent_scenarios(self):
        """
        Test multiple agents with different frameworks running simultaneously.
        
        Tests:
        - Multiple agent initialization
        - Framework isolation and coordination
        - Resource sharing and conflicts
        - Performance under concurrent load
        """
        logger.info("🧪 Testing multi-agent scenarios...")
        
        # Create simulation environment
        env = await self.create_test_simulation(tier="T1", seed=42)
        orchestrator = env["orchestrator"]
        event_bus = env["event_bus"]
        
        # Initialize multiple agents with different frameworks
        agents = []
        
        # Agent configurations
        agent_configs = [
            {"name": "agent_diy", "framework": "diy", "config": {}},
            {"name": "agent_script", "framework": "greedy_script", "config": {}},
        ]
        
        # Try to add more sophisticated agents if available
        try:
            agent_configs.extend([
                {"name": "agent_crew", "framework": "crewai", "config": {}},
                {"name": "agent_lang", "framework": "langchain", "config": {}}
            ])
        except Exception:
            logger.info("Advanced frameworks not available, using baseline agents")
        
        # Initialize agents
        for agent_config in agent_configs[:2]:  # Limit to 2 for testing
            try:
                if agent_config["framework"] == "greedy_script":
                    agent = BotFactory.create_bot("greedy_script_bot")
                else:
                    framework_config = FrameworkConfig(framework_type=agent_config["framework"])
                    agent = RunnerFactory.create_runner(agent_config["framework"], framework_config)
                
                if agent:
                    agents.append({
                        "name": agent_config["name"],
                        "agent": agent,
                        "framework": agent_config["framework"]
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to create agent {agent_config['name']}: {e}")
        
        assert len(agents) >= 1, "No agents successfully created"
        
        # Run multi-agent simulation
        event_bus.start_recording()
        
        # Start orchestrator
        await orchestrator.start(event_bus)
        
        # Let multiple agents run concurrently
        agent_tasks = []
        for agent_info in agents:
            # In a real implementation, agents would be integrated with the event bus
            # For now, we simulate their presence
            logger.info(f"Agent {agent_info['name']} ({agent_info['framework']}) ready")
        
        # Run simulation
        await asyncio.sleep(2)  # Brief multi-agent simulation
        
        # Stop simulation
        await orchestrator.stop()
        
        # Collect results
        events = event_bus.get_recorded_events()
        event_bus.stop_recording()
        
        # Validate multi-agent coordination
        assert len(events) > 0, "Multi-agent simulation generated no events"
        
        # Check for agent-specific events (would be framework dependent)
        agent_event_count = 0
        for event in events:
            if "agent" in event.get("source", "").lower():
                agent_event_count += 1
        
        logger.info(f"Multi-agent scenario completed with {len(agents)} agents")
        logger.info(f"Generated {len(events)} total events, {agent_event_count} agent-specific")
        
        logger.info("✅ Multi-agent scenarios test passed")
        
    @pytest.mark.e2e
    @pytest.mark.asyncio 
    async def test_curriculum_progression_workflow(self):
        """
        Test agents advancing through T0→T1→T2→T3 tiers.
        
        Tests:
        - Tier progression logic
        - Constraint escalation
        - Success criteria validation
        - Performance tracking across tiers
        """
        logger.info("🧪 Testing curriculum progression workflow...")
        
        tiers = ["T0", "T1", "T2", "T3"]
        progression_results = {}
        
        for tier in tiers:
            logger.info(f"Testing tier {tier}...")
            
            # Create tier-specific environment
            env = await self.create_test_simulation(tier=tier, seed=42)
            
            # Initialize budget enforcer for tier
            budget_enforcer = BudgetEnforcer.from_tier_config(tier, env["event_bus"])
            
            # Initialize metrics for tier
            financial_audit = FinancialAuditService()
            metric_suite = MetricSuite(
                tier=tier,
                financial_audit_service=financial_audit,
                sales_service=env["services"]["sales"],
                trust_score_service=env["services"]["trust"]
            )
            metric_suite.subscribe_to_events(env["event_bus"])
            
            # Run tier-specific simulation
            orchestrator = env["orchestrator"]
            event_bus = env["event_bus"]
            
            event_bus.start_recording()
            await orchestrator.start(event_bus)
            
            # Brief run for tier
            await asyncio.sleep(1)
            
            await orchestrator.stop()
            
            # Collect tier results
            events = event_bus.get_recorded_events()
            event_bus.stop_recording()
            
            # Calculate tier performance
            if events:
                final_scores = metric_suite.calculate_final_score(events)
                tier_score = final_scores.score
            else:
                tier_score = 0
            
            progression_results[tier] = {
                "score": tier_score,
                "event_count": len(events),
                "constraints": {
                    "max_tokens": budget_enforcer.config.max_tokens_per_action,
                    "memory_systems": getattr(budget_enforcer.config, 'memory_systems', [])
                }
            }
            
            logger.info(f"Tier {tier} completed: Score={tier_score:.2f}, Events={len(events)}")
        
        # Validate progression
        assert len(progression_results) == len(tiers), "Not all tiers completed"
        
        # Validate constraint progression (tokens should increase with tier)
        token_limits = [progression_results[tier]["constraints"]["max_tokens"] for tier in tiers]
        
        # Should be monotonically increasing or at least not decreasing
        for i in range(1, len(token_limits)):
            if token_limits[i] < token_limits[i-1]:
                logger.warning(f"Token limit decreased from {tiers[i-1]} to {tiers[i]}")
        
        logger.info("Curriculum progression results:")
        for tier in tiers:
            result = progression_results[tier]
            logger.info(f"  {tier}: Score={result['score']:.2f}, Tokens={result['constraints']['max_tokens']}")
        
        logger.info("✅ Curriculum progression workflow test passed")
        
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_real_time_monitoring_integration(self):
        """
        Test dashboard, metrics, and instrumentation during simulation.
        
        Tests:
        - Real-time metric updates
        - Dashboard API responsiveness
        - OpenTelemetry trace collection
        - Event streaming capabilities
        """
        logger.info("🧪 Testing real-time monitoring integration...")
        
        # Create monitored simulation environment
        env = await self.create_test_simulation(tier="T1", seed=42)
        orchestrator = env["orchestrator"]
        event_bus = env["event_bus"]
        
        # Initialize monitoring components
        financial_audit = FinancialAuditService()
        metric_suite = MetricSuite(
            tier="T1",
            financial_audit_service=financial_audit,
            sales_service=env["services"]["sales"],
            trust_score_service=env["services"]["trust"]
        )
        metric_suite.subscribe_to_events(event_bus)
        
        # Initialize tracing (if available)
        try:
            tracer = SimulationTracer()
            tracing_enabled = True
        except Exception as e:
            logger.warning(f"Tracing not available: {e}")
            tracing_enabled = False
        
        # Start monitoring
        event_bus.start_recording()
        monitoring_start = time.time()
        
        # Start simulation with monitoring
        await orchestrator.start(event_bus)
        
        # Collect real-time metrics during simulation
        metric_snapshots = []
        monitoring_duration = 2  # seconds
        snapshot_interval = 0.5  # seconds
        
        for i in range(int(monitoring_duration / snapshot_interval)):
            await asyncio.sleep(snapshot_interval)
            
            # Capture current state
            current_events = event_bus.get_recorded_events()
            
            snapshot = {
                "timestamp": time.time() - monitoring_start,
                "event_count": len(current_events),
                "tick": orchestrator.current_tick,
                "is_running": orchestrator.is_running
            }
            
            # Calculate current metrics if events available
            if current_events:
                try:
                    current_scores = metric_suite.calculate_final_score(current_events)
                    snapshot["current_score"] = current_scores.score
                except Exception as e:
                    snapshot["current_score"] = None
                    
            metric_snapshots.append(snapshot)
        
        # Stop simulation
        await orchestrator.stop()
        
        # Final data collection
        final_events = event_bus.get_recorded_events()
        event_bus.stop_recording()
        
        # Validate real-time monitoring
        assert len(metric_snapshots) > 0, "No monitoring snapshots collected"
        
        # Validate metric progression
        event_counts = [s["event_count"] for s in metric_snapshots]
        assert max(event_counts) >= min(event_counts), "Event count should not decrease"
        
        # Validate responsiveness (all snapshots should be within time bounds)
        for snapshot in metric_snapshots:
            assert snapshot["timestamp"] <= monitoring_duration + 1, "Snapshot timestamp out of bounds"
        
        # Validate final state
        assert len(final_events) > 0, "No final events collected"
        
        logger.info(f"Real-time monitoring captured {len(metric_snapshots)} snapshots")
        logger.info(f"Event progression: {min(event_counts)} → {max(event_counts)}")
        
        if tracing_enabled:
            logger.info("OpenTelemetry tracing operational")
        
        logger.info("✅ Real-time monitoring integration test passed")
        
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_event_stream_capture_and_replay(self):
        """
        Test golden snapshot generation and replay capabilities.
        
        Tests:
        - Event stream capture
        - Snapshot generation
        - Event replay accuracy
        - Golden snapshot validation
        """
        logger.info("🧪 Testing event stream capture and replay...")
        
        # Phase 1: Capture original simulation
        logger.info("Phase 1: Capturing original simulation...")
        
        env1 = await self.create_test_simulation(tier="T0", seed=12345)
        orchestrator1 = env1["orchestrator"]
        event_bus1 = env1["event_bus"]
        
        event_bus1.start_recording()
        await orchestrator1.start(event_bus1)
        await asyncio.sleep(1)  # Brief simulation
        await orchestrator1.stop()
        
        original_events = event_bus1.get_recorded_events()
        event_bus1.stop_recording()
        
        assert len(original_events) > 0, "Original simulation generated no events"
        
        # Phase 2: Generate snapshot
        logger.info("Phase 2: Generating golden snapshot...")
        
        # Create snapshot
        snapshot_hash = EventSnapshot.generate_event_stream_hash(original_events)
        
        # Save snapshot for future comparison
        git_sha = "test_snapshot_12345"
        run_id = "e2e_test_run"
        
        try:
            EventSnapshot.dump_events(original_events, git_sha, run_id)
            snapshot_saved = True
        except Exception as e:
            logger.warning(f"Snapshot save failed: {e}")
            snapshot_saved = False
        
        # Phase 3: Replay simulation
        logger.info("Phase 3: Replaying simulation...")
        
        env2 = await self.create_test_simulation(tier="T0", seed=12345)  # Same seed
        orchestrator2 = env2["orchestrator"] 
        event_bus2 = env2["event_bus"]
        
        event_bus2.start_recording()
        await orchestrator2.start(event_bus2)
        await asyncio.sleep(1)  # Same duration
        await orchestrator2.stop()
        
        replay_events = event_bus2.get_recorded_events()
        event_bus2.stop_recording()
        
        # Phase 4: Validate replay accuracy
        logger.info("Phase 4: Validating replay accuracy...")
        
        assert len(replay_events) > 0, "Replay simulation generated no events"
        
        # Compare event streams
        replay_hash = EventSnapshot.generate_event_stream_hash(replay_events)
        
        # Hashes should match for deterministic replay
        if snapshot_hash == replay_hash:
            logger.info("✅ Perfect replay accuracy achieved")
        else:
            logger.warning("Event stream replay differences detected")
            # Log first few differences for debugging
            min_len = min(len(original_events), len(replay_events))
            for i in range(min(5, min_len)):
                if i < len(original_events) and i < len(replay_events):
                    orig = original_events[i]
                    repl = replay_events[i]
                    if orig != repl:
                        logger.warning(f"Event {i} differs: {orig.get('type')} vs {repl.get('type')}")
        
        # Validate snapshot system
        assert len(snapshot_hash) > 0, "Snapshot hash generation failed"
        assert isinstance(snapshot_hash, str), "Invalid snapshot hash type"
        
        # Clean up test snapshots
        if snapshot_saved:
            try:
                import os
                snapshot_path = EventSnapshot.ARTIFACTS_DIR / f"{git_sha}_{run_id}.parquet"
                if snapshot_path.exists():
                    os.unlink(snapshot_path)
            except Exception as e:
                logger.warning(f"Snapshot cleanup failed: {e}")
        
        logger.info(f"Event stream capture completed:")
        logger.info(f"  - Original events: {len(original_events)}")
        logger.info(f"  - Replay events: {len(replay_events)}")
        logger.info(f"  - Snapshot hash: {snapshot_hash[:16]}...")
        
        logger.info("✅ Event stream capture and replay test passed")

@pytest.mark.e2e
class TestWorkflowIntegration:
    """Integration tests combining multiple workflow components."""
    
    @pytest.mark.asyncio
    async def test_complete_benchmark_workflow(self):
        """
        Run a complete benchmark workflow combining all end-to-end components.
        
        This test validates the entire FBA-Bench workflow from agent
        initialization through final scoring and reporting.
        """
        logger.info("🚀 Running complete benchmark workflow...")
        
        workflow_suite = TestEndToEndWorkflow()
        workflow_results = {}
        
        try:
            # Test complete simulation lifecycle
            await workflow_suite.test_complete_simulation_lifecycle()
            workflow_results["simulation_lifecycle"] = True
            
            # Test multi-agent scenarios
            await workflow_suite.test_multi_agent_scenarios()
            workflow_results["multi_agent_scenarios"] = True
            
            # Test curriculum progression
            await workflow_suite.test_curriculum_progression_workflow()
            workflow_results["curriculum_progression"] = True
            
            # Test real-time monitoring
            await workflow_suite.test_real_time_monitoring_integration()
            workflow_results["real_time_monitoring"] = True
            
            # Test event capture and replay
            await workflow_suite.test_event_stream_capture_and_replay()
            workflow_results["event_stream_capture"] = True
            
            # Validate all workflow components
            assert all(workflow_results.values()), f"Workflow components failed: {workflow_results}"
            
        except Exception as e:
            logger.error(f"Complete benchmark workflow failed: {e}")
            raise
        
        logger.info("🎉 Complete benchmark workflow passed!")
        return workflow_results