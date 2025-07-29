"""
Scientific Reproducibility Testing for FBA-Bench

This module validates the scientific reproducibility requirements that are
essential for FBA-Bench to serve as a credible tier-1 benchmark for the
research community.

Reproducibility Requirements:
- Identical Results: Same seed+config produces identical outputs across runs
- Golden Snapshots: Event streams match exactly for regression testing
- Statistical Consistency: Multiple runs with different seeds show expected variance
- Configuration Sensitivity: Changes in config properly affect results
- Platform Independence: Results consistent across different environments

Test Categories:
1. Deterministic Reproducibility - Bit-perfect identical results
2. Golden Snapshot Validation - Regression testing with reference data
3. Statistical Consistency - Variance analysis across multiple runs
4. Configuration Sensitivity - Parameter change impact validation
5. Cross-Platform Validation - Environment independence verification
"""

import pytest
import asyncio
import logging
import hashlib
import statistics
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

# Core imports
from integration_tests import IntegrationTestSuite, IntegrationTestConfig, logger
from simulation_orchestrator import SimulationOrchestrator, SimulationConfig
from event_bus import get_event_bus, EventBus
from metrics.metric_suite import MetricSuite
from constraints.budget_enforcer import BudgetEnforcer
from reproducibility.sim_seed import SimSeed
from reproducibility.event_snapshots import EventSnapshot
from reproducibility.ci_integration import CIIntegration

# Services imports
from services.world_store import WorldStore
from services.sales_service import SalesService
from services.trust_score_service import TrustScoreService
from financial_audit import FinancialAuditService

class ReproducibilityValidator:
    """Helper class for reproducibility validation."""
    
    def __init__(self):
        self.test_results = {}
        self.snapshots = {}
        
    def calculate_determinism_score(self, run_results: List[Dict[str, Any]]) -> float:
        """
        Calculate determinism score based on run consistency.
        
        Returns:
            Score from 0.0 to 1.0 where 1.0 is perfect determinism
        """
        if len(run_results) < 2:
            return 1.0  # Single run is deterministic by definition
        
        # Compare key metrics across runs
        metrics_to_compare = ["final_score", "event_count", "final_hash"]
        
        determinism_scores = []
        
        for metric in metrics_to_compare:
            values = [run.get(metric) for run in run_results if metric in run]
            
            if len(values) < 2:
                continue
                
            # For exact matches (hashes, counts)
            if metric in ["final_hash", "event_count"]:
                unique_values = len(set(str(v) for v in values))
                score = 1.0 if unique_values == 1 else 0.0
            else:
                # For numerical values, check variance
                if all(isinstance(v, (int, float)) for v in values):
                    variance = statistics.variance(values) if len(values) > 1 else 0
                    # Perfect determinism = zero variance
                    score = 1.0 if variance == 0 else max(0.0, 1.0 - min(variance, 1.0))
                else:
                    # String comparison
                    unique_values = len(set(str(v) for v in values))
                    score = 1.0 if unique_values == 1 else 0.0
            
            determinism_scores.append(score)
        
        return statistics.mean(determinism_scores) if determinism_scores else 0.0
    
    def validate_statistical_consistency(self, results: List[float]) -> Dict[str, Any]:
        """
        Validate statistical consistency across multiple runs.
        
        Returns:
            Statistics about the consistency of results
        """
        if len(results) < 3:
            return {"error": "Need at least 3 runs for statistical analysis"}
        
        mean_score = statistics.mean(results)
        std_dev = statistics.stdev(results)
        variance = statistics.variance(results)
        min_score = min(results)
        max_score = max(results)
        
        # Calculate coefficient of variation (relative variability)
        cv = std_dev / mean_score if mean_score != 0 else float('inf')
        
        # Statistical consistency score (lower variance = higher consistency)
        # But some variance is expected with different seeds
        expected_cv = 0.1  # 10% coefficient of variation is reasonable
        consistency_score = max(0.0, 1.0 - abs(cv - expected_cv) / expected_cv)
        
        return {
            "mean": mean_score,
            "std_dev": std_dev,
            "variance": variance,
            "min": min_score,
            "max": max_score,
            "coefficient_of_variation": cv,
            "consistency_score": consistency_score,
            "sample_size": len(results)
        }

class TestScientificReproducibility(IntegrationTestSuite):
    """Test suite for scientific reproducibility validation."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.config = IntegrationTestConfig(seed=42, verbose_logging=True)
        super().__init__(self.config)
        self.validator = ReproducibilityValidator()
        
    @pytest.mark.reproducibility
    @pytest.mark.asyncio
    async def test_deterministic_identical_results(self):
        """
        Test that same seed+config produces identical outputs across runs.
        
        Tests:
        - Multiple runs with identical configuration
        - Bit-perfect result matching
        - Event stream hash consistency
        - State hash validation
        """
        logger.info("🧪 Testing deterministic identical results...")
        
        # Test configuration
        test_seed = 12345
        num_runs = 3
        max_ticks = 50
        
        run_results = []
        
        for run_num in range(num_runs):
            logger.info(f"Running deterministic test {run_num + 1}/{num_runs}...")
            
            # Create identical simulation configuration
            sim_config = SimulationConfig(
                seed=test_seed,
                max_ticks=max_ticks,
                tick_interval_seconds=0.01,
                time_acceleration=100.0
            )
            
            # Create test environment
            env = await self.create_test_simulation(tier="T0", seed=test_seed)
            
            # Initialize services with identical configuration
            financial_audit = FinancialAuditService()
            metric_suite = MetricSuite(
                tier="T0",
                financial_audit_service=financial_audit,
                sales_service=env["services"]["sales"],
                trust_score_service=env["services"]["trust"]
            )
            metric_suite.subscribe_to_events(env["event_bus"])
            
            # Run simulation
            orchestrator = env["orchestrator"]
            event_bus = env["event_bus"]
            
            event_bus.start_recording()
            await orchestrator.start(event_bus)
            
            # Run for fixed duration
            await asyncio.sleep(1)  # 1 second test run
            
            await orchestrator.stop()
            
            # Collect results
            events = event_bus.get_recorded_events()
            event_bus.stop_recording()
            
            # Calculate final metrics
            if events:
                final_scores = metric_suite.calculate_final_score(events)
                final_score = final_scores.score
            else:
                final_score = 0
            
            # Generate event stream hash
            event_hash = EventSnapshot.generate_event_stream_hash(events)
            
            # Generate configuration hash
            config_data = {
                "seed": test_seed,
                "max_ticks": max_ticks,
                "tier": "T0"
            }
            config_hash = hashlib.sha256(str(config_data).encode()).hexdigest()
            
            run_result = {
                "run_number": run_num + 1,
                "final_score": final_score,
                "event_count": len(events),
                "event_hash": event_hash,
                "config_hash": config_hash,
                "final_hash": f"{final_score}_{len(events)}_{event_hash[:8]}",
                "tick_count": orchestrator.current_tick
            }
            
            run_results.append(run_result)
            
            logger.info(f"  Run {run_num + 1}: Score={final_score:.2f}, Events={len(events)}, Hash={event_hash[:16]}...")
        
        # Validate deterministic behavior
        assert len(run_results) == num_runs, "Not all runs completed"
        
        # Check identical results across runs
        reference_run = run_results[0]
        
        for i, run_result in enumerate(run_results[1:], 2):
            # Identical final scores
            assert run_result["final_score"] == reference_run["final_score"], f"Score mismatch in run {i}: {run_result['final_score']} != {reference_run['final_score']}"
            
            # Identical event counts
            assert run_result["event_count"] == reference_run["event_count"], f"Event count mismatch in run {i}: {run_result['event_count']} != {reference_run['event_count']}"
            
            # Identical event stream hashes
            assert run_result["event_hash"] == reference_run["event_hash"], f"Event hash mismatch in run {i}"
            
            # Identical configuration hashes
            assert run_result["config_hash"] == reference_run["config_hash"], f"Config hash mismatch in run {i}"
        
        # Calculate determinism score
        determinism_score = self.validator.calculate_determinism_score(run_results)
        
        # Validate perfect determinism
        assert determinism_score == 1.0, f"Determinism score not perfect: {determinism_score}"
        
        logger.info(f"Deterministic results validation:")
        logger.info(f"  Runs completed: {len(run_results)}")
        logger.info(f"  Determinism score: {determinism_score:.3f}")
        logger.info(f"  Reference hash: {reference_run['event_hash'][:16]}...")
        
        logger.info("✅ Deterministic identical results test passed")
        
    @pytest.mark.reproducibility
    @pytest.mark.asyncio
    async def test_golden_snapshot_validation(self):
        """
        Test golden snapshot validation for regression testing.
        
        Tests:
        - Golden snapshot generation
        - Snapshot comparison accuracy
        - Regression detection
        - Version drift identification
        """
        logger.info("🧪 Testing golden snapshot validation...")
        
        # Generate golden snapshot
        logger.info("Generating golden snapshot...")
        
        golden_seed = 54321
        snapshot_config = SimulationConfig(
            seed=golden_seed,
            max_ticks=30,
            tick_interval_seconds=0.01
        )
        
        # Create golden run
        env = await self.create_test_simulation(tier="T1", seed=golden_seed)
        
        # Run golden simulation
        orchestrator = env["orchestrator"]
        event_bus = env["event_bus"]
        
        event_bus.start_recording()
        await orchestrator.start(event_bus)
        await asyncio.sleep(0.5)  # Brief golden run
        await orchestrator.stop()
        
        golden_events = event_bus.get_recorded_events()
        event_bus.stop_recording()
        
        assert len(golden_events) > 0, "Golden snapshot generated no events"
        
        # Save golden snapshot
        golden_git_sha = "golden_snapshot_test"
        golden_run_id = "golden_reference"
        
        try:
            EventSnapshot.dump_events(golden_events, golden_git_sha, golden_run_id)
            golden_hash = EventSnapshot.generate_event_stream_hash(golden_events)
            snapshot_saved = True
        except Exception as e:
            logger.warning(f"Golden snapshot save failed: {e}")
            snapshot_saved = False
            golden_hash = "mock_golden_hash"
        
        # Test regression detection
        logger.info("Testing regression detection...")
        
        # Run identical simulation (should match)
        env2 = await self.create_test_simulation(tier="T1", seed=golden_seed)
        
        orchestrator2 = env2["orchestrator"]
        event_bus2 = env2["event_bus"]
        
        event_bus2.start_recording()
        await orchestrator2.start(event_bus2)
        await asyncio.sleep(0.5)  # Same duration
        await orchestrator2.stop()
        
        regression_events = event_bus2.get_recorded_events()
        event_bus2.stop_recording()
        
        # Compare with golden snapshot
        regression_hash = EventSnapshot.generate_event_stream_hash(regression_events)
        
        # Test CI integration if available
        try:
            is_reproducible = CIIntegration.verify_reproducibility(
                regression_events,
                golden_git_sha,
                golden_run_id
            )
            ci_available = True
        except Exception as e:
            logger.warning(f"CI integration not available: {e}")
            is_reproducible = (regression_hash == golden_hash)
            ci_available = False
        
        # Validate regression detection
        assert len(regression_events) > 0, "Regression test generated no events"
        
        # Hashes should match for identical runs
        assert regression_hash == golden_hash, f"Regression detected: {regression_hash[:16]} != {golden_hash[:16]}"
        
        if ci_available:
            assert is_reproducible, "CI reproducibility check failed"
        
        # Test with modified configuration (should detect differences)
        logger.info("Testing configuration change detection...")
        
        # Run with different seed (should produce different results)
        env3 = await self.create_test_simulation(tier="T1", seed=golden_seed + 1)
        
        orchestrator3 = env3["orchestrator"]
        event_bus3 = env3["event_bus"]
        
        event_bus3.start_recording()
        await orchestrator3.start(event_bus3)
        await asyncio.sleep(0.5)
        await orchestrator3.stop()
        
        modified_events = event_bus3.get_recorded_events()
        event_bus3.stop_recording()
        
        modified_hash = EventSnapshot.generate_event_stream_hash(modified_events)
        
        # Modified run should produce different hash
        assert modified_hash != golden_hash, "Configuration change not detected"
        
        # Clean up test snapshots
        if snapshot_saved:
            try:
                import os
                snapshot_path = EventSnapshot.ARTIFACTS_DIR / f"{golden_git_sha}_{golden_run_id}.parquet"
                if snapshot_path.exists():
                    os.unlink(snapshot_path)
            except Exception:
                pass
        
        logger.info(f"Golden snapshot validation results:")
        logger.info(f"  Golden events: {len(golden_events)}")
        logger.info(f"  Regression match: {regression_hash == golden_hash}")
        logger.info(f"  Change detection: {modified_hash != golden_hash}")
        logger.info(f"  CI integration: {ci_available}")
        
        logger.info("✅ Golden snapshot validation test passed")
        
    @pytest.mark.reproducibility
    @pytest.mark.asyncio
    async def test_statistical_consistency_across_seeds(self):
        """
        Test statistical consistency across multiple runs with different seeds.
        
        Tests:
        - Score distribution across different seeds
        - Expected variance validation
        - Statistical outlier detection
        - Consistency metrics calculation
        """
        logger.info("🧪 Testing statistical consistency across seeds...")
        
        # Test with multiple different seeds
        test_seeds = [42, 123, 456, 789, 999, 1337, 2023, 4096]
        num_seeds = len(test_seeds)
        
        if self.config.skip_slow_tests and num_seeds > 4:
            test_seeds = test_seeds[:4]
            logger.info(f"Limiting to {len(test_seeds)} seeds (skip slow tests)")
        
        score_results = []
        
        for i, seed in enumerate(test_seeds):
            logger.info(f"Running consistency test with seed {seed} ({i+1}/{len(test_seeds)})...")
            
            # Create simulation with different seed
            env = await self.create_test_simulation(tier="T1", seed=seed)
            
            # Initialize metrics
            financial_audit = FinancialAuditService()
            metric_suite = MetricSuite(
                tier="T1",
                financial_audit_service=financial_audit,
                sales_service=env["services"]["sales"],
                trust_score_service=env["services"]["trust"]
            )
            metric_suite.subscribe_to_events(env["event_bus"])
            
            # Run simulation
            orchestrator = env["orchestrator"]
            event_bus = env["event_bus"]
            
            event_bus.start_recording()
            await orchestrator.start(event_bus)
            await asyncio.sleep(0.8)  # Slightly longer for statistical validity
            await orchestrator.stop()
            
            # Calculate score
            events = event_bus.get_recorded_events()
            event_bus.stop_recording()
            
            if events:
                final_scores = metric_suite.calculate_final_score(events)
                score = final_scores.score
            else:
                score = 0
            
            score_results.append({
                "seed": seed,
                "score": score,
                "event_count": len(events),
                "tick_count": orchestrator.current_tick
            })
            
            logger.info(f"  Seed {seed}: Score={score:.2f}, Events={len(events)}")
        
        # Validate statistical consistency
        assert len(score_results) >= 3, "Need at least 3 runs for statistical analysis"
        
        scores = [result["score"] for result in score_results]
        event_counts = [result["event_count"] for result in score_results]
        
        # Calculate statistical metrics
        stats = self.validator.validate_statistical_consistency(scores)
        
        # Validate reasonable score distribution
        assert stats["mean"] > 0, "Mean score should be positive"
        assert stats["std_dev"] >= 0, "Standard deviation should be non-negative"
        
        # Check for reasonable variance (not too high, not zero)
        cv = stats["coefficient_of_variation"]
        assert 0 <= cv <= 1.0, f"Coefficient of variation out of reasonable range: {cv}"
        
        # Event counts should be reasonably consistent
        event_cv = statistics.stdev(event_counts) / statistics.mean(event_counts) if event_counts else 0
        assert event_cv <= 0.5, f"Event count variance too high: {event_cv}"
        
        # Check for outliers (simple outlier detection)
        q1 = statistics.quantiles(scores, n=4)[0] if len(scores) >= 4 else min(scores)
        q3 = statistics.quantiles(scores, n=4)[2] if len(scores) >= 4 else max(scores)
        iqr = q3 - q1
        outlier_threshold = 1.5 * iqr
        
        outliers = [s for s in scores if s < (q1 - outlier_threshold) or s > (q3 + outlier_threshold)]
        outlier_ratio = len(outliers) / len(scores)
        
        # No more than 25% outliers expected
        assert outlier_ratio <= 0.25, f"Too many statistical outliers: {outlier_ratio:.2%}"
        
        logger.info(f"Statistical consistency results:")
        logger.info(f"  Seeds tested: {len(test_seeds)}")
        logger.info(f"  Mean score: {stats['mean']:.2f} ± {stats['std_dev']:.2f}")
        logger.info(f"  Score range: {stats['min']:.2f} - {stats['max']:.2f}")
        logger.info(f"  Coefficient of variation: {cv:.3f}")
        logger.info(f"  Consistency score: {stats['consistency_score']:.3f}")
        logger.info(f"  Outliers: {len(outliers)}/{len(scores)} ({outlier_ratio:.1%})")
        
        logger.info("✅ Statistical consistency test passed")
        
    @pytest.mark.reproducibility
    @pytest.mark.asyncio
    async def test_configuration_sensitivity_validation(self):
        """
        Test that changes in configuration properly affect results.
        
        Tests:
        - Parameter sensitivity analysis
        - Configuration change detection
        - Expected impact validation
        - Boundary condition testing
        """
        logger.info("🧪 Testing configuration sensitivity validation...")
        
        base_seed = 42
        baseline_results = {}
        
        # Run baseline configuration
        logger.info("Running baseline configuration...")
        
        baseline_env = await self.create_test_simulation(tier="T1", seed=base_seed)
        
        # Baseline metrics
        financial_audit = FinancialAuditService()
        baseline_metrics = MetricSuite(
            tier="T1",
            financial_audit_service=financial_audit,
            sales_service=baseline_env["services"]["sales"],
            trust_score_service=baseline_env["services"]["trust"]
        )
        baseline_metrics.subscribe_to_events(baseline_env["event_bus"])
        
        # Run baseline
        orchestrator = baseline_env["orchestrator"]
        event_bus = baseline_env["event_bus"]
        
        event_bus.start_recording()
        await orchestrator.start(event_bus)
        await asyncio.sleep(0.5)
        await orchestrator.stop()
        
        baseline_events = event_bus.get_recorded_events()
        event_bus.stop_recording()
        
        if baseline_events:
            baseline_scores = baseline_metrics.calculate_final_score(baseline_events)
            baseline_score = baseline_scores.score
        else:
            baseline_score = 0
        
        baseline_results = {
            "score": baseline_score,
            "event_count": len(baseline_events),
            "tick_count": orchestrator.current_tick,
            "event_hash": EventSnapshot.generate_event_stream_hash(baseline_events)
        }
        
        logger.info(f"Baseline: Score={baseline_score:.2f}, Events={len(baseline_events)}")
        
        # Test configuration variations
        config_variations = [
            {
                "name": "different_seed",
                "changes": {"seed": base_seed + 100},
                "expected_impact": "different"
            },
            {
                "name": "different_tier",
                "changes": {"tier": "T2"},
                "expected_impact": "different"
            },
            {
                "name": "longer_run",
                "changes": {"duration_multiplier": 2.0},
                "expected_impact": "different"
            },
            {
                "name": "identical_config",
                "changes": {},
                "expected_impact": "identical"
            }
        ]
        
        sensitivity_results = {}
        
        for variation in config_variations:
            var_name = variation["name"]
            changes = variation["changes"]
            expected_impact = variation["expected_impact"]
            
            logger.info(f"Testing configuration variation: {var_name}...")
            
            # Apply configuration changes
            test_seed = changes.get("seed", base_seed)
            test_tier = changes.get("tier", "T1")
            duration_mult = changes.get("duration_multiplier", 1.0)
            
            # Create modified environment
            var_env = await self.create_test_simulation(tier=test_tier, seed=test_seed)
            
            # Initialize metrics for test tier
            var_financial_audit = FinancialAuditService()
            var_metrics = MetricSuite(
                tier=test_tier,
                financial_audit_service=var_financial_audit,
                sales_service=var_env["services"]["sales"],
                trust_score_service=var_env["services"]["trust"]
            )
            var_metrics.subscribe_to_events(var_env["event_bus"])
            
            # Run variation
            var_orchestrator = var_env["orchestrator"]
            var_event_bus = var_env["event_bus"]
            
            var_event_bus.start_recording()
            await var_orchestrator.start(var_event_bus)
            await asyncio.sleep(0.5 * duration_mult)
            await var_orchestrator.stop()
            
            var_events = var_event_bus.get_recorded_events()
            var_event_bus.stop_recording()
            
            if var_events:
                var_scores = var_metrics.calculate_final_score(var_events)
                var_score = var_scores.score
            else:
                var_score = 0
            
            var_results = {
                "score": var_score,
                "event_count": len(var_events),
                "tick_count": var_orchestrator.current_tick,
                "event_hash": EventSnapshot.generate_event_stream_hash(var_events)
            }
            
            # Analyze sensitivity
            score_diff = abs(var_score - baseline_score)
            event_count_diff = abs(len(var_events) - len(baseline_events))
            hash_identical = (var_results["event_hash"] == baseline_results["event_hash"])
            
            # Validate expected impact
            if expected_impact == "identical":
                # Should be identical results
                assert var_score == baseline_score, f"Scores should be identical for {var_name}: {var_score} != {baseline_score}"
                assert len(var_events) == len(baseline_events), f"Event counts should be identical for {var_name}"
                assert hash_identical, f"Event hashes should be identical for {var_name}"
            elif expected_impact == "different":
                # Should show some difference
                has_difference = (score_diff > 0.1 or event_count_diff > 0 or not hash_identical)
                assert has_difference, f"No difference detected for {var_name} (expected differences)"
            
            sensitivity_results[var_name] = {
                "results": var_results,
                "score_difference": score_diff,
                "event_count_difference": event_count_diff,
                "hash_identical": hash_identical,
                "expected_impact": expected_impact,
                "impact_detected": not hash_identical or score_diff > 0.1
            }
            
            logger.info(f"  {var_name}: Score={var_score:.2f} (Δ{score_diff:.2f}), Events={len(var_events)} (Δ{event_count_diff})")
        
        # Validate configuration sensitivity
        assert len(sensitivity_results) == len(config_variations), "Not all variations tested"
        
        # Check that we can detect both identical and different configurations
        identical_tests = [r for r in sensitivity_results.values() if r["expected_impact"] == "identical"]
        different_tests = [r for r in sensitivity_results.values() if r["expected_impact"] == "different"]
        
        if identical_tests:
            assert all(not r["impact_detected"] for r in identical_tests), "Failed to detect identical configurations"
        
        if different_tests:
            assert any(r["impact_detected"] for r in different_tests), "Failed to detect configuration differences"
        
        logger.info(f"Configuration sensitivity results:")
        for var_name, results in sensitivity_results.items():
            logger.info(f"  {var_name}: Expected={results['expected_impact']}, Detected={results['impact_detected']}")
        
        logger.info("✅ Configuration sensitivity validation test passed")
        
    @pytest.mark.reproducibility
    @pytest.mark.asyncio
    async def test_cross_platform_consistency(self):
        """
        Test platform independence verification.
        
        Tests:
        - Environment consistency validation
        - Floating-point determinism
        - Library version independence
        - System-specific behavior isolation
        """
        logger.info("🧪 Testing cross-platform consistency...")
        
        # Test multiple runs on current platform to simulate cross-platform
        # In a real scenario, this would run on different OS/hardware
        
        platform_seed = 98765
        platform_results = []
        
        # Simulate different "platform" conditions
        platform_conditions = [
            {"name": "condition_1", "time_acceleration": 1.0},
            {"name": "condition_2", "time_acceleration": 2.0},
            {"name": "condition_3", "time_acceleration": 0.5}
        ]
        
        for condition in platform_conditions:
            condition_name = condition["name"]
            time_accel = condition["time_acceleration"]
            
            logger.info(f"Testing platform condition: {condition_name}...")
            
            # Create environment with platform-specific settings
            env = await self.create_test_simulation(tier="T0", seed=platform_seed)
            
            # Configure for platform testing
            sim_config = SimulationConfig(
                seed=platform_seed,
                max_ticks=20,
                time_acceleration=time_accel
            )
            
            # Initialize metrics
            financial_audit = FinancialAuditService()
            metrics = MetricSuite(
                tier="T0",
                financial_audit_service=financial_audit,
                sales_service=env["services"]["sales"],
                trust_score_service=env["services"]["trust"]
            )
            metrics.subscribe_to_events(env["event_bus"])
            
            # Run simulation
            orchestrator = SimulationOrchestrator(sim_config)
            event_bus = env["event_bus"]
            
            event_bus.start_recording()
            await orchestrator.start(event_bus)
            await asyncio.sleep(0.3)  # Brief run
            await orchestrator.stop()
            
            events = event_bus.get_recorded_events()
            event_bus.stop_recording()
            
            if events:
                final_scores = metrics.calculate_final_score(events)
                score = final_scores.score
            else:
                score = 0
            
            platform_result = {
                "condition": condition_name,
                "score": score,
                "event_count": len(events),
                "tick_count": orchestrator.current_tick,
                "event_hash": EventSnapshot.generate_event_stream_hash(events),
                "time_acceleration": time_accel
            }
            
            platform_results.append(platform_result)
            
            logger.info(f"  {condition_name}: Score={score:.2f}, Events={len(events)}")
        
        # Validate cross-platform consistency
        assert len(platform_results) >= 2, "Need multiple platform conditions for comparison"
        
        # For deterministic systems, results should be consistent across "platforms"
        # when using the same seed and logical configuration
        
        # Check that we get reasonable results from all conditions
        for result in platform_results:
            assert result["score"] >= 0, f"Invalid score for {result['condition']}: {result['score']}"
            assert result["event_count"] > 0, f"No events for {result['condition']}"
        
        # Test floating-point determinism (scores should be consistent)
        scores = [result["score"] for result in platform_results]
        
        # Allow for small floating-point differences but should be mostly consistent
        if len(set(f"{s:.6f}" for s in scores)) == 1:
            logger.info("Perfect cross-platform score consistency achieved")
        else:
            score_variance = statistics.variance(scores) if len(scores) > 1 else 0
            # Small variance acceptable for floating-point operations
            assert score_variance < 0.01, f"Excessive score variance across platforms: {score_variance}"
        
        # Validate system behavior isolation
        # Different time accelerations shouldn't affect final results significantly
        # when using the same seed and tick count
        
        logger.info(f"Cross-platform consistency results:")
        for result in platform_results:
            logger.info(f"  {result['condition']}: Score={result['score']:.6f}, Events={result['event_count']}")
        
        # Calculate platform consistency score
        platform_consistency = self.validator.calculate_determinism_score(platform_results)
        logger.info(f"  Platform consistency score: {platform_consistency:.3f}")
        
        # Require high consistency for platform independence
        assert platform_consistency >= 0.9, f"Platform consistency too low: {platform_consistency}"
        
        logger.info("✅ Cross-platform consistency test passed")

@pytest.mark.reproducibility
class TestReproducibilityIntegration:
    """Integration tests combining multiple reproducibility components."""
    
    @pytest.mark.asyncio
    async def test_complete_reproducibility_validation(self):
        """
        Run complete reproducibility validation combining all requirements.
        
        This test validates that FBA-Bench meets all scientific reproducibility
        standards required for a tier-1 benchmark.
        """
        logger.info("🚀 Running complete reproducibility validation...")
        
        reproducibility_suite = TestScientificReproducibility()
        reproducibility_results = {}
        
        try:
            # Run all reproducibility tests
            await reproducibility_suite.test_deterministic_identical_results()
            reproducibility_results["deterministic_results"] = True
            
            await reproducibility_suite.test_golden_snapshot_validation()
            reproducibility_results["golden_snapshots"] = True
            
            await reproducibility_suite.test_statistical_consistency_across_seeds()
            reproducibility_results["statistical_consistency"] = True
            
            await reproducibility_suite.test_configuration_sensitivity_validation()
            reproducibility_results["configuration_sensitivity"] = True
            
            await reproducibility_suite.test_cross_platform_consistency()
            reproducibility_results["cross_platform_consistency"] = True
            
            # Validate all reproducibility components
            failed_components = [k for k, v in reproducibility_results.items() if not v]
            if failed_components:
                logger.warning(f"Some reproducibility components failed: {failed_components}")
            else:
                logger.info("All reproducibility tests passed!")
            
            # Overall reproducibility score
            reproducibility_score = sum(reproducibility_results.values()) / len(reproducibility_results) * 100
            
            assert reproducibility_score >= 90, f"Overall reproducibility score too low: {reproducibility_score}%"
            
        except Exception as e:
            logger.error(f"Complete reproducibility validation failed: {e}")
            raise
        
        logger.info("🎉 Complete reproducibility validation passed!")
        return reproducibility_results