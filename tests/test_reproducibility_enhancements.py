"""
Comprehensive Test Suite for FBA-Bench Reproducibility Enhancements

Tests all reproducibility features including LLM caching, deterministic clients,
seed management, golden master testing, and mode coordination.
"""

import asyncio
import tempfile
import json
import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
import pytest
import time
from datetime import datetime, timezone

# Import reproducibility components
from reproducibility.llm_cache import LLMResponseCache, CachedResponse, CacheStatistics
from reproducibility.sim_seed import SimSeed, DeterminismValidationResult
from reproducibility.golden_master import (
    GoldenMasterTester, ToleranceConfig, ComparisonResult, GoldenMasterRecord
)
from reproducibility.simulation_modes import (
    SimulationModeController, SimulationMode, ModeConfiguration, 
    get_mode_controller, set_global_mode
)
from reproducibility.reproducibility_config import (
    ReproducibilityConfig, create_deterministic_config, 
    create_research_config, get_global_config, set_global_config
)
from reproducibility.event_snapshots import EventSnapshot, LLMInteractionLog, SnapshotMetadata

from llm_interface.deterministic_client import (
    DeterministicLLMClient, OperationMode, ValidationSchema,
    create_deterministic_client, create_hybrid_client
)
from llm_interface.openrouter_client import OpenRouterClient
from llm_interface.contract import BaseLLMClient

# Mock LLM client for testing
class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing purposes."""
    
    def __init__(self, model_name: str = "mock-model", responses: List[str] = None):
        super().__init__(model_name)
        self.responses = responses or ["{'action': 'test_response'}"]
        self.call_count = 0
    
    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        response_content = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        
        return {
            "choices": [
                {
                    "message": {
                        "content": response_content,
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_content.split()),
                "total_tokens": len(prompt.split()) + len(response_content.split())
            }
        }
    
    async def get_token_count(self, text: str) -> int:
        return len(text.split())

class TestLLMCache:
    """Test suite for LLM response caching system."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def cache(self, temp_cache_dir):
        """Create LLM cache instance for testing."""
        return LLMResponseCache(
            cache_file="test_cache.db",
            cache_dir=temp_cache_dir,
            enable_validation=True
        )
    
    def test_cache_initialization(self, cache):
        """Test cache initialization and database setup."""
        assert cache.cache_dir.exists()
        assert cache.cache_file.exists()
        assert cache.enable_validation
    
    def test_prompt_hash_generation(self, cache):
        """Test deterministic prompt hash generation."""
        prompt = "Test prompt"
        model = "test-model"
        temperature = 0.5
        
        hash1 = cache.generate_prompt_hash(prompt, model, temperature)
        hash2 = cache.generate_prompt_hash(prompt, model, temperature)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
        
        # Different inputs should produce different hashes
        hash3 = cache.generate_prompt_hash("Different prompt", model, temperature)
        assert hash1 != hash3
    
    def test_cache_storage_and_retrieval(self, cache):
        """Test storing and retrieving responses from cache."""
        prompt_hash = "test_hash_123"
        response = {"choices": [{"message": {"content": "test response"}}]}
        metadata = {"model": "test-model", "temperature": 0.0}
        
        # Store response
        success = cache.cache_response(prompt_hash, response, metadata)
        assert success
        
        # Retrieve response
        cached_response = cache.get_cached_response(prompt_hash)
        assert cached_response == response
    
    def test_cache_miss(self, cache):
        """Test cache miss behavior."""
        non_existent_hash = "non_existent_hash"
        result = cache.get_cached_response(non_existent_hash)
        assert result is None
    
    def test_deterministic_mode(self, cache):
        """Test deterministic mode behavior."""
        cache.set_deterministic_mode(True)
        
        # Cache miss in deterministic mode should raise exception
        with pytest.raises(ValueError):
            cache.get_cached_response("non_existent_hash")
    
    def test_cache_validation(self, cache):
        """Test cache integrity validation."""
        # Add some test data
        prompt_hash = "validation_test"
        response = {"choices": [{"message": {"content": "validation test"}}]}
        cache.cache_response(prompt_hash, response)
        
        # Validate cache
        is_valid, errors = cache.validate_cache_integrity()
        assert is_valid
        assert len(errors) == 0
    
    def test_cache_statistics(self, cache):
        """Test cache statistics tracking."""
        initial_stats = cache.get_cache_statistics()
        assert initial_stats.total_requests == 0
        
        # Add some cache operations
        cache.cache_response("test1", {"test": "data1"})
        cache.get_cached_response("test1")  # Hit
        cache.get_cached_response("missing")  # Miss
        
        stats = cache.get_cache_statistics()
        assert stats.total_requests == 2
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1
        assert stats.hit_ratio == 0.5

class TestDeterministicLLMClient:
    """Test suite for deterministic LLM client."""
    
    @pytest.fixture
    def mock_client(self):
        """Create mock LLM client."""
        return MockLLMClient(responses=[
            '{"action": "response1"}',
            '{"action": "response2"}',
            '{"action": "response3"}'
        ])
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def deterministic_client(self, mock_client, temp_cache_dir):
        """Create deterministic LLM client for testing."""
        cache = LLMResponseCache(cache_dir=temp_cache_dir)
        return DeterministicLLMClient(
            underlying_client=mock_client,
            cache=cache,
            mode=OperationMode.HYBRID
        )
    
    @pytest.mark.asyncio
    async def test_deterministic_mode_caching(self, deterministic_client):
        """Test that deterministic mode properly caches responses."""
        prompt = "Test prompt for caching"
        
        # First call should cache the response
        response1 = await deterministic_client.call_llm(prompt)
        
        # Second call should return cached response
        response2 = await deterministic_client.call_llm(prompt)
        
        # Should be identical responses
        assert response1['choices'][0]['message']['content'] == response2['choices'][0]['message']['content']
        
        # Check metadata indicates cache hit
        metadata2 = response2.get('_fba_metadata', {})
        assert metadata2.get('cache_hit') == True
    
    @pytest.mark.asyncio
    async def test_mode_switching(self, deterministic_client):
        """Test switching between different operation modes."""
        # Start in hybrid mode
        assert deterministic_client.mode == OperationMode.HYBRID
        
        # Switch to deterministic mode
        deterministic_client.set_deterministic_mode(True)
        assert deterministic_client.mode == OperationMode.DETERMINISTIC
        
        # Switch back to hybrid
        deterministic_client.set_deterministic_mode(False)
        assert deterministic_client.mode == OperationMode.HYBRID
    
    @pytest.mark.asyncio
    async def test_response_validation(self, deterministic_client):
        """Test response format validation."""
        schema = ValidationSchema(
            required_fields=["choices"],
            field_types={"choices": list}
        )
        
        deterministic_client.set_validation_schema(schema)
        
        # Valid response should pass
        response = await deterministic_client.call_llm("Test prompt")
        assert 'choices' in response
        assert isinstance(response['choices'], list)
    
    def test_statistics_tracking(self, deterministic_client):
        """Test client statistics tracking."""
        initial_stats = deterministic_client.get_cache_statistics()
        assert 'total_calls' in initial_stats
        assert 'cache_hits' in initial_stats
        assert 'cache_misses' in initial_stats

class TestSimSeed:
    """Test suite for enhanced seed management."""
    
    def setup_method(self):
        """Reset seed state before each test."""
        SimSeed.reset_master_seed()
    
    def teardown_method(self):
        """Clean up after each test."""
        SimSeed.reset_master_seed()
    
    def test_master_seed_setting(self):
        """Test master seed setting and retrieval."""
        test_seed = 12345
        SimSeed.set_master_seed(test_seed)
        
        assert SimSeed.get_master_seed() == test_seed
        
        # Should not allow re-setting
        with pytest.raises(RuntimeError):
            SimSeed.set_master_seed(54321)
    
    def test_component_seed_isolation(self):
        """Test component-specific seed generation."""
        SimSeed.set_master_seed(42)
        
        # Different components should get different seeds
        seed1 = SimSeed.get_component_seed("component1")
        seed2 = SimSeed.get_component_seed("component2")
        
        assert seed1 != seed2
        
        # Same component should get same seed
        seed1_again = SimSeed.get_component_seed("component1")
        assert seed1 == seed1_again
    
    def test_seed_derivation(self):
        """Test deterministic seed derivation."""
        SimSeed.set_master_seed(42)
        
        # Same salt should produce same derived seed
        derived1 = SimSeed.derive_seed("test_salt")
        derived2 = SimSeed.derive_seed("test_salt")
        assert derived1 == derived2
        
        # Different salt should produce different seed
        derived3 = SimSeed.derive_seed("different_salt")
        assert derived1 != derived3
    
    def test_rng_source_registration(self):
        """Test RNG source registration and tracking."""
        SimSeed.set_master_seed(42)
        
        # Register a mock RNG source
        import random
        test_rng = random.Random()
        SimSeed.register_rng_source("test_rng", test_rng, "test_component")
        
        # Should be in registered sources
        stats = SimSeed.get_statistics()
        assert stats["registered_sources"] == 1
        assert "test_component" in stats["components"]
    
    def test_determinism_validation(self):
        """Test determinism validation functionality."""
        SimSeed.set_master_seed(42)
        SimSeed.get_component_seed("test_component")
        
        result = SimSeed.validate_determinism()
        assert isinstance(result, DeterminismValidationResult)
        assert result.is_deterministic
        assert len(result.issues) == 0
    
    def test_audit_trail(self):
        """Test audit trail functionality."""
        SimSeed.enable_audit(True)
        SimSeed.set_master_seed(42)
        SimSeed.get_component_seed("test_component")
        
        audit_trail = SimSeed.get_audit_trail()
        assert len(audit_trail) > 0
        
        # Check audit entries have required fields
        entry = audit_trail[0]
        assert hasattr(entry, 'timestamp')
        assert hasattr(entry, 'component')
        assert hasattr(entry, 'operation')
        assert hasattr(entry, 'seed_value')
    
    def test_component_context(self):
        """Test component context manager."""
        SimSeed.set_master_seed(42)
        
        with SimSeed.component_context("test_component") as seed:
            assert isinstance(seed, int)
            # Random operations here would use component-specific seed

class TestGoldenMaster:
    """Test suite for golden master testing system."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary directory for golden master storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def golden_master_tester(self, temp_storage_dir):
        """Create golden master tester instance."""
        return GoldenMasterTester(
            storage_dir=temp_storage_dir,
            enable_compression=False  # Disable for easier testing
        )
    
    @pytest.fixture
    def sample_simulation_data(self):
        """Create sample simulation data for testing."""
        return {
            "events": [
                {"timestamp": "2024-01-01T00:00:00Z", "type": "start", "data": {"tick": 0}},
                {"timestamp": "2024-01-01T00:01:00Z", "type": "action", "data": {"price": 1000}},
                {"timestamp": "2024-01-01T00:02:00Z", "type": "end", "data": {"tick": 2}}
            ],
            "final_state": {
                "revenue": 5000,
                "profit": 1000,
                "inventory": 100
            },
            "metadata": {
                "simulation_duration": 120,
                "agent_count": 3
            }
        }
    
    def test_golden_master_recording(self, golden_master_tester, sample_simulation_data):
        """Test recording a golden master baseline."""
        label = "test_baseline_v1"
        
        success = golden_master_tester.record_golden_master(
            simulation_run=sample_simulation_data,
            label=label,
            metadata={"test": True}
        )
        
        assert success
        assert label in golden_master_tester.list_golden_masters()
    
    def test_golden_master_comparison(self, golden_master_tester, sample_simulation_data):
        """Test comparing runs against golden master."""
        label = "comparison_test"
        
        # Record baseline
        golden_master_tester.record_golden_master(sample_simulation_data, label)
        
        # Compare identical data (should pass)
        result = golden_master_tester.compare_against_golden(sample_simulation_data, label)
        assert result.is_identical
        assert result.is_within_tolerance
        assert len(result.critical_differences) == 0
        
        # Compare modified data (should detect differences)
        modified_data = sample_simulation_data.copy()
        modified_data["final_state"]["revenue"] = 6000
        
        result2 = golden_master_tester.compare_against_golden(modified_data, label)
        assert not result2.is_identical
        assert len(result2.differences) > 0
    
    def test_tolerance_configuration(self, golden_master_tester, sample_simulation_data):
        """Test tolerance configuration for comparisons."""
        label = "tolerance_test"
        golden_master_tester.record_golden_master(sample_simulation_data, label)
        
        # Modify numeric value slightly
        modified_data = sample_simulation_data.copy()
        modified_data["final_state"]["revenue"] = 5000.1  # Small difference
        
        # Should fail with strict tolerance
        strict_tolerance = ToleranceConfig(numeric_tolerance=1e-10)
        result1 = golden_master_tester.compare_against_golden(
            modified_data, label, strict_tolerance
        )
        assert not result1.is_within_tolerance
        
        # Should pass with lenient tolerance
        lenient_tolerance = ToleranceConfig(numeric_tolerance=1.0)
        result2 = golden_master_tester.compare_against_golden(
            modified_data, label, lenient_tolerance
        )
        assert result2.is_within_tolerance
    
    def test_reproducibility_report_generation(self, golden_master_tester, sample_simulation_data):
        """Test comprehensive reproducibility report generation."""
        # Create multiple comparison results
        comparison_results = []
        
        for i in range(3):
            label = f"test_run_{i}"
            golden_master_tester.record_golden_master(sample_simulation_data, label)
            result = golden_master_tester.compare_against_golden(sample_simulation_data, label)
            comparison_results.append(result)
        
        report = golden_master_tester.generate_reproducibility_report(comparison_results)
        
        assert "summary" in report
        assert "statistics" in report
        assert "patterns" in report
        assert "recommendations" in report
        assert report["summary"]["perfect_reproducibility_rate"] == 1.0

class TestSimulationModeController:
    """Test suite for simulation mode controller."""
    
    def setup_method(self):
        """Reset state before each test."""
        # Reset global state
        SimSeed.reset_master_seed()
    
    def test_mode_controller_initialization(self):
        """Test mode controller initialization."""
        controller = SimulationModeController(initial_mode=SimulationMode.DETERMINISTIC)
        
        assert controller._current_mode == SimulationMode.DETERMINISTIC
        assert controller._current_config is not None
    
    def test_mode_switching(self):
        """Test switching between simulation modes."""
        controller = SimulationModeController()
        
        # Switch to stochastic mode
        result = controller.set_mode(SimulationMode.STOCHASTIC)
        assert result.success
        assert controller._current_mode == SimulationMode.STOCHASTIC
        
        # Switch to research mode
        result = controller.set_mode(SimulationMode.RESEARCH)
        assert result.success
        assert controller._current_mode == SimulationMode.RESEARCH
    
    def test_component_registration(self):
        """Test component registration and mode application."""
        controller = SimulationModeController()
        
        # Mock component with mode handlers
        class MockComponent:
            def __init__(self):
                self.current_mode = None
            
            def set_deterministic_mode(self, enabled):
                self.current_mode = "deterministic" if enabled else "stochastic"
        
        component = MockComponent()
        controller.register_component("test_component", component)
        
        # Switch mode and verify component is updated
        controller.set_mode(SimulationMode.DETERMINISTIC)
        assert component.current_mode == "deterministic"
    
    def test_mode_status_reporting(self):
        """Test mode status and performance reporting."""
        controller = SimulationModeController()
        
        status = controller.get_mode_status()
        assert "current_mode" in status
        assert "uptime_seconds" in status
        assert "performance_metrics" in status
        assert "registered_components" in status
    
    def test_temporary_mode_context(self):
        """Test temporary mode switching with context manager."""
        controller = SimulationModeController(initial_mode=SimulationMode.DETERMINISTIC)
        
        original_mode = controller._current_mode
        
        with controller.temporary_mode(SimulationMode.STOCHASTIC):
            assert controller._current_mode == SimulationMode.STOCHASTIC
        
        # Should be restored after context
        assert controller._current_mode == original_mode
    
    def test_health_check(self):
        """Test controller health check functionality."""
        controller = SimulationModeController()
        
        health = controller.health_check()
        assert "status" in health
        assert "current_mode" in health
        assert "component_count" in health

class TestReproducibilityConfig:
    """Test suite for reproducibility configuration system."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for config testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_config_creation(self):
        """Test configuration object creation and validation."""
        config = create_deterministic_config(master_seed=42)
        
        assert config.simulation_mode == SimulationMode.DETERMINISTIC
        assert config.seed_management.master_seed == 42
        assert config.llm_cache.enabled == True
        
        # Should be valid
        assert config.is_valid()
        assert len(config.validate()) == 0
    
    def test_config_validation(self):
        """Test configuration validation logic."""
        # Create invalid config
        config = ReproducibilityConfig()
        config.simulation_mode = SimulationMode.DETERMINISTIC
        config.llm_cache.enabled = False  # Invalid for deterministic mode
        
        issues = config.validate()
        assert len(issues) > 0
        assert not config.is_valid()
    
    def test_config_serialization(self, temp_config_dir):
        """Test configuration file save/load functionality."""
        config = create_deterministic_config(master_seed=12345)
        
        config_file = Path(temp_config_dir) / "test_config.json"
        
        # Save configuration
        success = config.save_to_file(config_file)
        assert success
        assert config_file.exists()
        
        # Load configuration
        loaded_config = ReproducibilityConfig.load_from_file(config_file)
        assert loaded_config is not None
        assert loaded_config.seed_management.master_seed == 12345
        assert loaded_config.simulation_mode == SimulationMode.DETERMINISTIC
    
    def test_predefined_configs(self):
        """Test predefined configuration templates."""
        # Test deterministic config
        det_config = create_deterministic_config()
        assert det_config.simulation_mode == SimulationMode.DETERMINISTIC
        assert det_config.llm_cache.allow_cache_misses == False
        
        # Test research config
        research_config = create_research_config()
        assert research_config.simulation_mode == SimulationMode.RESEARCH
        assert research_config.llm_cache.allow_cache_misses == True
    
    def test_global_config_management(self):
        """Test global configuration management."""
        original_config = get_global_config()
        
        # Set new global config
        test_config = create_deterministic_config(master_seed=999)
        set_global_config(test_config)
        
        # Verify global config changed
        current_config = get_global_config()
        assert current_config.seed_management.master_seed == 999

class TestEventSnapshots:
    """Test suite for enhanced event snapshots."""
    
    @pytest.fixture
    def sample_events(self):
        """Create sample events for testing."""
        return [
            {"timestamp": "2024-01-01T00:00:00Z", "event_type": "start", "data": {"tick": 0}},
            {"timestamp": "2024-01-01T00:01:00Z", "event_type": "action", "data": {"action": "price_change"}},
            {"timestamp": "2024-01-01T00:02:00Z", "event_type": "end", "data": {"tick": 2}}
        ]
    
    def test_llm_interaction_logging(self):
        """Test LLM interaction logging functionality."""
        EventSnapshot.clear_llm_interactions()
        
        # Log some interactions
        EventSnapshot.log_llm_interaction(
            prompt_hash="test_hash_1",
            model="test-model",
            temperature=0.0,
            cache_hit=True,
            response_hash="response_hash_1",
            deterministic_mode=True,
            validation_passed=True,
            response_time_ms=150.5
        )
        
        summary = EventSnapshot.get_llm_interaction_summary()
        assert summary["total_interactions"] == 1
        assert summary["cache_hits"] == 1
        assert summary["cache_hit_ratio"] == 1.0
    
    def test_enhanced_snapshot_creation(self, sample_events, tmp_path):
        """Test enhanced snapshot creation with metadata."""
        # Set up metadata
        EventSnapshot.set_snapshot_metadata(
            simulation_mode="deterministic",
            master_seed=42,
            llm_cache_status={"cache_size": 100, "hit_ratio": 0.95}
        )
        
        # Create enhanced snapshot
        EventSnapshot.clear_llm_interactions()
        EventSnapshot.log_llm_interaction(
            "hash1", "model1", 0.0, True, "resp1", True, True, 100.0
        )
        
        # Change to temp artifacts directory
        original_artifacts_dir = EventSnapshot.ARTIFACTS_DIR
        EventSnapshot.ARTIFACTS_DIR = tmp_path
        
        try:
            snapshot_path = EventSnapshot.dump_events_with_metadata(
                events=sample_events,
                git_sha="test123",
                run_id="run456"
            )
            
            assert snapshot_path is not None
            assert snapshot_path.exists()
            
            # Load and verify enhanced snapshot
            snapshot_data = EventSnapshot.load_enhanced_snapshot(snapshot_path)
            assert "events" in snapshot_data
            assert "llm_interactions" in snapshot_data
            assert "reproducibility_metadata" in snapshot_data
            assert len(snapshot_data["events"]) == 3
            assert len(snapshot_data["llm_interactions"]) == 1
            
        finally:
            EventSnapshot.ARTIFACTS_DIR = original_artifacts_dir
    
    def test_snapshot_reproducibility_validation(self, sample_events, tmp_path):
        """Test snapshot reproducibility validation."""
        EventSnapshot.ARTIFACTS_DIR = tmp_path
        
        try:
            # Create two identical snapshots
            snapshot1_path = EventSnapshot.dump_events_with_metadata(
                sample_events, "sha1", "run1"
            )
            snapshot2_path = EventSnapshot.dump_events_with_metadata(
                sample_events, "sha2", "run2"
            )
            
            # Validate reproducibility
            validation_result = EventSnapshot.validate_snapshot_reproducibility(
                snapshot1_path, snapshot2_path
            )
            
            assert validation_result["is_reproducible"]
            assert validation_result["events_match"]
            assert len(validation_result["issues"]) == 0
            
        finally:
            EventSnapshot.ARTIFACTS_DIR = Path("artifacts")

class TestIntegration:
    """Integration tests for the complete reproducibility system."""
    
    @pytest.fixture
    def temp_test_dir(self):
        """Create temporary directory for integration testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_end_to_end_deterministic_simulation(self, temp_test_dir):
        """Test complete end-to-end deterministic simulation workflow."""
        # 1. Set up global reproducibility configuration
        config = create_deterministic_config(master_seed=42)
        config.output_dir = temp_test_dir
        config.llm_cache.cache_dir = temp_test_dir
        config.golden_master.storage_dir = temp_test_dir
        set_global_config(config)
        
        # 2. Initialize simulation mode controller
        controller = get_mode_controller()
        controller.set_mode(SimulationMode.DETERMINISTIC, config)
        
        # 3. Set up deterministic seed management
        SimSeed.set_master_seed(42)
        
        # 4. Create mock simulation data
        simulation_data = {
            "events": [
                {"tick": 0, "action": "start"},
                {"tick": 1, "action": "price_change", "price": 1000},
                {"tick": 2, "action": "end"}
            ],
            "final_metrics": {
                "revenue": 5000,
                "profit": 1000
            }
        }
        
        # 5. Record as golden master
        golden_master = GoldenMasterTester(storage_dir=temp_test_dir)
        success = golden_master.record_golden_master(
            simulation_data, "integration_test_baseline"
        )
        assert success
        
        # 6. Simulate second run with same setup
        # (In real test, this would be actual simulation)
        result = golden_master.compare_against_golden(
            simulation_data, "integration_test_baseline"
        )
        
        assert result.is_identical
        assert result.is_within_tolerance
        
        # 7. Verify reproducibility statistics
        stats = controller.get_mode_status()
        assert stats["current_mode"] == SimulationMode.DETERMINISTIC.value
    
    @pytest.mark.asyncio
    async def test_llm_deterministic_workflow(self, temp_test_dir):
        """Test deterministic LLM workflow integration."""
        # Set up cache
        cache = LLMResponseCache(cache_dir=temp_test_dir)
        
        # Create deterministic client
        mock_client = MockLLMClient()
        det_client = DeterministicLLMClient(
            underlying_client=mock_client,
            cache=cache,
            mode=OperationMode.DETERMINISTIC
        )
        
        # Enable deterministic mode
        det_client.set_deterministic_mode(True)
        
        prompt = "Test deterministic prompt"
        
        # First call should go to underlying client and cache
        response1 = await det_client.call_llm(prompt)
        
        # Second call should come from cache
        response2 = await det_client.call_llm(prompt)
        
        # Responses should be identical
        assert response1['choices'][0]['message']['content'] == response2['choices'][0]['message']['content']
        
        # Second response should indicate cache hit
        metadata2 = response2.get('_fba_metadata', {})
        assert metadata2.get('cache_hit') == True
        
        # Verify cache statistics
        stats = det_client.get_cache_statistics()
        assert stats['cache_hits'] >= 1
        assert stats['total_calls'] >= 2
    
    def test_mode_switching_integration(self):
        """Test integration of mode switching across components."""
        # Create controller
        controller = SimulationModeController()
        
        # Create mock components
        class MockAgent:
            def __init__(self, agent_id):
                self.agent_id = agent_id
                self.mode = None
            
            def set_deterministic_mode(self, enabled):
                self.mode = "deterministic" if enabled else "stochastic"
        
        agents = [MockAgent(f"agent_{i}") for i in range(3)]
        
        # Register agents
        for agent in agents:
            controller.register_component(f"agent_{agent.agent_id}", agent)
        
        # Switch to deterministic mode
        result = controller.set_mode(SimulationMode.DETERMINISTIC)
        assert result.success
        
        # All agents should be in deterministic mode
        for agent in agents:
            assert agent.mode == "deterministic"
        
        # Switch to stochastic mode
        result = controller.set_mode(SimulationMode.STOCHASTIC)
        assert result.success
        
        # All agents should be in stochastic mode
        for agent in agents:
            assert agent.mode == "stochastic"

# Performance and stress tests
class TestPerformance:
    """Performance tests for reproducibility overhead."""
    
    def test_cache_performance(self, tmp_path):
        """Test cache performance with large number of operations."""
        cache = LLMResponseCache(cache_dir=str(tmp_path))
        
        # Measure time for cache operations
        start_time = time.time()
        
        # Perform many cache operations
        for i in range(1000):
            prompt_hash = f"hash_{i}"
            response = {"choices": [{"message": {"content": f"response_{i}"}}]}
            cache.cache_response(prompt_hash, response)
        
        # Retrieve all cached responses
        for i in range(1000):
            prompt_hash = f"hash_{i}"
            cached = cache.get_cached_response(prompt_hash)
            assert cached is not None
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert elapsed < 10.0  # 10 seconds for 2000 operations
        
        # Check final statistics
        stats = cache.get_cache_statistics()
        assert stats.cache_size >= 1000
        assert stats.cache_hits >= 1000
    
    def test_seed_generation_performance(self):
        """Test performance of seed generation operations."""
        SimSeed.set_master_seed(42)
        
        start_time = time.time()
        
        # Generate many component seeds
        for i in range(1000):
            seed = SimSeed.get_component_seed(f"component_{i}")
            assert isinstance(seed, int)
        
        elapsed = time.time() - start_time
        
        # Should be very fast
        assert elapsed < 1.0  # 1 second for 1000 operations
        
        SimSeed.reset_master_seed()
    
    def test_golden_master_comparison_performance(self, tmp_path):
        """Test performance of golden master comparisons."""
        # Create large simulation data
        large_data = {
            "events": [
                {"tick": i, "action": f"action_{i}", "data": {"value": i * 100}}
                for i in range(10000)
            ],
            "metrics": {f"metric_{i}": i * 1.5 for i in range(1000)}
        }
        
        golden_master = GoldenMasterTester(storage_dir=str(tmp_path))
        
        # Record baseline
        golden_master.record_golden_master(large_data, "performance_test")
        
        # Measure comparison time
        start_time = time.time()
        result = golden_master.compare_against_golden(large_data, "performance_test")
        elapsed = time.time() - start_time
        
        assert result.is_identical
        assert elapsed < 5.0  # Should complete within 5 seconds

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])