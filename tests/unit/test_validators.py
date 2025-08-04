"""
Unit tests for validation tools in FBA-Bench.

This module contains comprehensive tests for all validation components including
deterministic execution, statistical validation, reproducibility validation,
version control, and audit trail management.
"""

import pytest
import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import hashlib
import uuid
import threading
import time

from benchmarking.validators.deterministic import (
    DeterministicEnvironment, 
    EnvironmentState, 
    DeterministicContext
)
from benchmarking.validators.statistical_validator import (
    StatisticalValidator, 
    StatisticalSummary, 
    HypothesisTestResult
)
from benchmarking.validators.reproducibility_validator import (
    ReproducibilityValidator, 
    ReproducibilityReport, 
    ValidationResult
)
from benchmarking.validators.version_control import (
    VersionControlManager, 
    ComponentVersion, 
    VersionManifest
)
from benchmarking.validators.audit_trail import (
    AuditTrailManager, 
    AuditTrail, 
    AuditEvent
)


class TestEnvironmentState:
    """Test cases for EnvironmentState class."""
    
    def test_initialization(self):
        """Test EnvironmentState initialization."""
        state = EnvironmentState(
            random_seed=42,
            python_hash_seed=123
        )
        
        assert state.random_seed == 42
        assert state.python_hash_seed == 123
        assert isinstance(state.environment_variables, dict)
        assert isinstance(state.start_time, datetime)
        assert state.state_hash != ""  # Should be calculated
    
    def test_hash_calculation(self):
        """Test hash calculation consistency."""
        state1 = EnvironmentState(
            random_seed=42,
            python_hash_seed=123,
            environment_variables={"TEST_VAR": "test_value"}
        )
        
        state2 = EnvironmentState(
            random_seed=42,
            python_hash_seed=123,
            environment_variables={"TEST_VAR": "test_value"}
        )
        
        # Same state should produce same hash
        assert state1.state_hash == state2.state_hash
        
        # Different state should produce different hash
        state3 = EnvironmentState(
            random_seed=43,
            python_hash_seed=123,
            environment_variables={"TEST_VAR": "test_value"}
        )
        
        assert state1.state_hash != state3.state_hash


class TestDeterministicEnvironment:
    """Test cases for DeterministicEnvironment class."""
    
    def test_initialization(self):
        """Test DeterministicEnvironment initialization."""
        env = DeterministicEnvironment(seed=42)
        
        assert env.base_seed == 42
        assert not env.is_active
        assert env.current_state is None
    
    def test_initialization_with_random_seed(self):
        """Test DeterministicEnvironment initialization with random seed."""
        env = DeterministicEnvironment()
        
        assert env.base_seed is not None
        assert 0 <= env.base_seed < 2**32
    
    @patch('benchmarking.validators.deterministic.random.seed')
    @patch('benchmarking.validators.deterministic.os.environ')
    def test_activate(self, mock_environ, mock_seed):
        """Test environment activation."""
        env = DeterministicEnvironment(seed=42)
        
        # Mock environment capture
        with patch.object(env, '_capture_environment_state') as mock_capture:
            mock_state = EnvironmentState(random_seed=42, python_hash_seed=42)
            mock_capture.return_value = mock_state
            
            state = env.activate()
            
            assert env.is_active
            assert state == mock_state
            mock_seed.assert_called_with(42)
    
    @patch('benchmarking.validators.deterministic.random.seed')
    @patch('benchmarking.validators.deterministic.os.environ')
    def test_deactivate(self, mock_environ, mock_seed):
        """Test environment deactivation."""
        env = DeterministicEnvironment(seed=42)
        
        # Mock environment capture and restore
        with patch.object(env, '_capture_environment_state') as mock_capture, \
             patch.object(env, '_restore_environment_state') as mock_restore:
            
            initial_state = EnvironmentState(random_seed=0, python_hash_seed=0)
            final_state = EnvironmentState(random_seed=42, python_hash_seed=42)
            
            mock_capture.side_effect = [initial_state, final_state]
            
            env.activate()
            result = env.deactivate()
            
            assert not env.is_active
            assert env.current_state is None
            assert result == final_state
            mock_restore.assert_called_with(initial_state)
    
    def test_context_manager(self):
        """Test context manager functionality."""
        env = DeterministicEnvironment(seed=42)
        
        with patch.object(env, 'activate') as mock_activate, \
             patch.object(env, 'deactivate') as mock_deactivate:
            
            mock_state = EnvironmentState(random_seed=42, python_hash_seed=42)
            mock_activate.return_value = mock_state
            
            with env.context():
                pass
            
            mock_activate.assert_called_once()
            mock_deactivate.assert_called_once()
    
    def test_generate_derived_seeds(self):
        """Test derived seed generation."""
        env = DeterministicEnvironment(seed=42)
        
        seeds = env.generate_derived_seeds(5)
        
        assert len(seeds) == 5
        assert all(0 <= seed < 2**32 for seed in seeds)
        
        # Should be deterministic
        seeds2 = env.generate_derived_seeds(5)
        assert seeds == seeds2
    
    def test_validate_reproducibility(self):
        """Test reproducibility validation."""
        env = DeterministicEnvironment(seed=42)
        
        state1 = EnvironmentState(random_seed=42, python_hash_seed=42)
        state2 = EnvironmentState(random_seed=42, python_hash_seed=42)
        state3 = EnvironmentState(random_seed=43, python_hash_seed=42)
        
        assert env.validate_reproducibility(state1, state2) is True
        assert env.validate_reproducibility(state1, state3) is False
    
    def test_create_reproducible_config(self):
        """Test reproducible config creation."""
        env = DeterministicEnvironment(seed=42)
        
        config = {"param1": "value1", "param2": 123}
        reproducible_config = env.create_reproducible_config(config)
        
        assert "_reproducibility" in reproducible_config
        assert reproducible_config["_reproducibility"]["base_seed"] == 42
        assert "timestamp" in reproducible_config["_reproducibility"]
        
        # Original config should be preserved
        assert reproducible_config["param1"] == "value1"
        assert reproducible_config["param2"] == 123
    
    def test_load_reproducible_config(self):
        """Test reproducible config loading."""
        env = DeterministicEnvironment(seed=42)
        
        config = {
            "param1": "value1",
            "_reproducibility": {
                "base_seed": 123,
                "timestamp": "2023-01-01T00:00:00"
            }
        }
        
        result = env.load_reproducible_config(config)
        
        assert result is True
        assert env.base_seed == 123
    
    def test_load_reproducible_config_missing_info(self):
        """Test loading config without reproducibility info."""
        env = DeterministicEnvironment(seed=42)
        
        config = {"param1": "value1"}
        result = env.load_reproducible_config(config)
        
        assert result is False
    
    def test_get_environment_report(self):
        """Test environment report generation."""
        env = DeterministicEnvironment(seed=42)
        
        with patch('benchmarking.validators.deterministic.os.sys') as mock_sys:
            mock_sys.version = "3.9.0"
            mock_sys.platform = "linux"
            
            report = env.get_environment_report()
            
            assert report["base_seed"] == 42
            assert report["is_active"] is False
            assert report["python_version"] == "3.9.0"
            assert report["platform"] == "linux"


class TestDeterministicContext:
    """Test cases for DeterministicContext class."""
    
    def test_initialization(self):
        """Test DeterministicContext initialization."""
        context = DeterministicContext(base_seed=42)
        
        assert context.base_seed == 42
        assert len(context.get_context_stack()) == 0
    
    def test_initialization_with_random_seed(self):
        """Test DeterministicContext initialization with random seed."""
        context = DeterministicContext()
        
        assert context.base_seed is not None
        assert 0 <= context.base_seed < 2**32
    
    def test_push_context(self):
        """Test pushing a context."""
        context = DeterministicContext(base_seed=42)
        
        with patch.object(context._environment, 'activate') as mock_activate:
            seed = context.push_context("test_context")
            
            assert seed is not None
            stack = context.get_context_stack()
            assert len(stack) == 1
            assert stack[0]["name"] == "test_context"
            mock_activate.assert_called_once()
    
    def test_push_context_with_seed(self):
        """Test pushing a context with specific seed."""
        context = DeterministicContext(base_seed=42)
        
        with patch.object(context._environment, 'activate') as mock_activate:
            seed = context.push_context("test_context", seed=123)
            
            assert seed == 123
            stack = context.get_context_stack()
            assert stack[0]["seed"] == 123
    
    def test_pop_context(self):
        """Test popping a context."""
        context = DeterministicContext(base_seed=42)
        
        with patch.object(context._environment, 'activate') as mock_activate, \
             patch.object(context._environment, 'deactivate') as mock_deactivate:
            
            context.push_context("test1")
            context.push_context("test2")
            
            result = context.pop_context()
            
            assert result["name"] == "test2"
            stack = context.get_context_stack()
            assert len(stack) == 1
            assert stack[0]["name"] == "test1"
            
            # Should have deactivated and reactivated with parent context
            assert mock_deactivate.call_count == 1
            assert mock_activate.call_count == 3  # Initial + 2 pushes + 1 reactivate
    
    def test_pop_empty_context(self):
        """Test popping from empty context stack."""
        context = DeterministicContext(base_seed=42)
        
        result = context.pop_context()
        
        assert result is None
    
    def test_context_manager(self):
        """Test context manager functionality."""
        context = DeterministicContext(base_seed=42)
        
        with patch.object(context, 'push_context') as mock_push, \
             patch.object(context, 'pop_context') as mock_pop:
            
            mock_push.return_value = 123
            
            with context.context("test_context"):
                pass
            
            mock_push.assert_called_once_with("test_context", None)
            mock_pop.assert_called_once()
    
    def test_get_current_context(self):
        """Test getting current context."""
        context = DeterministicContext(base_seed=42)
        
        assert context.get_current_context() is None
        
        with patch.object(context._environment, 'activate'):
            context.push_context("test_context")
            
            current = context.get_current_context()
            assert current is not None
            assert current["name"] == "test_context"
    
    def test_generate_context_seed(self):
        """Test context seed generation."""
        context = DeterministicContext(base_seed=42)
        
        # Should be deterministic
        seed1 = context._generate_context_seed("test")
        seed2 = context._generate_context_seed("test")
        
        assert seed1 == seed2
        assert 0 <= seed1 < 2**32
        
        # Different names should produce different seeds
        seed3 = context._generate_context_seed("different")
        assert seed1 != seed3


class TestStatisticalSummary:
    """Test cases for StatisticalSummary class."""
    
    def test_initialization(self):
        """Test StatisticalSummary initialization."""
        summary = StatisticalSummary(
            mean=10.0,
            median=9.5,
            std_dev=2.0,
            variance=4.0,
            min_value=5.0,
            max_value=15.0,
            sample_size=10
        )
        
        assert summary.mean == 10.0
        assert summary.median == 9.5
        assert summary.std_dev == 2.0
        assert summary.variance == 4.0
        assert summary.min_value == 5.0
        assert summary.max_value == 15.0
        assert summary.sample_size == 10
        assert summary.confidence_interval is not None
        assert summary.confidence_level == 0.95
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        summary = StatisticalSummary(
            mean=10.0,
            median=9.5,
            std_dev=2.0,
            variance=4.0,
            min_value=5.0,
            max_value=15.0,
            sample_size=30
        )
        
        ci = summary.confidence_interval
        assert len(ci) == 2
        assert ci[0] < summary.mean < ci[1]
    
    def test_small_sample_confidence_interval(self):
        """Test confidence interval for small sample."""
        summary = StatisticalSummary(
            mean=10.0,
            median=9.5,
            std_dev=2.0,
            variance=4.0,
            min_value=5.0,
            max_value=15.0,
            sample_size=1
        )
        
        ci = summary.confidence_interval
        assert ci == (10.0, 10.0)
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        summary = StatisticalSummary(
            mean=10.0,
            median=9.5,
            std_dev=2.0,
            variance=4.0,
            min_value=5.0,
            max_value=15.0,
            sample_size=10
        )
        
        result = summary.to_dict()
        
        assert result["mean"] == 10.0
        assert result["median"] == 9.5
        assert result["std_dev"] == 2.0
        assert result["variance"] == 4.0
        assert result["min_value"] == 5.0
        assert result["max_value"] == 15.0
        assert result["sample_size"] == 10
        assert "confidence_interval" in result
        assert "margin_of_error" in result


class TestHypothesisTestResult:
    """Test cases for HypothesisTestResult class."""
    
    def test_initialization(self):
        """Test HypothesisTestResult initialization."""
        result = HypothesisTestResult(
            test_name="t-test",
            test_statistic=2.5,
            p_value=0.01,
            alpha=0.05,
            null_hypothesis="Means are equal",
            alternative_hypothesis="Means are not equal",
            conclusion="Reject null hypothesis"
        )
        
        assert result.test_name == "t-test"
        assert result.test_statistic == 2.5
        assert result.p_value == 0.01
        assert result.alpha == 0.05
        assert result.null_hypothesis == "Means are equal"
        assert result.alternative_hypothesis == "Means are not equal"
        assert result.conclusion == "Reject null hypothesis"
    
    def test_is_significant(self):
        """Test significance check."""
        # Significant result
        result1 = HypothesisTestResult(
            test_name="t-test",
            test_statistic=2.5,
            p_value=0.01,
            alpha=0.05,
            null_hypothesis="Means are equal",
            alternative_hypothesis="Means are not equal",
            conclusion="Reject null hypothesis"
        )
        assert result1.is_significant() is True
        
        # Non-significant result
        result2 = HypothesisTestResult(
            test_name="t-test",
            test_statistic=1.5,
            p_value=0.15,
            alpha=0.05,
            null_hypothesis="Means are equal",
            alternative_hypothesis="Means are not equal",
            conclusion="Fail to reject null hypothesis"
        )
        assert result2.is_significant() is False
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        result = HypothesisTestResult(
            test_name="t-test",
            test_statistic=2.5,
            p_value=0.01,
            alpha=0.05,
            null_hypothesis="Means are equal",
            alternative_hypothesis="Means are not equal",
            conclusion="Reject null hypothesis",
            effect_size=0.8
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["test_name"] == "t-test"
        assert result_dict["test_statistic"] == 2.5
        assert result_dict["p_value"] == 0.01
        assert result_dict["alpha"] == 0.05
        assert result_dict["significant"] is True
        assert result_dict["effect_size"] == 0.8


class TestStatisticalValidator:
    """Test cases for StatisticalValidator class."""
    
    def test_initialization(self):
        """Test StatisticalValidator initialization."""
        validator = StatisticalValidator(confidence_level=0.99)
        
        assert validator.confidence_level == 0.99
        assert validator.results_history == []
    
    def test_calculate_summary(self):
        """Test statistical summary calculation."""
        validator = StatisticalValidator()
        
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        summary = validator.calculate_summary(data)
        
        assert isinstance(summary, StatisticalSummary)
        assert summary.mean == 5.5
        assert summary.median == 5.5
        assert summary.sample_size == 10
        assert summary.min_value == 1.0
        assert summary.max_value == 10.0
    
    def test_calculate_summary_empty_data(self):
        """Test summary calculation with empty data."""
        validator = StatisticalValidator()
        
        with pytest.raises(ValueError, match="Data list cannot be empty"):
            validator.calculate_summary([])
    
    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        validator = StatisticalValidator()
        
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ci = validator.calculate_confidence_interval(data)
        
        assert len(ci) == 2
        assert ci[0] < ci[1]
    
    def test_calculate_confidence_interval_small_sample(self):
        """Test confidence interval with small sample."""
        validator = StatisticalValidator()
        
        ci = validator.calculate_confidence_interval([5.0])
        
        assert ci == (5.0, 5.0)
    
    def test_t_test_one_sample(self):
        """Test one-sample t-test."""
        validator = StatisticalValidator()
        
        data = [1, 2, 3, 4, 5]
        result = validator.t_test_one_sample(data, expected_mean=3.0)
        
        assert isinstance(result, HypothesisTestResult)
        assert result.test_name == "One-sample t-test"
        assert result.null_hypothesis == "Population mean = 3.0"
    
    def test_t_test_one_sample_insufficient_data(self):
        """Test one-sample t-test with insufficient data."""
        validator = StatisticalValidator()
        
        with pytest.raises(ValueError, match="Data must contain at least 2 values"):
            validator.t_test_one_sample([1], expected_mean=3.0)
    
    def test_t_test_two_samples(self):
        """Test two-sample t-test."""
        validator = StatisticalValidator()
        
        data1 = [1, 2, 3, 4, 5]
        data2 = [2, 3, 4, 5, 6]
        result = validator.t_test_two_samples(data1, data2)
        
        assert isinstance(result, HypothesisTestResult)
        assert result.test_name == "Two-sample t-test"
        assert result.null_hypothesis == "Population means are equal"
    
    def test_t_test_two_samples_insufficient_data(self):
        """Test two-sample t-test with insufficient data."""
        validator = StatisticalValidator()
        
        with pytest.raises(ValueError, match="Both samples must contain at least 2 values"):
            validator.t_test_two_samples([1], [2, 3])
    
    def test_paired_t_test(self):
        """Test paired t-test."""
        validator = StatisticalValidator()
        
        data1 = [1, 2, 3, 4, 5]
        data2 = [2, 3, 4, 5, 6]
        result = validator.paired_t_test(data1, data2)
        
        assert isinstance(result, HypothesisTestResult)
        assert result.test_name == "Paired t-test"
        assert result.null_hypothesis == "Mean difference = 0"
    
    def test_paired_t_test_different_lengths(self):
        """Test paired t-test with different length samples."""
        validator = StatisticalValidator()
        
        with pytest.raises(ValueError, match="Paired samples must have the same length"):
            validator.paired_t_test([1, 2, 3], [1, 2])
    
    def test_anova_test(self):
        """Test ANOVA test."""
        validator = StatisticalValidator()
        
        samples = {
            "group1": [1, 2, 3],
            "group2": [2, 3, 4],
            "group3": [3, 4, 5]
        }
        result = validator.anova_test(samples)
        
        assert isinstance(result, HypothesisTestResult)
        assert result.test_name == "One-way ANOVA"
        assert result.null_hypothesis == "All population means are equal"
    
    def test_anova_test_insufficient_groups(self):
        """Test ANOVA test with insufficient groups."""
        validator = StatisticalValidator()
        
        with pytest.raises(ValueError, match="ANOVA requires at least 2 sample groups"):
            validator.anova_test({"group1": [1, 2, 3]})
    
    def test_chi_square_test(self):
        """Test chi-square test."""
        validator = StatisticalValidator()
        
        observed = [10, 20, 30]
        expected = [15, 15, 30]
        result = validator.chi_square_test(observed, expected)
        
        assert isinstance(result, HypothesisTestResult)
        assert result.test_name == "Chi-square goodness-of-fit"
        assert result.null_hypothesis == "Observed frequencies match expected frequencies"
    
    def test_chi_square_test_different_lengths(self):
        """Test chi-square test with different length arrays."""
        validator = StatisticalValidator()
        
        with pytest.raises(ValueError, match="Observed and expected lists must have the same length"):
            validator.chi_square_test([10, 20], [10, 20, 30])
    
    def test_correlation_test(self):
        """Test correlation test."""
        validator = StatisticalValidator()
        
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        result = validator.correlation_test(x, y, method="pearson")
        
        assert isinstance(result, HypothesisTestResult)
        assert result.test_name == "Pearson correlation test"
        assert result.null_hypothesis == "No pearson correlation between variables"
    
    def test_correlation_test_different_lengths(self):
        """Test correlation test with different length arrays."""
        validator = StatisticalValidator()
        
        with pytest.raises(ValueError, match="x and y must have the same length"):
            validator.correlation_test([1, 2, 3], [1, 2])
    
    def test_correlation_test_insufficient_data(self):
        """Test correlation test with insufficient data."""
        validator = StatisticalValidator()
        
        with pytest.raises(ValueError, match="Correlation test requires at least 3 observations"):
            validator.correlation_test([1, 2], [2, 4])
    
    def test_correlation_test_unknown_method(self):
        """Test correlation test with unknown method."""
        validator = StatisticalValidator()
        
        with pytest.raises(ValueError, match="Unknown correlation method: unknown"):
            validator.correlation_test([1, 2, 3], [1, 2, 3], method="unknown")
    
    def test_normality_test(self):
        """Test normality test."""
        validator = StatisticalValidator()
        
        # Generate normal-looking data
        import numpy as np
        np.random.seed(42)
        data = np.random.normal(0, 1, 50).tolist()
        
        result = validator.normality_test(data, method="shapiro")
        
        assert isinstance(result, HypothesisTestResult)
        assert result.test_name == "Shapiro normality test"
        assert result.null_hypothesis == "Data follows a normal distribution"
    
    def test_normality_test_insufficient_data(self):
        """Test normality test with insufficient data."""
        validator = StatisticalValidator()
        
        with pytest.raises(ValueError, match="Normality test requires at least 3 observations"):
            validator.normality_test([1, 2])
    
    def test_normality_test_large_sample_shapiro(self):
        """Test Shapiro-Wilk test with large sample."""
        validator = StatisticalValidator()
        
        # Generate large sample
        import numpy as np
        np.random.seed(42)
        data = np.random.normal(0, 1, 6000).tolist()
        
        with pytest.raises(ValueError, match="Shapiro-Wilk test is not recommended for samples > 5000"):
            validator.normality_test(data, method="shapiro")
    
    def test_normality_test_unknown_method(self):
        """Test normality test with unknown method."""
        validator = StatisticalValidator()
        
        with pytest.raises(ValueError, match="Unknown normality test method: unknown"):
            validator.normality_test([1, 2, 3], method="unknown")
    
    def test_power_analysis_t_test(self):
        """Test power analysis for t-test."""
        validator = StatisticalValidator()
        
        result = validator.power_analysis(
            effect_size=0.5,
            alpha=0.05,
            power=0.8,
            test_type="t-test"
        )
        
        assert result["test_type"] == "t-test"
        assert result["effect_size"] == 0.5
        assert result["alpha"] == 0.05
        assert result["power"] == 0.8
        assert "required_sample_size" in result
        assert result["required_sample_size"] > 0
    
    def test_power_analysis_anova(self):
        """Test power analysis for ANOVA."""
        validator = StatisticalValidator()
        
        result = validator.power_analysis(
            effect_size=0.3,
            alpha=0.05,
            power=0.8,
            test_type="anova"
        )
        
        assert result["test_type"] == "anova"
        assert result["effect_size"] == 0.3
        assert "required_sample_size" in result
    
    def test_power_analysis_correlation(self):
        """Test power analysis for correlation."""
        validator = StatisticalValidator()
        
        result = validator.power_analysis(
            effect_size=0.5,
            alpha=0.05,
            power=0.8,
            test_type="correlation"
        )
        
        assert result["test_type"] == "correlation"
        assert result["effect_size"] == 0.5
        assert "required_sample_size" in result
    
    def test_power_analysis_unknown_test(self):
        """Test power analysis with unknown test type."""
        validator = StatisticalValidator()
        
        with pytest.raises(ValueError, match="Unknown test type: unknown"):
            validator.power_analysis(
                effect_size=0.5,
                alpha=0.05,
                power=0.8,
                test_type="unknown"
            )
    
    def test_validate_results(self):
        """Test results validation."""
        validator = StatisticalValidator()
        
        results = {
            "metric1": [1, 2, 3, 4, 5],
            "metric2": [10, 20, 30, 40, 50]
        }
        
        validation = validator.validate_results(results)
        
        assert "validation_timestamp" in validation
        assert "metrics" in validation
        assert "overall_validity" in validation
        assert "metric1" in validation["metrics"]
        assert "metric2" in validation["metrics"]
        
        # Check metric validation structure
        for metric_name, metric_data in validation["metrics"].items():
            assert "summary" in metric_data
            assert "is_normal" in metric_data
            assert "coefficient_of_variation" in metric_data
            assert "is_valid" in metric_data
            assert "validity_issues" in metric_data
    
    def test_validate_results_high_variability(self):
        """Test results validation with high variability."""
        validator = StatisticalValidator()
        
        # Data with high coefficient of variation
        results = {
            "metric1": [1, 100, 1, 100, 1]  # High variability
        }
        
        validation = validator.validate_results(results)
        
        assert validation["overall_validity"] is False
        assert not validation["metrics"]["metric1"]["is_valid"]
        assert "High variability detected" in validation["metrics"]["metric1"]["validity_issues"]
    
    def test_validate_results_small_sample(self):
        """Test results validation with small sample."""
        validator = StatisticalValidator()
        
        results = {
            "metric1": [1, 2]  # Small sample
        }
        
        validation = validator.validate_results(results)
        
        assert validation["overall_validity"] is False
        assert not validation["metrics"]["metric1"]["is_valid"]
        assert "Sample size too small" in validation["metrics"]["metric1"]["validity_issues"]


class TestComponentVersion:
    """Test cases for ComponentVersion class."""
    
    def test_initialization(self):
        """Test ComponentVersion initialization."""
        version = ComponentVersion(
            name="test_component",
            version="1.0.0",
            type="model",
            path="/path/to/component",
            hash="abc123"
        )
        
        assert version.name == "test_component"
        assert version.version == "1.0.0"
        assert version.type == "model"
        assert version.path == "/path/to/component"
        assert version.hash == "abc123"
        assert isinstance(version.metadata, dict)
        assert isinstance(version.timestamp, datetime)
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        version = ComponentVersion(
            name="test_component",
            version="1.0.0",
            type="model",
            path="/path/to/component",
            hash="abc123",
            metadata={"key": "value"}
        )
        
        result = version.to_dict()
        
        assert result["name"] == "test_component"
        assert result["version"] == "1.0.0"
        assert result["type"] == "model"
        assert result["path"] == "/path/to/component"
        assert result["hash"] == "abc123"
        assert result["metadata"] == {"key": "value"}
        assert "timestamp" in result
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "test_component",
            "version": "1.0.0",
            "type": "model",
            "path": "/path/to/component",
            "hash": "abc123",
            "metadata": {"key": "value"},
            "timestamp": "2023-01-01T00:00:00"
        }
        
        version = ComponentVersion.from_dict(data)
        
        assert version.name == "test_component"
        assert version.version == "1.0.0"
        assert version.type == "model"
        assert version.path == "/path/to/component"
        assert version.hash == "abc123"
        assert version.metadata == {"key": "value"}
        assert isinstance(version.timestamp, datetime)


class TestVersionManifest:
    """Test cases for VersionManifest class."""
    
    def test_initialization(self):
        """Test VersionManifest initialization."""
        manifest = VersionManifest(run_id="test_run")
        
        assert manifest.run_id == "test_run"
        assert isinstance(manifest.timestamp, datetime)
        assert manifest.components == []
        assert manifest.environment == {}
        assert manifest.git_info == {}
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        component = ComponentVersion(
            name="test_component",
            version="1.0.0",
            type="model",
            path="/path/to/component",
            hash="abc123"
        )
        
        manifest = VersionManifest(
            run_id="test_run",
            components=[component],
            environment={"key": "value"},
            git_info={"branch": "main"}
        )
        
        result = manifest.to_dict()
        
        assert result["run_id"] == "test_run"
        assert "timestamp" in result
        assert len(result["components"]) == 1
        assert result["components"][0]["name"] == "test_component"
        assert result["environment"] == {"key": "value"}
        assert result["git_info"] == {"branch": "main"}
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "run_id": "test_run",
            "timestamp": "2023-01-01T00:00:00",
            "components": [
                {
                    "name": "test_component",
                    "version": "1.0.0",
                    "type": "model",
                    "path": "/path/to/component",
                    "hash": "abc123",
                    "metadata": {},
                    "timestamp": "2023-01-01T00:00:00"
                }
            ],
            "environment": {"key": "value"},
            "git_info": {"branch": "main"}
        }
        
        manifest = VersionManifest.from_dict(data)
        
        assert manifest.run_id == "test_run"
        assert isinstance(manifest.timestamp, datetime)
        assert len(manifest.components) == 1
        assert manifest.components[0].name == "test_component"
        assert manifest.environment == {"key": "value"}
        assert manifest.git_info == {"branch": "main"}


class TestVersionControlManager:
    """Test cases for VersionControlManager class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = VersionControlManager(storage_path=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test VersionControlManager initialization."""
        assert self.manager.storage_path == Path(self.temp_dir)
        assert self.manager.get_current_manifest() is None
        assert self.manager._component_cache == {}
    
    def test_create_manifest(self):
        """Test manifest creation."""
        with patch.object(self.manager, '_capture_environment_info') as mock_env, \
             patch.object(self.manager, '_capture_git_info') as mock_git:
            
            mock_env.return_value = {"python_version": "3.9.0"}
            mock_git.return_value = {"branch": "main"}
            
            manifest = self.manager.create_manifest("test_run")
            
            assert manifest.run_id == "test_run"
            assert isinstance(manifest, VersionManifest)
            assert manifest.environment == {"python_version": "3.9.0"}
            assert manifest.git_info == {"branch": "main"}
            assert self.manager.get_current_manifest() == manifest
    
    def test_add_component(self):
        """Test adding a component."""
        self.manager.create_manifest("test_run")
        
        # Create a temporary file
        test_file = Path(self.temp_dir) / "test_file.txt"
        test_file.write_text("test content")
        
        component = self.manager.add_component(
            name="test_component",
            component_type="model",
            path=str(test_file),
            version="1.0.0",
            metadata={"key": "value"}
        )
        
        assert component.name == "test_component"
        assert component.type == "model"
        assert component.path == str(test_file)
        assert component.version == "1.0.0"
        assert component.metadata == {"key": "value"}
        
        # Component should be added to current manifest
        manifest = self.manager.get_current_manifest()
        assert len(manifest.components) == 1
        assert manifest.components[0] == component
    
    def test_add_component_no_manifest(self):
        """Test adding component without active manifest."""
        with pytest.raises(RuntimeError, match="No active manifest"):
            self.manager.add_component(
                name="test_component",
                component_type="model",
                path="/path/to/component"
            )
    
    def test_add_python_module(self):
        """Test adding Python module."""
        self.manager.create_manifest("test_run")
        
        with patch('benchmarking.validators.version_control.importlib.util.find_spec') as mock_find, \
             patch('benchmarking.validators.version_control.importlib.import_module') as mock_import:
            
            # Mock module spec
            mock_spec = Mock()
            mock_spec.origin = "/path/to/module.py"
            mock_spec.name = "test_module"
            mock_spec.submodule_search_locations = []
            mock_find.return_value = mock_spec
            
            # Mock imported module
            mock_module = Mock()
            mock_module.__version__ = "1.0.0"
            mock_import.return_value = mock_module
            
            component = self.manager.add_python_module("test_module", {"purpose": "testing"})
            
            assert component.name == "test_module"
            assert component.type == "python_module"
            assert component.version == "1.0.0"
            assert component.metadata["purpose"] == "testing"
            assert component.metadata["module_name"] == "test_module"
    
    def test_add_python_module_not_found(self):
        """Test adding non-existent Python module."""
        self.manager.create_manifest("test_run")
        
        with patch('benchmarking.validators.version_control.importlib.util.find_spec') as mock_find:
            mock_find.return_value = None
            
            with pytest.raises(ImportError, match="Module not found: non_existent_module"):
                self.manager.add_python_module("non_existent_module")
    
    def test_add_model(self):
        """Test adding a model."""
        self.manager.create_manifest("test_run")
        
        # Create a temporary file
        test_file = Path(self.temp_dir) / "test_model.pkl"
        test_file.write_text("model content")
        
        component = self.manager.add_model(
            model_path=str(test_file),
            model_name="test_model",
            metadata={"type": "neural_network"}
        )
        
        assert component.name == "test_model"
        assert component.type == "model"
        assert component.path == str(test_file)
        assert component.metadata["type"] == "neural_network"
    
    def test_add_dataset(self):
        """Test adding a dataset."""
        self.manager.create_manifest("test_run")
        
        # Create a temporary file
        test_file = Path(self.temp_dir) / "test_dataset.csv"
        test_file.write_text("dataset content")
        
        component = self.manager.add_dataset(
            dataset_path=str(test_file),
            dataset_name="test_dataset",
            metadata={"size": "1000_rows"}
        )
        
        assert component.name == "test_dataset"
        assert component.type == "dataset"
        assert component.path == str(test_file)
        assert component.metadata["size"] == "1000_rows"
    
    def test_add_configuration(self):
        """Test adding a configuration."""
        self.manager.create_manifest("test_run")
        
        # Create a temporary file
        test_file = Path(self.temp_dir) / "test_config.yaml"
        test_file.write_text("config content")
        
        component = self.manager.add_configuration(
            config_path=str(test_file),
            config_name="test_config",
            metadata={"environment": "production"}
        )
        
        assert component.name == "test_config"
        assert component.type == "config"
        assert component.path == str(test_file)
        assert component.metadata["environment"] == "production"
    
    def test_save_manifest(self):
        """Test manifest saving."""
        self.manager.create_manifest("test_run")
        
        # Add a component
        test_file = Path(self.temp_dir) / "test_file.txt"
        test_file.write_text("test content")
        
        self.manager.add_component(
            name="test_component",
            component_type="model",
            path=str(test_file)
        )
        
        # Save manifest
        manifest_path = self.manager.save_manifest()
        
        assert os.path.exists(manifest_path)
        assert manifest_path.endswith(".json")
        
        # Verify content
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        
        assert data["run_id"] == "test_run"
        assert len(data["components"]) == 1
        assert data["components"][0]["name"] == "test_component"
    
    def test_save_manifest_no_manifest(self):
        """Test saving without active manifest."""
        with pytest.raises(RuntimeError, match="No active manifest to save"):
            self.manager.save_manifest()
    
    def test_load_manifest(self):
        """Test manifest loading."""
        # Create and save a manifest first
        self.manager.create_manifest("test_run")
        
        test_file = Path(self.temp_dir) / "test_file.txt"
        test_file.write_text("test content")
        
        self.manager.add_component(
            name="test_component",
            component_type="model",
            path=str(test_file)
        )
        
        manifest_path = self.manager.save_manifest("test_manifest.json")
        
        # Create new manager and load manifest
        new_manager = VersionControlManager(storage_path=self.temp_dir)
        loaded_manifest = new_manager.load_manifest("test_manifest.json")
        
        assert loaded_manifest.run_id == "test_run"
        assert len(loaded_manifest.components) == 1
        assert loaded_manifest.components[0].name == "test_component"
        assert new_manager.get_current_manifest() == loaded_manifest
    
    def test_list_manifests(self):
        """Test listing manifests."""
        # Create and save multiple manifests
        self.manager.create_manifest("run1")
        self.manager.save_manifest("manifest1.json")
        
        self.manager.create_manifest("run2")
        self.manager.save_manifest("manifest2.json")
        
        manifests = self.manager.list_manifests()
        
        assert len(manifests) == 2
        assert "manifest1.json" in manifests
        assert "manifest2.json" in manifests
    
    def test_compare_manifests(self):
        """Test manifest comparison."""
        # Create first manifest
        self.manager.create_manifest("run1")
        
        test_file1 = Path(self.temp_dir) / "test_file1.txt"
        test_file1.write_text("content1")
        
        self.manager.add_component(
            name="component1",
            component_type="model",
            path=str(test_file1),
            version="1.0.0"
        )
        
        manifest1 = self.manager.get_current_manifest()
        
        # Create second manager and manifest
        manager2 = VersionControlManager(storage_path=self.temp_dir)
        manager2.create_manifest("run2")
        
        test_file2 = Path(self.temp_dir) / "test_file2.txt"
        test_file2.write_text("content2")
        
        manager2.add_component(
            name="component1",
            component_type="model",
            path=str(test_file2),
            version="2.0.0"  # Different version
        )
        
        manager2.add_component(
            name="component2",  # New component
            component_type="dataset",
            path=str(test_file2),
            version="1.0.0"
        )
        
        manifest2 = manager2.get_current_manifest()
        
        # Compare manifests
        comparison = self.manager.compare_manifests(manifest1, manifest2)
        
        assert comparison["run_ids"] == ["run1", "run2"]
        assert "differences" in comparison
        assert "components" in comparison["differences"]
        
        # Check component differences
        component_diffs = comparison["differences"]["components"]
        assert len(component_diffs["changed"]) == 1  # component1 changed
        assert len(component_diffs["added"]) == 1     # component2 added
        assert len(component_diffs["removed"]) == 0    # No components removed
    
    def test_verify_reproducibility(self):
        """Test reproducibility verification."""
        # Create reference manifest
        self.manager.create_manifest("reference_run")
        
        test_file = Path(self.temp_dir) / "test_file.txt"
        test_file.write_text("test content")
        
        self.manager.add_component(
            name="test_component",
            component_type="model",
            path=str(test_file),
            version="1.0.0"
        )
        
        reference_manifest = self.manager.get_current_manifest()
        
        # Verify reproducibility
        verification = self.manager.verify_reproducibility(reference_manifest)
        
        assert verification["reference_run_id"] == "reference_run"
        assert "results" in verification
        assert "components" in verification["results"]
        assert "overall_reproducible" in verification
        
        # Should be reproducible since we're using the same files
        assert verification["overall_reproducible"] is True
        assert verification["results"]["components"]["test_component"]["status"] == "reproducible"
    
    def test_verify_reproducibility_modified_file(self):
        """Test reproducibility verification with modified file."""
        # Create reference manifest
        self.manager.create_manifest("reference_run")
        
        test_file = Path(self.temp_dir) / "test_file.txt"
        test_file.write_text("original content")
        
        self.manager.add_component(
            name="test_component",
            component_type="model",
            path=str(test_file),
            version="1.0.0"
        )
        
        reference_manifest = self.manager.get_current_manifest()
        
        # Modify the file
        test_file.write_text("modified content")
        
        # Verify reproducibility
        verification = self.manager.verify_reproducibility(reference_manifest)
        
        # Should not be reproducible due to modified file
        assert verification["overall_reproducible"] is False
        assert verification["results"]["components"]["test_component"]["status"] == "modified"
    
    def test_verify_reproducibility_missing_file(self):
        """Test reproducibility verification with missing file."""
        # Create reference manifest
        self.manager.create_manifest("reference_run")
        
        test_file = Path(self.temp_dir) / "test_file.txt"
        test_file.write_text("test content")
        
        self.manager.add_component(
            name="test_component",
            component_type="model",
            path=str(test_file),
            version="1.0.0"
        )
        
        reference_manifest = self.manager.get_current_manifest()
        
        # Remove the file
        test_file.unlink()
        
        # Verify reproducibility
        verification = self.manager.verify_reproducibility(reference_manifest)
        
        # Should not be reproducible due to missing file
        assert verification["overall_reproducible"] is False
        assert verification["results"]["components"]["test_component"]["status"] == "missing"
    
    def test_calculate_component_hash_file(self):
        """Test hash calculation for file."""
        # Create a test file
        test_file = Path(self.temp_dir) / "test_file.txt"
        test_file.write_text("test content")
        
        hash_value = self.manager._calculate_component_hash(str(test_file))
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hash length
        
        # Same content should produce same hash
        hash_value2 = self.manager._calculate_component_hash(str(test_file))
        assert hash_value == hash_value2
    
    def test_calculate_component_hash_directory(self):
        """Test hash calculation for directory."""
        # Create a test directory with files
        test_dir = Path(self.temp_dir) / "test_dir"
        test_dir.mkdir()
        
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")
        
        hash_value = self.manager._calculate_component_hash(str(test_dir))
        
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hash length
        
        # Same content should produce same hash
        hash_value2 = self.manager._calculate_component_hash(str(test_dir))
        assert hash_value == hash_value2
    
    def test_calculate_component_hash_nonexistent(self):
        """Test hash calculation for non-existent path."""
        with pytest.raises(FileNotFoundError, match="Component not found"):
            self.manager._calculate_component_hash("/nonexistent/path")
    
    def test_capture_environment_info(self):
        """Test environment info capture."""
        with patch('benchmarking.validators.version_control.sys') as mock_sys, \
             patch('benchmarking.validators.version_control.os') as mock_os, \
             patch('benchmarking.validators.version_control.platform') as mock_platform:
            
            mock_sys.version = "3.9.0"
            mock_sys.platform = "linux"
            mock_os.environ = {"PATH": "/usr/bin", "HOME": "/home/user"}
            mock_platform.platform.return_value = "Linux-5.4.0-x86_64"
            mock_platform.machine.return_value = "x86_64"
            mock_platform.processor.return_value = "x86_64"
            mock_platform.python_version.return_value = "3.9.0"
            mock_platform.python_implementation.return_value = "CPython"
            
            env_info = self.manager._capture_environment_info()
            
            assert env_info["python_version"] == "3.9.0"
            assert env_info["platform"] == "linux"
            assert env_info["system"] == "Linux-5.4.0-x86_64"
            assert env_info["architecture"] == "x86_64"
            assert env_info["processor"] == "x86_64"
            assert env_info["python_implementation"] == "CPython"
            assert env_info["environment_variables"] == {"PATH": "/usr/bin", "HOME": "/home/user"}
    
    def test_capture_git_info(self):
        """Test git info capture."""
        with patch('benchmarking.validators.version_control.subprocess.run') as mock_run:
            # Mock git commands
            def mock_subprocess(command, **kwargs):
                result = Mock()
                if "rev-parse" in command:
                    result.stdout = "abc123def456"
                elif "branch" in command:
                    result.stdout = "main"
                elif "log" in command:
                    result.stdout = "Commit message"
                elif "status" in command:
                    result.stdout = ""
                return result
            
            mock_run.side_effect = mock_subprocess
            
            git_info = self.manager._capture_git_info()
            
            assert git_info["commit_hash"] == "abc123def456"
            assert git_info["branch"] == "main"
            assert git_info["commit_message"] == "Commit message"
            assert git_info["is_clean"] is True
    
    def test_capture_git_info_no_git(self):
        """Test git info capture when git is not available."""
        with patch('benchmarking.validators.version_control.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            
            git_info = self.manager._capture_git_info()
            
            assert git_info["error"] == "Git not available"
            assert "commit_hash" not in git_info
            assert "branch" not in git_info


class TestAuditEvent:
    """Test cases for AuditEvent class."""
    
    def test_initialization(self):
        """Test AuditEvent initialization."""
        event = AuditEvent(
            run_id="test_run",
            event_type="operation",
            component="test_component",
            action="test_action",
            severity="info",
            user="test_user",
            session_id="test_session",
            details={"key": "value"}
        )
        
        assert event.run_id == "test_run"
        assert event.event_type == "operation"
        assert event.component == "test_component"
        assert event.action == "test_action"
        assert event.severity == "info"
        assert event.user == "test_user"
        assert event.session_id == "test_session"
        assert event.details == {"key": "value"}
        assert isinstance(event.timestamp, datetime)
        assert isinstance(event.event_id, str)
        assert len(event.event_id) > 0
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        event = AuditEvent(
            run_id="test_run",
            event_type="operation",
            component="test_component",
            action="test_action",
            severity="info",
            user="test_user",
            session_id="test_session",
            details={"key": "value"}
        )
        
        result = event.to_dict()
        
        assert result["run_id"] == "test_run"
        assert result["event_type"] == "operation"
        assert result["component"] == "test_component"
        assert result["action"] == "test_action"
        assert result["severity"] == "info"
        assert result["user"] == "test_user"
        assert result["session_id"] == "test_session"
        assert result["details"] == {"key": "value"}
        assert "timestamp" in result
        assert "event_id" in result
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "run_id": "test_run",
            "event_type": "operation",
            "component": "test_component",
            "action": "test_action",
            "severity": "info",
            "user": "test_user",
            "session_id": "test_session",
            "details": {"key": "value"},
            "timestamp": "2023-01-01T00:00:00",
            "event_id": "test_event_id"
        }
        
        event = AuditEvent.from_dict(data)
        
        assert event.run_id == "test_run"
        assert event.event_type == "operation"
        assert event.component == "test_component"
        assert event.action == "test_action"
        assert event.severity == "info"
        assert event.user == "test_user"
        assert event.session_id == "test_session"
        assert event.details == {"key": "value"}
        assert isinstance(event.timestamp, datetime)
        assert event.event_id == "test_event_id"


class TestAuditTrail:
    """Test cases for AuditTrail class."""
    
    def test_initialization(self):
        """Test AuditTrail initialization."""
        trail = AuditTrail(run_id="test_run")
        
        assert trail.run_id == "test_run"
        assert isinstance(trail.start_time, datetime)
        assert trail.end_time is None
        assert trail.events == []
        assert trail.metadata == {}
        assert trail.checksum == ""
    
    def test_add_event(self):
        """Test adding an event."""
        trail = AuditTrail(run_id="test_run")
        
        event = AuditEvent(
            run_id="test_run",
            event_type="operation",
            component="test_component",
            action="test_action"
        )
        
        trail.add_event(event)
        
        assert len(trail.events) == 1
        assert trail.events[0] == event
        assert trail.checksum != ""  # Should be recalculated
    
    def test_add_event_wrong_run_id(self):
        """Test adding event with wrong run ID."""
        trail = AuditTrail(run_id="test_run")
        
        event = AuditEvent(
            run_id="wrong_run",
            event_type="operation",
            component="test_component",
            action="test_action"
        )
        
        with pytest.raises(ValueError, match="Event run_id does not match trail run_id"):
            trail.add_event(event)
    
    def test_end(self):
        """Test ending the trail."""
        trail = AuditTrail(run_id="test_run")
        
        trail.end()
        
        assert trail.end_time is not None
        assert isinstance(trail.end_time, datetime)
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        trail = AuditTrail(run_id="test_run")
        
        event = AuditEvent(
            run_id="test_run",
            event_type="operation",
            component="test_component",
            action="test_action"
        )
        
        trail.add_event(event)
        trail.end()
        
        result = trail.to_dict()
        
        assert result["run_id"] == "test_run"
        assert "start_time" in result
        assert "end_time" in result
        assert len(result["events"]) == 1
        assert result["events"][0]["event_type"] == "operation"
        assert result["metadata"] == {}
        assert "checksum" in result
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "run_id": "test_run",
            "start_time": "2023-01-01T00:00:00",
            "end_time": "2023-01-01T01:00:00",
            "events": [
                {
                    "run_id": "test_run",
                    "event_type": "operation",
                    "component": "test_component",
                    "action": "test_action",
                    "severity": "info",
                    "user": "system",
                    "session_id": "",
                    "details": {},
                    "timestamp": "2023-01-01T00:30:00",
                    "event_id": "test_event_id"
                }
            ],
            "metadata": {"key": "value"},
            "checksum": "abc123"
        }
        
        trail = AuditTrail.from_dict(data)
        
        assert trail.run_id == "test_run"
        assert isinstance(trail.start_time, datetime)
        assert isinstance(trail.end_time, datetime)
        assert len(trail.events) == 1
        assert trail.events[0].event_type == "operation"
        assert trail.metadata == {"key": "value"}
        assert trail.checksum == "abc123"
    
    def test_calculate_checksum(self):
        """Test checksum calculation."""
        trail = AuditTrail(run_id="test_run")
        
        event = AuditEvent(
            run_id="test_run",
            event_type="operation",
            component="test_component",
            action="test_action"
        )
        
        trail.add_event(event)
        
        checksum = trail._calculate_checksum()
        
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hash length
        
        # Same content should produce same checksum
        checksum2 = trail._calculate_checksum()
        assert checksum == checksum2


class TestAuditTrailManager:
    """Test cases for AuditTrailManager class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = AuditTrailManager(storage_path=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test AuditTrailManager initialization."""
        assert self.manager.storage_path == Path(self.temp_dir)
        assert self.manager._active_trails == {}
        assert self.manager._event_buffer == []
    
    def test_start_trail(self):
        """Test starting a trail."""
        trail = self.manager.start_trail("test_run", {"key": "value"})
        
        assert isinstance(trail, AuditTrail)
        assert trail.run_id == "test_run"
        assert trail.metadata == {"key": "value"}
        assert "test_run" in self.manager._active_trails
        assert self.manager._active_trails["test_run"] == trail
    
    def test_start_trail_existing(self):
        """Test starting a trail that already exists."""
        self.manager.start_trail("test_run")
        
        with pytest.raises(ValueError, match="Audit trail already exists for run"):
            self.manager.start_trail("test_run")
    
    def test_end_trail(self):
        """Test ending a trail."""
        trail = self.manager.start_trail("test_run")
        
        result = self.manager.end_trail("test_run")
        
        assert result == trail
        assert trail.end_time is not None
        assert "test_run" not in self.manager._active_trails
    
    def test_end_trail_nonexistent(self):
        """Test ending a non-existent trail."""
        with pytest.raises(ValueError, match="No active audit trail for run"):
            self.manager.end_trail("nonexistent_run")
    
    def test_log_event(self):
        """Test logging an event."""
        self.manager.start_trail("test_run")
        
        event = self.manager.log_event(
            run_id="test_run",
            component="test_component",
            action="test_action",
            event_type="operation",
            details={"key": "value"},
            severity="info",
            user="test_user",
            session_id="test_session"
        )
        
        assert isinstance(event, AuditEvent)
        assert event.run_id == "test_run"
        assert event.component == "test_component"
        assert event.action == "test_action"
        assert event.event_type == "operation"
        assert event.details == {"key": "value"}
        assert event.severity == "info"
        assert event.user == "test_user"
        assert event.session_id == "test_session"
        
        # Event should be added to trail
        trail = self.manager._active_trails["test_run"]
        assert len(trail.events) == 1
        assert trail.events[0] == event
    
    def test_log_event_no_trail(self):
        """Test logging event without active trail."""
        with pytest.raises(ValueError, match="No active audit trail for run"):
            self.manager.log_event(
                run_id="nonexistent_run",
                component="test_component",
                action="test_action"
            )
    
    def test_log_event_with_buffer(self):
        """Test logging event with buffering."""
        self.manager.start_trail("test_run")
        
        # Log multiple events
        for i in range(5):
            self.manager.log_event(
                run_id="test_run",
                component="test_component",
                action=f"test_action_{i}",
                event_type="operation"
            )
        
        # All events should be in the trail
        trail = self.manager._active_trails["test_run"]
        assert len(trail.events) == 5
    
    def test_get_trail(self):
        """Test getting a trail."""
        trail = self.manager.start_trail("test_run")
        
        result = self.manager.get_trail("test_run")
        
        assert result == trail
    
    def test_get_trail_nonexistent(self):
        """Test getting a non-existent trail."""
        result = self.manager.get_trail("nonexistent_run")
        
        assert result is None
    
    def test_save_trail(self):
        """Test saving a trail."""
        trail = self.manager.start_trail("test_run")
        
        # Add some events
        for i in range(3):
            self.manager.log_event(
                run_id="test_run",
                component="test_component",
                action=f"test_action_{i}",
                event_type="operation"
            )
        
        self.manager.end_trail("test_run")
        
        # Save trail
        trail_path = self.manager.save_trail(trail)
        
        assert os.path.exists(trail_path)
        assert trail_path.endswith(".json")
        
        # Verify content
        with open(trail_path, 'r') as f:
            data = json.load(f)
        
        assert data["run_id"] == "test_run"
        assert len(data["events"]) == 3
        assert data["events"][0]["action"] == "test_action_0"
    
    def test_load_trail(self):
        """Test loading a trail."""
        # Create and save a trail first
        trail = self.manager.start_trail("test_run")
        
        for i in range(3):
            self.manager.log_event(
                run_id="test_run",
                component="test_component",
                action=f"test_action_{i}",
                event_type="operation"
            )
        
        self.manager.end_trail("test_run")
        
        trail_path = self.manager.save_trail(trail, "test_trail.json")
        
        # Load trail
        loaded_trail = self.manager.load_trail("test_trail.json")
        
        assert loaded_trail.run_id == "test_run"
        assert len(loaded_trail.events) == 3
        assert loaded_trail.events[0].action == "test_action_0"
        assert loaded_trail.events[1].action == "test_action_1"
        assert loaded_trail.events[2].action == "test_action_2"
    
    def test_list_trails(self):
        """Test listing trails."""
        # Create and save multiple trails
        for i in range(3):
            trail = self.manager.start_trail(f"test_run_{i}")
            
            for j in range(2):
                self.manager.log_event(
                    run_id=f"test_run_{i}",
                    component="test_component",
                    action=f"test_action_{j}",
                    event_type="operation"
                )
            
            self.manager.end_trail(f"test_run_{i}")
            self.manager.save_trail(trail, f"trail_{i}.json")
        
        trails = self.manager.list_trails()
        
        assert len(trails) == 3
        assert "trail_0.json" in trails
        assert "trail_1.json" in trails
        assert "trail_2.json" in trails
    
    def test_get_trail_report(self):
        """Test getting trail report."""
        trail = self.manager.start_trail("test_run")
        
        # Add various events
        self.manager.log_event(
            run_id="test_run",
            component="component1",
            action="action1",
            event_type="operation",
            severity="info"
        )
        
        self.manager.log_event(
            run_id="test_run",
            component="component2",
            action="action2",
            event_type="error",
            severity="error"
        )
        
        self.manager.log_event(
            run_id="test_run",
            component="component1",
            action="action3",
            event_type="operation",
            severity="warning"
        )
        
        self.manager.end_trail("test_run")
        
        report = self.manager.get_trail_report(trail)
        
        assert report["run_id"] == "test_run"
        assert report["total_events"] == 3
        assert report["event_counts"]["operation"] == 2
        assert report["event_counts"]["error"] == 1
        assert report["severity_counts"]["info"] == 1
        assert report["severity_counts"]["error"] == 1
        assert report["severity_counts"]["warning"] == 1
        assert report["component_counts"]["component1"] == 2
        assert report["component_counts"]["component2"] == 1
        assert report["duration_seconds"] is not None
    
    def test_verify_trail_integrity(self):
        """Test trail integrity verification."""
        trail = self.manager.start_trail("test_run")
        
        # Add events
        for i in range(3):
            self.manager.log_event(
                run_id="test_run",
                component="test_component",
                action=f"test_action_{i}",
                event_type="operation"
            )
        
        self.manager.end_trail("test_run")
        
        # Verify integrity
        assert self.manager.verify_trail_integrity(trail) is True
    
    def test_verify_trail_integrity_checksum_mismatch(self):
        """Test trail integrity verification with checksum mismatch."""
        trail = self.manager.start_trail("test_run")
        
        # Add events
        for i in range(3):
            self.manager.log_event(
                run_id="test_run",
                component="test_component",
                action=f"test_action_{i}",
                event_type="operation"
            )
        
        self.manager.end_trail("test_run")
        
        # Tamper with checksum
        trail.checksum = "wrong_checksum"
        
        # Verify integrity
        assert self.manager.verify_trail_integrity(trail) is False
    
    def test_verify_trail_integrity_timestamp_out_of_order(self):
        """Test trail integrity verification with out-of-order timestamps."""
        trail = self.manager.start_trail("test_run")
        
        # Add events
        for i in range(3):
            self.manager.log_event(
                run_id="test_run",
                component="test_component",
                action=f"test_action_{i}",
                event_type="operation"
            )
        
        self.manager.end_trail("test_run")
        
        # Tamper with timestamps
        trail.events[1].timestamp = trail.events[0].timestamp - timedelta(seconds=1)
        
        # Verify integrity
        assert self.manager.verify_trail_integrity(trail) is False
    
    def test_verify_trail_integrity_wrong_run_id(self):
        """Test trail integrity verification with wrong run ID."""
        trail = self.manager.start_trail("test_run")
        
        # Add events
        for i in range(3):
            self.manager.log_event(
                run_id="test_run",
                component="test_component",
                action=f"test_action_{i}",
                event_type="operation"
            )
        
        self.manager.end_trail("test_run")
        
        # Tamper with event run ID
        trail.events[0].run_id = "wrong_run"
        
        # Verify integrity
        assert self.manager.verify_trail_integrity(trail) is False
    
    def test_export_trail_json(self):
        """Test trail export to JSON."""
        trail = self.manager.start_trail("test_run")
        
        # Add events
        for i in range(3):
            self.manager.log_event(
                run_id="test_run",
                component="test_component",
                action=f"test_action_{i}",
                event_type="operation"
            )
        
        self.manager.end_trail("test_run")
        
        # Export trail
        export_path = self.manager.export_trail("test_run", format="json", filename="test_export.json")
        
        assert os.path.exists(export_path)
        assert export_path.endswith("test_export.json")
        
        # Verify content
        with open(export_path, 'r') as f:
            data = json.load(f)
        
        assert data["run_id"] == "test_run"
        assert len(data["events"]) == 3
    
    def test_export_trail_csv(self):
        """Test trail export to CSV."""
        trail = self.manager.start_trail("test_run")
        
        # Add events
        for i in range(3):
            self.manager.log_event(
                run_id="test_run",
                component="test_component",
                action=f"test_action_{i}",
                event_type="operation"
            )
        
        self.manager.end_trail("test_run")
        
        # Export trail
        export_path = self.manager.export_trail("test_run", format="csv", filename="test_export.csv")
        
        assert os.path.exists(export_path)
        assert export_path.endswith("test_export.csv")
        
        # Verify content
        with open(export_path, 'r') as f:
            lines = f.readlines()
        
        # Should have header + 3 event lines
        assert len(lines) == 4
        assert "timestamp,event_id,event_type,component,action,severity,user,session_id,details" in lines[0]
    
    def test_export_trail_unknown_format(self):
        """Test trail export with unknown format."""
        trail = self.manager.start_trail("test_run")
        
        # Add events
        for i in range(3):
            self.manager.log_event(
                run_id="test_run",
                component="test_component",
                action=f"test_action_{i}",
                event_type="operation"
            )
        
        self.manager.end_trail("test_run")
        
        # Export trail with unknown format
        result = self.manager.export_trail("test_run", format="unknown")
        
        assert result is None
    
    def test_export_trail_nonexistent(self):
        """Test exporting non-existent trail."""
        result = self.manager.export_trail("nonexistent_run")
        
        assert result is None
    
    def test_audit_context_success(self):
        """Test audit context manager with successful operation."""
        self.manager.start_trail("test_run")
        
        with self.manager.audit_context(
            run_id="test_run",
            component="test_component",
            action="test_operation",
            user="test_user",
            session_id="test_session"
        ):
            pass  # Successful operation
        
        trail = self.manager.get_trail("test_run")
        
        # Should have start and complete events
        assert len(trail.events) == 2
        assert trail.events[0].action == "test_operation_start"
        assert trail.events[1].action == "test_operation_complete"
        assert trail.events[0].details["status"] == "started"
        assert trail.events[1].details["status"] == "completed"
    
    def test_audit_context_error(self):
        """Test audit context manager with error."""
        self.manager.start_trail("test_run")
        
        try:
            with self.manager.audit_context(
                run_id="test_run",
                component="test_component",
                action="test_operation",
                user="test_user",
                session_id="test_session"
            ):
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected error
        
        trail = self.manager.get_trail("test_run")
        
        # Should have start and error events
        assert len(trail.events) == 2
        assert trail.events[0].action == "test_operation_start"
        assert trail.events[1].action == "test_operation_error"
        assert trail.events[0].details["status"] == "started"
        assert trail.events[1].details["status"] == "error"
        assert trail.events[1].details["error"] == "Test error"
        assert trail.events[1].severity == "error"


class TestReproducibilityValidator:
    """Test cases for ReproducibilityValidator class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = ReproducibilityValidator(storage_path=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ReproducibilityValidator initialization."""
        assert self.validator.storage_path == Path(self.temp_dir)
        assert self.validator.validation_history == []
    
    def test_validate_reproducibility(self):
        """Test reproducibility validation."""
        # Create reference results
        reference_results = {
            "metric1": [1, 2, 3, 4, 5],
            "metric2": [10, 20, 30, 40, 50]
        }
        
        # Create current results (same as reference)
        current_results = {
            "metric1": [1, 2, 3, 4, 5],
            "metric2": [10, 20, 30, 40, 50]
        }
        
        report = self.validator.validate_reproducibility(
            reference_results=reference_results,
            current_results=current_results,
            tolerance=0.01
        )
        
        assert isinstance(report, ReproducibilityReport)
        assert report.reference_run_id is None
        assert report.current_run_id is None
        assert report.overall_reproducible is True
        assert "metric1" in report.metric_results
        assert "metric2" in report.metric_results
        assert report.metric_results["metric1"]["reproducible"] is True
        assert report.metric_results["metric2"]["reproducible"] is True
    
    def test_validate_reproducibility_with_ids(self):
        """Test reproducibility validation with run IDs."""
        reference_results = {"metric1": [1, 2, 3, 4, 5]}
        current_results = {"metric1": [1, 2, 3, 4, 5]}
        
        report = self.validator.validate_reproducibility(
            reference_results=reference_results,
            current_results=current_results,
            reference_run_id="ref_run",
            current_run_id="curr_run"
        )
        
        assert report.reference_run_id == "ref_run"
        assert report.current_run_id == "curr_run"
    
    def test_validate_reproducibility_not_reproducible(self):
        """Test reproducibility validation with non-reproducible results."""
        reference_results = {
            "metric1": [1, 2, 3, 4, 5],
            "metric2": [10, 20, 30, 40, 50]
        }
        
        # Create current results (different from reference)
        current_results = {
            "metric1": [10, 20, 30, 40, 50],  # Different values
            "metric2": [10, 20, 30, 40, 50]
        }
        
        report = self.validator.validate_reproducibility(
            reference_results=reference_results,
            current_results=current_results,
            tolerance=0.01
        )
        
        assert report.overall_reproducible is False
        assert report.metric_results["metric1"]["reproducible"] is False
        assert report.metric_results["metric2"]["reproducible"] is True
    
    def test_validate_reproducibility_missing_metric(self):
        """Test reproducibility validation with missing metric."""
        reference_results = {
            "metric1": [1, 2, 3, 4, 5],
            "metric2": [10, 20, 30, 40, 50]
        }
        
        # Create current results (missing metric2)
        current_results = {
            "metric1": [1, 2, 3, 4, 5]
        }
        
        report = self.validator.validate_reproducibility(
            reference_results=reference_results,
            current_results=current_results,
            tolerance=0.01
        )
        
        assert report.overall_reproducible is False
        assert report.metric_results["metric1"]["reproducible"] is True
        assert report.metric_results["metric2"]["reproducible"] is False
        assert "Missing in current results" in report.metric_results["metric2"]["issues"]
    
    def test_validate_reproducibility_extra_metric(self):
        """Test reproducibility validation with extra metric."""
        reference_results = {
            "metric1": [1, 2, 3, 4, 5]
        }
        
        # Create current results (extra metric2)
        current_results = {
            "metric1": [1, 2, 3, 4, 5],
            "metric2": [10, 20, 30, 40, 50]
        }
        
        report = self.validator.validate_reproducibility(
            reference_results=reference_results,
            current_results=current_results,
            tolerance=0.01
        )
        
        assert report.overall_reproducible is False
        assert report.metric_results["metric1"]["reproducible"] is True
        assert report.metric_results["metric2"]["reproducible"] is False
        assert "Missing in reference results" in report.metric_results["metric2"]["issues"]
    
    def test_validate_reproducibility_different_lengths(self):
        """Test reproducibility validation with different length arrays."""
        reference_results = {
            "metric1": [1, 2, 3, 4, 5]
        }
        
        # Create current results (different length)
        current_results = {
            "metric1": [1, 2, 3, 4, 5, 6]
        }
        
        report = self.validator.validate_reproducibility(
            reference_results=reference_results,
            current_results=current_results,
            tolerance=0.01
        )
        
        assert report.overall_reproducible is False
        assert report.metric_results["metric1"]["reproducible"] is False
        assert "Different array lengths" in report.metric_results["metric1"]["issues"]
    
    def test_validate_reproducibility_empty_results(self):
        """Test reproducibility validation with empty results."""
        with pytest.raises(ValueError, match="Results cannot be empty"):
            self.validator.validate_reproducibility(
                reference_results={},
                current_results={"metric1": [1, 2, 3]}
            )
    
    def test_validate_reproducibility_invalid_tolerance(self):
        """Test reproducibility validation with invalid tolerance."""
        with pytest.raises(ValueError, match="Tolerance must be non-negative"):
            self.validator.validate_reproducibility(
                reference_results={"metric1": [1, 2, 3]},
                current_results={"metric1": [1, 2, 3]},
                tolerance=-0.01
            )
    
    def test_validate_reproducibility_invalid_data(self):
        """Test reproducibility validation with invalid data."""
        with pytest.raises(ValueError, match="Reference results must be a dictionary"):
            self.validator.validate_reproducibility(
                reference_results="invalid",
                current_results={"metric1": [1, 2, 3]}
            )
    
    def test_validate_reproducibility_invalid_current_data(self):
        """Test reproducibility validation with invalid current data."""
        with pytest.raises(ValueError, match="Current results must be a dictionary"):
            self.validator.validate_reproducibility(
                reference_results={"metric1": [1, 2, 3]},
                current_results="invalid"
            )
    
    def test_validate_reproducibility_invalid_metric_data(self):
        """Test reproducibility validation with invalid metric data."""
        with pytest.raises(ValueError, match="Metric values must be lists of numbers"):
            self.validator.validate_reproducibility(
                reference_results={"metric1": "invalid"},
                current_results={"metric1": [1, 2, 3]}
            )
    
    def test_validate_reproducibility_invalid_current_metric_data(self):
        """Test reproducibility validation with invalid current metric data."""
        with pytest.raises(ValueError, match="Current metric values must be lists of numbers"):
            self.validator.validate_reproducibility(
                reference_results={"metric1": [1, 2, 3]},
                current_results={"metric1": "invalid"}
            )
    
    def test_validate_reproducibility_empty_metric_data(self):
        """Test reproducibility validation with empty metric data."""
        with pytest.raises(ValueError, match="Metric values cannot be empty"):
            self.validator.validate_reproducibility(
                reference_results={"metric1": []},
                current_results={"metric1": [1, 2, 3]}
            )
    
    def test_validate_reproducibility_empty_current_metric_data(self):
        """Test reproducibility validation with empty current metric data."""
        with pytest.raises(ValueError, match="Current metric values cannot be empty"):
            self.validator.validate_reproducibility(
                reference_results={"metric1": [1, 2, 3]},
                current_results={"metric1": []}
            )
    
    def test_validate_reproducibility_non_numeric_data(self):
        """Test reproducibility validation with non-numeric data."""
        with pytest.raises(ValueError, match="Metric values must be lists of numbers"):
            self.validator.validate_reproducibility(
                reference_results={"metric1": [1, 2, "invalid"]},
                current_results={"metric1": [1, 2, 3]}
            )
    
    def test_validate_reproducibility_non_numeric_current_data(self):
        """Test reproducibility validation with non-numeric current data."""
        with pytest.raises(ValueError, match="Current metric values must be lists of numbers"):
            self.validator.validate_reproducibility(
                reference_results={"metric1": [1, 2, 3]},
                current_results={"metric1": [1, 2, "invalid"]}
            )
    
    def test_save_report(self):
        """Test saving reproducibility report."""
        reference_results = {"metric1": [1, 2, 3, 4, 5]}
        current_results = {"metric1": [1, 2, 3, 4, 5]}
        
        report = self.validator.validate_reproducibility(
            reference_results=reference_results,
            current_results=current_results,
            reference_run_id="ref_run",
            current_run_id="curr_run"
        )
        
        # Save report
        report_path = self.validator.save_report(report, "test_report.json")
        
        assert os.path.exists(report_path)
        assert report_path.endswith("test_report.json")
        
        # Verify content
        with open(report_path, 'r') as f:
            data = json.load(f)
        
        assert data["reference_run_id"] == "ref_run"
        assert data["current_run_id"] == "curr_run"
        assert data["overall_reproducible"] is True
        assert "metric1" in data["metric_results"]
    
    def test_save_report_no_filename(self):
        """Test saving report without filename."""
        reference_results = {"metric1": [1, 2, 3, 4, 5]}
        current_results = {"metric1": [1, 2, 3, 4, 5]}
        
        report = self.validator.validate_reproducibility(
            reference_results=reference_results,
            current_results=current_results
        )
        
        # Save report without filename
        report_path = self.validator.save_report(report)
        
        assert os.path.exists(report_path)
        assert report_path.endswith(".json")
    
    def test_load_report(self):
        """Test loading reproducibility report."""
        reference_results = {"metric1": [1, 2, 3, 4, 5]}
        current_results = {"metric1": [1, 2, 3, 4, 5]}
        
        report = self.validator.validate_reproducibility(
            reference_results=reference_results,
            current_results=current_results,
            reference_run_id="ref_run",
            current_run_id="curr_run"
        )
        
        # Save report
        self.validator.save_report(report, "test_report.json")
        
        # Load report
        loaded_report = self.validator.load_report("test_report.json")
        
        assert loaded_report.reference_run_id == "ref_run"
        assert loaded_report.current_run_id == "curr_run"
        assert loaded_report.overall_reproducible is True
        assert "metric1" in loaded_report.metric_results
    
    def test_list_reports(self):
        """Test listing reproducibility reports."""
        # Create and save multiple reports
        for i in range(3):
            reference_results = {"metric1": [1, 2, 3, 4, 5]}
            current_results = {"metric1": [1, 2, 3, 4, 5]}
            
            report = self.validator.validate_reproducibility(
                reference_results=reference_results,
                current_results=current_results,
                reference_run_id=f"ref_run_{i}",
                current_run_id=f"curr_run_{i}"
            )
            
            self.validator.save_report(report, f"report_{i}.json")
        
        reports = self.validator.list_reports()
        
        assert len(reports) == 3
        assert "report_0.json" in reports
        assert "report_1.json" in reports
        assert "report_2.json" in reports
    
    def test_get_validation_history(self):
        """Test getting validation history."""
        # Create and save multiple reports
        for i in range(3):
            reference_results = {"metric1": [1, 2, 3, 4, 5]}
            current_results = {"metric1": [1, 2, 3, 4, 5]}
            
            report = self.validator.validate_reproducibility(
                reference_results=reference_results,
                current_results=current_results,
                reference_run_id=f"ref_run_{i}",
                current_run_id=f"curr_run_{i}"
            )
            
            self.validator.save_report(report, f"report_{i}.json")
        
        history = self.validator.get_validation_history()
        
        assert len(history) == 3
        assert all(isinstance(report, ReproducibilityReport) for report in history)
    
    def test_clear_validation_history(self):
        """Test clearing validation history."""
        # Create and save multiple reports
        for i in range(3):
            reference_results = {"metric1": [1, 2, 3, 4, 5]}
            current_results = {"metric1": [1, 2, 3, 4, 5]}
            
            report = self.validator.validate_reproducibility(
                reference_results=reference_results,
                current_results=current_results
            )
            
            self.validator.save_report(report, f"report_{i}.json")
        
        # Clear history
        self.validator.clear_validation_history()
        
        # History should be empty
        history = self.validator.get_validation_history()
        assert len(history) == 0
        
        # Files should still exist
        reports = self.validator.list_reports()
        assert len(reports) == 3


class TestReproducibilityReport:
    """Test cases for ReproducibilityReport class."""
    
    def test_initialization(self):
        """Test ReproducibilityReport initialization."""
        report = ReproducibilityReport(
            reference_run_id="ref_run",
            current_run_id="curr_run",
            overall_reproducible=True,
            metric_results={
                "metric1": {
                    "reproducible": True,
                    "issues": []
                }
            }
        )
        
        assert report.reference_run_id == "ref_run"
        assert report.current_run_id == "curr_run"
        assert report.overall_reproducible is True
        assert "metric1" in report.metric_results
        assert isinstance(report.validation_timestamp, datetime)
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        report = ReproducibilityReport(
            reference_run_id="ref_run",
            current_run_id="curr_run",
            overall_reproducible=True,
            metric_results={
                "metric1": {
                    "reproducible": True,
                    "issues": []
                }
            }
        )
        
        result = report.to_dict()
        
        assert result["reference_run_id"] == "ref_run"
        assert result["current_run_id"] == "curr_run"
        assert result["overall_reproducible"] is True
        assert "metric1" in result["metric_results"]
        assert "validation_timestamp" in result
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "reference_run_id": "ref_run",
            "current_run_id": "curr_run",
            "overall_reproducible": True,
            "metric_results": {
                "metric1": {
                    "reproducible": True,
                    "issues": []
                }
            },
            "validation_timestamp": "2023-01-01T00:00:00"
        }
        
        report = ReproducibilityReport.from_dict(data)
        
        assert report.reference_run_id == "ref_run"
        assert report.current_run_id == "curr_run"
        assert report.overall_reproducible is True
        assert "metric1" in report.metric_results
        assert isinstance(report.validation_timestamp, datetime)


class TestValidationResult:
    """Test cases for ValidationResult class."""
    
    def test_initialization(self):
        """Test ValidationResult initialization."""
        result = ValidationResult(
            metric_name="test_metric",
            reproducible=True,
            issues=[],
            reference_stats={"mean": 5.0, "std": 2.0},
            current_stats={"mean": 5.0, "std": 2.0},
            difference={"mean": 0.0, "std": 0.0}
        )
        
        assert result.metric_name == "test_metric"
        assert result.reproducible is True
        assert result.issues == []
        assert result.reference_stats == {"mean": 5.0, "std": 2.0}
        assert result.current_stats == {"mean": 5.0, "std": 2.0}
        assert result.difference == {"mean": 0.0, "std": 0.0}
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        result = ValidationResult(
            metric_name="test_metric",
            reproducible=True,
            issues=[],
            reference_stats={"mean": 5.0, "std": 2.0},
            current_stats={"mean": 5.0, "std": 2.0},
            difference={"mean": 0.0, "std": 0.0}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["metric_name"] == "test_metric"
        assert result_dict["reproducible"] is True
        assert result_dict["issues"] == []
        assert result_dict["reference_stats"] == {"mean": 5.0, "std": 2.0}
        assert result_dict["current_stats"] == {"mean": 5.0, "std": 2.0}
        assert result_dict["difference"] == {"mean": 0.0, "std": 0.0}
