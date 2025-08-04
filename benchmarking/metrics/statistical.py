"""
Statistical validation methods for benchmarking metrics.

This module provides statistical validation methods including confidence intervals,
significance testing, and other statistical analysis tools following HELM and
BIG-bench standards.
"""

import math
import statistics
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats


class StatisticalTest(Enum):
    """Enumeration of supported statistical tests."""
    T_TEST = "t_test"
    PAIRED_T_TEST = "paired_t_test"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY_U = "mann_whitney_u"
    ANOVA = "anova"
    CHI_SQUARE = "chi_square"


@dataclass
class StatisticalResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    interpretation: str = ""
    significant: bool = False
    alpha: float = 0.05


class StatisticalValidator:
    """
    Statistical validation for benchmarking metrics.
    
    This class provides methods for calculating confidence intervals, performing
    statistical tests, and validating the significance of results.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize the statistical validator.
        
        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
    
    def calculate_confidence_interval(
        self, 
        data: List[float], 
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for a sample mean.
        
        Args:
            data: Sample data
            confidence: Confidence level (0.0 to 1.0)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not data:
            raise ValueError("Data list cannot be empty")
        
        if len(data) < 2:
            return (data[0], data[0])
        
        n = len(data)
        mean = statistics.mean(data)
        std_err = statistics.stdev(data) / math.sqrt(n)
        
        # Calculate critical value
        if n >= 30:
            # Use z-score for large samples
            critical_value = stats.norm.ppf((1 + confidence) / 2)
        else:
            # Use t-score for small samples
            critical_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        
        margin_of_error = critical_value * std_err
        
        return (mean - margin_of_error, mean + margin_of_error)
    
    def t_test(
        self, 
        sample1: List[float], 
        sample2: List[float], 
        paired: bool = False
    ) -> StatisticalResult:
        """
        Perform t-test between two samples.
        
        Args:
            sample1: First sample data
            sample2: Second sample data
            paired: Whether to use paired t-test
            
        Returns:
            Statistical test result
        """
        if not sample1 or not sample2:
            raise ValueError("Sample lists cannot be empty")
        
        if paired and len(sample1) != len(sample2):
            raise ValueError("Paired samples must have the same length")
        
        if paired:
            # Paired t-test
            statistic, p_value = stats.ttest_rel(sample1, sample2)
            test_name = StatisticalTest.PAIRED_T_TEST.value
        else:
            # Independent t-test
            statistic, p_value = stats.ttest_ind(sample1, sample2)
            test_name = StatisticalTest.T_TEST.value
        
        # Calculate effect size (Cohen's d)
        if paired:
            differences = [s1 - s2 for s1, s2 in zip(sample1, sample2)]
            mean_diff = statistics.mean(differences)
            std_diff = statistics.stdev(differences) if len(differences) > 1 else 1.0
            effect_size = mean_diff / std_diff if std_diff > 0 else 0.0
        else:
            mean1, mean2 = statistics.mean(sample1), statistics.mean(sample2)
            var1, var2 = statistics.variance(sample1), statistics.variance(sample2)
            n1, n2 = len(sample1), len(sample2)
            
            pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            effect_size = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        
        # Calculate confidence interval for mean difference
        if paired:
            mean_diff = statistics.mean([s1 - s2 for s1, s2 in zip(sample1, sample2)])
            std_diff = statistics.stdev([s1 - s2 for s1, s2 in zip(sample1, sample2)]) if len(sample1) > 1 else 0.0
            n = len(sample1)
        else:
            mean_diff = statistics.mean(sample1) - statistics.mean(sample2)
            var1, var2 = statistics.variance(sample1), statistics.variance(sample2)
            n1, n2 = len(sample1), len(sample2)
            std_diff = math.sqrt(var1/n1 + var2/n2)
            n = min(n1, n2)
        
        if std_diff > 0 and n > 1:
            if n >= 30:
                critical_value = stats.norm.ppf(1 - self.alpha/2)
            else:
                critical_value = stats.t.ppf(1 - self.alpha/2, n - 1)
            
            margin_of_error = critical_value * std_diff
            confidence_interval = (mean_diff - margin_of_error, mean_diff + margin_of_error)
        else:
            confidence_interval = None
        
        # Interpret result
        significant = p_value < self.alpha
        if significant:
            interpretation = f"Significant difference detected (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference detected (p={p_value:.4f})"
        
        return StatisticalResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            interpretation=interpretation,
            significant=significant,
            alpha=self.alpha
        )
    
    def mann_whitney_u_test(self, sample1: List[float], sample2: List[float]) -> StatisticalResult:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        
        Args:
            sample1: First sample data
            sample2: Second sample data
            
        Returns:
            Statistical test result
        """
        if not sample1 or not sample2:
            raise ValueError("Sample lists cannot be empty")
        
        statistic, p_value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
        
        # Calculate effect size (r = Z / sqrt(N))
        n1, n2 = len(sample1), len(sample2)
        z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 0.0
        effect_size = abs(z_score) / math.sqrt(n1 + n2)
        
        # Interpret result
        significant = p_value < self.alpha
        if significant:
            interpretation = f"Significant difference detected (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference detected (p={p_value:.4f})"
        
        return StatisticalResult(
            test_name=StatisticalTest.MANN_WHITNEY_U.value,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            interpretation=interpretation,
            significant=significant,
            alpha=self.alpha
        )
    
    def anova_test(self, *samples: List[float]) -> StatisticalResult:
        """
        Perform one-way ANOVA test for multiple samples.
        
        Args:
            *samples: Variable number of sample lists
            
        Returns:
            Statistical test result
        """
        if len(samples) < 2:
            raise ValueError("At least two samples are required for ANOVA")
        
        if any(not sample for sample in samples):
            raise ValueError("Sample lists cannot be empty")
        
        statistic, p_value = stats.f_oneway(*samples)
        
        # Calculate effect size (eta-squared)
        total_variance = statistics.variance([val for sample in samples for val in sample])
        between_group_variance = sum(
            len(sample) * (statistics.mean(sample) - statistics.mean([val for sample in samples for val in sample]))**2
            for sample in samples
        ) / sum(len(sample) for sample in samples)
        
        effect_size = between_group_variance / total_variance if total_variance > 0 else 0.0
        
        # Interpret result
        significant = p_value < self.alpha
        if significant:
            interpretation = f"Significant difference detected between groups (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference detected between groups (p={p_value:.4f})"
        
        return StatisticalResult(
            test_name=StatisticalTest.ANOVA.value,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            interpretation=interpretation,
            significant=significant,
            alpha=self.alpha
        )
    
    def chi_square_test(self, observed: List[int], expected: List[int]) -> StatisticalResult:
        """
        Perform chi-square goodness-of-fit test.
        
        Args:
            observed: Observed frequencies
            expected: Expected frequencies
            
        Returns:
            Statistical test result
        """
        if not observed or not expected:
            raise ValueError("Observed and expected lists cannot be empty")
        
        if len(observed) != len(expected):
            raise ValueError("Observed and expected lists must have the same length")
        
        statistic, p_value = stats.chisquare(observed, expected)
        
        # Calculate effect size (Cramer's V)
        n = sum(observed)
        k = len(observed)
        cramers_v = math.sqrt(statistic / (n * (k - 1))) if n > 0 and k > 1 else 0.0
        
        # Interpret result
        significant = p_value < self.alpha
        if significant:
            interpretation = f"Significant deviation from expected distribution (p={p_value:.4f})"
        else:
            interpretation = f"No significant deviation from expected distribution (p={p_value:.4f})"
        
        return StatisticalResult(
            test_name=StatisticalTest.CHI_SQUARE.value,
            statistic=statistic,
            p_value=p_value,
            effect_size=cramers_v,
            interpretation=interpretation,
            significant=significant,
            alpha=self.alpha
        )
    
    def calculate_bayesian_factor(
        self, 
        sample1: List[float], 
        sample2: List[float]
    ) -> float:
        """
        Calculate Bayes factor for comparing two hypotheses.
        
        Args:
            sample1: Data under hypothesis 1
            sample2: Data under hypothesis 2
            
        Returns:
            Bayes factor (BF12)
        """
        if not sample1 or not sample2:
            raise ValueError("Sample lists cannot be empty")
        
        # Calculate likelihoods assuming normal distributions
        mean1, std1 = statistics.mean(sample1), statistics.stdev(sample1) if len(sample1) > 1 else 1.0
        mean2, std2 = statistics.mean(sample2), statistics.stdev(sample2) if len(sample2) > 1 else 1.0
        
        # Calculate log-likelihoods
        log_likelihood1 = sum(
            -0.5 * ((x - mean1) / std1)**2 - math.log(std1) - 0.5 * math.log(2 * math.pi)
            for x in sample1
        )
        
        log_likelihood2 = sum(
            -0.5 * ((x - mean2) / std2)**2 - math.log(std2) - 0.5 * math.log(2 * math.pi)
            for x in sample2
        )
        
        # Calculate Bayes factor
        log_bf12 = log_likelihood1 - log_likelihood2
        bf12 = math.exp(log_bf12)
        
        return bf12
    
    def calculate_power_analysis(
        self, 
        effect_size: float, 
        alpha: float = None, 
        power: float = 0.8
    ) -> Dict[str, Any]:
        """
        Calculate sample size needed for statistical power.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            alpha: Significance level (uses instance default if None)
            power: Desired statistical power (default: 0.8)
            
        Returns:
            Dictionary with power analysis results
        """
        if alpha is None:
            alpha = self.alpha
        
        # Calculate required sample size using approximation
        # For two-sample t-test
        if effect_size <= 0:
            raise ValueError("Effect size must be positive")
        
        # Approximation for sample size per group
        n_per_group = 16 / (effect_size ** 2)  # Simplified approximation
        
        # More accurate calculation using power analysis
        try:
            from statsmodels.stats.power import TTestIndPower
            power_analysis = TTestIndPower()
            n_per_group = power_analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                alternative='two-sided'
            )
        except ImportError:
            # Fallback to approximation if statsmodels not available
            pass
        
        return {
            "effect_size": effect_size,
            "alpha": alpha,
            "power": power,
            "sample_size_per_group": int(math.ceil(n_per_group)),
            "total_sample_size": int(math.ceil(2 * n_per_group))
        }
    
    def validate_reliability(self, data: List[float]) -> Dict[str, Any]:
        """
        Validate the reliability of measurements.
        
        Args:
            data: Measurement data
            
        Returns:
            Dictionary with reliability metrics
        """
        if not data:
            raise ValueError("Data list cannot be empty")
        
        if len(data) < 2:
            return {
                "reliable": False,
                "error": "Insufficient data for reliability analysis"
            }
        
        # Calculate basic statistics
        mean = statistics.mean(data)
        std_dev = statistics.stdev(data)
        coefficient_of_variation = std_dev / mean if mean != 0 else float('inf')
        
        # Calculate confidence interval
        ci_lower, ci_upper = self.calculate_confidence_interval(data)
        ci_width = ci_upper - ci_lower
        
        # Assess reliability
        reliable = True
        reliability_issues = []
        
        if coefficient_of_variation > 0.3:  # High variability
            reliable = False
            reliability_issues.append("High coefficient of variation")
        
        if ci_width > abs(mean) * 0.5:  # Wide confidence interval
            reliable = False
            reliability_issues.append("Wide confidence interval")
        
        return {
            "reliable": reliable,
            "mean": mean,
            "std_dev": std_dev,
            "coefficient_of_variation": coefficient_of_variation,
            "confidence_interval": (ci_lower, ci_upper),
            "confidence_interval_width": ci_width,
            "reliability_issues": reliability_issues
        }
    
    def compare_distributions(
        self, 
        sample1: List[float], 
        sample2: List[float]
    ) -> Dict[str, Any]:
        """
        Compare two distributions using multiple statistical tests.
        
        Args:
            sample1: First sample data
            sample2: Second sample data
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        # T-test
        try:
            results["t_test"] = self.t_test(sample1, sample2)
        except Exception as e:
            results["t_test"] = {"error": str(e)}
        
        # Mann-Whitney U test
        try:
            results["mann_whitney_u"] = self.mann_whitney_u_test(sample1, sample2)
        except Exception as e:
            results["mann_whitney_u"] = {"error": str(e)}
        
        # Kolmogorov-Smirnov test
        try:
            statistic, p_value = stats.ks_2samp(sample1, sample2)
            results["kolmogorov_smirnov"] = StatisticalResult(
                test_name="kolmogorov_smirnov",
                statistic=statistic,
                p_value=p_value,
                significant=p_value < self.alpha,
                interpretation=f"KS test p-value: {p_value:.4f}",
                alpha=self.alpha
            )
        except Exception as e:
            results["kolmogorov_smirnov"] = {"error": str(e)}
        
        # Basic statistics
        results["sample1_stats"] = {
            "mean": statistics.mean(sample1),
            "median": statistics.median(sample1),
            "std_dev": statistics.stdev(sample1) if len(sample1) > 1 else 0.0,
            "min": min(sample1),
            "max": max(sample1),
            "count": len(sample1)
        }
        
        results["sample2_stats"] = {
            "mean": statistics.mean(sample2),
            "median": statistics.median(sample2),
            "std_dev": statistics.stdev(sample2) if len(sample2) > 1 else 0.0,
            "min": min(sample2),
            "max": max(sample2),
            "count": len(sample2)
        }
        
        return results