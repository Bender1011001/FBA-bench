"""
Metric registry for managing available metrics.

This module provides a centralized registry for all available metrics,
allowing for dynamic registration, discovery, and instantiation of metrics.
"""

import logging
from typing import Dict, Any, List, Optional, Type, Union
from dataclasses import dataclass

from .base import BaseMetric, CognitiveMetrics, BusinessMetrics, TechnicalMetrics, EthicalMetrics, MetricConfig
from .advanced_cognitive import AdvancedCognitiveMetrics
from .business_intelligence import BusinessIntelligenceMetrics
from .technical_performance import TechnicalPerformanceMetrics
from .ethical_safety import EthicalSafetyMetrics
from .cross_domain import CrossDomainMetrics
from .statistical_analysis import StatisticalAnalysisFramework
from .comparative_analysis import ComparativeAnalysisEngine

logger = logging.getLogger(__name__)


@dataclass
class MetricRegistration:
    """Information about a registered metric."""
    name: str
    description: str
    category: str
    metric_class: Type[BaseMetric]
    default_config: MetricConfig
    tags: List[str] = None
    enabled: bool = True
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class MetricRegistry:
    """
    Registry for managing available metrics.
    
    This class provides a centralized way to register, discover, and instantiate
    metrics. It supports dynamic registration and categorization of metrics.
    """
    
    def __init__(self):
        """Initialize the metric registry."""
        self._metrics: Dict[str, MetricRegistration] = {}
        self._categories: Dict[str, List[str]] = {}
        self._tags: Dict[str, List[str]] = {}
        
        # Register built-in metrics
        self._register_builtin_metrics()
    
    def _register_builtin_metrics(self) -> None:
        """Register all built-in metrics."""
        # Cognitive metrics
        self.register_metric(
            name="cognitive_performance",
            description="Overall cognitive performance score",
            category="cognitive",
            metric_class=CognitiveMetrics,
            default_config=MetricConfig(
                name="cognitive_performance",
                description="Overall cognitive performance score",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=80.0
            ),
            tags=["cognitive", "performance", "reasoning", "planning", "memory", "learning"]
        )
        
        # Business metrics
        self.register_metric(
            name="business_performance",
            description="Overall business performance score",
            category="business",
            metric_class=BusinessMetrics,
            default_config=MetricConfig(
                name="business_performance",
                description="Overall business performance score",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=75.0
            ),
            tags=["business", "performance", "roi", "efficiency", "strategic"]
        )
        
        # Technical metrics
        self.register_metric(
            name="technical_performance",
            description="Overall technical performance score",
            category="technical",
            metric_class=TechnicalMetrics,
            default_config=MetricConfig(
                name="technical_performance",
                description="Overall technical performance score",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=85.0
            ),
            tags=["technical", "performance", "reliability", "resource", "execution"]
        )
        
        # Ethical metrics
        self.register_metric(
            name="ethical_performance",
            description="Overall ethical performance score",
            category="ethical",
            metric_class=EthicalMetrics,
            default_config=MetricConfig(
                name="ethical_performance",
                description="Overall ethical performance score",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=90.0
            ),
            tags=["ethical", "bias", "safety", "transparency", "fairness"]
        )
        
        # Advanced cognitive metrics
        self.register_metric(
            name="advanced_cognitive_performance",
            description="Advanced cognitive performance evaluation",
            category="cognitive",
            metric_class=AdvancedCognitiveMetrics,
            default_config=MetricConfig(
                name="advanced_cognitive_performance",
                description="Advanced cognitive performance evaluation",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=85.0
            ),
            tags=["cognitive", "advanced", "reasoning", "planning", "memory", "learning", "metacognition"]
        )
        
        # Business intelligence metrics
        self.register_metric(
            name="business_intelligence_performance",
            description="Business intelligence performance evaluation",
            category="business",
            metric_class=BusinessIntelligenceMetrics,
            default_config=MetricConfig(
                name="business_intelligence_performance",
                description="Business intelligence performance evaluation",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=80.0
            ),
            tags=["business", "intelligence", "strategic", "market", "risk", "roi"]
        )
        
        # Technical performance metrics
        self.register_metric(
            name="technical_performance_advanced",
            description="Advanced technical performance evaluation",
            category="technical",
            metric_class=TechnicalPerformanceMetrics,
            default_config=MetricConfig(
                name="technical_performance_advanced",
                description="Advanced technical performance evaluation",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=85.0
            ),
            tags=["technical", "performance", "scalability", "resource", "latency", "resilience"]
        )
        
        # Ethical and safety metrics
        self.register_metric(
            name="ethical_safety_performance",
            description="Ethical and safety performance evaluation",
            category="ethical",
            metric_class=EthicalSafetyMetrics,
            default_config=MetricConfig(
                name="ethical_safety_performance",
                description="Ethical and safety performance evaluation",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=90.0
            ),
            tags=["ethical", "safety", "bias", "fairness", "transparency", "privacy"]
        )
        
        # Cross-domain metrics
        self.register_metric(
            name="cross_domain_performance",
            description="Cross-domain performance evaluation",
            category="cross_domain",
            metric_class=CrossDomainMetrics,
            default_config=MetricConfig(
                name="cross_domain_performance",
                description="Cross-domain performance evaluation",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=75.0
            ),
            tags=["cross_domain", "adaptation", "transfer", "generalization", "consistency"]
        )
        
        # Statistical analysis metrics
        self.register_metric(
            name="statistical_analysis_performance",
            description="Statistical analysis performance evaluation",
            category="statistical",
            metric_class=StatisticalAnalysisFramework,
            default_config=MetricConfig(
                name="statistical_analysis_performance",
                description="Statistical analysis performance evaluation",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=85.0
            ),
            tags=["statistical", "analysis", "confidence", "significance", "correlation", "prediction"]
        )
        
        # Comparative analysis metrics
        self.register_metric(
            name="comparative_analysis_performance",
            description="Comparative analysis performance evaluation",
            category="comparative",
            metric_class=ComparativeAnalysisEngine,
            default_config=MetricConfig(
                name="comparative_analysis_performance",
                description="Comparative analysis performance evaluation",
                unit="score",
                min_value=0.0,
                max_value=100.0,
                target_value=80.0
            ),
            tags=["comparative", "analysis", "ranking", "comparison", "benchmarking", "evaluation"]
        )
    
    def register_metric(
        self,
        name: str,
        description: str,
        category: str,
        metric_class: Type[BaseMetric],
        default_config: MetricConfig,
        tags: List[str] = None,
        enabled: bool = True
    ) -> None:
        """
        Register a new metric.
        
        Args:
            name: Unique name for the metric
            description: Description of the metric
            category: Category of the metric
            metric_class: Class implementing the metric
            default_config: Default configuration for the metric
            tags: List of tags for categorization
            enabled: Whether the metric is enabled by default
        """
        if name in self._metrics:
            logger.warning(f"Metric '{name}' already registered, overwriting")
        
        registration = MetricRegistration(
            name=name,
            description=description,
            category=category,
            metric_class=metric_class,
            default_config=default_config,
            tags=tags or [],
            enabled=enabled
        )
        
        self._metrics[name] = registration
        
        # Update category index
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)
        
        # Update tag index
        for tag in registration.tags:
            if tag not in self._tags:
                self._tags[tag] = []
            if name not in self._tags[tag]:
                self._tags[tag].append(name)
        
        logger.info(f"Registered metric: {name} (category: {category})")
    
    def unregister_metric(self, name: str) -> bool:
        """
        Unregister a metric.
        
        Args:
            name: Name of the metric to unregister
            
        Returns:
            True if successful, False if metric not found
        """
        if name not in self._metrics:
            logger.warning(f"Metric '{name}' not found for unregistration")
            return False
        
        registration = self._metrics[name]
        
        # Remove from category index
        category = registration.category
        if category in self._categories and name in self._categories[category]:
            self._categories[category].remove(name)
            if not self._categories[category]:
                del self._categories[category]
        
        # Remove from tag index
        for tag in registration.tags:
            if tag in self._tags and name in self._tags[tag]:
                self._tags[tag].remove(name)
                if not self._tags[tag]:
                    del self._tags[tag]
        
        # Remove from metrics
        del self._metrics[name]
        
        logger.info(f"Unregistered metric: {name}")
        return True
    
    def get_metric(self, name: str) -> Optional[MetricRegistration]:
        """
        Get a metric registration by name.
        
        Args:
            name: Name of the metric
            
        Returns:
            Metric registration or None if not found
        """
        return self._metrics.get(name)
    
    def create_metric(
        self, 
        name: str, 
        config: MetricConfig = None
    ) -> Optional[BaseMetric]:
        """
        Create an instance of a metric.
        
        Args:
            name: Name of the metric
            config: Configuration for the metric (uses default if None)
            
        Returns:
            Metric instance or None if not found
        """
        registration = self.get_metric(name)
        if registration is None:
            logger.error(f"Metric '{name}' not found")
            return None
        
        if config is None:
            config = registration.default_config
        
        try:
            metric_instance = registration.metric_class(config)
            logger.debug(f"Created metric instance: {name}")
            return metric_instance
        except Exception as e:
            logger.error(f"Error creating metric '{name}': {e}")
            return None
    
    def list_metrics(self, category: str = None, tag: str = None) -> List[str]:
        """
        List available metrics.
        
        Args:
            category: Filter by category (optional)
            tag: Filter by tag (optional)
            
        Returns:
            List of metric names
        """
        if category is not None:
            return self._categories.get(category, []).copy()
        
        if tag is not None:
            return self._tags.get(tag, []).copy()
        
        return list(self._metrics.keys())
    
    def list_categories(self) -> List[str]:
        """
        List all available categories.
        
        Returns:
            List of category names
        """
        return list(self._categories.keys())
    
    def list_tags(self) -> List[str]:
        """
        List all available tags.
        
        Returns:
            List of tag names
        """
        return list(self._tags.keys())
    
    def get_metrics_by_category(self, category: str) -> Dict[str, MetricRegistration]:
        """
        Get all metrics in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            Dictionary of metric registrations
        """
        if category not in self._categories:
            return {}
        
        return {
            name: self._metrics[name]
            for name in self._categories[category]
            if name in self._metrics
        }
    
    def get_metrics_by_tag(self, tag: str) -> Dict[str, MetricRegistration]:
        """
        Get all metrics with a specific tag.
        
        Args:
            tag: Tag name
            
        Returns:
            Dictionary of metric registrations
        """
        if tag not in self._tags:
            return {}
        
        return {
            name: self._metrics[name]
            for name in self._tags[tag]
            if name in self._metrics
        }
    
    def get_enabled_metrics(self) -> Dict[str, MetricRegistration]:
        """
        Get all enabled metrics.
        
        Returns:
            Dictionary of enabled metric registrations
        """
        return {
            name: registration
            for name, registration in self._metrics.items()
            if registration.enabled
        }
    
    def enable_metric(self, name: str) -> bool:
        """
        Enable a metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            True if successful, False if metric not found
        """
        if name not in self._metrics:
            return False
        
        self._metrics[name].enabled = True
        logger.info(f"Enabled metric: {name}")
        return True
    
    def disable_metric(self, name: str) -> bool:
        """
        Disable a metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            True if successful, False if metric not found
        """
        if name not in self._metrics:
            return False
        
        self._metrics[name].enabled = False
        logger.info(f"Disabled metric: {name}")
        return True
    
    def get_metric_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Dictionary with metric information
        """
        registration = self.get_metric(name)
        if registration is None:
            return {"error": f"Metric '{name}' not found"}
        
        return {
            "name": registration.name,
            "description": registration.description,
            "category": registration.category,
            "class": registration.metric_class.__name__,
            "module": registration.metric_class.__module__,
            "default_config": {
                "name": registration.default_config.name,
                "description": registration.default_config.description,
                "unit": registration.default_config.unit,
                "min_value": registration.default_config.min_value,
                "max_value": registration.default_config.max_value,
                "target_value": registration.default_config.target_value,
                "weight": registration.default_config.weight,
                "enabled": registration.default_config.enabled
            },
            "tags": registration.tags,
            "enabled": registration.enabled
        }
    
    def create_metric_suite(
        self, 
        metric_names: List[str], 
        configs: Dict[str, MetricConfig] = None
    ) -> Dict[str, BaseMetric]:
        """
        Create a suite of metrics.
        
        Args:
            metric_names: List of metric names to include
            configs: Custom configurations for metrics (optional)
            
        Returns:
            Dictionary of metric instances
        """
        suite = {}
        configs = configs or {}
        
        for name in metric_names:
            config = configs.get(name)
            metric = self.create_metric(name, config)
            if metric is not None:
                suite[name] = metric
        
        return suite
    
    def validate_metric_config(self, name: str, config: MetricConfig) -> List[str]:
        """
        Validate a metric configuration.
        
        Args:
            name: Name of the metric
            config: Configuration to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        registration = self.get_metric(name)
        if registration is None:
            errors.append(f"Unknown metric: {name}")
            return errors
        
        # Validate required fields
        if not config.name:
            errors.append("Metric name cannot be empty")
        
        if not config.description:
            errors.append("Metric description cannot be empty")
        
        if not config.unit:
            errors.append("Metric unit cannot be empty")
        
        # Validate value ranges
        if (config.min_value is not None and 
            config.max_value is not None and 
            config.min_value >= config.max_value):
            errors.append("min_value must be less than max_value")
        
        if (config.target_value is not None and 
            config.min_value is not None and 
            config.target_value < config.min_value):
            errors.append("target_value must be greater than or equal to min_value")
        
        if (config.target_value is not None and 
            config.max_value is not None and 
            config.target_value > config.max_value):
            errors.append("target_value must be less than or equal to max_value")
        
        # Validate weight
        if config.weight < 0:
            errors.append("weight must be non-negative")
        
        return errors
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the registry.
        
        Returns:
            Dictionary with registry summary
        """
        enabled_count = sum(1 for r in self._metrics.values() if r.enabled)
        disabled_count = len(self._metrics) - enabled_count
        
        return {
            "total_metrics": len(self._metrics),
            "enabled_metrics": enabled_count,
            "disabled_metrics": disabled_count,
            "categories": {
                category: len(metrics)
                for category, metrics in self._categories.items()
            },
            "tags": {
                tag: len(metrics)
                for tag, metrics in self._tags.items()
            }
        }


# Global registry instance
metrics_registry = MetricRegistry()