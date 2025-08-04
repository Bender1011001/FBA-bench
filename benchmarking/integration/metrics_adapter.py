"""
Metrics adapter for integrating existing metrics with the benchmarking framework.

This module provides adapters to bridge the gap between the new benchmarking metrics
and the existing metrics system, ensuring seamless integration and compatibility.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Type
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..core.engine import BenchmarkEngine, BenchmarkResult
from ..integration.manager import IntegrationManager
from ..metrics.registry import metrics_registry
from ..metrics.base import BaseMetric, MetricResult

# Try to import existing metrics systems
try:
    from metrics.metric_suite import MetricSuite
    from metrics.cognitive_metrics import CognitiveMetrics
    from metrics.finance_metrics import FinanceMetrics
    from metrics.operations_metrics import OperationsMetrics
    from metrics.marketing_metrics import MarketingMetrics
    from metrics.trust_metrics import TrustMetrics
    from metrics.stress_metrics import StressMetrics
    from metrics.cost_metrics import CostMetrics
    from metrics.adversarial_metrics import AdversarialMetrics
    LEGACY_METRICS_AVAILABLE = True
except ImportError:
    LEGACY_METRICS_AVAILABLE = False
    logging.warning("legacy metrics module not available")

logger = logging.getLogger(__name__)


@dataclass
class MetricsAdapterConfig:
    """Configuration for metrics adapter."""
    enable_legacy_metrics: bool = True
    enable_new_metrics: bool = True
    merge_results: bool = True
    legacy_weights: Dict[str, float] = field(default_factory=dict)
    custom_transformers: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_legacy_metrics": self.enable_legacy_metrics,
            "enable_new_metrics": self.enable_new_metrics,
            "merge_results": self.merge_results,
            "legacy_weights": self.legacy_weights,
            "custom_transformers": self.custom_transformers
        }


@dataclass
class MetricsAdapterResult:
    """Result of metrics adapter execution."""
    success: bool
    legacy_metrics: Dict[str, Any] = field(default_factory=dict)
    new_metrics: Dict[str, Any] = field(default_factory=dict)
    merged_metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "legacy_metrics": self.legacy_metrics,
            "new_metrics": self.new_metrics,
            "merged_metrics": self.merged_metrics,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "warnings": self.warnings
        }


class MetricsAdapter:
    """
    Adapter for integrating existing metrics with the benchmarking framework.
    
    This class provides a bridge between the new benchmarking metrics and the
    existing metrics system, ensuring seamless integration and compatibility.
    """
    
    def __init__(self, config: MetricsAdapterConfig, integration_manager: IntegrationManager):
        """
        Initialize the metrics adapter.
        
        Args:
            config: Metrics adapter configuration
            integration_manager: Integration manager instance
        """
        self.config = config
        self.integration_manager = integration_manager
        self.legacy_metric_suite: Optional[MetricSuite] = None
        self._initialized = False
        
        # Default legacy weights
        self._default_legacy_weights = {
            "finance": 0.20,
            "ops": 0.15,
            "marketing": 0.10,
            "trust": 0.10,
            "cognitive": 0.15,
            "stress_recovery": 0.10,
            "adversarial_resistance": 0.15,
            "cost": -0.05
        }
        
        # Merge with provided weights
        self.legacy_weights = {**self._default_legacy_weights, **self.config.legacy_weights}
        
        logger.info("Initialized MetricsAdapter")
    
    async def initialize(self) -> bool:
        """
        Initialize the metrics adapter.
        
        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True
        
        try:
            # Initialize legacy metrics if enabled
            if self.config.enable_legacy_metrics and LEGACY_METRICS_AVAILABLE:
                self.legacy_metric_suite = MetricSuite(
                    tier="benchmarking",
                    weights=self.legacy_weights,
                    financial_audit_service=None,  # Will be provided during actual use
                    sales_service=None,  # Will be provided during actual use
                    trust_score_service=None  # Will be provided during actual use
                )
                logger.info("Initialized legacy metrics suite")
            
            # Initialize new metrics if enabled
            if self.config.enable_new_metrics:
                # New metrics are already initialized through the registry
                logger.info("Initialized new metrics registry")
            
            self._initialized = True
            logger.info("Successfully initialized MetricsAdapter")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MetricsAdapter: {e}")
            return False
    
    async def calculate_metrics(
        self, 
        tick_number: int, 
        events: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> MetricsAdapterResult:
        """
        Calculate metrics using both legacy and new metrics systems.
        
        Args:
            tick_number: Current tick number
            events: List of events
            context: Additional context information
            
        Returns:
            MetricsAdapterResult with calculated metrics
        """
        if not self._initialized:
            if not await self.initialize():
                return MetricsAdapterResult(
                    success=False,
                    error_message="Metrics adapter not initialized"
                )
        
        start_time = datetime.now()
        
        try:
            result = MetricsAdapterResult(success=True)
            
            # Calculate legacy metrics
            if self.config.enable_legacy_metrics and self.legacy_metric_suite:
                legacy_result = await self._calculate_legacy_metrics(tick_number, events, context)
                result.legacy_metrics = legacy_result
                result.warnings.extend(legacy_result.get("warnings", []))
            
            # Calculate new metrics
            if self.config.enable_new_metrics:
                new_result = await self._calculate_new_metrics(tick_number, events, context)
                result.new_metrics = new_result
                result.warnings.extend(new_result.get("warnings", []))
            
            # Merge results if enabled
            if self.config.merge_results and (result.legacy_metrics or result.new_metrics):
                result.merged_metrics = self._merge_metrics(result.legacy_metrics, result.new_metrics)
            
            # Calculate execution time
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Successfully calculated metrics for tick {tick_number}")
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = MetricsAdapterResult(
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            logger.error(f"Failed to calculate metrics for tick {tick_number}: {e}")
            return result
    
    async def _calculate_legacy_metrics(
        self, 
        tick_number: int, 
        events: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate metrics using the legacy metrics system.
        
        Args:
            tick_number: Current tick number
            events: List of events
            context: Additional context information
            
        Returns:
            Dictionary of legacy metrics
        """
        if not self.legacy_metric_suite:
            return {}
        
        try:
            # Process events through legacy metric suite
            for event in events:
                event_type = event.get("type", "unknown")
                
                # Create a simple event object for legacy system
                legacy_event = type('LegacyEvent', (), {
                    'tick_number': tick_number,
                    **event
                })()
                
                # Handle different event types
                if event_type == 'SaleOccurred':
                    self.legacy_metric_suite._handle_general_event(event_type, legacy_event)
                elif event_type == 'SetPriceCommand':
                    self.legacy_metric_suite._handle_general_event(event_type, legacy_event)
                elif event_type == 'ComplianceViolationEvent':
                    self.legacy_metric_suite._handle_general_event(event_type, legacy_event)
                elif event_type == 'NewBuyerFeedbackEvent':
                    self.legacy_metric_suite._handle_general_event(event_type, legacy_event)
                elif event_type == 'AgentDecisionEvent':
                    self.legacy_metric_suite._handle_general_event(event_type, legacy_event)
                elif event_type == 'AdSpendEvent':
                    self.legacy_metric_suite._handle_general_event(event_type, legacy_event)
                elif event_type == 'AgentPlannedGoalEvent':
                    self.legacy_metric_suite._handle_general_event(event_type, legacy_event)
                elif event_type == 'AgentGoalStatusUpdateEvent':
                    self.legacy_metric_suite._handle_general_event(event_type, legacy_event)
                elif event_type == 'ApiCallEvent':
                    self.legacy_metric_suite._handle_general_event(event_type, legacy_event)
                elif event_type == 'PlanningCoherenceScoreEvent':
                    self.legacy_metric_suite._handle_general_event(event_type, legacy_event)
                elif event_type in ['AdversarialEvent', 'PhishingEvent', 'MarketManipulationEvent', 'ComplianceTrapEvent']:
                    self.legacy_metric_suite._handle_general_event(event_type, legacy_event)
            
            # Calculate KPIs
            kpis = self.legacy_metric_suite.calculate_kpis(tick_number)
            
            # Transform to standard format
            legacy_metrics = {
                "overall_score": kpis.get("overall_score", 0.0),
                "breakdown": kpis.get("breakdown", {}),
                "timestamp": kpis.get("timestamp", datetime.now().isoformat()),
                "tick_number": kpis.get("tick_number", tick_number)
            }
            
            return legacy_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate legacy metrics: {e}")
            return {"error": str(e), "warnings": [f"Legacy metrics calculation failed: {e}"]}
    
    async def _calculate_new_metrics(
        self, 
        tick_number: int, 
        events: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate metrics using the new metrics system.
        
        Args:
            tick_number: Current tick number
            events: List of events
            context: Additional context information
            
        Returns:
            Dictionary of new metrics
        """
        try:
            new_metrics = {}
            
            # Get all registered metrics
            registered_metrics = metrics_registry.get_all_metrics()
            
            # Calculate each metric
            for metric_name, metric_instance in registered_metrics.items():
                try:
                    # Prepare metric context
                    metric_context = {
                        "tick_number": tick_number,
                        "events": events,
                        "context": context or {}
                    }
                    
                    # Calculate metric
                    metric_result = await metric_instance.calculate(metric_context)
                    
                    # Store result
                    if isinstance(metric_result, MetricResult):
                        new_metrics[metric_name] = metric_result.to_dict()
                    else:
                        new_metrics[metric_name] = metric_result
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate metric {metric_name}: {e}")
                    new_metrics[metric_name] = {"error": str(e)}
            
            # Apply custom transformers if configured
            for metric_name, transformer_name in self.config.custom_transformers.items():
                if metric_name in new_metrics:
                    try:
                        transformer = self._get_transformer(transformer_name)
                        if transformer:
                            new_metrics[metric_name] = transformer(new_metrics[metric_name])
                    except Exception as e:
                        logger.warning(f"Failed to apply transformer {transformer_name} to metric {metric_name}: {e}")
            
            return new_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate new metrics: {e}")
            return {"error": str(e), "warnings": [f"New metrics calculation failed: {e}"]}
    
    def _merge_metrics(self, legacy_metrics: Dict[str, Any], new_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge legacy and new metrics.
        
        Args:
            legacy_metrics: Legacy metrics dictionary
            new_metrics: New metrics dictionary
            
        Returns:
            Merged metrics dictionary
        """
        merged = {
            "legacy_metrics": legacy_metrics,
            "new_metrics": new_metrics,
            "merged_at": datetime.now().isoformat()
        }
        
        # Calculate combined overall score if both are available
        if legacy_metrics and new_metrics:
            legacy_score = legacy_metrics.get("overall_score", 0.0)
            
            # Calculate average of new metric scores
            new_scores = []
            for metric_data in new_metrics.values():
                if isinstance(metric_data, dict) and "score" in metric_data:
                    new_scores.append(metric_data["score"])
                elif isinstance(metric_data, (int, float)):
                    new_scores.append(metric_data)
            
            new_score = sum(new_scores) / len(new_scores) if new_scores else 0.0
            
            # Weighted average (50% legacy, 50% new)
            combined_score = (legacy_score * 0.5) + (new_score * 0.5)
            
            merged["overall_score"] = combined_score
            merged["score_breakdown"] = {
                "legacy_score": legacy_score,
                "new_score": new_score,
                "legacy_weight": 0.5,
                "new_weight": 0.5
            }
        
        return merged
    
    def _get_transformer(self, transformer_name: str) -> Optional[callable]:
        """
        Get a transformer function by name.
        
        Args:
            transformer_name: Name of the transformer
            
        Returns:
            Transformer function or None if not found
        """
        # Built-in transformers
        transformers = {
            "normalize": self._transformer_normalize,
            "scale": self._transformer_scale,
            "log": self._transformer_log,
            "percentage": self._transformer_percentage
        }
        
        return transformers.get(transformer_name)
    
    def _transformer_normalize(self, value: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize metric value to 0-1 range."""
        if "score" in value:
            score = value["score"]
            normalized = max(0.0, min(1.0, score / 100.0))
            value["normalized_score"] = normalized
        return value
    
    def _transformer_scale(self, value: Dict[str, Any], factor: float = 1.0) -> Dict[str, Any]:
        """Scale metric value by a factor."""
        if "score" in value:
            value["scaled_score"] = value["score"] * factor
        return value
    
    def _transformer_log(self, value: Dict[str, Any]) -> Dict[str, Any]:
        """Apply log transformation to metric value."""
        if "score" in value and value["score"] > 0:
            import math
            value["log_score"] = math.log(value["score"])
        return value
    
    def _transformer_percentage(self, value: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metric value to percentage."""
        if "score" in value:
            value["percentage"] = value["score"] * 100
        return value
    
    async def get_metric_definitions(self) -> Dict[str, Any]:
        """
        Get definitions of all available metrics.
        
        Returns:
            Dictionary of metric definitions
        """
        definitions = {
            "legacy_metrics": {},
            "new_metrics": {}
        }
        
        # Legacy metric definitions
        if self.config.enable_legacy_metrics and LEGACY_METRICS_AVAILABLE:
            definitions["legacy_metrics"] = {
                "finance": {
                    "description": "Financial performance metrics",
                    "weight": self.legacy_weights.get("finance", 0.0)
                },
                "ops": {
                    "description": "Operational efficiency metrics",
                    "weight": self.legacy_weights.get("ops", 0.0)
                },
                "marketing": {
                    "description": "Marketing effectiveness metrics",
                    "weight": self.legacy_weights.get("marketing", 0.0)
                },
                "trust": {
                    "description": "Trust and reputation metrics",
                    "weight": self.legacy_weights.get("trust", 0.0)
                },
                "cognitive": {
                    "description": "Cognitive performance metrics",
                    "weight": self.legacy_weights.get("cognitive", 0.0)
                },
                "stress_recovery": {
                    "description": "Stress recovery metrics",
                    "weight": self.legacy_weights.get("stress_recovery", 0.0)
                },
                "adversarial_resistance": {
                    "description": "Adversarial resistance metrics",
                    "weight": self.legacy_weights.get("adversarial_resistance", 0.0)
                },
                "cost": {
                    "description": "Cost efficiency metrics",
                    "weight": self.legacy_weights.get("cost", 0.0)
                }
            }
        
        # New metric definitions
        if self.config.enable_new_metrics:
            registered_metrics = metrics_registry.get_all_metrics()
            for metric_name, metric_instance in registered_metrics.items():
                definitions["new_metrics"][metric_name] = {
                    "description": getattr(metric_instance, "description", ""),
                    "category": getattr(metric_instance, "category", "unknown"),
                    "type": type(metric_instance).__name__
                }
        
        return definitions
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the metrics adapter.
        
        Returns:
            Health check result
        """
        health = {
            "initialized": self._initialized,
            "healthy": False,
            "issues": [],
            "components": {
                "legacy_metrics": {
                    "available": LEGACY_METRICS_AVAILABLE,
                    "enabled": self.config.enable_legacy_metrics,
                    "initialized": self.legacy_metric_suite is not None
                },
                "new_metrics": {
                    "available": True,
                    "enabled": self.config.enable_new_metrics,
                    "metrics_count": len(metrics_registry.get_all_metrics())
                }
            }
        }
        
        # Check overall health
        if not self._initialized:
            health["issues"].append("Metrics adapter not initialized")
            return health
        
        # Check legacy metrics health
        if self.config.enable_legacy_metrics:
            if not LEGACY_METRICS_AVAILABLE:
                health["issues"].append("Legacy metrics module not available")
            elif self.legacy_metric_suite is None:
                health["issues"].append("Legacy metrics suite not initialized")
        
        # Check new metrics health
        if self.config.enable_new_metrics:
            if len(metrics_registry.get_all_metrics()) == 0:
                health["issues"].append("No new metrics registered")
        
        # Determine overall health
        health["healthy"] = len(health["issues"]) == 0
        
        return health
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self.legacy_metric_suite = None
            self._initialized = False
            logger.info("Cleaned up MetricsAdapter")
            
        except Exception as e:
            logger.error(f"Failed to cleanup MetricsAdapter: {e}")


class MetricsAdapterFactory:
    """Factory for creating metrics adapters."""
    
    @staticmethod
    def create_adapter(
        config: MetricsAdapterConfig,
        integration_manager: IntegrationManager
    ) -> MetricsAdapter:
        """
        Create a metrics adapter.
        
        Args:
            config: Metrics adapter configuration
            integration_manager: Integration manager
            
        Returns:
            MetricsAdapter instance
        """
        return MetricsAdapter(config, integration_manager)
    
    @staticmethod
    async def create_and_initialize_adapter(
        config: MetricsAdapterConfig,
        integration_manager: IntegrationManager
    ) -> Optional[MetricsAdapter]:
        """
        Create and initialize a metrics adapter.
        
        Args:
            config: Metrics adapter configuration
            integration_manager: Integration manager
            
        Returns:
            MetricsAdapter instance or None if initialization failed
        """
        adapter = MetricsAdapterFactory.create_adapter(config, integration_manager)
        
        if await adapter.initialize():
            return adapter
        else:
            await adapter.cleanup()
            return None