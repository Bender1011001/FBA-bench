# metrics/metric_suite.py
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from metrics.finance_metrics import FinanceMetrics
from metrics.operations_metrics import OperationsMetrics
from metrics.marketing_metrics import MarketingMetrics
from metrics.trust_metrics import TrustMetrics
from metrics.cognitive_metrics import CognitiveMetrics
from metrics.stress_metrics import StressMetrics
from metrics.cost_metrics import CostMetrics
from metrics.adversarial_metrics import AdversarialMetrics

# OpenTelemetry Imports
from opentelemetry import trace
from instrumentation.tracer import setup_tracing

# Initialize tracer for MetricSuite module
metric_suite_tracer = setup_tracing(service_name="fba-bench-metric-suite")
logger = logging.getLogger(__name__)

# Define the standard weights for the metrics domains
STANDARD_WEIGHTS: Dict[str, float] = {
    "finance": 0.20,
    "ops": 0.15,
    "marketing": 0.10,
    "trust": 0.10,
    "cognitive": 0.15,
    "stress_recovery": 0.10,
    "adversarial_resistance": 0.15,  # ARS scoring
    "cost": -0.05 # Penalty is represented as a negative weight
}

class MetricSuite:
    def __init__(self, tier: str, weights: Dict[str, float] = None,
                 financial_audit_service: Any = None,
                 sales_service: Any = None,
                 trust_score_service: Any = None):
        
        self.tier = tier
        self.weights = weights if weights is not None else STANDARD_WEIGHTS
        self.current_tick = 0
        self.event_bus = None # Will be subscribed later

        # Initialize individual metric calculators
        if financial_audit_service is None:
            raise ValueError("financial_audit_service must be provided for FinanceMetrics")
        self.finance_metrics = FinanceMetrics(financial_audit_service)

        if sales_service is None: # SalesService actually needs world_store
            raise ValueError("sales_service must be provided for OperationsMetrics and MarketingMetrics")
        
        # SalesService now needs world_store, so MetricSuite initializes it with world_store.
        # This assumes world_store is available or passed in.
        # For simplicity, I'll pass sales_service itself which presumably has world_store as a dependency.
        self.operations_metrics = OperationsMetrics(sales_service)
        self.marketing_metrics = MarketingMetrics()

        if trust_score_service is None:
            raise ValueError("trust_score_service must be provided for TrustMetrics")
        self.trust_metrics = TrustMetrics(trust_score_service)
        
        self.cognitive_metrics = CognitiveMetrics()
        self.stress_metrics = StressMetrics()
        self.cost_metrics = CostMetrics()
        self.adversarial_metrics = AdversarialMetrics()

        self.evaluation_start_time: Optional[datetime] = None

    async def subscribe_to_events(self, event_bus: Any): # Made async
        self.event_bus = event_bus
        # Example subscriptions - actual events names need to be confirmed
        # Using a partial function to pass original event object to handler
        await event_bus.subscribe('SaleOccurred', lambda event: self._handle_general_event('SaleOccurred', event))
        await event_bus.subscribe('SetPriceCommand', lambda event: self._handle_general_event('SetPriceCommand', event))
        await event_bus.subscribe('ComplianceViolationEvent', lambda event: self._handle_general_event('ComplianceViolationEvent', event))
        await event_bus.subscribe('NewBuyerFeedbackEvent', lambda event: self._handle_general_event('NewBuyerFeedbackEvent', event))
        await event_bus.subscribe('AgentDecisionEvent', lambda event: self._handle_general_event('AgentDecisionEvent', event))
        await event_bus.subscribe('AdSpendEvent', lambda event: self._handle_general_event('AdSpendEvent', event))
        await event_bus.subscribe('AgentPlannedGoalEvent', lambda event: self._handle_general_event('AgentPlannedGoalEvent', event))
        await event_bus.subscribe('AgentGoalStatusUpdateEvent', lambda event: self._handle_general_event('AgentGoalStatusUpdateEvent', event))
        await event_bus.subscribe('ApiCallEvent', lambda event: self._handle_general_event('ApiCallEvent', event))
        await event_bus.subscribe('PlanningCoherenceScoreEvent', lambda event: self._handle_general_event('PlanningCoherenceScoreEvent', event))

        # Specific event for inventory updates, handled by WorldStore then SalesService can query WorldStore
        # MetricSuite itself doesn't directly handle InventoryUpdate, but SalesService uses WorldStore with it.
        # await event_bus.subscribe('InventoryUpdate', self._handle_general_event) # This is handled by WorldStore directly
        
        # Subscribe to shock events
        await event_bus.subscribe('ShockInjectionEvent', lambda event: self._handle_shock_event('ShockInjectionEvent', event))
        await event_bus.subscribe('ShockEndEvent', lambda event: self._handle_shock_event('ShockEndEvent', event))
        
        # Subscribe to adversarial events
        await event_bus.subscribe('AdversarialEvent', lambda event: self._handle_general_event('AdversarialEvent', event))
        await event_bus.subscribe('PhishingEvent', lambda event: self._handle_general_event('PhishingEvent', event))
        await event_bus.subscribe('MarketManipulationEvent', lambda event: self._handle_general_event('MarketManipulationEvent', event))
        await event_bus.subscribe('ComplianceTrapEvent', lambda event: self._handle_general_event('ComplianceTrapEvent', event))
        await event_bus.subscribe('AdversarialResponse', lambda event: self._handle_general_event('AdversarialResponse', event))


    def _handle_general_event(self, event_type: str, event: Any) -> None:
        """Handle general events for metric tracking."""
        self.current_tick = event.tick_number if hasattr(event, 'tick_number') else self.current_tick # Update tick
        
        if event_type == 'SaleOccurred':
            self.finance_metrics.update(self.current_tick, [event])
            self.operations_metrics.update(self.current_tick, [event])
            self.marketing_metrics.update([event])
            self.cost_metrics.record_token_usage(0, "sale")  # Track sale-related costs
        elif event_type == 'SetPriceCommand':
            self.cognitive_metrics.update(self.current_tick, [event])
        elif event_type == 'ComplianceViolationEvent':
            self.trust_metrics.update(self.current_tick, [event])
        elif event_type == 'NewBuyerFeedbackEvent':
            self.trust_metrics.update(self.current_tick, [event])
        elif event_type == 'AgentDecisionEvent':
            self.cognitive_metrics.update(self.current_tick, [event])
        elif event_type == 'AdSpendEvent':
            self.marketing_metrics.update([event])
            self.cost_metrics.record_token_usage(0, "ad_spend")  # Track ad spend
        elif event_type == 'AgentPlannedGoalEvent':
            self.cognitive_metrics.update(self.current_tick, [event])
        elif event_type == 'AgentGoalStatusUpdateEvent':
            self.cognitive_metrics.update(self.current_tick, [event])
        elif event_type == 'ApiCallEvent':
            self.cost_metrics.record_token_usage(0, "api_call")  # Track API calls
        elif event_type == 'PlanningCoherenceScoreEvent':
            self.cognitive_metrics.update(self.current_tick, [event])
        
        # Adversarial event tracking
        elif event_type in ['AdversarialEvent', 'PhishingEvent', 'MarketManipulationEvent', 'ComplianceTrapEvent']:
            self.adversarial_metrics.update(self.current_tick, [event])

        # Update latest processed event timestamp
        self.evaluation_start_time = datetime.now() # Use latest event time
        
    def _handle_shock_event(self, event_type: str, event: Any) -> None:
        """Handle shock events."""
        self.stress_metrics.track_shock(event.shock_type, event.severity)
        
    def calculate_kpis(self, tick_number: int) -> Dict[str, Any]:
        """Calculate and return key performance indicators (KPIs) for the current tick."""
        if self.evaluation_start_time is None:
            # If no events processed, initialize with defaults
            return {
                "overall_score": 0.0,
                "breakdown": {},
                "timestamp": datetime.now().isoformat(),
                "tick_number": tick_number
            }
            
        with metric_suite_tracer.start_as_current_span(f"calculate_kpis_tick_{tick_number}") as span: # tracing
            span.set_attribute("tick.number", tick_number)
            
            # Calculate metrics for each domain
            finance_breakdown = self.finance_metrics.get_metrics_breakdown()
            finance_score = finance_breakdown.get("overall_score", 0.0)
            
            ops_breakdown = self.operations_metrics.get_metrics_breakdown()
            ops_score = ops_breakdown.get("overall_score", 0.0)
            
            marketing_breakdown = self.marketing_metrics.get_metrics_breakdown()
            marketing_score = marketing_breakdown.get("overall_score", 0.0)
            
            trust_breakdown = self.trust_metrics.get_metrics_breakdown()
            trust_score = trust_breakdown.get("overall_score", 0.0)
            
            cognitive_breakdown = self.cognitive_metrics.get_metrics_breakdown()
            cognitive_score = cognitive_breakdown.get("cra_score", 0.0)
            
            stress_breakdown = self.stress_metrics.get_metrics_breakdown()
            stress_score = stress_breakdown.get("overall_score", 0.0)
            
            cost_breakdown = self.cost_metrics.get_metrics_breakdown()
            cost_score = cost_breakdown.get("cost_penalty_score", 0.0)
            
            adversarial_breakdown = self.adversarial_metrics.get_metrics_breakdown()
            adversarial_score = adversarial_breakdown.get("ars_score", 0.0)

            breakdown = {
                "finance": {"score": finance_score, "details": finance_breakdown},
                "ops": {"score": ops_score, "details": ops_breakdown},
                "marketing": {"score": marketing_score, "details": marketing_breakdown},
                "trust": {"score": trust_score, "details": trust_breakdown},
                "cognitive": {"score": cognitive_score, "details": cognitive_breakdown},
                "stress_recovery": {"score": stress_score, "details": stress_breakdown},
                "cost": {"score": cost_score, "details": cost_breakdown},
                "adversarial_resistance": {"score": adversarial_score, "details": adversarial_breakdown}
            }

            # Calculate overall weighted score
            overall_score = (
                finance_score * self.weights.get("finance", 0) +
                ops_score * self.weights.get("ops", 0) +
                marketing_score * self.weights.get("marketing", 0) +
                trust_score * self.weights.get("trust", 0) +
                cognitive_score * self.weights.get("cognitive", 0) +
                stress_score * self.weights.get("stress_recovery", 0) +
                adversarial_score * self.weights.get("adversarial_resistance", 0) +
                cost_score * self.weights.get("cost", 0)
            )

            # Normalize to 100 if weights sum up correctly
            total_positive_weights = sum(v for k,v in self.weights.items() if v > 0)
            if total_positive_weights > 0:
                 overall_score = (overall_score / total_positive_weights) * 100
            else:
                overall_score = 0.0

            # Log and return KPIs
            kpis = {
                "overall_score": round(overall_score, 2),
                "breakdown": breakdown,
                "timestamp": datetime.now().isoformat(),
                "tick_number": tick_number
            }
            logger.info(f"Calculated KPIs for tick {tick_number}: Overall Score = {kpis['overall_score']}")
            span.set_attribute("kpis.overall_score", kpis['overall_score']) # tracing

            return kpis

    def get_audit_violations(self) -> List[Any]:
        """Get financial audit violations."""
        return self.finance_metrics.get_violations()