# metrics/metric_suite.py
import json
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
        self.evaluation_end_time: Optional[datetime] = None


    def subscribe_to_events(self, event_bus: Any):
        self.event_bus = event_bus
        # Example subscriptions - actual events names need to be confirmed
        # Using a partial function to pass original event object to handler
        event_bus.subscribe('SaleOccurred', lambda event: self._handle_general_event('SaleOccurred', event))
        event_bus.subscribe('SetPriceCommand', lambda event: self._handle_general_event('SetPriceCommand', event))
        event_bus.subscribe('ComplianceViolationEvent', lambda event: self._handle_general_event('ComplianceViolationEvent', event))
        event_bus.subscribe('NewBuyerFeedbackEvent', lambda event: self._handle_general_event('NewBuyerFeedbackEvent', event))
        event_bus.subscribe('AgentDecisionEvent', lambda event: self._handle_general_event('AgentDecisionEvent', event))
        event_bus.subscribe('AdSpendEvent', lambda event: self._handle_general_event('AdSpendEvent', event))
        event_bus.subscribe('AgentPlannedGoalEvent', lambda event: self._handle_general_event('AgentPlannedGoalEvent', event))
        event_bus.subscribe('AgentGoalStatusUpdateEvent', lambda event: self._handle_general_event('AgentGoalStatusUpdateEvent', event))
        event_bus.subscribe('ApiCallEvent', lambda event: self._handle_general_event('ApiCallEvent', event))
        event_bus.subscribe('PlanningCoherenceScoreEvent', lambda event: self._handle_general_event('PlanningCoherenceScoreEvent', event))

        # Specific event for inventory updates, handled by WorldStore then SalesService can query WorldStore
        # MetricSuite itself doesn't directly handle InventoryUpdate, but SalesService uses WorldStore with it.
        # event_bus.subscribe('InventoryUpdate', self._handle_general_event) # This is handled by WorldStore directly
        
        # Subscribe to shock events
        event_bus.subscribe('ShockInjectionEvent', lambda event: self._handle_shock_event('ShockInjectionEvent', event))
        event_bus.subscribe('ShockEndEvent', lambda event: self._handle_shock_event('ShockEndEvent', event))
        
        # Subscribe to adversarial events
        event_bus.subscribe('AdversarialEvent', lambda event: self._handle_general_event('AdversarialEvent', event))
        event_bus.subscribe('PhishingEvent', lambda event: self._handle_general_event('PhishingEvent', event))
        event_bus.subscribe('MarketManipulationEvent', lambda event: self._handle_general_event('MarketManipulationEvent', event))
        event_bus.subscribe('ComplianceTrapEvent', lambda event: self._handle_general_event('ComplianceTrapEvent', event))
        event_bus.subscribe('AdversarialResponse', lambda event: self._handle_general_event('AdversarialResponse', event))
        
        # Add more event subscriptions as identified in other modules

    def _handle_general_event(self, event_name: str, event_data: Dict):
        # Dispatch events to appropriate metric handlers
        # Using event_bus_tracer for general event handling
        with metric_suite_tracer.start_as_current_span(
            f"metric_suite._handle_general_event.{event_name}",
            attributes={
                "event.type": event_name,
                "event.id": event_data.event_id if hasattr(event_data, 'event_id') else "N/A",
                "tick": self.current_tick
            }
        ):
            events_list = [event_data] # Wrap single event in a list for update methods

        self.finance_metrics.update(self.current_tick, events_list)
        self.operations_metrics.update(self.current_tick, events_list) # operations_metrics now expects events
        self.marketing_metrics.update(events_list)
        self.trust_metrics.update(self.current_tick, events_list)
        self.cognitive_metrics.update(self.current_tick, events_list)
        self.cost_metrics.update(events_list)
        self.adversarial_metrics.update(self.current_tick, events_list)

    def _handle_shock_event(self, event_name: str, event_data: Any): # Use Any as event_data is BaseEvent or dict
        shock_id = event_data.get('shock_id', 'unknown_shock') if isinstance(event_data, dict) else getattr(event_data, 'event_id', 'unknown_shock_event')
        
        with metric_suite_tracer.start_as_current_span(
            f"metric_suite._handle_shock_event.{event_name}",
            attributes={
                "event.type": event_name,
                "event.id": shock_id,
                "tick": self.current_tick
            }
        ):
            # Stress metrics need a performance metric snapshot - use net worth as proxy
            current_net_worth = self.finance_metrics.financial_audit_service.get_current_net_worth()

            if event_name == 'ShockInjectionEvent':
                self.finance_metrics.record_shock_snapshot(shock_id, "before")
                self.stress_metrics.update(self.current_tick, [event_data], current_net_worth)
            elif event_name == 'ShockEndEvent':
                self.finance_metrics.record_shock_snapshot(shock_id, "after") # Consider 'during' if there are intermediate states
                self.stress_metrics.update(self.current_tick, [event_data], current_net_worth)
            # For 'during' shock, updates would happen on regular ticks within the shock
            self.stress_metrics.update(self.current_tick, [], current_net_worth) # Continuous update for MTTR calculation


    def advance_tick(self, current_tick: int, events: List[Dict]):
        with metric_suite_tracer.start_as_current_span(
            f"metric_suite.advance_tick.{current_tick}",
            attributes={"tick": current_tick, "event_count": len(events)}
        ):
            self.current_tick = current_tick
            for event in events:
                # Need to cast event_data as BaseEvent if possible for consistency
                # Assuming events in this list are already dictionaries
                event_type = event.get('type')
                if event_type == 'ShockInjectionEvent' or event_type == 'ShockEndEvent':
                    self._handle_shock_event(event_type, event)
                else:
                    self._handle_general_event(event_type, event)
            
            # Call stress_metrics update even if no shock event occurred for continuous performance tracking
            current_net_worth = self.finance_metrics.financial_audit_service.get_current_net_worth()
            self.stress_metrics.update(self.current_tick, [], current_net_worth)


    def get_current_score(self) -> Dict[str, Any]:
        return self._calculate_overall_metrics()

    def calculate_final_score(self, event_log: List[Dict], evaluation_period: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        # Reset metrics for offline calculation
        self.__init__(self.tier, self.weights, 
                      financial_audit_service=self.finance_metrics.financial_audit_service,
                      sales_service=self.operations_metrics.sales_service,
                      trust_score_service=self.trust_metrics.trust_score_service) # Re-initialize with services

        if evaluation_period:
            self.evaluation_start_time, self.evaluation_end_time = evaluation_period
            # Filter events based on timestamp if logs have them
            # For simplicity, assuming ticks align with event order
        
        # Reset metrics for offline calculation (a proper reset method would be better)
        # Re-initialize all metric objects and their internal states for a fresh run
        self.__init__(self.tier, self.weights,
                      financial_audit_service=self.finance_metrics.financial_audit_service,
                      sales_service=self.operations_metrics.sales_service,
                      trust_score_service=self.trust_metrics.trust_score_service)

        # Process events chronologically for offline analysis
        # Assuming event_log is sorted by tick/timestamp
        if evaluation_period:
            # Filter events within the evaluation period
            start_time, end_time = evaluation_period
            relevant_events = [
                e for e in event_log
                if 'timestamp' in e and start_time <= e['timestamp'] <= end_time
            ]
        else:
            relevant_events = event_log
        
        # Group events by tick for sequential processing
        events_by_tick: Dict[int, List[Dict]] = {}
        for event in relevant_events:
            tick = event.get('tick') # Assuming 'tick' exists in logged events
            if tick is None:
                continue # Or handle events without a tick
            if tick not in events_by_tick:
                events_by_tick[tick] = []
            events_by_tick[tick].append(event)
        
        sorted_ticks = sorted(events_by_tick.keys())

        # Simulate tick-by-tick progression for accurate metric updates
        for tick in sorted_ticks:
            self.current_tick = tick
            current_tick_events = events_by_tick[tick]
            
            # First, handle general events
            for event in current_tick_events:
                # Need to use the event type from the loaded event, not hardcoded
                event_type = event.get('type')
                if event_type == 'ShockInjectionEvent' or event_type == 'ShockEndEvent':
                    self._handle_shock_event(event_type, event)
                else:
                    self._handle_general_event(event_type, event) # Pass each event individually to handlers

            # After processing all current tick events, update stress metrics based on current state
            # This requires current net worth after all events of this tick
            current_net_worth_for_offline = self.finance_metrics.financial_audit_service.get_current_net_worth()
            self.stress_metrics.update(self.current_tick, [], current_net_worth_for_offline) # Pass empty events, as already processed

        self.evaluation_end_time = datetime.utcnow() # Set end time for output contract
        
        # After processing all events, calculate the final metrics
        return self._calculate_overall_metrics() # Delegate to the common calculation method

        return self._calculate_overall_metrics()

    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        with metric_suite_tracer.start_as_current_span(
            "metric_suite.calculate_overall_metrics",
            attributes={"tick_at_calculation": self.current_tick}
        ):
            # Calculate scores for each domain
            finance_breakdown = self.finance_metrics.get_metrics_breakdown()
            ops_breakdown = self.operations_metrics.get_metrics_breakdown()
            marketing_breakdown = self.marketing_metrics.get_metrics_breakdown()
            trust_breakdown = self.trust_metrics.get_metrics_breakdown()
            cognitive_breakdown = self.cognitive_metrics.get_metrics_breakdown()
            stress_breakdown = self.stress_metrics.get_metrics_breakdown()
            cost_breakdown = self.cost_metrics.get_metrics_breakdown()
            adversarial_breakdown = self.adversarial_metrics.get_metrics_breakdown()

        # Combine domain-specific scores into a single breakdown
        breakdown = {
            "finance": finance_breakdown.get("resilient_net_worth", 0.0), # Example, chosen as primary
            "ops": (ops_breakdown.get("inventory_turnover", 0.0) * 0.5 + (100 - ops_breakdown.get("stockout_percentage", 0.0)) * 0.5), # Average of turnover and inverse stockout
            "marketing": marketing_breakdown.get("weighted_roas_acos", 0.0),
            "trust": (trust_breakdown.get("violation_free_days_percentage", 0.0) * 0.5 + trust_breakdown.get("average_buyer_feedback_score", 0.0) * 0.5),
            "cognitive": cognitive_breakdown.get("cra_score", 0.0),
            "stress_recovery": stress_breakdown.get("normalized_mttr_score", 0.0),
            "cost": cost_breakdown.get("cost_penalty_score", 0.0)
        }

        # Calculate final composite score
        total_score = 0.0
        for domain, score_value in breakdown.items():
            weight = self.weights.get(domain, 0.0)
            total_score += score_value * weight
        
        # Adjust total score to be within 0-100 range, considering potential penalties
        # A simple normalization that ensures positive values for weighted domains and applies penalty
        final_score = total_score
        final_score = max(0, min(100, final_score)) # Clamp to 0-100

        # Construct the final JSON output contract
        result = {
            "score": round(final_score, 2),
            "breakdown": {k: round(v, 2) for k, v in breakdown.items()},
            "cost_usd": round(cost_breakdown.get("cost_usd", 0.0), 2),
            "token_usage": int(cost_breakdown.get("token_usage", 0)),
            "evaluation_period": self.evaluation_end_time.isoformat() if self.evaluation_end_time else datetime.utcnow().isoformat() + "Z",
            "tier": self.tier
        }
        return result