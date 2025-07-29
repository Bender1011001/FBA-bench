# metrics/stress_metrics.py
from typing import List, Dict, Any

class StressMetrics:
    def __init__(self):
        self.shock_events: Dict[str, Dict] = {} # shock_id: {'start_tick': int, 'end_tick': int, 'recovery_tick': int, 'impact_metric': float, 'baseline_metric': float}
        self.baseline_performance_history: List[float] = [] # Snapshot of a key performance metric before shocks
        self.performance_during_shock: Dict[str, List[float]] = {}
        self.performance_post_shock: Dict[str, List[float]] = {}

    def update(self, current_tick: int, events: List[Dict], current_performance_metric: float):
        self.baseline_performance_history.append(current_performance_metric) # e.g., Net Worth

        for event in events:
            if event.get('type') == 'ShockInjectionEvent':
                shock_id = event['shock_id']
                if shock_id not in self.shock_events:
                    self.shock_events[shock_id] = {
                        'start_tick': current_tick,
                        'end_tick': -1,
                        'recovery_tick': -1,
                        'baseline_metric_at_shock': current_performance_metric # Performance at moment of shock
                    }
                else: # Likely 'during' or 'end' of shock
                    pass # Handled by specific end shock event or continuous monitoring

            elif event.get('type') == 'ShockEndEvent':
                shock_id = event['shock_id']
                if shock_id in self.shock_events and self.shock_events[shock_id]['end_tick'] == -1:
                    self.shock_events[shock_id]['end_tick'] = current_tick
                    # Store performance at the end of the shock (lowest point or immediate aftermath)
                    self.shock_events[shock_id]['impact_metric'] = current_performance_metric

        # Track performance during and after active shocks
        for shock_id, shock_data in self.shock_events.items():
            if shock_data['start_tick'] != -1 and shock_data['end_tick'] == -1: # Shock is active
                if shock_id not in self.performance_during_shock:
                    self.performance_during_shock[shock_id] = []
                self.performance_during_shock[shock_id].append(current_performance_metric)
            elif shock_data['end_tick'] != -1 and shock_data['recovery_tick'] == -1: # Post shock, awaiting recovery
                if shock_id not in self.performance_post_shock:
                    self.performance_post_shock[shock_id] = []
                self.performance_post_shock[shock_id].append(current_performance_metric)
                
                # Check for recovery
                if current_performance_metric >= shock_data['baseline_metric_at_shock'] * 0.95: # Arbitrary 95% recovery threshold
                    self.shock_events[shock_id]['recovery_tick'] = current_tick


    def calculate_mttr(self) -> Dict[str, float]:
        mttr_scores = {}
        for shock_id, shock_data in self.shock_events.items():
            if shock_data['start_tick'] != -1 and shock_data['recovery_tick'] != -1:
                mttr = shock_data['recovery_tick'] - shock_data['start_tick']
                mttr_scores[shock_id] = mttr
            elif shock_data['start_tick'] != -1 and shock_data['end_tick'] != -1 and shock_data['recovery_tick'] == -1 and shock_data['recovery_tick'] != -1: # Shock ended but no recovery yet
                mttr_scores[shock_id] = float('inf') # Indicate non-recovery
        
        return mttr_scores

    def get_metrics_breakdown(self) -> Dict[str, float]:
        mttr_results = self.calculate_mttr()
        
        # Average MTTR if multiple shocks, otherwise just the one
        avg_mttr = sum(v for v in mttr_results.values() if v != float('inf')) / len(mttr_results) if mttr_results else 0.0
        
        # A higher MTTR is worse, so we inverse it for a score (example)
        # We want to transform MTTR into a 0-100 score where lower MTTR is better
        # Let's say max acceptable MTTR is 50 ticks, recover in 1 tick = 100, 50 ticks = 0
        normalized_mttr_score = 100 - (avg_mttr * 2) if avg_mttr <= 50 else 0
        normalized_mttr_score = max(0, min(100, normalized_mttr_score))


        return {
            "mean_time_to_recovery": avg_mttr,
            "normalized_mttr_score": normalized_mttr_score
        }