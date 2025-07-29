# metrics/trust_metrics.py
from typing import List, Dict, Any

class TrustMetrics:
    def __init__(self, trust_score_service: Any):
        self.trust_score_service = trust_score_service
        self.violation_free_days = 0
        self.total_days = 0
        self.violations_count = 0
        self.buyer_feedback_scores: List[float] = [] # Raw scores from service

    def update(self, current_tick: int, events: List[Dict]):
        self.total_days = current_tick

        has_violation_today = False
        for event in events:
            if event.get('type') == 'ComplianceViolationEvent': # Assuming such an event exists
                self.violations_count += 1
                has_violation_today = True
            # For buyer feedback, we would need events indicating new feedback
            # Assuming 'NewBuyerFeedbackEvent' with a 'score' field
            elif event.get('type') == 'NewBuyerFeedbackEvent':
                score = event.get('score')
                if score is not None:
                    self.buyer_feedback_scores.append(score)

        if not has_violation_today:
            self.violation_free_days += 1
        
        # Integrate with trust_score_service for overall buyer feedback
        # Assuming trust_score_service aggregates and provides a current score
        current_buyer_feedback = self.trust_score_service.get_current_trust_score()
        if current_buyer_feedback is not None:
            # We might already have it from events, but this is a fallback/aggregation point
            pass # The individual scores are tracked in self.buyer_feedback_scores


    def calculate_violation_free_days(self) -> float:
        if self.total_days == 0:
            return 0.0
        return (self.violation_free_days / self.total_days) * 100 # Percentage of days without violations

    def calculate_buyer_feedback_score(self) -> float:
        if not self.buyer_feedback_scores:
            return 0.0
        # This is an average of individual feedback scores.
        # trust_score_service might have a more sophisticated aggregate.
        return sum(self.buyer_feedback_scores) / len(self.buyer_feedback_scores)

    def get_metrics_breakdown(self) -> Dict[str, float]:
        violation_free_days_pct = self.calculate_violation_free_days()
        avg_buyer_feedback_score = self.calculate_buyer_feedback_score()

        return {
            "violation_free_days_percentage": violation_free_days_pct,
            "average_buyer_feedback_score": avg_buyer_feedback_score,
        }