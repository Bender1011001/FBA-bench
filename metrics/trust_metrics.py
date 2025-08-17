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
        
        # The TrustScoreService is now used to calculate a holistic score in get_metrics_breakdown.
        # No direct integration needed here unless the service itself needs to be updated with events.


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
        # avg_buyer_feedback_score = self.calculate_buyer_feedback_score() # Old way

        # New way: Use TrustScoreService for a more holistic trust score
        holistic_trust_score = 0.0
        if hasattr(self.trust_score_service, 'calculate_trust_score'):
             try:
                holistic_trust_score = self.trust_score_service.calculate_trust_score(
                    violations_count=self.violations_count,
                    buyer_feedback_scores=self.buyer_feedback_scores,
                    total_days=self.total_days
                )
             except Exception as e:
                logger.error(f"Error calculating holistic trust score: {e}")
                holistic_trust_score = 0.0 # Fallback
        else:
            logger.warning("TrustScoreService does not have calculate_trust_score method. Using fallback.")
            # Fallback to old average if service is not as expected, or some other default
            if self.buyer_feedback_scores:
                 holistic_trust_score = sum(self.buyer_feedback_scores) / len(self.buyer_feedback_scores)
            else: # If no feedback, use violation_free_days_pct as a proxy or a default
                 holistic_trust_score = violation_free_days_pct


        return {
            "violation_free_days_percentage": violation_free_days_pct,
            # "average_buyer_feedback_score": avg_buyer_feedback_score, # Replaced
            "holistic_trust_score": holistic_trust_score, # New
        }