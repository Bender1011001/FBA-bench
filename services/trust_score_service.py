from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TrustScoreService:
    """
    Service responsible for calculating an agent's trust score.
    It does not maintain its own state but calculates based on input data
    provided by the caller (e.g., TrustMetrics).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.base_score = self.config.get("base_score", 100.0)
        self.violation_penalty = self.config.get("violation_penalty", 5.0) # Points deducted per violation
        self.feedback_weight = self.config.get("feedback_weight", 0.2) # Max 20% of base score influence from feedback
        self.min_score = self.config.get("min_score", 0.0)
        self.max_score = self.config.get("max_score", 100.0) # Can be higher than base_score for bonuses
        logger.info(f"TrustScoreService initialized with config: {self.config}")

    def calculate_trust_score(
        self,
        violations_count: int,
        buyer_feedback_scores: List[float],
        total_days: int # For potential future time-based factors
    ) -> float:
        """
        Calculates the trust score based on violations and buyer feedback.
        """
        current_score = self.base_score

        # Deduct for violations
        current_score -= violations_count * self.violation_penalty

        # Adjust based on buyer feedback (assuming feedback is 1-5 stars)
        if buyer_feedback_scores:
            avg_feedback = sum(buyer_feedback_scores) / len(buyer_feedback_scores)
            # Normalize feedback (1-5) to a scale that adjusts the score.
            # (avg_feedback - 3) maps 3 stars to 0, 1 star to -2, 5 stars to +2.
            # Multiply by a fraction of the base score to determine adjustment magnitude.
            feedback_normalization_factor = (avg_feedback - 3.0) / 2.0 # Results in -1 to 1
            max_feedback_adjustment = self.base_score * self.feedback_weight
            feedback_adjustment = feedback_normalization_factor * max_feedback_adjustment
            current_score += feedback_adjustment
        
        # Ensure score is within the configured valid range
        final_score = max(self.min_score, min(self.max_score, current_score))
        logger.debug(f"Calculated trust score: {final_score:.2f} (Violations: {violations_count}, Avg Feedback: {sum(buyer_feedback_scores)/len(buyer_feedback_scores) if buyer_feedback_scores else 0:.2f})")
        return final_score

    def get_current_trust_score(self) -> Optional[float]:
        """
        This method is not suitable for a stateless calculator.
        TrustMetrics should call `calculate_trust_score` with relevant data.
        Raising an error to highlight incorrect usage or need for refactoring if this interface is strict.
        """
        logger.error("TrustScoreService.get_current_trust_score() called on a stateless calculator. This indicates a design issue or misuse. Use calculate_trust_score method.")
        raise NotImplementedError("This TrustScoreService is stateless. Use calculate_trust_score method with relevant data.")

    async def start(self, event_bus=None): # EventBus might not be needed if stateless and called directly
        logger.info("TrustScoreService (stateless calculator) started.")

    async def stop(self):
        logger.info("TrustScoreService (stateless calculator) stopped.")