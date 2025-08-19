import pytest

from services.trust_score_service import TrustScoreService


def test_get_current_trust_score_deprecated_and_raises():
    svc = TrustScoreService()
    # Ensure a DeprecationWarning is emitted and NotImplementedError is raised
    with pytest.warns(DeprecationWarning):
        with pytest.raises(NotImplementedError):
            svc.get_current_trust_score()


@pytest.mark.parametrize(
    "violations,feedback,total_days",
    [
        (0, [5, 5, 5], 30),       # Best case feedback, no violations
        (3, [3, 4, 2], 10),       # Mixed feedback, some violations
        (10, [1, 1, 2], 90),      # Many violations, poor feedback
        (0, [], 1),               # No feedback provided
    ],
)
def test_calculate_trust_score_within_bounds(violations, feedback, total_days):
    config = {
        "base_score": 100.0,
        "violation_penalty": 5.0,
        "feedback_weight": 0.2,
        "min_score": 0.0,
        "max_score": 100.0,
    }
    svc = TrustScoreService(config=config)
    score = svc.calculate_trust_score(
        violations_count=violations,
        buyer_feedback_scores=feedback,
        total_days=total_days,
    )
    assert isinstance(score, float)
    assert config["min_score"] <= score <= config["max_score"]