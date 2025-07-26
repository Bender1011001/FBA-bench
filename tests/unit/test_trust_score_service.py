"""
Unit tests for TrustScoreService.

These tests isolate the trust score calculation logic to verify correctness of:
- Trust score updates based on customer events
- Negative event rate calculations
- Score blending and smoothing
- Boundary conditions and edge cases
"""

import unittest
from fba_bench.services.trust_score_service import TrustScoreService


class TestTrustScoreService(unittest.TestCase):
    """Test suite for TrustScoreService."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = TrustScoreService()
    
    def test_new_product_default_trust_score(self):
        """Test that new products with no events get default high trust score."""
        # No customer events
        result = self.service.update_score(
            current_score=0.5,
            customer_events=[],
            total_orders=0
        )
        self.assertEqual(result, 1.0)
        
        # No total orders
        result = self.service.update_score(
            current_score=0.5,
            customer_events=[{"type": "positive_review"}],
            total_orders=0
        )
        self.assertEqual(result, 1.0)
    
    def test_no_negative_events_maintains_high_trust(self):
        """Test that products with no negative events maintain high trust."""
        positive_events = [
            {"type": "positive_review", "rating": 5},
            {"type": "order_completed"},
            {"type": "positive_feedback"}
        ]
        
        result = self.service.update_score(
            current_score=0.8,
            customer_events=positive_events,
            total_orders=10
        )
        
        # Should blend towards 1.0 (no negative events)
        # trust_score = 1.0, blended = 0.8 * 0.7 + 1.0 * 0.3 = 0.56 + 0.3 = 0.86
        expected = 0.86
        self.assertAlmostEqual(result, expected, places=2)
    
    def test_negative_events_decrease_trust_score(self):
        """Test that negative events decrease trust score."""
        negative_events = [
            {"type": "negative_review", "rating": 1},
            {"type": "a_to_z_claim", "reason": "defective"},
            {"type": "return_request", "reason": "not_as_described"}
        ]
        
        result = self.service.update_score(
            current_score=1.0,
            customer_events=negative_events,
            total_orders=10
        )
        
        # negative_rate = 3/10 = 0.3
        # trust_score = max(0.1, 1.0 - (0.3 * 2.0)) = max(0.1, 0.4) = 0.4
        # blended = 1.0 * 0.7 + 0.4 * 0.3 = 0.7 + 0.12 = 0.82
        expected = 0.82
        self.assertAlmostEqual(result, expected, places=2)
    
    def test_high_negative_rate_caps_at_minimum(self):
        """Test that very high negative rates are capped at minimum trust score."""
        negative_events = [
            {"type": "negative_review"},
            {"type": "a_to_z_claim"},
            {"type": "return_request"},
            {"type": "negative_review"},
            {"type": "a_to_z_claim"}
        ]
        
        result = self.service.update_score(
            current_score=0.2,
            customer_events=negative_events,
            total_orders=5  # 100% negative rate
        )
        
        # negative_rate = 5/5 = 1.0
        # trust_score = max(0.1, 1.0 - (1.0 * 2.0)) = max(0.1, -1.0) = 0.1
        # blended = 0.2 * 0.7 + 0.1 * 0.3 = 0.14 + 0.03 = 0.17
        expected = 0.17
        self.assertAlmostEqual(result, expected, places=2)
    
    def test_mixed_events_only_counts_negative(self):
        """Test that only negative events are counted in trust score calculation."""
        mixed_events = [
            {"type": "positive_review", "rating": 5},
            {"type": "negative_review", "rating": 1},
            {"type": "order_completed"},
            {"type": "a_to_z_claim"},
            {"type": "positive_feedback"},
            {"type": "return_request"}
        ]
        
        result = self.service.update_score(
            current_score=0.8,
            customer_events=mixed_events,
            total_orders=20
        )
        
        # Only 3 negative events out of 6 total events, but 20 total orders
        # negative_rate = 3/20 = 0.15
        # trust_score = max(0.1, 1.0 - (0.15 * 2.0)) = max(0.1, 0.7) = 0.7
        # blended = 0.8 * 0.7 + 0.7 * 0.3 = 0.56 + 0.21 = 0.77
        expected = 0.77
        self.assertAlmostEqual(result, expected, places=2)
    
    def test_score_blending_smooths_changes(self):
        """Test that score blending provides smooth transitions."""
        events = [{"type": "negative_review"}]
        
        # Test with high current score
        high_score_result = self.service.update_score(
            current_score=0.9,
            customer_events=events,
            total_orders=10
        )
        
        # Test with low current score
        low_score_result = self.service.update_score(
            current_score=0.3,
            customer_events=events,
            total_orders=10
        )
        
        # Both should be different due to blending, but closer to each other than raw calculation
        # negative_rate = 1/10 = 0.1
        # trust_score = max(0.1, 1.0 - (0.1 * 2.0)) = 0.8
        
        # High: 0.9 * 0.7 + 0.8 * 0.3 = 0.63 + 0.24 = 0.87
        # Low: 0.3 * 0.7 + 0.8 * 0.3 = 0.21 + 0.24 = 0.45
        
        self.assertAlmostEqual(high_score_result, 0.87, places=2)
        self.assertAlmostEqual(low_score_result, 0.45, places=2)
        
        # Verify blending effect: results are closer than original scores
        original_diff = abs(0.9 - 0.3)  # 0.6
        blended_diff = abs(high_score_result - low_score_result)  # ~0.42
        self.assertLess(blended_diff, original_diff)
    
    def test_boundary_conditions(self):
        """Test boundary conditions and edge cases."""
        # Test with zero current score, but with events to trigger blending
        result = self.service.update_score(
            current_score=0.0,
            customer_events=[{"type": "positive_review"}], # Add a non-negative event
            total_orders=1
        )
        # Should blend towards 1.0: 0.0 * 0.7 + 1.0 * 0.3 = 0.3
        self.assertAlmostEqual(result, 0.3, places=2)
        
        # Test with maximum current score
        result = self.service.update_score(
            current_score=1.0,
            customer_events=[],
            total_orders=1
        )
        # Should blend towards 1.0: 1.0 * 0.7 + 1.0 * 0.3 = 1.0
        self.assertEqual(result, 1.0)
        
        # Test result is always bounded [0.0, 1.0]
        result = self.service.update_score(
            current_score=0.0,
            customer_events=[{"type": "negative_review"}] * 100,
            total_orders=100
        )
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)
    
    def test_negative_event_types_recognition(self):
        """Test that all negative event types are properly recognized."""
        # Test each negative event type individually
        negative_event_types = ["negative_review", "a_to_z_claim", "return_request"]
        
        for event_type in negative_event_types:
            events = [{"type": event_type}]
            result = self.service.update_score(
                current_score=1.0,
                customer_events=events,
                total_orders=10
            )
            
            # Should be less than 1.0 due to negative event
            self.assertLess(result, 1.0, f"Event type {event_type} should decrease trust score")
    
    def test_unknown_event_types_ignored(self):
        """Test that unknown event types are ignored."""
        unknown_events = [
            {"type": "unknown_event"},
            {"type": "custom_event"},
            {"type": ""}
        ]
        
        result = self.service.update_score(
            current_score=0.8,
            customer_events=unknown_events,
            total_orders=10
        )
        
        # Should treat as no negative events
        # trust_score = 1.0, blended = 0.8 * 0.7 + 1.0 * 0.3 = 0.86
        expected = 0.86
        self.assertAlmostEqual(result, expected, places=2)
    
    def test_events_without_type_field(self):
        """Test handling of events without type field."""
        malformed_events = [
            {"rating": 1},  # No type field
            {"description": "bad product"},  # No type field
            {}  # Empty event
        ]
        
        result = self.service.update_score(
            current_score=0.8,
            customer_events=malformed_events,
            total_orders=10
        )
        
        # Should treat as no negative events since no valid type fields
        expected = 0.86  # Same as no negative events
        self.assertAlmostEqual(result, expected, places=2)
    
    def test_negative_rate_calculation_precision(self):
        """Test precise negative rate calculations."""
        # Test with exact fractions
        events = [{"type": "negative_review"}] * 3
        
        result = self.service.update_score(
            current_score=0.5,
            customer_events=events,
            total_orders=12  # 3/12 = 0.25 negative rate
        )
        
        # negative_rate = 3/12 = 0.25
        # trust_score = max(0.1, 1.0 - (0.25 * 2.0)) = max(0.1, 0.5) = 0.5
        # blended = 0.5 * 0.7 + 0.5 * 0.3 = 0.35 + 0.15 = 0.5
        expected = 0.5
        self.assertAlmostEqual(result, expected, places=2)
    
    def test_large_numbers_handling(self):
        """Test handling of large numbers of events and orders."""
        # Test with large numbers
        large_events = [{"type": "negative_review"}] * 1000
        
        result = self.service.update_score(
            current_score=0.7,
            customer_events=large_events,
            total_orders=10000  # 10% negative rate
        )
        
        # negative_rate = 1000/10000 = 0.1
        # trust_score = max(0.1, 1.0 - (0.1 * 2.0)) = max(0.1, 0.8) = 0.8
        # blended = 0.7 * 0.7 + 0.8 * 0.3 = 0.49 + 0.24 = 0.73
        expected = 0.73
        self.assertAlmostEqual(result, expected, places=2)
    
    def test_score_progression_over_time(self):
        """Test how trust score evolves over multiple updates."""
        current_score = 1.0
        
        # Simulate gradual degradation with negative events
        for i in range(5):
            events = [{"type": "negative_review"}] * (i + 1)
            current_score = self.service.update_score(
                current_score=current_score,
                customer_events=events,
                total_orders=10
            )
        
        # Score should have decreased from initial 1.0
        self.assertLess(current_score, 1.0)
        self.assertGreater(current_score, 0.0)
        
        # Now simulate recovery with no negative events
        for i in range(3):
            current_score = self.service.update_score(
                current_score=current_score,
                customer_events=[],  # No negative events
                total_orders=10
            )
        
        # Score should have improved but due to blending won't reach 1.0 immediately
        self.assertGreater(current_score, 0.5)


if __name__ == '__main__':
    unittest.main()