"""
Customer Event Service.

Handles all customer event generation including reviews, returns, claims,
and other customer interactions with realistic behavior patterns.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import random
from fba_bench.money import Money


class CustomerEventService:
    """
    Service responsible for generating realistic customer events and interactions.
    
    This service encapsulates the complex customer behavior simulation logic
    that was previously embedded in the monolithic tick_day() method.
    """
    
    def __init__(self, rng: random.Random, config: Dict[str, Any]):
        """
        Initialize customer event service.
        
        Args:
            rng: Seeded random number generator for deterministic behavior
            config: Configuration dictionary containing event probabilities and settings
        """
        self.rng = rng
        self.config = config
        
        # Customer segment definitions
        self.customer_segments = {
            "price_sensitive": {"weight": 0.4, "price_factor": 2.0, "review_rate": 0.8},
            "quality_focused": {"weight": 0.3, "price_factor": 0.5, "review_rate": 1.5},
            "convenience_focused": {"weight": 0.2, "price_factor": 1.0, "review_rate": 0.6},
            "brand_loyal": {"weight": 0.1, "price_factor": 0.3, "review_rate": 1.2}
        }
        
        # Category-specific behavior factors
        self.category_factors = {
            "Electronics": {"defect_rate": 1.2, "return_rate": 1.3, "review_rate": 1.1},
            "Health": {"defect_rate": 0.8, "return_rate": 1.5, "review_rate": 1.3},
            "Beauty": {"defect_rate": 0.9, "return_rate": 1.4, "review_rate": 1.2},
            "Toys": {"defect_rate": 1.1, "return_rate": 1.2, "review_rate": 0.9},
            "Books": {"defect_rate": 0.7, "return_rate": 0.8, "review_rate": 1.4},
            "DEFAULT": {"defect_rate": 1.0, "return_rate": 1.0, "review_rate": 1.0}
        }
    
    def generate_customer_events(
        self,
        asin: str,
        product: Any,
        units_sold: int,
        current_date: datetime,
        avg_comp_price: Optional[Money] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate customer events for a product based on sales and market conditions.
        
        Args:
            asin: Product ASIN
            product: Product object
            units_sold: Number of units sold
            current_date: Current simulation date
            avg_comp_price: Average competitor price for comparison
            
        Returns:
            List of generated customer events
        """
        events = []
        
        for _ in range(units_sold):
            event = self._generate_single_customer_event(
                asin, product, current_date, avg_comp_price
            )
            if event:
                events.append(event)
        
        return events
    
    def _generate_single_customer_event(
        self,
        asin: str,
        product: Any,
        current_date: datetime,
        avg_comp_price: Optional[Money] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate a single customer event based on probabilities and product characteristics."""
        # Select customer segment
        selected_segment = self._select_customer_segment()
        segment_props = self.customer_segments[selected_segment]
        
        # Calculate price penalty if competitor pricing available
        price_penalty = self._calculate_price_penalty(
            product.price, avg_comp_price, segment_props["price_factor"]
        )
        
        # Get category-specific factors
        category_factor = self.category_factors.get(
            product.category, self.category_factors["DEFAULT"]
        )
        
        # Calculate seasonal adjustments
        season_factor = self._calculate_seasonal_factor(current_date)
        
        # Calculate trust-based factors
        trust_score = getattr(product, 'trust_score', 1.0)
        trust_factor = max(0.1, trust_score)
        review_multiplier = (
            segment_props["review_rate"] * 
            category_factor["review_rate"] * 
            season_factor
        )
        
        # Calculate event probabilities
        probabilities = self._calculate_event_probabilities(
            trust_factor, review_multiplier, price_penalty, category_factor
        )
        
        # Generate event based on probabilities
        return self._select_event_type(
            probabilities, asin, selected_segment, current_date
        )
    
    def _select_customer_segment(self) -> str:
        """Select a customer segment based on weighted probabilities."""
        segment_roll = self.rng.random()
        cumulative = 0
        
        for segment, props in self.customer_segments.items():
            cumulative += props["weight"]
            if segment_roll <= cumulative:
                return segment
        
        return "price_sensitive"  # Default fallback
    
    def _calculate_price_penalty(
        self,
        product_price: Money,
        avg_comp_price: Optional[Money],
        price_factor: float
    ) -> float:
        """Calculate price penalty based on competitive pricing."""
        if avg_comp_price is None or product_price <= avg_comp_price:
            return 0.0
        
        price_diff_ratio = (product_price - avg_comp_price) / avg_comp_price
        base_penalty = min(0.15, float(price_diff_ratio) * 0.3)
        return base_penalty * price_factor
    
    def _calculate_seasonal_factor(self, current_date: datetime) -> float:
        """Calculate seasonal adjustment factor."""
        month = current_date.month
        if month in [11, 12]:  # Holiday season
            return 1.2  # More reviews and issues during holidays
        elif month in [6, 7, 8]:  # Summer
            return 0.9  # Slightly less activity
        return 1.0
    
    def _calculate_event_probabilities(
        self,
        trust_factor: float,
        review_multiplier: float,
        price_penalty: float,
        category_factor: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate probabilities for different event types."""
        return {
            "positive_review": max(0.02, min(0.25,
                0.12 * trust_factor * review_multiplier - price_penalty * 0.5)),
            "negative_review": max(0.01, min(0.15,
                (0.08 - 0.06 * trust_factor) * category_factor["defect_rate"] + price_penalty)),
            "a_to_z_claim": max(0.001, min(0.05,
                (0.015 - 0.012 * trust_factor) * category_factor["return_rate"] + price_penalty * 0.3)),
            "return_request": max(0.005, min(0.08,
                0.03 * category_factor["return_rate"] + price_penalty * 0.2)),
            "customer_message": 0.02 + (1.0 - trust_factor) * 0.02,
            "seller_feedback": max(0.01, 0.015 * review_multiplier),
            "product_question": 0.025  # Base rate, adjusted by segment in selection
        }
    
    def _select_event_type(
        self,
        probabilities: Dict[str, float],
        asin: str,
        selected_segment: str,
        current_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """Select and generate a specific event type based on probabilities."""
        r = self.rng.random()
        cumulative = 0
        
        # Positive review
        cumulative += probabilities["positive_review"]
        if r < cumulative:
            return self._generate_positive_review(asin, selected_segment, current_date)
        
        # Negative review
        cumulative += probabilities["negative_review"]
        if r < cumulative:
            return self._generate_negative_review(asin, selected_segment, current_date)
        
        # A-to-Z claim
        cumulative += probabilities["a_to_z_claim"]
        if r < cumulative:
            return self._generate_a_to_z_claim(asin, selected_segment, current_date)
        
        # Return request
        cumulative += probabilities["return_request"]
        if r < cumulative:
            return self._generate_return_request(asin, selected_segment, current_date)
        
        # Customer message
        cumulative += probabilities["customer_message"]
        if r < cumulative:
            return self._generate_customer_message(asin, selected_segment, current_date)
        
        # Seller feedback
        cumulative += probabilities["seller_feedback"]
        if r < cumulative:
            return self._generate_seller_feedback(asin, selected_segment, current_date)
        
        # Product question (adjusted for quality-focused customers)
        question_prob = probabilities["product_question"]
        if selected_segment == "quality_focused":
            question_prob = 0.025
        else:
            question_prob = 0.01
        
        cumulative += question_prob
        if r < cumulative:
            return self._generate_product_question(asin, selected_segment, current_date)
        
        return None  # No event generated
    
    def _generate_positive_review(
        self, asin: str, segment: str, date: datetime
    ) -> Dict[str, Any]:
        """Generate a positive review event."""
        positive_messages = [
            "Great product, exactly as described!",
            "Fast shipping and excellent quality.",
            "Highly recommend this item.",
            "Perfect for my needs, very satisfied.",
            "Good value for the price.",
            "Works as expected, no issues.",
            "Quick delivery and well packaged."
        ]
        
        score = self.rng.choices([4, 5], weights=[0.3, 0.7])[0]
        
        return {
            "type": "positive_review",
            "asin": asin,
            "score": score,
            "text": self.rng.choice(positive_messages),
            "customer_segment": segment,
            "date": date
        }
    
    def _generate_negative_review(
        self, asin: str, segment: str, date: datetime
    ) -> Dict[str, Any]:
        """Generate a negative review event."""
        negative_messages = [
            "Product did not meet expectations.",
            "Poor quality, broke after a few uses.",
            "Not as described in the listing.",
            "Overpriced for what you get.",
            "Shipping was delayed significantly.",
            "Product arrived damaged.",
            "Doesn't work as advertised."
        ]
        
        score = self.rng.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        
        return {
            "type": "negative_review",
            "asin": asin,
            "score": score,
            "text": self.rng.choice(negative_messages),
            "customer_segment": segment,
            "date": date
        }
    
    def _generate_a_to_z_claim(
        self, asin: str, segment: str, date: datetime
    ) -> Dict[str, Any]:
        """Generate an A-to-Z claim event."""
        claim_reasons = [
            "Item not as described",
            "Product defective",
            "Never received item",
            "Item significantly different from listing",
            "Product stopped working"
        ]
        
        return {
            "type": "a_to_z_claim",
            "asin": asin,
            "reason": self.rng.choice(claim_reasons),
            "customer_segment": segment,
            "date": date
        }
    
    def _generate_return_request(
        self, asin: str, segment: str, date: datetime
    ) -> Dict[str, Any]:
        """Generate a return request event."""
        return_reasons = [
            "Changed mind",
            "Found better price elsewhere",
            "Product not needed",
            "Size/fit issues",
            "Quality concerns"
        ]
        
        return {
            "type": "return_request",
            "asin": asin,
            "reason": self.rng.choice(return_reasons),
            "customer_segment": segment,
            "date": date
        }
    
    def _generate_customer_message(
        self, asin: str, segment: str, date: datetime
    ) -> Dict[str, Any]:
        """Generate a customer message event."""
        message_types = [
            "Can you provide more details about this product?",
            "When will this item ship?",
            "Is this compatible with [other product]?",
            "What is your return policy?",
            "Can you expedite shipping?"
        ]
        
        return {
            "type": "message",
            "asin": asin,
            "content": self.rng.choice(message_types),
            "customer_segment": segment,
            "date": date
        }
    
    def _generate_seller_feedback(
        self, asin: str, segment: str, date: datetime
    ) -> Dict[str, Any]:
        """Generate seller feedback event."""
        # Trust score affects feedback distribution
        feedback_scores = [1, 2, 3, 4, 5]
        feedback_weights = [0.05, 0.1, 0.15, 0.3, 0.4]  # Default positive distribution
        
        score = self.rng.choices(feedback_scores, weights=feedback_weights)[0]
        
        feedback_messages = {
            1: "Poor communication and slow shipping.",
            2: "Below average service.",
            3: "Average seller experience.",
            4: "Good seller, fast shipping.",
            5: "Excellent seller, highly recommend!"
        }
        
        return {
            "type": "seller_feedback",
            "asin": asin,
            "score": score,
            "text": feedback_messages[score],
            "customer_segment": segment,
            "date": date
        }
    
    def _generate_product_question(
        self, asin: str, segment: str, date: datetime
    ) -> Dict[str, Any]:
        """Generate a product question event."""
        questions = [
            "What are the exact dimensions?",
            "Is this item in stock?",
            "What materials is this made from?",
            "How long is the warranty?",
            "Are there any color options?"
        ]
        
        return {
            "type": "product_question",
            "asin": asin,
            "question": self.rng.choice(questions),
            "customer_segment": segment,
            "date": date
        }
    
    def calculate_trust_score(
        self,
        customer_events: List[Dict[str, Any]],
        total_orders: int
    ) -> float:
        """
        Calculate trust score based on customer event history.
        
        Args:
            customer_events: List of customer events for the product
            total_orders: Total number of orders for normalization
            
        Returns:
            Trust score between 0.0 and 1.0
        """
        if not customer_events or total_orders == 0:
            return 1.0  # Default high trust for new products
        
        # Count negative events
        negative_events = [
            e for e in customer_events 
            if e["type"] in ["negative_review", "a_to_z_claim", "return_request"]
        ]
        
        # Calculate trust score based on negative event rate
        negative_rate = len(negative_events) / total_orders
        
        # Trust score decreases with negative event rate
        trust_score = max(0.1, 1.0 - (negative_rate * 2.0))  # Cap at 0.1 minimum
        
        return min(1.0, trust_score)  # Cap at 1.0 maximum