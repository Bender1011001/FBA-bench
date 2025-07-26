"""
Competitor data model for FBA-Bench simulation.

This module defines the Competitor dataclass that represents competitor products
in the marketplace simulation.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from fba_bench.money import Money


@dataclass
class Competitor:
    """
    Represents a competitor product in the FBA-Bench simulation.
    
    This class models competitor behavior and characteristics in the marketplace,
    including pricing strategies, sales performance, and market positioning.
    
    Attributes:
        asin (str): Amazon Standard Identification Number for the competitor.
        category (str): Product category (e.g., "Electronics", "Health", "Beauty").
        cost (Money): Wholesale cost per unit (assumed similar to agent's cost).
        price (Money): Current retail price of the competitor's product.
        base_demand (float): Baseline demand if price were $1.
        bsr (int): Best Seller Rank, dynamically updated based on performance.
        sales_velocity (float): Current sales velocity (units per day).
        strategy (str): Pricing strategy ("aggressive", "follower", "premium", "value").
        sales_history (List[float]): Historical sales data for trend analysis.
        price_history (List[float]): Historical pricing data for strategy analysis.
        trust_score (float): Competitor's trust/reputation score (0.0-1.0).
        brand_strength (float): Brand recognition and loyalty factor (0.0-1.0).
        inventory_level (int): Estimated current inventory level.
        market_share (float): Estimated market share percentage (0.0-1.0).
        review_count (int): Total number of customer reviews.
        average_rating (float): Average customer rating (1.0-5.0).
        fulfillment_method (str): "FBA", "FBM", or "Hybrid".
        seasonal_factor (float): Seasonal demand multiplier.
        price_elasticity (float): Price sensitivity factor.
    """
    
    # Core identification and categorization
    asin: str
    category: str
    
    # Financial attributes
    cost: Money
    price: Money
    base_demand: float = 20.0
    
    # Performance metrics
    bsr: int = 100000
    sales_velocity: float = 0.0
    
    # Behavioral characteristics
    strategy: str = "follower"  # Default strategy
    
    # Historical data for analysis
    sales_history: List[float] = field(default_factory=list)
    price_history: List[float] = field(default_factory=list)
    
    # Market positioning attributes
    trust_score: float = 1.0
    brand_strength: float = 0.5
    inventory_level: int = 1000
    market_share: float = 0.0
    
    # Customer feedback metrics
    review_count: int = 100
    average_rating: float = 4.0
    
    # Operational characteristics
    fulfillment_method: str = "FBA"
    
    # Market dynamics factors
    seasonal_factor: float = 1.0
    price_elasticity: float = 1.0
    
    def __post_init__(self):
        """Initialize computed fields and validate data."""
        # Ensure strategy is valid
        valid_strategies = ["aggressive", "follower", "premium", "value"]
        if self.strategy not in valid_strategies:
            self.strategy = "follower"
        
        # Initialize history if empty
        if not self.sales_history:
            self.sales_history = [self.sales_velocity]
        if not self.price_history:
            self.price_history = [float(self.price.to_decimal())]
    
    def update_sales_history(self, sales: float, max_history: int = 30) -> None:
        """
        Update sales history with new sales data.
        
        Args:
            sales: New sales value to add
            max_history: Maximum number of historical points to keep
        """
        self.sales_history.append(sales)
        if len(self.sales_history) > max_history:
            self.sales_history.pop(0)
    
    def update_price_history(self, price: Money, max_history: int = 30) -> None:
        """
        Update price history with new price data.
        
        Args:
            price: New price value to add
            max_history: Maximum number of historical points to keep
        """
        self.price_history.append(float(price.to_decimal()))
        if len(self.price_history) > max_history:
            self.price_history.pop(0)
    
    def get_average_sales_velocity(self, days: int = 7) -> float:
        """
        Calculate average sales velocity over specified period.
        
        Args:
            days: Number of recent days to average
            
        Returns:
            Average sales velocity
        """
        if not self.sales_history:
            return self.sales_velocity
        
        recent_sales = self.sales_history[-days:] if len(self.sales_history) >= days else self.sales_history
        return sum(recent_sales) / len(recent_sales) if recent_sales else self.sales_velocity
    
    def get_price_trend(self, days: int = 7) -> str:
        """
        Determine price trend over specified period.
        
        Args:
            days: Number of recent days to analyze
            
        Returns:
            Price trend: "increasing", "decreasing", or "stable"
        """
        if len(self.price_history) < 2:
            return "stable"
        
        recent_prices = self.price_history[-days:] if len(self.price_history) >= days else self.price_history
        if len(recent_prices) < 2:
            return "stable"
        
        first_price = recent_prices[0]
        last_price = recent_prices[-1]
        
        change_threshold = 0.02  # 2% change threshold
        price_change = (last_price - first_price) / first_price
        
        if price_change > change_threshold:
            return "increasing"
        elif price_change < -change_threshold:
            return "decreasing"
        else:
            return "stable"
    
    def calculate_competitiveness_score(self, market_avg_price: Money) -> float:
        """
        Calculate how competitive this competitor is relative to market average.
        
        Args:
            market_avg_price: Average price in the market
            
        Returns:
            Competitiveness score (higher = more competitive)
        """
        if market_avg_price <= Money.zero():
            return 1.0
        
        # Price competitiveness (lower price = more competitive)
        price_ratio = float(self.price.to_decimal()) / float(market_avg_price.to_decimal())
        price_competitiveness = max(0.1, 2.0 - price_ratio)  # Inverted ratio
        
        # Performance factors
        bsr_competitiveness = max(0.1, 1.0 - (self.bsr / 1000000.0))  # Better BSR = higher score
        rating_competitiveness = self.average_rating / 5.0
        trust_competitiveness = self.trust_score
        
        # Weighted combination
        competitiveness = (
            price_competitiveness * 0.4 +
            bsr_competitiveness * 0.3 +
            rating_competitiveness * 0.2 +
            trust_competitiveness * 0.1
        )
        
        return min(2.0, max(0.1, competitiveness))