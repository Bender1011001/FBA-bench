from typing import Dict, List

import numpy as np

from .config_loader import load_config

# Load configuration
_config = load_config()
REL_PRICE_FACTOR_MIN = _config.market_dynamics.rel_price_factor_min
REL_PRICE_FACTOR_MAX = _config.market_dynamics.rel_price_factor_max

class Competitor:
    """
    Enhanced competitor model with realistic business characteristics and behaviors.

    Attributes:
        asin (str): Amazon Standard Identification Number for the competitor.
        price (float): Current price of the competitor's product.
        sales_velocity (float): Current sales velocity of the competitor.
        trust_score (float): Trust score of the competitor (default: 1.0).
        strategy (str): Pricing strategy ("aggressive", "follower", "premium", "value").
        brand_strength (float): Brand recognition and loyalty (0.0-1.0).
        inventory_level (int): Current inventory level.
        cost_structure (float): Cost multiplier affecting pricing flexibility.
        market_share (float): Estimated market share (0.0-1.0).
        review_count (int): Total number of reviews.
        average_rating (float): Average customer rating (1.0-5.0).
        fulfillment_method (str): "FBA", "FBM", or "Hybrid".
        geographic_focus (str): "US", "Global", "Regional".
        seasonal_factor (float): Seasonal demand multiplier.
        price_elasticity (float): Price sensitivity factor.
        launch_date (int): Days since product launch.
        last_price_change (int): Days since last price change.
        promotion_active (bool): Whether currently running promotions.
    """
    def __init__(self, asin: str, price: float, sales_velocity: float, trust_score: float = 1.0,
                 strategy: str = "follower", brand_strength: float = 0.5):
        """
        Initialize an enhanced Competitor instance.

        Args:
            asin (str): Amazon Standard Identification Number.
            price (float): Product price.
            sales_velocity (float): Sales velocity.
            trust_score (float, optional): Trust score (default: 1.0).
            strategy (str, optional): Pricing strategy (default: "follower").
            brand_strength (float, optional): Brand strength (default: 0.5).
        """
        import random
        
        # Basic attributes
        self.asin = asin
        self.price = price
        self.sales_velocity = sales_velocity
        self.trust_score = trust_score
        self.strategy = strategy
        self.brand_strength = brand_strength
        
        # Enhanced business characteristics
        self.inventory_level = random.randint(50, 500)
        self.cost_structure = random.uniform(0.4, 0.8)  # Cost as % of price
        self.market_share = random.uniform(0.05, 0.25)  # 5-25% market share
        self.review_count = random.randint(10, 1000)
        self.average_rating = random.uniform(3.5, 4.8)
        
        # Operational characteristics
        fulfillment_options = ["FBA", "FBM", "Hybrid"]
        self.fulfillment_method = random.choice(fulfillment_options)
        
        geographic_options = ["US", "Global", "Regional"]
        self.geographic_focus = random.choice(geographic_options)
        
        # Market dynamics
        self.seasonal_factor = 1.0
        self.price_elasticity = random.uniform(0.5, 2.0)  # Price sensitivity
        self.launch_date = random.randint(30, 365)  # Days since launch
        self.last_price_change = random.randint(0, 30)
        self.promotion_active = random.random() < 0.15  # 15% chance of promotion
        
        # Performance tracking
        self.sales_history = [sales_velocity] * 7  # Last 7 days
        self.price_history = [price] * 7  # Last 7 days
        self.bsr = random.randint(10000, 100000)
        
    def update_market_position(self, market_conditions: dict):
        """Update competitor's market position based on conditions."""
        # Update seasonal factor
        season = market_conditions.get("season", "normal")
        if season == "holiday":
            self.seasonal_factor = 1.3
        elif season == "summer":
            self.seasonal_factor = 0.9
        else:
            self.seasonal_factor = 1.0
            
        # Update sales history
        self.sales_history.append(self.sales_velocity)
        if len(self.sales_history) > 7:
            self.sales_history.pop(0)
            
        # Update price history
        self.price_history.append(self.price)
        if len(self.price_history) > 7:
            self.price_history.pop(0)
    
    def calculate_competitive_response(self, agent_price: float, agent_sales: float) -> dict:
        """Calculate how this competitor responds to agent's actions."""
        import random
        
        response = {
            "price_change": 0.0,
            "promotion_change": False,
            "inventory_action": None,
            "reason": ""
        }
        
        price_ratio = agent_price / self.price if self.price > 0 else 1.0
        sales_ratio = agent_sales / max(1, self.sales_velocity)
        
        # Strategy-based responses
        if self.strategy == "aggressive":
            if price_ratio < 0.95:  # Agent undercuts by 5%+
                response["price_change"] = -0.03  # Cut price by 3%
                response["reason"] = "Aggressive price matching"
            elif sales_ratio > 1.5:  # Agent outselling significantly
                response["promotion_change"] = True
                response["reason"] = "Promotional response to competition"
                
        elif self.strategy == "premium":
            if sales_ratio > 2.0:  # Only respond to major threats
                response["price_change"] = -0.01  # Small price reduction
                response["reason"] = "Defensive pricing adjustment"
            # Premium brands rarely engage in price wars
            
        elif self.strategy == "follower":
            if abs(price_ratio - 1.0) > 0.1:  # Price difference > 10%
                target_price = agent_price * random.uniform(0.98, 1.02)
                response["price_change"] = (target_price - self.price) / self.price
                response["reason"] = "Following market pricing"
                
        elif self.strategy == "value":
            if price_ratio > 1.1:  # Agent prices higher
                response["price_change"] = 0.02  # Increase price slightly
                response["reason"] = "Value positioning adjustment"
        
        # Inventory-based responses
        if self.inventory_level < 20:
            response["inventory_action"] = "restock_urgent"
        elif self.inventory_level > 300:
            response["inventory_action"] = "clearance_pricing"
            response["price_change"] = min(response["price_change"], -0.05)
        
        return response
    
    def get_competitive_strength(self) -> float:
        """Calculate overall competitive strength (0.0-1.0)."""
        factors = [
            self.trust_score * 0.25,
            self.brand_strength * 0.20,
            min(1.0, self.average_rating / 5.0) * 0.15,
            min(1.0, self.review_count / 500) * 0.10,
            self.market_share * 4.0 * 0.15,  # Scale market share
            (1.0 if self.fulfillment_method == "FBA" else 0.7) * 0.15
        ]
        return min(1.0, sum(factors))

def calculate_trust_score(**kwargs) -> float:
    """
    Calculates a seller trust score based on negative events.
    Accepts keyword arguments:
        - cancellations (int)
        - policy_violations (int)
        - review_manipulation (int)
        - customer_issues (int)
        - base_score (float, default 1.0)
    Returns a trust score between 0.0 and 1.0.
    """
    cancellations = kwargs.get("cancellations", 0)
    policy_violations = kwargs.get("policy_violations", 0)
    review_manipulation = kwargs.get("review_manipulation", 0)
    customer_issues = kwargs.get("customer_issues", 0)
    base_score = kwargs.get("base_score", 1.0)
    # Penalty weights (tunable)
    penalty = (
        0.1 * cancellations +
        0.15 * policy_violations +
        0.1 * review_manipulation +
        0.05 * customer_issues
    )
    trust_score = max(0.0, min(1.0, base_score - penalty))
    return trust_score

def calculate_dynamic_elasticity(
    bsr: int,
    e_min: float = 1.1,
    e_max: float = 4.5,
    bsr_mid_point: int = 10000,
    bsr_scale: float = 1.5
) -> float:
    """
    Calculates the price elasticity coefficient based on Best Seller Rank (BSR).

    The function uses a sigmoid curve to model the relationship where demand becomes
    more elastic (a higher coefficient) as BSR improves (a lower BSR number).

    Args:
        bsr: The product's current Best Seller Rank. Must be > 0.
        e_min: The minimum elasticity for products with a very poor BSR.
        e_max: The maximum elasticity for top-selling products.
        bsr_mid_point: The BSR value at the center of the transition.
        bsr_scale: Controls how steep the transition is from low to high elasticity.

    Returns:
        The calculated price elasticity coefficient.
    """
    if bsr <= 0:
        # Handle the edge case of a product with no rank or an invalid rank
        return e_min

    # We use the log of BSR to handle its wide range and create a more stable curve
    log_bsr = np.log(bsr)
    log_bsr_mid = np.log(bsr_mid_point)

    # Standard sigmoid function: 1 / (1 + e^-x)
    # We use (1 - sigmoid) because a lower BSR (better rank) should yield higher elasticity
    sigmoid_value = 1 / (1 + np.exp(-(log_bsr - log_bsr_mid) / bsr_scale))

    # Calculate the final elasticity, interpolating between E_min and E_max
    elasticity = e_min + (e_max - e_min) * (1 - sigmoid_value)

    return elasticity

def get_seasonality_multiplier(date, category):
    """
    Returns a demand multiplier for the given date and category based on seasonality and event calendar.
    Example: Q4 boost, Prime Day, category-specific events.
    """
    # Q4 (Oct-Dec): 1.3x demand
    if date.month in [10, 11, 12]:
        return 1.3
    # Prime Day (July 15): 2.0x demand
    if date.month == 7 and date.day == 15:
        return 2.0
    # Apparel: Spring boost
    if category == "Apparel" and date.month in [3, 4, 5]:
        return 1.2
    # Default: no boost
    return 1.0

def calculate_demand(
    base_demand: float,
    price: float,
    elasticity: float,
    seasonality_multiplier: float = 1.0,
    competitors: List[Competitor] = None,
    trust_score: float = 1.0
) -> int:
    """
    Calculates the final demand based on price, elasticity, seasonality/event multiplier,
    relative price vs. competitors, and seller trust score.
    """
    # Convert price to float for calculations if it's a Money type
    price_value = float(price.to_decimal()) if hasattr(price, 'to_decimal') else price
    
    if price_value <= 0:
        # Assume very high demand for a free product to avoid division by zero
        return int(round(base_demand * 2 * seasonality_multiplier))

    # BUGFIX: Add price floor validation to prevent elasticity explosion at very low prices
    MIN_PRICE = 0.50  # $0.50 minimum price floor to prevent mathematical explosion
    safe_price = max(price_value, MIN_PRICE)
    
    # Log warning if price floor was applied
    if price_value < MIN_PRICE:
        import warnings
        warnings.warn(f"Price ${price_value:.2f} below minimum floor ${MIN_PRICE:.2f}. "
                     f"Using floor price to prevent elasticity explosion.", UserWarning)

    # Relative price effect: compare to average competitor price
    rel_price_factor = 1.0
    if competitors and len(competitors) > 0:
        competitor_prices = [float(c.price.to_decimal()) if hasattr(c.price, 'to_decimal') else c.price for c in competitors]
        avg_competitor_price = np.mean(competitor_prices)
        if avg_competitor_price > 0:
            rel_price_factor = avg_competitor_price / safe_price  # Use safe_price for calculation
            # Clamp to [REL_PRICE_FACTOR_MIN, REL_PRICE_FACTOR_MAX] to avoid extreme effects
            rel_price_factor = max(REL_PRICE_FACTOR_MIN, min(REL_PRICE_FACTOR_MAX, rel_price_factor))

    # Trust score effect: linearly scale demand (0.0 = suppressed, 1.0 = full)
    trust_factor = max(0.0, min(1.0, trust_score))

    # BUGFIX: Use safe_price in elasticity calculation to prevent explosion
    # Demand formula: Demand = Base * (Price ^ -Elasticity) * Seasonality * RelativePrice * Trust
    demand = base_demand * (safe_price ** -elasticity) * seasonality_multiplier * rel_price_factor * trust_factor

    # Return as an integer, as we can't sell fractional units
    return int(round(demand))