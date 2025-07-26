"""
Competitor management service for FBA-Bench simulation.

This module provides the CompetitorManager class that handles all competitor-related
business logic, including price updates, sales velocity changes, BSR calculations,
and strategy assignments.
"""

import random
from typing import List, Protocol, runtime_checkable
from decimal import Decimal
from datetime import datetime

from fba_bench.money import Money
from fba_bench.models.competitor import Competitor
from fba_bench.config_loader import FBABenchConfig


@runtime_checkable
class MarketDynamics(Protocol):
    """Protocol for market dynamics dependency."""
    
    def calculate_demand(self, base_demand: float, price: Money, elasticity: float,
                        seasonality_multiplier: float, competitors: List[Competitor],
                        trust_score: float) -> int:
        """Calculate demand based on market conditions."""
        ...


class CompetitorManager:
    """
    Manages competitor behavior and market dynamics in the FBA-Bench simulation.
    
    This service encapsulates all competitor-related business logic, providing
    a clean interface for updating competitor states based on market conditions
    and agent actions.
    
    The CompetitorManager uses dependency injection to receive configuration
    and market dynamics dependencies, making it easily testable and modular.
    """
    
    def __init__(self, config: FBABenchConfig, rng: random.Random):
        """
        Initialize the CompetitorManager.
        
        Args:
            config: Unified FBA-Bench configuration
            rng: Random number generator for reproducible behavior
        """
        self.config = config
        self.rng = rng
        
        # Extract competitor configuration for easy access
        self.competitor_config = config.competitor_model
        
        # Cache configuration values for performance (convert to Decimal for precise arithmetic)
        self._price_change_base = self.competitor_config.price_change_base
        self._sales_change_base = self.competitor_config.sales_change_base
        self._strategies = self.competitor_config.strategies
        self._aggressive_undercut_threshold = Decimal(str(self.competitor_config.aggressive_undercut_threshold))
        self._aggressive_undercut_amount = Decimal(str(self.competitor_config.aggressive_undercut_amount))
        self._follower_price_sensitivity = Decimal(str(self.competitor_config.follower_price_sensitivity))
        self._premium_price_maintenance = Decimal(str(self.competitor_config.premium_price_maintenance))
        self._value_competitive_threshold = Decimal(str(self.competitor_config.value_competitive_threshold))
        
        # BSR configuration
        self._bsr_base = config.market_dynamics.bsr_base
        self._bsr_min_value = config.market_dynamics.bsr_min_value
        self._bsr_max_value = config.market_dynamics.bsr_max_value
        self._bsr_smoothing_factor = config.market_dynamics.bsr_smoothing_factor
    
    def update_competitors(self, competitors: List[Competitor], agent_price: Money,
                          agent_sales: int, agent_bsr: int, market_demand_multiplier: float,
                          current_date: datetime) -> None:
        """
        Update all competitor prices, sales velocities, and BSRs.
        
        This is the main entry point for competitor updates. It orchestrates
        the update of all competitor attributes based on current market conditions
        and agent performance.
        
        Args:
            competitors: List of competitor objects to update
            agent_price: Current price of the agent's product
            agent_sales: Agent's current sales velocity (EMA)
            agent_bsr: Agent's current Best Seller Rank
            market_demand_multiplier: Market condition multiplier (e.g., holiday season)
            current_date: Current simulation date for seasonal calculations
        """
        if not competitors:
            return
        
        # Calculate market conditions
        is_holiday_season = current_date.month in [11, 12]
        effective_multiplier = market_demand_multiplier * (1.5 if is_holiday_season else 1.0)
        
        for comp in competitors:
            # Ensure competitor has a strategy assigned
            if not hasattr(comp, 'strategy') or not comp.strategy:
                comp.strategy = self._assign_competitor_strategy(comp)
            
            # Update competitor price based on strategy and market conditions
            price_change = self._calculate_competitor_price_change(
                comp, agent_price, agent_sales, agent_bsr, effective_multiplier
            )
            self._apply_price_change(comp, price_change)
            
            # Update competitor sales velocity
            sales_change = self._calculate_competitor_sales_change(
                comp, agent_price, agent_sales, effective_multiplier
            )
            self._apply_sales_velocity_change(comp, sales_change)
            
            # Update competitor BSR based on performance
            self._update_competitor_bsr(comp, effective_multiplier)
            
            # Update historical data
            comp.update_sales_history(comp.sales_velocity)
            comp.update_price_history(comp.price)
    
    def _assign_competitor_strategy(self, comp: Competitor) -> str:
        """
        Assign a pricing strategy to a competitor.
        
        Uses the competitor's ASIN hash for consistent strategy assignment
        across simulation runs with the same seed.
        
        Args:
            comp: Competitor to assign strategy to
            
        Returns:
            Strategy name from the configured strategy list
        """
        strategy_index = hash(comp.asin) % len(self._strategies)
        return self._strategies[strategy_index]
    
    def _calculate_competitor_price_change(self, comp: Competitor, agent_price: Money,
                                         agent_sales: int, agent_bsr: int,
                                         market_multiplier: float) -> float:
        """
        Calculate price change for a competitor based on strategy and market conditions.
        
        This method implements the core competitor pricing logic, with different
        strategies responding differently to agent actions and market conditions.
        
        Args:
            comp: Competitor to calculate price change for
            agent_price: Agent's current price
            agent_sales: Agent's sales velocity
            agent_bsr: Agent's Best Seller Rank
            market_multiplier: Market demand multiplier
            
        Returns:
            Price change as a multiplier (e.g., 0.05 = 5% increase)
        """
        base_change = Decimal(str(self.rng.uniform(-self._price_change_base, self._price_change_base)))
        
        strategy = comp.strategy
        # Ensure agent_price is a Money object for division
        agent_price_money = Money(agent_price, "USD") if not isinstance(agent_price, Money) else agent_price
        # Fix: Convert float price to Money for division
        comp_price_money = Money.from_dollars(comp.price) if isinstance(comp.price, (int, float)) else comp.price
        price_ratio = agent_price_money / comp_price_money

        # Strategy-specific pricing behavior
        if strategy == 'aggressive':
            # Aggressive competitors try to undercut
            if price_ratio < Decimal(str(self._aggressive_undercut_threshold)):  # Agent is significantly cheaper
                base_change -= Decimal(str(self._aggressive_undercut_amount))  # Lower price aggressively
            elif price_ratio > Decimal('1.05'):  # Agent is more expensive
                base_change += Decimal('0.01')  # Raise price slightly

        elif strategy == 'follower':
            # Followers mimic agent pricing with delay
            if price_ratio < Decimal('0.98'):
                base_change -= self._follower_price_sensitivity  # Follow price down slowly
            elif price_ratio > Decimal('1.02'):
                base_change += self._follower_price_sensitivity  # Follow price up slowly

        elif strategy == 'premium':
            # Premium competitors maintain higher prices
            if price_ratio > Decimal('0.9'):  # Agent is close to their price
                base_change += self._premium_price_maintenance  # Maintain premium
            else:
                base_change -= Decimal('0.005')  # Small adjustment down

        elif strategy == 'value':
            # Value competitors focus on low prices
            if price_ratio < self._value_competitive_threshold:  # Agent is competitive
                base_change -= Decimal('0.025')  # Stay competitive
            else:
                base_change += Decimal('0.005')  # Small increase if agent is expensive
        
        # Market condition adjustments
        if market_multiplier > Decimal('1.2'):  # High demand period
            base_change += Decimal('0.01')  # Increase prices in high demand
        
        # BSR-based adjustments (better BSR = more pricing power)
        if agent_bsr > 50000:  # Agent has poor BSR
            base_change += Decimal('0.005')  # Competitors can raise prices
        elif agent_bsr < 10000:  # Agent has good BSR
            base_change -= Decimal('0.005')  # Competitors need to be more competitive
        
        return base_change

    def _calculate_competitor_sales_change(self, comp: Competitor, agent_price: Money,
                                         agent_sales: int, market_multiplier: float) -> float:
        """
        Calculate sales velocity change for a competitor.
        
        Sales velocity changes based on price competitiveness, market share dynamics,
        and overall market conditions.
        
        Args:
            comp: Competitor to calculate sales change for
            agent_price: Agent's current price
            agent_sales: Agent's sales velocity
            market_multiplier: Market demand multiplier
            
        Returns:
            Sales change as a multiplier (e.g., 0.04 = 4% increase)
        """
        base_change = self.rng.uniform(-self._sales_change_base, self._sales_change_base)
        
        # Price competitiveness effect
        price_ratio = comp.price / agent_price
        if price_ratio < 0.9:  # Competitor is much cheaper
            base_change += 0.04  # Higher sales
        elif price_ratio > 1.1:  # Competitor is much more expensive
            base_change -= 0.04  # Lower sales
        
        # Market share competition - this needs to be calculated externally
        # since we don't have access to all competitors here
        # For now, use the competitor's current market share if available
        if hasattr(comp, 'market_share') and comp.market_share:
            if comp.market_share < 0.1:  # Low market share
                base_change += 0.02  # Try to gain share
            elif comp.market_share > 0.3:  # High market share
                base_change -= 0.01  # Natural decline
        
        # Market conditions
        base_change *= market_multiplier
        
        return base_change
    
    def _update_competitor_bsr(self, comp: Competitor, market_multiplier: float) -> None:
        """
        Update competitor BSR based on their performance.
        
        BSR is updated based on sales velocity, price competitiveness,
        and market conditions, similar to how the agent's BSR is calculated.
        
        Args:
            comp: Competitor to update BSR for
            market_multiplier: Market demand multiplier
        """
        # Simple BSR model for competitors
        # Better sales velocity and price competitiveness = better BSR
        
        # For now, use a simplified model since we don't have access to all competitors
        # In a full implementation, this would use the same logic as the agent's BSR calculation
        
        # Estimate price competitiveness (this is simplified)
        # In the full implementation, this would compare against average competitor price
        price_competitiveness = 1.0  # Default neutral competitiveness
        
        # BSR improves with sales velocity and price competitiveness
        bsr_factor = comp.sales_velocity * price_competitiveness * market_multiplier
        
        if bsr_factor > 1.0:
            # Improve BSR (lower number)
            comp.bsr = max(self._bsr_min_value, int(comp.bsr * 0.95))
        elif bsr_factor < 0.5:
            # Worsen BSR (higher number)
            comp.bsr = min(self._bsr_max_value, int(comp.bsr * 1.05))
        
        # Add some randomness
        if self.rng.random() < 0.1:  # 10% chance of random BSR change
            comp.bsr = max(self._bsr_min_value, min(self._bsr_max_value,
                          int(comp.bsr * self.rng.uniform(0.9, 1.1))))
    
    def _apply_price_change(self, comp: Competitor, price_change: float) -> None:
        """
        Apply price change to competitor with bounds checking.
        
        Args:
            comp: Competitor to update
            price_change: Price change multiplier
        """
        multiplier = Decimal(str(1 + price_change))
        # Fix: Convert float price to Money for multiplication
        if isinstance(comp.price, (int, float)):
            price_money = Money.from_dollars(comp.price)
            new_price_money = price_money * multiplier
            new_price = new_price_money.to_float()
        else:
            # comp.price is already Money
            new_price_money = comp.price * multiplier
            new_price = new_price_money.to_float()
        comp.price = Money.from_dollars(max(1.0, min(999.99, round(new_price, 2))))
    
    def _apply_sales_velocity_change(self, comp: Competitor, sales_change: float) -> None:
        """
        Apply sales velocity change to competitor with bounds checking.
        
        Args:
            comp: Competitor to update
            sales_change: Sales velocity change multiplier
        """
        new_sales_velocity = comp.sales_velocity * (1 + sales_change)
        comp.sales_velocity = max(0.1, min(100.0, new_sales_velocity))
    
    def initialize_competitors(self, num_competitors: int, agent_asin: str,
                             agent_category: str, agent_price: Money,
                             agent_cost: Money, agent_base_demand: float,
                             agent_bsr: int) -> List[Competitor]:
        """
        Initialize a list of competitor products for the simulation.
        
        Creates competitors with realistic variations in price and sales velocity
        relative to the agent's product.
        
        Args:
            num_competitors: Number of competitors to create
            agent_asin: ASIN of the agent's product
            agent_category: Product category
            agent_price: Agent's current price
            agent_cost: Agent's cost (assumed similar for competitors)
            agent_base_demand: Agent's base demand
            agent_bsr: Agent's BSR
            
        Returns:
            List of initialized Competitor objects
        """
        competitors = []
        
        for i in range(num_competitors):
            comp_asin = f"CMP{i+1:03d}{agent_asin[-3:]}"
            
            # Price: ±10% of agent's price
            multiplier = Decimal(str(self.rng.uniform(0.9, 1.1)))
            price = Money.from_dollars(round((agent_price * multiplier).to_float(), 2))
            
            # Sales velocity: ±20% of agent's base demand
            sales_velocity = max(0.1, agent_base_demand * self.rng.uniform(0.8, 1.2))
            
            # Create competitor with realistic attributes
            competitor = Competitor(
                asin=comp_asin,
                category=agent_category,
                cost=agent_cost,  # Assume similar cost
                price=price,
                base_demand=agent_base_demand,
                bsr=agent_bsr + self.rng.randint(-10000, 10000),  # Vary BSR slightly
                sales_velocity=sales_velocity,
                strategy=self._assign_competitor_strategy(Competitor(comp_asin, agent_category, agent_cost, price)),
                trust_score=self.rng.uniform(0.7, 1.0),  # Random trust score
                brand_strength=self.rng.uniform(0.3, 0.8),  # Random brand strength
                inventory_level=self.rng.randint(500, 2000),  # Random inventory
                review_count=self.rng.randint(50, 500),  # Random review count
                average_rating=self.rng.uniform(3.5, 4.8),  # Random rating
            )
            
            competitors.append(competitor)
        
        return competitors
    
    def calculate_market_share(self, competitors: List[Competitor], agent_sales: float) -> None:
        """
        Calculate and update market share for all competitors.
        
        This method calculates the market share for each competitor based on
        their sales velocity relative to total market sales.
        
        Args:
            competitors: List of competitors to update
            agent_sales: Agent's sales velocity
        """
        total_sales = agent_sales + sum(c.sales_velocity for c in competitors)
        
        if total_sales > 0:
            for comp in competitors:
                comp.market_share = comp.sales_velocity / total_sales
        else:
            # If no sales, distribute equally
            equal_share = 1.0 / (len(competitors) + 1)  # +1 for agent
            for comp in competitors:
                comp.market_share = equal_share