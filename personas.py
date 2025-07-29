"""
Competitor Persona Library - High-Fidelity Market Chaos

This module implements the "Embrace Irrationality" architectural mandate by providing
a library of distinct competitor personas with irrational, human-like behaviors that
deviate from simple optimization.

Each persona encapsulates specific market behavior patterns that introduce realistic
chaos and unpredictability into the simulation environment.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from decimal import Decimal

from money import Money
from events import CompetitorState


@dataclass
class MarketConditions:
    """
    Market state information provided to personas for decision-making.
    
    This encapsulates all the market intelligence a competitor would have
    access to when making pricing and strategy decisions.
    """
    current_tick: int
    current_state: CompetitorState
    market_competitors: list[CompetitorState]
    market_average_price: Money
    market_min_price: Money
    market_max_price: Money
    own_sales_velocity: float  # Recent sales trend
    market_trend: str  # "rising", "falling", "stable"


class CompetitorPersona(ABC):
    """
    Base class for all competitor personas.
    
    Each persona represents a distinct competitive behavior pattern with
    its own decision-making logic, risk tolerance, and market response style.
    
    Personas maintain their own internal state and can exhibit time-dependent
    behaviors, memory of past actions, and complex strategic patterns.
    """
    
    def __init__(self, competitor_id: str, cost_basis: Money):
        """
        Initialize the persona with basic competitor information.
        
        Args:
            competitor_id: Unique identifier for this competitor
            cost_basis: Minimum viable price (cost to produce/acquire product)
        """
        self.competitor_id = competitor_id
        self.cost_basis = cost_basis
        self.internal_state: Dict[str, Any] = {}
        self.last_action_tick: int = 0
    
    @abstractmethod
    async def act(self, market_conditions: MarketConditions) -> Optional[CompetitorState]:
        """
        Evaluate market conditions and decide on competitor actions.
        
        This is the core decision-making method that each persona must implement.
        The persona receives complete market intelligence and returns an updated
        CompetitorState if it decides to take action, or None if no action.
        
        Args:
            market_conditions: Current market state and competitor intelligence
            
        Returns:
            Updated CompetitorState if action taken, None if no changes
        """
        pass
    
    def _calculate_minimum_price(self) -> Money:
        """Calculate the absolute minimum price (cost basis + small margin)."""
        print(f"DEBUG: self.cost_basis={self.cost_basis} type={type(self.cost_basis)}")
        print(f"DEBUG: self.cost_basis._cents={getattr(self.cost_basis, '_cents', 'N/A')} type={type(getattr(self.cost_basis, '_cents', 'N/A'))}")
        if not hasattr(self.cost_basis, "_cents") or not isinstance(self.cost_basis._cents, int):
            print(f"WARNING: Invalid cost_basis for competitor {getattr(self, 'competitor_id', 'unknown')}: {self.cost_basis}")
            cents = 0
        else:
            cents = self.cost_basis._cents
        return (Decimal(cents) / Decimal('100')) * Decimal('1.01')  # 1% minimum margin
    
    def _get_state_value(self, key: str, default: Any = None) -> Any:
        """Safely retrieve internal state value."""
        return self.internal_state.get(key, default)
    
    def _set_state_value(self, key: str, value: Any) -> None:
        """Update internal state value."""
        self.internal_state[key] = value


class IrrationalSlasher(CompetitorPersona):
    """
    The Irrational Price Slasher Persona
    
    This persona represents competitors who occasionally engage in destructive
    price wars, ignoring market logic and slashing prices to unsustainable levels.
    
    Behavior Patterns:
    - 15% chance per tick to enter "slash mode" 
    - During slash mode: prices set to just above cost basis
    - Slash episodes last 3-7 ticks
    - Higher chance to slash if losing market share
    - May trigger cascading price wars
    """
    
    def __init__(self, competitor_id: str, cost_basis: Money):
        super().__init__(competitor_id, cost_basis)
        self.slash_probability = 0.15  # 15% base chance per tick
        self.slash_duration_range = (3, 7)  # 3-7 ticks
    
    async def act(self, market_conditions: MarketConditions) -> Optional[CompetitorState]:
        """
        Implement irrational slashing behavior.
        
        The slasher evaluates whether to enter slash mode, continue slashing,
        or return to rational pricing based on market conditions and internal state.
        """
        current_tick = market_conditions.current_tick
        is_slashing = self._get_state_value('is_slashing', False)
        slash_end_tick = self._get_state_value('slash_end_tick', 0)
        
        # Check if currently in slash mode
        if is_slashing:
            if current_tick >= slash_end_tick:
                # End slash mode - return to rational pricing
                self._set_state_value('is_slashing', False)
                self._set_state_value('slash_end_tick', 0)
                return await self._rational_pricing(market_conditions)
            else:
                # Continue slashing - maintain rock-bottom pricing
                return await self._slash_pricing(market_conditions)
        
        # Not currently slashing - decide whether to start
        should_slash = await self._should_start_slashing(market_conditions)
        
        if should_slash:
            # Enter slash mode
            duration = random.randint(*self.slash_duration_range)
            self._set_state_value('is_slashing', True)
            self._set_state_value('slash_end_tick', current_tick + duration)
            return await self._slash_pricing(market_conditions)
        else:
            # Continue rational behavior
            return await self._rational_pricing(market_conditions)
    
    async def _should_start_slashing(self, market_conditions: MarketConditions) -> bool:
        """
        Determine if the competitor should enter destructive slash mode.
        
        Factors increasing slash probability:
        - Low sales velocity (losing market share)
        - Being significantly above market average price
        - Random irrational impulses
        """
        base_probability = self.slash_probability
        
        # Increase probability if sales are poor
        if market_conditions.own_sales_velocity < 0.5:  # Below average performance
            base_probability *= 2.0
        
        # Increase probability if significantly above market average
        current_price = market_conditions.current_state.price
        market_avg = market_conditions.market_average_price
        if Decimal(str(current_price)) > Decimal(str(market_avg)) * Decimal('1.2'):  # 20% above average
            base_probability *= 1.5
        
        # Cap maximum probability
        final_probability = min(base_probability, 0.4)  # Max 40% chance
        
        return random.random() < final_probability
    
    async def _slash_pricing(self, market_conditions: MarketConditions) -> CompetitorState:
        """
        Set destructively low pricing during slash mode.
        
        Price is set to just above cost basis, ignoring market conditions.
        """
        slash_price = self._calculate_minimum_price()
        
        # Create updated state with slashed price
        current_state = market_conditions.current_state
        
        return CompetitorState(
            asin=current_state.asin,
            price=slash_price,
            bsr=current_state.bsr,  # BSR will improve due to low price
            sales_velocity=float(Decimal(str(float(current_state.sales_velocity))) * Decimal('1.5'))  # Boost sales
        )
    
    async def _rational_pricing(self, market_conditions: MarketConditions) -> CompetitorState:
        """
        Implement rational competitive pricing when not in slash mode.
        
        Follows market-responsive pricing similar to default behavior.
        """
        current_state = market_conditions.current_state
        market_avg = market_conditions.market_average_price
        
        # Price slightly below market average for competitiveness
        rational_price = Decimal(str(market_avg)) * Decimal('0.95')  # 5% below average
        
        # Ensure we don't price below cost basis
        final_price = max(rational_price, self._calculate_minimum_price())
        
        return CompetitorState(
            asin=current_state.asin,
            price=final_price,
            bsr=current_state.bsr,
            sales_velocity=float(current_state.sales_velocity)
        )


class SlowFollower(CompetitorPersona):
    """
    The Slow Market Follower Persona
    
    This persona represents competitors with delayed market response due to:
    - Organizational bureaucracy and slow decision-making
    - Infrequent market monitoring
    - Conservative risk management
    - Limited market intelligence resources
    
    Behavior Patterns:
    - Only evaluates market every 4-8 ticks (realistic lag)
    - When active, follows market trends conservatively
    - Gradual price adjustments, never dramatic changes
    - Tends to lag behind market movements
    """
    
    def __init__(self, competitor_id: str, cost_basis: Money):
        super().__init__(competitor_id, cost_basis)
        self.evaluation_interval_range = (4, 8)  # Ticks between evaluations
        self.max_price_change_percent = Decimal('0.10')  # Maximum 10% price change
    
    async def act(self, market_conditions: MarketConditions) -> Optional[CompetitorState]:
        """
        Implement slow, lagged market following behavior.
        
        The slow follower only acts periodically and makes conservative
        adjustments when it does evaluate the market.
        """
        current_tick = market_conditions.current_tick
        last_evaluation_tick = self._get_state_value('last_evaluation_tick', 0)
        next_evaluation_tick = self._get_state_value('next_evaluation_tick', 0)
        
        # Initialize evaluation schedule on first run
        if next_evaluation_tick == 0:
            interval = random.randint(*self.evaluation_interval_range)
            self._set_state_value('next_evaluation_tick', current_tick + interval)
            return None  # No action on first tick
        
        # Check if it's time for evaluation
        if current_tick < next_evaluation_tick:
            return None  # Not time to evaluate yet
        
        # Time to evaluate and potentially act
        self._set_state_value('last_evaluation_tick', current_tick)
        
        # Schedule next evaluation
        interval = random.randint(*self.evaluation_interval_range)
        self._set_state_value('next_evaluation_tick', current_tick + interval)
        
        # Perform conservative market following
        return await self._conservative_market_following(market_conditions)
    
    async def _conservative_market_following(self, market_conditions: MarketConditions) -> CompetitorState:
        """
        Implement conservative price adjustments following market trends.
        
        The slow follower makes gradual adjustments toward market conditions
        but never dramatic price changes.
        """
        current_state = market_conditions.current_state
        current_price = current_state.price
        market_avg = market_conditions.market_average_price
        
        # Calculate target price (market average with slight conservative bias)
        target_price = Decimal(str(market_avg)) * Decimal('1.02')  # 2% above average (conservative)
        
        # Calculate maximum allowed price change
        max_increase = current_price * (Decimal('1') + self.max_price_change_percent)
        max_decrease = current_price * (Decimal('1') - self.max_price_change_percent)
        
        # Apply conservative adjustment limits
        if target_price > current_price:
            # Trending up - gradual increase
            new_price = min(target_price, max_increase)
        else:
            # Trending down - gradual decrease
            new_price = max(target_price, max_decrease)
        
        # Never price below cost basis
        final_price = max(new_price, self._calculate_minimum_price())
        
        # Conservative sales velocity adjustment (slow to respond)
        velocity_adjustment = Decimal('1.0')
        if final_price < current_price:
            velocity_adjustment = Decimal('1.05')  # Modest boost for price reduction
        elif final_price > current_price:
            velocity_adjustment = Decimal('0.95')  # Modest reduction for price increase
        
        return CompetitorState(
            asin=current_state.asin,
            price=final_price,
            bsr=current_state.bsr,
            sales_velocity=float(current_state.sales_velocity * velocity_adjustment)
        )