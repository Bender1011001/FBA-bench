"""Event-driven CompetitorManager for FBA-Bench v3 architecture with High-Fidelity Chaos."""

import asyncio
import logging
import random
import time
import uuid
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from decimal import Decimal

from money import Money
from models.competitor import Competitor
from events import TickEvent, CompetitorPricesUpdated, CompetitorState
from event_bus import EventBus, get_event_bus
from personas import CompetitorPersona, MarketConditions, IrrationalSlasher, SlowFollower


logger = logging.getLogger(__name__)


class CompetitorStrategy(Enum):
    """Competitor pricing strategies."""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"
    RANDOM = "random"


@dataclass
class CompetitorAction:
    """Represents a competitor action with financial impact."""
    competitor_id: str
    action_type: str
    old_price: Money
    new_price: Money
    confidence: float
    timestamp: float


class CompetitorManager:
    """
    Event-driven competitor management service for FBA-Bench v3 with High-Fidelity Chaos.
    
    Subscribes to TickEvent and publishes CompetitorPricesUpdated events.
    Now delegates all competitor decision-making to pluggable persona objects,
    introducing realistic market irrationality and chaos.
    """
    
    def __init__(self, config: Dict):
        """Initialize event-driven competitor manager with persona support."""
        self.config = config
        self.competitors: Dict[str, Competitor] = {}
        self.competitor_personas: Dict[str, CompetitorPersona] = {}
        self.competitor_states: Dict[str, CompetitorState] = {}
        self.action_history: List[CompetitorAction] = []
        self.event_bus: Optional[EventBus] = None
        self._subscription_task: Optional[asyncio.Task] = None
        
        # Persona configuration for random assignment
        self.persona_distribution = config.get('persona_distribution', {
            'IrrationalSlasher': 0.3,  # 30% chance
            'SlowFollower': 0.4,       # 40% chance
            'Default': 0.3             # 30% use old rational behavior
        })
        
        # Market intelligence configuration
        self.market_sensitivity = config.get('market_sensitivity', 0.7)
        self.sales_velocity_window = config.get('sales_velocity_window', 5)  # Track last 5 ticks
        
        # Market state tracking for persona intelligence
        self.market_history: List[Dict] = []
        self.tick_counter: int = 0
        
    async def start(self) -> None:
        """Start the event-driven competitor manager."""
        if self.event_bus is None:
            self.event_bus = get_event_bus()
        
        # Subscribe to TickEvent
        self._subscription_task = asyncio.create_task(
            self.event_bus.subscribe("TickEvent", self._handle_tick_event)
        )
        
        logger.info("CompetitorManager started - subscribed to TickEvent")
        
    async def stop(self) -> None:
        """Stop the event-driven competitor manager."""
        if self._subscription_task:
            self._subscription_task.cancel()
            try:
                await self._subscription_task
            except asyncio.CancelledError:
                pass
            self._subscription_task = None
            
        logger.info("CompetitorManager stopped")
        
    async def _handle_tick_event(self, event: TickEvent) -> None:
        """Handle TickEvent by delegating to personas and publishing results."""
        try:
            self.tick_counter = event.tick_number
            logger.debug(f"CompetitorManager processing tick {event.tick_number} with {len(self.competitors)} competitors")
            
            # Build market intelligence for personas
            market_conditions = self._build_market_conditions(event)
            
            # Update market history for trend analysis
            self._update_market_history(market_conditions, event)
            
            # Process each competitor through their assigned persona
            await self._process_competitors_with_personas(market_conditions)
            
            # Create competitor state snapshots from updated states
            competitor_states = self._create_competitor_snapshots()
            
            # Publish CompetitorPricesUpdated event
            await self._publish_competitor_update(event.tick_number, competitor_states)
            
        except Exception as e:
            logger.error(f"Error in CompetitorManager tick processing: {e}")
            raise
    
    async def _process_competitors_with_personas(self, base_market_conditions: Dict) -> None:
        """Process each competitor using their assigned persona for decision-making."""
        for competitor_id, competitor in self.competitors.items():
            if not hasattr(competitor, 'is_active') or not competitor.is_active:
                continue
                
            # Get the persona for this competitor
            persona = self.competitor_personas.get(competitor_id)
            if not persona:
                logger.warning(f"No persona assigned to competitor {competitor_id}, skipping")
                continue
            
            # Get current competitor state
            current_state = self.competitor_states.get(competitor_id)
            if not current_state:
                # Initialize state from competitor object
                current_state = self._initialize_competitor_state(competitor)
                self.competitor_states[competitor_id] = current_state
            
            # Build persona-specific market conditions
            market_conditions = self._build_persona_market_conditions(
                base_market_conditions, current_state
            )
            
            try:
                # Let the persona decide on action
                updated_state = await persona.act(market_conditions)
                
                if updated_state:
                    # Record the action for history
                    old_price = current_state.price
                    action = CompetitorAction(
                        competitor_id=competitor_id,
                        action_type=f"persona_{type(persona).__name__}",
                        old_price=old_price,
                        new_price=updated_state.price,
                        confidence=0.8,  # Personas are confident in their choices
                        timestamp=time.time()
                    )
                    self.action_history.append(action)
                    
                    # Update stored state
                    self.competitor_states[competitor_id] = updated_state
                    
                    # Update the competitor object for compatibility
                    competitor.price = updated_state.price
                    
                    logger.debug(f"Persona {type(persona).__name__} updated competitor {competitor_id}: {old_price} -> {updated_state.price}")
                    
            except Exception as e:
                logger.error(f"Error processing competitor {competitor_id} with persona {type(persona).__name__}: {e}")
                continue
    
    def _build_market_conditions(self, tick_event: TickEvent) -> Dict:
        """Build comprehensive market conditions from tick event and history."""
        # Extract market data from event metadata
        metadata = tick_event.metadata.get('market_conditions', {})
        our_price = metadata.get('our_price', Money.from_dollars(20.00))
        sales_velocity = metadata.get('sales_velocity', 1.0)
        
        # Calculate market trend from history
        market_trend = self._calculate_market_trend()
        
        # Get current competitor prices for market analysis
        competitor_prices = []
        for state in self.competitor_states.values():
            competitor_prices.append(state.price)
        
        # Calculate market statistics
        if competitor_prices:
            market_avg = Money(sum(p.cents for p in competitor_prices) // len(competitor_prices))
            market_min = min(competitor_prices)
            market_max = max(competitor_prices)
        else:
            market_avg = our_price
            market_min = our_price
            market_max = our_price
        
        return {
            'current_tick': tick_event.tick_number,
            'our_price': our_price,
            'sales_velocity': sales_velocity,
            'market_trend': market_trend,
            'market_average_price': market_avg,
            'market_min_price': market_min,
            'market_max_price': market_max,
            'competitor_count': len(self.competitor_states)
        }
    
    def _update_market_history(self, market_conditions: Dict, tick_event: TickEvent) -> None:
        """Update market history for trend analysis."""
        history_entry = {
            'tick': tick_event.tick_number,
            'average_price': market_conditions['market_average_price'],
            'sales_velocity': market_conditions['sales_velocity'],
            'competitor_count': market_conditions['competitor_count'],
            'timestamp': time.time()
        }
        
        self.market_history.append(history_entry)
        
        # Keep only recent history (last 20 ticks)
        if len(self.market_history) > 20:
            self.market_history = self.market_history[-20:]
    
    def _calculate_market_trend(self) -> str:
        """Calculate market trend from recent history."""
        if len(self.market_history) < 3:
            return 'stable'
        
        # Look at price changes over last few ticks
        recent_prices = [entry['average_price'].cents for entry in self.market_history[-5:]]
        
        if len(recent_prices) < 2:
            return 'stable'
        
        # Calculate trend
        price_changes = []
        for i in range(1, len(recent_prices)):
            change = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
            price_changes.append(change)
        
        avg_change = sum(price_changes) / len(price_changes)
        
        if avg_change > 0.02:  # 2% average increase
            return 'rising'
        elif avg_change < -0.02:  # 2% average decrease
            return 'falling'
        else:
            return 'stable'
    
    def _build_persona_market_conditions(self, base_conditions: Dict, current_state: CompetitorState) -> MarketConditions:
        """Build MarketConditions object for persona decision-making."""
        # Get all competitor states as list
        all_competitor_states = list(self.competitor_states.values())
        
        return MarketConditions(
            current_tick=base_conditions['current_tick'],
            current_state=current_state,
            market_competitors=all_competitor_states,
            market_average_price=base_conditions['market_average_price'],
            market_min_price=base_conditions['market_min_price'],
            market_max_price=base_conditions['market_max_price'],
            own_sales_velocity=current_state.sales_velocity,
            market_trend=base_conditions['market_trend']
        )
    
    def _initialize_competitor_state(self, competitor: Competitor) -> CompetitorState:
        """Initialize CompetitorState from Competitor object."""
        competitor_id = getattr(competitor, 'competitor_id', competitor.asin)
        
        return CompetitorState(
            asin=getattr(competitor, 'asin', competitor_id),
            price=competitor.price,
            bsr=getattr(competitor, 'bsr', 100000),
            sales_velocity=float(getattr(competitor, 'sales_velocity', 1.0))
        )
        
    def _create_competitor_snapshots(self) -> List[CompetitorState]:
        """Create CompetitorState snapshots from stored states."""
        snapshots = []
        
        for competitor_id, state in self.competitor_states.items():
            competitor = self.competitors.get(competitor_id)
            if competitor and (not hasattr(competitor, 'is_active') or competitor.is_active):
                snapshots.append(state)
            
        return snapshots
    
    async def _publish_competitor_update(self, tick_number: int, competitor_states: List[CompetitorState]) -> None:
        """Publish CompetitorPricesUpdated event."""
        if self.event_bus is None:
            logger.warning("Cannot publish competitor update - event bus not initialized")
            return
            
        # Create market summary for quick analysis
        market_summary = {}
        if competitor_states:
            prices = [comp.price for comp in competitor_states]
            market_summary = {
                'competitor_count': len(competitor_states),
                'average_price': str(Money(sum(p.cents for p in prices) // len(prices))),
                'min_price': str(min(prices)),
                'max_price': str(max(prices)),
                'average_bsr': sum(comp.bsr for comp in competitor_states) / len(competitor_states),
                'average_sales_velocity': sum(comp.sales_velocity for comp in competitor_states) / len(competitor_states)
            }
        
        event = CompetitorPricesUpdated(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            tick_number=tick_number,
            competitors=competitor_states,
            market_summary=market_summary
        )
        
        await self.event_bus.publish(event)
        logger.debug(f"Published CompetitorPricesUpdated with {len(competitor_states)} competitors")
        
    def add_competitor(self, competitor: Competitor, persona: Optional[CompetitorPersona] = None) -> None:
        """Add a competitor to the manager with optional persona."""
        competitor_id = getattr(competitor, 'competitor_id', competitor.asin)
        self.competitors[competitor_id] = competitor
        
        # Assign persona
        if persona:
            self.competitor_personas[competitor_id] = persona
        else:
            # Auto-assign random persona based on distribution
            self.competitor_personas[competitor_id] = self._create_random_persona(competitor_id, competitor)
        
        # Initialize competitor state
        self.competitor_states[competitor_id] = self._initialize_competitor_state(competitor)
        
        logger.info(f"Added competitor {competitor_id} with persona {type(self.competitor_personas[competitor_id]).__name__}")
        
    def add_competitor_with_persona(self, competitor: Competitor, persona_type: str, **persona_kwargs) -> None:
        """Add a competitor with a specific persona type."""
        competitor_id = getattr(competitor, 'competitor_id', competitor.asin)
        
        # Create the specified persona
        persona = self._create_persona_by_type(competitor_id, competitor, persona_type, **persona_kwargs)
        
    
    def _create_random_persona(self, competitor_id: str, competitor: Competitor) -> CompetitorPersona:
        """Create a random persona based on configured distribution."""
        # Get cost basis from competitor (with fallback)
        cost_basis = getattr(competitor, 'cost_basis', competitor.price * Decimal('0.7'))  # 30% margin default
        
        # Select persona type based on distribution
        persona_types = list(self.persona_distribution.keys())
        weights = list(self.persona_distribution.values())
        selected_type = random.choices(persona_types, weights=weights)[0]
        
        return self._create_persona_by_type(competitor_id, competitor, selected_type)
    
    def _create_persona_by_type(self, competitor_id: str, competitor: Competitor, persona_type: str, **kwargs) -> CompetitorPersona:
        """Create a specific persona type."""
        # Get cost basis from competitor (with fallback)
        cost_basis = getattr(competitor, 'cost_basis', competitor.price * Decimal('0.7'))  # 30% margin default
        
        if persona_type == 'IrrationalSlasher':
            return IrrationalSlasher(competitor_id, cost_basis)
        elif persona_type == 'SlowFollower':
            return SlowFollower(competitor_id, cost_basis)
        elif persona_type == 'Default':
            # Create a basic rational persona (could implement DefaultPersona later)
            return SlowFollower(competitor_id, cost_basis)  # Use SlowFollower as default for now
        else:
            logger.warning(f"Unknown persona type {persona_type}, using SlowFollower as default")
            return SlowFollower(competitor_id, cost_basis)
    
    def get_persona_statistics(self) -> Dict:
        """Get statistics about persona distribution."""
        persona_counts = {}
        for persona in self.competitor_personas.values():
            persona_type = type(persona).__name__
            persona_counts[persona_type] = persona_counts.get(persona_type, 0) + 1
        
        return {
            'total_competitors': len(self.competitor_personas),
            'persona_distribution': persona_counts,
            'persona_types': list(set(type(p).__name__ for p in self.competitor_personas.values()))
        }
        # Add with the persona
        self.add_competitor(competitor, persona)
        
    def assign_persona(self, competitor_id: str, persona: CompetitorPersona) -> bool:
        """Assign a persona to an existing competitor."""
        if competitor_id in self.competitors:
            self.competitor_personas[competitor_id] = persona
            logger.info(f"Assigned {type(persona).__name__} persona to competitor {competitor_id}")
            return True
        return False
        
    def get_competitor_persona(self, competitor_id: str) -> Optional[CompetitorPersona]:
        """Get the persona assigned to a competitor."""
        return self.competitor_personas.get(competitor_id)
        
    def remove_competitor(self, competitor_id: str) -> bool:
        """Remove a competitor from the manager."""
        removed = False
        if competitor_id in self.competitors:
            del self.competitors[competitor_id]
            removed = True
        if competitor_id in self.competitor_personas:
            del self.competitor_personas[competitor_id]
        if competitor_id in self.competitor_states:
            del self.competitor_states[competitor_id]
        return removed
        
    def get_competitor(self, competitor_id: str) -> Optional[Competitor]:
        """Get a competitor by ID."""
        return self.competitors.get(competitor_id)
        
    def get_all_competitors(self) -> List[Competitor]:
        """Get all competitors."""
        return list(self.competitors.values())
        
    def _select_strategy(self, competitor: Competitor, market_trend: str) -> CompetitorStrategy:
        """Select strategy based on competitor profile and market conditions."""
        # Use competitor's preferred strategy if available
        if hasattr(competitor, 'preferred_strategy') and competitor.preferred_strategy:
            return CompetitorStrategy(competitor.preferred_strategy)
            
        # Otherwise, select based on market conditions and weights
        if market_trend == 'rising':
            # More aggressive in rising markets
            weights = {
                CompetitorStrategy.AGGRESSIVE: 0.5,
                CompetitorStrategy.ADAPTIVE: 0.3,
                CompetitorStrategy.CONSERVATIVE: 0.15,
                CompetitorStrategy.RANDOM: 0.05
            }
        elif market_trend == 'falling':
            # More conservative in falling markets
            weights = {
                CompetitorStrategy.CONSERVATIVE: 0.5,
                CompetitorStrategy.ADAPTIVE: 0.3,
                CompetitorStrategy.AGGRESSIVE: 0.15,
                CompetitorStrategy.RANDOM: 0.05
            }
        else:
            # Use default weights for stable markets
            weights = {
                CompetitorStrategy(k): v for k, v in self.strategy_weights.items()
            }
            
        # Weighted random selection
        strategies = list(weights.keys())
        weights_list = list(weights.values())
        return random.choices(strategies, weights=weights_list)[0]
        
    def _calculate_new_price(
        self,
        competitor: Competitor,
        strategy: CompetitorStrategy,
        our_price: Money,
        sales_velocity: float,
        market_trend: str
    ) -> Money:
        """Calculate new price based on strategy."""
        current_price = competitor.current_price
        
        if strategy == CompetitorStrategy.AGGRESSIVE:
            # Try to undercut our price
            target_price = our_price * 0.95  # 5% below our price
            max_decrease = current_price * (1 - self.max_price_change_percent)
            new_price = max(target_price, max_decrease)
            
        elif strategy == CompetitorStrategy.CONSERVATIVE:
            # Small adjustments based on market trend
            if market_trend == 'rising':
                adjustment = 1 + (self.max_price_change_percent * 0.3)
            elif market_trend == 'falling':
                adjustment = 1 - (self.max_price_change_percent * 0.3)
            else:
                adjustment = 1 + random.uniform(-0.02, 0.02)  # ±2%
            new_price = current_price * adjustment
            
        elif strategy == CompetitorStrategy.ADAPTIVE:
            # Respond to sales velocity and market conditions
            if sales_velocity > 1.0:  # High sales velocity
                # Market is hot, can increase prices
                adjustment = 1 + (self.max_price_change_percent * 0.6)
            elif sales_velocity < 0.5:  # Low sales velocity
                # Market is slow, decrease prices
                adjustment = 1 - (self.max_price_change_percent * 0.6)
            else:
                # Match our price approximately
                target_price = our_price * random.uniform(0.98, 1.02)
                return target_price
            new_price = current_price * adjustment
            
        else:  # RANDOM
            # Random price change within limits
            adjustment = 1 + random.uniform(
                -self.max_price_change_percent, 
                self.max_price_change_percent
            )
            new_price = current_price * adjustment
            
        # Ensure minimum price constraints
        min_price = Money.from_dollars(1.00)  # Minimum $1.00
        return max(new_price, min_price)
        
    def _is_valid_price_change(self, old_price: Money, new_price: Money) -> bool:
        """Validate that a price change is within acceptable limits."""
        if old_price == new_price:
            return False
            
        # Check minimum change amount
        price_diff = abs(new_price - old_price)
        if price_diff < self.min_price_change:
            return False
            
        # Check maximum change percentage
        max_change = old_price * self.max_price_change_percent
        if price_diff > max_change:
            return False
            
        return True
        
    def _calculate_action_confidence(
        self,
        competitor: Competitor,
        strategy: CompetitorStrategy,
        market_trend: str,
        sales_velocity: float
    ) -> float:
        """Calculate confidence level for the action."""
        base_confidence = 0.7
        
        # Adjust based on strategy
        strategy_confidence = {
            CompetitorStrategy.AGGRESSIVE: 0.8,
            CompetitorStrategy.CONSERVATIVE: 0.9,
            CompetitorStrategy.ADAPTIVE: 0.85,
            CompetitorStrategy.RANDOM: 0.5
        }
        
        confidence = strategy_confidence.get(strategy, base_confidence)
        
        # Adjust based on market conditions
        if market_trend == 'stable':
            confidence *= 1.1
        elif market_trend in ['rising', 'falling']:
            confidence *= 0.9
            
        # Adjust based on sales velocity
        if 0.7 <= sales_velocity <= 1.3:  # Normal range
            confidence *= 1.05
        else:
            confidence *= 0.95
            
        return min(confidence, 1.0)
        
    def get_market_position(self, our_price: Money) -> Dict:
        """Get our market position relative to competitors."""
        if not self.competitors:
            return {
                'position': 'only_seller',
                'price_rank': 1,
                'total_competitors': 0,
                'price_percentile': 100.0,
                'average_competitor_price': None,
                'lowest_competitor_price': None,
                'highest_competitor_price': None
            }
            
        competitor_prices = [
            comp.current_price for comp in self.competitors.values() 
            if comp.is_active
        ]
        
        if not competitor_prices:
            return {
                'position': 'only_active_seller',
                'price_rank': 1,
                'total_competitors': len(self.competitors),
                'price_percentile': 100.0,
                'average_competitor_price': None,
                'lowest_competitor_price': None,
                'highest_competitor_price': None
            }
            
        # Calculate statistics
        all_prices = competitor_prices + [our_price]
        all_prices.sort()
        
        our_rank = all_prices.index(our_price) + 1
        total_sellers = len(all_prices)
        percentile = (total_sellers - our_rank + 1) / total_sellers * 100
        
        # Determine position category
        if percentile >= 80:
            position = 'premium'
        elif percentile >= 60:
            position = 'above_average'
        elif percentile >= 40:
            position = 'average'
        elif percentile >= 20:
            position = 'below_average'
        else:
            position = 'discount'
            
        return {
            'position': position,
            'price_rank': our_rank,
            'total_competitors': len(competitor_prices),
            'price_percentile': percentile,
            'average_competitor_price': Money(sum(p.cents for p in competitor_prices) // len(competitor_prices)),
            'lowest_competitor_price': min(competitor_prices),
            'highest_competitor_price': max(competitor_prices)
        }
        
    def get_action_history(self, limit: Optional[int] = None) -> List[CompetitorAction]:
        """Get competitor action history."""
        if limit:
            return self.action_history[-limit:]
        return self.action_history.copy()
        
    def clear_action_history(self) -> None:
        """Clear the action history."""
        self.action_history.clear()
        
    def get_statistics(self) -> Dict:
        """Get competitor manager statistics."""
        active_competitors = sum(1 for c in self.competitors.values() if c.is_active)
        total_actions = len(self.action_history)
        
        # Action type distribution
        action_types = {}
        for action in self.action_history:
            action_types[action.action_type] = action_types.get(action.action_type, 0) + 1
            
        return {
            'total_competitors': len(self.competitors),
            'active_competitors': active_competitors,
            'total_actions': total_actions,
            'action_type_distribution': action_types,
            'average_action_confidence': (
                sum(a.confidence for a in self.action_history) / total_actions
                if total_actions > 0 else 0.0
            )
        }