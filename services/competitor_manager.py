import logging
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from datetime import datetime
from enum import Enum

from money import Money # Ensure Money is imported
from models.competitor import Competitor
from events import TickEvent, CompetitorPricesUpdated, CompetitorState
from personas import CompetitorPersona # New import

# Import get_event_bus for instantiation and EventBus for type hinting
from event_bus import get_event_bus
if TYPE_CHECKING:
    from event_bus import EventBus 
    from services.world_store import WorldStore # For type hinting only


logger = logging.getLogger(__name__)


class CompetitorStrategy(Enum):
    """Competitor pricing strategies."""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"
    RANDOM = "random"


class CompetitorManager:
    """
    Manages competitor agents within the FBA-Bench simulation.
    
    This service simulates competitor behavior, updating their prices, BSRs,
    and sales velocities based on predefined strategies or market conditions.
    It publishes CompetitorPricesUpdated events to reflect these changes.
    """
    
    def __init__(self, config: Dict, world_store: 'WorldStore'): # Use forward reference
        """
        Initialize the CompetitorManager.
        
        Args:
            config: Service configuration
            world_store: WorldStore instance for product state
        """
        self.config = config
        self.world_store = world_store
        self.event_bus: Optional['EventBus'] = None # Use forward reference
        
        # Competitors managed by this service
        self.competitors: Dict[str, Competitor] = {}  # ASIN -> Competitor instance
        
        # Strategies for generating competitor data
        self.competitor_strategies = {
            CompetitorStrategy.AGGRESSIVE: self._aggressive_strategy,
            CompetitorStrategy.CONSERVATIVE: self._conservative_strategy,
            CompetitorStrategy.ADAPTIVE: self._adaptive_strategy,
            CompetitorStrategy.RANDOM: self._random_strategy
        }
        
        self.pricing_volatility = config.get('pricing_volatility', 0.05)
        self.bsr_volatility = config.get('bsr_volatility', 0.1)
        self.sales_volatility = config.get('sales_volatility', 0.1)

        # Statistics
        self.updates_published = 0
        self.total_competitors_tracked = 0
        
        logger.info("CompetitorManager initialized")
    
    async def start(self, event_bus: 'EventBus') -> None: # Use forward reference
        """Starts the CompetitorManager and subscribes to events."""
        self.event_bus = event_bus
        await self.event_bus.subscribe(TickEvent, self._handle_tick_event)
        logger.info("CompetitorManager started and subscribed to TickEvent")
    
    async def stop(self) -> None:
        """Stops the CompetitorManager."""
        logger.info("CompetitorManager stopped")
        
    def register_competitor(self, competitor_id: str, initial_price: Money, initial_bsr: int = 100000, initial_sales_velocity: float = 10.0, strategy: CompetitorStrategy = CompetitorStrategy.ADAPTIVE, persona: Optional[CompetitorPersona] = None) -> None:
        """
        Registers a new competitor to be managed by this service.
        
        Args:
            competitor_id: Unique ID for the competitor (e.g., ASIN)
            initial_price: Starting price of their product
            initial_bsr: Initial Best Seller Rank
            initial_sales_velocity: Initial daily sales velocity
            strategy: Pricing strategy this competitor follows
            persona: Optional competitor persona influencing behavior
        """
        if competitor_id in self.competitors:
            logger.warning(f"Competitor {competitor_id} already registered. Skipping.")
            return

        self.competitors[competitor_id] = Competitor(
            id=competitor_id,
            price=initial_price,
            bsr=initial_bsr,
            sales_velocity=initial_sales_velocity,
            strategy=strategy,
            persona=persona
        )
        self.total_competitors_tracked += 1
        logger.info(f"Competitor {competitor_id} registered with strategy {strategy.value}.")

    async def _handle_tick_event(self, event: TickEvent) -> None:
        """Handle tick events by updating competitor data and publishing updates."""
        try:
            logger.debug(f"CompetitorManager received TickEvent for tick {event.tick_number}. Updating competitors...")
            
            updated_competitor_states: List[CompetitorState] = []
            
            for competitor_id, competitor in self.competitors.items():
                # Get our product's current price from WorldStore to factor into competitor decisions
                our_product_state = self.world_store.get_product_state("B0DEFAULT") # Assuming a default product ASIN for comparison
                our_price = our_product_state.price if our_product_state else Money(100) # Default if our product not found

                new_price, new_bsr, new_sales_velocity = self._apply_strategy(competitor, our_price, event)
                
                # Update competitor's state
                competitor.price = new_price
                competitor.bsr = int(new_bsr)
                competitor.sales_velocity = new_sales_velocity
                
                updated_competitor_states.append(
                    CompetitorState(
                        asin=competitor.id,
                        price=new_price,
                        bsr=int(new_bsr),
                        sales_velocity=new_sales_velocity
                    )
                )
            
            # Publish update event
            if self.event_bus and updated_competitor_states:
                market_summary = self._calculate_market_summary(updated_competitor_states)
                update_event = CompetitorPricesUpdated(
                    event_id=f"competitor_update_{event.tick_number}_{int(datetime.now().timestamp())}",
                    timestamp=datetime.now(),
                    tick_number=event.tick_number,
                    competitors=updated_competitor_states,
                    market_summary=market_summary
                )
                await self.event_bus.publish(update_event)
                self.updates_published += 1
                logger.debug(f"Published CompetitorPricesUpdated event for tick {event.tick_number} with {len(updated_competitor_states)} competitors.")
                
        except Exception as e:
            logger.error(f"Error processing TickEvent in CompetitorManager: {e}", exc_info=True)

    def _apply_strategy(self, competitor: Competitor, our_price: Money, tick_event: TickEvent) -> Tuple[Money, float, float]:
        """Apply competitor's defined strategy to update its prices and metrics."""
        strategy_func = self.competitor_strategies.get(competitor.strategy)
        if not strategy_func:
            logger.warning(f"Unknown strategy {competitor.strategy} for competitor {competitor.id}. Defaulting to adaptive.")
            strategy_func = self._adaptive_strategy
        
        return strategy_func(competitor, our_price, tick_event)

    def _aggressive_strategy(self, competitor: Competitor, our_price: Money, tick_event: TickEvent) -> Tuple[Money, float, float]:
        """Competitor aggressively undercuts our price."""
        new_price = our_price - Money(10)  # Always $0.10 less
        new_price = max(Money(1), new_price) # Ensure price is not too low
        new_bsr = max(1.0, competitor.bsr * (1 - self.bsr_volatility * random.random())) # Slight improvement
        new_sales_velocity = competitor.sales_velocity * (1 + self.sales_volatility * random.random()) # Increase sales
        return new_price, new_bsr, new_sales_velocity

    def _conservative_strategy(self, competitor: Competitor, our_price: Money, tick_event: TickEvent) -> Tuple[Money, float, float]:
        """Competitor maintains stable prices, reacts slowly."""
        new_price = competitor.price # Price remains stable
        new_bsr = competitor.bsr * (1 + self.bsr_volatility * random.random()) # Slight degradation
        new_sales_velocity = competitor.sales_velocity * (1 - self.sales_volatility * random.random()) # Slight decrease
        return new_price, new_bsr, new_sales_velocity

    def _adaptive_strategy(self, competitor: Competitor, our_price: Money, tick_event: TickEvent) -> Tuple[Money, float, float]:
        """Competitor adapts to market, adjusts prices to stay competitive."""
        # Adjust price towards average of our price and its current price
        target_price_cents = (competitor.price.cents + our_price.cents) / 2
        new_price = Money(target_price_cents)
        
        # Add some randomness to BSR and sales velocity
        new_bsr = competitor.bsr + random.randint(-1000, 1000)
        new_bsr = max(1, new_bsr)
        new_sales_velocity = competitor.sales_velocity * (1 + random.uniform(-self.sales_volatility, self.sales_volatility))
        return new_price, new_bsr, new_sales_velocity

    def _random_strategy(self, competitor: Competitor, our_price: Money, tick_event: TickEvent) -> Tuple[Money, float, float]:
        """Competitor changes prices randomly."""
        new_price = Money.from_dollars(random.uniform(competitor.price.dollars * 0.8, competitor.price.dollars * 1.2)) # +/- 20%
        new_bsr = competitor.bsr + random.randint(-5000, 5000)
        new_bsr = max(1, new_bsr)
        new_sales_velocity = competitor.sales_velocity * (1 + random.uniform(-0.2, 0.2))
        return new_price, new_bsr, new_sales_velocity

    def _calculate_market_summary(self, competitors: List[CompetitorState]) -> Dict[str, Any]:
        """Calculate summary metrics for the market based on competitor data."""
        if not competitors:
            return {
                'competitor_count': 0,
                'average_price': Money.zero().dollars,
                'min_price': Money.zero().dollars,
                'max_price': Money.zero().dollars,
                'average_bsr': 0,
                'average_sales_velocity': 0.0
            }
        
        prices = [c.price.dollars for c in competitors]
        bsrs = [c.bsr for c in competitors]
        sales_velocities = [c.sales_velocity for c in competitors]
        
        return {
            'competitor_count': len(competitors),
            'average_price': sum(prices) / len(prices),
            'min_price': min(prices),
            'max_price': max(prices),
            'average_bsr': sum(bsrs) / len(bsrs),
            'average_sales_velocity': sum(sales_velocities) / len(sales_velocities)
        }
