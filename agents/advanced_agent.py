"""
AdvancedAgent for FBA-Bench v3 multi-agent platform.

Demonstrates the sandboxed agent pattern where agents can only interact
with the world through commands published to the EventBus.
"""

import asyncio
import logging
import uuid
import random
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

from money import Money
from events import BaseEvent, SetPriceCommand, ProductPriceUpdated, TickEvent
from event_bus import EventBus, get_event_bus


logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for AdvancedAgent behavior."""
    agent_id: str
    target_asin: str
    strategy: str = "profit_maximizer"
    price_sensitivity: float = 0.1  # How much to adjust prices (0.0 to 1.0)
    reaction_speed: int = 1  # How often to react (every N ticks)
    min_price: Money = Money(500)  # Minimum price $5.00
    max_price: Money = Money(5000)  # Maximum price $50.00


class AdvancedAgent:
    """
    Sandboxed AI agent for FBA-Bench v3 multi-agent platform.
    
    Demonstrates the core multi-agent principles:
    1. No direct access to world state - all perception through events
    2. No direct actions - all intents expressed as commands
    3. Command-arbitration-event loop for all interactions
    4. Complete isolation from other services and agents
    
    The agent subscribes to relevant events to build its world model,
    then publishes SetPriceCommand events to express pricing intentions.
    The WorldStore arbitrates these commands and publishes canonical updates.
    """
    
    def __init__(self, config: AgentConfig, event_bus: Optional[EventBus] = None):
        """
        Initialize the AdvancedAgent.
        
        Args:
            config: Agent configuration including strategy and constraints
            event_bus: EventBus for communication (sandboxed interface)
        """
        self.config = config
        self.event_bus = event_bus or get_event_bus()
        
        # Agent state (built from events only)
        self.current_tick = 0
        self.last_action_tick = 0
        self.current_price: Optional[Money] = None
        self.price_history: List[Money] = []
        
        # Market perception (from events)
        self.market_data: Dict[str, Any] = {}
        self.competitor_prices: List[Money] = []
        
        # Agent decision-making state
        self.decision_history: List[Dict[str, Any]] = []
        self.commands_sent = 0
        self.commands_accepted = 0
        
        logger.info(f"AdvancedAgent initialized: id={config.agent_id}, target={config.target_asin}, strategy={config.strategy}")
    
    async def start(self):
        """Start the agent and subscribe to relevant events."""
        # Subscribe to events for world perception
        await self.event_bus.subscribe('TickEvent', self.handle_tick_event)
        await self.event_bus.subscribe('ProductPriceUpdated', self.handle_price_updated)
        
        logger.info(f"AdvancedAgent {self.config.agent_id} started - subscribed to events")
    
    async def stop(self):
        """Stop the agent."""
        logger.info(f"AdvancedAgent {self.config.agent_id} stopped")
    
    # Event Handlers (World Perception)
    
    async def handle_tick_event(self, event: TickEvent):
        """
        Process tick events to trigger agent decision-making.
        
        This is the main decision loop where the agent evaluates
        its current state and decides whether to take action.
        """
        self.current_tick = event.tick_number
        
        # Check if it's time to make a decision
        if self._should_act():
            await self._make_pricing_decision()
    
    async def handle_price_updated(self, event: ProductPriceUpdated):
        """
        Process ProductPriceUpdated events to maintain world model.
        
        This is how the agent learns about price changes in the market,
        including its own accepted commands and competitor actions.
        """
        if event.asin == self.config.target_asin:
            # Our product price was updated
            previous_price = self.current_price
            self.current_price = event.new_price
            self.price_history.append(event.new_price)
            
            # Track if this was our command that was accepted
            if event.agent_id == self.config.agent_id:
                self.commands_accepted += 1
                logger.info(f"Agent {self.config.agent_id} command accepted: price={event.new_price}")
            else:
                logger.info(f"Agent {self.config.agent_id} observed external price change: {previous_price} -> {event.new_price}")
        
        else:
            # Competitor price change - update market perception
            self.competitor_prices.append(event.new_price)
            if len(self.competitor_prices) > 10:  # Keep last 10 competitor prices
                self.competitor_prices.pop(0)
    
    # Decision Making
    
    def _should_act(self) -> bool:
        """Determine if the agent should take action this tick."""
        ticks_since_last_action = self.current_tick - self.last_action_tick
        return ticks_since_last_action >= self.config.reaction_speed
    
    async def _make_pricing_decision(self):
        """
        Make a pricing decision and publish SetPriceCommand if needed.
        
        This implements the agent's pricing strategy and demonstrates
        the command pattern for expressing agent intentions.
        """
        try:
            # Calculate desired price based on strategy
            desired_price = self._calculate_desired_price()
            
            if desired_price and self._should_change_price(desired_price):
                # Publish SetPriceCommand to express pricing intention
                await self._publish_price_command(desired_price)
                self.last_action_tick = self.current_tick
        
        except Exception as e:
            logger.error(f"Agent {self.config.agent_id} decision error: {e}", exc_info=True)
    
    def _calculate_desired_price(self) -> Optional[Money]:
        """
        Calculate desired price based on agent's strategy.
        
        This demonstrates different agent strategies and how they
        can lead to diverse market behaviors.
        """
        if self.config.strategy == "profit_maximizer":
            return self._profit_maximizer_strategy()
        elif self.config.strategy == "market_follower":
            return self._market_follower_strategy()
        elif self.config.strategy == "aggressive_pricer":
            return self._aggressive_pricer_strategy()
        else:
            return self._random_strategy()
    
    def _profit_maximizer_strategy(self) -> Optional[Money]:
        """Strategy that tries to maximize profit through gradual price increases."""
        if not self.current_price:
            return Money(2000)  # Start at $20.00
        
        # Gradually increase price to test market elasticity
        increase_factor = 1.0 + (self.config.price_sensitivity * 0.5)
        new_price = Money(int(self.current_price.cents * increase_factor))
        
        # Bound within agent's constraints
        return min(max(new_price, self.config.min_price), self.config.max_price)
    
    def _market_follower_strategy(self) -> Optional[Money]:
        """Strategy that follows competitor pricing with slight undercut."""
        if not self.competitor_prices:
            return Money(2000)  # Default price
        
        # Calculate average competitor price
        avg_competitor_price = Money(sum(p.cents for p in self.competitor_prices) // len(self.competitor_prices))
        
        # Undercut by small amount
        undercut_factor = 1.0 - (self.config.price_sensitivity * 0.2)
        new_price = Money(int(avg_competitor_price.cents * undercut_factor))
        
        return min(max(new_price, self.config.min_price), self.config.max_price)
    
    def _aggressive_pricer_strategy(self) -> Optional[Money]:
        """Strategy that aggressively undercuts competition."""
        if not self.competitor_prices:
            return Money(1800)  # Start low at $18.00
        
        # Find minimum competitor price and undercut significantly
        min_competitor_price = min(self.competitor_prices)
        undercut_factor = 1.0 - (self.config.price_sensitivity * 0.3)
        new_price = Money(int(min_competitor_price.cents * undercut_factor))
        
        return min(max(new_price, self.config.min_price), self.config.max_price)
    
    def _random_strategy(self) -> Optional[Money]:
        """Random pricing strategy for chaos testing."""
        price_range = self.config.max_price.cents - self.config.min_price.cents
        random_cents = self.config.min_price.cents + random.randint(0, price_range)
        return Money(random_cents)
    
    def _should_change_price(self, desired_price: Money) -> bool:
        """Determine if the price change is significant enough to warrant a command."""
        if not self.current_price:
            return True  # Always set initial price
        
        # Only change if difference is meaningful
        price_diff_ratio = abs(desired_price.cents - self.current_price.cents) / self.current_price.cents
        return price_diff_ratio >= 0.01  # 1% minimum change
    
    async def _publish_price_command(self, new_price: Money):
        """
        Publish SetPriceCommand to express pricing intention.
        
        This demonstrates the sandboxed agent pattern where agents
        can only express intentions through commands, not take direct actions.
        """
        command = SetPriceCommand(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            agent_id=self.config.agent_id,
            asin=self.config.target_asin,
            new_price=new_price,
            reason=f"Strategy: {self.config.strategy}, Target: profit optimization"
        )
        
        await self.event_bus.publish(command)
        self.commands_sent += 1
        
        # Record decision for analysis
        self.decision_history.append({
            'tick': self.current_tick,
            'command_id': command.event_id,
            'old_price': str(self.current_price) if self.current_price else None,
            'new_price': str(new_price),
            'strategy': self.config.strategy,
            'reason': command.reason
        })
        
        logger.info(f"Agent {self.config.agent_id} published SetPriceCommand: price={new_price}, strategy={self.config.strategy}")
    
    # Status and Analytics
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status for monitoring."""
        return {
            'agent_id': self.config.agent_id,
            'target_asin': self.config.target_asin,
            'strategy': self.config.strategy,
            'current_tick': self.current_tick,
            'current_price': str(self.current_price) if self.current_price else None,
            'commands_sent': self.commands_sent,
            'commands_accepted': self.commands_accepted,
            'acceptance_rate': self.commands_accepted / max(1, self.commands_sent),
            'competitor_prices_observed': len(self.competitor_prices),
            'decisions_made': len(self.decision_history)
        }
    
    def get_decision_history(self) -> List[Dict[str, Any]]:
        """Get agent's decision history for analysis."""
        return self.decision_history.copy()


# Factory functions for common agent types

def create_profit_maximizer_agent(agent_id: str, target_asin: str) -> AdvancedAgent:
    """Create a profit-maximizing agent."""
    config = AgentConfig(
        agent_id=agent_id,
        target_asin=target_asin,
        strategy="profit_maximizer",
        price_sensitivity=0.15,
        reaction_speed=2
    )
    return AdvancedAgent(config)


def create_market_follower_agent(agent_id: str, target_asin: str) -> AdvancedAgent:
    """Create a market-following agent."""
    config = AgentConfig(
        agent_id=agent_id,
        target_asin=target_asin,
        strategy="market_follower",
        price_sensitivity=0.1,
        reaction_speed=3
    )
    return AdvancedAgent(config)


def create_aggressive_agent(agent_id: str, target_asin: str) -> AdvancedAgent:
    """Create an aggressive pricing agent."""
    config = AgentConfig(
        agent_id=agent_id,
        target_asin=target_asin,
        strategy="aggressive_pricer",
        price_sensitivity=0.25,
        reaction_speed=1
    )
    return AdvancedAgent(config)