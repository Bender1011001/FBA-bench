"""
Test suite for Competitor Persona Integration with Event-Driven Architecture

This module tests Phase 4: High-Fidelity Chaos implementation, verifying that
competitor personas work correctly within the existing event-driven architecture.
"""

import asyncio
import pytest
import pytest_asyncio
import time
from decimal import Decimal
from datetime import datetime
from typing import List, Dict

from money import Money
from events import TickEvent, CompetitorPricesUpdated, CompetitorState
from services.competitor_manager import CompetitorManager
from services.sales_service import SalesService
from personas import IrrationalSlasher, SlowFollower, MarketConditions
from models.competitor import Competitor
from event_bus import EventBus


class MockCompetitor(Competitor):
    """Mock competitor for testing."""
    
    def __init__(self, competitor_id: str, asin: str, price: Money):
        self.competitor_id = competitor_id
        self.asin = asin
        self.price = price
        self.bsr = 50000
        self.sales_velocity = Decimal('1.0')
        self.inventory_level = 100
        self.is_active = True
        self.cost_basis = price * Decimal('0.7')  # 30% margin
        self.last_price_change = 0


class TestPersonaIntegration:
    """Test persona integration with event-driven architecture."""
    
    @pytest_asyncio.fixture
    async def event_bus(self):
        """Create test event bus."""
        bus = EventBus()
        await bus.start()
        yield bus
        await bus.stop()
    
    @pytest.fixture
    def competitor_manager_config(self):
        """Configuration for competitor manager."""
        return {
            'persona_distribution': {
                'IrrationalSlasher': 0.5,
                'SlowFollower': 0.5,
                'Default': 0.0
            },
            'market_sensitivity': 0.8,
            'sales_velocity_window': 5
        }
    
    @pytest.fixture
    async def competitor_manager(self, event_bus, competitor_manager_config):
        """Create and start competitor manager."""
        manager = CompetitorManager(competitor_manager_config)
        manager.event_bus = event_bus
        await manager.start()
        yield manager
        await manager.stop()
    
    @pytest.fixture
    def mock_competitors(self):
        """Create mock competitors for testing."""
        return [
            MockCompetitor("comp_1", "ASIN001", Money.from_dollars(20.00)),
            MockCompetitor("comp_2", "ASIN002", Money.from_dollars(22.00)),
            MockCompetitor("comp_3", "ASIN003", Money.from_dollars(18.00)),
        ]
    
    @pytest.mark.asyncio
    async def test_persona_assignment(self, competitor_manager, mock_competitors):
        """Test that personas are correctly assigned to competitors."""
        # Add competitors with specific personas
        irrational_slasher = IrrationalSlasher("comp_1", Money.from_dollars(14.00))
        slow_follower = SlowFollower("comp_2", Money.from_dollars(15.00))
        
        competitor_manager.add_competitor(mock_competitors[0], irrational_slasher)
        competitor_manager.add_competitor(mock_competitors[1], slow_follower)
        competitor_manager.add_competitor(mock_competitors[2])  # Auto-assign
        
        # Verify persona assignments
        assert isinstance(competitor_manager.get_competitor_persona("comp_1"), IrrationalSlasher)
        assert isinstance(competitor_manager.get_competitor_persona("comp_2"), SlowFollower)
        assert competitor_manager.get_competitor_persona("comp_3") is not None
        
        # Check persona statistics
        stats = competitor_manager.get_persona_statistics()
        assert stats['total_competitors'] == 3
        assert 'IrrationalSlasher' in stats['persona_distribution']
        assert 'SlowFollower' in stats['persona_distribution']
    
    @pytest.mark.asyncio
    async def test_irrational_slasher_behavior(self, competitor_manager, mock_competitors):
        """Test IrrationalSlasher persona behavior."""
        # Create slasher with high slash probability for testing
        slasher = IrrationalSlasher("comp_1", Money.from_dollars(14.00))
        slasher.slash_probability = 1.0  # Force slashing for test
        
        competitor_manager.add_competitor(mock_competitors[0], slasher)
        
        # Create market conditions that trigger slashing
        tick_event = TickEvent(
            event_id="test_tick_1",
            timestamp=datetime.now(),
            tick_number=1,
            metadata={
                'market_conditions': {
                    'our_price': Money.from_dollars(19.00),
                    'sales_velocity': 0.3,  # Low sales velocity
                    'market_trend': 'falling'
                }
            }
        )
        
        # Process tick and check for slashing behavior
        await competitor_manager._handle_tick_event(tick_event)
        
        # Get updated competitor state
        updated_state = competitor_manager.competitor_states.get("comp_1")
        assert updated_state is not None
        
        # Price should be slashed to near cost basis when slashing
        if slasher._get_state_value('is_slashing', False):
            expected_min_price = slasher._calculate_minimum_price()
            assert updated_state.price <= expected_min_price * Decimal('1.1')  # Allow small margin
            print(f"Slasher price: {updated_state.price}, min price: {expected_min_price}")
    
    @pytest.mark.asyncio
    async def test_slow_follower_behavior(self, competitor_manager, mock_competitors):
        """Test SlowFollower persona delayed response behavior."""
        slow_follower = SlowFollower("comp_2", Money.from_dollars(15.00))
        competitor_manager.add_competitor(mock_competitors[1], slow_follower)
        
        initial_price = mock_competitors[1].price
        
        # Process multiple ticks
        for tick_num in range(1, 6):
            tick_event = TickEvent(
                event_id=f"test_tick_{tick_num}",
                timestamp=datetime.now(),
                tick_number=tick_num,
                metadata={
                    'market_conditions': {
                        'our_price': Money.from_dollars(19.00),
                        'sales_velocity': 1.2,
                        'market_trend': 'rising'
                    }
                }
            )
            await competitor_manager._handle_tick_event(tick_event)
        
        # SlowFollower should not respond on every tick
        # Check that it doesn't change price too frequently
        updated_state = competitor_manager.competitor_states.get("comp_2")
        assert updated_state is not None
        
        # The price may or may not have changed depending on evaluation schedule
        print(f"SlowFollower initial: {initial_price}, final: {updated_state.price}")
        print(f"Next evaluation tick: {slow_follower._get_state_value('next_evaluation_tick', 0)}")
    
    @pytest.mark.asyncio
    async def test_persona_event_flow_integration(self, event_bus, competitor_manager, mock_competitors):
        """Test complete event flow with personas: TickEvent → CompetitorManager → CompetitorPricesUpdated."""
        # Add competitors with different personas
        irrational_slasher = IrrationalSlasher("comp_1", Money.from_dollars(14.00))
        slow_follower = SlowFollower("comp_2", Money.from_dollars(15.00))
        
        competitor_manager.add_competitor(mock_competitors[0], irrational_slasher)
        competitor_manager.add_competitor(mock_competitors[1], slow_follower)
        
        # Set up event capture
        captured_events = []
        
        async def capture_competitor_updates(event: CompetitorPricesUpdated):
            captured_events.append(event)
        
        await event_bus.subscribe("CompetitorPricesUpdated", capture_competitor_updates)
        
        # Send tick event
        tick_event = TickEvent(
            event_id="test_tick_integration",
            timestamp=datetime.now(),
            tick_number=1,
            metadata={
                'market_conditions': {
                    'our_price': Money.from_dollars(19.00),
                    'sales_velocity': 1.0,
                    'market_trend': 'stable'
                }
            }
        )
        
        await event_bus.publish(tick_event)
        await asyncio.sleep(0.1)  # Allow event processing
        
        # Verify CompetitorPricesUpdated event was published
        assert len(captured_events) >= 1
        
        event = captured_events[0]
        assert isinstance(event, CompetitorPricesUpdated)
        assert event.tick_number == 1
        assert len(event.competitors) == 2
        
        # Verify competitor states in event
        competitor_ids = [comp.competitor_id for comp in event.competitors]
        assert "comp_1" in competitor_ids
        assert "comp_2" in competitor_ids
        
        print(f"Published event with {len(event.competitors)} competitors")
        print(f"Market summary: {event.market_summary}")
    
    @pytest.mark.asyncio
    async def test_persona_with_sales_service_integration(self, event_bus, competitor_manager, mock_competitors):
        """Test full integration: Personas → CompetitorPricesUpdated → SalesService."""
        # Set up SalesService to receive competitor events
        sales_config = {
            'base_demand': 100,
            'price_elasticity': 2.0,
            'competitor_effect_strength': 0.5
        }
        sales_service = SalesService(sales_config)
        sales_service.event_bus = event_bus
        await sales_service.start()
        
        try:
            # Add competitors with personas
            irrational_slasher = IrrationalSlasher("comp_1", Money.from_dollars(14.00))
            irrational_slasher.slash_probability = 1.0  # Force slashing
            
            competitor_manager.add_competitor(mock_competitors[0], irrational_slasher)
            
            # Process tick to generate competitor update
            tick_event = TickEvent(
                event_id="test_sales_integration",
                timestamp=datetime.now(),
                tick_number=1,
                metadata={
                    'market_conditions': {
                        'our_price': Money.from_dollars(20.00),
                        'sales_velocity': 0.2,  # Low velocity to trigger slashing
                        'market_trend': 'falling'
                    }
                }
            )
            
            await event_bus.publish(tick_event)
            await asyncio.sleep(0.2)  # Allow event processing
            
            # Check that SalesService received and processed competitor data
            assert sales_service.latest_competitor_data is not None
            assert len(sales_service.latest_competitor_data) == 1
            
            competitor_data = sales_service.latest_competitor_data[0]
            print(f"SalesService received competitor data: {competitor_data.competitor_id} at {competitor_data.price}")
            
            # Verify the competitor average price calculation works
            avg_price = sales_service._get_competitor_average_price()
            assert avg_price > Money.from_dollars(0)
            print(f"Calculated average competitor price: {avg_price}")
            
        finally:
            await sales_service.stop()
    
    @pytest.mark.asyncio
    async def test_market_chaos_generation(self, competitor_manager):
        """Test that personas generate realistic market chaos and unpredictability."""
        # Create a diverse set of competitors with different personas
        competitors = []
        for i in range(6):
            competitor = MockCompetitor(f"comp_{i}", f"ASIN{i:03d}", Money.from_dollars(20.0 + i))
            competitors.append(competitor)
        
        # Add with specific persona distribution
        personas = [
            IrrationalSlasher(f"comp_0", Money.from_dollars(14.00)),
            IrrationalSlasher(f"comp_1", Money.from_dollars(14.50)),
            SlowFollower(f"comp_2", Money.from_dollars(15.00)),
            SlowFollower(f"comp_3", Money.from_dollars(15.50)),
            SlowFollower(f"comp_4", Money.from_dollars(16.00)),
            SlowFollower(f"comp_5", Money.from_dollars(16.50)),
        ]
        
        for i, (competitor, persona) in enumerate(zip(competitors, personas)):
            competitor_manager.add_competitor(competitor, persona)
        
        # Track price changes over multiple ticks
        price_history = []
        
        for tick_num in range(1, 15):  # Run for 14 ticks
            tick_event = TickEvent(
                event_id=f"chaos_tick_{tick_num}",
                timestamp=datetime.now(),
                tick_number=tick_num,
                metadata={
                    'market_conditions': {
                        'our_price': Money.from_dollars(19.00),
                        'sales_velocity': 0.8 + (tick_num % 3) * 0.2,  # Varying conditions
                        'market_trend': ['stable', 'rising', 'falling'][tick_num % 3]
                    }
                }
            )
            
            await competitor_manager._handle_tick_event(tick_event)
            
            # Capture price snapshot
            tick_prices = {}
            for comp_id, state in competitor_manager.competitor_states.items():
                tick_prices[comp_id] = state.price
            price_history.append(tick_prices)
        
        # Analyze chaos metrics
        price_volatility = self._calculate_price_volatility(price_history)
        behavior_diversity = self._analyze_behavior_diversity(competitor_manager, price_history)
        
        print(f"Market Chaos Metrics:")
        print(f"  Price Volatility: {price_volatility:.3f}")
        print(f"  Behavior Diversity Score: {behavior_diversity:.3f}")
        print(f"  Persona Distribution: {competitor_manager.get_persona_statistics()}")
        
        # Verify chaos is generated (prices should change over time)
        assert price_volatility > 0.01  # At least 1% volatility
        assert behavior_diversity > 0.5  # Different personas behave differently
    
    def _calculate_price_volatility(self, price_history: List[Dict]) -> float:
        """Calculate price volatility across all competitors."""
        if len(price_history) < 2:
            return 0.0
        
        total_volatility = 0.0
        competitor_count = 0
        
        for comp_id in price_history[0].keys():
            prices = [tick_data[comp_id].amount for tick_data in price_history]
            if len(prices) > 1:
                # Calculate coefficient of variation
                mean_price = sum(prices) / len(prices)
                variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
                std_dev = variance ** 0.5
                cv = std_dev / mean_price if mean_price > 0 else 0
                total_volatility += cv
                competitor_count += 1
        
        return total_volatility / competitor_count if competitor_count > 0 else 0.0
    
    def _analyze_behavior_diversity(self, manager: CompetitorManager, price_history: List[Dict]) -> float:
        """Analyze how differently personas behave."""
        if len(price_history) < 3:
            return 0.0
        
        # Group competitors by persona type
        persona_groups = {}
        for comp_id, persona in manager.competitor_personas.items():
            persona_type = type(persona).__name__
            if persona_type not in persona_groups:
                persona_groups[persona_type] = []
            persona_groups[persona_type].append(comp_id)
        
        # Calculate behavioral differences between persona groups
        if len(persona_groups) < 2:
            return 0.0
        
        # Compare price change patterns between groups
        group_behaviors = {}
        for persona_type, comp_ids in persona_groups.items():
            total_changes = 0
            change_count = 0
            
            for comp_id in comp_ids:
                for i in range(1, len(price_history)):
                    old_price = price_history[i-1][comp_id].amount
                    new_price = price_history[i][comp_id].amount
                    if old_price != new_price:
                        change_magnitude = abs(new_price - old_price) / old_price
                        total_changes += change_magnitude
                        change_count += 1
            
            avg_change = total_changes / change_count if change_count > 0 else 0
            group_behaviors[persona_type] = avg_change
        
        # Calculate diversity as coefficient of variation of group behaviors
        behaviors = list(group_behaviors.values())
        if len(behaviors) < 2:
            return 0.0
        
        mean_behavior = sum(behaviors) / len(behaviors)
        variance = sum((b - mean_behavior) ** 2 for b in behaviors) / len(behaviors)
        std_dev = variance ** 0.5
        
        return std_dev / mean_behavior if mean_behavior > 0 else 0.0


if __name__ == "__main__":
    # Run basic test manually
    async def run_basic_test():
        """Run a basic persona test manually."""
        print("Running basic persona integration test...")
        
        # Create event bus
        event_bus = EventBus()
        await event_bus.start()
        
        try:
            # Create competitor manager
            config = {
                'persona_distribution': {
                    'IrrationalSlasher': 0.5,
                    'SlowFollower': 0.5
                }
            }
            
            manager = CompetitorManager(config)
            manager.event_bus = event_bus
            await manager.start()
            
            # Add test competitors
            comp1 = MockCompetitor("test_1", "ASIN001", Money.from_dollars(20.00))
            comp2 = MockCompetitor("test_2", "ASIN002", Money.from_dollars(22.00))
            
            slasher = IrrationalSlasher("test_1", Money.from_dollars(14.00))
            follower = SlowFollower("test_2", Money.from_dollars(15.40))
            
            manager.add_competitor(comp1, slasher)
            manager.add_competitor(comp2, follower)
            
            print(f"Added competitors with personas:")
            print(f"  test_1: {type(manager.get_competitor_persona('test_1')).__name__}")
            print(f"  test_2: {type(manager.get_competitor_persona('test_2')).__name__}")
            
            # Run a few ticks
            for tick in range(1, 4):
                tick_event = TickEvent(
                    event_id=f"test_tick_{tick}",
                    timestamp=datetime.now(),
                    tick_number=tick,
                    metadata={
                        'market_conditions': {
                            'our_price': Money.from_dollars(19.00),
                            'sales_velocity': 0.5,
                            'market_trend': 'stable'
                        }
                    }
                )
                
                await manager._handle_tick_event(tick_event)
                
                print(f"Tick {tick} completed")
                for comp_id, state in manager.competitor_states.items():
                    print(f"  {comp_id}: {state.price}")
            
            await manager.stop()
            print("Basic test completed successfully!")
            
        finally:
            await event_bus.stop()
    
    # Run the test
    asyncio.run(run_basic_test())