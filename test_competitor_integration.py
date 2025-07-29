"""Integration tests for CompetitorManager -> SalesService event flow."""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from money import Money
from events import TickEvent, CompetitorPricesUpdated, CompetitorState
from event_bus import EventBus, AsyncioQueueBackend
from models.competitor import Competitor
from models.product import Product
from services.competitor_manager import CompetitorManager
from services.sales_service import SalesService
from services.fee_calculation_service import FeeCalculationService


@pytest_asyncio.fixture
async def event_bus():
    """Create a test event bus."""
    backend = AsyncioQueueBackend()
    bus = EventBus(backend)
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
def competitor_manager():
    """Create a test competitor manager."""
    config = {
        'competitor_strategy_weights': {
            'aggressive': 0.3,
            'conservative': 0.4,
            'adaptive': 0.2,
            'random': 0.1
        },
        'max_price_change_percent': 0.15,
        'min_price_change_dollars': 0.01,
        'market_sensitivity': 0.7,
        'competitor_response_delay': 0.1  # Short delay for testing
    }
    return CompetitorManager(config)


@pytest.fixture
def sales_service():
    """Create a test sales service."""
    config = {
        'demand_model': 'exponential',
        'base_conversion_rate': 0.15,
        'price_elasticity': -1.5,
        'trust_score_impact': 0.3,
        'inventory_impact_threshold': 10,
        'demand_volatility': 0.1,
        'conversion_volatility': 0.05,
        'max_history_size': 100
    }
    
    # Mock fee service
    fee_service = MagicMock(spec=FeeCalculationService)
    fee_service.calculate_fees = AsyncMock(return_value=(
        Money.from_dollars(5.00),  # total_fees
        {  # fee_breakdown
            'referral_fee': Money.from_dollars(3.00),
            'fulfillment_fee': Money.from_dollars(2.00)
        }
    ))
    
    return SalesService(config, fee_service)


@pytest.fixture
def test_competitors():
    """Create test competitors."""
    return [
        Competitor(
            asin="B001",
            price=Money.from_dollars(18.99),
            sales_velocity=1.2,
            bsr=5000,
            strategy="aggressive",
            trust_score=0.85
        ),
        Competitor(
            asin="B002", 
            price=Money.from_dollars(22.50),
            sales_velocity=0.8,
            bsr=10000,
            strategy="conservative",
            trust_score=0.75
        ),
        Competitor(
            asin="B003",
            price=Money.from_dollars(25.00),
            sales_velocity=0.6,
            bsr=15000,
            strategy="premium",
            trust_score=0.90
        )
    ]


@pytest.fixture
def test_product():
    """Create a test product."""
    return Product(
        asin="TEST001",
        category="Electronics",
        cost=Money.from_dollars(10.00),
        price=Money.from_dollars(20.00),
        base_demand=5.0,
        bsr=8000,
        trust_score=0.80,
        inventory_units=50,
        reserved_units=0,
        sales_velocity=1.0,
        conversion_rate=0.15
    )


class TestCompetitorEventFlow:
    """Test the complete competitor event flow."""
    
    @pytest.mark.asyncio
    async def test_competitor_manager_publishes_on_tick(
        self, 
        event_bus, 
        competitor_manager, 
        test_competitors
    ):
        """Test that CompetitorManager publishes CompetitorPricesUpdated on TickEvent."""
        # Setup competitors
        for competitor in test_competitors:
            # Add required attributes for event-driven flow
            competitor.competitor_id = competitor.asin
            competitor.is_active = True
            competitor.last_price_change = 0
            competitor_manager.add_competitor(competitor)
        
        # Start competitor manager
        competitor_manager.event_bus = event_bus
        await competitor_manager.start()
        
        # Create a collector for published events
        published_events = []
        
        async def event_collector(event):
            published_events.append(event)
        
        await event_bus.subscribe("CompetitorPricesUpdated", event_collector)
        
        # Publish TickEvent
        tick_event = TickEvent(
            event_id="tick_001",
            timestamp=datetime.now(),
            tick_number=1,
            simulation_time=datetime.now(),
            metadata={
                'market_conditions': {
                    'our_price': Money.from_dollars(20.00),
                    'sales_velocity': 1.0,
                    'market_trend': 'stable'
                }
            }
        )
        
        await event_bus.publish(tick_event)
        
        # Wait for event processing
        await asyncio.sleep(0.1)
        
        # Verify CompetitorPricesUpdated was published
        assert len(published_events) == 1
        competitor_event = published_events[0]
        assert isinstance(competitor_event, CompetitorPricesUpdated)
        assert competitor_event.tick_number == 1
        assert len(competitor_event.competitors) == 3
        
        # Verify competitor states have correct structure
        for comp_state in competitor_event.competitors:
            assert isinstance(comp_state, CompetitorState)
            assert isinstance(comp_state.price, Money)
            assert comp_state.bsr > 0
            assert comp_state.sales_velocity >= 0
        
        await competitor_manager.stop()
    
    @pytest.mark.asyncio
    async def test_sales_service_receives_competitor_data(
        self,
        event_bus,
        sales_service,
        test_competitors
    ):
        """Test that SalesService receives and stores competitor data."""
        # Start sales service
        await sales_service.start(event_bus)
        
        # Create competitor states
        competitor_states = [
            CompetitorState(
                asin=comp.asin,
                price=comp.price,
                bsr=comp.bsr,
                sales_velocity=comp.sales_velocity
            )
            for comp in test_competitors
        ]
        
        # Create and publish CompetitorPricesUpdated event
        competitor_event = CompetitorPricesUpdated(
            event_id="comp_001",
            timestamp=datetime.now(),
            tick_number=1,
            competitors=competitor_states,
            market_summary={
                'competitor_count': len(competitor_states),
                'average_price': '22.16',
                'min_price': '18.99',
                'max_price': '25.00'
            }
        )
        
        await event_bus.publish(competitor_event)
        
        # Wait for event processing
        await asyncio.sleep(0.1)
        
        # Verify sales service has updated competitor data
        assert len(sales_service.current_competitor_states) == 3
        assert sales_service.competitor_market_summary['competitor_count'] == 3
        
        # Verify competitor average price calculation
        avg_price = sales_service._get_competitor_average_price()
        assert avg_price is not None
        assert avg_price.cents > 0
        
        await sales_service.stop()
    
    @pytest.mark.asyncio
    async def test_end_to_end_competitor_flow(
        self,
        event_bus,
        competitor_manager,
        sales_service,
        test_competitors,
        test_product
    ):
        """Test complete end-to-end competitor flow."""
        # Setup competitors
        for competitor in test_competitors:
            competitor.competitor_id = competitor.asin
            competitor.is_active = True
            competitor.last_price_change = 0
            competitor_manager.add_competitor(competitor)
        
        # Start services
        competitor_manager.event_bus = event_bus
        await competitor_manager.start()
        await sales_service.start(event_bus)
        
        # Initial state - no competitor data
        assert len(sales_service.current_competitor_states) == 0
        initial_avg_price = sales_service._get_competitor_average_price()
        assert initial_avg_price is None
        
        # Publish TickEvent to trigger competitor updates
        tick_event = TickEvent(
            event_id="tick_001",
            timestamp=datetime.now(),
            tick_number=1,
            simulation_time=datetime.now(),
            metadata={
                'market_conditions': {
                    'our_price': test_product.price,
                    'sales_velocity': 1.0,
                    'market_trend': 'rising'
                }
            }
        )
        
        await event_bus.publish(tick_event)
        
        # Wait for event processing
        await asyncio.sleep(0.2)
        
        # Verify sales service received competitor data
        assert len(sales_service.current_competitor_states) == 3
        
        # Test competitor price analysis
        avg_price = sales_service._get_competitor_average_price()
        assert avg_price is not None
        assert avg_price.cents > 0
        
        # Test competition factor calculation
        competition_factor = sales_service._calculate_competition_factor(test_product.price)
        assert 0.5 <= competition_factor <= 1.5  # Should be within reasonable range
        
        # Test market summary
        market_summary = sales_service.get_competitor_market_summary()
        assert market_summary['competitor_count'] == 3
        assert market_summary['market_updated'] == True
        assert market_summary['average_price'] is not None
        
        # Cleanup
        await competitor_manager.stop()
        await sales_service.stop()
    
    @pytest.mark.asyncio
    async def test_competitor_price_changes_affect_demand(
        self,
        event_bus,
        competitor_manager,
        sales_service,
        test_competitors
    ):
        """Test that competitor price changes affect demand calculations."""
        # Setup competitors with different price points
        low_price_competitor = Competitor(
            asin="LOW001",
            price=Money.from_dollars(15.00),  # Lower than our price
            sales_velocity=1.5,
            bsr=3000,
            strategy="aggressive",
            trust_score=0.80
        )
        low_price_competitor.competitor_id = low_price_competitor.asin
        low_price_competitor.is_active = True
        low_price_competitor.last_price_change = 0
        
        high_price_competitor = Competitor(
            asin="HIGH001",
            price=Money.from_dollars(30.00),  # Higher than our price
            sales_velocity=0.5,
            bsr=20000,
            strategy="premium",
            trust_score=0.95
        )
        high_price_competitor.competitor_id = high_price_competitor.asin
        high_price_competitor.is_active = True
        high_price_competitor.last_price_change = 0
        
        competitor_manager.add_competitor(low_price_competitor)
        competitor_manager.add_competitor(high_price_competitor)
        
        # Start services
        competitor_manager.event_bus = event_bus
        await competitor_manager.start()
        await sales_service.start(event_bus)
        
        # Simulate tick with our price at $20
        our_price = Money.from_dollars(20.00)
        tick_event = TickEvent(
            event_id="tick_001",
            timestamp=datetime.now(),
            tick_number=1,
            simulation_time=datetime.now(),
            metadata={
                'market_conditions': {
                    'our_price': our_price,
                    'sales_velocity': 1.0,
                    'market_trend': 'stable'
                }
            }
        )
        
        await event_bus.publish(tick_event)
        await asyncio.sleep(0.2)
        
        # Test competition factor - we should be in middle range
        competition_factor = sales_service._calculate_competition_factor(our_price)
        assert 0.8 <= competition_factor <= 1.2  # Should be relatively neutral
        
        # Test with all low-price competitors (we become expensive)
        expensive_context_competitor = Competitor(
            asin="CHEAP001",
            price=Money.from_dollars(12.00),
            sales_velocity=2.0,
            bsr=1000,
            strategy="aggressive",
            trust_score=0.70
        )
        expensive_context_competitor.competitor_id = expensive_context_competitor.asin
        expensive_context_competitor.is_active = True
        expensive_context_competitor.last_price_change = 0
        
        competitor_manager.add_competitor(expensive_context_competitor)
        
        # Trigger another tick
        tick_event.tick_number = 2
        tick_event.event_id = "tick_002"
        await event_bus.publish(tick_event)
        await asyncio.sleep(0.2)
        
        # Now we should have lower competition factor (less demand due to high relative price)
        new_competition_factor = sales_service._calculate_competition_factor(our_price)
        assert new_competition_factor < competition_factor  # Should be lower now
        
        await competitor_manager.stop()
        await sales_service.stop()


if __name__ == "__main__":
    pytest.main([__file__])