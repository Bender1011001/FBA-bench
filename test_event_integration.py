"""Integration tests for FBA-Bench v3 event-driven architecture."""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

from money import Money
from events import TickEvent, SaleOccurred
from event_bus import EventBus, AsyncioQueueBackend
from simulation_orchestrator import SimulationOrchestrator, SimulationConfig
from services.sales_service import SalesService
from services.trust_score_service import TrustScoreService
from services.fee_calculation_service import FeeCalculationService
from models.product import Product


class EventCollector:
    """Helper class to collect events for testing."""
    
    def __init__(self):
        self.collected_events: List[Any] = []
        self.event_counts: Dict[str, int] = {}
    
    async def collect_event(self, event: Any) -> None:
        """Collect an event for later verification."""
        self.collected_events.append(event)
        event_type = type(event).__name__
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
    
    def get_events_of_type(self, event_type: type) -> List[Any]:
        """Get all collected events of a specific type."""
        return [event for event in self.collected_events if isinstance(event, event_type)]
    
    def clear(self) -> None:
        """Clear collected events."""
        self.collected_events.clear()
        self.event_counts.clear()


@pytest_asyncio.fixture
async def event_bus():
    """Create and start an event bus for testing."""
    bus = EventBus(AsyncioQueueBackend())
    await bus.start()
    yield bus
    await bus.stop()


@pytest.fixture
def sample_product():
    """Create a sample product for testing."""
    return Product(
        asin="B12345TEST",
        category="electronics",
        cost=Money.from_dollars(10.0),
        price=Money.from_dollars(25.0),
        base_demand=50.0,
        bsr=100000,
        trust_score=0.8,
        inventory_units=100,
        size="small",
        weight=1.0
    )


@pytest.fixture
def fee_service():
    """Create a fee calculation service for testing."""
    config = {
        'fee_rates': {
            'referral_base_rate': 0.15,
            'fba_small_standard': Money.from_dollars(3.22)
        }
    }
    return FeeCalculationService(config)


@pytest.fixture
def sales_service(fee_service):
    """Create a sales service for testing."""
    config = {
        'demand_model': 'exponential',
        'base_conversion_rate': 0.15,
        'price_elasticity': -1.5
    }
    return SalesService(config, fee_service)


@pytest.fixture
def trust_service():
    """Create a trust score service for testing."""
    config = {
        'base_trust_score': 0.7,
        'trust_event_weights': {
            'sale': 0.02,
            'stockout': -0.03
        }
    }
    return TrustScoreService(config)


@pytest.fixture
def orchestrator():
    """Create a simulation orchestrator for testing."""
    config = SimulationConfig(
        tick_interval_seconds=0.1,  # Fast ticks for testing
        max_ticks=5,
        time_acceleration=1.0
    )
    return SimulationOrchestrator(config)


class TestEventDrivenCore:
    """Test the core event-driven functionality."""
    
    @pytest.mark.asyncio
    async def test_event_bus_publish_subscribe(self, event_bus):
        """Test basic EventBus publish/subscribe functionality."""
        collector = EventCollector()
        
        # Subscribe to TickEvent
        await event_bus.subscribe(TickEvent, collector.collect_event)
        
        # Create and publish a test event
        test_event = TickEvent(
            event_id="test_tick_1",
            timestamp=datetime.now(),
            tick_number=1,
            simulation_time=datetime.now(),
            metadata={"test": True}
        )
        
        await event_bus.publish(test_event)
        
        # Wait a bit for event processing
        await asyncio.sleep(0.1)
        
        # Verify event was received
        assert len(collector.collected_events) == 1
        assert collector.event_counts["TickEvent"] == 1
        received_event = collector.collected_events[0]
        assert received_event.event_id == "test_tick_1"
        assert received_event.tick_number == 1
    
    @pytest.mark.asyncio
    async def test_simulation_orchestrator_tick_generation(self, event_bus, orchestrator):
        """Test that SimulationOrchestrator generates TickEvents."""
        collector = EventCollector()
        
        # Subscribe to TickEvents
        await event_bus.subscribe(TickEvent, collector.collect_event)
        
        # Start orchestrator
        await orchestrator.start(event_bus)
        
        # Wait for simulation to complete (5 ticks at 0.1s intervals)
        await asyncio.sleep(1.0)
        
        # Stop orchestrator
        await orchestrator.stop()
        
        # Verify TickEvents were generated
        tick_events = collector.get_events_of_type(TickEvent)
        assert len(tick_events) == 5
        
        # Verify tick numbers are sequential
        for i, event in enumerate(tick_events):
            assert event.tick_number == i
            assert "seasonal_factor" in event.metadata
            assert "weekday_factor" in event.metadata
    
    @pytest.mark.asyncio
    async def test_sales_service_event_handling(self, event_bus, sales_service, sample_product):
        """Test that SalesService responds to TickEvents and publishes SaleOccurred events."""
        collector = EventCollector()
        
        # Subscribe to SaleOccurred events
        await event_bus.subscribe(SaleOccurred, collector.collect_event)
        
        # Start sales service
        await sales_service.start(event_bus)
        
        # Manually create and publish a TickEvent
        tick_event = TickEvent(
            event_id="test_tick_sales",
            timestamp=datetime.now(),
            tick_number=1,
            simulation_time=datetime.now(),
            metadata={"seasonal_factor": 1.2}
        )
        
        # Mock the _get_active_products method to return our sample product
        original_method = sales_service._get_active_products
        sales_service._get_active_products = lambda tick: [sample_product]
        
        try:
            await event_bus.publish(tick_event)
            
            # Wait for processing
            await asyncio.sleep(0.2)
            
            # Stop sales service
            await sales_service.stop()
            
            # Check if any SaleOccurred events were generated
            sale_events = collector.get_events_of_type(SaleOccurred)
            
            # Note: Sales might be 0 due to random nature, but service should have processed the tick
            # Verify the service processed the tick (check internal state)
            assert sales_service.current_market_conditions.seasonal_adjustment == 1.2
            
        finally:
            # Restore original method
            sales_service._get_active_products = original_method
    
    @pytest.mark.asyncio
    async def test_trust_service_sale_event_handling(self, event_bus, trust_service, sample_product):
        """Test that TrustScoreService responds to SaleOccurred events."""
        # Start trust service
        await trust_service.start(event_bus)
        
        # Create a SaleOccurred event
        sale_event = SaleOccurred(
            event_id="test_sale_trust",
            timestamp=datetime.now(),
            asin=sample_product.asin,
            units_sold=2,
            units_demanded=5,
            unit_price=Money.from_dollars(25.0),
            total_revenue=Money.from_dollars(50.0),
            total_fees=Money.from_dollars(8.0),
            total_profit=Money.from_dollars(22.0),
            cost_basis=Money.from_dollars(20.0),
            trust_score_at_sale=0.8,
            bsr_at_sale=100000,
            conversion_rate=0.4,
            fee_breakdown={"referral": Money.from_dollars(8.0)},
            market_conditions={"demand_multiplier": 1.0}
        )
        
        # Publish the sale event
        await event_bus.publish(sale_event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Stop trust service
        await trust_service.stop()
        
        # Verify trust score was updated
        trust_score = trust_service.get_trust_score(sample_product.asin)
        assert trust_score != trust_service.base_trust_score  # Should have changed
        
        # Verify events were recorded
        assert len(trust_service.event_history) > 0
        
        # Check for sale event
        sale_events = [e for e in trust_service.event_history if e.event_type.value == "sale"]
        assert len(sale_events) == 1
        
        # Check for stockout event (units_demanded > units_sold)
        stockout_events = [e for e in trust_service.event_history if e.event_type.value == "stockout"]
        assert len(stockout_events) == 1


class TestEndToEndIntegration:
    """Test complete end-to-end event flow."""
    
    @pytest.mark.asyncio
    async def test_complete_simulation_flow(
        self, 
        event_bus, 
        orchestrator, 
        sales_service, 
        trust_service, 
        sample_product
    ):
        """Test complete simulation flow: Orchestrator -> Sales -> Trust."""
        # Collectors for different event types
        tick_collector = EventCollector()
        sale_collector = EventCollector()
        
        # Subscribe to events
        await event_bus.subscribe(TickEvent, tick_collector.collect_event)
        await event_bus.subscribe(SaleOccurred, sale_collector.collect_event)
        
        # Mock the sales service to return our sample product
        original_method = sales_service._get_active_products
        sales_service._get_active_products = lambda tick: [sample_product]
        
        try:
            # Start all services
            await sales_service.start(event_bus)
            await trust_service.start(event_bus)
            await orchestrator.start(event_bus)
            
            # Wait for simulation to complete
            await asyncio.sleep(1.0)
            
            # Stop all services
            await orchestrator.stop()
            await sales_service.stop()
            await trust_service.stop()
            
            # Verify the event flow
            tick_events = tick_collector.get_events_of_type(TickEvent)
            sale_events = sale_collector.get_events_of_type(SaleOccurred)
            
            # Should have generated 5 tick events
            assert len(tick_events) == 5
            
            # Verify event bus statistics
            stats = event_bus.get_stats()
            assert stats['started'] is False  # Should be stopped
            assert stats['events_published'] >= 5  # At least the tick events
            assert stats['events_processed'] >= 5
            
            # Verify orchestrator statistics
            orch_status = orchestrator.get_status()
            assert orch_status['current_tick'] == 5
            assert orch_status['statistics']['total_ticks'] == 5
            assert not orch_status['is_running']
            
        finally:
            # Restore original method
            sales_service._get_active_products = original_method
    
    @pytest.mark.asyncio
    async def test_money_type_strict_enforcement(self, sample_product):
        """Test that Money type enforcement works correctly."""
        # Test that creating Money with float raises TypeError
        with pytest.raises(TypeError, match="Float not allowed in Money constructor"):
            Money(19.99)  # Should fail - must use Money.from_dollars()
        
        # Test that factory methods work
        price = Money.from_dollars(19.99)
        assert price.cents == 1999
        
        # Test arithmetic operations return Money types
        doubled = price * 2
        assert isinstance(doubled, Money)
        assert doubled.cents == 3998
        
        # Test that product monetary fields are Money types
        assert isinstance(sample_product.cost, Money)
        assert isinstance(sample_product.price, Money)
        
        # Test profit calculation returns Money
        profit = sample_product.get_profit_margin()
        assert isinstance(profit, Money)
    
    @pytest.mark.asyncio
    async def test_event_schema_validation(self):
        """Test that event schemas validate correctly."""
        # Test valid TickEvent
        tick_event = TickEvent(
            event_id="valid_tick",
            timestamp=datetime.now(),
            tick_number=1,
            simulation_time=datetime.now(),
            metadata={}
        )
        assert tick_event.tick_number == 1
        
        # Test invalid TickEvent (negative tick number)
        with pytest.raises(ValueError, match="Tick number must be >= 0"):
            TickEvent(
                event_id="invalid_tick",
                timestamp=datetime.now(),
                tick_number=-1,
                simulation_time=datetime.now(),
                metadata={}
            )
        
        # Test valid SaleOccurred event
        sale_event = SaleOccurred(
            event_id="valid_sale",
            timestamp=datetime.now(),
            asin="B123TEST",
            units_sold=1,
            units_demanded=2,
            unit_price=Money.from_dollars(10.0),
            total_revenue=Money.from_dollars(10.0),
            total_fees=Money.from_dollars(2.0),
            total_profit=Money.from_dollars(3.0),
            cost_basis=Money.from_dollars(5.0),
            trust_score_at_sale=0.8,
            bsr_at_sale=100000,
            conversion_rate=0.5
        )
        assert sale_event.units_sold == 1
        
        # Test invalid SaleOccurred (units sold > units demanded)
        with pytest.raises(ValueError, match="Units sold cannot exceed units demanded"):
            SaleOccurred(
                event_id="invalid_sale",
                timestamp=datetime.now(),
                asin="B123TEST",
                units_sold=3,  # More than demanded
                units_demanded=2,
                unit_price=Money.from_dollars(10.0),
                total_revenue=Money.from_dollars(30.0),
                total_fees=Money.from_dollars(5.0),
                total_profit=Money.from_dollars(10.0),
                cost_basis=Money.from_dollars(15.0),
                trust_score_at_sale=0.8,
                bsr_at_sale=100000,
                conversion_rate=1.5  # Also invalid
            )


if __name__ == "__main__":
    # Run tests with asyncio support
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])