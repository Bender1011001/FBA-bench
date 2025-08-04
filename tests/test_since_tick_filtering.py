"""
Test script to verify since_tick filtering functionality works correctly.
"""

import asyncio
import sys
import pytest
from datetime import datetime, timezone
from services.dashboard_api_service import DashboardAPIService
from event_bus import AsyncioQueueBackend, EventBus
from events import TickEvent, SaleOccurred, SetPriceCommand
from money import Money


@pytest.mark.asyncio
async def test_since_tick_filtering():
    """Test that since_tick filtering works correctly."""
    print("🧪 Testing since_tick filtering functionality...")
    
    # Setup test environment
    backend = AsyncioQueueBackend()
    event_bus = EventBus(backend)
    dashboard_service = DashboardAPIService(event_bus)
    
    await event_bus.start()
    await dashboard_service.start()
    
    try:
        # Simulate several ticks with events
        for tick in range(1, 6):
            # Publish tick event
            tick_event = TickEvent(
                event_id=f"tick_{tick}",
                timestamp=datetime.now(timezone.utc),
                tick_number=tick,
                simulation_time=datetime.now(timezone.utc)
            )
            await event_bus.publish(tick_event)
            # Wait for tick to be processed
            await asyncio.sleep(0.05)
            
            # Add a sale event for this tick
            sale_event = SaleOccurred(
                event_id=f"sale_{tick}",
                timestamp=datetime.now(timezone.utc),
                asin="B001TEST",
                units_sold=1,
                units_demanded=2,
                unit_price=Money.from_dollars(20.00),
                total_revenue=Money.from_dollars(20.00),
                total_fees=Money.from_dollars(3.00),
                total_profit=Money.from_dollars(12.00),
                cost_basis=Money.from_dollars(5.00),
                trust_score_at_sale=0.9,
                bsr_at_sale=1000,
                conversion_rate=0.5
            )
            await event_bus.publish(sale_event)
            
            # Add a command event for this tick
            command_event = SetPriceCommand(
                event_id=f"cmd_{tick}",
                timestamp=datetime.now(timezone.utc),
                agent_id="test_agent",
                asin="B001TEST",
                new_price=Money.from_dollars(19.00 + tick),
                reason=f"Test command for tick {tick}"
            )
            await event_bus.publish(command_event)
            
            # Wait for events to be processed
            await asyncio.sleep(0.05)
            print(f"   Processed tick {tick} - Current tick in dashboard: {dashboard_service.simulation_state['current_tick']}")
        
        # Test filtering functionality
        print("\n📊 Testing event retrieval with since_tick filtering:")
        
        # Debug: Check dashboard state
        print(f"   Dashboard current tick: {dashboard_service.simulation_state['current_tick']}")
        print(f"   Events processed: {dashboard_service.events_processed_count}")
        
        # Test 1: Get all events (no filtering)
        all_sales = dashboard_service.get_recent_events(event_type='sales', limit=100)
        all_commands = dashboard_service.get_recent_events(event_type='commands', limit=100)
        print(f"   Total sales events: {len(all_sales)}")
        print(f"   Total command events: {len(all_commands)}")
        
        # Debug: Show event details
        if all_sales:
            print("   Sales events tick numbers:", [s.get('tick_number', 'missing') for s in all_sales])
        if all_commands:
            print("   Command events tick numbers:", [c.get('tick_number', 'missing') for c in all_commands])
        
        # Test 2: Filter sales since tick 3
        filtered_sales = dashboard_service.get_recent_events(
            event_type='sales', limit=100, since_tick=3
        )
        print(f"   Sales since tick 3: {len(filtered_sales)} (expected: 3)")
        for sale in filtered_sales:
            print(f"      Tick {sale['tick_number']}: {sale['event_id']}")
        
        # Test 3: Filter commands since tick 4
        filtered_commands = dashboard_service.get_recent_events(
            event_type='commands', limit=100, since_tick=4
        )
        print(f"   Commands since tick 4: {len(filtered_commands)} (expected: 2)")
        for cmd in filtered_commands:
            print(f"      Tick {cmd['tick_number']}: {cmd['event_id']}")
        
        # Test 4: Mixed events since tick 2
        mixed_events = dashboard_service.get_recent_events(
            event_type=None, limit=100, since_tick=2
        )
        print(f"   Mixed events since tick 2: {len(mixed_events)} (expected: 8)")
        
        # Test 5: Edge case - since_tick higher than any tick
        no_events = dashboard_service.get_recent_events(
            event_type='sales', limit=100, since_tick=10
        )
        print(f"   Events since tick 10: {len(no_events)} (expected: 0)")
        
        # Validation
        assert len(filtered_sales) == 3, f"Expected 3 sales since tick 3, got {len(filtered_sales)}"
        assert len(filtered_commands) == 2, f"Expected 2 commands since tick 4, got {len(filtered_commands)}"
        assert len(mixed_events) == 8, f"Expected 8 mixed events since tick 2, got {len(mixed_events)}"
        assert len(no_events) == 0, f"Expected 0 events since tick 10, got {len(no_events)}"
        
        # Verify tick numbers in filtered results
        for sale in filtered_sales:
            assert sale['tick_number'] >= 3, f"Sale has tick {sale['tick_number']}, expected >= 3"
        
        for cmd in filtered_commands:
            assert cmd['tick_number'] >= 4, f"Command has tick {cmd['tick_number']}, expected >= 4"
        
        print("\n✅ All since_tick filtering tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    finally:
        await dashboard_service.stop()
        await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(test_since_tick_filtering())