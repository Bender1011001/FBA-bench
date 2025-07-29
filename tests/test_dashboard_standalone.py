"""
Standalone Dashboard API Integration Test

Tests the core DashboardAPIService and API functionality without
complex import dependencies that cause relative import issues.
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timezone
from typing import Dict, Any

# Add current directory to path to avoid import issues
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from money import Money
from event_bus import EventBus
from events import TickEvent, SaleOccurred, CompetitorPricesUpdated, SetPriceCommand, ProductPriceUpdated, CompetitorState

# Import DashboardAPIService directly to avoid services.__init__ issues
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'services'))
from dashboard_api_service import DashboardAPIService


async def test_dashboard_core_functionality():
    """Test core dashboard functionality without complex dependencies."""
    print("🧪 FBA-Bench v3 Dashboard API Standalone Test")
    print("=" * 55)
    
    # Initialize components
    event_bus = EventBus()
    dashboard_service = DashboardAPIService(event_bus)
    
    print("✅ EventBus and DashboardAPIService initialized")
    
    # Start EventBus first
    await event_bus.start()
    print("✅ EventBus started")
    
    # Start dashboard service
    await dashboard_service.start()
    print("✅ Dashboard service started")
    
    # Test 1: Basic event processing
    print("\n🧪 Test 1: Basic Event Processing")
    print("-" * 40)
    
    # Create and publish test events
    tick_event = TickEvent(
        event_id="test-tick-1",
        timestamp=datetime.now(timezone.utc),
        tick_number=1,
        simulation_time=datetime.now(timezone.utc)
    )
    await event_bus.publish(tick_event)
    
    sale_event = SaleOccurred(
        event_id="test-sale-1",
        timestamp=datetime.now(timezone.utc),
        asin="B001DASHBOARD",
        units_sold=2,
        units_demanded=3,
        unit_price=Money(2000),  # $20.00
        total_revenue=Money(4000),  # $40.00
        total_fees=Money(600),  # $6.00
        total_profit=Money(2400),  # $24.00
        cost_basis=Money(1000),  # $10.00
        trust_score_at_sale=0.95,
        bsr_at_sale=1200,
        conversion_rate=0.67
    )
    await event_bus.publish(sale_event)
    
    command_event = SetPriceCommand(
        event_id="test-cmd-1",
        timestamp=datetime.now(timezone.utc),
        agent_id="test-agent-dashboard",
        asin="B001DASHBOARD",
        new_price=Money(2100),  # $21.00
        reason="dashboard_test"
    )
    await event_bus.publish(command_event)
    
    # Allow processing time
    await asyncio.sleep(0.2)
    
    print("✅ Published test events (tick, sale, command)")
    
    # Test 2: Snapshot generation
    print("\n🧪 Test 2: Snapshot Generation")
    print("-" * 40)
    
    snapshot = dashboard_service.get_simulation_snapshot()
    
    # Validate snapshot structure
    required_keys = [
        'current_tick', 'simulation_time', 'last_update',
        'products', 'competitors', 'financial_summary',
        'agents', 'command_stats', 'event_stats', 'metadata'
    ]
    
    missing_keys = [key for key in required_keys if key not in snapshot]
    if missing_keys:
        print(f"❌ Missing snapshot keys: {missing_keys}")
        return False
    
    print("✅ Snapshot contains all required keys")
    
    # Validate specific data
    if snapshot['current_tick'] != 1:
        print(f"❌ Expected tick 1, got {snapshot['current_tick']}")
        return False
    
    if snapshot['financial_summary']['total_transactions'] == 0:
        print("❌ No sales transactions recorded")
        return False
    
    if len(snapshot['sales_history']) == 0:
        print("❌ No sales history recorded")
        return False
    
    if len(snapshot['command_history']) == 0:
        print("❌ No command history recorded")
        return False
    
    print("✅ Snapshot data validation passed")
    print(f"   📊 Current tick: {snapshot['current_tick']}")
    print(f"   💰 Total revenue: ${snapshot['financial_summary']['total_revenue']/100:.2f}")
    print(f"   💵 Total profit: ${snapshot['financial_summary']['total_profit']/100:.2f}")
    print(f"   📦 Units sold: {snapshot['financial_summary']['total_units_sold']}")
    print(f"   🤖 Agent commands: {len(snapshot['command_history'])}")
    print(f"   📈 Sales recorded: {len(snapshot['sales_history'])}")
    
    # Test 3: Event filtering
    print("\n🧪 Test 3: Event Filtering")
    print("-" * 40)
    
    sales_events = dashboard_service.get_recent_events(event_type='sales', limit=10)
    command_events = dashboard_service.get_recent_events(event_type='commands', limit=10)
    
    if not sales_events:
        print("❌ No sales events returned")
        return False
    
    if not command_events:
        print("❌ No command events returned")
        return False
    
    print(f"✅ Event filtering working: {len(sales_events)} sales, {len(command_events)} commands")
    
    # Test 4: Real-time metrics
    print("\n🧪 Test 4: Real-time Metrics Calculation")
    print("-" * 40)
    
    # Publish more events to test aggregation
    for i in range(2, 5):
        tick = TickEvent(
            event_id=f"test-tick-{i}",
            timestamp=datetime.now(timezone.utc),
            tick_number=i,
            simulation_time=datetime.now(timezone.utc)
        )
        await event_bus.publish(tick)
        
        sale = SaleOccurred(
            event_id=f"test-sale-{i}",
            timestamp=datetime.now(timezone.utc),
            asin="B001DASHBOARD",
            units_sold=1,
            units_demanded=2,
            unit_price=Money(2100),
            total_revenue=Money(2100),
            total_fees=Money(315),
            total_profit=Money(1285),
            cost_basis=Money(500),
            trust_score_at_sale=0.93,
            bsr_at_sale=1100,
            conversion_rate=0.5
        )
        await event_bus.publish(sale)
    
    await asyncio.sleep(0.2)
    
    # Check updated metrics
    updated_snapshot = dashboard_service.get_simulation_snapshot()
    
    if updated_snapshot['current_tick'] != 4:
        print(f"❌ Expected tick 4, got {updated_snapshot['current_tick']}")
        return False
    
    total_transactions = updated_snapshot['financial_summary']['total_transactions']
    if total_transactions < 2:
        print(f"❌ Expected at least 2 transactions, got {total_transactions}")
        return False
    
    print("✅ Real-time metrics updating correctly")
    print(f"   📊 Final tick: {updated_snapshot['current_tick']}")
    print(f"   📈 Total transactions: {total_transactions}")
    print(f"   💰 Total revenue: ${updated_snapshot['financial_summary']['total_revenue']/100:.2f}")
    print(f"   📊 Events processed: {updated_snapshot['event_stats']['events_processed']}")
    
    # Cleanup
    await dashboard_service.stop()
    await event_bus.stop()
    print("\n✅ Dashboard service and EventBus stopped")
    
    return True


async def test_api_server_basic():
    """Test if API server endpoints are working (if running)."""
    print("\n🧪 Test 5: API Server Endpoints (if available)")
    print("-" * 50)
    
    try:
        import requests
        
        # Test health endpoint
        response = requests.get("http://localhost:8000/api/v1/health", timeout=2)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health endpoint accessible")
            print(f"   🏥 Status: {health_data.get('status')}")
            
            # Test snapshot endpoint
            response = requests.get("http://localhost:8000/api/v1/simulation/snapshot", timeout=2)
            if response.status_code == 200:
                snapshot_data = response.json()
                print("✅ Snapshot endpoint working")
                print(f"   📸 Snapshot keys: {list(snapshot_data.keys())}")
                return True
            else:
                print(f"⚠️  Snapshot endpoint returned {response.status_code}")
        else:
            print(f"⚠️  Health endpoint returned {response.status_code}")
    
    except ImportError:
        print("⚠️  requests module not available - skipping API tests")
    except Exception as e:
        print(f"⚠️  API server not running - {str(e)}")
        print("   💡 To test API: python api_server.py")
    
    return False


async def main():
    """Run the standalone dashboard test."""
    
    # Test core dashboard functionality
    dashboard_success = await test_dashboard_core_functionality()
    
    # Test API server if available
    api_success = await test_api_server_basic()
    
    # Summary
    print(f"\n📋 TEST RESULTS")
    print("=" * 55)
    print(f"Dashboard Core Functionality: {'✅ PASS' if dashboard_success else '❌ FAIL'}")
    print(f"API Server Endpoints: {'✅ PASS' if api_success else '⚠️  SKIP (server not running)'}")
    
    if dashboard_success:
        print("\n🎉 Dashboard API Service: CORE FUNCTIONALITY VERIFIED!")
        print("\n💡 Next steps:")
        print("   1. Run 'python api_server.py' to start the API server")
        print("   2. Open http://localhost:8000/dashboard for live dashboard")
        print("   3. Connect to http://localhost:8000/docs for API documentation")
        print("   4. Integrate with live simulation for real-time data")
        
        if not api_success:
            print("\n🚀 To test the complete research toolkit:")
            print("   Terminal 1: python api_server.py")
            print("   Terminal 2: python test_dashboard_standalone.py")
    else:
        print("\n❌ Dashboard test failed - check error messages above")
    
    return dashboard_success


if __name__ == "__main__":
    asyncio.run(main())