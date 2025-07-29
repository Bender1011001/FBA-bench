"""
End-to-end integration test for FBA-Bench v3 Research Toolkit.

Tests the complete data pipeline from simulation events to dashboard display:
1. EventBus + DashboardAPIService data aggregation
2. FastAPI REST endpoints
3. WebSocket real-time streaming  
4. Complete simulation state snapshot

This verifies that Phase 6 research infrastructure is fully operational.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any

import requests
import websockets
from money import Money

from event_bus import EventBus
from events import TickEvent, SaleOccurred, CompetitorPricesUpdated, SetPriceCommand, ProductPriceUpdated, CompetitorState
from services.dashboard_api_service import DashboardAPIService


class ResearchToolkitIntegrationTest:
    """Comprehensive integration test for the research toolkit."""
    
    def __init__(self):
        self.event_bus = None
        self.dashboard_service = None
        self.test_results = {
            'dashboard_aggregation': False,
            'rest_api_snapshot': False,
            'rest_api_events': False,
            'websocket_connection': False,
            'event_flow_end_to_end': False,
        }
        
    async def run_full_test(self) -> Dict[str, Any]:
        """Run complete end-to-end test suite."""
        print("🧪 FBA-Bench v3 Research Toolkit Integration Test")
        print("=" * 60)
        
        try:
            # Test 1: EventBus + DashboardAPIService integration
            await self._test_dashboard_aggregation()
            
            # Test 2: API server endpoints (requires manual server start)
            await self._test_api_endpoints_if_available()
            
            # Test 3: WebSocket streaming (requires manual server start)
            await self._test_websocket_if_available()
            
            # Test 4: Complete event flow simulation
            await self._test_complete_event_flow()
            
            # Generate test report
            return self._generate_test_report()
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return {'status': 'failed', 'error': str(e), 'results': self.test_results}
        finally:
            await self._cleanup()
    
    async def _test_dashboard_aggregation(self):
        """Test EventBus to DashboardAPIService data aggregation."""
        print("\n🧪 Test 1: Dashboard Service Data Aggregation")
        print("-" * 50)
        
        # Setup EventBus and DashboardAPIService
        self.event_bus = EventBus()
        self.dashboard_service = DashboardAPIService(self.event_bus)
        await self.dashboard_service.start()
        
        print("✅ EventBus and DashboardAPIService initialized")
        
        # Simulate events and verify aggregation
        test_events = await self._create_test_events()
        
        for event in test_events:
            await self.event_bus.publish(event)
            await asyncio.sleep(0.1)  # Allow processing time
        
        print(f"✅ Published {len(test_events)} test events")
        
        # Verify dashboard state
        snapshot = self.dashboard_service.get_simulation_snapshot()
        
        # Validate snapshot structure
        required_keys = [
            'current_tick', 'products', 'competitors', 'financial_summary',
            'agents', 'command_stats', 'event_stats'
        ]
        
        for key in required_keys:
            if key not in snapshot:
                raise AssertionError(f"Missing required key in snapshot: {key}")
        
        print("✅ Snapshot contains all required data structures")
        
        # Validate specific data
        if snapshot['current_tick'] != 5:
            raise AssertionError(f"Expected tick 5, got {snapshot['current_tick']}")
        
        if len(snapshot['products']) == 0:
            raise AssertionError("No products found in snapshot")
        
        if snapshot['financial_summary']['total_transactions'] == 0:
            raise AssertionError("No sales transactions recorded")
        
        print("✅ Dashboard aggregation data validation passed")
        print(f"   📊 Current tick: {snapshot['current_tick']}")
        print(f"   💰 Total revenue: ${snapshot['financial_summary']['total_revenue']/100:.2f}")
        print(f"   📦 Products tracked: {len(snapshot['products'])}")
        print(f"   🤖 Agents active: {len(snapshot['agents'])}")
        
        self.test_results['dashboard_aggregation'] = True
    
    async def _test_api_endpoints_if_available(self):
        """Test REST API endpoints if server is running."""
        print("\n🧪 Test 2: REST API Endpoints")
        print("-" * 50)
        
        try:
            # Test health endpoint
            response = requests.get("http://localhost:8000/api/v1/health", timeout=2)
            if response.status_code == 200:
                print("✅ Health endpoint accessible")
                
                # Test snapshot endpoint
                response = requests.get("http://localhost:8000/api/v1/simulation/snapshot", timeout=2)
                if response.status_code == 200:
                    snapshot_data = response.json()
                    print("✅ Snapshot endpoint working")
                    print(f"   📸 Snapshot keys: {list(snapshot_data.keys())}")
                    self.test_results['rest_api_snapshot'] = True
                
                # Test events endpoint
                response = requests.get("http://localhost:8000/api/v1/simulation/events", timeout=2)
                if response.status_code == 200:
                    events_data = response.json()
                    print("✅ Events endpoint working")
                    print(f"   📝 Events returned: {len(events_data.get('events', []))}")
                    self.test_results['rest_api_events'] = True
            
        except requests.exceptions.RequestException:
            print("⚠️  API server not running - skipping REST API tests")
            print("   💡 To test APIs, run: python api_server.py")
    
    async def _test_websocket_if_available(self):
        """Test WebSocket connection if server is running."""
        print("\n🧪 Test 3: WebSocket Real-Time Streaming")
        print("-" * 50)
        
        try:
            uri = "ws://localhost:8000/ws/events"
            async with websockets.connect(uri, timeout=2) as websocket:
                print("✅ WebSocket connection established")
                
                # Send ping
                await websocket.send(json.dumps({"type": "ping"}))
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=1)
                message = json.loads(response)
                
                if message.get("type") == "pong":
                    print("✅ WebSocket ping/pong working")
                elif message.get("type") == "snapshot":
                    print("✅ WebSocket initial snapshot received")
                
                self.test_results['websocket_connection'] = True
                
        except (websockets.exceptions.ConnectionRefused, asyncio.TimeoutError):
            print("⚠️  WebSocket server not available - skipping WebSocket tests")
            print("   💡 To test WebSocket, run: python api_server.py")
        except Exception as e:
            print(f"⚠️  WebSocket test error: {e}")
    
    async def _test_complete_event_flow(self):
        """Test complete event flow from simulation to dashboard."""
        print("\n🧪 Test 4: Complete Event Flow Simulation")
        print("-" * 50)
        
        if not self.dashboard_service:
            print("❌ Dashboard service not available for event flow test")
            return
        
        # Simulate a complete business scenario
        print("📈 Simulating business scenario...")
        
        # Agent makes pricing decision
        set_price_cmd = SetPriceCommand(
            event_id="cmd-flow-test-001",
            timestamp=datetime.now(timezone.utc),
            agent_id="flow-test-agent",
            asin="B001FLOW",
            new_price=Money(2500),  # $25.00
            reason="flow_test_price_optimization"
        )
        await self.event_bus.publish(set_price_cmd)
        
        # WorldStore processes command and updates price
        price_update = ProductPriceUpdated(
            event_id="price-flow-test-001",
            timestamp=datetime.now(timezone.utc),
            asin="B001FLOW",
            new_price=Money(2500),
            previous_price=Money(2000),
            agent_id="flow-test-agent",
            command_id="cmd-flow-test-001"
        )
        await self.event_bus.publish(price_update)
        
        # Sale occurs at new price
        sale = SaleOccurred(
            event_id="sale-flow-test-001",
            timestamp=datetime.now(timezone.utc),
            asin="B001FLOW",
            units_sold=2,
            units_demanded=3,
            unit_price=Money(2500),
            total_revenue=Money(5000),
            total_fees=Money(750),
            total_profit=Money(3250),
            cost_basis=Money(1000),
            trust_score_at_sale=0.95,
            bsr_at_sale=1500,
            conversion_rate=0.67
        )
        await self.event_bus.publish(sale)
        
        # Allow processing time
        await asyncio.sleep(0.2)
        
        # Verify complete data flow
        snapshot = self.dashboard_service.get_simulation_snapshot()
        
        # Check agent tracking
        if "flow-test-agent" not in snapshot['agents']:
            raise AssertionError("Agent activity not tracked in dashboard")
        
        # Check product state
        if "B001FLOW" not in snapshot['products']:
            raise AssertionError("Product price update not tracked")
        
        # Check sales data
        recent_sales = [sale for sale in snapshot['sales_history'] if sale['asin'] == 'B001FLOW']
        if not recent_sales:
            raise AssertionError("Sale not recorded in dashboard")
        
        # Check financial impact
        if snapshot['financial_summary']['total_transactions'] == 0:
            raise AssertionError("Financial summary not updated")
        
        print("✅ Complete event flow verified")
        print(f"   🤖 Agent commands tracked: {len(snapshot['command_history'])}")
        print(f"   💰 Sales recorded: {len(snapshot['sales_history'])}")
        print(f"   📊 Products monitored: {len(snapshot['products'])}")
        
        self.test_results['event_flow_end_to_end'] = True
    
    async def _create_test_events(self):
        """Create test events for simulation."""
        events = []
        
        # Tick events
        for i in range(1, 6):
            events.append(TickEvent(
                event_id=f"tick-{i}",
                timestamp=datetime.now(timezone.utc),
                tick_number=i,
                simulation_time=datetime.now(timezone.utc)
            ))
        
        # Competitor update
        competitors = [
            CompetitorState(asin="COMP001", price=Money(1800), bsr=2000, sales_velocity=5.5),
            CompetitorState(asin="COMP002", price=Money(2200), bsr=1500, sales_velocity=7.2),
        ]
        events.append(CompetitorPricesUpdated(
            event_id="comp-update-1",
            timestamp=datetime.now(timezone.utc),
            tick_number=3,
            competitors=competitors
        ))
        
        # Agent command
        events.append(SetPriceCommand(
            event_id="cmd-001",
            timestamp=datetime.now(timezone.utc),
            agent_id="test-agent-001",
            asin="B001TEST",
            new_price=Money(2000),
            reason="test_pricing"
        ))
        
        # Price update
        events.append(ProductPriceUpdated(
            event_id="price-001",
            timestamp=datetime.now(timezone.utc),
            asin="B001TEST",
            new_price=Money(2000),
            previous_price=Money(1800),
            agent_id="test-agent-001",
            command_id="cmd-001"
        ))
        
        # Sales
        events.append(SaleOccurred(
            event_id="sale-001",
            timestamp=datetime.now(timezone.utc),
            asin="B001TEST",
            units_sold=3,
            units_demanded=5,
            unit_price=Money(2000),
            total_revenue=Money(6000),
            total_fees=Money(900),
            total_profit=Money(3600),
            cost_basis=Money(1500),
            trust_score_at_sale=0.92,
            bsr_at_sale=1200,
            conversion_rate=0.6
        ))
        
        return events
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        passed_tests = sum(1 for result in self.test_results.values() if result)
        total_tests = len(self.test_results)
        
        print(f"\n📋 TEST REPORT")
        print("=" * 60)
        
        for test_name, passed in self.test_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED - Research Toolkit is fully operational!")
            status = "success"
        else:
            print("⚠️  Some tests failed - check API server availability")
            status = "partial"
        
        return {
            'status': status,
            'tests_passed': passed_tests,
            'tests_total': total_tests,
            'results': dict(self.test_results),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _cleanup(self):
        """Clean up test resources."""
        if self.dashboard_service:
            await self.dashboard_service.stop()
        print("\n🧹 Test cleanup completed")


async def main():
    """Run the integration test."""
    test = ResearchToolkitIntegrationTest()
    result = await test.run_full_test()
    
    if result['status'] == 'success':
        print("\n🚀 FBA-Bench v3 Research Toolkit: READY FOR PRODUCTION!")
        print("\n💡 To start the full system:")
        print("   1. python api_server.py (starts API server)")
        print("   2. Open http://localhost:8000/dashboard (view dashboard)")
        print("   3. Connect simulation to EventBus (for live data)")
    else:
        print(f"\n⚠️  Test result: {result['status']}")
        if 'error' in result:
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())