"""
Test the complete multi-agent command-arbitration-event loop for Phase 5.

Validates the core multi-agent pattern:
1. Agent publishes SetPriceCommand 
2. WorldStore arbitrates and updates canonical state
3. WorldStore publishes ProductPriceUpdated event
4. SalesService receives and processes the canonical price update
"""

import asyncio
import uuid
import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from money import Money
from events import SetPriceCommand, ProductPriceUpdated
from event_bus import EventBus, AsyncioQueueBackend
from services.world_store import WorldStore
from services.sales_service import SalesService
from agents.advanced_agent import AdvancedAgent, AgentConfig


class MultiAgentLoopTest:
    """Test suite for multi-agent command-arbitration-event loop."""
    
    def __init__(self):
        self.event_bus = None
        self.world_store = None
        self.sales_service = None
        self.agent = None
        self.test_results = []
    
    async def setup(self):
        """Set up test environment with all services."""
        print("🚀 Setting up multi-agent test environment...")
        
        # Create event bus
        backend = AsyncioQueueBackend()
        self.event_bus = EventBus(backend)
        await self.event_bus.start()
        
        # Create and start WorldStore
        self.world_store = WorldStore(self.event_bus)
        await self.world_store.start()
        
        # Create and start SalesService (minimal config for testing)
        sales_config = {
            'demand_model': 'exponential',
            'base_conversion_rate': 0.15,
            'price_elasticity': -1.5
        }
        # Note: SalesService needs a fee service, but for this test we'll create a mock
        from services.fee_calculation_service import FeeCalculationService
        fee_service = FeeCalculationService({})
        
        self.sales_service = SalesService(sales_config, fee_service)
        await self.sales_service.start(self.event_bus)
        
        # Create test agent
        agent_config = AgentConfig(
            agent_id="test-agent-001",
            target_asin="B001TEST",
            strategy="profit_maximizer",
            price_sensitivity=0.1,
            reaction_speed=1
        )
        self.agent = AdvancedAgent(agent_config, self.event_bus)
        await self.agent.start()
        
        # Initialize product in WorldStore
        test_asin = "B001TEST"
        initial_price = Money(2000)  # $20.00
        self.world_store.initialize_product(test_asin, initial_price)
        
        print(f"✅ Test environment ready")
        print(f"   📦 Product: {test_asin} @ {initial_price}")
        print(f"   🤖 Agent: {agent_config.agent_id} ({agent_config.strategy})")
        print(f"   🌍 WorldStore: Managing canonical state")
        print(f"   📊 SalesService: Subscribed to price updates")
        print()
    
    async def teardown(self):
        """Clean up test environment."""
        if self.agent:
            await self.agent.stop()
        if self.sales_service:
            await self.sales_service.stop()
        if self.world_store:
            await self.world_store.stop()
        if self.event_bus:
            await self.event_bus.stop()
        print("🧹 Test environment cleaned up")
    
    async def test_basic_command_loop(self):
        """Test basic agent command -> WorldStore arbitration -> service update loop."""
        print("🧪 Test 1: Basic Command-Arbitration-Event Loop")
        
        test_asin = "B001TEST"
        new_price = Money(2200)  # $22.00
        
        # Record initial state
        initial_world_price = self.world_store.get_product_price(test_asin)
        initial_sales_price = self.sales_service.canonical_product_prices.get(test_asin)
        
        print(f"   📊 Initial state:")
        print(f"      WorldStore price: {initial_world_price}")
        print(f"      SalesService price: {initial_sales_price}")
        
        # Step 1: Agent publishes SetPriceCommand
        command = SetPriceCommand(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            agent_id=self.agent.config.agent_id,
            asin=test_asin,
            new_price=new_price,
            reason="Test price increase"
        )
        
        print(f"   🤖 Agent publishing SetPriceCommand: {initial_world_price} -> {new_price}")
        await self.event_bus.publish(command)
        
        # Wait for event processing
        await asyncio.sleep(0.1)
        
        # Step 2: Verify WorldStore updated canonical state
        final_world_price = self.world_store.get_product_price(test_asin)
        world_stats = self.world_store.get_statistics()
        
        print(f"   🌍 WorldStore processed command:")
        print(f"      New canonical price: {final_world_price}")
        print(f"      Commands processed: {world_stats['commands_processed']}")
        print(f"      Commands rejected: {world_stats['commands_rejected']}")
        
        # Step 3: Verify SalesService received ProductPriceUpdated
        final_sales_price = self.sales_service.canonical_product_prices.get(test_asin)
        price_history = self.sales_service.product_price_history.get(test_asin, [])
        
        print(f"   📊 SalesService updated from event:")
        print(f"      Canonical price: {final_sales_price}")
        print(f"      Price history entries: {len(price_history)}")
        
        # Validate results
        success = True
        errors = []
        
        if final_world_price != new_price:
            success = False
            errors.append(f"WorldStore price mismatch: expected {new_price}, got {final_world_price}")
        
        if final_sales_price != new_price:
            success = False
            errors.append(f"SalesService price mismatch: expected {new_price}, got {final_sales_price}")
        
        if world_stats['commands_processed'] != 1:
            success = False
            errors.append(f"WorldStore should have processed 1 command, got {world_stats['commands_processed']}")
        
        if len(price_history) == 0:
            success = False
            errors.append("SalesService should have price history entries")
        
        result = {
            'test': 'basic_command_loop',
            'success': success,
            'errors': errors,
            'initial_price': str(initial_world_price),
            'final_price': str(final_world_price),
            'commands_processed': world_stats['commands_processed']
        }
        self.test_results.append(result)
        
        if success:
            print(f"   ✅ PASS: Command-arbitration-event loop working correctly")
        else:
            print(f"   ❌ FAIL: {'; '.join(errors)}")
        
        print()
        return success
    
    async def test_command_rejection(self):
        """Test WorldStore command rejection and proper error handling."""
        print("🧪 Test 2: Command Rejection and Arbitration")
        
        test_asin = "B001TEST"
        invalid_price = Money(50)  # $0.50 - below minimum threshold
        
        initial_stats = self.world_store.get_statistics()
        initial_price = self.world_store.get_product_price(test_asin)
        
        print(f"   📊 Testing command rejection:")
        print(f"      Current price: {initial_price}")
        print(f"      Invalid price: {invalid_price} (below minimum)")
        
        # Publish invalid command
        command = SetPriceCommand(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            agent_id=self.agent.config.agent_id,
            asin=test_asin,
            new_price=invalid_price,
            reason="Test invalid price"
        )
        
        await self.event_bus.publish(command)
        await asyncio.sleep(0.1)
        
        # Verify command was rejected
        final_stats = self.world_store.get_statistics()
        final_price = self.world_store.get_product_price(test_asin)
        
        commands_rejected = final_stats['commands_rejected'] - initial_stats['commands_rejected']
        
        print(f"   🌍 WorldStore arbitration result:")
        print(f"      Price unchanged: {final_price}")
        print(f"      Commands rejected: {commands_rejected}")
        
        success = True
        errors = []
        
        if final_price != initial_price:
            success = False
            errors.append(f"Price should not have changed: {initial_price} -> {final_price}")
        
        if commands_rejected != 1:
            success = False
            errors.append(f"Should have rejected 1 command, got {commands_rejected}")
        
        result = {
            'test': 'command_rejection',
            'success': success,
            'errors': errors,
            'commands_rejected': commands_rejected,
            'price_unchanged': str(final_price)
        }
        self.test_results.append(result)
        
        if success:
            print(f"   ✅ PASS: Command arbitration and rejection working correctly")
        else:
            print(f"   ❌ FAIL: {'; '.join(errors)}")
        
        print()
        return success
    
    async def test_multiple_agents(self):
        """Test multiple agents competing for the same resource."""
        print("🧪 Test 3: Multiple Agent Competition")
        
        test_asin = "B001TEST"
        
        # Create second agent
        agent2_config = AgentConfig(
            agent_id="test-agent-002",
            target_asin=test_asin,
            strategy="aggressive_pricer",
            price_sensitivity=0.2,
            reaction_speed=1
        )
        agent2 = AdvancedAgent(agent2_config, self.event_bus)
        await agent2.start()
        
        try:
            initial_price = self.world_store.get_product_price(test_asin)
            print(f"   📊 Initial price: {initial_price}")
            print(f"   🤖 Agent 1: {self.agent.config.agent_id} ({self.agent.config.strategy})")
            print(f"   🤖 Agent 2: {agent2.config.agent_id} ({agent2.config.strategy})")
            
            # Both agents submit commands simultaneously
            command1 = SetPriceCommand(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                agent_id=self.agent.config.agent_id,
                asin=test_asin,
                new_price=Money(2100),  # $21.00
                reason="Agent 1 price increase"
            )
            
            command2 = SetPriceCommand(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                agent_id=agent2.config.agent_id,
                asin=test_asin,
                new_price=Money(1900),  # $19.00
                reason="Agent 2 aggressive undercut"
            )
            
            # Submit both commands
            await self.event_bus.publish(command1)
            await self.event_bus.publish(command2)
            await asyncio.sleep(0.1)
            
            # Check final state
            final_price = self.world_store.get_product_price(test_asin)
            final_stats = self.world_store.get_statistics()
            
            print(f"   🌍 WorldStore arbitration result:")
            print(f"      Final price: {final_price}")
            print(f"      Total commands processed: {final_stats['commands_processed']}")
            
            # At least one command should be processed
            success = final_price != initial_price
            
            result = {
                'test': 'multiple_agents',
                'success': success,
                'initial_price': str(initial_price),
                'final_price': str(final_price),
                'commands_processed': final_stats['commands_processed']
            }
            self.test_results.append(result)
            
            if success:
                print(f"   ✅ PASS: Multiple agent competition handled correctly")
            else:
                print(f"   ❌ FAIL: No price change detected from agent competition")
        
        finally:
            await agent2.stop()
        
        print()
        return success
    
    def print_summary(self):
        """Print test summary and results."""
        print("📋 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print()
        
        for result in self.test_results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{status} {result['test']}")
            if result['errors']:
                for error in result['errors']:
                    print(f"   • {error}")
        
        print()
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Multi-agent infrastructure is working correctly.")
        else:
            print("⚠️  Some tests failed. Check implementation for issues.")
        
        return passed_tests == total_tests


async def main():
    """Run the complete multi-agent loop test suite."""
    print("🧪 FBA-Bench v3 Phase 5: Multi-Agent Command Loop Tests")
    print("=" * 60)
    print()
    
    test_suite = MultiAgentLoopTest()
    
    try:
        await test_suite.setup()
        
        # Run all tests
        test1_result = await test_suite.test_basic_command_loop()
        test2_result = await test_suite.test_command_rejection()
        test3_result = await test_suite.test_multiple_agents()
        
        # Print summary
        all_passed = test_suite.print_summary()
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await test_suite.teardown()


if __name__ == "__main__":
    # Run the test suite
    result = asyncio.run(main())
    exit(0 if result else 1)