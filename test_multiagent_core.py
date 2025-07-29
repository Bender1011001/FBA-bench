"""
Simplified test for multi-agent command-arbitration-event loop.

Tests the core Phase 5 functionality without complex dependencies.
"""

import asyncio
import uuid
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from money import Money
from events import SetPriceCommand, ProductPriceUpdated
from event_bus import EventBus, AsyncioQueueBackend
from services.world_store import WorldStore


class SimpleAgent:
    """Simplified agent for testing core multi-agent patterns."""
    
    def __init__(self, agent_id: str, event_bus: EventBus):
        self.agent_id = agent_id
        self.event_bus = event_bus
        self.price_updates_received = []
    
    async def start(self):
        """Subscribe to ProductPriceUpdated events."""
        await self.event_bus.subscribe('ProductPriceUpdated', self.handle_price_update)
    
    async def handle_price_update(self, event: ProductPriceUpdated):
        """Handle price update events."""
        self.price_updates_received.append(event)
        print(f"   🤖 Agent {self.agent_id} received price update: {event.asin} = {event.new_price}")
    
    async def send_price_command(self, asin: str, new_price: Money, reason: str = "Agent pricing decision"):
        """Send a SetPriceCommand to the WorldStore."""
        command = SetPriceCommand(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            agent_id=self.agent_id,
            asin=asin,
            new_price=new_price,
            reason=reason
        )
        
        print(f"   🤖 Agent {self.agent_id} sending command: {asin} -> {new_price}")
        await self.event_bus.publish(command)
        return command


async def test_multiagent_core():
    """Test the core multi-agent command-arbitration-event loop."""
    print("🧪 FBA-Bench v3 Phase 5: Core Multi-Agent Loop Test")
    print("=" * 60)
    print()
    
    # Setup
    print("🚀 Setting up test environment...")
    backend = AsyncioQueueBackend()
    event_bus = EventBus(backend)
    await event_bus.start()
    
    # Create WorldStore
    world_store = WorldStore(event_bus)
    await world_store.start()
    
    # Create test agents
    agent1 = SimpleAgent("agent-001", event_bus)
    agent2 = SimpleAgent("agent-002", event_bus)
    
    await agent1.start()
    await agent2.start()
    
    # Initialize test product
    test_asin = "B001TEST"
    initial_price = Money(2000)  # $20.00
    world_store.initialize_product(test_asin, initial_price)
    
    print(f"✅ Environment ready: Product {test_asin} @ {initial_price}")
    print()
    
    try:
        # Test 1: Basic Command Loop
        print("🧪 Test 1: Basic Command-Arbitration-Event Loop")
        
        new_price = Money(2200)  # $22.00
        command1 = await agent1.send_price_command(test_asin, new_price, "Price increase test")
        
        # Wait for event processing
        await asyncio.sleep(0.2)
        
        # Verify WorldStore state
        final_price = world_store.get_product_price(test_asin)
        stats = world_store.get_statistics()
        
        print(f"   🌍 WorldStore result: {initial_price} -> {final_price}")
        print(f"   📊 Commands processed: {stats['commands_processed']}")
        
        # Verify agents received update
        agent1_updates = len(agent1.price_updates_received)
        agent2_updates = len(agent2.price_updates_received)
        
        print(f"   📡 Agent 1 updates received: {agent1_updates}")
        print(f"   📡 Agent 2 updates received: {agent2_updates}")
        
        test1_success = (
            final_price == new_price and
            stats['commands_processed'] >= 1 and
            agent1_updates >= 1 and
            agent2_updates >= 1
        )
        
        print(f"   {'✅ PASS' if test1_success else '❌ FAIL'}: Basic command loop")
        print()
        
        # Test 2: Command Rejection
        print("🧪 Test 2: Command Rejection")
        
        invalid_price = Money(50)  # $0.50 - below minimum
        command2 = await agent2.send_price_command(test_asin, invalid_price, "Invalid price test")
        
        await asyncio.sleep(0.2)
        
        # Price should not change
        rejected_price = world_store.get_product_price(test_asin)
        rejected_stats = world_store.get_statistics()
        
        print(f"   🌍 Price after rejection: {rejected_price}")
        print(f"   📊 Commands rejected: {rejected_stats['commands_rejected']}")
        
        test2_success = (
            rejected_price == final_price and  # Price unchanged
            rejected_stats['commands_rejected'] >= 1
        )
        
        print(f"   {'✅ PASS' if test2_success else '❌ FAIL'}: Command rejection")
        print()
        
        # Test 3: Multiple Agent Competition
        print("🧪 Test 3: Multiple Agent Competition")
        
        # Both agents send commands simultaneously
        price_a = Money(2300)  # $23.00
        price_b = Money(2100)  # $21.00
        
        await agent1.send_price_command(test_asin, price_a, "Agent 1 competing")
        await agent2.send_price_command(test_asin, price_b, "Agent 2 competing")
        
        await asyncio.sleep(0.2)
        
        # Check final state
        competition_price = world_store.get_product_price(test_asin)
        competition_stats = world_store.get_statistics()
        
        print(f"   🌍 Final price after competition: {competition_price}")
        print(f"   📊 Total commands processed: {competition_stats['commands_processed']}")
        
        # At least one command should have been processed
        test3_success = (
            competition_price in [price_a, price_b] and
            competition_stats['commands_processed'] >= 2
        )
        
        print(f"   {'✅ PASS' if test3_success else '❌ FAIL'}: Multiple agent competition")
        print()
        
        # Summary
        print("📋 TEST SUMMARY")
        print("=" * 30)
        all_tests_passed = test1_success and test2_success and test3_success
        
        print(f"Basic Command Loop: {'✅ PASS' if test1_success else '❌ FAIL'}")
        print(f"Command Rejection: {'✅ PASS' if test2_success else '❌ FAIL'}")
        print(f"Agent Competition: {'✅ PASS' if test3_success else '❌ FAIL'}")
        print()
        
        if all_tests_passed:
            print("🎉 ALL TESTS PASSED!")
            print("✅ Multi-agent command-arbitration-event loop is working correctly")
            print("✅ WorldStore arbitration is functional")
            print("✅ Event propagation to all agents is working")
        else:
            print("⚠️  Some tests failed - check implementation")
        
        print()
        print("🔍 Final State:")
        print(f"   Product {test_asin}: {world_store.get_product_price(test_asin)}")
        print(f"   WorldStore stats: {world_store.get_statistics()}")
        
        return all_tests_passed
        
    finally:
        # Cleanup
        await world_store.stop()
        await event_bus.stop()
        print("🧹 Test environment cleaned up")


if __name__ == "__main__":
    result = asyncio.run(test_multiagent_core())
    exit(0 if result else 1)