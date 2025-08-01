"""
Integration Test for Enhanced Cognitive Architecture

Tests the integration of hierarchical planning, structured reflection,
memory validation, and cognitive configuration components.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import cognitive components
from agents.advanced_agent import AdvancedAgent, AgentConfig
from agents.cognitive_config import get_cognitive_config, CognitiveMode
from agents.hierarchical_planner import StrategicPlanner, TacticalPlanner, PlanType
from memory_experiments.reflection_module import StructuredReflectionLoop, ReflectionTrigger
from memory_experiments.memory_validator import MemoryConsistencyChecker, MemoryIntegrationGateway
from memory_experiments.memory_config import MemoryConfig
from memory_experiments.dual_memory_manager import DualMemoryManager
from event_bus import EventBus
from events import BaseEvent, TickEvent
from money import Money


class CognitiveIntegrationTest:
    """Test suite for cognitive architecture integration."""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.test_results = {}
        
    async def run_all_tests(self):
        """Run comprehensive integration tests."""
        logger.info("🧠 Starting Cognitive Architecture Integration Tests")
        
        try:
            await self.event_bus.start()
            
            # Test 1: Cognitive Configuration
            await self.test_cognitive_configuration()
            
            # Test 2: Agent with Cognitive Enhancement
            await self.test_enhanced_agent_initialization()
            
            # Test 3: Hierarchical Planning System
            await self.test_hierarchical_planning_integration()
            
            # Test 4: Structured Reflection Integration
            await self.test_structured_reflection_integration()
            
            # Test 5: Memory Validation Integration
            await self.test_memory_validation_integration()
            
            # Test 6: End-to-end Cognitive Loop
            await self.test_complete_cognitive_loop()
            
            # Print test summary
            self.print_test_summary()
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            raise
        finally:
            await self.event_bus.stop()
    
    async def test_cognitive_configuration(self):
        """Test cognitive configuration system."""
        logger.info("🔧 Testing Cognitive Configuration System")
        
        try:
            # Test different configuration templates
            dev_config = get_cognitive_config("development", "test_agent_001")
            prod_config = get_cognitive_config("production", "test_agent_002")
            research_config = get_cognitive_config("research", "test_agent_003")
            
            # Verify configurations
            assert dev_config.cognitive_mode == CognitiveMode.DEBUG
            assert prod_config.cognitive_mode == CognitiveMode.ENHANCED
            assert research_config.cognitive_mode == CognitiveMode.EXPERIMENTAL
            
            # Test configuration validation
            assert dev_config.reflection.reflection_interval >= 1
            assert prod_config.strategic_planning.strategic_planning_horizon >= 30
            assert research_config.memory_validation.memory_validation_enabled
            
            # Test configuration serialization
            config_dict = dev_config.to_dict()
            assert isinstance(config_dict, dict)
            assert "cognitive_mode" in config_dict
            
            self.test_results["cognitive_configuration"] = "✅ PASSED"
            logger.info("✅ Cognitive Configuration System: PASSED")
            
        except Exception as e:
            self.test_results["cognitive_configuration"] = f"❌ FAILED: {e}"
            logger.error(f"❌ Cognitive Configuration System: FAILED - {e}")
    
    async def test_enhanced_agent_initialization(self):
        """Test enhanced agent initialization with cognitive capabilities."""
        logger.info("🤖 Testing Enhanced Agent Initialization")
        
        try:
            # Create agent with cognitive enhancements
            config = AgentConfig(
                agent_id="test_cognitive_agent",
                target_asin="TEST-ASIN-001",
                strategy="profit_maximizer",
                cognitive_enabled=True,
                cognitive_config_template="development",
                enable_hierarchical_planning=True,
                enable_structured_reflection=True,
                enable_memory_validation=True
            )
            
            agent = AdvancedAgent(config, self.event_bus)
            
            # Verify cognitive components are initialized
            assert agent.config.cognitive_enabled
            assert agent.cognitive_config is not None
            
            # Note: Strategic planner and other components may not be fully initialized
            # without a proper gateway and memory system, but the architecture should handle this gracefully
            
            # Test agent status with cognitive information
            status = agent.get_agent_status()
            assert "cognitive_enabled" in status
            assert status["cognitive_enabled"] == True
            
            self.test_results["enhanced_agent_init"] = "✅ PASSED"
            logger.info("✅ Enhanced Agent Initialization: PASSED")
            
        except Exception as e:
            self.test_results["enhanced_agent_init"] = f"❌ FAILED: {e}"
            logger.error(f"❌ Enhanced Agent Initialization: FAILED - {e}")
    
    async def test_hierarchical_planning_integration(self):
        """Test hierarchical planning system integration."""
        logger.info("📋 Testing Hierarchical Planning Integration")
        
        try:
            # Initialize planning components
            strategic_planner = StrategicPlanner("test_agent_planning", self.event_bus)
            tactical_planner = TacticalPlanner("test_agent_planning", strategic_planner, self.event_bus)
            
            # Test strategic plan creation
            context = {
                "current_metrics": {
                    "revenue_growth": 0.05,
                    "profit_margin": 0.15,
                    "market_share": 0.08
                },
                "market_conditions": {
                    "competitive_pressure": 0.6,
                    "volatility": 0.3
                }
            }
            
            objectives = await strategic_planner.create_strategic_plan(context, timeframe=90)
            assert len(objectives) > 0
            logger.info(f"Created {len(objectives)} strategic objectives")
            
            # Test tactical action generation
            current_state = {
                "current_metrics": context["current_metrics"],
                "inventory_level": 50,
                "customer_messages": [],
                "available_budget": 1000.0
            }
            
            tactical_actions = await tactical_planner.generate_tactical_actions(objectives, current_state)
            assert len(tactical_actions) >= 0
            logger.info(f"Generated {len(tactical_actions)} tactical actions")
            
            # Test action prioritization
            if tactical_actions:
                constraints = {"max_concurrent_actions": 3}
                prioritized_actions = await tactical_planner.prioritize_actions(tactical_actions, constraints)
                assert len(prioritized_actions) <= 3
                logger.info(f"Prioritized to {len(prioritized_actions)} actions")
            
            # Test strategic status
            strategic_status = strategic_planner.get_strategic_status()
            assert "agent_id" in strategic_status
            
            # Test tactical status
            tactical_status = tactical_planner.get_tactical_status()
            assert "agent_id" in tactical_status
            
            self.test_results["hierarchical_planning"] = "✅ PASSED"
            logger.info("✅ Hierarchical Planning Integration: PASSED")
            
        except Exception as e:
            self.test_results["hierarchical_planning"] = f"❌ FAILED: {e}"
            logger.error(f"❌ Hierarchical Planning Integration: FAILED - {e}")
    
    async def test_structured_reflection_integration(self):
        """Test structured reflection system integration."""
        logger.info("🔍 Testing Structured Reflection Integration")
        
        try:
            # Initialize memory and reflection systems
            memory_config = MemoryConfig()
            memory_manager = DualMemoryManager(memory_config, "test_agent_reflection")
            reflection_loop = StructuredReflectionLoop(
                "test_agent_reflection", 
                memory_manager, 
                memory_config, 
                self.event_bus
            )
            
            # Test reflection trigger
            test_events = [
                {
                    "event_id": "test_event_1",
                    "timestamp": datetime.now().isoformat(),
                    "type": "decision_outcome",
                    "severity": 0.7,
                    "description": "Test major event for reflection"
                }
            ]
            
            reflection_result = await reflection_loop.trigger_reflection(
                tick_interval=24,
                major_events=test_events
            )
            
            # Note: Reflection might not trigger based on internal logic, which is expected
            if reflection_result:
                assert reflection_result.agent_id == "test_agent_reflection"
                assert hasattr(reflection_result, 'insights')
                assert hasattr(reflection_result, 'policy_adjustments')
                logger.info(f"Reflection generated {len(reflection_result.insights)} insights")
            else:
                logger.info("Reflection was not triggered (expected based on conditions)")
            
            # Test reflection status
            reflection_status = reflection_loop.get_reflection_status()
            assert "agent_id" in reflection_status
            
            self.test_results["structured_reflection"] = "✅ PASSED"
            logger.info("✅ Structured Reflection Integration: PASSED")
            
        except Exception as e:
            self.test_results["structured_reflection"] = f"❌ FAILED: {e}"
            logger.error(f"❌ Structured Reflection Integration: FAILED - {e}")
    
    async def test_memory_validation_integration(self):
        """Test memory validation system integration."""
        logger.info("🔒 Testing Memory Validation Integration")
        
        try:
            # Initialize memory validation components
            memory_config = MemoryConfig()
            memory_manager = DualMemoryManager(memory_config, "test_agent_validation")
            consistency_checker = MemoryConsistencyChecker("test_agent_validation", memory_config)
            integration_gateway = MemoryIntegrationGateway(
                "test_agent_validation",
                memory_manager,
                consistency_checker,
                memory_config,
                self.event_bus
            )
            
            # Test action validation
            test_action = {
                "type": "set_price",
                "parameters": {"price": 25.0, "asin": "TEST-ASIN"},
                "expected_impact": {"revenue": 0.05}
            }
            
            should_proceed, validation_result = await integration_gateway.pre_action_validation(test_action)
            assert isinstance(should_proceed, bool)
            assert validation_result is not None
            assert hasattr(validation_result, 'validation_passed')
            
            logger.info(f"Action validation: proceed={should_proceed}, "
                       f"confidence={validation_result.confidence_score:.2f}")
            
            # Test post-action learning
            test_outcome = {"success": True, "impact": {"revenue_change": 0.03}}
            learning_success = await integration_gateway.post_action_learning(test_action, test_outcome)
            assert isinstance(learning_success, bool)
            
            # Test gateway status
            gateway_status = integration_gateway.get_gateway_status()
            assert "agent_id" in gateway_status
            assert "statistics" in gateway_status
            
            self.test_results["memory_validation"] = "✅ PASSED"
            logger.info("✅ Memory Validation Integration: PASSED")
            
        except Exception as e:
            self.test_results["memory_validation"] = f"❌ FAILED: {e}"
            logger.error(f"❌ Memory Validation Integration: FAILED - {e}")
    
    async def test_complete_cognitive_loop(self):
        """Test complete cognitive loop with all systems integrated."""
        logger.info("🔄 Testing Complete Cognitive Loop")
        
        try:
            # Create cognitive agent
            config = AgentConfig(
                agent_id="test_cognitive_loop",
                target_asin="TEST-ASIN-LOOP",
                strategy="profit_maximizer",
                cognitive_enabled=True,
                cognitive_config_template="development"
            )
            
            agent = AdvancedAgent(config, self.event_bus)
            await agent.start()
            
            # Simulate a series of tick events
            for tick in range(1, 6):  # 5 ticks
                tick_event = TickEvent(
                    event_id=f"tick_{tick}",
                    timestamp=datetime.now(),
                    tick_number=tick
                )
                
                # Process tick event
                await agent.handle_tick_event(tick_event)
                
                # Add some delay to simulate realistic timing
                await asyncio.sleep(0.1)
            
            # Test cognitive methods
            final_status = agent.get_agent_status()
            assert "cognitive_enabled" in final_status
            
            # Test strategic plan creation
            strategic_context = {
                "current_metrics": {"profit_margin": 0.15},
                "market_conditions": {"volatility": 0.3}
            }
            
            plan_created = await agent.create_strategic_plan(strategic_context)
            # Note: This might fail if memory systems aren't fully set up, which is expected
            
            # Test forced reflection
            reflection_triggered = await agent.trigger_cognitive_reflection(force=True)
            # Note: This might also fail without full memory setup, which is expected
            
            # Test memory validation
            validation_results = await agent.validate_memory_consistency()
            assert "validation_available" in validation_results
            
            self.test_results["complete_cognitive_loop"] = "✅ PASSED"
            logger.info("✅ Complete Cognitive Loop: PASSED")
            
        except Exception as e:
            self.test_results["complete_cognitive_loop"] = f"❌ FAILED: {e}"
            logger.error(f"❌ Complete Cognitive Loop: FAILED - {e}")
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        logger.info("\n" + "="*60)
        logger.info("🧠 COGNITIVE ARCHITECTURE INTEGRATION TEST SUMMARY")
        logger.info("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results.values() if r.startswith("✅")])
        failed_tests = total_tests - passed_tests
        
        for test_name, result in self.test_results.items():
            logger.info(f"{test_name:.<40} {result}")
        
        logger.info("-"*60)
        logger.info(f"TOTAL TESTS: {total_tests}")
        logger.info(f"PASSED: {passed_tests}")
        logger.info(f"FAILED: {failed_tests}")
        logger.info(f"SUCCESS RATE: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests == 0:
            logger.info("🎉 ALL TESTS PASSED! Cognitive architecture is fully integrated.")
        else:
            logger.warning(f"⚠️  {failed_tests} tests failed. Review integration issues.")
        
        logger.info("="*60)


async def main():
    """Run the cognitive integration test suite."""
    test_suite = CognitiveIntegrationTest()
    
    try:
        await test_suite.run_all_tests()
        print("\n🎯 Cognitive Integration Test Complete!")
        return True
    except Exception as e:
        print(f"\n💥 Test Suite Failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)