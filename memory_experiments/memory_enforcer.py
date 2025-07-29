"""
Memory Enforcer

Integrates memory constraints with FBA-Bench's existing AgentGateway system,
enforcing memory limits and injecting memory context into agent prompts.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List

from .memory_config import MemoryConfig
from .dual_memory_manager import DualMemoryManager
from .reflection_module import ReflectionModule
from constraints.token_counter import TokenCounter
from event_bus import EventBus


logger = logging.getLogger(__name__)


class MemoryEnforcer:
    """
    Enforces memory constraints and manages memory injection into agent prompts.
    
    Integrates with existing AgentGateway to provide memory-aware constraint
    enforcement alongside token budget management.
    """
    
    def __init__(self, config: MemoryConfig, agent_id: str, event_bus: EventBus):
        self.config = config
        self.agent_id = agent_id
        self.event_bus = event_bus
        
        # Initialize dual memory system
        self.memory_manager = DualMemoryManager(config, agent_id)
        
        # Initialize reflection module if enabled
        self.reflection_module = None
        if config.should_use_reflection():
            self.reflection_module = ReflectionModule(self.memory_manager, config)
        
        # Memory usage tracking
        self.memory_tokens_used = 0
        self.total_memory_retrievals = 0
        self.memory_injection_history: List[Dict[str, Any]] = []
        
        # Token counter for memory content
        self.token_counter = TokenCounter()
        
        logger.info(f"MemoryEnforcer initialized for agent {agent_id} with mode {config.memory_mode}")
    
    async def preprocess_memory_for_prompt(self, prompt: str, action_type: str) -> Dict[str, Any]:
        """
        Retrieve relevant memories and prepare them for injection into agent prompt.
        
        Args:
            prompt: Original agent prompt
            action_type: Type of action being performed
            
        Returns:
            Dict containing memory content and metadata for prompt injection
        """
        if not self.config.enable_memory_injection or not self.config.is_memory_enabled():
            return {
                "memory_content": "",
                "memory_tokens": 0,
                "retrieved_memories": 0,
                "memory_domains": [],
                "within_budget": True
            }
        
        # Extract query from prompt for memory retrieval
        memory_query = self._extract_memory_query(prompt, action_type)
        
        # Retrieve relevant memories
        retrieved_memories = await self.memory_manager.retrieve_memories(
            query=memory_query,
            max_memories=self.config.max_retrieval_events
        )
        
        if not retrieved_memories:
            return {
                "memory_content": "",
                "memory_tokens": 0,
                "retrieved_memories": 0,
                "memory_domains": [],
                "within_budget": True
            }
        
        # Format memories for prompt injection
        memory_content = self._format_memories_for_prompt(retrieved_memories)
        
        # Count tokens in memory content
        memory_tokens = self.token_counter.count_tokens(memory_content)
        
        # Check if memory content fits within budget
        within_budget = memory_tokens <= self.config.memory_budget_tokens
        
        if not within_budget:
            # Truncate memory content to fit budget
            memory_content = self._truncate_memory_content(memory_content, self.config.memory_budget_tokens)
            memory_tokens = self.token_counter.count_tokens(memory_content)
            logger.warning(f"Memory content truncated to fit budget: {memory_tokens} tokens")
        
        # Track memory usage
        self.memory_tokens_used = memory_tokens
        self.total_memory_retrievals += 1
        
        # Record memory injection for analysis
        injection_record = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "query": memory_query,
            "retrieved_memories": len(retrieved_memories),
            "memory_tokens": memory_tokens,
            "memory_domains": list(set(m.domain for m in retrieved_memories)),
            "within_budget": within_budget
        }
        self.memory_injection_history.append(injection_record)
        
        # Keep only recent injection history
        if len(self.memory_injection_history) > 100:
            self.memory_injection_history = self.memory_injection_history[-100:]
        
        return {
            "memory_content": memory_content,
            "memory_tokens": memory_tokens,
            "retrieved_memories": len(retrieved_memories),
            "memory_domains": list(set(m.domain for m in retrieved_memories)),
            "within_budget": within_budget
        }
    
    async def inject_memory_into_prompt(self, prompt: str, memory_data: Dict[str, Any]) -> str:
        """
        Inject memory content into agent prompt.
        
        Args:
            prompt: Original prompt
            memory_data: Memory data from preprocess_memory_for_prompt
            
        Returns:
            Enhanced prompt with memory content
        """
        if not memory_data["memory_content"]:
            return prompt
        
        memory_section = f"""

RELEVANT MEMORIES ({memory_data['retrieved_memories']} entries, {memory_data['memory_tokens']} tokens):
{memory_data['memory_content']}

MEMORY GUIDANCE: Consider the above memories when making your decision. Recent experiences and patterns may inform your strategy.
"""
        
        # Insert memory section before the main instruction
        enhanced_prompt = f"{prompt}{memory_section}"
        
        return enhanced_prompt
    
    async def store_event_in_memory(self, event, domain: str = "general") -> bool:
        """Store an event in the agent's memory system."""
        if not self.config.is_memory_enabled():
            return True
        
        return await self.memory_manager.store_event(event, domain)
    
    async def check_reflection_trigger(self, current_time: Optional[datetime] = None) -> bool:
        """Check if reflection should be triggered and perform it if needed."""
        if not self.reflection_module:
            return False
        
        current_time = current_time or datetime.now()
        
        if await self.memory_manager.should_reflect(current_time):
            logger.info(f"Triggering reflection for agent {self.agent_id}")
            
            try:
                result = await self.reflection_module.perform_reflection(current_time)
                
                # Publish reflection event
                await self.event_bus.publish("MemoryReflectionCompleted", {
                    "agent_id": self.agent_id,
                    "timestamp": current_time.isoformat(),
                    "memories_promoted": result.memories_promoted,
                    "memories_discarded": result.memories_discarded,
                    "algorithm_used": result.algorithm_used,
                    "quality_metrics": result.quality_metrics
                })
                
                return True
                
            except Exception as e:
                logger.error(f"Reflection failed for agent {self.agent_id}: {e}")
                return False
        
        return False
    
    def get_memory_budget_status(self) -> str:
        """Get memory budget status for prompt injection (similar to token budget)."""
        if not self.config.track_memory_usage:
            return ""
        
        memory_summary = asyncio.create_task(self.memory_manager.get_memory_summary())
        # Note: This is synchronous, so we'll need to handle async properly in real usage
        
        status = f"""
MEMORY STATUS:
- Memory Mode: {self.config.memory_mode.value}
- Memory tokens used: {self.memory_tokens_used} / {self.config.memory_budget_tokens}
- Total retrievals: {self.total_memory_retrievals}
- Reflection enabled: {self.config.reflection_enabled}
"""
        
        if self.config.memory_mode.value in ["reflection_enabled", "hybrid_reflection"]:
            last_reflection = self.memory_manager.last_reflection_time
            if last_reflection:
                status += f"- Last reflection: {last_reflection.strftime('%Y-%m-%d %H:%M')}\n"
            else:
                status += "- Last reflection: Never\n"
        
        return status
    
    async def validate_memory_constraints(self, estimated_tokens: int) -> tuple[bool, str]:
        """
        Validate if action can proceed given memory constraints.
        
        Args:
            estimated_tokens: Estimated tokens for the action
            
        Returns:
            Tuple of (can_proceed, message)
        """
        if not self.config.is_memory_enabled():
            return True, ""
        
        # Check if adding memory content would exceed effective token budget
        effective_budget = self.config.get_effective_token_budget()
        total_tokens = estimated_tokens + self.memory_tokens_used
        
        if total_tokens > effective_budget:
            return False, f"Action would exceed effective token budget: {total_tokens} > {effective_budget}"
        
        # Check memory store capacity constraints
        short_term_size = await self.memory_manager.short_term_store.size()
        if short_term_size >= self.config.short_term_capacity:
            logger.warning(f"Short-term memory at capacity: {short_term_size}")
        
        long_term_size = await self.memory_manager.long_term_store.size()
        if long_term_size >= self.config.long_term_capacity:
            logger.warning(f"Long-term memory at capacity: {long_term_size}")
        
        return True, ""
    
    def _extract_memory_query(self, prompt: str, action_type: str) -> str:
        """Extract a query for memory retrieval from the agent prompt."""
        # Simple extraction - in production this could be more sophisticated
        
        # Look for key business terms in the prompt
        business_keywords = [
            "price", "pricing", "competitor", "sales", "profit", "inventory",
            "demand", "market", "strategy", "budget", "cost", "revenue"
        ]
        
        prompt_lower = prompt.lower()
        found_keywords = [keyword for keyword in business_keywords if keyword in prompt_lower]
        
        if found_keywords:
            # Use found keywords as query
            return " ".join(found_keywords)
        
        # Fallback to action type
        return action_type
    
    def _format_memories_for_prompt(self, memories: List) -> str:
        """Format retrieved memories for inclusion in agent prompt."""
        if not memories:
            return ""
        
        formatted_memories = []
        
        for i, memory in enumerate(memories, 1):
            # Format each memory with timestamp and content
            age_hours = (datetime.now() - memory.timestamp).total_seconds() / 3600
            
            if age_hours < 24:
                time_desc = f"{age_hours:.1f} hours ago"
            else:
                time_desc = f"{age_hours/24:.1f} days ago"
            
            formatted_memory = f"{i}. [{memory.domain.upper()}] {time_desc}: {memory.content[:200]}..."
            formatted_memories.append(formatted_memory)
        
        return "\n".join(formatted_memories)
    
    def _truncate_memory_content(self, memory_content: str, max_tokens: int) -> str:
        """Truncate memory content to fit within token budget."""
        words = memory_content.split()
        
        # Rough approximation: 1 token ≈ 0.75 words
        max_words = int(max_tokens * 0.75)
        
        if len(words) <= max_words:
            return memory_content
        
        truncated_words = words[:max_words]
        truncated_content = " ".join(truncated_words)
        truncated_content += "\n[Memory content truncated to fit token budget]"
        
        return truncated_content
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics."""
        memory_summary = await self.memory_manager.get_memory_summary()
        
        stats = {
            "memory_config": {
                "mode": self.config.memory_mode.value,
                "short_term_capacity": self.config.short_term_capacity,
                "long_term_capacity": self.config.long_term_capacity,
                "reflection_enabled": self.config.reflection_enabled,
                "memory_budget_tokens": self.config.memory_budget_tokens
            },
            "memory_usage": {
                "current_memory_tokens": self.memory_tokens_used,
                "total_retrievals": self.total_memory_retrievals,
                "injection_history_size": len(self.memory_injection_history)
            },
            "memory_stores": memory_summary
        }
        
        # Add reflection statistics if available
        if self.reflection_module:
            reflection_stats = self.reflection_module.get_reflection_statistics()
            stats["reflection"] = reflection_stats
        
        # Add recent injection history
        if self.memory_injection_history:
            stats["recent_injections"] = self.memory_injection_history[-5:]  # Last 5
        
        return stats
    
    async def clear_memory_history(self):
        """Clear memory history for fresh experiments."""
        await self.memory_manager.clear_memories("all")
        self.memory_injection_history.clear()
        self.memory_tokens_used = 0
        self.total_memory_retrievals = 0
        
        if self.reflection_module:
            self.reflection_module.clear_reflection_history()
        
        logger.info(f"Memory history cleared for agent {self.agent_id}")
    
    def update_tick(self, tick: int):
        """Update current tick for memory system."""
        self.memory_manager.update_tick(tick)