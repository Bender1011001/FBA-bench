"""
Framework-specific configuration templates and examples.

Provides pre-built configurations for different agent frameworks
with sensible defaults and common use cases.
"""

from typing import Dict, Any
from .config_schema import AgentRunnerConfig, LLMConfig, MemoryConfig, AgentConfig, CrewConfig


class FrameworkConfig:
    """Simple configuration to specify a framework type."""
    def __init__(self, framework_type: str):
        self.framework_type = framework_type


class DIYConfig:
    """Pre-built configurations for DIY agent runner."""
    
    @staticmethod
    def advanced_agent(agent_id: str, target_asin: str = "B0DEFAULT") -> AgentRunnerConfig:
        """Configuration for event-driven AdvancedAgent."""
        return AgentRunnerConfig(
            agent_id=agent_id,
            framework="diy",
            agent_config=AgentConfig(
                agent_type="advanced",
                target_asin=target_asin,
                strategy="profit_maximizer",
                price_sensitivity=0.1,
                reaction_speed=1
            ),
            verbose=False
        )
    
    @staticmethod
    def baseline_greedy(agent_id: str) -> AgentRunnerConfig:
        """Configuration for greedy baseline bot."""
        return AgentRunnerConfig(
            agent_id=agent_id,
            framework="diy",
            agent_config=AgentConfig(
                agent_type="baseline",
                strategy="greedy"
            ),
            custom_config={
                "bot_type": "greedy",
                "reorder_threshold": 10,
                "reorder_quantity": 50
            },
            verbose=False
        )
    
    @staticmethod
    def llm_claude(agent_id: str, api_key: str, target_asin: str = "B0DEFAULT") -> AgentRunnerConfig:
        """Configuration for Claude-based LLM agent."""
        return AgentRunnerConfig(
            agent_id=agent_id,
            framework="diy",
            agent_config=AgentConfig(
                agent_type="llm",
                target_asin=target_asin
            ),
            custom_config={
                "llm_type": "claude",
                "model_name": "anthropic/claude-3-sonnet:beta",
                "api_key": api_key
            },
            verbose=False
        )
    
    @staticmethod
    def llm_gpt(agent_id: str, api_key: str, target_asin: str = "B0DEFAULT") -> AgentRunnerConfig:
        """Configuration for GPT-based LLM agent."""
        return AgentRunnerConfig(
            agent_id=agent_id,
            framework="diy",
            agent_config=AgentConfig(
                agent_type="llm",
                target_asin=target_asin
            ),
            custom_config={
                "llm_type": "gpt",
                "model_name": "gpt-4o-mini",
                "api_key": api_key
            },
            verbose=False
        )


class CrewAIConfig:
    """Pre-built configurations for CrewAI agent runner."""
    
    @staticmethod
    def standard_crew(agent_id: str, api_key: str) -> AgentRunnerConfig:
        """Standard 4-agent crew configuration."""
        return AgentRunnerConfig(
            agent_id=agent_id,
            framework="crewai",
            llm_config=LLMConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.1,
                api_key=api_key
            ),
            crew_config=CrewConfig(
                process="sequential",
                crew_size=4,
                roles=["pricing_specialist", "inventory_manager", "market_analyst", "strategy_coordinator"],
                collaboration_mode="sequential",
                allow_delegation=True
            ),
            verbose=False
        )
    
    @staticmethod
    def hierarchical_crew(agent_id: str, api_key: str) -> AgentRunnerConfig:
        """Hierarchical crew with manager coordination."""
        return AgentRunnerConfig(
            agent_id=agent_id,
            framework="crewai",
            llm_config=LLMConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.1,
                api_key=api_key
            ),
            crew_config=CrewConfig(
                process="hierarchical",
                crew_size=4,
                roles=["pricing_specialist", "inventory_manager", "market_analyst", "strategy_coordinator"],
                collaboration_mode="hierarchical",
                allow_delegation=True
            ),
            verbose=True
        )
    
    @staticmethod
    def focused_pricing_crew(agent_id: str, api_key: str) -> AgentRunnerConfig:
        """Specialized crew focused on pricing optimization."""
        return AgentRunnerConfig(
            agent_id=agent_id,
            framework="crewai",
            llm_config=LLMConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.05,  # Lower temperature for more deterministic pricing
                api_key=api_key
            ),
            crew_config=CrewConfig(
                process="sequential",
                crew_size=3,
                roles=["pricing_specialist", "market_analyst", "strategy_coordinator"],
                collaboration_mode="sequential",
                allow_delegation=False  # More focused decision making
            ),
            verbose=False
        )
    
    @staticmethod
    def claude_crew(agent_id: str, api_key: str) -> AgentRunnerConfig:
        """CrewAI configuration using Claude models."""
        return AgentRunnerConfig(
            agent_id=agent_id,
            framework="crewai",
            llm_config=LLMConfig(
                provider="anthropic",
                model="claude-3-sonnet",
                temperature=0.1,
                api_key=api_key
            ),
            crew_config=CrewConfig(
                process="sequential",
                crew_size=4,
                allow_delegation=True
            ),
            verbose=False
        )


class LangChainConfig:
    """Pre-built configurations for LangChain agent runner."""
    
    @staticmethod
    def reasoning_agent(agent_id: str, api_key: str) -> AgentRunnerConfig:
        """LangChain agent with reasoning chains and memory."""
        return AgentRunnerConfig(
            agent_id=agent_id,
            framework="langchain",
            llm_config=LLMConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.1,
                api_key=api_key
            ),
            memory_config=MemoryConfig(
                type="buffer",
                window_size=10
            ),
            max_iterations=5,
            verbose=False
        )
    
    @staticmethod
    def memory_agent(agent_id: str, api_key: str) -> AgentRunnerConfig:
        """LangChain agent with advanced memory system."""
        return AgentRunnerConfig(
            agent_id=agent_id,
            framework="langchain",
            llm_config=LLMConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.2,
                api_key=api_key
            ),
            memory_config=MemoryConfig(
                type="summary",
                window_size=20
            ),
            max_iterations=7,
            verbose=False
        )
    
    @staticmethod
    def fast_agent(agent_id: str, api_key: str) -> AgentRunnerConfig:
        """Fast LangChain agent with minimal memory for quick decisions."""
        return AgentRunnerConfig(
            agent_id=agent_id,
            framework="langchain",
            llm_config=LLMConfig(
                provider="openai",
                model="gpt-3.5-turbo",
                temperature=0.1,
                api_key=api_key
            ),
            memory_config=MemoryConfig(
                type="buffer",
                window_size=5
            ),
            max_iterations=3,
            timeout_seconds=15,
            verbose=False
        )
    
    @staticmethod
    def analytical_agent(agent_id: str, api_key: str) -> AgentRunnerConfig:
        """Analytical LangChain agent for deep market analysis."""
        return AgentRunnerConfig(
            agent_id=agent_id,
            framework="langchain",
            llm_config=LLMConfig(
                provider="openai",
                model="gpt-4",
                temperature=0.05,  # Very low temperature for analytical work
                api_key=api_key
            ),
            memory_config=MemoryConfig(
                type="summary",
                window_size=15
            ),
            max_iterations=10,  # Allow more iterations for thorough analysis
            timeout_seconds=60,
            verbose=True  # Enable verbose for debugging analytical process
        )


def get_framework_examples() -> Dict[str, Dict[str, Any]]:
    """Get example configurations for all frameworks."""
    return {
        "diy": {
            "advanced_agent": "Event-driven agent with pricing strategy",
            "baseline_greedy": "Simple greedy pricing bot",
            "llm_claude": "Claude-powered decision making",
            "llm_gpt": "GPT-powered decision making"
        },
        "crewai": {
            "standard_crew": "4-agent collaborative crew",
            "hierarchical_crew": "Manager-coordinated crew",
            "focused_pricing_crew": "Specialized pricing optimization",
            "claude_crew": "CrewAI with Claude models"
        },
        "langchain": {
            "reasoning_agent": "Reasoning chains with memory",
            "memory_agent": "Advanced memory system",
            "fast_agent": "Quick decisions, minimal memory",
            "analytical_agent": "Deep analytical capabilities"
        }
    }


def create_example_config(framework: str, config_type: str, agent_id: str, 
                         api_key: str = None, **kwargs) -> AgentRunnerConfig:
    """
    Create an example configuration.
    
    Args:
        framework: Framework name (diy, crewai, langchain)
        config_type: Configuration type (e.g., 'standard_crew', 'reasoning_agent')
        agent_id: Agent identifier
        api_key: API key for LLM providers
        **kwargs: Additional parameters
        
    Returns:
        Pre-configured AgentRunnerConfig instance
    """
    if framework == "diy":
        if config_type == "advanced_agent":
            return DIYConfig.advanced_agent(agent_id, kwargs.get("target_asin", "B0DEFAULT"))
        elif config_type == "baseline_greedy":
            return DIYConfig.baseline_greedy(agent_id)
        elif config_type == "llm_claude":
            return DIYConfig.llm_claude(agent_id, api_key, kwargs.get("target_asin", "B0DEFAULT"))
        elif config_type == "llm_gpt":
            return DIYConfig.llm_gpt(agent_id, api_key, kwargs.get("target_asin", "B0DEFAULT"))
    
    elif framework == "crewai":
        if config_type == "standard_crew":
            return CrewAIConfig.standard_crew(agent_id, api_key)
        elif config_type == "hierarchical_crew":
            return CrewAIConfig.hierarchical_crew(agent_id, api_key)
        elif config_type == "focused_pricing_crew":
            return CrewAIConfig.focused_pricing_crew(agent_id, api_key)
        elif config_type == "claude_crew":
            return CrewAIConfig.claude_crew(agent_id, api_key)
    
    elif framework == "langchain":
        if config_type == "reasoning_agent":
            return LangChainConfig.reasoning_agent(agent_id, api_key)
        elif config_type == "memory_agent":
            return LangChainConfig.memory_agent(agent_id, api_key)
        elif config_type == "fast_agent":
            return LangChainConfig.fast_agent(agent_id, api_key)
        elif config_type == "analytical_agent":
            return LangChainConfig.analytical_agent(agent_id, api_key)
    
    raise ValueError(f"Unknown configuration: {framework}.{config_type}")