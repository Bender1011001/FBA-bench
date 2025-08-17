"""
CrewAI Agent Runner for FBA-Bench.

This module implements the AgentRunner interface for CrewAI agents,
enabling them to participate in the benchmarking system.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_runner import AgentRunner, AgentRunnerStatus, AgentRunnerError, AgentRunnerInitializationError, AgentRunnerDecisionError
from benchmarking.config.pydantic_config import FrameworkType, LLMConfig, CrewConfig
from pydantic import ValidationError
from fba_bench.core.llm_outputs import FbaDecision

logger = logging.getLogger(__name__)


class CrewAIRunner(AgentRunner):
    """
    Agent runner for CrewAI agents.
    
    This class integrates CrewAI agents into the FBA-Bench system,
    allowing them to make pricing decisions in the simulation.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the CrewAI agent runner."""
        super().__init__(agent_id, config)
        self.crewai_agent = None
        self.crew = None
        self.llm_config = None
        self.crew_config = None
        
    def _do_initialize(self) -> None:
        """Initialize the CrewAI agent and crew."""
        try:
            # Extract configurations
            self.llm_config = self._extract_llm_config()
            self.crew_config = self._extract_crew_config()
            
            # Create the CrewAI agent
            self._create_crewai_agent()
            
            # Create the crew
            self._create_crew()
            
            logger.info(f"CrewAI agent runner {self.agent_id} initialized successfully")
            
        except ImportError as e:
            raise AgentRunnerInitializationError(
                f"CrewAI not available: {e}. Install with: pip install crewai",
                agent_id=self.agent_id,
                framework="CrewAI"
            ) from e
        except Exception as e:
            raise AgentRunnerInitializationError(
                f"Failed to initialize CrewAI agent {self.agent_id}: {e}",
                agent_id=self.agent_id,
                framework="CrewAI"
            ) from e
    
    def _extract_llm_config(self) -> LLMConfig:
        """Extract LLM configuration from the agent config."""
        llm_config_dict = self.config.get('llm_config', {})
        
        # Create LLMConfig with defaults
        return LLMConfig(
            name=f"{self.agent_id}_llm",
            model=llm_config_dict.get('model', 'gpt-4'),
            api_key=llm_config_dict.get('api_key'),
            base_url=llm_config_dict.get('base_url'),
            max_tokens=llm_config_dict.get('max_tokens', 2048),
            temperature=llm_config_dict.get('temperature', 0.7),
            top_p=llm_config_dict.get('top_p', 1.0),
            timeout=llm_config_dict.get('timeout', 30),
            max_retries=llm_config_dict.get('max_retries', 3)
        )
    
    def _extract_crew_config(self) -> CrewConfig:
        """Extract Crew configuration from the agent config."""
        crew_config_dict = self.config.get('crew_config', {})
        
        # Create CrewConfig with defaults
        return CrewConfig(
            name=f"{self.agent_id}_crew",
            process=crew_config_dict.get('process', 'sequential'),
            crew_size=crew_config_dict.get('crew_size', 1),
            roles=crew_config_dict.get('roles', ['pricing_specialist']),
            collaboration_mode=crew_config_dict.get('collaboration_mode', 'sequential'),
            allow_delegation=crew_config_dict.get('allow_delegation', False)
        )
    
    def _create_crewai_agent(self) -> None:
        """Create the CrewAI agent."""
        from crewai import Agent
        
        # Get agent-specific parameters
        agent_params = self.config.get('parameters', {})
        
        # Create the CrewAI agent
        self.crewai_agent = Agent(
            role=agent_params.get('role', 'FBA Pricing Specialist'),
            goal=agent_params.get('goal', 'Optimize FBA product pricing for maximum profit'),
            backstory=agent_params.get('backstory', 'You are an experienced FBA pricing expert with deep knowledge of e-commerce markets.'),
            verbose=False,
            allow_delegation=self.crew_config.allow_delegation
        )
        
        logger.debug(f"Created CrewAI agent with role: {self.crewai_agent.role}")
    
    def _create_crew(self) -> None:
        """Create the CrewAI crew."""
        from crewai import Crew, Task
        
        # Create a default task for the agent
        task = Task(
            description="Analyze FBA market data and make optimal pricing decisions",
            agent=self.crewai_agent,
            expected_output="A JSON object with pricing decisions for each product"
        )
        
        # Create the crew
        self.crew = Crew(
            agents=[self.crewai_agent],
            tasks=[task],
            verbose=False,
            process=self.crew_config.process
        )
        
        logger.debug(f"Created CrewAI crew with process: {self.crew_config.process}")
    
    def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a pricing decision using the CrewAI agent.
        
        Args:
            context: Context information including market state and products
            
        Returns:
            Dictionary containing the decision and metadata
        """
        try:
            # Update context
            self.update_context(context)
            
            # Format the task description based on the context
            task_description = self._format_task_description(context)
            
            # Update the crew task
            from crewai import Task
            task = Task(
                description=task_description,
                agent=self.crewai_agent,
                expected_output="A JSON object with pricing decisions for each product"
            )
            
            # Update the crew with the new task
            self.crew.tasks = [task]
            
            # Execute the crew
            result = self.crew.kickoff()
            
            # Parse and validate structured output using Pydantic
            decision = self._parse_crewai_result(result)
            
            # Update metrics
            self.update_metrics({
                'decision_timestamp': datetime.now().isoformat(),
                'decision_type': 'pricing',
                'raw_result': str(result),
                'parsed_decision': decision
            })
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in CrewAI decision making: {e}")
            raise AgentRunnerDecisionError(
                f"Decision making failed for CrewAI agent {self.agent_id}: {e}",
                agent_id=self.agent_id,
                framework="CrewAI"
            ) from e
    
    def _format_task_description(self, context: Dict[str, Any]) -> str:
        """Format the task description based on the context."""
        # Extract relevant information from context
        products = context.get('products', [])
        market_conditions = context.get('market_conditions', {})
        recent_events = context.get('recent_events', [])
        tick = context.get('tick', 0)
        
        # Format the task description
        task_description = f"You are an FBA pricing expert. Analyze the following data and make optimal pricing decisions for tick {tick}:\n\n"
        
        # Add products information
        if products:
            task_description += "PRODUCTS:\n"
            for product in products:
                task_description += f"- ASIN: {product.get('asin', 'unknown')}, "
                task_description += f"Current Price: ${product.get('current_price', 0):.2f}, "
                task_description += f"Cost: ${product.get('cost', 0):.2f}, "
                task_description += f"Sales Rank: {product.get('sales_rank', 'unknown')}, "
                task_description += f"Inventory: {product.get('inventory', 0)}\n"
        
        # Add market conditions
        if market_conditions:
            task_description += "\nMARKET CONDITIONS:\n"
            for key, value in market_conditions.items():
                task_description += f"- {key}: {value}\n"
        
        # Add recent events
        if recent_events:
            task_description += "\nRECENT EVENTS:\n"
            for event in recent_events[-5:]:  # Only include last 5 events
                task_description += f"- {event}\n"
        
        # Require exact schema compliance for robust parsing
        schema_text = json.dumps(FbaDecision.model_json_schema(), indent=2)
        task_description += (
            "\nRespond ONLY with a valid JSON object that conforms exactly to this schema:\n"
            f"{schema_text}\n"
            "Do not include any prose or explanation outside the JSON. "
            "Use floating-point dollars for prices (e.g., 19.99)."
        )

        return task_description
    
    def _parse_crewai_result(self, result: Any) -> Dict[str, Any]:
        """
        Validate the LLM output against the FbaDecision schema and adapt to legacy dict format.
        Raises AgentRunnerDecisionError if validation fails.
        """
        raw_text = result if isinstance(result, str) else str(result)

        json_text = raw_text
        if not raw_text.strip().startswith("{"):
            try:
                start = raw_text.index("{")
                end = raw_text.rindex("}") + 1
                json_text = raw_text[start:end]
            except Exception:
                json_text = raw_text

        try:
            validated: FbaDecision = FbaDecision.model_validate_json(json_text)
        except ValidationError as e:
            logger.error(f"LLM output validation failed: {e}")
            raise AgentRunnerDecisionError(
                f"Decision making failed for CrewAI agent {self.agent_id}: invalid structured output",
                agent_id=self.agent_id,
                framework="CrewAI",
            ) from e

        # Build adapted legacy-shaped decision dict
        adapted: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "framework": "CrewAI",
            "timestamp": datetime.now().isoformat(),
            "pricing_decisions": {},
            "reasoning": "Validated structured output",
        }
        for pd in validated.pricing_decisions:
            adapted["pricing_decisions"][pd.asin] = {
                "price": float(pd.new_price),
                "confidence": 0.9,
                "reasoning": pd.reasoning,
            }
        if validated.meta:
            adapted["meta"] = validated.meta
        return adapted
    
    def _do_cleanup(self) -> None:
        """Clean up CrewAI resources."""
        # CrewAI doesn't require explicit cleanup
        self.crewai_agent = None
        self.crew = None
        logger.info(f"CrewAI agent runner {self.agent_id} cleaned up")