"""
DIY (Do It Yourself) Agent Runner for FBA-Bench.

This module implements the AgentRunner interface for custom-built agents,
enabling them to participate in the benchmarking system.
"""

import logging
import json
import math
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

from .base_runner import AgentRunner, AgentRunnerStatus, AgentRunnerError, AgentRunnerInitializationError, AgentRunnerDecisionError
from benchmarking.config.pydantic_config import FrameworkType, LLMConfig, AgentConfig

logger = logging.getLogger(__name__)


class PricingStrategy:
    """Base class for pricing strategies."""
    
    def calculate_price(self, product_data: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate the optimal price for a product."""
        raise NotImplementedError


class CompetitivePricingStrategy(PricingStrategy):
    """Competitive pricing strategy based on market conditions."""
    
    def __init__(self, margin_target: float = 0.3, competitor_sensitivity: float = 0.5):
        self.margin_target = margin_target  # Target profit margin (30% by default)
        self.competitor_sensitivity = competitor_sensitivity  # How much to adjust based on competitors
    
    def calculate_price(self, product_data: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate price based on competitive analysis."""
        cost = product_data.get('cost', 0)
        current_price = product_data.get('current_price', cost * 1.5)
        sales_rank = product_data.get('sales_rank', 1000000)
        inventory = product_data.get('inventory', 100)
        
        # Get competitor prices from market data
        competitor_prices = market_data.get('competitor_prices', [])
        avg_competitor_price = sum(competitor_prices) / len(competitor_prices) if competitor_prices else current_price
        
        # Base price calculation (cost + target margin)
        base_price = cost * (1 + self.margin_target)
        
        # Adjust based on sales rank (lower rank = higher demand = higher price)
        rank_factor = 1.0
        if sales_rank < 10000:  # Top 10,000
            rank_factor = 1.2
        elif sales_rank < 50000:  # Top 50,000
            rank_factor = 1.1
        elif sales_rank > 500000:  # Poor selling
            rank_factor = 0.9
        
        # Adjust based on inventory levels
        inventory_factor = 1.0
        if inventory < 10:  # Low inventory, can increase price
            inventory_factor = 1.1
        elif inventory > 100:  # High inventory, need to reduce price
            inventory_factor = 0.95
        
        # Competitive adjustment
        competitive_factor = 1.0
        if avg_competitor_price > 0:
            price_ratio = base_price / avg_competitor_price
            if price_ratio > 1.2:  # We're much more expensive
                competitive_factor = 1.0 - (self.competitor_sensitivity * 0.2)
            elif price_ratio < 0.8:  # We're much cheaper
                competitive_factor = 1.0 + (self.competitor_sensitivity * 0.1)
        
        # Calculate final price
        final_price = base_price * rank_factor * inventory_factor * competitive_factor
        
        # Ensure minimum profit margin
        minimum_price = cost * 1.1  # At least 10% margin
        final_price = max(final_price, minimum_price)
        
        return round(final_price, 2)


class DynamicPricingStrategy(PricingStrategy):
    """Dynamic pricing strategy that adapts to market conditions."""
    
    def __init__(self, base_margin: float = 0.25, elasticity_factor: float = 0.3):
        self.base_margin = base_margin
        self.elasticity_factor = elasticity_factor
        self.price_history = {}  # Track price history for each product
    
    def calculate_price(self, product_data: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate price using dynamic pricing algorithm."""
        asin = product_data.get('asin', 'unknown')
        cost = product_data.get('cost', 0)
        current_price = product_data.get('current_price', cost * 1.5)
        sales_rank = product_data.get('sales_rank', 1000000)
        inventory = product_data.get('inventory', 100)
        
        # Get market demand indicator
        market_demand = market_data.get('market_demand', 1.0)  # 1.0 = neutral
        
        # Get seasonality factor
        seasonality = market_data.get('seasonality', 1.0)  # 1.0 = neutral
        
        # Calculate base price
        base_price = cost * (1 + self.base_margin)
        
        # Demand-based adjustment
        demand_factor = 1.0 + (market_demand - 1.0) * self.elasticity_factor
        
        # Seasonality adjustment
        seasonality_factor = seasonality
        
        # Sales rank adjustment (logarithmic scale)
        rank_adjustment = 1.0
        if sales_rank > 0:
            rank_adjustment = 1.0 + 0.1 * math.log(1000000 / sales_rank)
        
        # Inventory adjustment
        inventory_adjustment = 1.0
        if inventory > 0:
            inventory_adjustment = 1.0 - 0.05 * math.log(inventory / 10)
        
        # First compute a preliminary price without history dampening
        preliminary_price = base_price * demand_factor * seasonality_factor * rank_adjustment * inventory_adjustment

        # Apply history-based dampening to avoid large swings
        history_adjustment = 1.0
        last_price = None
        if asin in self.price_history and self.price_history[asin]:
            last_price = self.price_history[asin][-1]
            change_ratio = abs(preliminary_price - last_price) / last_price if last_price and last_price > 0 else 0.0
            if change_ratio > 0.10:
                history_adjustment = 0.9  # Dampen large changes by 10%

        final_price = preliminary_price * history_adjustment

        # Ensure minimum profit margin
        minimum_price = cost * 1.1  # At least 10% margin
        final_price = max(final_price, minimum_price)

        # Update price history
        if asin not in self.price_history:
            self.price_history[asin] = []
        self.price_history[asin].append(final_price)

        # Keep only last 10 prices
        if len(self.price_history[asin]) > 10:
            self.price_history[asin] = self.price_history[asin][-10:]

        return round(final_price, 2)


class DIYRunner(AgentRunner):
    """
    Agent runner for DIY (Do It Yourself) agents.
    
    This class integrates custom-built agents into the FBA-Bench system,
    allowing them to make pricing decisions using custom algorithms.
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """Initialize the DIY agent runner."""
        super().__init__(agent_id, config)
        self.llm_config = None
        self.agent_config = None
        self.pricing_strategy = None
        self.decision_history = []
        
    def _do_initialize(self) -> None:
        """Initialize the DIY agent and its components."""
        try:
            # Extract configurations
            self.llm_config = self._extract_llm_config()
            self.agent_config = self._extract_agent_config()
            
            # Create the pricing strategy
            self._create_pricing_strategy()
            
            logger.info(f"DIY agent runner {self.agent_id} initialized successfully")
            
        except Exception as e:
            raise AgentRunnerInitializationError(
                f"Failed to initialize DIY agent {self.agent_id}: {e}",
                agent_id=self.agent_id,
                framework="DIY"
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
    
    def _extract_agent_config(self) -> AgentConfig:
        """Extract Agent configuration from the agent config."""
        agent_config_dict = self.config.get('agent_config', {})
        
        # Create AgentConfig with defaults
        return AgentConfig(
            name=f"{self.agent_id}_agent",
            agent_id=self.agent_id,
            type=agent_config_dict.get('type', 'pricing_agent'),
            framework=FrameworkType.DIY,
            parameters=agent_config_dict.get('parameters', {})
        )
    
    def _create_pricing_strategy(self) -> None:
        """Create the pricing strategy based on configuration."""
        strategy_type = self.agent_config.parameters.get('pricing_strategy', 'competitive')
        
        if strategy_type == 'competitive':
            margin_target = self.agent_config.parameters.get('margin_target', 0.3)
            competitor_sensitivity = self.agent_config.parameters.get('competitor_sensitivity', 0.5)
            self.pricing_strategy = CompetitivePricingStrategy(
                margin_target=margin_target,
                competitor_sensitivity=competitor_sensitivity
            )
        elif strategy_type == 'dynamic':
            base_margin = self.agent_config.parameters.get('base_margin', 0.25)
            elasticity_factor = self.agent_config.parameters.get('elasticity_factor', 0.3)
            self.pricing_strategy = DynamicPricingStrategy(
                base_margin=base_margin,
                elasticity_factor=elasticity_factor
            )
        else:
            # Default to competitive pricing
            self.pricing_strategy = CompetitivePricingStrategy()
        
        logger.debug(f"Created pricing strategy: {strategy_type}")
    
    def make_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a pricing decision using the DIY agent.
        
        Args:
            context: Context information including market state and products
            
        Returns:
            Dictionary containing the decision and metadata
        """
        try:
            # Update context
            self.update_context(context)
            
            # Extract products and market data
            products = context.get('products', [])
            market_conditions = context.get('market_conditions', {})
            tick = context.get('tick', 0)
            
            # Make pricing decisions for each product
            pricing_decisions = {}
            reasoning = ""
            
            for product in products:
                asin = product.get('asin', 'unknown')
                
                # Calculate price using the pricing strategy
                new_price = self.pricing_strategy.calculate_price(product, market_conditions)
                
                # Calculate confidence based on data quality
                confidence = self._calculate_confidence(product, market_conditions)
                
                # Generate reasoning
                product_reasoning = self._generate_reasoning(product, market_conditions, new_price)
                
                pricing_decisions[asin] = {
                    'price': new_price,
                    'confidence': confidence,
                    'reasoning': product_reasoning
                }
                
                reasoning += f"{asin}: {product_reasoning}\n"
            
            # Create decision object
            decision = {
                'agent_id': self.agent_id,
                'framework': 'DIY',
                'timestamp': datetime.now().isoformat(),
                'pricing_decisions': pricing_decisions,
                'reasoning': reasoning.strip()
            }
            
            # Update decision history
            self.decision_history.append({
                'tick': tick,
                'decision': decision,
                'timestamp': datetime.now()
            })
            
            # Keep only last 50 decisions
            if len(self.decision_history) > 50:
                self.decision_history = self.decision_history[-50:]
            
            # Update metrics
            self.update_metrics({
                'decision_timestamp': datetime.now().isoformat(),
                'decision_type': 'pricing',
                'products_count': len(products),
                'average_confidence': sum(d['confidence'] for d in pricing_decisions.values()) / len(pricing_decisions) if pricing_decisions else 0,
                'strategy_type': self.agent_config.parameters.get('pricing_strategy', 'competitive')
            })
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in DIY decision making: {e}")
            raise AgentRunnerDecisionError(
                f"Decision making failed for DIY agent {self.agent_id}: {e}",
                agent_id=self.agent_id,
                framework="DIY"
            ) from e
    
    def _calculate_confidence(self, product: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate confidence score for the pricing decision."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if we have good product data
        if product.get('cost', 0) > 0:
            confidence += 0.1
        if product.get('current_price', 0) > 0:
            confidence += 0.1
        if product.get('sales_rank', 0) > 0:
            confidence += 0.1
        if product.get('inventory', 0) >= 0:
            confidence += 0.1
        
        # Increase confidence if we have good market data
        if market_data.get('market_demand', 0) > 0:
            confidence += 0.05
        if market_data.get('competitor_prices', []):
            confidence += 0.05
        
        # Cap confidence at 0.95
        return min(confidence, 0.95)
    
    def _generate_reasoning(self, product: Dict[str, Any], market_data: Dict[str, Any], new_price: float) -> str:
        """Generate reasoning for the pricing decision."""
        asin = product.get('asin', 'unknown')
        cost = product.get('cost', 0)
        current_price = product.get('current_price', cost * 1.5)
        sales_rank = product.get('sales_rank', 1000000)
        inventory = product.get('inventory', 100)
        
        reasoning = f"Calculated price ${new_price:.2f} for {asin}. "
        
        # Add cost-based reasoning
        margin = (new_price - cost) / cost if cost > 0 else 0
        reasoning += f"Cost: ${cost:.2f}, Margin: {margin:.1%}. "
        
        # Add sales rank reasoning
        if sales_rank < 10000:
            reasoning += f"High demand product (rank {sales_rank}). "
        elif sales_rank > 500000:
            reasoning += f"Low demand product (rank {sales_rank}). "
        
        # Add inventory reasoning
        if inventory < 10:
            reasoning += f"Low inventory ({inventory} units). "
        elif inventory > 100:
            reasoning += f"High inventory ({inventory} units). "
        
        # Add market conditions reasoning
        market_demand = market_data.get('market_demand', 1.0)
        if market_demand > 1.2:
            reasoning += f"High market demand. "
        elif market_demand < 0.8:
            reasoning += f"Low market demand. "
        
        # Add competitor pricing reasoning
        competitor_prices = market_data.get('competitor_prices', [])
        if competitor_prices:
            avg_competitor_price = sum(competitor_prices) / len(competitor_prices)
            if new_price > avg_competitor_price * 1.1:
                reasoning += f"Priced above competitors (avg ${avg_competitor_price:.2f}). "
            elif new_price < avg_competitor_price * 0.9:
                reasoning += f"Priced below competitors (avg ${avg_competitor_price:.2f}). "
        
        return reasoning.strip()
    
    def _do_cleanup(self) -> None:
        """Clean up DIY agent resources."""
        self.pricing_strategy = None
        self.decision_history = []
        logger.info(f"DIY agent runner {self.agent_id} cleaned up")