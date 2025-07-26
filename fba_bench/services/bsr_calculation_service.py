"""
BSR (Best Seller Rank) Calculation Service.

Handles all BSR-related calculations including sales velocity tracking,
conversion rate analysis, and competitive positioning metrics.
"""
from typing import List, Dict, Any
from decimal import Decimal
from fba_bench.money import Money
from fba_bench.models.competitor import Competitor


class BSRCalculationService:
    """
    Service responsible for calculating and updating Best Seller Rank (BSR) metrics.
    
    This service encapsulates the complex BSR calculation logic that was previously
    embedded in the monolithic tick_day() method.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BSR calculation service with configuration.
        
        Args:
            config: Configuration dictionary containing BSR calculation parameters
        """
        self.ema_decay = config.get('ema_decay', 0.1)
        self.bsr_smoothing_factor = config.get('bsr_smoothing_factor', 1.0)
        self.bsr_base = config.get('bsr_base', 1000000)
        self.bsr_min_value = config.get('bsr_min_value', 1)
        self.bsr_max_value = config.get('bsr_max_value', 10000000)
    
    def update_product_bsr(
        self,
        product: Any,
        units_sold: int,
        demand: float,
        competitors: List[Competitor]
    ) -> None:
        """
        Update BSR metrics for a product based on sales performance and competition.
        
        Args:
            product: Product object to update
            units_sold: Number of units sold in this period
            demand: Calculated demand for the product
            competitors: List of competing products
        """
        # Update sales and demand history
        self._update_sales_history(product, units_sold, demand)
        
        # Update EMA metrics
        self._update_ema_metrics(product, units_sold, demand)
        
        # Calculate and update BSR
        new_bsr = self._calculate_bsr(product, competitors)
        product.bsr = new_bsr
    
    def _update_sales_history(self, product: Any, units_sold: int, demand: float) -> None:
        """Update product sales and demand history."""
        product.sales_history.append(units_sold)
        product.demand_history.append(demand)
        
        # Calculate conversion rate
        conversion = units_sold / demand if demand > 0 else 0.0
        product.conversion_history.append(conversion)
    
    def _update_ema_metrics(self, product: Any, units_sold: int, demand: float) -> None:
        """Update Exponential Moving Average metrics for sales velocity and conversion."""
        conversion = units_sold / demand if demand > 0 else 0.0
        
        if len(product.sales_history) == 1:
            # First data point - initialize EMA
            product.ema_sales_velocity = units_sold
            product.ema_conversion = conversion
        else:
            # Update EMA using decay factor
            product.ema_sales_velocity = (
                (1 - self.ema_decay) * product.ema_sales_velocity + 
                self.ema_decay * units_sold
            )
            product.ema_conversion = (
                (1 - self.ema_decay) * product.ema_conversion + 
                self.ema_decay * conversion
            )
    
    def _calculate_bsr(self, product: Any, competitors: List[Competitor]) -> int:
        """
        Calculate BSR using the blueprint formula:
        BSR = base / (ema_sales_velocity * ema_conversion * rel_sales_index * rel_price_index)
        """
        # Calculate competitor averages for relative indices
        avg_comp_sales = self.bsr_smoothing_factor
        avg_comp_price = product.price
        
        if competitors:
            avg_comp_sales = max(
                self.bsr_smoothing_factor,
                sum(c.sales_velocity for c in competitors) / len(competitors)
            )
            
            # Calculate average competitor price using proper Money arithmetic
            total_comp_price = Money.zero()
            for competitor in competitors:
                total_comp_price += competitor.price
            avg_comp_price = total_comp_price / len(competitors)
        
        # Calculate relative indices
        rel_sales_index = (
            max(self.bsr_smoothing_factor, product.ema_sales_velocity) / avg_comp_sales
        )
        
        rel_price_index = (
            avg_comp_price / max(Money.from_dollars(self.bsr_smoothing_factor), product.price)
        )
        
        # Apply BSR formula with bounds checking
        if (product.ema_sales_velocity > self.bsr_smoothing_factor and 
            product.ema_conversion > self.bsr_smoothing_factor):
            
            denominator = (
                product.ema_sales_velocity *
                product.ema_conversion *
                rel_sales_index *
                float(rel_price_index)
            )
            
            calculated_bsr = self.bsr_base / max(self.bsr_smoothing_factor, denominator)
            return max(self.bsr_min_value, min(self.bsr_max_value, int(calculated_bsr)))
        else:
            return self.bsr_base
    
    def get_bsr_metrics(self, product: Any) -> Dict[str, Any]:
        """
        Get comprehensive BSR metrics for a product.
        
        Returns:
            Dictionary containing BSR metrics and performance indicators
        """
        return {
            'current_bsr': product.bsr,
            'ema_sales_velocity': getattr(product, 'ema_sales_velocity', 0),
            'ema_conversion': getattr(product, 'ema_conversion', 0),
            'sales_history_length': len(getattr(product, 'sales_history', [])),
            'demand_history_length': len(getattr(product, 'demand_history', [])),
            'conversion_history_length': len(getattr(product, 'conversion_history', []))
        }