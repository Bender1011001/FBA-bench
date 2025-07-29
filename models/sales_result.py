"""Sales result model for transaction tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

from money import Money


@dataclass
class SalesResult:
    """
    Result of a sales transaction with comprehensive tracking.
    
    Captures all relevant information about a sales transaction
    including units sold, revenue, fees, and profit calculations.
    """
    
    asin: str
    units_sold: int
    units_demanded: int
    revenue: Money
    total_fees: Money
    profit: Money
    timestamp: datetime
    
    # Fee breakdown for analysis
    fee_breakdown: Dict[str, Money] = field(default_factory=dict)
    
    # Additional transaction details
    unit_price: Optional[Money] = None
    conversion_rate: float = 0.0
    trust_score_at_sale: float = 1.0
    bsr_at_sale: int = 1000000
    
    def __post_init__(self):
        """Validate and calculate derived fields."""
        # Ensure Money types
        for field_name, value in [('revenue', self.revenue), ('total_fees', self.total_fees), ('profit', self.profit)]:
            if not isinstance(value, Money):
                if isinstance(value, (int, float, str)):
                    setattr(self, field_name, Money.from_dollars(value))
                else:
                    raise TypeError(f"{field_name} must be Money type or convertible, got {type(value)}")
        
        # Calculate unit price if not provided
        if self.unit_price is None and self.units_sold > 0:
            self.unit_price = self.revenue / self.units_sold
        
        # Calculate conversion rate if not provided
        if self.conversion_rate == 0.0 and self.units_demanded > 0:
            self.conversion_rate = self.units_sold / self.units_demanded
        
        # Validate fee breakdown contains Money types
        for fee_type, amount in self.fee_breakdown.items():
            if not isinstance(amount, Money):
                if isinstance(amount, (int, float, str)):
                    self.fee_breakdown[fee_type] = Money.from_dollars(amount)
                else:
                    raise TypeError(f"Fee breakdown '{fee_type}' must be Money type, got {type(amount)}")
    
    def get_profit_margin_percentage(self) -> float:
        """Calculate profit margin as percentage of revenue."""
        if self.revenue.cents == 0:
            return 0.0
        return (self.profit.to_float() / self.revenue.to_float()) * 100
    
    def get_fee_percentage(self) -> float:
        """Calculate total fees as percentage of revenue."""
        if self.revenue.cents == 0:
            return 0.0
        return (self.total_fees.to_float() / self.revenue.to_float()) * 100
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the sales result."""
        return {
            'asin': self.asin,
            'units_sold': self.units_sold,
            'units_demanded': self.units_demanded,
            'conversion_rate': round(self.conversion_rate, 3),
            'revenue': str(self.revenue),
            'total_fees': str(self.total_fees),
            'profit': str(self.profit),
            'profit_margin_pct': round(self.get_profit_margin_percentage(), 2),
            'fee_percentage': round(self.get_fee_percentage(), 2),
            'unit_price': str(self.unit_price) if self.unit_price else None,
            'trust_score_at_sale': self.trust_score_at_sale,
            'bsr_at_sale': self.bsr_at_sale,
            'timestamp': self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        return f"SalesResult({self.asin}, {self.units_sold} units, {self.profit} profit)"
    
    def __repr__(self) -> str:
        return (f"SalesResult(asin='{self.asin}', units_sold={self.units_sold}, "
                f"revenue={self.revenue!r}, profit={self.profit!r})")