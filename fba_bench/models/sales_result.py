"""Sales processing result data structures."""
from __future__ import annotations

from dataclasses import dataclass
from fba_bench.money import Money


@dataclass
class SalesResult:
    """Result of processing sales for a single product."""
    
    units_sold: int
    revenue: Money
    total_fees: Money
    demand: int
    trust_score: float
    bsr_change: int = 0
    
    @property
    def net_profit(self) -> Money:
        """Calculate net profit (revenue - fees)."""
        return self.revenue - self.total_fees
    
    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate (units_sold / demand)."""
        return self.units_sold / self.demand if self.demand > 0 else 0.0