"""
DemandService for FBA-Bench: handles demand calculation and BSR updates.
"""

from typing import Any, List, Dict
from fba_bench.money import Money

class DemandService:
    """
    Service for updating product demand metrics and BSR.
    """

    def __init__(self, bsr_base=100000, bsr_min_value=1, bsr_max_value=1000000, bsr_smoothing_factor=0.1, ema_decay=0.2):
        self.bsr_base = bsr_base
        self.bsr_min_value = bsr_min_value
        self.bsr_max_value = bsr_max_value
        self.bsr_smoothing_factor = bsr_smoothing_factor
        self.ema_decay = ema_decay

    def update_bsr(
        self,
        product: Any,
        units_sold: int,
        demand: float,
        competitors: List[Any]
    ) -> None:
        """
        Update BSR metrics for a product based on sales performance and competition.
        """
        self._update_sales_history(product, units_sold, demand)
        self._update_ema_metrics(product, units_sold, demand)
        new_bsr = self._calculate_bsr(product, competitors)
        product.bsr = new_bsr

    def _update_sales_history(self, product: Any, units_sold: int, demand: float) -> None:
        product.sales_history.append(units_sold)
        product.demand_history.append(demand)
        conversion = units_sold / demand if demand > 0 else 0.0
        product.conversion_history.append(conversion)

    def _update_ema_metrics(self, product: Any, units_sold: int, demand: float) -> None:
        conversion = units_sold / demand if demand > 0 else 0.0
        if len(product.sales_history) == 1:
            product.ema_sales_velocity = units_sold
            product.ema_conversion = conversion
        else:
            product.ema_sales_velocity = (
                (1 - self.ema_decay) * product.ema_sales_velocity +
                self.ema_decay * units_sold
            )
            product.ema_conversion = (
                (1 - self.ema_decay) * product.ema_conversion +
                self.ema_decay * conversion
            )

    def _calculate_bsr(self, product: Any, competitors: List[Any]) -> int:
        avg_comp_sales = self.bsr_smoothing_factor
        avg_comp_price = product.price

        if competitors:
            avg_comp_sales = max(
                self.bsr_smoothing_factor,
                sum(c.sales_velocity for c in competitors) / len(competitors)
            )
            total_comp_price = Money.zero()
            for competitor in competitors:
                total_comp_price += competitor.price
            avg_comp_price = total_comp_price / len(competitors)

        rel_sales_index = (
            max(self.bsr_smoothing_factor, product.ema_sales_velocity) / avg_comp_sales
        )
        rel_price_index = (
            avg_comp_price / max(Money.from_dollars(self.bsr_smoothing_factor), product.price)
        )

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
        return {
            'current_bsr': product.bsr,
            'ema_sales_velocity': getattr(product, 'ema_sales_velocity', 0),
            'ema_conversion': getattr(product, 'ema_conversion', 0),
            'sales_history_length': len(getattr(product, 'sales_history', [])),
            'demand_history_length': len(getattr(product, 'demand_history', [])),
            'conversion_history_length': len(getattr(product, 'conversion_history', []))
        }