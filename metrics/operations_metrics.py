# metrics/operations_metrics.py
from typing import List, Dict, Any, Optional, Callable

class OperationsMetrics:
    def __init__(self, sales_service: Any, initial_inventory_value: float = 0.0):
        self.sales_service = sales_service
        self.total_revenue = 0.0
        self.inventory_history: List[Dict] = [] # {'tick': int, 'value': float}
        self.stockout_days = 0
        self.total_operational_days = 0
        self.initial_inventory_value = initial_inventory_value

    def update(self, current_tick: int, events: List[Dict]):
        self.total_operational_days = current_tick  # Assuming tick represents a day

        # Update revenue from SaleOccurred events
        for event in events:
            if event.get('type') == 'SaleOccurred':
                # Prefer explicit total_revenue/amount fields, fallback to unit_price * units if present
                amount = event.get('amount')
                if isinstance(amount, (int, float)):
                    self.total_revenue += float(amount)
                else:
                    units = event.get('units_sold') or event.get('units', 0)
                    unit_price = event.get('unit_price') or event.get('price') or 0.0
                    try:
                        self.total_revenue += float(units) * float(unit_price)
                    except Exception:
                        # ignore if not computable
                        pass

        # Resolve inventory value via service if available, else via events fallback, else last known value
        current_inventory_value = self._resolve_inventory_value(events)
        self.inventory_history.append({'tick': current_tick, 'value': current_inventory_value})

        # Check for stockouts (simplified: if inventory value drops to zero)
        if current_inventory_value == 0:
            self.stockout_days += 1

    def _resolve_inventory_value(self, events: List[Dict]) -> float:
        """
        Determine current inventory value using the following precedence:
        1) sales_service.get_current_inventory_value() if available
        2) Any event that carries 'inventory_value' or ('inventory_quantity' and 'unit_cost'/'cost_basis')
        3) Last known inventory value (previous history) or initial_inventory_value
        """
        # 1) Service method if present
        try:
            getter: Optional[Callable[[], float]] = getattr(self.sales_service, "get_current_inventory_value", None)
            if callable(getter):
                val = getter()
                if isinstance(val, (int, float)):
                    return float(val)
        except Exception:
            pass

        # 2) Fallback via events data
        latest_val: Optional[float] = None
        for e in reversed(events):  # prefer most recent event first
            if isinstance(e, dict):
                if "inventory_value" in e and isinstance(e["inventory_value"], (int, float)):
                    latest_val = float(e["inventory_value"])
                    break
                qty = e.get("inventory_quantity")
                unit_cost = e.get("unit_cost") or e.get("cost_basis")
                if isinstance(qty, (int, float)) and isinstance(unit_cost, (int, float)):
                    latest_val = float(qty) * float(unit_cost)
                    break

        if latest_val is not None:
            return latest_val

        # 3) Prior value or initial
        if self.inventory_history:
            return float(self.inventory_history[-1]['value'])
        return float(self.initial_inventory_value)

    def calculate_inventory_turnover(self) -> float:
        if not self.inventory_history:
            return 0.0

        # Calculate average inventory value
        inventory_values = [d['value'] for d in self.inventory_history]
        inventory_values.insert(0, self.initial_inventory_value) # Include initial inventory
        average_inventory_value = sum(inventory_values) / len(inventory_values)

        if average_inventory_value > 0:
            return self.total_revenue / average_inventory_value
        return 0.0

    def calculate_stockout_percentage(self) -> float:
        if self.total_operational_days == 0:
            return 0.0
        return (self.stockout_days / self.total_operational_days) * 100

    def get_metrics_breakdown(self) -> Dict[str, float]:
        inventory_turnover = self.calculate_inventory_turnover()
        stockout_percentage = self.calculate_stockout_percentage()

        return {
            "inventory_turnover": inventory_turnover,
            "stockout_percentage": stockout_percentage,
        }