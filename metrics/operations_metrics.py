# metrics/operations_metrics.py
from typing import List, Dict, Any

class OperationsMetrics:
    def __init__(self, sales_service: Any, initial_inventory_value: float = 0.0):
        self.sales_service = sales_service
        self.total_revenue = 0.0
        self.inventory_history: List[Dict] = [] # {'tick': int, 'value': float}
        self.stockout_days = 0
        self.total_operational_days = 0
        self.initial_inventory_value = initial_inventory_value

    def update(self, current_tick: int, events: List[Dict]):
        self.total_operational_days = current_tick # Assuming tick represents a day

        # Update revenue from SaleOccurred events
        for event in events:
            if event.get('type') == 'SaleOccurred':
                self.total_revenue += event.get('amount', 0.0)

        # Assuming sales_service can provide current inventory value
        current_inventory_value = self.sales_service.get_current_inventory_value() # Placeholder
        self.inventory_history.append({'tick': current_tick, 'value': current_inventory_value})

        # Check for stockouts (simplified: if inventory value drops to zero)
        if current_inventory_value == 0:
            self.stockout_days += 1

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