# metrics/finance_metrics.py
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable

class FinanceMetrics:
    def __init__(self, financial_audit_service: Any):
        self.financial_audit_service = financial_audit_service
        self.net_worth_history: List[float] = []
        self.cash_flow_history: List[float] = []
        self.shock_net_worth_snapshots: Dict[str, Tuple[float, float, float]] = {} # shock_id: (before, during, after)

    def update(self, current_tick: int, events: List[Dict]):
        # Resolve current net worth from audit service
        current_net_worth = float(self.financial_audit_service.get_current_net_worth())
        self.net_worth_history.append(current_net_worth)

        # Compute cash flow via audit service if available; otherwise compute from events
        current_cash_flow = self._compute_cash_flow(events)
        self.cash_flow_history.append(current_cash_flow)

    def record_shock_snapshot(self, shock_id: str, phase: str):
        current_net_worth = self.financial_audit_service.get_current_net_worth()
        if shock_id not in self.shock_net_worth_snapshots:
            self.shock_net_worth_snapshots[shock_id] = [0.0, 0.0, 0.0]

        if phase == "before":
            self.shock_net_worth_snapshots[shock_id] = (current_net_worth, 0.0, 0.0)
        elif phase == "during":
            existing_before, _, _ = self.shock_net_worth_snapshots[shock_id]
            self.shock_net_worth_snapshots[shock_id] = (existing_before, current_net_worth, 0.0)
        elif phase == "after":
            existing_before, existing_during, _ = self.shock_net_worth_snapshots[shock_id]
            self.shock_net_worth_snapshots[shock_id] = (existing_before, existing_during, current_net_worth)

    def calculate_resilient_net_worth(self) -> float:
        # Simplistic resilient net worth: average of post-shock net worth relative to pre-shock
        # A more complex model would involve stress testing scenarios and projected recovery
        if not self.shock_net_worth_snapshots:
            return self.net_worth_history[-1] if self.net_worth_history else 0.0

        resilience_scores = []
        for shock_id, (before, during, after) in self.shock_net_worth_snapshots.items():
            if before > 0 and after > 0:
                resilience_scores.append((after - during) / before)  # Percentage recovery from minimum during shock
            elif before > 0: # If there's no "after" value, consider "during" as the "after" for partial shock
                resilience_scores.append((during - during) / before)

        return np.mean(resilience_scores) * 100 if resilience_scores else 0.0 # Return as percentage

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.01) -> float:
        # Requires more robust return calculation, assuming net worth changes represent returns
        returns = np.diff(self.net_worth_history) / self.net_worth_history[:-1]
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0.0

    def calculate_drawdown_recovery(self) -> float:
        if not self.net_worth_history:
            return 0.0
        peak = self.net_worth_history[0]
        max_drawdown = 0.0
        for nw in self.net_worth_history:
            if nw > peak:
                peak = nw
            drawdown = (peak - nw) / peak if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # This is a simplified recovery; true recovery would track time to new peak
        return 1.0 - max_drawdown if max_drawdown > 0 else 1.0 # Return as a recovery percentage (1.0 = full recovery)

    def _compute_cash_flow(self, events: List[Dict]) -> float:
        """
        Compute cash flow using the following precedence:
        1) financial_audit_service.get_current_cash_flow() if available
        2) Sum of cash-in (SaleOccurred.amount or unit_price*units) minus cash-out (PurchaseOccurred.cost or unit_cost*units)
        """
        try:
            getter: Optional[Callable[[], float]] = getattr(self.financial_audit_service, "get_current_cash_flow", None)
            if callable(getter):
                val = getter()
                if isinstance(val, (int, float)):
                    return float(val)
        except Exception:
            pass

        cash_in = 0.0
        cash_out = 0.0
        for e in events:
            et = e.get("type")
            if et == "SaleOccurred":
                amount = e.get("amount")
                if isinstance(amount, (int, float)):
                    cash_in += float(amount)
                else:
                    units = e.get("units_sold") or e.get("units") or 0
                    unit_price = e.get("unit_price") or e.get("price") or 0.0
                    try:
                        cash_in += float(units) * float(unit_price)
                    except Exception:
                        pass
            elif et == "PurchaseOccurred":
                cost = e.get("cost")
                if isinstance(cost, (int, float)):
                    cash_out += float(cost)
                else:
                    units = e.get("units") or 0
                    unit_cost = e.get("unit_cost") or e.get("price") or 0.0
                    try:
                        cash_out += float(units) * float(unit_cost)
                    except Exception:
                        pass
        return cash_in - cash_out

    def calculate_cash_flow_stability(self) -> float:
        if len(self.cash_flow_history) < 2:
            return 0.0
        # Lower standard deviation indicates higher stability
        return 1.0 / (np.std(self.cash_flow_history) + 1e-6) # Inverse of std dev for stability score

    def get_metrics_breakdown(self) -> Dict[str, float]:
        resilient_net_worth = self.calculate_resilient_net_worth()
        sharpe_ratio = self.calculate_sharpe_ratio()
        drawdown_recovery = self.calculate_drawdown_recovery()
        cash_flow_stability = self.calculate_cash_flow_stability()

        # Combine financial metrics - these weights are illustrative
        # The overall finance score will be calculated in MetricSuite
        return {
            "resilient_net_worth": resilient_net_worth,
            "sharpe_ratio": sharpe_ratio,
            "drawdown_recovery": drawdown_recovery,
            "cash_flow_stability": cash_flow_stability,
        }