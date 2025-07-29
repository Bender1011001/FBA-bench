# metrics/finance_metrics.py
import numpy as np
from typing import List, Dict, Any, Tuple

class FinanceMetrics:
    def __init__(self, financial_audit_service: Any):
        self.financial_audit_service = financial_audit_service
        self.net_worth_history: List[float] = []
        self.cash_flow_history: List[float] = []
        self.shock_net_worth_snapshots: Dict[str, Tuple[float, float, float]] = {} # shock_id: (before, during, after)

    def update(self, current_tick: int, events: List[Dict]):
        # Assuming financial_audit_service gives current net worth and cash flow
        current_net_worth = self.financial_audit_service.get_current_net_worth()
        self.net_worth_history.append(current_net_worth)

        # For a more robust cash flow, we would need detailed ledger analysis from financial_audit.py
        # For now, let's simulate a simple cash flow change based on events if available
        # This is a placeholder and should be replaced with actual cash flow calculation
        cash_inflow = sum(e.get('amount', 0) for e in events if e.get('type') == 'SaleOccurred') # Example
        cash_outflow = sum(e.get('cost', 0) for e in events if e.get('type') == 'PurchaseOccurred') # Example
        current_cash_flow = cash_inflow - cash_outflow
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