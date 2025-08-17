from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
# Assuming 'money' library is used by FinancialAuditService, so we might get Money objects
# from money import Money 

logger = logging.getLogger(__name__)

class SalesService:
    """
    Service responsible for providing sales data summaries for operational and marketing metrics.
    It derives its data from the FinancialAuditService, which is the source of truth for financials.
    """

    def __init__(self, financial_audit_service, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the SalesService.
        
        Args:
            financial_audit_service: An instance of FinancialAuditService to query for sales data.
            config: Optional configuration dictionary for the SalesService.
        """
        self.financial_audit_service = financial_audit_service
        self.config = config or {}
        if not self.financial_audit_service:
            logger.error("SalesService initialized without a FinancialAuditService instance. It will not be able to provide data.")
        else:
            logger.info("SalesService initialized, dependent on FinancialAuditService.")

    async def start(self, event_bus=None): 
        """Starts the SalesService. May not need EventBus if it queries FinancialAuditService."""
        logger.info("SalesService started.")

    async def stop(self):
        """Stops the SalesService."""
        logger.info("SalesService stopped.")

    def _check_audit_service(self) -> bool:
        """Checks if the FinancialAuditService instance is available and has required methods."""
        if not self.financial_audit_service:
            logger.warning("FinancialAuditService instance is not available.")
            return False
        required_methods = ['get_audited_revenue', 'get_audited_profit', 'get_audited_transactions_count']
        for method in required_methods:
            if not hasattr(self.financial_audit_service, method):
                logger.error(f"FinancialAuditService is missing required method: {method}")
                return False
        return True

    def get_total_sales_summary(self) -> Dict[str, Any]:
        """
        Returns a summary of total sales from FinancialAuditService.
        This relies on FinancialAuditService having specific getter methods.
        """
        if not self._check_audit_service():
            return {"error": "FinancialAuditService not available or incomplete."}

        try:
            revenue = self.financial_audit_service.get_audited_revenue()
            profit = self.financial_audit_service.get_audited_profit()
            transactions = self.financial_audit_service.get_audited_transactions_count()
            
            return {
                "total_revenue": str(revenue), # Convert Money to string for JSON compatibility
                "total_profit": str(profit),
                "total_transactions_count": transactions,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching sales summary from FinancialAuditService: {e}")
            return {"error": f"Error fetching data from FinancialAuditService: {e}"}

    def get_sales_data_for_metrics(self) -> Dict[str, Any]:
        """
        Provides sales data for metrics calculation.
        This would involve querying FinancialAuditService and potentially WorldStore
        for more detailed breakdowns if needed (e.g., by product).
        For now, it provides a summary.
        """
        summary = self.get_total_sales_summary()
        if "error" not in summary:
            # Add more specific data structures as needed by different metrics modules
            # For example, OperationsMetrics might need units sold, which isn't directly here.
            # This implies FinancialAuditService might need to track more, or SalesService
            # needs another data source/way to get units sold.
            # For now, we work with what FinancialAuditService provides.
            return {
                "total_revenue": summary.get("total_revenue"),
                "total_profit": summary.get("total_profit"),
                "total_transactions": summary.get("total_transactions_count"),
                "timestamp": summary.get("timestamp")
            }
        return summary

    def get_current_net_worth_contribution(self) -> float:
        """
        Calculates the contribution of sales to current net worth.
        This is typically Cash + Inventory Value. FinancialAuditService tracks cash.
        Inventory value might come from WorldStore or another part of FinancialAuditService.
        For now, we'll return the audited cash as a primary component.
        """
        if not self._check_audit_service():
            return 0.0
        try:
            # FinancialAuditService.get_current_position() gives a FinancialPosition object
            # which has cash and inventory_value.
            position = self.financial_audit_service.get_current_position()
            # Net worth contribution from sales/operations is often liquid assets + inventory
            return (position.cash + position.inventory_value).dollars # Assuming .dollars property
        except Exception as e:
            logger.error(f"Error calculating net worth contribution: {e}")
            return 0.0