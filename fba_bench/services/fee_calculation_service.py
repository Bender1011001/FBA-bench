"""
Fee Calculation Service

Handles all fee calculations and ledger entries for sales transactions.
Extracted from the monolithic tick_day method to improve maintainability.
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Union, Optional

from ..money import Money
from ..fee_engine import FeeEngine
from ..ledger import Entry, Transaction


class FeeCalculationService:
    """Service for calculating fees and creating ledger entries for sales transactions."""
    
    def __init__(self, fee_engine: FeeEngine, ledger):
        self.fee_engine = fee_engine
        self.ledger = ledger
    
    def calculate_and_record_sale(
        self,
        asin: str,
        product,
        units_sold: int,
        revenue: Money,
        trust_fee_multiplier: float,
        selling_plan_fee: Money,
        current_date: datetime,
        **fee_params
    ) -> Dict[str, Union[Money, float]]:
        """
        Calculate all fees for a sale and record the transaction in the ledger.
        
        Args:
            asin: Product ASIN
            product: Product object
            units_sold: Number of units sold
            revenue: Total revenue from sale
            trust_fee_multiplier: Trust-based fee multiplier
            selling_plan_fee: Selling plan fee
            current_date: Current simulation date
            **fee_params: Additional fee calculation parameters
            
        Returns:
            Dictionary containing fee breakdown and totals
        """
        # Calculate all fees using the fee engine
        fees = self.fee_engine.total_fees(
            category=product.category,
            price=product.price,
            size_tier=product.size_tier,
            size=product.size,
            **fee_params
        )
        
        # Calculate total fees with trust multiplier
        base_fees = Money.from_dollars(fees["total"]) if isinstance(fees["total"], (int, float)) else fees["total"]
        total_fees = (base_fees * units_sold * Decimal(str(trust_fee_multiplier))) + selling_plan_fee
        
        # Calculate COGS
        cogs = units_sold * product.cost
        
        # Create ledger entries
        self._record_sale_transaction(
            asin=asin,
            revenue=revenue,
            total_fees=total_fees,
            cogs=cogs,
            timestamp=current_date
        )
        
        return {
            "fees": fees,
            "total_fees": total_fees,
            "cogs": cogs,
            "revenue": revenue,
            "net_profit": revenue - total_fees - cogs
        }
    
    def _record_sale_transaction(
        self,
        asin: str,
        revenue: Money,
        total_fees: Money,
        cogs: Money,
        timestamp: datetime
    ) -> None:
        """Record a sale transaction in the ledger with proper double-entry accounting."""
        
        # Prepare ledger entries
        debits = [
            Entry("COGS", cogs, timestamp),        # Cost of goods sold expense
            Entry("Fees", total_fees, timestamp),  # Fees expense
        ]
        credits = [
            Entry("Revenue", revenue, timestamp),   # Gross revenue earned
            Entry("Inventory", cogs, timestamp),    # Inventory asset reduced
        ]
        
        # Handle cash flow based on whether net is positive or negative
        net_cash = revenue - total_fees
        if net_cash >= Money.zero():
            # Positive cash flow: debit Cash (asset increase)
            debits.insert(0, Entry("Cash", net_cash, timestamp))
        else:
            # Negative cash flow: credit Cash (asset decrease) with positive amount
            credits.insert(0, Entry("Cash", -net_cash, timestamp))
        
        # Post balanced transaction with all positive amounts
        self.ledger.post(Transaction(
            f"Sales and fees for {asin}",
            debits=debits,
            credits=credits
        ))
    
    def calculate_trust_fee_multiplier(self, trust_score: float) -> float:
        """
        Calculate fee multiplier based on seller trust score.
        Lower trust = higher fees as penalty.
        """
        if trust_score >= 0.9:
            return 1.0  # No penalty for high trust
        elif trust_score >= 0.7:
            return 1.1  # 10% penalty for medium trust
        elif trust_score >= 0.5:
            return 1.25  # 25% penalty for low trust
        else:
            return 1.5  # 50% penalty for very low trust
    
    def calculate_selling_plan_fee(self, selling_plan: str, revenue: Money) -> Money:
        """Calculate selling plan specific fees."""
        if selling_plan == "Individual":
            return Money.from_dollars(0.99)  # $0.99 per item for Individual plan
        elif selling_plan == "Professional":
            return Money.zero()  # No per-item fee for Professional plan
        else:
            return Money.zero()  # Default to no fee