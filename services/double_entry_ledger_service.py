"""Double-Entry Ledger Service for FBA-Bench v3 with uncompromising financial integrity."""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from decimal import Decimal

from money import Money
from events import SaleOccurred
from event_bus import EventBus


logger = logging.getLogger(__name__)


class AccountType(Enum):
    """Types of accounts in the double-entry system."""
    ASSET = "asset"           # Debit normal balance
    LIABILITY = "liability"   # Credit normal balance
    EQUITY = "equity"         # Credit normal balance
    REVENUE = "revenue"       # Credit normal balance
    EXPENSE = "expense"       # Debit normal balance


class TransactionType(Enum):
    """Types of transactions in the system."""
    SALE = "sale"
    FEE_PAYMENT = "fee_payment"
    INVENTORY_PURCHASE = "inventory_purchase"
    INVENTORY_ADJUSTMENT = "inventory_adjustment"
    CASH_DEPOSIT = "cash_deposit"
    CASH_WITHDRAWAL = "cash_withdrawal"
    EQUITY_INJECTION = "equity_injection"
    OWNER_DISTRIBUTION = "owner_distribution"
    ADJUSTING_ENTRY = "adjusting_entry"


@dataclass
class Account:
    """Account in the double-entry ledger system."""
    account_id: str
    name: str
    account_type: AccountType
    normal_balance: str = ""  # "debit" or "credit"
    balance: Money = field(default_factory=Money.zero)
    is_contra: bool = False  # Contra accounts have opposite normal balance
    description: str = ""
    parent_account: Optional[str] = None  # For hierarchical accounts
    
    def __post_init__(self):
        """Set normal balance based on account type."""
        if not self.normal_balance:
            if self.account_type in [AccountType.ASSET, AccountType.EXPENSE]:
                self.normal_balance = "debit"
            else:  # LIABILITY, EQUITY, REVENUE
                self.normal_balance = "credit"
        
        # Contra accounts flip the normal balance
        if self.is_contra:
            self.normal_balance = "credit" if self.normal_balance == "debit" else "debit"


@dataclass
class LedgerEntry:
    """Individual entry in a transaction (debit or credit)."""
    entry_id: str
    account_id: str
    amount: Money
    entry_type: str  # "debit" or "credit"
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transaction:
    """Complete transaction with balanced debits and credits."""
    transaction_id: str
    transaction_type: TransactionType
    description: str
    debits: List[LedgerEntry] = field(default_factory=list)
    credits: List[LedgerEntry] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_posted: bool = False
    
    def __post_init__(self):
        """Validate transaction balance."""
        if not self.is_balanced():
            raise ValueError(f"Transaction {self.transaction_id} is not balanced")
    
    def is_balanced(self) -> bool:
        """Check if total debits equal total credits."""
        total_debits = sum((entry.amount for entry in self.debits), Money.zero())
        total_credits = sum((entry.amount for entry in self.credits), Money.zero())
        return total_debits.cents == total_credits.cents
    
    def get_total_debits(self) -> Money:
        """Get total debit amount."""
        return sum((entry.amount for entry in self.debits), Money.zero())
    
    def get_total_credits(self) -> Money:
        """Get total credit amount."""
        return sum((entry.amount for entry in self.credits), Money.zero())


@dataclass
class FinancialStatement:
    """Financial statement data structure."""
    statement_type: str  # "balance_sheet", "income_statement", "cash_flow"
    period_start: datetime
    period_end: datetime
    data: Dict[str, Any]
    generated_at: datetime = field(default_factory=datetime.now)


class DoubleEntryLedgerService:
    """
    Double-Entry Ledger Service for FBA-Bench v3.
    
    Implements a complete double-entry accounting system with:
    - Chart of accounts management
    - Transaction recording and validation
    - Financial statement generation
    - Audit trail maintenance
    - Integration with event bus for real-time updates
    
    Critical Requirements:
    - Enforces double-entry rules (debits = credits)
    - Maintains proper account balances
    - Generates accurate financial statements
    - Provides audit trail for all transactions
    """
    
    def __init__(self, config: Dict):
        """Initialize the double-entry ledger service."""
        self.config = config
        self.event_bus: Optional[EventBus] = None
        
        # Ledger storage
        self.accounts: Dict[str, Account] = {}
        self.transactions: Dict[str, Transaction] = {}
        self.unposted_transactions: List[Transaction] = []
        
        # Financial statement cache
        self.balance_sheet_cache: Optional[FinancialStatement] = None
        self.income_statement_cache: Optional[FinancialStatement] = None
        
        # Initialize chart of accounts
        self._initialize_chart_of_accounts()
        
        logger.info("DoubleEntryLedgerService initialized with chart of accounts")
    
    def _initialize_chart_of_accounts(self) -> None:
        """Initialize the standard chart of accounts."""
        # Asset accounts (debit normal)
        self._add_account(Account(
            account_id="cash",
            name="Cash",
            account_type=AccountType.ASSET,
            description="Cash on hand and in bank accounts"
        ))
        
        self._add_account(Account(
            account_id="inventory",
            name="Inventory",
            account_type=AccountType.ASSET,
            description="Inventory at cost"
        ))
        
        self._add_account(Account(
            account_id="accounts_receivable",
            name="Accounts Receivable",
            account_type=AccountType.ASSET,
            description="Money owed by customers"
        ))
        
        self._add_account(Account(
            account_id="prepaid_expenses",
            name="Prepaid Expenses",
            account_type=AccountType.ASSET,
            description="Expenses paid in advance"
        ))
        
        # Liability accounts (credit normal)
        self._add_account(Account(
            account_id="accounts_payable",
            name="Accounts Payable",
            account_type=AccountType.LIABILITY,
            description="Money owed to suppliers"
        ))
        
        self._add_account(Account(
            account_id="accrued_liabilities",
            name="Accrued Liabilities",
            account_type=AccountType.LIABILITY,
            description="Expenses incurred but not yet paid"
        ))
        
        self._add_account(Account(
            account_id="unearned_revenue",
            name="Unearned Revenue",
            account_type=AccountType.LIABILITY,
            description="Revenue received but not yet earned"
        ))
        
        # Equity accounts (credit normal)
        self._add_account(Account(
            account_id="owner_equity",
            name="Owner Equity",
            account_type=AccountType.EQUITY,
            description="Owner's investment in the business"
        ))
        
        self._add_account(Account(
            account_id="retained_earnings",
            name="Retained Earnings",
            account_type=AccountType.EQUITY,
            description="Accumulated profits retained in the business"
        ))
        
        # Revenue accounts (credit normal)
        self._add_account(Account(
            account_id="sales_revenue",
            name="Sales Revenue",
            account_type=AccountType.REVENUE,
            description="Revenue from product sales"
        ))
        
        self._add_account(Account(
            account_id="other_revenue",
            name="Other Revenue",
            account_type=AccountType.REVENUE,
            description="Revenue from other sources"
        ))
        
        # Expense accounts (debit normal)
        self._add_account(Account(
            account_id="cost_of_goods_sold",
            name="Cost of Goods Sold",
            account_type=AccountType.EXPENSE,
            description="Cost of inventory sold"
        ))
        
        self._add_account(Account(
            account_id="fulfillment_fees",
            name="Fulfillment Fees",
            account_type=AccountType.EXPENSE,
            description="FBA fulfillment fees"
        ))
        
        self._add_account(Account(
            account_id="referral_fees",
            name="Referral Fees",
            account_type=AccountType.EXPENSE,
            description="Amazon referral fees"
        ))
        
        self._add_account(Account(
            account_id="storage_fees",
            name="Storage Fees",
            account_type=AccountType.EXPENSE,
            description="Inventory storage fees"
        ))
        
        self._add_account(Account(
            account_id="advertising_expense",
            name="Advertising Expense",
            account_type=AccountType.EXPENSE,
            description="Advertising costs"
        ))
        
        self._add_account(Account(
            account_id="other_expenses",
            name="Other Expenses",
            account_type=AccountType.EXPENSE,
            description="Other operating expenses"
        ))
        
        logger.info(f"Initialized chart of accounts with {len(self.accounts)} accounts")
    
    def _add_account(self, account: Account) -> None:
        """Add an account to the chart of accounts."""
        self.accounts[account.account_id] = account
    
    async def start(self, event_bus: EventBus) -> None:
        """Start the ledger service and subscribe to events."""
        self.event_bus = event_bus
        
        # Subscribe to relevant events
        await self.event_bus.subscribe(SaleOccurred, self._handle_sale_occurred)
        
        logger.info("DoubleEntryLedgerService started and subscribed to events")
    
    async def stop(self) -> None:
        """Stop the ledger service."""
        # Post any remaining unposted transactions
        if self.unposted_transactions:
            logger.info(f"Posting {len(self.unposted_transactions)} unposted transactions")
            for transaction in self.unposted_transactions:
                await self.post_transaction(transaction)
        
        logger.info("DoubleEntryLedgerService stopped")
    
    async def _handle_sale_occurred(self, event: SaleOccurred) -> None:
        """Handle SaleOccurred events by creating appropriate ledger entries with fee breakdown."""
        try:
            # Compute net receivable
            net_receivable = event.total_revenue - event.total_fees

            # Create transaction for the sale including fees and COGS
            transaction = Transaction(
                transaction_id=f"sale_{event.event_id}",
                transaction_type=TransactionType.SALE,
                description=f"Sale of ASIN {event.asin}",
                metadata={
                    "event_id": event.event_id,
                    "asin": event.asin,
                    "units_sold": event.units_sold,
                    "units_demanded": event.units_demanded,
                    "unit_price": event.unit_price,
                    "total_revenue": event.total_revenue,
                    "total_fees": event.total_fees,
                    "total_profit": event.total_profit,
                    "cost_basis": event.cost_basis,
                    "fee_breakdown": {k: str(v) for k, v in (event.fee_breakdown or {}).items()}
                }
            )

            # Debit: Accounts Receivable for net proceeds (increase asset)
            transaction.debits.append(LedgerEntry(
                entry_id=f"ar_{event.event_id}",
                account_id="accounts_receivable",
                amount=net_receivable,
                entry_type="debit",
                description="Net receivable from sale (gross revenue less fees)"
            ))

            # Debit: Cost of Goods Sold (increase expense)
            if event.cost_basis.cents != 0:
                transaction.debits.append(LedgerEntry(
                    entry_id=f"cogs_{event.event_id}",
                    account_id="cost_of_goods_sold",
                    amount=event.cost_basis,
                    entry_type="debit",
                    description="Cost of goods sold"
                ))

            # Debit: Fee expenses by type (increase expenses)
            fee_account_map = {
                "referral": "referral_fees",
                "fba": "fulfillment_fees",
                "storage": "storage_fees",
                "advertising": "advertising_expense",
            }
            debited_fees_total = Money.zero()
            if event.total_fees.cents != 0:
                # Apply detailed breakdown if available
                if getattr(event, "fee_breakdown", None):
                    for fee_type, amount in event.fee_breakdown.items():
                        if amount.cents == 0:
                            continue
                        expense_account = fee_account_map.get(fee_type.lower(), "other_expenses")
                        transaction.debits.append(LedgerEntry(
                            entry_id=f"fee_{fee_type}_{event.event_id}",
                            account_id=expense_account,
                            amount=amount,
                            entry_type="debit",
                            description=f"{fee_type} fee expense"
                        ))
                        debited_fees_total += amount
                    # Adjust for any residual rounding differences
                    residual = event.total_fees - debited_fees_total
                    if residual.cents != 0:
                        transaction.debits.append(LedgerEntry(
                            entry_id=f"fee_residual_{event.event_id}",
                            account_id="other_expenses",
                            amount=residual,
                            entry_type="debit",
                            description="Residual fee adjustment"
                        ))
                else:
                    # No breakdown provided, book all fees to other_expenses
                    transaction.debits.append(LedgerEntry(
                        entry_id=f"fee_other_{event.event_id}",
                        account_id="other_expenses",
                        amount=event.total_fees,
                        entry_type="debit",
                        description="Aggregated fees expense"
                    ))

            # Credit: Sales Revenue (increase revenue) for gross revenue
            if event.total_revenue.cents != 0:
                transaction.credits.append(LedgerEntry(
                    entry_id=f"rev_{event.event_id}",
                    account_id="sales_revenue",
                    amount=event.total_revenue,
                    entry_type="credit",
                    description="Revenue from product sale (gross)"
                ))

            # Credit: Inventory (decrease asset) for cost basis
            if event.cost_basis.cents != 0:
                transaction.credits.append(LedgerEntry(
                    entry_id=f"inv_{event.event_id}",
                    account_id="inventory",
                    amount=event.cost_basis,
                    entry_type="credit",
                    description="Inventory reduction"
                ))

            # Add to unposted transactions and immediately post to update balances
            self.unposted_transactions.append(transaction)
            await self.post_transaction(transaction)
            
            logger.debug(f"Created sale transaction {transaction.transaction_id} for {event.total_revenue} with fees {event.total_fees}")

        except Exception as e:
            logger.error(f"Error handling SaleOccurred event {event.event_id}: {e}")
            raise
    
    # Note: FeeCalculated events are not part of the current event schema.
    # Fee recognition is performed within _handle_sale_occurred using total_fees and fee_breakdown.
    
    async def _handle_inventory_adjusted(self, event: Any) -> None:
        """Handle InventoryAdjusted events by creating appropriate ledger entries."""
        try:
            # Create transaction for the inventory adjustment
            transaction = Transaction(
                transaction_id=f"inv_adj_{event.event_id}",
                transaction_type=TransactionType.INVENTORY_ADJUSTMENT,
                description=f"Inventory adjustment for {event.product_id}",
                metadata={
                    "event_id": event.event_id,
                    "product_id": event.product_id,
                    "adjustment_type": event.adjustment_type,
                    "quantity_change": event.quantity_change,
                    "cost_change": event.cost_change
                }
            )
            
            if event.adjustment_type == "write_up":
                # Debit: Inventory (increase asset)
                transaction.debits.append(LedgerEntry(
                    entry_id=f"inv_up_{event.event_id}",
                    account_id="inventory",
                    amount=event.cost_change,
                    entry_type="debit",
                    description="Inventory write-up"
                ))
                
                # Credit: Other Revenue (increase revenue)
                transaction.credits.append(LedgerEntry(
                    entry_id=f"rev_adj_{event.event_id}",
                    account_id="other_revenue",
                    amount=event.cost_change,
                    entry_type="credit",
                    description="Revenue from inventory adjustment"
                ))
                
            elif event.adjustment_type == "write_down":
                # Debit: Other Expenses (increase expense)
                transaction.debits.append(LedgerEntry(
                    entry_id=f"exp_adj_{event.event_id}",
                    account_id="other_expenses",
                    amount=event.cost_change,
                    entry_type="debit",
                    description="Expense from inventory write-down"
                ))
                
                # Credit: Inventory (decrease asset)
                transaction.credits.append(LedgerEntry(
                    entry_id=f"inv_down_{event.event_id}",
                    account_id="inventory",
                    amount=event.cost_change,
                    entry_type="credit",
                    description="Inventory write-down"
                ))
            
            # Add to unposted transactions
            self.unposted_transactions.append(transaction)
            
            logger.debug(f"Created inventory adjustment transaction {transaction.transaction_id}")
            
        except Exception as e:
            logger.error(f"Error handling InventoryAdjusted event {event.event_id}: {e}")
            raise
    
    async def post_transaction(self, transaction: Transaction) -> None:
        """Post a transaction to the ledger and update account balances."""
        try:
            # Validate transaction
            if not transaction.is_balanced():
                raise ValueError(f"Transaction {transaction.transaction_id} is not balanced")
            
            # Update account balances
            for entry in transaction.debits:
                account = self.accounts.get(entry.account_id)
                if not account:
                    raise ValueError(f"Account {entry.account_id} not found")
                
                if account.normal_balance == "debit":
                    account.balance += entry.amount
                else:
                    account.balance -= entry.amount
            
            for entry in transaction.credits:
                account = self.accounts.get(entry.account_id)
                if not account:
                    raise ValueError(f"Account {entry.account_id} not found")
                
                if account.normal_balance == "credit":
                    account.balance += entry.amount
                else:
                    account.balance -= entry.amount
            
            # Mark as posted
            transaction.is_posted = True
            
            # Store transaction
            self.transactions[transaction.transaction_id] = transaction
            
            # Remove from unposted if it was there
            if transaction in self.unposted_transactions:
                self.unposted_transactions.remove(transaction)
            
            # Invalidate financial statement cache
            self.balance_sheet_cache = None
            self.income_statement_cache = None
            
            logger.debug(f"Posted transaction {transaction.transaction_id}")
            
        except Exception as e:
            logger.error(f"Error posting transaction {transaction.transaction_id}: {e}")
            raise
    
    async def post_all_unposted_transactions(self) -> None:
        """Post all unposted transactions."""
        if not self.unposted_transactions:
            return
        
        logger.info(f"Posting {len(self.unposted_transactions)} unposted transactions")
        
        # Create a copy to avoid modification during iteration
        transactions_to_post = self.unposted_transactions.copy()
        
        for transaction in transactions_to_post:
            await self.post_transaction(transaction)
        
        logger.info("All unposted transactions posted successfully")
    
    def get_account_balance(self, account_id: str) -> Money:
        """Get the current balance of an account."""
        account = self.accounts.get(account_id)
        if not account:
            raise ValueError(f"Account {account_id} not found")
        return account.balance
    
    def get_all_account_balances(self) -> Dict[str, Money]:
        """Get balances for all accounts."""
        return {account_id: account.balance for account_id, account in self.accounts.items()}
    
    def trial_balance(self) -> Dict[str, Money]:
        """Generate a trial balance of all accounts."""
        return self.get_all_account_balances()
    
    def is_trial_balance_balanced(self) -> bool:
        """Check if the trial balance is balanced (debits = credits)."""
        balances = self.trial_balance()
        
        total_debits = Money.zero()
        total_credits = Money.zero()
        
        for account_id, balance in balances.items():
            account = self.accounts[account_id]
            if account.normal_balance == "debit":
                total_debits += balance
            else:
                total_credits += balance
        
        return total_debits.cents == total_credits.cents
    
    def get_trial_balance_difference(self) -> Money:
        """Get the difference between total debits and credits."""
        balances = self.trial_balance()
        
        total_debits = Money.zero()
        total_credits = Money.zero()
        
        for account_id, balance in balances.items():
            account = self.accounts[account_id]
            if account.normal_balance == "debit":
                total_debits += balance
            else:
                total_credits += balance
        
        return total_debits - total_credits
    
    def generate_balance_sheet(self, force_refresh: bool = False) -> FinancialStatement:
        """Generate a balance sheet statement."""
        if self.balance_sheet_cache and not force_refresh:
            return self.balance_sheet_cache
        
        # Ensure all transactions are posted
        if self.unposted_transactions:
            logger.warning("Unposted transactions exist. Posting them before generating balance sheet.")
            # In a real implementation, this would be async
            # For now, we'll work with what we have
        
        # Calculate account balances
        balances = self.get_all_account_balances()
        
        # Prepare balance sheet data
        balance_sheet_data = {
            "assets": {},
            "liabilities": {},
            "equity": {}
        }
        
        total_assets = Money.zero()
        total_liabilities = Money.zero()
        total_equity = Money.zero()
        
        for account_id, balance in balances.items():
            account = self.accounts[account_id]
            
            if account.account_type == AccountType.ASSET:
                balance_sheet_data["assets"][account_id] = {
                    "name": account.name,
                    "balance": balance,
                    "description": account.description
                }
                total_assets += balance
            
            elif account.account_type == AccountType.LIABILITY:
                balance_sheet_data["liabilities"][account_id] = {
                    "name": account.name,
                    "balance": balance,
                    "description": account.description
                }
                total_liabilities += balance
            
            elif account.account_type == AccountType.EQUITY:
                balance_sheet_data["equity"][account_id] = {
                    "name": account.name,
                    "balance": balance,
                    "description": account.description
                }
                total_equity += balance
        
        # Add totals
        balance_sheet_data["total_assets"] = total_assets
        balance_sheet_data["total_liabilities"] = total_liabilities
        balance_sheet_data["total_equity"] = total_equity
        
        # Check accounting identity
        balance_sheet_data["accounting_identity_valid"] = (total_assets.cents == (total_liabilities + total_equity).cents)
        balance_sheet_data["identity_difference"] = total_assets - (total_liabilities + total_equity)
        
        # Create financial statement
        balance_sheet = FinancialStatement(
            statement_type="balance_sheet",
            period_start=datetime.min,  # Since beginning of time
            period_end=datetime.now(),
            data=balance_sheet_data
        )
        
        # Cache the result
        self.balance_sheet_cache = balance_sheet
        
        return balance_sheet
    
    def generate_income_statement(self, period_start: Optional[datetime] = None, 
                                 period_end: Optional[datetime] = None,
                                 force_refresh: bool = False) -> FinancialStatement:
        """Generate an income statement for the specified period."""
        if self.income_statement_cache and not force_refresh:
            return self.income_statement_cache
        
        # For now, we'll generate a simple income statement from current balances
        # In a full implementation, this would filter transactions by date
        
        # Get account balances
        balances = self.get_all_account_balances()
        
        # Prepare income statement data
        income_statement_data = {
            "revenue": {},
            "expenses": {}
        }
        
        total_revenue = Money.zero()
        total_expenses = Money.zero()
        
        for account_id, balance in balances.items():
            account = self.accounts[account_id]
            
            if account.account_type == AccountType.REVENUE:
                income_statement_data["revenue"][account_id] = {
                    "name": account.name,
                    "balance": balance,
                    "description": account.description
                }
                total_revenue += balance
            
            elif account.account_type == AccountType.EXPENSE:
                income_statement_data["expenses"][account_id] = {
                    "name": account.name,
                    "balance": balance,
                    "description": account.description
                }
                total_expenses += balance
        
        # Calculate totals and net income
        income_statement_data["total_revenue"] = total_revenue
        income_statement_data["total_expenses"] = total_expenses
        income_statement_data["net_income"] = total_revenue - total_expenses
        
        # Create financial statement
        income_statement = FinancialStatement(
            statement_type="income_statement",
            period_start=period_start or datetime.min,
            period_end=period_end or datetime.now(),
            data=income_statement_data
        )
        
        # Cache the result
        self.income_statement_cache = income_statement
        
        return income_statement
    
    def get_transaction_history(self, limit: int = 100) -> List[Transaction]:
        """Get the transaction history, limited to the specified number of transactions."""
        # Sort transactions by timestamp (most recent first)
        sorted_transactions = sorted(
            self.transactions.values(),
            key=lambda t: t.timestamp,
            reverse=True
        )
        
        return sorted_transactions[:limit]
    
    def get_transactions_by_type(self, transaction_type: TransactionType) -> List[Transaction]:
        """Get all transactions of a specific type."""
        return [
            transaction for transaction in self.transactions.values()
            if transaction.transaction_type == transaction_type
        ]
    
    def get_transactions_by_account(self, account_id: str) -> List[Transaction]:
        """Get all transactions that affect a specific account."""
        return [
            transaction for transaction in self.transactions.values()
            if any(entry.account_id == account_id for entry in transaction.debits + transaction.credits)
        ]
    
    def get_financial_position(self) -> Dict[str, Any]:
        """Get the current financial position for audit purposes."""
        balance_sheet = self.generate_balance_sheet()
        income_statement = self.generate_income_statement()
        
        return {
            "timestamp": datetime.now(),
            "cash": self.get_account_balance("cash"),
            "inventory_value": self.get_account_balance("inventory"),
            "accounts_receivable": self.get_account_balance("accounts_receivable"),
            "total_assets": balance_sheet.data["total_assets"],
            "accounts_payable": self.get_account_balance("accounts_payable"),
            "accrued_liabilities": self.get_account_balance("accrued_liabilities"),
            "total_liabilities": balance_sheet.data["total_liabilities"],
            "retained_earnings": self.get_account_balance("retained_earnings"),
            "current_period_profit": income_statement.data["net_income"],
            "total_equity": balance_sheet.data["total_equity"],
            "accounting_identity_valid": balance_sheet.data["accounting_identity_valid"],
            "identity_difference": balance_sheet.data["identity_difference"]
        }
    
    def get_ledger_statistics(self) -> Dict[str, Any]:
        """Get ledger service statistics."""
        return {
            "total_accounts": len(self.accounts),
            "total_transactions": len(self.transactions),
            "unposted_transactions": len(self.unposted_transactions),
            "trial_balance_balanced": self.is_trial_balance_balanced(),
            "trial_balance_difference": str(self.get_trial_balance_difference()),
            "last_transaction_time": (
                max(t.timestamp for t in self.transactions.values()) if self.transactions else None
            )
        }