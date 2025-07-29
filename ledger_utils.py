"""Utility functions for ledger analysis and accounting calculations."""
from decimal import Decimal
from typing import Dict, Tuple, Optional, Union
import hashlib
import json
from money import Money


def _balance_to_decimal(balance: Union[float, Money]) -> Decimal:
    """Convert balance (float or Money) to Decimal for calculations."""
    if isinstance(balance, Money):
        return balance.to_decimal()
    else:
        # Handle float/int for backward compatibility
        return Decimal(str(balance))


def balance_sheet(ledger) -> Dict[str, Decimal]:
    """Generate balance sheet from ledger entries."""
    return balance_sheet_from_ledger(ledger)


def balance_sheet_from_ledger(ledger) -> Dict[str, Decimal]:
    """Generate balance sheet from ledger with proper account classification."""
    balances = {}
    
    # Get all account balances
    trial_balances = ledger.trial_balance()
    
    # Classify accounts into balance sheet categories
    for account, balance in trial_balances.items():
        decimal_balance = _balance_to_decimal(balance)
        
        # Asset accounts (normal debit balance)
        if account in ["Cash", "Inventory", "Accounts Receivable", "Prepaid Expenses"]:
            balances[account] = decimal_balance
        
        # Liability accounts (normal credit balance, but stored as positive in our system)
        elif account in ["Accounts Payable", "Accrued Liabilities", "Notes Payable"]:
            balances[account] = decimal_balance
        
        # Equity accounts (normal credit balance)
        elif account in ["Equity", "Retained Earnings"]:
            balances[account] = decimal_balance
        
        # Revenue and expense accounts are not part of balance sheet
        # They flow through to retained earnings
    
    return balances


def income_statement(ledger, start_tick: int = 0, end_tick: Optional[int] = None) -> Dict[str, Decimal]:
    """Generate income statement from ledger entries for specified period."""
    return income_statement_from_ledger(ledger, start_tick, end_tick)


def income_statement_from_ledger(ledger, start_tick: int = 0, end_tick: Optional[int] = None) -> Dict[str, Decimal]:
    """Generate income statement with revenue and expense classification."""
    statement = {}
    
    # Get all account balances
    trial_balances = ledger.trial_balance()
    
    total_revenue = Decimal("0")
    total_expenses = Decimal("0")
    
    for account, balance in trial_balances.items():
        decimal_balance = _balance_to_decimal(balance)
        
        # Revenue accounts (normal credit balance, stored as positive)
        if account in ["Revenue", "Sales", "Interest Income"]:
            statement[account] = decimal_balance
            total_revenue += decimal_balance
        
        # Expense accounts (normal debit balance, but stored as negative in our system)
        elif account in ["COGS", "Fees", "Operating Expenses", "Interest Expense"]:
            # Convert to positive for income statement display
            expense_amount = abs(decimal_balance)
            statement[account] = expense_amount
            total_expenses += expense_amount
    
    # Calculate totals
    statement["Total Revenue"] = total_revenue
    statement["Total Expenses"] = total_expenses
    statement["Net Income"] = total_revenue - total_expenses
    
    return statement


def trial_balance(ledger, tick: Optional[int] = None) -> Tuple[Decimal, Decimal]:
    """Calculate trial balance returning (total_debits, total_credits)."""
    balances = ledger.trial_balance()
    
    total_debits = Decimal("0")
    total_credits = Decimal("0")
    
    for account, balance in balances.items():
        decimal_balance = _balance_to_decimal(balance)
        
        # Account type aware trial balance calculation:
        # - Assets/Expenses: Positive balances are debits, negative are credits
        # - Liabilities/Equity/Revenue: Positive balances are credits, negative are debits
        if account in ["Equity", "Revenue", "Liabilities", "Accounts Payable", "Accrued Liabilities", "Notes Payable"]:
            # For credit-normal accounts: positive balance = credit, negative balance = debit
            if decimal_balance >= 0:
                total_credits += decimal_balance
            else:
                total_debits += abs(decimal_balance)
        else:
            # For debit-normal accounts: positive balance = debit, negative balance = credit
            if decimal_balance >= 0:
                total_debits += decimal_balance
            else:
                total_credits += abs(decimal_balance)
    
    return total_debits, total_credits


def accounting_identity_gap(bs: Dict[str, Decimal]) -> Decimal:
    """Calculate the gap in the accounting identity A = L + E."""
    assets = sum(v for k, v in bs.items() if k in ["Cash", "Inventory", "Accounts Receivable", "Prepaid Expenses"])
    liabilities = sum(v for k, v in bs.items() if k in ["Accounts Payable", "Accrued Liabilities", "Notes Payable"])
    equity = sum(v for k, v in bs.items() if k in ["Equity", "Retained Earnings"])
    
    return assets - (liabilities + equity)


def equity_delta_ex_owner(bs_start: Dict[str, Decimal], bs_end: Dict[str, Decimal], 
                         owner_contrib: Decimal, owner_dist: Decimal) -> Decimal:
    """Calculate equity change excluding owner contributions/distributions."""
    equity_start = sum(v for k, v in bs_start.items() if k in ["Equity", "Retained Earnings"])
    equity_end = sum(v for k, v in bs_end.items() if k in ["Equity", "Retained Earnings"])
    
    return (equity_end - equity_start) - (owner_contrib - owner_dist)


def hash_ledger_slice(ledger, start_tick: int, end_tick: Optional[int] = None) -> str:
    """Generate hash of ledger entries for specified tick range."""
    # For now, hash all transactions - in a real implementation this would filter by tick
    transactions = []
    
    for txn in ledger.entries:
        txn_data = {
            "description": txn.description,
            "debits": [(e.account, str(e.amount), e.timestamp.isoformat()) for e in txn.debits],
            "credits": [(e.account, str(e.amount), e.timestamp.isoformat()) for e in txn.credits]
        }
        transactions.append(txn_data)
    
    # Sort for deterministic hashing
    transactions.sort(key=lambda x: x["description"])
    
    hash_data = json.dumps(transactions, sort_keys=True)
    return hashlib.sha256(hash_data.encode()).hexdigest()[:16]


def hash_rng_state(rng) -> str:
    """Generate hash of RNG state for determinism checking."""
    state = rng.getstate()
    # Convert state to string representation for hashing
    state_str = str(state)
    return hashlib.sha256(state_str.encode()).hexdigest()[:16]


def hash_inventory_state(inventory_manager) -> str:
    """Generate hash of inventory state for drift detection."""
    inventory_data = []
    
    for sku, batches in inventory_manager._batches.items():
        for batch in batches:
            batch_data = (
                sku,
                batch.quantity,
                int(batch.cost_per_unit.cents) if hasattr(batch.cost_per_unit, 'cents') else int(batch.cost_per_unit * 100)  # Convert to cents for deterministic hashing
            )
            inventory_data.append(batch_data)
    
    # Sort for deterministic hashing
    inventory_data.sort()
    
    hash_data = json.dumps(inventory_data, sort_keys=True)
    return hashlib.sha256(hash_data.encode()).hexdigest()[:16]