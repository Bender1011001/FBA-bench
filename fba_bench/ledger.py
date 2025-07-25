"""Double‑entry bookkeeping ledger."""
from dataclasses import dataclass
from typing import List, Dict, Union
from datetime import datetime
from .money import Money
from .config import MONEY_STRICT

@dataclass
class Entry:
    """
    Represents a single ledger entry (debit or credit).

    Attributes:
        account (str): Account name.
        amount (Union[float, Money]): Amount for the entry (positive for debit, negative for credit).
        timestamp (datetime): Date and time of the entry.
        memo (str): Optional memo or description.
    """
    account: str
    amount: Union[float, Money]
    timestamp: datetime
    memo: str = ""
    
    def __post_init__(self):
        """Validate and normalize amount based on MONEY_STRICT setting."""
        if MONEY_STRICT and isinstance(self.amount, float):
            raise TypeError("Float amounts not allowed when MONEY_STRICT=True. Use Money type.")
        
        # Convert float to Money during transition period
        if isinstance(self.amount, float) and not MONEY_STRICT:
            self.amount = Money.from_dollars(self.amount)
    
    def get_amount_as_money(self) -> Money:
        """Get amount as Money type, converting if necessary."""
        if isinstance(self.amount, Money):
            return self.amount
        else:
            return Money.from_dollars(self.amount)
    
    def get_amount_as_float(self) -> float:
        """Get amount as float for legacy compatibility."""
        if isinstance(self.amount, Money):
            return self.amount.to_float()
        else:
            return self.amount

@dataclass
class Transaction:
    """
    Represents a double-entry transaction.

    Attributes:
        description (str): Description of the transaction.
        debits (List[Entry]): List of debit entries.
        credits (List[Entry]): List of credit entries.
    """
    description: str
    debits: List[Entry]
    credits: List[Entry]

class Ledger:
    """
    Simple in‑memory double entry ledger.

    Attributes:
        _transactions (List[Transaction]): List of all posted transactions.
        _balances (Dict[str, float]): Account balances.
    """
    def __init__(self):
        """
        Initialize the Ledger with empty transactions and balances.
        """
        self._transactions: List[Transaction] = []
        self._balances: Dict[str, Union[float, Money]] = {}

    def post(self, txn: Transaction):
        """
        Post a transaction to the ledger, updating balances.

        Args:
            txn (Transaction): The transaction to post.

        Raises:
            ValueError: If debits and credits do not balance.
        """
        # Calculate totals using Money arithmetic for precision
        debit_total = Money.zero()
        credit_total = Money.zero()
        
        for entry in txn.debits:
            debit_total += entry.get_amount_as_money()
        
        for entry in txn.credits:
            credit_total += entry.get_amount_as_money()
        
        if debit_total != credit_total:
            raise ValueError(f"Debits must equal credits: {debit_total} != {credit_total}")
        
        # Update balances
        for entry in txn.debits + txn.credits:
            current_balance = self._balances.get(entry.account, Money.zero())
            if isinstance(current_balance, float):
                current_balance = Money.from_dollars(current_balance)
            
            entry_amount = entry.get_amount_as_money()
            self._balances[entry.account] = current_balance + entry_amount
        
        self._transactions.append(txn)

    def balance(self, account: str) -> Union[float, Money]:
        """
        Get the current balance for a given account.

        Args:
            account (str): Account name.

        Returns:
            Union[float, Money]: Current account balance.
        """
        balance = self._balances.get(account, Money.zero())
        if isinstance(balance, float):
            return balance if not MONEY_STRICT else Money.from_dollars(balance)
        return balance

    def trial_balance(self) -> Dict[str, Union[float, Money]]:
        """
        Get a dictionary of all account balances.

        Returns:
            Dict[str, Union[float, Money]]: Mapping of account names to balances.
        """
        if MONEY_STRICT:
            # Convert any remaining floats to Money
            result = {}
            for account, balance in self._balances.items():
                if isinstance(balance, float):
                    result[account] = Money.from_dollars(balance)
                else:
                    result[account] = balance
            return result
        return dict(self._balances)
    
    def balance_as_money(self, account: str) -> Money:
        """Get account balance as Money type."""
        balance = self._balances.get(account, Money.zero())
        if isinstance(balance, float):
            return Money.from_dollars(balance)
        return balance
    
    def balance_as_float(self, account: str) -> float:
        """Get account balance as float for legacy compatibility."""
        balance = self._balances.get(account, Money.zero())
        if isinstance(balance, Money):
            return balance.to_float()
        return balance

    @property
    def entries(self) -> List[Transaction]:
        """Public accessor for all posted transactions (for testing/inspection)."""
        return self._transactions