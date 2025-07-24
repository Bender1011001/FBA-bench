"""Double‑entry bookkeeping ledger."""
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime

@dataclass
class Entry:
    """
    Represents a single ledger entry (debit or credit).

    Attributes:
        account (str): Account name.
        amount (float): Amount for the entry (positive for debit, negative for credit).
        timestamp (datetime): Date and time of the entry.
        memo (str): Optional memo or description.
    """
    account: str
    amount: float
    timestamp: datetime
    memo: str = ""

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
        self._balances: Dict[str, float] = {}

    def post(self, txn: Transaction):
        """
        Post a transaction to the ledger, updating balances.

        Args:
            txn (Transaction): The transaction to post.

        Raises:
            ValueError: If debits and credits do not balance.
        """
        debit_total = sum(e.amount for e in txn.debits)
        credit_total = sum(e.amount for e in txn.credits)
        if round(debit_total, 2) != round(credit_total, 2):
            raise ValueError("Debits must equal credits")
        for entry in txn.debits + txn.credits:
            self._balances[entry.account] = self._balances.get(entry.account, 0.0) + entry.amount
        self._transactions.append(txn)

    def balance(self, account: str) -> float:
        """
        Get the current balance for a given account.

        Args:
            account (str): Account name.

        Returns:
            float: Current account balance.
        """
        return self._balances.get(account, 0.0)

    def trial_balance(self) -> Dict[str, float]:
        """
        Get a dictionary of all account balances.

        Returns:
            Dict[str, float]: Mapping of account names to balances.
        """
        return dict(self._balances)

    @property
    def entries(self) -> List[Transaction]:
        """Public accessor for all posted transactions (for testing/inspection)."""
        return self._transactions