"""Tests for ledger migration to Money type with feature flag support."""
import pytest
from datetime import datetime
from fba_bench.ledger import Ledger, Entry, Transaction
from fba_bench.money import Money
from fba_bench import config


def test_ledger_accepts_float_during_transition():
    """Test that ledger accepts float amounts when MONEY_STRICT=False."""
    # Ensure we're in transition mode
    original_strict = config.MONEY_STRICT
    config.MONEY_STRICT = False
    
    try:
        ledger = Ledger()
        now = datetime.now()
        
        # Create transaction with float amounts
        txn = Transaction(
            "Test transaction",
            [Entry("Cash", 100.0, now)],
            [Entry("Revenue", 100.0, now)]
        )
        
        # Should not raise an error
        ledger.post(txn)
        
        # Balance should be accessible
        cash_balance = ledger.balance("Cash")
        assert isinstance(cash_balance, Money)  # Should be converted to Money internally
        assert cash_balance == Money.from_dollars(100.0)
        
    finally:
        config.MONEY_STRICT = original_strict


def test_ledger_rejects_float_when_strict():
    """Test that ledger rejects float amounts when MONEY_STRICT=True."""
    # Enable strict mode
    original_strict = config.MONEY_STRICT
    config.MONEY_STRICT = True
    
    try:
        now = datetime.now()
        
        # Creating Entry with float should raise TypeError
        with pytest.raises(TypeError, match="Float amounts not allowed when MONEY_STRICT=True"):
            Entry("Cash", 100.0, now)
            
    finally:
        config.MONEY_STRICT = original_strict


def test_ledger_accepts_money_type():
    """Test that ledger accepts Money type amounts."""
    ledger = Ledger()
    now = datetime.now()
    
    # Create transaction with Money amounts
    cash_amount = Money.from_dollars("100.00")
    revenue_amount = Money.from_dollars("100.00")
    
    txn = Transaction(
        "Test transaction",
        [Entry("Cash", cash_amount, now)],
        [Entry("Revenue", revenue_amount, now)]
    )
    
    # Should not raise an error
    ledger.post(txn)
    
    # Balance should be Money type
    cash_balance = ledger.balance("Cash")
    assert isinstance(cash_balance, Money)
    assert cash_balance == cash_amount


def test_ledger_exact_arithmetic():
    """Test that ledger performs exact arithmetic with Money types."""
    ledger = Ledger()
    now = datetime.now()
    
    # Create multiple transactions with precise amounts
    amounts = [
        Money.from_dollars("33.33"),
        Money.from_dollars("33.33"),
        Money.from_dollars("33.34")
    ]
    
    for i, amount in enumerate(amounts):
        txn = Transaction(
            f"Transaction {i+1}",
            [Entry("Cash", amount, now)],
            [Entry("Revenue", amount, now)]
        )
        ledger.post(txn)
    
    # Total should be exactly $100.00
    cash_balance = ledger.balance("Cash")
    expected_total = Money.from_dollars("100.00")
    assert cash_balance == expected_total
    
    # Verify trial balance
    trial_bal = ledger.trial_balance()
    assert trial_bal["Cash"] == expected_total
    assert trial_bal["Revenue"] == expected_total


def test_ledger_balance_conversion_methods():
    """Test balance conversion methods for compatibility."""
    ledger = Ledger()
    now = datetime.now()
    
    amount = Money.from_dollars("19.99")
    txn = Transaction(
        "Test transaction",
        [Entry("Cash", amount, now)],
        [Entry("Revenue", amount, now)]
    )
    ledger.post(txn)
    
    # Test Money conversion
    cash_as_money = ledger.balance_as_money("Cash")
    assert isinstance(cash_as_money, Money)
    assert cash_as_money == amount
    
    # Test float conversion for legacy compatibility
    cash_as_float = ledger.balance_as_float("Cash")
    assert isinstance(cash_as_float, float)
    assert abs(cash_as_float - 19.99) < 0.001


def test_ledger_mixed_transaction_validation():
    """Test that transactions with mixed float/Money are handled correctly."""
    original_strict = config.MONEY_STRICT
    config.MONEY_STRICT = False
    
    try:
        ledger = Ledger()
        now = datetime.now()
        
        # Create transaction with mixed types (both should be converted to Money)
        txn = Transaction(
            "Mixed transaction",
            [Entry("Cash", 50.0, now)],  # float
            [Entry("Revenue", Money.from_dollars("50.00"), now)]  # Money
        )
        
        # Should not raise an error - both converted to Money internally
        ledger.post(txn)
        
        # Both balances should be Money type
        cash_balance = ledger.balance("Cash")
        revenue_balance = ledger.balance("Revenue")
        
        assert isinstance(cash_balance, Money)
        assert isinstance(revenue_balance, Money)
        assert cash_balance == revenue_balance
        
    finally:
        config.MONEY_STRICT = original_strict


def test_ledger_precision_preservation():
    """Test that ledger preserves precision with Money types."""
    ledger = Ledger()
    now = datetime.now()
    
    # Use amounts that would cause floating point precision issues
    amount1 = Money.from_dollars("0.1")
    amount2 = Money.from_dollars("0.2")
    
    txn1 = Transaction("Transaction 1", [Entry("Cash", amount1, now)], [Entry("Revenue", amount1, now)])
    txn2 = Transaction("Transaction 2", [Entry("Cash", amount2, now)], [Entry("Revenue", amount2, now)])
    
    ledger.post(txn1)
    ledger.post(txn2)
    
    # Should be exactly 0.30, not 0.30000000000000004
    total_balance = ledger.balance("Cash")
    expected = Money.from_dollars("0.30")
    assert total_balance == expected
    assert total_balance.cents == 30  # Exactly 30 cents


def test_ledger_zero_balance_handling():
    """Test that zero balances are handled correctly."""
    ledger = Ledger()
    
    # Non-existent account should return Money.zero()
    balance = ledger.balance("NonExistent")
    assert isinstance(balance, Money)
    assert balance == Money.zero()
    
    # balance_as_money should also return Money.zero()
    balance_money = ledger.balance_as_money("NonExistent")
    assert isinstance(balance_money, Money)
    assert balance_money == Money.zero()
    
    # balance_as_float should return 0.0
    balance_float = ledger.balance_as_float("NonExistent")
    assert isinstance(balance_float, float)
    assert balance_float == 0.0