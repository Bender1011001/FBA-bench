"""Property-based tests for Money class arithmetic laws and invariants."""
import pytest
from hypothesis import given, strategies as st, assume
from decimal import Decimal
from fba_bench.money import Money, USD, EUR


# Strategies for generating test data
cents_strategy = st.integers(min_value=-10**9, max_value=10**9)
positive_cents_strategy = st.integers(min_value=1, max_value=10**9)
currency_strategy = st.sampled_from(["USD", "EUR", "GBP", "JPY"])
decimal_strategy = st.decimals(min_value=-10000, max_value=10000, places=2)
int_multiplier_strategy = st.integers(min_value=-1000, max_value=1000)


@given(cents_strategy, cents_strategy)
def test_money_addition_commutativity(a, b):
    """Test that a + b == b + a for Money."""
    money_a = Money(a)
    money_b = Money(b)
    assert money_a + money_b == money_b + money_a


@given(cents_strategy, cents_strategy, cents_strategy)
def test_money_addition_associativity(a, b, c):
    """Test that (a + b) + c == a + (b + c) for Money."""
    money_a = Money(a)
    money_b = Money(b)
    money_c = Money(c)
    assert (money_a + money_b) + money_c == money_a + (money_b + money_c)


@given(cents_strategy)
def test_money_addition_identity(a):
    """Test that a + 0 == a for Money."""
    money_a = Money(a)
    zero = Money.zero()
    assert money_a + zero == money_a
    assert zero + money_a == money_a


@given(cents_strategy)
def test_money_additive_inverse(a):
    """Test that a + (-a) == 0 for Money."""
    money_a = Money(a)
    assert money_a + (-money_a) == Money.zero()


@given(cents_strategy, cents_strategy)
def test_money_subtraction_inverse_of_addition(a, b):
    """Test that (a + b) - b == a for Money."""
    money_a = Money(a)
    money_b = Money(b)
    assert (money_a + money_b) - money_b == money_a


@given(positive_cents_strategy, int_multiplier_strategy)
def test_money_multiplication_by_integer(cents, multiplier):
    """Test Money multiplication by integer."""
    money = Money(cents)
    result = money * multiplier
    assert result.cents == cents * multiplier
    assert result.currency == money.currency


@given(positive_cents_strategy, int_multiplier_strategy)
def test_money_multiplication_commutativity_with_int(cents, multiplier):
    """Test that money * int == int * money."""
    money = Money(cents)
    assert money * multiplier == multiplier * money


@given(positive_cents_strategy, positive_cents_strategy)
def test_money_division_by_money_gives_decimal(a, b):
    """Test that Money / Money gives Decimal ratio."""
    money_a = Money(a)
    money_b = Money(b)
    result = money_a / money_b
    assert isinstance(result, Decimal)
    assert result == Decimal(a) / Decimal(b)


@given(positive_cents_strategy, st.integers(min_value=1, max_value=1000))
def test_money_division_by_int_and_multiplication_inverse(cents, divisor):
    """Test that (money / n) * n ≈ money for integer division."""
    money = Money(cents)
    divided = money / divisor
    # Due to rounding, we may not get exact equality
    reconstructed = divided * divisor
    # Allow for rounding error of up to divisor-1 cents
    assert abs(reconstructed.cents - money.cents) < divisor


@given(cents_strategy, cents_strategy)
def test_money_comparison_consistency(a, b):
    """Test comparison consistency."""
    money_a = Money(a)
    money_b = Money(b)
    
    if a < b:
        assert money_a < money_b
        assert not money_a > money_b
        assert money_a <= money_b
        assert not money_a >= money_b
        assert money_a != money_b
    elif a > b:
        assert money_a > money_b
        assert not money_a < money_b
        assert money_a >= money_b
        assert not money_a <= money_b
        assert money_a != money_b
    else:  # a == b
        assert money_a == money_b
        assert not money_a < money_b
        assert not money_a > money_b
        assert money_a <= money_b
        assert money_a >= money_b


@given(cents_strategy)
def test_money_absolute_value_properties(cents):
    """Test absolute value properties."""
    money = Money(cents)
    abs_money = abs(money)
    
    # abs(x) >= 0
    assert abs_money.cents >= 0
    
    # abs(x) == abs(-x)
    assert abs(money) == abs(-money)
    
    # abs(abs(x)) == abs(x)
    assert abs(abs_money) == abs_money


@given(decimal_strategy)
def test_money_from_dollars_roundtrip(dollars):
    """Test that from_dollars -> to_decimal roundtrip preserves precision to cents."""
    assume(abs(dollars) < 10**7)  # Avoid overflow
    
    money = Money.from_dollars(dollars)
    result_decimal = money.to_decimal()
    
    # Should be equal to nearest cent
    expected = dollars.quantize(Decimal('0.01'))
    assert result_decimal == expected


@given(cents_strategy)
def test_money_hash_consistency(cents):
    """Test that equal Money objects have equal hashes."""
    money1 = Money(cents)
    money2 = Money(cents)
    
    assert money1 == money2
    assert hash(money1) == hash(money2)


def test_money_currency_isolation():
    """Test that different currencies cannot be mixed in operations."""
    usd = Money(100, "USD")
    eur = Money(100, "EUR")
    
    with pytest.raises(ValueError, match="different currencies"):
        usd + eur
    
    with pytest.raises(ValueError, match="different currencies"):
        usd - eur
    
    with pytest.raises(ValueError, match="different currencies"):
        usd < eur


def test_money_float_contamination_prevention():
    """Test that Money prevents float contamination."""
    money = Money(100)
    
    # Constructor should reject floats
    with pytest.raises(TypeError, match="Float not allowed"):
        Money(100.0)
    
    # Arithmetic should reject floats
    with pytest.raises(TypeError, match="Cannot multiply Money by float"):
        money * 1.5
    
    with pytest.raises(TypeError, match="Cannot divide Money by float"):
        money / 1.5


@given(st.integers(min_value=0, max_value=999))
def test_money_caching_for_common_values(cents):
    """Test that common USD values are cached."""
    money1 = USD(cents)
    money2 = USD(cents)
    
    # Should be the same object due to caching
    assert money1 is money2


def test_money_string_representations():
    """Test string representations are readable."""
    money = Money(1999, "USD")  # $19.99
    
    assert str(money) == "19.99 USD"
    assert "Money" in repr(money)
    assert "19.99" in repr(money)
    assert "USD" in repr(money)


@given(cents_strategy)
def test_money_immutability(cents):
    """Test that Money objects are immutable."""
    money = Money(cents)
    
    # Should not be able to modify attributes
    with pytest.raises(AttributeError):
        money._cents = 999
    
    with pytest.raises(AttributeError):
        money._currency = "EUR"


def test_money_zero_factory():
    """Test Money.zero() factory method."""
    zero_usd = Money.zero("USD")
    zero_eur = Money.zero("EUR")
    
    assert zero_usd.cents == 0
    assert zero_usd.currency == "USD"
    assert zero_eur.cents == 0
    assert zero_eur.currency == "EUR"
    
    # Different currencies should not be equal
    assert zero_usd != zero_eur


@given(decimal_strategy)
def test_money_decimal_multiplication_precision(dollars):
    """Test multiplication by Decimal maintains precision."""
    assume(abs(dollars) < 1000)  # Avoid overflow
    
    money = Money.from_dollars("10.00")
    multiplier = Decimal("1.5")
    
    result = money * multiplier
    expected_cents = int((1000 * multiplier).quantize(Decimal('1')))
    
    assert result.cents == expected_cents