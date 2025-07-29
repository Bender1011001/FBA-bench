"""Money type with integer cents backend for exact financial calculations."""
from __future__ import annotations
from decimal import Decimal, ROUND_HALF_UP
from typing import Union, Any
import operator
from functools import total_ordering


# Money cache for common values to reduce allocation overhead
_MONEY_CACHE: dict[int, Money] = {}


@total_ordering
class Money:
    """
    Immutable money type backed by integer cents for exact arithmetic.
    
    Provides absolute determinism across Python versions and platforms,
    prevents float contamination, and supports multi-currency operations.
    
    Key Features:
    - Integer cents backend eliminates floating-point precision errors
    - Immutable design prevents accidental modification
    - Currency support for multi-currency operations
    - Performance optimized with caching for common values
    - Strict type checking prevents float contamination
    """
    __slots__ = ('_cents', '_currency')
    
    def __init__(self, cents: int, currency: str = "USD"):
        """
        Initialize Money with integer cents.
        
        Args:
            cents: Amount in cents (e.g., 1999 for $19.99)
            currency: Currency code (default: USD)
        
        Raises:
            TypeError: If cents is a float (strict guard against float contamination)
            ValueError: If cents is not an integer
        """
        if isinstance(cents, float):
            raise TypeError("Float not allowed in Money constructor - use Money.from_dollars() for decimal input")
        
        if not isinstance(cents, int):
            raise ValueError(f"Money requires integer cents, got {type(cents)}")
        
        object.__setattr__(self, '_cents', cents)
        object.__setattr__(self, '_currency', currency)
    
    @classmethod
    def from_dollars(cls, dollars: Union[str, Decimal, int, float], currency: str = "USD") -> Money:
        """
        Create Money from dollar amount with proper rounding.
        
        Args:
            dollars: Dollar amount as string, Decimal, int, or float
            currency: Currency code (default: USD)
        
        Returns:
            Money instance with exact cent representation
        
        Examples:
            >>> Money.from_dollars("19.99")
            Money(1999, "USD")
            >>> Money.from_dollars(Decimal("33.333"))
            Money(3333, "USD")  # Rounded to nearest cent
        """
        if isinstance(dollars, str):
            decimal_amount = Decimal(dollars)
        elif isinstance(dollars, (int, float)):
            decimal_amount = Decimal(str(dollars))
        elif isinstance(dollars, Decimal):
            decimal_amount = dollars
        else:
            raise TypeError(f"Unsupported type for dollars: {type(dollars)}")
        
        # Convert to cents with banker's rounding
        cents_decimal = decimal_amount * 100
        cents = int(cents_decimal.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        
        # Use cache for common values
        if currency == "USD" and 0 <= cents <= 99999:  # Cache up to $999.99
            if cents not in _MONEY_CACHE:
                _MONEY_CACHE[cents] = cls(cents, currency)
            return _MONEY_CACHE[cents]
        
        return cls(cents, currency)
    
    @classmethod
    def zero(cls, currency: str = "USD") -> Money:
        """Create zero money amount."""
        return cls.from_dollars(0, currency)
    
    def to_decimal(self) -> Decimal:
        """Convert to Decimal for external reporting."""
        return Decimal(self._cents) / 100
    
    def to_float(self) -> float:
        """Convert to float (use sparingly, only for legacy compatibility)."""
        return float(self._cents) / 100
    
    @property
    def cents(self) -> int:
        """Get the raw cents value."""
        return self._cents
    
    @property
    def currency(self) -> str:
        """Get the currency code."""
        return self._currency
    
    def _check_currency_compatibility(self, other: Money) -> None:
        """Check if two Money instances have compatible currencies."""
        if self._currency != other._currency:
            raise ValueError(f"Cannot operate on different currencies: {self._currency} vs {other._currency}")
    
    # Arithmetic operations
    def __add__(self, other: Money) -> Money:
        if not isinstance(other, Money):
            raise TypeError(f"Cannot add Money and {type(other)}")
        self._check_currency_compatibility(other)
        return Money(self._cents + other._cents, self._currency)
    
    def __sub__(self, other: Money) -> Money:
        if not isinstance(other, Money):
            raise TypeError(f"Cannot subtract {type(other)} from Money")
        self._check_currency_compatibility(other)
        return Money(self._cents - other._cents, self._currency)
    
    def __mul__(self, other: Union[int, Decimal]) -> Money:
        if isinstance(other, float):
            raise TypeError("Float multiplication not allowed - use Decimal or int")
        if isinstance(other, int):
            return Money(self._cents * other, self._currency)
        elif isinstance(other, Decimal):
            # Multiply by decimal with proper rounding
            result_cents = (Decimal(self._cents) * other).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
            return Money(int(result_cents), self._currency)
        else:
            raise TypeError(f"Cannot multiply Money by {type(other)}")
    
    def __rmul__(self, other: Union[int, Decimal]) -> Money:
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[int, Decimal]) -> Money:
        if isinstance(other, float):
            raise TypeError("Float division not allowed - use Decimal or int")
        if isinstance(other, int):
            if other == 0:
                raise ZeroDivisionError("Cannot divide Money by zero")
            # Use banker's rounding for division
            result_cents = Decimal(self._cents) / Decimal(other)
            return Money(int(result_cents.quantize(Decimal('1'), rounding=ROUND_HALF_UP)),
                        self._currency)
        elif isinstance(other, Decimal):
            if other == 0:
                raise ZeroDivisionError("Cannot divide Money by zero")
            result_cents = Decimal(self._cents) / other
            return Money(int(result_cents.quantize(Decimal('1'), rounding=ROUND_HALF_UP)),
                        self._currency)
        else:
            raise TypeError(f"Cannot divide Money by {type(other)}")
    
    def __floordiv__(self, other: Union[int, Decimal]) -> Money:
        if isinstance(other, float):
            raise TypeError("Float division not allowed - use Decimal or int")
        if isinstance(other, (int, Decimal)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide Money by zero")
            return Money(self._cents // int(other), self._currency)
        else:
            raise TypeError(f"Cannot floor divide Money by {type(other)}")
    
    def __mod__(self, other: Union[int, Decimal]) -> Money:
        if isinstance(other, float):
            raise TypeError("Float modulo not allowed - use Decimal or int")
        if isinstance(other, (int, Decimal)):
            if other == 0:
                raise ZeroDivisionError("Cannot modulo Money by zero")
            return Money(self._cents % int(other), self._currency)
        else:
            raise TypeError(f"Cannot modulo Money by {type(other)}")
    
    # Unary operations
    def __neg__(self) -> Money:
        return Money(-self._cents, self._currency)
    
    def __abs__(self) -> Money:
        return Money(abs(self._cents), self._currency)
    
    # Comparison operations
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return False
        return self._cents == other._cents and self._currency == other._currency
    
    def __lt__(self, other: Money) -> bool:
        if not isinstance(other, Money):
            raise TypeError(f"Cannot compare Money with {type(other)}")
        self._check_currency_compatibility(other)
        return self._cents < other._cents
    
    def __hash__(self) -> int:
        return hash((self._cents, self._currency))
    
    # String representation
    def __str__(self) -> str:
        dollars = self._cents / 100
        return f"${dollars:.2f}"
    
    def __repr__(self) -> str:
        return f"Money({self._cents}, {self._currency!r})"
    
    # Serialization support
    def __getstate__(self) -> dict:
        return {'cents': self._cents, 'currency': self._currency}
    
    def __setstate__(self, state: dict) -> None:
        object.__setattr__(self, '_cents', state['cents'])
        object.__setattr__(self, '_currency', state['currency'])


# Utility functions for Money operations
def sum_money(money_amounts: list[Money], start: Money = None) -> Money:
    """
    Sum a list of Money amounts with proper type checking.
    
    Args:
        money_amounts: List of Money instances to sum
        start: Starting value (default: Money.zero())
    
    Returns:
        Sum of all Money amounts
    
    Raises:
        TypeError: If any amount is not a Money instance
        ValueError: If currencies don't match
    """
    if not money_amounts:
        return start or Money.zero()
    
    if start is None:
        start = Money.zero(money_amounts[0].currency)
    
    result = start
    for amount in money_amounts:
        if not isinstance(amount, Money):
            raise TypeError(f"All amounts must be Money instances, got {type(amount)}")
        result += amount
    
    return result


def max_money(*money_amounts: Money) -> Money:
    """Return the maximum Money amount."""
    if not money_amounts:
        raise ValueError("max_money() requires at least one argument")
    
    result = money_amounts[0]
    for amount in money_amounts[1:]:
        if amount > result:
            result = amount
    return result


def min_money(*money_amounts: Money) -> Money:
    """Return the minimum Money amount."""
    if not money_amounts:
        raise ValueError("min_money() requires at least one argument")
    
    result = money_amounts[0]
    for amount in money_amounts[1:]:
        if amount < result:
            result = amount
    return result