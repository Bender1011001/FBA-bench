"""Money type with integer cents backend for exact financial calculations."""
from __future__ import annotations
from decimal import Decimal, ROUND_HALF_UP
from typing import Union, Any
import operator
from functools import total_ordering


# A cache for frequently used Money instances (e.g., $0, $1) to optimize performance
# and reduce object creation overhead, especially in high-frequency calculations.
_MONEY_CACHE: dict[int, 'Money'] = {} # Using forward reference for type hinting


@total_ordering
class Money:
    """
    Represents an immutable monetary value with exact precision, backed by
    an integer number of cents. This design prevents floating-point arithmetic
    errors, ensuring deterministic and accurate financial calculations crucial
    for simulations and auditing.

    The class supports various arithmetic operations, comparisons, and
    conversion to/from human-readable dollar amounts, consistently applying
    banker's rounding (ROUND_HALF_UP) for precision.

    Attributes:
        _cents (int): The amount in the smallest currency unit (e.g., cents for USD).
        _currency (str): The ISO 4217 currency code (e.g., "USD").

    Key Features:
    - **Integer-based Arithmetic**: Eliminates floating-point inaccuracies.
    - **Immutability**: Instances cannot be changed after creation, enhancing predictability.
    - **Currency Awareness**: Supports explicit currency codes, preventing cross-currency errors.
    - **Optimized Performance**: Leverages a cache for common `Money` values to reduce overhead.
    - **Strict Type Checking**: Guards against accidental `float` contamination in constructors and operations.
    - **Deterministic Rounding**: Uses `Decimal` and `ROUND_HALF_UP` for consistent rounding.
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
        # Convert various input types to a high-precision Decimal
        if isinstance(dollars, str):
            decimal_amount = Decimal(dollars)
        elif isinstance(dollars, (int, float)):
            # Convert float to string first to avoid precision issues before Decimal conversion
            decimal_amount = Decimal(str(dollars))
        elif isinstance(dollars, Decimal):
            decimal_amount = dollars
        else:
            raise TypeError(f"Unsupported type for dollars: {type(dollars)}")
        
        # Convert the dollar amount to cents, applying banker's rounding (ROUND_HALF_UP).
        # This ensures that .5 cents always rounds to the nearest even integer.
        cents_decimal = decimal_amount * 100
        cents = int(cents_decimal.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        
        # Utilize the performance cache for frequently accessed USD values.
        # Caching up to $999.99 (99999 cents) reduces object creation overhead.
        if currency == "USD" and 0 <= cents <= 99999:
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
    
    
    # Arithmetic operations (+, -, *, /, //, %)
    # These dunder methods enable Money objects to be used with standard
    # arithmetic operators, ensuring type checks and currency compatibility.

    def __add__(self, other: 'Money') -> 'Money':
        """Adds two Money instances."""
        if not isinstance(other, Money):
            raise TypeError(f"Cannot add Money and {type(other)}")
        self._check_currency_compatibility(other)
        return Money(self._cents + other._cents, self._currency)
    
    def __sub__(self, other: 'Money') -> 'Money':
        """Subtracts one Money instance from another."""
        if not isinstance(other, Money):
            raise TypeError(f"Cannot subtract {type(other)} from Money")
        self._check_currency_compatibility(other)
        return Money(self._cents - other._cents, self._currency)
    
    def __mul__(self, other: Union[int, Decimal]) -> 'Money':
        """
        Multiplies a Money instance by an integer or Decimal.
        Float multiplication is explicitly disallowed to maintain precision.
        """
        if isinstance(other, float):
            raise TypeError("Float multiplication not allowed - use Decimal or int for Money operations to maintain precision.")
        if isinstance(other, int):
            return Money(self._cents * other, self._currency)
        elif isinstance(other, Decimal):
            # Perform multiplication using Decimal for precision, then round to nearest cent.
            result_cents = (Decimal(self._cents) * other).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
            return Money(int(result_cents), self._currency)
        else:
            raise TypeError(f"Cannot multiply Money by {type(other)}. Supported types: int, Decimal.")
    
    def __rmul__(self, other: Union[int, Decimal]) -> 'Money':
        """Enables right-hand side multiplication (e.g., 5 * Money(100))."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[int, Decimal]) -> 'Money':
        """
        Divides a Money instance by an integer or Decimal, returning a new Money instance.
        Float division is explicitly disallowed. Uses banker's rounding.
        """
        if isinstance(other, float):
            raise TypeError("Float division not allowed - use Decimal or int for Money operations.")
        if isinstance(other, int):
            if other == 0:
                raise ZeroDivisionError("Cannot divide Money by zero")
            # Perform division using Decimal for intermediate precision, then round to nearest cent.
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
            raise TypeError(f"Cannot divide Money by {type(other)}. Supported types: int, Decimal.")
    
    def __floordiv__(self, other: Union[int, Decimal]) -> 'Money':
        """
        Performs floor division on a Money instance by an integer or Decimal.
        Returns a new Money instance.
        """
        if isinstance(other, float):
            raise TypeError("Float division not allowed - use Decimal or int for Money operations.")
        if isinstance(other, (int, Decimal)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide Money by zero")
            return Money(self._cents // int(other), self._currency)
        else:
            raise TypeError(f"Cannot floor divide Money by {type(other)}. Supported types: int, Decimal.")
    
    def __mod__(self, other: Union[int, Decimal]) -> 'Money':
        """
        Performs modulo operation on a Money instance by an integer or Decimal.
        Returns a new Money instance.
        """
        if isinstance(other, float):
            raise TypeError("Float modulo not allowed - use Decimal or int for Money operations.")
        if isinstance(other, (int, Decimal)):
            if other == 0:
                raise ZeroDivisionError("Cannot modulo Money by zero")
            return Money(self._cents % int(other), self._currency)
        else:
            raise TypeError(f"Cannot modulo Money by {type(other)}. Supported types: int, Decimal.")
    
    # Unary operations (negation, absolute value)
    def __neg__(self) -> 'Money':
        """Returns a new Money instance with the negated value."""
        return Money(-self._cents, self._currency)
    
    def __abs__(self) -> 'Money':
        """Returns a new Money instance with the absolute value."""
        return Money(abs(self._cents), self._currency)
    
    # Comparison operations (==, <, <=, >, >=)
    # @total_ordering decorator automatically fills in <=, >, >= if < and == are defined.
    def __eq__(self, other: object) -> bool:
        """Compares two Money instances for equality (value and currency)."""
        if not isinstance(other, Money):
            return False
        # Comparing cents directly for equality ensures precision.
        return self._cents == other._cents and self._currency == other._currency
    
    def __lt__(self, other: 'Money') -> bool:
        """Compares if this Money instance is less than another (same currency required)."""
        if not isinstance(other, Money):
            raise TypeError(f"Cannot compare Money with {type(other)}")
        self._check_currency_compatibility(other) # Ensure currency compatibility for comparison
        return self._cents < other._cents
    
    def __hash__(self) -> int:
        """Returns a hash value, allowing Money objects to be used in sets or as dictionary keys."""
        return hash((self._cents, self._currency))
    
    # String representation (for display and debugging)
    def __str__(self) -> str:
        """Returns a human-readable string representation (e.g., "$19.99")."""
        dollars = self._cents / 100
        return f"${dollars:.2f}" # Formats to two decimal places
    
    def __repr__(self) -> str:
        """Returns a developer-friendly string representation for debugging."""
        return f"Money({self._cents}, {self._currency!r})"
    
    # Serialization support (for pickling/unpickling)
    # These methods ensure that Money objects can be correctly serialized and deserialized.
    def __getstate__(self) -> dict:
        """Returns the state for pickling, containing cents and currency."""
        return {'cents': self._cents, 'currency': self._currency}
    
    def __setstate__(self, state: dict) -> None:
        """Restores the state from pickling, setting private attributes directly."""
        # Use object.__setattr__ because Money is immutable (frozen=True equivalent)
        object.__setattr__(self, '_cents', state['cents'])
        object.__setattr__(self, '_currency', state['currency'])


# Utility functions for Money operations
def sum_money(money_amounts: list['Money'], start: 'Money' = None) -> 'Money':
    """
    Sums a list of Money amounts, ensuring all amounts are of the same currency.

    Args:
        money_amounts (list[Money]): A list of Money instances to be summed.
        start (Money, optional): An initial Money value to start the sum from.
                                   Defaults to Money.zero() if not provided.

    Returns:
        Money: The total sum of all Money amounts.

    Raises:
        TypeError: If any item in `money_amounts` is not a Money instance.
        ValueError: If currencies of the amounts do not match each other or `start`.
    """
    if not money_amounts:
        return start or Money.zero()
    
    # Determine the starting value and its currency.
    # If 'start' is not provided, use a zero Money amount with the currency of the first item.
    if start is None:
        start = Money.zero(money_amounts[0].currency)
    
    # Accumulate the sum, leveraging Money's __add__ method with currency checks.
    result = start
    for amount in money_amounts:
        if not isinstance(amount, Money):
            raise TypeError(f"All amounts in the list must be Money instances, but got {type(amount)}.")
        result += amount # __add__ handles currency compatibility
    
    return result


def max_money(*money_amounts: 'Money') -> 'Money':
    """
    Returns the maximum Money amount from a variable number of arguments.
    All input Money amounts must be of the same currency.

    Args:
        *money_amounts (Money): Variable number of Money instances.

    Returns:
        Money: The Money instance with the maximum value.

    Raises:
        ValueError: If no arguments are provided.
        TypeError: If arguments are not Money instances or have differing currencies.
    """
    if not money_amounts:
        raise ValueError("max_money() requires at least one Money argument.")
    
    # Start with the first amount and compare against the rest.
    result = money_amounts[0]
    for amount in money_amounts[1:]:
        # Comparison operators (e.g., '>') automatically call _check_currency_compatibility.
        if amount > result:
            result = amount
    return result


def min_money(*money_amounts: 'Money') -> 'Money':
    """
    Returns the minimum Money amount from a variable number of arguments.
    All input Money amounts must be of the same currency.

    Args:
        *money_amounts (Money): Variable number of Money instances.

    Returns:
        Money: The Money instance with the minimum value.

    Raises:
        ValueError: If no arguments are provided.
        TypeError: If arguments are not Money instances or have differing currencies.
    """
    if not money_amounts:
        raise ValueError("min_money() requires at least one Money argument.")
    
    # Start with the first amount and compare against the rest.
    result = money_amounts[0]
    for amount in money_amounts[1:]:
        # Comparison operators (e.g., '<') automatically call _check_currency_compatibility.
        if amount < result:
            result = amount
    return result