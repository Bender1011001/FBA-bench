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
    """
    __slots__ = ('_cents', '_currency')
    
    def __init__(self, cents: int, currency: str = "USD", *, _skip_guard: bool = False):
        """
        Initialize Money with integer cents.
        
        Args:
            cents: Amount in cents (e.g., 1999 for $19.99)
            currency: Currency code (default: USD)
            _skip_guard: Internal flag to skip validation (used by from_dollars)
        
        Raises:
            TypeError: If cents is a float (strict guard against float contamination)
            ValueError: If cents is not an integer
        """
        if not _skip_guard and isinstance(cents, float):
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
            currency: Currency code
            
        Returns:
            Money instance with properly rounded cents
            
        Examples:
            Money.from_dollars("19.99") -> Money(1999)
            Money.from_dollars(Decimal("19.995")) -> Money(2000)  # Rounds to nearest cent
        """
        if isinstance(dollars, (int, float)):
            # Convert to Decimal for precise rounding
            decimal_amount = Decimal(str(dollars))
        elif isinstance(dollars, str):
            decimal_amount = Decimal(dollars)
        elif isinstance(dollars, Decimal):
            decimal_amount = dollars
        else:
            raise TypeError(f"Unsupported type for dollars: {type(dollars)}")
        
        # Round to nearest cent using banker's rounding
        cents_decimal = decimal_amount * 100
        rounded_cents = int(cents_decimal.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        
        return cls(rounded_cents, currency, _skip_guard=True)
    
    @classmethod
    def zero(cls, currency: str = "USD") -> Money:
        """Create zero money amount."""
        return cls._get_cached(0, currency)
    
    @classmethod
    def _get_cached(cls, cents: int, currency: str = "USD") -> Money:
        """Get cached Money instance for common values."""
        if 0 <= cents <= 999 and currency == "USD":
            cache_key = cents
            if cache_key not in _MONEY_CACHE:
                _MONEY_CACHE[cache_key] = cls(cents, currency, _skip_guard=True)
            return _MONEY_CACHE[cache_key]
        return cls(cents, currency, _skip_guard=True)
    
    @property
    def cents(self) -> int:
        """Get the raw cents value."""
        return self._cents
    
    @property
    def currency(self) -> str:
        """Get the currency code."""
        return self._currency
    
    def to_decimal(self) -> Decimal:
        """Convert to Decimal for external reporting or calculations requiring precision."""
        return Decimal(self._cents) / 100
    
    def to_float(self) -> float:
        """Convert to float (use sparingly, mainly for legacy compatibility)."""
        return self._cents / 100.0
    
    def _check_currency_compatibility(self, other: Money) -> None:
        """Ensure currencies match for arithmetic operations."""
        if self._currency != other._currency:
            raise ValueError(f"Cannot operate on different currencies: {self._currency} vs {other._currency}")
    
    # Arithmetic operations
    def __add__(self, other: Money) -> Money:
        if not isinstance(other, Money):
            return NotImplemented
        self._check_currency_compatibility(other)
        return Money._get_cached(self._cents + other._cents, self._currency)
    
    def __sub__(self, other: Money) -> Money:
        if not isinstance(other, Money):
            return NotImplemented
        self._check_currency_compatibility(other)
        return Money._get_cached(self._cents - other._cents, self._currency)
    
    def __mul__(self, other: Union[int, Decimal]) -> Money:
        """Multiply by scalar (int or Decimal), not float to prevent contamination."""
        if isinstance(other, float):
            raise TypeError("Cannot multiply Money by float - use Decimal for precise multiplication")
        
        if isinstance(other, int):
            return Money._get_cached(self._cents * other, self._currency)
        elif isinstance(other, Decimal):
            # Use Decimal arithmetic for precision
            result_cents = int((Decimal(self._cents) * other).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
            return Money._get_cached(result_cents, self._currency)
        else:
            return NotImplemented
    
    def __rmul__(self, other: Union[int, Decimal]) -> Money:
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[int, Decimal, Money]) -> Union[Money, Decimal]:
        """
        Division behavior:
        - Money / scalar -> Money (rounded to nearest cent)
        - Money / Money -> Decimal (ratio)
        """
        if isinstance(other, Money):
            self._check_currency_compatibility(other)
            if other._cents == 0:
                raise ZeroDivisionError("Cannot divide by zero money")
            return Decimal(self._cents) / Decimal(other._cents)
        
        if isinstance(other, float):
            raise TypeError("Cannot divide Money by float - use Decimal for precise division")
        
        if isinstance(other, (int, Decimal)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            
            if isinstance(other, int):
                result_cents = self._cents // other
            else:  # Decimal
                result_cents = int((Decimal(self._cents) / other).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
            
            return Money._get_cached(result_cents, self._currency)
        
        return NotImplemented
    
    def __floordiv__(self, other: Union[int, Money]) -> Union[Money, int]:
        """Floor division for integer results."""
        if isinstance(other, Money):
            self._check_currency_compatibility(other)
            if other._cents == 0:
                raise ZeroDivisionError("Cannot divide by zero money")
            return self._cents // other._cents
        
        if isinstance(other, int):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return Money._get_cached(self._cents // other, self._currency)
        
        return NotImplemented
    
    def __mod__(self, other: Union[int, Money]) -> Money:
        """Modulo operation."""
        if isinstance(other, Money):
            self._check_currency_compatibility(other)
            if other._cents == 0:
                raise ZeroDivisionError("Cannot modulo by zero money")
            return Money._get_cached(self._cents % other._cents, self._currency)
        
        if isinstance(other, int):
            if other == 0:
                raise ZeroDivisionError("Cannot modulo by zero")
            return Money._get_cached(self._cents % other, self._currency)
        
        return NotImplemented
    
    def __neg__(self) -> Money:
        """Unary negation."""
        return Money._get_cached(-self._cents, self._currency)
    
    def __abs__(self) -> Money:
        """Absolute value."""
        return Money._get_cached(abs(self._cents), self._currency)
    
    # Comparison operations
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Money):
            return False
        return self._cents == other._cents and self._currency == other._currency
    
    def __lt__(self, other: Money) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        self._check_currency_compatibility(other)
        return self._cents < other._cents
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash((self._cents, self._currency))
    
    # String representations
    def __str__(self) -> str:
        """Human-readable string representation."""
        dollars = self._cents / 100
        return f"{dollars:.2f} {self._currency}"
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        dollars = self._cents / 100
        return f"Money('{dollars:.2f}', '{self._currency}')"
    
    # Serialization support
    def __reduce_ex__(self, protocol: int) -> tuple:
        """Support for pickle with minimal size."""
        return (Money, (self._cents, self._currency, True))  # _skip_guard=True
    
    # Type checking support
    def __class_getitem__(cls, currency: str) -> type:
        """Support for type hints like Money['USD']."""
        return cls


# Convenience constructors
def USD(cents: int) -> Money:
    """Create USD Money instance."""
    return Money._get_cached(cents, "USD")


def EUR(cents: int) -> Money:
    """Create EUR Money instance."""
    return Money._get_cached(cents, "EUR")


# Pre-populate cache with common values
for i in range(1000):
    _MONEY_CACHE[i] = Money(i, "USD", _skip_guard=True)