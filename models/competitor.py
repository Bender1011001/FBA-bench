"""Competitor model for market simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from money import Money


@dataclass
class Competitor:
    """
    Competitor product in the marketplace.
    
    Represents a competing product with simplified attributes
    focused on market dynamics and pricing competition.
    """
    
    asin: str
    price: Money
    sales_velocity: float
    bsr: int = 1000000
    strategy: str = "follower"  # "aggressive", "follower", "premium", "value"
    trust_score: float = 0.8
    category: str = "DEFAULT"
    
    def __post_init__(self):
        """Validate competitor data."""
        if not isinstance(self.price, Money):
            if isinstance(self.price, (int, float, str)):
                self.price = Money.from_dollars(self.price)
            else:
                raise TypeError(f"Competitor price must be Money type or convertible, got {type(self.price)}")
        
        if not 0.0 <= self.trust_score <= 1.0:
            raise ValueError(f"Trust score must be between 0.0 and 1.0, got {self.trust_score}")
        
        if self.bsr < 1:
            raise ValueError(f"BSR must be >= 1, got {self.bsr}")
        
        if self.sales_velocity < 0:
            raise ValueError(f"Sales velocity must be >= 0, got {self.sales_velocity}")
    
    def update_price(self, new_price: Money) -> None:
        """Update competitor price."""
        if not isinstance(new_price, Money):
            if isinstance(new_price, (int, float, str)):
                new_price = Money.from_dollars(new_price)
            else:
                raise TypeError(f"Price must be Money type or convertible, got {type(new_price)}")
        
        self.price = new_price
    
    def update_sales_velocity(self, new_velocity: float) -> None:
        """Update sales velocity."""
        if new_velocity < 0:
            raise ValueError(f"Sales velocity must be >= 0, got {new_velocity}")
        self.sales_velocity = new_velocity
    
    def update_bsr(self, new_bsr: int) -> None:
        """Update BSR."""
        if new_bsr < 1:
            raise ValueError(f"BSR must be >= 1, got {new_bsr}")
        self.bsr = new_bsr
    
    def __str__(self) -> str:
        return f"Competitor({self.asin}, {self.price}, {self.strategy})"
    
    def __repr__(self) -> str:
        return (f"Competitor(asin='{self.asin}', price={self.price!r}, "
                f"sales_velocity={self.sales_velocity}, strategy='{self.strategy}')")