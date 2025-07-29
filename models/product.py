"""Unified Product model combining best features from both repositories."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

from money import Money


@dataclass
class Product:
    """
    Unified Product model with Money type integration and clean architecture.
    
    Combines the financial accuracy of FBA-bench-main with the clean design
    of fba_bench_good_sim. All monetary values use the Money type for exact
    arithmetic and prevent floating-point precision errors.
    
    Attributes:
        asin: Amazon Standard Identification Number
        category: Product category for fee calculations
        cost: Unit cost using Money type for exact arithmetic
        price: Current selling price using Money type
        base_demand: Base demand for the product
        bsr: Best Seller Rank
        trust_score: Seller trust score (0.0 to 1.0)
        
        # Inventory tracking
        inventory_units: Current inventory units
        reserved_units: Units reserved for pending orders
        
        # Performance metrics with EMA tracking
        ema_sales_velocity: Exponential moving average of sales velocity
        ema_conversion: Exponential moving average of conversion rate
        sales_velocity: Current sales velocity
        conversion_rate: Current conversion rate
        
        # Historical data for analysis
        sales_history: Historical sales data
        demand_history: Historical demand data
        conversion_history: Historical conversion rates
        price_history: Historical pricing data
        
        # Product metadata
        size_tier: Size tier for FBA fees ("standard" or "oversize")
        size: Size category ("small", "large", etc.)
        weight: Product weight in pounds
        dimensions: Product dimensions (L x W x H in inches)
        
        # Timestamps
        launch_date: When the product was launched
        last_updated: Last update timestamp
    """
    
    # Core product identification
    asin: str
    category: str
    
    # Financial data using Money type for exact arithmetic
    cost: Money
    price: Money
    
    # Market performance
    base_demand: float
    bsr: int = 1000000
    trust_score: float = 1.0
    
    # Inventory management
    inventory_units: int = 0
    reserved_units: int = 0
    
    # Performance metrics with EMA tracking
    ema_sales_velocity: float = 0.0
    ema_conversion: float = 0.0
    sales_velocity: float = 0.0
    conversion_rate: float = 0.0
    
    # Historical tracking
    sales_history: List[float] = field(default_factory=list)
    demand_history: List[float] = field(default_factory=list)
    conversion_history: List[float] = field(default_factory=list)
    price_history: List[Money] = field(default_factory=list)
    
    # Physical characteristics for fee calculations
    size_tier: str = "standard"  # "standard" or "oversize"
    size: str = "small"  # "small", "large", "medium"
    weight: float = 1.0  # Weight in pounds
    dimensions: str = "12x8x4"  # L x W x H in inches
    
    # Timestamps
    launch_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize timestamps and validate data."""
        if self.launch_date is None:
            self.launch_date = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()
        
        # Ensure Money types for cost and price
        if not isinstance(self.cost, Money):
            if isinstance(self.cost, (int, float, str)):
                self.cost = Money.from_dollars(self.cost)
            else:
                raise TypeError(f"Product cost must be Money type or convertible, got {type(self.cost)}")
        
        if not isinstance(self.price, Money):
            if isinstance(self.price, (int, float, str)):
                self.price = Money.from_dollars(self.price)
            else:
                raise TypeError(f"Product price must be Money type or convertible, got {type(self.price)}")
        
        # Validate ranges
        if not 0.0 <= self.trust_score <= 1.0:
            raise ValueError(f"Trust score must be between 0.0 and 1.0, got {self.trust_score}")
        
        if self.bsr < 1:
            raise ValueError(f"BSR must be >= 1, got {self.bsr}")
        
        if self.base_demand < 0:
            raise ValueError(f"Base demand must be >= 0, got {self.base_demand}")
    
    def update_price(self, new_price: Money) -> None:
        """
        Update product price with history tracking.
        
        Args:
            new_price: New price as Money type
        """
        if not isinstance(new_price, Money):
            if isinstance(new_price, (int, float, str)):
                new_price = Money.from_dollars(new_price)
            else:
                raise TypeError(f"Price must be Money type or convertible, got {type(new_price)}")
        
        # Track price history
        self.price_history.append(self.price)
        self.price = new_price
        self.last_updated = datetime.now()
    
    def update_inventory(self, units_sold: int, units_received: int = 0) -> None:
        """
        Update inventory levels.
        
        Args:
            units_sold: Number of units sold (reduces inventory)
            units_received: Number of units received (increases inventory)
        """
        self.inventory_units = max(0, self.inventory_units - units_sold + units_received)
        self.last_updated = datetime.now()
    
    def update_performance_metrics(
        self, 
        sales: float, 
        demand: float, 
        ema_decay: float = 0.2
    ) -> None:
        """
        Update performance metrics with EMA tracking.
        
        Args:
            sales: Current period sales
            demand: Current period demand
            ema_decay: EMA decay factor (default: 0.2)
        """
        # Calculate conversion rate
        conversion = sales / demand if demand > 0 else 0.0
        
        # Update current metrics
        self.sales_velocity = sales
        self.conversion_rate = conversion
        
        # Update EMA metrics
        if self.ema_sales_velocity == 0.0:
            # First update - initialize with current values
            self.ema_sales_velocity = sales
            self.ema_conversion = conversion
        else:
            # EMA update: new_ema = (1 - decay) * old_ema + decay * new_value
            self.ema_sales_velocity = (1 - ema_decay) * self.ema_sales_velocity + ema_decay * sales
            self.ema_conversion = (1 - ema_decay) * self.ema_conversion + ema_decay * conversion
        
        # Track history
        self.sales_history.append(sales)
        self.demand_history.append(demand)
        self.conversion_history.append(conversion)
        
        # Limit history size to prevent memory bloat
        max_history = 365  # Keep 1 year of daily data
        if len(self.sales_history) > max_history:
            self.sales_history = self.sales_history[-max_history:]
            self.demand_history = self.demand_history[-max_history:]
            self.conversion_history = self.conversion_history[-max_history:]
            self.price_history = self.price_history[-max_history:]
        
        self.last_updated = datetime.now()
    
    def update_trust_score(self, new_trust_score: float) -> None:
        """
        Update trust score with validation.
        
        Args:
            new_trust_score: New trust score (0.0 to 1.0)
        """
        if not 0.0 <= new_trust_score <= 1.0:
            raise ValueError(f"Trust score must be between 0.0 and 1.0, got {new_trust_score}")
        
        self.trust_score = new_trust_score
        self.last_updated = datetime.now()
    
    def update_bsr(self, new_bsr: int) -> None:
        """
        Update Best Seller Rank with validation.
        
        Args:
            new_bsr: New BSR value (must be >= 1)
        """
        if new_bsr < 1:
            raise ValueError(f"BSR must be >= 1, got {new_bsr}")
        
        self.bsr = new_bsr
        self.last_updated = datetime.now()
    
    def get_profit_margin(self) -> Money:
        """Calculate profit margin per unit."""
        return self.price - self.cost
    
    def get_profit_margin_percentage(self) -> float:
        """Calculate profit margin as percentage."""
        if self.price.cents == 0:
            return 0.0
        return (self.get_profit_margin().to_float() / self.price.to_float()) * 100
    
    def get_inventory_value(self) -> Money:
        """Calculate total inventory value at cost."""
        return self.cost * self.inventory_units
    
    def get_available_units(self) -> int:
        """Get available units (total - reserved)."""
        return max(0, self.inventory_units - self.reserved_units)
    
    def is_in_stock(self) -> bool:
        """Check if product has available inventory."""
        return self.get_available_units() > 0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of product performance metrics."""
        return {
            'asin': self.asin,
            'current_price': str(self.price),
            'profit_margin': str(self.get_profit_margin()),
            'profit_margin_pct': round(self.get_profit_margin_percentage(), 2),
            'bsr': self.bsr,
            'trust_score': round(self.trust_score, 3),
            'sales_velocity': round(self.sales_velocity, 2),
            'conversion_rate': round(self.conversion_rate, 3),
            'ema_sales_velocity': round(self.ema_sales_velocity, 2),
            'ema_conversion': round(self.ema_conversion, 3),
            'inventory_units': self.inventory_units,
            'available_units': self.get_available_units(),
            'inventory_value': str(self.get_inventory_value()),
            'in_stock': self.is_in_stock(),
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }
    
    def __str__(self) -> str:
        """String representation of the product."""
        return f"Product({self.asin}, {self.price}, BSR: {self.bsr:,})"
    
    def __repr__(self) -> str:
        """Detailed representation of the product."""
        return (f"Product(asin='{self.asin}', category='{self.category}', "
                f"cost={self.cost!r}, price={self.price!r}, "
                f"bsr={self.bsr}, trust_score={self.trust_score})")