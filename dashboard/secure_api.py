"""
Secure Dashboard API Interface.

Provides a completely decoupled, secure interface between the dashboard and simulation core.
This replaces the insecure direct memory access pattern with proper API boundaries.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import uuid
import json
from decimal import Decimal

from fba_bench.money import Money


@dataclass
class SimulationSnapshot:
    """Immutable snapshot of simulation state for dashboard consumption."""
    simulation_id: str
    current_day: int
    current_date: str
    cash_balance: str  # Serialized Money as string
    inventory_value: str
    total_revenue: str
    total_fees: str
    total_cogs: str
    products: List[Dict[str, Any]]
    competitors: List[Dict[str, Any]]
    recent_events: List[Dict[str, Any]]
    kpi_metrics: Dict[str, Any]
    timestamp: str


@dataclass
class ProductSnapshot:
    """Immutable snapshot of product state."""
    asin: str
    category: str
    price: str  # Serialized Money
    cost: str   # Serialized Money
    inventory_quantity: int
    bsr: int
    sales_velocity: float
    conversion_rate: float
    trust_score: float


@dataclass
class CompetitorSnapshot:
    """Immutable snapshot of competitor state."""
    asin: str
    price: str  # Serialized Money
    bsr: int
    sales_velocity: float
    market_share: float


class SimulationDataProvider(ABC):
    """Abstract interface for providing simulation data to dashboard."""
    
    @abstractmethod
    def get_simulation_snapshot(self, simulation_id: str) -> Optional[SimulationSnapshot]:
        """Get current simulation state snapshot."""
        pass
    
    @abstractmethod
    def get_financial_summary(self, simulation_id: str) -> Dict[str, Any]:
        """Get financial summary data."""
        pass
    
    @abstractmethod
    def get_product_performance(self, simulation_id: str, asin: str) -> Optional[ProductSnapshot]:
        """Get product performance data."""
        pass
    
    @abstractmethod
    def get_market_analysis(self, simulation_id: str) -> Dict[str, Any]:
        """Get market analysis data."""
        pass


class SecureSimulationDataProvider(SimulationDataProvider):
    """
    Secure implementation of simulation data provider.
    
    This class provides read-only access to simulation data through immutable snapshots,
    completely decoupling the dashboard from the simulation core.
    """
    
    def __init__(self):
        """Initialize secure data provider."""
        self._simulation_registry: Dict[str, Any] = {}
        self._access_log: List[Dict[str, Any]] = []
    
    def register_simulation(self, simulation: Any) -> str:
        """
        Register a simulation instance and return a secure ID.
        
        Args:
            simulation: Simulation instance to register
            
        Returns:
            Secure simulation ID for dashboard access
        """
        simulation_id = str(uuid.uuid4())
        self._simulation_registry[simulation_id] = simulation
        
        self._log_access("register_simulation", simulation_id, "SUCCESS")
        return simulation_id
    
    def unregister_simulation(self, simulation_id: str) -> bool:
        """
        Unregister a simulation instance.
        
        Args:
            simulation_id: Simulation ID to unregister
            
        Returns:
            True if successfully unregistered
        """
        if simulation_id in self._simulation_registry:
            del self._simulation_registry[simulation_id]
            self._log_access("unregister_simulation", simulation_id, "SUCCESS")
            return True
        
        self._log_access("unregister_simulation", simulation_id, "FAILED")
        return False
    
    def get_simulation_snapshot(self, simulation_id: str) -> Optional[SimulationSnapshot]:
        """Get immutable simulation snapshot."""
        simulation = self._get_simulation(simulation_id)
        if not simulation:
            return None
        
        try:
            # Create immutable snapshot - no direct access to simulation objects
            snapshot = SimulationSnapshot(
                simulation_id=simulation_id,
                current_day=getattr(simulation, 'day', 0),
                current_date=str(getattr(simulation, 'now', datetime.now())),
                cash_balance=str(simulation.ledger.balance('Cash')),
                inventory_value=str(simulation.ledger.balance('Inventory')),
                total_revenue=str(simulation.ledger.balance('Revenue')),
                total_fees=str(simulation.ledger.balance('Fees')),
                total_cogs=str(simulation.ledger.balance('COGS')),
                products=self._serialize_products(simulation),
                competitors=self._serialize_competitors(simulation),
                recent_events=self._serialize_recent_events(simulation),
                kpi_metrics=self._calculate_kpi_metrics(simulation),
                timestamp=datetime.now().isoformat()
            )
            
            self._log_access("get_simulation_snapshot", simulation_id, "SUCCESS")
            return snapshot
            
        except Exception as e:
            self._log_access("get_simulation_snapshot", simulation_id, f"ERROR: {e}")
            return None
    
    def get_financial_summary(self, simulation_id: str) -> Dict[str, Any]:
        """Get financial summary with proper serialization."""
        simulation = self._get_simulation(simulation_id)
        if not simulation:
            return {}
        
        try:
            cash = simulation.ledger.balance('Cash')
            revenue = simulation.ledger.balance('Revenue')
            fees = simulation.ledger.balance('Fees')
            cogs = simulation.ledger.balance('COGS')
            inventory = simulation.ledger.balance('Inventory')
            
            # Calculate derived metrics
            gross_profit = revenue - cogs
            net_profit = gross_profit - fees
            profit_margin = float(net_profit / revenue) if revenue > Money.zero() else 0.0
            
            summary = {
                "cash_balance": str(cash),
                "total_revenue": str(revenue),
                "total_fees": str(fees),
                "total_cogs": str(cogs),
                "inventory_value": str(inventory),
                "gross_profit": str(gross_profit),
                "net_profit": str(net_profit),
                "profit_margin_pct": round(profit_margin * 100, 2),
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_access("get_financial_summary", simulation_id, "SUCCESS")
            return summary
            
        except Exception as e:
            self._log_access("get_financial_summary", simulation_id, f"ERROR: {e}")
            return {}
    
    def get_product_performance(self, simulation_id: str, asin: str) -> Optional[ProductSnapshot]:
        """Get product performance snapshot."""
        simulation = self._get_simulation(simulation_id)
        if not simulation or asin not in simulation.products:
            return None
        
        try:
            product = simulation.products[asin]
            
            # Get inventory quantity safely
            inventory_qty = 0
            if hasattr(simulation, 'inventory'):
                batches = simulation.inventory._batches.get(asin, [])
                inventory_qty = sum(getattr(batch, "quantity", 0) for batch in batches)
            
            snapshot = ProductSnapshot(
                asin=asin,
                category=product.category,
                price=str(product.price),
                cost=str(product.cost),
                inventory_quantity=inventory_qty,
                bsr=getattr(product, 'bsr', 1000000),
                sales_velocity=getattr(product, 'ema_sales_velocity', 0.0),
                conversion_rate=getattr(product, 'ema_conversion', 0.0),
                trust_score=getattr(product, 'trust_score', 1.0)
            )
            
            self._log_access("get_product_performance", simulation_id, "SUCCESS")
            return snapshot
            
        except Exception as e:
            self._log_access("get_product_performance", simulation_id, f"ERROR: {e}")
            return None
    
    def get_market_analysis(self, simulation_id: str) -> Dict[str, Any]:
        """Get market analysis data."""
        simulation = self._get_simulation(simulation_id)
        if not simulation:
            return {}
        
        try:
            analysis = {
                "total_competitors": len(getattr(simulation, 'competitors', [])),
                "market_categories": list(set(p.category for p in simulation.products.values())),
                "average_competitor_price": self._calculate_avg_competitor_price(simulation),
                "market_concentration": self._calculate_market_concentration(simulation),
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_access("get_market_analysis", simulation_id, "SUCCESS")
            return analysis
            
        except Exception as e:
            self._log_access("get_market_analysis", simulation_id, f"ERROR: {e}")
            return {}
    
    def _get_simulation(self, simulation_id: str) -> Optional[Any]:
        """Safely get simulation instance."""
        return self._simulation_registry.get(simulation_id)
    
    def _serialize_products(self, simulation: Any) -> List[Dict[str, Any]]:
        """Serialize products to safe dictionary format."""
        products = []
        for asin, product in simulation.products.items():
            products.append({
                "asin": asin,
                "category": product.category,
                "price": str(product.price),
                "cost": str(product.cost),
                "bsr": getattr(product, 'bsr', 1000000)
            })
        return products
    
    def _serialize_competitors(self, simulation: Any) -> List[Dict[str, Any]]:
        """Serialize competitors to safe dictionary format."""
        competitors = []
        for competitor in getattr(simulation, 'competitors', []):
            competitors.append({
                "asin": getattr(competitor, 'asin', 'UNKNOWN'),
                "price": str(getattr(competitor, 'price', Money.zero())),
                "bsr": getattr(competitor, 'bsr', 1000000),
                "sales_velocity": getattr(competitor, 'sales_velocity', 0.0)
            })
        return competitors
    
    def _serialize_recent_events(self, simulation: Any) -> List[Dict[str, Any]]:
        """Serialize recent events to safe format."""
        events = []
        event_log = getattr(simulation, 'event_log', [])
        
        # Get last 10 events
        for event in event_log[-10:]:
            if isinstance(event, str):
                events.append({
                    "message": event,
                    "timestamp": datetime.now().isoformat()
                })
            elif isinstance(event, dict):
                # Ensure no direct object references
                safe_event = {
                    "type": event.get("type", "unknown"),
                    "message": str(event.get("message", "")),
                    "timestamp": str(event.get("date", datetime.now()))
                }
                events.append(safe_event)
        
        return events
    
    def _calculate_kpi_metrics(self, simulation: Any) -> Dict[str, Any]:
        """Calculate KPI metrics safely."""
        try:
            revenue = simulation.ledger.balance('Revenue')
            fees = simulation.ledger.balance('Fees')
            cogs = simulation.ledger.balance('COGS')
            
            return {
                "revenue_per_day": str(revenue / max(1, getattr(simulation, 'day', 1))),
                "average_order_value": str(revenue / max(1, len(simulation.products))),
                "fee_percentage": float(fees / revenue * 100) if revenue > Money.zero() else 0.0,
                "gross_margin": float((revenue - cogs) / revenue * 100) if revenue > Money.zero() else 0.0
            }
        except Exception:
            return {}
    
    def _calculate_avg_competitor_price(self, simulation: Any) -> str:
        """Calculate average competitor price safely."""
        competitors = getattr(simulation, 'competitors', [])
        if not competitors:
            return str(Money.zero())
        
        total_price = Money.zero()
        count = 0
        
        for competitor in competitors:
            price = getattr(competitor, 'price', None)
            if price and isinstance(price, Money):
                total_price += price
                count += 1
        
        if count > 0:
            return str(total_price / count)
        return str(Money.zero())
    
    def _calculate_market_concentration(self, simulation: Any) -> float:
        """Calculate market concentration index."""
        competitors = getattr(simulation, 'competitors', [])
        if len(competitors) <= 1:
            return 1.0
        
        # Simple concentration measure: 1/N where N is number of competitors
        return 1.0 / len(competitors)
    
    def _log_access(self, operation: str, simulation_id: str, status: str) -> None:
        """Log API access for security auditing."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "simulation_id": simulation_id,
            "status": status
        }
        self._access_log.append(log_entry)
        
        # Keep only last 1000 log entries
        if len(self._access_log) > 1000:
            self._access_log = self._access_log[-1000:]
    
    def get_access_log(self) -> List[Dict[str, Any]]:
        """Get access log for security monitoring."""
        return self._access_log.copy()


class DashboardSecurityManager:
    """Manages dashboard security and access control."""
    
    def __init__(self):
        """Initialize security manager."""
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._rate_limits: Dict[str, List[datetime]] = {}
    
    def create_session(self, simulation_id: str) -> str:
        """Create a secure dashboard session."""
        session_id = str(uuid.uuid4())
        self._active_sessions[session_id] = {
            "simulation_id": simulation_id,
            "created_at": datetime.now(),
            "last_access": datetime.now(),
            "access_count": 0
        }
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate dashboard session."""
        if session_id not in self._active_sessions:
            return False
        
        session = self._active_sessions[session_id]
        
        # Check session age (max 24 hours)
        age = datetime.now() - session["created_at"]
        if age.total_seconds() > 86400:  # 24 hours
            del self._active_sessions[session_id]
            return False
        
        # Update last access
        session["last_access"] = datetime.now()
        session["access_count"] += 1
        
        return True
    
    def check_rate_limit(self, session_id: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
        """Check rate limiting for session."""
        now = datetime.now()
        
        if session_id not in self._rate_limits:
            self._rate_limits[session_id] = []
        
        # Clean old requests outside window
        cutoff = now.timestamp() - (window_minutes * 60)
        self._rate_limits[session_id] = [
            req_time for req_time in self._rate_limits[session_id]
            if req_time.timestamp() > cutoff
        ]
        
        # Check if under limit
        if len(self._rate_limits[session_id]) >= max_requests:
            return False
        
        # Add current request
        self._rate_limits[session_id].append(now)
        return True
    
    def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions."""
        now = datetime.now()
        expired_sessions = []
        
        for session_id, session in self._active_sessions.items():
            age = now - session["created_at"]
            if age.total_seconds() > 86400:  # 24 hours
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self._active_sessions[session_id]
            if session_id in self._rate_limits:
                del self._rate_limits[session_id]


# Global secure data provider instance
secure_data_provider = SecureSimulationDataProvider()
security_manager = DashboardSecurityManager()