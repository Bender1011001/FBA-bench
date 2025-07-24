"""
Global Supply Chain implementation for FBA-Bench.

Implements the supplier model described in the blueprint with:
- International ("Sim-Alibaba") and Domestic suppliers
- MOQ, lead times, QC risk, and capital lock characteristics
- Supplier reputation and stability tracking
- Blacklisting cascade for supplier abuse
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

class SupplierType(Enum):
    INTERNATIONAL = "international"
    DOMESTIC = "domestic"

class SupplierStatus(Enum):
    ACTIVE = "active"
    BLACKLISTED = "blacklisted"
    SUSPENDED = "suspended"
    BANKRUPT = "bankrupt"

@dataclass
class Supplier:
    """
    Supplier object with reputation, stability, and characteristics.
    
    Attributes match the blueprint specification in section 2.3.
    """
    supplier_id: str
    name: str
    supplier_type: SupplierType
    
    # Cost characteristics
    unit_cost_multiplier: float  # Multiplier applied to base product cost
    
    # Order characteristics
    moq_min: int  # Minimum order quantity
    moq_max: int  # Maximum order quantity
    
    # Lead time characteristics (in days)
    production_lead_time_min: int
    production_lead_time_max: int
    shipping_lead_time_min: int
    shipping_lead_time_max: int
    
    # Quality and reliability
    qc_risk_probability: float  # Probability of quality control issues (0-1)
    reputation_score: float = 1.0  # Reputation score (0-1, starts at 1.0)
    stability_score: float = 1.0  # Stability score (0-1, starts at 1.0)
    
    # Status tracking
    status: SupplierStatus = SupplierStatus.ACTIVE
    cancellation_count: int = 0
    successful_orders: int = 0
    total_orders: int = 0
    
    # Capital lock period (days)
    capital_lock_days: int = 30
    
    # Order history
    order_history: List[Dict] = field(default_factory=list)

    def calculate_total_lead_time(self, shipping_method: str = "sea") -> int:
        """Calculate total lead time including production and shipping."""
        production_time = random.randint(self.production_lead_time_min, self.production_lead_time_max)
        
        if self.supplier_type == SupplierType.INTERNATIONAL:
            if shipping_method == "air":
                shipping_time = random.randint(5, 10)
            else:  # sea freight
                shipping_time = random.randint(30, 40)
        else:  # domestic
            shipping_time = random.randint(self.shipping_lead_time_min, self.shipping_lead_time_max)
        
        return production_time + shipping_time

    def calculate_unit_cost(self, base_cost: float) -> float:
        """Calculate actual unit cost including supplier markup/discount."""
        return base_cost * self.unit_cost_multiplier

    def update_reputation(self, order_successful: bool, quality_issue: bool = False):
        """Update supplier reputation based on order outcome."""
        self.total_orders += 1
        
        if order_successful:
            self.successful_orders += 1
            # Slight reputation improvement for successful orders
            self.reputation_score = min(1.0, self.reputation_score + 0.01)
        else:
            self.cancellation_count += 1
            # Reputation penalty for cancellations
            self.reputation_score = max(0.0, self.reputation_score - 0.1)
        
        if quality_issue:
            # Additional penalty for quality issues
            self.reputation_score = max(0.0, self.reputation_score - 0.05)
        
        # Check for blacklisting
        if self.cancellation_count >= 3 or self.reputation_score < 0.3:
            self.status = SupplierStatus.BLACKLISTED

    def can_fulfill_order(self, quantity: int) -> bool:
        """Check if supplier can fulfill an order of given quantity."""
        return (self.status == SupplierStatus.ACTIVE and 
                self.moq_min <= quantity <= self.moq_max)

class GlobalSupplyChain:
    """
    Global supply chain manager implementing the blueprint specification.
    
    Manages international and domestic suppliers with different characteristics.
    """
    
    def __init__(self):
        self.suppliers: Dict[str, Supplier] = {}
        self.active_orders: Dict[str, Dict] = {}
        self._initialize_default_suppliers()
    
    def _initialize_default_suppliers(self):
        """Initialize default suppliers matching blueprint characteristics."""
        
        # International suppliers ("Sim-Alibaba")
        international_suppliers = [
            Supplier(
                supplier_id="INTL_001",
                name="Shenzhen Manufacturing Co.",
                supplier_type=SupplierType.INTERNATIONAL,
                unit_cost_multiplier=0.6,  # Low cost
                moq_min=500, moq_max=2000,  # High MOQ
                production_lead_time_min=30, production_lead_time_max=45,
                shipping_lead_time_min=30, shipping_lead_time_max=40,
                qc_risk_probability=0.15,  # Medium-high QC risk
                capital_lock_days=90  # 2-4 months capital lock
            ),
            Supplier(
                supplier_id="INTL_002", 
                name="Guangzhou Export Ltd.",
                supplier_type=SupplierType.INTERNATIONAL,
                unit_cost_multiplier=0.7,
                moq_min=1000, moq_max=5000,
                production_lead_time_min=35, production_lead_time_max=50,
                shipping_lead_time_min=25, shipping_lead_time_max=35,
                qc_risk_probability=0.12,
                capital_lock_days=120
            )
        ]
        
        # Domestic suppliers
        domestic_suppliers = [
            Supplier(
                supplier_id="DOM_001",
                name="American Manufacturing Inc.",
                supplier_type=SupplierType.DOMESTIC,
                unit_cost_multiplier=1.2,  # High cost
                moq_min=10, moq_max=500,  # Variable MOQ (lowered min for testing)
                production_lead_time_min=10, production_lead_time_max=20,
                shipping_lead_time_min=3, shipping_lead_time_max=7,
                qc_risk_probability=0.05,  # Low QC risk
                capital_lock_days=20  # <1 month capital lock
            ),
            Supplier(
                supplier_id="DOM_002",
                name="Regional Supply Co.",
                supplier_type=SupplierType.DOMESTIC,
                unit_cost_multiplier=1.1,
                moq_min=5, moq_max=300,  # Variable MOQ (lowered min for testing)
                production_lead_time_min=7, production_lead_time_max=15,
                shipping_lead_time_min=2, shipping_lead_time_max=5,
                qc_risk_probability=0.03,
                capital_lock_days=15
            )
        ]
        
        # Add all suppliers to the registry
        for supplier in international_suppliers + domestic_suppliers:
            self.suppliers[supplier.supplier_id] = supplier
    
    def get_available_suppliers(self, quantity: int) -> List[Supplier]:
        """Get list of suppliers that can fulfill the given quantity."""
        return [supplier for supplier in self.suppliers.values() 
                if supplier.can_fulfill_order(quantity)]
    
    def place_order(self, supplier_id: str, quantity: int, base_cost: float, 
                   shipping_method: str = "sea") -> Optional[Dict]:
        """
        Place an order with a supplier.
        
        Returns order details or None if order cannot be placed.
        """
        if supplier_id not in self.suppliers:
            return None
        
        supplier = self.suppliers[supplier_id]
        
        if not supplier.can_fulfill_order(quantity):
            return None
        
        # Calculate order details
        unit_cost = supplier.calculate_unit_cost(base_cost)
        total_cost = unit_cost * quantity
        lead_time = supplier.calculate_total_lead_time(shipping_method)
        
        # Simulate quality control risk
        qc_issue = random.random() < supplier.qc_risk_probability
        
        order = {
            "order_id": f"ORD_{len(self.active_orders) + 1:06d}",
            "supplier_id": supplier_id,
            "quantity": quantity,
            "unit_cost": unit_cost,
            "total_cost": total_cost,
            "lead_time_days": lead_time,
            "shipping_method": shipping_method,
            "qc_issue": qc_issue,
            "status": "placed",
            "capital_lock_days": supplier.capital_lock_days
        }
        
        self.active_orders[order["order_id"]] = order
        supplier.order_history.append(order)
        
        return order
    
    def complete_order(self, order_id: str, successful: bool = True) -> bool:
        """Complete an order and update supplier reputation."""
        if order_id not in self.active_orders:
            return False
        
        order = self.active_orders[order_id]
        supplier = self.suppliers[order["supplier_id"]]
        
        # Update supplier reputation
        supplier.update_reputation(successful, order.get("qc_issue", False))
        
        # Update order status
        order["status"] = "completed" if successful else "cancelled"
        
        # Remove from active orders
        del self.active_orders[order_id]
        
        return True
    
    def blacklist_supplier(self, supplier_id: str, reason: str = "manual"):
        """Blacklist a supplier (blacklisting cascade)."""
        if supplier_id in self.suppliers:
            supplier = self.suppliers[supplier_id]
            supplier.status = SupplierStatus.BLACKLISTED
            
            # Cancel all active orders with this supplier
            orders_to_cancel = [order_id for order_id, order in self.active_orders.items() 
                              if order["supplier_id"] == supplier_id]
            
            for order_id in orders_to_cancel:
                self.complete_order(order_id, successful=False)
    
    def get_supplier_analytics(self) -> Dict:
        """Get analytics on supplier performance."""
        analytics = {
            "total_suppliers": len(self.suppliers),
            "active_suppliers": len([s for s in self.suppliers.values() 
                                   if s.status == SupplierStatus.ACTIVE]),
            "blacklisted_suppliers": len([s for s in self.suppliers.values() 
                                        if s.status == SupplierStatus.BLACKLISTED]),
            "international_suppliers": len([s for s in self.suppliers.values() 
                                          if s.supplier_type == SupplierType.INTERNATIONAL]),
            "domestic_suppliers": len([s for s in self.suppliers.values() 
                                     if s.supplier_type == SupplierType.DOMESTIC]),
            "active_orders": len(self.active_orders),
            "avg_reputation": sum(s.reputation_score for s in self.suppliers.values()) / len(self.suppliers)
        }
        
        return analytics
    
    def find_best_supplier(self, quantity: int, base_cost: float, 
                          priority: str = "cost") -> Optional[Tuple[Supplier, Dict]]:
        """
        Find the best supplier for a given order based on priority.
        
        Args:
            quantity: Order quantity
            base_cost: Base product cost
            priority: "cost", "speed", "quality", or "reliability"
        
        Returns:
            Tuple of (best_supplier, order_preview) or None
        """
        available_suppliers = self.get_available_suppliers(quantity)
        
        if not available_suppliers:
            return None
        
        best_supplier = None
        best_score = float('-inf')
        
        for supplier in available_suppliers:
            score = 0
            
            if priority == "cost":
                # Lower cost is better
                unit_cost = supplier.calculate_unit_cost(base_cost)
                score = -unit_cost
            elif priority == "speed":
                # Lower lead time is better
                lead_time = supplier.calculate_total_lead_time()
                score = -lead_time
            elif priority == "quality":
                # Lower QC risk and higher reputation is better
                score = supplier.reputation_score - supplier.qc_risk_probability
            elif priority == "reliability":
                # Higher reputation and stability is better
                score = (supplier.reputation_score + supplier.stability_score) / 2
            
            if score > best_score:
                best_score = score
                best_supplier = supplier
        
        if best_supplier:
            order_preview = {
                "supplier_id": best_supplier.supplier_id,
                "unit_cost": best_supplier.calculate_unit_cost(base_cost),
                "total_cost": best_supplier.calculate_unit_cost(base_cost) * quantity,
                "estimated_lead_time": best_supplier.calculate_total_lead_time(),
                "qc_risk": best_supplier.qc_risk_probability,
                "reputation": best_supplier.reputation_score
            }
            return best_supplier, order_preview
        
        return None