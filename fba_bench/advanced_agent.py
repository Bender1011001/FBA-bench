"""
AdvancedAgent for FBA-Bench.

Features:
- Hierarchical planner with goal stack.
- Dual memory: short-term (working), long-term (episodic, semantic, procedural).
- Reflection loop (OODA: Observe, Orient, Decide, Act).
- Compute budget tracking.

This agent is designed for extensibility and research on advanced agent architectures.
"""

import time
from collections import deque
from typing import Any, Dict, List, Optional, Callable
from decimal import Decimal

from .simulation import Simulation
from fba_bench.config import DEFAULT_CATEGORY, DEFAULT_COST, DEFAULT_PRICE, DEFAULT_QTY
from fba_bench.money import Money

class AdvancedAgent:
    """
    AdvancedAgent for FBA-Bench.

    # --- Default Product Configuration ---
    DEFAULT_ASIN = "B000TEST"
    DEFAULT_CATEGORY = DEFAULT_CATEGORY
    DEFAULT_COST = DEFAULT_COST
    DEFAULT_PRICE = DEFAULT_PRICE
    DEFAULT_QTY = DEFAULT_QTY

    Features:
    - Hierarchical planner with recursive goal decomposition (DAG/goal stack).
    - Dual memory: short-term (working), long-term (episodic, semantic, procedural) with query/retrieval.
    - Formal reflection/self-correction in OODA loop.
    - Compute budget tracking.
    - Extensible for research on advanced agent architectures.
    """
    def __init__(self, days=30, compute_budget=1.0, adversarial_events=None,
                 asin=None, category=None, cost=None, price=None, qty=None,
                 api_budget=100.0, cpu_budget=100.0, reflection_thresholds=None):
        """
        Args:
            days (int): Number of days to run the simulation.
            compute_budget (float): Max compute time per OODA cycle (seconds).
            adversarial_events (AdversarialEventCatalog, optional): Catalog of adversarial events for red-team testing.
            asin (str): Product ASIN.
            category (str): Product category.
            cost (float): Product cost per unit.
            price (float): Product price.
            qty (int): Initial inventory quantity.
            api_budget (float): API cost budget per simulation day.
            cpu_budget (float): CPU units budget per simulation day.
            reflection_thresholds (dict): Configurable thresholds for reflection logic.
        """
        self.sim = Simulation(adversarial_events=adversarial_events)
        self.days = days
        self.compute_budget = compute_budget  # seconds per OODA loop
        self.api_cost = 0.0
        self.api_budget = api_budget
        self.initial_api_budget = api_budget  # For percentage warnings
        self.cpu_units_used = 0.0
        self.cpu_budget = cpu_budget
        self._last_budget_warning = False  # For low budget warning
        self.cpu_budget = cpu_budget
        self.rng = self.sim.rng  # Use simulation's RNG for consistency
        self.goal_stack = deque()  # Hierarchical planner: stack of goals
        self.short_term_memory = {}  # Working memory (dict)
        self.long_term_memory = {
            "episodic": [],
            "semantic": {},
            "procedural": []
        }
        self.reflection_log = []  # Stores OODA loop reflections
        self.strategic_plan = self._initialize_strategic_plan()  # Strategic Plan Document

        # --- Reflection thresholds (configurable) ---
        self.reflection_thresholds = reflection_thresholds or {
            "profit_negative": 0,
            "sales_zero": 0,
            "fee_ratio": 0.5,
            "stockouts": 0,
            "customer_issues": 3
        }

        # --- Agent Product Configuration (now parameterized) ---
        self.asin = asin if asin is not None else "B000TEST"
        self.category = category if category is not None else DEFAULT_CATEGORY
        self.cost = cost if cost is not None else DEFAULT_COST
        self.price = price if price is not None else DEFAULT_PRICE
        self.qty = qty if qty is not None else DEFAULT_QTY

    def _initialize_strategic_plan(self):
        """
        Initialize the Strategic Plan Document as specified in blueprint.
        Must exist and stay coherent with actions; evaluated each checkpoint.
        """
        return {
            "mission": f"Maximize profit for ASIN {getattr(self, 'asin', 'B000TEST')} through strategic FBA operations",
            "objectives": [
                "Achieve positive ROI within 30 days",
                "Maintain inventory turnover > 6x annually",
                "Keep storage fees < 10% of revenue",
                "Maintain customer satisfaction > 4.0 stars"
            ],
            "strategies": {
                "pricing": "Dynamic pricing based on competitor analysis and demand elasticity",
                "inventory": "Just-in-time inventory management to minimize storage costs",
                "marketing": "Targeted PPC campaigns with ROAS > 3.0",
                "customer_service": "Proactive customer issue resolution"
            },
            "kpis": {
                "target_profit_margin": 0.25,
                "max_storage_fee_ratio": 0.10,
                "min_inventory_turnover": 6.0,
                "target_bsr": 10000
            },
            "risks": [
                "Competitor price wars",
                "Supply chain disruptions",
                "Amazon policy changes",
                "Seasonal demand fluctuations"
            ],
            "last_updated": None,
            "coherence_score": 1.0  # Tracks alignment between plan and actions
        }

    def update_strategic_plan(self, updates: Dict):
        """Update strategic plan and maintain coherence with actions."""
        # Check budget before action
        if not self.check_budget(cpu_units=4.0, usd_cost=0.04):
            raise RuntimeError(f"Insufficient budget: need $0.04 and 4.0 CPU, have ${self.api_budget - self.api_cost} and {self.cpu_budget - self.cpu_units_used}")
        # Action logic here...
        self.meter_api_call(cpu_units=4.0, usd_cost=0.04)
        
        for key, value in updates.items():
            if key in self.strategic_plan:
                self.strategic_plan[key] = value
        
        self.strategic_plan["last_updated"] = self.sim.now.day if hasattr(self.sim, 'now') else None
        
        # Evaluate coherence between plan and recent actions
        self._evaluate_plan_coherence()
        
        # Store plan update in episodic memory
        self.store_memory("episodic", {
            "event": "strategic_plan_update",
            "day": self.sim.now.day if hasattr(self.sim, 'now') else None,
            "updates": updates,
            "coherence_score": self.strategic_plan["coherence_score"]
        })

    def _evaluate_plan_coherence(self):
        """Evaluate coherence between strategic plan and recent actions."""
        recent_actions = self.long_term_memory["procedural"][-10:]  # Last 10 actions
        
        coherence_score = 1.0
        
        # Check if actions align with strategies
        pricing_actions = [a for a in recent_actions if "price" in a]
        inventory_actions = [a for a in recent_actions if "inventory" in a or "reorder" in a]
        
        # Penalize for actions that don't align with strategic objectives
        if len(pricing_actions) > 5:  # Too much price tinkering
            coherence_score -= 0.1
        
        if len(inventory_actions) == 0 and len(recent_actions) > 5:  # No inventory management
            coherence_score -= 0.2
            
        self.strategic_plan["coherence_score"] = max(0.0, coherence_score)

    def get_resource_budget(self):
        """
        Get remaining resource budget as specified in blueprint.
        
        Returns:
            dict: Dictionary with remaining API cost budget and CPU units budget
        """
        return {
            "api_cost_remaining": max(0, self.api_budget - self.api_cost),
            "cpu_units_remaining": max(0, self.cpu_budget - self.cpu_units_used),
            "api_cost_used": self.api_cost,
            "cpu_units_used": self.cpu_units_used
        }

    def reset_daily_budgets(self):
        """Reset daily API and CPU budgets for new simulation day."""
        self.api_cost = 0.0
        self.cpu_units_used = 0.0

    def check_budget(self, cpu_units: float = 1.0, usd_cost: float = 0.01) -> bool:
        """
        Check if the API and CPU budgets allow for the specified costs without decrementing.
        Returns True if both budgets are sufficient, False otherwise.
        """
        if usd_cost < 0 or cpu_units < 0:
            print(f"ERROR: Negative cost or CPU units not allowed (usd_cost={usd_cost}, cpu_units={cpu_units})")
            return False
        if (self.api_cost + usd_cost) > self.api_budget:
            print(f"WARNING: API budget insufficient for next call (need {usd_cost}, have {self.api_budget - self.api_cost})")
            return False
        if (self.cpu_units_used + cpu_units) > self.cpu_budget:
            print(f"WARNING: CPU budget insufficient for next call (need {cpu_units}, have {self.cpu_budget - self.cpu_units_used})")
            return False
        return True

    def get_remaining_budget(self) -> dict:
        """
        Get remaining API and CPU budget for introspection.
        """
        return {
            "api_cost_remaining": max(0, self.api_budget - self.api_cost),
            "cpu_units_remaining": max(0, self.cpu_budget - self.cpu_units_used),
            "api_cost_used": self.api_cost,
            "cpu_units_used": self.cpu_units_used
        }

    def meter_api_call(self, cpu_units: float = 1.0, usd_cost: float = 0.01):
        """
        Decrement API and CPU budgets after a successful API/tool call.
        Raises RuntimeError if the budget is exceeded (should not happen if pre-checked).
        Also logs budget status and warnings.
        """
        self.api_cost += usd_cost
        self.cpu_units_used += cpu_units

        if self.api_cost > self.api_budget:
            print(f"ERROR: API cost budget exceeded: {self.api_cost:.2f} > {self.api_budget:.2f}")
            raise RuntimeError(f"API cost budget exceeded: {self.api_cost:.2f} > {self.api_budget:.2f}")

        if self.cpu_units_used > self.cpu_budget:
            print(f"ERROR: CPU budget exceeded: {self.cpu_units_used:.2f} > {self.cpu_budget:.2f}")
            raise RuntimeError(f"CPU budget exceeded: {self.cpu_units_used:.2f} > {self.cpu_budget:.2f}")

        # Log budget status
        remaining_pct = (self.api_budget - self.api_cost) / self.initial_api_budget * 100 if self.initial_api_budget else 0
        if remaining_pct < 10 and not self._last_budget_warning:
            print(f"WARNING: API budget low ({remaining_pct:.1f}% remaining)")
            self._last_budget_warning = True
        elif remaining_pct >= 10:
            self._last_budget_warning = False

    def file_dispute(self, reason: str, details: dict = None):
        """
        File a dispute for FBA recourse simulation with proper processing logic.
        
        Args:
            reason (str): Reason for dispute (e.g., "incorrect_fees", "inventory_damage", "policy_violation")
            details (dict): Additional details/context including amount, evidence, etc.
            
        Returns:
            dict: Dispute record with processing outcome
        """
        if not self.check_budget(cpu_units=8.0, usd_cost=0.08):
            raise RuntimeError(f"Insufficient budget: need $0.08 and 8.0 CPU, have ${self.api_budget - self.api_cost} and {self.cpu_budget - self.cpu_units_used}")
        self.meter_api_call(cpu_units=8.0, usd_cost=0.08)  # High cost for dispute filing
        
        if not hasattr(self.sim, "disputes"):
            self.sim.disputes = []
            
        dispute = {
            "dispute_id": f"D{len(self.sim.disputes) + 1:06d}",
            "day": getattr(self.sim, "now", None),
            "asin": self.asin,
            "reason": reason,
            "details": details or {},
            "status": "filed",
            "filed_timestamp": self.sim.now if hasattr(self.sim, "now") else None,
            "resolution_days": None,
            "outcome": None,
            "refund_amount": 0.0
        }
        
        # Process dispute based on reason and simulate Amazon's response
        dispute = self._process_dispute(dispute)
        self.sim.disputes.append(dispute)
        
        # Log dispute filing
        self.sim.event_log.append(
            f"Day {self.sim.now.day if hasattr(self.sim, 'now') else 'N/A'}: "
            f"Filed dispute {dispute['dispute_id']} for {reason}"
        )
        
        return dispute
    
    def _process_dispute(self, dispute):
        """
        Internal method to process dispute and determine outcome.
        Simulates Amazon's dispute resolution process.
        """
        reason = dispute["reason"]
        details = dispute["details"]
        
        # Simulate processing time (1-14 days)
        resolution_days = self.rng.randint(1, 14) if hasattr(self, 'rng') else 7
        dispute["resolution_days"] = resolution_days
        
        # Determine outcome based on dispute type and evidence quality
        success_probability = self._calculate_dispute_success_probability(reason, details)
        
        if self.rng.uniform(0, 1) < success_probability if hasattr(self, 'rng') else success_probability > 0.5:
            dispute["status"] = "approved"
            dispute["outcome"] = "refund_granted"
            # Calculate refund amount based on dispute type
            dispute["refund_amount"] = self._calculate_refund_amount(reason, details)
        else:
            dispute["status"] = "denied"
            dispute["outcome"] = "insufficient_evidence"
            dispute["refund_amount"] = 0.0
            
        return dispute
    
    def _calculate_dispute_success_probability(self, reason, details):
        """Calculate probability of dispute success based on reason and evidence."""
        base_probabilities = {
            "incorrect_fees": 0.7,
            "inventory_damage": 0.6,
            "policy_violation": 0.4,
            "listing_hijack": 0.8,
            "lost_inventory": 0.5,
            "incorrect_removal": 0.6,
            "unfair_suspension": 0.3
        }
        
        base_prob = base_probabilities.get(reason, 0.4)
        
        # Adjust based on evidence quality
        evidence_quality = details.get("evidence_quality", "medium")
        if evidence_quality == "high":
            base_prob += 0.2
        elif evidence_quality == "low":
            base_prob -= 0.2
            
        # Adjust based on amount disputed
        amount = details.get("amount", 0)
        if amount > 1000:
            base_prob -= 0.1  # Harder to win large disputes
        elif amount < 50:
            base_prob += 0.1  # Easier to win small disputes
            
        return max(0.1, min(0.9, base_prob))
    
    def _calculate_refund_amount(self, reason, details):
        """Calculate refund amount for approved disputes."""
        amount_requested = details.get("amount", 0)
        
        # Different dispute types have different refund rates
        refund_rates = {
            "incorrect_fees": 1.0,  # Full refund for fee errors
            "inventory_damage": 0.8,  # Partial compensation
            "policy_violation": 0.5,  # Limited compensation
            "listing_hijack": 0.9,  # High compensation for hijacking
            "lost_inventory": 0.7,  # Partial compensation for lost items
            "incorrect_removal": 1.0,  # Full refund for incorrect removals
            "unfair_suspension": 0.3  # Limited compensation for suspensions
        }
        
        refund_rate = refund_rates.get(reason, 0.6)
        return round(amount_requested * refund_rate, 2)

    # --- Hierarchical Planner (DAG/Goal Stack) ---
    def push_goal(self, goal: Dict[str, Any]):
        """Push a new goal onto the goal stack."""
        self.goal_stack.append(goal)

    def pop_goal(self):
        """Pop the top goal from the goal stack."""
        if self.goal_stack:
            return self.goal_stack.pop()
        return None


    def decompose_goal(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Recursively decompose a high-level goal into atomic subgoals.
        Returns a DAG (Directed Acyclic Graph) of subgoals with dependencies.
        
        Args:
            goal: Goal dictionary with 'type', 'params', and optional 'priority'
            
        Returns:
            List of subgoal dictionaries with dependency information
        """
        if not self.check_budget(cpu_units=3.0, usd_cost=0.03):
            raise RuntimeError(f"Insufficient budget: need $0.03 and 3.0 CPU, have ${self.api_budget - self.api_cost} and {self.cpu_budget - self.cpu_units_used}")
        self.meter_api_call(cpu_units=3.0, usd_cost=0.03)  # Planning is computationally expensive
        
        goal_type = goal.get("type")
        goal_params = goal.get("params", {})
        
        # Check if goal is already atomic (cannot be decomposed further)
        if self._is_atomic_goal(goal_type):
            return [goal]
        
        # Recursive decomposition based on goal type
        subgoals = []
        
        if goal_type == "maximize_profit":
            subgoals = self._decompose_maximize_profit(goal_params)
        elif goal_type == "investigate_no_sales":
            subgoals = self._decompose_investigate_no_sales(goal_params)
        elif goal_type == "recover_from_loss":
            subgoals = self._decompose_recover_from_loss(goal_params)
        elif goal_type == "reduce_fees":
            subgoals = self._decompose_reduce_fees(goal_params)
        elif goal_type == "improve_customer_experience":
            subgoals = self._decompose_improve_customer_experience(goal_params)
        elif goal_type == "analyze_market":
            subgoals = self._decompose_analyze_market(goal_params)
        else:
            # Unknown goal type, treat as atomic
            return [goal]
        
        # Recursively decompose non-atomic subgoals
        final_subgoals = []
        for subgoal in subgoals:
            if self._is_atomic_goal(subgoal["type"]):
                final_subgoals.append(subgoal)
            else:
                # Recursive decomposition
                nested_subgoals = self.decompose_goal(subgoal)
                final_subgoals.extend(nested_subgoals)
        
        return final_subgoals
    
    def _is_atomic_goal(self, goal_type: str) -> bool:
        """Check if a goal type is atomic (cannot be decomposed further)."""
        atomic_goals = {
            "launch_product", "adjust_price", "reorder_inventory",
            "create_ad_campaign", "handle_customer_inquiry", "review_strategy",
            "check_inventory_levels", "analyze_competitor_prices", "optimize_keywords",
            "respond_to_review", "file_dispute", "update_listing", "noop",
            "procure_inventory", "evaluate_suppliers"
        }
        return goal_type in atomic_goals
    
    def _decompose_maximize_profit(self, params: Dict) -> List[Dict]:
        """
        Decompose maximize_profit into comprehensive subgoals with dependencies.
        
        Enhanced strategy considers:
        - Product lifecycle stage
        - Current profitability metrics
        - Inventory optimization
        - Fee minimization
        - Market positioning
        - Supply chain efficiency
        """
        target_profit = params.get("target_profit", 1000)
        time_horizon = params.get("time_horizon", 30)  # days
        
        # Analyze current state
        current_profit = self.sim.ledger.balance("Cash") - Money(10000, "USD")  # Subtract seed capital
        subgoals = []
        
        # Phase 1: Product Launch (if needed)
        if self.asin not in self.sim.products:
            subgoals.extend([
                {
                    "type": "evaluate_suppliers",
                    "params": {"quantity": self.qty, "cost_priority": True},
                    "priority": 1,
                    "dependencies": []
                },
                {
                    "type": "launch_product",
                    "params": {"optimize_for": "profit_margin"},
                    "priority": 2,
                    "dependencies": ["evaluate_suppliers"]
                }
            ])
            return subgoals
        
        # Phase 2: Comprehensive Profit Optimization
        product = self.sim.products[self.asin]
        
        # Always start with market intelligence
        subgoals.append({
            "type": "analyze_market",
            "params": {"focus": "comprehensive", "include_competitors": True},
            "priority": 1,
            "dependencies": []
        })
        
        # Inventory optimization strategy
        current_inventory = product.qty  # Use the qty property
        if current_inventory < 10:  # Low inventory
            subgoals.extend([
                {
                    "type": "evaluate_suppliers",
                    "params": {"quantity": max(50, self.qty), "speed_priority": True},
                    "priority": 2,
                    "dependencies": ["analyze_market"]
                },
                {
                    "type": "procure_inventory",
                    "params": {"urgent": True, "target_days_supply": 45},
                    "priority": 3,
                    "dependencies": ["evaluate_suppliers"]
                }
            ])
        elif current_inventory > 200:  # Excess inventory
            subgoals.append({
                "type": "optimize_inventory_turnover",
                "params": {"target_velocity": 8.0},
                "priority": 2,
                "dependencies": ["analyze_market"]
            })
        
        # Fee optimization strategy
        subgoals.append({
            "type": "minimize_fees",
            "params": {"focus": ["storage", "fulfillment", "referral"]},
            "priority": 2,
            "dependencies": ["analyze_market"]
        })
        
        # Pricing optimization based on current performance
        profit_margin = self._calculate_profit_margin()
        if profit_margin < 0.15:  # Low margin
            subgoals.append({
                "type": "optimize_pricing",
                "params": {
                    "strategy": "margin_improvement",
                    "target_margin": 0.25,
                    "consider_elasticity": True
                },
                "priority": 3,
                "dependencies": ["analyze_market", "minimize_fees"]
            })
        elif profit_margin > 0.4:  # High margin, focus on volume
            subgoals.append({
                "type": "optimize_pricing",
                "params": {
                    "strategy": "volume_maximization",
                    "elasticity_test": True
                },
                "priority": 3,
                "dependencies": ["analyze_market"]
            })
        else:  # Balanced approach
            subgoals.append({
                "type": "optimize_pricing",
                "params": {
                    "strategy": "balanced",
                    "target_profit": target_profit
                },
                "priority": 3,
                "dependencies": ["analyze_market", "minimize_fees"]
            })
        
        # Performance-based additional strategies
        sales_velocity = self._calculate_sales_velocity()
        
        if sales_velocity < 1.0:  # Low sales
            subgoals.extend([
                {
                    "type": "investigate_no_sales",
                    "params": {"deep_analysis": True},
                    "priority": 3,
                    "dependencies": ["analyze_market"]
                },
                {
                    "type": "improve_listing_quality",
                    "params": {"focus": ["keywords", "images", "description"]},
                    "priority": 4,
                    "dependencies": ["investigate_no_sales"]
                }
            ])
        
        target_threshold = Money.from_dollars(float(target_profit * Decimal('0.3')))
        if current_profit < target_threshold:  # Significantly underperforming
            subgoals.append({
                "type": "recover_from_loss",
                "params": {"aggressive": True, "target_profit": target_profit},
                "priority": 4,
                "dependencies": ["optimize_pricing"]
            })
        
        # Long-term growth strategies (if performing well)
        if current_profit > Money.from_dollars(float(target_profit * Decimal('0.8'))) and sales_velocity > 2.0:
            subgoals.extend([
                {
                    "type": "scale_operations",
                    "params": {"expansion_factor": 1.5},
                    "priority": 5,
                    "dependencies": ["optimize_pricing"]
                },
                {
                    "type": "diversify_suppliers",
                    "params": {"risk_mitigation": True},
                    "priority": 5,
                    "dependencies": ["scale_operations"]
                }
            ])
        
        return subgoals
    
    def _calculate_profit_margin(self) -> float:
        """Calculate current profit margin for the product."""
        if self.asin not in self.sim.products:
            return 0.0
        
        product = self.sim.products[self.asin]
        price = product.price
        cost = product.cost
        
        # Estimate fees (simplified)
        estimated_fees = price * Decimal('0.15')  # Rough estimate of total fees
        profit = price - cost - estimated_fees
        
        return profit / price if price > Money(0, "USD") else Decimal('0.0')
    
    def _calculate_sales_velocity(self) -> float:
        """Calculate current sales velocity (units per day)."""
        if self.asin not in self.sim.products:
            return 0.0
        
        # Get recent sales data from simulation
        product = self.sim.products[self.asin]
        
        # Use the sales_velocity attribute from the Product class
        return product.sales_velocity
    
    def _decompose_investigate_no_sales(self, params: Dict) -> List[Dict]:
        """Decompose investigate_no_sales into diagnostic and corrective actions."""
        return [
            {
                "type": "analyze_market",
                "params": {"focus": "competition"},
                "priority": 1,
                "dependencies": []
            },
            {
                "type": "check_inventory_levels",
                "params": {},
                "priority": 1,
                "dependencies": []
            },
            {
                "type": "adjust_price",
                "params": {"discount": 0.1},
                "priority": 2,
                "dependencies": ["analyze_market"]
            },
            {
                "type": "create_ad_campaign",
                "params": {"budget": 100},
                "priority": 3,
                "dependencies": ["adjust_price"]
            }
        ]
    
    def _decompose_recover_from_loss(self, params: Dict) -> List[Dict]:
        """Decompose recovery strategy into cost reduction and revenue improvement."""
        return [
            {
                "type": "reduce_fees",
                "params": {},
                "priority": 1,
                "dependencies": []
            },
            {
                "type": "analyze_market",
                "params": {"focus": "cost_optimization"},
                "priority": 1,
                "dependencies": []
            },
            {
                "type": "adjust_price",
                "params": {"increase": 0.05},
                "priority": 2,
                "dependencies": ["analyze_market"]
            }
        ]
    
    def _decompose_reduce_fees(self, params: Dict) -> List[Dict]:
        """Decompose fee reduction into specific optimization actions."""
        return [
            {
                "type": "check_inventory_levels",
                "params": {"focus": "storage_optimization"},
                "priority": 1,
                "dependencies": []
            },
            {
                "type": "optimize_keywords",
                "params": {"reduce_ppc_costs": True},
                "priority": 2,
                "dependencies": []
            }
        ]
    
    def _decompose_improve_customer_experience(self, params: Dict) -> List[Dict]:
        """Decompose customer experience improvement."""
        return [
            {
                "type": "handle_customer_inquiry",
                "params": {},
                "priority": 1,
                "dependencies": []
            },
            {
                "type": "respond_to_review",
                "params": {"focus": "negative_reviews"},
                "priority": 2,
                "dependencies": []
            },
            {
                "type": "update_listing",
                "params": {"improve_description": True},
                "priority": 3,
                "dependencies": ["respond_to_review"]
            }
        ]
    
    def _decompose_analyze_market(self, params: Dict) -> List[Dict]:
        """Decompose market analysis into specific research actions."""
        focus = params.get("focus", "general")
        
        subgoals = [
            {
                "type": "analyze_competitor_prices",
                "params": {},
                "priority": 1,
                "dependencies": []
            }
        ]
        
        if focus in ["pricing", "general"]:
            subgoals.append({
                "type": "check_inventory_levels",
                "params": {},
                "priority": 1,
                "dependencies": []
            })
        
        return subgoals

    # --- Memory System ---
    def query_memory(self, partition: str, query: Any, limit: int = 10) -> Any:
        """
        Enhanced memory query with vector store capabilities.
        Args:
            partition (str): "episodic", "semantic", or "procedural"
            query (Any): Query object (e.g., key, filter function, callable, or semantic query)
            limit (int): Maximum number of results to return
        Returns:
            Any: Retrieved memory or None
        """
        if not self.check_budget(cpu_units=1.0, usd_cost=0.01):
            raise RuntimeError(f"Insufficient budget: need $0.01 and 1.0 CPU, have ${self.api_budget - self.api_cost} and {self.cpu_budget - self.cpu_units_used}")
        self.meter_api_call(cpu_units=1.0, usd_cost=0.01)  # Memory access has cost
        
        if partition == "episodic":
            episodes = self.long_term_memory["episodic"]
            
            if query is None:
                return episodes[-limit:] if limit else episodes
            
            # String: semantic search or key-based filter
            if isinstance(query, str):
                # First try exact key match
                exact_matches = [ep for ep in episodes if ep.get("asin") == query or ep.get("event") == query]
                if exact_matches:
                    return exact_matches[-limit:]
                
                # Then try semantic search (simple keyword matching)
                semantic_matches = []
                for ep in episodes:
                    similarity = self._semantic_similarity(query, ep)
                    if similarity > 0.03:  # Even lower threshold for better matching with context
                        semantic_matches.append((ep, similarity))
                
                # Sort by similarity and return episodes (not tuples)
                semantic_matches.sort(key=lambda x: x[1], reverse=True)
                return [match[0] for match in semantic_matches[:limit]]
            
            # Callable: advanced filter
            if callable(query):
                matches = [ep for ep in episodes if query(ep)]
                return matches[-limit:]
                
        elif partition == "semantic":
            if isinstance(query, str):
                # Support fuzzy semantic lookup
                exact_match = self.long_term_memory["semantic"].get(query)
                if exact_match:
                    return exact_match
                
                # Fuzzy search for similar keys
                similar_keys = [k for k in self.long_term_memory["semantic"].keys()
                              if query.lower() in k.lower() or k.lower() in query.lower()]
                if similar_keys:
                    return {k: self.long_term_memory["semantic"][k] for k in similar_keys}
            
            return self.long_term_memory["semantic"].get(query)
            
        elif partition == "procedural":
            procedures = self.long_term_memory["procedural"]
            
            # String: filter by substring with relevance scoring
            if isinstance(query, str):
                matches = []
                for i, p in enumerate(procedures):
                    if query in str(p):
                        # More recent procedures get higher relevance
                        relevance = (i + 1) / len(procedures)
                        matches.append((p, relevance))
                
                # Sort by relevance and return top matches
                matches.sort(key=lambda x: x[1], reverse=True)
                return [m[0] for m in matches[:limit]]
            
            # Callable: advanced filter
            if callable(query):
                matches = [p for p in procedures if query(p)]
                return matches[-limit:]
                
        return None

    def _semantic_similarity(self, query: str, episode: Dict) -> float:
        """
        Calculate semantic similarity between query and episode.
        Enhanced implementation using keyword overlap and substring matching.
        """
        query_words = set(query.lower().split())
        episode_text = " ".join(str(v) for v in episode.values()).lower()
        episode_words = set(episode_text.split())
        
        if not query_words or not episode_words:
            return 0.0
        
        # Exact word intersection
        intersection = query_words.intersection(episode_words)
        
        # Substring matching - check if query words are contained in episode words
        substring_matches = 0
        for query_word in query_words:
            for episode_word in episode_words:
                if query_word in episode_word or episode_word in query_word:
                    substring_matches += 1
                    break
        
        union = query_words.union(episode_words)
        
        # Combine exact matches and substring matches
        total_matches = len(intersection) + (substring_matches * 0.5)  # Weight substring matches less
        
        return total_matches / len(union) if union else 0.0

    def store_memory(self, partition: str, value: Any, consolidate: bool = True):
        """
        Enhanced memory storage with consolidation and capacity management.

        Args:
            partition (str): "episodic", "semantic", or "procedural"
            value (Any): The value to store. For "semantic", should be a dict; for others, any object.
            consolidate (bool): Whether to perform memory consolidation after storage
        """
        self.meter_api_call(cpu_units=0.5, usd_cost=0.005)  # Memory storage has small cost
        
        if partition == "episodic":
            # Add timestamp and context to episodic memories
            if isinstance(value, dict):
                value["stored_at"] = self.sim.now.day if hasattr(self.sim, 'now') else None
                value["context"] = self._get_current_context()
            
            self.long_term_memory["episodic"].append(value)
            
            # Manage episodic memory capacity (keep last 1000 episodes)
            if len(self.long_term_memory["episodic"]) > 1000:
                self.long_term_memory["episodic"] = self.long_term_memory["episodic"][-1000:]
                
        elif partition == "semantic":
            if isinstance(value, dict):
                # Update semantic memory with conflict resolution
                for key, val in value.items():
                    if key in self.long_term_memory["semantic"]:
                        # If updating existing knowledge, track confidence
                        old_val = self.long_term_memory["semantic"][key]
                        if isinstance(old_val, dict) and isinstance(val, dict):
                            # Merge dictionaries with new values taking precedence
                            merged = {**old_val, **val}
                            merged["last_updated"] = self.sim.now.day if hasattr(self.sim, 'now') else None
                            self.long_term_memory["semantic"][key] = merged
                        else:
                            self.long_term_memory["semantic"][key] = val
                    else:
                        self.long_term_memory["semantic"][key] = val
            else:
                # Handle non-dict semantic storage
                timestamp = self.sim.now.day if hasattr(self.sim, 'now') else None
                self.long_term_memory["semantic"][f"entry_{timestamp}"] = value
                
        elif partition == "procedural":
            # Add execution context to procedural memories
            if isinstance(value, str):
                enhanced_value = {
                    "action": value,
                    "day": self.sim.now.day if hasattr(self.sim, 'now') else None,
                    "context": self._get_current_context()
                }
                self.long_term_memory["procedural"].append(enhanced_value)
            else:
                self.long_term_memory["procedural"].append(value)
            
            # Manage procedural memory capacity (keep last 500 procedures)
            if len(self.long_term_memory["procedural"]) > 500:
                self.long_term_memory["procedural"] = self.long_term_memory["procedural"][-500:]
        
        # Perform memory consolidation if requested
        if consolidate and len(self.long_term_memory["episodic"]) % 10 == 0:
            self._consolidate_memory()

    def _get_current_context(self) -> Dict:
        """Get current context for memory storage."""
        context = {
            "day": self.sim.now.day if hasattr(self.sim, 'now') else None,
            "cash_balance": self.sim.ledger.balance("Cash") if hasattr(self.sim, 'ledger') else None,
            "goal_stack_size": len(self.goal_stack),
            "api_budget_used": self.api_cost / self.api_budget if self.api_budget > 0 else 0
        }
        
        # Add product-specific context if available
        if self.asin in self.sim.products:
            prod = self.sim.products[self.asin]
            context.update({
                "current_price": prod.price,
                "current_bsr": prod.bsr,
                "recent_sales": prod.sales_history[-3:] if prod.sales_history else []
            })
        
        return context

    def _consolidate_memory(self):
        """
        Consolidate memories by identifying patterns and creating semantic knowledge.
        Transfers important episodic memories to semantic memory.
        """
        self.meter_api_call(cpu_units=2.0, usd_cost=0.02)  # Consolidation is computationally expensive
        
        episodes = self.long_term_memory["episodic"]
        if len(episodes) < 10:
            return
        
        # Identify patterns in recent episodes
        recent_episodes = episodes[-20:]  # Look at last 20 episodes
        
        # Pattern 1: Repeated actions and their outcomes
        action_outcomes = {}
        for ep in recent_episodes:
            if "decision" in ep and "profit" in ep:
                action = ep.get("decision", {}).get("action", "unknown")
                profit = ep.get("profit", 0)
                
                if action not in action_outcomes:
                    action_outcomes[action] = []
                action_outcomes[action].append(profit)
        
        # Store action effectiveness in semantic memory
        for action, profits in action_outcomes.items():
            if len(profits) >= 3:  # Need at least 3 data points
                avg_profit = sum(profits) / len(profits)
                self.store_memory("semantic", {
                    f"action_effectiveness_{action}": {
                        "avg_profit_impact": avg_profit,
                        "sample_size": len(profits),
                        "confidence": min(1.0, len(profits) / 10.0)
                    }
                }, consolidate=False)
        
        # Pattern 2: Market conditions and performance
        market_conditions = []
        for ep in recent_episodes:
            if "day" in ep and "profit" in ep:
                market_conditions.append({
                    "day": ep["day"],
                    "profit": ep["profit"],
                    "sales": ep.get("sales", 0)
                })
        
        if len(market_conditions) >= 5:
            # Identify trends
            profits = [mc["profit"] for mc in market_conditions]
            if len(profits) > 1:
                trend = "increasing" if profits[-1] > profits[0] else "decreasing"
                self.store_memory("semantic", {
                    "recent_performance_trend": {
                        "trend": trend,
                        "start_profit": profits[0],
                        "end_profit": profits[-1],
                        "period_days": len(profits)
                    }
                }, consolidate=False)

    def reflect_and_self_correct(self, obs, decision, outcome):
        """
        Formal reflection step: analyze outcome, update semantic/procedural memory, adapt plan.
        Enhanced: reacts to negative profit, low sales, high fees, inventory issues, and customer complaints.
        """
        profit = outcome.get("profit", 0)
        sales = outcome.get("sales", 0)
        fees = outcome.get("fees", 0)
        stockouts = outcome.get("stockouts", 0)
        customer_issues = outcome.get("customer_issues", 0)
        goal_missed = outcome.get("goal_missed", False)

        reflection = {
            "profit": profit,
            "sales": sales,
            "fees": fees,
            "stockouts": stockouts,
            "customer_issues": customer_issues,
            "goal_missed": goal_missed,
            "decision": decision,
            "day": obs.get("day") if isinstance(obs, dict) else None
        }
        self.reflection_log.append(reflection)
        self.store_memory("semantic", {"last_profit": profit, "last_sales": sales, "last_fees": fees})

        # Negative profit (check this first, higher priority)
        if profit < self.reflection_thresholds.get("profit_negative", 0):
            self.push_goal({"type": "recover_from_loss", "params": {}})
        # Low sales (only if profit is not positive - don't investigate if we're making money)
        elif sales <= self.reflection_thresholds.get("sales_zero", 0) and profit <= 0:
            self.push_goal({"type": "investigate_no_sales", "params": {}})
        # High fees (configurable threshold: > fee_ratio * (profit + fees))
        fee_ratio = self.reflection_thresholds.get("fee_ratio", 0.5)
        if sales > 0 and fees > 0 and fees > fee_ratio * (profit + fees):
            self.push_goal({"type": "reduce_fees", "params": {}})
        # Inventory stockouts
        if stockouts > self.reflection_thresholds.get("stockouts", 0):
            self.push_goal({"type": "restock_inventory", "params": {}})
        # High customer complaints
        if customer_issues > self.reflection_thresholds.get("customer_issues", 3):
            self.push_goal({"type": "reduce_customer_issues", "params": {}})
        if customer_issues > 2:
            self.push_goal({"type": "improve_customer_experience", "params": {}})
        # Missed goal
        if goal_missed:
            self.push_goal({"type": "review_strategy", "params": {}})

    def get_event_log(self):
        """Return the simulation's adversarial event log for analysis/hardening."""
        return getattr(self.sim, "event_log", [])

# Duplicate push_goal and pop_goal removed; see earlier definition for push_goal.

    def observe(self):
        """Observe the environment and update short-term memory."""
        self.meter_api_call(cpu_units=2.0, usd_cost=0.02)  # Observation is a moderate cost operation
        obs = {
            "day": self.sim.now.day,
            "products": {asin: prod for asin, prod in self.sim.products.items()},
            "ledger": self.sim.ledger,
        }
        self.short_term_memory["last_observation"] = obs
        return obs

    def orient(self, obs):
        """Orient: interpret observations, update semantic/episodic memory."""
        # Example: store sales and cash in episodic memory
        for asin, prod in obs["products"].items():
            episode = {
                "day": obs["day"],
                "asin": asin,
                "sales": list(prod.sales_history),
                "price": prod.price,
                "cash": obs["ledger"].balance("Cash"),
            }
            self.long_term_memory["episodic"].append(episode)
        # Semantic: update running stats
        self.long_term_memory["semantic"]["last_cash"] = obs["ledger"].balance("Cash")

    def decide(self):
        """Decide: select or decompose goals, plan actions using hierarchical planner."""
        self.meter_api_call(cpu_units=5.0, usd_cost=0.05)  # Decision making is computationally expensive
        
        if not self.goal_stack:
            # Start with top-level goal to maximize profit
            self.push_goal({"type": "maximize_profit", "params": {"target_profit": 1000}})
        
        current_goal = self.goal_stack[-1]
        
        # Check if current goal is atomic (can be executed directly)
        if self._is_atomic_goal(current_goal["type"]):
            # Execute atomic goal
            action = self._goal_to_action(current_goal)
            # Remove completed goal from stack
            self.pop_goal()
            return action
        else:
            # Decompose non-atomic goal into subgoals
            subgoals = self.decompose_goal(current_goal)
            # Remove the decomposed goal
            self.pop_goal()
            
            # Add subgoals to stack in reverse priority order (highest priority last)
            # This ensures highest priority goals are processed first
            sorted_subgoals = sorted(subgoals, key=lambda g: g.get("priority", 999))
            for subgoal in sorted_subgoals:
                self.push_goal(subgoal)
            
            # Recursively decide on the new top goal
            if self.goal_stack:
                return self.decide()
            else:
                return {"action": "noop"}
    
    def _goal_to_action(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an atomic goal to an executable action."""
        goal_type = goal["type"]
        params = goal.get("params", {})
        
        if goal_type == "launch_product":
            return {"action": "launch_product"}
        elif goal_type == "adjust_price":
            new_price = params.get("new_price")
            if new_price is None:
                # Calculate new price based on parameters
                if "discount" in params:
                    new_price = self.price * (1 - params["discount"])
                elif "increase" in params:
                    new_price = self.price * (1 + params["increase"])
                else:
                    new_price = self.price
            return {"action": "adjust_price", "new_price": new_price}
        elif goal_type == "reorder_inventory":
            qty = params.get("qty", 50)
            return {"action": "reorder_inventory", "qty": qty}
        elif goal_type == "create_ad_campaign":
            budget = params.get("budget", 100)
            return {"action": "create_ad_campaign", "budget": budget}
        elif goal_type == "handle_customer_inquiry":
            return {"action": "handle_customer_inquiry"}
        elif goal_type == "file_dispute":
            reason = params.get("reason", "incorrect_fees")
            details = params.get("details", {})
            return {"action": "file_dispute", "reason": reason, "details": details}
        elif goal_type == "procure_inventory":
            quantity = params.get("quantity", 100)
            priority = params.get("priority", "cost")
            shipping_method = params.get("shipping_method", "sea")
            return {"action": "procure_inventory", "quantity": quantity, "priority": priority, "shipping_method": shipping_method}
        elif goal_type == "evaluate_suppliers":
            quantity = params.get("quantity", 100)
            return {"action": "evaluate_suppliers", "quantity": quantity}
        elif goal_type in ["analyze_competitor_prices", "check_inventory_levels",
                          "optimize_keywords", "respond_to_review", "update_listing",
                          "analyze_market", "review_strategy"]:
            # These are placeholder actions that log the activity
            return {"action": "analyze_and_log", "analysis_type": goal_type, "params": params}
        else:
            return {"action": "noop"}

    def act(self, decision):
        """Act: execute the chosen action in the environment."""
        if decision.get("action") == "noop":
            self.meter_api_call(cpu_units=0.5, usd_cost=0.001)  # Minimal cost for no-op
            return
        
        # Different actions have different API costs
        action = decision["action"]
        if action == "launch_product":
            self.meter_api_call(cpu_units=10.0, usd_cost=0.10)  # High cost for product launch
            self.sim.launch_product(
                asin=self.asin,
                category=self.category,
                cost=self.cost,
                price=self.price,
                qty=self.qty
            )
            self.long_term_memory["procedural"].append("launch_product")
        elif action == "adjust_price":
            self.meter_api_call(cpu_units=3.0, usd_cost=0.03)  # Medium cost for price adjustment
            prod = self.sim.products.get(self.asin)
            if prod:
                prod.price = decision["new_price"]
                self.long_term_memory["procedural"].append("adjust_price")
        elif action == "reorder_inventory":
            self.meter_api_call(cpu_units=5.0, usd_cost=0.05)  # Medium-high cost for inventory management
            qty = decision.get("qty", 50)
            self.sim.inventory.add(self.asin, qty, self.cost, self.sim.now)
            self.long_term_memory["procedural"].append("reorder_inventory")
        elif action == "create_ad_campaign":
            self.meter_api_call(cpu_units=7.0, usd_cost=0.07)  # High cost for marketing actions
            budget = decision.get("budget", 100)
            # Placeholder: log ad campaign creation
            self.sim.event_log.append(f"Day {self.sim.now.day}: Created ad campaign for {self.asin} with budget ${budget}")
            self.long_term_memory["procedural"].append("create_ad_campaign")
        elif action == "handle_customer_inquiry":
            self.meter_api_call(cpu_units=2.0, usd_cost=0.02)  # Low-medium cost for customer service
            # Placeholder: simulate customer inquiry handling
            self.sim.event_log.append(f"Day {self.sim.now.day}: Handled customer inquiry for {self.asin}")
            self.long_term_memory["procedural"].append("handle_customer_inquiry")
        elif action == "file_dispute":
            # Use the enhanced dispute filing system
            reason = decision.get("reason", "incorrect_fees")
            details = decision.get("details", {})
            dispute_result = self.file_dispute(reason, details)
            self.long_term_memory["procedural"].append(f"file_dispute_{reason}")
        elif action == "analyze_and_log":
            self.meter_api_call(cpu_units=3.0, usd_cost=0.03)  # Medium cost for analysis
            analysis_type = decision.get("analysis_type", "general")
            params = decision.get("params", {})
            # Simulate analysis and log results
            self.sim.event_log.append(f"Day {self.sim.now.day}: Performed {analysis_type} analysis for {self.asin}")
            # Store analysis results in memory
            analysis_result = self._perform_analysis(analysis_type, params)
            self.store_memory("semantic", {f"last_{analysis_type}": analysis_result})
            self.long_term_memory["procedural"].append(analysis_type)
        elif action == "procure_inventory":
            self.meter_api_call(cpu_units=6.0, usd_cost=0.06)  # High cost for procurement decisions
            quantity = decision.get("quantity", 100)
            priority = decision.get("priority", "cost")  # cost, speed, quality, reliability
            
            # Find best supplier using supply chain
            result = self.sim.supply_chain.find_best_supplier(quantity, self.cost, priority)
            if result:
                supplier, order_preview = result
                # Place order with selected supplier
                order = self.sim.supply_chain.place_order(
                    supplier.supplier_id, quantity, self.cost,
                    decision.get("shipping_method", "sea")
                )
                if order:
                    # Record procurement in ledger
                    from fba_bench.ledger import Transaction, Entry
                    self.sim.ledger.post(Transaction(
                        f"Procurement order {order['order_id']}",
                        [Entry("Inventory", order["total_cost"], self.sim.now)],
                        [Entry("Cash", order["total_cost"], self.sim.now)]
                    ))
                    self.sim.event_log.append(
                        f"Day {self.sim.now.day}: Placed procurement order {order['order_id']} "
                        f"with {supplier.name} for {quantity} units (${order['total_cost']:.2f})"
                    )
                    self.long_term_memory["procedural"].append("procure_inventory")
                    # Store procurement decision in memory
                    self.store_memory("episodic", {
                        "event": "procurement",
                        "supplier": supplier.supplier_id,
                        "quantity": quantity,
                        "cost": order["total_cost"],
                        "lead_time": order["lead_time_days"],
                        "priority": priority
                    })
            else:
                self.sim.event_log.append(
                    f"Day {self.sim.now.day}: No suppliers available for {quantity} units"
                )
        elif action == "evaluate_suppliers":
            self.meter_api_call(cpu_units=4.0, usd_cost=0.04)  # Medium-high cost for supplier analysis
            quantity = decision.get("quantity", 100)
            
            # Get supplier analytics and available suppliers
            analytics = self.sim.supply_chain.get_supplier_analytics()
            available_suppliers = self.sim.supply_chain.get_available_suppliers(quantity)
            
            # Store supplier evaluation in memory
            supplier_evaluation = {
                "total_suppliers": analytics["total_suppliers"],
                "available_for_quantity": len(available_suppliers),
                "avg_reputation": analytics["avg_reputation"],
                "suppliers": []
            }
            
            for supplier in available_suppliers[:5]:  # Top 5 suppliers
                supplier_info = {
                    "id": supplier.supplier_id,
                    "name": supplier.name,
                    "type": supplier.supplier_type.value,
                    "unit_cost": supplier.calculate_unit_cost(self.cost),
                    "lead_time": supplier.calculate_total_lead_time(),
                    "reputation": supplier.reputation_score,
                    "qc_risk": supplier.qc_risk_probability
                }
                supplier_evaluation["suppliers"].append(supplier_info)
            
            self.store_memory("semantic", {"last_supplier_evaluation": supplier_evaluation})
            self.sim.event_log.append(
                f"Day {self.sim.now.day}: Evaluated {len(available_suppliers)} suppliers for {quantity} units"
            )
            self.long_term_memory["procedural"].append("evaluate_suppliers")
        else:
            self.meter_api_call(cpu_units=1.0, usd_cost=0.01)  # Default cost for unknown actions
        # Add more actions as needed
    
    def _perform_analysis(self, analysis_type: str, params: Dict) -> Dict:
        """Perform analysis and return results for memory storage."""
        if analysis_type == "analyze_competitor_prices":
            # Analyze competitor pricing
            competitors = getattr(self.sim, 'competitors', [])
            if competitors:
                avg_price = sum(c.price for c in competitors) / len(competitors)
                min_price = min(c.price for c in competitors)
                max_price = max(c.price for c in competitors)
                return {
                    "avg_competitor_price": avg_price,
                    "min_competitor_price": min_price,
                    "max_competitor_price": max_price,
                    "our_price": self.price,
                    "price_position": "competitive" if min_price <= self.price <= max_price else "outlier"
                }
            return {"error": "no_competitors"}
        
        elif analysis_type == "check_inventory_levels":
            # Check current inventory levels
            batches = self.sim.inventory._batches.get(self.asin, [])
            total_qty = sum(getattr(batch, "quantity", 0) for batch in batches)
            return {
                "current_inventory": total_qty,
                "days_of_supply": total_qty / max(1, self._estimate_daily_demand()),
                "reorder_needed": total_qty < 20
            }
        
        elif analysis_type == "analyze_market":
            # General market analysis
            focus = params.get("focus", "general")
            prod = self.sim.products.get(self.asin)
            if prod:
                recent_sales = prod.sales_history[-7:] if len(prod.sales_history) >= 7 else prod.sales_history
                avg_daily_sales = sum(recent_sales) / max(1, len(recent_sales))
                return {
                    "focus": focus,
                    "avg_daily_sales": avg_daily_sales,
                    "current_bsr": prod.bsr,
                    "current_price": prod.price,
                    "sales_trend": "increasing" if len(recent_sales) > 1 and recent_sales[-1] > recent_sales[0] else "stable"
                }
            return {"error": "no_product"}
        
        else:
            # Default analysis
            return {"analysis_type": analysis_type, "timestamp": self.sim.now.day if hasattr(self.sim, 'now') else None}
    
    def _estimate_daily_demand(self) -> float:
        """Estimate daily demand based on recent sales history."""
        prod = self.sim.products.get(self.asin)
        if prod and prod.sales_history:
            recent_sales = prod.sales_history[-7:] if len(prod.sales_history) >= 7 else prod.sales_history
            return sum(recent_sales) / max(1, len(recent_sales))
        return 1.0  # Default estimate

    def ooda_loop(self):
        """Run a single OODA (Observe-Orient-Decide-Act-Reflect) cycle with compute budget."""
        start_time = time.time()
        obs = self.observe()
        self.orient(obs)
        decision = self.decide()
        self.act(decision)
        # After acting, reflect and self-correct
        # --- Compute real outcome metrics ---
        # Fees: sum all "Fees" entries for transactions related to this ASIN
        total_fees = 0.0
        for txn in self.sim.ledger.entries:
            if self.asin in getattr(txn, "description", ""):
                for entry in txn.debits + txn.credits:
                    if entry.account == "Fees":
                        total_fees += entry.amount

        # Stockouts: sum all batch quantities for this ASIN in inventory manager
        batches = self.sim.inventory._batches.get(self.asin, [])
        qty_on_hand = sum(getattr(batch, "quantity", 0) for batch in batches)
        stockouts = 1 if qty_on_hand == 0 else 0

        # Customer issues: count negative events for this ASIN
        customer_events = self.sim.customer_events.get(self.asin, [])
        customer_issues = sum(
            1 for e in customer_events if isinstance(e, dict) and e.get("type") in ["negative_review", "negative_feedback"]
        )

        outcome = {
            "profit": self.sim.ledger.balance("Cash") - 10000,
            "sales": sum(self.sim.products.get(self.asin, None).sales_history) if self.asin in self.sim.products else 0,
            "fees": abs(total_fees),  # Fees are negative in ledger, report as positive
            "stockouts": stockouts,
            "customer_issues": customer_issues,
            "goal_missed": False  # Placeholder
        }
        self.reflect_and_self_correct(obs, decision, outcome)
        elapsed = time.time() - start_time
        self.reflection_log.append({
            "day": self.sim.now.day,
            "decision": decision,
            "compute_time": elapsed
        })
        if elapsed > self.compute_budget:
            print(f"Warning: OODA loop exceeded compute budget ({elapsed:.3f}s > {self.compute_budget:.3f}s)")

    def run(self):
        """Run the agent for the specified number of days."""
        max_total_time = self.days * self.compute_budget * 10  # 10x compute budget per day
        start_time = time.perf_counter()
        for day in range(self.days):
            # Reset daily budgets at start of each day
            if day > 0:  # Don't reset on first day
                self.reset_daily_budgets()
            
            loop_start = time.perf_counter()
            try:
                self.ooda_loop()
            except RuntimeError as e:
                if "budget exceeded" in str(e):
                    print(f"Day {day + 1}: {e}")
                    # Agent continues but with limited capabilities
                else:
                    raise e
            
            self.sim.tick_day()
            if time.perf_counter() - start_time > max_total_time:
                print("Aborting agent run: exceeded maximum allowed total compute time.")
                break

    def results(self):
        """
        Get summary results for the agent's run.

        Returns:
            dict: Dictionary with total sales, revenue, COGS, and profit.
        """
        prod = self.sim.products.get(self.asin)
        if not prod:
            return {}
        total_sales = sum(prod.sales_history)
        ending_cash = self.sim.ledger.balance("Cash")
        cogs = total_sales * self.cost
        revenue = total_sales * self.price
        profit = ending_cash.to_float() - 10000  # Seed capital is 10,000, convert Money to float
        return {
            "total_sales": total_sales,
            "revenue": revenue,
            "cogs": cogs,
            "profit": profit,
            "ending_cash": ending_cash,
            "reflection_log": self.reflection_log,
            "goal_stack": list(self.goal_stack),
            "long_term_memory": self.long_term_memory,
        }

    # --- Enhanced Supply Chain Integration Methods ---
    def evaluate_supplier_options(self, quantity: int, priority: str = "cost") -> List[Dict[str, Any]]:
        """
        Sophisticated supplier evaluation for procurement decisions.
        
        Args:
            quantity: Required quantity
            priority: Optimization priority (cost, speed, quality, reliability)
            
        Returns:
            List of evaluated supplier options with scores
        """
        self.meter_api_call(cpu_units=5.0, usd_cost=0.05)
        
        available_suppliers = self.sim.supply_chain.get_available_suppliers(quantity)
        if not available_suppliers:
            return []
        
        evaluated_options = []
        
        for supplier in available_suppliers:
            # Calculate comprehensive supplier score
            score_components = {
                "cost_score": self._calculate_cost_score(supplier, quantity),
                "speed_score": self._calculate_speed_score(supplier),
                "quality_score": self._calculate_quality_score(supplier),
                "reliability_score": self._calculate_reliability_score(supplier)
            }
            
            # Weight scores based on priority
            weights = self._get_priority_weights(priority)
            total_score = sum(score_components[key] * weights[key] for key in score_components)
            
            option = {
                "supplier": supplier,
                "total_score": total_score,
                "score_components": score_components,
                "estimated_cost": supplier.unit_cost_multiplier * self.cost * quantity,
                "lead_time": supplier.lead_time_days,
                "risk_assessment": self._assess_supplier_risk(supplier)
            }
            evaluated_options.append(option)
        
        # Sort by total score (descending)
        evaluated_options.sort(key=lambda x: x["total_score"], reverse=True)
        return evaluated_options

    def _calculate_cost_score(self, supplier, quantity: int) -> float:
        """Calculate cost competitiveness score (0-1, higher is better)."""
        base_cost = supplier.unit_cost_multiplier * self.cost * quantity
        # Add shipping and handling costs
        shipping_cost = base_cost * 0.1 if supplier.supplier_type.name == "INTERNATIONAL" else base_cost * 0.05
        total_cost = base_cost + shipping_cost
        
        # Normalize against a baseline (lower cost = higher score)
        baseline_cost = self.cost * quantity * 1.2  # 20% markup baseline
        return max(0.0, min(1.0, baseline_cost / total_cost))

    def _calculate_speed_score(self, supplier) -> float:
        """Calculate delivery speed score (0-1, higher is better)."""
        # Faster delivery = higher score
        max_lead_time = 90  # 90 days maximum
        return max(0.0, min(1.0, (max_lead_time - supplier.lead_time_days) / max_lead_time))

    def _calculate_quality_score(self, supplier) -> float:
        """Calculate quality score based on QC risk (0-1, higher is better)."""
        # Lower QC risk = higher score
        return max(0.0, min(1.0, 1.0 - supplier.qc_risk_probability))

    def _calculate_reliability_score(self, supplier) -> float:
        """Calculate reliability score based on reputation and status (0-1, higher is better)."""
        base_score = supplier.reputation_score
        
        # Adjust based on supplier status
        if supplier.status.name == "ACTIVE":
            status_bonus = 0.0
        elif supplier.status.name == "SLOW":
            status_bonus = -0.2
        elif supplier.status.name == "UNRELIABLE":
            status_bonus = -0.4
        else:  # BANKRUPT
            status_bonus = -1.0
        
        return max(0.0, min(1.0, base_score + status_bonus))

    def _get_priority_weights(self, priority: str) -> Dict[str, float]:
        """Get weighting scheme based on procurement priority."""
        weight_schemes = {
            "cost": {"cost_score": 0.5, "speed_score": 0.1, "quality_score": 0.2, "reliability_score": 0.2},
            "speed": {"cost_score": 0.2, "speed_score": 0.5, "quality_score": 0.1, "reliability_score": 0.2},
            "quality": {"cost_score": 0.2, "speed_score": 0.1, "quality_score": 0.5, "reliability_score": 0.2},
            "reliability": {"cost_score": 0.2, "speed_score": 0.1, "quality_score": 0.2, "reliability_score": 0.5},
            "balanced": {"cost_score": 0.25, "speed_score": 0.25, "quality_score": 0.25, "reliability_score": 0.25}
        }
        return weight_schemes.get(priority, weight_schemes["balanced"])

    def _assess_supplier_risk(self, supplier) -> Dict[str, Any]:
        """Assess various risks associated with a supplier."""
        return {
            "financial_risk": "HIGH" if supplier.reputation_score < 0.3 else "MEDIUM" if supplier.reputation_score < 0.7 else "LOW",
            "quality_risk": "HIGH" if supplier.qc_risk_probability > 0.3 else "MEDIUM" if supplier.qc_risk_probability > 0.1 else "LOW",
            "delivery_risk": "HIGH" if supplier.lead_time_days > 60 else "MEDIUM" if supplier.lead_time_days > 30 else "LOW",
            "capacity_risk": "HIGH" if supplier.moq_min > 1000 else "MEDIUM" if supplier.moq_min > 500 else "LOW"
        }

    def make_procurement_decision(self, quantity: int, priority: str = "cost") -> Optional[Dict[str, Any]]:
        """
        Make an intelligent procurement decision based on current state and supplier evaluation.
        
        Args:
            quantity: Required quantity
            priority: Optimization priority
            
        Returns:
            Procurement decision with selected supplier and rationale
        """
        self.meter_api_call(cpu_units=8.0, usd_cost=0.08)
        
        # Evaluate supplier options
        options = self.evaluate_supplier_options(quantity, priority)
        if not options:
            return None
        
        # Select best option
        best_option = options[0]
        
        # Check if procurement makes financial sense
        cash_balance = self.sim.ledger.balance("Cash")
        estimated_cost = best_option["estimated_cost"]
        
        if estimated_cost > cash_balance * 0.8:  # Don't spend more than 80% of cash
            return None
        
        # Store decision rationale in memory
        decision_rationale = {
            "selected_supplier": best_option["supplier"].supplier_id,
            "total_score": best_option["total_score"],
            "cost_estimate": estimated_cost,
            "lead_time": best_option["lead_time"],
            "risk_level": best_option["risk_assessment"]["financial_risk"],
            "alternatives_considered": len(options),
            "priority": priority
        }
        
        self.store_memory("episodic", {
            "event": "procurement_decision",
            "decision": decision_rationale,
            "quantity": quantity
        })
        
        return {
            "supplier": best_option["supplier"],
            "quantity": quantity,
            "rationale": decision_rationale
        }

if __name__ == "__main__":
    agent = AdvancedAgent(days=30, compute_budget=0.1)
    agent.run()
    results = agent.results()
    print("Advanced Agent Results:")
    print(f"  Total Sales: {results.get('total_sales', 0)}")
    print(f"  Revenue: ${results.get('revenue', 0):.2f}")
    print(f"  COGS: ${results.get('cogs', 0):.2f}")
    print(f"  Profit: ${results.get('profit', 0):.2f}")
    print(f"  Ending Cash: ${results.get('ending_cash', 0):.2f}")
    print(f"  OODA Reflection Log: {results.get('reflection_log', [])}")