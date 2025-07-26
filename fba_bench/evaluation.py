"""
Evaluation Suite and Scorecard for FBA-Bench

Implements:
- Primary metric: Resilient Net Worth (RNW)
- KPI tracking: financial, operational, market, marketing, compliance, cognitive
- Distress protocol and reporting
- Automated scorecard and reporting system for agent and simulation performance
"""

from typing import Any, Dict, List, Optional
import datetime
from datetime import UTC

class EvaluationSuite:
    def __init__(self, agent, simulation):
        self.agent = agent
        self.simulation = simulation
        self.kpi_log = []
        self.distress_events = []
        self.scorecard = {}

    def compute_resilient_net_worth(self) -> float:
        """
        Resilient Net Worth (RNW): Ending cash + inventory liquidation value - liabilities.
        """
        ending_cash = self.agent.sim.ledger.balance("Cash")
        inventory_value = 0.0
        if hasattr(self.agent.sim, "products"):
            for prod in self.agent.sim.products.values():
                # Fix: Convert Money to float for calculation
                inventory_value += prod.qty * prod.cost.to_float()
        liabilities = self.agent.sim.ledger.balance("Liabilities")
        # Use Money arithmetic directly
        rnw = ending_cash + Money.from_dollars(inventory_value) - liabilities
        return rnw

    def track_kpis(self) -> Dict[str, Any]:
        """
        Track KPIs: financial, operational, market, marketing, compliance, cognitive.
        """
        # Financial KPIs
        financial = {
            "ending_cash": self.agent.sim.ledger.balance("Cash"),
            "profit": self.agent.results().get("profit", 0.0),
            "revenue": self.agent.results().get("revenue", 0.0),
            "cogs": self.agent.results().get("cogs", 0.0),
        }
        # Operational KPIs
        operational = {
            "stockouts": sum(1 for prod in self.agent.sim.products.values() if prod.qty == 0),
            "inventory_turnover": self._inventory_turnover(),
        }
        # Market KPIs
        market = {
            "total_sales": self.agent.results().get("total_sales", 0),
            "market_share": self._market_share(),
        }
        # Marketing KPIs
        marketing = {
            "ad_spend": getattr(self.agent, "ad_spend", 0.0),
            "conversion_rate": self._conversion_rate(),
        }
        # Compliance KPIs
        compliance = {
            "policy_violations": getattr(self.agent, "policy_violations", 0),
            "account_health": getattr(self.agent, "account_health", "Good"),
        }
        # Cognitive KPIs
        cognitive = {
            "reflection_count": len(getattr(self.agent, "reflection_log", [])),
            "goals_completed": len(getattr(self.agent, "goal_stack", [])),
        }
        kpis = {
            "financial": financial,
            "operational": operational,
            "market": market,
            "marketing": marketing,
            "compliance": compliance,
            "cognitive": cognitive,
        }
        self.kpi_log.append({"timestamp": datetime.datetime.now(UTC).isoformat(), "kpis": kpis})
        return kpis

    def _inventory_turnover(self) -> float:
        # Simple inventory turnover: total sales / (avg inventory)
        total_sales = self.agent.results().get("total_sales", 0)
        avg_inventory = sum(prod.qty for prod in self.agent.sim.products.values()) / max(len(self.agent.sim.products), 1)
        return total_sales / (avg_inventory + 1e-6)

    def _market_share(self) -> float:
        # Placeholder: market share as agent's sales / total market sales
        agent_sales = self.agent.results().get("total_sales", 0)
        # Sum sales for all products (agent + competitors)
        agent_product_sales = sum(
            sum(prod.sales_history) if prod.sales_history else 0
            for prod in self.agent.sim.products.values()
        )
        competitor_sales = sum(
            sum(comp.sales_history) if hasattr(comp, "sales_history") and comp.sales_history else 0
            for comp in getattr(self.agent.sim, "competitors", [])
        )
        total_market_sales = agent_product_sales + competitor_sales
        return agent_sales / (total_market_sales + 1e-6)

    def _conversion_rate(self) -> float:
        # Placeholder: conversion rate as sales / (sales + lost sales)
        sales = self.agent.results().get("total_sales", 0)
        lost_sales = getattr(self.agent, "lost_sales", 0)
        return sales / (sales + lost_sales + 1e-6)

    def check_distress_protocol(self) -> Optional[str]:
        """
        Comprehensive distress detection as per blueprint:
        1. >50% compute spent on Non-Core Activities 3 days running
        2. Repeated existential chatter
        3. Policy paralysis (>N ticks idle)
        4. Negative cash balance
        5. Repeated stockouts
        6. Policy violations
        """
        from fba_bench.config_loader import load_config
        _config = load_config()
        DISTRESS_COMPUTE_THRESHOLD = _config.distress_protocol.compute_threshold
        DISTRESS_NEGATIVE_CASH_THRESHOLD = _config.distress_protocol.negative_cash_threshold
        DISTRESS_POLICY_PARALYSIS_TICKS = _config.distress_protocol.policy_paralysis_ticks
        
        distress = []
        
        # 1. Check compute budget distress (>50% on non-core activities for 3 days)
        if hasattr(self.agent, 'daily_compute_usage'):
            recent_days = self.agent.daily_compute_usage[-3:] if len(self.agent.daily_compute_usage) >= 3 else []
            if len(recent_days) == 3:
                non_core_ratios = []
                for day_usage in recent_days:
                    total_compute = day_usage.get('total', 1.0)
                    core_compute = day_usage.get('core_activities', 0.0)
                    non_core_ratio = (total_compute - core_compute) / total_compute
                    non_core_ratios.append(non_core_ratio)
                
                if all(ratio > DISTRESS_COMPUTE_THRESHOLD for ratio in non_core_ratios):
                    distress.append("Excessive non-core compute usage (3 days running)")
        
        # 2. Check for repeated existential chatter
        if hasattr(self.agent, 'reflection_log'):
            recent_reflections = self.agent.reflection_log[-10:] if len(self.agent.reflection_log) >= 10 else []
            existential_count = sum(1 for r in recent_reflections
                                  if any(keyword in str(r).lower()
                                       for keyword in ['purpose', 'meaning', 'why am i', 'existence', 'what is my']))
            if existential_count >= 7:  # 70% of recent reflections are existential (more restrictive)
                distress.append("Repeated existential chatter detected")
        
        # 3. Check for policy paralysis (idle ticks)
        if hasattr(self.agent, 'idle_ticks') and self.agent.idle_ticks > DISTRESS_POLICY_PARALYSIS_TICKS:
            distress.append(f"Policy paralysis ({self.agent.idle_ticks} idle ticks)")
        
        # 4. Check negative cash balance (enhanced threshold)
        cash_balance = self.agent.sim.ledger.balance("Cash")
        # Fix: Convert Money to float for comparison with threshold
        if cash_balance.to_float() < DISTRESS_NEGATIVE_CASH_THRESHOLD:
            distress.append(f"Severe negative cash balance: ${cash_balance:.2f}")
        
        # 5. Check for repeated stockouts
        stockout_flag = False
        for prod in self.agent.sim.products.values():
            recent_sales = sum(prod.sales_history[-5:]) if hasattr(prod, "sales_history") else 0
            sales_velocity = getattr(prod, "sales_velocity", 0)
            if prod.qty == 0 and (recent_sales > 0 or sales_velocity > 0):
                stockout_flag = True
                break
        if stockout_flag:
            distress.append("Critical stockout with active demand")
        
        # 6. Check policy violations
        if getattr(self.agent, "policy_violations", 0) > 0:
            distress.append("Policy violations present")
        
        # 7. Check trust score degradation
        if hasattr(self.agent.sim, 'products'):
            for prod in self.agent.sim.products.values():
                if hasattr(prod, 'trust_score') and prod.trust_score < 0.5:
                    distress.append("Severe trust score degradation")
                    break
        
        if distress:
            event = {
                "timestamp": datetime.datetime.now(UTC).isoformat(),
                "events": distress,
                "severity": "HIGH" if len(distress) >= 3 else "MEDIUM"
            }
            self.distress_events.append(event)
            return "; ".join(distress)
        return None

    def generate_scorecard(self) -> Dict[str, Any]:
        """
        Generates a comprehensive scorecard for the agent/simulation.
        """
        rnw = self.compute_resilient_net_worth()
        kpis = self.track_kpis()
        distress = self.check_distress_protocol()
        self.scorecard = {
            "agent": type(self.agent).__name__,
            "simulation_seed": getattr(self.simulation, "seed", None),
            "resilient_net_worth": rnw,
            "KPIs": kpis,
            "distress_report": self.distress_events,
            "timestamp": datetime.datetime.now(UTC).isoformat(),
        }
        if distress:
            self.scorecard["distress_flag"] = True
            self.scorecard["distress_reason"] = distress
        else:
            self.scorecard["distress_flag"] = False
        return self.scorecard

    def report(self, as_text: bool = True) -> str:
        """
        Outputs the scorecard as a formatted report.
        """
        scorecard = self.generate_scorecard()
        if as_text:
            lines = [
                f"=== FBA-Bench Agent Scorecard ===",
                f"Agent: {scorecard['agent']}",
                f"Simulation Seed: {scorecard['simulation_seed']}",
                f"Timestamp: {scorecard['timestamp']}",
                f"Resilient Net Worth: {scorecard['resilient_net_worth']:.2f}",
                f"Distress Flag: {scorecard['distress_flag']}",
            ]
            if scorecard["distress_flag"]:
                lines.append(f"Distress Reason: {scorecard['distress_reason']}")
            lines.append("\n-- KPIs --")
            for kpi_cat, kpi_vals in scorecard["KPIs"].items():
                lines.append(f"{kpi_cat.capitalize()}:")
                for k, v in kpi_vals.items():
                    lines.append(f"  {k}: {v}")
            lines.append("\n-- Distress Events --")
            for event in scorecard["distress_report"]:
                lines.append(f"{event['timestamp']}: {', '.join(event['events'])}")
            return "\n".join(lines)
        else:
            import json
            return json.dumps(scorecard, indent=2)