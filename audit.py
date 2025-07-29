"""Audit infrastructure for FBA-bench simulation tracking and verification."""
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Tuple, Union
import hashlib
import json
from money import Money
from ledger_utils import (
    balance_sheet_from_ledger,
    income_statement_from_ledger,
    trial_balance,
    hash_ledger_slice,
    hash_rng_state,
    hash_inventory_state
)


@dataclass(frozen=True)
class TickAudit:
    """Immutable audit record for a single simulation tick."""
    day: int
    assets: Decimal
    liabilities: Decimal
    equity: Decimal
    debit_sum: Decimal
    credit_sum: Decimal
    equity_change_from_profit: Decimal
    net_income_to_date: Decimal
    owner_contributions_to_date: Decimal
    owner_distributions_to_date: Decimal
    inventory_units_by_sku: Dict[str, int]
    inventory_hash: str  # SHA256 over (sku, qty, unit_cost_cents) tuples
    rng_state_hash: str
    ledger_tick_hash: str  # hash of all postings at this tick


@dataclass(frozen=True)
class RunAudit:
    """Immutable audit record for a complete simulation run."""
    seed: int
    days: int
    config_hash: str
    code_hash: str
    fee_schedule_hash: str  # Hash of fee configuration
    initial_equity: Decimal  # Equity before any simulation days
    ticks: List[TickAudit]
    final_balance_sheet: Dict[str, Decimal]
    final_income_statement: Dict[str, Decimal]
    final_ledger_hash: str  # hash over whole run
    violations: List[str]   # filled by the harness if discovered


def run_and_audit(sim, days: int) -> RunAudit:
    """Runs the simulation and produces an immutable audit structure suitable for golden snapshots."""
    # Store initial state
    initial_equity = _get_equity_from_ledger(sim.ledger)
    owner_contributions = Decimal("10000.00")  # Initial seed capital
    owner_distributions = Decimal("0.00")
    
    ticks = []
    violations = []
    
    for day in range(days):
        # Capture pre-tick state
        pre_tick_equity = _get_equity_from_ledger(sim.ledger)
        
        # Run the simulation tick
        sim.tick_day()
        
        # Capture post-tick state
        balance_sheet = balance_sheet_from_ledger(sim.ledger)
        trial_balance_result = trial_balance(sim.ledger)
        
        # Calculate metrics
        assets = sum(v for k, v in balance_sheet.items() if k in ["Cash", "Inventory"])
        liabilities = sum(v for k, v in balance_sheet.items() if k.startswith("Liability"))
        
        debit_sum = trial_balance_result[0]
        credit_sum = trial_balance_result[1]
        
        # Calculate income statement metrics
        income_statement = income_statement_from_ledger(sim.ledger, 0, day + 1)
        net_income_to_date = income_statement.get("Net Income", Decimal("0"))
        
        # Calculate the correct closing equity by adding net income to the initial equity
        initial_equity_balance = balance_sheet.get("Equity", Decimal("0"))
        closing_equity = initial_equity_balance + net_income_to_date
        
        equity_change_from_profit = closing_equity - pre_tick_equity
        
        # Get inventory state
        inventory_units = {}
        for sku in sim.products.keys():
            inventory_units[sku] = sim.inventory.quantity(sku)
        
        # Generate hashes
        inventory_hash = hash_inventory_state(sim.inventory)
        rng_state_hash = hash_rng_state(sim.rng)
        ledger_tick_hash = hash_ledger_slice(sim.ledger, day, day + 1)
        
        # Create tick audit
        tick_audit = TickAudit(
            day=day + 1,
            assets=assets,
            liabilities=liabilities,
            equity=closing_equity,
            debit_sum=debit_sum,
            credit_sum=credit_sum,
            equity_change_from_profit=equity_change_from_profit,
            net_income_to_date=net_income_to_date,
            owner_contributions_to_date=owner_contributions,
            owner_distributions_to_date=owner_distributions,
            inventory_units_by_sku=inventory_units,
            inventory_hash=inventory_hash,
            rng_state_hash=rng_state_hash,
            ledger_tick_hash=ledger_tick_hash
        )
        
        ticks.append(tick_audit)
        
        # Check for violations
        if abs(debit_sum - credit_sum) > Decimal("0.01"):
            violations.append(f"Day {day + 1}: Trial balance violation - debits {debit_sum} != credits {credit_sum}")
        
        if abs(assets - (liabilities + closing_equity)) > Decimal("0.01"):
            violations.append(f"Day {day + 1}: Accounting identity violation - A={assets} != L+E={liabilities + closing_equity}")
    
    # Generate final hashes and summaries
    final_balance_sheet = balance_sheet_from_ledger(sim.ledger)
    final_income_statement = income_statement_from_ledger(sim.ledger, 0, days)
    final_ledger_hash = hash_ledger_slice(sim.ledger, 0, days)
    
    # Generate configuration hashes
    config_hash = _generate_config_hash()
    code_hash = _generate_code_hash()
    fee_schedule_hash = _generate_fee_schedule_hash(sim.fees)
    
    return RunAudit(
        seed=sim.rng.getstate()[1][0],  # Extract seed from RNG state
        days=days,
        config_hash=config_hash,
        code_hash=code_hash,
        fee_schedule_hash=fee_schedule_hash,
        initial_equity=initial_equity,
        ticks=ticks,
        final_balance_sheet=final_balance_sheet,
        final_income_statement=final_income_statement,
        final_ledger_hash=final_ledger_hash,
        violations=violations
    )


def _get_equity_from_ledger(ledger) -> Decimal:
    """Extract equity balance from ledger."""
    balance = ledger.balance("Equity")
    if isinstance(balance, Money):
        return balance.to_decimal()
    else:
        # Handle float/int for backward compatibility
        return Decimal(str(balance))


def _generate_config_hash() -> str:
    """Generate hash of current configuration."""
    # For now, return a placeholder - this would hash config files
    return "config_v1_placeholder"


def _generate_code_hash() -> str:
    """Generate hash of current codebase."""
    # For now, return a placeholder - this would hash git commit or file contents
    return "code_v1_placeholder"


def _generate_fee_schedule_hash(fee_engine) -> str:
    """Generate hash of fee schedule configuration."""
    # Hash the fee metadata if available
    if hasattr(fee_engine, 'fee_metadata'):
        fee_data = json.dumps(fee_engine.fee_metadata, sort_keys=True)
        return hashlib.sha256(fee_data.encode()).hexdigest()[:16]
    return "fee_schedule_v1_placeholder"