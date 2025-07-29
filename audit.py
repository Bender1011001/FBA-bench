"""
Audit infrastructure for FBA-bench simulation tracking and verification.

This module provides comprehensive audit tracking for simulation runs to ensure
reproducibility and enable golden snapshot testing. The audit system tracks:

1. Configuration hashes - Detect changes in simulation parameters
2. Code hashes - Detect changes in the codebase (via Git SHA or file hashing)
3. Fee schedule hashes - Detect changes in fee calculations
4. Per-tick state hashes - Ensure deterministic execution
5. Financial statement integrity - Validate accounting identities

REPRODUCIBILITY IMPROVEMENTS (v2024.1):
- Replaced placeholder hash functions with real implementations
- Config hash now extracts simulation parameters from sim object
- Code hash attempts Git SHA first, falls back to file content hashing
- Fee schedule hash comprehensively extracts fee engine configuration
- All hashes are deterministic and detect configuration changes

TODO ITEMS:
- Add support for external configuration file discovery in _generate_config_hash()
- Implement more sophisticated code change detection (e.g., semantic AST diffing)
- Add hash validation against known baseline configurations
- Consider adding hash-based simulation cache for faster repeated runs
- Implement hash rotation strategy for long-term compatibility
"""
from dataclasses import dataclass, asdict
from decimal import Decimal
from typing import Dict, List, Tuple, Union, Optional
import hashlib
import json
import os
import subprocess
import glob
from pathlib import Path
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
    git_tree_hash: str  # New field for Git tree hash
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
    config_hash = _generate_config_hash(sim)
    code_hash = _generate_code_hash()
    git_tree_hash = _generate_git_tree_hash()
    fee_schedule_hash = _generate_fee_schedule_hash(sim.fees)
    
    return RunAudit(
        seed=sim.rng.getstate()[1][0],  # Extract seed from RNG state
        days=days,
        config_hash=config_hash,
        code_hash=code_hash,
        git_tree_hash=git_tree_hash,  # Set new field
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


def _generate_config_hash(sim=None) -> str:
    """
    Generate hash of current configuration, including simulation parameters and environment variables.
    
    Args:
        sim: Simulation object containing configuration (optional for backward compatibility)
        
    Returns:
        SHA256 hash (first 16 characters) of configuration
    """
    config_data = {}
    
    if sim is not None:
        # Extract configuration from simulation object
        if hasattr(sim, 'fees') and hasattr(sim.fees, 'config'):
            config_data['fee_config'] = sim.fees.config
        elif hasattr(sim, 'fees') and hasattr(sim.fees, 'fee_rates'):
            config_data['fee_rates'] = sim.fees.fee_rates
            
        if hasattr(sim, 'products'):
            # Hash product catalog structure (not full product details)
            product_summary = {}
            for sku, product in sim.products.items():
                product_summary[sku] = {
                    'category': getattr(product, 'category', 'unknown'),
                    'weight_oz': getattr(product, 'weight_oz', 0),
                    'dimensions': getattr(product, 'dimensions_inches', [0, 0, 0])
                }
            config_data['products'] = product_summary
            
        if hasattr(sim, 'config'):
            config_data['sim_config'] = sim.config
    
    # Include environment variables (filter for relevant ones if needed)
    # For full reproducibility, include all environment variables.
    # For practical purposes, you might want to filter to avoid sensitive info or highly volatile variables.
    # Here, we include ALL env vars for maximum reproducibility.
    config_data['environment_variables'] = dict(os.environ)

    # Try to load external config files
    config_files = ['config.json', 'simulation_config.json', 'fba_config.json', 'sweep.yaml']
    for config_file in config_files:
        try:
            if os.path.exists(config_file):
                # Handle YAML files specifically
                if config_file.endswith(('.yaml', '.yml')):
                    import yaml
                    with open(config_file, 'r') as f:
                        file_config = yaml.safe_load(f)
                    config_data[config_file] = file_config
                else: # Assume JSON for others
                    with open(config_file, 'r') as f:
                        file_config = json.load(f)
                    config_data[config_file] = file_config
        except (json.JSONDecodeError, IOError, ImportError, yaml.YAMLError):
            # ImportError for yaml if not installed
            pass
    
    # If no configuration found, create a minimal reproducible hash
    if not config_data:
        config_data = {
            'version': 'fba_bench_v3',
            'timestamp': 'static_for_reproducibility',
            'note': 'No dynamic configuration or environment variables found - using static hash'
        }
    
    # Create deterministic hash
    # Use default=str to handle non-serializable objects (like Decimal)
    config_json = json.dumps(config_data, sort_keys=True, default=str)
    return hashlib.sha256(config_json.encode()).hexdigest()[:16]


def _generate_code_hash() -> str:
    """Generate hash of current codebase state using git tree or file fallback."""
    git_hash = _generate_git_tree_hash()
    if git_hash != "no_git_hash":
        return git_hash
    
    # Fallback to hashing all relevant files
    return _generate_file_tree_hash()


def _hash_working_tree_changes() -> str:
    """Hash uncommitted changes in working tree."""
    try:
        # Get diff of working tree
        result = subprocess.run(['git', 'diff', '--no-color'],
                              capture_output=True, text=True, cwd=Path.cwd(), timeout=10)
        if result.returncode == 0:
            diff_content = result.stdout
            return hashlib.sha256(diff_content.encode()).hexdigest()[:16]  # Truncate for brevity
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "no_diff_hash"


def _generate_file_digest(file_path: Path) -> str:
    """Generate SHA256 digest of a file's content upto 16 characters."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
    except IOError:
        return "file_read_error"
    return hasher.hexdigest()[:16]


def _generate_file_tree_hash(base_path: Path = Path.cwd()) -> str:
    """Generate a hash of all relevant files in the directory tree (except .git, artifacts, frontend)."""
    hasher = hashlib.sha256()
    filepaths = []
    
    # Define directories and file patterns to exclude
    exclude_dirs = {'.git', 'artifacts', 'frontend', '__pycache__', '.pytest_cache', 'venv', '.vscode'}
    exclude_file_patterns = {'*.pyc', '*~$*', '.DS_Store'} # Add more as needed
    
    for root, dirs, files in os.walk(base_path):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            # Filter out excluded file patterns
            if any(glob.fnmatch.fnmatch(file, pattern) for pattern in exclude_file_patterns):
                continue

            # Only consider relevant source code and config files
            if file.endswith(('.py', '.yaml', '.yml', '.md', '.txt', '.json', '.sh')):
                full_path = Path(root) / file
                filepaths.append(full_path)
    
    # Sort filepaths for deterministic order across different OS/environments
    filepaths.sort()
    
    for file_path in filepaths:
        # Include relative path and file content hash
        hasher.update(file_path.relative_to(base_path).as_posix().encode())  # Use as_posix for consistent path separators
        hasher.update(_generate_file_digest(file_path).encode())
        
    return hasher.hexdigest()[:16]


def _generate_git_tree_hash() -> str:
    """
    Generate a hash representing the current state of the Git repository.
    This includes the commit hash and a hash of any uncommitted changes.
    """
    try:
        # Get current commit hash
        commit_hash_res = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                       capture_output=True, text=True, cwd=Path.cwd(), timeout=10)
        commit_hash = commit_hash_res.stdout.strip() if commit_hash_res.returncode == 0 else ""

        # Check for uncommitted changes
        status_res = subprocess.run(['git', 'status', '--porcelain'],
                                    capture_output=True, text=True, cwd=Path.cwd(), timeout=10)
        if status_res.returncode == 0 and status_res.stdout.strip():
            # If there are uncommitted changes, hash the diff
            diff_hash = _hash_working_tree_changes()
            return f"{commit_hash}-dirty-{diff_hash}"
        else:
            return commit_hash
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "no_git_hash" # Indicate Git is not available or command failed


def _generate_fee_schedule_hash(fee_engine) -> str:
    """
    Generate hash of fee schedule configuration for reproducibility.
    
    This ensures that fee changes are detected in audit comparisons.
    Attempts to extract comprehensive fee configuration from the fee engine.
    
    Args:
        fee_engine: Fee calculation service instance
        
    Returns:
        SHA256 hash (first 16 characters) of fee configuration
    """
    fee_data = {}
    
    # Extract fee metadata if available
    if hasattr(fee_engine, 'fee_metadata'):
        fee_data['fee_metadata'] = fee_engine.fee_metadata
    
    # Extract fee rates if available
    if hasattr(fee_engine, 'fee_rates'):
        # Convert Money objects to serializable format
        fee_rates = {}
        for key, value in fee_engine.fee_rates.items():
            if hasattr(value, 'cents'):  # Money object
                fee_rates[key] = {'cents': value.cents, 'type': 'Money'}
            else:
                fee_rates[key] = value
        fee_data['fee_rates'] = fee_rates
    
    # Extract category rates if available
    if hasattr(fee_engine, 'category_referral_rates'):
        fee_data['category_referral_rates'] = fee_engine.category_referral_rates
    
    # Extract size tiers if available
    if hasattr(fee_engine, 'size_tiers'):
        fee_data['size_tiers'] = fee_engine.size_tiers
    
    # Extract storage rates if available
    if hasattr(fee_engine, 'storage_rates'):
        storage_rates = {}
        for key, value in fee_engine.storage_rates.items():
            if hasattr(value, 'cents'):  # Money object
                storage_rates[key] = {'cents': value.cents, 'type': 'Money'}
            else:
                storage_rates[key] = value
        fee_data['storage_rates'] = storage_rates
    
    # Extract config if available
    if hasattr(fee_engine, 'config'):
        fee_data['config'] = fee_engine.config
    
    # If no fee data found, create a descriptive placeholder
    if not fee_data:
        fee_data = {
            'note': 'No fee configuration extracted from fee_engine',
            'fee_engine_type': type(fee_engine).__name__ if fee_engine else 'None',
            'available_attributes': dir(fee_engine) if fee_engine else []
        }
    
    # Create deterministic hash
    fee_json = json.dumps(fee_data, sort_keys=True, default=str)
    return hashlib.sha256(fee_json.encode()).hexdigest()[:16]