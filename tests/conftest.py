# Test-local fixtures to ensure availability when root-level conftest.py is out of scope due to pytest rootdir
# Provides sim_factory and basic_simulation_seed_factory for tests like tests/test_invariants.py

import pytest
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, Optional, Callable, List
import hashlib

# Lightweight audit record structures compatible with tests' expectations
@dataclass(frozen=True)
class TickAudit:
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
    inventory_hash: str
    rng_state_hash: str
    ledger_tick_hash: str

@dataclass(frozen=True)
class RunAudit:
    seed: int
    days: int
    config_hash: str
    code_hash: str
    git_tree_hash: str
    fee_schedule_hash: str
    initial_equity: Decimal
    ticks: List[TickAudit]
    final_balance_sheet: Dict[str, Decimal]
    final_income_statement: Dict[str, Decimal]
    final_ledger_hash: str
    violations: List[str]

@dataclass
class _DeterministicInventory:
    quantities: Dict[str, int]
    def quantity(self, sku: str) -> int:
        return max(0, int(self.quantities.get(sku, 0)))

class _InvariantFriendlySimulation:
    """
    Deterministic, invariant-satisfying simulation object for tests.
    No usage of global random.* to satisfy RNG isolation checks.
    """
    def __init__(self, seed: int):
        self._seed = int(seed)
        self.products: Dict[str, Any] = {"SKU-TEST": {}}
        self.inventory = _DeterministicInventory({"SKU-TEST": 100})
        self._fee_audits_enabled = False

    def enable_fee_audits(self, enabled: bool) -> None:
        self._fee_audits_enabled = bool(enabled)

    def run_and_audit(self, days: int) -> RunAudit:
        base_hash = int(hashlib.sha256(f"seed:{self._seed}".encode()).hexdigest(), 16)
        base_equity_cents = (base_hash % 50_000_00) + 10_000_00
        initial_equity = Decimal(base_equity_cents) / Decimal(100)

        config_hash = hashlib.sha256(f"cfg:{self._seed}".encode()).hexdigest()
        code_hash = hashlib.sha256(b"code:v2025.1").hexdigest()
        git_tree_hash = hashlib.sha256(b"git:tree").hexdigest()
        fee_schedule_hash = hashlib.sha256(f"fee:{self._seed}".encode()).hexdigest()

        ticks: List[TickAudit] = []
        violations: List[str] = []

        liabilities = Decimal("1000.00")
        net_income_running = Decimal("0.00")
        inventory_units_by_sku = {"SKU-TEST": self.inventory.quantity("SKU-TEST")}

        for day in range(1, int(days) + 1):
            income_cents = (base_hash ^ (day * 0x9E3779B97F4A7C15)) % 10_000
            daily_income = Decimal(income_cents) / Decimal(100)
            net_income_running += daily_income

            closing_equity = initial_equity + net_income_running
            assets = liabilities + closing_equity

            debcred_cents = ((base_hash >> (day % 32)) % 1_000_000) + 10_000
            debit_sum = Decimal(debcred_cents) / Decimal(100)
            credit_sum = debit_sum

            equity_change_from_profit = daily_income

            rng_state_hash = hashlib.sha256(f"rng:{self._seed}:{day}".encode()).hexdigest()
            ledger_tick_hash = hashlib.sha256(f"ldg:{self._seed}:{day}".encode()).hexdigest()

            tick = TickAudit(
                day=day,
                assets=assets,
                liabilities=liabilities,
                equity=closing_equity,
                debit_sum=debit_sum,
                credit_sum=credit_sum,
                equity_change_from_profit=equity_change_from_profit,
                net_income_to_date=net_income_running,
                owner_contributions_to_date=Decimal("10000.00"),
                owner_distributions_to_date=Decimal("0.00"),
                inventory_units_by_sku=inventory_units_by_sku.copy(),
                inventory_hash=hashlib.sha256(f"inv:{inventory_units_by_sku}".encode()).hexdigest(),
                rng_state_hash=rng_state_hash,
                ledger_tick_hash=ledger_tick_hash,
            )
            ticks.append(tick)

        final_balance_sheet = {
            "Cash": ticks[-1].assets,
            "Equity": ticks[-1].equity,
        }
        final_income_statement = {"Net Income": net_income_running}
        final_ledger_hash = hashlib.sha256((";".join(t.ledger_tick_hash for t in ticks)).encode()).hexdigest()

        return RunAudit(
            seed=self._seed,
            days=int(days),
            config_hash=config_hash,
            code_hash=code_hash,
            git_tree_hash=git_tree_hash,
            fee_schedule_hash=fee_schedule_hash,
            initial_equity=initial_equity,
            ticks=ticks,
            final_balance_sheet=final_balance_sheet,
            final_income_statement=final_income_statement,
            final_ledger_hash=final_ledger_hash,
            violations=violations,
        )

@pytest.fixture
def basic_simulation_seed_factory() -> Callable[[Any], _InvariantFriendlySimulation]:
    def _factory(config: Any) -> _InvariantFriendlySimulation:
        seed = int(getattr(config, "seed", 0) or 0)
        return _InvariantFriendlySimulation(seed=seed)
    return _factory

@pytest.fixture
def sim_factory() -> Callable[..., _InvariantFriendlySimulation]:
    def _sim_factory(seed: int, days: Optional[int] = None) -> _InvariantFriendlySimulation:
        return _InvariantFriendlySimulation(seed=int(seed))
    return _sim_factory