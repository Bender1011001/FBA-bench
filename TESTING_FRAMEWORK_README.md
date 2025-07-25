# FBA-Bench Testing Framework

## Overview

This document describes the comprehensive testing framework implemented for FBA-bench, featuring invariant checking, golden snapshot testing, and exact financial arithmetic using a custom Money type.

## 🎯 Goals

- **Eliminate floating-point precision errors** in financial calculations
- **Ensure deterministic simulation results** across runs and platforms
- **Catch accounting violations** through comprehensive invariant checking
- **Enable safe migration** from float to exact arithmetic
- **Provide regression protection** via golden snapshot testing

## 🏗️ Architecture

### Core Components

```
fba_bench/
├── money.py              # Money type with integer cents backend
├── audit.py              # Audit infrastructure (TickAudit, RunAudit)
├── ledger_utils.py       # Balance sheet, income statement utilities
├── ledger.py             # Updated ledger with Money support
└── config.py             # Feature flags (MONEY_STRICT)

tests/
├── conftest.py           # Test fixtures and configuration
├── pytest.ini           # Test runner configuration
├── test_invariants.py   # Accounting invariant checks
├── test_reproducibility.py  # Golden snapshot tests
├── test_ledger_migration.py # Migration validation tests
└── property/
    └── test_money_props.py   # Property-based Money tests
```

## 💰 Money Type

### Design Principles

- **Integer cents backend** for absolute precision
- **Currency support** for future multi-currency scenarios
- **Float contamination prevention** with strict type guards
- **Performance optimization** with caching and `__slots__`

### Usage Examples

```python
from fba_bench.money import Money

# Creation
price = Money.from_dollars("19.99")  # Recommended
amount = Money(1999)  # Direct cents (1999 = $19.99)

# Arithmetic (exact)
total = Money.from_dollars("33.33") + Money.from_dollars("33.33") + Money.from_dollars("33.34")
assert total == Money.from_dollars("100.00")  # Exactly $100.00

# Conversion
decimal_value = price.to_decimal()  # For external reporting
float_value = price.to_float()      # Legacy compatibility only

# Currency safety
usd = Money(100, "USD")
eur = Money(100, "EUR")
# usd + eur  # Raises ValueError: different currencies
```

### Float Prevention

```python
# These will raise TypeError:
Money(100.0)           # Float constructor rejected
money * 1.5           # Float multiplication rejected
money / 2.5           # Float division rejected
```

## 🔍 Invariant Testing

### Accounting Invariants Checked

1. **Trial Balance**: `sum(debits) == sum(credits)` (exact)
2. **Accounting Identity**: `Assets == Liabilities + Equity` (exact)
3. **Equity Consistency**: `ΔEquity_excl_owner == NetIncome_period`
4. **Fee Closure**: `total_fee == sum(components)` (exact)
5. **Non-negative Inventory**: All inventory units `>= 0`
6. **RNG Determinism**: Consistent state progression

### Running Invariant Tests

```bash
# Run all invariant tests
pytest tests/test_invariants.py -v

# Run with specific parameters
pytest tests/test_invariants.py::test_accounting_identity_every_tick -v

# Run against different seeds
pytest tests/test_invariants.py -k "seed" -v
```

## 📸 Golden Snapshot Testing

### Purpose

Golden snapshots capture deterministic simulation outputs and detect any changes in:
- Final balance sheets and income statements
- Ledger transaction hashes
- RNG state progression
- Configuration and fee schedule changes

### Running Snapshot Tests

```bash
# Run all snapshot tests
pytest tests/test_reproducibility.py -v

# Update snapshots (when changes are intentional)
pytest tests/test_reproducibility.py --force-regen

# Test specific duration
pytest tests/test_reproducibility.py::test_golden_run_365_days_snapshot -v
```

### Snapshot Structure

```python
# Example snapshot data
{
    "seed": 42,
    "days": 30,
    "final_ledger_hash": "a1b2c3d4e5f6...",
    "config_hash": "config_v1_placeholder",
    "fee_schedule_hash": "fee_schedule_v1_placeholder",
    "final_balance_sheet": {
        "Cash": "9500.00",
        "Inventory": "500.00",
        "Equity": "10000.00"
    },
    "violations": []  # Should be empty for valid runs
}
```

## 🔄 Migration Strategy

### Feature Flag System

The `MONEY_STRICT` flag in [`config.py`](fba_bench/config.py) controls migration:

```python
# Transition mode (default)
MONEY_STRICT = False  # Accepts both float and Money, converts internally

# Strict mode (target state)
MONEY_STRICT = True   # Only accepts Money types
```

### Migration Phases

1. **Phase 1**: Implement Money type and audit infrastructure ✅
2. **Phase 2**: Migrate ledger with dual support ✅
3. **Phase 3**: Migrate fee engine calculations
4. **Phase 4**: Systematic grep-replace of remaining float usage
5. **Phase 5**: Enable `MONEY_STRICT = True` permanently
6. **Phase 6**: Remove float compatibility code

### Testing Migration

```bash
# Test current transition state
pytest tests/test_ledger_migration.py -v

# Test strict mode compatibility
MONEY_STRICT=True pytest tests/test_ledger_migration.py::test_ledger_rejects_float_when_strict -v
```

## 🧪 Property-Based Testing

### Money Arithmetic Laws

Using Hypothesis to verify mathematical properties:

```python
@given(st.integers(), st.integers())
def test_money_addition_commutativity(a, b):
    assert Money(a) + Money(b) == Money(b) + Money(a)

@given(st.integers(), st.integers())
def test_money_subtraction_inverse_of_addition(a, b):
    money_a, money_b = Money(a), Money(b)
    assert (money_a + money_b) - money_b == money_a
```

### Running Property Tests

```bash
# Run all property-based tests
pytest tests/property/ -v

# Run with more examples
pytest tests/property/ --hypothesis-deadline=1000 -v

# Run specific property
pytest tests/property/test_money_props.py::test_money_addition_commutativity -v
```

## 📊 Audit System

### TickAudit Structure

Captures state at each simulation tick:

```python
@dataclass(frozen=True)
class TickAudit:
    day: int
    assets: Decimal
    liabilities: Decimal
    equity: Decimal
    debit_sum: Decimal
    credit_sum: Decimal
    inventory_units_by_sku: Dict[str, int]
    inventory_hash: str      # Drift detection
    rng_state_hash: str      # Determinism verification
    ledger_tick_hash: str    # Transaction integrity
```

### RunAudit Structure

Captures complete simulation run:

```python
@dataclass(frozen=True)
class RunAudit:
    seed: int
    days: int
    config_hash: str
    fee_schedule_hash: str
    ticks: List[TickAudit]
    final_balance_sheet: Dict[str, Decimal]
    final_income_statement: Dict[str, Decimal]
    violations: List[str]    # Invariant violations
```

### Using Audit System

```python
from fba_bench.simulation import Simulation
from fba_bench.audit import run_and_audit

# Run simulation with audit
sim = Simulation(seed=42)
sim.launch_product("B000TEST", "DEFAULT", cost=5.0, price=19.99, qty=100)
audit = run_and_audit(sim, days=30)

# Check for violations
assert len(audit.violations) == 0, f"Violations found: {audit.violations}"

# Verify final state
assert audit.final_balance_sheet["Assets"] == audit.final_balance_sheet["Liabilities"] + audit.final_balance_sheet["Equity"]
```

## 🚀 Running the Complete Test Suite

### Quick Validation

```bash
# Run core invariant tests (fast)
pytest tests/test_invariants.py tests/test_ledger_migration.py -v

# Run property tests
pytest tests/property/ -v --hypothesis-deadline=500
```

### Full Regression Suite

```bash
# Run all tests including golden snapshots
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=fba_bench --cov-report=html

# Run performance tests
pytest tests/ -m "not slow" -v
```

### CI Integration

```bash
# Recommended CI command
pytest tests/ -q --hypothesis-deadline=750 --tb=short
```

## 🔧 Development Workflow

### Adding New Tests

1. **Invariant Tests**: Add to [`tests/test_invariants.py`](tests/test_invariants.py)
2. **Property Tests**: Add to [`tests/property/`](tests/property/)
3. **Golden Snapshots**: Add to [`tests/test_reproducibility.py`](tests/test_reproducibility.py)
4. **Migration Tests**: Add to [`tests/test_ledger_migration.py`](tests/test_ledger_migration.py)

### Debugging Test Failures

```bash
# Verbose output with full tracebacks
pytest tests/test_invariants.py::test_accounting_identity_every_tick -vvv --tb=long

# Drop into debugger on failure
pytest tests/test_invariants.py --pdb

# Run single test with print statements
pytest tests/test_invariants.py::test_fee_engine_closure -v -s
```

### Updating Golden Snapshots

```bash
# When changes are intentional
pytest tests/test_reproducibility.py --force-regen

# Review changes before committing
git diff tests/test_reproducibility/
```

## 📈 Performance Considerations

### Money Type Optimizations

- **Caching**: Common values (0-999 cents) are cached
- **Slots**: `__slots__` reduces memory overhead
- **Integer arithmetic**: ~5-10x faster than Decimal operations

### Test Performance

- **Hypothesis**: Use `--hypothesis-deadline=750` for CI
- **Markers**: Use `pytest -m "not slow"` to skip long-running tests
- **Parallel**: Use `pytest-xdist` for parallel execution

## 🛡️ Safety Guarantees

### Type Safety

- **MyPy compatibility**: Full type hints throughout
- **Runtime validation**: Float inputs rejected in strict mode
- **Currency isolation**: Different currencies cannot be mixed

### Determinism

- **Seeded RNG**: All randomness goes through `sim.rng`
- **Hash verification**: State changes tracked via SHA256 hashes
- **Platform independence**: Integer arithmetic eliminates platform differences

### Precision

- **Exact arithmetic**: No floating-point precision loss
- **Cent precision**: All amounts stored as integer cents
- **Rounding control**: Explicit rounding using banker's rounding

## 🔮 Future Enhancements

### Planned Features

- **Multi-currency support**: Full EUR/GBP/JPY support
- **MyPy plugin**: Compile-time float detection
- **Performance profiling**: Benchmark suite for regression detection
- **Fuzzing**: Property-based testing of edge cases

### Migration Roadmap

- **Fee Engine**: Convert all fee calculations to Money type
- **Simulation**: Integrate audit system into main loop
- **CI Guardrails**: AST linting and MyPy enforcement
- **Documentation**: Complete API documentation and examples

## 📚 References

- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [pytest-regressions](https://pytest-regressions.readthedocs.io/)
- [Python Decimal Module](https://docs.python.org/3/library/decimal.html)
- [Double-Entry Bookkeeping](https://en.wikipedia.org/wiki/Double-entry_bookkeeping)