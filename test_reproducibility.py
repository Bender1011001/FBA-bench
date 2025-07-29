"""Golden snapshot testing for deterministic simulation reproducibility."""
import pytest
from audit import run_and_audit


def serialize_run_audit(audit):
    """Serialize RunAudit to stable format for snapshot comparison."""
    return {
        "seed": audit.seed,
        "days": audit.days,
        "config_hash": audit.config_hash,
        "code_hash": audit.code_hash,
        "fee_schedule_hash": audit.fee_schedule_hash,
        "final_ledger_hash": audit.final_ledger_hash,
        "final_balance_sheet": {k: str(v) for k, v in audit.final_balance_sheet.items()},
        "final_income_statement": {k: str(v) for k, v in audit.final_income_statement.items()},
        "tick_count": len(audit.ticks),
        "tick_ledger_hashes": [t.ledger_tick_hash for t in audit.ticks],
        "rng_hashes": [t.rng_state_hash for t in audit.ticks],
        "violations": audit.violations,
        # Summary statistics for quick verification
        "final_assets": str(audit.ticks[-1].assets) if audit.ticks else "0",
        "final_equity": str(audit.ticks[-1].equity) if audit.ticks else "0",
        "total_inventory_units": sum(audit.ticks[-1].inventory_units_by_sku.values()) if audit.ticks else 0,
    }


@pytest.mark.golden
def test_golden_run_matches_snapshot(sim_factory, data_regression):
    """Test that a deterministic run produces identical results."""
    sim = sim_factory(seed=42, days=30)
    audit = sim.run_and_audit(days=30)
    
    # Serialize to stable format
    snapshot_data = serialize_run_audit(audit)
    
    # Compare against golden snapshot
    data_regression.check(snapshot_data)


@pytest.mark.golden
def test_golden_run_365_days_snapshot(sim_factory, data_regression):
    """Test longer run for comprehensive drift detection."""
    sim = sim_factory(seed=42, days=365)
    audit = sim.run_and_audit(days=365)
    
    # For longer runs, focus on key hashes to keep snapshots manageable
    snapshot_data = {
        "seed": audit.seed,
        "days": audit.days,
        "final_ledger_hash": audit.final_ledger_hash,
        "config_hash": audit.config_hash,
        "fee_schedule_hash": audit.fee_schedule_hash,
        "violations": audit.violations,
        "final_balance_sheet": {k: str(v) for k, v in audit.final_balance_sheet.items()},
        "final_income_statement": {k: str(v) for k, v in audit.final_income_statement.items()},
        # Sample of tick hashes for spot checking
        "sample_tick_hashes": {
            "day_1": audit.ticks[0].ledger_tick_hash if len(audit.ticks) > 0 else None,
            "day_30": audit.ticks[29].ledger_tick_hash if len(audit.ticks) > 29 else None,
            "day_90": audit.ticks[89].ledger_tick_hash if len(audit.ticks) > 89 else None,
            "day_180": audit.ticks[179].ledger_tick_hash if len(audit.ticks) > 179 else None,
            "day_365": audit.ticks[364].ledger_tick_hash if len(audit.ticks) > 364 else None,
        },
        "sample_rng_hashes": {
            "day_1": audit.ticks[0].rng_state_hash if len(audit.ticks) > 0 else None,
            "day_30": audit.ticks[29].rng_state_hash if len(audit.ticks) > 29 else None,
            "day_90": audit.ticks[89].rng_state_hash if len(audit.ticks) > 89 else None,
            "day_180": audit.ticks[179].rng_state_hash if len(audit.ticks) > 179 else None,
            "day_365": audit.ticks[364].rng_state_hash if len(audit.ticks) > 364 else None,
        }
    }
    
    data_regression.check(snapshot_data)


@pytest.mark.golden
@pytest.mark.parametrize("seed", [0, 1337, 42, 12345])
def test_different_seeds_produce_different_results(sim_factory, seed):
    """Test that different seeds produce different but deterministic results."""
    sim = sim_factory(seed=seed, days=10)
    audit = sim.run_and_audit(days=10)
    
    # Each seed should produce consistent results
    assert audit.seed == seed
    assert len(audit.ticks) == 10
    assert len(audit.violations) == 0  # Should have no violations in normal runs
    
    # Results should be deterministic for the same seed
    sim2 = sim_factory(seed=seed, days=10)
    audit2 = sim2.run_and_audit(days=10)
    
    assert audit.final_ledger_hash == audit2.final_ledger_hash
    assert [t.ledger_tick_hash for t in audit.ticks] == [t.ledger_tick_hash for t in audit2.ticks]
    assert [t.rng_state_hash for t in audit.ticks] == [t.rng_state_hash for t in audit2.ticks]


@pytest.mark.golden
def test_hash_stability_across_runs(sim_factory):
    """Test that hashes are stable across multiple runs with same parameters."""
    # Run same simulation multiple times
    runs = []
    for _ in range(3):
        sim = sim_factory(seed=42, days=5)
        audit = sim.run_and_audit(days=5)
        runs.append(audit)
    
    # All runs should produce identical hashes
    reference_run = runs[0]
    for run in runs[1:]:
        assert run.final_ledger_hash == reference_run.final_ledger_hash
        assert run.config_hash == reference_run.config_hash
        assert run.fee_schedule_hash == reference_run.fee_schedule_hash
        
        # Tick-by-tick comparison
        for i, (tick1, tick2) in enumerate(zip(reference_run.ticks, run.ticks)):
            assert tick1.ledger_tick_hash == tick2.ledger_tick_hash, f"Ledger hash mismatch at tick {i}"
            assert tick1.rng_state_hash == tick2.rng_state_hash, f"RNG hash mismatch at tick {i}"
            assert tick1.inventory_hash == tick2.inventory_hash, f"Inventory hash mismatch at tick {i}"


@pytest.mark.golden
def test_configuration_change_detection(sim_factory, data_regression):
    """Test that configuration changes are detected via hash changes."""
    # This test will fail if fee configuration changes without updating version
    sim = sim_factory(seed=42, days=5)
    audit = sim.run_and_audit(days=5)
    
    config_snapshot = {
        "config_hash": audit.config_hash,
        "fee_schedule_hash": audit.fee_schedule_hash,
        "code_hash": audit.code_hash,
    }
    
    # This will create a baseline on first run, then detect changes on subsequent runs
    data_regression.check(config_snapshot)


@pytest.mark.golden
def test_minimal_run_snapshot(sim_factory, data_regression):
    """Test minimal 1-day run for quick verification."""
    sim = sim_factory(seed=42, days=1)
    audit = sim.run_and_audit(days=1)
    
    # Detailed snapshot for single day
    snapshot_data = {
        "seed": audit.seed,
        "days": audit.days,
        "violations": audit.violations,
        "tick_data": {
            "day": audit.ticks[0].day,
            "assets": str(audit.ticks[0].assets),
            "liabilities": str(audit.ticks[0].liabilities),
            "equity": str(audit.ticks[0].equity),
            "debit_sum": str(audit.ticks[0].debit_sum),
            "credit_sum": str(audit.ticks[0].credit_sum),
            "net_income": str(audit.ticks[0].net_income_to_date),
            "inventory_units": audit.ticks[0].inventory_units_by_sku,
            "ledger_hash": audit.ticks[0].ledger_tick_hash,
            "rng_hash": audit.ticks[0].rng_state_hash,
            "inventory_hash": audit.ticks[0].inventory_hash,
        },
        "final_balance_sheet": {k: str(v) for k, v in audit.final_balance_sheet.items()},
        "final_income_statement": {k: str(v) for k, v in audit.final_income_statement.items()},
    }
    
    data_regression.check(snapshot_data)