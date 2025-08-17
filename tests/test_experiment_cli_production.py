import os
import json
import yaml
from pathlib import Path
from typing import List, Tuple

import pytest

import experiment_cli as ec

def _safe_dump(config: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

def test_experiment_cli_sequential_run(tmp_path, monkeypatch):
    # Prepare a minimal sweep.yaml with a single run
    config = {
        "experiment_name": "prod_cli_test",
        "description": "unit test sweep for production CLI",
        "base_parameters": {"scenario_file": "dummy_scenario.yaml"},
        "parameter_sweep": {},
        "output": {}
    }
    config_path = Path(tmp_path) / "sweep.yaml"
    _safe_dump(config, config_path)

    # Patch ScenarioEngine.run_simulation to avoid real environment
    def mock_run(self, scenario_file, agents):
        return {
            "final_state": {"ok": True},
            "simulation_duration": 1,
            "metrics": {"total_profit": 123.45}
        }
    monkeypatch.setattr(ec.ScenarioEngine, "run_simulation", mock_run, raising=True)

    exp_config, run_items_iter = ec._load_config_and_expand_run_params(str(config_path))
    run_items = list(run_items_iter)
    assert len(run_items) == 1  # ensure one run

    results_dir = Path(tmp_path) / "results"
    results_dir.mkdir()
    run_numbers = run_items

    successes = ec._parallel_run(str(config_path), run_numbers, str(results_dir), parallel=1)
    assert successes == 1
    # Verify result file exists
    result_file = results_dir / "run_1.json"
    assert result_file.exists()
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "simulation_duration" in data
    assert "metrics" in data and "total_profit" in data["metrics"]
