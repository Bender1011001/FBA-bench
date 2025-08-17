#!/usr/bin/env python3
"""
FBA-Bench v3: Production-ready Experiment CLI

This module provides a robust, release-grade command-line interface for
executing large-scale simulation experiments. It focuses on determinism,
scalability, and clear observability, while delegating core simulation work
to the ScenarioEngine (scenarios/scenario_engine.py).

Features
- Run experiments from a sweep.yaml configuration
- Deterministic seeding per run (master seed + run index)
- Parallel execution using multiprocessing (spawn) for isolation
- Deterministic result storage with per-run JSON artifacts
- Lightweight, production-grade error handling and logging
- Simple analyze command to summarize results

Assumptions
- Each run config must include a "scenario_file" path and an optional "agents"
  dictionary that will be passed through to ScenarioEngine.run_simulation.
- The sweep.yaml format mirrors the structure used by the legacy system:
  {
    experiment_name: "My Experiment",
    description: "Description",
    base_parameters: { ... },
    parameter_sweep: { "paramA": [val1, val2], "paramB": [valX, valY] },
    output: { ... }
  }

Note
- This module is designed to be self-contained and production-friendly, with a
  minimal surface area for integration into release workflows.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import itertools
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Optional

import yaml

# Core dependency: ScenarioEngine performs the actual simulation steps
from scenarios.scenario_engine import ScenarioEngine

# Configure top-level logging for production usage
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("experiment_cli")

# ----------------------------
# Lightweight ExperimentConfig
# ----------------------------
class ExperimentConfig:
    def __init__(self, config_data: Dict[str, Any]):
        self.experiment_name: str = config_data.get("experiment_name", "untitled_experiment")
        self.description: str = config_data.get("description", "")
        self.base_parameters: Dict[str, Any] = config_data.get("base_parameters", {})
        self.parameter_sweep: Dict[str, List[Any]] = config_data.get("parameter_sweep", {})
        self.output_config: Dict[str, Any] = config_data.get("output", {})

    def generate_parameter_combinations(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """
        Generate all parameter combinations defined by the sweep.
        Yields (run_number, final_parameters).
        """
        sweep = self.parameter_sweep or {}
        param_names = list(sweep.keys())
        param_values = [sweep[name] for name in param_names]

        run_number = 1
        if not param_names:
            # No sweep specified; emit a single run with base parameters
            yield run_number, dict(self.base_parameters)
            return

        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            final_params = dict(self.base_parameters)
            final_params.update(params)
            yield run_number, final_params
            run_number += 1

    def get_total_combinations(self) -> int:
        total = 1
        for values in self.parameter_sweep.values():
            total *= len(values)
        if total <= 0:
            total = 1
        return total

# ----------------------------
# Helpers
# ----------------------------
def _load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _init_seed(seed: int) -> None:
    import random
    random.seed(seed)
    # Optional: also seed numpy if present in the runtime (no hard dependency here)

def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _write_json(target_path: Path, data: Any) -> None:
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

# ----------------------------
# Worker: production run
# ----------------------------
def _run_single_simulation_worker(args: Tuple[str, int, Dict[str, Any], str]) -> bool:
    """
    Worker function to execute a single simulation run in a separate process.

    Args:
        args: (config_file_path, run_number, final_parameters, results_dir)
    """
    config_file_path, run_number, final_parameters, results_dir = args
    try:
        # Deterministic seed per run
        master_seed = int(os.environ.get("MASTER_SEED", "0"))
        _init_seed(master_seed + run_number)

        engine = ScenarioEngine()
        scenario_file = final_parameters.get("scenario_file")
        agents = final_parameters.get("agents", {})

        if not scenario_file:
            raise ValueError("Missing 'scenario_file' in run parameters.")

        logger.info(f"Run {run_number}: executing scenario '{scenario_file}' with {len(agents)} agents.")
        result = engine.run_simulation(scenario_file, agents)

        results_path = Path(results_dir) / f"run_{run_number}.json"
        _write_json(results_path, result)
        logger.info(f"Run {run_number} completed; result saved to {results_path}")
        return True
    except Exception as e:
        logger.exception(f"Worker failed on run {run_number}: {e}")
        return False

# ----------------------------
# Analyze: lightweight summary
# ----------------------------
def analyze_results(results_dir: str) -> Dict[str, Any]:
    """
    Analyze all run result JSON files in the given directory and return a summary.

    Returns a dict: { "total_runs": int, "successful_runs": int, "average_profit": float, ... }
    """
    dir_path = Path(results_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    run_files = sorted(dir_path.glob("run_*.json"))
    summary: Dict[str, Any] = {
        "total_runs": len(run_files),
        "successful_runs": 0,
        "average_profit": None,
        "overall_profit": 0.0,
    }

    profits = []
    for f in run_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            profit = None
            if isinstance(data, dict):
                metrics = data.get("metrics") or {}
                profit = metrics.get("total_profit") or data.get("simulation_data", {}).get("final_profit")
            if isinstance(profit, (int, float)):
                profits.append(float(profit))
            # If run reported success in its own result, we mark it
            if data.get("final_state") is not None or data.get("success") == True:
                summary["successful_runs"] += 1
        except Exception:
            continue

    if profits:
        summary["average_profit"] = sum(profits) / len(profits)

    summary["overall_profit"] = sum(profits) if profits else 0.0
    return summary

# ----------------------------
# CLI entry point
# ----------------------------
def _load_config_and_expand_run_params(config_path: str) -> Tuple[ExperimentConfig, List[Tuple[int, Dict[str, Any]]]]:
    config_data = _load_yaml_config(config_path)
    exp_config = ExperimentConfig(config_data)

    # Generate the list of (run_number, final_parameters)
    run_items: List[Tuple[int, Dict[str, Any]]] = list(exp_config.generate_parameter_combinations())
    return exp_config, run_items

def _parallel_run(config_path: str, run_items: List[Tuple[int, Dict[str, Any]]], results_dir: Path, parallel: int) -> int:
    """
    Run runs in parallel using multiprocessing while respecting Windows spawn semantics.
    Returns number of successful runs.
    """
    if parallel <= 1:
        # Sequential fallback
        successes = 0
        for run_number, final_params in run_items:
            ok = _run_single_simulation_worker((config_path, run_number, final_params, str(results_dir)))
            if ok:
                successes += 1
        return successes

    # Parallel execution
    worker_args = [
        (config_path, run_number, final_params, str(results_dir))
        for (run_number, final_params) in run_items
    ]
    successes = 0
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=parallel, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(_run_single_simulation_worker, aw): aw[1] for aw in worker_args}
        for future in as_completed(futures):
            run_num = futures[future]
            try:
                if future.result():
                    successes += 1
            except Exception:
                logger.exception(f"Worker for run {run_num} raised an exception.")
    return successes

def _ensure_results_dir(base_dir: Path, experiment_name: str) -> Path:
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = base_dir / f"{experiment_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

# ----------------------------
# Main CLI
# ----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        prog="experiment_cli",
        description="FBA-Bench v3 Production Experiment CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser("run", help="Execute an experiment sweep from sweep.yaml")
    run_parser.add_argument("config_file", help="Path to sweep.yaml configuration")
    run_parser.add_argument("--max-runs", type=int, default=None, help="Limit total number of runs (for testing)")
    run_parser.add_argument("--parallel", type=int, default=1, help="Number of parallel worker processes")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results from a previous run")
    analyze_parser.add_argument("results_dir", help="Path to results directory to analyze")

    args = parser.parse_args()

    if args.command == "run":
        config_path = args.config_file
        if not os.path.exists(config_path):
            logger.error(f"Sweep config not found: {config_path}")
            return 2

        exp_config, run_items = _load_config_and_expand_run_params(config_path)

        total_available = len(run_items)
        if args.max_runs is not None:
            total_available = min(total_available, args.max_runs)
            run_items = run_items[:total_available]

        # Prepare results directory
        results_base = Path("results")
        results_dir = _ensure_results_dir(results_base, exp_config.experiment_name)

        # Persist the used sweep configuration for traceability
        try:
            with open(results_dir / "experiment_config.yaml", "w", encoding="utf-8") as f:
                f.write(yaml.safe_dump({
                    "experiment_name": exp_config.experiment_name,
                    "description": exp_config.description,
                    "base_parameters": exp_config.base_parameters,
                    "parameter_sweep": exp_config.parameter_sweep,
                    "output": exp_config.output_config,
                }, sort_keys=False))
        except Exception as e:
            logger.warning(f"Could not save experiment config: {e}")

        logger.info(f"Starting experiment: {exp_config.experiment_name}")
        logger.info(f"Total runs to execute: {len(run_items)} (parallel={args.parallel})")

        start = __import__("time").time()
        successes = _parallel_run(config_path, run_items, results_dir, args.parallel)
        duration = __import__("time").time() - start

        logger.info(f"Experiment completed: {successes}/{len(run_items)} successful runs in {duration:.2f}s.")
        # Optional: summarize
        try:
            summary = analyze_results(str(results_dir))
            logger.info(f"Aggregate results: {summary}")
            with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to compute summary: {e}")
        return 0

    elif args.command == "analyze":
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            logger.error(f"Results directory not found: {results_dir}")
            return 2
        try:
            summary = analyze_results(str(results_dir))
            logger.info(f"Analysis Summary: {summary}")
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return 1
        return 0

    else:
        logger.error("Unknown command. Use 'run' or 'analyze'.")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())