from __future__ import annotations
import uuid
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException

from fba_bench_api.core.state import experiment_configs_db
from fba_bench_api.models.experiments import (
    ExperimentCreateRequest, ExperimentStatusResponse, ExperimentResultsResponse
)

router = APIRouter(prefix="/api/v1/experiments", tags=["Experiments"])

@router.post("", response_model=ExperimentStatusResponse, status_code=201)
async def create_experiment(req: ExperimentCreateRequest):
    exp_id = str(uuid.uuid4())
    now = datetime.now()
    total = 1
    for values in req.parameter_sweep.values():
        total *= len(values)
    if req.max_runs:
        total = min(total, req.max_runs)

    data: Dict[str, Any] = {
        "experiment_id": exp_id,
        "config": {
            "experiment_name": req.experiment_name,
            "description": req.description,
            "base_parameters": req.base_parameters,
            "parameter_sweep": req.parameter_sweep,
            "output_config": req.output_config,
            "parallel_workers": req.parallel_workers,
            "max_runs": req.max_runs,
        },
        "status": "running",
        "total_runs": total,
        "completed_runs": 0,
        "successful_runs": 0,
        "failed_runs": 0,
        "progress_percentage": 0.0,
        "start_time": now,
        "end_time": None,
        "current_run_details": None,
        "message": f"Experiment started with {total} total runs",
    }
    experiment_configs_db[exp_id] = data
    return ExperimentStatusResponse(**data)

@router.get("/{experiment_id}", response_model=ExperimentStatusResponse)
async def get_experiment_status(experiment_id: str):
    if experiment_id not in experiment_configs_db:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found")
    return ExperimentStatusResponse(**experiment_configs_db[experiment_id])

@router.post("/{experiment_id}/stop", response_model=ExperimentStatusResponse)
async def stop_experiment(experiment_id: str):
    if experiment_id not in experiment_configs_db:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found")
    ex = experiment_configs_db[experiment_id]
    if ex["status"] not in {"running", "paused"}:
        raise HTTPException(400, f"Experiment '{experiment_id}' is not running")
    ex["status"] = "stopped"
    ex["end_time"] = datetime.now()
    ex["message"] = "Experiment stopped by user request"
    return ExperimentStatusResponse(**ex)

@router.get("/{experiment_id}/results", response_model=ExperimentResultsResponse)
async def get_experiment_results(experiment_id: str):
    if experiment_id not in experiment_configs_db:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found")
    ex = experiment_configs_db[experiment_id]
    summary = {
        "experiment_name": ex["config"]["experiment_name"],
        "total_runs": ex["total_runs"],
        "successful_runs": ex["successful_runs"],
        "failed_runs": ex["failed_runs"],
        "average_execution_time": 0.0,
        "parameter_sweep_summary": ex["config"]["parameter_sweep"],
    }
    return ExperimentResultsResponse(
        experiment_id=experiment_id,
        status=ex["status"],
        results_summary=summary,
        individual_run_results=[],
        results_uri=f"results/{ex['config']['experiment_name']}_{experiment_id}"
    )

# Convenience lists for your ExperimentManagement page
@router.get("/../api/experiments/active")
async def get_active_experiments():
    out = []
    for exp_id, ex in experiment_configs_db.items():
        if ex.get("status") in {"running", "queued", "paused"}:
            out.append({
                "id": exp_id,
                "experimentName": ex["config"]["experiment_name"],
                "description": ex["config"].get("description", ""),
                "config": ex["config"],
                "status": ex["status"],
                "progress": ex.get("progress_percentage", 0),
                "startTime": ex.get("start_time").isoformat() if ex.get("start_time") else None,
                "lastUpdated": ex.get("last_updated", ""),
            })
    return out

@router.get("/../api/experiments/completed-full")
async def get_completed_experiments():
    out = []
    for exp_id, ex in experiment_configs_db.items():
        if ex.get("status") in {"completed", "failed", "cancelled", "stopped"}:
            out.append({
                "id": exp_id,
                "experimentName": ex["config"]["experiment_name"],
                "description": ex["config"].get("description", ""),
                "config": ex["config"],
                "status": ex["status"],
                "progress": 100 if ex["status"] == "completed" else ex.get("progress_percentage", 0),
                "startTime": ex.get("start_time").isoformat() if ex.get("start_time") else None,
                "endTime": ex.get("end_time").isoformat() if ex.get("end_time") else None,
                "lastUpdated": ex.get("last_updated", ""),
            })
    return out