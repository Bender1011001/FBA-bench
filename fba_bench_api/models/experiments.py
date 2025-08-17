from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class ConfigTemplateSave(BaseModel):
    config_id: str
    template_name: str
    description: Optional[str] = None

class ConfigTemplateResponse(BaseModel):
    template_name: str
    description: Optional[str]
    config_data: Dict[str, Any]
    created_at: datetime

class BenchmarkConfigRequest(BaseModel):
    simulationSettings: Dict[str, Any]
    agentConfigs: List[Dict[str, Any]]
    llmSettings: Dict[str, Any]
    constraints: Dict[str, Any]
    experimentSettings: Dict[str, Any]

class BenchmarkConfigResponse(BaseModel):
    success: bool
    message: str
    config_id: str
    created_at: datetime

class ExperimentCreateRequest(BaseModel):
    experiment_name: str
    description: Optional[str] = None
    base_parameters: Dict[str, Any]
    parameter_sweep: Dict[str, List[Any]]
    output_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    parallel_workers: int = Field(1, ge=1)
    max_runs: Optional[int] = Field(None, ge=1)

class ExperimentStatusResponse(BaseModel):
    experiment_id: str
    status: str
    total_runs: int
    completed_runs: int
    successful_runs: int
    failed_runs: int
    progress_percentage: float
    start_time: datetime
    end_time: Optional[datetime] = None
    current_run_details: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class ExperimentResultsResponse(BaseModel):
    experiment_id: str
    status: str
    results_summary: Dict[str, Any]
    individual_run_results: List[Dict[str, Any]]
    results_uri: Optional[str] = None