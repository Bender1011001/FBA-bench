// Shared API types aligned with backend Pydantic models (approximate and tolerant to extra fields)

export type ExperimentStatus = 'draft' | 'running' | 'completed' | 'failed';
export type SimulationStatus = 'queued' | 'running' | 'completed' | 'failed';

// Generic, backend-extensible map types
export type MetricsSummary = Record<string, number>;
export type ValidatorsSummary = Record<string, boolean | string>;

// Experiment entity
export interface Experiment {
  id: string;
  name: string;
  description?: string;
  status: ExperimentStatus;
  created_at: string; // ISO datetime
  updated_at: string; // ISO datetime
  [key: string]: unknown; // allow backend-extensible fields
}

// Agent entity
export interface Agent {
  id: string;
  name: string;
  description?: string;
  runner: string;
  config?: Record<string, unknown>;
  created_at: string; // ISO datetime
  updated_at: string; // ISO datetime
  [key: string]: unknown; // allow backend-extensible fields
}

// Simulation run entity
export interface Simulation {
  id: string;
  experiment_id: string;
  status: SimulationStatus;
  started_at?: string; // ISO datetime
  finished_at?: string; // ISO datetime
  [key: string]: unknown; // allow backend-extensible fields
}

// Report payloads produced by the engine
export interface EngineScenarioReport {
  scenario_key: string;
  metrics: MetricsSummary;
  validators: ValidatorsSummary;
  summary?: string;
  [key: string]: unknown; // allow backend-extensible fields
}

export interface EngineReport {
  experiment_id: string;
  scenario_reports: Array<EngineScenarioReport>;
  totals?: MetricsSummary;
  created_at: string; // ISO datetime
  [key: string]: unknown; // allow backend-extensible fields
}

// Helper: common API shape for delete responses
export interface OkResponse {
  ok: true;
  [key: string]: unknown;
}