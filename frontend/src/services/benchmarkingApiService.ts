import { apiService } from './apiService';
import type {
  BenchmarkResult,
  MultiDimensionalMetric,
  ExecutionProgress,
  CapabilityAssessment,
  PerformanceHeatmap
} from '../types';

// Benchmark Configuration Management
export interface CreateBenchmarkConfigRequest {
  name: string;
  description?: string;
  scenarios: string[];
  agents: string[];
  metrics: string[];
  duration: number;
  iterations: number;
  parameters?: Record<string, unknown>;
}

export interface UpdateBenchmarkConfigRequest extends Partial<CreateBenchmarkConfigRequest> {
  id: string;
}

export interface BenchmarkConfigResponse extends CreateBenchmarkConfigRequest {
  id: string;
  created_at: string;
  updated_at: string;
  is_active: boolean;
}

// Scenario Management
export interface CreateScenarioRequest {
  name: string;
  description?: string;
  type: 'static' | 'dynamic' | 'adaptive';
  parameters: Record<string, unknown>;
  difficulty: 'easy' | 'medium' | 'hard';
  tags?: string[];
}

export interface UpdateScenarioRequest extends Partial<CreateScenarioRequest> {
  id: string;
}

export interface ScenarioResponse extends CreateScenarioRequest {
  id: string;
  created_at: string;
  updated_at: string;
  is_active: boolean;
}

// Metrics Data Retrieval
export interface MetricsQueryRequest {
  benchmark_id?: string;
  agent_id?: string;
  scenario_id?: string;
  metric_names?: string[];
  time_range?: {
    start: string;
    end: string;
  };
  group_by?: 'agent' | 'scenario' | 'time';
  aggregation?: 'avg' | 'sum' | 'min' | 'max' | 'count';
}

export interface MetricsResponse {
  data: MultiDimensionalMetric[];
  summary: {
    total_points: number;
    time_range: {
      start: string;
      end: string;
    };
    agents: string[];
    scenarios: string[];
    metrics: string[];
  };
}

// Results Export
export interface ExportRequest {
  benchmark_id: string;
  format: 'csv' | 'json' | 'pdf';
  include_metadata?: boolean;
  include_raw_data?: boolean;
  filters?: {
    agents?: string[];
    scenarios?: string[];
    metrics?: string[];
    time_range?: {
      start: string;
      end: string;
    };
  };
}

export interface ExportResponse {
  download_url: string;
  expires_at: string;
  file_size: number;
  format: string;
}

class BenchmarkingApiService {
  // Benchmark Configuration Management
  async createBenchmarkConfig(config: CreateBenchmarkConfigRequest): Promise<BenchmarkConfigResponse> {
    const response = await apiService.post<BenchmarkConfigResponse>('/benchmarking/configs', config);
    return response.data;
  }

  async getBenchmarkConfig(id: string): Promise<BenchmarkConfigResponse> {
    const response = await apiService.get<BenchmarkConfigResponse>(`/benchmarking/configs/${id}`);
    return response.data;
  }

  async updateBenchmarkConfig(config: UpdateBenchmarkConfigRequest): Promise<BenchmarkConfigResponse> {
    const response = await apiService.patch<BenchmarkConfigResponse>(`/benchmarking/configs/${config.id}`, config);
    return response.data;
  }

  async deleteBenchmarkConfig(id: string): Promise<void> {
    await apiService.delete<void>(`/benchmarking/configs/${id}`);
  }

  async listBenchmarkConfigs(): Promise<BenchmarkConfigResponse[]> {
    const response = await apiService.get<BenchmarkConfigResponse[]>('/benchmarking/configs');
    return response.data;
  }

  async validateBenchmarkConfig(id: string): Promise<{ is_valid: boolean; errors: string[] }> {
    const response = await apiService.post<{ is_valid: boolean; errors: string[] }>(`/benchmarking/configs/${id}/validate`);
    return response.data;
  }

  // Scenario Management
  async createScenario(scenario: CreateScenarioRequest): Promise<ScenarioResponse> {
    const response = await apiService.post<ScenarioResponse>('/benchmarking/scenarios', scenario);
    return response.data;
  }

  async getScenario(id: string): Promise<ScenarioResponse> {
    const response = await apiService.get<ScenarioResponse>(`/benchmarking/scenarios/${id}`);
    return response.data;
  }

  async updateScenario(scenario: UpdateScenarioRequest): Promise<ScenarioResponse> {
    const response = await apiService.patch<ScenarioResponse>(`/benchmarking/scenarios/${scenario.id}`, scenario);
    return response.data;
  }

  async deleteScenario(id: string): Promise<void> {
    await apiService.delete<void>(`/benchmarking/scenarios/${id}`);
  }

  async listScenarios(): Promise<ScenarioResponse[]> {
    const response = await apiService.get<ScenarioResponse[]>('/benchmarking/scenarios');
    return response.data;
  }

  async validateScenario(id: string): Promise<{ is_valid: boolean; errors: string[] }> {
    const response = await apiService.post<{ is_valid: boolean; errors: string[] }>(`/benchmarking/scenarios/${id}/validate`);
    return response.data;
  }

  // Benchmark Execution
  async startBenchmark(configId: string): Promise<{ execution_id: string; status: string }> {
    const response = await apiService.post<{ execution_id: string; status: string }>(`/benchmarking/configs/${configId}/start`);
    return response.data;
  }

  async stopBenchmark(executionId: string): Promise<void> {
    await apiService.post<void>(`/benchmarking/executions/${executionId}/stop`);
  }

  async getExecutionStatus(executionId: string): Promise<ExecutionProgress> {
    const response = await apiService.get<ExecutionProgress>(`/benchmarking/executions/${executionId}/status`);
    return response.data;
  }

  async getExecutionResults(executionId: string): Promise<BenchmarkResult> {
    const response = await apiService.get<BenchmarkResult>(`/benchmarking/executions/${executionId}/results`);
    return response.data;
  }

  async listExecutions(): Promise<Array<{ id: string; config_id: string; status: string; started_at: string; completed_at?: string }>> {
    const response = await apiService.get<Array<{ id: string; config_id: string; status: string; started_at: string; completed_at?: string }>>('/benchmarking/executions');
    return response.data;
  }

  // Metrics Data Retrieval
  async getMetrics(query: MetricsQueryRequest): Promise<MetricsResponse> {
    const response = await apiService.post<MetricsResponse>('/benchmarking/metrics/query', query);
    return response.data;
  }

  async getCapabilityAssessment(agentId: string, scenarioId: string): Promise<CapabilityAssessment> {
    const response = await apiService.get<CapabilityAssessment>(`/benchmarking/capabilities/${agentId}/${scenarioId}`);
    return response.data;
  }

  async getPerformanceHeatmap(agentIds: string[], scenarioIds: string[], metricNames: string[]): Promise<PerformanceHeatmap> {
    const response = await apiService.post<PerformanceHeatmap>('/benchmarking/heatmap', {
      agents: agentIds,
      scenarios: scenarioIds,
      metrics: metricNames
    });
    return response.data;
  }

  // Results Export and Sharing
  async exportResults(exportRequest: ExportRequest): Promise<ExportResponse> {
    const response = await apiService.post<ExportResponse>('/benchmarking/export', exportRequest);
    return response.data;
  }

  async shareResults(benchmarkId: string, recipients: string[], message?: string): Promise<{ share_id: string; expires_at: string }> {
    const response = await apiService.post<{ share_id: string; expires_at: string }>(`/benchmarking/${benchmarkId}/share`, {
      recipients,
      message
    });
    return response.data;
  }

  async getSharedResults(shareId: string): Promise<BenchmarkResult> {
    const response = await apiService.get<BenchmarkResult>(`/benchmarking/shared/${shareId}`);
    return response.data;
  }

  // Templates and Presets
  async getBenchmarkTemplates(): Promise<Array<{ id: string; name: string; description: string; category: string }>> {
    const response = await apiService.get<Array<{ id: string; name: string; description: string; category: string }>>('/benchmarking/templates');
    return response.data;
  }

  async getScenarioTemplates(): Promise<Array<{ id: string; name: string; description: string; category: string }>> {
    const response = await apiService.get<Array<{ id: string; name: string; description: string; category: string }>>('/benchmarking/scenario-templates');
    return response.data;
  }

  async applyBenchmarkTemplate(templateId: string, customizations?: Record<string, unknown>): Promise<BenchmarkConfigResponse> {
    const response = await apiService.post<BenchmarkConfigResponse>(`/benchmarking/templates/${templateId}/apply`, {
      customizations
    });
    return response.data;
  }

  // Configuration Import/Export
  async importBenchmarkConfig(configData: unknown): Promise<BenchmarkConfigResponse> {
    const response = await apiService.post<BenchmarkConfigResponse>('/benchmarking/import', configData);
    return response.data;
  }

  async exportBenchmarkConfig(configId: string, format: 'json' | 'yaml' = 'json'): Promise<string> {
    const response = await apiService.get<string>(`/benchmarking/configs/${configId}/export?format=${format}`);
    return response.data;
  }

  // Real-time Updates (WebSocket integration)
  async subscribeToExecutionUpdates(executionId: string): Promise<void> {
    // This would integrate with the WebSocket service
    // Implementation depends on the WebSocket service structure
    console.log(`Subscribing to execution updates for ${executionId}`);
  }

  async subscribeToMetricsUpdates(executionId: string): Promise<void> {
    // This would integrate with the WebSocket service
    console.log(`Subscribing to metrics updates for ${executionId}`);
  }
}

// Export singleton instance
export const benchmarkingApiService = new BenchmarkingApiService();

// Export convenience functions
export async function createBenchmarkConfig(config: CreateBenchmarkConfigRequest): Promise<BenchmarkConfigResponse> {
  return benchmarkingApiService.createBenchmarkConfig(config);
}

export async function getBenchmarkConfig(id: string): Promise<BenchmarkConfigResponse> {
  return benchmarkingApiService.getBenchmarkConfig(id);
}

export async function updateBenchmarkConfig(config: UpdateBenchmarkConfigRequest): Promise<BenchmarkConfigResponse> {
  return benchmarkingApiService.updateBenchmarkConfig(config);
}

export async function deleteBenchmarkConfig(id: string): Promise<void> {
  return benchmarkingApiService.deleteBenchmarkConfig(id);
}

export async function listBenchmarkConfigs(): Promise<BenchmarkConfigResponse[]> {
  return benchmarkingApiService.listBenchmarkConfigs();
}

export async function createScenario(scenario: CreateScenarioRequest): Promise<ScenarioResponse> {
  return benchmarkingApiService.createScenario(scenario);
}

export async function getScenario(id: string): Promise<ScenarioResponse> {
  return benchmarkingApiService.getScenario(id);
}

export async function updateScenario(scenario: UpdateScenarioRequest): Promise<ScenarioResponse> {
  return benchmarkingApiService.updateScenario(scenario);
}

export async function deleteScenario(id: string): Promise<void> {
  return benchmarkingApiService.deleteScenario(id);
}

export async function listScenarios(): Promise<ScenarioResponse[]> {
  return benchmarkingApiService.listScenarios();
}

export async function startBenchmark(configId: string): Promise<{ execution_id: string; status: string }> {
  return benchmarkingApiService.startBenchmark(configId);
}

export async function stopBenchmark(executionId: string): Promise<void> {
  return benchmarkingApiService.stopBenchmark(executionId);
}

export async function getExecutionStatus(executionId: string): Promise<ExecutionProgress> {
  return benchmarkingApiService.getExecutionStatus(executionId);
}

export async function getExecutionResults(executionId: string): Promise<BenchmarkResult> {
  return benchmarkingApiService.getExecutionResults(executionId);
}

export async function listExecutions(): Promise<Array<{ id: string; config_id: string; status: string; started_at: string; completed_at?: string }>> {
  return benchmarkingApiService.listExecutions();
}

export async function getMetrics(query: MetricsQueryRequest): Promise<MetricsResponse> {
  return benchmarkingApiService.getMetrics(query);
}

export async function getCapabilityAssessment(agentId: string, scenarioId: string): Promise<CapabilityAssessment> {
  return benchmarkingApiService.getCapabilityAssessment(agentId, scenarioId);
}

export async function getPerformanceHeatmap(agentIds: string[], scenarioIds: string[], metricNames: string[]): Promise<PerformanceHeatmap> {
  return benchmarkingApiService.getPerformanceHeatmap(agentIds, scenarioIds, metricNames);
}

export async function exportResults(exportRequest: ExportRequest): Promise<ExportResponse> {
  return benchmarkingApiService.exportResults(exportRequest);
}

export async function shareResults(benchmarkId: string, recipients: string[], message?: string): Promise<{ share_id: string; expires_at: string }> {
  return benchmarkingApiService.shareResults(benchmarkId, recipients, message);
}

export async function getSharedResults(shareId: string): Promise<BenchmarkResult> {
  return benchmarkingApiService.getSharedResults(shareId);
}

export async function getBenchmarkTemplates(): Promise<Array<{ id: string; name: string; description: string; category: string }>> {
  return benchmarkingApiService.getBenchmarkTemplates();
}

export async function getScenarioTemplates(): Promise<Array<{ id: string; name: string; description: string; category: string }>> {
  return benchmarkingApiService.getScenarioTemplates();
}

export async function applyBenchmarkTemplate(templateId: string, customizations?: Record<string, unknown>): Promise<BenchmarkConfigResponse> {
  return benchmarkingApiService.applyBenchmarkTemplate(templateId, customizations);
}

export async function importBenchmarkConfig(configData: unknown): Promise<BenchmarkConfigResponse> {
  return benchmarkingApiService.importBenchmarkConfig(configData);
}

export async function exportBenchmarkConfig(configId: string, format: 'json' | 'yaml' = 'json'): Promise<string> {
  return benchmarkingApiService.exportBenchmarkConfig(configId, format);
}