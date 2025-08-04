import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { benchmarkingApiService } from '../../services/benchmarkingApiService';
import { webSocketService } from '../../services/webSocketService';

// Mock the underlying API service
jest.mock('../../services/apiService');
import { apiService } from '../../services/apiService';

describe('Benchmarking API Integration Tests', () => {
  const mockApiService = apiService as jest.Mocked<typeof apiService>;

  beforeEach(() => {
    jest.clearAllMocks();
    mockApiService.get.mockClear();
    mockApiService.post.mockClear();
    mockApiService.put.mockClear();
    mockApiService.patch.mockClear();
    mockApiService.delete.mockClear();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Benchmark Configuration Management', () => {
    const mockConfig = {
      id: '1',
      name: 'Test Benchmark',
      description: 'A test benchmark configuration',
      scenarios: ['scenario1', 'scenario2'],
      agents: ['agent1', 'agent2'],
      metrics: ['performance', 'accuracy'],
      duration: 300,
      iterations: 10,
      created_at: '2023-01-01T00:00:00Z',
      updated_at: '2023-01-01T00:00:00Z',
      is_active: true
    };

    it('should create a benchmark configuration', async () => {
      const createRequest = {
        name: 'Test Benchmark',
        description: 'A test benchmark configuration',
        scenarios: ['scenario1', 'scenario2'],
        agents: ['agent1', 'agent2'],
        metrics: ['performance', 'accuracy'],
        duration: 300,
        iterations: 10
      };

      mockApiService.post.mockResolvedValue({
        data: mockConfig
      });

      const result = await benchmarkingApiService.createBenchmarkConfig(createRequest);

      expect(mockApiService.post).toHaveBeenCalledWith('/benchmarking/configs', createRequest);
      expect(result).toEqual(mockConfig);
    });

    it('should get a benchmark configuration by ID', async () => {
      mockApiService.get.mockResolvedValue({
        data: mockConfig
      });

      const result = await benchmarkingApiService.getBenchmarkConfig('1');

      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/configs/1');
      expect(result).toEqual(mockConfig);
    });

    it('should update a benchmark configuration', async () => {
      const updateRequest = {
        id: '1',
        name: 'Updated Benchmark',
        description: 'An updated benchmark configuration',
        scenarios: ['scenario1', 'scenario2'],
        agents: ['agent1', 'agent2'],
        metrics: ['performance', 'accuracy'],
        duration: 300,
        iterations: 10
      };

      mockApiService.patch.mockResolvedValue({
        data: { ...mockConfig, ...updateRequest }
      });

      const result = await benchmarkingApiService.updateBenchmarkConfig(updateRequest);

      expect(mockApiService.patch).toHaveBeenCalledWith('/benchmarking/configs/1', updateRequest);
      expect(result.name).toBe('Updated Benchmark');
    });

    it('should delete a benchmark configuration', async () => {
      mockApiService.delete.mockResolvedValue({
        data: null
      });

      await benchmarkingApiService.deleteBenchmarkConfig('1');

      expect(mockApiService.delete).toHaveBeenCalledWith('/benchmarking/configs/1');
    });

    it('should list all benchmark configurations', async () => {
      const mockConfigs = [mockConfig, { ...mockConfig, id: '2', name: 'Second Benchmark' }];
      
      mockApiService.get.mockResolvedValue({
        data: mockConfigs
      });

      const result = await benchmarkingApiService.listBenchmarkConfigs();

      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/configs');
      expect(result).toHaveLength(2);
      expect(result[0].name).toBe('Test Benchmark');
    });

    it('should validate a benchmark configuration', async () => {
      mockApiService.post.mockResolvedValue({
        data: { is_valid: true, errors: [] }
      });

      const result = await benchmarkingApiService.validateBenchmarkConfig('1');

      expect(mockApiService.post).toHaveBeenCalledWith('/benchmarking/configs/1/validate');
      expect(result.is_valid).toBe(true);
      expect(result.errors).toEqual([]);
    });
  });

  describe('Scenario Management', () => {
    const mockScenario = {
      id: '1',
      name: 'Test Scenario',
      description: 'A test scenario',
      type: 'static' as const,
      parameters: { difficulty: 'medium' },
      difficulty: 'medium' as const,
      created_at: '2023-01-01T00:00:00Z',
      updated_at: '2023-01-01T00:00:00Z',
      is_active: true
    };

    it('should create a scenario', async () => {
      const createRequest = {
        name: 'Test Scenario',
        description: 'A test scenario',
        type: 'static' as const,
        parameters: { difficulty: 'medium' },
        difficulty: 'medium' as const
      };

      mockApiService.post.mockResolvedValue({
        data: mockScenario
      });

      const result = await benchmarkingApiService.createScenario(createRequest);

      expect(mockApiService.post).toHaveBeenCalledWith('/benchmarking/scenarios', createRequest);
      expect(result).toEqual(mockScenario);
    });

    it('should get a scenario by ID', async () => {
      mockApiService.get.mockResolvedValue({
        data: mockScenario
      });

      const result = await benchmarkingApiService.getScenario('1');

      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/scenarios/1');
      expect(result).toEqual(mockScenario);
    });

    it('should update a scenario', async () => {
      const updateRequest = {
        id: '1',
        name: 'Updated Scenario',
        description: 'An updated scenario',
        type: 'dynamic' as const,
        parameters: { difficulty: 'hard' },
        difficulty: 'hard' as const
      };

      mockApiService.patch.mockResolvedValue({
        data: { ...mockScenario, ...updateRequest }
      });

      const result = await benchmarkingApiService.updateScenario(updateRequest);

      expect(mockApiService.patch).toHaveBeenCalledWith('/benchmarking/scenarios/1', updateRequest);
      expect(result.name).toBe('Updated Scenario');
    });

    it('should delete a scenario', async () => {
      mockApiService.delete.mockResolvedValue({
        data: null
      });

      await benchmarkingApiService.deleteScenario('1');

      expect(mockApiService.delete).toHaveBeenCalledWith('/benchmarking/scenarios/1');
    });

    it('should list all scenarios', async () => {
      const mockScenarios = [mockScenario, { ...mockScenario, id: '2', name: 'Second Scenario' }];
      
      mockApiService.get.mockResolvedValue({
        data: mockScenarios
      });

      const result = await benchmarkingApiService.listScenarios();

      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/scenarios');
      expect(result).toHaveLength(2);
      expect(result[0].name).toBe('Test Scenario');
    });
  });

  describe('Benchmark Execution', () => {
    it('should start a benchmark', async () => {
      mockApiService.post.mockResolvedValue({
        data: { execution_id: 'exec1', status: 'starting' }
      });

      const result = await benchmarkingApiService.startBenchmark('1');

      expect(mockApiService.post).toHaveBeenCalledWith('/benchmarking/executions/config/1/start');
      expect(result.execution_id).toBe('exec1');
      expect(result.status).toBe('starting');
    });

    it('should stop a benchmark execution', async () => {
      mockApiService.post.mockResolvedValue({
        data: null
      });

      await benchmarkingApiService.stopBenchmark('exec1');

      expect(mockApiService.post).toHaveBeenCalledWith('/benchmarking/executions/exec1/stop');
    });

    it('should get execution status', async () => {
      const mockProgress = {
        execution_id: 'exec1',
        status: 'running',
        progress: 65,
        current_iteration: 7,
        total_iterations: 10,
        elapsed_time: 180,
        estimated_remaining_time: 90,
        current_scenario: 'scenario1',
        active_agents: ['agent1', 'agent2'],
        metrics: {
          performance: 0.85,
          accuracy: 0.92
        }
      };

      mockApiService.get.mockResolvedValue({
        data: mockProgress
      });

      const result = await benchmarkingApiService.getExecutionStatus('exec1');

      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/executions/exec1/status');
      expect(result.status).toBe('running');
      expect(result.progress).toBe(65);
    });

    it('should get execution results', async () => {
      const mockResults = {
        execution_id: 'exec1',
        benchmark_config_id: '1',
        start_time: '2023-01-01T00:00:00Z',
        end_time: '2023-01-01T00:05:00Z',
        status: 'completed',
        total_iterations: 10,
        completed_iterations: 10,
        scenario_results: []
      };

      mockApiService.get.mockResolvedValue({
        data: mockResults
      });

      const result = await benchmarkingApiService.getExecutionResults('exec1');

      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/executions/exec1/results');
      expect(result.status).toBe('completed');
      expect(result.completed_iterations).toBe(10);
    });

    it('should list all executions', async () => {
      const mockExecutions = [
        {
          id: 'exec1',
          config_id: '1',
          status: 'completed',
          started_at: '2023-01-01T00:00:00Z',
          completed_at: '2023-01-01T00:05:00Z'
        },
        {
          id: 'exec2',
          config_id: '1',
          status: 'running',
          started_at: '2023-01-01T00:10:00Z'
        }
      ];

      mockApiService.get.mockResolvedValue({
        data: mockExecutions
      });

      const result = await benchmarkingApiService.listExecutions();

      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/executions');
      expect(result).toHaveLength(2);
      expect(result[0].status).toBe('completed');
      expect(result[1].status).toBe('running');
    });
  });

  describe('Metrics Data Retrieval', () => {
    it('should get metrics data', async () => {
      const mockMetrics = [
        {
          agent_id: 'agent1',
          scenario_name: 'scenario1',
          timestamp: '2023-01-01T00:00:00Z',
          values: [
            { name: 'performance', value: 0.85 },
            { name: 'accuracy', value: 0.92 }
          ]
        }
      ];

      const mockResponse = {
        data: mockMetrics,
        summary: {
          total_points: 1,
          time_range: {
            start: '2023-01-01T00:00:00Z',
            end: '2023-01-01T00:05:00Z'
          },
          agents: ['agent1'],
          scenarios: ['scenario1'],
          metrics: ['performance', 'accuracy']
        }
      };

      mockApiService.get.mockResolvedValue(mockResponse);

      const query = {
        benchmark_id: 'exec1',
        metric_names: ['performance', 'accuracy'],
        group_by: 'agent' as const
      };

      const result = await benchmarkingApiService.getMetrics(query);

      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/metrics', {
        params: {
          benchmark_id: 'exec1',
          metric_names: ['performance', 'accuracy'],
          group_by: 'agent'
        }
      });
      expect(result.data).toHaveLength(1);
      expect(result.data[0].agent_id).toBe('agent1');
    });

    it('should get capability assessment', async () => {
      const mockAssessment = {
        agent_id: 'agent1',
        scenario_id: 'scenario1',
        capabilities: [
          { name: 'decision_making', score: 0.85, confidence: 0.92 },
          { name: 'adaptability', score: 0.78, confidence: 0.88 }
        ],
        overall_score: 0.82,
        assessment_date: '2023-01-01T00:00:00Z'
      };

      mockApiService.get.mockResolvedValue({
        data: mockAssessment
      });

      const result = await benchmarkingApiService.getCapabilityAssessment('agent1', 'scenario1');

      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/capabilities/agent1/scenario1');
      expect(result.agent_id).toBe('agent1');
      expect(result.scenario_id).toBe('scenario1');
      expect(result.capabilities).toHaveLength(2);
    });

    it('should get performance heatmap', async () => {
      const mockHeatmap = {
        agents: ['agent1', 'agent2'],
        scenarios: ['scenario1', 'scenario2'],
        metrics: ['performance', 'accuracy'],
        data: [
          {
            agent: 'agent1',
            scenario: 'scenario1',
            performance: 0.85,
            accuracy: 0.92
          },
          {
            agent: 'agent1',
            scenario: 'scenario2',
            performance: 0.78,
            accuracy: 0.88
          }
        ]
      };

      mockApiService.post.mockResolvedValue({
        data: mockHeatmap
      });

      const result = await benchmarkingApiService.getPerformanceHeatmap(
        ['agent1', 'agent2'],
        ['scenario1', 'scenario2'],
        ['performance', 'accuracy']
      );

      expect(mockApiService.post).toHaveBeenCalledWith('/benchmarking/heatmap', {
        agents: ['agent1', 'agent2'],
        scenarios: ['scenario1', 'scenario2'],
        metrics: ['performance', 'accuracy']
      });
      expect(result.agents).toHaveLength(2);
      expect(result.scenarios).toHaveLength(2);
      expect(result.data).toHaveLength(2);
    });
  });

  describe('Results Export and Sharing', () => {
    it('should export results', async () => {
      const mockExport = {
        download_url: 'http://example.com/download',
        expires_at: '2023-01-01T01:00:00Z',
        file_size: 1024,
        format: 'csv' as const
      };

      mockApiService.post.mockResolvedValue({
        data: mockExport
      });

      const exportRequest = {
        benchmark_id: 'exec1',
        format: 'csv' as const
      };

      const result = await benchmarkingApiService.exportResults(exportRequest);

      expect(mockApiService.post).toHaveBeenCalledWith('/benchmarking/export', exportRequest);
      expect(result.download_url).toBe('http://example.com/download');
      expect(result.format).toBe('csv');
    });

    it('should share results', async () => {
      const mockShare = {
        share_id: 'share1',
        expires_at: '2023-01-01T01:00:00Z'
      };

      mockApiService.post.mockResolvedValue({
        data: mockShare
      });

      const result = await benchmarkingApiService.shareResults('exec1', ['user1', 'user2'], 'Check out these results!');

      expect(mockApiService.post).toHaveBeenCalledWith('/benchmarking/share/exec1', {
        recipients: ['user1', 'user2'],
        message: 'Check out these results!'
      });
      expect(result.share_id).toBe('share1');
    });

    it('should get shared results', async () => {
      const mockResults = {
        execution_id: 'exec1',
        benchmark_config_id: '1',
        start_time: '2023-01-01T00:00:00Z',
        end_time: '2023-01-01T00:05:00Z',
        status: 'completed',
        total_iterations: 10,
        completed_iterations: 10,
        scenario_results: []
      };

      mockApiService.get.mockResolvedValue({
        data: mockResults
      });

      const result = await benchmarkingApiService.getSharedResults('share1');

      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/share/share1');
      expect(result.status).toBe('completed');
    });
  });

  describe('Templates and Configuration Management', () => {
    it('should get benchmark templates', async () => {
      const mockTemplates = [
        { id: '1', name: 'Basic Template', description: 'A basic benchmark template', category: 'basic' },
        { id: '2', name: 'Advanced Template', description: 'An advanced benchmark template', category: 'advanced' }
      ];

      mockApiService.get.mockResolvedValue({
        data: mockTemplates
      });

      const result = await benchmarkingApiService.getBenchmarkTemplates();

      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/templates');
      expect(result).toHaveLength(2);
      expect(result[0].name).toBe('Basic Template');
    });

    it('should get scenario templates', async () => {
      const mockTemplates = [
        { id: '1', name: 'Market Competition', description: 'Competitive market scenario', category: 'market' },
        { id: '2', name: 'Supply Chain', description: 'Supply chain scenario', category: 'operations' }
      ];

      mockApiService.get.mockResolvedValue({
        data: mockTemplates
      });

      const result = await benchmarkingApiService.getScenarioTemplates();

      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/scenario-templates');
      expect(result).toHaveLength(2);
      expect(result[0].name).toBe('Market Competition');
    });

    it('should apply a benchmark template', async () => {
      const mockConfig = {
        id: '1',
        name: 'Applied Template',
        description: 'A template-based configuration',
        scenarios: ['scenario1'],
        agents: ['agent1'],
        metrics: ['performance'],
        duration: 300,
        iterations: 10,
        created_at: '2023-01-01T00:00:00Z',
        updated_at: '2023-01-01T00:00:00Z',
        is_active: true
      };

      mockApiService.post.mockResolvedValue({
        data: mockConfig
      });

      const result = await benchmarkingApiService.applyBenchmarkTemplate('1', { custom_param: 'value' });

      expect(mockApiService.post).toHaveBeenCalledWith('/benchmarking/templates/1/apply', {
        customizations: { custom_param: 'value' }
      });
      expect(result.name).toBe('Applied Template');
    });

    it('should import a benchmark configuration', async () => {
      const mockConfigData = {
        name: 'Imported Configuration',
        scenarios: ['scenario1'],
        agents: ['agent1'],
        metrics: ['performance'],
        duration: 300,
        iterations: 10
      };

      const mockConfig = {
        ...mockConfigData,
        id: '1',
        created_at: '2023-01-01T00:00:00Z',
        updated_at: '2023-01-01T00:00:00Z',
        is_active: true
      };

      mockApiService.post.mockResolvedValue({
        data: mockConfig
      });

      const result = await benchmarkingApiService.importBenchmarkConfig(mockConfigData);

      expect(mockApiService.post).toHaveBeenCalledWith('/benchmarking/import', mockConfigData);
      expect(result.name).toBe('Imported Configuration');
    });

    it('should export a benchmark configuration', async () => {
      const mockConfig = {
        name: 'Exported Configuration',
        scenarios: ['scenario1'],
        agents: ['agent1'],
        metrics: ['performance'],
        duration: 300,
        iterations: 10
      };

      mockApiService.get.mockResolvedValue({
        data: JSON.stringify(mockConfig)
      });

      const result = await benchmarkingApiService.exportBenchmarkConfig('1', 'json');

      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/configs/1/export', {
        params: { format: 'json' }
      });
      expect(JSON.parse(result)).toEqual(mockConfig);
    });
  });

  describe('Error Handling', () => {
    it('should handle API errors gracefully', async () => {
      const apiError = new Error('API Error');
      mockApiService.get.mockRejectedValue(apiError);

      await expect(benchmarkingApiService.getBenchmarkConfig('1')).rejects.toThrow('API Error');
    });

    it('should handle network errors gracefully', async () => {
      const networkError = new Error('Network Error');
      mockApiService.get.mockRejectedValue(networkError);

      await expect(benchmarkingApiService.listBenchmarkConfigs()).rejects.toThrow('Network Error');
    });

    it('should handle validation errors gracefully', async () => {
      const validationError = {
        name: 'ValidationError',
        message: 'Invalid configuration',
        details: { field: 'name', error: 'Name is required' }
      };

      mockApiService.post.mockRejectedValue(validationError);

      await expect(benchmarkingApiService.createBenchmarkConfig({
        name: '',
        scenarios: ['scenario1'],
        agents: ['agent1'],
        metrics: ['performance'],
        duration: 300,
        iterations: 10
      })).rejects.toThrow('Invalid configuration');
    });
  });

  describe('WebSocket Integration', () => {
    it('should subscribe to execution updates', () => {
      const mockCallback = jest.fn();
      
      benchmarkingApiService.subscribeToExecutionUpdates('exec1', mockCallback);

      expect(webSocketService.subscribe).toHaveBeenCalledWith('exec1', {
        onExecutionUpdate: expect.any(Function),
        onMetricsUpdate: expect.any(Function),
        onBenchmarkComplete: expect.any(Function)
      });
    });

    it('should unsubscribe from execution updates', () => {
      benchmarkingApiService.unsubscribeFromExecutionUpdates('exec1');

      expect(webSocketService.unsubscribe).toHaveBeenCalledWith('exec1');
    });

    it('should handle WebSocket connection status', () => {
      const mockStatus = 'connected';
      (webSocketService.getConnectionStatus as jest.Mock).mockReturnValue(mockStatus);

      const status = benchmarkingApiService.getWebSocketStatus();

      expect(status).toBe('connected');
    });
  });
});