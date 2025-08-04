import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
// Note: These components will be imported after they are created
// For now, we'll use dynamic imports or create mock components
import { benchmarkingApiService } from '../../services/benchmarkingApiService';
import { webSocketService } from '../../services/webSocketService';

// Mock the API services
vi.mock('../../services/benchmarkingApiService');
vi.mock('../../services/webSocketService');

// Mock data for testing
const mockBenchmarkConfigs = [
  {
    id: '1',
    name: 'Basic Benchmark',
    description: 'A basic benchmark configuration',
    scenarios: ['scenario1', 'scenario2'],
    agents: ['agent1', 'agent2'],
    metrics: ['performance', 'accuracy'],
    duration: 300,
    iterations: 10,
    created_at: '2023-01-01T00:00:00Z',
    updated_at: '2023-01-01T00:00:00Z',
    is_active: true
  }
];

const mockScenarios = [
  {
    id: '1',
    name: 'Market Competition',
    description: 'Competitive market scenario',
    type: 'dynamic' as const,
    parameters: { difficulty: 'medium' },
    difficulty: 'medium' as const,
    created_at: '2023-01-01T00:00:00Z',
    updated_at: '2023-01-01T00:00:00Z',
    is_active: true
  }
];

const mockExecutionProgress = {
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

const mockBenchmarkResults = {
  execution_id: 'exec1',
  benchmark_config_id: '1',
  start_time: '2023-01-01T00:00:00Z',
  end_time: '2023-01-01T00:05:00Z',
  status: 'completed',
  total_iterations: 10,
  completed_iterations: 10,
  scenario_results: [
    {
      scenario_name: 'scenario1',
      agent_results: [
        {
          agent_id: 'agent1',
          metrics: [
            { name: 'performance', value: 0.85 },
            { name: 'accuracy', value: 0.92 }
          ]
        },
        {
          agent_id: 'agent2',
          metrics: [
            { name: 'performance', value: 0.78 },
            { name: 'accuracy', value: 0.88 }
          ]
        }
      ]
    }
  ]
};

describe('BenchmarkDashboard Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (benchmarkingApiService.listBenchmarkConfigs as vi.Mock).mockResolvedValue(mockBenchmarkConfigs);
    (benchmarkingApiService.listScenarios as vi.Mock).mockResolvedValue(mockScenarios);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders dashboard with title and navigation', () => {
    render(<BenchmarkDashboard />);
    
    expect(screen.getByText('FBA-Bench Benchmark Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Benchmark Configuration')).toBeInTheDocument();
    expect(screen.getByText('Scenario Management')).toBeInTheDocument();
    expect(screen.getByText('Execution Monitor')).toBeInTheDocument();
    expect(screen.getByText('Results Analysis')).toBeInTheDocument();
  });

  it('displays benchmark configurations', async () => {
    render(<BenchmarkDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('Basic Benchmark')).toBeInTheDocument();
      expect(screen.getByText('A basic benchmark configuration')).toBeInTheDocument();
    });
  });

  it('displays scenarios', async () => {
    render(<BenchmarkDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('Market Competition')).toBeInTheDocument();
      expect(screen.getByText('Competitive market scenario')).toBeInTheDocument();
    });
  });

  it('handles configuration creation', async () => {
    (benchmarkingApiService.createBenchmarkConfig as vi.Mock).mockResolvedValue({
      id: '2',
      name: 'New Benchmark',
      description: 'A new benchmark',
      scenarios: ['scenario1'],
      agents: ['agent1'],
      metrics: ['performance'],
      duration: 300,
      iterations: 5,
      created_at: '2023-01-01T00:00:00Z',
      updated_at: '2023-01-01T00:00:00Z',
      is_active: true
    });

    render(<BenchmarkDashboard />);
    
    // Find and click the create configuration button
    const createButton = screen.getByText('Create Configuration');
    fireEvent.click(createButton);
    
    // Fill out the form (simplified for testing)
    const nameInput = screen.getByLabelText('Configuration Name');
    fireEvent.change(nameInput, { target: { value: 'New Benchmark' } });
    
    const submitButton = screen.getByText('Create Configuration');
    fireEvent.click(submitButton);
    
    await waitFor(() => {
      expect(benchmarkingApiService.createBenchmarkConfig).toHaveBeenCalledWith({
        name: 'New Benchmark',
        scenarios: ['scenario1'],
        agents: ['agent1'],
        metrics: ['performance'],
        duration: 300,
        iterations: 5
      });
    });
  });
});

describe('MetricsVisualization Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders metrics visualization with title', () => {
    const mockMetrics = [
      {
        agent_id: 'agent1',
        scenario_name: 'scenario1',
        values: [
          { name: 'performance', value: 0.85 },
          { name: 'accuracy', value: 0.92 }
        ]
      }
    ];

    render(<MetricsVisualization data={mockMetrics} />);
    
    expect(screen.getByText('Metrics Visualization')).toBeInTheDocument();
    expect(screen.getByText('performance')).toBeInTheDocument();
    expect(screen.getByText('accuracy')).toBeInTheDocument();
  });

  it('handles empty data state', () => {
    render(<MetricsVisualization data={[]} />);
    
    expect(screen.getByText('No metrics data available')).toBeInTheDocument();
  });

  it('handles metric selection', async () => {
    const mockMetrics = [
      {
        agent_id: 'agent1',
        scenario_name: 'scenario1',
        values: [
          { name: 'performance', value: 0.85 },
          { name: 'accuracy', value: 0.92 }
        ]
      }
    ];

    render(<MetricsVisualization data={mockMetrics} />);
    
    const performanceButton = screen.getByText('performance');
    fireEvent.click(performanceButton);
    
    await waitFor(() => {
      expect(screen.getByText('Performance Metrics')).toBeInTheDocument();
    });
  });
});

describe('ScenarioBuilder Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (benchmarkingApiService.listScenarios as vi.Mock).mockResolvedValue(mockScenarios);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders scenario builder with title', () => {
    render(<ScenarioBuilder />);
    
    expect(screen.getByText('Scenario Builder')).toBeInTheDocument();
    expect(screen.getByText('Create New Scenario')).toBeInTheDocument();
  });

  it('displays existing scenarios', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      expect(screen.getByText('Market Competition')).toBeInTheDocument();
      expect(screen.getByText('Competitive market scenario')).toBeInTheDocument();
    });
  });

  it('handles scenario creation', async () => {
    (benchmarkingApiService.createScenario as vi.Mock).mockResolvedValue({
      id: '2',
      name: 'New Scenario',
      description: 'A new scenario',
      type: 'static' as const,
      parameters: { difficulty: 'easy' },
      difficulty: 'easy' as const,
      created_at: '2023-01-01T00:00:00Z',
      updated_at: '2023-01-01T00:00:00Z',
      is_active: true
    });

    render(<ScenarioBuilder />);
    
    const createButton = screen.getByText('Create Scenario');
    fireEvent.click(createButton);
    
    const nameInput = screen.getByLabelText('Scenario Name');
    fireEvent.change(nameInput, { target: { value: 'New Scenario' } });
    
    const typeSelect = screen.getByLabelText('Scenario Type');
    fireEvent.change(typeSelect, { target: { value: 'static' } });
    
    const submitButton = screen.getByText('Create Scenario');
    fireEvent.click(submitButton);
    
    await waitFor(() => {
      expect(benchmarkingApiService.createScenario).toHaveBeenCalledWith({
        name: 'New Scenario',
        type: 'static',
        parameters: { difficulty: 'easy' },
        difficulty: 'easy'
      });
    });
  });
});

describe('ExecutionMonitor Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (benchmarkingApiService.listExecutions as vi.Mock).mockResolvedValue([
      {
        id: 'exec1',
        config_id: '1',
        status: 'running',
        started_at: '2023-01-01T00:00:00Z'
      }
    ]);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders execution monitor with title', () => {
    render(<ExecutionMonitor />);
    
    expect(screen.getByText('Execution Monitor')).toBeInTheDocument();
    expect(screen.getByText('Active Executions')).toBeInTheDocument();
  });

  it('displays active executions', async () => {
    render(<ExecutionMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('exec1')).toBeInTheDocument();
      expect(screen.getByText('running')).toBeInTheDocument();
    });
  });

  it('handles execution start', async () => {
    (benchmarkingApiService.startBenchmark as vi.Mock).mockResolvedValue({
      execution_id: 'exec2',
      status: 'starting'
    });

    render(<ExecutionMonitor />);
    
    const startButton = screen.getByText('Start Execution');
    fireEvent.click(startButton);
    
    await waitFor(() => {
      expect(benchmarkingApiService.startBenchmark).toHaveBeenCalledWith('1');
    });
  });

  it('handles execution stop', async () => {
    (benchmarkingApiService.stopBenchmark as vi.Mock).mockResolvedValue(undefined);

    render(<ExecutionMonitor />);
    
    const stopButton = screen.getByText('Stop Execution');
    fireEvent.click(stopButton);
    
    await waitFor(() => {
      expect(benchmarkingApiService.stopBenchmark).toHaveBeenCalledWith('exec1');
    });
  });

  it('subscribes to WebSocket updates', () => {
    render(<ExecutionMonitor />);
    
    expect(webSocketService.subscribe).toHaveBeenCalledWith(
      'exec1',
      expect.objectContaining({
        onExecutionUpdate: expect.any(Function),
        onMetricsUpdate: expect.any(Function),
        onBenchmarkComplete: expect.any(Function)
      })
    );
  });
});

describe('ResultsComparison Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (benchmarkingApiService.listExecutions as vi.Mock).mockResolvedValue([
      {
        id: 'exec1',
        config_id: '1',
        status: 'completed',
        started_at: '2023-01-01T00:00:00Z',
        completed_at: '2023-01-01T00:05:00Z'
      }
    ]);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('renders results comparison with title', () => {
    render(<ResultsComparison />);
    
    expect(screen.getByText('Results Comparison')).toBeInTheDocument();
    expect(screen.getByText('Completed Executions')).toBeInTheDocument();
  });

  it('displays completed executions', async () => {
    render(<ResultsComparison />);
    
    await waitFor(() => {
      expect(screen.getByText('exec1')).toBeInTheDocument();
      expect(screen.getByText('completed')).toBeInTheDocument();
    });
  });

  it('handles execution selection', async () => {
    (benchmarkingApiService.getExecutionResults as vi.Mock).mockResolvedValue(mockBenchmarkResults);

    render(<ResultsComparison />);
    
    await waitFor(() => {
      const executionItem = screen.getByText('exec1');
      fireEvent.click(executionItem);
    });
    
    await waitFor(() => {
      expect(benchmarkingApiService.getExecutionResults).toHaveBeenCalledWith('exec1');
      expect(screen.getByText('Agent Performance Comparison')).toBeInTheDocument();
    });
  });

  it('handles results export', async () => {
    (benchmarkingApiService.exportResults as vi.Mock).mockResolvedValue({
      download_url: 'http://example.com/download',
      expires_at: '2023-01-01T01:00:00Z',
      file_size: 1024,
      format: 'csv'
    });

    render(<ResultsComparison />);
    
    await waitFor(() => {
      const executionItem = screen.getByText('exec1');
      fireEvent.click(executionItem);
    });
    
    await waitFor(() => {
      const exportButton = screen.getByText('Export Results');
      fireEvent.click(exportButton);
    });
    
    await waitFor(() => {
      expect(benchmarkingApiService.exportResults).toHaveBeenCalledWith({
        benchmark_id: 'exec1',
        format: 'csv'
      });
    });
  });

  it('handles results sharing', async () => {
    (benchmarkingApiService.shareResults as vi.Mock).mockResolvedValue({
      share_id: 'share1',
      expires_at: '2023-01-01T01:00:00Z'
    });

    render(<ResultsComparison />);
    
    await waitFor(() => {
      const executionItem = screen.getByText('exec1');
      fireEvent.click(executionItem);
    });
    
    await waitFor(() => {
      const shareButton = screen.getByText('Share Results');
      fireEvent.click(shareButton);
    });
    
    await waitFor(() => {
      expect(benchmarkingApiService.shareResults).toHaveBeenCalledWith('exec1', [], undefined);
    });
  });
});

describe('Integration Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('handles WebSocket connection errors gracefully', async () => {
    (webSocketService.getConnectionStatus as vi.Mock).mockReturnValue('disconnected');
    
    render(<ExecutionMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('WebSocket Disconnected')).toBeInTheDocument();
    });
  });

  it('handles API errors gracefully', async () => {
    (benchmarkingApiService.listBenchmarkConfigs as vi.Mock).mockRejectedValue(new Error('API Error'));
    
    render(<BenchmarkDashboard />);
    
    await waitFor(() => {
      expect(screen.getByText('Failed to load benchmark configurations')).toBeInTheDocument();
    });
  });

  it('handles loading states', async () => {
    (benchmarkingApiService.listBenchmarkConfigs as vi.Mock).mockImplementation(() => 
      new Promise(resolve => setTimeout(resolve, 1000))
    );
    
    render(<BenchmarkDashboard />);
    
    expect(screen.getByText('Loading configurations...')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.queryByText('Loading configurations...')).not.toBeInTheDocument();
    });
  });
});