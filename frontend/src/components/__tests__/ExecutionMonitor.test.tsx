import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { apiService } from '../../services/apiService';
import { notificationService } from '../../utils/notificationService';
import ExecutionMonitor from '../ExecutionMonitor';

// Mock the dependencies
jest.mock('../../hooks/useWebSocket');
jest.mock('../../services/apiService');
jest.mock('../../utils/notificationService');

const mockUseWebSocket = useWebSocket as jest.MockedFunction<typeof useWebSocket>;
const mockApiService = apiService as jest.Mocked<typeof apiService>;
const mockNotificationService = notificationService as jest.Mocked<typeof notificationService>;

describe('ExecutionMonitor Component', () => {
  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Setup default mock implementations
    mockUseWebSocket.mockReturnValue({
      lastMessage: null,
      connectionStatus: 'connected',
      sendMessage: jest.fn(),
      connect: jest.fn(),
      disconnect: jest.fn(),
      readyState: WebSocket.OPEN
    });
    
    mockApiService.get.mockResolvedValue({
      data: []
    });
  });

  test('renders without crashing', () => {
    render(<ExecutionMonitor />);
    expect(screen.getByText('Execution Monitor')).toBeInTheDocument();
  });

  test('displays connection status', () => {
    mockUseWebSocket.mockReturnValue({
      lastMessage: null,
      connectionStatus: 'disconnected',
      sendMessage: jest.fn(),
      connect: jest.fn(),
      disconnect: jest.fn(),
      readyState: WebSocket.CLOSED
    });

    render(<ExecutionMonitor />);
    expect(screen.getByText('Disconnected from benchmarking server')).toBeInTheDocument();
  });

  test('fetches active benchmarks on mount', async () => {
    const mockActiveBenchmarks = [
      {
        benchmark_id: 'test-benchmark-1',
        status: 'running',
        progress: 50.0,
        start_time: '2023-01-01T00:00:00Z'
      }
    ];

    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/active') {
        return Promise.resolve({ data: mockActiveBenchmarks });
      }
      return Promise.resolve({ data: [] });
    });

    render(<ExecutionMonitor />);
    
    await waitFor(() => {
      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/active');
    });
  });

  test('fetches completed benchmarks on mount', async () => {
    const mockCompletedBenchmarks = [
      {
        benchmark_name: 'test-benchmark-1',
        status: 'completed',
        start_time: '2023-01-01T00:00:00Z',
        end_time: '2023-01-01T01:00:00Z',
        duration_seconds: 3600,
        scenario_results: []
      }
    ];

    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/completed') {
        return Promise.resolve({ data: mockCompletedBenchmarks });
      }
      return Promise.resolve({ data: [] });
    });

    render(<ExecutionMonitor />);
    
    await waitFor(() => {
      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/completed');
    });
  });

  test('handles WebSocket benchmark_started event', () => {
    const mockEvent = {
      type: 'benchmark_started',
      payload: {
        benchmark_id: 'test-benchmark-1'
      }
    };

    mockUseWebSocket.mockReturnValue({
      lastMessage: { data: JSON.stringify(mockEvent) } as MessageEvent,
      connectionStatus: 'connected',
      sendMessage: jest.fn(),
      connect: jest.fn(),
      disconnect: jest.fn(),
      readyState: WebSocket.OPEN
    });

    render(<ExecutionMonitor />);
    
    expect(mockNotificationService.info).toHaveBeenCalledWith(
      'Benchmark test-benchmark-1 started',
      3000
    );
  });

  test('handles WebSocket benchmark_completed event', () => {
    const mockEvent = {
      type: 'benchmark_completed',
      payload: {
        benchmark_name: 'test-benchmark-1',
        status: 'completed'
      }
    };

    mockUseWebSocket.mockReturnValue({
      lastMessage: { data: JSON.stringify(mockEvent) } as MessageEvent,
      connectionStatus: 'connected',
      sendMessage: jest.fn(),
      connect: jest.fn(),
      disconnect: jest.fn(),
      readyState: WebSocket.OPEN
    });

    render(<ExecutionMonitor />);
    
    expect(mockNotificationService.success).toHaveBeenCalledWith(
      'Benchmark test-benchmark-1 completed!',
      5000
    );
  });

  test('handles WebSocket benchmark_error event', () => {
    const mockEvent = {
      type: 'benchmark_error',
      payload: {
        error: 'Test error message'
      }
    };

    mockUseWebSocket.mockReturnValue({
      lastMessage: { data: JSON.stringify(mockEvent) } as MessageEvent,
      connectionStatus: 'connected',
      sendMessage: jest.fn(),
      connect: jest.fn(),
      disconnect: jest.fn(),
      readyState: WebSocket.OPEN
    });

    render(<ExecutionMonitor />);
    
    expect(mockNotificationService.error).toHaveBeenCalledWith(
      'Benchmark error: Test error message',
      5000
    );
    
    expect(screen.getByText('Test error message')).toBeInTheDocument();
  });

  test('stops benchmark when stop button is clicked', async () => {
    const mockActiveBenchmarks = [
      {
        benchmark_id: 'test-benchmark-1',
        status: 'running',
        progress: 50.0,
        start_time: '2023-01-01T00:00:00Z'
      }
    ];

    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/active') {
        return Promise.resolve({ data: mockActiveBenchmarks });
      }
      return Promise.resolve({ data: [] });
    });

    mockApiService.post.mockResolvedValue({});

    render(<ExecutionMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('Stop')).toBeInTheDocument();
    });
    
    fireEvent.click(screen.getByText('Stop'));
    
    expect(mockApiService.post).toHaveBeenCalledWith(
      '/benchmarking/test-benchmark-1/stop'
    );
    
    expect(mockNotificationService.info).toHaveBeenCalledWith(
      'Benchmark test-benchmark-1 stopping',
      3000
    );
  });

  test('toggles auto refresh', () => {
    render(<ExecutionMonitor />);
    
    const autoRefreshCheckbox = screen.getByLabelText('Auto Refresh');
    expect(autoRefreshCheckbox).toBeChecked();
    
    fireEvent.click(autoRefreshCheckbox);
    expect(autoRefreshCheckbox).not.toBeChecked();
  });

  test('changes refresh interval', () => {
    render(<ExecutionMonitor />);
    
    const intervalSelect = screen.getByLabelText('Refresh Interval');
    expect(intervalSelect).toHaveValue('5');
    
    fireEvent.change(intervalSelect, { target: { value: '10' } });
    expect(intervalSelect).toHaveValue('10');
  });

  test('displays agent results when benchmark is selected', async () => {
    const mockCompletedBenchmarks = [
      {
        benchmark_name: 'test-benchmark-1',
        status: 'completed',
        start_time: '2023-01-01T00:00:00Z',
        end_time: '2023-01-01T01:00:00Z',
        duration_seconds: 3600,
        scenario_results: []
      }
    ];

    const mockAgentResults = [
      {
        agent_id: 'test-agent-1',
        scenario_name: 'test-benchmark-1',
        run_number: 1,
        duration_seconds: 60,
        success: true,
        metrics: [
          { name: 'accuracy', value: 0.95, unit: '%' },
          { name: 'efficiency', value: 0.85, unit: 'score' }
        ]
      }
    ];

    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/completed') {
        return Promise.resolve({ data: mockCompletedBenchmarks });
      }
      if (url === '/benchmarking/test-benchmark-1/agent-results') {
        return Promise.resolve({ data: mockAgentResults });
      }
      return Promise.resolve({ data: [] });
    });

    render(<ExecutionMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('View Details')).toBeInTheDocument();
    });
    
    fireEvent.click(screen.getByText('View Details'));
    
    await waitFor(() => {
      expect(mockApiService.get).toHaveBeenCalledWith(
        '/benchmarking/test-benchmark-1/agent-results'
      );
    });
    
    expect(screen.getByText('test-agent-1')).toBeInTheDocument();
    expect(screen.getByText('accuracy: 0.950 %')).toBeInTheDocument();
    expect(screen.getByText('efficiency: 0.850 score')).toBeInTheDocument();
  });

  test('displays error message when API call fails', async () => {
    mockApiService.get.mockRejectedValue(new Error('API Error'));

    render(<ExecutionMonitor />);
    
    await waitFor(() => {
      expect(screen.getByText('Failed to fetch active benchmarks')).toBeInTheDocument();
    });
  });
});