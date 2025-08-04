import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { apiService } from '../../services/apiService';
import { notificationService } from '../../utils/notificationService';
import BenchmarkRunner from '../BenchmarkRunner';

// Mock the dependencies
jest.mock('../../services/apiService');
jest.mock('../../utils/notificationService');

const mockApiService = apiService as jest.Mocked<typeof apiService>;
const mockNotificationService = notificationService as jest.Mocked<typeof notificationService>;

describe('BenchmarkRunner Component', () => {
  const mockConfigurations = [
    {
      id: 'config-1',
      benchmark_id: 'test-benchmark-1',
      name: 'Test Benchmark 1',
      description: 'A test benchmark configuration',
      version: '1.0.0',
      created_at: '2025-01-01T00:00:00Z',
      updated_at: '2025-01-01T00:00:00Z'
    },
    {
      id: 'config-2',
      benchmark_id: 'test-benchmark-2',
      name: 'Test Benchmark 2',
      description: 'Another test benchmark configuration',
      version: '1.0.0',
      created_at: '2025-01-01T00:00:00Z',
      updated_at: '2025-01-01T00:00:00Z'
    }
  ];

  const mockBenchmarkResults = [
    {
      benchmark_name: 'test-benchmark-1',
      status: 'completed',
      start_time: '2023-01-01T00:00:00Z',
      end_time: '2023-01-01T01:00:00Z',
      duration_seconds: 3600,
      scenario_results: []
    },
    {
      benchmark_name: 'test-benchmark-2',
      status: 'completed',
      start_time: '2023-01-02T00:00:00Z',
      end_time: '2023-01-02T01:00:00Z',
      duration_seconds: 3600,
      scenario_results: []
    }
  ];

  const mockRunningBenchmark = {
    benchmark_id: 'running-benchmark',
    status: 'running',
    start_time: '2023-01-01T00:00:00Z',
    end_time: null,
    duration_seconds: 0,
    scenario_results: []
  };

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Setup default mock implementations
    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/configurations') {
        return Promise.resolve({ data: mockConfigurations });
      }
      if (url === '/benchmarking/results') {
        return Promise.resolve({ data: mockBenchmarkResults });
      }
      if (url === '/benchmarking/results/running-benchmark') {
        return Promise.resolve({ data: mockRunningBenchmark });
      }
      return Promise.resolve({ data: {} });
    });
    
    mockApiService.post.mockResolvedValue({
      data: {
        benchmark_id: 'new-benchmark',
        status: 'running',
        start_time: '2023-01-01T00:00:00Z',
        end_time: null,
        duration_seconds: 0,
        scenario_results: []
      }
    });
    
    mockApiService.delete.mockResolvedValue({});
  });

  test('renders without crashing', () => {
    render(<BenchmarkRunner />);
    expect(screen.getByText('Benchmark Runner')).toBeInTheDocument();
  });

  test('fetches configurations on mount', async () => {
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/configurations');
    });
  });

  test('fetches benchmark results on mount', async () => {
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/results');
    });
  });

  test('displays configurations for selection', async () => {
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      expect(screen.getByText('Test Benchmark 1')).toBeInTheDocument();
      expect(screen.getByText('Test Benchmark 2')).toBeInTheDocument();
    });
  });

  test('displays benchmark results', async () => {
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      expect(screen.getByText('test-benchmark-1')).toBeInTheDocument();
      expect(screen.getByText('test-benchmark-2')).toBeInTheDocument();
    });
  });

  test('selects a configuration when clicked', async () => {
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      fireEvent.click(screen.getByText('Test Benchmark 1'));
    });
    
    expect(screen.getByText('Selected: Test Benchmark 1')).toBeInTheDocument();
  });

  test('runs benchmark with selected configuration', async () => {
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      // Select a configuration
      fireEvent.click(screen.getByText('Test Benchmark 1'));
      
      // Run benchmark
      fireEvent.click(screen.getByText('Run Benchmark'));
    });
    
    await waitFor(() => {
      expect(mockApiService.post).toHaveBeenCalledWith(
        '/benchmarking/run',
        {
          configuration_id: 'config-1',
          options: {}
        }
      );
      
      expect(mockNotificationService.success).toHaveBeenCalledWith(
        'Benchmark started successfully',
        3000
      );
    });
  });

  test('disables run button when no configuration is selected', async () => {
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      const runButton = screen.getByText('Run Benchmark');
      expect(runButton).toBeDisabled();
    });
  });

  test('enables run button when a configuration is selected', async () => {
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      // Select a configuration
      fireEvent.click(screen.getByText('Test Benchmark 1'));
      
      const runButton = screen.getByText('Run Benchmark');
      expect(runButton).not.toBeDisabled();
    });
  });

  test('disables run button when a benchmark is already running', async () => {
    // Mock a running benchmark
    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/configurations') {
        return Promise.resolve({ data: mockConfigurations });
      }
      if (url === '/benchmarking/results') {
        return Promise.resolve({ data: [mockRunningBenchmark] });
      }
      return Promise.resolve({ data: {} });
    });
    
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      // Select a configuration
      fireEvent.click(screen.getByText('Test Benchmark 1'));
      
      const runButton = screen.getByText('Run Benchmark');
      expect(runButton).toBeDisabled();
    });
  });

  test('stops a running benchmark', async () => {
    // Mock a running benchmark
    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/configurations') {
        return Promise.resolve({ data: mockConfigurations });
      }
      if (url === '/benchmarking/results') {
        return Promise.resolve({ data: [mockRunningBenchmark] });
      }
      return Promise.resolve({ data: {} });
    });
    
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      // Stop the running benchmark
      fireEvent.click(screen.getByText('Stop Benchmark'));
    });
    
    await waitFor(() => {
      expect(mockApiService.delete).toHaveBeenCalledWith(
        '/benchmarking/run/running-benchmark'
      );
      
      expect(mockNotificationService.success).toHaveBeenCalledWith(
        'Benchmark stopped successfully',
        3000
      );
    });
  });

  test('disables stop button when no benchmark is running', async () => {
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      const stopButton = screen.getByText('Stop Benchmark');
      expect(stopButton).toBeDisabled();
    });
  });

  test('enables stop button when a benchmark is running', async () => {
    // Mock a running benchmark
    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/configurations') {
        return Promise.resolve({ data: mockConfigurations });
      }
      if (url === '/benchmarking/results') {
        return Promise.resolve({ data: [mockRunningBenchmark] });
      }
      return Promise.resolve({ data: {} });
    });
    
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      const stopButton = screen.getByText('Stop Benchmark');
      expect(stopButton).not.toBeDisabled();
    });
  });

  test('displays benchmark status', async () => {
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      expect(screen.getByText('completed')).toBeInTheDocument();
    });
  });

  test('displays benchmark duration', async () => {
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      expect(screen.getByText('1h 0m')).toBeInTheDocument();
    });
  });

  test('refreshes benchmark results when refresh button is clicked', async () => {
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      // Click refresh button
      fireEvent.click(screen.getByText('Refresh'));
    });
    
    await waitFor(() => {
      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/results');
    });
  });

  test('displays error message when API call fails', async () => {
    mockApiService.get.mockRejectedValue(new Error('API Error'));
    
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      expect(screen.getByText('Failed to fetch configurations')).toBeInTheDocument();
    });
  });

  test('displays error message when benchmark run fails', async () => {
    mockApiService.post.mockRejectedValue(new Error('Run Error'));
    
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      // Select a configuration
      fireEvent.click(screen.getByText('Test Benchmark 1'));
      
      // Run benchmark
      fireEvent.click(screen.getByText('Run Benchmark'));
    });
    
    await waitFor(() => {
      expect(screen.getByText('Failed to start benchmark')).toBeInTheDocument();
    });
  });

  test('displays error message when benchmark stop fails', async () => {
    // Mock a running benchmark
    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/configurations') {
        return Promise.resolve({ data: mockConfigurations });
      }
      if (url === '/benchmarking/results') {
        return Promise.resolve({ data: [mockRunningBenchmark] });
      }
      return Promise.resolve({ data: {} });
    });
    
    mockApiService.delete.mockRejectedValue(new Error('Stop Error'));
    
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      // Stop the running benchmark
      fireEvent.click(screen.getByText('Stop Benchmark'));
    });
    
    await waitFor(() => {
      expect(screen.getByText('Failed to stop benchmark')).toBeInTheDocument();
    });
  });

  test('displays loading state while fetching configurations', () => {
    // Mock a delayed response
    mockApiService.get.mockImplementation(() => {
      return new Promise(resolve => {
        setTimeout(() => {
          resolve({ data: mockConfigurations });
        }, 100);
      });
    });
    
    render(<BenchmarkRunner />);
    
    // Check that loading indicator is displayed
    expect(screen.getByText('Loading configurations...')).toBeInTheDocument();
  });

  test('displays empty state when no configurations are available', async () => {
    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/configurations') {
        return Promise.resolve({ data: [] });
      }
      if (url === '/benchmarking/results') {
        return Promise.resolve({ data: mockBenchmarkResults });
      }
      return Promise.resolve({ data: {} });
    });
    
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      expect(screen.getByText('No configurations available')).toBeInTheDocument();
    });
  });

  test('displays empty state when no benchmark results are available', async () => {
    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/configurations') {
        return Promise.resolve({ data: mockConfigurations });
      }
      if (url === '/benchmarking/results') {
        return Promise.resolve({ data: [] });
      }
      return Promise.resolve({ data: {} });
    });
    
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      expect(screen.getByText('No benchmark results available')).toBeInTheDocument();
    });
  });

  test('polls for benchmark status when a benchmark is running', async () => {
    // Mock a running benchmark
    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/configurations') {
        return Promise.resolve({ data: mockConfigurations });
      }
      if (url === '/benchmarking/results') {
        return Promise.resolve({ data: [mockRunningBenchmark] });
      }
      return Promise.resolve({ data: {} });
    });
    
    // Mock setTimeout
    jest.useFakeTimers();
    
    render(<BenchmarkRunner />);
    
    await waitFor(() => {
      // Fast-forward timers
      act(() => {
        jest.advanceTimersByTime(5000);
      });
      
      // Check that API was called again
      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/results');
    });
    
    // Clean up
    jest.useRealTimers();
  });
});