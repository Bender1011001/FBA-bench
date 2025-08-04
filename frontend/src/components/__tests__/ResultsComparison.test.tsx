import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { apiService } from '../../services/apiService';
import { notificationService } from '../../utils/notificationService';
import ResultsComparison from '../ResultsComparison';

// Mock the dependencies
jest.mock('../../services/apiService');
jest.mock('../../utils/notificationService');
jest.mock('recharts', () => ({
  BarChart: ({ children, ...props }: any) => <div data-testid="bar-chart" {...props}>{children}</div>,
  Bar: (props: any) => <div data-testid="bar" {...props} />,
  LineChart: ({ children, ...props }: any) => <div data-testid="line-chart" {...props}>{children}</div>,
  Line: (props: any) => <div data-testid="line" {...props} />,
  XAxis: (props: any) => <div data-testid="x-axis" {...props} />,
  YAxis: (props: any) => <div data-testid="y-axis" {...props} />,
  CartesianGrid: (props: any) => <div data-testid="cartesian-grid" {...props} />,
  Tooltip: (props: any) => <div data-testid="tooltip" {...props} />,
  Legend: (props: any) => <div data-testid="legend" {...props} />,
  ResponsiveContainer: ({ children, ...props }: any) => <div data-testid="responsive-container" {...props}>{children}</div>,
  ScatterChart: ({ children, ...props }: any) => <div data-testid="scatter-chart" {...props}>{children}</div>,
  Scatter: (props: any) => <div data-testid="scatter" {...props} />,
  RadarChart: ({ children, ...props }: any) => <div data-testid="radar-chart" {...props}>{children}</div>,
  PolarGrid: (props: any) => <div data-testid="polar-grid" {...props} />,
  PolarAngleAxis: (props: any) => <div data-testid="polar-angle-axis" {...props} />,
  PolarRadiusAxis: (props: any) => <div data-testid="polar-radius-axis" {...props} />,
  Radar: (props: any) => <div data-testid="radar" {...props} />,
  ComposedChart: ({ children, ...props }: any) => <div data-testid="composed-chart" {...props}>{children}</div>,
  Area: (props: any) => <div data-testid="area" {...props} />,
  AreaChart: ({ children, ...props }: any) => <div data-testid="area-chart" {...props}>{children}</div>
}));

const mockApiService = apiService as jest.Mocked<typeof apiService>;
const mockNotificationService = notificationService as jest.Mocked<typeof notificationService>;

describe('ResultsComparison Component', () => {
  const mockBenchmarkResults = [
    {
      benchmark_name: 'test-benchmark-1',
      status: 'completed',
      start_time: '2023-01-01T00:00:00Z',
      end_time: '2023-01-01T01:00:00Z',
      duration_seconds: 3600,
      scenario_results: [
        {
          scenario_name: 'test-scenario-1',
          agent_results: [
            {
              agent_id: 'test-agent-1',
              success: true,
              metrics: [
                { name: 'overall_score', value: 0.85, unit: 'score' },
                { name: 'cognitive_score', value: 0.9, unit: 'score' }
              ]
            }
          ]
        }
      ]
    },
    {
      benchmark_name: 'test-benchmark-2',
      status: 'completed',
      start_time: '2023-01-02T00:00:00Z',
      end_time: '2023-01-02T01:00:00Z',
      duration_seconds: 3600,
      scenario_results: [
        {
          scenario_name: 'test-scenario-1',
          agent_results: [
            {
              agent_id: 'test-agent-1',
              success: true,
              metrics: [
                { name: 'overall_score', value: 0.75, unit: 'score' },
                { name: 'cognitive_score', value: 0.8, unit: 'score' }
              ]
            }
          ]
        }
      ]
    }
  ];

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Setup default mock implementations
    mockApiService.get.mockResolvedValue({
      data: mockBenchmarkResults[0]
    });
    
    mockApiService.post.mockResolvedValue({
      data: {}
    });
  });

  test('renders without crashing', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    expect(screen.getByText('Results Comparison')).toBeInTheDocument();
  });

  test('displays benchmark results for selection', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    expect(screen.getByText('test-benchmark-1')).toBeInTheDocument();
    expect(screen.getByText('test-benchmark-2')).toBeInTheDocument();
  });

  test('auto-selects first two results for comparison', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    const checkboxes = screen.getAllByRole('checkbox');
    expect(checkboxes[0]).toBeChecked();
    expect(checkboxes[1]).toBeChecked();
    expect(checkboxes[2]).not.toBeChecked();
  });

  test('allows toggling result selection', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    const checkboxes = screen.getAllByRole('checkbox');
    
    // Unselect first result
    fireEvent.click(checkboxes[0]);
    expect(checkboxes[0]).not.toBeChecked();
    
    // Select third result (if there was one)
    if (checkboxes.length > 2) {
      fireEvent.click(checkboxes[2]);
      expect(checkboxes[2]).toBeChecked();
    }
  });

  test('displays comparison controls when results are selected', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    expect(screen.getByText('Comparison Type')).toBeInTheDocument();
    expect(screen.getByText('Chart Type')).toBeInTheDocument();
  });

  test('changes comparison type', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    const comparisonTypeSelect = screen.getByLabelText('Comparison Type');
    expect(comparisonTypeSelect).toHaveValue('overview');
    
    fireEvent.change(comparisonTypeSelect, { target: { value: 'agents' } });
    expect(comparisonTypeSelect).toHaveValue('agents');
  });

  test('changes metric selection when comparison type requires it', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    // Change to agents comparison type
    const comparisonTypeSelect = screen.getByLabelText('Comparison Type');
    fireEvent.change(comparisonTypeSelect, { target: { value: 'agents' } });
    
    // Metric selection should now be visible
    const metricSelect = screen.getByLabelText('Metric');
    expect(metricSelect).toBeInTheDocument();
    expect(metricSelect).toHaveValue('overall_score');
    
    fireEvent.change(metricSelect, { target: { value: 'cognitive_score' } });
    expect(metricSelect).toHaveValue('cognitive_score');
  });

  test('changes chart type', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    const chartTypeSelect = screen.getByLabelText('Chart Type');
    expect(chartTypeSelect).toHaveValue('bar');
    
    fireEvent.change(chartTypeSelect, { target: { value: 'line' } });
    expect(chartTypeSelect).toHaveValue('line');
  });

  test('displays bar chart by default', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
  });

  test('displays line chart when selected', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    const chartTypeSelect = screen.getByLabelText('Chart Type');
    fireEvent.change(chartTypeSelect, { target: { value: 'line' } });
    
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
  });

  test('displays radar chart when selected', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    const chartTypeSelect = screen.getByLabelText('Chart Type');
    fireEvent.change(chartTypeSelect, { target: { value: 'radar' } });
    
    expect(screen.getByTestId('radar-chart')).toBeInTheDocument();
  });

  test('disables scatter plot when not exactly 2 results selected', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    // With 2 results selected, scatter plot should be enabled
    const chartTypeSelect = screen.getByLabelText('Chart Type');
    expect(screen.queryByText('Select exactly 2 results for scatter plot')).not.toBeInTheDocument();
    
    // Unselect one result
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[0]);
    
    // Now scatter plot should show a message
    fireEvent.change(chartTypeSelect, { target: { value: 'scatter' } });
    expect(screen.getByText('Select exactly 2 results for scatter plot')).toBeInTheDocument();
  });

  test('exports results as JSON', async () => {
    const mockBlob = new Blob(['{"test": "data"}'], { type: 'application/json' });
    global.URL.createObjectURL = jest.fn(() => 'blob:test-url');
    global.URL.revokeObjectURL = jest.fn();
    
    // Mock document.createElement to capture the download link
    const mockLink = {
      href: '',
      download: '',
      click: jest.fn()
    };
    document.createElement = jest.fn().mockReturnValue(mockLink);
    
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    fireEvent.click(screen.getByText('Export JSON'));
    
    await waitFor(() => {
      expect(mockApiService.post).toHaveBeenCalledWith(
        '/benchmarking/results/export',
        {
          resultIds: ['test-benchmark-1', 'test-benchmark-2'],
          options: {
            format: 'json',
            include_metadata: true,
            include_raw_data: true,
            include_charts: false,
            include_analysis: true
          }
        },
        { headers: { 'Accept': 'application/json' } }
      );
    });
    
    expect(mockNotificationService.success).toHaveBeenCalledWith(
      'Results exported as JSON',
      3000
    );
  });

  test('exports results as CSV', async () => {
    const mockBlob = new Blob(['test,data'], { type: 'text/csv' });
    global.URL.createObjectURL = jest.fn(() => 'blob:test-url');
    global.URL.revokeObjectURL = jest.fn();
    
    // Mock document.createElement to capture the download link
    const mockLink = {
      href: '',
      download: '',
      click: jest.fn()
    };
    document.createElement = jest.fn().mockReturnValue(mockLink);
    
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    fireEvent.click(screen.getByText('Export CSV'));
    
    await waitFor(() => {
      expect(mockApiService.post).toHaveBeenCalledWith(
        '/benchmarking/results/export',
        {
          resultIds: ['test-benchmark-1', 'test-benchmark-2'],
          options: {
            format: 'csv',
            include_metadata: true,
            include_raw_data: true,
            include_charts: false,
            include_analysis: true
          }
        },
        { headers: { 'Accept': 'text/csv' } }
      );
    });
    
    expect(mockNotificationService.success).toHaveBeenCalledWith(
      'Results exported as CSV',
      3000
    );
  });

  test('exports results as PDF', async () => {
    const mockBlob = new Blob(['pdf content'], { type: 'application/pdf' });
    global.URL.createObjectURL = jest.fn(() => 'blob:test-url');
    global.URL.revokeObjectURL = jest.fn();
    
    // Mock document.createElement to capture the download link
    const mockLink = {
      href: '',
      download: '',
      click: jest.fn()
    };
    document.createElement = jest.fn().mockReturnValue(mockLink);
    
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    fireEvent.click(screen.getByText('Export PDF'));
    
    await waitFor(() => {
      expect(mockApiService.post).toHaveBeenCalledWith(
        '/benchmarking/results/export',
        {
          resultIds: ['test-benchmark-1', 'test-benchmark-2'],
          options: {
            format: 'pdf',
            include_metadata: true,
            include_raw_data: true,
            include_charts: false,
            include_analysis: true
          }
        },
        { headers: { 'Accept': 'application/pdf' } }
      );
    });
    
    expect(mockNotificationService.success).toHaveBeenCalledWith(
      'Results exported as PDF',
      3000
    );
  });

  test('displays error when no results are selected for export', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    // Unselect all results
    const checkboxes = screen.getAllByRole('checkbox');
    checkboxes.forEach(checkbox => {
      if (checkbox.checked) {
        fireEvent.click(checkbox);
      }
    });
    
    fireEvent.click(screen.getByText('Export JSON'));
    
    expect(mockNotificationService.error).toHaveBeenCalledWith(
      'No results selected for export',
      3000
    );
  });

  test('displays detailed results table', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    expect(screen.getByText('Detailed Results')).toBeInTheDocument();
    expect(screen.getByText('test-benchmark-1')).toBeInTheDocument();
    expect(screen.getByText('test-benchmark-2')).toBeInTheDocument();
  });

  test('calculates average score correctly', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    // The average score for test-benchmark-1 should be 0.85
    expect(screen.getByText('0.850')).toBeInTheDocument();
  });

  test('calculates success rate correctly', () => {
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    // Both benchmarks have 100% success rate
    expect(screen.getByText('100.0%')).toBeInTheDocument();
  });

  test('displays error message when API call fails', async () => {
    mockApiService.get.mockRejectedValue(new Error('API Error'));
    
    render(<ResultsComparison benchmarkResults={mockBenchmarkResults} />);
    
    // Wait for the error to be displayed
    await waitFor(() => {
      expect(screen.getByText('Failed to fetch detailed results')).toBeInTheDocument();
    });
  });
});