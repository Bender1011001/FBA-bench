import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { apiService } from '../../services/apiService';
import MetricsVisualization from '../MetricsVisualization';

// Mock the dependencies
jest.mock('../../services/apiService');
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

describe('MetricsVisualization Component', () => {
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
                { name: 'cognitive_score', value: 0.9, unit: 'score' },
                { name: 'business_score', value: 0.8, unit: 'score' },
                { name: 'technical_score', value: 0.75, unit: 'score' }
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
                { name: 'cognitive_score', value: 0.8, unit: 'score' },
                { name: 'business_score', value: 0.7, unit: 'score' },
                { name: 'technical_score', value: 0.65, unit: 'score' }
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
  });

  test('renders without crashing', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    expect(screen.getByText('Metrics Visualization')).toBeInTheDocument();
  });

  test('displays benchmark results for selection', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    expect(screen.getByText('test-benchmark-1')).toBeInTheDocument();
    expect(screen.getByText('test-benchmark-2')).toBeInTheDocument();
  });

  test('auto-selects first result if available', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    const checkboxes = screen.getAllByRole('checkbox');
    expect(checkboxes[0]).toBeChecked();
  });

  test('allows toggling result selection', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    const checkboxes = screen.getAllByRole('checkbox');
    
    // Unselect first result
    fireEvent.click(checkboxes[0]);
    expect(checkboxes[0]).not.toBeChecked();
    
    // Select second result
    fireEvent.click(checkboxes[1]);
    expect(checkboxes[1]).toBeChecked();
  });

  test('displays visualization controls when results are selected', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    expect(screen.getByText('Visualization Type')).toBeInTheDocument();
    expect(screen.getByText('Chart Type')).toBeInTheDocument();
  });

  test('changes visualization type', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    const visualizationTypeSelect = screen.getByLabelText('Visualization Type');
    expect(visualizationTypeSelect).toHaveValue('overview');
    
    fireEvent.change(visualizationTypeSelect, { target: { value: 'agents' } });
    expect(visualizationTypeSelect).toHaveValue('agents');
  });

  test('changes metric selection when visualization type requires it', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    // Change to agents visualization type
    const visualizationTypeSelect = screen.getByLabelText('Visualization Type');
    fireEvent.change(visualizationTypeSelect, { target: { value: 'agents' } });
    
    // Metric selection should now be visible
    const metricSelect = screen.getByLabelText('Metric');
    expect(metricSelect).toBeInTheDocument();
    expect(metricSelect).toHaveValue('overall_score');
    
    fireEvent.change(metricSelect, { target: { value: 'cognitive_score' } });
    expect(metricSelect).toHaveValue('cognitive_score');
  });

  test('changes chart type', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    const chartTypeSelect = screen.getByLabelText('Chart Type');
    expect(chartTypeSelect).toHaveValue('bar');
    
    fireEvent.change(chartTypeSelect, { target: { value: 'line' } });
    expect(chartTypeSelect).toHaveValue('line');
  });

  test('displays bar chart by default', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
  });

  test('displays line chart when selected', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    const chartTypeSelect = screen.getByLabelText('Chart Type');
    fireEvent.change(chartTypeSelect, { target: { value: 'line' } });
    
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
  });

  test('displays radar chart when selected', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    const chartTypeSelect = screen.getByLabelText('Chart Type');
    fireEvent.change(chartTypeSelect, { target: { value: 'radar' } });
    
    expect(screen.getByTestId('radar-chart')).toBeInTheDocument();
  });

  test('disables scatter plot when not exactly 2 results selected', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    // With 1 result selected, scatter plot should show a message
    const chartTypeSelect = screen.getByLabelText('Chart Type');
    fireEvent.change(chartTypeSelect, { target: { value: 'scatter' } });
    expect(screen.getByText('Select exactly 2 results for scatter plot')).toBeInTheDocument();
    
    // Select second result
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[1]);
    
    // Now scatter plot should be enabled
    expect(screen.queryByText('Select exactly 2 results for scatter plot')).not.toBeInTheDocument();
  });

  test('displays overview visualization by default', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    expect(screen.getByText('Overview Metrics')).toBeInTheDocument();
    expect(screen.getByText('Overall Score')).toBeInTheDocument();
    expect(screen.getByText('Cognitive Score')).toBeInTheDocument();
    expect(screen.getByText('Business Score')).toBeInTheDocument();
    expect(screen.getByText('Technical Score')).toBeInTheDocument();
  });

  test('displays agents visualization when selected', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    // Change to agents visualization type
    const visualizationTypeSelect = screen.getByLabelText('Visualization Type');
    fireEvent.change(visualizationTypeSelect, { target: { value: 'agents' } });
    
    expect(screen.getByText('Agent Performance')).toBeInTheDocument();
    expect(screen.getByText('test-agent-1')).toBeInTheDocument();
  });

  test('displays trends visualization when selected', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    // Change to trends visualization type
    const visualizationTypeSelect = screen.getByLabelText('Visualization Type');
    fireEvent.change(visualizationTypeSelect, { target: { value: 'trends' } });
    
    expect(screen.getByText('Performance Trends')).toBeInTheDocument();
  });

  test('displays comparison visualization when selected', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    // Change to comparison visualization type
    const visualizationTypeSelect = screen.getByLabelText('Visualization Type');
    fireEvent.change(visualizationTypeSelect, { target: { value: 'comparison' } });
    
    expect(screen.getByText('Benchmark Comparison')).toBeInTheDocument();
  });

  test('displays detailed metrics table', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    expect(screen.getByText('Detailed Metrics')).toBeInTheDocument();
    expect(screen.getByText('test-benchmark-1')).toBeInTheDocument();
    expect(screen.getByText('test-benchmark-2')).toBeInTheDocument();
  });

  test('calculates average score correctly', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    // The average score for test-benchmark-1 should be 0.85
    expect(screen.getByText('0.850')).toBeInTheDocument();
  });

  test('calculates success rate correctly', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    // Both benchmarks have 100% success rate
    expect(screen.getByText('100.0%')).toBeInTheDocument();
  });

  test('displays error message when no results are selected', () => {
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    // Unselect all results
    const checkboxes = screen.getAllByRole('checkbox');
    checkboxes.forEach(checkbox => {
      if (checkbox.checked) {
        fireEvent.click(checkbox);
      }
    });
    
    expect(screen.getByText('No benchmark results selected')).toBeInTheDocument();
  });

  test('displays error message when API call fails', async () => {
    mockApiService.get.mockRejectedValue(new Error('API Error'));
    
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    // Wait for the error to be displayed
    await waitFor(() => {
      expect(screen.getByText('Failed to fetch detailed results')).toBeInTheDocument();
    });
  });

  test('exports visualization as image', async () => {
    // Mock canvas and toBlob
    const mockCanvas = document.createElement('canvas');
    const mockToBlob = jest.fn((callback) => {
      callback(new Blob(['test'], { type: 'image/png' }));
    });
    mockCanvas.toBlob = mockToBlob;
    
    // Mock document.createElement to return our mock canvas
    const originalCreateElement = document.createElement;
    document.createElement = jest.fn((tagName) => {
      if (tagName === 'canvas') {
        return mockCanvas;
      }
      return originalCreateElement.call(document, tagName);
    });
    
    // Mock URL.createObjectURL and URL.revokeObjectURL
    global.URL.createObjectURL = jest.fn(() => 'blob:test-url');
    global.URL.revokeObjectURL = jest.fn();
    
    // Mock document.createElement to capture the download link
    const mockLink = {
      href: '',
      download: '',
      click: jest.fn()
    };
    document.createElement = jest.fn().mockReturnValue(mockLink);
    
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    fireEvent.click(screen.getByText('Export as Image'));
    
    await waitFor(() => {
      expect(mockLink.download).toBe('metrics-visualization.png');
      expect(mockLink.href).toBe('blob:test-url');
      expect(mockLink.click).toHaveBeenCalled();
    });
    
    // Restore original functions
    document.createElement = originalCreateElement;
  });

  test('exports visualization as data', async () => {
    // Mock document.createElement to capture the download link
    const mockLink = {
      href: '',
      download: '',
      click: jest.fn()
    };
    document.createElement = jest.fn().mockReturnValue(mockLink);
    
    render(<MetricsVisualization benchmarkResults={mockBenchmarkResults} />);
    
    fireEvent.click(screen.getByText('Export as Data'));
    
    await waitFor(() => {
      expect(mockLink.download).toBe('metrics-data.json');
      expect(mockLink.click).toHaveBeenCalled();
    });
  });
});