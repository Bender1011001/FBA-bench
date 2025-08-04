import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { apiService } from '../../services/apiService';
import { notificationService } from '../../utils/notificationService';
import ReportGenerator from '../ReportGenerator';

// Mock the dependencies
jest.mock('../../services/apiService');
jest.mock('../../utils/notificationService');

const mockApiService = apiService as jest.Mocked<typeof apiService>;
const mockNotificationService = notificationService as jest.Mocked<typeof notificationService>;

describe('ReportGenerator Component', () => {
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

  const mockReportTemplates = [
    {
      id: 'template-1',
      name: 'Executive Summary',
      category: 'business',
      description: 'High-level summary for executives'
    },
    {
      id: 'template-2',
      name: 'Technical Analysis',
      category: 'technical',
      description: 'Detailed technical analysis'
    }
  ];

  const mockGeneratedReports = [
    {
      id: 'report-1',
      benchmark_id: 'test-benchmark-1',
      title: 'Test Report 1',
      description: 'Test description 1',
      template_id: 'template-1',
      format: 'pdf',
      generated_at: '2023-01-01T00:00:00Z'
    },
    {
      id: 'report-2',
      benchmark_id: 'test-benchmark-2',
      title: 'Test Report 2',
      description: 'Test description 2',
      template_id: 'template-2',
      format: 'html',
      generated_at: '2023-01-02T00:00:00Z'
    }
  ];

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Setup default mock implementations
    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/report-templates') {
        return Promise.resolve({ data: mockReportTemplates });
      }
      if (url === '/benchmarking/reports') {
        return Promise.resolve({ data: mockGeneratedReports });
      }
      return Promise.resolve({ data: {} });
    });
    
    mockApiService.post.mockResolvedValue({
      data: {
        id: 'new-report-1',
        benchmark_id: 'test-benchmark-1',
        title: 'Test Report',
        description: 'Test description',
        template_id: 'template-1',
        format: 'pdf',
        generated_at: '2023-01-01T00:00:00Z'
      }
    });
    
    mockApiService.delete.mockResolvedValue({});
  });

  test('renders without crashing', () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    expect(screen.getByText('Report Generator')).toBeInTheDocument();
  });

  test('fetches report templates on mount', async () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/report-templates');
    });
  });

  test('fetches generated reports on mount', async () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/reports');
    });
  });

  test('displays benchmark results for selection', async () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      expect(screen.getByText('test-benchmark-1')).toBeInTheDocument();
      expect(screen.getByText('test-benchmark-2')).toBeInTheDocument();
    });
  });

  test('displays report templates for selection', async () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      expect(screen.getByText('Executive Summary')).toBeInTheDocument();
      expect(screen.getByText('Technical Analysis')).toBeInTheDocument();
    });
  });

  test('auto-selects first template if available', async () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      const templateSelect = screen.getByLabelText('Template *');
      expect(templateSelect).toHaveValue('template-1');
    });
  });

  test('updates report title when benchmark is selected', async () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      const benchmarkSelect = screen.getByText('Select a benchmark result');
      fireEvent.change(benchmarkSelect, { target: { value: 'test-benchmark-1' } });
      
      const titleInput = screen.getByLabelText('Report Title *');
      expect(titleInput).toHaveValue('Benchmark Report - test-benchmark-1');
    });
  });

  test('updates report description when benchmark is selected', async () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      const benchmarkSelect = screen.getByText('Select a benchmark result');
      fireEvent.change(benchmarkSelect, { target: { value: 'test-benchmark-1' } });
      
      const descriptionInput = screen.getByLabelText('Description');
      expect(descriptionInput).toHaveValue(
        'Comprehensive report for benchmark execution on 1/1/2023'
      );
    });
  });

  test('allows changing report title', async () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      const titleInput = screen.getByLabelText('Report Title *');
      fireEvent.change(titleInput, { target: { value: 'Custom Report Title' } });
      expect(titleInput).toHaveValue('Custom Report Title');
    });
  });

  test('allows changing report description', async () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      const descriptionInput = screen.getByLabelText('Description');
      fireEvent.change(descriptionInput, { target: { value: 'Custom description' } });
      expect(descriptionInput).toHaveValue('Custom description');
    });
  });

  test('allows changing report template', async () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      const templateSelect = screen.getByLabelText('Template *');
      fireEvent.change(templateSelect, { target: { value: 'template-2' } });
      expect(templateSelect).toHaveValue('template-2');
    });
  });

  test('allows changing report format', async () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      const formatSelect = screen.getByLabelText('Format');
      expect(formatSelect).toHaveValue('pdf');
      
      fireEvent.change(formatSelect, { target: { value: 'html' } });
      expect(formatSelect).toHaveValue('html');
    });
  });

  test('allows toggling report sections', async () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      const summaryCheckbox = screen.getByLabelText('summary');
      expect(summaryCheckbox).toBeChecked();
      
      fireEvent.click(summaryCheckbox);
      expect(summaryCheckbox).not.toBeChecked();
    });
  });

  test('validates configuration before generating report', async () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      // Try to generate without selecting a benchmark
      fireEvent.click(screen.getByText('Generate Report'));
      
      expect(screen.getByText('Please select a benchmark result')).toBeInTheDocument();
    });
  });

  test('generates report with valid configuration', async () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      // Select a benchmark
      const benchmarkSelect = screen.getByText('Select a benchmark result');
      fireEvent.change(benchmarkSelect, { target: { value: 'test-benchmark-1' } });
      
      // Set title
      const titleInput = screen.getByLabelText('Report Title *');
      fireEvent.change(titleInput, { target: { value: 'Test Report' } });
      
      // Generate report
      fireEvent.click(screen.getByText('Generate Report'));
    });
    
    await waitFor(() => {
      expect(mockApiService.post).toHaveBeenCalledWith('/benchmarking/reports', {
        benchmark_id: 'test-benchmark-1',
        title: 'Test Report',
        description: 'Comprehensive report for benchmark execution on 1/1/2023',
        template_id: 'template-1',
        format: 'pdf',
        sections: [
          {
            id: 'summary',
            title: 'Executive Summary',
            type: 'summary',
            content: {},
            order: 1,
            visible: true
          },
          {
            id: 'charts',
            title: 'Performance Charts',
            type: 'charts',
            content: {},
            order: 2,
            visible: true
          },
          {
            id: 'tables',
            title: 'Detailed Results',
            type: 'tables',
            content: {},
            order: 3,
            visible: true
          },
          {
            id: 'analysis',
            title: 'Analysis & Insights',
            type: 'analysis',
            content: {},
            order: 5,
            visible: true
          }
        ],
        metadata: {
          generated_by: 'user',
          generated_at: expect.any(String),
          benchmark_result: 'test-benchmark-1'
        }
      });
      
      expect(mockNotificationService.success).toHaveBeenCalledWith(
        'Report generated successfully',
        3000
      );
    });
  });

  test('displays generated reports in history tab', async () => {
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    // Switch to history tab
    fireEvent.click(screen.getByText('Report History'));
    
    await waitFor(() => {
      expect(screen.getByText('Test Report 1')).toBeInTheDocument();
      expect(screen.getByText('Test Report 2')).toBeInTheDocument();
    });
  });

  test('downloads report', async () => {
    const mockBlob = new Blob(['test content'], { type: 'application/pdf' });
    global.URL.createObjectURL = jest.fn(() => 'blob:test-url');
    global.URL.revokeObjectURL = jest.fn();
    
    // Mock document.createElement to capture the download link
    const mockLink = {
      href: '',
      download: '',
      click: jest.fn()
    };
    document.createElement = jest.fn().mockReturnValue(mockLink);
    
    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/reports/report-1/download') {
        return Promise.resolve({ data: mockBlob });
      }
      return Promise.resolve({ data: {} });
    });
    
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    // Switch to history tab
    fireEvent.click(screen.getByText('Report History'));
    
    await waitFor(() => {
      // Click download button
      fireEvent.click(screen.getAllByText('Download')[0]);
    });
    
    await waitFor(() => {
      expect(mockApiService.get).toHaveBeenCalledWith(
        '/benchmarking/reports/report-1/download',
        {
          headers: { 'Accept': 'application/pdf' }
        }
      );
      
      expect(mockNotificationService.success).toHaveBeenCalledWith(
        'Report downloaded as PDF',
        3000
      );
    });
  });

  test('deletes report', async () => {
    // Mock window.confirm
    window.confirm = jest.fn(() => true);
    
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    // Switch to history tab
    fireEvent.click(screen.getByText('Report History'));
    
    await waitFor(() => {
      // Click delete button
      fireEvent.click(screen.getAllByText('Delete')[0]);
    });
    
    await waitFor(() => {
      expect(mockApiService.delete).toHaveBeenCalledWith('/benchmarking/reports/report-1');
      expect(mockNotificationService.success).toHaveBeenCalledWith(
        'Report deleted successfully',
        3000
      );
    });
  });

  test('displays error message when API call fails', async () => {
    mockApiService.get.mockRejectedValue(new Error('API Error'));
    
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      expect(screen.getByText('Failed to fetch report templates')).toBeInTheDocument();
    });
  });

  test('displays error message when report generation fails', async () => {
    mockApiService.post.mockRejectedValue(new Error('Generation Error'));
    
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    await waitFor(() => {
      // Select a benchmark
      const benchmarkSelect = screen.getByText('Select a benchmark result');
      fireEvent.change(benchmarkSelect, { target: { value: 'test-benchmark-1' } });
      
      // Set title
      const titleInput = screen.getByLabelText('Report Title *');
      fireEvent.change(titleInput, { target: { value: 'Test Report' } });
      
      // Generate report
      fireEvent.click(screen.getByText('Generate Report'));
    });
    
    await waitFor(() => {
      expect(screen.getByText('Failed to generate report')).toBeInTheDocument();
    });
  });

  test('displays error message when report download fails', async () => {
    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/reports/report-1/download') {
        return Promise.reject(new Error('Download Error'));
      }
      return Promise.resolve({ data: {} });
    });
    
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    // Switch to history tab
    fireEvent.click(screen.getByText('Report History'));
    
    await waitFor(() => {
      // Click download button
      fireEvent.click(screen.getAllByText('Download')[0]);
    });
    
    await waitFor(() => {
      expect(screen.getByText('Failed to download report')).toBeInTheDocument();
    });
  });

  test('displays error message when report deletion fails', async () => {
    // Mock window.confirm
    window.confirm = jest.fn(() => true);
    
    mockApiService.delete.mockRejectedValue(new Error('Deletion Error'));
    
    render(<ReportGenerator benchmarkResults={mockBenchmarkResults} />);
    
    // Switch to history tab
    fireEvent.click(screen.getByText('Report History'));
    
    await waitFor(() => {
      // Click delete button
      fireEvent.click(screen.getAllByText('Delete')[0]);
    });
    
    await waitFor(() => {
      expect(screen.getByText('Failed to delete report')).toBeInTheDocument();
    });
  });
});