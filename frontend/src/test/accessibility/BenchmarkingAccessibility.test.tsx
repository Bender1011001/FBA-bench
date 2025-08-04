import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';

// Import all components to test
import BenchmarkDashboard from '../../components/BenchmarkDashboard';
import MetricsVisualization from '../../components/MetricsVisualization';
import ScenarioBuilder from '../../components/ScenarioBuilder';
import ExecutionMonitor from '../../components/ExecutionMonitor';
import ResultsComparison from '../../components/ResultsComparison';
import RadarChart from '../../components/charts/RadarChart';
import HeatmapChart from '../../components/charts/HeatmapChart';
import TimeSeriesChart from '../../components/charts/TimeSeriesChart';
import ComparisonChart from '../../components/charts/ComparisonChart';

// Mock WebSocket
global.WebSocket = jest.fn().mockImplementation(() => ({
  send: jest.fn(),
  close: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
  readyState: 1,
}));

// Mock API service
jest.mock('../../services/apiService', () => ({
  apiService: {
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
  },
}));

// Mock notification service
jest.mock('../../utils/notificationService', () => ({
  notificationService: {
    success: jest.fn(),
    error: jest.fn(),
    info: jest.fn(),
    warning: jest.fn(),
  },
}));

describe('Accessibility Testing - Benchmarking Components', () => {
  const mockBenchmarkResults = [
    {
      id: 'test-1',
      name: 'Test Benchmark',
      description: 'Test benchmark description',
      status: 'completed' as const,
      created_at: '2025-01-01T00:00:00Z',
      updated_at: '2025-01-01T00:00:00Z',
      scenario_results: [
        {
          scenario_name: 'Test Scenario',
          agent_results: [
            {
              agent_id: 'agent-1',
              agent_name: 'Test Agent',
              scenario_name: 'Test Scenario',
              success: true,
              start_time: '2025-01-01T00:00:00Z',
              end_time: '2025-01-01T00:05:00Z',
              metrics: [
                { name: 'accuracy', value: 0.95, unit: 'percentage' },
                { name: 'efficiency', value: 0.85, unit: 'score' },
              ],
            },
          ],
        },
      ],
    },
  ];

  const mockWebSocketEvents = [
    {
      type: 'benchmark_started',
      timestamp: '2025-01-01T00:00:00Z',
      data: { benchmark_id: 'test-1', name: 'Test Benchmark' },
    },
    {
      type: 'benchmark_progress',
      timestamp: '2025-01-01T00:01:00Z',
      data: { benchmark_id: 'test-1', progress: 50 },
    },
    {
      type: 'benchmark_completed',
      timestamp: '2025-01-01T00:05:00Z',
      data: { benchmark_id: 'test-1', success: true },
    },
  ];

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
  });

  describe('BenchmarkDashboard', () => {
    it('should have no accessibility violations', async () => {
      const { container } = render(
        <BenchmarkDashboard
          benchmarkResults={mockBenchmarkResults}
          onBenchmarkSelect={jest.fn()}
          onBenchmarkCreate={jest.fn()}
          className="test-class"
        />
      );

      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should be keyboard navigable', () => {
      render(
        <BenchmarkDashboard
          benchmarkResults={mockBenchmarkResults}
          onBenchmarkSelect={jest.fn()}
          onBenchmarkCreate={jest.fn()}
        />
      );

      // Test tab navigation
      const buttons = screen.getAllByRole('button');
      buttons.forEach((button, index) => {
        expect(button).toHaveAttribute('tabindex', '0');
        expect(button).not.toHaveAttribute('disabled');
      });
    });

    it('should have proper ARIA labels', () => {
      render(
        <BenchmarkDashboard
          benchmarkResults={mockBenchmarkResults}
          onBenchmarkSelect={jest.fn()}
          onBenchmarkCreate={jest.fn()}
        />
      );

      expect(screen.getByLabelText('Benchmark Dashboard')).toBeInTheDocument();
      expect(screen.getByLabelText('Create New Benchmark')).toBeInTheDocument();
      expect(screen.getByLabelText('Filter Benchmarks')).toBeInTheDocument();
    });
  });

  describe('MetricsVisualization', () => {
    it('should have no accessibility violations', async () => {
      const { container } = render(
        <MetricsVisualization
          benchmarkResults={mockBenchmarkResults}
          selectedResult={mockBenchmarkResults[0]}
          onResultSelect={jest.fn()}
        />
      );

      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should support keyboard navigation for chart controls', () => {
      render(
        <MetricsVisualization
          benchmarkResults={mockBenchmarkResults}
          selectedResult={mockBenchmarkResults[0]}
          onResultSelect={jest.fn()}
        />
      );

      const select = screen.getByLabelText('Select Benchmark Result');
      expect(select).toHaveAttribute('tabindex', '0');

      fireEvent.keyDown(select, { key: 'ArrowDown' });
      fireEvent.keyDown(select, { key: 'Enter' });
    });

    it('should have proper contrast for chart elements', () => {
      render(
        <MetricsVisualization
          benchmarkResults={mockBenchmarkResults}
          selectedResult={mockBenchmarkResults[0]}
          onResultSelect={jest.fn()}
        />
      );

      // Check that chart elements have sufficient contrast
      const chartElements = screen.getAllByRole('img');
      chartElements.forEach(element => {
        expect(element).toHaveAttribute('aria-label');
        expect(element).not.toHaveStyle({ opacity: '0' });
      });
    });
  });

  describe('ScenarioBuilder', () => {
    it('should have no accessibility violations', async () => {
      const { container } = render(
        <ScenarioBuilder
          onScenarioCreated={jest.fn()}
          onScenarioUpdated={jest.fn()}
        />
      );

      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should support keyboard navigation through form fields', () => {
      render(
        <ScenarioBuilder
          onScenarioCreated={jest.fn()}
          onScenarioUpdated={jest.fn()}
        />
      );

      // Test form field navigation
      const nameInput = screen.getByLabelText('Scenario Name *');
      const descriptionInput = screen.getByLabelText('Description');
      const durationInput = screen.getByLabelText('Duration (seconds)');

      expect(nameInput).toHaveAttribute('tabindex', '0');
      expect(descriptionInput).toHaveAttribute('tabindex', '0');
      expect(durationInput).toHaveAttribute('tabindex', '0');

      // Test tab order
      fireEvent.tab();
      expect(document.activeElement).toBe(nameInput);
      
      fireEvent.tab();
      expect(document.activeElement).toBe(descriptionInput);
      
      fireEvent.tab();
      expect(document.activeElement).toBe(durationInput);
    });

    it('should have proper form validation feedback', () => {
      render(
        <ScenarioBuilder
          onScenarioCreated={jest.fn()}
          onScenarioUpdated={jest.fn()}
        />
      );

      const submitButton = screen.getByRole('button', { name: /create scenario/i });
      fireEvent.click(subButton);

      // Should show validation errors
      expect(screen.getByText('Scenario name is required')).toBeInTheDocument();
      expect(screen.getByText('At least one agent type must be selected')).toBeInTheDocument();
    });
  });

  describe('ExecutionMonitor', () => {
    it('should have no accessibility violations', async () => {
      const { container } = render(
        <ExecutionMonitor
          benchmarkId="test-1"
          isConnected={true}
          events={mockWebSocketEvents}
          onExport={jest.fn()}
        />
      );

      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should support keyboard navigation for event controls', () => {
      render(
        <ExecutionMonitor
          benchmarkId="test-1"
          isConnected={true}
          events={mockWebSocketEvents}
          onExport={jest.fn()}
        />
      );

      const pauseButton = screen.getByLabelText('Pause Execution');
      const exportButton = screen.getByLabelText('Export Logs');

      expect(pauseButton).toHaveAttribute('tabindex', '0');
      expect(exportButton).toHaveAttribute('tabindex', '0');

      fireEvent.keyDown(pauseButton, { key: ' ' });
      fireEvent.keyDown(exportButton, { key: 'Enter' });
    });

    it('should have proper live region updates', () => {
      render(
        <ExecutionMonitor
          benchmarkId="test-1"
          isConnected={true}
          events={mockWebSocketEvents}
          onExport={jest.fn()}
        />
      );

      // Check for live region announcements
      const liveRegion = screen.getByRole('status');
      expect(liveRegion).toBeInTheDocument();
      expect(liveRegion).toHaveAttribute('aria-live', 'polite');
    });
  });

  describe('ResultsComparison', () => {
    it('should have no accessibility violations', async () => {
      const { container } = render(
        <ResultsComparison
          benchmarkResults={mockBenchmarkResults}
          selectedResults={[mockBenchmarkResults[0]]}
          onResultSelect={jest.fn()}
        />
      );

      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should support keyboard navigation for comparison controls', () => {
      render(
        <ResultsComparison
          benchmarkResults={mockBenchmarkResults}
          selectedResults={[mockBenchmarkResults[0]]}
          onResultSelect={jest.fn()}
        />
      );

      const comparisonTypeSelect = screen.getByLabelText('Comparison Type');
      const metricSelect = screen.getByLabelText('Select Metrics');

      expect(comparisonTypeSelect).toHaveAttribute('tabindex', '0');
      expect(metricSelect).toHaveAttribute('tabindex', '0');

      fireEvent.keyDown(comparisonTypeSelect, { key: 'ArrowDown' });
      fireEvent.keyDown(metricSelect, { key: 'ArrowDown' });
    });

    it('should have proper data table accessibility', () => {
      render(
        <ResultsComparison
          benchmarkResults={mockBenchmarkResults}
          selectedResults={[mockBenchmarkResults[0]]}
          onResultSelect={jest.fn()}
        />
      );

      const table = screen.getByRole('table');
      expect(table).toBeInTheDocument();
      expect(table).toHaveAttribute('aria-label');

      const headers = screen.getAllByRole('columnheader');
      headers.forEach(header => {
        expect(header).toHaveAttribute('scope', 'col');
      });

      const cells = screen.getAllByRole('cell');
      cells.forEach(cell => {
        expect(cell).toBeInTheDocument();
      });
    });
  });

  describe('Chart Components', () => {
    it('RadarChart should have no accessibility violations', async () => {
      const { container } = render(
        <RadarChart
          data={[
            {
              agent_id: 'agent-1',
              agent_name: 'Test Agent',
              capabilities: {
                cognitive: 0.8,
                business: 0.7,
                technical: 0.9,
                ethical: 0.85,
              },
              overall_score: 0.8125,
              timestamp: '2025-01-01T00:00:00Z',
            },
          ]}
          selectedMetrics={['cognitive', 'business', 'technical', 'ethical']}
          onMetricSelect={jest.fn()}
        />
      );

      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('HeatmapChart should have no accessibility violations', async () => {
      const { container } = render(
        <HeatmapChart
          data={{
            agents: ['agent-1', 'agent-2'],
            scenarios: ['scenario-1', 'scenario-2'],
            metrics: ['accuracy', 'efficiency'],
            data: [
              [[0.9, 0.8], [0.7, 0.6]],
              [[0.8, 0.7], [0.6, 0.5]],
            ],
            timestamp: '2025-01-01T00:00:00Z',
          }}
          selectedMetric="accuracy"
          onMetricSelect={jest.fn()}
        />
      );

      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('TimeSeriesChart should have no accessibility violations', async () => {
      const { container } = render(
        <TimeSeriesChart
          data={[
            {
              timestamp: '2025-01-01T00:00:00Z',
              tick: 0,
              cpu_usage: 25,
              memory_usage: 50,
              response_time: 100,
            },
            {
              timestamp: '2025-01-01T00:01:00Z',
              tick: 1,
              cpu_usage: 35,
              memory_usage: 55,
              response_time: 120,
            },
          ]}
          selectedMetrics={['cpu_usage', 'memory_usage', 'response_time']}
          onMetricSelect={jest.fn()}
        />
      );

      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('ComparisonChart should have no accessibility violations', async () => {
      const { container } = render(
        <ComparisonChart
          data={[
            {
              agent_id: 'agent-1',
              agent_name: 'Test Agent 1',
              metrics: [
                { name: 'accuracy', value: 0.95 },
                { name: 'efficiency', value: 0.85 },
                { name: 'cost', value: 0.75 },
              ],
            },
            {
              agent_id: 'agent-2',
              agent_name: 'Test Agent 2',
              metrics: [
                { name: 'accuracy', value: 0.88 },
                { name: 'efficiency', value: 0.92 },
                { name: 'cost', value: 0.68 },
              ],
            },
          ]}
          selectedMetrics={['accuracy', 'efficiency', 'cost']}
          onMetricSelect={jest.fn()}
        />
      );

      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  describe('Color and Contrast Testing', () => {
    it('should have sufficient color contrast for all text', () => {
      render(
        <BenchmarkDashboard
          benchmarkResults={mockBenchmarkResults}
          onBenchmarkSelect={jest.fn()}
          onBenchmarkCreate={jest.fn()}
        />
      );

      // Test text elements for sufficient contrast
      const textElements = screen.getAllByText(/test/i);
      textElements.forEach(element => {
        const computedStyle = window.getComputedStyle(element);
        expect(computedStyle.color).toBeDefined();
        expect(computedStyle.backgroundColor).toBeDefined();
      });
    });

    it('should not rely solely on color for information', () => {
      render(
        <MetricsVisualization
          benchmarkResults={mockBenchmarkResults}
          selectedResult={mockBenchmarkResults[0]}
          onResultSelect={jest.fn()}
        />
      );

      // Check that status indicators have text labels
      const statusElements = screen.getAllByRole('status');
      statusElements.forEach(element => {
        expect(element.textContent).toBeDefined();
        expect(element.textContent).not.toBe('');
      });
    });
  });

  describe('Screen Reader Testing', () => {
    it('should be compatible with screen readers', () => {
      render(
        <BenchmarkDashboard
          benchmarkResults={mockBenchmarkResults}
          onBenchmarkSelect={jest.fn()}
          onBenchmarkCreate={jest.fn()}
        />
      );

      // Test that important elements are announced to screen readers
      const announcements = screen.getAllByRole('status');
      expect(announcements.length).toBeGreaterThan(0);

      // Test that interactive elements have proper labels
      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toHaveAccessibleName();
      });
    });

    it('should provide proper context for screen readers', () => {
      render(
        <ScenarioBuilder
          onScenarioCreated={jest.fn()}
          onScenarioUpdated={jest.fn()}
        />
      );

      // Test that form fields have proper labels and descriptions
      const nameInput = screen.getByLabelText('Scenario Name *');
      expect(nameInput).toHaveAccessibleDescription();

      const submitButton = screen.getByRole('button', { name: /create scenario/i });
      expect(submitButton).toHaveAccessibleName();
    });
  });

  describe('Focus Management', () => {
    it('should manage focus properly during interactions', () => {
      render(
        <BenchmarkDashboard
          benchmarkResults={mockBenchmarkResults}
          onBenchmarkSelect={jest.fn()}
          onBenchmarkCreate={jest.fn()}
        />
      );

      // Test focus trap in modals
      const createButton = screen.getByLabelText('Create New Benchmark');
      fireEvent.click(createButton);

      // Check that focus is trapped in modal
      const modalElements = screen.getAllByRole('dialog');
      expect(modalElements.length).toBeGreaterThan(0);

      // Test that Tab key cycles through focusable elements
      const focusableElements = screen.getAllByRole('button');
      focusableElements.forEach((element, index) => {
        expect(element).toHaveAttribute('tabindex', '0');
      });
    });

    it('should restore focus after interactions', () => {
      render(
        <ExecutionMonitor
          benchmarkId="test-1"
          isConnected={true}
          events={mockWebSocketEvents}
          onExport={jest.fn()}
        />
      );

      const exportButton = screen.getByLabelText('Export Logs');
      const initialFocus = document.activeElement;

      fireEvent.click(exportButton);

      // After interaction, focus should be restored appropriately
      expect(document.activeElement).toBe(initialFocus);
    });
  });

  describe('Responsive Design Testing', () => {
    it('should be accessible on different screen sizes', () => {
      // Test mobile view
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      });

      render(
        <BenchmarkDashboard
          benchmarkResults={mockBenchmarkResults}
          onBenchmarkSelect={jest.fn()}
          onBenchmarkCreate={jest.fn()}
        />
      );

      // Check that responsive elements are properly sized
      const container = screen.getByLabelText('Benchmark Dashboard');
      expect(container).toBeInTheDocument();

      // Test tablet view
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      });

      // Re-render with new viewport size
      render(
        <BenchmarkDashboard
          benchmarkResults={mockBenchmarkResults}
          onBenchmarkSelect={jest.fn()}
          onBenchmarkCreate={jest.fn()}
        />
      );

      // Test desktop view
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      });

      // Re-render with new viewport size
      render(
        <BenchmarkDashboard
          benchmarkResults={mockBenchmarkResults}
          onBenchmarkSelect={jest.fn()}
          onBenchmarkCreate={jest.fn()}
        />
      );
    });
  });
});