import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { apiService } from '../../services/apiService';
import { notificationService } from '../../utils/notificationService';
import ConfigurationEditor from '../ConfigurationEditor';

// Mock the dependencies
jest.mock('../../services/apiService');
jest.mock('../../utils/notificationService');

const mockApiService = apiService as jest.Mocked<typeof apiService>;
const mockNotificationService = notificationService as jest.Mocked<typeof notificationService>;

describe('ConfigurationEditor Component', () => {
  const mockConfiguration = {
    benchmark_id: 'test-benchmark',
    name: 'Test Benchmark',
    description: 'A test benchmark configuration',
    version: '1.0.0',
    environment: {
      deterministic: true,
      random_seed: 42,
      parallel_execution: false,
      max_workers: 1
    },
    scenarios: [
      {
        id: 'test-scenario',
        name: 'Test Scenario',
        type: 'test',
        enabled: true,
        priority: 1,
        config: {
          duration: 100,
          complexity: 'medium'
        }
      }
    ],
    agents: [
      {
        id: 'test-agent',
        name: 'Test Agent',
        framework: 'test',
        enabled: true,
        config: {
          model: 'test_model',
          temperature: 0.5
        }
      }
    ],
    metrics: {
      categories: ['cognitive', 'business'],
      custom_metrics: []
    },
    execution: {
      runs_per_scenario: 2,
      max_duration: 0,
      timeout: 300,
      retry_on_failure: true,
      max_retries: 3
    },
    output: {
      format: 'json',
      path: './test_results',
      include_detailed_logs: false,
      include_audit_trail: true
    },
    validation: {
      enabled: true,
      statistical_significance: true,
      confidence_level: 0.95,
      reproducibility_check: true
    },
    metadata: {
      author: 'Test Author',
      created: '2025-01-01T00:00:00Z',
      tags: ['test']
    }
  };

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Setup default mock implementations
    mockApiService.get.mockResolvedValue({
      data: mockConfiguration
    });
    
    mockApiService.put.mockResolvedValue({
      data: {
        ...mockConfiguration,
        name: 'Updated Test Benchmark'
      }
    });
  });

  test('renders without crashing', () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    expect(screen.getByText('Configuration Editor')).toBeInTheDocument();
  });

  test('fetches configuration on mount', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/configurations/test-config');
    });
  });

  test('displays configuration data when loaded', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      expect(screen.getByDisplayValue('Test Benchmark')).toBeInTheDocument();
      expect(screen.getByDisplayValue('A test benchmark configuration')).toBeInTheDocument();
    });
  });

  test('updates configuration name when changed', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      const nameInput = screen.getByLabelText('Benchmark Name *');
      fireEvent.change(nameInput, { target: { value: 'Updated Benchmark' } });
      expect(nameInput).toHaveValue('Updated Benchmark');
    });
  });

  test('updates configuration description when changed', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      const descriptionInput = screen.getByLabelText('Description');
      fireEvent.change(descriptionInput, { target: { value: 'Updated description' } });
      expect(descriptionInput).toHaveValue('Updated description');
    });
  });

  test('updates environment settings when changed', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      const deterministicCheckbox = screen.getByLabelText('Deterministic');
      fireEvent.click(deterministicCheckbox);
      expect(deterministicCheckbox).not.toBeChecked();
    });
  });

  test('updates scenario settings when changed', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      const scenarioNameInput = screen.getByLabelText('Scenario Name *');
      fireEvent.change(scenarioNameInput, { target: { value: 'Updated Scenario' } });
      expect(scenarioNameInput).toHaveValue('Updated Scenario');
    });
  });

  test('updates agent settings when changed', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      const agentNameInput = screen.getByLabelText('Agent Name *');
      fireEvent.change(agentNameInput, { target: { value: 'Updated Agent' } });
      expect(agentNameInput).toHaveValue('Updated Agent');
    });
  });

  test('updates metrics settings when changed', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      const cognitiveCheckbox = screen.getByLabelText('Cognitive');
      fireEvent.click(cognitiveCheckbox);
      expect(cognitiveCheckbox).not.toBeChecked();
    });
  });

  test('updates execution settings when changed', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      const runsInput = screen.getByLabelText('Runs per Scenario *');
      fireEvent.change(runsInput, { target: { value: '5' } });
      expect(runsInput).toHaveValue('5');
    });
  });

  test('updates output settings when changed', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      const formatSelect = screen.getByLabelText('Output Format *');
      fireEvent.change(formatSelect, { target: { value: 'csv' } });
      expect(formatSelect).toHaveValue('csv');
    });
  });

  test('updates validation settings when changed', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      const validationCheckbox = screen.getByLabelText('Enable Validation');
      fireEvent.click(validationCheckbox);
      expect(validationCheckbox).not.toBeChecked();
    });
  });

  test('updates metadata settings when changed', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      const authorInput = screen.getByLabelText('Author');
      fireEvent.change(authorInput, { target: { value: 'Updated Author' } });
      expect(authorInput).toHaveValue('Updated Author');
    });
  });

  test('validates configuration before saving', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      // Clear the benchmark name
      const nameInput = screen.getByLabelText('Benchmark Name *');
      fireEvent.change(nameInput, { target: { value: '' } });
      
      // Try to save
      fireEvent.click(screen.getByText('Save Configuration'));
      
      // Check that validation error is displayed
      expect(screen.getByText('Benchmark name is required')).toBeInTheDocument();
    });
  });

  test('saves configuration with valid data', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      // Save the configuration
      fireEvent.click(screen.getByText('Save Configuration'));
    });
    
    await waitFor(() => {
      expect(mockApiService.put).toHaveBeenCalledWith(
        '/benchmarking/configurations/test-config',
        expect.objectContaining({
          benchmark_id: 'test-benchmark',
          name: 'Test Benchmark'
        })
      );
      
      expect(mockNotificationService.success).toHaveBeenCalledWith(
        'Configuration updated successfully',
        3000
      );
    });
  });

  test('resets form when reset button is clicked', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      // Change the benchmark name
      const nameInput = screen.getByLabelText('Benchmark Name *');
      fireEvent.change(nameInput, { target: { value: 'Updated Benchmark' } });
      
      // Reset the form
      fireEvent.click(screen.getByText('Reset'));
      
      // Check that the form is reset
      expect(nameInput).toHaveValue('Test Benchmark');
    });
  });

  test('exports configuration as JSON', async () => {
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      // Mock document.createElement to capture the download link
      const mockLink = {
        href: '',
        download: '',
        click: jest.fn()
      };
      document.createElement = jest.fn().mockReturnValue(mockLink);
      
      // Export the configuration
      fireEvent.click(screen.getByText('Export JSON'));
      
      // Check that the download link was created
      expect(mockLink.download).toBe('test-benchmark.json');
      expect(mockLink.click).toHaveBeenCalled();
    });
  });

  test('imports configuration from JSON', async () => {
    const mockFile = new File(['{"benchmark_id": "imported-benchmark"}'], 'config.json', {
      type: 'application/json'
    });
    
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      // Mock the file input
      const fileInput = screen.getByLabelText('Import Configuration');
      Object.defineProperty(fileInput, 'files', {
        value: [mockFile],
        writable: false
      });
      
      // Trigger the change event
      fireEvent.change(fileInput);
    });
    
    // Check that the form is populated with imported data
    expect(screen.getByDisplayValue('imported-benchmark')).toBeInTheDocument();
  });

  test('displays error message when API call fails', async () => {
    mockApiService.get.mockRejectedValue(new Error('API Error'));
    
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      expect(screen.getByText('Failed to fetch configuration')).toBeInTheDocument();
    });
  });

  test('displays error message when configuration save fails', async () => {
    mockApiService.put.mockRejectedValue(new Error('Save Error'));
    
    render(<ConfigurationEditor configurationId="test-config" />);
    
    await waitFor(() => {
      // Save the configuration
      fireEvent.click(screen.getByText('Save Configuration'));
    });
    
    await waitFor(() => {
      expect(screen.getByText('Failed to update configuration')).toBeInTheDocument();
    });
  });

  test('displays loading state while fetching configuration', () => {
    // Mock a delayed response
    mockApiService.get.mockImplementation(() => {
      return new Promise(resolve => {
        setTimeout(() => {
          resolve({ data: mockConfiguration });
        }, 100);
      });
    });
    
    render(<ConfigurationEditor configurationId="test-config" />);
    
    // Check that loading indicator is displayed
    expect(screen.getByText('Loading configuration...')).toBeInTheDocument();
  });

  test('displays error state when configuration is not found', async () => {
    mockApiService.get.mockRejectedValue(new Error('Configuration not found'));
    
    render(<ConfigurationEditor configurationId="non-existent-config" />);
    
    await waitFor(() => {
      expect(screen.getByText('Configuration not found')).toBeInTheDocument();
    });
  });
});