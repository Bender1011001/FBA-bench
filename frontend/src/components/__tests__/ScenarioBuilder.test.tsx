import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { apiService } from '../../services/apiService';
import { notificationService } from '../../utils/notificationService';
import ScenarioBuilder from '../ScenarioBuilder';

// Mock the dependencies
jest.mock('../../services/apiService');
jest.mock('../../utils/notificationService');

const mockApiService = apiService as jest.Mocked<typeof apiService>;
const mockNotificationService = notificationService as jest.Mocked<typeof notificationService>;

describe('ScenarioBuilder Component', () => {
  const mockConfigurationTemplates = [
    {
      id: 'template-1',
      name: 'Basic Benchmark',
      description: 'A basic benchmark configuration',
      category: 'general',
      config: {
        benchmark_id: 'basic-benchmark',
        name: 'Basic Benchmark',
        description: 'A basic benchmark configuration',
        version: '1.0.0',
        environment: {
          deterministic: true,
          random_seed: 42,
          parallel_execution: false,
          max_workers: 1
        },
        scenarios: [
          {
            id: 'basic-scenario',
            name: 'Basic Scenario',
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
            id: 'basic-agent',
            name: 'Basic Agent',
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
      }
    },
    {
      id: 'template-2',
      name: 'Advanced Benchmark',
      description: 'An advanced benchmark configuration',
      category: 'advanced',
      config: {
        benchmark_id: 'advanced-benchmark',
        name: 'Advanced Benchmark',
        description: 'An advanced benchmark configuration',
        version: '1.0.0',
        environment: {
          deterministic: false,
          random_seed: null,
          parallel_execution: true,
          max_workers: 4
        },
        scenarios: [
          {
            id: 'advanced-scenario',
            name: 'Advanced Scenario',
            type: 'test',
            enabled: true,
            priority: 1,
            config: {
              duration: 200,
              complexity: 'high'
            }
          }
        ],
        agents: [
          {
            id: 'advanced-agent',
            name: 'Advanced Agent',
            framework: 'test',
            enabled: true,
            config: {
              model: 'advanced_model',
              temperature: 0.7
            }
          }
        ],
        metrics: {
          categories: ['cognitive', 'business', 'technical'],
          custom_metrics: []
        },
        execution: {
          runs_per_scenario: 5,
          max_duration: 0,
          timeout: 600,
          retry_on_failure: true,
          max_retries: 5
        },
        output: {
          format: 'json',
          path: './test_results',
          include_detailed_logs: true,
          include_audit_trail: true
        },
        validation: {
          enabled: true,
          statistical_significance: true,
          confidence_level: 0.99,
          reproducibility_check: true
        },
        metadata: {
          author: 'Test Author',
          created: '2025-01-01T00:00:00Z',
          tags: ['test', 'advanced']
        }
      }
    }
  ];

  const mockAgentFrameworks = [
    {
      id: 'framework-1',
      name: 'Test Framework',
      description: 'A test agent framework',
      version: '1.0.0',
      capabilities: ['reasoning', 'planning'],
      config_schema: {
        type: 'object',
        properties: {
          model: {
            type: 'string',
            description: 'Model name'
          },
          temperature: {
            type: 'number',
            description: 'Temperature setting',
            minimum: 0,
            maximum: 1
          }
        },
        required: ['model']
      }
    },
    {
      id: 'framework-2',
      name: 'Advanced Framework',
      description: 'An advanced agent framework',
      version: '2.0.0',
      capabilities: ['reasoning', 'planning', 'learning'],
      config_schema: {
        type: 'object',
        properties: {
          model: {
            type: 'string',
            description: 'Model name'
          },
          temperature: {
            type: 'number',
            description: 'Temperature setting',
            minimum: 0,
            maximum: 1
          },
          max_tokens: {
            type: 'integer',
            description: 'Maximum tokens',
            minimum: 1
          }
        },
        required: ['model']
      }
    }
  ];

  const mockScenarioTypes = [
    {
      id: 'type-1',
      name: 'Test Scenario',
      description: 'A test scenario type',
      config_schema: {
        type: 'object',
        properties: {
          duration: {
            type: 'integer',
            description: 'Duration in seconds',
            minimum: 1
          },
          complexity: {
            type: 'string',
            description: 'Complexity level',
            enum: ['low', 'medium', 'high']
          }
        },
        required: ['duration']
      }
    },
    {
      id: 'type-2',
      name: 'Advanced Scenario',
      description: 'An advanced scenario type',
      config_schema: {
        type: 'object',
        properties: {
          duration: {
            type: 'integer',
            description: 'Duration in seconds',
            minimum: 1
          },
          complexity: {
            type: 'string',
            description: 'Complexity level',
            enum: ['low', 'medium', 'high']
          },
          difficulty: {
            type: 'integer',
            description: 'Difficulty level',
            minimum: 1,
            maximum: 10
          }
        },
        required: ['duration']
      }
    }
  ];

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Setup default mock implementations
    mockApiService.get.mockImplementation((url) => {
      if (url === '/benchmarking/configuration-templates') {
        return Promise.resolve({ data: mockConfigurationTemplates });
      }
      if (url === '/benchmarking/agent-frameworks') {
        return Promise.resolve({ data: mockAgentFrameworks });
      }
      if (url === '/benchmarking/scenario-types') {
        return Promise.resolve({ data: mockScenarioTypes });
      }
      return Promise.resolve({ data: {} });
    });
    
    mockApiService.post.mockResolvedValue({
      data: {
        benchmark_id: 'new-benchmark',
        status: 'created'
      }
    });
  });

  test('renders without crashing', () => {
    render(<ScenarioBuilder />);
    expect(screen.getByText('Scenario Builder')).toBeInTheDocument();
  });

  test('fetches configuration templates on mount', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/configuration-templates');
    });
  });

  test('fetches agent frameworks on mount', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/agent-frameworks');
    });
  });

  test('fetches scenario types on mount', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      expect(mockApiService.get).toHaveBeenCalledWith('/benchmarking/scenario-types');
    });
  });

  test('displays configuration templates for selection', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      expect(screen.getByText('Basic Benchmark')).toBeInTheDocument();
      expect(screen.getByText('Advanced Benchmark')).toBeInTheDocument();
    });
  });

  test('loads template when selected', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      // Select the first template
      fireEvent.click(screen.getByText('Basic Benchmark'));
    });
    
    // Check that the form is populated with template data
    expect(screen.getByDisplayValue('Basic Benchmark')).toBeInTheDocument();
    expect(screen.getByDisplayValue('A basic benchmark configuration')).toBeInTheDocument();
  });

  test('updates benchmark name when changed', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      // Select the first template
      fireEvent.click(screen.getByText('Basic Benchmark'));
    });
    
    // Change the benchmark name
    const nameInput = screen.getByLabelText('Benchmark Name *');
    fireEvent.change(nameInput, { target: { value: 'Custom Benchmark' } });
    
    expect(nameInput).toHaveValue('Custom Benchmark');
  });

  test('updates benchmark description when changed', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      // Select the first template
      fireEvent.click(screen.getByText('Basic Benchmark'));
    });
    
    // Change the benchmark description
    const descriptionInput = screen.getByLabelText('Description');
    fireEvent.change(descriptionInput, { target: { value: 'Custom description' } });
    
    expect(descriptionInput).toHaveValue('Custom description');
  });

  test('adds a new scenario', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      // Select the first template
      fireEvent.click(screen.getByText('Basic Benchmark'));
    });
    
    // Add a new scenario
    fireEvent.click(screen.getByText('Add Scenario'));
    
    // Check that a new scenario form is displayed
    expect(screen.getByText('New Scenario')).toBeInTheDocument();
  });

  test('removes a scenario', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      // Select the first template
      fireEvent.click(screen.getByText('Basic Benchmark'));
    });
    
    // Remove the existing scenario
    fireEvent.click(screen.getByText('Remove Scenario'));
    
    // Check that the scenario is removed
    expect(screen.queryByText('Basic Scenario')).not.toBeInTheDocument();
  });

  test('adds a new agent', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      // Select the first template
      fireEvent.click(screen.getByText('Basic Benchmark'));
    });
    
    // Add a new agent
    fireEvent.click(screen.getByText('Add Agent'));
    
    // Check that a new agent form is displayed
    expect(screen.getByText('New Agent')).toBeInTheDocument();
  });

  test('removes an agent', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      // Select the first template
      fireEvent.click(screen.getByText('Basic Benchmark'));
    });
    
    // Remove the existing agent
    fireEvent.click(screen.getByText('Remove Agent'));
    
    // Check that the agent is removed
    expect(screen.queryByText('Basic Agent')).not.toBeInTheDocument();
  });

  test('validates configuration before saving', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      // Select the first template
      fireEvent.click(screen.getByText('Basic Benchmark'));
    });
    
    // Clear the benchmark name
    const nameInput = screen.getByLabelText('Benchmark Name *');
    fireEvent.change(nameInput, { target: { value: '' } });
    
    // Try to save
    fireEvent.click(screen.getByText('Save Configuration'));
    
    // Check that validation error is displayed
    expect(screen.getByText('Benchmark name is required')).toBeInTheDocument();
  });

  test('saves configuration with valid data', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      // Select the first template
      fireEvent.click(screen.getByText('Basic Benchmark'));
    });
    
    // Save the configuration
    fireEvent.click(screen.getByText('Save Configuration'));
    
    await waitFor(() => {
      expect(mockApiService.post).toHaveBeenCalledWith(
        '/benchmarking/configurations',
        expect.objectContaining({
          benchmark_id: 'basic-benchmark',
          name: 'Basic Benchmark'
        })
      );
      
      expect(mockNotificationService.success).toHaveBeenCalledWith(
        'Configuration saved successfully',
        3000
      );
    });
  });

  test('resets form when reset button is clicked', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      // Select the first template
      fireEvent.click(screen.getByText('Basic Benchmark'));
    });
    
    // Change the benchmark name
    const nameInput = screen.getByLabelText('Benchmark Name *');
    fireEvent.change(nameInput, { target: { value: 'Custom Benchmark' } });
    
    // Reset the form
    fireEvent.click(screen.getByText('Reset'));
    
    // Check that the form is reset
    expect(nameInput).toHaveValue('Basic Benchmark');
  });

  test('exports configuration as JSON', async () => {
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      // Select the first template
      fireEvent.click(screen.getByText('Basic Benchmark'));
    });
    
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
    expect(mockLink.download).toBe('basic-benchmark.json');
    expect(mockLink.click).toHaveBeenCalled();
  });

  test('imports configuration from JSON', async () => {
    const mockFile = new File(['{"benchmark_id": "imported-benchmark"}'], 'config.json', {
      type: 'application/json'
    });
    
    render(<ScenarioBuilder />);
    
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
    
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      expect(screen.getByText('Failed to fetch configuration templates')).toBeInTheDocument();
    });
  });

  test('displays error message when configuration save fails', async () => {
    mockApiService.post.mockRejectedValue(new Error('Save Error'));
    
    render(<ScenarioBuilder />);
    
    await waitFor(() => {
      // Select the first template
      fireEvent.click(screen.getByText('Basic Benchmark'));
    });
    
    // Save the configuration
    fireEvent.click(screen.getByText('Save Configuration'));
    
    await waitFor(() => {
      expect(screen.getByText('Failed to save configuration')).toBeInTheDocument();
    });
  });
});