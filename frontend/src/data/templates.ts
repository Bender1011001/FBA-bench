import type { Configuration } from '../types';

export interface Template {
  id: string;
  name: string;
  description: string;
  useCase: string;
  configuration: Configuration;
}

export const predefinedTemplates: Template[] = [
  {
    id: 'quick-start',
    name: 'Quick Start',
    description: 'A basic single-agent simulation with Grok4 LLM, ideal for initial setup and quick validation.',
    useCase: 'Basic single-agent simulation, LLM validation, quick testing',
    configuration: {
      simulationSettings: {
        simulationName: 'Quick Start Simulation',
        description: 'A basic simulation for quick starting.',
        duration: 100, // ticks
        randomSeed: 42,
        // Example additional settings
        metricsInterval: 10,
        snapshotInterval: 50,
      },
      agentConfigs: [
        {
          agentName: 'BasicAgent',
          agentType: 'baseline_bots',
          model: 'grok-4',
          max_tokens: 1000,
          temperature: 0.7,
          // Add other agent specific settings
          llmInterface: 'openrouter', // Assuming openrouter is the default LLM provider
          role: 'participant',
          behavior: 'default',
        },
      ],
      llmSettings: {
        provider: 'openrouter',
        api_key: 'sk-xxxxxxxx', // Placeholder
        model: 'grok-4',
        temperature: 0.7,
        max_tokens: 1000,
      },
      constraints: {
        maxBudget: 100,
        maxTime: 60000, // milliseconds
        tokenLimits: {
          grok4: 1000000, // Example token limit
        },
      },
       experimentSettings: { // Include experiment settings
        experimentName: 'Quick Start Experiment',
        description: 'Default experiment for quick start.',
        iterations: 1,
        parallelRuns: 1,
        // Other experiment settings
        parameters: [],
      }
    },
  },
  {
    id: 'multi-agent-competition',
    name: 'Multi-Agent Competition',
    description: 'A scenario with 3 agents competing, designed to evaluate strategic interactions and resource allocation under pressure.',
    useCase: 'Multi-agent competition, strategic interaction analysis, resource contention',
    configuration: {
      simulationSettings: {
        simulationName: 'Competitive Scenario',
        description: 'Three agents competing for resources.',
        duration: 200,
        randomSeed: 123,
        metricsInterval: 20,
        snapshotInterval: 100,
      },
      agentConfigs: [
        {
          agentName: 'AgentA',
          agentType: 'baseline_bots',
          model: 'grok-4',
          max_tokens: 1500,
          temperature: 0.8,
          llmInterface: 'openrouter',
          role: 'competitor',
          behavior: 'aggressive',
        },
        {
          agentName: 'AgentB',
          agentType: 'baseline_bots',
          model: 'grok-4',
          max_tokens: 1500,
          temperature: 0.8,
          llmInterface: 'openrouter',
          role: 'competitor',
          behavior: 'defensive',
        },
        {
          agentName: 'AgentC',
          agentType: 'baseline_bots',
          model: 'grok-4',
          max_tokens: 1500,
          temperature: 0.8,
          llmInterface: 'openrouter',
          role: 'competitor',
          behavior: 'balanced',
        },
      ],
      llmSettings: {
        provider: 'openrouter',
        api_key: 'sk-xxxxxxxx',
        model: 'grok-4',
        temperature: 0.8,
        max_tokens: 1500,
      },
      constraints: {
        maxBudget: 500,
        maxTime: 120000,
        tokenLimits: {
          grok4: 2000000,
        },
      },
      experimentSettings: {
        experimentName: 'Multi-Agent Competition Exp',
        description: 'Experimenting with competitive agent strategies.',
        iterations: 3,
        parallelRuns: 1,
        parameters: [],
      }
    },
  },
  {
    id: 'parameter-sweep-experiment',
    name: 'Parameter Sweep Experiment',
    description: 'Designed for systematic evaluation of how changing specific parameters (e.g., agent temperature, simulation duration) affects outcomes.',
    useCase: 'Parameter optimization, sensitivity analysis, hypothesis testing',
    configuration: {
      simulationSettings: {
        simulationName: 'Parameter Sweep',
        description: 'Systematic evaluation of parameters.',
        duration: 50,
        randomSeed: 1, // Will be overridden by sweep parameters
        metricsInterval: 5,
        snapshotInterval: 25,
      },
      agentConfigs: [
        {
          agentName: 'SweepingAgent',
          agentType: 'baseline_bots',
          model: 'grok-4',
          max_tokens: 800,
          temperature: 0.5, // Will be swept
          llmInterface: 'openrouter',
          role: 'researcher',
          behavior: 'exploratory',
        },
      ],
      llmSettings: {
        provider: 'openrouter',
        api_key: 'sk-xxxxxxxx',
        model: 'grok-4',
        temperature: 0.5, // Will be swept
        max_tokens: 800,
      },
      constraints: {
        maxBudget: 200,
        maxTime: 90000,
        tokenLimits: {
          grok4: 1500000,
        },
      },
      experimentSettings: {
        experimentName: 'Temperature Sweep',
        description: 'Sweeping agent temperature and simulation duration.',
        iterations: 5,
        parallelRuns: 2,
        parameters: [
          {
            name: 'agentConfigs[0].temperature',
            values: [0.5, 0.7, 0.9],
          },
          {
            name: 'simulationSettings.duration',
            values: [50, 100, 150],
          },
        ],
      }
    },
  },
  {
    id: 'memory-evaluation',
    name: 'Memory Evaluation',
    description: 'A template focused on testing agent memory capabilities, assessing how well agents retain and utilize information over time.',
    useCase: 'Memory testing, long-term context evaluation, information retention',
    configuration: {
      simulationSettings: {
        simulationName: 'Memory Test Simulation',
        description: 'Evaluating agent memory capabilities.',
        duration: 150,
        randomSeed: 777,
        metricsInterval: 15,
        snapshotInterval: 75,
      },
      agentConfigs: [
        {
          agentName: 'MemoryAgent',
          agentType: 'baseline_bots',
          model: 'grok-4',
          max_tokens: 2000,
          temperature: 0.6,
          llmInterface: 'openrouter',
          role: 'learner',
          behavior: 'adaptive',
        },
      ],
      llmSettings: {
        provider: 'openrouter',
        api_key: 'sk-xxxxxxxx',
        model: 'grok-4',
        temperature: 0.6,
        max_tokens: 2000,
      },
      constraints: {
        maxBudget: 300,
        maxTime: 180000,
        tokenLimits: {
          grok4: 3000000,
        },
      },
      experimentSettings: {
        experimentName: 'Memory Retention Test',
        description: 'Tests agent context retention over extended interactions.',
        iterations: 1,
        parallelRuns: 1,
        parameters: [],
      }
    },
  },
  {
    id: 'custom-configuration',
    name: 'Custom Configuration',
    description: 'An empty template providing a blank slate for users to manually set up all simulation, agent, LLM, and constraint parameters from scratch.',
    useCase: 'Manual setup, highly custom scenarios, starting from scratch',
    configuration: {
      simulationSettings: {
        simulationName: 'New Custom Simulation',
        description: 'Start with a blank slate for full customization.',
        duration: 0,
        randomSeed: 0,
        metricsInterval: 0,
        snapshotInterval: 0,
      },
      agentConfigs: [],
      llmSettings: {
        provider: 'openrouter',
        api_key: '',
        model: 'grok-4',
        temperature: 0.7,
        max_tokens: 1000,
      },
      constraints: {
        maxBudget: 0,
        maxTime: 0,
        tokenLimits: {
          grok4: 0, // No limit by default for custom
        },
      },
      experimentSettings: {
        experimentName: 'Custom Experiment',
        description: 'User-defined experiment configuration.',
        iterations: 1,
        parallelRuns: 1,
        parameters: [],
      }
    },
  },
];