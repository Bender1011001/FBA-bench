import React, { useState } from 'react';
import { SimulationConfigForm } from '../components/SimulationConfigForm';
import { AgentSelectionForm } from '../components/AgentSelectionForm';
import { ExperimentConfigForm } from '../components/ExperimentConfigForm';
import TemplateSelectionForm from '../components/TemplateSelectionForm';
import ConstraintConfigForm from '../components/ConstraintConfigForm';
import TierSelector from '../components/TierSelector';
// import ConstraintMonitor from '../components/ConstraintMonitor'; // Not directly used in wizard steps, but good for context if ever needed for live config check

import type { Configuration, SimulationSettings, TemplateAgentConfig, Template, ConstraintConfig } from '../types';
import { defaultConstraintConfig, constraintTiers } from '../data/constraintTiers'; // Import constraintTiers
import { sanitizeUserInput, sanitizeExternalData } from '../utils/sanitization';

const WizardStep = {
  SimulationType: 0,
  Templates: 1,
  Simulation: 2, // Renamed from BasicParameters
  Agents: 3, // Renamed from AgentSelection
  Constraints: 4, // New step for Constraints & Tiers - Step 5 in description for user, but step 4 in 0-indexed enum
  Experiment: 5, // Renamed from OutputSettings
  Review: 6 // Renamed from ReviewAndLaunch
} as const;

type WizardStep = typeof WizardStep[keyof typeof WizardStep];

export function ConfigurationWizard() {
  const [currentStep, setCurrentStep] = useState<WizardStep>(WizardStep.SimulationType);
  const [isExperiment, setIsExperiment] = useState<boolean>(false);
  const [isConstraintConfigValid, setIsConstraintConfigValid] = useState<boolean>(true); // New state for validation
  
  // Validation errors for each step
  const [validationErrors, setValidationErrors] = useState<Record<WizardStep, string[]>>({
    [WizardStep.SimulationType]: [],
    [WizardStep.Templates]: [],
    [WizardStep.Simulation]: [],
    [WizardStep.Agents]: [],
    [WizardStep.Constraints]: [],
    [WizardStep.Experiment]: [],
    [WizardStep.Review]: [],
  });

  const [currentConfiguration, setCurrentConfiguration] = useState<Configuration>({
    simulationSettings: {
      simulationName: '',
      description: '',
      duration: 0,
      tickInterval: 1,
      randomSeed: 0,
      metricsInterval: 0,
      snapshotInterval: 0,
      initialPrice: 10.0,
      inventory: 100,
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
      ...defaultConstraintConfig,
      tier: defaultConstraintConfig.tier as ConstraintConfig['tier'], // Explicitly cast tier
    },
    experimentSettings: {
      experimentName: '',
      description: '',
      iterations: 1,
      batchSize: 1, // Changed from parallelRuns
      parameters: [],
    }
  });

  // Validation functions for each step
  const validateSimulationStep = (): boolean => {
    const errors: string[] = [];
    const { simulationSettings } = currentConfiguration;
    
    if (!simulationSettings.simulationName || simulationSettings.simulationName.trim() === '') {
      errors.push('Simulation name is required');
    } else if (simulationSettings.simulationName.length < 3) {
      errors.push('Simulation name must be at least 3 characters long');
    } else if (simulationSettings.simulationName.length > 100) {
      errors.push('Simulation name must be less than 100 characters');
    }
    
    if (simulationSettings.description && simulationSettings.description.length > 500) {
      errors.push('Description must be less than 500 characters');
    }
    
    if (simulationSettings.duration <= 0) {
      errors.push('Duration must be greater than 0');
    } else if (simulationSettings.duration > 1000000) {
      errors.push('Duration must be less than 1,000,000');
    }
    
    if (simulationSettings.randomSeed < 0) {
      errors.push('Random seed must be non-negative');
    }
    
    if (simulationSettings.metricsInterval <= 0) {
      errors.push('Metrics interval must be greater than 0');
    } else if (simulationSettings.metricsInterval > simulationSettings.duration) {
      errors.push('Metrics interval cannot be greater than duration');
    }
    
    if (simulationSettings.snapshotInterval <= 0) {
      errors.push('Snapshot interval must be greater than 0');
    } else if (simulationSettings.snapshotInterval > simulationSettings.duration) {
      errors.push('Snapshot interval cannot be greater than duration');
    }
    
    setValidationErrors(prev => ({ ...prev, [WizardStep.Simulation]: errors }));
    return errors.length === 0;
  };
  
  const validateAgentsStep = (): boolean => {
    const errors: string[] = [];
    
    if (currentConfiguration.agentConfigs.length === 0) {
      errors.push('At least one agent configuration is required');
    }
    
    // Validate each agent configuration
    currentConfiguration.agentConfigs.forEach((agent, index) => {
      if (!agent.agentName || agent.agentName.trim() === '') {
        errors.push(`Agent ${index + 1}: Name is required`);
      }
      
      if (!agent.agentType || agent.agentType.trim() === '') {
        errors.push(`Agent ${index + 1}: Type is required`);
      }
      
      if (agent.temperature < 0 || agent.temperature > 2) {
        errors.push(`Agent ${index + 1}: Temperature must be between 0 and 2`);
      }
      
      if (agent.max_tokens <= 0 || agent.max_tokens > 100000) {
        errors.push(`Agent ${index + 1}: Max tokens must be between 1 and 100,000`);
      }
    });
    
    setValidationErrors(prev => ({ ...prev, [WizardStep.Agents]: errors }));
    return errors.length === 0;
  };
  
  const validateExperimentStep = (): boolean => {
    const errors: string[] = [];
    const { experimentSettings } = currentConfiguration;
    
    if (!experimentSettings.experimentName || experimentSettings.experimentName.trim() === '') {
      errors.push('Experiment name is required');
    } else if (experimentSettings.experimentName.length < 3) {
      errors.push('Experiment name must be at least 3 characters long');
    } else if (experimentSettings.experimentName.length > 100) {
      errors.push('Experiment name must be less than 100 characters');
    }
    
    if (experimentSettings.description && experimentSettings.description.length > 500) {
      errors.push('Description must be less than 500 characters');
    }
    
    if (experimentSettings.iterations !== undefined && experimentSettings.iterations <= 0) {
      errors.push('Iterations must be greater than 0');
    } else if (experimentSettings.iterations !== undefined && experimentSettings.iterations > 1000) {
      errors.push('Iterations must be less than 1,000');
    }
    
    if (experimentSettings.batchSize !== undefined && experimentSettings.batchSize <= 0) {
      errors.push('Batch size must be greater than 0');
    } else if (experimentSettings.batchSize !== undefined && experimentSettings.iterations !== undefined && experimentSettings.batchSize > experimentSettings.iterations) {
      errors.push('Batch size cannot be greater than iterations');
    }
    
    // Validate parameters
    if (experimentSettings.parameters && experimentSettings.parameters.length > 0) {
      experimentSettings.parameters.forEach((param, index) => {
        if (!param.name || param.name.trim() === '') {
          errors.push(`Parameter ${index + 1}: Name is required`);
        }
        
        if (!param.type || !['number', 'string', 'boolean'].includes(param.type)) {
          errors.push(`Parameter ${index + 1}: Type must be number, string, or boolean`);
        }
        
        if (param.type === 'range') {
          if (param.min === undefined || param.max === undefined) {
            errors.push(`Parameter ${index + 1}: Min and max values are required for range type`);
          } else if (param.min >= param.max) {
            errors.push(`Parameter ${index + 1}: Min value must be less than max value`);
          }
        }
      });
    }
    
    setValidationErrors(prev => ({ ...prev, [WizardStep.Experiment]: errors }));
    return errors.length === 0;
  };
  
  const validateLLMSettings = (): boolean => {
    const errors: string[] = [];
    const { llmSettings } = currentConfiguration;
    
    if (!llmSettings.provider || llmSettings.provider.trim() === '') {
      errors.push('LLM provider is required');
    }
    
    if (!llmSettings.model || llmSettings.model.trim() === '') {
      errors.push('LLM model is required');
    }
    
    if (llmSettings.temperature !== undefined && (llmSettings.temperature < 0 || llmSettings.temperature > 2)) {
      errors.push('Temperature must be between 0 and 2');
    }
    
    if (llmSettings.max_tokens !== undefined && (llmSettings.max_tokens <= 0 || llmSettings.max_tokens > 100000)) {
      errors.push('Max tokens must be between 1 and 100,000');
    }
    
    // Add these errors to the simulation step as LLM settings are part of simulation configuration
    setValidationErrors(prev => ({ ...prev, [WizardStep.Simulation]: errors }));
    return errors.length === 0;
  };

  const handleNext = () => {
    setCurrentStep((prev) => {
      // Validate current step before proceeding
      let isValid = true;
      
      switch (prev) {
        case WizardStep.Simulation:
          isValid = validateSimulationStep() && validateLLMSettings();
          break;
        case WizardStep.Agents:
          isValid = validateAgentsStep();
          break;
        case WizardStep.Experiment:
          isValid = validateExperimentStep();
          break;
        case WizardStep.Constraints:
          if (!isConstraintConfigValid) {
            alert('Please resolve all constraint configuration errors and warnings before proceeding.');
            return prev;
          }
          break;
      }
      
      if (!isValid) {
        return prev; // Stay on the current step if validation fails
      }

      // Logic for step progression based on simulation type
      if (prev === WizardStep.SimulationType) {
        return WizardStep.Templates;
      }
      return (prev + 1) as WizardStep;
    });
  };

  const handleBack = () => {
    setCurrentStep((prev) => {
      if (prev === WizardStep.Templates && !isExperiment) {
        return WizardStep.SimulationType;
      }
      return (prev - 1) as WizardStep;
    });
  };

  const handleSelectTemplate = (config: Configuration) => {
    // Sanitize the configuration data before applying it
    const sanitizedConfig = sanitizeExternalData(config);
    setCurrentConfiguration(sanitizedConfig);
    setCurrentStep(WizardStep.Simulation); // Go to the Simulation step after template selection
  };

  const handleConstraintConfigChange = (config: ConstraintConfig, isValid: boolean) => {
    // Sanitize the constraint configuration before applying it
    const sanitizedConfig = sanitizeExternalData(config);
    setCurrentConfiguration((prev) => ({
      ...prev,
      constraints: sanitizedConfig,
    }));
    setIsConstraintConfigValid(isValid);
  };

  const handleTierSelect = (tierName: ConstraintConfig['tier']) => {
    // Logic to update the tier in the current configuration and set default limits based on the selected tier
    const newTier = constraintTiers.find(tier => tier.name === tierName);
    if (newTier) {
      setCurrentConfiguration(prev => ({
        ...prev,
        constraints: {
          ...prev.constraints,
          tier: newTier.name as ConstraintConfig['tier'],
          budgetLimitUSD: newTier.budgetLimit,
          tokenLimit: newTier.tokenLimit,
          rateLimitPerMinute: newTier.rateLimit,
          memoryLimitMB: newTier.memoryLimit,
        },
      }));
    }
  };

  const handleSubmit = () => {
    // Logic to send configurations to backend
    console.log("Submitting Configuration:", currentConfiguration);
    // Call API service here
    alert("Simulation/Experiment configuration submitted!");
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white shadow-lg rounded-lg">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">
        Simulation Configuration Wizard
      </h1>

      <div className="mb-8">
        <nav aria-label="Progress" className="mb-8">
            <ol role="list" className="flex items-center">
                {Object.entries(WizardStep).filter(([, value]) => typeof value === 'number').map(([label, value], index) => (
                    <li key={label} className="relative pr-8 sm:pr-20">
                    {index !== 0 && <div className="absolute inset-0 flex items-center" aria-hidden="true"><div className={`h-0.5 w-full ${currentStep >= value ? 'bg-blue-600' : 'bg-gray-200'}`} /></div>}
                    <a href="#" className="relative flex flex-col items-center" aria-current={currentStep === value ? 'step' : undefined}>
                        <div className={`w-8 h-8 flex items-center justify-center rounded-full border-2 ${currentStep === value ? 'bg-blue-600 border-blue-600' : currentStep > value ? 'bg-blue-600 border-blue-600' : 'border-gray-300 bg-white'}`}>
                        {currentStep === value ? (
                            <span className="h-2.5 w-2.5 rounded-full bg-white" aria-hidden="true" />
                        ) : currentStep > value ? (
                            <svg className="h-5 w-5 text-white" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                                <path fillRule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z" clipRule="evenodd" />
                            </svg>
                        ) : (
                            <span className="text-gray-500">{index + 1}</span>
                        )}
                        </div>
                        <span className={`mt-2 text-sm font-medium ${currentStep === value ? 'text-blue-600' : 'text-gray-900'}`}>{label.replace(/([A-Z])/g, ' $1').trim()}</span>
                    </a>
                    </li>
                ))}
            </ol>
        </nav>
      </div>

      <div className="py-8">
        {currentStep === WizardStep.SimulationType && (
          <div>
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Select Simulation Type</h2>
            <div className="flex space-x-4">
              <button
                className={`py-3 px-6 rounded-lg text-lg font-medium transition-all duration-200 ${
                  !isExperiment
                    ? 'bg-blue-600 text-white shadow-md'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                onClick={() => setIsExperiment(false)}
              >
                Single Simulation Run
              </button>
              <button
                className={`py-3 px-6 rounded-lg text-lg font-medium transition-all duration-200 ${
                  isExperiment
                    ? 'bg-blue-600 text-white shadow-md'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                onClick={() => setIsExperiment(true)}
              >
                Experiment (Parameter Sweep)
              </button>
            </div>
          </div>
        )}

        {currentStep === WizardStep.Templates && (
          <TemplateSelectionForm onSelectTemplate={handleSelectTemplate} onNext={handleNext} />
        )}

        {currentStep === WizardStep.Simulation && (
          <SimulationConfigForm
            config={currentConfiguration.simulationSettings}
            onConfigChange={(newSettings: SimulationSettings) => {
              // Sanitize the simulation settings before applying them
              const sanitizedSettings = sanitizeExternalData(newSettings);
              setCurrentConfiguration((prev) => ({
                ...prev,
                simulationSettings: sanitizedSettings,
              }));
            }}
          />
        )}

        {currentStep === WizardStep.Agents && (
          <>
            {currentConfiguration.agentConfigs.length === 0 ? (
              <div>No agents configured for this template. Please go back to select a template with agents or adjust manually.</div>
            ) : (
              <AgentSelectionForm
                // Only showing the first agent for a simple demo since AgentSelectionForm expects a single agent
                agentConfig={currentConfiguration.agentConfigs[0]}
                onAgentConfigChange={(newAgentConfig: TemplateAgentConfig) =>
                  setCurrentConfiguration((prev) => {
                    // Sanitize the agent configuration before applying it
                    const sanitizedAgentConfig = sanitizeExternalData(newAgentConfig);
                    const updatedAgentConfigs = [...prev.agentConfigs];
                    if (updatedAgentConfigs[0]) {
                      updatedAgentConfigs[0] = { ...sanitizedAgentConfig };
                    }
                    return { ...prev, agentConfigs: updatedAgentConfigs };
                  })
                }
              />
            )}
          </>
        )}

        {currentStep === WizardStep.Constraints && (
          <div className="space-y-6">
            <ConstraintConfigForm
              initialConfig={currentConfiguration.constraints}
              onConfigChange={handleConstraintConfigChange}
            />
            <TierSelector
              currentConfig={currentConfiguration.constraints}
              onTierSelect={handleTierSelect as (tierName: string) => void}
            />
            {/* Optionally, display ConstraintMonitor here for immediate feedback based on current constraints */}
            {/* <ConstraintMonitor usage={{...}} /> */}
          </div>
        )}

        {currentStep === WizardStep.Experiment && (
          <ExperimentConfigForm
            config={currentConfiguration.experimentSettings}
            onConfigChange={(newSettings) => {
              // Sanitize the experiment settings before applying them
              const sanitizedSettings = sanitizeExternalData(newSettings);
              setCurrentConfiguration((prev) => ({
                ...prev,
                experimentSettings: { ...sanitizedSettings, outputConfig: sanitizedSettings.outputConfig || {} }, // Ensure outputConfig is not undefined
              }));
            }}
          />
        )}


        {currentStep === WizardStep.Review && (
          <div>
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Review & Launch</h2>
            <div className="bg-gray-50 p-4 rounded-md mb-6">
              <h3 className="text-xl font-medium text-gray-800 mb-2">Summary</h3>
              <pre className="bg-gray-100 p-3 rounded-md text-sm whitespace-pre-wrap">
                {JSON.stringify(currentConfiguration, null, 2)}
              </pre>
            </div>
            {/* Add Save/Load/Save as Template buttons */}
            <button
              type="button"
              className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              onClick={() => {
                // Sanitize configuration before saving
                const sanitizedConfig = sanitizeExternalData(currentConfiguration);
                localStorage.setItem('fbaBenchConfiguration', JSON.stringify(sanitizedConfig));
                alert('Configuration saved to local storage!');
              }}
            >
              Save Configuration
            </button>
            <button
              type="button"
              className="ml-4 px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2"
              onClick={() => {
                const savedConfig = localStorage.getItem('fbaBenchConfiguration');
                if (savedConfig) {
                  try {
                    const parsedConfig: Configuration = JSON.parse(savedConfig);
                    // Sanitize configuration after loading
                    const sanitizedConfig = sanitizeExternalData(parsedConfig);
                    setCurrentConfiguration(sanitizedConfig);
                    alert('Configuration loaded from local storage!');
                    setCurrentStep(WizardStep.SimulationType); // Reset to first step after loading
                  } catch (e) {
                    alert('Failed to load configuration: Invalid JSON');
                    console.error('Failed to parse saved configuration', e);
                  }
                } else {
                  alert('No saved configuration found.');
                }
              }}
            >
              Load Configuration
            </button>
            <button
              type="button"
              className="ml-4 px-4 py-2 bg-purple-500 text-white rounded-md hover:bg-purple-600 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2"
              onClick={() => {
                const templateName = prompt("Enter a name for your template:");
                if (templateName) {
                  // Sanitize user input for template name
                  const sanitizedName = sanitizeUserInput(templateName);
                  if (!sanitizedName || sanitizedName.trim() === '') {
                    alert("Template name cannot be empty or contain invalid characters.");
                    return;
                  }
                  
                  if (!currentConfiguration.simulationSettings.simulationName || currentConfiguration.agentConfigs.length === 0) {
                    alert("Cannot save as template: Please provide a simulation name and at least one agent configuration.");
                    return;
                  }
                  
                  // Sanitize configuration before saving as template
                  const sanitizedConfig = sanitizeExternalData(currentConfiguration);
                  const newTemplate: Template = {
                    id: `custom-${new Date().getTime()}`,
                    name: sanitizedName,
                    description: `User-saved template: ${sanitizedName}`,
                    useCase: 'Custom saved configuration',
                    configuration: sanitizedConfig,
                  };
                  const userTemplates: Template[] = JSON.parse(localStorage.getItem('fbaBenchUserTemplates') || '[]');
                  userTemplates.push(newTemplate);
                  localStorage.setItem('fbaBenchUserTemplates', JSON.stringify(userTemplates));
                  alert(`Template '${sanitizedName}' saved locally.`);
                }
              }}
            >
              Save as Template
            </button>
          </div>
        )}
      </div>

      <div className="flex justify-between mt-8">
        <button
          onClick={handleBack}
          disabled={currentStep === WizardStep.SimulationType}
          className="px-6 py-2 border border-gray-300 rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Previous
        </button>
        {currentStep === WizardStep.Review ? (
          <button
            onClick={handleSubmit}
            className="px-6 py-2 border border-transparent rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
          >
            Launch Simulation
          </button>
        ) : (
          <button
            onClick={() => handleNext()}
            className="px-6 py-2 border border-transparent rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Next
          </button>
        )}
      </div>
    </div>
  );
}
