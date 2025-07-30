import React, { useState } from 'react';
import { SimulationConfigForm } from '../components/SimulationConfigForm';
import { AgentSelectionForm } from '../components/AgentSelectionForm';
import { ExperimentConfigForm } from '../components/ExperimentConfigForm';
import TemplateSelectionForm from '../components/TemplateSelectionForm'; // Import the new component
import type { Configuration, SimulationSettings, TemplateAgentConfig, Template } from '../types'; // Import only used types explicitly

const WizardStep = {
  SimulationType: 0,
  Templates: 1, // New step
  BasicParameters: 2, // Shifted
  AgentSelection: 3, // Shifted
  ConstraintConfiguration: 4, // Shifted
  OutputSettings: 5, // Shifted
  ReviewAndLaunch: 6 // Shifted
} as const;

type WizardStep = typeof WizardStep[keyof typeof WizardStep];

export function ConfigurationWizard() {
  const [currentStep, setCurrentStep] = useState<WizardStep>(WizardStep.SimulationType);
  const [isExperiment, setIsExperiment] = useState<boolean>(false); // Explicitly define type
  const [currentConfiguration, setCurrentConfiguration] = useState<Configuration>({
    simulationSettings: {
      simulationName: '',
      description: '',
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
      tokenLimits: { grok4: 0 },
    },
    experimentSettings: {
      experimentName: '',
      description: '',
      iterations: 1,
      parallelRuns: 1,
      parameters: [],
    }
  });

  const handleNext = () => {
    setCurrentStep((prev) => {
        if (prev === WizardStep.SimulationType) {
            // After SimulationType, always go to Templates
            return WizardStep.Templates;
        }
        return (prev + 1) as WizardStep;
    });
  };

  const handleBack = () => {
    setCurrentStep((prev) => {
        if (prev === WizardStep.Templates && !isExperiment) {
            // If coming from Templates, go back to SimulationType if not an experiment
            return WizardStep.SimulationType;
        }
        return (prev - 1) as WizardStep;
    });
  };
  
  const handleSelectTemplate = (config: Configuration) => {
    setCurrentConfiguration(config);
    // After selecting a template, decide next step based on template content or force next.
    // For now, always proceed to the next step (BasicParameters)
    setCurrentStep(WizardStep.BasicParameters);
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

        {currentStep === WizardStep.BasicParameters && (
          <SimulationConfigForm
            config={currentConfiguration.simulationSettings}
            onConfigChange={(newSettings: SimulationSettings) =>
              setCurrentConfiguration((prev) => ({
                ...prev,
                simulationSettings: newSettings,
              }))
            }
          />
        )}

        {currentStep === WizardStep.AgentSelection && (
          <>
            {currentConfiguration.agentConfigs.length === 0 ? (
                <div>No agents configured for this template. Please go back to select a template with agents or adjust manually.</div>
            ) : (
                <AgentSelectionForm
                    // Only showing the first agent for a simple demo since AgentSelectionForm expects a single agent
                    agentConfig={currentConfiguration.agentConfigs[0]}
                    onAgentConfigChange={(newAgentConfig: TemplateAgentConfig) =>
                        setCurrentConfiguration((prev) => {
                            const updatedAgentConfigs = [...prev.agentConfigs];
                            if (updatedAgentConfigs[0]) {
                                updatedAgentConfigs[0] = { ...newAgentConfig };
                            }
                            return { ...prev, agentConfigs: updatedAgentConfigs };
                        })
                    }
                />
            )}
          </>
        )}
        
        {currentStep === WizardStep.ConstraintConfiguration && (
          <div>
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Constraint & Tier Configuration</h2>
            {/* You would usually have a form component here */}
            <p className="text-gray-600">
              Configure budget constraints, token limits, and performance tiers for:
              <pre>{JSON.stringify(currentConfiguration.constraints, null, 2)}</pre>
            </p>
          </div>
        )}

        {currentStep === WizardStep.OutputSettings && (
          <div>
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Output & Monitoring Settings</h2>
            {/* You would usually have a form component here */}
            <p className="text-gray-600">
              Define how simulation data and results are saved and displayed for:
              <pre>{JSON.stringify(currentConfiguration.experimentSettings?.outputConfig, null, 2)}</pre>
            </p>
          </div>
        )}

        {currentStep === WizardStep.ReviewAndLaunch && (
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
                // Save currentConfiguration to local storage
                localStorage.setItem('fbaBenchConfiguration', JSON.stringify(currentConfiguration));
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
                    setCurrentConfiguration(parsedConfig);
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
                    // Basic validation: Ensure minimal settings are present
                    if (!currentConfiguration.simulationSettings.simulationName || currentConfiguration.agentConfigs.length === 0) {
                        alert("Cannot save as template: Please provide a simulation name and at least one agent configuration.");
                        return;
                    }
                    const newTemplate: Template = {
                        id: `custom-${new Date().getTime()}`, // Use getTime() for unique id
                        name: templateName,
                        description: `User-saved template: ${templateName}`,
                        useCase: 'Custom saved configuration',
                        configuration: currentConfiguration,
                    };
                    // Optionally, persist user templates in local storage
                    const userTemplates: Template[] = JSON.parse(localStorage.getItem('fbaBenchUserTemplates') || '[]');
                    userTemplates.push(newTemplate);
                    localStorage.setItem('fbaBenchUserTemplates', JSON.stringify(userTemplates));
                    alert(`Template '${templateName}' saved locally.`);
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
          disabled={currentStep === WizardStep.SimulationType} // Disable if on the first step
          className="px-6 py-2 border border-gray-300 rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Previous
        </button>
        {currentStep === WizardStep.ReviewAndLaunch ? (
          <button
            onClick={handleSubmit}
            className="px-6 py-2 border border-transparent rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
          >
            Launch Simulation
          </button>
        ) : (
          <button
            onClick={() => {
              if (currentStep === WizardStep.SimulationType) {
                // After selecting simulation type, move to templates step if it's not an experiment
                // Or directly to BasicParameters if it is an experiment (skipping templates for experiments)
                // For now, always go to templates. We can refine this logic later.
                if (isExperiment) {
                  // If it's an experiment, perhaps we skip the general templates and go straight to experiment-specific setup
                  // For now, still go to templates to allow selecting a parameter sweep template
                  handleNext(); 
                } else {
                  handleNext();
                }
              } else {
                handleNext();
              }
            }}
            className="px-6 py-2 border border-transparent rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Next
          </button>
        )}
      </div>
    </div>
  );
}
