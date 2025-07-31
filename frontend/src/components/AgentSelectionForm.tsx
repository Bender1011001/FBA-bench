// frontend/src/components/AgentSelectionForm.tsx

import React, { useEffect, useState } from 'react';
import { apiService } from '../services/apiService';
import type { FrameworkData, AgentConfigurationResponse, TemplateAgentConfig } from '../types';
import HelpTooltip from './HelpTooltip';
import { getHelpText } from '../data/helpContent';
import { validateField, validateForm, agentValidationSchema, type AgentValidationRules, type AgentFormData } from '../utils/validators';

interface AgentSelectionFormProps {
  agentConfig?: TemplateAgentConfig;
  onAgentConfigChange: (newConfig: TemplateAgentConfig) => void;
  onValidationChange?: (isValid: boolean) => void; // Callback to notify parent of validation status
}

export function AgentSelectionForm({ agentConfig, onAgentConfigChange, onValidationChange }: AgentSelectionFormProps) {
  const [availableFrameworks, setAvailableFrameworks] = useState<string[]>([]);
  const [availableAgents, setAvailableAgents] = useState<AgentConfigurationResponse[]>([]);
  const [selectedFramework, setSelectedFramework] = useState<string>(agentConfig?.agentType || '');
  const [selectedAgentType, setSelectedAgentType] = useState<string>(agentConfig?.agentName || '');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Internal state for LLM config, since TemplateAgentConfig has flattened LLM properties
  const [llmProvider, setLlmProvider] = useState<string>(agentConfig?.llmInterface || '');
  const [llmModel, setLlmModel] = useState<string>(agentConfig?.model || '');
  const [llmTemperature, setLlmTemperature] = useState<number | string | undefined>(agentConfig?.temperature ?? undefined); // Can be undefined for empty input
  const [llmMaxTokens, setLlmMaxTokens] = useState<number | string | undefined>(agentConfig?.max_tokens ?? undefined); // Can be undefined for empty input

  // Initialize numeric states with default values if agentConfig has no value or it's 0 (handle 0 explicitly)
  useEffect(() => {
    if (agentConfig?.temperature !== undefined) {
      setLlmTemperature(agentConfig.temperature);
    } else if (llmTemperature === undefined) { // Only set default if not already set or implicitly undefined due to agentConfig
      setLlmTemperature(0.7);
    }
    if (agentConfig?.max_tokens !== undefined) {
      setLlmMaxTokens(agentConfig.max_tokens);
    } else if (llmMaxTokens === undefined) { // Only set default if not already set
      setLlmMaxTokens(1000);
    }
  }, [agentConfig?.temperature, agentConfig?.max_tokens]);


  // State for validation errors
  const [validationErrors, setValidationErrors] = useState<Record<keyof AgentFormData, string | undefined>>(() => {
    const initialErrors: Record<keyof AgentFormData, string | undefined> = {};
    for (const key in agentValidationSchema) {
      if (Object.prototype.hasOwnProperty.call(agentValidationSchema, key)) {
        initialErrors[key as keyof AgentFormData] = undefined;
      }
    }
    return initialErrors;
  });

  const mapAgentDataToValidation = (): AgentFormData => ({
    framework: selectedFramework,
    llmModel: llmModel,
    temperature: llmTemperature === '' || llmTemperature === undefined ? (0 as number) : (llmTemperature as number), // Ensure number for validation
    maxTokens: llmMaxTokens === '' || llmMaxTokens === undefined ? (0 as number) : (llmMaxTokens as number), // Ensure number for validation
  });

  // Effect to fetch agent data
  useEffect(() => {
    const fetchAgentData = async () => {
      setLoading(true);
      setError(null);
      try {
        const frameworksResponse = await apiService.get<FrameworkData>('/api/v1/agents/frameworks');
        setAvailableFrameworks(frameworksResponse.data.frameworks);

        const agentsResponse = await apiService.get<AgentConfigurationResponse[]>('/api/v1/agents/available');
        setAvailableAgents(agentsResponse.data);
      } catch (err) {
        console.error("Failed to fetch agent data:", err);
        setError("Failed to load agent data. Please check the API server.");
      } finally {
        setLoading(false);
      }
    };
    fetchAgentData();
  }, []);

  // Effect to re-validate form and notify parent whenever relevant config changes
  useEffect(() => {
    const currentData = mapAgentDataToValidation();
    const errors = validateForm(currentData, agentValidationSchema);

    // Filter out errors for fields that are not currently displayed (e.g., LLM fields when no agent is selected)
    const filteredErrors: Record<keyof AgentFormData, string | undefined> = {} as Record<keyof AgentFormData, string | undefined>;
    for (const key in errors) {
      if (Object.prototype.hasOwnProperty.call(errors, key)) {
        const fieldName = key as keyof AgentFormData;
        // Only include validation errors for the currently visible/relevant fields
        // framework is always relevant
        // llmModel, temperature, maxTokens are relevant only if a framework is selected
        if (fieldName === 'framework' || (selectedFramework && (fieldName === 'llmModel' || fieldName === 'temperature' || fieldName === 'maxTokens'))) {
          filteredErrors[fieldName] = errors[fieldName];
        } else {
            filteredErrors[fieldName] = undefined; // Clear errors for hidden/irrelevant fields
        }
      }
    }

    setValidationErrors(filteredErrors);
    const isValid = Object.values(filteredErrors).every(error => !error);
    if (onValidationChange) {
      onValidationChange(isValid);
    }

    // Update the parent's agentConfig
    const newConfig: TemplateAgentConfig = {
      agentName: selectedAgentType,
      agentType: selectedFramework,
      model: llmModel,
      max_tokens: llmMaxTokens as number, // Ensure number for TemplateAgentConfig
      temperature: llmTemperature as number, // Ensure number for TemplateAgentConfig
      llmInterface: llmProvider,
      role: agentConfig?.role || '',
      behavior: agentConfig?.behavior || '',
    };
    onAgentConfigChange(newConfig);

  }, [selectedFramework, selectedAgentType, llmModel, llmMaxTokens, llmTemperature, llmProvider, availableAgents, agentConfig, onAgentConfigChange, onValidationChange]);


  const handleAgentTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedAgentType(e.target.value);
  };

  const handleFrameworkChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newValue = e.target.value;
    setSelectedFramework(newValue);
    setSelectedAgentType(''); // Reset agent type when framework changes
    // Validate individual field immediately
    setValidationErrors(prevErrors => ({
      ...prevErrors,
      framework: validateField(newValue, agentValidationSchema.framework),
    }));
  };

  const handleLlmDetailStringChange = (e: React.ChangeEvent<HTMLInputElement>, setter: React.Dispatch<React.SetStateAction<string>>, fieldName: keyof AgentFormData) => {
    const value = e.target.value;
    setter(value);
    setValidationErrors(prevErrors => ({
      ...prevErrors,
      [fieldName]: validateField(value, agentValidationSchema[fieldName as keyof AgentValidationRules]),
    }));
  };

  const handleLlmDetailNumberChange = (e: React.ChangeEvent<HTMLInputElement>, setter: React.Dispatch<React.SetStateAction<number | string | undefined>>, fieldName: keyof AgentFormData) => {
    const value = e.target.value;
    let parsedValue: number | string | undefined;
    if (value === '') {
        parsedValue = undefined; // Allow empty string for number fields initially
    } else {
        const numValue = parseFloat(value);
        if (isNaN(numValue)) {
            parsedValue = value; // Keep invalid string to show error
        } else {
            parsedValue = numValue;
        }
    }
    setter(parsedValue);

    setValidationErrors(prevErrors => ({
      ...prevErrors,
      [fieldName]: validateField(parsedValue, agentValidationSchema[fieldName as keyof AgentValidationRules]),
    }));
  };


  if (loading) {
    return <div className="text-center py-8">Loading agent data...</div>;
  }

  if (error) {
    return <div className="text-center py-8 text-red-600">{error}</div>;
  }

  const agentsForSelectedFramework = availableAgents.filter(
    (agent) => agent.agent_framework === selectedFramework
  );

  const currentAgentExample = agentsForSelectedFramework.find(
    (agent) => agent.agent_type === selectedAgentType
  );

  const getError = (fieldName: keyof AgentFormData) => validationErrors[fieldName];
  const hasError = (fieldName: keyof AgentFormData) => !!getError(fieldName);

  return (
    <div>
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        Agent & Bot Selection
      </h2>
      <div className="space-y-4">
        <div>
          <label htmlFor="framework" className={`block text-sm font-medium ${hasError('framework') ? 'text-red-600' : 'text-gray-700'}`}>
            Agent Framework *
            <HelpTooltip content={getHelpText('agent', 'framework')} />
          </label>
          <select
            name="framework"
            id="framework"
            value={selectedFramework}
            onChange={handleFrameworkChange}
            onBlur={handleFrameworkChange} // Validate on blur
            className={`mt-1 block w-full border ${hasError('framework') ? 'border-red-500' : 'border-gray-300'} rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500`}
          >
            <option value="">Select a framework</option>
            {availableFrameworks.map((framework) => (
              <option key={framework} value={framework}>
                {framework.toUpperCase()}
              </option>
            ))}
          </select>
          {hasError('framework') && (
            <p className="mt-1 text-sm text-red-600">{getError('framework')}</p>
          )}
        </div>

        {selectedFramework && (
          <div>
            <label htmlFor="agentType" className="block text-sm font-medium text-gray-700">
              Agent Type
              <HelpTooltip content={getHelpText('agent', 'agentType')} />
            </label>
            <select
              name="agent_type"
              id="agentType"
              value={selectedAgentType}
              onChange={handleAgentTypeChange}
              onBlur={handleAgentTypeChange} // Validation on blur
              className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">Select an agent type</option>
              {agentsForSelectedFramework.map((agent) => (
                <option key={agent.agent_type} value={agent.agent_type}>
                  {agent.agent_type}
                </option>
              ))}
            </select>
          </div>
        )}

        {selectedAgentType && (
          <div className="bg-gray-50 p-4 rounded-md">
            <h3 className="text-lg font-medium text-gray-800 mb-2">Agent Details</h3>
            <p className="text-sm text-gray-600 mb-4">
              {currentAgentExample?.description}
            </p>

            {/* Render LLM Configuration */}
            <div className="mb-4">
              <h4 className="text-md font-medium text-gray-700 mb-2">LLM Configuration</h4>
              <div className="space-y-2">
                <div>
                  <label htmlFor="llmProvider" className="block text-xs font-medium text-gray-600">
                    Provider
                    <HelpTooltip content={getHelpText('agent', 'llmProvider')} />
                  </label>
                  <input
                    type="text"
                    name="llmInterface" // Name matches TemplateAgentConfig
                    id="llmProvider"
                    value={llmProvider}
                    onChange={(e) => handleLlmDetailStringChange(e, setLlmProvider, 'llmModel')} // Using llmModel as temp placeholder since no rule for provider
                    onBlur={(e) => handleLlmDetailStringChange(e, setLlmProvider, 'llmModel')} // No direct validation schema rule for provider yet
                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-1 px-2 text-sm"
                  />
                  {/* No specific hasError check for llmProvider, as no rule exists yet. */}
                </div>
                <div>
                  <label htmlFor="llmModel" className={`block text-xs font-medium ${hasError('llmModel') ? 'text-red-600' : 'text-gray-600'}`}>
                    Model *
                    <HelpTooltip content={getHelpText('agent', 'llmModel')} />
                  </label>
                  <input
                    type="text"
                    name="model" // Name matches TemplateAgentConfig
                    id="llmModel"
                    value={llmModel}
                    onChange={(e) => handleLlmDetailStringChange(e, setLlmModel, 'llmModel')}
                    onBlur={(e) => handleLlmDetailStringChange(e, setLlmModel, 'llmModel')}
                    className={`mt-1 block w-full border ${hasError('llmModel') ? 'border-red-500' : 'border-gray-300'} rounded-md shadow-sm py-1 px-2 text-sm`}
                  />
                  {hasError('llmModel') && (
                    <p className="mt-1 text-sm text-red-600">{getError('llmModel')}</p>
                  )}
                </div>
                <div>
                  <label htmlFor="llmTemperature" className={`block text-xs font-medium ${hasError('temperature') ? 'text-red-600' : 'text-gray-600'}`}>
                    Temperature
                    <HelpTooltip content={getHelpText('agent', 'temperature')} />
                  </label>
                  <input
                    type="number"
                    name="temperature" // Name matches TemplateAgentConfig
                    id="llmTemperature"
                    value={llmTemperature ?? ''} // Use ?? '' for proper display of undefined/null
                    onChange={(e) => handleLlmDetailNumberChange(e, setLlmTemperature, 'temperature')}
                    onBlur={(e) => handleLlmDetailNumberChange(e, setLlmTemperature, 'temperature')}
                    step="0.01"
                    min="0"
                    max="2"
                    className={`mt-1 block w-full border ${hasError('temperature') ? 'border-red-500' : 'border-gray-300'} rounded-md shadow-sm py-1 px-2 text-sm`}
                  />
                  {hasError('temperature') && (
                    <p className="mt-1 text-sm text-red-600">{getError('temperature')}</p>
                  )}
                </div>
                <div>
                  <label htmlFor="llmMaxTokens" className={`block text-xs font-medium ${hasError('maxTokens') ? 'text-red-600' : 'text-gray-600'}`}>
                    Max Tokens *
                    <HelpTooltip content={getHelpText('agent', 'max_tokens')} />
                  </label>
                  <input
                    type="number"
                    name="max_tokens" // Name matches TemplateAgentConfig
                    id="llmMaxTokens"
                    value={llmMaxTokens ?? ''} // Use ?? '' for proper display of undefined/null
                    onChange={(e) => handleLlmDetailNumberChange(e, setLlmMaxTokens, 'maxTokens')}
                    onBlur={(e) => handleLlmDetailNumberChange(e, setLlmMaxTokens, 'maxTokens')}
                    min="1"
                    max="32000"
                    className={`mt-1 block w-full border ${hasError('maxTokens') ? 'border-red-500' : 'border-gray-300'} rounded-md shadow-sm py-1 px-2 text-sm`}
                  />
                   {hasError('maxTokens') && (
                    <p className="mt-1 text-sm text-red-600">{getError('maxTokens')}</p>
                  )}
                </div>
              </div>
            </div>

            {/* Removed CrewAI and Custom config for simplicity to match TemplateAgentConfig's flattened structure */}
            {/* If these are needed, TemplateAgentConfig needs to be expanded or a mapping function introduced */}
            
            {/* Displaying agent's specific role and behavior from TemplateAgentConfig */}
            <div className="mb-4">
              <h4 className="text-md font-medium text-gray-700 mb-2">Agent-Specific Details (from Template)</h4>
              <div className="space-y-2">
                <div>
                  <label className="block text-xs font-medium text-gray-600">
                    Role
                    <HelpTooltip content={getHelpText('agent', 'role')} />
                  </label>
                  <p className="text-sm text-gray-900">{agentConfig?.role || 'N/A'}</p>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-600">
                    Behavior
                    <HelpTooltip content={getHelpText('agent', 'behavior')} />
                  </label>
                  <p className="text-sm text-gray-900">{agentConfig?.behavior || 'N/A'}</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}