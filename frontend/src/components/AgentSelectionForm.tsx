import React, { useEffect, useState } from 'react';
import { apiService } from '../services/apiService';
import type { AgentRunnerConfig, FrameworkData, AgentConfigurationResponse, TemplateAgentConfig } from '../types';

interface AgentSelectionFormProps {
  // Now expects a single TemplateAgentConfig (or null/undefined) for editing
  // or an array, in which case you need to specify which one is being edited.
  // For simplicity, let's assume it gets a single agent, and main wizard manages the array.
  agentConfig?: TemplateAgentConfig; 
  onAgentConfigChange: (newConfig: TemplateAgentConfig) => void;
}

export function AgentSelectionForm({ agentConfig, onAgentConfigChange }: AgentSelectionFormProps) {
  const [availableFrameworks, setAvailableFrameworks] = useState<string[]>([]);
  const [availableAgents, setAvailableAgents] = useState<AgentConfigurationResponse[]>([]);
  // Initialize with values from agentConfig if provided, otherwise empty
  const [selectedFramework, setSelectedFramework] = useState<string>(agentConfig?.agentType || '');
  const [selectedAgentType, setSelectedAgentType] = useState<string>(agentConfig?.agentName || ''); // Use agentName as agent_type from Template
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Internal state for LLM config, since TemplateAgentConfig has flattened LLM properties
  const [llmProvider, setLlmProvider] = useState(agentConfig?.llmInterface || '');
  const [llmModel, setLlmModel] = useState(agentConfig?.model || '');
  const [llmTemperature, setLlmTemperature] = useState(agentConfig?.temperature || 0.7);
  const [llmMaxTokens, setLlmMaxTokens] = useState(agentConfig?.max_tokens || 1000);


  useEffect(() => {
    const fetchAgentData = async () => {
      setLoading(true);
      setError(null);
      try {
        const frameworksResponse = await apiService.get<FrameworkData>('/api/v1/agents/frameworks');
        setAvailableFrameworks(frameworksResponse.frameworks);

        const agentsResponse = await apiService.get<AgentConfigurationResponse[]>('/api/v1/agents/available');
        setAvailableAgents(agentsResponse);
      } catch (err) {
        console.error("Failed to fetch agent data:", err);
        setError("Failed to load agent data. Please check the API server.");
      } finally {
        setLoading(false);
      }
    };
    fetchAgentData();
  }, []);

  useEffect(() => {
    // When framework or agent type changes, update the agentConfig
    // We need to map from AgentRunnerConfig (from API) back to TemplateAgentConfig for this form.
    const newConfig: TemplateAgentConfig = {
      agentName: selectedAgentType,
      agentType: selectedFramework,
      model: llmModel,
      max_tokens: llmMaxTokens,
      temperature: llmTemperature,
      llmInterface: llmProvider,
      role: agentConfig?.role || '',     // Keep existing if not changed
      behavior: agentConfig?.behavior || '', // Keep existing if not changed
    };

    const matchingAgent = availableAgents.find(
      (agent) => agent.agent_framework === selectedFramework && agent.agent_type === selectedAgentType
    );

    if (matchingAgent) {
      // Pre-populate LLM details from the example config if available
      newConfig.model = matchingAgent.example_config.llm_config?.model || newConfig.model;
      newConfig.max_tokens = matchingAgent.example_config.llm_config?.max_tokens || newConfig.max_tokens;
      newConfig.temperature = matchingAgent.example_config.llm_config?.temperature || newConfig.temperature;
      newConfig.llmInterface = matchingAgent.example_config.llm_config?.provider || newConfig.llmInterface;

      // Update internal LLM states based on the matched agent or existing config
      setLlmModel(newConfig.model);
      setLlmMaxTokens(newConfig.max_tokens);
      setLlmTemperature(newConfig.temperature);
      setLlmProvider(newConfig.llmInterface);
    }
    
    // Only call onAgentConfigChange if there's a meaningful change from initial
    // This needs to be a deep comparison, but for simplicity, let's just send always for now.
    onAgentConfigChange(newConfig);
  }, [selectedFramework, selectedAgentType, llmModel, llmMaxTokens, llmTemperature, llmProvider, availableAgents]);


  const handleAgentTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedAgentType(e.target.value);
  };
   const handleFrameworkChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedFramework(e.target.value);
    setSelectedAgentType(''); // Reset agent type when framework changes
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

  return (
    <div>
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        Agent & Bot Selection
      </h2>
      <div className="space-y-4">
        <div>
          <label htmlFor="framework" className="block text-sm font-medium text-gray-700">
            Agent Framework
          </label>
          <select
            name="framework"
            id="framework"
            value={selectedFramework}
            onChange={handleFrameworkChange}
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">Select a framework</option>
            {availableFrameworks.map((framework) => (
              <option key={framework} value={framework}>
                {framework.toUpperCase()}
              </option>
            ))}
          </select>
        </div>

        {selectedFramework && (
          <div>
            <label htmlFor="agentType" className="block text-sm font-medium text-gray-700">
              Agent Type
            </label>
            <select
              name="agent_type"
              id="agentType"
              value={selectedAgentType}
              onChange={handleAgentTypeChange}
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
                  </label>
                  <input
                    type="text"
                    name="llmInterface" // Name matches TemplateAgentConfig
                    id="llmProvider"
                    value={llmProvider}
                    onChange={(e) => setLlmProvider(e.target.value)}
                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-1 px-2 text-sm"
                  />
                </div>
                <div>
                  <label htmlFor="llmModel" className="block text-xs font-medium text-gray-600">
                    Model
                  </label>
                  <input
                    type="text"
                    name="model" // Name matches TemplateAgentConfig
                    id="llmModel"
                    value={llmModel}
                    onChange={(e) => setLlmModel(e.target.value)}
                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-1 px-2 text-sm"
                  />
                </div>
                <div>
                  <label htmlFor="llmTemperature" className="block text-xs font-medium text-gray-600">
                    Temperature
                  </label>
                  <input
                    type="number"
                    name="temperature" // Name matches TemplateAgentConfig
                    id="llmTemperature"
                    value={llmTemperature}
                    onChange={(e) => setLlmTemperature(parseFloat(e.target.value))}
                    step="0.01"
                    min="0"
                    max="1"
                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-1 px-2 text-sm"
                  />
                </div>
                <div>
                  <label htmlFor="llmMaxTokens" className="block text-xs font-medium text-gray-600">
                    Max Tokens
                  </label>
                  <input
                    type="number"
                    name="max_tokens" // Name matches TemplateAgentConfig
                    id="llmMaxTokens"
                    value={llmMaxTokens}
                    onChange={(e) => setLlmMaxTokens(parseFloat(e.target.value))}
                    min="1"
                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-1 px-2 text-sm"
                  />
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
                  <label className="block text-xs font-medium text-gray-600">Role</label>
                  <p className="text-sm text-gray-900">{agentConfig?.role || 'N/A'}</p>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-600">Behavior</label>
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