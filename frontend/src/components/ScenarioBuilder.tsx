import React, { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';
import type { 
  ScenarioConfig, 
  AgentConfig, 
  ConfigurationTemplate,
  BenchmarkConfig 
} from '../types';
import { notificationService } from '../utils/notificationService';
import LoadingSpinner from './LoadingSpinner';
import ErrorBoundary from './ErrorBoundary';

interface ScenarioBuilderProps {
  onScenarioCreated?: (scenario: ScenarioConfig) => void;
  className?: string;
}

const ScenarioBuilder: React.FC<ScenarioBuilderProps> = ({ 
  onScenarioCreated, 
  className = '' 
}) => {
  const [templates, setTemplates] = useState<ConfigurationTemplate[]>([]);
  const [agents, setAgents] = useState<AgentConfig[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [selectedAgents, setSelectedAgents] = useState<string[]>([]);
  const [scenario, setScenario] = useState<Partial<ScenarioConfig>>({
    id: '',
    name: '',
    type: 'custom',
    description: '',
    config: {},
    enabled: true,
    priority: 1,
    parameters: {
      duration: 300,
      complexity: 'medium',
      domain: 'general',
      difficulty: 'medium'
    },
    metadata: {
      author: '',
      version: '1.0.0',
      tags: []
    }
  });
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'basic' | 'parameters' | 'metadata'>('basic');

  useEffect(() => {
    fetchTemplates();
    fetchAgents();
  }, []);

  useEffect(() => {
    if (selectedTemplate) {
      loadTemplate(selectedTemplate);
    }
  }, [selectedTemplate]);

  const fetchTemplates = async () => {
    setIsLoading(true);
    try {
      const response = await apiService.get<ConfigurationTemplate[]>('/benchmarking/templates');
      setTemplates(response.data);
    } catch (err) {
      console.error('Error fetching templates:', err);
      setError('Failed to fetch templates');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchAgents = async () => {
    try {
      const response = await apiService.get<AgentConfig[]>('/benchmarking/agents');
      setAgents(response.data);
    } catch (err) {
      console.error('Error fetching agents:', err);
      setError('Failed to fetch agents');
    }
  };

  const loadTemplate = (templateId: string) => {
    const template = templates.find(t => t.id === templateId);
    if (template) {
      setScenario(prev => ({
        ...prev,
        type: template.config.scenarios[0]?.type || 'custom',
        parameters: {
          ...prev.parameters,
          ...template.config.scenarios[0]?.parameters
        }
      }));
    }
  };

  const handleInputChange = (field: string, value: any) => {
    setScenario(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleParameterChange = (field: string, value: any) => {
    setScenario(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        [field]: value
      }
    }));
  };

  const handleMetadataChange = (field: string, value: any) => {
    setScenario(prev => ({
      ...prev,
      metadata: {
        ...prev.metadata,
        [field]: value
      }
    }));
  };

  const handleAgentToggle = (agentId: string) => {
    setSelectedAgents(prev => 
      prev.includes(agentId)
        ? prev.filter(id => id !== agentId)
        : [...prev, agentId]
    );
  };

  const handleTagsChange = (tagsString: string) => {
    const tags = tagsString.split(',').map(tag => tag.trim()).filter(Boolean);
    handleMetadataChange('tags', tags);
  };

  const validateScenario = (): boolean => {
    if (!scenario.name?.trim()) {
      setError('Scenario name is required');
      return false;
    }
    if (!scenario.type?.trim()) {
      setError('Scenario type is required');
      return false;
    }
    if (selectedAgents.length === 0) {
      setError('At least one agent must be selected');
      return false;
    }
    return true;
  };

  const saveScenario = async () => {
    if (!validateScenario()) return;

    setIsSaving(true);
    setError(null);

    try {
      const scenarioConfig: ScenarioConfig = {
        id: scenario.id || `scenario_${Date.now()}`,
        name: scenario.name!,
        type: scenario.type!,
        description: scenario.description || '',
        config: scenario.config || {},
        enabled: scenario.enabled ?? true,
        priority: scenario.priority || 1,
        parameters: scenario.parameters!,
        metadata: scenario.metadata!
      };

      const response = await apiService.post<ScenarioConfig>('/benchmarking/scenarios', scenarioConfig);
      
      notificationService.success('Scenario created successfully', 3000);
      onScenarioCreated?.(response.data);
      
      // Reset form
      setScenario({
        id: '',
        name: '',
        type: 'custom',
        description: '',
        config: {},
        enabled: true,
        priority: 1,
        parameters: {
          duration: 300,
          complexity: 'medium',
          domain: 'general',
          difficulty: 'medium'
        },
        metadata: {
          author: '',
          version: '1.0.0',
          tags: []
        }
      });
      setSelectedTemplate('');
      setSelectedAgents([]);
    } catch (err) {
      console.error('Error saving scenario:', err);
      setError('Failed to save scenario');
    } finally {
      setIsSaving(false);
    }
  };

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center h-96 ${className}`}>
        <LoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}>
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Scenario Builder</h2>
        
        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-md">
            <h3 className="text-sm font-medium text-red-800">Error</h3>
            <p className="text-sm text-red-700 mt-1">{error}</p>
            <button
              onClick={() => setError(null)}
              className="mt-2 text-sm text-red-600 hover:text-red-800"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Template Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Start from Template (Optional)
          </label>
          <select
            value={selectedTemplate}
            onChange={(e) => setSelectedTemplate(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Create Custom Scenario</option>
            {templates.map(template => (
              <option key={template.id} value={template.id}>
                {template.name} - {template.category}
              </option>
            ))}
          </select>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-gray-200 mb-6">
          {[
            { id: 'basic', label: 'Basic Info' },
            { id: 'parameters', label: 'Parameters' },
            { id: 'metadata', label: 'Metadata' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`px-4 py-2 font-medium transition-colors ${
                activeTab === tab.id
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-700 hover:text-blue-600 border-b-2 border-transparent hover:border-blue-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Basic Info Tab */}
        {activeTab === 'basic' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Scenario Name *
              </label>
              <input
                type="text"
                value={scenario.name || ''}
                onChange={(e) => handleInputChange('name', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter scenario name"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Scenario Type *
              </label>
              <select
                value={scenario.type || ''}
                onChange={(e) => handleInputChange('type', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select type</option>
                <option value="business">Business</option>
                <option value="cognitive">Cognitive</option>
                <option value="technical">Technical</option>
                <option value="ethical">Ethical</option>
                <option value="custom">Custom</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description
              </label>
              <textarea
                value={scenario.description || ''}
                onChange={(e) => handleInputChange('description', e.target.value)}
                rows={3}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Describe the scenario"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Agents *
              </label>
              <div className="space-y-2 max-h-40 overflow-y-auto border border-gray-200 rounded-md p-3">
                {agents.map(agent => (
                  <label key={agent.id} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={selectedAgents.includes(agent.id)}
                      onChange={() => handleAgentToggle(agent.id)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">
                      {agent.name} ({agent.type})
                    </span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Parameters Tab */}
        {activeTab === 'parameters' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Duration (seconds)
              </label>
              <input
                type="number"
                value={scenario.parameters?.duration || ''}
                onChange={(e) => handleParameterChange('duration', parseInt(e.target.value) || 0)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                min="1"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Complexity
              </label>
              <select
                value={scenario.parameters?.complexity || ''}
                onChange={(e) => handleParameterChange('complexity', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Domain
              </label>
              <input
                type="text"
                value={scenario.parameters?.domain || ''}
                onChange={(e) => handleParameterChange('domain', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="e.g., finance, healthcare, retail"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Difficulty
              </label>
              <select
                value={scenario.parameters?.difficulty || ''}
                onChange={(e) => handleParameterChange('difficulty', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
                <option value="expert">Expert</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Priority
              </label>
              <input
                type="number"
                value={scenario.priority || ''}
                onChange={(e) => handleInputChange('priority', parseInt(e.target.value) || 1)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                min="1"
                max="10"
              />
            </div>
          </div>
        )}

        {/* Metadata Tab */}
        {activeTab === 'metadata' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Author
              </label>
              <input
                type="text"
                value={scenario.metadata?.author || ''}
                onChange={(e) => handleMetadataChange('author', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Author name"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Version
              </label>
              <input
                type="text"
                value={scenario.metadata?.version || ''}
                onChange={(e) => handleMetadataChange('version', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="1.0.0"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Tags (comma-separated)
              </label>
              <input
                type="text"
                value={scenario.metadata?.tags?.join(', ') || ''}
                onChange={(e) => handleTagsChange(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="tag1, tag2, tag3"
              />
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="mt-8 flex justify-end space-x-3">
          <button
            onClick={() => {
              setScenario({
                id: '',
                name: '',
                type: 'custom',
                description: '',
                config: {},
                enabled: true,
                priority: 1,
                parameters: {
                  duration: 300,
                  complexity: 'medium',
                  domain: 'general',
                  difficulty: 'medium'
                },
                metadata: {
                  author: '',
                  version: '1.0.0',
                  tags: []
                }
              });
              setSelectedTemplate('');
              setSelectedAgents([]);
              setError(null);
            }}
            className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 transition-colors"
          >
            Reset
          </button>
          <button
            onClick={saveScenario}
            disabled={isSaving}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            {isSaving ? 'Saving...' : 'Save Scenario'}
          </button>
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default ScenarioBuilder;