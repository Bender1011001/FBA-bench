import React, { useState, useCallback, useEffect } from 'react';
import { ExperimentConfigForm } from '../components/ExperimentConfigForm';
import { ExperimentRunner } from '../components/ExperimentRunner';
import { ExperimentProgress } from '../components/ExperimentProgress';
import { ExperimentResults } from '../components/ExperimentResults';
import type { ExperimentConfig, ExperimentExecution, ExperimentStatus, Template } from '../types'; // Import Template type
import { predefinedTemplates } from '../data/templates'; // Import predefined templates
import { apiService } from '../services/apiService'; // For fetching historical data if needed

export function ExperimentManagement() {
  const [currentConfig, setCurrentConfig] = useState<ExperimentConfig>({});
  const [activeExperiments, setActiveExperiments] = useState<ExperimentExecution[]>([]);
  const [completedExperiments, setCompletedExperiments] = useState<ExperimentExecution[]>([]);
  const [activeTab, setActiveTab] = useState<'configure' | 'queue' | 'history'>('configure');
  const [selectedTemplateId, setSelectedTemplateId] = useState<string>(''); // For template selection

  // Load initial active/completed experiments from backend on mount (if persisted)
  // For now, these will be empty arrays. A future enhancement could retrieve from backend.
  useEffect(() => {
    const fetchInitialExperiments = async () => {
      // In a real application, you'd fetch running/completed experiments here
      // For now, we simulate an empty initial state or load from local storage
      console.log("Fetching initial experiment states (placeholder)...");
       // Example: Fetch from a backend endpoint if they persist experiments
      try {
        const active: ExperimentExecution[] = await apiService.get('/api/experiments/active');
        const completed: ExperimentExecution[] = await apiService.get('/api/experiments/completed-full'); // Get full details for completed
        setActiveExperiments(active);
        setCompletedExperiments(completed);
      } catch (error) {
        console.error("Failed to load initial experiments:", error);
        // Handle error, maybe set an empty array
        setActiveExperiments([]);
        setCompletedExperiments([]);
      }
    };
    fetchInitialExperiments();
  }, []);

  const handleConfigChange = useCallback((newConfig: ExperimentConfig) => {
    setCurrentConfig(newConfig);
  }, []);

  const handleTemplateSelect = useCallback((event: React.ChangeEvent<HTMLSelectElement>) => {
    const templateId = event.target.value;
    setSelectedTemplateId(templateId);
    if (templateId === '') {
      setCurrentConfig({}); // Reset to empty if no template selected
    } else {
      const template = predefinedTemplates.find(t => t.id === templateId);
      if (template) {
        if (template.configuration.experimentSettings) {
          // Deep copy to avoid direct mutation
          setCurrentConfig(JSON.parse(JSON.stringify(template.configuration.experimentSettings)));
        } else {
          setCurrentConfig({}); // No experiment settings in template, reset
        }
      }
    }
  }, []);

  const handleExperimentRun = useCallback((newExperiment: ExperimentExecution) => {
    setActiveExperiments(prev => [...prev, newExperiment]);
    setActiveTab('queue'); // Switch to queue view after starting an experiment
  }, []);

  const handleExperimentUpdate = useCallback((updatedExperiment: ExperimentExecution) => {
    setActiveExperiments(prev => 
      prev.map(exp => exp.id === updatedExperiment.id ? updatedExperiment : exp)
    );
  }, []);

  const handleExperimentEnded = useCallback((experimentId: string, status: ExperimentStatus) => {
    setActiveExperiments(prev => {
      const endedExperiment = prev.find(exp => exp.id === experimentId);
      if (endedExperiment) {
        // Move to completed and update status based on final event
        const updatedEndedExperiment = { ...endedExperiment, status: status, endTime: new Date().toISOString() };
        setCompletedExperiments(compPrev => [...compPrev, updatedEndedExperiment]);
        return prev.filter(exp => exp.id !== experimentId);
      }
      return prev;
    });
  }, []);

  // Handler to view a specific detailed result from ExperimentResults component
  const handleViewExperimentDetails = useCallback((experimentId: string) => {
    // This could navigate to a dedicated detail page, or open a modal.
    // For now, ExperimentResults handles its own detailed view internally using selectedExperimentId.
    console.log(`Requested to view details for experiment: ${experimentId}`);
  }, []); // Currently, this handler is mainly for logging or future extensibility

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-6">Experiment Management System</h1>
      
      <div className="mb-6 border-b border-gray-200">
        <nav className="-mb-px flex space-x-8" aria-label="Tabs">
          <button
            onClick={() => setActiveTab('configure')}
            className={`whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm focus:outline-none 
              ${activeTab === 'configure' ? 'border-indigo-500 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}
          >
            Configure Experiment
          </button>
          <button
            onClick={() => setActiveTab('queue')}
            className={`whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm focus:outline-none 
              ${activeTab === 'queue' ? 'border-indigo-500 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}
          >
            Active/Queued Experiments ({activeExperiments.length})
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`whitespace-nowrap py-3 px-1 border-b-2 font-medium text-sm focus:outline-none 
              ${activeTab === 'history' ? 'border-indigo-500 text-indigo-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}
          >
            Experiment History ({completedExperiments.length})
          </button>
        </nav>
      </div>

      <div className="bg-white p-6 rounded-lg shadow">
        {activeTab === 'configure' && (
          <div>
            <div className="mb-6">
              <label htmlFor="template-select" className="block text-sm font-medium text-gray-700">
                Load Experiment Template
              </label>
              <select
                id="template-select"
                value={selectedTemplateId}
                onChange={handleTemplateSelect}
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
              >
                <option value="">-- Select a Template --</option>
                {predefinedTemplates.map(template => (
                  <option key={template.id} value={template.id}>
                    {template.name} - {template.description}
                  </option>
                ))}
              </select>
            </div>
            <ExperimentConfigForm config={currentConfig} onConfigChange={handleConfigChange} />
            <div className="mt-8 pt-4 border-t border-gray-200">
              <ExperimentRunner
                config={currentConfig}
                onExperimentRun={handleExperimentRun}
                onCancel={() => { /* Potentially reset config or navigate away */ }}
              />
            </div>
          </div>
        )}

        {activeTab === 'queue' && (
          <div>
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Active & Queued Experiments</h2>
            {activeExperiments.length === 0 ? (
              <p className="text-gray-600">No active or queued experiments. Start one from the "Configure Experiment" tab!</p>
            ) : (
                <div className="space-y-4">
                    {activeExperiments.map(exp => (
                        <ExperimentProgress 
                            key={exp.id} 
                            experiment={exp} 
                            onExperimentUpdate={handleExperimentUpdate}
                            onExperimentEnded={handleExperimentEnded}
                        />
                    ))}
                </div>
            )}
          </div>
        )}

        {activeTab === 'history' && (
          <ExperimentResults onViewDetails={handleViewExperimentDetails} />
        )}
      </div>
    </div>
  );
}