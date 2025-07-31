import React, { useState } from 'react';
import type { ResultsData } from '../types';

interface ComparisonToolProps {
  // This component will likely fetch its own data for multiple experiments
  // Or accept an array of ResultsData if pre-fetched for side-by-side view
  data?: ResultsData[];
  onCompare?: (experimentIds: string[]) => void;
}

// Placeholder for a generic chart component that can display different chart types
const GenericChart: React.FC<any> = ({ data, config }) => (
  <div className="bg-white p-4 rounded shadow-md h-64 flex items-center justify-center">
    <p>{config.title || 'Generic Chart Placeholder'}</p>
  </div>
);

const ComparisonTool: React.FC<ComparisonToolProps> = ({ data, onCompare }) => {
  const [selectedExperiments, setSelectedExperiments] = useState<string[]>([]);
  const [baselineExperiment, setBaselineExperiment] = useState<string>('');

  // Example data for available experiments - in a real app, this would come from an API
  const availableExperiments = [
    { id: 'exp-001', name: 'Experiment Alpha - Pricing Strategy A' },
    { id: 'exp-002', name: 'Experiment Beta - Pricing Strategy B' },
    { id: 'exp-003', name: 'Experiment Gamma - Marketing Campaign' },
    { id: 'baseline-run', name: 'Baseline Run - Default Configuration' },
  ];

  const handleExperimentSelection = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const { value, selectedOptions } = e.target;
    // Allow multi-selection
    const selectedIds = Array.from(selectedOptions).map(option => option.value);
    setSelectedExperiments(selectedIds);
    if (onCompare) {
        onCompare(selectedIds);
    }
  };

  const simulateStatisticalTest = (exp1: ResultsData, exp2: ResultsData) => {
    // Placeholder for actual statistical significance testing
    // e.g., t-test on profit data
    const profit1 = exp1.aggregatedMetrics.totalProfit || 0;
    const profit2 = exp2.aggregatedMetrics.totalProfit || 0;
    const diff = profit2 - profit1;
    const isSignificant = Math.abs(diff) > 1000; // Example threshold
    return (
      <p className={`text-sm ${isSignificant ? 'text-green-600' : 'text-orange-600'}`}>
        Profit Delta: ${diff.toFixed(2)} ({isSignificant ? 'Statistically Significant (Placeholder)' : 'Not Significant (Placeholder)'})
      </p>
    );
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md mb-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Experiment Comparison & Analysis</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* Experiment Selection for Comparison */}
        <div>
          <label htmlFor="selectExperiments" className="block text-sm font-medium text-gray-700 mb-1">Select Experiments to Compare</label>
          <select
            id="selectExperiments"
            multiple
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md h-32"
            value={selectedExperiments}
            onChange={handleExperimentSelection}
          >
            {availableExperiments.map((exp) => (
              <option key={exp.id} value={exp.id}>{exp.name}</option>
            ))}
          </select>
          <p className="mt-2 text-sm text-gray-500">Hold Ctrl/Cmd to select multiple.</p>
        </div>

        {/* Baseline Selection */}
        <div>
          <label htmlFor="baselineExperiment" className="block text-sm font-medium text-gray-700 mb-1">Select Baseline Experiment</label>
          <select
            id="baselineExperiment"
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
            value={baselineExperiment}
            onChange={(e) => setBaselineExperiment(e.target.value)}
          >
            <option value="">No Baseline</option>
            {availableExperiments.map((exp) => (
              <option key={exp.id} value={exp.id}>{exp.name}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Side-by-side Comparison View */}
      {selectedExperiments.length > 0 && (
        <div className="mt-8">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Side-by-Side Comparison</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {selectedExperiments.map((expId) => {
              // Placeholder: In a real app, you'd fetch resultsData for expId
              const experimentData = data?.find(d => d.experimentId === expId) || {
                  experimentId: expId,
                  simulationResults: [],
                  aggregatedMetrics: { totalRevenue: Math.random() * 10000, totalCosts: Math.random() * 5000, totalProfit: Math.random() * 5000, averageTicksPerSecond: 10, topPerformingAgent: 'AgentX', experimentDuration: 1000 },
                  agentPerformance: [],
                  financialMetrics: { totalRevenue: 0, totalCosts: 0, totalProfit: 0 },
                  timeSeriesData: [],
              }; // Mock data
              
              const isBaseline = expId === baselineExperiment;

              return (
                <div key={expId} className={`border p-4 rounded-md ${isBaseline ? 'border-indigo-500 bg-indigo-50' : 'border-gray-200 bg-gray-50'}`}>
                  <h4 className="font-bold text-gray-900 mb-2">{availableExperiments.find(e => e.id === expId)?.name || expId} {isBaseline && '(Baseline)'}</h4>
                  <p className="text-sm">Total Profit: ${experimentData.aggregatedMetrics.totalProfit?.toFixed(2)}</p>
                  <p className="text-sm">Total Revenue: ${experimentData.aggregatedMetrics.totalRevenue?.toFixed(2)}</p>
                  {/* More metrics */}
                  
                  {/* Statistical Significance & Performance Delta */}
                  {baselineExperiment && !isBaseline && data?.find(d => d.experimentId === baselineExperiment) && (
                    <div className="mt-2">
                        {simulateStatisticalTest(data?.find(d => d.experimentId === baselineExperiment)!, experimentData)}
                    </div>
                  )}

                  <div className="mt-4">
                    {/* Charts for individual experiment */}
                    <GenericChart data={experimentData.timeSeriesData} config={{ title: 'Financial Performance', chartType: 'line', metrics: ['revenue'] }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {selectedExperiments.length === 0 && (
        <p className="text-center text-gray-500 mt-8">Select experiments above to enable comparison.</p>
      )}
    </div>
  );
};

export default ComparisonTool;