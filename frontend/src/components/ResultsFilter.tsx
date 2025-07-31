import React, { useState, useEffect } from 'react';

interface ResultsFilterProps {
  onFilterChange: (filters: Record<string, unknown>) => void;
  // Potentially add props for available options, e.g., agent list, experiment types
  availableAgents?: string[];
  availableExperimentTypes?: string[];
  availableMetrics?: string[];
}

const ResultsFilter: React.FC<ResultsFilterProps> = ({
  onFilterChange,
  availableAgents = ['AgentA', 'AgentB', 'AgentC'], // Example data
  availableExperimentTypes = ['TypeX', 'TypeY', 'TypeZ'], // Example data
  availableMetrics = ['profit', 'revenue', 'costs', 'decisionsMade'], // Example data
}) => {
  const [dateRange, setDateRange] = useState<string>('');
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [experimentType, setExperimentType] = useState<string>('');
  const [metricThreshold, setMetricThreshold] = useState<number | string>('');
  const [selectedMetric, setSelectedMetric] = useState<string>('');
  const [customQuery, setCustomQuery] = useState<string>('');

  useEffect(() => {
    // Debounce filter changes for better performance on complex UIs
    const handler = setTimeout(() => {
      const filters: Record<string, unknown> = {};
      if (dateRange) filters.dateRange = dateRange;
      if (selectedAgent) filters.agentId = selectedAgent;
      if (experimentType) filters.experimentType = experimentType;
      if (selectedMetric && metricThreshold !== '') filters[selectedMetric] = metricThreshold;
      if (customQuery) filters.customQuery = customQuery; // For backend processing

      onFilterChange(filters);
    }, 300); // 300ms debounce

    return () => {
      clearTimeout(handler);
    };
  }, [dateRange, selectedAgent, experimentType, metricThreshold, selectedMetric, customQuery, onFilterChange]);

  return (
    <div className="bg-white p-6 rounded-lg shadow-md mb-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Filter Results</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Date Range Filtering */}
        <div>
          <label htmlFor="dateRange" className="block text-sm font-medium text-gray-700 mb-1">Date Range</label>
          <select
            id="dateRange"
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
            value={dateRange}
            onChange={(e) => setDateRange(e.target.value)}
          >
            <option value="">All Time</option>
            <option value="last24h">Last 24 Hours</option>
            <option value="last7d">Last 7 Days</option>
            <option value="last30d">Last 30 Days</option>
            <option value="custom">Custom Range...</option> {/* Future: add date pickers */}
          </select>
        </div>

        {/* Agent-specific Result Filtering */}
        <div>
          <label htmlFor="selectedAgent" className="block text-sm font-medium text-gray-700 mb-1">Agent</label>
          <select
            id="selectedAgent"
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
            value={selectedAgent}
            onChange={(e) => setSelectedAgent(e.target.value)}
          >
            <option value="">All Agents</option>
            {availableAgents.map((agent) => (
              <option key={agent} value={agent}>{agent}</option>
            ))}
          </select>
        </div>

        {/* Experiment Type and Configuration Filtering */}
        <div>
          <label htmlFor="experimentType" className="block text-sm font-medium text-gray-700 mb-1">Experiment Type</label>
          <select
            id="experimentType"
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
            value={experimentType}
            onChange={(e) => setExperimentType(e.target.value)}
          >
            <option value="">All Types</option>
            {availableExperimentTypes.map((type) => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
        </div>

        {/* Performance Metric Threshold Filtering */}
        <div>
          <label htmlFor="selectedMetric" className="block text-sm font-medium text-gray-700 mb-1">Metric Threshold</label>
          <div className="flex items-center mt-1">
            <select
              id="selectedMetric"
              className="block w-1/2 pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-l-md"
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
            >
              <option value="">Select Metric</option>
              {availableMetrics.map((metric) => (
                <option key={metric} value={metric}>{metric.charAt(0).toUpperCase() + metric.slice(1)}</option>
              ))}
            </select>
            <input
              type="number"
              className="block w-1/2 shadow-sm sm:text-sm border-gray-300 rounded-r-md p-2"
              placeholder="Min value"
              value={metricThreshold}
              onChange={(e) => setMetricThreshold(e.target.value === '' ? '' : parseFloat(e.target.value))}
              disabled={!selectedMetric}
            />
          </div>
        </div>

        {/* Custom Query Builder */}
        <div className="col-span-full">
          <label htmlFor="customQuery" className="block text-sm font-medium text-gray-700 mb-1">Custom Query</label>
          <input
            type="text"
            id="customQuery"
            className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md p-2"
            placeholder="e.g., agent.profit > 1000 AND experiment.duration < 500"
            value={customQuery}
            onChange={(e) => setCustomQuery(e.target.value)}
          />
          <p className="mt-2 text-sm text-gray-500">
            Enter advanced queries (syntax depends on backend implementation).
          </p>
        </div>
      </div>
    </div>
  );
};

export default ResultsFilter;