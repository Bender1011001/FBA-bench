import React, { useState, useEffect, useCallback } from 'react';
import type { ExperimentResultSummary, DetailedExperimentResult, ExperimentStatus } from '../types';
import { apiService } from '../services/apiService';

interface ExperimentResultsProps {
  // This could be used to switch to a detailed view, or open a modal
  onViewDetails: (experimentId: string) => void; 
}

export function ExperimentResults({ onViewDetails }: ExperimentResultsProps) {
  const [results, setResults] = useState<ExperimentResultSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filterText, setFilterText] = useState('');
  const [sortColumn, setSortColumn] = useState<keyof ExperimentResultSummary | null>('startTime');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [selectedExperimentId, setSelectedExperimentId] = useState<string | null>(null);
  const [detailedResult, setDetailedResult] = useState<DetailedExperimentResult | null>(null);
  const [detailedLoading, setDetailedLoading] = useState(false);
  const [detailedError, setDetailedError] = useState<string | null>(null);


  const fetchResults = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Assuming a backend endpoint for getting a summary of all completed experiments
      const data = await apiService.get<ExperimentResultSummary[]>('/api/experiments/results-summary');
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch experiment results.');
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchDetailedResult = useCallback(async (experimentId: string) => {
    setDetailedLoading(true);
    setDetailedError(null);
    try {
      // Assuming a backend endpoint for getting detailed results of a specific experiment
      const data = await apiService.get<DetailedExperimentResult>(`/api/experiments/${experimentId}/details`);
      setDetailedResult(data);
    } catch (err) {
      setDetailedError(err instanceof Error ? err.message : `Failed to fetch detailed results for ${experimentId}.`);
    } finally {
      setDetailedLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchResults();
  }, [fetchResults]);

  useEffect(() => {
    if (selectedExperimentId) {
      fetchDetailedResult(selectedExperimentId);
    } else {
      setDetailedResult(null); // Clear detailed view if no experiment is selected
    }
  }, [selectedExperimentId, fetchDetailedResult]);


  const sortedAndFilteredResults = React.useMemo(() => {
    let filtered = results.filter(res =>
      res.experimentName.toLowerCase().includes(filterText.toLowerCase()) ||
      res.description.toLowerCase().includes(filterText.toLowerCase()) ||
      res.id.toLowerCase().includes(filterText.toLowerCase())
    );

    if (sortColumn) {
      filtered = filtered.sort((a, b) => {
        const aValue = a[sortColumn];
        const bValue = b[sortColumn];

        if (typeof aValue === 'string' && typeof bValue === 'string') {
          return sortDirection === 'asc' ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
        }
        if (typeof aValue === 'number' && typeof bValue === 'number') {
          return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
        }
        return 0;
      });
    }
    return filtered;
  }, [results, filterText, sortColumn, sortDirection]);

  const handleSort = (column: keyof ExperimentResultSummary) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  const renderStatusBadge = (status: ExperimentStatus) => {
    let colorClass = '';
    switch (status) {
      case 'queued': colorClass = 'bg-gray-200 text-gray-800'; break;
      case 'running': colorClass = 'bg-blue-200 text-blue-800'; break;
      case 'completed': colorClass = 'bg-green-200 text-green-800'; break;
      case 'failed': colorClass = 'bg-red-200 text-red-800'; break;
      case 'cancelled': colorClass = 'bg-yellow-200 text-yellow-800'; break;
      case 'paused': colorClass = 'bg-orange-200 text-orange-800'; break;
      default: colorClass = 'bg-gray-200 text-gray-800';
    }
    return <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colorClass}`}>{status}</span>;
  };

  if (loading) return <div className="text-center py-8">Loading results...</div>;
  if (error) return <div className="text-center py-8 text-red-600">Error: {error}</div>;

  return (
    <div className="p-6 bg-white shadow overflow-hidden sm:rounded-lg">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">Experiment Results</h2>

      <div className="mb-4 flex space-x-4">
        <input
          type="text"
          placeholder="Filter results..."
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          className="mt-1 block w-1/3 border border-gray-300 rounded-md shadow-sm py-2 px-3"
        />
        <button
          onClick={fetchResults}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          disabled={loading}
        >
          Refresh
        </button>
        {/* Placeholder for Export Button */}
        <button
            onClick={() => alert('Export functionality coming soon!')}
            className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
            disabled={loading}
        >
            Export All
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSort('experimentName')}
              >
                Experiment Name {sortColumn === 'experimentName' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSort('status')}
              >
                Status {sortColumn === 'status' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSort('startTime')}
              >
                Start Time {sortColumn === 'startTime' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSort('endTime')}
              >
                End Time {sortColumn === 'endTime' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer"
                onClick={() => handleSort('totalCombinations')}
              >
                Total Runs {sortColumn === 'totalCombinations' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Key Metrics
              </th>
              <th className="relative px-6 py-3"><span className="sr-only">Actions</span></th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sortedAndFilteredResults.map((res) => (
              <tr key={res.id}>
                <td className="px-6 py-4 whitespace-nowrap"><div className="text-sm font-medium text-gray-900">{res.experimentName}</div><div className="text-xs text-gray-500">{res.id}</div></td>
                <td className="px-6 py-4 whitespace-nowrap">{renderStatusBadge(res.status)}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{new Date(res.startTime).toLocaleString()}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{res.endTime ? new Date(res.endTime).toLocaleString() : 'N/A'}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{res.completedCombinations} / {res.totalCombinations}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {res.keyMetrics && Object.entries(res.keyMetrics).map(([key, value]) => (
                    <div key={key} className="text-xs">{key}: {typeof value === 'number' ? value.toFixed(2) : String(value)}</div>
                  ))}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                  <button onClick={() => setSelectedExperimentId(res.id)} className="text-indigo-600 hover:text-indigo-900">
                    View Details
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Detailed Experiment Results Modal/Panel */}
      {selectedExperimentId && (
        <div className="mt-8 border-t border-gray-200 pt-8" id="detailed-results">
          <h3 className="text-xl font-medium text-gray-800 mb-4">
            Experiment Details: {detailedResult?.summary.experimentName || selectedExperimentId}
          </h3>
          {detailedLoading && <div className="text-center py-4">Loading detailed results...</div>}
          {detailedError && <div className="text-center py-4 text-red-600">Error: {detailedError}</div>}
          {detailedResult && (
            <div className="bg-gray-50 p-6 rounded-lg shadow-inner">
              <h4 className="text-lg font-semibold text-gray-700 mb-2">Configuration:</h4>
              <pre className="text-sm bg-white p-4 rounded-md overflow-auto max-h-60 mb-4">
                {JSON.stringify(detailedResult.config, null, 2)}
              </pre>

              <h4 className="text-lg font-semibold text-gray-700 mb-2">Metrics Summary:</h4>
              <pre className="text-sm bg-white p-4 rounded-md overflow-auto max-h-60 mb-4">
                {JSON.stringify(detailedResult.resultsData, null, 2)}
              </pre>

              {/* Placeholder for Visualization */}
              <div className="bg-white p-4 rounded-md mb-4 shadow-sm">
                <h4 className="text-lg font-semibold text-gray-700 mb-2">Visualization (Coming Soon):</h4>
                <p className="text-gray-600">Charts and graphs will be displayed here.</p>
              </div>

              <div className="text-right">
                <button
                  onClick={() => setSelectedExperimentId(null)}
                  className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
                >
                  Close Details
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}