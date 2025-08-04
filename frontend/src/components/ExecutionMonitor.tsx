import React, { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';
import { useWebSocket } from '../hooks/useWebSocket';
import type { 
  ExecutionProgress, 
  BenchmarkResult, 
  BenchmarkWebSocketEvent,
  AgentRunResult 
} from '../types';
import { notificationService } from '../utils/notificationService';
import LoadingSpinner from './LoadingSpinner';
import ErrorBoundary from './ErrorBoundary';

interface ExecutionMonitorProps {
  className?: string;
}

const ExecutionMonitor: React.FC<ExecutionMonitorProps> = ({ className = '' }) => {
  const [activeBenchmarks, setActiveBenchmarks] = useState<ExecutionProgress[]>([]);
  const [completedBenchmarks, setCompletedBenchmarks] = useState<BenchmarkResult[]>([]);
  const [selectedBenchmark, setSelectedBenchmark] = useState<string | null>(null);
  const [agentResults, setAgentResults] = useState<AgentRunResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000);

  // WebSocket connection for real-time updates
  const { lastMessage, connectionStatus } = useWebSocket({
    url: `${import.meta.env.VITE_WS_URL || 'ws://localhost:8000'}/ws/benchmarking`,
    autoConnect: true,
  });

  useEffect(() => {
    fetchActiveBenchmarks();
    fetchCompletedBenchmarks();
  }, []);

  useEffect(() => {
    if (lastMessage) {
      try {
        const event: BenchmarkWebSocketEvent = JSON.parse(lastMessage.data);
        handleWebSocketEvent(event);
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    }
  }, [lastMessage]);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    
    if (autoRefresh) {
      intervalId = setInterval(() => {
        fetchActiveBenchmarks();
        fetchCompletedBenchmarks();
      }, refreshInterval);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [autoRefresh, refreshInterval]);

  useEffect(() => {
    if (selectedBenchmark) {
      fetchAgentResults(selectedBenchmark);
    }
  }, [selectedBenchmark]);

  const handleWebSocketEvent = (event: BenchmarkWebSocketEvent) => {
    switch (event.type) {
      case 'benchmark_started':
        notificationService.info(`Benchmark ${event.payload.benchmark_id} started`, 3000);
        fetchActiveBenchmarks();
        break;
      case 'benchmark_progress':
        setActiveBenchmarks(prev => 
          prev.map(b => 
            b.benchmark_id === event.payload.benchmark_id 
              ? { ...b, ...event.payload }
              : b
          )
        );
        break;
      case 'agent_run_completed':
        if (selectedBenchmark === event.payload.scenario_name) {
          setAgentResults(prev => [...prev, event.payload]);
        }
        break;
      case 'benchmark_completed':
        notificationService.success(`Benchmark ${event.payload.benchmark_name} completed!`, 5000);
        setActiveBenchmarks(prev => 
          prev.filter(b => b.benchmark_id !== event.payload.benchmark_name)
        );
        setCompletedBenchmarks(prev => [event.payload, ...prev]);
        break;
      case 'benchmark_error':
        notificationService.error(`Benchmark error: ${event.payload.error}`, 5000);
        setError(event.payload.error);
        break;
      default:
        console.log('Unhandled WebSocket event:', event);
    }
  };

  const fetchActiveBenchmarks = async () => {
    try {
      const response = await apiService.get<ExecutionProgress[]>('/benchmarking/active');
      setActiveBenchmarks(response.data);
    } catch (err) {
      console.error('Error fetching active benchmarks:', err);
      setError('Failed to fetch active benchmarks');
    }
  };

  const fetchCompletedBenchmarks = async () => {
    try {
      const response = await apiService.get<BenchmarkResult[]>('/benchmarking/completed');
      setCompletedBenchmarks(response.data);
    } catch (err) {
      console.error('Error fetching completed benchmarks:', err);
      setError('Failed to fetch completed benchmarks');
    }
  };

  const fetchAgentResults = async (benchmarkId: string) => {
    setIsLoading(true);
    try {
      const response = await apiService.get<AgentRunResult[]>(`/benchmarking/${benchmarkId}/agent-results`);
      setAgentResults(response.data);
    } catch (err) {
      console.error('Error fetching agent results:', err);
      setError('Failed to fetch agent results');
    } finally {
      setIsLoading(false);
    }
  };

  const stopBenchmark = async (benchmarkId: string) => {
    try {
      await apiService.post(`/benchmarking/${benchmarkId}/stop`);
      notificationService.info(`Benchmark ${benchmarkId} stopping`, 3000);
      fetchActiveBenchmarks();
    } catch (err) {
      console.error('Error stopping benchmark:', err);
      setError('Failed to stop benchmark');
    }
  };

  const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'running':
        return 'bg-green-100 text-green-800';
      case 'completed':
        return 'bg-blue-100 text-blue-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'paused':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <ErrorBoundary>
      <div className={`space-y-6 ${className}`}>
        {/* Header with Controls */}
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold text-gray-900">Execution Monitor</h2>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="autoRefresh"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <label htmlFor="autoRefresh" className="ml-2 text-sm text-gray-700">
                Auto Refresh
              </label>
            </div>
            
            {autoRefresh && (
              <select
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(parseInt(e.target.value))}
                className="px-3 py-1 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="1000">1s</option>
                <option value="5000">5s</option>
                <option value="10000">10s</option>
                <option value="30000">30s</option>
              </select>
            )}
            
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${
                connectionStatus === 'connected' ? 'bg-green-500' : 
                connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
              }`}></div>
              <span className="text-sm text-gray-600">
                {connectionStatus === 'connected' ? 'Live' :
                 connectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-md">
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

        {/* Active Benchmarks */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Active Benchmarks</h3>
          
          {activeBenchmarks.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No active benchmarks</p>
          ) : (
            <div className="space-y-4">
              {activeBenchmarks.map(benchmark => (
                <div key={benchmark.benchmark_id} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <h4 className="font-medium text-gray-900">{benchmark.benchmark_id}</h4>
                      <div className="flex items-center mt-1">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(benchmark.status)}`}>
                          {benchmark.status}
                        </span>
                        {benchmark.current_scenario && (
                          <span className="ml-2 text-sm text-gray-600">
                            Current: {benchmark.current_scenario}
                          </span>
                        )}
                      </div>
                    </div>
                    
                    <button
                      onClick={() => stopBenchmark(benchmark.benchmark_id)}
                      className="px-3 py-1 bg-red-600 text-white text-sm rounded-md hover:bg-red-700 transition-colors"
                    >
                      Stop
                    </button>
                  </div>
                  
                  {/* Progress Bar */}
                  <div className="mb-3">
                    <div className="flex justify-between text-sm text-gray-600 mb-1">
                      <span>Progress</span>
                      <span>{benchmark.progress.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${benchmark.progress}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  {/* Additional Info */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    {benchmark.start_time && (
                      <div>
                        <span className="text-gray-500">Started:</span>
                        <span className="ml-1 text-gray-900">
                          {new Date(benchmark.start_time).toLocaleString()}
                        </span>
                      </div>
                    )}
                    {benchmark.estimated_completion_time && (
                      <div>
                        <span className="text-gray-500">Est. Completion:</span>
                        <span className="ml-1 text-gray-900">
                          {new Date(benchmark.estimated_completion_time).toLocaleString()}
                        </span>
                      </div>
                    )}
                    {benchmark.current_agent && (
                      <div>
                        <span className="text-gray-500">Current Agent:</span>
                        <span className="ml-1 text-gray-900">{benchmark.current_agent}</span>
                      </div>
                    )}
                    {benchmark.current_run !== undefined && (
                      <div>
                        <span className="text-gray-500">Run:</span>
                        <span className="ml-1 text-gray-900">{benchmark.current_run}</span>
                      </div>
                    )}
                  </div>
                  
                  {/* Errors */}
                  {benchmark.errors && benchmark.errors.length > 0 && (
                    <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-md">
                      <h5 className="text-sm font-medium text-red-800">Errors</h5>
                      <ul className="mt-1 text-sm text-red-700 list-disc list-inside">
                        {benchmark.errors.map((error, index) => (
                          <li key={index}>{error}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Completed Benchmarks */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Completed Benchmarks</h3>
          
          {completedBenchmarks.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No completed benchmarks</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Benchmark
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Duration
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Scenarios
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Completed
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {completedBenchmarks.map(benchmark => (
                    <tr key={benchmark.benchmark_name} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900">
                          {benchmark.benchmark_name}
                        </div>
                        <div className="text-sm text-gray-500">
                          {new Date(benchmark.start_time).toLocaleString()}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {formatDuration(benchmark.duration_seconds)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {benchmark.scenario_results.length}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                          Completed
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        <button
                          onClick={() => setSelectedBenchmark(
                            selectedBenchmark === benchmark.benchmark_name 
                              ? null 
                              : benchmark.benchmark_name
                          )}
                          className="text-blue-600 hover:text-blue-900 mr-3"
                        >
                          {selectedBenchmark === benchmark.benchmark_name ? 'Hide' : 'View'} Details
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Agent Results */}
        {selectedBenchmark && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              Agent Results - {selectedBenchmark}
            </h3>
            
            {isLoading ? (
              <div className="flex items-center justify-center h-32">
                <LoadingSpinner size="medium" />
              </div>
            ) : agentResults.length === 0 ? (
              <p className="text-gray-500 text-center py-8">No agent results available</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Agent
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Scenario
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Run
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Duration
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Status
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Metrics
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {agentResults.map((result, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {result.agent_id}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {result.scenario_name}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {result.run_number}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {formatDuration(result.duration_seconds)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            result.success 
                              ? 'bg-green-100 text-green-800' 
                              : 'bg-red-100 text-red-800'
                          }`}>
                            {result.success ? 'Success' : 'Failed'}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-900">
                          <div className="max-h-20 overflow-y-auto">
                            {result.metrics.map((metric, metricIndex) => (
                              <div key={metricIndex} className="text-xs">
                                {metric.name}: {metric.value.toFixed(3)} {metric.unit}
                              </div>
                            ))}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
};

export default ExecutionMonitor;