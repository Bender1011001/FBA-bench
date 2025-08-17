import React, { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';
import useWebSocket from '../hooks/useWebSocket';
import type {
  BenchmarkConfig,
  BenchmarkResult,
  ExecutionProgress,
  BenchmarkWebSocketEvent,
  ConfigurationTemplate
} from '../types';
import { notificationService } from '../utils/notificationService';
import LoadingSpinner from './LoadingSpinner';
import ErrorBoundary from './ErrorBoundary';
import MetricsVisualization from './MetricsVisualization';
import ScenarioBuilder from './ScenarioBuilder';
import ExecutionMonitor from './ExecutionMonitor';
import ResultsComparison from './ResultsComparison';
import ReportGenerator from './ReportGenerator';

interface BenchmarkDashboardProps {
  className?: string;
}

const BenchmarkDashboard: React.FC<BenchmarkDashboardProps> = ({ className = '' }) => {
  const [activeTab, setActiveTab] = useState(0);
  const [benchmarkConfigs, setBenchmarkConfigs] = useState<BenchmarkConfig[]>([]);
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult[]>([]);
  const [executionProgress, setExecutionProgress] = useState<ExecutionProgress | null>(null);
  const [templates, setTemplates] = useState<ConfigurationTemplate[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // WebSocket connection for real-time updates
  const { lastMessage, connectionStatus } = useWebSocket({
    url: `${import.meta.env.VITE_WS_URL || 'ws://localhost:8000'}/ws/benchmarking`,
    autoConnect: true,
  });

  // Fetch initial data
  useEffect(() => {
    fetchBenchmarkConfigs();
    fetchBenchmarkResults();
    fetchTemplates();
  }, []);

  // Handle WebSocket messages
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

  const handleWebSocketEvent = (event: BenchmarkWebSocketEvent) => {
    switch (event.type) {
      case 'benchmark_started':
        notificationService.info(`Benchmark ${event.payload.benchmark_id} started`, 3000);
        break;
      case 'benchmark_progress':
        setExecutionProgress(event.payload);
        break;
      case 'scenario_completed':
        notificationService.success(
          `Scenario ${event.payload.scenario_name} completed in ${event.payload.duration_seconds.toFixed(2)}s`,
          3000
        );
        break;
      case 'benchmark_completed':
        setBenchmarkResults(prev => [event.payload, ...prev]);
        notificationService.success('Benchmark completed successfully!', 5000);
        break;
      case 'benchmark_error':
        notificationService.error(`Benchmark error: ${event.payload.error}`, 5000);
        setError(event.payload.error);
        break;
      default:
        console.log('Unhandled WebSocket event:', event);
    }
  };

  const fetchBenchmarkConfigs = async () => {
    setIsLoading(true);
    try {
      const response = await apiService.get<BenchmarkConfig[]>('/benchmarking/configs');
      setBenchmarkConfigs(response.data);
    } catch (err) {
      console.error('Error fetching benchmark configs:', err);
      setError('Failed to fetch benchmark configurations');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchBenchmarkResults = async () => {
    setIsLoading(true);
    try {
      const response = await apiService.get<BenchmarkResult[]>('/benchmarking/results');
      setBenchmarkResults(response.data);
    } catch (err) {
      console.error('Error fetching benchmark results:', err);
      setError('Failed to fetch benchmark results');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchTemplates = async () => {
    try {
      const response = await apiService.get<ConfigurationTemplate[]>('/benchmarking/templates');
      setTemplates(response.data);
    } catch (err) {
      console.error('Error fetching templates:', err);
      // Non-critical error, don't set main error state
    }
  };

  const startBenchmark = async (configId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await apiService.post<{ benchmark_id: string }>(`/benchmarking/configs/${configId}/start`);
      notificationService.success(`Benchmark ${response.data.benchmark_id} started`, 3000);
    } catch (err) {
      console.error('Error starting benchmark:', err);
      setError('Failed to start benchmark');
    } finally {
      setIsLoading(false);
    }
  };

  const stopBenchmark = async (benchmarkId: string) => {
    try {
      await apiService.post(`/benchmarking/${benchmarkId}/stop`);
      notificationService.info(`Benchmark ${benchmarkId} stopping`, 3000);
    } catch (err) {
      console.error('Error stopping benchmark:', err);
      setError('Failed to stop benchmark');
    }
  };

  const deleteBenchmarkResult = async (resultId: string) => {
    try {
      await apiService.delete(`/benchmarking/results/${resultId}`);
      setBenchmarkResults(prev => prev.filter(r => r.benchmark_name !== resultId));
      notificationService.success('Benchmark result deleted', 3000);
    } catch (err) {
      console.error('Error deleting benchmark result:', err);
      setError('Failed to delete benchmark result');
    }
  };

  const exportResults = async (resultId: string, format: 'json' | 'csv' | 'pdf') => {
    try {
      const response = await apiService.get(`/benchmarking/results/${resultId}/export/${format}`, {
        headers: { 'Accept': format === 'pdf' ? 'application/pdf' : 'application/json' }
      });
      
      // Create download link
      const blob = new Blob([JSON.stringify(response.data, null, 2)], { 
        type: format === 'pdf' ? 'application/pdf' : 'application/json' 
      });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `benchmark_${resultId}_results.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      notificationService.success(`Results exported as ${format.toUpperCase()}`, 3000);
    } catch (err) {
      console.error('Error exporting results:', err);
      setError('Failed to export results');
    }
  };

  if (isLoading && benchmarkConfigs.length === 0) {
    return (
      <div className={`flex items-center justify-center h-96 ${className}`}>
        <LoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <div className={`bg-white rounded-lg shadow-md p-6 ${className}`}>
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900">FBA-Bench Dashboard</h1>
          <p className="text-gray-600 mt-2">
            Comprehensive benchmarking framework for agent evaluation and comparison
          </p>
          
          {/* Connection Status */}
          <div className="mt-4 flex items-center">
            <div className={`w-3 h-3 rounded-full mr-2 ${
              connectionStatus === 'connected' ? 'bg-green-500' : 
              connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
            }`}></div>
            <span className="text-sm text-gray-600">
              {connectionStatus === 'connected' ? 'Connected to benchmarking server' :
               connectionStatus === 'connecting' ? 'Connecting to benchmarking server...' :
               'Disconnected from benchmarking server'}
            </span>
          </div>
        </div>

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

        {/* Active Execution Progress */}
        {executionProgress && (
          <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-md">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-medium text-blue-900">
                  Benchmark in Progress: {executionProgress.benchmark_id}
                </h3>
                <p className="text-sm text-blue-700 mt-1">
                  Status: {executionProgress.status} • Progress: {executionProgress.progress.toFixed(1)}%
                </p>
                {executionProgress.current_scenario && (
                  <p className="text-sm text-blue-700">
                    Current Scenario: {executionProgress.current_scenario}
                  </p>
                )}
                {executionProgress.estimated_completion_time && (
                  <p className="text-sm text-blue-700">
                    Estimated Completion: {new Date(executionProgress.estimated_completion_time).toLocaleString()}
                  </p>
                )}
              </div>
              <button
                onClick={() => stopBenchmark(executionProgress.benchmark_id)}
                className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
              >
                Stop Benchmark
              </button>
            </div>
            <div className="mt-4 w-full bg-gray-200 rounded-full h-2.5">
              <div 
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${executionProgress.progress}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="flex border-b border-gray-200 mb-6">
          {['Metrics Visualization', 'Scenario Builder', 'Execution Monitor', 'Results Comparison', 'Report Generator'].map((tab, index) => (
            <button
              key={index}
              onClick={() => setActiveTab(index)}
              className={`px-4 py-2 font-medium transition-colors ${
                activeTab === index
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-700 hover:text-blue-600 border-b-2 border-transparent hover:border-blue-300'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="space-y-6">
          {/* Metrics Visualization Section */}
          {activeTab === 0 && (
            <div className="bg-gray-50 rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Metrics Visualization</h2>
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <MetricsVisualization benchmarkResults={benchmarkResults} />
              </div>
            </div>
          )}

          {/* Scenario Builder Section */}
          {activeTab === 1 && (
            <div className="bg-gray-50 rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Scenario Builder</h2>
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <ScenarioBuilder />
              </div>
            </div>
          )}

          {/* Execution Monitor Section */}
          {activeTab === 2 && (
            <div className="bg-gray-50 rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Execution Monitor</h2>
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <ExecutionMonitor />
              </div>
            </div>
          )}

          {/* Results Comparison Section */}
          {activeTab === 3 && (
            <div className="bg-gray-50 rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Results Comparison</h2>
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <ResultsComparison benchmarkResults={benchmarkResults} />
              </div>
            </div>
          )}

          {/* Report Generator Section */}
          {activeTab === 4 && (
            <div className="bg-gray-50 rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Report Generator</h2>
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <ReportGenerator benchmarkResults={benchmarkResults} />
              </div>
            </div>
          )}
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default BenchmarkDashboard;