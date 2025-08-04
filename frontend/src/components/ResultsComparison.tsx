import React, { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';
import type { 
  BenchmarkResult, 
  ScenarioResult, 
  AgentRunResult,
  MetricResult,
  ExportOptions 
} from '../types';
import { notificationService } from '../utils/notificationService';
import LoadingSpinner from './LoadingSpinner';
import ErrorBoundary from './ErrorBoundary';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ComposedChart,
  Area,
  AreaChart
} from 'recharts';

interface ResultsComparisonProps {
  benchmarkResults: BenchmarkResult[];
  className?: string;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

const ResultsComparison: React.FC<ResultsComparisonProps> = ({ 
  benchmarkResults, 
  className = '' 
}) => {
  const [selectedResults, setSelectedResults] = useState<string[]>([]);
  const [comparisonData, setComparisonData] = useState<any[]>([]);
  const [detailedResults, setDetailedResults] = useState<Record<string, BenchmarkResult>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [comparisonType, setComparisonType] = useState<'overview' | 'agents' | 'scenarios' | 'metrics'>('overview');
  const [selectedMetric, setSelectedMetric] = useState<string>('overall_score');
  const [chartType, setChartType] = useState<'bar' | 'line' | 'radar' | 'scatter'>('bar');

  useEffect(() => {
    if (benchmarkResults.length > 0) {
      // Auto-select the first 2 results for comparison
      const initialSelection = benchmarkResults
        .slice(0, 2)
        .map(result => result.benchmark_name);
      setSelectedResults(initialSelection);
    }
  }, [benchmarkResults]);

  useEffect(() => {
    if (selectedResults.length > 0) {
      fetchDetailedResults();
      generateComparisonData();
    }
  }, [selectedResults, comparisonType, selectedMetric]);

  const fetchDetailedResults = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const detailedData: Record<string, BenchmarkResult> = {};
      
      for (const resultId of selectedResults) {
        const response = await apiService.get<BenchmarkResult>(`/benchmarking/results/${resultId}`);
        detailedData[resultId] = response.data;
      }
      
      setDetailedResults(detailedData);
    } catch (err) {
      console.error('Error fetching detailed results:', err);
      setError('Failed to fetch detailed results');
    } finally {
      setIsLoading(false);
    }
  };

  const generateComparisonData = () => {
    const data: any[] = [];
    
    if (comparisonType === 'overview') {
      selectedResults.forEach(resultId => {
        const result = detailedResults[resultId];
        if (result) {
          data.push({
            name: result.benchmark_name,
            duration: result.duration_seconds,
            scenarios: result.scenario_results.length,
            avgScore: calculateAverageScore(result),
            successRate: calculateSuccessRate(result)
          });
        }
      });
    } else if (comparisonType === 'agents') {
      // Get all unique agent IDs across selected results
      const allAgents = new Set<string>();
      selectedResults.forEach(resultId => {
        const result = detailedResults[resultId];
        if (result) {
          result.scenario_results.forEach(scenario => {
            scenario.agent_results.forEach(agent => {
              allAgents.add(agent.agent_id);
            });
          });
        }
      });
      
      allAgents.forEach(agentId => {
        const agentData: any = { name: agentId };
        
        selectedResults.forEach(resultId => {
          const result = detailedResults[resultId];
          if (result) {
            const agentResults = result.scenario_results.flatMap(scenario => 
              scenario.agent_results.filter(agent => agent.agent_id === agentId)
            );
            
            if (agentResults.length > 0) {
              const avgScore = agentResults.reduce((sum, agent) => {
                const score = agent.metrics.find(m => m.name === selectedMetric)?.value || 0;
                return sum + score;
              }, 0) / agentResults.length;
              
              agentData[resultId] = avgScore;
            } else {
              agentData[resultId] = 0;
            }
          }
        });
        
        data.push(agentData);
      });
    } else if (comparisonType === 'scenarios') {
      // Get all unique scenario names across selected results
      const allScenarios = new Set<string>();
      selectedResults.forEach(resultId => {
        const result = detailedResults[resultId];
        if (result) {
          result.scenario_results.forEach(scenario => {
            allScenarios.add(scenario.scenario_name);
          });
        }
      });
      
      allScenarios.forEach(scenarioName => {
        const scenarioData: any = { name: scenarioName };
        
        selectedResults.forEach(resultId => {
          const result = detailedResults[resultId];
          if (result) {
            const scenario = result.scenario_results.find(s => s.scenario_name === scenarioName);
            if (scenario) {
              const avgScore = scenario.agent_results.reduce((sum, agent) => {
                const score = agent.metrics.find(m => m.name === selectedMetric)?.value || 0;
                return sum + score;
              }, 0) / scenario.agent_results.length;
              
              scenarioData[resultId] = avgScore;
            } else {
              scenarioData[resultId] = 0;
            }
          }
        });
        
        data.push(scenarioData);
      });
    } else if (comparisonType === 'metrics') {
      // Get all unique metric names across selected results
      const allMetrics = new Set<string>();
      selectedResults.forEach(resultId => {
        const result = detailedResults[resultId];
        if (result) {
          result.scenario_results.forEach(scenario => {
            scenario.agent_results.forEach(agent => {
              agent.metrics.forEach(metric => {
                allMetrics.add(metric.name);
              });
            });
          });
        }
      });
      
      allMetrics.forEach(metricName => {
        const metricData: any = { name: metricName };
        
        selectedResults.forEach(resultId => {
          const result = detailedResults[resultId];
          if (result) {
            const allMetricValues = result.scenario_results.flatMap(scenario => 
              scenario.agent_results.flatMap(agent => 
                agent.metrics.filter(m => m.name === metricName).map(m => m.value)
              )
            );
            
            if (allMetricValues.length > 0) {
              const avgValue = allMetricValues.reduce((sum, value) => sum + value, 0) / allMetricValues.length;
              metricData[resultId] = avgValue;
            } else {
              metricData[resultId] = 0;
            }
          }
        });
        
        data.push(metricData);
      });
    }
    
    setComparisonData(data);
  };

  const calculateAverageScore = (result: BenchmarkResult): number => {
    const allScores = result.scenario_results.flatMap(scenario => 
      scenario.agent_results.flatMap(agent => 
        agent.metrics.filter(m => m.name === 'overall_score').map(m => m.value)
      )
    );
    
    if (allScores.length === 0) return 0;
    return allScores.reduce((sum, score) => sum + score, 0) / allScores.length;
  };

  const calculateSuccessRate = (result: BenchmarkResult): number => {
    const allRuns = result.scenario_results.flatMap(scenario => scenario.agent_results);
    if (allRuns.length === 0) return 0;
    
    const successfulRuns = allRuns.filter(run => run.success).length;
    return (successfulRuns / allRuns.length) * 100;
  };

  const handleResultToggle = (resultId: string) => {
    setSelectedResults(prev => 
      prev.includes(resultId)
        ? prev.filter(id => id !== resultId)
        : [...prev, resultId]
    );
  };

  const exportResults = async (format: 'json' | 'csv' | 'pdf') => {
    if (selectedResults.length === 0) {
      notificationService.error('No results selected for export', 3000);
      return;
    }

    try {
      const exportOptions: ExportOptions = {
        format,
        include_metadata: true,
        include_raw_data: true,
        include_charts: false,
        include_analysis: true
      };

      const response = await apiService.post(
        '/benchmarking/results/export',
        { resultIds: selectedResults, options: exportOptions },
        { headers: { 'Accept': format === 'pdf' ? 'application/pdf' : 'application/json' } }
      );
      
      // Create download link
      const blob = new Blob([JSON.stringify(response.data, null, 2)], { 
        type: format === 'pdf' ? 'application/pdf' : 'application/json' 
      });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `benchmark_comparison_${new Date().toISOString().split('T')[0]}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      notificationService.success(`Results exported as ${format.toUpperCase()}`, 3000);
    } catch (err) {
      console.error('Error exporting results:', err);
      setError('Failed to export results');
    }
  };

  const renderChart = () => {
    if (comparisonData.length === 0) {
      return <p className="text-gray-500 text-center py-8">No data to display</p>;
    }

    const chartProps = {
      width: '100%',
      height: 400,
      data: comparisonData,
      margin: { top: 20, right: 30, left: 20, bottom: 50 }
    };

    switch (chartType) {
      case 'bar':
        return (
          <ResponsiveContainer {...chartProps}>
            <BarChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
              <YAxis />
              <Tooltip />
              <Legend />
              {selectedResults.map((resultId, index) => (
                <Bar 
                  key={resultId} 
                  dataKey={resultId} 
                  fill={COLORS[index % COLORS.length]} 
                  name={detailedResults[resultId]?.benchmark_name || resultId}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        );
      
      case 'line':
        return (
          <ResponsiveContainer {...chartProps}>
            <LineChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
              <YAxis />
              <Tooltip />
              <Legend />
              {selectedResults.map((resultId, index) => (
                <Line 
                  key={resultId} 
                  type="monotone" 
                  dataKey={resultId} 
                  stroke={COLORS[index % COLORS.length]} 
                  strokeWidth={2}
                  name={detailedResults[resultId]?.benchmark_name || resultId}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        );
      
      case 'radar':
        return (
          <ResponsiveContainer {...chartProps}>
            <RadarChart data={comparisonData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="name" />
              <PolarRadiusAxis />
              {selectedResults.map((resultId, index) => (
                <Radar
                  key={resultId}
                  name={detailedResults[resultId]?.benchmark_name || resultId}
                  dataKey={resultId}
                  stroke={COLORS[index % COLORS.length]}
                  fill={COLORS[index % COLORS.length]}
                  fillOpacity={0.6}
                />
              ))}
              <Tooltip />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        );
      
      case 'scatter':
        if (selectedResults.length === 2) {
          const scatterData = comparisonData.map(item => ({
            x: item[selectedResults[0]] || 0,
            y: item[selectedResults[1]] || 0,
            name: item.name
          }));
          
          return (
            <ResponsiveContainer {...chartProps}>
              <ScatterChart data={scatterData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  type="number" 
                  dataKey="x" 
                  name={detailedResults[selectedResults[0]]?.benchmark_name || selectedResults[0]} 
                />
                <YAxis 
                  type="number" 
                  dataKey="y" 
                  name={detailedResults[selectedResults[1]]?.benchmark_name || selectedResults[1]} 
                />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter dataKey="y" fill="#8884d8" />
              </ScatterChart>
            </ResponsiveContainer>
          );
        }
        return <p className="text-gray-500 text-center py-8">Select exactly 2 results for scatter plot</p>;
      
      default:
        return <p className="text-gray-500 text-center py-8">Invalid chart type</p>;
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
      <div className={`space-y-6 ${className}`}>
        <h2 className="text-2xl font-bold text-gray-900">Results Comparison</h2>
        
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

        {/* Result Selection */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Select Results to Compare</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {benchmarkResults.map(result => (
              <label key={result.benchmark_name} className="flex items-center space-x-2 p-3 border border-gray-200 rounded-md hover:bg-gray-50 cursor-pointer">
                <input
                  type="checkbox"
                  checked={selectedResults.includes(result.benchmark_name)}
                  onChange={() => handleResultToggle(result.benchmark_name)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <div className="flex-1">
                  <div className="text-sm font-medium text-gray-900">{result.benchmark_name}</div>
                  <div className="text-xs text-gray-500">
                    {new Date(result.start_time).toLocaleDateString()} • {result.scenario_results.length} scenarios
                  </div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Comparison Controls */}
        {selectedResults.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex flex-wrap gap-4 items-center">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Comparison Type
                </label>
                <select
                  value={comparisonType}
                  onChange={(e) => setComparisonType(e.target.value as any)}
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="overview">Overview</option>
                  <option value="agents">Agents</option>
                  <option value="scenarios">Scenarios</option>
                  <option value="metrics">Metrics</option>
                </select>
              </div>

              {(comparisonType === 'agents' || comparisonType === 'scenarios' || comparisonType === 'metrics') && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Metric
                  </label>
                  <select
                    value={selectedMetric}
                    onChange={(e) => setSelectedMetric(e.target.value)}
                    className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="overall_score">Overall Score</option>
                    <option value="cognitive_score">Cognitive Score</option>
                    <option value="business_score">Business Score</option>
                    <option value="technical_score">Technical Score</option>
                    <option value="execution_time">Execution Time</option>
                    <option value="accuracy">Accuracy</option>
                    <option value="efficiency">Efficiency</option>
                  </select>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Chart Type
                </label>
                <select
                  value={chartType}
                  onChange={(e) => setChartType(e.target.value as any)}
                  className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="bar">Bar Chart</option>
                  <option value="line">Line Chart</option>
                  <option value="radar">Radar Chart</option>
                  {selectedResults.length === 2 && (
                    <option value="scatter">Scatter Plot</option>
                  )}
                </select>
              </div>

              <div className="flex items-end space-x-2">
                <button
                  onClick={() => exportResults('json')}
                  className="px-3 py-2 bg-gray-600 text-white text-sm rounded-md hover:bg-gray-700 transition-colors"
                >
                  Export JSON
                </button>
                <button
                  onClick={() => exportResults('csv')}
                  className="px-3 py-2 bg-gray-600 text-white text-sm rounded-md hover:bg-gray-700 transition-colors"
                >
                  Export CSV
                </button>
                <button
                  onClick={() => exportResults('pdf')}
                  className="px-3 py-2 bg-gray-600 text-white text-sm rounded-md hover:bg-gray-700 transition-colors"
                >
                  Export PDF
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Comparison Chart */}
        {selectedResults.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              {comparisonType.charAt(0).toUpperCase() + comparisonType.slice(1)} Comparison
            </h3>
            {renderChart()}
          </div>
        )}

        {/* Detailed Results Table */}
        {selectedResults.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Results</h3>
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
                      Avg Score
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Success Rate
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {selectedResults.map(resultId => {
                    const result = detailedResults[resultId];
                    if (!result) return null;
                    
                    return (
                      <tr key={resultId} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {result.benchmark_name}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {Math.floor(result.duration_seconds / 60)}m {result.duration_seconds % 60}s
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {result.scenario_results.length}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {calculateAverageScore(result).toFixed(3)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {calculateSuccessRate(result).toFixed(1)}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
};

export default ResultsComparison;