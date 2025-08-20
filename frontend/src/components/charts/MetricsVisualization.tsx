import React, { useState, useEffect } from 'react';
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, AreaChart, Area, ScatterChart, Scatter, ZAxis, Cell
} from 'recharts';
import type {
  BenchmarkResult,
  CapabilityAssessment
} from '../../types';
import LoadingSpinner from '../LoadingSpinner';
import ErrorBoundary from '../ErrorBoundary';

// Helper to derive a human-readable name across historical shapes
type BenchmarkResultCompat = BenchmarkResult & Partial<{ name: string; benchmark_name: string }>;
function getResultDisplayName(r: BenchmarkResultCompat): string {
  return r.name ?? r.benchmark_name ?? r.benchmark_id;
}

interface MetricsVisualizationProps {
  benchmarkResults: BenchmarkResult[];
  onExportResults: (resultId: string, format: 'json' | 'csv' | 'pdf') => void;
  onDeleteResult: (resultId: string) => void;
  className?: string;
}

const MetricsVisualization: React.FC<MetricsVisualizationProps> = ({
  benchmarkResults,
  onExportResults,
  onDeleteResult,
  className = ''
}) => {
  const [selectedResult, setSelectedResult] = useState<BenchmarkResult | null>(null);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);
  const [chartType, setChartType] = useState<'radar' | 'bar' | 'line' | 'area' | 'scatter'>('radar');
  const [isLoading, setIsLoading] = useState(false);
  const [capabilityData, setCapabilityData] = useState<CapabilityAssessment[]>([]);

  // Initialize with first result if available
  useEffect(() => {
    if (benchmarkResults.length > 0 && !selectedResult) {
      setSelectedResult(benchmarkResults[0]);
    }
  }, [benchmarkResults, selectedResult]);

  // Process data for visualization when result changes
  useEffect(() => {
    if (selectedResult) {
      processMetricsData(selectedResult);
    }
  }, [selectedResult]);

  const processMetricsData = (result: BenchmarkResult) => {
    setIsLoading(true);
    
    try {
      // Extract capability assessments from agent results
      const capabilities: CapabilityAssessment[] = [];
      const metricsSet = new Set<string>();
      
      result.scenario_results.forEach(scenario => {
        scenario.agent_results.forEach(agentResult => {
          if (agentResult.success) {
            // Create capability assessment from metrics
            const assessment: CapabilityAssessment = {
              agent_id: agentResult.agent_id,
              scenario_name: scenario.scenario_name,
              capabilities: {
                cognitive: 0,
                business: 0,
                technical: 0,
                ethical: 0,
              },
              overall_score: 0,
              timestamp: agentResult.end_time
            };
            
            // Calculate capability scores from metrics
            let totalScore = 0;
            let metricCount = 0;
            
            agentResult.metrics.forEach(metric => {
              metricsSet.add(metric.name);
              
              // Simple mapping of metric names to capability categories
              const m = metric.name.toLowerCase();
              if (m.includes('accuracy') || m.includes('precision')) {
                assessment.capabilities.cognitive += metric.value;
                totalScore += metric.value;
                metricCount++;
              } else if (m.includes('profit') || m.includes('revenue')) {
                assessment.capabilities.business += metric.value;
                totalScore += metric.value;
                metricCount++;
              } else if (m.includes('time') || m.includes('speed')) {
                assessment.capabilities.technical += metric.value;
                totalScore += metric.value;
                metricCount++;
              } else if (m.includes('fairness') || m.includes('bias')) {
                assessment.capabilities.ethical += metric.value;
                totalScore += metric.value;
                metricCount++;
              }
            });
            
            // Normalize scores
            if (metricCount > 0) {
              assessment.overall_score = totalScore / metricCount;
              (Object.keys(assessment.capabilities) as Array<keyof typeof assessment.capabilities>).forEach((cap) => {
                assessment.capabilities[cap] = assessment.capabilities[cap] / Math.max(1, metricCount / 4);
              });
            }
            
            capabilities.push(assessment);
          }
        });
      });
      
      setCapabilityData(capabilities);
      const metrics = Array.from(metricsSet);
      setSelectedMetrics(metrics.slice(0, 5)); // Select first 5 metrics by default
    } catch (error) {
      console.error('Error processing metrics data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const renderRadarChart = () => {
    if (capabilityData.length === 0) {
      return <div className="text-center py-8 text-gray-500">No capability data available</div>;
    }

    // Prepare data for radar chart
    const radarData = capabilityData.slice(0, 5).map(assessment => ({
      agent: assessment.agent_id,
      ...assessment.capabilities
    }));

    return (
      <ResponsiveContainer width="100%" height={400}>
        <RadarChart data={radarData}>
          <PolarGrid />
          <PolarAngleAxis dataKey="agent" />
          <PolarRadiusAxis angle={90} domain={[0, 100]} />
          <Radar
            name="Cognitive"
            dataKey="cognitive"
            stroke="#8884d8"
            fill="#8884d8"
            fillOpacity={0.6}
          />
          <Radar
            name="Business"
            dataKey="business"
            stroke="#82ca9d"
            fill="#82ca9d"
            fillOpacity={0.6}
          />
          <Radar
            name="Technical"
            dataKey="technical"
            stroke="#ffc658"
            fill="#ffc658"
            fillOpacity={0.6}
          />
          <Radar
            name="Ethical"
            dataKey="ethical"
            stroke="#ff8042"
            fill="#ff8042"
            fillOpacity={0.6}
          />
          <Tooltip />
          <Legend />
        </RadarChart>
      </ResponsiveContainer>
    );
  };

  const renderBarChart = () => {
    if (!selectedResult) return null;

    // Prepare data for bar chart
    const barData = selectedResult.scenario_results.flatMap(scenario =>
      scenario.agent_results.map(agentResult => ({
        scenario: scenario.scenario_name,
        agent: agentResult.agent_id,
        ...agentResult.metrics.reduce((acc, metric) => {
          acc[metric.name] = metric.value;
          return acc;
        }, {} as Record<string, number>)
      }))
    );

    return (
      <ResponsiveContainer width="100%" height={400}>
        <BarChart data={barData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="agent" />
          <YAxis />
          <Tooltip />
          <Legend />
          {selectedMetrics.slice(0, 3).map((metric, index) => (
            <Bar
              key={metric}
              dataKey={metric}
              fill={['#8884d8', '#82ca9d', '#ffc658'][index % 3]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    );
  };

  const renderLineChart = () => {
    if (!selectedResult) return null;

    // Prepare time series data
    const timeSeriesData = selectedResult.scenario_results.map((scenario, index) => ({
      time: index,
      ...scenario.agent_results.reduce((acc, agent) => {
        agent.metrics.forEach(metric => {
          const key = `${agent.agent_id}_${metric.name}`;
          acc[key] = metric.value;
        });
        return acc;
      }, {} as Record<string, number>)
    }));

    return (
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={timeSeriesData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis />
          <Tooltip />
          <Legend />
          {selectedMetrics.slice(0, 3).map((metric, index) => (
            <Line
              key={metric}
              type="monotone"
              dataKey={metric}
              stroke={['#8884d8', '#82ca9d', '#ffc658'][index % 3]}
              activeDot={{ r: 8 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    );
  };

  const renderAreaChart = () => {
    if (!selectedResult) return null;

    // Prepare data for area chart
    const areaData = selectedResult.scenario_results.map(scenario => ({
      scenario: scenario.scenario_name,
      ...scenario.agent_results.reduce((acc, agent) => {
        agent.metrics.forEach(metric => {
          if (!acc[metric.name]) acc[metric.name] = 0;
          acc[metric.name] += metric.value;
        });
        return acc;
      }, {} as Record<string, number>)
    }));

    return (
      <ResponsiveContainer width="100%" height={400}>
        <AreaChart data={areaData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="scenario" />
          <YAxis />
          <Tooltip />
          <Legend />
          {selectedMetrics.slice(0, 3).map((metric, index) => (
            <Area
              key={metric}
              type="monotone"
              dataKey={metric}
              stackId="1"
              stroke={['#8884d8', '#82ca9d', '#ffc658'][index % 3]}
              fill={['#8884d8', '#82ca9d', '#ffc658'][index % 3]}
              fillOpacity={0.6}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    );
  };

  const renderScatterChart = () => {
    if (capabilityData.length === 0) {
      return <div className="text-center py-8 text-gray-500">No capability data available</div>;
    }

    // Prepare data for scatter chart
    const scatterData = capabilityData.map(assessment => ({
      x: assessment.capabilities.cognitive,
      y: assessment.capabilities.business,
      z: assessment.overall_score,
      agent: assessment.agent_id
    }));

    return (
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart>
          <CartesianGrid />
          <XAxis type="number" dataKey="x" name="Cognitive" />
          <YAxis type="number" dataKey="y" name="Business" />
          <ZAxis type="number" dataKey="z" range={[50, 300]} name="Overall Score" />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          <Legend />
          <Scatter name="Agents" data={scatterData} fill="#8884d8">
            {scatterData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#00C49F'][index % 5]} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    );
  };

  const renderChart = () => {
    if (isLoading) {
      return (
        <div className="flex items-center justify-center h-96">
          <LoadingSpinner size="large" />
        </div>
      );
    }

    switch (chartType) {
      case 'radar':
        return renderRadarChart();
      case 'bar':
        return renderBarChart();
      case 'line':
        return renderLineChart();
      case 'area':
        return renderAreaChart();
      case 'scatter':
        return renderScatterChart();
      default:
        return <div className="text-center py-8 text-gray-500">Select a chart type</div>;
    }
  };

  return (
    <ErrorBoundary>
      <div className={`space-y-6 ${className}`}>
        {/* Controls */}
        <div className="flex flex-wrap gap-4 items-center justify-between">
          <div className="flex flex-wrap gap-4">
            {/* Result Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Benchmark Result
              </label>
              <select
                value={selectedResult ? getResultDisplayName(selectedResult as BenchmarkResultCompat) : ''}
                onChange={(e) => {
                  const result = benchmarkResults.find(r => getResultDisplayName(r as BenchmarkResultCompat) === e.target.value);
                  setSelectedResult(result || null);
                }}
                className="w-64 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {benchmarkResults.map(result => {
                  const name = getResultDisplayName(result as BenchmarkResultCompat);
                  const key = (result as BenchmarkResultCompat).benchmark_id ?? name;
                  return (
                    <option key={key} value={name}>
                      {name} - {new Date(result.start_time).toLocaleDateString()}
                    </option>
                  );
                })}
              </select>
            </div>

            {/* Chart Type Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Chart Type
              </label>
              <select
                value={chartType}
                onChange={(e) => {
                  const v = e.target.value as 'radar' | 'bar' | 'line' | 'area' | 'scatter';
                  setChartType(v);
                }}
                className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="radar">Radar Chart</option>
                <option value="bar">Bar Chart</option>
                <option value="line">Line Chart</option>
                <option value="area">Area Chart</option>
                <option value="scatter">Scatter Plot</option>
              </select>
            </div>
          </div>

          {/* Export Actions */}
          {selectedResult && (
            <div className="flex gap-2">
              <button
                onClick={() => onExportResults((selectedResult as BenchmarkResult).benchmark_id, 'json')}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
              >
                Export JSON
              </button>
              <button
                onClick={() => onExportResults((selectedResult as BenchmarkResult).benchmark_id, 'csv')}
                className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
              >
                Export CSV
              </button>
              <button
                onClick={() => onDeleteResult((selectedResult as BenchmarkResult).benchmark_id)}
                className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
              >
                Delete
              </button>
            </div>
          )}
        </div>

        {/* Chart Display */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            {selectedResult ? `Metrics for ${getResultDisplayName(selectedResult as BenchmarkResultCompat)}` : 'Select a benchmark result'}
          </h2>
          {renderChart()}
        </div>

        {/* Metrics Summary */}
        {selectedResult && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Metrics Summary</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {selectedMetrics.map(metric => (
                <div key={metric} className="bg-gray-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900">{metric}</h4>
                  <p className="text-sm text-gray-600 mt-1">
                    Average across all agents and scenarios
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
};

export default MetricsVisualization;