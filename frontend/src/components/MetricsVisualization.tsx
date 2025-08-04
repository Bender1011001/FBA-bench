import React, { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';
import type { 
  MultiDimensionalMetric, 
  CapabilityAssessment, 
  PerformanceHeatmap,
  BenchmarkResult 
} from '../types';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  HeatmapChart,
  Heatmap,
  ZAxis
} from 'recharts';
import LoadingSpinner from './LoadingSpinner';
import ErrorBoundary from './ErrorBoundary';

interface MetricsVisualizationProps {
  benchmarkResults: BenchmarkResult[];
  className?: string;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

const MetricsVisualization: React.FC<MetricsVisualizationProps> = ({ 
  benchmarkResults, 
  className = '' 
}) => {
  const [metrics, setMetrics] = useState<MultiDimensionalMetric[]>([]);
  const [capabilityAssessments, setCapabilityAssessments] = useState<CapabilityAssessment[]>([]);
  const [performanceHeatmap, setPerformanceHeatmap] = useState<PerformanceHeatmap | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<string>('overall_score');
  const [timeRange, setTimeRange] = useState<string>('all');

  useEffect(() => {
    if (benchmarkResults.length > 0) {
      fetchMetricsData();
    }
  }, [benchmarkResults]);

  const fetchMetricsData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Fetch multi-dimensional metrics
      const metricsResponse = await apiService.get<MultiDimensionalMetric[]>('/benchmarking/metrics');
      setMetrics(metricsResponse.data);
      
      // Fetch capability assessments
      const assessmentsResponse = await apiService.get<CapabilityAssessment[]>('/benchmarking/assessments');
      setCapabilityAssessments(assessmentsResponse.data);
      
      // Fetch performance heatmap
      const heatmapResponse = await apiService.get<PerformanceHeatmap>('/benchmarking/heatmap');
      setPerformanceHeatmap(heatmapResponse.data);
    } catch (err) {
      console.error('Error fetching metrics data:', err);
      setError('Failed to fetch metrics data');
    } finally {
      setIsLoading(false);
    }
  };

  // Process data for charts
  const getMetricsOverTime = () => {
    return metrics
      .filter(metric => metric.name === selectedMetric)
      .map(metric => ({
        timestamp: new Date(metric.timestamp).toLocaleString(),
        value: metric.values[0] || 0,
        category: metric.category
      }));
  };

  const getCapabilityRadarData = () => {
    if (capabilityAssessments.length === 0) return [];
    
    const latestAssessment = capabilityAssessments[capabilityAssessments.length - 1];
    return Object.entries(latestAssessment.capabilities).map(([key, value]) => ({
      capability: key.charAt(0).toUpperCase() + key.slice(1),
      score: value * 100,
      fullMark: 100
    }));
  };

  const getPerformanceDistribution = () => {
    const scoreRanges = [
      { range: '0-20', min: 0, max: 20, count: 0 },
      { range: '21-40', min: 21, max: 40, count: 0 },
      { range: '41-60', min: 41, max: 60, count: 0 },
      { range: '61-80', min: 61, max: 80, count: 0 },
      { range: '81-100', min: 81, max: 100, count: 0 }
    ];

    capabilityAssessments.forEach(assessment => {
      const score = assessment.overall_score * 100;
      const range = scoreRanges.find(r => score >= r.min && score <= r.max);
      if (range) range.count++;
    });

    return scoreRanges;
  };

  const getHeatmapData = () => {
    if (!performanceHeatmap) return [];
    
    const data = [];
    for (let i = 0; i < performanceHeatmap.agents.length; i++) {
      for (let j = 0; j < performanceHeatmap.scenarios.length; j++) {
        for (let k = 0; k < performanceHeatmap.metrics.length; k++) {
          data.push({
            agent: performanceHeatmap.agents[i],
            scenario: performanceHeatmap.scenarios[j],
            metric: performanceHeatmap.metrics[k],
            value: performanceHeatmap.data[i][j][k]
          });
        }
      }
    }
    return data;
  };

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center h-96 ${className}`}>
        <LoadingSpinner size="large" />
      </div>
    );
  }

  if (error) {
    return (
      <div className={`bg-red-50 border border-red-200 rounded-md p-4 ${className}`}>
        <h3 className="text-sm font-medium text-red-800">Error</h3>
        <p className="text-sm text-red-700 mt-1">{error}</p>
        <button
          onClick={fetchMetricsData}
          className="mt-2 text-sm text-red-600 hover:text-red-800"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <div className={`space-y-6 ${className}`}>
        {/* Controls */}
        <div className="flex flex-wrap gap-4 items-center">
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
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Time Range
            </label>
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Time</option>
              <option value="today">Today</option>
              <option value="week">Last Week</option>
              <option value="month">Last Month</option>
            </select>
          </div>
        </div>

        {/* Metrics Over Time Chart */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Metrics Over Time</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={getMetricsOverTime()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#8884d8" 
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Capability Assessment Radar Chart */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Capability Assessment</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={getCapabilityRadarData()}>
                <PolarGrid />
                <PolarAngleAxis dataKey="capability" />
                <PolarRadiusAxis angle={90} domain={[0, 100]} />
                <Radar
                  name="Capabilities"
                  dataKey="score"
                  stroke="#8884d8"
                  fill="#8884d8"
                  fillOpacity={0.6}
                />
                <Tooltip />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Performance Distribution */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Distribution</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={getPerformanceDistribution()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Performance Heatmap */}
        {performanceHeatmap && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Heatmap</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <HeatmapChart data={getHeatmapData()}>
                  <XAxis dataKey="agent" />
                  <YAxis dataKey="scenario" />
                  <Tooltip />
                  <Heatmap dataKey="value">
                    {getHeatmapData().map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Heatmap>
                </HeatmapChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
};

export default MetricsVisualization;