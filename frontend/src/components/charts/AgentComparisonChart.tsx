import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis
} from 'recharts';
import type { ChartConfiguration, AgentPerformanceData } from '../../types';

// Helper to determine if a value is a number
const isNumber = (value: any): value is number => typeof value === 'number';

interface AgentComparisonChartProps {
  data: AgentPerformanceData[];
  config: ChartConfiguration;
}

const AgentComparisonChart: React.FC<AgentComparisonChartProps> = ({ data, config }) => {
  if (!data || data.length === 0) {
    return (
      <div className="bg-white p-4 rounded shadow-md h-64 flex items-center justify-center text-gray-500">
        No agent performance data available for this chart.
      </div>
    );
  }

  const { chartType, metrics, title, xAxisLabel, yAxisLabel, groupBy } = config;

  // Prepare data for agent comparison, potentially grouping or sorting
  const processedData = groupBy && metrics.length > 0
    ? [...data].sort((a, b) => {
        const valueA = (a as any)[metrics[0]];
        const valueB = (b as any)[metrics[0]];
        return (isNumber(valueB) ? valueB : 0) - (isNumber(valueA) ? valueA : 0);
      }) // Sort by the first metric for ranking
    : data;

  const renderChart = () => {
    switch (chartType) {
      case 'bar':
        return (
          <BarChart data={processedData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={groupBy || 'agentId'} label={{ value: xAxisLabel || 'Agent ID', position: 'insideBottom', offset: 0 }} />
            <YAxis label={{ value: yAxisLabel || 'Value', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            {metrics.map((metric, index) => (
              <Bar
                key={metric}
                dataKey={metric}
                fill={['#8884d8', '#82ca9d', '#ffc658', '#d0ed57', '#a4de6c'][index % 5]}
                name={metric.charAt(0).toUpperCase() + metric.slice(1)}
              />
            ))}
          </BarChart>
        );
      case 'scatter':
        // Assuming metrics[0] is for X-axis, metrics[1] for Y-axis, and optionally metrics[2] for Z-axis (size/color)
        if (metrics.length < 2) {
          return (
            <div className="bg-white p-4 rounded shadow-md h-64 flex items-center justify-center text-red-500">
              Scatter chart requires at least two metrics.
            </div>
          );
        }
        return (
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid />
            <XAxis type="number" dataKey={metrics[0]} name={xAxisLabel || metrics[0].charAt(0).toUpperCase() + metrics[0].slice(1)} />
            <YAxis type="number" dataKey={metrics[1]} name={yAxisLabel || metrics[1].charAt(0).toUpperCase() + metrics[1].slice(1)} />
            {metrics[2] && <ZAxis type="number" dataKey={metrics[2]} range={[60, 400]} name={metrics[2].charAt(0).toUpperCase() + metrics[2].slice(1)} />}
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Legend />
            <Scatter name={title || 'Agents'} data={data} fill="#8884d8" />
          </ScatterChart>
        );
      default:
        return (
          <div className="bg-white p-4 rounded shadow-md h-64 flex items-center justify-center text-red-500">
            Unsupported chart type: {chartType}
          </div>
        );
    }
  };

  return (
    <div className="bg-white p-4 rounded shadow-md h-96 flex flex-col">
      <h3 className="text-lg font-semibold text-gray-800 mb-2">{title || 'Agent Performance Chart'}</h3>
      <ResponsiveContainer width="100%" height="100%">
        {renderChart()}
      </ResponsiveContainer>
    </div>
  );
};

export default AgentComparisonChart;