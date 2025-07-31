import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar
} from 'recharts';
import type { ChartConfiguration, TimeSeriesData } from '../../types';

interface FinancialChartProps {
  data: TimeSeriesData[];
  config: ChartConfiguration;
}

const FinancialChart: React.FC<FinancialChartProps> = ({ data, config }) => {
  if (!data || data.length === 0) {
    return (
      <div className="bg-white p-4 rounded shadow-md h-64 flex items-center justify-center text-gray-500">
        No financial data available for this chart.
      </div>
    );
  }

  const { chartType, metrics, title, xAxisLabel, yAxisLabel } = config;

  const renderChart = () => {
    switch (chartType) {
      case 'line':
        return (
          <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="tick" label={{ value: xAxisLabel || 'Simulation Tick', position: 'insideBottom', offset: 0 }} />
            <YAxis label={{ value: yAxisLabel || 'Value', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            {metrics.map((metric, index) => (
              <Line
                key={metric}
                type="monotone"
                dataKey={metric}
                stroke={['#8884d8', '#82ca9d', '#ffc658', '#d0ed57', '#a4de6c'][index % 5]}
                activeDot={{ r: 8 }}
                name={metric.charAt(0).toUpperCase() + metric.slice(1)}
              />
            ))}
          </LineChart>
        );
      case 'area':
        return (
          <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="tick" label={{ value: xAxisLabel || 'Simulation Tick', position: 'insideBottom', offset: 0 }} />
            <YAxis label={{ value: yAxisLabel || 'Value', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            {metrics.map((metric, index) => (
              <Area
                key={metric}
                type="monotone"
                dataKey={metric}
                stroke={['#8884d8', '#82ca9d', '#ffc658', '#d0ed57', '#a4de6c'][index % 5]}
                fill={['#8884d8', '#82ca9d', '#ffc658', '#d0ed57', '#a4de6c'][index % 5]}
                fillOpacity={0.3}
                name={metric.charAt(0).toUpperCase() + metric.slice(1)}
              />
            ))}
          </AreaChart>
        );
      case 'bar':
        return (
          <BarChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="tick" label={{ value: xAxisLabel || 'Simulation Tick', position: 'insideBottom', offset: 0 }} />
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
      <h3 className="text-lg font-semibold text-gray-800 mb-2">{title || 'Financial Chart'}</h3>
      <ResponsiveContainer width="100%" height="100%">
        {renderChart()}
      </ResponsiveContainer>
    </div>
  );
};

export default FinancialChart;