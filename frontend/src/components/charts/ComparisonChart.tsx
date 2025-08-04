import React, { useMemo } from 'react';
import type { BenchmarkResult, AgentRunResult, MultiDimensionalMetric } from '../../types';

interface ComparisonChartProps {
  data: BenchmarkResult[] | AgentRunResult[] | MultiDimensionalMetric[];
  title?: string;
  className?: string;
  chartType?: 'bar' | 'line' | 'scatter' | 'grouped-bar';
  metric?: string;
  groupBy?: 'agent' | 'scenario' | 'both';
  showLegend?: boolean;
  showGrid?: boolean;
  interactive?: boolean;
  onBarClick?: (data: { agent: string; scenario: string; value: number }) => void;
  selectedAgent?: string;
  selectedScenario?: string;
  height?: number;
}

interface ComparisonDataPoint {
  agent: string;
  scenario: string;
  value: number;
  category: string;
  group: string;
}

const ComparisonChart: React.FC<ComparisonChartProps> = ({
  data,
  title,
  className = '',
  chartType = 'bar',
  metric = 'performance',
  groupBy = 'agent',
  showLegend = true,
  showGrid = true,
  interactive = true,
  onBarClick,
  selectedAgent,
  selectedScenario,
  height = 400
}) => {
  const { chartData, categories, groups, maxValue, minValue } = useMemo(() => {
    if (!data || data.length === 0) {
      return {
        chartData: [],
        categories: [],
        groups: [],
        maxValue: 100,
        minValue: 0
      };
    }

    // Process data based on input type
    const processedData: ComparisonDataPoint[] = [];
    
    if (Array.isArray(data)) {
      data.forEach((item) => {
        if ('agent_results' in item) {
          // BenchmarkResult
          const benchmarkResult = item as BenchmarkResult;
          benchmarkResult.scenario_results.forEach(scenario => {
            scenario.agent_results.forEach(agent => {
              processedData.push({
                agent: agent.agent_id,
                scenario: scenario.scenario_name,
                value: agent.metrics.find(m => m.name === metric)?.value || 0,
                category: scenario.scenario_name,
                group: agent.agent_id
              });
            });
          });
        } else if ('agent_id' in item) {
          // AgentRunResult
          const agentResult = item as AgentRunResult;
          processedData.push({
            agent: agentResult.agent_id,
            scenario: agentResult.scenario_name,
            value: agentResult.metrics.find(m => m.name === metric)?.value || 0,
            category: agentResult.scenario_name,
            group: agentResult.agent_id
          });
        } else {
          // MultiDimensionalMetric
          const metricData = item as MultiDimensionalMetric;
          processedData.push({
            agent: metricData.agent_id,
            scenario: metricData.scenario_name,
            value: metricData.values.find(v => v.name === metric)?.value || 0,
            category: metricData.scenario_name,
            group: metricData.agent_id
          });
        }
      });
    }

    // Get unique categories and groups
    const categories = Array.from(new Set(processedData.map(d => d.category)));
    const groups = Array.from(new Set(processedData.map(d => d.group)));

    // Calculate min/max values
    const values = processedData.map(d => d.value);
    const maxValue = Math.max(...values);
    const minValue = Math.min(...values);

    return {
      chartData: processedData,
      categories,
      groups,
      maxValue,
      minValue
    };
  }, [data, metric, groupBy]);

  const handleBarClick = (dataPoint: ComparisonDataPoint) => {
    if (interactive && onBarClick) {
      onBarClick({
        agent: dataPoint.agent,
        scenario: dataPoint.scenario,
        value: dataPoint.value
      });
    }
  };

  if (!data || data.length === 0) {
    return (
      <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}>
        <h3 className="text-lg font-medium text-gray-900 mb-4">{title || 'Comparison Chart'}</h3>
        <div className="flex items-center justify-center h-64 text-gray-500">
          No data available for visualization
        </div>
      </div>
    );
  }

  const colors = [
    'rgb(59, 130, 246)',   // Blue
    'rgb(16, 185, 129)',   // Green
    'rgb(245, 101, 101)',  // Red
    'rgb(251, 191, 36)',   // Yellow
    'rgb(139, 92, 246)',   // Purple
    'rgb(236, 72, 153)'    // Pink
  ];

  const renderChart = () => {
    const chartWidth = 100;
    const chartHeight = 100;
    const padding = { top: 20, right: 30, bottom: 60, left: 60 };

    return (
      <div className="relative" style={{ height: `${height}px` }}>
        <svg
          viewBox={`${padding.left} ${padding.top} ${chartWidth - padding.left - padding.right} ${chartHeight - padding.top - padding.bottom}`}
          className="w-full h-full"
          style={{ maxHeight: `${height}px` }}
        >
          {/* Grid lines */}
          {showGrid && (
            <>
              {/* Horizontal grid lines */}
              {[0.2, 0.4, 0.6, 0.8].map((ratio) => {
                const y = padding.top + (chartHeight - padding.top - padding.bottom) * (1 - ratio);
                return (
                  <line
                    key={`h-${ratio}`}
                    x1={padding.left}
                    y1={y}
                    x2={chartWidth - padding.right}
                    y2={y}
                    stroke="#e5e7eb"
                    strokeWidth="0.5"
                  />
                );
              })}
            </>
          )}

          {/* Y-axis labels */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
            const value = minValue + (maxValue - minValue) * (1 - ratio);
            const y = padding.top + (chartHeight - padding.top - padding.bottom) * ratio;
            
            return (
              <text
                key={`y-${ratio}`}
                x={padding.left - 10}
                y={y + 5}
                textAnchor="end"
                className="text-xs fill-gray-600"
              >
                {value.toFixed(1)}
              </text>
            );
          })}

          {/* X-axis labels */}
          {categories.map((category, index) => {
            const x = padding.left + ((chartWidth - padding.left - padding.right) / categories.length) * (index + 0.5);
            
            return (
              <text
                key={`x-${index}`}
                x={x}
                y={chartHeight - padding.bottom + 20}
                textAnchor="middle"
                className="text-xs fill-gray-600"
              >
                {category}
              </text>
            );
          })}

          {/* Chart elements */}
          {chartData.map((dataPoint, index) => {
            const categoryIndex = categories.indexOf(dataPoint.category);
            const groupIndex = groups.indexOf(dataPoint.group);
            
            let x: number, y: number, width: number, height: number;
            
            if (chartType === 'bar' || chartType === 'grouped-bar') {
              const categoryWidth = (chartWidth - padding.left - padding.right) / categories.length;
              const barWidth = chartType === 'grouped-bar' 
                ? categoryWidth / groups.length * 0.8 
                : categoryWidth * 0.8;
              
              x = padding.left + categoryWidth * categoryIndex + (chartType === 'grouped-bar' 
                ? (categoryWidth / groups.length) * groupIndex + (categoryWidth / groups.length - barWidth) / 2
                : (categoryWidth - barWidth) / 2);
              
              y = padding.top + ((maxValue - dataPoint.value) / (maxValue - minValue || 1)) * (chartHeight - padding.top - padding.bottom);
              
              width = barWidth;
              height = ((dataPoint.value - minValue) / (maxValue - minValue || 1)) * (chartHeight - padding.top - padding.bottom);
            } else {
              // Line and scatter charts
              x = padding.left + ((chartWidth - padding.left - padding.right) / categories.length) * (categoryIndex + 0.5);
              y = padding.top + ((maxValue - dataPoint.value) / (maxValue - minValue || 1)) * (chartHeight - padding.top - padding.bottom);
              width = 0;
              height = 0;
            }

            const color = colors[groupIndex % colors.length];
            const isSelected = selectedAgent === dataPoint.agent && selectedScenario === dataPoint.scenario;

            if (chartType === 'bar' || chartType === 'grouped-bar') {
              return (
                <rect
                  key={index}
                  x={x}
                  y={y}
                  width={width}
                  height={height}
                  fill={color}
                  className={`cursor-pointer transition-all duration-200 hover:opacity-80 ${
                    isSelected ? 'opacity-100' : 'opacity-80'
                  }`}
                  onClick={() => interactive && handleBarClick(dataPoint)}
                />
              );
            } else if (chartType === 'scatter') {
              return (
                <circle
                  key={index}
                  cx={x}
                  cy={y}
                  r="4"
                  fill={color}
                  className={`cursor-pointer transition-all duration-200 hover:r-6 ${
                    isSelected ? 'r-6' : ''
                  }`}
                  onClick={() => interactive && handleBarClick(dataPoint)}
                />
              );
            } else {
              // Line chart - we'll need to aggregate points by group
              return null;
            }
          })}

          {/* Line chart lines */}
          {chartType === 'line' && groups.map((group, groupIndex) => {
            const groupData = chartData.filter(d => d.group === group);
            const sortedData = groupData.sort((a, b) => categories.indexOf(a.category) - categories.indexOf(b.category));
            
            const pathData = sortedData.length > 0 
              ? `M ${padding.left + ((chartWidth - padding.left - padding.right) / categories.length) * 0.5} ${padding.top + ((maxValue - sortedData[0].value) / (maxValue - minValue || 1)) * (chartHeight - padding.top - padding.bottom)} ` +
                sortedData.slice(1).map((point, index) => {
                  const x = padding.left + ((chartWidth - padding.left - padding.right) / categories.length) * (categories.indexOf(point.category) + 0.5);
                  const y = padding.top + ((maxValue - point.value) / (maxValue - minValue || 1)) * (chartHeight - padding.top - padding.bottom);
                  return `L ${x} ${y}`;
                }).join(' ')
              : '';

            return (
              <path
                key={groupIndex}
                d={pathData}
                fill="none"
                stroke={colors[groupIndex % colors.length]}
                strokeWidth="2"
                className="transition-all duration-200"
              />
            );
          })}
        </svg>

        {/* Y-axis label */}
        <div className="absolute left-2 top-1/2 transform -translate-y-1/2 -rotate-90 text-sm text-gray-600">
          {metric}
        </div>

        {/* X-axis label */}
        <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 text-sm text-gray-600">
          {groupBy === 'agent' ? 'Agents' : groupBy === 'scenario' ? 'Scenarios' : 'Categories'}
        </div>
      </div>
    );
  };

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}>
      {title && (
        <h3 className="text-lg font-medium text-gray-900 mb-4">{title}</h3>
      )}
      
      {renderChart()}

      {/* Legend */}
      {showLegend && (
        <div className="mt-4 flex flex-wrap gap-4 justify-center">
          {groups.map((group, index) => (
            <div key={index} className="flex items-center space-x-2">
              <div
                className="w-3 h-3 rounded"
                style={{ backgroundColor: colors[index % colors.length] }}
              />
              <span className="text-sm text-gray-600">{group}</span>
            </div>
          ))}
        </div>
      )}

      {/* Selected agent/scenario info */}
      {(selectedAgent || selectedScenario) && (
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <div className="flex flex-wrap gap-4 text-sm text-gray-600">
            {selectedAgent && (
              <div>
                <span className="font-medium">Agent:</span> {selectedAgent}
              </div>
            )}
            {selectedScenario && (
              <div>
                <span className="font-medium">Scenario:</span> {selectedScenario}
              </div>
            )}
            {chartData.find(d => d.agent === selectedAgent && d.scenario === selectedScenario) && (
              <div>
                <span className="font-medium">Value:</span> {
                  chartData.find(d => d.agent === selectedAgent && d.scenario === selectedScenario)?.value.toFixed(2)
                }
              </div>
            )}
          </div>
        </div>
      )}

      {/* Interactive tooltip */}
      {interactive && (
        <div className="mt-4 text-center">
          <button
            onClick={() => {
              if (chartData.length > 0) {
                const firstPoint = chartData[0];
                handleBarClick(firstPoint);
              }
            }}
            className="text-sm text-blue-600 hover:text-blue-800 transition-colors"
          >
            Click on elements to view detailed information
          </button>
        </div>
      )}
    </div>
  );
};

export default ComparisonChart;