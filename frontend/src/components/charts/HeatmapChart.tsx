import React, { useMemo } from 'react';
import type { PerformanceHeatmap, BenchmarkResult, AgentRunResult } from '../../types';

interface HeatmapChartProps {
  data: PerformanceHeatmap | BenchmarkResult[] | AgentRunResult[];
  title?: string;
  className?: string;
  showLegend?: boolean;
  colorScale?: 'sequential' | 'diverging' | 'categorical';
  metric?: string;
  interactive?: boolean;
  onCellClick?: (agent: string, scenario: string, metric: string, value: number) => void;
  selectedAgent?: string;
  selectedScenario?: string;
}

interface HeatmapCell {
  agent: string;
  scenario: string;
  value: number;
  x: number;
  y: number;
  width: number;
  height: number;
  color: string;
}

const HeatmapChart: React.FC<HeatmapChartProps> = ({
  data,
  title,
  className = '',
  showLegend = true,
  colorScale = 'sequential',
  metric = 'performance',
  interactive = true,
  onCellClick,
  selectedAgent,
  selectedScenario
}) => {
  const { cells, minValue, maxValue, colorScaleFunction, agents, scenarios } = useMemo(() => {
    if (!data || (Array.isArray(data) && data.length === 0)) {
      return {
        cells: [],
        minValue: 0,
        maxValue: 100,
        colorScaleFunction: (value: number) => '#e5e7eb',
        agents: [],
        scenarios: []
      };
    }

    // Extract data based on input type
    let heatmapData: PerformanceHeatmap | null = null;
    let processedData: { agent: string; scenario: string; value: number }[] = [];

    if ('agents' in data && 'scenarios' in data && 'data' in data) {
      // PerformanceHeatmap type
      heatmapData = data as PerformanceHeatmap;
      const { agents, scenarios, metrics, data: values } = heatmapData;
      
      // Flatten the 3D array into individual cells
      processedData = [];
      for (let i = 0; i < agents.length; i++) {
        for (let j = 0; j < scenarios.length; j++) {
          for (let k = 0; k < metrics.length; k++) {
            processedData.push({
              agent: agents[i],
              scenario: scenarios[j],
              value: values[i][j][k]
            });
          }
        }
      }
    } else {
      // Array of BenchmarkResult or AgentRunResult
      processedData = (data as (BenchmarkResult | AgentRunResult)[]).flatMap(item => {
        if ('agent_results' in item) {
          // BenchmarkResult
          return item.scenario_results.flatMap(scenario =>
            scenario.agent_results.map(agent => ({
              agent: agent.agent_id,
              scenario: scenario.scenario_name,
              value: agent.metrics.find(m => m.name === metric)?.value || 0
            }))
          );
        } else {
          // AgentRunResult
          return [{
            agent: item.agent_id,
            scenario: item.scenario_name,
            value: item.metrics.find(m => m.name === metric)?.value || 0
          }];
        }
      });
    }

    // Get unique agents and scenarios
    const agents = Array.from(new Set(processedData.map(d => d.agent)));
    const scenarios = Array.from(new Set(processedData.map(d => d.scenario)));

    // Calculate min and max values
    const values = processedData.map(d => d.value);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);

    // Create color scale function
    const colorScaleFunction = (value: number): string => {
      const normalized = (value - minValue) / (maxValue - minValue || 1);
      
      switch (colorScale) {
        case 'sequential':
          return `rgb(${Math.round(59 + (245 - 59) * normalized)}, ${Math.round(130 + (101 - 130) * normalized)}, ${Math.round(246 + (101 - 246) * normalized)})`;
        
        case 'diverging':
          if (normalized < 0.5) {
            // Blue to white
            const t = normalized * 2;
            return `rgb(${Math.round(59 + (255 - 59) * t)}, ${Math.round(130 + (255 - 130) * t)}, ${Math.round(246 + (255 - 246) * t)})`;
          } else {
            // White to red
            const t = (normalized - 0.5) * 2;
            return `rgb(${Math.round(255 - (255 - 245) * t)}, ${Math.round(255 - (255 - 101) * t)}, ${Math.round(255 - (255 - 101) * t)})`;
          }
        
        case 'categorical':
          const colors = [
            'rgb(59, 130, 246)',   // Blue
            'rgb(16, 185, 129)',   // Green
            'rgb(245, 101, 101)',  // Red
            'rgb(251, 191, 36)',   // Yellow
            'rgb(139, 92, 246)',   // Purple
            'rgb(236, 72, 153)'    // Pink
          ];
          return colors[agents.indexOf(processedData.find(d => d.value === value)?.agent || '') % colors.length];
        
        default:
          return '#e5e7eb';
      }
    };

    // Create cells for the heatmap
    const cells: HeatmapCell[] = [];
    const cellWidth = 100 / scenarios.length;
    const cellHeight = 100 / agents.length;

    scenarios.forEach((scenario, scenarioIndex) => {
      agents.forEach((agent, agentIndex) => {
        const dataPoint = processedData.find(d => d.agent === agent && d.scenario === scenario);
        const value = dataPoint ? dataPoint.value : 0;
        
        cells.push({
          agent,
          scenario,
          value,
          x: scenarioIndex * cellWidth,
          y: agentIndex * cellHeight,
          width: cellWidth,
          height: cellHeight,
          color: colorScaleFunction(value)
        });
      });
    });

    return {
      cells,
      minValue,
      maxValue,
      colorScaleFunction,
      agents,
      scenarios
    };
  }, [data, colorScale, metric]);

  const handleCellClick = (cell: HeatmapCell) => {
    if (interactive && onCellClick) {
      onCellClick(cell.agent, cell.scenario, metric, cell.value);
    }
  };

  if (!data || data.length === 0) {
    return (
      <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}>
        <h3 className="text-lg font-medium text-gray-900 mb-4">{title || 'Heatmap Chart'}</h3>
        <div className="flex items-center justify-center h-64 text-gray-500">
          No data available for visualization
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}>
      {title && (
        <h3 className="text-lg font-medium text-gray-900 mb-4">{title}</h3>
      )}
      
      <div className="overflow-x-auto">
        <div className="min-w-full">
          {/* Header row - Scenario names */}
          <div className="flex">
            <div className="w-32 flex-shrink-0"></div>
            {scenarios.map((scenario, index) => (
              <div
                key={index}
                className="flex-1 text-xs text-gray-600 font-medium text-center py-2 border-b border-gray-200"
                style={{ minWidth: '80px' }}
              >
                {scenario}
              </div>
            ))}
          </div>

          {/* Data rows - Agent names and cells */}
          {agents.map((agent, agentIndex) => (
            <div key={agentIndex} className="flex">
              {/* Agent name */}
              <div className="w-32 flex-shrink-0 flex items-center justify-end pr-2">
                <span className="text-sm text-gray-700 font-medium truncate max-w-[120px]">
                  {agent}
                </span>
              </div>
              
              {/* Heatmap cells */}
              {scenarios.map((scenario, scenarioIndex) => {
                const cell = cells.find(c => c.agent === agent && c.scenario === scenario);
                if (!cell) return null;
                
                const isSelected = selectedAgent === agent && selectedScenario === scenario;
                
                return (
                  <div
                    key={`${agentIndex}-${scenarioIndex}`}
                    className="relative flex-1 border border-gray-200"
                    style={{ minWidth: '80px' }}
                  >
                    <div
                      className={`cursor-pointer transition-all duration-200 hover:opacity-80 ${
                        isSelected ? 'ring-2 ring-blue-500 ring-offset-1' : ''
                      }`}
                      style={{
                        backgroundColor: cell.color,
                        height: '100%',
                        width: '100%'
                      }}
                      onClick={() => handleCellClick(cell)}
                    >
                      {/* Value label */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-xs font-medium text-white drop-shadow">
                          {cell.value.toFixed(1)}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      {showLegend && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Low</span>
            <span className="text-sm text-gray-600">High</span>
          </div>
          <div className="h-4 rounded" style={{
            background: `linear-gradient(to right, ${colorScaleFunction(minValue)}, ${colorScaleFunction(maxValue)})`
          }} />
          <div className="flex justify-between mt-1">
            <span className="text-xs text-gray-500">{minValue.toFixed(1)}</span>
            <span className="text-xs text-gray-500">{maxValue.toFixed(1)}</span>
          </div>
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
            {cells.find(c => c.agent === selectedAgent && c.scenario === selectedScenario) && (
              <div>
                <span className="font-medium">Value:</span> {
                  cells.find(c => c.agent === selectedAgent && c.scenario === selectedScenario)?.value.toFixed(2)
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
              if (cells.length > 0) {
                const firstCell = cells[0];
                handleCellClick(firstCell);
              }
            }}
            className="text-sm text-blue-600 hover:text-blue-800 transition-colors"
          >
            Click on cells to view detailed information
          </button>
        </div>
      )}
    </div>
  );
};

export default HeatmapChart;