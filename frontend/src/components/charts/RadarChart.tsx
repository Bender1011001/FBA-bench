import React, { useMemo } from 'react';
import type { MultiDimensionalMetric, CapabilityAssessment } from '../../types';

interface RadarChartProps {
  data: MultiDimensionalMetric[] | CapabilityAssessment[];
  title?: string;
  className?: string;
  showLegend?: boolean;
  showGrid?: boolean;
  interactive?: boolean;
  onDimensionClick?: (dimension: string) => void;
  selectedAgent?: string;
  selectedScenario?: string;
}

interface Point {
  x: number;
  y: number;
}

interface Polygon {
  points: Point[];
  fill: string;
  stroke: string;
  opacity: number;
}

const RadarChart: React.FC<RadarChartProps> = ({
  data,
  title,
  className = '',
  showLegend = true,
  showGrid = true,
  interactive = true,
  onDimensionClick,
  selectedAgent,
  selectedScenario
}) => {
  const { dimensions, polygons, legendItems, maxRadius } = useMemo(() => {
    if (!data || data.length === 0) {
      return {
        dimensions: [],
        polygons: [],
        legendItems: [],
        maxRadius: 100
      };
    }

    // Extract dimensions from the first data point
    const firstData = data[0];
    let dimensions: string[] = [];
    
    if ('dimensions' in firstData) {
      dimensions = firstData.dimensions;
    } else if ('capabilities' in firstData) {
      dimensions = Object.keys(firstData.capabilities);
    }
    
    // Calculate max radius based on data values
    const maxValues = dimensions.map((_, index) => {
      return Math.max(...data.map(item => {
        if ('capabilities' in item) {
          return Object.values(item.capabilities)[index] || 0;
        } else {
          return item.values[index] || 0;
        }
      }));
    });
    
    const maxRadius = Math.max(...maxValues, 100);

    // Generate polygons for each data series
    const polygons: Polygon[] = data.map((item, index) => {
      const points: Point[] = dimensions.map((dimension, dimIndex) => {
        let value: number;
        
        if ('capabilities' in item) {
          // Handle CapabilityAssessment data
          const capabilityValues = Object.values(item.capabilities);
          value = capabilityValues[dimIndex] || 0;
        } else {
          // Handle MultiDimensionalMetric data
          value = item.values[dimIndex] || 0;
        }

        // Convert to polar coordinates
        const angle = (dimIndex * 2 * Math.PI) / dimensions.length - Math.PI / 2;
        const radius = (value / maxRadius) * 90; // Scale to fit chart
        
        return {
          x: 50 + radius * Math.cos(angle),
          y: 50 + radius * Math.sin(angle)
        };
      });

      // Generate colors based on index
      const colors = [
        'rgba(59, 130, 246, 0.6)',   // Blue
        'rgba(16, 185, 129, 0.6)',   // Green
        'rgba(245, 101, 101, 0.6)',  // Red
        'rgba(251, 191, 36, 0.6)',   // Yellow
        'rgba(139, 92, 246, 0.6)',   // Purple
        'rgba(236, 72, 153, 0.6)'    // Pink
      ];
      
      const color = colors[index % colors.length];
      
      return {
        points,
        fill: color,
        stroke: color.replace('0.6', '1'),
        opacity: 0.6
      };
    });

    // Generate legend items
    const legendItems = data.map((item, index) => {
      let label: string;
      
      if ('capabilities' in item) {
        label = `${item.agent_id} - ${item.scenario_name}`;
      } else {
        label = item.name || `Series ${index + 1}`;
      }

      const colors = [
        'rgb(59, 130, 246)',
        'rgb(16, 185, 129)',
        'rgb(245, 101, 101)',
        'rgb(251, 191, 36)',
        'rgb(139, 92, 246)',
        'rgb(236, 72, 153)'
      ];
      
      return {
        label,
        color: colors[index % colors.length]
      };
    });

    return {
      dimensions,
      polygons,
      legendItems,
      maxRadius
    };
  }, [data]);

  const handleDimensionClick = (dimension: string) => {
    if (interactive && onDimensionClick) {
      onDimensionClick(dimension);
    }
  };

  if (!data || data.length === 0) {
    return (
      <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}>
        <h3 className="text-lg font-medium text-gray-900 mb-4">{title || 'Radar Chart'}</h3>
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
      
      <div className="relative">
        <svg
          viewBox="0 0 100 100"
          className="w-full h-96"
          style={{ maxWidth: '100%', height: 'auto' }}
        >
          {/* Grid circles */}
          {showGrid && (
            <>
              {[20, 40, 60, 80].map((radius) => (
                <circle
                  key={radius}
                  cx="50"
                  cy="50"
                  r={radius}
                  fill="none"
                  stroke="#e5e7eb"
                  strokeWidth="0.5"
                />
              ))}
            </>
          )}

          {/* Grid lines */}
          {showGrid && dimensions.map((_, index) => {
            const angle = (index * 2 * Math.PI) / dimensions.length - Math.PI / 2;
            const x2 = 50 + 90 * Math.cos(angle);
            const y2 = 50 + 90 * Math.sin(angle);
            
            return (
              <line
                key={index}
                x1="50"
                y1="50"
                x2={x2}
                y2={y2}
                stroke="#e5e7eb"
                strokeWidth="0.5"
              />
            );
          })}

          {/* Data polygons */}
          {polygons.map((polygon, index) => (
            <polygon
              key={index}
              points={polygon.points.map(p => `${p.x},${p.y}`).join(' ')}
              fill={polygon.fill}
              stroke={polygon.stroke}
              strokeWidth="1"
              opacity={polygon.opacity}
              className="transition-all duration-200 hover:opacity-80 cursor-pointer"
              style={{
                filter: interactive ? 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))' : 'none'
              }}
            />
          ))}

          {/* Data points */}
          {polygons.map((polygon, polygonIndex) =>
            polygon.points.map((point, pointIndex) => (
              <circle
                key={`${polygonIndex}-${pointIndex}`}
                cx={point.x}
                cy={point.y}
                r="2"
                fill="#1f2937"
                className="cursor-pointer"
                onClick={() => interactive && handleDimensionClick(dimensions[pointIndex])}
              />
            ))
          )}

          {/* Dimension labels */}
          {dimensions.map((dimension, index) => {
            const angle = (index * 2 * Math.PI) / dimensions.length - Math.PI / 2;
            const labelRadius = 95;
            const x = 50 + labelRadius * Math.cos(angle);
            const y = 50 + labelRadius * Math.sin(angle);
            
            return (
              <g key={index}>
                <text
                  x={x}
                  y={y}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="text-xs fill-gray-600 cursor-pointer select-none"
                  onClick={() => interactive && handleDimensionClick(dimension)}
                  style={{ pointerEvents: interactive ? 'auto' : 'none' }}
                >
                  {dimension}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Value labels on axes */}
        {showGrid && (
          <div className="absolute inset-0 pointer-events-none">
            {[20, 40, 60, 80].map((value, index) => (
              <div
                key={index}
                className="absolute text-xs text-gray-400"
                style={{
                  left: '50%',
                  top: `${50 - (value / 100) * 45}%`,
                  transform: 'translateX(-50%)'
                }}
              >
                {Math.round((value / 80) * maxRadius)}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Legend */}
      {showLegend && (
        <div className="mt-4 flex flex-wrap gap-4 justify-center">
          {legendItems.map((item, index) => (
            <div key={index} className="flex items-center space-x-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: item.color }}
              />
              <span className="text-sm text-gray-600">{item.label}</span>
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
          </div>
        </div>
      )}

      {/* Interactive tooltip (placeholder) */}
      {interactive && (
        <div className="mt-4 text-center">
          <button
            onClick={() => {
              if (dimensions.length > 0) {
                handleDimensionClick(dimensions[0]);
              }
            }}
            className="text-sm text-blue-600 hover:text-blue-800 transition-colors"
          >
            Click on dimensions to drill down
          </button>
        </div>
      )}
    </div>
  );
};

export default RadarChart;