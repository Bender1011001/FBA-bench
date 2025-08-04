import React, { useMemo } from 'react';
import type { TimeSeriesData, ExecutionProgress } from '../../types';

interface TimeSeriesChartProps {
  data: TimeSeriesData[] | ExecutionProgress[];
  title?: string;
  className?: string;
  showLegend?: boolean;
  showGrid?: boolean;
  showArea?: boolean;
  showPoints?: boolean;
  interactive?: boolean;
  onPointClick?: (point: { timestamp: string; value: number; metric: string }) => void;
  selectedMetric?: string;
  timeRange?: { start: string; end: string };
  metricsToShow?: string[];
  height?: number;
}

interface TimeSeriesPoint {
  timestamp: string;
  value: number;
  metric: string;
  x: number;
  y: number;
}

interface TimeSeriesLine {
  metric: string;
  points: TimeSeriesPoint[];
  color: string;
  strokeWidth: number;
}

const TimeSeriesChart: React.FC<TimeSeriesChartProps> = ({
  data,
  title,
  className = '',
  showLegend = true,
  showGrid = true,
  showArea = false,
  showPoints = true,
  interactive = true,
  onPointClick,
  selectedMetric,
  timeRange,
  metricsToShow,
  height = 400
}) => {
  const { lines, minX, maxX, minY, maxY, timeFormatFunction } = useMemo(() => {
    if (!data || data.length === 0) {
      return {
        lines: [],
        minX: 0,
        maxX: 100,
        minY: 0,
        maxY: 100,
        timeFormatFunction: (timestamp: string) => timestamp
      };
    }

    // Extract time series data
    let processedData: TimeSeriesPoint[] = [];
    
    if (Array.isArray(data) && data.length > 0) {
      const firstItem = data[0];
      
      if ('timestamp' in firstItem && 'value' in firstItem) {
        // Already in TimeSeriesPoint format
        processedData = data.map((item: any) => ({
          timestamp: item.timestamp,
          value: item.value,
          metric: item.metric || 'default'
        }));
      } else if ('timestamp' in firstItem) {
        // TimeSeriesData format
        processedData = data.flatMap((item: any) => {
          const points: TimeSeriesPoint[] = [];
          Object.keys(item).forEach(key => {
            if (key !== 'timestamp' && key !== 'tick') {
              points.push({
                timestamp: item.timestamp,
                value: item[key],
                metric: key
              });
            }
          });
          return points;
        });
      } else if ('cpu_usage' in firstItem) {
        // ResourceUsage format
        processedData = data.flatMap((item: any) => {
          const points: TimeSeriesPoint[] = [];
          Object.keys(item).forEach(key => {
            if (key !== 'timestamp' && typeof item[key] === 'number') {
              points.push({
                timestamp: item.timestamp,
                value: item[key],
                metric: key
              });
            }
          });
          return points;
        });
      } else if ('progress' in firstItem) {
        // ExecutionProgress format
        processedData = data.map((item: any) => ({
          timestamp: item.timestamp,
          value: item.progress,
          metric: 'progress'
        }));
      }
    }

    // Filter metrics if specified
    if (metricsToShow && metricsToShow.length > 0) {
      processedData = processedData.filter(point => 
        metricsToShow.includes(point.metric)
      );
    }

    // Group by metric
    const metricGroups = new Map<string, TimeSeriesPoint[]>();
    processedData.forEach(point => {
      if (!metricGroups.has(point.metric)) {
        metricGroups.set(point.metric, []);
      }
      metricGroups.get(point.metric)!.push(point);
    });

    // Sort points by timestamp for each metric
    metricGroups.forEach(points => {
      points.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
    });

    // Create lines with colors
    const colors = [
      'rgb(59, 130, 246)',   // Blue
      'rgb(16, 185, 129)',   // Green
      'rgb(245, 101, 101)',  // Red
      'rgb(251, 191, 36)',   // Yellow
      'rgb(139, 92, 246)',   // Purple
      'rgb(236, 72, 153)'    // Pink
    ];

    const lines: TimeSeriesLine[] = Array.from(metricGroups.entries()).map(([metric, points], index) => ({
      metric,
      points,
      color: colors[index % colors.length],
      strokeWidth: 2
    }));

    // Calculate min/max values
    const allValues = processedData.map(p => p.value);
    const allTimestamps = processedData.map(p => new Date(p.timestamp).getTime());
    
    const minX = Math.min(...allTimestamps);
    const maxX = Math.max(...allTimestamps);
    const minY = Math.min(...allValues);
    const maxY = Math.max(...allValues);

    // Create time format function
    const timeFormatFunction = (timestamp: string): string => {
      const date = new Date(timestamp);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffMins = Math.floor(diffMs / (1000 * 60));
      const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
      
      if (diffMins < 1) {
        return 'just now';
      } else if (diffMins < 60) {
        return `${diffMins}m ago`;
      } else if (diffHours < 24) {
        return `${diffHours}h ago`;
      } else {
        return date.toLocaleDateString();
      }
    };

    return {
      lines,
      minX,
      maxX,
      minY,
      maxY,
      timeFormatFunction
    };
  }, [data, metricsToShow]);

  const handlePointClick = (point: TimeSeriesPoint) => {
    if (interactive && onPointClick) {
      onPointClick(point);
    }
  };

  if (!data || data.length === 0) {
    return (
      <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}>
        <h3 className="text-lg font-medium text-gray-900 mb-4">{title || 'Time Series Chart'}</h3>
        <div className="flex items-center justify-center h-64 text-gray-500">
          No data available for visualization
        </div>
      </div>
    );
  }

  const chartWidth = 100;
  const chartHeight = 100;
  const padding = { top: 20, right: 30, bottom: 60, left: 60 };

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}>
      {title && (
        <h3 className="text-lg font-medium text-gray-900 mb-4">{title}</h3>
      )}
      
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
              
              {/* Vertical grid lines */}
              {[0.2, 0.4, 0.6, 0.8].map((ratio) => {
                const x = padding.left + (chartWidth - padding.left - padding.right) * ratio;
                return (
                  <line
                    key={`v-${ratio}`}
                    x1={x}
                    y1={padding.top}
                    x2={x}
                    y2={chartHeight - padding.bottom}
                    stroke="#e5e7eb"
                    strokeWidth="0.5"
                  />
                );
              })}
            </>
          )}

          {/* Y-axis labels */}
          {[0, 0.25, 0.5, 0.75, 1].map((ratio) => {
            const value = minY + (maxY - minY) * (1 - ratio);
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
          {lines.length > 0 && lines[0].points.length > 0 && (
            [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
              const index = Math.floor((lines[0].points.length - 1) * ratio);
              const point = lines[0].points[index];
              const x = padding.left + (chartWidth - padding.left - padding.right) * ratio;
              
              return (
                <text
                  key={`x-${ratio}`}
                  x={x}
                  y={chartHeight - padding.bottom + 20}
                  textAnchor="middle"
                  className="text-xs fill-gray-600"
                >
                  {timeFormatFunction(point.timestamp)}
                </text>
              );
            })
          )}

          {/* Data lines */}
          {lines.map((line, lineIndex) => {
            const points = line.points.map(point => {
              const x = padding.left + ((new Date(point.timestamp).getTime() - minX) / (maxX - minX || 1)) * (chartWidth - padding.left - padding.right);
              const y = padding.top + ((maxY - point.value) / (maxY - minY || 1)) * (chartHeight - padding.top - padding.bottom);
              
              return { x, y, ...point };
            });

            // Create path data
            const pathData = points.length > 0 
              ? `M ${points[0].x} ${points[0].y} ` + points.slice(1).map(p => `L ${p.x} ${p.y}`).join(' ')
              : '';

            return (
              <g key={lineIndex}>
                {/* Area fill */}
                {showArea && points.length > 0 && (
                  <path
                    d={`M ${points[0].x} ${chartHeight - padding.bottom} L ${pathData} L ${points[points.length - 1].x} ${chartHeight - padding.bottom} Z`}
                    fill={line.color.replace('rgb', 'rgba').replace(')', ', 0.2)')}
                    stroke="none"
                  />
                )}
                
                {/* Line */}
                <path
                  d={pathData}
                  fill="none"
                  stroke={line.color}
                  strokeWidth={line.strokeWidth}
                  className="transition-all duration-200"
                />
                
                {/* Points */}
                {showPoints && points.map((point, pointIndex) => (
                  <circle
                    key={`${lineIndex}-${pointIndex}`}
                    cx={point.x}
                    cy={point.y}
                    r="3"
                    fill={line.color}
                    stroke="white"
                    strokeWidth="1"
                    className={`cursor-pointer transition-all duration-200 hover:r-4 ${
                      selectedMetric === point.metric ? 'r-4' : ''
                    }`}
                    onClick={() => interactive && handlePointClick(point)}
                  />
                ))}
              </g>
            );
          })}
        </svg>

        {/* Y-axis label */}
        <div className="absolute left-2 top-1/2 transform -translate-y-1/2 -rotate-90 text-sm text-gray-600">
          Value
        </div>

        {/* X-axis label */}
        <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 text-sm text-gray-600">
          Time
        </div>
      </div>

      {/* Legend */}
      {showLegend && (
        <div className="mt-4 flex flex-wrap gap-4 justify-center">
          {lines.map((line, index) => (
            <div key={index} className="flex items-center space-x-2">
              <div
                className="w-3 h-0.5"
                style={{ backgroundColor: line.color }}
              />
              <span className="text-sm text-gray-600">{line.metric}</span>
            </div>
          ))}
        </div>
      )}

      {/* Selected metric info */}
      {selectedMetric && (
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-600">
            <span className="font-medium">Selected Metric:</span> {selectedMetric}
          </div>
        </div>
      )}

      {/* Interactive tooltip */}
      {interactive && (
        <div className="mt-4 text-center">
          <button
            onClick={() => {
              if (lines.length > 0 && lines[0].points.length > 0) {
                const lastPoint = lines[0].points[lines[0].points.length - 1];
                handlePointClick(lastPoint);
              }
            }}
            className="text-sm text-blue-600 hover:text-blue-800 transition-colors"
          >
            Click on points to view detailed information
          </button>
        </div>
      )}
    </div>
  );
};

export default TimeSeriesChart;