import React from 'react';
import type { DashboardMetric } from '../../types';

interface MetricWidgetProps {
  metric: DashboardMetric;
}

const MetricWidget: React.FC<MetricWidgetProps> = ({ metric }) => {
  const { label, value, formatType, trend, unit, description, color } = metric;

  const formatValue = (val: string | number, type?: DashboardMetric['formatType']): string => {
    if (typeof val === 'number') {
      switch (type) {
        case 'currency':
          return `$${val.toFixed(2)}`;
        case 'percentage':
          return `${val.toFixed(2)}%`;
        case 'time': {
          // Basic time formatting (e.g., seconds to HH:MM:SS)
          const absVal = Math.abs(val);
          const hours = Math.floor(absVal / 3600);
          const minutes = Math.floor((absVal % 3600) / 60);
          const seconds = Math.floor(absVal % 60);
          const sign = val < 0 ? '-' : '';
          return `${sign}${hours > 0 ? `${hours}h ` : ''}${minutes}m ${seconds}s`;
        }
        case 'number':
        default:
          return val.toLocaleString();
      }
    }
    return String(val);
  };

  const getTrendIcon = (trendDirection?: 'up' | 'down' | 'neutral') => {
    switch (trendDirection) {
      case 'up':
        return <span className="text-green-500 ml-1">&#9650;</span>; // Up arrow
      case 'down':
        return <span className="text-red-500 ml-1">&#9660;</span>; // Down arrow
      case 'neutral':
        return <span className="text-gray-500 ml-1">&#9644;</span>; // Square/neutral
      default:
        return null;
    }
  };

  return (
    <div className={`bg-white p-4 rounded-lg shadow-sm border border-gray-200 ${color || ''} relative`}>
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium text-gray-500">{label}</h4>
        {description && (
          <span className="tooltip">
            <span className="text-gray-400 text-sm cursor-help">&#9432;</span> {/* Info icon */}
            <div className="tooltip-text bg-gray-700 text-white text-xs rounded py-1 px-2 absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block transition-opacity duration-300 opacity-0 group-hover:opacity-100 z-10 w-48">
              {description}
            </div>
          </span>
        )}
      </div>
      <div className="flex items-baseline mb-1">
        <p className="text-2xl font-bold text-gray-900">{formatValue(value, formatType)}</p>
        {unit && <span className="ml-1 text-base text-gray-500">{unit}</span>}
        {getTrendIcon(trend)}
      </div>
      {/* Gauge chart placeholder (add condition for gauge type if needed later) */}
      {/* Alert indicator placeholder (add condition for alert status if needed later) */}
    </div>
  );
};

export default MetricWidget;