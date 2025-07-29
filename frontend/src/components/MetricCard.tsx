import React from 'react';
import type { DashboardMetric } from '../types';

interface MetricCardProps {
  metric: DashboardMetric;
  className?: string;
}

const formatValue = (value: string | number, formatType?: DashboardMetric['formatType']): string => {
  if (typeof value === 'string') {
    return value;
  }

  switch (formatType) {
    case 'currency':
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
      }).format(value);
    
    case 'percentage':
      return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1,
      }).format(value / 100);
    
    case 'number':
      return new Intl.NumberFormat('en-US').format(value);
    
    default:
      return value.toString();
  }
};

const getTrendIcon = (trend?: DashboardMetric['trend']) => {
  switch (trend) {
    case 'up':
      return (
        <svg className="w-4 h-4 text-green-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M3.293 9.707a1 1 0 010-1.414l6-6a1 1 0 011.414 0l6 6a1 1 0 01-1.414 1.414L11 5.414V17a1 1 0 11-2 0V5.414L4.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
        </svg>
      );
    
    case 'down':
      return (
        <svg className="w-4 h-4 text-red-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M16.707 10.293a1 1 0 010 1.414l-6 6a1 1 0 01-1.414 0l-6-6a1 1 0 111.414-1.414L9 14.586V3a1 1 0 012 0v11.586l4.293-4.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
      );
    
    case 'neutral':
      return (
        <svg className="w-4 h-4 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
        </svg>
      );
    
    default:
      return null;
  }
};

const getTrendColorClass = (trend?: DashboardMetric['trend']): string => {
  switch (trend) {
    case 'up':
      return 'text-green-600';
    case 'down':
      return 'text-red-600';
    case 'neutral':
      return 'text-gray-600';
    default:
      return 'text-gray-900';
  }
};

export const MetricCard: React.FC<MetricCardProps> = ({ metric, className = '' }) => {
  const { label, value, trend, formatType } = metric;
  const formattedValue = formatValue(value, formatType);
  const trendIcon = getTrendIcon(trend);
  const trendColorClass = getTrendColorClass(trend);

  return (
    <div className={`
      bg-white rounded-lg shadow-sm border border-gray-200 p-6 
      hover:shadow-md transition-shadow duration-200
      ${className}
    `}>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-2">
            {label}
          </p>
          <p className={`text-2xl font-bold ${trendColorClass}`}>
            {formattedValue}
          </p>
        </div>
        
        {trendIcon && (
          <div className="ml-4 flex-shrink-0">
            <div className="flex items-center justify-center w-8 h-8 bg-gray-50 rounded-full">
              {trendIcon}
            </div>
          </div>
        )}
      </div>
      
      {trend && (
        <div className="mt-3 flex items-center text-sm">
          <span className={`font-medium ${
            trend === 'up' ? 'text-green-600' : 
            trend === 'down' ? 'text-red-600' : 
            'text-gray-600'
          }`}>
            {trend === 'up' ? 'Trending up' : 
             trend === 'down' ? 'Trending down' : 
             'No change'}
          </span>
        </div>
      )}
    </div>
  );
};

// Utility component for loading state
export const MetricCardSkeleton: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`
    bg-white rounded-lg shadow-sm border border-gray-200 p-6 
    animate-pulse ${className}
  `}>
    <div className="flex items-center justify-between">
      <div className="flex-1">
        <div className="h-4 bg-gray-200 rounded w-24 mb-2"></div>
        <div className="h-8 bg-gray-200 rounded w-32"></div>
      </div>
      <div className="ml-4 flex-shrink-0">
        <div className="w-8 h-8 bg-gray-200 rounded-full"></div>
      </div>
    </div>
    <div className="mt-3">
      <div className="h-4 bg-gray-200 rounded w-20"></div>
    </div>
  </div>
);

// Utility component for error state
export const MetricCardError: React.FC<{ 
  label: string; 
  error: string; 
  className?: string; 
}> = ({ label, error, className = '' }) => (
  <div className={`
    bg-white rounded-lg shadow-sm border border-red-200 p-6 
    ${className}
  `}>
    <div className="flex items-center justify-between">
      <div className="flex-1">
        <p className="text-sm font-medium text-gray-600 mb-2">
          {label}
        </p>
        <p className="text-2xl font-bold text-red-600">
          --
        </p>
      </div>
      <div className="ml-4 flex-shrink-0">
        <div className="flex items-center justify-center w-8 h-8 bg-red-50 rounded-full">
          <svg className="w-4 h-4 text-red-500" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
        </div>
      </div>
    </div>
    <div className="mt-3">
      <span className="text-sm text-red-600 font-medium">
        Error: {error}
      </span>
    </div>
  </div>
);

export default MetricCard;