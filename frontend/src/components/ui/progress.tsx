import React from 'react';

interface ProgressProps {
  value: number;
  className?: string;
  color?: 'blue' | 'green' | 'yellow' | 'red';
}

export const Progress: React.FC<ProgressProps> = ({ 
  value, 
  className = '', 
  color = 'blue' 
}) => {
  // Ensure value is between 0 and 100
  const normalizedValue = Math.min(100, Math.max(0, value));
  
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    yellow: 'bg-yellow-500',
    red: 'bg-red-500'
  };

  return (
    <div className={`w-full bg-gray-200 rounded-full overflow-hidden ${className}`}>
      <div
        className={`h-full transition-all duration-300 ease-out ${colorClasses[color]}`}
        style={{ width: `${normalizedValue}%` }}
        aria-valuenow={normalizedValue}
        aria-valuemin={0}
        aria-valuemax={100}
        role="progressbar"
      />
    </div>
  );
};

export default Progress;