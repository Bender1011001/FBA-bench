// frontend/src/components/LoadingSpinner.tsx

import React from 'react';

interface LoadingSpinnerProps {
  size?: 'small' | 'medium' | 'large';
  message?: string;
  className?: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ size = 'medium', message, className }) => {
  let spinnerSizeClasses = '';
  let textSizeClasses = 'text-sm';

  switch (size) {
    case 'small':
      spinnerSizeClasses = 'w-5 h-5 border-2';
      textSizeClasses = 'text-xs';
      break;
    case 'large':
      spinnerSizeClasses = 'w-12 h-12 border-4';
      textSizeClasses = 'text-lg';
      break;
    case 'medium': // default
    default:
      spinnerSizeClasses = 'w-8 h-8 border-3';
      textSizeClasses = 'text-sm';
      break;
  }

  return (
    <div className={`flex flex-col items-center justify-center ${className}`}>
      <div
        className={`inline-block animate-spin rounded-full border-solid border-current border-r-transparent text-blue-500 patience ${spinnerSizeClasses}`}
        role="status"
      >
        <span className="!absolute !-m-px !h-px !w-px !overflow-hidden !whitespace-nowrap !border-0 !p-0 ![clip:rect(0,0,0,0)]">Loading...</span>
      </div>
      {message && <p className={`mt-2 text-gray-600 ${textSizeClasses}`}>{message}</p>}
    </div>
  );
};

export default LoadingSpinner;