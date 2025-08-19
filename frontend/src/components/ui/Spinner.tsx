import React from 'react';

type SpinnerSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl';

export interface SpinnerProps {
  size?: SpinnerSize;
  className?: string;
  label?: string;
  'aria-live'?: 'assertive' | 'polite' | 'off';
}

/**
 * Accessible, theme-aware spinner.
 * - Uses SVG for minimal layout shift
 * - Includes optional aria-live region for screen readers
 */
const Spinner: React.FC<SpinnerProps> = ({
  size = 'md',
  className = '',
  label,
  'aria-live': ariaLive = 'polite',
}) => {
  const sizeMap: Record<SpinnerSize, string> = {
    xs: 'h-3 w-3',
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-10 w-10',
    xl: 'h-12 w-12',
  };

  return (
    <div className={`inline-flex items-center gap-2 ${className}`} role="status" aria-live={ariaLive}>
      <svg
        className={`animate-spin ${sizeMap[size]} text-blue-600 dark:text-blue-400`}
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        aria-hidden="true"
        focusable="false"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0a12 12 0 100 24v-4a8 8 0 01-8-8z"
        />
      </svg>
      {label ? <span className="text-sm text-gray-600 dark:text-gray-300">{label}</span> : null}
    </div>
  );
};

export default Spinner;