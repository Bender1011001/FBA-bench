// frontend/src/components/ErrorBoundary.tsx

import React from 'react'; // Dummy edit to force TS refresh
import { ErrorBoundary as ReactErrorBoundary } from 'react-error-boundary';
import { handleError, ErrorCategory } from '../utils/errorHandler';
import type { AppError } from '../utils/errorHandler';
import ErrorFallback from './ErrorFallback';

interface Props {
  children: React.ReactNode;
}

const ErrorBoundary: React.FC<Props> = ({ children }) => {
  const logErrorToHandler = (error: Error, info: { componentStack: string }) => {
    // Convert generic Error to AppError for centralized handling
    const appError: AppError = {
      name: error.name,
      message: error.message,
      category: ErrorCategory.System, // Categorize as System error for unhandled React errors
      isHandled: false, // Mark as unhandled by the boundary itself for logging purposes
      details: {
        componentStack: info.componentStack,
        originalError: error,
      },
      stack: error.stack,
      userMessage: 'An unexpected error occurred in a UI component. Please try refreshing the page.',
    };
    handleError(appError); // Use the centralized error handler
  };

  return (
    <ReactErrorBoundary
      FallbackComponent={ErrorFallback}
      onError={logErrorToHandler}
    >
      {children}
    </ReactErrorBoundary>
  );
};

export default ErrorBoundary;