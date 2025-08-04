// frontend/src/components/SpecializedErrorBoundaries.tsx

import React from 'react';
import { ErrorBoundary as ReactErrorBoundary } from 'react-error-boundary';
import { handleError, ErrorCategory } from '../utils/errorHandler';
import type { AppError } from '../utils/errorHandler';
import ErrorFallback from './ErrorFallback';

interface Props {
  children: React.ReactNode;
  fallback?: React.ComponentType<{ error: Error; resetErrorBoundary: () => void }>;
}

// Error boundary specifically for WebSocket-related components
export const WebSocketErrorBoundary: React.FC<Props> = ({ children, fallback }) => {
  const logErrorToHandler = (error: Error, info: React.ErrorInfo) => {
    const appError: AppError = {
      name: error.name,
      message: error.message,
      category: ErrorCategory.Network,
      isHandled: false,
      details: {
        componentStack: info.componentStack || 'No component stack available',
        originalError: error,
        errorType: 'WebSocket',
      },
      stack: error.stack,
      userMessage: 'A connection error occurred. Please check your network connection and try again.',
    };
    handleError(appError);
  };

  return (
    <ReactErrorBoundary
      FallbackComponent={fallback || ErrorFallback}
      onError={logErrorToHandler}
    >
      {children}
    </ReactErrorBoundary>
  );
};

// Error boundary specifically for data visualization components
export const DashboardErrorBoundary: React.FC<Props> = ({ children, fallback }) => {
  const logErrorToHandler = (error: Error, info: React.ErrorInfo) => {
    const appError: AppError = {
      name: error.name,
      message: error.message,
      category: ErrorCategory.System,
      isHandled: false,
      details: {
        componentStack: info.componentStack || 'No component stack available',
        originalError: error,
        errorType: 'Dashboard',
      },
      stack: error.stack,
      userMessage: 'An error occurred while rendering the dashboard. Please refresh the page.',
    };
    handleError(appError);
  };

  return (
    <ReactErrorBoundary
      FallbackComponent={fallback || ErrorFallback}
      onError={logErrorToHandler}
    >
      {children}
    </ReactErrorBoundary>
  );
};

// Error boundary specifically for form components
export const FormErrorBoundary: React.FC<Props> = ({ children, fallback }) => {
  const logErrorToHandler = (error: Error, info: React.ErrorInfo) => {
    const appError: AppError = {
      name: error.name,
      message: error.message,
      category: ErrorCategory.Validation,
      isHandled: false,
      details: {
        componentStack: info.componentStack || 'No component stack available',
        originalError: error,
        errorType: 'Form',
      },
      stack: error.stack,
      userMessage: 'An error occurred in the form. Please check your input and try again.',
    };
    handleError(appError);
  };

  return (
    <ReactErrorBoundary
      FallbackComponent={fallback || ErrorFallback}
      onError={logErrorToHandler}
    >
      {children}
    </ReactErrorBoundary>
  );
};

// Error boundary specifically for simulation control components
export const SimulationErrorBoundary: React.FC<Props> = ({ children, fallback }) => {
  const logErrorToHandler = (error: Error, info: React.ErrorInfo) => {
    const appError: AppError = {
      name: error.name,
      message: error.message,
      category: ErrorCategory.System,
      isHandled: false,
      details: {
        componentStack: info.componentStack || 'No component stack available',
        originalError: error,
        errorType: 'Simulation',
      },
      stack: error.stack,
      userMessage: 'A simulation error occurred. Please reset the simulation and try again.',
    };
    handleError(appError);
  };

  return (
    <ReactErrorBoundary
      FallbackComponent={fallback || ErrorFallback}
      onError={logErrorToHandler}
    >
      {children}
    </ReactErrorBoundary>
  );
};

// Error boundary specifically for event log components
export const EventLogErrorBoundary: React.FC<Props> = ({ children, fallback }) => {
  const logErrorToHandler = (error: Error, info: React.ErrorInfo) => {
    const appError: AppError = {
      name: error.name,
      message: error.message,
      category: ErrorCategory.System,
      isHandled: false,
      details: {
        componentStack: info.componentStack || 'No component stack available',
        originalError: error,
        errorType: 'EventLog',
      },
      stack: error.stack,
      userMessage: 'An error occurred in the event log. Please refresh the page.',
    };
    handleError(appError);
  };

  return (
    <ReactErrorBoundary
      FallbackComponent={fallback || ErrorFallback}
      onError={logErrorToHandler}
    >
      {children}
    </ReactErrorBoundary>
  );
};

// Error boundary specifically for agent monitoring components
export const AgentMonitorErrorBoundary: React.FC<Props> = ({ children, fallback }) => {
  const logErrorToHandler = (error: Error, info: React.ErrorInfo) => {
    const appError: AppError = {
      name: error.name,
      message: error.message,
      category: ErrorCategory.System,
      isHandled: false,
      details: {
        componentStack: info.componentStack || 'No component stack available',
        originalError: error,
        errorType: 'AgentMonitor',
      },
      stack: error.stack,
      userMessage: 'An error occurred in the agent monitor. Please refresh the page.',
    };
    handleError(appError);
  };

  return (
    <ReactErrorBoundary
      FallbackComponent={fallback || ErrorFallback}
      onError={logErrorToHandler}
    >
      {children}
    </ReactErrorBoundary>
  );
};

// Error boundary specifically for system health components
export const SystemHealthErrorBoundary: React.FC<Props> = ({ children, fallback }) => {
  const logErrorToHandler = (error: Error, info: React.ErrorInfo) => {
    const appError: AppError = {
      name: error.name,
      message: error.message,
      category: ErrorCategory.System,
      isHandled: false,
      details: {
        componentStack: info.componentStack || 'No component stack available',
        originalError: error,
        errorType: 'SystemHealth',
      },
      stack: error.stack,
      userMessage: 'An error occurred in the system health monitor. Please refresh the page.',
    };
    handleError(appError);
  };

  return (
    <ReactErrorBoundary
      FallbackComponent={fallback || ErrorFallback}
      onError={logErrorToHandler}
    >
      {children}
    </ReactErrorBoundary>
  );
};

// Error boundary specifically for configuration components
export const ConfigurationErrorBoundary: React.FC<Props> = ({ children, fallback }) => {
  const logErrorToHandler = (error: Error, info: React.ErrorInfo) => {
    const appError: AppError = {
      name: error.name,
      message: error.message,
      category: ErrorCategory.Validation,
      isHandled: false,
      details: {
        componentStack: info.componentStack || 'No component stack available',
        originalError: error,
        errorType: 'Configuration',
      },
      stack: error.stack,
      userMessage: 'A configuration error occurred. Please check your settings and try again.',
    };
    handleError(appError);
  };

  return (
    <ReactErrorBoundary
      FallbackComponent={fallback || ErrorFallback}
      onError={logErrorToHandler}
    >
      {children}
    </ReactErrorBoundary>
  );
};

// Error boundary specifically for experiment management components
export const ExperimentErrorBoundary: React.FC<Props> = ({ children, fallback }) => {
  const logErrorToHandler = (error: Error, info: React.ErrorInfo) => {
    const appError: AppError = {
      name: error.name,
      message: error.message,
      category: ErrorCategory.System,
      isHandled: false,
      details: {
        componentStack: info.componentStack || 'No component stack available',
        originalError: error,
        errorType: 'Experiment',
      },
      stack: error.stack,
      userMessage: 'An experiment management error occurred. Please refresh the page.',
    };
    handleError(appError);
  };

  return (
    <ReactErrorBoundary
      FallbackComponent={fallback || ErrorFallback}
      onError={logErrorToHandler}
    >
      {children}
    </ReactErrorBoundary>
  );
};

// Error boundary specifically for results analysis components
export const ResultsAnalysisErrorBoundary: React.FC<Props> = ({ children, fallback }) => {
  const logErrorToHandler = (error: Error, info: React.ErrorInfo) => {
    const appError: AppError = {
      name: error.name,
      message: error.message,
      category: ErrorCategory.System,
      isHandled: false,
      details: {
        componentStack: info.componentStack || 'No component stack available',
        originalError: error,
        errorType: 'ResultsAnalysis',
      },
      stack: error.stack,
      userMessage: 'A results analysis error occurred. Please refresh the page.',
    };
    handleError(appError);
  };

  return (
    <ReactErrorBoundary
      FallbackComponent={fallback || ErrorFallback}
      onError={logErrorToHandler}
    >
      {children}
    </ReactErrorBoundary>
  );
};

// Error boundary specifically for statistics components
export const StatsErrorBoundary: React.FC<Props> = ({ children, fallback }) => {
  const logErrorToHandler = (error: Error, info: React.ErrorInfo) => {
    const appError: AppError = {
      name: error.name,
      message: error.message,
      category: ErrorCategory.System,
      isHandled: false,
      details: {
        componentStack: info.componentStack || 'No component stack available',
        originalError: error,
        errorType: 'Statistics',
      },
      stack: error.stack,
      userMessage: 'A statistics error occurred. Please refresh the page.',
    };
    handleError(appError);
  };

  return (
    <ReactErrorBoundary
      FallbackComponent={fallback || ErrorFallback}
      onError={logErrorToHandler}
    >
      {children}
    </ReactErrorBoundary>
  );
};