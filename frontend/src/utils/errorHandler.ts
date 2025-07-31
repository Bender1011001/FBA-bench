// frontend/src/utils/errorHandler.ts

export const ErrorCategory = {
  Network: 'Network',
  Validation: 'Validation',
  System: 'System',
  User: 'User',
  Unknown: 'Unknown',
} as const;

export type ErrorCategory = typeof ErrorCategory[keyof typeof ErrorCategory];

export interface AppError extends Error {
  statusCode?: number;
  category: ErrorCategory;
  isHandled: boolean;
  details?: unknown;
  userMessage?: string;
}

// Function to determine user-friendly message based on error
const getUserFriendlyMessage = (error: AppError): string => {
  if (error.userMessage) {
    return error.userMessage;
  }
  switch (error.category) {
    case ErrorCategory.Network:
      return 'A network error occurred. Please check your internet connection and try again.';
    case ErrorCategory.Validation:
      return 'There was an issue with your input. Please review the highlighted fields.';
    case ErrorCategory.System:
      return 'An unexpected system error occurred. We are working to fix it. Please try again later.';
    case ErrorCategory.User:
      return 'Please check your input and try again.';
    default:
      return 'An unknown error occurred. Please try again or contact support if the issue persists.';
  }
};

// Centralized error logging and reporting
export const logError = (error: AppError) => {
  console.error(`[${error.category} Error]: ${error.message}`, error.details || error);
  // In a real application, you would integrate with a centralized logging service like Sentry, LogRocket, etc.
  // For example:
  // Sentry.captureException(error);
};

// Error recovery strategies
export const handleRecovery = (error: AppError): boolean => {
  // Implement recovery logic based on error category
  switch (error.category) {
    case ErrorCategory.Network:
      // Potentially trigger a retry mechanism if appropriate
      console.log('Attempting network error recovery...');
      return true; // Indicates that recovery was attempted
    case ErrorCategory.Validation:
      // Front-end validation errors typically don't require automatic recovery, but guide user.
      console.log('Guiding user for validation error correction.');
      return false;
    case ErrorCategory.System:
      // For system errors, suggest refresh or report
      console.log('Suggesting system error recovery: refresh or report.');
      return false;
    case ErrorCategory.User:
      // User-related errors often require input correction
      console.log('Prompting user for input correction.');
      return false;
    default:
      return false;
  }
};

// Main error handling function
// Type guard to check if an unknown error is an AppError
const isAppError = (error: unknown): error is AppError => {
  return typeof error === 'object' && error !== null && 'category' in error && 'message' in error;
};

// Main error handling function
export const handleError = (error: unknown): AppError => {
  let appError: AppError;

  if (error instanceof Error) {
    // Check if it's already an AppError (e.g., re-thrown or pre-categorized)
    if (isAppError(error)) {
      appError = { ...error, isHandled: true };
    } else if (error.name === 'TypeError' || error.name === 'ReferenceError') {
      // Attempt to categorize and enrich common JS Errors
      appError = {
        ...error,
        category: ErrorCategory.System,
        isHandled: true,
        details: error,
      };
    } else {
      // Default for other generic Errors
      appError = {
        ...error,
        category: ErrorCategory.Unknown,
        isHandled: true,
        details: error,
      };
    }
  } else if (isAppError(error)) {
    // If it's already an AppError or a similar structured object
    appError = {
      ...error,
      isHandled: true,
    };
  } else if (typeof error === 'string') {
    // Handle string errors
    appError = {
      name: 'AppError',
      message: error,
      category: ErrorCategory.Unknown,
      isHandled: true,
      details: error,
    };
  } else if (typeof error === 'object' && error !== null) {
      // Handle generic objects that aren't necessarily Errors or AppErrors
      const message = (error as { message?: string }).message || 'An unknown object error occurred.';
      appError = {
          name: (error as { name?: string }).name || 'AppError',
          message: message,
          category: (error as { category?: ErrorCategory }).category || ErrorCategory.Unknown,
          isHandled: true,
          statusCode: (error as { statusCode?: number }).statusCode,
          details: error,
          userMessage: (error as { userMessage?: string }).userMessage,
      };
  } else {
    // Catch-all for truly unknown error types (e.g., number, boolean, null, undefined)
    appError = {
      name: 'AppError',
      message: 'An unknown error type was caught.',
      category: ErrorCategory.Unknown,
      isHandled: true,
      details: error,
    };
  }

  appError.userMessage = appError.userMessage || getUserFriendlyMessage(appError);
  logError(appError);
  // Do not attempt recovery immediately here since it's already handled within `handleRecovery`
  // And avoid calling `handleRecovery` twice if it was already called by the original error source

  // Here you would integrate with a notification system (e.g., Toast, Modal)
  // For now, we'll just log and return the error
  console.log(`Displaying user message: ${appError.userMessage}`);

  return appError;
};

// Example usage (for testing)
// const testNetworkError = new Error('Failed to fetch data');
// (testNetworkError as AppError).category = ErrorCategory.Network;
// (testNetworkError as AppError).statusCode = 503;
// handleError(testNetworkError);

// handleError('Something went wrong with a string error');