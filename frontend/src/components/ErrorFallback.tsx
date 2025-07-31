// frontend/src/components/ErrorFallback.tsx

import React, { useState } from 'react';
import type { AppError } from '../utils/errorHandler';

interface ErrorFallbackProps {
  error: AppError;
  resetErrorBoundary: (...args: Array<unknown>) => void;
}

const ErrorFallback: React.FC<ErrorFallbackProps> = ({ error, resetErrorBoundary }) => {
  const [showDetails, setShowDetails] = useState(false);

  const handleReportError = () => {
    // In a real application, this would send the error to a backend service.
    console.log('Reporting error:', JSON.stringify(error, null, 2));
    alert('Error reported. Thank you for your feedback!');
  };

  return (
    <div className="flex flex-col items-center justify-center p-8 bg-red-100 border border-red-400 text-red-700 rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4">Oops! Something went wrong.</h2>
      <p className="text-lg text-center mb-6">
        {error.userMessage || 'An unexpected error occurred. Please try again.'}
      </p>

      <div className="flex space-x-4 mb-6">
        <button
          onClick={resetErrorBoundary}
          className="px-6 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
        >
          Retry
        </button>
        <button
          onClick={() => window.location.reload()} // Force a full page reload to reset state
          className="px-6 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50"
        >
          Reset Application
        </button>
        <button
          onClick={handleReportError}
          className="px-6 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50"
        >
          Report Error
        </button>
      </div>

      <button
        onClick={() => setShowDetails(!showDetails)}
        className="text-blue-700 hover:underline mb-4 focus:outline-none"
      >
        {showDetails ? 'Hide Details' : 'Show Details'}
      </button>

      {showDetails && (
        <div className="w-full max-w-lg bg-gray-50 p-4 border border-gray-200 rounded-md overflow-auto text-sm text-gray-800">
          <h3 className="font-semibold mb-2">Error Details:</h3>
          <p className="font-mono break-words">
            <strong>Name:</strong> {error.name}
          </p>
          <p className="font-mono break-words">
            <strong>Message:</strong> {error.message}
          </p>
          <p className="font-mono break-words">
            <strong>Category:</strong> {error.category}
          </p>
          {error.statusCode && (
            <p className="font-mono break-words">
              <strong>Status Code:</strong> {error.statusCode}
            </p>
          )}
          {typeof error.details === 'object' && error.details !== null && (
            <>
              <h4 className="font-semibold mt-3 mb-1">Raw Error:</h4>
              <pre className="whitespace-pre-wrap">{JSON.stringify(error.details, null, 2)}</pre>
            </>
          )}
          {typeof error.details === 'string' && (
            <>
              <h4 className="font-semibold mt-3 mb-1">Raw Error Message:</h4>
              <pre className="whitespace-pre-wrap">{error.details}</pre>
            </>
          )}
           {error.stack && (
            <>
              <h4 className="font-semibold mt-3 mb-1">Stack Trace:</h4>
              <pre className="whitespace-pre-wrap">{error.stack}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default ErrorFallback;