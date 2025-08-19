import { useMemo, useState } from 'react';
import { ErrorBoundary as REBErrorBoundary } from 'react-error-boundary';
import type { ReactNode, ReactElement, ErrorInfo as ReactErrorInfo } from 'react';
import type { FallbackProps } from 'react-error-boundary';

/**
 * Copy text to clipboard with a safe fallback
 */
async function copyToClipboardSafe(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.left = '-9999px';
      document.body.appendChild(textarea);
      textarea.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(textarea);
      return ok;
    } catch {
      return false;
    }
  }
}

/**
 * Default fallback UI for the error boundary.
 * - Shows a friendly message
 * - Provides Try Again (reset) and Copy Error actions
 * - Collapsible details panel with stack trace when available
 */
const DefaultFallback = ({ error, resetErrorBoundary }: FallbackProps): ReactElement => {
  const [showDetails, setShowDetails] = useState(false);
  const [copied, setCopied] = useState<'idle' | 'ok' | 'fail'>('idle');

  const errorPayload = useMemo(() => {
    const payload: Record<string, unknown> = {
      name: (error as Error)?.name ?? 'Error',
      message: (error as Error)?.message ?? String(error),
    };
    const err = error as unknown as { category?: unknown; statusCode?: unknown; details?: unknown; stack?: string };
    if (err?.category !== undefined) payload.category = err.category;
    if (err?.statusCode !== undefined) payload.statusCode = err.statusCode;
    if (err?.details !== undefined) payload.details = err.details;
    if ((error as Error)?.stack) payload.stack = (error as Error).stack;
    return payload;
  }, [error]);

  const handleCopy = async () => {
    const ok = await copyToClipboardSafe(JSON.stringify(errorPayload, null, 2));
    setCopied(ok ? 'ok' : 'fail');
    setTimeout(() => setCopied('idle'), 2000);
  };

  return (
    <div role="alert" className="min-h-[40vh] flex items-center justify-center bg-gray-50 py-10 px-4">
      <div className="w-full max-w-2xl bg-white shadow rounded-lg p-6 border border-gray-200">
        <div className="flex items-start gap-3">
          <svg className="h-6 w-6 text-red-600 mt-1 shrink-0" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M3.34 16l6.928-12a2 2 0 0 1 3.464 0L20.66 16A2 2 0 0 1 18.928 19H5.072A2 2 0 0 1 3.34 16Z" />
          </svg>
          <div className="flex-1">
            <h2 className="text-lg font-semibold text-gray-900">Something went wrong</h2>
            <p className="mt-1 text-sm text-gray-600">
              An unexpected error occurred. You can try again, or copy error details for reporting.
            </p>
            <div className="mt-3 rounded bg-red-50 text-red-800 text-sm p-3 break-words" aria-live="polite">
              {(error as Error)?.message ?? String(error)}
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={resetErrorBoundary}
                className="inline-flex items-center px-3 py-2 rounded-md bg-blue-600 text-white text-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Try again
              </button>
              <button
                type="button"
                onClick={handleCopy}
                className="inline-flex items-center px-3 py-2 rounded-md bg-gray-100 text-gray-900 text-sm hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400"
              >
                {copied === 'ok' ? 'Copied!' : copied === 'fail' ? 'Copy failed' : 'Copy error'}
              </button>
              <button
                type="button"
                aria-expanded={showDetails}
                aria-controls="error-details"
                onClick={() => setShowDetails((s) => !s)}
                className="inline-flex items-center px-3 py-2 rounded-md bg-gray-100 text-gray-900 text-sm hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400"
              >
                {showDetails ? 'Hide details' : 'Show details'}
              </button>
            </div>

            {showDetails && (
              <div id="error-details" className="mt-4">
                <details open className="text-sm">
                  <summary className="cursor-default font-medium text-gray-800 mb-2">Error details</summary>
                  <pre className="max-h-64 overflow-auto bg-gray-50 border border-gray-200 rounded p-3 text-gray-800 whitespace-pre-wrap">
                    {JSON.stringify(errorPayload, null, 2)}
                  </pre>
                </details>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

type BoundaryProps = {
  children: ReactNode;
  fallback?: React.ComponentType<FallbackProps>;
  onError?: (error: Error, info: ReactErrorInfo) => void;
  onReset?: () => void;
  resetKeys?: Array<unknown>;
};

/**
 * Global ErrorBoundary wrapper using react-error-boundary.
 * Wraps application routes and layout. Accepts optional custom fallback.
 */
export const ErrorBoundary = ({
  children,
  fallback: FallbackComponent,
  onError,
  onReset,
  resetKeys,
}: BoundaryProps) => {
  return (
    <REBErrorBoundary
      FallbackComponent={FallbackComponent ?? DefaultFallback}
      onError={onError}
      onReset={onReset}
      resetKeys={resetKeys}
    >
      {children}
    </REBErrorBoundary>
  );
};

export default ErrorBoundary;