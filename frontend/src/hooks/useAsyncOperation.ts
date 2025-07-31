// frontend/src/hooks/useAsyncOperation.ts

import { useState, useCallback, useRef } from 'react';
import { handleError } from '../utils/errorHandler';
import type { AppError } from '../utils/errorHandler';

interface AsyncOperationOptions<T> {
  onSuccess?: (data: T) => void;
  onError?: (error: AppError) => void;
  // TODO: Add options for retry attempts, retry delay, etc. if not handled by apiService level
}

interface AsyncOperationResult<T, Args extends unknown[]> {
  loading: boolean;
  error: AppError | null;
  data: T | null;
  execute: (...args: Args) => Promise<T | null>;
  reset: () => void;
  // TODO: Add other states like `idle`, `success`, `retrying` etc. for more granular UI feedback
}

function useAsyncOperation<T, Args extends unknown[] = []>(
  operation: (...args: Args) => Promise<T>,
  options?: AsyncOperationOptions<T>
): AsyncOperationResult<T, Args> {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<AppError | null>(null);
  const [data, setData] = useState<T | null>(null);

  // Using a ref to prevent stale closures if options change, or to manage ongoing operations
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const execute = useCallback(async (...args: Args): Promise<T | null> => {
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const result = await operation(...args);
      setData(result);
      optionsRef.current?.onSuccess?.(result);
      return result;
    } catch (err: unknown) {
      // Use the centralized error handler to process and log the error
      const processedError = handleError(err);
      setError(processedError);
      
      optionsRef.current?.onError?.(processedError);
      return null;
    } finally {
      setLoading(false);
    }
  }, [operation]);

  const reset = useCallback(() => {
    setLoading(false);
    setError(null);
    setData(null);
  }, []);

  return { loading, error, data, execute, reset };
}

export default useAsyncOperation;