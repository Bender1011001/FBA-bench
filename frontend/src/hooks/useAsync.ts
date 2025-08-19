import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

export type AsyncStatus = 'idle' | 'loading' | 'success' | 'error';

export interface UseAsyncOptions<TArgs extends unknown[], TData> {
  /**
   * Optional initial data to seed before first run
   */
  initialData?: TData | null;
  /**
   * Run immediately on mount with these arguments
   */
  immediateArgs?: TArgs;
  /**
   * Abort previous in-flight call when a new one starts
   */
  abortOnReRun?: boolean;
}

export interface UseAsyncResult<TArgs extends unknown[], TData> {
  status: AsyncStatus;
  data: TData | null;
  error: unknown;
  isIdle: boolean;
  isLoading: boolean;
  isSuccess: boolean;
  isError: boolean;
  run: (...args: TArgs) => Promise<TData | null>;
  reset: () => void;
  /**
   * Abort the current in-flight operation if any
   */
  abort: () => void;
}

export function useAsync<TArgs extends unknown[], TData = unknown>(
  asyncFn: (...args: TArgs) => Promise<TData>,
  options: UseAsyncOptions<TArgs, TData> = {},
): UseAsyncResult<TArgs, TData> {
  const { initialData = null, immediateArgs, abortOnReRun = true } = options;
  const [status, setStatus] = useState<AsyncStatus>('idle');
  const [data, setData] = useState<TData | null>(initialData);
  const [error, setError] = useState<unknown>(null);

  const abortRef = useRef<AbortController | null>(null);
  const mountedRef = useRef<boolean>(false);

  const isIdle = status === 'idle';
  const isLoading = status === 'loading';
  const isSuccess = status === 'success';
  const isError = status === 'error';

  const abort = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  }, []);

  const run = useCallback(
    async (...args: TArgs) => {
      if (abortOnReRun) {
        abort();
      }
      const controller = new AbortController();
      abortRef.current = controller;

      setStatus('loading');
      setError(null);

      try {
        // If the asyncFn supports an AbortSignal as last arg, pass it
        // Consumers can ignore it if not needed.
        const maybeArgs = [...args] as unknown[];
        if (typeof asyncFn === 'function') {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const result = await (asyncFn as any)(...maybeArgs, controller.signal);
          if (!mountedRef.current) return null;
          setData(result as TData);
          setStatus('success');
          return result as TData;
        }
        throw new Error('Invalid async function');
      } catch (e) {
        if ((e as DOMException)?.name === 'AbortError') {
          // Treat abort as idle again to allow retries without error state
          if (!mountedRef.current) return null;
          setStatus('idle');
          return null;
        }
        if (!mountedRef.current) return null;
        setError(e);
        setStatus('error');
        return null;
      } finally {
        abortRef.current = null;
      }
    },
    [abort, abortOnReRun, asyncFn],
  );

  const reset = useCallback(() => {
    abort();
    setStatus('idle');
    setData(initialData);
    setError(null);
  }, [abort, initialData]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      abort();
    };
  }, [abort]);

  useEffect(() => {
    if (immediateArgs) {
      // eslint-disable-next-line @typescript-eslint/no-floating-promises
      run(...immediateArgs);
    }
    // We only want to run immediately once on mount if provided
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return useMemo(
    () => ({
      status,
      data,
      error,
      isIdle,
      isLoading,
      isSuccess,
      isError,
      run,
      reset,
      abort,
    }),
    [status, data, error, isIdle, isLoading, isSuccess, isError, run, reset, abort],
  );
}

export default useAsync;