import React, { useCallback, useMemo, useState } from 'react';
import { ToastContext } from './toast-core';
import type { ToastContextValue, Toast } from './toast-core';

function genId(): string {
  // Lightweight, good-enough id
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

/**
 * Component-only export to satisfy React Fast Refresh rules.
 * Context and hook live in separate files.
 */
const ToastProvider: React.FC<React.PropsWithChildren> = ({ children }) => {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((ts) => ts.filter((t) => t.id !== id));
  }, []);

  const addToast = useCallback<ToastContextValue['addToast']>((toastInput) => {
    const id = toastInput.id ?? genId();
    const toast: Toast = {
      id,
      type: toastInput.type,
      message: toastInput.message,
      createdAt: Date.now(),
      durationMs: toastInput.durationMs ?? 4000,
      action: toastInput.action,
    };
    setToasts((ts) => [toast, ...ts].slice(0, 50));
    if (toast.durationMs && toast.durationMs > 0) {
      window.setTimeout(() => removeToast(id), toast.durationMs);
    }
    return id;
  }, [removeToast]);

  const api = useMemo<ToastContextValue>(() => ({
    toasts,
    addToast,
    removeToast,
    clearToasts: () => setToasts([]),
    success: (message, durationMs) => addToast({ type: 'success', message, durationMs }),
    error: (message, durationMs) => addToast({ type: 'error', message, durationMs }),
    info: (message, durationMs) => addToast({ type: 'info', message, durationMs }),
    warning: (message, durationMs) => addToast({ type: 'warning', message, durationMs }),
  }), [toasts, addToast, removeToast]);

  return <ToastContext.Provider value={api}>{children}</ToastContext.Provider>;
};

export default ToastProvider;