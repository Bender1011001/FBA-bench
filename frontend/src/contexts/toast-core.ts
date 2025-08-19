import { createContext } from 'react';

export type ToastType = 'info' | 'success' | 'error' | 'warning';

export type Toast = {
  id: string;
  type: ToastType;
  message: string;
  durationMs?: number;
  createdAt: number;
  action?: { label: string; onClick: () => void };
};

export type ToastContextValue = {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id' | 'createdAt'> & { id?: string }) => string;
  removeToast: (id: string) => void;
  clearToasts: () => void;
  // Helpers
  success: (message: string, durationMs?: number) => string;
  error: (message: string, durationMs?: number) => string;
  info: (message: string, durationMs?: number) => string;
  warning: (message: string, durationMs?: number) => string;
};

// Context only (no hooks/components here to satisfy Fast Refresh rule)
export const ToastContext = createContext<ToastContextValue | null>(null);