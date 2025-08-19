import { useContext } from 'react';
import { ToastContext } from './toast-core';
import type { ToastContextValue } from './toast-core';

const useToast = (): ToastContextValue => {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error('useToast must be used within ToastProvider');
  }
  return ctx;
};

export default useToast;