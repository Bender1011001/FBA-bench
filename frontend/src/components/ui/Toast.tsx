import React from 'react';
import useToast from '../../contexts/useToast';
import type { Toast as ToastItem } from '../../contexts/toast-core';

const Toast: React.FC = () => {
  const { toasts, removeToast } = useToast();

  if (!toasts.length) return null;

  return (
    <div
      aria-live="polite"
      aria-atomic="true"
      className="fixed bottom-4 right-4 z-50 flex flex-col items-end gap-2"
    >
      {toasts.map((t: ToastItem) => (
        <div
          key={t.id}
          role="status"
          className={[
            'w-[320px] max-w-[90vw] rounded-lg shadow-lg border p-3 text-sm bg-white',
            t.type === 'success' ? 'border-green-200' : '',
            t.type === 'error' ? 'border-red-200' : '',
            t.type === 'warning' ? 'border-yellow-200' : '',
            t.type === 'info' ? 'border-blue-200' : '',
          ].join(' ')}
        >
          <div className="flex items-start gap-2">
            <span aria-hidden="true" className="mt-0.5">
              {t.type === 'success' ? '✅' : t.type === 'error' ? '❌' : t.type === 'warning' ? '⚠️' : 'ℹ️'}
            </span>
            <div className="flex-1">
              <div className="text-gray-900">{t.message}</div>
              {t.action && (
                <button
                  type="button"
                  className="mt-2 inline-flex items-center gap-1 rounded border px-2 py-1 text-xs hover:bg-gray-50"
                  onClick={t.action.onClick}
                >
                  {t.action.label}
                </button>
              )}
            </div>
            <button
              type="button"
              aria-label="Close notification"
              className="rounded p-1 text-gray-500 hover:bg-gray-100"
              onClick={() => removeToast(t.id)}
            >
              ×
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};

export default Toast;