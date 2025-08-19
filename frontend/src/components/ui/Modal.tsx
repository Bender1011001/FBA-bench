import React, { useCallback, useEffect, useRef } from 'react';

/**
 * Accessible, focus-trapped modal dialog.
 * - role="dialog" with aria-modal
 * - ESC to close
 * - Focus trap within modal content
 * - Restores focus to the previously focused element on close
 */
export interface ModalProps {
  open: boolean;
  title?: string;
  onClose: () => void;
  children: React.ReactNode;
  footer?: React.ReactNode;
  initialFocusRef?: React.RefObject<HTMLElement>;
  'aria-label'?: string; // optional explicit label if no visible title
}

const Modal: React.FC<ModalProps> = ({
  open,
  title,
  onClose,
  children,
  footer,
  initialFocusRef,
  'aria-label': ariaLabel,
}) => {
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const lastFocusedRef = useRef<HTMLElement | null>(null);

  // Save last focused element to restore on close
  useEffect(() => {
    if (open) {
      lastFocusedRef.current = (document.activeElement as HTMLElement) ?? null;
    }
  }, [open]);

  // Focus management: focus initial or first focusable
  useEffect(() => {
    if (!open) return;

    const focusFirst = () => {
      const el = initialFocusRef?.current ?? contentRef.current;
      if (!el) return;

      // if given element is focusable, focus it; else find first focusable in content
      const isFocusable = (node: Element) => {
        const tag = node.tagName?.toLowerCase();
        const focusableTags = ['button', 'a', 'input', 'select', 'textarea'];
        const hasTabIndex = (node as HTMLElement).tabIndex >= 0;
        const disabled = (node as HTMLButtonElement).disabled;
        const ariaHidden = (node as HTMLElement).getAttribute('aria-hidden') === 'true';
        return (!ariaHidden && !disabled && (focusableTags.includes(tag) || hasTabIndex));
      };

      let target: HTMLElement | null = null;
      if (initialFocusRef?.current && isFocusable(initialFocusRef.current)) {
        target = initialFocusRef.current as HTMLElement;
      } else if (contentRef.current) {
        const list = contentRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        target = list.length ? list[0] : (contentRef.current as HTMLElement);
      }
      target?.focus();
    };

    const id = window.setTimeout(focusFirst, 0);
    return () => clearTimeout(id);
  }, [open, initialFocusRef]);

  // ESC to close
  const onKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!open) return;
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
      // Trap Tab focus
      if (e.key === 'Tab' && contentRef.current) {
        const focusable = contentRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        const list = Array.from(focusable).filter(el => !el.hasAttribute('disabled') && el.offsetParent !== null);
        if (list.length === 0) {
          e.preventDefault();
          (contentRef.current as HTMLElement).focus();
          return;
        }
        const first = list[0];
        const last = list[list.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    },
    [open, onClose]
  );

  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => onKeyDown(e);
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [open, onKeyDown]);

  // Restore focus on close
  useEffect(() => {
    if (!open && lastFocusedRef.current) {
      const id = window.setTimeout(() => lastFocusedRef.current?.focus(), 0);
      return () => clearTimeout(id);
    }
    return;
  }, [open]);

  if (!open) return null;

  const labelledBy = title ? 'modal-title' : undefined;

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      role="presentation"
      onMouseDown={(e) => {
        // click outside content closes modal
        if (e.target === overlayRef.current) onClose();
      }}
    >
      <div
        ref={contentRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelledBy}
        aria-label={ariaLabel}
        className="w-full max-w-lg rounded-lg bg-white shadow-xl outline-none ring-1 ring-black/10 focus:outline-none"
        tabIndex={-1}
      >
        <div className="px-4 py-3 border-b">
          <div className="flex items-start justify-between">
            {title ? (
              <h2 id="modal-title" className="text-lg font-semibold text-gray-900">
                {title}
              </h2>
            ) : null}
            <button
              type="button"
              aria-label="Close"
              className="ml-3 inline-flex items-center justify-center rounded p-2 text-gray-500 hover:bg-gray-100 focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              onClick={onClose}
            >
              ×
            </button>
          </div>
        </div>

        <div className="px-4 py-3">{children}</div>

        {footer ? <div className="px-4 py-3 border-t bg-gray-50">{footer}</div> : null}
      </div>
    </div>
  );
};

export default Modal;