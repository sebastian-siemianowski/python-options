import { createContext, useContext, useState, useCallback, useRef, type ReactNode } from 'react';

/* ─── Types ─────────────────────────────────────────────────────── */
export type ToastVariant = 'info' | 'success' | 'warning' | 'error' | 'progress';

export interface Toast {
  id: string;
  variant: ToastVariant;
  title: string;
  description?: string;
  linkTo?: string;
  progress?: number; // 0-100 for determinate, -1 for indeterminate
  autoDismissMs?: number;
  createdAt: number;
}

interface ToastContextValue {
  toasts: Toast[];
  addToast: (opts: Omit<Toast, 'id' | 'createdAt'>) => string;
  updateToast: (id: string, patch: Partial<Toast>) => void;
  removeToast: (id: string) => void;
}

/* ─── Context ───────────────────────────────────────────────────── */
const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be inside ToastProvider');
  return ctx;
}

/* ─── Provider ──────────────────────────────────────────────────── */
let nextId = 1;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const timers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const removeToast = useCallback((id: string) => {
    const t = timers.current.get(id);
    if (t) { clearTimeout(t); timers.current.delete(id); }
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const addToast = useCallback((opts: Omit<Toast, 'id' | 'createdAt'>) => {
    const id = `toast-${nextId++}`;
    const toast: Toast = { ...opts, id, createdAt: Date.now() };

    setToasts(prev => {
      const next = [...prev, toast];
      // Max 4 visible
      return next.length > 4 ? next.slice(next.length - 4) : next;
    });

    // Auto-dismiss
    const dismiss = opts.autoDismissMs
      ?? (opts.variant === 'success' ? 4000 : opts.variant === 'warning' ? 6000 : 0);
    if (dismiss > 0) {
      timers.current.set(id, setTimeout(() => removeToast(id), dismiss));
    }

    return id;
  }, [removeToast]);

  const updateToast = useCallback((id: string, patch: Partial<Toast>) => {
    setToasts(prev => prev.map(t => t.id === id ? { ...t, ...patch } : t));
  }, []);

  return (
    <ToastContext.Provider value={{ toasts, addToast, updateToast, removeToast }}>
      {children}
    </ToastContext.Provider>
  );
}
