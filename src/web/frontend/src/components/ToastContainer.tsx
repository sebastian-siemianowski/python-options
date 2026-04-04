import { useNavigate } from 'react-router-dom';
import { useToast, type Toast, type ToastVariant } from '../stores/toastStore';
import {
  Info,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Loader2,
  X,
  ArrowRight,
} from 'lucide-react';
import { useEffect, useState } from 'react';

/* ─── Variant config ────────────────────────────────────────────── */
const VARIANTS: Record<ToastVariant, {
  borderGradient: string;
  icon: typeof Info;
  iconColor: string;
  glow?: string;
}> = {
  info: {
    borderGradient: 'linear-gradient(180deg, var(--accent-violet) 0%, var(--accent-indigo) 100%)',
    icon: Info,
    iconColor: 'var(--accent-violet)',
  },
  success: {
    borderGradient: 'linear-gradient(180deg, var(--accent-emerald) 0%, #059669 100%)',
    icon: CheckCircle2,
    iconColor: 'var(--accent-emerald)',
  },
  warning: {
    borderGradient: 'linear-gradient(180deg, var(--accent-amber) 0%, #D97706 100%)',
    icon: AlertTriangle,
    iconColor: 'var(--accent-amber)',
  },
  error: {
    borderGradient: 'linear-gradient(180deg, var(--accent-rose) 0%, #E11D48 100%)',
    icon: XCircle,
    iconColor: 'var(--accent-rose)',
    glow: '0 0 20px rgba(255,107,138,0.08)',
  },
  progress: {
    borderGradient: 'linear-gradient(180deg, var(--accent-violet) 0%, var(--accent-cyan) 100%)',
    icon: Loader2,
    iconColor: 'var(--accent-cyan)',
  },
};

/* ─── Single Toast Card ─────────────────────────────────────────── */
function ToastCard({ toast, index }: { toast: Toast; index: number }) {
  const { removeToast } = useToast();
  const navigate = useNavigate();
  const variant = VARIANTS[toast.variant];
  const Icon = variant.icon;
  const [exiting, setExiting] = useState(false);

  /* Time-bar for auto-dismissible toasts */
  const autoDismiss = toast.autoDismissMs
    ?? (toast.variant === 'success' ? 4000 : toast.variant === 'warning' ? 6000 : 0);

  const handleDismiss = () => {
    setExiting(true);
    setTimeout(() => removeToast(toast.id), 150);
  };

  const handleClick = () => {
    if (toast.linkTo) {
      navigate(toast.linkTo);
      handleDismiss();
    }
  };

  return (
    <div
      role="status"
      aria-live="polite"
      className={`relative flex items-start gap-3 p-3 pr-8 rounded-xl overflow-hidden ${exiting ? 'toast-exit' : 'toast-enter'}`}
      style={{
        background: 'linear-gradient(135deg, rgba(10,10,35,0.9) 0%, rgba(13,27,62,0.8) 50%, rgba(10,37,64,0.7) 100%)',
        backdropFilter: 'blur(20px) saturate(1.4)',
        WebkitBackdropFilter: 'blur(20px) saturate(1.4)',
        border: '1px solid var(--border-void)',
        boxShadow: variant.glow
          ? `${variant.glow}, 0 8px 32px rgba(0,0,0,0.3)`
          : '0 8px 32px rgba(0,0,0,0.3)',
        minWidth: 320,
        maxWidth: 400,
        cursor: toast.linkTo ? 'pointer' : 'default',
        animationDelay: `${index * 40}ms`,
      }}
      onClick={toast.linkTo ? handleClick : undefined}
    >
      {/* Left accent border */}
      <div
        className="absolute left-0 top-0 bottom-0 w-[3px]"
        style={{ background: variant.borderGradient }}
      />

      {/* Icon */}
      <div className="flex-shrink-0 mt-0.5">
        <Icon
          className={`w-[18px] h-[18px] ${toast.variant === 'progress' ? 'animate-spin' : ''}`}
          style={{ color: variant.iconColor }}
        />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <p className="text-[13px] font-medium truncate" style={{ color: 'var(--text-primary)' }}>
          {toast.title}
        </p>
        {toast.description && (
          <p className="text-[11px] mt-0.5 line-clamp-2" style={{ color: 'var(--text-secondary)' }}>
            {toast.description}
          </p>
        )}

        {/* Progress bar */}
        {toast.variant === 'progress' && toast.progress != null && (
          <div className="mt-2 h-[3px] rounded-full overflow-hidden" style={{ background: 'rgba(139,92,246,0.1)' }}>
            {toast.progress >= 0 ? (
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: `${toast.progress}%`,
                  background: 'linear-gradient(90deg, var(--accent-violet) 0%, var(--accent-cyan) 100%)',
                }}
              />
            ) : (
              <div className="h-full rounded-full progress-indeterminate" />
            )}
          </div>
        )}

        {/* Auto-dismiss timer bar */}
        {autoDismiss > 0 && toast.variant !== 'progress' && (
          <div className="mt-2 h-[2px] rounded-full overflow-hidden" style={{ background: 'rgba(139,92,246,0.06)' }}>
            <div
              className="h-full rounded-full"
              style={{
                background: variant.borderGradient,
                animation: `toast-timer ${autoDismiss}ms linear`,
                opacity: 0.5,
              }}
            />
          </div>
        )}
      </div>

      {/* Link arrow */}
      {toast.linkTo && (
        <ArrowRight className="w-3.5 h-3.5 flex-shrink-0 mt-1" style={{ color: 'var(--text-muted)' }} />
      )}

      {/* Close button */}
      <button
        onClick={(e) => { e.stopPropagation(); handleDismiss(); }}
        className="absolute top-2 right-2 p-1 rounded-md transition-colors hover:bg-[rgba(139,92,246,0.08)]"
        style={{ color: 'var(--text-muted)' }}
      >
        <X className="w-3 h-3" />
      </button>
    </div>
  );
}

/* ─── Container ─────────────────────────────────────────────────── */
export default function ToastContainer() {
  const { toasts } = useToast();

  if (toasts.length === 0) return null;

  return (
    <div
      className="fixed bottom-6 right-6 z-[200] flex flex-col gap-2"
      style={{ pointerEvents: 'none' }}
    >
      {toasts.map((toast, i) => (
        <div key={toast.id} style={{ pointerEvents: 'auto' }}>
          <ToastCard toast={toast} index={i} />
        </div>
      ))}
    </div>
  );
}
