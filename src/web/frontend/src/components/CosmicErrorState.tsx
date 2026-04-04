/**
 * Cosmic Error States (Story 10.3)
 *
 * Error messages that are clear, non-technical, and provide recovery paths.
 * Rose accents but calm -- not screaming alerts.
 */
import { useState, useEffect, useCallback } from 'react';
import { RefreshCw, ChevronDown, WifiOff, AlertCircle } from 'lucide-react';

/* ── Error Card ──────────────────────────────────────────────── */

interface ErrorCardProps {
  title?: string;
  description?: string;
  error?: Error | string | null;
  onRetry?: () => void;
  onSecondary?: () => void;
  secondaryLabel?: string;
  /** Show network-disconnected variant */
  isNetworkError?: boolean;
}

export function CosmicErrorCard({
  title,
  description,
  error,
  onRetry,
  onSecondary,
  secondaryLabel = 'View Status',
  isNetworkError = false,
}: ErrorCardProps) {
  const [showDetails, setShowDetails] = useState(false);
  const [retryState, setRetryState] = useState<{ attempt: number; countdown: number } | null>(null);

  const errorMessage = typeof error === 'string' ? error : error?.message ?? '';
  const displayTitle = title ?? (isNetworkError ? 'No connection to the server' : 'Unable to load data');
  const displayDesc = description ?? (isNetworkError
    ? 'Check that the backend is running.'
    : 'The backend may be restarting. This usually resolves within seconds.');

  const BACKOFF = [1, 3, 8];

  const doRetry = useCallback(() => {
    if (onRetry) onRetry();
  }, [onRetry]);

  useEffect(() => {
    if (!retryState) return;
    if (retryState.countdown <= 0) {
      doRetry();
      if (retryState.attempt < BACKOFF.length - 1) {
        setRetryState({ attempt: retryState.attempt + 1, countdown: BACKOFF[retryState.attempt + 1] });
      } else {
        setRetryState(null);
      }
      return;
    }
    const t = setTimeout(() => setRetryState({ ...retryState, countdown: retryState.countdown - 1 }), 1000);
    return () => clearTimeout(t);
  }, [retryState, doRetry]);

  const handleRetry = () => {
    if (retryState) return;
    doRetry();
    setRetryState({ attempt: 0, countdown: BACKOFF[0] });
  };

  return (
    <div className="fade-up" style={{ maxWidth: 480, margin: '0 auto' }}>
      <div
        className="glass-card p-6 error-pulse-glow"
        style={{
          background: `radial-gradient(ellipse at 50% 50%, rgba(255,107,138,0.04) 0%, transparent 60%), linear-gradient(135deg, rgba(10,10,35,0.85), rgba(13,27,62,0.75))`,
        }}
      >
        <div className="flex items-start gap-4">
          {/* Icon */}
          <div className="flex-shrink-0 mt-0.5">
            {isNetworkError ? (
              <WifiOff size={24} style={{ color: 'var(--accent-rose)' }} />
            ) : (
              <AlertCircle size={24} style={{ color: 'var(--accent-rose)' }} />
            )}
          </div>

          <div className="flex-1 space-y-3">
            {/* Title */}
            <h3 className="text-base font-semibold" style={{ color: 'var(--text-luminous)' }}>
              {displayTitle}
            </h3>

            {/* Description */}
            <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
              {displayDesc}
            </p>

            {/* Network hint */}
            {isNetworkError && (
              <code
                className="block text-xs px-3 py-2 rounded-lg"
                style={{
                  background: 'rgba(10,10,26,0.6)',
                  border: '1px solid var(--border-void)',
                  color: 'var(--text-muted)',
                  fontFamily: 'monospace',
                }}
              >
                make web-backend
              </code>
            )}

            {/* Buttons */}
            <div className="flex items-center gap-3 pt-1">
              {onRetry && (
                <button
                  onClick={handleRetry}
                  disabled={!!retryState}
                  className="flex items-center gap-2 h-9 px-4 rounded-xl text-sm font-semibold text-white
                    transition-all duration-200 hover:shadow-lg hover:shadow-rose-500/20 hover:-translate-y-px
                    disabled:opacity-60 disabled:cursor-default"
                  style={{ background: 'linear-gradient(135deg, #ff6b8a, #ff5577)' }}
                >
                  {retryState ? (
                    <>
                      <span className="w-2 h-2 rounded-full bg-white animate-spin" />
                      Retrying in {retryState.countdown}s...
                    </>
                  ) : (
                    <>
                      <RefreshCw size={14} />
                      Try Again
                    </>
                  )}
                </button>
              )}
              {onSecondary && (
                <button
                  onClick={onSecondary}
                  className="h-9 px-4 rounded-xl text-sm font-medium transition-all duration-200
                    hover:bg-[rgba(139,92,246,0.08)]"
                  style={{ color: 'var(--text-secondary)', border: '1px solid var(--border-void)' }}
                >
                  {secondaryLabel}
                </button>
              )}
            </div>

            {/* Technical details (collapsible) */}
            {errorMessage && (
              <div className="pt-2">
                <button
                  onClick={() => setShowDetails(!showDetails)}
                  className="flex items-center gap-1 text-xs transition-colors duration-150"
                  style={{ color: 'var(--text-muted)' }}
                >
                  <ChevronDown
                    size={12}
                    className="transition-transform duration-200"
                    style={{ transform: showDetails ? 'rotate(180deg)' : 'rotate(0)' }}
                  />
                  {showDetails ? 'Hide details' : 'Show details'}
                </button>
                {showDetails && (
                  <pre
                    className="mt-2 text-[10px] px-3 py-2 rounded-lg overflow-x-auto"
                    style={{
                      background: 'rgba(10,10,26,0.6)',
                      border: '1px solid var(--border-void)',
                      color: 'var(--text-muted)',
                      fontFamily: 'monospace',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-all',
                    }}
                  >
                    {errorMessage}
                    {'\n'}Timestamp: {new Date().toISOString()}
                  </pre>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Partial Error Banner ────────────────────────────────────── */

export function PartialErrorBanner({ message, onRetry }: { message?: string; onRetry?: () => void }) {
  return (
    <div
      className="flex items-center gap-3 px-4 h-10 text-sm fade-up"
      style={{
        background: 'rgba(255,107,138,0.04)',
        borderLeft: '3px solid var(--accent-rose)',
        color: 'var(--text-secondary)',
      }}
    >
      <AlertCircle size={14} style={{ color: 'var(--accent-rose)' }} />
      <span className="flex-1">{message ?? 'Some data unavailable'}</span>
      {onRetry && (
        <button
          onClick={onRetry}
          className="text-xs font-medium transition-colors duration-150 hover:underline"
          style={{ color: '#ff6b8a' }}
        >
          Retry
        </button>
      )}
    </div>
  );
}
