import { useEffect, useRef, useState } from 'react';
import { X, Play, CheckCircle2, XCircle, Loader2 } from 'lucide-react';

export type JobMode = 'tune' | 'stocks' | 'retune' | 'calibrate';

interface JobEvent {
  type: string;
  message: string;
  count: number;
  success: number;
  fail: number;
}

interface LogLine {
  id: number;
  type: string;
  text: string;
}

const MODE_LABELS: Record<JobMode, { title: string; desc: string; color: string }> = {
  tune: { title: 'make tune', desc: 'Re-estimate model parameters for all assets', color: '#8b5cf6' },
  stocks: { title: 'make stocks', desc: 'Refresh prices and regenerate signals', color: '#3b82f6' },
  retune: { title: 'make retune', desc: 'Full retune pipeline', color: '#f59e0b' },
  calibrate: { title: 'make calibrate', desc: 'Re-tune failing assets', color: '#10b981' },
};

export function JobRunnerModal({
  open,
  mode,
  onClose,
}: {
  open: boolean;
  mode: JobMode | null;
  onClose: () => void;
}) {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'failed' | 'error'>('idle');
  const [counters, setCounters] = useState({ count: 0, success: 0, fail: 0 });
  const [elapsedSec, setElapsedSec] = useState(0);
  const esRef = useRef<EventSource | null>(null);
  const preRef = useRef<HTMLPreElement | null>(null);
  const idSeqRef = useRef(0);
  const startTimeRef = useRef<number>(0);
  const timerRef = useRef<number | null>(null);

  // Start stream when modal opens with a mode
  useEffect(() => {
    if (!open || !mode) return;
    // Reset state
    setLines([]);
    setStatus('running');
    setCounters({ count: 0, success: 0, fail: 0 });
    setElapsedSec(0);
    idSeqRef.current = 0;
    startTimeRef.current = Date.now();

    const es = new EventSource(`/api/tune/retune/stream?mode=${mode}`);
    esRef.current = es;

    timerRef.current = window.setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);

    es.onmessage = (ev) => {
      try {
        const data: JobEvent = JSON.parse(ev.data);
        idSeqRef.current += 1;
        setLines((prev) => {
          const next = [...prev, { id: idSeqRef.current, type: data.type, text: data.message }];
          // Cap lines to avoid memory blow-up
          if (next.length > 5000) return next.slice(-5000);
          return next;
        });
        if (data.count || data.success || data.fail) {
          setCounters({ count: data.count, success: data.success, fail: data.fail });
        }
        if (data.type === 'completed' || data.type === 'failed' || data.type === 'error') {
          setStatus(data.type);
          es.close();
          esRef.current = null;
          if (timerRef.current) {
            window.clearInterval(timerRef.current);
            timerRef.current = null;
          }
        }
      } catch (e) {
        // ignore malformed
      }
    };

    es.onerror = () => {
      // EventSource will auto-reconnect by default; we only mark error if not already finished
      setStatus((prev) => (prev === 'running' ? 'error' : prev));
    };

    return () => {
      es.close();
      esRef.current = null;
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [open, mode]);

  // Auto-scroll to bottom on new lines
  useEffect(() => {
    if (preRef.current) {
      preRef.current.scrollTop = preRef.current.scrollHeight;
    }
  }, [lines]);

  if (!open || !mode) return null;

  const info = MODE_LABELS[mode];
  const isRunning = status === 'running';

  const handleClose = () => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    onClose();
  };

  const lineColor = (type: string): string => {
    if (type === 'phase' || type === 'start') return '#a78bfa';
    if (type === 'completed') return '#10b981';
    if (type === 'failed' || type === 'error') return '#f43f5e';
    if (type === 'progress') return '#60a5fa';
    return 'rgba(255,255,255,0.72)';
  };

  const formatElapsed = (s: number) => {
    const m = Math.floor(s / 60);
    const ss = (s % 60).toString().padStart(2, '0');
    return `${m}:${ss}`;
  };

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.72)' }}
      onClick={handleClose}
    >
      <div
        className="w-full max-w-4xl max-h-[85vh] flex flex-col rounded-lg border overflow-hidden"
        style={{
          background: 'var(--void-bg)',
          borderColor: 'rgba(255,255,255,0.08)',
          boxShadow: '0 20px 60px rgba(0,0,0,0.6)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-5 py-3 border-b"
          style={{ borderColor: 'rgba(255,255,255,0.06)', background: 'var(--void-hover)' }}
        >
          <div className="flex items-center gap-3">
            <div
              className="w-8 h-8 rounded-md flex items-center justify-center"
              style={{ background: `${info.color}22`, color: info.color }}
            >
              {isRunning ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : status === 'completed' ? (
                <CheckCircle2 className="w-4 h-4" />
              ) : status === 'failed' || status === 'error' ? (
                <XCircle className="w-4 h-4" />
              ) : (
                <Play className="w-4 h-4" />
              )}
            </div>
            <div>
              <div className="text-sm font-semibold text-[var(--text-primary)] font-mono">
                {info.title}
              </div>
              <div className="text-[11px] text-[var(--text-muted)]">{info.desc}</div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-[11px] text-[var(--text-muted)] font-mono tabular-nums">
              {formatElapsed(elapsedSec)}
            </div>
            {(counters.count > 0 || counters.success > 0 || counters.fail > 0) && (
              <div className="flex items-center gap-2 text-[11px] font-mono tabular-nums">
                <span className="text-[var(--text-muted)]">total:</span>
                <span className="text-[var(--text-primary)]">{counters.count}</span>
                <span className="text-[#10b981]">ok {counters.success}</span>
                {counters.fail > 0 && <span className="text-[#f43f5e]">fail {counters.fail}</span>}
              </div>
            )}
            <span
              className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full font-semibold"
              style={{
                background:
                  status === 'running'
                    ? '#60a5fa22'
                    : status === 'completed'
                    ? '#10b98122'
                    : status === 'failed' || status === 'error'
                    ? '#f43f5e22'
                    : 'rgba(255,255,255,0.06)',
                color:
                  status === 'running'
                    ? '#60a5fa'
                    : status === 'completed'
                    ? '#10b981'
                    : status === 'failed' || status === 'error'
                    ? '#f43f5e'
                    : 'var(--text-muted)',
              }}
            >
              {status}
            </span>
            <button
              onClick={handleClose}
              className="p-1 rounded hover:bg-white/[0.06] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
              aria-label="Close"
              title={isRunning ? 'Cancel and close' : 'Close'}
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Log area */}
        <pre
          ref={preRef}
          className="flex-1 overflow-y-auto overflow-x-auto m-0 px-4 py-3 text-[12px] leading-[1.45] font-mono tabular-nums"
          style={{
            background: '#05060a',
            color: 'rgba(255,255,255,0.72)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all',
          }}
        >
          {lines.length === 0 && isRunning && (
            <span className="text-[var(--text-muted)]">Waiting for output…</span>
          )}
          {lines.map((l) => (
            <div key={l.id} style={{ color: lineColor(l.type) }}>
              {l.text}
            </div>
          ))}
        </pre>

        {/* Footer */}
        <div
          className="flex items-center justify-between px-5 py-2 border-t text-[11px] text-[var(--text-muted)]"
          style={{ borderColor: 'rgba(255,255,255,0.06)', background: 'var(--void-hover)' }}
        >
          <span>{lines.length} lines</span>
          <span>
            {isRunning
              ? 'Streaming… click outside or X to cancel'
              : 'Finished — click outside or X to close'}
          </span>
        </div>
      </div>
    </div>
  );
}
