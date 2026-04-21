import { useEffect, useMemo, useRef, useState } from 'react';
import { X, CheckCircle2, XCircle, Loader2, ChevronDown, ChevronRight, Play, Activity } from 'lucide-react';

export type JobMode = 'tune' | 'stocks' | 'retune' | 'calibrate';

type EventType =
  | 'start'
  | 'phase'
  | 'asset'
  | 'heartbeat'
  | 'log'
  | 'completed'
  | 'failed'
  | 'error';

interface JobEvent {
  type: EventType;
  mode?: string;
  title?: string;
  symbol?: string;
  status?: 'ok' | 'fail';
  detail?: string;
  message?: string;
  done?: number;
  fail?: number;
  total?: number;
  elapsed_s?: number;
  exit_code?: number;
}

interface LogLine {
  id: number;
  text: string;
}

interface AssetEntry {
  symbol: string;
  status: 'ok' | 'fail';
  detail?: string;
}

interface PhaseEntry {
  id: number;
  title: string;
}

const MODE_LABELS: Record<JobMode, { title: string; desc: string; color: string }> = {
  tune:      { title: 'Retune models',    desc: 'Re-estimate model parameters',          color: '#8b5cf6' },
  stocks:    { title: 'Refresh stocks',   desc: 'Download prices and regenerate signals', color: '#3b82f6' },
  retune:    { title: 'Run tune',         desc: 'Full retune pipeline (make retune)',    color: '#8b5cf6' },
  calibrate: { title: 'Calibrate',        desc: 'Re-tune failing assets',                color: '#10b981' },
};

export function JobRunnerPanel({
  open,
  mode,
  onClose,
}: {
  open: boolean;
  mode: JobMode | null;
  onClose: () => void;
}) {
  const [phases, setPhases] = useState<PhaseEntry[]>([]);
  const [assets, setAssets] = useState<AssetEntry[]>([]);
  const [logLines, setLogLines] = useState<LogLine[]>([]);
  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'failed' | 'error'>('idle');
  const [counters, setCounters] = useState({ done: 0, fail: 0, total: 0 });
  const [elapsedSec, setElapsedSec] = useState(0);
  const [showRawLog, setShowRawLog] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const esRef = useRef<EventSource | null>(null);
  const preRef = useRef<HTMLDivElement | null>(null);
  const logIdRef = useRef(0);
  const phaseIdRef = useRef(0);
  const startTimeRef = useRef<number>(0);
  const timerRef = useRef<number | null>(null);

  // Buffered updates — SSE can fire hundreds of events per second
  const bufferRef = useRef<{
    newAssets: AssetEntry[];
    newPhases: PhaseEntry[];
    newLogs: LogLine[];
    counters: { done: number; fail: number; total: number };
    elapsed: number | null;
    finalStatus: 'completed' | 'failed' | 'error' | null;
  }>({
    newAssets: [],
    newPhases: [],
    newLogs: [],
    counters: { done: 0, fail: 0, total: 0 },
    elapsed: null,
    finalStatus: null,
  });
  const flushRef = useRef<number | null>(null);

  const modeColor = mode ? MODE_LABELS[mode].color : '#8b5cf6';

  const flushBuffer = () => {
    const b = bufferRef.current;
    if (b.newAssets.length > 0) {
      setAssets((prev) => {
        const next = prev.concat(b.newAssets);
        return next.length > 500 ? next.slice(-500) : next;
      });
      b.newAssets = [];
    }
    if (b.newPhases.length > 0) {
      setPhases((prev) => {
        const next = prev.concat(b.newPhases);
        return next.length > 30 ? next.slice(-30) : next;
      });
      b.newPhases = [];
    }
    if (b.newLogs.length > 0) {
      setLogLines((prev) => {
        const next = prev.concat(b.newLogs);
        return next.length > 2000 ? next.slice(-2000) : next;
      });
      b.newLogs = [];
    }
    setCounters(b.counters);
    if (b.elapsed !== null) {
      setElapsedSec(b.elapsed);
      b.elapsed = null;
    }
    if (b.finalStatus) {
      setStatus(b.finalStatus);
      b.finalStatus = null;
    }
  };

  useEffect(() => {
    if (!open || !mode) return;

    setPhases([]);
    setAssets([]);
    setLogLines([]);
    setStatus('running');
    setCounters({ done: 0, fail: 0, total: 0 });
    setElapsedSec(0);
    setErrorMsg(null);
    logIdRef.current = 0;
    phaseIdRef.current = 0;
    startTimeRef.current = Date.now();
    bufferRef.current = {
      newAssets: [],
      newPhases: [],
      newLogs: [],
      counters: { done: 0, fail: 0, total: 0 },
      elapsed: null,
      finalStatus: null,
    };

    const es = new EventSource(`/api/tune/retune/stream?mode=${mode}`);
    esRef.current = es;

    timerRef.current = window.setInterval(() => {
      setElapsedSec(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);

    flushRef.current = window.setInterval(flushBuffer, 120);

    es.onmessage = (ev) => {
      let data: JobEvent;
      try {
        data = JSON.parse(ev.data);
      } catch {
        return;
      }
      const buf = bufferRef.current;

      switch (data.type) {
        case 'start':
          if (typeof data.total === 'number') {
            buf.counters = { ...buf.counters, total: data.total };
          }
          break;

        case 'phase':
          if (data.title) {
            phaseIdRef.current += 1;
            buf.newPhases.push({ id: phaseIdRef.current, title: data.title });
          }
          break;

        case 'asset':
          if (data.symbol) {
            buf.newAssets.push({
              symbol: data.symbol,
              status: data.status === 'fail' ? 'fail' : 'ok',
              detail: data.detail,
            });
          }
          buf.counters = {
            done: data.done ?? buf.counters.done,
            fail: data.fail ?? buf.counters.fail,
            total: data.total ?? buf.counters.total,
          };
          break;

        case 'heartbeat':
          buf.counters = {
            done: data.done ?? buf.counters.done,
            fail: buf.counters.fail,
            total: data.total ?? buf.counters.total,
          };
          if (typeof data.elapsed_s === 'number') {
            buf.elapsed = data.elapsed_s;
          }
          break;

        case 'log':
          if (data.message) {
            logIdRef.current += 1;
            buf.newLogs.push({ id: logIdRef.current, text: data.message });
          }
          break;

        case 'completed':
        case 'failed':
        case 'error':
          buf.finalStatus = data.type;
          if (data.type === 'error' && data.message) {
            setErrorMsg(data.message);
          }
          es.close();
          esRef.current = null;
          if (timerRef.current) {
            window.clearInterval(timerRef.current);
            timerRef.current = null;
          }
          flushBuffer();
          if (flushRef.current) {
            window.clearInterval(flushRef.current);
            flushRef.current = null;
          }
          break;
      }
    };

    es.onerror = () => {
      setStatus((prev) => (prev === 'running' ? 'error' : prev));
    };

    return () => {
      es.close();
      esRef.current = null;
      if (timerRef.current) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
      if (flushRef.current) {
        window.clearInterval(flushRef.current);
        flushRef.current = null;
      }
    };
  }, [open, mode]);

  useEffect(() => {
    if (showRawLog && preRef.current) {
      preRef.current.scrollTop = preRef.current.scrollHeight;
    }
  }, [logLines, showRawLog]);

  const handleClose = () => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (flushRef.current) {
      window.clearInterval(flushRef.current);
      flushRef.current = null;
    }
    onClose();
  };

  const formatElapsed = (s: number) => {
    const m = Math.floor(s / 60);
    const ss = (s % 60).toString().padStart(2, '0');
    return `${m}:${ss}`;
  };

  const recentAssets = useMemo(() => assets.slice(-32).reverse(), [assets]);
  const currentPhase = phases.length > 0 ? phases[phases.length - 1] : null;

  const progressPct = counters.total > 0
    ? Math.min(100, ((counters.done + counters.fail) / counters.total) * 100)
    : (status === 'running' ? 0 : 100);

  if (!open || !mode) return null;

  const info = MODE_LABELS[mode];
  const isRunning = status === 'running';

  const statusColor =
    status === 'running' ? '#60a5fa' :
    status === 'completed' ? '#10b981' :
    status === 'failed' || status === 'error' ? '#f43f5e' :
    'var(--text-muted)';

  const eta = (() => {
    if (!isRunning || counters.done === 0 || counters.total === 0) return null;
    const rate = counters.done / Math.max(elapsedSec, 1);
    const remaining = counters.total - counters.done - counters.fail;
    if (rate <= 0 || remaining <= 0) return null;
    const secs = Math.round(remaining / rate);
    return formatElapsed(secs);
  })();

  return (
    <div
      className="mb-3 rounded-lg border overflow-hidden"
      style={{
        background: 'var(--void-bg)',
        borderColor: `${modeColor}33`,
        boxShadow: `0 4px 16px ${modeColor}1a`,
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-2.5"
        style={{ background: `${modeColor}0f`, borderBottom: '1px solid rgba(255,255,255,0.06)' }}
      >
        <div className="flex items-center gap-3">
          <div
            className="w-7 h-7 rounded-md flex items-center justify-center"
            style={{ background: `${modeColor}22`, color: modeColor }}
          >
            {isRunning ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> :
             status === 'completed' ? <CheckCircle2 className="w-3.5 h-3.5" /> :
             status === 'failed' || status === 'error' ? <XCircle className="w-3.5 h-3.5" /> :
             <Play className="w-3.5 h-3.5" />}
          </div>
          <div>
            <div className="text-[13px] font-semibold text-[var(--text-primary)] leading-tight">
              {info.title}
            </div>
            <div className="text-[11px] text-[var(--text-muted)] leading-tight">{info.desc}</div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span
            className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full font-semibold"
            style={{ background: `${statusColor}22`, color: statusColor }}
          >
            {status}
          </span>
          <span className="text-[11px] text-[var(--text-muted)] font-mono tabular-nums">
            {formatElapsed(elapsedSec)}
            {eta && <span style={{ opacity: 0.6 }}> · eta {eta}</span>}
          </span>
          <button
            onClick={handleClose}
            className="p-1 rounded hover:bg-white/[0.06] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
            aria-label="Close"
            title={isRunning ? 'Cancel and close' : 'Close'}
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-1" style={{ background: 'rgba(255,255,255,0.04)' }}>
        <div
          className="h-full transition-all duration-300"
          style={{
            width: `${progressPct}%`,
            background: status === 'failed' || status === 'error' ? '#f43f5e' : modeColor,
            boxShadow: `0 0 8px ${modeColor}66`,
          }}
        />
      </div>

      {/* Body */}
      <div className="p-4 space-y-3">
        {/* Hero stats row */}
        <div className="grid grid-cols-4 gap-2">
          <StatTile
            label="Processed"
            value={counters.done + counters.fail}
            suffix={counters.total > 0 ? `/ ${counters.total}` : undefined}
            color="var(--text-primary)"
          />
          <StatTile
            label="Successful"
            value={counters.done}
            color="#10b981"
            dot={isRunning && counters.done > 0}
          />
          <StatTile
            label="Failed"
            value={counters.fail}
            color={counters.fail > 0 ? '#f43f5e' : 'var(--text-muted)'}
          />
          <StatTile
            label="Progress"
            value={`${Math.round(progressPct)}%`}
            color={modeColor}
          />
        </div>

        {/* Current phase */}
        {currentPhase && (
          <div
            className="flex items-center gap-2 px-3 py-2 rounded"
            style={{
              background: `${modeColor}0f`,
              border: `1px solid ${modeColor}22`,
            }}
          >
            {isRunning ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" style={{ color: modeColor }} />
            ) : (
              <Activity className="w-3.5 h-3.5 shrink-0" style={{ color: modeColor }} />
            )}
            <div className="flex-1 min-w-0">
              <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-semibold leading-tight">
                Current phase
              </div>
              <div className="text-[12px] font-medium text-[var(--text-primary)] truncate">
                {currentPhase.title}
              </div>
            </div>
            {phases.length > 1 && (
              <span className="text-[10px] text-[var(--text-muted)] tabular-nums">
                {phases.length} total
              </span>
            )}
          </div>
        )}

        {/* Assets grid */}
        {recentAssets.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-semibold">
                Recent completions
              </div>
              <span className="text-[10px] text-[var(--text-muted)] tabular-nums">
                showing {recentAssets.length} of {assets.length}
              </span>
            </div>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(120px,1fr))] gap-1">
              {recentAssets.map((a, i) => (
                <div
                  key={`${a.symbol}-${i}`}
                  className="flex items-center gap-1.5 px-2 py-1 rounded text-[11px] font-mono"
                  style={{
                    background: a.status === 'ok' ? 'rgba(16,185,129,0.06)' : 'rgba(244,63,94,0.08)',
                    border: `1px solid ${a.status === 'ok' ? 'rgba(16,185,129,0.2)' : 'rgba(244,63,94,0.28)'}`,
                  }}
                  title={a.detail ? `${a.symbol} - ${a.detail}` : a.symbol}
                >
                  {a.status === 'ok' ? (
                    <CheckCircle2 className="w-3 h-3 shrink-0" style={{ color: '#10b981' }} />
                  ) : (
                    <XCircle className="w-3 h-3 shrink-0" style={{ color: '#f43f5e' }} />
                  )}
                  <span className="text-[var(--text-primary)] font-semibold truncate">{a.symbol}</span>
                  {a.detail && (
                    <span className="text-[var(--text-muted)] ml-auto shrink-0" style={{ fontSize: '10px' }}>
                      {a.detail}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Starting state */}
        {isRunning && !currentPhase && recentAssets.length === 0 && (
          <div className="text-[12px] text-[var(--text-muted)] italic flex items-center gap-2 py-2">
            <Loader2 className="w-3 h-3 animate-spin" />
            Starting job&hellip;
          </div>
        )}

        {/* Terminal state banner */}
        {!isRunning && status === 'completed' && (
          <div
            className="flex items-center gap-2 px-3 py-2 rounded text-[12px]"
            style={{
              background: 'rgba(16,185,129,0.08)',
              border: '1px solid rgba(16,185,129,0.24)',
              color: '#10b981',
            }}
          >
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span className="font-medium">
              Completed · {counters.done} successful{counters.fail > 0 ? `, ${counters.fail} failed` : ''} in {formatElapsed(elapsedSec)}
            </span>
          </div>
        )}
        {!isRunning && (status === 'failed' || status === 'error') && (
          <div
            className="flex items-center gap-2 px-3 py-2 rounded text-[12px]"
            style={{
              background: 'rgba(244,63,94,0.08)',
              border: '1px solid rgba(244,63,94,0.28)',
              color: '#f43f5e',
            }}
          >
            <XCircle className="w-3.5 h-3.5" />
            <span className="font-medium">
              {status === 'error' ? (errorMsg || 'Stream error') : `Failed after ${formatElapsed(elapsedSec)}`}
            </span>
          </div>
        )}

        {/* Collapsible raw log */}
        <div className="pt-1">
          <button
            onClick={() => setShowRawLog((v) => !v)}
            className="inline-flex items-center gap-1 text-[11px] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
          >
            {showRawLog ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
            Raw log ({logLines.length})
          </button>
          {showRawLog && (
            <div
              ref={preRef}
              className="mt-2 max-h-[240px] overflow-y-auto overflow-x-auto rounded px-3 py-2 text-[11px] leading-[1.5] font-mono"
              style={{
                background: '#05060a',
                color: 'rgba(255,255,255,0.62)',
                border: '1px solid rgba(255,255,255,0.06)',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-all',
              }}
            >
              {logLines.length === 0 ? (
                <span className="text-[var(--text-muted)]">(no subprocess output yet)</span>
              ) : (
                logLines.map((l) => (
                  <div key={l.id}>{l.text}</div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatTile({
  label,
  value,
  suffix,
  color,
  dot,
}: {
  label: string;
  value: number | string;
  suffix?: string;
  color: string;
  dot?: boolean;
}) {
  return (
    <div
      className="px-3 py-2 rounded"
      style={{
        background: 'rgba(255,255,255,0.02)',
        border: '1px solid rgba(255,255,255,0.05)',
      }}
    >
      <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-semibold">
        {label}
      </div>
      <div className="flex items-baseline gap-1.5 mt-0.5">
        {dot && (
          <span
            className="w-1.5 h-1.5 rounded-full animate-pulse shrink-0"
            style={{ background: color, boxShadow: `0 0 6px ${color}`, alignSelf: 'center' }}
          />
        )}
        <span className="text-[16px] font-semibold tabular-nums leading-none" style={{ color }}>
          {value}
        </span>
        {suffix && (
          <span className="text-[11px] text-[var(--text-muted)] tabular-nums">{suffix}</span>
        )}
      </div>
    </div>
  );
}

/** Back-compat alias for legacy import paths. */
export const JobRunnerModal = JobRunnerPanel;
