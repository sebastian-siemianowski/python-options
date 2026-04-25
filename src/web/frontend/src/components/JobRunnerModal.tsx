import { useEffect, useMemo, useRef, useState } from 'react';
import { X, CheckCircle2, XCircle, Loader2, ChevronDown, ChevronRight, Play, Activity, Download, Sparkles } from 'lucide-react';

export type JobMode = 'tune' | 'stocks' | 'retune' | 'calibrate';

type EventType =
  | 'start'
  | 'phase'
  | 'asset'
  | 'model'
  | 'refresh'
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
  // Multi-phase metadata (retune pipeline: refresh -> backup -> tune)
  step?: number;
  total_steps?: number;
  phase_count?: number;
  kind?: string;
  phase_step?: number;
  phase_title?: string;
  // Per-asset model selection (emitted from tune logs).
  model?: string;
  // Refresh pass telemetry (emitted from refresh logs).
  pass?: number;
  total_passes?: number;
  ok?: number;
  pending?: number;
  // Structured error surface.
  error_type?: string;
  error?: string;
}

interface LogLine {
  id: number;
  text: string;
}

interface AssetEntry {
  symbol: string;
  status: 'ok' | 'fail';
  detail?: string;
  model?: string;
}

interface PhaseEntry {
  id: number;
  title: string;
  step?: number;
  totalSteps?: number;
  kind?: string;
  startedAt: number;
}

interface RefreshPassState {
  pass: number;
  totalPasses: number;
  ok: number;
  pending: number;
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
  // Rich telemetry derived from subprocess logs.
  const [refreshPass, setRefreshPass] = useState<RefreshPassState | null>(null);
  const [modelBySymbol, setModelBySymbol] = useState<Record<string, string>>({});
  const [modelCounts, setModelCounts] = useState<Record<string, number>>({});

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
    modelUpdates: Record<string, string>;
    refreshPass: RefreshPassState | null;
  }>({
    newAssets: [],
    newPhases: [],
    newLogs: [],
    counters: { done: 0, fail: 0, total: 0 },
    elapsed: null,
    finalStatus: null,
    modelUpdates: {},
    refreshPass: null,
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
    if (Object.keys(b.modelUpdates).length > 0) {
      const updates = b.modelUpdates;
      b.modelUpdates = {};
      setModelBySymbol((prev) => ({ ...prev, ...updates }));
      setModelCounts((prev) => {
        const next = { ...prev };
        for (const [sym, model] of Object.entries(updates)) {
          // Only count each symbol once — if we already have a model for it,
          // skip (tune occasionally re-prints the same selection).
          if (modelBySymbol[sym] === undefined || modelBySymbol[sym] !== model) {
            next[model] = (next[model] ?? 0) + 1;
          }
        }
        return next;
      });
    }
    if (b.refreshPass) {
      setRefreshPass(b.refreshPass);
      b.refreshPass = null;
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
    setRefreshPass(null);
    setModelBySymbol({});
    setModelCounts({});
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
      modelUpdates: {},
      refreshPass: null,
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
            buf.newPhases.push({
              id: phaseIdRef.current,
              title: data.title,
              step: data.step,
              totalSteps: data.total_steps,
              kind: data.kind,
              startedAt: Date.now(),
            });
          }
          break;

        case 'asset':
          if (data.symbol) {
            buf.newAssets.push({
              symbol: data.symbol,
              status: data.status === 'fail' ? 'fail' : 'ok',
              detail: data.detail,
              model: data.model,
            });
            if (data.model) {
              buf.modelUpdates[data.symbol] = data.model;
            }
          }
          buf.counters = {
            done: data.done ?? buf.counters.done,
            fail: data.fail ?? buf.counters.fail,
            total: data.total ?? buf.counters.total,
          };
          break;

        case 'model':
          if (data.symbol && data.model) {
            buf.modelUpdates[data.symbol] = data.model;
          }
          break;

        case 'refresh':
          if (typeof data.pass === 'number') {
            const prev = buf.refreshPass;
            buf.refreshPass = {
              pass: data.pass,
              totalPasses: data.total_passes ?? prev?.totalPasses ?? 1,
              ok: data.ok ?? prev?.ok ?? 0,
              pending: data.pending ?? prev?.pending ?? 0,
            };
          }
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

        case 'error':
          // Non-terminal: structured error extracted from a traceback during
          // the run. Surface the message but keep the EventSource open so the
          // subsequent `failed` terminal event can still arrive.
          setErrorMsg(
            data.message || data.error_type || 'Unknown error'
          );
          break;

        case 'completed':
        case 'failed':
          buf.finalStatus = data.type;
          if (data.type === 'failed' && data.error) {
            setErrorMsg((prev) => prev || data.error || null);
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

  // Stable mm:ss for header. Integer-only; safe for float server elapsed_s.
  const formatElapsed = (raw: number) => {
    const s = Math.max(0, Math.round(Number(raw) || 0));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = (s % 60).toString().padStart(2, '0');
    if (h > 0) {
      const mm = m.toString().padStart(2, '0');
      return `${h}:${mm}:${ss}`;
    }
    return `${m}:${ss}`;
  };

  // Human-friendly "2m 34s" / "1h 05m" for banners/tooltips.
  const formatDuration = (raw: number) => {
    const s = Math.max(0, Math.round(Number(raw) || 0));
    if (s < 60) return `${s}s`;
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    if (h > 0) return `${h}h ${m.toString().padStart(2, '0')}m`;
    return ss === 0 ? `${m}m` : `${m}m ${ss.toString().padStart(2, '0')}s`;
  };

  const recentAssets = useMemo(() => assets.slice(-32).reverse(), [assets]);
  const currentPhase = phases.length > 0 ? phases[phases.length - 1] : null;

  // Top-N model distribution chips during the tune phase.
  const topModels = useMemo(() => {
    const entries = Object.entries(modelCounts);
    if (entries.length === 0) return [] as Array<{ name: string; count: number; hue: number }>;
    entries.sort((a, b) => b[1] - a[1]);
    // Deterministic hash -> hue for stable coloring.
    const hashHue = (s: string) => {
      let h = 0;
      for (let i = 0; i < s.length; i += 1) h = (h * 31 + s.charCodeAt(i)) >>> 0;
      return h % 360;
    };
    return entries.slice(0, 6).map(([name, count]) => ({ name, count, hue: hashHue(name) }));
  }, [modelCounts]);
  const totalModelled = useMemo(
    () => Object.values(modelCounts).reduce((acc, v) => acc + v, 0),
    [modelCounts],
  );

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

  const statusLabel =
    status === 'running' ? 'Running' :
    status === 'completed' ? 'Complete' :
    status === 'failed' ? 'Failed' :
    status === 'error' ? 'Error' :
    'Idle';

  // Throughput: completed assets per minute. Skip tiny-elapsed noise.
  const processed = counters.done + counters.fail;
  const throughput = elapsedSec >= 5 && processed > 0
    ? processed / (elapsedSec / 60)
    : null;
  const avgPerAsset = processed > 0 ? elapsedSec / processed : null;
  const successRate = processed > 0 ? (counters.done / processed) * 100 : null;

  const eta = (() => {
    if (!isRunning || processed === 0 || counters.total === 0) return null;
    const rate = processed / Math.max(elapsedSec, 1);
    const remaining = counters.total - processed;
    if (rate <= 0 || remaining <= 0) return null;
    return Math.round(remaining / rate);
  })();

  return (
    <div
      className="mb-3 rounded-xl border overflow-hidden"
      style={{
        background:
          'linear-gradient(180deg, rgba(10,12,20,0.96) 0%, rgba(6,7,14,0.98) 100%)',
        borderColor: `${modeColor}33`,
        boxShadow: `0 12px 40px -16px ${modeColor}40, 0 0 0 1px rgba(255,255,255,0.02), inset 0 1px 0 rgba(255,255,255,0.03)`,
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3"
        style={{
          background: `linear-gradient(180deg, ${modeColor}14, ${modeColor}06)`,
          borderBottom: '1px solid rgba(255,255,255,0.05)',
        }}
      >
        <div className="flex items-center gap-3 min-w-0">
          <div
            className="relative w-8 h-8 rounded-lg flex items-center justify-center shrink-0"
            style={{
              background: `linear-gradient(180deg, ${modeColor}2a, ${modeColor}14)`,
              color: modeColor,
              border: `1px solid ${modeColor}33`,
              boxShadow: isRunning ? `0 0 18px -2px ${modeColor}55` : undefined,
            }}
          >
            {isRunning ? <Loader2 className="w-4 h-4 animate-spin" /> :
             status === 'completed' ? <CheckCircle2 className="w-4 h-4" /> :
             status === 'failed' || status === 'error' ? <XCircle className="w-4 h-4" /> :
             <Play className="w-4 h-4" />}
            {isRunning && (
              <span
                className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 rounded-full animate-pulse"
                style={{ background: modeColor, boxShadow: `0 0 6px ${modeColor}` }}
              />
            )}
          </div>
          <div className="min-w-0">
            <div className="text-[13px] font-semibold text-[var(--text-primary)] leading-tight tracking-tight truncate">
              {info.title}
            </div>
            <div className="text-[11px] text-[var(--text-muted)] leading-tight truncate">{info.desc}</div>
          </div>
        </div>
        <div className="flex items-center gap-2.5 shrink-0">
          <span
            className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-[0.08em] px-2 py-0.5 rounded-full font-semibold"
            style={{
              background: `${statusColor}1F`,
              color: statusColor,
              border: `1px solid ${statusColor}33`,
            }}
          >
            {isRunning && (
              <span
                className="w-1.5 h-1.5 rounded-full animate-pulse"
                style={{ background: statusColor }}
              />
            )}
            {statusLabel}
          </span>
          <div className="flex flex-col items-end leading-tight">
            <span
              className="text-[12px] text-[var(--text-primary)] font-semibold font-mono tabular-nums"
              title={`Elapsed ${formatDuration(elapsedSec)}`}
            >
              {formatElapsed(elapsedSec)}
            </span>
            {eta !== null && (
              <span className="text-[10px] text-[var(--text-muted)] tabular-nums">
                eta {formatElapsed(eta)}
              </span>
            )}
          </div>
          <button
            onClick={handleClose}
            className="p-1.5 rounded-md hover:bg-white/[0.06] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
            aria-label="Close"
            title={isRunning ? 'Cancel and close' : 'Close'}
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Progress bar */}
      <div
        className="relative h-2 overflow-hidden rounded-full mx-4 mt-3 p-[1px]"
        style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.13), rgba(255,255,255,0.04))', boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.35)' }}
      >
        <div
          className="relative h-full rounded-full transition-[width] duration-500 ease-out"
          style={{
            width: `${progressPct}%`,
            background:
              status === 'failed' || status === 'error'
                ? 'linear-gradient(90deg, #f43f5e, #fb7185, #fecdd3)'
                : `linear-gradient(90deg, ${modeColor} 0%, #38d9f5 62%, rgba(255,255,255,0.9) 100%)`,
            boxShadow: `0 0 18px -8px ${modeColor}, inset 0 1px 0 rgba(255,255,255,0.34)`,
          }}
        >
          <span aria-hidden className="absolute inset-x-1 top-0 h-px rounded-full" style={{ background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.72), transparent)' }} />
          {isRunning && progressPct > 2 && <span aria-hidden className="absolute right-0 top-1/2 h-3 w-3 -translate-y-1/2 translate-x-1/2 rounded-full" style={{ background: '#fff', boxShadow: `0 0 18px 4px ${modeColor}66` }} />}
        </div>
      </div>

      {/* Body */}
      <div className="p-4 space-y-3">
        {/* Hero stats row */}
        <div className="grid grid-cols-4 gap-2">
          <StatTile
            label="Processed"
            value={processed}
            suffix={counters.total > 0 ? `/ ${counters.total}` : undefined}
            color="var(--text-primary)"
          />
          <StatTile
            label="Successful"
            value={counters.done}
            suffix={successRate !== null ? `${successRate.toFixed(0)}%` : undefined}
            color="#10b981"
            dot={isRunning && counters.done > 0}
          />
          <StatTile
            label="Failed"
            value={counters.fail}
            color={counters.fail > 0 ? '#f43f5e' : 'var(--text-muted)'}
          />
          <StatTile
            label={throughput !== null ? 'Throughput' : 'Progress'}
            value={throughput !== null ? throughput.toFixed(1) : `${Math.round(progressPct)}%`}
            suffix={throughput !== null ? '/ min' : undefined}
            color={modeColor}
            tooltip={
              throughput !== null
                ? `${throughput.toFixed(1)} assets/min · avg ${avgPerAsset ? avgPerAsset.toFixed(1) : '—'}s per asset`
                : undefined
            }
          />
        </div>

        {/* Current phase */}
        {currentPhase && (
          <div
            className="rounded-lg overflow-hidden"
            style={{
              background: `linear-gradient(180deg, ${modeColor}12, ${modeColor}04)`,
              border: `1px solid ${modeColor}26`,
            }}
          >
            <div className="flex items-center gap-2.5 px-3 py-2.5">
              {isRunning ? (
                <Loader2
                  className="w-4 h-4 animate-spin shrink-0"
                  style={{ color: modeColor }}
                />
              ) : (
                <Activity
                  className="w-4 h-4 shrink-0"
                  style={{ color: modeColor }}
                />
              )}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-[0.08em] text-[var(--text-muted)] font-semibold">
                    {currentPhase.totalSteps && currentPhase.totalSteps > 1
                      ? `Step ${currentPhase.step ?? 1} of ${currentPhase.totalSteps}`
                      : 'Current phase'}
                  </span>
                  {currentPhase.kind && currentPhase.kind !== 'work' && (
                    <span
                      className="text-[9px] uppercase tracking-[0.1em] px-1.5 py-[1px] rounded-full font-semibold"
                      style={{
                        background: `${modeColor}1f`,
                        color: modeColor,
                        border: `1px solid ${modeColor}2a`,
                      }}
                    >
                      {currentPhase.kind.replace('_', ' ')}
                    </span>
                  )}
                </div>
                <div className="text-[13px] font-medium text-[var(--text-primary)] truncate mt-0.5 tracking-tight">
                  {currentPhase.title}
                </div>
              </div>
            </div>
            {/* Multi-step stepper: only shown for pipelines with >1 phase */}
            {currentPhase.totalSteps && currentPhase.totalSteps > 1 && (
              <div
                className="flex items-stretch gap-0 px-3 pb-2.5"
                role="list"
                aria-label="Pipeline steps"
              >
                {Array.from({ length: currentPhase.totalSteps }).map((_, i) => {
                  const step = i + 1;
                  const cur = currentPhase.step ?? 1;
                  const isDone = step < cur;
                  const isActive = step === cur;
                  // Prefer the most-recent title seen for each step index.
                  const title = [...phases]
                    .reverse()
                    .find((p) => p.step === step)?.title;
                  return (
                    <div
                      key={step}
                      role="listitem"
                      className="flex-1 min-w-0"
                      style={{
                        paddingRight: step < currentPhase.totalSteps! ? 6 : 0,
                      }}
                      title={title || `Step ${step}`}
                    >
                      <div
                        className="h-[3px] rounded-full transition-all duration-500 ease-out"
                        style={{
                          background:
                            isDone || isActive
                              ? `linear-gradient(90deg, ${modeColor}, ${modeColor}cc)`
                              : 'rgba(255,255,255,0.06)',
                          boxShadow:
                            isActive && isRunning
                              ? `0 0 8px ${modeColor}66`
                              : undefined,
                          opacity: isDone ? 0.6 : 1,
                        }}
                      />
                      <div className="flex items-center gap-1 mt-1.5">
                        {isDone && (
                          <CheckCircle2
                            className="w-2.5 h-2.5 shrink-0"
                            style={{ color: modeColor, opacity: 0.7 }}
                          />
                        )}
                        {isActive && isRunning && (
                          <span
                            className="w-1.5 h-1.5 rounded-full animate-pulse shrink-0"
                            style={{
                              background: modeColor,
                              boxShadow: `0 0 4px ${modeColor}`,
                            }}
                          />
                        )}
                        <span
                          className="text-[10px] truncate tracking-tight"
                          style={{
                            color: isActive
                              ? 'var(--text-primary)'
                              : 'var(--text-muted)',
                            fontWeight: isActive ? 600 : 500,
                          }}
                        >
                          {title || `Step ${step}`}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Refresh pass monitor — appears when backend is in the download phase */}
        {isRunning && refreshPass && currentPhase?.kind === 'download' && (() => {
          const { pass, totalPasses, ok, pending } = refreshPass;
          const processedRefresh = ok + pending;
          const passFrac = totalPasses > 0 ? Math.min(1, pass / totalPasses) : 0;
          const okFrac = processedRefresh > 0 ? ok / processedRefresh : 0;
          return (
            <div
              className="rounded-xl px-3 py-2.5"
              style={{
                background:
                  'linear-gradient(180deg, rgba(96,165,250,0.07) 0%, rgba(96,165,250,0.02) 100%)',
                border: '1px solid rgba(96,165,250,0.22)',
              }}
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <Download className="w-3.5 h-3.5" style={{ color: '#60a5fa' }} />
                  <span className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-semibold">
                    Refresh pass
                  </span>
                  <span
                    className="text-[11px] font-semibold tabular-nums"
                    style={{ color: '#bfdbfe' }}
                  >
                    {pass} <span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>/ {totalPasses}</span>
                  </span>
                </div>
                <div className="flex items-center gap-3 text-[10px] tabular-nums">
                  <span style={{ color: '#10b981' }}>
                    <span style={{ fontWeight: 700 }}>{ok}</span>
                    <span style={{ opacity: 0.7 }}> ok</span>
                  </span>
                  <span style={{ color: pending > 0 ? '#fbbf24' : 'var(--text-muted)' }}>
                    <span style={{ fontWeight: 700 }}>{pending}</span>
                    <span style={{ opacity: 0.7 }}> pending</span>
                  </span>
                </div>
              </div>
              {/* Pass progress bar */}
              <div
                className="h-1 rounded-full overflow-hidden mb-1"
                style={{ background: 'rgba(96,165,250,0.12)' }}
              >
                <div
                  style={{
                    width: `${passFrac * 100}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, #60a5fa 0%, #818cf8 100%)',
                    transition: 'width 420ms cubic-bezier(0.22, 1, 0.36, 1)',
                  }}
                />
              </div>
              {/* Ok / pending split bar */}
              {processedRefresh > 0 && (
                <div
                  className="h-[3px] rounded-full overflow-hidden flex"
                  style={{ background: 'rgba(148,163,184,0.12)' }}
                >
                  <div
                    style={{
                      width: `${okFrac * 100}%`,
                      background: 'linear-gradient(90deg, #10b981 0%, #34d399 100%)',
                      transition: 'width 420ms cubic-bezier(0.22, 1, 0.36, 1)',
                    }}
                  />
                  <div
                    style={{
                      flex: 1,
                      background:
                        'repeating-linear-gradient(-45deg, rgba(251,191,36,0.55) 0 4px, rgba(251,191,36,0.22) 4px 8px)',
                    }}
                  />
                </div>
              )}
            </div>
          );
        })()}

        {/* Model mix — during tune, shows top distributions as a stacked chip bar */}
        {isRunning && topModels.length > 0 && currentPhase?.kind === 'tune' && (
          <div
            className="rounded-xl px-3 py-2.5"
            style={{
              background:
                'linear-gradient(180deg, rgba(139,92,246,0.07) 0%, rgba(139,92,246,0.02) 100%)',
              border: '1px solid rgba(139,92,246,0.22)',
            }}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Sparkles className="w-3.5 h-3.5" style={{ color: modeColor }} />
                <span className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-semibold">
                  Model mix
                </span>
                <span className="text-[11px] font-semibold tabular-nums text-[var(--text-primary)]">
                  {totalModelled}
                </span>
                <span className="text-[10px] text-[var(--text-muted)]">
                  assets fitted
                </span>
              </div>
              <span className="text-[10px] text-[var(--text-muted)] tabular-nums">
                {Object.keys(modelCounts).length} distributions
              </span>
            </div>
            {/* Stacked proportional bar */}
            <div
              className="h-1.5 rounded-full overflow-hidden flex mb-2"
              style={{ background: 'rgba(148,163,184,0.10)' }}
            >
              {topModels.map((m) => {
                const frac = totalModelled > 0 ? m.count / totalModelled : 0;
                return (
                  <div
                    key={m.name}
                    style={{
                      width: `${frac * 100}%`,
                      background: `linear-gradient(90deg, hsl(${m.hue} 72% 58%) 0%, hsl(${(m.hue + 18) % 360} 72% 66%) 100%)`,
                      transition: 'width 420ms cubic-bezier(0.22, 1, 0.36, 1)',
                    }}
                    title={`${m.name} · ${m.count} (${(frac * 100).toFixed(0)}%)`}
                  />
                );
              })}
            </div>
            {/* Chips */}
            <div className="flex flex-wrap gap-1.5">
              {topModels.map((m) => {
                const frac = totalModelled > 0 ? m.count / totalModelled : 0;
                return (
                  <div
                    key={m.name}
                    className="flex items-center gap-1.5 rounded-full px-2 py-[3px]"
                    style={{
                      background: `hsla(${m.hue}, 72%, 58%, 0.10)`,
                      border: `1px solid hsla(${m.hue}, 72%, 58%, 0.32)`,
                    }}
                  >
                    <span
                      className="w-1.5 h-1.5 rounded-full"
                      style={{ background: `hsl(${m.hue} 72% 60%)` }}
                    />
                    <span
                      className="text-[10.5px] font-medium tracking-tight"
                      style={{ color: 'var(--text-primary)' }}
                    >
                      {m.name}
                    </span>
                    <span
                      className="text-[10px] tabular-nums"
                      style={{ color: 'var(--text-muted)' }}
                    >
                      {m.count}
                      <span style={{ opacity: 0.6 }}> · {(frac * 100).toFixed(0)}%</span>
                    </span>
                  </div>
                );
              })}
            </div>
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
            <div className="grid grid-cols-[repeat(auto-fill,minmax(132px,1fr))] gap-1">
              {recentAssets.map((a, i) => {
                const model = a.model ?? modelBySymbol[a.symbol];
                return (
                  <div
                    key={`${a.symbol}-${i}`}
                    className="flex flex-col gap-[2px] px-2 py-1 rounded text-[11px] font-mono"
                    style={{
                      background:
                        a.status === 'ok'
                          ? 'rgba(16,185,129,0.06)'
                          : 'rgba(244,63,94,0.08)',
                      border: `1px solid ${a.status === 'ok' ? 'rgba(16,185,129,0.2)' : 'rgba(244,63,94,0.28)'}`,
                    }}
                    title={
                      model
                        ? `${a.symbol} · ${model}`
                        : a.detail
                        ? `${a.symbol} — ${a.detail}`
                        : a.symbol
                    }
                  >
                    <div className="flex items-center gap-1.5">
                      {a.status === 'ok' ? (
                        <CheckCircle2
                          className="w-3 h-3 shrink-0"
                          style={{ color: '#10b981' }}
                        />
                      ) : (
                        <XCircle
                          className="w-3 h-3 shrink-0"
                          style={{ color: '#f43f5e' }}
                        />
                      )}
                      <span className="text-[var(--text-primary)] font-semibold truncate">
                        {a.symbol}
                      </span>
                      {a.detail && !model && (
                        <span
                          className="text-[var(--text-muted)] ml-auto shrink-0"
                          style={{ fontSize: '10px' }}
                        >
                          {a.detail}
                        </span>
                      )}
                    </div>
                    {model && (
                      <span
                        className="truncate tracking-tight"
                        style={{
                          fontSize: '9.5px',
                          color: modeColor,
                          opacity: 0.82,
                          paddingLeft: 18,
                          letterSpacing: '-0.01em',
                        }}
                      >
                        {model}
                      </span>
                    )}
                  </div>
                );
              })}
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
              Completed · {counters.done} successful
              {counters.fail > 0 ? `, ${counters.fail} failed` : ''}
              {' '}in {formatDuration(elapsedSec)}
              {avgPerAsset !== null && processed > 1 && (
                <span style={{ opacity: 0.75 }}>
                  {' '}· {avgPerAsset.toFixed(1)}s per asset
                </span>
              )}
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
              {status === 'error' ? (errorMsg || 'Stream error') : `Failed after ${formatDuration(elapsedSec)}`}
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
  tooltip,
}: {
  label: string;
  value: number | string;
  suffix?: string;
  color: string;
  dot?: boolean;
  tooltip?: string;
}) {
  return (
    <div
      className="px-3 py-2 rounded-lg transition-colors"
      style={{
        background:
          'linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015))',
        border: '1px solid rgba(255,255,255,0.06)',
      }}
      title={tooltip}
    >
      <div className="text-[10px] uppercase tracking-[0.08em] text-[var(--text-muted)] font-semibold">
        {label}
      </div>
      <div className="flex items-baseline gap-1.5 mt-1">
        {dot && (
          <span
            className="w-1.5 h-1.5 rounded-full animate-pulse shrink-0"
            style={{ background: color, boxShadow: `0 0 6px ${color}`, alignSelf: 'center' }}
          />
        )}
        <span
          className="text-[17px] font-semibold tabular-nums leading-none tracking-tight"
          style={{ color, fontVariantNumeric: 'tabular-nums' }}
        >
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
