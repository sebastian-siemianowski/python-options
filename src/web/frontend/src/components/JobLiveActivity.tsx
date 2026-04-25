import { useEffect, useMemo, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import {
  Activity,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  Download,
  Loader2,
  Minus,
  Sparkles,
  Square,
  X,
  XCircle,
} from 'lucide-react';
import {
  formatJobDuration,
  formatJobElapsed,
  JOB_MODE_LABELS,
  useJobStore,
  type JobAssetEntry,
  type JobLogLine,
  type JobModelMeta,
  type JobMode,
  type JobRefreshPassState,
  type JobStatus,
} from '../stores/jobStore';

const SIGNAL_QUERY_KEYS = [
  ['signalSummary'],
  ['signalStats'],
  ['signalsBySector'],
  ['strongSignals'],
  ['highConvictionBuy'],
  ['highConvictionSell'],
  ['tuneList'],
  ['tuneStats'],
] as const;

const EMPTY_ASSETS: JobAssetEntry[] = [];
const EMPTY_LOGS: JobLogLine[] = [];
const EMPTY_MODEL_COUNTS: Record<string, number> = {};
const EMPTY_MODEL_BY_SYMBOL: Record<string, string> = {};
const EMPTY_MODEL_META_BY_SYMBOL: Record<string, JobModelMeta> = {};

function statusTone(status: JobStatus) {
  switch (status) {
    case 'running': return { label: 'Running', color: '#60a5fa', bg: 'rgba(96,165,250,0.13)' };
    case 'completed': return { label: 'Complete', color: '#10b981', bg: 'rgba(16,185,129,0.13)' };
    case 'failed': return { label: 'Failed', color: '#f43f5e', bg: 'rgba(244,63,94,0.13)' };
    case 'error': return { label: 'Needs attention', color: '#f59e0b', bg: 'rgba(245,158,11,0.14)' };
    case 'stopped': return { label: 'Stopped', color: '#94a3b8', bg: 'rgba(148,163,184,0.12)' };
    default: return { label: 'Idle', color: '#94a3b8', bg: 'rgba(148,163,184,0.10)' };
  }
}

function hashHue(value: string) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) hash = (hash * 31 + value.charCodeAt(i)) >>> 0;
  return hash % 360;
}

function compactMetric(value: number) {
  return Math.abs(value) >= 1000 ? `${Math.round(value / 100) / 10}k` : `${value}`;
}

function fixedMetric(value: number, digits: number) {
  return Number.isFinite(value) ? value.toFixed(digits) : '—';
}

function MetricPill({ label, value, hue }: { label: string; value: string; hue: number }) {
  return (
    <span className="rounded-full px-1.5 py-0.5 text-[9px] font-semibold tabular-nums" style={{ color: `hsl(${hue} 78% 82%)`, background: `hsla(${hue},72%,58%,0.12)`, border: `1px solid hsla(${hue},72%,62%,0.22)` }}>
      <span className="opacity-60">{label}</span> {value}
    </span>
  );
}

function titleForMode(mode: JobMode | null) {
  if (!mode) return JOB_MODE_LABELS.retune;
  return JOB_MODE_LABELS[mode];
}

export default function JobLiveActivity() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const mode = useJobStore((state) => state.mode);
  const status = useJobStore((state) => state.status);
  const phases = useJobStore((state) => state.phases);
  const counters = useJobStore((state) => state.counters);
  const elapsedSec = useJobStore((state) => state.elapsedSec);
  const errorMsg = useJobStore((state) => state.errorMsg);
  const surfaceVisible = useJobStore((state) => state.surfaceVisible);
  const expanded = useJobStore((state) => state.expanded);
  const rawLogOpen = useJobStore((state) => state.rawLogOpen);
  const refreshPass = useJobStore((state) => state.refreshPass);
  const stopJob = useJobStore((state) => state.stopJob);
  const setExpanded = useJobStore((state) => state.setExpanded);
  const toggleExpanded = useJobStore((state) => state.toggleExpanded);
  const setRawLogOpen = useJobStore((state) => state.setRawLogOpen);
  const clearTerminalJob = useJobStore((state) => state.clearTerminalJob);
  const assets = useJobStore((state) => state.expanded ? state.assets : EMPTY_ASSETS);
  const modelBySymbol = useJobStore((state) => state.expanded ? state.modelBySymbol : EMPTY_MODEL_BY_SYMBOL);
  const modelMetaBySymbol = useJobStore((state) => state.expanded ? state.modelMetaBySymbol : EMPTY_MODEL_META_BY_SYMBOL);
  const modelCounts = useJobStore((state) => state.expanded ? state.modelCounts : EMPTY_MODEL_COUNTS);
  const logLines = useJobStore((state) => state.rawLogOpen ? state.logLines : EMPTY_LOGS);
  const prevStatusRef = useRef<JobStatus>(status);
  const rawLogRef = useRef<HTMLDivElement | null>(null);

  const info = titleForMode(mode);
  const tone = statusTone(status);
  const isRunning = status === 'running';
  const processed = counters.done + counters.fail;
  const progressPct = counters.total > 0
    ? Math.min(100, (processed / counters.total) * 100)
    : isRunning ? 3 : status === 'idle' ? 0 : 100;
  const rate = processed > 0 ? processed / Math.max(elapsedSec, 1) : 0;
  const successRate = processed > 0 ? Math.round((counters.done / processed) * 100) : null;
  const eta = isRunning && rate > 0 && counters.total > processed
    ? Math.round((counters.total - processed) / rate)
    : null;
  const currentPhase = phases.length > 0 ? phases[phases.length - 1] : null;
  const liveTitle = mode === 'retune' || mode === 'tune' || mode === 'calibrate'
    ? 'Live Tune Activity'
    : 'Live Market Refresh';
  const liveSubtitle = isRunning
    ? 'Running quietly in the background — keep exploring Signals.'
    : status === 'completed'
      ? 'Finished cleanly. Fresh data is being reflected across the dashboard.'
      : status === 'stopped'
        ? 'Stopped safely. No blocking overlay, no lost context.'
        : status === 'failed' || status === 'error'
          ? 'The job needs attention. Raw logs are available below.'
          : info.desc;
  const recentAssets = useMemo(() => expanded ? assets.slice(-28).reverse() : EMPTY_ASSETS, [assets, expanded]);
  const topModels = useMemo(() => {
    if (!expanded) return [];
    return Object.entries(modelCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([name, count]) => ({ name, count, hue: hashHue(name) }));
  }, [modelCounts, expanded]);
  const totalModelled = useMemo(() => expanded ? Object.values(modelCounts).reduce((sum, value) => sum + value, 0) : 0, [modelCounts, expanded]);
  const recentModelWinners = useMemo(() => {
    if (!expanded) return [];
    return Object.entries(modelMetaBySymbol)
      .sort((a, b) => b[1].updatedAt - a[1].updatedAt)
      .slice(0, 12)
      .map(([symbol, meta]) => ({ symbol, meta, hue: hashHue(`${symbol}:${meta.model}`) }));
  }, [modelMetaBySymbol, expanded]);
  const fallbackModelWinners = useMemo(() => {
    if (!expanded || recentModelWinners.length > 0) return [];
    return Object.entries(modelBySymbol)
      .slice(-12)
      .reverse()
      .map(([symbol, model]) => ({ symbol, model, hue: hashHue(`${symbol}:${model}`) }));
  }, [modelBySymbol, expanded, recentModelWinners.length]);

  useEffect(() => {
    if (prevStatusRef.current === 'running' && status === 'completed') {
      SIGNAL_QUERY_KEYS.forEach((queryKey) => {
        queryClient.invalidateQueries({ queryKey: [...queryKey] });
      });
    }
    prevStatusRef.current = status;
  }, [status, queryClient]);

  useEffect(() => {
    if (rawLogOpen && rawLogRef.current) {
      rawLogRef.current.scrollTop = rawLogRef.current.scrollHeight;
    }
  }, [logLines, rawLogOpen]);

  if (!surfaceVisible || status === 'idle' || !mode) return null;

  const job = {
    mode,
    status,
    phases,
    assets,
    logLines,
    counters,
    elapsedSec,
    errorMsg,
    refreshPass: refreshPass as JobRefreshPassState | null,
    modelBySymbol,
    modelMetaBySymbol,
    modelCounts,
    surfaceVisible,
    expanded,
    rawLogOpen,
    stopJob,
    setExpanded,
    toggleExpanded,
    setRawLogOpen,
    clearTerminalJob,
  };

  return (
    <div className="fixed bottom-5 right-5 z-50 pointer-events-none max-w-[calc(100vw-2rem)]">
      <style>{`
        @keyframes liveActivityGlow {
          0%, 100% { transform: translate3d(0,0,0) scale(1); opacity: 0.78; }
          50% { transform: translate3d(10px,-8px,0) scale(1.08); opacity: 1; }
        }
        @keyframes liveActivityRise {
          from { opacity: 0; transform: translateY(12px) scale(0.98); filter: blur(4px); }
          to { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
        }
        @keyframes liveActivityHalo {
          0%, 100% { opacity: 0.55; transform: translate3d(-10px,0,0) scale(1); }
          50% { opacity: 1; transform: translate3d(12px,-8px,0) scale(1.08); }
        }
        @media (prefers-reduced-motion: reduce) {
          .job-live-activity-anim, .job-live-activity-anim * { animation: none !important; transition-duration: 0.01ms !important; }
        }
      `}</style>

      {job.expanded && (
        <div
          className="job-live-activity-anim pointer-events-auto mb-3 w-[min(760px,calc(100vw-2rem))] max-h-[min(78vh,820px)] overflow-hidden rounded-[28px]"
          style={{
            animation: 'liveActivityRise 260ms cubic-bezier(.2,.8,.2,1) both',
            background: 'linear-gradient(180deg, rgba(18,20,33,0.86) 0%, rgba(8,9,17,0.94) 100%)',
            border: `1px solid ${info.color}38`,
            boxShadow: `0 28px 90px -28px ${info.color}66, 0 24px 80px -40px rgba(0,0,0,0.95), inset 0 1px 0 rgba(255,255,255,0.09)`,
            backdropFilter: 'blur(28px) saturate(1.35)',
            WebkitBackdropFilter: 'blur(28px) saturate(1.35)',
          }}
          role="region"
          aria-label="Live job progress"
        >
          <div className="relative overflow-hidden">
            <div
              aria-hidden
              className="absolute -top-24 -right-20 h-56 w-56 rounded-full job-live-activity-anim"
              style={{
                background: `radial-gradient(circle, ${info.color}42 0%, ${info.color}10 38%, transparent 72%)`,
                filter: 'blur(4px)',
                animation: 'liveActivityGlow 5.5s ease-in-out infinite',
              }}
            />
            <div
              aria-hidden
              className="absolute -bottom-24 left-8 h-44 w-72 rounded-full job-live-activity-anim"
              style={{
                background: 'radial-gradient(ellipse, rgba(56,217,245,0.12), rgba(139,92,246,0.06) 46%, transparent 72%)',
                filter: 'blur(10px)',
                animation: 'liveActivityHalo 7s ease-in-out infinite',
              }}
            />
            <div className="relative flex items-start justify-between gap-5 px-5 py-4" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
              <div className="flex items-start gap-3 min-w-0">
                <div
                  className="relative h-11 w-11 rounded-[15px] flex items-center justify-center shrink-0"
                  style={{
                    background: `linear-gradient(145deg, ${info.color}2e, rgba(255,255,255,0.035))`,
                    border: `1px solid ${info.color}3d`,
                    boxShadow: isRunning ? `0 0 24px -4px ${info.color}aa, inset 0 1px 0 rgba(255,255,255,0.12)` : 'inset 0 1px 0 rgba(255,255,255,0.1)',
                    color: info.color,
                  }}
                >
                  {isRunning ? <Loader2 className="w-5 h-5 animate-spin" /> : job.status === 'completed' ? <CheckCircle2 className="w-5 h-5" /> : job.status === 'failed' || job.status === 'error' ? <XCircle className="w-5 h-5" /> : <Activity className="w-5 h-5" />}
                  {isRunning && <span className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full pulse-dot" style={{ background: info.color, boxShadow: `0 0 10px ${info.color}` }} />}
                </div>
                <div className="min-w-0">
                  <div className="label-micro mb-1" style={{ color: info.color }}>Live Activity</div>
                  <div className="text-[22px] font-semibold tracking-[-0.04em] text-white leading-tight">{liveTitle}</div>
                  <div className="mt-1 text-[12px] text-[var(--text-muted)] truncate max-w-[440px]">
                    {currentPhase?.title ?? info.desc}
                    {eta !== null && <span className="text-[var(--text-secondary)]"> · ETA {formatJobElapsed(eta)}</span>}
                  </div>
                  <div className="mt-1.5 text-[11px] text-[var(--text-secondary)] truncate max-w-[480px]">
                    {liveSubtitle}
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <StatusPill label={tone.label} color={tone.color} bg={tone.bg} pulse={isRunning} />
                {isRunning && (
                  <button
                    type="button"
                    onClick={job.stopJob}
                    className="inline-flex h-8 items-center gap-1.5 rounded-full px-3 text-[11px] font-semibold transition-all hover:brightness-110 active:scale-[0.98]"
                    style={{ background: 'linear-gradient(180deg,#fb7185,#e11d48)', color: 'white', boxShadow: '0 8px 22px -10px rgba(244,63,94,0.85)' }}
                    aria-label="Stop running job"
                  >
                    <Square className="h-3 w-3" fill="currentColor" />
                    Stop
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => job.setExpanded(false)}
                  className="inline-flex h-8 w-8 items-center justify-center rounded-full transition-colors hover:bg-white/[0.08]"
                  style={{ color: 'var(--text-secondary)', border: '1px solid rgba(255,255,255,0.08)' }}
                  aria-label="Collapse live activity"
                >
                  <Minus className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>

            <div className="relative h-2 overflow-hidden rounded-full mx-5 mb-4 p-[1px]" style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.13), rgba(255,255,255,0.035))', boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.35)' }}>
              <div
                className="relative h-full rounded-full transition-[width] duration-700 ease-out"
                style={{
                  width: `${progressPct}%`,
                  background: job.status === 'failed' || job.status === 'error'
                    ? 'linear-gradient(90deg,#f43f5e,#fb7185,#fecdd3)'
                    : `linear-gradient(90deg, ${info.color} 0%, #38d9f5 58%, rgba(255,255,255,0.92) 100%)`,
                  boxShadow: `0 0 24px -8px ${info.color}, inset 0 1px 0 rgba(255,255,255,0.34)`,
                }}
              >
                <span aria-hidden className="absolute inset-x-1 top-0 h-px rounded-full" style={{ background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.72), transparent)' }} />
                {isRunning && progressPct > 2 && <span aria-hidden className="absolute right-0 top-1/2 h-3 w-3 -translate-y-1/2 translate-x-1/2 rounded-full" style={{ background: '#ffffff', boxShadow: `0 0 18px 4px ${info.color}88` }} />}
              </div>
            </div>
          </div>

          <div className="max-h-[calc(min(78vh,820px)-116px)] overflow-y-auto p-5 space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5" aria-live="polite">
              <MetricTile label="Processed" value={processed} suffix={job.counters.total > 0 ? `/ ${job.counters.total}` : undefined} color="#ffffff" />
              <MetricTile label="Successful" value={job.counters.done} suffix={successRate !== null ? `${successRate}%` : undefined} color="#10b981" pulse={isRunning && job.counters.done > 0} />
              <MetricTile label="Failed" value={job.counters.fail} color={job.counters.fail > 0 ? '#f43f5e' : 'var(--text-muted)'} />
              <MetricTile label={rate > 0 ? 'Throughput' : 'Progress'} value={rate > 0 ? (rate * 60).toFixed(1) : `${Math.round(progressPct)}%`} suffix={rate > 0 ? '/ min' : undefined} color={info.color} />
            </div>

            <section className="rounded-[18px] p-4" style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015))', border: '1px solid rgba(255,255,255,0.075)', boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.055)' }}>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-[10px] uppercase tracking-[0.13em] text-[var(--text-muted)] font-semibold">Non-blocking promise</div>
                  <div className="mt-1 text-[13px] font-medium tracking-[-0.02em] text-[var(--text-primary)]">
                    {isRunning ? 'Tune is running. Signals, filters, charts, and watchlists stay responsive.' : 'The job lifecycle is complete and your workspace is still exactly where you left it.'}
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2 text-[10.5px] font-semibold">
                  <span className="rounded-full px-2.5 py-1" style={{ color: '#a7f3d0', background: 'rgba(16,185,129,0.09)', border: '1px solid rgba(16,185,129,0.22)' }}>Background stream</span>
                  <span className="rounded-full px-2.5 py-1" style={{ color: '#bfdbfe', background: 'rgba(96,165,250,0.09)', border: '1px solid rgba(96,165,250,0.22)' }}>Live cache refresh</span>
                  <span className="rounded-full px-2.5 py-1" style={{ color: '#ddd6fe', background: 'rgba(139,92,246,0.10)', border: '1px solid rgba(139,92,246,0.24)' }}>Safe cancel</span>
                </div>
              </div>
            </section>

            {currentPhase && (
              <section className="rounded-[18px] p-4" style={{ background: `linear-gradient(180deg, ${info.color}12, ${info.color}05)`, border: `1px solid ${info.color}26` }}>
                <div className="flex items-center gap-3">
                  {isRunning ? <Loader2 className="h-4 w-4 animate-spin shrink-0" style={{ color: info.color }} /> : <Activity className="h-4 w-4 shrink-0" style={{ color: info.color }} />}
                  <div className="min-w-0 flex-1">
                    <div className="text-[10px] uppercase tracking-[0.11em] font-semibold text-[var(--text-muted)]">
                      {currentPhase.totalSteps && currentPhase.totalSteps > 1 ? `Step ${currentPhase.step ?? 1} of ${currentPhase.totalSteps}` : 'Current phase'}
                    </div>
                    <div className="mt-0.5 truncate text-[14px] font-medium tracking-[-0.02em] text-[var(--text-primary)]">{currentPhase.title}</div>
                  </div>
                </div>
                {currentPhase.totalSteps && currentPhase.totalSteps > 1 && (
                  <div className="mt-4 flex gap-2" role="list" aria-label="Pipeline steps">
                    {Array.from({ length: currentPhase.totalSteps }).map((_, index) => {
                      const step = index + 1;
                      const current = currentPhase.step ?? 1;
                      const title = [...job.phases].reverse().find((phase) => phase.step === step)?.title ?? `Step ${step}`;
                      const active = step === current;
                      const done = step < current;
                      return (
                        <div key={step} className="min-w-0 flex-1" role="listitem" title={title}>
                          <div className="h-1 rounded-full" style={{ background: active || done ? `linear-gradient(90deg, ${info.color}, #38d9f5)` : 'rgba(255,255,255,0.075)', opacity: done ? 0.62 : 1 }} />
                          <div className="mt-1.5 flex items-center gap-1.5 min-w-0">
                            {done && <CheckCircle2 className="h-3 w-3 shrink-0" style={{ color: info.color, opacity: 0.7 }} />}
                            {active && isRunning && <span className="h-1.5 w-1.5 rounded-full animate-pulse shrink-0" style={{ background: info.color, boxShadow: `0 0 6px ${info.color}` }} />}
                            <span className="truncate text-[10px]" style={{ color: active ? 'var(--text-primary)' : 'var(--text-muted)', fontWeight: active ? 650 : 500 }}>{title}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </section>
            )}

            {isRunning && job.refreshPass && currentPhase?.kind === 'download' && (
              <RefreshPassCard pass={job.refreshPass.pass} totalPasses={job.refreshPass.totalPasses} ok={job.refreshPass.ok} pending={job.refreshPass.pending} />
            )}

            {topModels.length > 0 && (
              <section className="rounded-[18px] p-4" style={{ background: 'linear-gradient(180deg, rgba(139,92,246,0.10), rgba(139,92,246,0.025))', border: '1px solid rgba(139,92,246,0.24)' }}>
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <Sparkles className="h-4 w-4" style={{ color: info.color }} />
                    <span className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-muted)] font-semibold">Model Mix</span>
                    <span className="text-[12px] font-semibold text-[var(--text-primary)] tabular-nums">{totalModelled}</span>
                    <span className="text-[11px] text-[var(--text-muted)]">assets fitted</span>
                  </div>
                  <span className="text-[10px] text-[var(--text-muted)] tabular-nums">{Object.keys(job.modelCounts).length} distributions</span>
                </div>
                <div className="mb-3 flex h-2 overflow-hidden rounded-full" style={{ background: 'rgba(148,163,184,0.12)' }}>
                  {topModels.map((model) => {
                    const pct = totalModelled > 0 ? (model.count / totalModelled) * 100 : 0;
                    return <div key={model.name} title={`${model.name} · ${model.count}`} style={{ width: `${pct}%`, background: `linear-gradient(90deg, hsl(${model.hue} 72% 58%), hsl(${(model.hue + 22) % 360} 72% 66%))` }} />;
                  })}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {topModels.map((model) => (
                    <span key={model.name} className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10.5px]" style={{ background: `hsla(${model.hue},72%,58%,0.11)`, border: `1px solid hsla(${model.hue},72%,58%,0.32)`, color: 'var(--text-primary)' }}>
                      <span className="h-1.5 w-1.5 rounded-full" style={{ background: `hsl(${model.hue} 72% 62%)` }} />
                      <span className="max-w-[170px] truncate">{model.name}</span>
                      <span className="text-[var(--text-muted)] tabular-nums">{model.count}</span>
                    </span>
                  ))}
                </div>
              </section>
            )}

            {(recentModelWinners.length > 0 || fallbackModelWinners.length > 0) && (
              <section className="rounded-[20px] p-4" style={{ background: 'radial-gradient(520px 180px at 10% -20%, rgba(167,139,250,0.14), transparent 58%), linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015))', border: '1px solid rgba(255,255,255,0.085)', boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.06)' }}>
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 min-w-0">
                    <Sparkles className="h-4 w-4 shrink-0" style={{ color: info.color }} />
                    <div className="min-w-0">
                      <div className="text-[10px] uppercase tracking-[0.13em] text-[var(--text-muted)] font-semibold">Winning models</div>
                      <div className="mt-0.5 truncate text-[12px] text-[var(--text-secondary)]">Latest stock → champion distribution</div>
                    </div>
                  </div>
                  <span className="rounded-full px-2.5 py-1 text-[10px] font-semibold tabular-nums" style={{ color: '#ddd6fe', background: 'rgba(139,92,246,0.12)', border: '1px solid rgba(139,92,246,0.24)' }}>
                    {recentModelWinners.length || fallbackModelWinners.length} latest
                  </span>
                </div>
                <div className="grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-2">
                  {recentModelWinners.map((winner) => (
                    <div
                      key={`${winner.symbol}-${winner.meta.model}`}
                      className="group relative overflow-hidden rounded-[16px] px-3 py-2.5"
                      style={{
                        background: `linear-gradient(150deg, hsla(${winner.hue},72%,58%,0.13), rgba(255,255,255,0.025))`,
                        border: `1px solid hsla(${winner.hue},72%,62%,0.30)`,
                        boxShadow: `0 12px 30px -26px hsla(${winner.hue},72%,58%,0.95), inset 0 1px 0 rgba(255,255,255,0.055)`,
                      }}
                      title={`${winner.symbol} winner: ${winner.meta.model}`}
                    >
                      <div aria-hidden className="absolute inset-x-3 top-0 h-px" style={{ background: `linear-gradient(90deg, transparent, hsla(${winner.hue},72%,70%,0.75), transparent)` }} />
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-mono text-[12px] font-semibold tracking-[-0.02em] text-white truncate">{winner.symbol}</span>
                        <span className="rounded-full px-1.5 py-0.5 text-[8.5px] font-bold uppercase tracking-[0.09em]" style={{ color: `hsl(${winner.hue} 80% 78%)`, background: `hsla(${winner.hue},72%,58%,0.16)`, border: `1px solid hsla(${winner.hue},72%,62%,0.32)` }}>Winner</span>
                      </div>
                      <div className="mt-1.5 truncate text-[11px] font-medium tracking-[-0.01em]" style={{ color: `hsl(${winner.hue} 80% 78%)` }}>
                        {winner.meta.model}
                      </div>
                      {(winner.meta.weightPct != null || winner.meta.bic != null || winner.meta.hyv != null || winner.meta.crps != null || winner.meta.pitP != null) && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {winner.meta.weightPct != null && <MetricPill label="W" value={`${winner.meta.weightPct}%`} hue={winner.hue} />}
                          {winner.meta.bic != null && <MetricPill label="BIC" value={compactMetric(winner.meta.bic)} hue={winner.hue} />}
                          {winner.meta.hyv != null && <MetricPill label="Hyv" value={compactMetric(winner.meta.hyv)} hue={winner.hue} />}
                          {winner.meta.crps != null && <MetricPill label="CRPS" value={fixedMetric(winner.meta.crps, 4)} hue={winner.hue} />}
                          {winner.meta.pitP != null && <MetricPill label="PIT" value={fixedMetric(winner.meta.pitP, 3)} hue={winner.hue} />}
                        </div>
                      )}
                    </div>
                  ))}
                  {fallbackModelWinners.map((winner) => (
                    <div key={`${winner.symbol}-${winner.model}`} className="rounded-[16px] px-3 py-2.5" style={{ background: `linear-gradient(150deg, hsla(${winner.hue},72%,58%,0.10), rgba(255,255,255,0.025))`, border: `1px solid hsla(${winner.hue},72%,62%,0.24)` }} title={`${winner.symbol} winner: ${winner.model}`}>
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-mono text-[12px] font-semibold text-white truncate">{winner.symbol}</span>
                        <span className="rounded-full px-1.5 py-0.5 text-[8.5px] font-bold uppercase tracking-[0.09em]" style={{ color: `hsl(${winner.hue} 80% 78%)`, background: `hsla(${winner.hue},72%,58%,0.16)` }}>Winner</span>
                      </div>
                      <div className="mt-1.5 truncate text-[11px] font-medium" style={{ color: `hsl(${winner.hue} 80% 78%)` }}>{winner.model}</div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {recentAssets.length > 0 && (
              <section>
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-muted)] font-semibold">Recent completions</span>
                  <span className="text-[10px] text-[var(--text-muted)] tabular-nums">showing {recentAssets.length} of {job.assets.length}</span>
                </div>
                <div className="grid grid-cols-[repeat(auto-fill,minmax(138px,1fr))] gap-1.5">
                  {recentAssets.map((asset, index) => {
                    const meta = job.modelMetaBySymbol[asset.symbol];
                    const model = asset.model ?? meta?.model ?? job.modelBySymbol[asset.symbol];
                    const ok = asset.status === 'ok';
                    const hue = hashHue(`${asset.symbol}:${model ?? asset.detail ?? asset.status}`);
                    const weightPct = asset.weightPct ?? meta?.weightPct;
                    const bic = asset.bic ?? meta?.bic;
                    const hyv = asset.hyv ?? meta?.hyv;
                    const crps = asset.crps ?? meta?.crps;
                    const pitP = asset.pitP ?? meta?.pitP;
                    return (
                      <div key={`${asset.symbol}-${index}`} className="rounded-[16px] px-3 py-2.5 font-mono" style={{ background: ok ? `linear-gradient(150deg, hsla(${hue},70%,58%,0.13), rgba(255,255,255,0.026))` : 'rgba(244,63,94,0.08)', border: `1px solid ${ok ? `hsla(${hue},70%,62%,0.28)` : 'rgba(244,63,94,0.28)'}`, boxShadow: ok ? `0 14px 32px -28px hsla(${hue},70%,58%,0.95), inset 0 1px 0 rgba(255,255,255,0.045)` : undefined }} title={model ? `${asset.symbol} · winner: ${model}` : asset.detail ?? asset.symbol}>
                        <div className="flex items-center gap-1.5 min-w-0">
                          {ok ? <CheckCircle2 className="h-3 w-3 shrink-0" style={{ color: '#10b981' }} /> : <XCircle className="h-3 w-3 shrink-0" style={{ color: '#f43f5e' }} />}
                          <span className="truncate text-[11px] font-semibold text-[var(--text-primary)]">{asset.symbol}</span>
                          {model && <span className="ml-auto rounded px-1 text-[8px] uppercase tracking-[0.08em]" style={{ color: `hsl(${hue} 80% 78%)`, background: `hsla(${hue},70%,58%,0.13)` }}>win</span>}
                        </div>
                        {model && <div className="mt-1 truncate pl-[18px] text-[10px] font-medium tracking-[-0.01em]" style={{ color: `hsl(${hue} 80% 78%)`, opacity: 0.96 }}>{model}</div>}
                        {(weightPct != null || bic != null || hyv != null || crps != null || pitP != null) && (
                          <div className="mt-2 flex flex-wrap gap-1 pl-[18px]">
                            {weightPct != null && <MetricPill label="W" value={`${weightPct}%`} hue={hue} />}
                            {bic != null && <MetricPill label="BIC" value={compactMetric(bic)} hue={hue} />}
                            {hyv != null && <MetricPill label="Hyv" value={compactMetric(hyv)} hue={hue} />}
                            {crps != null && <MetricPill label="CRPS" value={fixedMetric(crps, 4)} hue={hue} />}
                            {pitP != null && <MetricPill label="PIT" value={fixedMetric(pitP, 3)} hue={hue} />}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </section>
            )}

            {!isRunning && job.status === 'completed' && (
              <TerminalBanner color="#10b981" icon={<CheckCircle2 className="h-4 w-4" />}>
                Completed · {job.counters.done} successful{job.counters.fail > 0 ? `, ${job.counters.fail} failed` : ''} in {formatJobDuration(job.elapsedSec)}
              </TerminalBanner>
            )}
            {!isRunning && (job.status === 'failed' || job.status === 'error' || job.status === 'stopped') && (
              <TerminalBanner color={job.status === 'stopped' ? '#94a3b8' : '#f43f5e'} icon={<XCircle className="h-4 w-4" />}>
                {job.errorMsg ?? (job.status === 'stopped' ? 'Stopped by user' : 'Job did not complete')}
              </TerminalBanner>
            )}

            {!isRunning && (
              <div className="flex flex-wrap items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => {
                    job.clearTerminalJob();
                    navigate('/tuning');
                  }}
                  className="inline-flex items-center gap-2 rounded-full px-3.5 py-2 text-[12px] font-semibold transition-all hover:-translate-y-0.5 active:scale-[0.98]"
                  style={{ color: '#ddd6fe', background: 'rgba(139,92,246,0.12)', border: '1px solid rgba(139,92,246,0.28)' }}
                >
                  Open tuning dashboard
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>
                <button
                  type="button"
                  onClick={job.clearTerminalJob}
                  className="inline-flex items-center gap-2 rounded-full px-3.5 py-2 text-[12px] font-semibold transition-all hover:bg-white/[0.08] active:scale-[0.98]"
                  style={{ color: 'var(--text-secondary)', border: '1px solid rgba(255,255,255,0.08)' }}
                >
                  Dismiss
                </button>
              </div>
            )}

            <div>
              <button
                type="button"
                onClick={() => job.setRawLogOpen(!job.rawLogOpen)}
                className="inline-flex items-center gap-1.5 text-[11px] text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
              >
                {job.rawLogOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                Raw log ({job.logLines.length})
              </button>
              {job.rawLogOpen && (
                <div ref={rawLogRef} className="mt-2 max-h-[260px] overflow-auto rounded-2xl px-3 py-2 font-mono text-[11px] leading-[1.5]" style={{ background: 'rgba(3,4,8,0.9)', color: 'rgba(255,255,255,0.62)', border: '1px solid rgba(255,255,255,0.07)', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                  {job.logLines.length === 0 ? <span className="text-[var(--text-muted)]">(no subprocess output yet)</span> : job.logLines.map((line) => <div key={line.id}>{line.text}</div>)}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <div
        className="job-live-activity-anim pointer-events-auto w-[min(372px,calc(100vw-2rem))] overflow-hidden rounded-[26px]"
        style={{
          animation: 'liveActivityRise 220ms cubic-bezier(.2,.8,.2,1) both',
          background: 'linear-gradient(180deg, rgba(18,20,33,0.76) 0%, rgba(7,8,15,0.88) 100%)',
          border: `1px solid ${info.color}2f`,
          boxShadow: `0 18px 56px -28px ${info.color}80, 0 16px 48px -34px rgba(0,0,0,0.95), inset 0 1px 0 rgba(255,255,255,0.11)`,
          backdropFilter: 'blur(30px) saturate(1.45)',
          WebkitBackdropFilter: 'blur(30px) saturate(1.45)',
        }}
        aria-live="polite"
      >
        <div className="relative px-3.5 py-3">
          <div aria-hidden className="absolute inset-x-5 top-0 h-px" style={{ background: `linear-gradient(90deg, transparent, ${info.color}8c, transparent)` }} />
          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={job.toggleExpanded}
              className="relative h-9 w-9 rounded-[14px] flex items-center justify-center shrink-0 transition-transform hover:scale-[1.03] active:scale-95"
              style={{ background: `linear-gradient(145deg, ${info.color}2a, rgba(255,255,255,0.03))`, border: `1px solid ${info.color}36`, color: info.color, boxShadow: isRunning ? `0 0 18px -6px ${info.color}` : undefined }}
              aria-label="Expand live job progress"
            >
              {isRunning ? <Loader2 className="h-4.5 w-4.5 animate-spin" /> : job.status === 'completed' ? <CheckCircle2 className="h-4.5 w-4.5" /> : job.status === 'failed' || job.status === 'error' ? <XCircle className="h-4.5 w-4.5" /> : <Activity className="h-4.5 w-4.5" />}
              {isRunning && <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full pulse-dot" style={{ background: info.color, boxShadow: `0 0 8px ${info.color}` }} />}
            </button>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="text-[12.5px] font-semibold tracking-[-0.025em] text-white truncate">
                  {isRunning ? currentPhase?.title ?? info.title : info.shortTitle}
                </span>
                <StatusPill label={tone.label} color={tone.color} bg={tone.bg} pulse={isRunning} compact />
              </div>
              <div className="mt-0.5 flex items-center gap-2 text-[10.5px] text-[var(--text-muted)]">
                <span className="truncate">
                  {isRunning
                    ? eta !== null ? `ETA ${formatJobElapsed(eta)}` : 'Running in background'
                    : liveSubtitle}
                </span>
                {isRunning && counters.total > 0 && (
                  <span className="shrink-0 font-mono tabular-nums text-[var(--text-secondary)]">{Math.round(progressPct)}%</span>
                )}
              </div>
              <div className="mt-2 flex items-center gap-2">
                <div className="h-2 flex-1 overflow-hidden rounded-full p-[1px]" style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.13), rgba(255,255,255,0.04))', boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.35)' }}>
                  <div className="relative h-full rounded-full transition-[width] duration-700" style={{ width: `${progressPct}%`, background: `linear-gradient(90deg, ${info.color} 0%, #38d9f5 62%, rgba(255,255,255,0.88) 100%)`, boxShadow: `0 0 16px -7px ${info.color}, inset 0 1px 0 rgba(255,255,255,0.35)` }}>
                    <span aria-hidden className="absolute inset-x-1 top-0 h-px rounded-full" style={{ background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.75), transparent)' }} />
                  </div>
                </div>
                <span className="text-[10px] font-mono text-[var(--text-secondary)] tabular-nums min-w-[74px] text-right">
                  {processed}{job.counters.total > 0 ? ` / ${job.counters.total}` : ''}
                </span>
              </div>
            </div>
            <div className="flex flex-col items-end gap-1 shrink-0">
              <span className="text-[11px] font-mono font-semibold tabular-nums text-[var(--text-primary)]" title={`Elapsed ${formatJobDuration(job.elapsedSec)}`}>{formatJobElapsed(job.elapsedSec)}</span>
              <div className="flex items-center gap-1">
                <button type="button" onClick={job.toggleExpanded} className="inline-flex h-6 w-6 items-center justify-center rounded-full hover:bg-white/[0.08]" style={{ color: 'var(--text-secondary)' }} aria-label={job.expanded ? 'Collapse job details' : 'Expand job details'}>
                  {job.expanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronUp className="h-3.5 w-3.5" />}
                </button>
                {isRunning ? (
                  <button type="button" onClick={job.stopJob} className="inline-flex h-6 w-6 items-center justify-center rounded-full hover:bg-red-500/15" style={{ color: '#fb7185' }} aria-label="Stop running job">
                    <Square className="h-3 w-3" fill="currentColor" />
                  </button>
                ) : (
                  <button type="button" onClick={job.clearTerminalJob} className="inline-flex h-6 w-6 items-center justify-center rounded-full hover:bg-white/[0.08]" style={{ color: 'var(--text-muted)' }} aria-label="Dismiss completed job">
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatusPill({ label, color, bg, pulse, compact }: { label: string; color: string; bg: string; pulse?: boolean; compact?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full font-semibold uppercase tracking-[0.08em]" style={{ background: bg, color, border: `1px solid ${color}35`, padding: compact ? '2px 6px' : '4px 9px', fontSize: compact ? 8.5 : 10 }}>
      {pulse && <span className="h-1.5 w-1.5 rounded-full animate-pulse" style={{ background: color, boxShadow: `0 0 6px ${color}` }} />}
      {label}
    </span>
  );
}

function MetricTile({ label, value, suffix, color, pulse }: { label: string; value: number | string; suffix?: string; color: string; pulse?: boolean }) {
  return (
    <div className="rounded-[16px] px-3 py-3" style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.018))', border: '1px solid rgba(255,255,255,0.07)', boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.05)' }}>
      <div className="text-[10px] uppercase tracking-[0.09em] text-[var(--text-muted)] font-semibold">{label}</div>
      <div className="mt-2 flex items-baseline gap-1.5">
        {pulse && <span className="h-1.5 w-1.5 rounded-full animate-pulse" style={{ background: color, boxShadow: `0 0 6px ${color}` }} />}
        <span className="text-[21px] leading-none font-semibold tracking-[-0.03em] tabular-nums" style={{ color }}>{value}</span>
        {suffix && <span className="text-[11px] text-[var(--text-muted)] tabular-nums">{suffix}</span>}
      </div>
    </div>
  );
}

function RefreshPassCard({ pass, totalPasses, ok, pending }: { pass: number; totalPasses: number; ok: number; pending: number }) {
  const passPct = totalPasses > 0 ? Math.min(100, (pass / totalPasses) * 100) : 0;
  const processed = ok + pending;
  const okPct = processed > 0 ? (ok / processed) * 100 : 0;
  return (
    <section className="rounded-[18px] p-4" style={{ background: 'linear-gradient(180deg, rgba(96,165,250,0.10), rgba(96,165,250,0.025))', border: '1px solid rgba(96,165,250,0.25)' }}>
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Download className="h-4 w-4 text-[#60a5fa]" />
          <span className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-muted)] font-semibold">Refresh pass</span>
          <span className="text-[12px] font-semibold text-[#bfdbfe] tabular-nums">{pass} <span className="text-[var(--text-muted)]">/ {totalPasses}</span></span>
        </div>
        <div className="flex items-center gap-3 text-[10px] tabular-nums">
          <span className="text-[#10b981]"><b>{ok}</b> ok</span>
          <span className={pending > 0 ? 'text-[#fbbf24]' : 'text-[var(--text-muted)]'}><b>{pending}</b> pending</span>
        </div>
      </div>
      <div className="mb-1.5 h-1.5 overflow-hidden rounded-full" style={{ background: 'rgba(96,165,250,0.12)' }}>
        <div className="h-full transition-[width] duration-500" style={{ width: `${passPct}%`, background: 'linear-gradient(90deg,#60a5fa,#818cf8)' }} />
      </div>
      {processed > 0 && (
        <div className="flex h-1 overflow-hidden rounded-full" style={{ background: 'rgba(148,163,184,0.12)' }}>
          <div className="transition-[width] duration-500" style={{ width: `${okPct}%`, background: 'linear-gradient(90deg,#10b981,#34d399)' }} />
          <div style={{ flex: 1, background: 'repeating-linear-gradient(-45deg, rgba(251,191,36,0.55) 0 4px, rgba(251,191,36,0.22) 4px 8px)' }} />
        </div>
      )}
    </section>
  );
}

function TerminalBanner({ color, icon, children }: { color: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 rounded-[16px] px-3 py-2.5 text-[12px] font-medium" style={{ background: `${color}14`, border: `1px solid ${color}40`, color }}>
      {icon}
      <span>{children}</span>
    </div>
  );
}
