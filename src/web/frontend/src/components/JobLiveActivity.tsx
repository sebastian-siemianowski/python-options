import { useEffect, useMemo, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
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
  type JobMode,
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

function titleForMode(mode: JobMode | null) {
  if (!mode) return JOB_MODE_LABELS.retune;
  return JOB_MODE_LABELS[mode];
}

export default function JobLiveActivity() {
  const queryClient = useQueryClient();
  const job = useJobStore();
  const prevStatusRef = useRef<JobStatus>(job.status);
  const rawLogRef = useRef<HTMLDivElement | null>(null);

  const info = titleForMode(job.mode);
  const tone = statusTone(job.status);
  const isRunning = job.status === 'running';
  const processed = job.counters.done + job.counters.fail;
  const progressPct = job.counters.total > 0
    ? Math.min(100, (processed / job.counters.total) * 100)
    : isRunning ? 3 : job.status === 'idle' ? 0 : 100;
  const rate = processed > 0 ? processed / Math.max(job.elapsedSec, 1) : 0;
  const eta = isRunning && rate > 0 && job.counters.total > processed
    ? Math.round((job.counters.total - processed) / rate)
    : null;
  const currentPhase = job.phases.length > 0 ? job.phases[job.phases.length - 1] : null;
  const recentAssets = useMemo(() => job.assets.slice(-28).reverse(), [job.assets]);
  const topModels = useMemo(() => {
    return Object.entries(job.modelCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([name, count]) => ({ name, count, hue: hashHue(name) }));
  }, [job.modelCounts]);
  const totalModelled = useMemo(() => Object.values(job.modelCounts).reduce((sum, value) => sum + value, 0), [job.modelCounts]);

  useEffect(() => {
    if (prevStatusRef.current === 'running' && job.status === 'completed') {
      SIGNAL_QUERY_KEYS.forEach((queryKey) => {
        queryClient.invalidateQueries({ queryKey: [...queryKey] });
      });
    }
    prevStatusRef.current = job.status;
  }, [job.status, queryClient]);

  useEffect(() => {
    if (job.rawLogOpen && rawLogRef.current) {
      rawLogRef.current.scrollTop = rawLogRef.current.scrollHeight;
    }
  }, [job.logLines, job.rawLogOpen]);

  if (!job.surfaceVisible || job.status === 'idle' || !job.mode) return null;

  return (
    <div className="fixed bottom-5 right-5 z-50 pointer-events-none max-w-[calc(100vw-2rem)]">
      <style>{`
        @keyframes liveActivityGlow {
          0%, 100% { transform: translate3d(0,0,0) scale(1); opacity: 0.78; }
          50% { transform: translate3d(10px,-8px,0) scale(1.08); opacity: 1; }
        }
        @keyframes liveActivityShimmer {
          0% { transform: translateX(-120%); }
          100% { transform: translateX(120%); }
        }
        @keyframes liveActivityRise {
          from { opacity: 0; transform: translateY(12px) scale(0.98); filter: blur(4px); }
          to { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
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
                  <div className="text-[20px] font-semibold tracking-[-0.03em] text-white leading-tight">{info.title}</div>
                  <div className="mt-1 text-[12px] text-[var(--text-muted)] truncate max-w-[440px]">
                    {currentPhase?.title ?? info.desc}
                    {eta !== null && <span className="text-[var(--text-secondary)]"> · ETA {formatJobElapsed(eta)}</span>}
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

            <div className="relative h-[4px] overflow-hidden" style={{ background: 'rgba(255,255,255,0.045)' }}>
              <div
                className="h-full transition-[width] duration-700 ease-out"
                style={{
                  width: `${progressPct}%`,
                  background: job.status === 'failed' || job.status === 'error'
                    ? 'linear-gradient(90deg,#f43f5e,#fb7185)'
                    : `linear-gradient(90deg, ${info.color}, #38d9f5)`,
                  boxShadow: `0 0 16px ${info.color}77`,
                }}
              />
              {isRunning && (
                <div
                  className="absolute inset-y-0 job-live-activity-anim"
                  style={{
                    width: `${Math.max(progressPct, 8)}%`,
                    background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.28), transparent)',
                    animation: 'liveActivityShimmer 1.8s linear infinite',
                  }}
                />
              )}
            </div>
          </div>

          <div className="max-h-[calc(min(78vh,820px)-116px)] overflow-y-auto p-5 space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5" aria-live="polite">
              <MetricTile label="Processed" value={processed} suffix={job.counters.total > 0 ? `/ ${job.counters.total}` : undefined} color="#ffffff" />
              <MetricTile label="Successful" value={job.counters.done} suffix={processed > 0 ? `${Math.round((job.counters.done / processed) * 100)}%` : undefined} color="#10b981" pulse={isRunning && job.counters.done > 0} />
              <MetricTile label="Failed" value={job.counters.fail} color={job.counters.fail > 0 ? '#f43f5e' : 'var(--text-muted)'} />
              <MetricTile label={rate > 0 ? 'Throughput' : 'Progress'} value={rate > 0 ? (rate * 60).toFixed(1) : `${Math.round(progressPct)}%`} suffix={rate > 0 ? '/ min' : undefined} color={info.color} />
            </div>

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

            {recentAssets.length > 0 && (
              <section>
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-muted)] font-semibold">Recent completions</span>
                  <span className="text-[10px] text-[var(--text-muted)] tabular-nums">showing {recentAssets.length} of {job.assets.length}</span>
                </div>
                <div className="grid grid-cols-[repeat(auto-fill,minmax(138px,1fr))] gap-1.5">
                  {recentAssets.map((asset, index) => {
                    const model = asset.model ?? job.modelBySymbol[asset.symbol];
                    const ok = asset.status === 'ok';
                    return (
                      <div key={`${asset.symbol}-${index}`} className="rounded-xl px-2.5 py-2 font-mono" style={{ background: ok ? 'rgba(16,185,129,0.065)' : 'rgba(244,63,94,0.08)', border: `1px solid ${ok ? 'rgba(16,185,129,0.20)' : 'rgba(244,63,94,0.28)'}` }} title={model ? `${asset.symbol} · ${model}` : asset.detail ?? asset.symbol}>
                        <div className="flex items-center gap-1.5 min-w-0">
                          {ok ? <CheckCircle2 className="h-3 w-3 shrink-0" style={{ color: '#10b981' }} /> : <XCircle className="h-3 w-3 shrink-0" style={{ color: '#f43f5e' }} />}
                          <span className="truncate text-[11px] font-semibold text-[var(--text-primary)]">{asset.symbol}</span>
                        </div>
                        {model && <div className="mt-1 truncate pl-[18px] text-[9.5px] tracking-[-0.01em]" style={{ color: info.color, opacity: 0.86 }}>{model}</div>}
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
        className="job-live-activity-anim pointer-events-auto w-[min(430px,calc(100vw-2rem))] overflow-hidden rounded-[22px]"
        style={{
          animation: 'liveActivityRise 220ms cubic-bezier(.2,.8,.2,1) both',
          background: 'linear-gradient(180deg, rgba(20,22,35,0.82) 0%, rgba(8,9,17,0.90) 100%)',
          border: `1px solid ${info.color}34`,
          boxShadow: `0 20px 70px -26px ${info.color}88, 0 18px 64px -36px rgba(0,0,0,0.95), inset 0 1px 0 rgba(255,255,255,0.10)`,
          backdropFilter: 'blur(26px) saturate(1.35)',
          WebkitBackdropFilter: 'blur(26px) saturate(1.35)',
        }}
        aria-live="polite"
      >
        <div className="relative px-4 py-3.5">
          <div aria-hidden className="absolute inset-x-4 top-0 h-px" style={{ background: `linear-gradient(90deg, transparent, ${info.color}80, transparent)` }} />
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={job.toggleExpanded}
              className="relative h-10 w-10 rounded-[14px] flex items-center justify-center shrink-0 transition-transform active:scale-95"
              style={{ background: `linear-gradient(145deg, ${info.color}30, rgba(255,255,255,0.035))`, border: `1px solid ${info.color}40`, color: info.color, boxShadow: isRunning ? `0 0 20px -5px ${info.color}` : undefined }}
              aria-label="Expand live job progress"
            >
              {isRunning ? <Loader2 className="h-5 w-5 animate-spin" /> : job.status === 'completed' ? <CheckCircle2 className="h-5 w-5" /> : job.status === 'failed' || job.status === 'error' ? <XCircle className="h-5 w-5" /> : <Activity className="h-5 w-5" />}
            </button>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="text-[12px] font-semibold tracking-[-0.02em] text-white truncate">{info.shortTitle}</span>
                <StatusPill label={tone.label} color={tone.color} bg={tone.bg} pulse={isRunning} compact />
              </div>
              <div className="mt-0.5 truncate text-[11px] text-[var(--text-muted)]">
                {currentPhase?.title ?? info.desc}
              </div>
              <div className="mt-2 flex items-center gap-2">
                <div className="h-1.5 flex-1 overflow-hidden rounded-full" style={{ background: 'rgba(255,255,255,0.07)' }}>
                  <div className="h-full rounded-full transition-[width] duration-700" style={{ width: `${progressPct}%`, background: `linear-gradient(90deg, ${info.color}, #38d9f5)`, boxShadow: `0 0 10px ${info.color}77` }} />
                </div>
                <span className="text-[10px] font-mono text-[var(--text-secondary)] tabular-nums min-w-[86px] text-right">
                  {processed}{job.counters.total > 0 ? ` / ${job.counters.total}` : ''}
                </span>
              </div>
            </div>
            <div className="flex flex-col items-end gap-1.5 shrink-0">
              <span className="text-[12px] font-mono font-semibold tabular-nums text-[var(--text-primary)]" title={`Elapsed ${formatJobDuration(job.elapsedSec)}`}>{formatJobElapsed(job.elapsedSec)}</span>
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
