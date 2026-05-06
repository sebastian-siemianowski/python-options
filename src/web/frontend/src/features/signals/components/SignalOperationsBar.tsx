import type { ReactNode } from 'react';
import { Activity, Loader2, Play, RefreshCw, Square, Zap } from 'lucide-react';
import type { SummaryRow } from '../../../api';
import CyberProgressRing from '../../../components/CyberProgressRing';
import { formatJobElapsed, type JobCounters, type JobMode, type JobStageMetric, type JobStatus } from '../../../stores/jobStore';

interface SignalOperationsBarProps {
  status: JobStatus;
  mode: JobMode | null;
  counters: JobCounters;
  stageMetrics: JobStageMetric[];
  activeStageKey: string | null;
  elapsedSec: number;
  phaseTitle: string | null;
  filteredRows: SummaryRow[];
  totalRows: number;
  onRefreshStocks: () => void;
  onRunTune: () => void;
  onRunBoth: () => void;
  onViewProgress: () => void;
  onStop: () => void;
}

export default function SignalOperationsBar({
  status,
  mode,
  counters,
  stageMetrics,
  activeStageKey,
  elapsedSec,
  phaseTitle,
  filteredRows,
  totalRows,
  onRefreshStocks,
  onRunTune,
  onRunBoth,
  onViewProgress,
  onStop,
}: SignalOperationsBarProps) {
  const isRunning = status === 'running';
  const isStocks = mode === 'stocks';
  const isBoth = mode === 'tune-stocks';
  const isTune = mode === 'retune' || mode === 'tune' || mode === 'calibrate' || isBoth;
  const activeStage = stageMetrics.find((stage) => stage.key === activeStageKey) ?? stageMetrics.find((stage) => stage.status === 'running') ?? stageMetrics[stageMetrics.length - 1] ?? null;
  const activeCounters = activeStage ? { done: activeStage.done, fail: activeStage.fail, total: activeStage.total } : counters;
  const processed = activeCounters.done + activeCounters.fail;
  const hasCountableActiveStage = activeCounters.total > 0;
  const progressPct = activeCounters.total > 0 ? Math.min(100, (processed / activeCounters.total) * 100) : isRunning ? 7 : 0;
  const completionRate = processed > 0 && activeStage?.kind !== 'download' ? Math.round((activeCounters.done / processed) * 100) : null;
  const etaSec = isRunning && processed > 0 && activeCounters.total > processed && activeStage?.kind !== 'download'
    ? Math.max(0, Math.round(((activeCounters.total - processed) * elapsedSec) / processed))
    : null;
  const statusColor = status === 'running' ? '#60a5fa'
    : status === 'completed' ? '#10b981'
      : status === 'failed' || status === 'error' ? '#f43f5e'
        : status === 'stopped' ? '#94a3b8'
          : '#a78bfa';
  const statusLabel = isRunning
    ? isBoth ? 'Refreshing 7d, retuning, then signals' : mode === 'stocks' ? 'Refreshing market data' : 'Tuning models'
    : status === 'completed' ? 'Last job complete'
      : status === 'stopped' ? 'Stopped'
        : status === 'failed' || status === 'error' ? 'Needs attention'
          : 'Ready';
  const pipelineStages = isBoth
    ? [
        { key: 'download', label: 'Refresh 7d data', tone: '#60a5fa' },
        { key: 'backup', label: 'Backup cache', tone: '#a78bfa' },
        { key: 'tune', label: 'Retune models', tone: '#c084fc' },
        { key: 'signals', label: 'Generate signals', tone: '#38d9f5' },
      ]
    : isStocks
    ? [
        { key: 'download', label: 'Refresh prices', tone: '#60a5fa' },
        { key: 'signals', label: 'Generate signals', tone: '#38d9f5' },
      ]
    : [
        { key: 'download', label: 'Refresh 7d data', tone: '#60a5fa' },
        { key: 'backup', label: 'Backup cache', tone: '#a78bfa' },
        { key: 'tune', label: 'Retune models', tone: '#c084fc' },
        { key: 'calibration', label: 'Calibration', tone: '#10b981' },
      ];
  const activeStageIndex = (() => {
    const title = (phaseTitle ?? '').toLowerCase();
    if (!isRunning) return status === 'completed' ? pipelineStages.length : 0;
    if (activeStage?.kind) {
      const stageIndex = pipelineStages.findIndex((stage) => stage.key === activeStage.kind);
      if (stageIndex >= 0) return stageIndex + 1;
    }
    if (title.includes('refresh') || title.includes('download')) return 1;
    if (title.includes('signal')) return isBoth ? 4 : isStocks ? 2 : 1;
    if (title.includes('backup')) return 2;
    if (title.includes('calibrat')) return 4;
    if (title.includes('fit') || title.includes('tune') || title.includes('model')) return 3;
    return Math.max(1, Math.min(pipelineStages.length, Math.ceil((progressPct / 100) * pipelineStages.length)));
  })();
  const activeStageTone = isRunning && activeStage?.kind
    ? pipelineStages.find((stage) => stage.key === activeStage.kind)?.tone ?? statusColor
    : status === 'completed' ? '#10b981'
      : status === 'failed' || status === 'error' ? '#fb7185'
        : '#a78bfa';
  const progressAccent = isRunning ? '#38d9f5' : status === 'completed' ? '#6ee7b7' : '#38d9f5';
  const progressStageTitle = isRunning ? activeStage?.title ?? phaseTitle ?? 'Pipeline runway' : 'Pipeline runway';
  const progressStageBadge = isRunning || status === 'completed'
    ? `Stage ${Math.max(0, activeStageIndex)} / ${pipelineStages.length}`
    : status === 'stopped' ? 'Stopped' : status === 'failed' || status === 'error' ? 'Needs attention' : 'Ready';
  const activeStageLabel = activeStage?.kind === 'download'
    ? (isBoth || mode === 'retune' ? 'refreshed' : 'ready')
    : activeStage?.kind === 'signals'
      ? 'generated'
    : activeStage?.kind === 'backup'
      ? 'backed up'
      : activeStage?.kind === 'calibration'
        ? 'calibrated'
        : 'processed';
  const runningProgressText = hasCountableActiveStage
    ? `${processed}${activeCounters.total > 0 ? ` / ${activeCounters.total}` : ''} ${activeStageLabel}`
    : activeStage?.kind === 'signals'
      ? 'Generating signals'
      : activeStage?.kind === 'backup'
        ? 'Backing up cache'
      : 'Working';
  const runTuneSubtitle = isRunning
    ? isTune && !isBoth
      ? activeStage?.kind === 'download'
        ? 'Refreshing last 7 days first'
        : activeStage?.kind === 'backup'
          ? 'Backing up tune cache'
          : activeStage?.kind === 'tune'
            ? 'Retuning models live with CPU minus one workers'
            : 'Live pipeline in progress'
      : 'Open the live activity drawer'
    : 'Refresh 7d data, back up, then retune';
  const stocksSubtitle = isRunning
    ? isStocks ? 'Refreshing market data' : 'Open the live activity drawer'
    : 'Prices, cache, and signals';
  const runBothSubtitle = isRunning
    ? isBoth ? 'Max-speed refresh, retune, then signals' : 'Open the live activity drawer'
    : 'CPU - 1 retune, then regenerate signals';

  return (
    <div className="fade-up mb-6">
      <div
        className="relative overflow-hidden rounded-[30px] px-4 py-4 md:px-6 md:py-5"
        style={{
          background: 'radial-gradient(900px 280px at 18% -20%, rgba(167,139,250,0.20), transparent 62%), radial-gradient(760px 260px at 92% 118%, rgba(56,217,245,0.13), transparent 62%), linear-gradient(135deg, rgba(25,26,44,0.82), rgba(8,9,18,0.91) 56%, rgba(24,16,42,0.82))',
          border: '1px solid rgba(255,255,255,0.09)',
          boxShadow: '0 34px 96px -58px rgba(139,92,246,0.95), 0 22px 82px -56px rgba(56,217,245,0.55), 0 18px 58px -42px rgba(0,0,0,0.95), inset 0 1px 0 rgba(255,255,255,0.11)',
          backdropFilter: 'blur(24px) saturate(1.35)',
          WebkitBackdropFilter: 'blur(24px) saturate(1.35)',
        }}
      >
        <div aria-hidden className="tune-orb-slow absolute -left-24 -top-28 h-56 w-56 rounded-full" style={{ background: 'radial-gradient(circle, rgba(139,92,246,0.22), rgba(139,92,246,0.05) 42%, transparent 72%)', filter: 'blur(4px)' }} />
        <div aria-hidden className="tune-orb-slow tune-orb-delay absolute -bottom-28 -right-20 h-64 w-64 rounded-full" style={{ background: 'radial-gradient(circle, rgba(56,217,245,0.16), rgba(56,217,245,0.035) 42%, transparent 72%)', filter: 'blur(6px)' }} />
        <div aria-hidden className="absolute inset-x-12 top-0 h-px" style={{ background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.28), rgba(196,181,253,0.38), transparent)' }} />
        <div aria-hidden className="absolute inset-x-0 bottom-0 h-px" style={{ background: 'linear-gradient(90deg, transparent, rgba(56,217,245,0.18), rgba(139,92,246,0.24), transparent)' }} />

        <div className="relative flex flex-col gap-5 2xl:flex-row 2xl:items-stretch 2xl:justify-between">
          <div className="min-w-0 flex-1">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={isRunning ? onViewProgress : undefined}
                className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.10em] ${isRunning ? 'cursor-pointer hover:brightness-125' : 'cursor-default'}`}
                style={{ color: statusColor, background: `${statusColor}18`, border: `1px solid ${statusColor}36` }}
              >
                <span className={`h-1.5 w-1.5 rounded-full ${isRunning ? 'animate-pulse' : ''}`} style={{ background: statusColor, boxShadow: isRunning ? `0 0 8px ${statusColor}` : undefined }} />
                {statusLabel}
              </button>
              {isRunning && (
                <span className="text-[11px] tabular-nums text-[var(--text-muted)]">
                  {formatJobElapsed(elapsedSec)} · {runningProgressText}{etaSec !== null ? ` · ETA ${formatJobElapsed(etaSec)}` : ''}
                </span>
              )}
            </div>
            <div className="mt-4 grid max-w-[760px] grid-cols-2 gap-2 sm:grid-cols-4" aria-label="Tune pipeline stages">
              {pipelineStages.map((stage, index) => {
                const stageNumber = index + 1;
                const active = activeStageIndex === stageNumber;
                const done = activeStageIndex > stageNumber;
                return (
                  <div
                    key={stage.label}
                    className="rounded-2xl px-3 py-2 transition-all duration-300"
                    style={{
                      background: done || active ? `${stage.tone}14` : 'rgba(255,255,255,0.025)',
                      border: `1px solid ${done || active ? `${stage.tone}3d` : 'rgba(255,255,255,0.055)'}`,
                      boxShadow: active ? `0 12px 30px -24px ${stage.tone}` : 'inset 0 1px 0 rgba(255,255,255,0.035)',
                    }}
                  >
                    <div className="mb-1 flex items-center gap-1.5">
                      <span
                        className={`h-1.5 w-1.5 rounded-full ${active && isRunning ? 'animate-pulse' : ''}`}
                        style={{ background: done || active ? stage.tone : 'rgba(255,255,255,0.18)', boxShadow: active ? `0 0 8px ${stage.tone}` : undefined }}
                      />
                      <span className="text-[9px] font-semibold uppercase tracking-[0.13em] text-[var(--text-muted)]">Step {stageNumber}</span>
                    </div>
                    <div className="truncate text-[11px] font-semibold tracking-[-0.01em]" style={{ color: done || active ? 'var(--text-primary)' : 'var(--text-secondary)' }}>{stage.label}</div>
                  </div>
                );
              })}
            </div>
            <div
              className="relative mt-4 flex max-w-[800px] items-center gap-4 overflow-hidden rounded-[28px] p-3"
              style={{
                background: `radial-gradient(320px 160px at 0% 50%, ${activeStageTone}18, transparent 68%), linear-gradient(135deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018))`,
                border: `1px solid ${activeStageTone}24`,
                boxShadow: `0 22px 58px -42px ${activeStageTone}, inset 0 1px 0 rgba(255,255,255,0.07)`,
              }}
              aria-label={`Live job progress ${Math.round(progressPct)} percent`}
            >
              <CyberProgressRing
                percent={progressPct}
                color={activeStageTone}
                accent={progressAccent}
                size={162}
                stroke={11}
                label={isRunning ? 'Live' : status === 'completed' ? 'Done' : status === 'stopped' ? 'Stopped' : 'Ready'}
                caption={isBoth ? 'Run both' : isTune ? 'Retune' : 'Signals'}
                running={isRunning}
                status={status}
                stages={pipelineStages}
                activeStageIndex={activeStageIndex}
              />
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.12em]" style={{ color: activeStageTone, background: `${activeStageTone}14`, border: `1px solid ${activeStageTone}2d` }}>
                    Pipeline state
                  </span>
                  <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--text-muted)]">{progressStageBadge}</span>
                </div>
                <div className="mt-2 truncate text-[18px] font-semibold tracking-[-0.035em] text-white">{progressStageTitle}</div>
                <div className="mt-1 text-[12px] text-[var(--text-secondary)]">
                  {isRunning ? runningProgressText : status === 'completed' ? 'Fresh results are ready.' : status === 'stopped' ? 'Stopped safely. Ready for the next run.' : 'Ready to run at full CPU - 1 speed.'}
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-2 text-[10.5px] text-[var(--text-muted)]">
                  <span className="rounded-full px-2.5 py-1 tabular-nums" style={{ background: 'rgba(255,255,255,0.045)', border: '1px solid rgba(255,255,255,0.07)' }}>{Math.round(progressPct)}% complete</span>
                  {isRunning && <span className="rounded-full px-2.5 py-1 tabular-nums" style={{ color: '#bfdbfe', background: 'rgba(96,165,250,0.09)', border: '1px solid rgba(147,197,253,0.20)' }}>{formatJobElapsed(elapsedSec)} elapsed</span>}
                  {etaSec !== null && <span className="rounded-full px-2.5 py-1 tabular-nums" style={{ color: '#ddd6fe', background: 'rgba(139,92,246,0.10)', border: '1px solid rgba(167,139,250,0.22)' }}>ETA {formatJobElapsed(etaSec)}</span>}
                </div>
              </div>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] text-[var(--text-muted)]">
              <span className="truncate">
                {isRunning ? `${phaseTitle ?? 'Preparing live pipeline...'} · Signals remains usable` : `${filteredRows.length.toLocaleString()} visible signals · ${totalRows.toLocaleString()} total assets`}
              </span>
              {completionRate !== null && isRunning && (
                <span className="rounded-full px-2 py-0.5 tabular-nums" style={{ color: '#a7f3d0', background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.18)' }}>{completionRate}% success</span>
              )}
              {activeCounters.fail > 0 && (
                <span className="rounded-full px-2 py-0.5 tabular-nums" style={{ color: '#fb7185', background: 'rgba(244,63,94,0.10)', border: '1px solid rgba(244,63,94,0.22)' }}>{activeCounters.fail} failed</span>
              )}
            </div>
          </div>

          <div className="flex flex-col justify-center gap-3 2xl:w-[520px]">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 2xl:grid-cols-1">
              <OperationButton
                icon={isTune && !isBoth && isRunning ? <Loader2 className="h-5 w-5 animate-spin" /> : <Play className="h-5 w-5" />}
                title={isRunning ? isTune && !isBoth ? 'Tuning live...' : 'View live activity' : 'Run Tune'}
                subtitle={runTuneSubtitle}
                eyebrow={isRunning && isTune && !isBoth ? 'Streaming now' : 'Recommended'}
                color="#a78bfa"
                active={isTune && !isBoth && isRunning}
                primary
                onClick={onRunTune}
              />
              <OperationButton
                icon={isStocks && isRunning ? <Loader2 className="h-5 w-5 animate-spin" /> : <RefreshCw className="h-5 w-5" />}
                title={isRunning ? isStocks ? 'Refreshing...' : 'View live activity' : 'Refresh Stocks'}
                subtitle={stocksSubtitle}
                eyebrow="Market data"
                color="#60a5fa"
                active={isStocks && isRunning}
                onClick={onRefreshStocks}
              />
              <OperationButton
                icon={isBoth && isRunning ? <Loader2 className="h-5 w-5 animate-spin" /> : <Zap className="h-5 w-5" />}
                title={isRunning ? isBoth ? 'Running both...' : 'View live activity' : 'Run Both'}
                subtitle={runBothSubtitle}
                eyebrow="One click"
                color="#14b8a6"
                active={isBoth && isRunning}
                onClick={onRunBoth}
              />
            </div>

            <div className="flex flex-wrap items-center justify-end gap-2">
              {isRunning && (
                <button
                  type="button"
                  onClick={onViewProgress}
                  className="inline-flex items-center justify-center gap-2 rounded-full px-3.5 py-2 text-[12px] font-semibold transition-all hover:-translate-y-0.5 active:scale-[0.98]"
                  style={{ color: '#dbeafe', background: 'rgba(96,165,250,0.10)', border: '1px solid rgba(96,165,250,0.26)', boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.06)' }}
                >
                  <Activity className="h-3.5 w-3.5" />
                  View Live Activity
                </button>
              )}
              {isRunning && (
                <button
                  type="button"
                  onClick={onStop}
                  className="group inline-flex items-center justify-center gap-2 rounded-full px-3.5 py-2 text-[12px] font-semibold text-white transition-all hover:-translate-y-0.5 active:scale-[0.98]"
                  style={{ background: 'linear-gradient(180deg,#fb7185,#e11d48)', boxShadow: '0 16px 34px -22px rgba(244,63,94,0.95)' }}
                >
                  <Square className="h-3.5 w-3.5" fill="currentColor" />
                  Stop
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function OperationButton({
  icon,
  title,
  subtitle,
  eyebrow,
  color,
  active,
  primary,
  onClick,
}: {
  icon: ReactNode;
  title: string;
  subtitle: string;
  eyebrow?: string;
  color: string;
  active: boolean;
  primary?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`focus-ring group relative overflow-hidden rounded-[24px] text-left transition-all duration-300 hover:-translate-y-0.5 active:scale-[0.985] ${primary ? 'px-5 py-4' : 'px-4 py-3'}`}
      style={{
        background: primary
          ? `radial-gradient(520px 160px at 18% -18%, ${color}36, transparent 58%), linear-gradient(150deg, ${color}24, rgba(255,255,255,0.055) 54%, rgba(56,217,245,0.075))`
          : active
            ? `linear-gradient(150deg, ${color}20, rgba(255,255,255,0.035))`
            : 'linear-gradient(150deg, rgba(255,255,255,0.052), rgba(255,255,255,0.018))',
        border: `1px solid ${primary ? `${color}70` : active ? `${color}58` : 'rgba(255,255,255,0.085)'}`,
        boxShadow: primary
          ? `0 26px 62px -38px ${color}, 0 0 0 1px ${color}18 inset, inset 0 1px 0 rgba(255,255,255,0.14)`
          : active ? `0 16px 38px -30px ${color}, inset 0 1px 0 rgba(255,255,255,0.12)` : 'inset 0 1px 0 rgba(255,255,255,0.065)',
      }}
    >
      <div aria-hidden className="absolute inset-x-4 top-0 h-px" style={{ background: `linear-gradient(90deg, transparent, ${color}b8, rgba(255,255,255,0.42), transparent)`, opacity: primary || active ? 1 : 0.45 }} />
      {primary && <div aria-hidden className="tune-orb-slow absolute -right-16 -top-20 h-36 w-36 rounded-full" style={{ background: `radial-gradient(circle, ${color}2f, transparent 70%)`, filter: 'blur(5px)' }} />}
      <div className="relative flex items-center gap-3">
        <span
          className={`inline-flex shrink-0 items-center justify-center rounded-[17px] transition-transform duration-200 group-hover:scale-105 ${primary ? 'h-12 w-12' : 'h-10 w-10'}`}
          style={{ color, background: `${color}1d`, border: `1px solid ${color}3d`, boxShadow: primary || active ? `0 0 26px -8px ${color}` : undefined }}
        >
          {icon}
        </span>
        <span className="min-w-0">
          {eyebrow && <span className="mb-1 block text-[9px] font-semibold uppercase tracking-[0.13em]" style={{ color }}>{eyebrow}</span>}
          <span className={`block font-semibold tracking-[-0.035em] text-white ${primary ? 'text-[17px]' : 'text-[14px]'}`}>{title}</span>
          <span className={`mt-0.5 block truncate text-[var(--text-muted)] ${primary ? 'text-[12px]' : 'text-[11px]'}`}>{subtitle}</span>
        </span>
      </div>
    </button>
  );
}
