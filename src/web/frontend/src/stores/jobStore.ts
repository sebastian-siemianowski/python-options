import { create } from 'zustand';

export type JobMode = 'tune' | 'stocks' | 'retune' | 'calibrate' | 'tune-stocks';
export type JobStatus = 'idle' | 'running' | 'completed' | 'failed' | 'error' | 'stopped';

type EventType =
  | 'start'
  | 'phase'
  | 'asset'
  | 'model'
  | 'refresh'
  | 'progress'
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
  step?: number;
  total_steps?: number;
  phase_count?: number;
  kind?: string;
  phase_step?: number;
  phase_title?: string;
  model?: string;
  pass?: number;
  total_passes?: number;
  ok?: number;
  pending?: number;
  weight_pct?: number;
  bic?: number;
  hyv?: number;
  crps?: number;
  pit_p?: number;
  fit_status?: string;
  error_type?: string;
  error?: string;
}

export interface JobLogLine {
  id: number;
  text: string;
}

export interface JobAssetEntry {
  symbol: string;
  status: 'ok' | 'fail';
  detail?: string;
  model?: string;
  weightPct?: number;
  bic?: number;
  hyv?: number;
  crps?: number;
  pitP?: number;
  fitStatus?: string;
}

export interface JobModelMeta {
  model: string;
  weightPct?: number;
  bic?: number;
  hyv?: number;
  crps?: number;
  pitP?: number;
  fitStatus?: string;
  updatedAt: number;
}

export interface JobPhaseEntry {
  id: number;
  title: string;
  step?: number;
  totalSteps?: number;
  kind?: string;
  startedAt: number;
}

export interface JobRefreshPassState {
  pass: number;
  totalPasses: number;
  ok: number;
  pending: number;
}

export interface JobCounters {
  done: number;
  fail: number;
  total: number;
}

export type JobStageStatus = 'pending' | 'running' | 'completed' | 'failed' | 'stopped';

export interface JobStageMetric {
  key: string;
  title: string;
  kind: string;
  step?: number;
  totalSteps?: number;
  done: number;
  fail: number;
  total: number;
  status: JobStageStatus;
  startedAt: number;
  updatedAt: number;
}

interface JobState {
  mode: JobMode | null;
  status: JobStatus;
  phases: JobPhaseEntry[];
  assets: JobAssetEntry[];
  logLines: JobLogLine[];
  counters: JobCounters;
  stageMetrics: JobStageMetric[];
  activeStageKey: string | null;
  elapsedSec: number;
  startedAt: number | null;
  finishedAt: number | null;
  errorMsg: string | null;
  refreshPass: JobRefreshPassState | null;
  modelBySymbol: Record<string, string>;
  modelMetaBySymbol: Record<string, JobModelMeta>;
  modelCounts: Record<string, number>;
  surfaceVisible: boolean;
  expanded: boolean;
  rawLogOpen: boolean;
  startJob: (mode: JobMode) => void;
  stopJob: () => void;
  dismissSurface: () => void;
  showSurface: () => void;
  setExpanded: (expanded: boolean) => void;
  toggleExpanded: () => void;
  setRawLogOpen: (open: boolean) => void;
  clearTerminalJob: () => void;
}

const STORAGE_KEY = 'python-options-live-job-snapshot-v1';
const MAX_ASSETS = 500;
const MAX_LOGS = 300;
const MAX_PHASES = 30;
const REFRESH_PASS_RX = /Pass\s+(\d+)\s*\/\s*(\d+)/i;
const REFRESH_PROGRESS_RX = /●\s*(\d+)\s*ok\s*[·|/•]\s*(\d+)\s*pending/i;
const REFRESH_COMPLETE_RX = /✓\s*(\d+)\s*\/\s*(\d+)\s*complete/i;
const TUNE_TOTAL_RX = /\b(\d+)\s+to\s+process\b/i;
const TUNE_MODEL_RX = /^\s*✓\s*(\S+)\s+(?:→|->)\s+(.+?)\s*$/;

let eventSource: EventSource | null = null;
let elapsedTimer: number | null = null;
let logFlushTimer: number | null = null;
let pendingLogLines: JobLogLine[] = [];
let logId = 0;
let phaseId = 0;

function requestBackendCancel() {
  if (typeof navigator !== 'undefined' && typeof navigator.sendBeacon === 'function') {
    try {
      const payload = new Blob(['{}'], { type: 'application/json' });
      if (navigator.sendBeacon('/api/tune/retune/cancel', payload)) return;
    } catch {
      // Fall through to fetch.
    }
  }

  void fetch('/api/tune/retune/cancel', {
    method: 'POST',
    keepalive: true,
    cache: 'no-store',
  }).catch(() => {
    // The UI still closes promptly; the backend may already have ended.
  });
}

function cleanupStream() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  if (elapsedTimer !== null) {
    window.clearInterval(elapsedTimer);
    elapsedTimer = null;
  }
  if (logFlushTimer !== null) {
    window.clearTimeout(logFlushTimer);
    logFlushTimer = null;
  }
  pendingLogLines = [];
}

function safeLoadSnapshot(): Partial<JobState> | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<JobState>;
    if (parsed.status === 'running') {
      return {
        ...parsed,
        status: 'error',
        errorMsg: 'Browser reloaded while the job stream was active. Start again to reconnect.',
        surfaceVisible: true,
        expanded: false,
        finishedAt: Date.now(),
      };
    }
    return parsed;
  } catch {
    return null;
  }
}

function persistSnapshot(state: JobState) {
  try {
    const snapshot = {
      mode: state.mode,
      status: state.status,
      phases: state.phases.slice(-MAX_PHASES),
      assets: state.assets.slice(-64),
      logLines: state.logLines.slice(-200),
      counters: state.counters,
      stageMetrics: state.stageMetrics,
      activeStageKey: state.activeStageKey,
      elapsedSec: state.elapsedSec,
      startedAt: state.startedAt,
      finishedAt: state.finishedAt,
      errorMsg: state.errorMsg,
      refreshPass: state.refreshPass,
      modelBySymbol: state.modelBySymbol,
      modelMetaBySymbol: state.modelMetaBySymbol,
      modelCounts: state.modelCounts,
      surfaceVisible: state.surfaceVisible,
      expanded: state.expanded,
      rawLogOpen: false,
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
  } catch {
    // Local storage is best-effort only.
  }
}

function initialState(): Omit<JobState,
  | 'startJob'
  | 'stopJob'
  | 'dismissSurface'
  | 'showSurface'
  | 'setExpanded'
  | 'toggleExpanded'
  | 'setRawLogOpen'
  | 'clearTerminalJob'
> {
  const snapshot = typeof window !== 'undefined' ? safeLoadSnapshot() : null;
  return {
    mode: snapshot?.mode ?? null,
    status: snapshot?.status ?? 'idle',
    phases: snapshot?.phases ?? [],
    assets: snapshot?.assets ?? [],
    logLines: snapshot?.logLines ?? [],
    counters: snapshot?.counters ?? { done: 0, fail: 0, total: 0 },
    stageMetrics: snapshot?.stageMetrics ?? [],
    activeStageKey: snapshot?.activeStageKey ?? null,
    elapsedSec: snapshot?.elapsedSec ?? 0,
    startedAt: snapshot?.startedAt ?? null,
    finishedAt: snapshot?.finishedAt ?? null,
    errorMsg: snapshot?.errorMsg ?? null,
    refreshPass: snapshot?.refreshPass ?? null,
    modelBySymbol: snapshot?.modelBySymbol ?? {},
    modelMetaBySymbol: snapshot?.modelMetaBySymbol ?? {},
    modelCounts: snapshot?.modelCounts ?? {},
    surfaceVisible: snapshot?.surfaceVisible ?? false,
    expanded: snapshot?.expanded ?? false,
    rawLogOpen: false,
  };
}

export const JOB_MODE_LABELS: Record<JobMode, { title: string; shortTitle: string; desc: string; color: string }> = {
  tune: { title: 'Retune models', shortTitle: 'Tune', desc: 'Re-estimating model parameters', color: '#8b5cf6' },
  stocks: { title: 'Refresh stocks', shortTitle: 'Stocks', desc: 'Refreshing prices and signals', color: '#3b82f6' },
  retune: { title: 'Run Tune', shortTitle: 'Tune', desc: 'Full retune pipeline', color: '#8b5cf6' },
  calibrate: { title: 'Calibrate', shortTitle: 'Calibrate', desc: 'Repairing calibration failures', color: '#10b981' },
  'tune-stocks': { title: 'Run Both', shortTitle: 'Both', desc: 'Tune models, then refresh prices and signals', color: '#14b8a6' },
};

function metricKindFromPhase(kind?: string, title?: string): string | null {
  const lowerTitle = (title ?? '').toLowerCase();
  if (kind === 'download') return 'download';
  if (kind === 'backup') return 'backup';
  if (kind === 'tune') return 'tune';
  if (kind === 'signals') return 'signals';
  if (kind === 'calibration') return 'calibration';
  if (kind === 'tune_sub' && lowerTitle.includes('calibrat')) return 'calibration';
  if (kind === 'work' || kind === 'launch') return kind;
  return null;
}

function stageKeyFor(kind: string, step?: number): string {
  if (kind === 'download') return 'download';
  if (kind === 'backup') return 'backup';
  if (kind === 'tune') return 'tune';
  if (kind === 'signals') return 'signals';
  if (kind === 'calibration') return 'calibration';
  return step ? `${kind}:${step}` : kind;
}

function stageTitleFor(kind: string, title: string): string {
  if (kind === 'download') return 'Refresh data';
  if (kind === 'backup') return 'Backup tune cache';
  if (kind === 'tune') return 'Tune stocks';
  if (kind === 'signals') return 'Generate signals';
  if (kind === 'calibration') return 'Calibration';
  return title;
}

function upsertRunningStage(
  stages: JobStageMetric[],
  stage: { key: string; title: string; kind: string; step?: number; totalSteps?: number; total?: number },
  now: number,
): JobStageMetric[] {
  const previous = stages.map((item) => (
    item.status === 'running' && item.key !== stage.key
      ? {
          ...item,
          done: item.done === 0 && item.total <= 1 ? Math.max(1, item.total) : item.done,
          total: item.total === 0 && item.done === 0 ? 1 : item.total,
          status: 'completed' as JobStageStatus,
          updatedAt: now,
        }
      : item
  ));
  const existing = previous.find((item) => item.key === stage.key);
  if (!existing) {
    return previous.concat({
      key: stage.key,
      title: stage.title,
      kind: stage.kind,
      step: stage.step,
      totalSteps: stage.totalSteps,
      done: 0,
      fail: 0,
      total: stage.total ?? 0,
      status: 'running',
      startedAt: now,
      updatedAt: now,
    });
  }
  return previous.map((item) => (
    item.key === stage.key
      ? {
          ...item,
          title: stage.title,
          kind: stage.kind,
          step: stage.step ?? item.step,
          totalSteps: stage.totalSteps ?? item.totalSteps,
          total: Math.max(item.total, stage.total ?? 0),
          status: 'running' as JobStageStatus,
          updatedAt: now,
        }
      : item
  ));
}

function updateStageCounts(
  stages: JobStageMetric[],
  key: string | null,
  counts: Partial<JobCounters>,
  now: number,
  options?: { exact?: boolean },
): JobStageMetric[] {
  if (!key) return stages;
  return stages.map((item) => (
    item.key === key
      ? {
          ...item,
          done: options?.exact && counts.done !== undefined ? counts.done : Math.max(item.done, counts.done ?? item.done),
          fail: options?.exact && counts.fail !== undefined ? counts.fail : Math.max(item.fail, counts.fail ?? item.fail),
          total: options?.exact && counts.total !== undefined ? counts.total : Math.max(item.total, counts.total ?? item.total),
          updatedAt: now,
        }
      : item
  ));
}

function parseRefreshLogLine(message: string, previous: JobRefreshPassState | null): JobRefreshPassState | null {
  let next = previous ? { ...previous } : null;
  let changed = false;

  for (const line of message.split(/\r?\n/)) {
    const passMatch = REFRESH_PASS_RX.exec(line);
    if (passMatch) {
      changed = true;
      next = {
        pass: Number(passMatch[1]),
        totalPasses: Number(passMatch[2]),
        ok: next?.ok ?? 0,
        pending: next?.pending ?? 0,
      };
    }

    const progressMatch = REFRESH_PROGRESS_RX.exec(line);
    if (progressMatch) {
      changed = true;
      next = {
        pass: next?.pass ?? 1,
        totalPasses: next?.totalPasses ?? 1,
        ok: Number(progressMatch[1]),
        pending: Number(progressMatch[2]),
      };
    }

    const completeMatch = REFRESH_COMPLETE_RX.exec(line);
    if (completeMatch) {
      changed = true;
      const ok = Number(completeMatch[1]);
      const total = Number(completeMatch[2]);
      next = {
        pass: next?.pass ?? next?.totalPasses ?? 1,
        totalPasses: next?.totalPasses ?? next?.pass ?? 1,
        ok,
        pending: Math.max(0, total - ok),
      };
    }
  }

  return changed ? next : null;
}

function collectTuneLogTelemetry(messages: string[]): { total: number; completions: Map<string, string> } {
  const completions = new Map<string, string>();
  let total = 0;

  for (const message of messages) {
    for (const line of message.split(/\r?\n/)) {
      const totalMatch = TUNE_TOTAL_RX.exec(line);
      if (totalMatch) {
        total = Math.max(total, Number(totalMatch[1]));
      }

      const modelMatch = TUNE_MODEL_RX.exec(line);
      if (modelMatch) {
        completions.set(modelMatch[1], modelMatch[2].trim());
      }
    }
  }

  return { total, completions };
}

export const useJobStore = create<JobState>((set, get) => ({
  ...initialState(),

  startJob: (mode) => {
    const current = get();
    if (current.status === 'running') {
      set({ surfaceVisible: true, expanded: true });
      return;
    }

    cleanupStream();
    logId = 0;
    phaseId = 0;
    const startedAt = Date.now();
    const launchInfo = JOB_MODE_LABELS[mode];
    let persistTimer: number | null = null;

    const schedulePersist = () => {
      if (persistTimer !== null) return;
      persistTimer = window.setTimeout(() => {
        persistTimer = null;
        persistSnapshot(get());
      }, 500);
    };

    const enqueueLogLine = (message: string) => {
      logId += 1;
      pendingLogLines.push({ id: logId, text: message });
      if (pendingLogLines.length > MAX_LOGS) {
        pendingLogLines = pendingLogLines.slice(-MAX_LOGS);
      }
      const refreshFromLog = parseRefreshLogLine(message, get().refreshPass);
      const shouldReconcileTune = TUNE_TOTAL_RX.test(message) || TUNE_MODEL_RX.test(message);
      if (refreshFromLog || shouldReconcileTune) {
        set((state) => {
          if (state.status !== 'running') return state;
          const now = Date.now();
          let stageMetrics = state.stageMetrics;
          let counters = state.counters;
          let activeStageKey = state.activeStageKey;
          let refreshPass = state.refreshPass;
          let assets = state.assets;
          let modelBySymbol = state.modelBySymbol;
          let modelMetaBySymbol = state.modelMetaBySymbol;
          let modelCounts = state.modelCounts;

          if (refreshFromLog) {
            const refreshTotal = refreshFromLog.ok + refreshFromLog.pending;
            refreshPass = refreshFromLog;
            if (refreshTotal > 0) {
              stageMetrics = updateStageCounts(
                stageMetrics,
                'download',
                { done: refreshFromLog.ok, total: refreshTotal },
                now,
                { exact: true },
              );
            }
          }

          if (shouldReconcileTune) {
            const tuneTelemetry = collectTuneLogTelemetry([
              ...state.logLines.map((line) => line.text),
              ...pendingLogLines.map((line) => line.text),
            ]);
            if (tuneTelemetry.total > 0 || tuneTelemetry.completions.size > 0) {
              const tuneTotal = Math.max(tuneTelemetry.total, state.counters.total, stageMetrics.find((stage) => stage.key === 'tune')?.total ?? 0);
              stageMetrics = upsertRunningStage(stageMetrics, {
                key: 'tune',
                title: 'Tune stocks',
                kind: 'tune',
                step: 3,
                totalSteps: state.mode === 'retune' ? 4 : undefined,
                total: tuneTotal,
              }, now);
              activeStageKey = 'tune';

              const nextBySymbol = { ...modelBySymbol };
              const nextMeta = { ...modelMetaBySymbol };
              let nextAssets = assets;
              const assetSymbols = new Set(nextAssets.map((asset) => asset.symbol));

              for (const [symbol, model] of tuneTelemetry.completions) {
                nextBySymbol[symbol] = model;
                nextMeta[symbol] = {
                  ...nextMeta[symbol],
                  model,
                  updatedAt: now,
                };
                if (!assetSymbols.has(symbol)) {
                  assetSymbols.add(symbol);
                  nextAssets = nextAssets.concat({ symbol, status: 'ok', detail: model, model }).slice(-MAX_ASSETS);
                }
              }

              const nextCounts: Record<string, number> = {};
              for (const model of Object.values(nextBySymbol)) {
                nextCounts[model] = (nextCounts[model] ?? 0) + 1;
              }
              const done = Object.keys(nextBySymbol).length;
              stageMetrics = updateStageCounts(stageMetrics, 'tune', { done, total: tuneTotal }, now, { exact: true });
              counters = { ...counters, done, total: Math.max(counters.total, tuneTotal) };
              assets = nextAssets;
              modelBySymbol = nextBySymbol;
              modelMetaBySymbol = nextMeta;
              modelCounts = nextCounts;
            }
          }

          return {
            refreshPass,
            stageMetrics,
            activeStageKey,
            counters,
            assets,
            modelBySymbol,
            modelMetaBySymbol,
            modelCounts,
          };
        });
      }
      if (logFlushTimer !== null) return;
      logFlushTimer = window.setTimeout(() => {
        const batch = pendingLogLines;
        pendingLogLines = [];
        logFlushTimer = null;
        if (batch.length === 0) return;
        set((state) => {
          if (state.status !== 'running') return state;
          return { logLines: state.logLines.concat(batch).slice(-MAX_LOGS) };
        });
        schedulePersist();
      }, 180);
    };

    set({
      mode,
      status: 'running',
      phases: [{
        id: 0,
        title: `Launching ${launchInfo.shortTitle.toLowerCase()} pipeline`,
        kind: 'launch',
        startedAt,
      }],
      assets: [],
      logLines: [],
      counters: { done: 0, fail: 0, total: 0 },
      stageMetrics: [],
      activeStageKey: null,
      elapsedSec: 0,
      startedAt,
      finishedAt: null,
      errorMsg: null,
      refreshPass: null,
      modelBySymbol: {},
      modelMetaBySymbol: {},
      modelCounts: {},
      surfaceVisible: true,
      expanded: false,
      rawLogOpen: false,
    });

    elapsedTimer = window.setInterval(() => {
      const state = get();
      if (state.status !== 'running' || !state.startedAt) return;
      const elapsedSec = Math.floor((Date.now() - state.startedAt) / 1000);
      set({ elapsedSec });
      persistSnapshot(get());
    }, 1000);

    const es = new EventSource(`/api/tune/retune/stream?mode=${mode}`);
    eventSource = es;

    es.onmessage = (ev) => {
      let data: JobEvent;
      try {
        data = JSON.parse(ev.data) as JobEvent;
      } catch {
        return;
      }

      if (data.type === 'log') {
        if (data.message) enqueueLogLine(data.message);
        return;
      }

      set((state) => {
        if (state.status !== 'running') return state;
        const next: Partial<JobState> = {};

        switch (data.type) {
          case 'start':
            if (typeof data.total === 'number') {
              next.counters = { ...state.counters, total: data.total };
            }
            break;

          case 'phase':
            if (data.title) {
              const now = Date.now();
              phaseId += 1;
              next.phases = state.phases.concat({
                id: phaseId,
                title: data.title,
                step: data.step,
                totalSteps: data.total_steps,
                kind: data.kind,
                startedAt: now,
              }).slice(-MAX_PHASES);
              const metricKind = metricKindFromPhase(data.kind, data.title);
              if (metricKind) {
                const key = stageKeyFor(metricKind, data.step);
                next.activeStageKey = key;
                next.stageMetrics = upsertRunningStage(state.stageMetrics, {
                  key,
                  title: stageTitleFor(metricKind, data.title),
                  kind: metricKind,
                  step: metricKind === 'calibration' ? 4 : data.step,
                  totalSteps: metricKind === 'calibration' ? Math.max(data.total_steps ?? 4, 4) : data.total_steps,
                  total: metricKind === 'backup' ? 1 : data.total,
                }, now);
              }
              if (data.kind !== 'download') {
                next.refreshPass = null;
              }
            }
            break;

          case 'asset': {
            const now = Date.now();
            const activeStage = state.stageMetrics.find((stage) => stage.key === state.activeStageKey);
            const isDownloadAsset = activeStage?.kind === 'download';
            const counters = {
              done: data.done ?? state.counters.done,
              fail: data.fail ?? state.counters.fail,
              total: data.total ?? state.counters.total,
            };
            if (!isDownloadAsset) {
              next.counters = counters;
              next.stageMetrics = updateStageCounts(
                next.stageMetrics ?? state.stageMetrics,
                state.activeStageKey,
                counters,
                now,
              );
            } else if (data.status === 'fail') {
              next.stageMetrics = updateStageCounts(
                next.stageMetrics ?? state.stageMetrics,
                state.activeStageKey,
                { fail: (activeStage?.fail ?? 0) + 1 },
                now,
              );
            }
            if (data.symbol) {
              const entry: JobAssetEntry = {
                symbol: data.symbol,
                status: data.status === 'fail' ? 'fail' : 'ok',
                detail: data.detail,
                model: data.model,
                weightPct: data.weight_pct,
                bic: data.bic,
                hyv: data.hyv,
                crps: data.crps,
                pitP: data.pit_p,
                fitStatus: data.fit_status,
              };
              if (!isDownloadAsset || data.status === 'fail' || data.model) {
                next.assets = state.assets.concat(entry).slice(-MAX_ASSETS);
              }
              if (data.model) {
                const oldModel = state.modelBySymbol[data.symbol];
                const nextBySymbol = { ...state.modelBySymbol, [data.symbol]: data.model };
                const nextCounts = { ...state.modelCounts };
                if (oldModel !== data.model) {
                  if (oldModel && nextCounts[oldModel] != null) {
                    const decremented = nextCounts[oldModel] - 1;
                    if (decremented > 0) nextCounts[oldModel] = decremented;
                    else delete nextCounts[oldModel];
                  }
                  nextCounts[data.model] = (nextCounts[data.model] ?? 0) + 1;
                }
                next.modelBySymbol = nextBySymbol;
                const existingMeta = state.modelMetaBySymbol[data.symbol];
                next.modelMetaBySymbol = {
                  ...state.modelMetaBySymbol,
                  [data.symbol]: {
                    model: data.model,
                    weightPct: data.weight_pct ?? existingMeta?.weightPct,
                    bic: data.bic ?? existingMeta?.bic,
                    hyv: data.hyv ?? existingMeta?.hyv,
                    crps: data.crps ?? existingMeta?.crps,
                    pitP: data.pit_p ?? existingMeta?.pitP,
                    fitStatus: data.fit_status ?? existingMeta?.fitStatus,
                    updatedAt: Date.now(),
                  },
                };
                next.modelCounts = nextCounts;
              }
            }
            break;
          }

          case 'model':
            if (data.symbol && data.model) {
              const oldModel = state.modelBySymbol[data.symbol];
              if (oldModel !== data.model) {
                const nextCounts = { ...state.modelCounts };
                if (oldModel && nextCounts[oldModel] != null) {
                  const decremented = nextCounts[oldModel] - 1;
                  if (decremented > 0) nextCounts[oldModel] = decremented;
                  else delete nextCounts[oldModel];
                }
                nextCounts[data.model] = (nextCounts[data.model] ?? 0) + 1;
                next.modelCounts = nextCounts;
              }
              next.modelBySymbol = { ...state.modelBySymbol, [data.symbol]: data.model };
              const existingMeta = state.modelMetaBySymbol[data.symbol];
              next.modelMetaBySymbol = {
                ...state.modelMetaBySymbol,
                [data.symbol]: {
                  model: data.model,
                  weightPct: data.weight_pct ?? existingMeta?.weightPct,
                  bic: data.bic ?? existingMeta?.bic,
                  hyv: data.hyv ?? existingMeta?.hyv,
                  crps: data.crps ?? existingMeta?.crps,
                  pitP: data.pit_p ?? existingMeta?.pitP,
                  fitStatus: data.fit_status ?? existingMeta?.fitStatus,
                  updatedAt: Date.now(),
                },
              };
            }
            break;

          case 'refresh':
            if (typeof data.pass === 'number') {
              const now = Date.now();
              const ok = data.ok ?? state.refreshPass?.ok ?? 0;
              const pending = data.pending ?? state.refreshPass?.pending ?? 0;
              const refreshTotal = ok + pending;
              next.refreshPass = {
                pass: data.pass,
                totalPasses: data.total_passes ?? state.refreshPass?.totalPasses ?? 1,
                ok,
                pending,
              };
              if (refreshTotal > 0) {
                next.stageMetrics = updateStageCounts(
                  next.stageMetrics ?? state.stageMetrics,
                  'download',
                  { done: ok, total: refreshTotal },
                  now,
                  { exact: true },
                );
              }
            }
            break;

          case 'progress':
            {
              const now = Date.now();
              const key = data.kind ? stageKeyFor(data.kind, data.step) : state.activeStageKey;
              const done = data.done ?? 0;
              const fail = data.fail ?? 0;
              const total = data.total ?? Math.max(done + fail, state.counters.total);
              next.stageMetrics = updateStageCounts(
                next.stageMetrics ?? state.stageMetrics,
                key,
                { done, fail, total },
                now,
                { exact: true },
              );
              if (key === state.activeStageKey || data.kind === 'signals') {
                next.counters = { done, fail, total };
              }
            }
            break;

          case 'heartbeat':
            {
              const now = Date.now();
              const activeStage = state.stageMetrics.find((stage) => stage.key === state.activeStageKey);
              if (activeStage?.kind === 'tune' || activeStage?.kind === 'calibration' || activeStage?.kind === 'work' || activeStage?.kind === 'launch') {
                const nextDone = Math.max(activeStage?.done ?? 0, data.done ?? 0);
                const nextTotal = Math.max(activeStage?.total ?? 0, data.total ?? 0);
                next.stageMetrics = updateStageCounts(
                  next.stageMetrics ?? state.stageMetrics,
                  state.activeStageKey,
                  { done: nextDone, total: nextTotal },
                  now,
                );
                next.counters = {
                  done: Math.max(state.counters.done, data.done ?? 0),
                  fail: state.counters.fail,
                  total: Math.max(state.counters.total, data.total ?? 0),
                };
              }
            if (typeof data.elapsed_s === 'number') next.elapsedSec = data.elapsed_s;
            }
            break;

          case 'error':
            next.errorMsg = data.message || data.error_type || data.error || 'Unknown job error';
            break;

          case 'completed':
          case 'failed':
            cleanupStream();
            next.status = data.type;
            next.finishedAt = Date.now();
            next.stageMetrics = state.stageMetrics.map((stage) => (
              stage.status === 'running'
                ? { ...stage, status: data.type === 'completed' ? 'completed' : 'failed', updatedAt: Date.now() }
                : stage
            ));
            if (data.type === 'failed') next.errorMsg = state.errorMsg || data.error || 'Job failed';
            if (typeof data.elapsed_s === 'number') next.elapsedSec = data.elapsed_s;
            break;
        }

        return next;
      });
      schedulePersist();
    };

    es.onerror = () => {
      const state = get();
      if (state.status === 'running') {
        set({
          status: 'error',
          finishedAt: Date.now(),
          errorMsg: 'Live stream disconnected. The backend job may still be running.',
          surfaceVisible: true,
        });
        cleanupStream();
        persistSnapshot(get());
      }
    };
  },

  stopJob: () => {
    const wasRunning = get().status === 'running';
    cleanupStream();
    set((state) => ({
      status: state.status === 'running' ? 'stopped' : state.status,
      finishedAt: Date.now(),
      errorMsg: state.status === 'running' ? 'Stopped by user' : state.errorMsg,
      stageMetrics: state.stageMetrics.map((stage) => stage.status === 'running' ? { ...stage, status: 'stopped', updatedAt: Date.now() } : stage),
      surfaceVisible: true,
      expanded: false,
    }));
    persistSnapshot(get());
    if (wasRunning) {
      requestBackendCancel();
    }
  },

  dismissSurface: () => {
    set({ surfaceVisible: false, expanded: false });
    persistSnapshot(get());
  },

  showSurface: () => {
    set({ surfaceVisible: true });
    persistSnapshot(get());
  },

  setExpanded: (expanded) => {
    set({ expanded, surfaceVisible: true });
    persistSnapshot(get());
  },

  toggleExpanded: () => {
    set((state) => ({ expanded: !state.expanded, surfaceVisible: true }));
    persistSnapshot(get());
  },

  setRawLogOpen: (rawLogOpen) => {
    set({ rawLogOpen });
  },

  clearTerminalJob: () => {
    const state = get();
    if (state.status === 'running') return;
    cleanupStream();
    set({
      mode: null,
      status: 'idle',
      phases: [],
      assets: [],
      logLines: [],
      counters: { done: 0, fail: 0, total: 0 },
      stageMetrics: [],
      activeStageKey: null,
      elapsedSec: 0,
      startedAt: null,
      finishedAt: null,
      errorMsg: null,
      refreshPass: null,
      modelBySymbol: {},
      modelMetaBySymbol: {},
      modelCounts: {},
      surfaceVisible: false,
      expanded: false,
      rawLogOpen: false,
    });
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
  },
}));

export function formatJobElapsed(raw: number): string {
  const s = Math.max(0, Math.round(Number(raw) || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = (s % 60).toString().padStart(2, '0');
  if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${ss}`;
  return `${m}:${ss}`;
}

export function formatJobDuration(raw: number): string {
  const s = Math.max(0, Math.round(Number(raw) || 0));
  if (s < 60) return `${s}s`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (h > 0) return `${h}h ${m.toString().padStart(2, '0')}m`;
  return ss === 0 ? `${m}m` : `${m}m ${ss.toString().padStart(2, '0')}s`;
}
