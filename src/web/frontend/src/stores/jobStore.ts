import { create } from 'zustand';

export type JobMode = 'tune' | 'stocks' | 'retune' | 'calibrate';
export type JobStatus = 'idle' | 'running' | 'completed' | 'failed' | 'error' | 'stopped';

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

interface JobState {
  mode: JobMode | null;
  status: JobStatus;
  phases: JobPhaseEntry[];
  assets: JobAssetEntry[];
  logLines: JobLogLine[];
  counters: JobCounters;
  elapsedSec: number;
  startedAt: number | null;
  finishedAt: number | null;
  errorMsg: string | null;
  refreshPass: JobRefreshPassState | null;
  modelBySymbol: Record<string, string>;
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
const MAX_LOGS = 2000;
const MAX_PHASES = 30;

let eventSource: EventSource | null = null;
let elapsedTimer: number | null = null;
let logId = 0;
let phaseId = 0;

function requestBackendCancel() {
  void fetch('/api/tune/retune/cancel', {
    method: 'POST',
    keepalive: true,
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
      elapsedSec: state.elapsedSec,
      startedAt: state.startedAt,
      finishedAt: state.finishedAt,
      errorMsg: state.errorMsg,
      refreshPass: state.refreshPass,
      modelBySymbol: state.modelBySymbol,
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
    elapsedSec: snapshot?.elapsedSec ?? 0,
    startedAt: snapshot?.startedAt ?? null,
    finishedAt: snapshot?.finishedAt ?? null,
    errorMsg: snapshot?.errorMsg ?? null,
    refreshPass: snapshot?.refreshPass ?? null,
    modelBySymbol: snapshot?.modelBySymbol ?? {},
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
};

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

    set({
      mode,
      status: 'running',
      phases: [],
      assets: [],
      logLines: [],
      counters: { done: 0, fail: 0, total: 0 },
      elapsedSec: 0,
      startedAt,
      finishedAt: null,
      errorMsg: null,
      refreshPass: null,
      modelBySymbol: {},
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

      set((state) => {
        if (state.status !== 'running' && data.type !== 'completed' && data.type !== 'failed') return state;
        const next: Partial<JobState> = {};

        switch (data.type) {
          case 'start':
            if (typeof data.total === 'number') {
              next.counters = { ...state.counters, total: data.total };
            }
            break;

          case 'phase':
            if (data.title) {
              phaseId += 1;
              next.phases = state.phases.concat({
                id: phaseId,
                title: data.title,
                step: data.step,
                totalSteps: data.total_steps,
                kind: data.kind,
                startedAt: Date.now(),
              }).slice(-MAX_PHASES);
            }
            break;

          case 'asset': {
            const counters = {
              done: data.done ?? state.counters.done,
              fail: data.fail ?? state.counters.fail,
              total: data.total ?? state.counters.total,
            };
            next.counters = counters;
            if (data.symbol) {
              const entry: JobAssetEntry = {
                symbol: data.symbol,
                status: data.status === 'fail' ? 'fail' : 'ok',
                detail: data.detail,
                model: data.model,
              };
              next.assets = state.assets.concat(entry).slice(-MAX_ASSETS);
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
            }
            break;

          case 'refresh':
            if (typeof data.pass === 'number') {
              next.refreshPass = {
                pass: data.pass,
                totalPasses: data.total_passes ?? state.refreshPass?.totalPasses ?? 1,
                ok: data.ok ?? state.refreshPass?.ok ?? 0,
                pending: data.pending ?? state.refreshPass?.pending ?? 0,
              };
            }
            break;

          case 'heartbeat':
            next.counters = {
              done: data.done ?? state.counters.done,
              fail: state.counters.fail,
              total: data.total ?? state.counters.total,
            };
            if (typeof data.elapsed_s === 'number') next.elapsedSec = data.elapsed_s;
            break;

          case 'log':
            if (data.message) {
              logId += 1;
              next.logLines = state.logLines.concat({ id: logId, text: data.message }).slice(-MAX_LOGS);
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
            if (data.type === 'failed') next.errorMsg = state.errorMsg || data.error || 'Job failed';
            if (typeof data.elapsed_s === 'number') next.elapsedSec = data.elapsed_s;
            break;
        }

        return next;
      });
      persistSnapshot(get());
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
    if (get().status === 'running') {
      requestBackendCancel();
    }
    cleanupStream();
    set((state) => ({
      status: state.status === 'running' ? 'stopped' : state.status,
      finishedAt: Date.now(),
      errorMsg: state.status === 'running' ? 'Stopped by user' : state.errorMsg,
      surfaceVisible: true,
      expanded: false,
    }));
    persistSnapshot(get());
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
      elapsedSec: 0,
      startedAt: null,
      finishedAt: null,
      errorMsg: null,
      refreshPass: null,
      modelBySymbol: {},
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
