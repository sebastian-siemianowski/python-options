/**
 * Module-level store for retune state.
 * Persists across page navigation since it lives outside React component lifecycle.
 */

export type RetuneStatus = 'idle' | 'running' | 'completed' | 'failed';

export interface RetuneLogEntry {
  type: string;
  message: string;
  count: number;
  success: number;
  fail: number;
}

interface RetuneState {
  status: RetuneStatus;
  logs: RetuneLogEntry[];
  mode: 'retune' | 'tune' | 'calibrate';
  showPanel: boolean;
  startedAt: number | null;
  finishedAt: number | null;
  totalAssets: number;
  successCount: number;
  failCount: number;
  currentAsset: string | null;
  currentPhase: string | null;
}

type Listener = () => void;

const MAX_LOGS = 500;

let state: RetuneState = {
  status: 'idle',
  logs: [],
  mode: 'retune',
  showPanel: false,
  startedAt: null,
  finishedAt: null,
  totalAssets: 0,
  successCount: 0,
  failCount: 0,
  currentAsset: null,
  currentPhase: null,
};

let eventSource: EventSource | null = null;
const listeners = new Set<Listener>();

function emit() {
  state = { ...state };
  listeners.forEach((l) => l());
}

export function getRetuneSnapshot(): RetuneState {
  return state;
}

export function subscribeRetune(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function setRetuneMode(mode: RetuneState['mode']) {
  state.mode = mode;
  emit();
}

export function setShowPanel(show: boolean) {
  state.showPanel = show;
  emit();
}

function addLog(entry: RetuneLogEntry) {
  const newLogs = [...state.logs, entry];
  // Cap logs to prevent memory bloat and rendering hang
  state.logs = newLogs.length > MAX_LOGS ? newLogs.slice(-MAX_LOGS) : newLogs;
}

function extractAssetName(message: string): string | null {
  // Match ticker-like patterns: 1-5 uppercase letters, optionally with =X, -USD, .PA etc.
  const m = message.match(/\b([A-Z][A-Z0-9]{0,5}(?:[=\-\.][A-Z0-9]+)?)\b/);
  return m ? m[1] : null;
}

export function startRetune(onComplete?: () => void) {
  if (state.status === 'running') return;

  state.status = 'running';
  state.logs = [];
  state.showPanel = true;
  state.startedAt = Date.now();
  state.finishedAt = null;
  state.totalAssets = 0;
  state.successCount = 0;
  state.failCount = 0;
  state.currentAsset = null;
  state.currentPhase = null;
  emit();

  const es = new EventSource(`/api/tune/retune/stream?mode=${state.mode}`);
  eventSource = es;

  es.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as RetuneLogEntry;
      addLog(data);

      if (data.type === 'progress') {
        state.totalAssets = data.count || state.totalAssets;
        state.successCount = data.success || state.successCount;
        state.failCount = data.fail || state.failCount;
        state.currentAsset = extractAssetName(data.message);
      } else if (data.type === 'phase') {
        state.currentPhase = data.message;
      }

      if (data.type === 'completed' || data.type === 'failed' || data.type === 'error') {
        state.status = data.type === 'completed' ? 'completed' : 'failed';
        state.finishedAt = Date.now();
        state.totalAssets = data.count || state.totalAssets;
        state.successCount = data.success || state.successCount;
        state.failCount = data.fail || state.failCount;
        state.currentAsset = null;
        es.close();
        eventSource = null;
        onComplete?.();
      }
      emit();
    } catch {
      // ignore parse errors
    }
  };

  es.onerror = () => {
    if (state.status === 'running') {
      state.status = 'failed';
      state.finishedAt = Date.now();
      addLog({ type: 'error', message: 'Connection lost', count: 0, success: 0, fail: 0 });
      emit();
    }
    es.close();
    eventSource = null;
  };
}

export function stopRetune() {
  eventSource?.close();
  eventSource = null;
  state.status = 'idle';
  state.finishedAt = Date.now();
  addLog({ type: 'log', message: 'Cancelled by user', count: 0, success: 0, fail: 0 });
  emit();
}
