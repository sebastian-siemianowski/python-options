/**
 * Module-level store for retune state.
 * Persists across page navigation since it lives outside React component lifecycle.
 */

export type RetuneStatus = 'idle' | 'running' | 'completed' | 'failed';

export interface RetuneLogEntry {
  type: string;
  message: string;
  count: number;
}

interface RetuneState {
  status: RetuneStatus;
  logs: RetuneLogEntry[];
  mode: 'retune' | 'tune' | 'calibrate';
  showPanel: boolean;
}

type Listener = () => void;

let state: RetuneState = {
  status: 'idle',
  logs: [],
  mode: 'retune',
  showPanel: false,
};

let eventSource: EventSource | null = null;
const listeners = new Set<Listener>();

function emit() {
  // Create a new state reference so React detects the change
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

export function startRetune(onComplete?: () => void) {
  if (state.status === 'running') return;

  state.status = 'running';
  state.logs = [];
  state.showPanel = true;
  emit();

  const es = new EventSource(`/api/tune/retune/stream?mode=${state.mode}`);
  eventSource = es;

  es.onmessage = (event) => {
    try {
      const data: RetuneLogEntry = JSON.parse(event.data);
      state.logs = [...state.logs, data];

      if (data.type === 'completed' || data.type === 'failed' || data.type === 'error') {
        state.status = data.type === 'completed' ? 'completed' : 'failed';
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
      state.logs = [...state.logs, { type: 'error', message: 'Connection lost', count: 0 }];
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
  state.logs = [...state.logs, { type: 'log', message: 'Cancelled by user', count: 0 }];
  emit();
}
