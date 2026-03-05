const BASE = '';  // Vite proxy handles /api -> backend

export async function fetchApi<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function postApi<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

// ── Overview ────────────────────────────────────────────────────────
export const api = {
  overview: () => fetchApi<OverviewData>('/api/overview'),
  health: () => fetchApi<{ status: string }>('/api/health'),

  // Signals
  signalSummary: () => fetchApi<SignalSummaryData>('/api/signals/summary'),
  signalStats: () => fetchApi<SignalStats>('/api/signals/stats'),
  signalAssets: () => fetchApi<{ assets: AssetBlock[]; total: number }>('/api/signals/assets'),
  signalFailed: () => fetchApi<{ failed_assets: string[]; count: number }>('/api/signals/failed'),
  highConviction: (type: 'buy' | 'sell') =>
    fetchApi<{ signals: HighConvictionSignal[]; count: number }>(`/api/signals/high-conviction/${type}`),

  // Risk
  riskDashboard: () => fetchApi<RiskDashboard>('/api/risk/dashboard'),
  riskSummary: () => fetchApi<RiskSummary>('/api/risk/summary'),

  // Charts
  chartSymbols: () => fetchApi<{ symbols: string[]; count: number }>('/api/charts/symbols'),
  chartOhlcv: (symbol: string, tail = 365) =>
    fetchApi<{ symbol: string; data: OHLCVBar[]; count: number }>(`/api/charts/ohlcv/${symbol}?tail=${tail}`),
  chartIndicators: (symbol: string, tail = 365) =>
    fetchApi<{ symbol: string; indicators: Indicators }>(`/api/charts/indicators/${symbol}?tail=${tail}`),
  chartForecast: (symbol: string) =>
    fetchApi<ForecastData>(`/api/charts/forecast/${symbol}`),
  chartImages: () => fetchApi<{ images: ChartImage[]; count: number }>('/api/charts/images'),

  // Tuning
  tuneList: () => fetchApi<{ assets: TuneAsset[]; total: number }>('/api/tune/list'),
  tuneStats: () => fetchApi<TuneStats>('/api/tune/stats'),
  tuneDetail: (symbol: string) => fetchApi<TuneDetail>(`/api/tune/detail/${symbol}`),
  pitFailures: () => fetchApi<{ failures: TuneAsset[]; count: number }>('/api/tune/pit-failures'),

  // Data
  dataStatus: () => fetchApi<DataSummary>('/api/data/status'),
  dataPrices: () => fetchApi<{ files: PriceFile[]; total: number }>('/api/data/prices'),
  dataDirectories: () => fetchApi<Record<string, DirInfo>>('/api/data/directories'),

  // Arena
  arenaStatus: () => fetchApi<ArenaStatus>('/api/arena/status'),
  arenaSafeStorage: () => fetchApi<{ models: SafeStorageModel[]; count: number }>('/api/arena/safe-storage'),
  arenaResults: () => fetchApi<Record<string, unknown>>('/api/arena/results'),

  // Tasks
  triggerSignals: (args?: string[]) => postApi<TaskResponse>('/api/tasks/signals/compute', { args }),
  triggerDataRefresh: (symbols?: string[]) => postApi<TaskResponse>('/api/tasks/data/refresh', { symbols }),
  triggerTuning: (symbols?: string[]) => postApi<TaskResponse>('/api/tasks/tune/run', { symbols }),
  triggerRisk: () => postApi<TaskResponse>('/api/tasks/risk/compute'),
  triggerCharts: () => postApi<TaskResponse>('/api/tasks/charts/generate'),
  taskStatus: (taskId: string) => fetchApi<TaskStatusResponse>(`/api/tasks/status/${taskId}`),
};

// ── Types ───────────────────────────────────────────────────────────

export interface OverviewData {
  signals: SignalStats;
  tuning: TuneStats;
  data: DataSummary;
}

export interface SignalStats {
  cached: boolean;
  total_assets: number;
  failed: number;
  buy_signals: number;
  sell_signals: number;
  hold_signals: number;
  cache_age_seconds: number | null;
}

export interface SignalSummaryData {
  summary_rows: SummaryRow[];
  horizons: number[];
  total: number;
}

export interface SummaryRow {
  asset_label: string;
  horizon_signals: Record<string, HorizonSignal>;
  nearest_label: string;
  sector: string;
  crash_risk_score: number;
  momentum_score: number;
}

export interface HorizonSignal {
  label: string;
  profit_pln: number;
  p_up: number;
  exp_ret: number;
  ue_up: number;
  ue_down: number;
}

export interface AssetBlock {
  symbol: string;
  title: string;
  signals: Record<string, unknown>[];
  [key: string]: unknown;
}

export interface HighConvictionSignal {
  ticker: string;
  asset_label: string;
  sector: string;
  horizon_days: number;
  signal_type: string;
  probability_up: number;
  probability_down: number;
  expected_return_pct: number;
  expected_profit_pln: number;
  [key: string]: unknown;
}

export interface RiskDashboard {
  risk_temperature: Record<string, unknown>;
  metals_risk_temperature: Record<string, unknown>;
  market_temperature: Record<string, unknown>;
  computed_at: string;
}

export interface RiskSummary {
  combined_temperature: number;
  status: string;
  risk_temperature: number;
  metals_temperature: number;
  market_temperature: number;
  computed_at: string;
}

export interface OHLCVBar {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Indicators {
  sma20?: { time: string; value: number }[];
  sma50?: { time: string; value: number }[];
  sma200?: { time: string; value: number }[];
  bollinger?: {
    upper: { time: string; value: number }[];
    lower: { time: string; value: number }[];
  };
  rsi?: { time: string; value: number }[];
  atr?: { time: string; value: number }[];
}

export interface ForecastData {
  symbol: string;
  asset_label: string;
  forecasts: {
    horizon_days: number;
    expected_return_pct: number;
    probability_up: number;
    signal_label: string;
  }[];
}

export interface ChartImage {
  filename: string;
  category: string;
  url: string;
}

export interface TuneAsset {
  symbol: string;
  best_model: string;
  pit_calibration_grade: string;
  ad_stat: number | null;
  ad_critical: number | null;
  ad_pass: boolean | null;
  num_models: number;
  cache_version: string;
  last_tuned: string;
  file_size_kb: number;
}

export interface TuneStats {
  total: number;
  pit_pass: number;
  pit_fail: number;
  pit_unknown: number;
  models_distribution: Record<string, number>;
}

export interface TuneDetail {
  symbol: string;
  data: Record<string, unknown>;
}

export interface DataSummary {
  total_files: number;
  stale_files: number;
  fresh_files: number;
  freshest_hours: number | null;
  oldest_hours: number | null;
  total_size_mb: number;
}

export interface PriceFile {
  symbol: string;
  filename: string;
  last_modified: string;
  age_hours: number;
  size_kb: number;
  rows: number;
}

export interface DirInfo {
  path: string;
  file_count: number;
  exists: boolean;
}

export interface ArenaStatus {
  safe_storage_count: number;
  experimental_count: number;
  benchmark_symbols: string[];
}

export interface SafeStorageModel {
  name: string;
  filename: string;
  size_kb: number;
}

export interface TaskResponse {
  task_id: string;
  task_type: string;
  status: string;
}

export interface TaskStatusResponse {
  task_id: string;
  status: string;
  meta?: Record<string, unknown>;
  result?: Record<string, unknown>;
  error?: string;
}
