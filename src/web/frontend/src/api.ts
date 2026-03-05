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

// ── API client ──────────────────────────────────────────────────────
export const api = {
  overview: () => fetchApi<OverviewData>('/api/overview'),
  health: () => fetchApi<{ status: string; service: string }>('/api/health'),

  // Signals
  signalSummary: () => fetchApi<SignalSummaryData>('/api/signals/summary'),
  signalStats: () => fetchApi<SignalStats>('/api/signals/stats'),
  signalAssets: () => fetchApi<{ assets: AssetBlock[]; total: number }>('/api/signals/assets'),
  signalFailed: () => fetchApi<{ failed_assets: string[]; count: number }>('/api/signals/failed'),
  signalsBySector: () => fetchApi<SectorSignalsData>('/api/signals/by-sector'),
  strongSignals: () => fetchApi<StrongSignalsData>('/api/signals/strong-signals'),
  highConviction: (type: 'buy' | 'sell') =>
    fetchApi<{ signals: HighConvictionSignal[]; count: number }>(`/api/signals/high-conviction/${type}`),

  // Risk
  riskDashboard: () => fetchApi<RiskDashboard>('/api/risk/dashboard'),
  riskSummary: () => fetchApi<RiskSummary>('/api/risk/summary'),

  // Charts
  chartSymbols: () => fetchApi<{ symbols: string[]; count: number }>('/api/charts/symbols'),
  chartSymbolsBySector: () => fetchApi<ChartSectorData>('/api/charts/symbols-by-sector'),
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

  // Services / Health
  servicesHealth: () => fetchApi<ServicesHealth>('/api/services/health'),
  servicesErrors: () => fetchApi<{ errors: ServiceError[]; count: number }>('/api/services/errors'),

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
  errors?: string[];
}

export interface SignalStats {
  cached: boolean;
  total_assets: number;
  failed: number;
  buy_signals: number;
  sell_signals: number;
  hold_signals: number;
  strong_buy_signals: number;
  strong_sell_signals: number;
  exit_signals: number;
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

// ── Sector signals ──────────────────────────────────────────────────
export interface SectorGroup {
  name: string;
  assets: SummaryRow[];
  asset_count: number;
  strong_buy: number;
  buy: number;
  hold: number;
  sell: number;
  strong_sell: number;
  exit: number;
  avg_momentum: number;
  avg_crash_risk: number;
}

export interface SectorSignalsData {
  sectors: SectorGroup[];
  total_sectors: number;
}

export interface StrongSignalEntry {
  symbol: string;
  asset_label: string;
  sector: string;
  horizon: string;
  p_up: number;
  exp_ret: number;
  momentum: number;
}

export interface StrongSignalsData {
  strong_buy: StrongSignalEntry[];
  strong_sell: StrongSignalEntry[];
}

// ── Chart sector grouping ───────────────────────────────────────────
export interface ChartSectorGroup {
  name: string;
  symbols: string[];
  count: number;
}

export interface ChartSectorData {
  sectors: ChartSectorGroup[];
  total_sectors: number;
}

// ── Other types ─────────────────────────────────────────────────────

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

// ── Services / Health ───────────────────────────────────────────────
export interface ServicesHealth {
  api: {
    status: string;
    uptime_seconds: number;
    uptime_human: string;
    memory_mb: number;
    cpu_percent: number;
    pid: number;
  };
  signal_cache: {
    status: string;
    exists: boolean;
    age_seconds: number | null;
    age_human?: string;
    size_mb: number;
    last_modified?: string;
  };
  price_data: {
    status: string;
    total_files: number;
    stale_files: number;
    fresh_files?: number;
    freshest_hours?: number;
    oldest_hours?: number;
    total_size_mb?: number;
  };
  workers: {
    status: string;
    redis: { status: string; used_memory_human?: string; error?: string; message?: string };
    celery: { status: string; workers?: number; worker_names?: string[]; error?: string; message?: string };
  };
  recent_errors: ServiceError[];
}

export interface ServiceError {
  source: string;
  message: string;
  timestamp: string;
}

// ── Tasks ───────────────────────────────────────────────────────────
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
