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
  qualityScores: () => fetchApi<QualityScoresData>('/api/signals/quality-scores'),
  intrinsicValues: () => fetchApi<IntrinsicValuesData>('/api/signals/intrinsic-values'),

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

  // Cache refresh (invalidate in-memory server cache)
  refreshTuneCache: () => postApi<{ status: string }>('/api/tuning/refresh-cache', {}),
  refreshSignalCache: () => postApi<{ status: string }>('/api/signals/refresh-cache', {}),

  // Diagnostics
  diagPitSummary: () => fetchApi<DiagPitSummary>('/api/diagnostics/pit-summary'),
  diagCalibrationFailures: () => fetchApi<DiagCalibrationFailures>('/api/diagnostics/calibration-failures'),
  diagModelComparison: () => fetchApi<DiagModelComparison>('/api/diagnostics/model-comparison'),
  diagRegimeDistribution: () => fetchApi<DiagRegimeDistribution>('/api/diagnostics/regime-distribution'),
  diagCrossAssetSummary: () => fetchApi<DiagCrossAssetSummary>('/api/diagnostics/cross-asset-summary'),
  diagProfitability: () => fetchApi<ProfitabilityMetrics>('/api/diagnostics/profitability'),

  // Risk (full dashboard + refresh)
  riskRefresh: () => postApi<{ status: string; summary: RiskSummary }>('/api/risk/refresh'),

  // Indicators
  indicatorsLeaderboard: (top = 0, family?: string) => {
    const params = new URLSearchParams();
    if (top > 0) params.set('top', String(top));
    if (family) params.set('family', family);
    return fetchApi<IndicatorsLeaderboard>(`/api/indicators/leaderboard?${params}`);
  },
  indicatorsTop10: () => fetchApi<IndicatorStrategy[]>('/api/indicators/top10'),
  indicatorsFamilies: () => fetchApi<IndicatorFamily[]>('/api/indicators/families'),
  indicatorsStrategy: (id: number) => fetchApi<IndicatorStrategyDetail>(`/api/indicators/strategy/${id}`),
  indicatorsHeatmap: (id: number) => fetchApi<IndicatorHeatmap>(`/api/indicators/strategy/${id}/heatmap`),
  indicatorsRefresh: () => postApi<{ status: string }>('/api/indicators/refresh'),
  indicatorsRunBacktest: (mode: 'quick' | 'full' = 'full') =>
    postApi<IndicatorBacktestStart>(`/api/indicators/backtest?mode=${mode}`),
  indicatorsBacktestStatus: () => fetchApi<IndicatorBacktestStatus>('/api/indicators/backtest/status'),
};

// ── Types ───────────────────────────────────────────────────────────

export interface QualityFormulaComponent {
  name: string;
  weight: number;
  desc: string;
}
export interface QualityFormulaTier {
  range: string;
  label: string;
  desc: string;
}
export interface QualityFormula {
  title: string;
  description: string;
  components: QualityFormulaComponent[];
  tiers: QualityFormulaTier[];
  non_company_notes: Record<string, string>;
}
export interface QualityScoresData {
  scores: Record<string, number>;
  formula: QualityFormula;
}

export interface IntrinsicValuation {
  intrinsic_value: number | null;
  price: number | null;
  gap_pct: number | null;
}
export interface IntrinsicMethodStep {
  step: string;
  desc: string;
}
export interface IntrinsicFormula {
  title: string;
  description: string;
  methodology: IntrinsicMethodStep[];
  non_company_methods: Record<string, string>;
  interpretation: Record<string, string>;
}
export interface IntrinsicValuesData {
  valuations: Record<string, IntrinsicValuation>;
  formula: IntrinsicFormula;
}

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

export interface KellyHorizon {
  horizon: number;
  half_kelly: number;
  capped_size: number;
  edge: number;
}

export interface SummaryRow {
  asset_label: string;
  horizon_signals: Record<string, HorizonSignal>;
  nearest_label: string;
  sector: string;
  crash_risk_score: number;
  momentum_score: number;
  conviction?: number;
  kelly?: KellyHorizon[];
  signal_ttl?: unknown;
}

export interface HorizonSignal {
  label: string;
  profit_pln: number;
  p_up: number;
  exp_ret: number;
  ue_up: number;
  ue_down: number;
  position_strength?: number;
  risk_temperature?: number;
  kelly_half?: number;
  eu_balanced?: number;
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
  macd?: {
    macd: { time: string; value: number }[];
    signal: { time: string; value: number }[];
    histogram: { time: string; value: number }[];
  };
  stochastic?: {
    k: { time: string; value: number }[];
    d: { time: string; value: number }[];
  };
  adx?: {
    adx: { time: string; value: number }[];
    plus_di: { time: string; value: number }[];
    minus_di: { time: string; value: number }[];
  };
  obv?: { time: string; value: number }[];
  cci?: { time: string; value: number }[];
  mfi?: { time: string; value: number }[];
  cmf?: { time: string; value: number }[];
  roc?: { time: string; value: number }[];
  bbpctb?: { time: string; value: number }[];
  composite?: { time: string; value: number }[];
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
  has_scores?: boolean;
  final?: number | null;
  bic?: number | null;
  crps?: number | null;
  hyv?: number | null;
  pit?: string | null;
  pit_rate?: number | null;
  css?: number | null;
  fec?: number | null;
  time_ms?: number | null;
  n_tests?: number | null;
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

// ── Diagnostics ─────────────────────────────────────────────────────

export interface DiagModelMetric {
  model: string;
  bic: number | null;
  crps: number | null;
  hyvarinen: number | null;
  pit_ks_pvalue: number | null;
  ad_pvalue: number | null;
  histogram_mad: number | null;
  weight: number;
  nu: number | null;
  phi: number | null;
}

export interface DiagAsset {
  symbol: string;
  best_model: string;
  pit_grade: string;
  ad_stat: number | null;
  ad_critical: number | null;
  ad_pass: boolean | null;
  pit_ks_pvalue: number | null;
  num_models: number;
  bma_weights: Record<string, number>;
  models: DiagModelMetric[];
  regime: string | null;
  last_tuned: string;
}

export interface DiagPitSummary {
  assets: DiagAsset[];
  total: number;
  passing: number;
  failing: number;
  unknown: number;
  computed_at: string;
}

export interface DiagCalibrationFailures {
  failures: Array<Record<string, unknown>>;
  count: number;
  file_exists: boolean;
  error?: string;
}

export interface DiagModelStats {
  name: string;
  win_count: number;
  total_weight: number;
  appearances: number;
  avg_weight: number;
  win_rate: number;
  max_weight: number;
  min_weight: number;
}

export interface DiagModelComparison {
  models: Record<string, DiagModelStats>;
  total_assets: number;
  computed_at: string;
}

export interface DiagRegimeInfo {
  count: number;
  percentage: number;
  assets: string[];
}

export interface DiagRegimeDistribution {
  regimes: Record<string, DiagRegimeInfo>;
  total: number;
  computed_at: string;
}

export interface DiagCrossAssetModelScore {
  crps: number | null;
  pit_ks_p: number | null;
  ad_p: number | null;
  bic: number | null;
  hyv: number | null;
  weight: number;
}

export interface DiagCrossAssetRow {
  symbol: string;
  best_model: string;
  regime: string | null;
  ad_pass: boolean | null;
  scores: Record<string, DiagCrossAssetModelScore | null>;
}

export interface DiagCrossAssetSummary {
  rows: DiagCrossAssetRow[];
  models: string[];
  model_averages: Record<string, { avg_crps: number | null; avg_pit_p: number | null; avg_bic: number | null; count: number }>;
  total: number;
  computed_at: string;
}

// ── Risk Full Dashboard ─────────────────────────────────────────────

export interface RiskStressIndicator {
  name: string;
  value: number | null;
  zscore: number | null;
  contribution: number;
  data_available: boolean;
  interpretation?: string;
}

export interface RiskStressCategory {
  name: string;
  weight: number;
  stress_level: number;
  weighted_contribution: number;
  indicators: RiskStressIndicator[];
}

export interface MetalDetail {
  name: string;
  price: number | null;
  return_1d: number;
  return_5d: number;
  return_21d: number;
  volatility: number;
  stress_level: number;
  momentum_signal: string;
  data_available: boolean;
  forecast_7d: number;
  forecast_30d: number;
  forecast_90d: number;
  forecast_180d: number;
  forecast_365d: number;
  forecast_confidence: string;
}

export interface UniverseMetrics {
  name: string;
  weight: number;
  current_level: number | null;
  return_1d: number;
  return_5d: number;
  return_21d: number;
  return_63d: number;
  volatility_20d: number;
  volatility_percentile: number;
  vol_term_structure_ratio: number;
  vol_inverted: boolean;
  breadth_pct_above_50ma: number | null;
  breadth_pct_above_200ma: number | null;
  stress_level: number;
  stress_contribution: number;
  momentum_signal: string;
  data_available: boolean;
  forecast_7d: number;
  forecast_30d: number;
  forecast_90d: number;
  forecast_180d: number;
  forecast_365d: number;
  forecast_confidence: string;
}

export interface SectorMetrics {
  name: string;
  ticker: string;
  return_1d: number;
  return_5d: number;
  return_21d: number;
  volatility_20d: number;
  volatility_percentile: number;
  momentum_signal: string;
  risk_score: number;
  data_available: boolean;
  forecast_7d: number;
  forecast_30d: number;
  forecast_90d: number;
  forecast_180d: number;
  forecast_365d: number;
  forecast_confidence: string;
}

export interface CurrencyMetrics {
  name: string;
  ticker: string;
  rate: number;
  return_1d: number;
  return_5d: number;
  return_21d: number;
  volatility_20d: number;
  momentum_signal: string;
  risk_score: number;
  data_available: boolean;
  forecast_7d: number;
  forecast_30d: number;
  forecast_90d: number;
  forecast_180d: number;
  forecast_365d: number;
  forecast_confidence: string;
  is_inverse: boolean;
}

export interface MarketBreadth {
  pct_above_50ma: number;
  pct_above_200ma: number;
  new_highs: number;
  new_lows: number;
  advance_decline_ratio: number;
  breadth_thrust: boolean;
  breadth_warning: boolean;
  interpretation: string;
}

export interface CorrelationStress {
  avg_correlation: number;
  max_correlation: number;
  correlation_percentile: number;
  systemic_risk_elevated: boolean;
  interpretation: string;
}

export interface RiskDashboardFull {
  risk_temperature: {
    temperature: number;
    scale_factor: number;
    overnight_budget_active: boolean;
    computed_at: string;
    data_quality: number;
    categories: Record<string, RiskStressCategory>;
    crash_risk_pct: number;
    crash_risk_level: string;
  };
  metals_risk_temperature: {
    temperature: number;
    scale_factor: number;
    status: string;
    action_text: string;
    computed_at: string;
    data_quality: number;
    indicators: RiskStressIndicator[];
    metals: Record<string, MetalDetail>;
    crash_risk_pct: number;
    crash_risk_level: string;
    regime_state: string;
  };
  market_temperature: {
    temperature: number;
    scale_factor: number;
    status: string;
    action_text: string;
    computed_at: string;
    data_quality: number;
    universes: Record<string, UniverseMetrics>;
    breadth: MarketBreadth;
    correlation: CorrelationStress;
    crash_risk_pct: number;
    crash_risk_level: string;
    sectors: Record<string, SectorMetrics>;
    currencies: Record<string, CurrencyMetrics>;
    overall_momentum: string;
    exit_signal: boolean;
    exit_reason: string | null;
  };
  computed_at: string;
  _cached?: boolean;
  _cache_age_seconds?: number;
}

// Story 8.3: Profitability monitoring
export interface ProfitabilityMetrics {
  timestamps: string[];
  hit_rates: { '7d': number[]; '21d': number[] };
  signal_rates: number[];
  sharpe: { '7d': number[]; '21d': number[] };
  crps: number[];
  ece: number[];
  targets: Record<string, number>;
}

// ── Indicators ──────────────────────────────────────────────────────
export interface IndicatorStrategy {
  rank: number;
  id: number;
  name: string;
  family: string;
  composite: number;
  sharpe: number | null;
  sortino: number | null;
  cagr: number | null;
  bh_cagr: number | null;
  cagr_diff: number | null;
  max_dd: number | null;
  buy_hit: number | null;
  sell_hit: number | null;
  win_rate: number | null;
  profit_factor: number | null;
  exposure: number | null;
  n_trades: number | null;
  n_assets: number;
  sharpe_beat_bh: string | null;
}

export interface IndicatorsLeaderboard {
  strategies: IndicatorStrategy[];
  total: number;
}

export interface IndicatorFamily {
  name: string;
  count: number;
  avg_composite: number;
  ids: number[];
}

export interface IndicatorAssetResult {
  symbol: string;
  sharpe: number;
  cagr: number;
  max_dd: number;
  total_return: number;
  win_rate: number | null;
  n_trades: number;
}

export interface IndicatorStrategyDetail {
  id: number;
  name: string;
  family: string;
  aggregate: Record<string, unknown>;
  per_asset: IndicatorAssetResult[];
}

export interface IndicatorHeatmap {
  id: number;
  name: string;
  assets: IndicatorAssetResult[];
}

export interface IndicatorBacktestStart {
  status: string;
  mode?: string;
  started_at?: number;
  progress?: string;
}

export interface IndicatorBacktestStatus {
  running: boolean;
  pid: number | null;
  started_at: number | null;
  finished_at: number | null;
  exit_code: number | null;
  progress: string;
  error: string | null;
  mode: string | null;
  elapsed_seconds: number | null;
}
