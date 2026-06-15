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

  // Politicians
  politiciansNotice: () => fetchApi<PoliticiansNoticeResponse>('/api/politicians/notice'),
  politiciansSummary: () => fetchApi<PoliticiansSummaryResponse>('/api/politicians/summary'),
  politiciansSourceHealth: () => fetchApi<PoliticiansSourceHealthResponse>('/api/politicians/source-health'),
  politiciansRefreshCache: () => postApi<{ status: string; cache: string; cleared_entries: number }>('/api/politicians/refresh-cache'),
  politiciansSync: () => postApi<PoliticiansSyncResponse>('/api/politicians/sync', {}),
  politiciansTrades: (params: PoliticiansTradeQuery = {}) => {
    const query = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== '') query.set(key, String(value));
    });
    const suffix = query.toString() ? `?${query.toString()}` : '';
    return fetchApi<PoliticiansTradesResponse>(`/api/politicians/trades${suffix}`);
  },
  politiciansAsset: (symbol: string, windowDays = 180) =>
    fetchApi<PoliticiansAssetResponse>(`/api/politicians/assets/${encodeURIComponent(symbol)}?window_days=${windowDays}`),
  politiciansFiler: (filerId: string, windowDays = 180) =>
    fetchApi<PoliticiansFilerResponse>(`/api/politicians/filers/${encodeURIComponent(filerId)}?window_days=${windowDays}`),

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
  emaStates: () => fetchApi<EmaStatesData>('/api/signals/ema-states'),
  smaReversals: () => fetchApi<SmaReversalsData>('/api/signals/sma-reversals'),
  reversalFlips: (recentDays = 4, tail = 365) =>
    fetchApi<ReversalFlipsData>(`/api/signals/reversal-flips?recent_days=${recentDays}&tail=${tail}`),

  // Risk
  riskDashboard: () => fetchApi<RiskDashboard>('/api/risk/dashboard'),
  riskSummary: () => fetchApi<RiskSummary>('/api/risk/summary'),

  // Charts
  chartSymbols: () => fetchApi<{ symbols: string[]; count: number }>('/api/charts/symbols'),
  chartSymbolsBySector: () => fetchApi<ChartSectorData>('/api/charts/symbols-by-sector'),
  chartOhlcv: (symbol: string, tail = 365) =>
    fetchApi<{ symbol: string; data: OHLCVBar[]; count: number }>(`/api/charts/ohlcv/${encodeURIComponent(symbol)}?tail=${tail}`),
  chartIndicators: (symbol: string, tail = 365) =>
    fetchApi<{ symbol: string; indicators: Indicators }>(`/api/charts/indicators/${encodeURIComponent(symbol)}?tail=${tail}`),
  chartForecast: (symbol: string) =>
    fetchApi<ForecastData>(`/api/charts/forecast/${encodeURIComponent(symbol)}`),
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
  refreshTuneCache: () => postApi<{ status: string }>('/api/tune/refresh-cache', {}),
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

  // Watchlist — user-curated tickers persisted server-side.
  watchlistGet: () => fetchApi<WatchlistResponse>('/api/watchlist'),
  watchlistProxyMap: () =>
    fetchApi<WatchlistProxyMapResponse>('/api/watchlist/proxy-map'),
  watchlistAdd: async (symbol: string): Promise<WatchlistResponse> => {
    const res = await fetch('/api/watchlist', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol }),
    });
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try { const body = await res.json(); if (body?.detail) detail = body.detail; } catch { /* ignore */ }
      throw new Error(detail);
    }
    return res.json();
  },
  watchlistRemove: async (symbol: string): Promise<WatchlistResponse> => {
    const res = await fetch(`/api/watchlist/${encodeURIComponent(symbol)}`, { method: 'DELETE' });
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try { const body = await res.json(); if (body?.detail) detail = body.detail; } catch { /* ignore */ }
      throw new Error(detail);
    }
    return res.json();
  },
};

export interface WatchlistResponse {
  symbols: string[];
}

export interface WatchlistProxyMapResponse {
  proxies: Record<string, string>;
}

export interface PoliticiansDataUseNotice {
  title: string;
  summary: string;
  bullets: string[];
  official_sources: string[];
  reviewed_at: string;
}

export interface PoliticiansNoticeResponse {
  feature: 'politicians';
  status: 'notice_only' | 'available' | 'disabled' | 'ok' | 'missing_data';
  enabled: boolean;
  compliance_mode: 'research_only' | 'internal' | 'public';
  requested_compliance_mode: string;
  compliance_mode_valid: boolean;
  valid_compliance_modes: string[];
  disabled_reason?: string | null;
  endpoint?: string | null;
  message?: string;
  generated_at?: string;
  data_age_seconds?: number | null;
  data_use_notice: PoliticiansDataUseNotice;
}

export interface PoliticiansSourceHealth {
  updated_at: string | null;
  sources: Record<string, unknown>;
}

export interface PoliticiansChamberSummary {
  trade_count: number;
  buy_count: number;
  sell_count: number;
  buy_amount_mid_usd: number;
  sell_amount_mid_usd: number;
  net_buy_amount_mid_usd: number;
}

export interface PoliticiansSummaryMetrics {
  total_trades: number;
  new_disclosures_7d: number;
  new_disclosures_last_7_days: number;
  new_tracked_asset_disclosures_7d: number;
  new_watchlist_disclosures_7d: number;
  tracked_asset_trades: number;
  watchlist_trades: number;
  late_filings: number;
  newest_disclosure_date: string | null;
  source_health: PoliticiansSourceHealth;
  by_chamber: Record<string, PoliticiansChamberSummary>;
}

export interface PoliticiansSummaryResponse extends PoliticiansNoticeResponse, PoliticiansSummaryMetrics {
  status: 'ok' | 'disabled';
  summary: PoliticiansSummaryMetrics;
}

export interface PoliticiansTradeQuery {
  limit?: number;
  offset?: number;
  symbol?: string;
  filer?: string;
  chamber?: string;
  party?: string;
  state?: string;
  transaction_type?: string;
  transaction_side?: 'purchase' | 'sale';
  owner?: string;
  flag?: string;
  tracked_only?: boolean;
  watchlist_only?: boolean;
  top_traders_only?: boolean;
  stock_linked_only?: boolean;
  from?: string;
  to?: string;
}

export interface PoliticiansTradePage {
  limit: number | null;
  offset: number;
  returned: number;
  total: number;
  has_next: boolean;
}

export interface PoliticiansTradeRow extends Record<string, unknown> {
  trade_id?: string;
  ticker?: string;
  filer_name?: string;
  chamber?: string;
  party?: string;
  state?: string;
  transaction_type?: string;
  owner?: string;
  disclosure_date?: string;
  successful_trader_rank?: number;
  successful_trader_overall_rank?: number;
  successful_trader_score?: number;
  successful_trader_return_pct?: number;
  successful_trader_win_rate?: number;
  successful_trader_required_profile?: boolean;
  official_source_url: string | null;
}

export interface PoliticiansSuccessfulTrader {
  rank: number;
  overall_rank: number;
  filer_key: string;
  filer_name: string;
  chamber?: string | null;
  party?: string | null;
  state?: string | null;
  total_trades: number;
  scored_trades: number;
  coverage: number;
  win_rate: number;
  average_signed_return_pct: number;
  median_signed_return_pct: number;
  success_score: number;
  amount_mid_usd: number;
  top_tickers: string[];
  included_by_requested_profile: boolean;
}

export interface PoliticiansSuccessfulTraders {
  methodology: {
    label: string;
    description: string;
    horizon_days: number;
    min_holding_days: number;
    as_of_date: string;
    required_profiles: string[];
  };
  limit: number;
  scored_trade_count: number;
  unscored_trade_count: number;
  eligible_trader_count: number;
  leaderboard: PoliticiansSuccessfulTrader[];
}

export interface PoliticiansTradesResponse extends PoliticiansNoticeResponse {
  status: 'ok' | 'disabled' | 'missing_data';
  filter?: PoliticiansTradeQuery;
  page?: PoliticiansTradePage;
  total?: number;
  trades?: PoliticiansTradeRow[];
  successful_traders?: PoliticiansSuccessfulTraders;
}

export interface PoliticiansAssetResponse extends PoliticiansNoticeResponse {
  status: 'ok' | 'disabled' | 'missing_data';
  symbol?: string;
  window_days?: number;
  total?: number;
  total_symbol_trades?: number;
  recent_trades?: PoliticiansTradeRow[];
  trades?: PoliticiansTradeRow[];
  unique_filers?: string[];
  unique_filer_count?: number;
  buy_sell_imbalance?: Record<string, number>;
  amount_estimates?: Record<string, number>;
  activity?: Record<string, unknown>;
  disclosure_timeline?: Array<Record<string, number | string>>;
  known_limitations?: Array<{ code: string; message: string }>;
}

export interface PoliticiansFilerMetadata {
  filer_id: string;
  filer_name: string | null;
  chamber?: string | null;
  party?: string | null;
  state?: string | null;
  source?: string | null;
  committee_enrichment?: string[];
  committee_data_source?: string;
  metadata_complete: boolean;
}

export interface PoliticiansFilerResponse extends PoliticiansNoticeResponse {
  status: 'ok' | 'disabled' | 'missing_data';
  filer_id?: string;
  window_days?: number;
  metadata?: PoliticiansFilerMetadata;
  total?: number;
  total_filer_trades?: number;
  recent_trades?: PoliticiansTradeRow[];
  top_tickers?: Array<Record<string, number | string>>;
  top_sectors?: Array<Record<string, number | string>>;
  delay_stats?: Record<string, number | null>;
  ownership_breakdown?: Record<'self' | 'spouse' | 'dependent_child' | 'joint' | 'unknown', number>;
  source_documents?: Array<Record<string, string | null | undefined>>;
}

export interface PoliticiansSourceEntry {
  status: 'ok' | 'degraded' | 'offline' | 'disabled';
  last_sync_time: string | null;
  newest_filing: string | null;
  parse_success_rate: number | null;
  trade_count: number;
  parse_error_count: number;
  low_confidence_rows: number;
  recent_errors: unknown[];
  remediation: string;
}

export interface PoliticiansSourceHealthResponse extends PoliticiansNoticeResponse {
  status: 'ok' | 'disabled';
  overall_status?: 'ok' | 'degraded' | 'offline' | 'disabled';
  sources?: Record<string, PoliticiansSourceEntry>;
  source_health?: PoliticiansSourceHealth;
  confidence_buckets?: Record<'high' | 'warning' | 'quarantined', number>;
  parse_error_count?: number;
}

export interface PoliticiansSyncResponse {
  task?: 'politicians_daily_sync';
  status: 'ok' | 'degraded' | 'disabled';
  offline_mode?: boolean;
  safe_to_rerun?: boolean;
  started_at?: string;
  finished_at?: string;
  date_window?: { from: string; to: string };
  steps?: Record<string, Record<string, unknown>>;
  counts?: Record<string, number>;
  errors?: string[];
  message?: string;
}

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

export interface EmaState {
  price: number;
  ema9: number | null;
  ema50: number | null;
  ema600: number | null;
  below_9: boolean | null;
  below_50: boolean | null;
  below_600: boolean | null;
}
export interface EmaStatesData {
  states: Record<string, EmaState>;
  count: number;
  periods: number[];
  built_at: number;
}

// ── SMA reversal detection ───────────────────────────────────────────────
export interface HistoricalEdge {
  samples: number;
  win_rate: number | null;       // 0..1, null when samples < 5
  median_fwd_pct: number | null;
  mean_fwd_pct: number | null;
  std_fwd_pct: number | null;
}
export interface SmaReversal {
  symbol: string;
  period: 9 | 50 | 600 | number;
  direction: 'bull' | 'bear';
  price: number;
  sma: number;
  distance_pct: number;
  atr_distance: number | null;
  atr: number | null;
  slope_pct_5d: number;
  volume_ratio: number | null;
  days_since_cross: number;
  persistence: number;
  persistence_window: number;
  persistence_threshold: number;
  passes_persistence: boolean;
  false_break: boolean;
  score: number; // 0..100
  cross_date: string | null;
  cross_index_from_end: number;

  // Buy-signal quality fields
  regime_sma: number | null;
  regime_ok: boolean;
  overextended: boolean;
  stop_price: number | null;
  target_price: number | null;
  risk_reward: number | null;      // 2.0 by construction when set
  grade: 'A' | 'B' | 'C' | null;
  grade_reasons: string[];
  historical_edge: HistoricalEdge;
  edge_forward_days: number;
}
export interface SmaReversalsData {
  reversals: SmaReversal[];
  counts_by_period: Record<string, { bull: number; bear: number }>;
  grade_counts: { A: number; B: number; C: number; ungraded: number };
  buy_setups: number;
  periods: number[];
  lookback_bars: number;
  persistence_window: number;
  persistence_threshold: number;
  regime_period: number;
  overextended_atr: number;
  edge_forward_days: number;
  total: number;
  built_at: number;
}

export interface ReversalFlipEntry {
  symbol: string;
  signal: 'buy' | 'sell' | null;
  signal_date: string | null;
  bars_ago: number | null;
  price: number | null;
}

export interface ReversalFlipsData {
  signals: Record<string, ReversalFlipEntry>;
  counts: { buy: number; sell: number };
  recent_days: number;
  tail: number;
  total: number;
  latest_date: string | null;
  built_at: number;
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
  pct_30d?: number | null;
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
  pit_calibration_grade: string | null;
  ad_pass: boolean | null;
  ks_pvalue: number | null;
  ks_stat: number | null;
  num_models: number;
  bic: number | null;
  phi: number | null;
  nu: number | null;
  n_obs: number | null;
  top_weight: number | null;
  cache_version: string;
  last_tuned: string;
  file_size_kb: number;
}

export interface ModelAnalytics {
  count: number;
  avg_bic: number | null;
  median_bic: number | null;
  best_bic: number | null;
  worst_bic: number | null;
  avg_phi: number | null;
  avg_nu: number | null;
  avg_weight: number | null;
  avg_ks_pvalue: number | null;
  median_ks_pvalue: number | null;
  pit_pass: number;
  pit_fail: number;
  pit_pass_rate: number | null;
  avg_n_obs: number | null;
  top_symbols: string[];
}

export interface TuneStats {
  total: number;
  pit_pass: number;
  pit_fail: number;
  pit_unknown: number;
  models_distribution: Record<string, number>;
  models_analytics?: Record<string, ModelAnalytics>;
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
  politicians?: {
    status: string;
    data_age_seconds: number | null;
    sources: Record<string, {
      status?: string;
      last_sync_time?: string | null;
      newest_filing?: string | null;
      parse_success_rate?: number | null;
      recent_error_count?: number;
    }>;
    overall_source_status?: string;
    details_url: string;
    source_health_url: string;
    degraded_blocks_app: boolean;
    message?: string;
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
  filing_id?: string;
  parser_version?: string;
  exception_class?: string;
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
  family: string;
  win_count: number;
  total_weight: number;
  appearances: number;
  avg_weight: number;
  win_rate: number;
  appearance_rate: number;
  max_weight: number;
  min_weight: number;
  avg_bic: number | null;
  avg_crps: number | null;
  avg_hyvarinen: number | null;
  avg_pit_p: number | null;
  avg_ad_p: number | null;
  avg_histogram_mad: number | null;
  top_symbols: string[];
}

export interface DiagModelFamilyStats {
  family: string;
  model_count: number;
  appearances: number;
  win_count: number;
  avg_weight: number;
}

export interface DiagModelComparisonCell {
  family: string;
  winner: boolean;
  weight: number;
  bic: number | null;
  crps: number | null;
  hyvarinen: number | null;
  pit_ks_pvalue: number | null;
  ad_pvalue: number | null;
  histogram_mad: number | null;
}

export interface DiagModelComparisonRow {
  symbol: string;
  best_model: string;
  regime: string | null;
  ad_pass: boolean | null;
  models: Record<string, DiagModelComparisonCell>;
}

export interface DiagModelComparison {
  models: Record<string, DiagModelStats>;
  model_names: string[];
  families: DiagModelFamilyStats[];
  matrix_rows: DiagModelComparisonRow[];
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
  family: string;
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
  model_averages: Record<string, { family: string; avg_crps: number | null; avg_pit_p: number | null; avg_bic: number | null; avg_weight: number; count: number }>;
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
