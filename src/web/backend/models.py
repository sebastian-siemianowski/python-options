"""
Pydantic v2 response models for API endpoints.

These mirror the existing Python dataclass structures and to_dict() outputs.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Task management ──────────────────────────────────────────────────────────


class TaskResponse(BaseModel):
    task_id: str
    task_type: str
    status: str = "queued"
    message: str = ""


class TaskStatusResponse(BaseModel):
    task_id: str
    task_type: str
    status: str  # queued | started | progress | completed | failed
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# ── Signals ──────────────────────────────────────────────────────────────────


class SignalItem(BaseModel):
    horizon_days: int
    label: str
    score: float = 0.0
    p_up: float = 0.5
    exp_ret: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    profit_pln: float = 0.0
    position_strength: float = 0.0
    vol_mean: float = 0.0
    regime: str = ""
    expected_utility: float = 0.0
    expected_gain: float = 0.0
    expected_loss: float = 0.0
    risk_temperature: float = 0.0
    risk_scale_factor: float = 1.0
    ue_up: float = 0.0
    ue_down: float = 0.0
    pit_penalty_effective: float = 1.0
    evt_enabled: bool = False
    hansen_enabled: bool = False
    cst_enabled: bool = False
    crps_score: Optional[float] = None


class AssetSignalRow(BaseModel):
    """Mirrors the summary_rows structure used throughout the terminal UX."""
    symbol: str
    asset_label: str
    sector: str = ""
    crash_risk_score: float = 0.0
    nearest_label: str = "HOLD"
    horizon_signals: Dict[int, Dict[str, Any]] = Field(default_factory=dict)


class HighConvictionSignal(BaseModel):
    ticker: str
    asset_label: str
    sector: str = ""
    horizon_days: int
    signal_type: str  # STRONG_BUY | STRONG_SELL
    probability_up: float = 0.5
    probability_down: float = 0.5
    expected_return_pct: float = 0.0
    signal_strength: float = 0.0
    generated_at: str = ""
    options_chain: Optional[Dict[str, Any]] = None


class SignalsResponse(BaseModel):
    assets: List[AssetSignalRow]
    total_assets: int
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    generated_at: str = ""
    horizons: List[int] = Field(default_factory=lambda: [1, 3, 7])


class HighConvictionResponse(BaseModel):
    buy_signals: List[HighConvictionSignal]
    sell_signals: List[HighConvictionSignal]
    total: int


# ── Risk Dashboard ───────────────────────────────────────────────────────────


class StressCategoryResponse(BaseModel):
    name: str
    temperature: float
    weight: float
    indicators: List[Dict[str, Any]] = Field(default_factory=list)


class RiskTemperatureResponse(BaseModel):
    temperature: float
    scale_factor: float
    overnight_budget_active: bool = False
    crash_risk_pct: float = 0.0
    crash_risk_level: str = "LOW"
    categories: Dict[str, Any] = Field(default_factory=dict)
    computed_at: str = ""


class MetalsRiskResponse(BaseModel):
    temperature: float
    scale_factor: float
    status: str = ""
    action_text: str = ""
    regime_state: str = ""
    metals: Dict[str, Any] = Field(default_factory=dict)
    computed_at: str = ""


class MarketTemperatureResponse(BaseModel):
    temperature: float
    scale_factor: float
    status: str = ""
    action_text: str = ""
    crash_risk_pct: float = 0.0
    crash_risk_level: str = "LOW"
    overall_momentum: str = ""
    exit_signal: bool = False
    exit_reason: str = ""
    sectors: Dict[str, Any] = Field(default_factory=dict)
    currencies: Dict[str, Any] = Field(default_factory=dict)
    breadth: Dict[str, Any] = Field(default_factory=dict)
    computed_at: str = ""


class RiskDashboardResponse(BaseModel):
    risk_temperature: RiskTemperatureResponse
    metals_risk_temperature: MetalsRiskResponse
    market_temperature: MarketTemperatureResponse
    combined_temperature: float = 0.0
    combined_scale_factor: float = 1.0
    combined_status: str = ""
    computed_at: str = ""


# ── Tuning ───────────────────────────────────────────────────────────────────


class TuneAssetSummary(BaseModel):
    symbol: str
    pit_calibration_grade: Optional[str] = None
    best_model: str = ""
    bic: Optional[float] = None
    phi: Optional[float] = None
    nu: Optional[float] = None
    n_obs: Optional[int] = None
    degraded: bool = False
    last_tuned: str = ""


class TuneDetailResponse(BaseModel):
    symbol: str
    global_params: Dict[str, Any] = Field(default_factory=dict)
    model_weights: Dict[str, float] = Field(default_factory=dict)
    models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    regime_params: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class TuneListResponse(BaseModel):
    assets: List[TuneAssetSummary]
    total: int
    pit_a_count: int = 0
    pit_b_count: int = 0
    pit_c_count: int = 0
    pit_d_count: int = 0
    pit_f_count: int = 0
    pit_none_count: int = 0


# ── Data Management ──────────────────────────────────────────────────────────


class AssetDataStatus(BaseModel):
    symbol: str
    last_date: Optional[str] = None
    file_size_kb: float = 0.0
    age_days: float = 0.0
    row_count: int = 0
    status: str = "ok"  # ok | stale | missing


class DataStatusResponse(BaseModel):
    assets: List[AssetDataStatus]
    total_assets: int
    fresh_count: int = 0
    stale_count: int = 0
    missing_count: int = 0
    total_size_mb: float = 0.0
    prices_dir: str = ""


# ── Charts ───────────────────────────────────────────────────────────────────


class OHLCVBar(BaseModel):
    time: str  # ISO date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class IndicatorPoint(BaseModel):
    time: str
    value: Optional[float] = None


class ForecastPoint(BaseModel):
    time: str
    median: float
    ci_50_lo: float
    ci_50_hi: float
    ci_80_lo: float
    ci_80_hi: float
    ci_95_lo: float
    ci_95_hi: float


class ChartDataResponse(BaseModel):
    ticker: str
    bars: List[OHLCVBar]
    last_close: float = 0.0


class IndicatorsResponse(BaseModel):
    ticker: str
    sma_10: List[IndicatorPoint] = Field(default_factory=list)
    sma_20: List[IndicatorPoint] = Field(default_factory=list)
    sma_50: List[IndicatorPoint] = Field(default_factory=list)
    sma_200: List[IndicatorPoint] = Field(default_factory=list)
    bb_upper: List[IndicatorPoint] = Field(default_factory=list)
    bb_lower: List[IndicatorPoint] = Field(default_factory=list)
    rsi: List[IndicatorPoint] = Field(default_factory=list)
    atr: List[IndicatorPoint] = Field(default_factory=list)


class ForecastResponse(BaseModel):
    ticker: str
    forecasts: List[ForecastPoint] = Field(default_factory=list)
    drift_annualized: float = 0.0
    best_nu: float = 30.0
    markers: Dict[int, Dict[str, float]] = Field(default_factory=dict)


# ── Arena ────────────────────────────────────────────────────────────────────


class ArenaModelResult(BaseModel):
    model_name: str
    final_score: float = 0.0
    bic: Optional[float] = None
    crps: Optional[float] = None
    hyvarinen: Optional[float] = None
    pit_pass_rate: Optional[float] = None
    css: Optional[float] = None
    fec: Optional[float] = None
    vs_std: Optional[float] = None
    hard_gates_passed: bool = False


class ArenaResponse(BaseModel):
    standard_models: List[ArenaModelResult] = Field(default_factory=list)
    experimental_models: List[ArenaModelResult] = Field(default_factory=list)
    safe_storage: List[ArenaModelResult] = Field(default_factory=list)
    last_run: str = ""


# ── Overview ─────────────────────────────────────────────────────────────────


class SystemHealth(BaseModel):
    signals_last_run: str = ""
    tune_last_run: str = ""
    data_freshness: str = ""
    total_assets: int = 0
    tuned_assets: int = 0
    price_files: int = 0
    signal_buy_count: int = 0
    signal_sell_count: int = 0
    avg_pit_grade: str = ""
    redis_connected: bool = False
    celery_connected: bool = False
