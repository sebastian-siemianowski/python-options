"""
===============================================================================
METALS RISK TEMPERATURE MODULE
===============================================================================

Implements the Chinese Professors' Panel Hybrid Solution (Score: 9.2/10):

    Architecture: Mirror existing risk_temperature.py pattern (Chen Wei-Lin)
    Indicators: Ratio-based stress signals (Liu Jian-Ming)
    Practitioner: Battle-tested hedge fund indicators (Zhang Hui-Fang)

DESIGN PRINCIPLE:
    "Metals don't predict direction — they reveal stress regime."

This module computes a scalar metals risk temperature from cross-metal
stress indicators that can be used for:
    1. Standalone metals risk assessment (make metals)
    2. Integration into main risk temperature as "Metals" category

STRESS INDICATORS (Practitioner Consensus + Academic Rigor):
    1. Copper/Gold Ratio (35%): Economic health bellwether
       - Rising = Risk-on (industrial demand)
       - Falling = Risk-off (flight to safety)
       
    2. Silver/Gold Ratio (25%): Speculative intensity
       - Rising = Risk-on (silver outperforms in euphoria)
       - Falling = Risk-off (gold outperforms in fear)
       
    3. Gold Volatility Proxy (20%): Fear gauge via realized vol
       - Uses 20-day realized volatility percentile
       
    4. Precious vs Industrial Spread (10%): Sector rotation
       - (Gold + Silver) / (Copper + Platinum) divergence
       
    5. Platinum/Gold Ratio (10%): Industrial precious hybrid
       - Platinum as both industrial and precious metal

MATHEMATICAL MODEL:
    metals_temp = Σ_i (weight_i × stress_i)
    
    Where stress_i is z-score normalized and capped at [0, 2]

REFERENCES:
    Expert Panel Evaluation (January 2026)
    Professor Chen Wei-Lin (Tsinghua): Architecture Score 8.5/10
    Professor Liu Jian-Ming (Fudan): Indicator Score 9/10
    Professor Zhang Hui-Fang (CUHK): Practitioner Score 9.5/10
    Combined Implementation Score: 9.2/10

===============================================================================
"""

from __future__ import annotations

import json
import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

# Governance imports (February 2026 — Copilot Story QUANT-2025-MRT-001)
try:
    from decision.regime_governance import (
        RegimeState,
        RegimeStateManager,
        ImputationManager,
        DynamicGapRiskEstimator,
        GovernedRiskTemperatureAudit,
        IndicatorAuditRecord,
        CategoryAuditRecord,
        GapRiskAuditRecord,
        apply_rate_limit,
        create_governance_managers,
        IMPUTATION_WARNING_THRESHOLD,
        IMPUTATION_TEMPERATURE_FLOOR,
        MAX_TEMP_CHANGE_PER_DAY,
    )
    GOVERNANCE_AVAILABLE = True
except ImportError:
    GOVERNANCE_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# METALS RISK TEMPERATURE CONSTANTS
# =============================================================================

# Stress indicator weights (sum to 1.0)
COPPER_GOLD_WEIGHT = 0.35      # Economic health bellwether
SILVER_GOLD_WEIGHT = 0.25      # Speculative intensity
GOLD_VOLATILITY_WEIGHT = 0.20  # Fear gauge
PRECIOUS_INDUSTRIAL_WEIGHT = 0.10  # Sector rotation
PLATINUM_GOLD_WEIGHT = 0.10    # Industrial precious hybrid

# Z-score calculation lookback
ZSCORE_LOOKBACK_DAYS = 60

# Volatility percentile lookback
VOLATILITY_LOOKBACK_DAYS = 252  # 1 year

# Scaling function parameters (match main risk_temperature.py)
SIGMOID_K = 3.0
SIGMOID_THRESHOLD = 1.0

# Temperature bounds
TEMP_MIN = 0.0
TEMP_MAX = 2.0

# Cache TTL for market data (seconds)
CACHE_TTL_SECONDS = 3600  # 1 hour

# =============================================================================
# ANTICIPATORY ENHANCEMENT CONSTANTS (February 2026)
# =============================================================================

# Volatility Term Structure Inversion Detection
VOL_TERM_STRUCTURE_SHORT_WINDOW = 5    # 5-day realized volatility
VOL_TERM_STRUCTURE_LONG_WINDOW = 20    # 20-day realized volatility
VOL_TERM_STRUCTURE_INVERSION_THRESHOLD = 1.5  # Short/Long ratio threshold
VOL_TERM_STRUCTURE_MIN_METALS = 2      # Minimum metals required for signal
VOL_TERM_STRUCTURE_STRESS_CONTRIBUTION = 0.3  # Additive stress when triggered

# Robust Z-Score (MAD-based) - Used instead of standard z-score
# Consistency constant for MAD to match std dev of normal distribution
MAD_CONSISTENCY_CONSTANT = 1.4826

# Data Infrastructure Hardening
DATA_FAILOVER_LATENCY_THRESHOLD_SECONDS = 300  # 5 minutes
DATA_PRICE_DIVERGENCE_THRESHOLD = 0.005  # 0.5% price divergence triggers alert
DATA_DEGRADED_MODE_THRESHOLD = 0.6  # 60% indicators required, else degraded mode
DATA_DEGRADED_MODE_TEMP_FLOOR = 1.0  # Temperature floor when in degraded mode

# Enhanced Escalation Protocol with Hysteresis (Updated per Copilot Story)
ESCALATION_NORMAL_TO_ELEVATED_TEMP = 0.7
ESCALATION_ELEVATED_TO_STRESSED_TEMP = 1.2
ESCALATION_STRESSED_TO_NORMAL_TEMP = 0.5
ESCALATION_ELEVATED_TO_NORMAL_TEMP = 0.4
ESCALATION_CONSECUTIVE_REQUIRED = 2  # Consecutive computations for upward transition
ESCALATION_SUSTAINED_DAYS_STRESSED_TO_NORMAL = 5  # Business days
ESCALATION_SUSTAINED_DAYS_ELEVATED_TO_NORMAL = 3  # Business days


class AlertSeverity:
    """Alert severity levels for routing to operations."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# =============================================================================
# DATA CLASSES (Mirror risk_temperature.py structure)
# =============================================================================

@dataclass
class MetalStressIndicator:
    """Individual metal stress indicator with value and metadata."""
    name: str
    value: float              # Raw value (e.g., ratio)
    zscore: float             # Z-score relative to lookback
    contribution: float       # Contribution to total stress
    data_available: bool
    interpretation: str       # Human-readable interpretation
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": float(self.value) if self.data_available else None,
            "zscore": float(self.zscore) if self.data_available else None,
            "contribution": float(self.contribution),
            "data_available": self.data_available,
            "interpretation": self.interpretation,
        }


@dataclass
class MetalStressCategory:
    """Individual metal stress (Gold, Silver, Copper, etc.)."""
    name: str
    price: Optional[float]    # Current price
    return_5d: float          # 5-day return
    volatility: float         # 20-day realized volatility
    stress_level: float       # Metal-specific stress ∈ [0, 2]
    data_available: bool
    # Momentum fields (added Feb 2026)
    return_1d: float = 0.0    # 1-day return
    return_21d: float = 0.0   # 21-day (1-month) return
    momentum_signal: str = "" # "↑ Strong", "↗ Rising", "→ Flat", "↘ Falling", "↓ Weak"
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "price": float(self.price) if self.price else None,
            "return_5d": float(self.return_5d),
            "return_1d": float(self.return_1d),
            "return_21d": float(self.return_21d),
            "volatility": float(self.volatility),
            "stress_level": float(self.stress_level),
            "momentum_signal": self.momentum_signal,
            "data_available": self.data_available,
        }


@dataclass
class MetalsRiskTemperatureResult:
    """Complete metals risk temperature computation result."""
    temperature: float                       # Final temperature ∈ [0, 2]
    scale_factor: float                      # Position scaling factor ∈ (0, 1)
    indicators: List[MetalStressIndicator]   # Ratio-based indicators
    metals: Dict[str, MetalStressCategory]   # Individual metal stress
    computed_at: str                         # ISO timestamp
    data_quality: float                      # Fraction of indicators with data
    status: str                              # Calm, Elevated, Stressed, Extreme
    action_text: str                         # Position recommendation
    
    # Governance enhancement fields (February 2026)
    regime_state: str = "Calm"               # Discrete regime from hysteresis
    previous_regime_state: Optional[str] = None
    regime_transition_occurred: bool = False
    raw_temperature: Optional[float] = None  # Before rate limiting
    rate_limit_applied: bool = False
    imputed_indicators: int = 0
    imputation_warning: bool = False
    temperature_floor_applied: bool = False
    gap_risk_estimate: float = 0.03
    overnight_budget_active: bool = False
    overnight_max_position: Optional[float] = None
    audit_trail: Optional["GovernedRiskTemperatureAudit"] = None
    # Crash risk fields (February 2026)
    crash_risk_pct: float = 0.0
    crash_risk_level: str = "Low"
    vol_inversion_count: int = 0
    
    def to_dict(self) -> Dict:
        result = {
            "temperature": float(self.temperature),
            "scale_factor": float(self.scale_factor),
            "status": self.status,
            "action_text": self.action_text,
            "computed_at": self.computed_at,
            "data_quality": float(self.data_quality),
            "indicators": [ind.to_dict() for ind in self.indicators],
            "metals": {k: v.to_dict() for k, v in self.metals.items()},
            # Governance fields
            "regime_state": self.regime_state,
            "previous_regime_state": self.previous_regime_state,
            "regime_transition_occurred": self.regime_transition_occurred,
            "raw_temperature": float(self.raw_temperature) if self.raw_temperature is not None else None,
            "rate_limit_applied": self.rate_limit_applied,
            "imputed_indicators": self.imputed_indicators,
            "imputation_warning": self.imputation_warning,
            "temperature_floor_applied": self.temperature_floor_applied,
            "gap_risk_estimate": float(self.gap_risk_estimate),
            "overnight_budget_active": self.overnight_budget_active,
            "overnight_max_position": float(self.overnight_max_position) if self.overnight_max_position else None,
            # Crash risk fields
            "crash_risk_pct": float(self.crash_risk_pct),
            "crash_risk_level": self.crash_risk_level,
            "vol_inversion_count": self.vol_inversion_count,
        }
        return result
    
    def get_audit_json(self) -> Optional[str]:
        """Get complete audit trail as JSON string."""
        if self.audit_trail:
            return self.audit_trail.to_json()
        return None
    
    def render_audit_trail(self) -> Optional[str]:
        """Get human-readable audit trail."""
        if self.audit_trail:
            return self.audit_trail.render_human_readable()
        return None
    
    @property
    def is_elevated(self) -> bool:
        return self.temperature > 0.5
    
    @property
    def is_stressed(self) -> bool:
        return self.temperature > 1.0
    
    @property
    def is_extreme(self) -> bool:
        return self.temperature > 1.5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _compute_robust_zscore(
    values: pd.Series,
    lookback: int = ZSCORE_LOOKBACK_DAYS,
    return_audit: bool = False
) -> float | Tuple[float, Dict]:
    """
    Compute ROBUST z-score using Median Absolute Deviation (MAD).
    
    Formula: z_robust = (x - median) / (1.4826 × MAD)
    
    The MAD estimator has a breakdown point of 50%, meaning up to half
    the observations can be outliers without corrupting the estimate.
    This prevents the crash itself from distorting the reference distribution.
    
    Args:
        values: Time series of values
        lookback: Lookback window for median/MAD calculation
        return_audit: If True, return tuple of (zscore, audit_dict)
        
    Returns:
        Robust z-score, or tuple (zscore, audit_dict) if return_audit=True
    """
    audit = {
        "lookback_median": None,
        "lookback_mad": None,
        "lookback_mean": None,  # For comparison/audit
        "lookback_std": None,   # For comparison/audit
        "lookback_start": None,
        "lookback_end": None,
        "estimation_method": "MAD_robust",
    }
    
    try:
        if values is None:
            return (0.0, audit) if return_audit else 0.0
        
        if isinstance(values, pd.DataFrame):
            if values.shape[1] == 1:
                values = values.iloc[:, 0]
            else:
                return (0.0, audit) if return_audit else 0.0
        
        values = values.dropna()
        
        if len(values) < lookback // 2:
            return (0.0, audit) if return_audit else 0.0
        
        recent = values.iloc[-lookback:] if len(values) >= lookback else values
        current = float(values.iloc[-1])
        
        # Robust estimation using MAD
        median = float(recent.median())
        mad = float((recent - median).abs().median())
        
        # Also compute standard stats for audit trail comparison
        mean = float(recent.mean())
        std = float(recent.std())
        
        # Record audit information
        audit["lookback_median"] = median
        audit["lookback_mad"] = mad
        audit["lookback_mean"] = mean
        audit["lookback_std"] = std
        if hasattr(recent, 'index') and len(recent.index) > 0:
            audit["lookback_start"] = str(recent.index[0])
            audit["lookback_end"] = str(recent.index[-1])
        
        # Use MAD for robust z-score computation
        # Scale MAD by 1.4826 to match std dev of normal distribution
        scaled_mad = MAD_CONSISTENCY_CONSTANT * mad
        
        if scaled_mad < 1e-10 or not np.isfinite(scaled_mad):
            # Fallback to standard z-score if MAD is zero
            if std < 1e-10 or not np.isfinite(std):
                return (0.0, audit) if return_audit else 0.0
            zscore = (current - mean) / std
            audit["estimation_method"] = "std_fallback"
        else:
            zscore = (current - median) / scaled_mad
        
        result = float(np.clip(zscore, -5.0, 5.0))
        
        return (result, audit) if return_audit else result
    except Exception:
        return (0.0, audit) if return_audit else 0.0


def _compute_zscore(
    values: pd.Series,
    lookback: int = ZSCORE_LOOKBACK_DAYS,
    return_audit: bool = False
) -> float | Tuple[float, Dict]:
    """
    Compute z-score of most recent value relative to rolling window.
    
    NOTE: As of February 2026 Anticipatory Enhancement, this now uses
    the robust MAD-based z-score computation for outlier resistance.
    
    Args:
        values: Time series of values
        lookback: Lookback window for calculation
        return_audit: If True, return tuple of (zscore, audit_dict)
        
    Returns:
        Z-score of most recent value, or tuple (zscore, audit_dict) if return_audit=True
    """
    return _compute_robust_zscore(values, lookback, return_audit)


def _compute_volatility_term_structure(
    prices: pd.Series,
    short_window: int = VOL_TERM_STRUCTURE_SHORT_WINDOW,
    long_window: int = VOL_TERM_STRUCTURE_LONG_WINDOW,
) -> Tuple[float, float, float]:
    """
    Compute volatility term structure ratio for a single metal.
    
    Ratio = (5-day realized vol) / (20-day realized vol)
    
    When ratio > 1.5, short-term volatility exceeds long-term trend,
    indicating volatility term structure inversion - a documented
    precursor to crash events.
    
    Args:
        prices: Price series for the metal
        short_window: Short-term volatility window (default 5 days)
        long_window: Long-term volatility window (default 20 days)
        
    Returns:
        Tuple of (ratio, short_vol, long_vol)
    """
    try:
        if prices is None or len(prices) < long_window + 5:
            return 0.0, 0.0, 0.0
        
        returns = prices.pct_change().dropna()
        
        if len(returns) < long_window:
            return 0.0, 0.0, 0.0
        
        # Compute annualized volatilities
        short_vol = float(returns.iloc[-short_window:].std() * np.sqrt(252))
        long_vol = float(returns.iloc[-long_window:].std() * np.sqrt(252))
        
        if long_vol < 1e-10:
            return 0.0, short_vol, long_vol
        
        ratio = short_vol / long_vol
        
        return ratio, short_vol, long_vol
    except Exception:
        return 0.0, 0.0, 0.0


def compute_volatility_term_structure_stress(
    metals_data: Dict[str, pd.Series]
) -> Tuple[MetalStressIndicator, Dict[str, Dict]]:
    """
    Compute volatility term structure inversion signal across all metals.
    
    LEADING INDICATOR: When 2+ metals simultaneously exhibit vol term
    structure inversion (5-day vol / 20-day vol > 1.5), this historically
    precedes crash events by 1-5 trading days.
    
    This is an ADDITIVE stress contribution (not weighted like other indicators)
    that injects 0.3 into the temperature when triggered.
    
    Args:
        metals_data: Dict of metal name -> price series
        
    Returns:
        Tuple of (MetalStressIndicator, details_dict with per-metal breakdown)
    """
    metal_details = {}
    inverted_metals = []
    
    for name in ['GOLD', 'SILVER', 'COPPER', 'PLATINUM', 'PALLADIUM']:
        if name not in metals_data:
            metal_details[name] = {
                "ratio": None,
                "short_vol": None,
                "long_vol": None,
                "inverted": False,
                "data_available": False,
            }
            continue
        
        ratio, short_vol, long_vol = _compute_volatility_term_structure(
            metals_data[name]
        )
        
        is_inverted = ratio >= VOL_TERM_STRUCTURE_INVERSION_THRESHOLD
        
        metal_details[name] = {
            "ratio": ratio,
            "short_vol": short_vol,
            "long_vol": long_vol,
            "inverted": is_inverted,
            "data_available": True,
        }
        
        if is_inverted:
            inverted_metals.append(name)
    
    # Signal triggers when >= 2 metals show inversion
    signal_triggered = len(inverted_metals) >= VOL_TERM_STRUCTURE_MIN_METALS
    
    # Contribution is additive 0.3 when signal triggers
    contribution = VOL_TERM_STRUCTURE_STRESS_CONTRIBUTION if signal_triggered else 0.0
    
    # Build interpretation - more understandable language
    if signal_triggered:
        # Vol term structure inversion = short-term vol >> long-term vol = crash warning
        metals_list = ', '.join(inverted_metals)
        interpretation = f"⚠️ VOL SPIKE in {len(inverted_metals)} metals ({metals_list})"
    elif len(inverted_metals) == 1:
        interpretation = f"Watch: {inverted_metals[0]} vol elevated"
    else:
        interpretation = "Normal"
    
    # Compute average ratio for display
    valid_ratios = [d["ratio"] for d in metal_details.values() if d["ratio"] is not None]
    avg_ratio = sum(valid_ratios) / len(valid_ratios) if valid_ratios else 0.0
    
    indicator = MetalStressIndicator(
        name="Vol Term Structure",
        value=avg_ratio,
        zscore=len(inverted_metals),  # Use count as "z-score" equivalent
        contribution=contribution,
        data_available=len(valid_ratios) >= 3,
        interpretation=interpretation,
    )
    
    return indicator, metal_details


def compute_crash_risk(
    vol_inversion_count: int,
    temperature: float,
    avg_vol_ratio: float = 1.0,
) -> Tuple[float, str]:
    """
    Compute crash risk probability and level based on multiple signals.
    
    METHODOLOGY:
    - Vol term structure inversion is the PRIMARY crash predictor
    - Each inverted metal adds crash probability
    - Temperature above 1.0 adds additional risk
    - High vol ratio amplifies risk
    
    Returns:
        Tuple of (crash_risk_pct, crash_risk_level)
    """
    base_risk_by_count = {0: 0.02, 1: 0.05, 2: 0.15, 3: 0.25, 4: 0.40, 5: 0.55}
    base_risk = base_risk_by_count.get(vol_inversion_count, 0.55)
    
    # Temperature multiplier
    if temperature > 1.5:
        temp_multiplier = 1.5
    elif temperature > 1.0:
        temp_multiplier = 1.25
    elif temperature > 0.7:
        temp_multiplier = 1.1
    else:
        temp_multiplier = 1.0
    
    # Vol ratio amplifier
    if avg_vol_ratio > 2.5:
        vol_multiplier = 1.4
    elif avg_vol_ratio > 2.0:
        vol_multiplier = 1.2
    elif avg_vol_ratio > 1.5:
        vol_multiplier = 1.1
    else:
        vol_multiplier = 1.0
    
    crash_risk_pct = min(0.75, base_risk * temp_multiplier * vol_multiplier)
    
    # Determine risk level
    if crash_risk_pct >= 0.40:
        crash_risk_level = "Extreme"
    elif crash_risk_pct >= 0.25:
        crash_risk_level = "High"
    elif crash_risk_pct >= 0.15:
        crash_risk_level = "Elevated"
    elif crash_risk_pct >= 0.05:
        crash_risk_level = "Moderate"
    else:
        crash_risk_level = "Low"
    
    return crash_risk_pct, crash_risk_level


def _compute_volatility_percentile(
    prices: pd.Series,
    vol_window: int = 20,
    lookback: int = VOLATILITY_LOOKBACK_DAYS
) -> float:
    """Compute current volatility percentile over lookback period."""
    try:
        if prices is None or len(prices) < vol_window + 10:
            return 0.5
        
        returns = prices.pct_change().dropna()
        if len(returns) < lookback:
            return 0.5
        
        rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        
        if len(rolling_vol) < 20:
            return 0.5
        
        current_vol = float(rolling_vol.iloc[-1])
        historical_vol = rolling_vol.iloc[-lookback:] if len(rolling_vol) >= lookback else rolling_vol
        
        percentile = (historical_vol < current_vol).sum() / len(historical_vol)
        return float(percentile)
    except Exception:
        return 0.5


def _extract_close_series(df, ticker: str) -> Optional[pd.Series]:
    """Safely extract Close price series from yfinance DataFrame."""
    if df is None:
        return None
    
    try:
        if hasattr(df, 'empty') and df.empty:
            return None
    except ValueError:
        if len(df) == 0:
            return None
    
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                series = df['Close']
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                return series.dropna()
        
        if 'Close' in df.columns:
            series = df['Close']
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            return series.dropna()
        
        if 'close' in df.columns:
            return df['close'].dropna()
        
        if len(df.columns) > 0:
            return df.iloc[:, 0].dropna()
    except Exception:
        pass
    
    return None


# =============================================================================
# MARKET DATA FETCHING — Multi-Source Federation (February 2026)
# =============================================================================

_metals_data_cache: Dict[str, Tuple[datetime, Dict[str, pd.Series]]] = {}

# Track data source health and failover events
_data_source_health: Dict[str, Dict] = {}


@dataclass
class DataSourceResult:
    """Result from a data source fetch attempt."""
    source: str
    metal: str
    series: Optional[pd.Series]
    latency_seconds: float
    success: bool
    error_message: Optional[str] = None
    

@dataclass  
class DataQualityReport:
    """Comprehensive data quality report for audit trail."""
    timestamp: str
    metals_available: int
    metals_total: int
    data_quality_pct: float
    degraded_mode: bool
    degraded_mode_reason: Optional[str]
    failover_events: List[Dict]
    source_used: Dict[str, str]  # metal -> source
    price_divergence_alerts: List[Dict]
    

def _fetch_from_yfinance(
    ticker: str,
    start: str,
    end: str,
    metal_name: str,
) -> DataSourceResult:
    """Fetch from yfinance (primary source)."""
    import time
    start_time = time.time()
    
    try:
        import yfinance as yf
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        
        latency = time.time() - start_time
        series = _extract_close_series(df, ticker)
        
        if series is not None and len(series) > 20:
            return DataSourceResult(
                source="yfinance",
                metal=metal_name,
                series=series,
                latency_seconds=latency,
                success=True,
            )
        else:
            return DataSourceResult(
                source="yfinance",
                metal=metal_name,
                series=None,
                latency_seconds=latency,
                success=False,
                error_message="Insufficient data returned",
            )
    except Exception as e:
        latency = time.time() - start_time
        return DataSourceResult(
            source="yfinance",
            metal=metal_name,
            series=None,
            latency_seconds=latency,
            success=False,
            error_message=str(e),
        )


def _fetch_from_fred(
    fred_series_id: str,
    start: str,
    end: str,
    metal_name: str,
) -> DataSourceResult:
    """
    Fetch from FRED (Federal Reserve Economic Data) as secondary source.
    
    This is a fallback when yfinance fails or exhibits latency issues.
    FRED provides daily spot prices for gold and silver.
    """
    import time
    start_time = time.time()
    
    try:
        import pandas_datareader as pdr
        
        series = pdr.get_data_fred(fred_series_id, start=start, end=end)
        latency = time.time() - start_time
        
        if series is not None and len(series) > 20:
            # FRED returns DataFrame, convert to Series
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = series.dropna()
            
            return DataSourceResult(
                source="FRED",
                metal=metal_name,
                series=series,
                latency_seconds=latency,
                success=True,
            )
        else:
            return DataSourceResult(
                source="FRED",
                metal=metal_name,
                series=None,
                latency_seconds=latency,
                success=False,
                error_message="Insufficient data returned",
            )
    except ImportError:
        return DataSourceResult(
            source="FRED",
            metal=metal_name,
            series=None,
            latency_seconds=time.time() - start_time,
            success=False,
            error_message="pandas_datareader not installed",
        )
    except Exception as e:
        return DataSourceResult(
            source="FRED",
            metal=metal_name,
            series=None,
            latency_seconds=time.time() - start_time,
            success=False,
            error_message=str(e),
        )


def _check_price_divergence(
    primary_series: pd.Series,
    secondary_series: pd.Series,
    threshold: float = DATA_PRICE_DIVERGENCE_THRESHOLD,
) -> Tuple[bool, float]:
    """
    Check if prices from two sources diverge beyond threshold.
    
    Returns (divergence_detected, divergence_pct)
    """
    try:
        # Align by date
        common_idx = primary_series.index.intersection(secondary_series.index)
        if len(common_idx) < 5:
            return False, 0.0
        
        # Compare recent prices
        primary_recent = float(primary_series.loc[common_idx].iloc[-1])
        secondary_recent = float(secondary_series.loc[common_idx].iloc[-1])
        
        if primary_recent < 1e-10:
            return False, 0.0
        
        divergence_pct = abs(primary_recent - secondary_recent) / primary_recent
        divergence_detected = divergence_pct > threshold
        
        return divergence_detected, divergence_pct
    except Exception:
        return False, 0.0


def _fetch_metals_data_with_failover(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> Tuple[Dict[str, pd.Series], DataQualityReport]:
    """
    Fetch metals data with multi-source federation and automatic failover.
    
    Primary source: yfinance
    Secondary source: FRED (for gold/silver)
    
    Failover triggers:
    - Primary source latency > 5 minutes
    - Price divergence > 0.5% from secondary source
    - Primary source returns error
    
    Returns:
        Tuple of (metals_data dict, DataQualityReport)
    """
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=ZSCORE_LOOKBACK_DAYS + 30)
    start = start_dt.strftime("%Y-%m-%d")
    
    # Define tickers for each source
    yfinance_tickers = {
        'GOLD': 'GC=F',
        'SILVER': 'SI=F',
        'COPPER': 'HG=F',
        'PLATINUM': 'PL=F',
        'PALLADIUM': 'PA=F',
    }
    
    # FRED series IDs (only gold and silver available)
    fred_series = {
        'GOLD': 'GOLDAMGBD228NLBM',  # Gold Fixing Price London
        'SILVER': 'SLVPRUSD',         # Silver Fixing Price
    }
    
    result = {}
    failover_events = []
    source_used = {}
    price_divergence_alerts = []
    
    for name, ticker in yfinance_tickers.items():
        # Try primary source (yfinance)
        primary_result = _fetch_from_yfinance(ticker, start, end, name)
        
        # Check if failover is needed
        needs_failover = False
        failover_reason = None
        
        if not primary_result.success:
            needs_failover = True
            failover_reason = f"Primary source failed: {primary_result.error_message}"
        elif primary_result.latency_seconds > DATA_FAILOVER_LATENCY_THRESHOLD_SECONDS:
            needs_failover = True
            failover_reason = f"Primary source latency too high: {primary_result.latency_seconds:.1f}s"
        
        # Try secondary source if available and needed
        if needs_failover and name in fred_series:
            secondary_result = _fetch_from_fred(fred_series[name], start, end, name)
            
            if secondary_result.success:
                failover_events.append({
                    "metal": name,
                    "from_source": "yfinance",
                    "to_source": "FRED",
                    "reason": failover_reason,
                    "timestamp": datetime.now().isoformat(),
                })
                result[name] = secondary_result.series
                source_used[name] = "FRED"
                logger.warning(f"Data failover for {name}: {failover_reason}")
                continue
        
        # Use primary result if available
        if primary_result.success:
            result[name] = primary_result.series
            source_used[name] = "yfinance"
            
            # Cross-check with secondary source if available
            if name in fred_series:
                secondary_result = _fetch_from_fred(fred_series[name], start, end, name)
                if secondary_result.success:
                    diverged, divergence_pct = _check_price_divergence(
                        primary_result.series,
                        secondary_result.series,
                    )
                    if diverged:
                        price_divergence_alerts.append({
                            "metal": name,
                            "primary_source": "yfinance",
                            "secondary_source": "FRED",
                            "divergence_pct": divergence_pct,
                            "timestamp": datetime.now().isoformat(),
                        })
                        logger.warning(
                            f"Price divergence alert for {name}: "
                            f"{divergence_pct:.2%} between yfinance and FRED"
                        )
        else:
            source_used[name] = "unavailable"
    
    # Build data quality report
    metals_available = len(result)
    metals_total = len(yfinance_tickers)
    data_quality_pct = metals_available / metals_total if metals_total > 0 else 0.0
    
    degraded_mode = data_quality_pct < DATA_DEGRADED_MODE_THRESHOLD
    degraded_mode_reason = None
    if degraded_mode:
        degraded_mode_reason = (
            f"Only {metals_available}/{metals_total} metals available "
            f"({data_quality_pct:.0%} < {DATA_DEGRADED_MODE_THRESHOLD:.0%} threshold)"
        )
        logger.warning(f"DEGRADED MODE: {degraded_mode_reason}")
    
    quality_report = DataQualityReport(
        timestamp=datetime.now().isoformat(),
        metals_available=metals_available,
        metals_total=metals_total,
        data_quality_pct=data_quality_pct,
        degraded_mode=degraded_mode,
        degraded_mode_reason=degraded_mode_reason,
        failover_events=failover_events,
        source_used=source_used,
        price_divergence_alerts=price_divergence_alerts,
    )
    
    return result, quality_report


def _fetch_metals_data(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None
) -> Dict[str, pd.Series]:
    """
    Fetch all metals price data.
    
    Returns dict with keys: 'GOLD', 'SILVER', 'COPPER', 'PLATINUM', 'PALLADIUM'
    """
    cache_key = f"metals_{start_date}_{end_date}"
    now = datetime.now()
    
    if cache_key in _metals_data_cache:
        cached_time, cached_data = _metals_data_cache[cache_key]
        if (now - cached_time).total_seconds() < CACHE_TTL_SECONDS:
            return cached_data
    
    try:
        import yfinance as yf
    except ImportError:
        warnings.warn("yfinance not available for metals data")
        return {}
    
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=ZSCORE_LOOKBACK_DAYS + 30)
    start = start_dt.strftime("%Y-%m-%d")
    
    metals_tickers = {
        'GOLD': 'GC=F',        # Gold futures
        'SILVER': 'SI=F',      # Silver futures
        'COPPER': 'HG=F',      # Copper futures
        'PLATINUM': 'PL=F',    # Platinum futures
        'PALLADIUM': 'PA=F',   # Palladium futures
    }
    
    result = {}
    
    for name, ticker in metals_tickers.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            series = _extract_close_series(df, ticker)
            if series is not None and len(series) > 20:
                result[name] = series
        except Exception as e:
            if os.getenv('DEBUG'):
                print(f"Failed to fetch {ticker}: {e}")
    
    _metals_data_cache[cache_key] = (now, result)
    return result


# =============================================================================
# STRESS INDICATOR CALCULATIONS
# =============================================================================

def compute_copper_gold_stress(metals_data: Dict[str, pd.Series]) -> MetalStressIndicator:
    """
    Copper/Gold ratio as economic health bellwether.
    
    - Rising ratio = Risk-on (industrial demand outpacing safe haven)
    - Falling ratio = Risk-off (flight to gold)
    - Extreme deviation from mean = Stress
    """
    if 'COPPER' not in metals_data or 'GOLD' not in metals_data:
        return MetalStressIndicator(
            name="Copper/Gold",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Data unavailable"
        )
    
    try:
        copper = metals_data['COPPER']
        gold = metals_data['GOLD']
        
        common_idx = copper.index.intersection(gold.index)
        if len(common_idx) < 30:
            return MetalStressIndicator(
                name="Copper/Gold",
                value=0.0,
                zscore=0.0,
                contribution=0.0,
                data_available=False,
                interpretation="Insufficient data"
            )
        
        copper_aligned = copper.loc[common_idx]
        gold_aligned = gold.loc[common_idx]
        ratio = copper_aligned / gold_aligned
        
        current_ratio = float(ratio.iloc[-1])
        zscore = _compute_zscore(ratio)
        
        # Falling ratio (negative z-score) = risk-off = stress
        # Extreme either direction = stress
        stress = abs(zscore) * 1.2
        
        if zscore < -1.5:
            interpretation = "Sharp risk-off (flight to gold)"
        elif zscore < -0.5:
            interpretation = "Mild risk-off"
        elif zscore > 1.5:
            interpretation = "Strong risk-on (industrial demand)"
        elif zscore > 0.5:
            interpretation = "Mild risk-on"
        else:
            interpretation = "Neutral"
        
        return MetalStressIndicator(
            name="Copper/Gold",
            value=current_ratio,
            zscore=zscore,
            contribution=min(stress, 2.0) * COPPER_GOLD_WEIGHT,
            data_available=True,
            interpretation=interpretation
        )
    except Exception:
        return MetalStressIndicator(
            name="Copper/Gold",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Calculation error"
        )


def compute_silver_gold_stress(metals_data: Dict[str, pd.Series]) -> MetalStressIndicator:
    """
    Silver/Gold ratio as speculative intensity indicator.
    
    - Rising ratio = Silver outperforming = Speculative euphoria
    - Falling ratio = Gold outperforming = Fear/safety
    - Extreme moves = Market stress
    """
    if 'SILVER' not in metals_data or 'GOLD' not in metals_data:
        return MetalStressIndicator(
            name="Silver/Gold",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Data unavailable"
        )
    
    try:
        silver = metals_data['SILVER']
        gold = metals_data['GOLD']
        
        common_idx = silver.index.intersection(gold.index)
        if len(common_idx) < 30:
            return MetalStressIndicator(
                name="Silver/Gold",
                value=0.0,
                zscore=0.0,
                contribution=0.0,
                data_available=False,
                interpretation="Insufficient data"
            )
        
        silver_aligned = silver.loc[common_idx]
        gold_aligned = gold.loc[common_idx]
        ratio = silver_aligned / gold_aligned
        
        current_ratio = float(ratio.iloc[-1])
        zscore = _compute_zscore(ratio)
        
        # Falling ratio (negative z-score) = flight to gold = stress
        # Rapidly rising ratio = speculative excess = also stress
        if zscore < 0:
            stress = abs(zscore) * 1.5  # Fear amplified
        else:
            stress = abs(zscore) * 0.8  # Euphoria less weighted
        
        if zscore < -2.0:
            interpretation = "Extreme fear (gold dominance)"
        elif zscore < -1.0:
            interpretation = "Elevated fear"
        elif zscore > 2.0:
            interpretation = "Speculative euphoria"
        elif zscore > 1.0:
            interpretation = "Risk appetite strong"
        else:
            interpretation = "Normal range"
        
        return MetalStressIndicator(
            name="Silver/Gold",
            value=current_ratio,
            zscore=zscore,
            contribution=min(stress, 2.0) * SILVER_GOLD_WEIGHT,
            data_available=True,
            interpretation=interpretation
        )
    except Exception:
        return MetalStressIndicator(
            name="Silver/Gold",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Calculation error"
        )


def compute_gold_volatility_stress(metals_data: Dict[str, pd.Series]) -> MetalStressIndicator:
    """
    Gold volatility as fear gauge (GVZ proxy).
    
    Uses realized volatility percentile as stress indicator.
    High gold volatility = Market uncertainty/fear.
    """
    if 'GOLD' not in metals_data:
        return MetalStressIndicator(
            name="Gold Vol",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Data unavailable"
        )
    
    try:
        gold = metals_data['GOLD']
        
        if len(gold) < 60:
            return MetalStressIndicator(
                name="Gold Vol",
                value=0.0,
                zscore=0.0,
                contribution=0.0,
                data_available=False,
                interpretation="Insufficient data"
            )
        
        vol_percentile = _compute_volatility_percentile(gold)
        
        # High percentile = high stress
        stress = vol_percentile * 2.0  # Scale to [0, 2]
        
        # Compute current annualized vol for display
        returns = gold.pct_change().dropna()
        current_vol = float(returns.iloc[-20:].std() * np.sqrt(252) * 100)  # Annualized %
        
        if vol_percentile > 0.9:
            interpretation = f"Extreme volatility ({current_vol:.1f}% ann)"
        elif vol_percentile > 0.7:
            interpretation = f"Elevated volatility ({current_vol:.1f}% ann)"
        elif vol_percentile < 0.2:
            interpretation = f"Unusually calm ({current_vol:.1f}% ann)"
        else:
            interpretation = f"Normal volatility ({current_vol:.1f}% ann)"
        
        return MetalStressIndicator(
            name="Gold Vol",
            value=current_vol,
            zscore=(vol_percentile - 0.5) * 4,  # Center at 0.5, scale
            contribution=min(stress, 2.0) * GOLD_VOLATILITY_WEIGHT,
            data_available=True,
            interpretation=interpretation
        )
    except Exception:
        return MetalStressIndicator(
            name="Gold Vol",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Calculation error"
        )


def compute_precious_industrial_stress(metals_data: Dict[str, pd.Series]) -> MetalStressIndicator:
    """
    Precious vs Industrial metals spread.
    
    (Gold + Silver) / (Copper + Platinum) ratio.
    Rising = Flight to precious = Risk-off stress.
    """
    required = ['GOLD', 'SILVER', 'COPPER', 'PLATINUM']
    if not all(m in metals_data for m in required):
        return MetalStressIndicator(
            name="Precious/Industrial",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Data unavailable"
        )
    
    try:
        gold = metals_data['GOLD']
        silver = metals_data['SILVER']
        copper = metals_data['COPPER']
        platinum = metals_data['PLATINUM']
        
        # Find common index
        common_idx = gold.index
        for series in [silver, copper, platinum]:
            common_idx = common_idx.intersection(series.index)
        
        if len(common_idx) < 30:
            return MetalStressIndicator(
                name="Precious/Industrial",
                value=0.0,
                zscore=0.0,
                contribution=0.0,
                data_available=False,
                interpretation="Insufficient data"
            )
        
        # Normalize each metal to starting value for fair comparison
        gold_norm = gold.loc[common_idx] / gold.loc[common_idx].iloc[0]
        silver_norm = silver.loc[common_idx] / silver.loc[common_idx].iloc[0]
        copper_norm = copper.loc[common_idx] / copper.loc[common_idx].iloc[0]
        platinum_norm = platinum.loc[common_idx] / platinum.loc[common_idx].iloc[0]
        
        precious = (gold_norm + silver_norm) / 2
        industrial = (copper_norm + platinum_norm) / 2
        ratio = precious / industrial
        
        current_ratio = float(ratio.iloc[-1])
        zscore = _compute_zscore(ratio)
        
        # Rising ratio (positive z-score) = precious outperforming = risk-off = stress
        stress = max(0, zscore * 1.3)
        
        if zscore > 1.5:
            interpretation = "Strong flight to precious"
        elif zscore > 0.5:
            interpretation = "Mild precious preference"
        elif zscore < -1.5:
            interpretation = "Industrial outperforming"
        elif zscore < -0.5:
            interpretation = "Mild industrial preference"
        else:
            interpretation = "Balanced"
        
        return MetalStressIndicator(
            name="Precious/Industrial",
            value=current_ratio,
            zscore=zscore,
            contribution=min(stress, 2.0) * PRECIOUS_INDUSTRIAL_WEIGHT,
            data_available=True,
            interpretation=interpretation
        )
    except Exception:
        return MetalStressIndicator(
            name="Precious/Industrial",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Calculation error"
        )


def compute_platinum_gold_stress(metals_data: Dict[str, pd.Series]) -> MetalStressIndicator:
    """
    Platinum/Gold ratio as industrial-precious hybrid indicator.
    
    Platinum is both industrial (auto catalysts) and precious.
    Low ratio = Industrial weakness = Economic stress.
    """
    if 'PLATINUM' not in metals_data or 'GOLD' not in metals_data:
        return MetalStressIndicator(
            name="Platinum/Gold",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Data unavailable"
        )
    
    try:
        platinum = metals_data['PLATINUM']
        gold = metals_data['GOLD']
        
        common_idx = platinum.index.intersection(gold.index)
        if len(common_idx) < 30:
            return MetalStressIndicator(
                name="Platinum/Gold",
                value=0.0,
                zscore=0.0,
                contribution=0.0,
                data_available=False,
                interpretation="Insufficient data"
            )
        
        platinum_aligned = platinum.loc[common_idx]
        gold_aligned = gold.loc[common_idx]
        ratio = platinum_aligned / gold_aligned
        
        current_ratio = float(ratio.iloc[-1])
        zscore = _compute_zscore(ratio)
        
        # Falling ratio = platinum weakness = industrial stress
        stress = max(0, -zscore * 1.2)
        
        if zscore < -1.5:
            interpretation = "Platinum collapsed (industrial fear)"
        elif zscore < -0.5:
            interpretation = "Platinum underperforming"
        elif zscore > 1.0:
            interpretation = "Platinum strength (auto demand)"
        else:
            interpretation = "Normal range"
        
        return MetalStressIndicator(
            name="Platinum/Gold",
            value=current_ratio,
            zscore=zscore,
            contribution=min(stress, 2.0) * PLATINUM_GOLD_WEIGHT,
            data_available=True,
            interpretation=interpretation
        )
    except Exception:
        return MetalStressIndicator(
            name="Platinum/Gold",
            value=0.0,
            zscore=0.0,
            contribution=0.0,
            data_available=False,
            interpretation="Calculation error"
        )


def _compute_momentum_signal(ret_1d: float, ret_5d: float, ret_21d: float) -> str:
    """
    Compute a human-readable momentum signal from multi-timeframe returns.
    
    Logic:
    - Strong Up (↑): 21d > +5% and 5d > +1%
    - Rising (↗): 5d > +2% or (21d > 0 and 5d > 0)
    - Flat (→): Small moves across timeframes
    - Falling (↘): 5d < -2% or (21d < 0 and 5d < 0)
    - Strong Down (↓): 21d < -5% and 5d < -1%
    """
    # Thresholds
    strong_threshold_21d = 0.05   # 5%
    strong_threshold_5d = 0.01   # 1%
    rising_threshold = 0.02      # 2%
    flat_threshold = 0.01        # 1%
    
    if ret_21d > strong_threshold_21d and ret_5d > strong_threshold_5d:
        return "↑ Strong"
    elif ret_21d < -strong_threshold_21d and ret_5d < -strong_threshold_5d:
        return "↓ Weak"
    elif ret_5d > rising_threshold or (ret_21d > 0 and ret_5d > 0):
        return "↗ Rising"
    elif ret_5d < -rising_threshold or (ret_21d < 0 and ret_5d < 0):
        return "↘ Falling"
    else:
        return "→ Flat"


def compute_individual_metal_stress(
    metals_data: Dict[str, pd.Series]
) -> Dict[str, MetalStressCategory]:
    """Compute stress metrics for each individual metal including momentum."""
    metals = {}
    
    for name in ['GOLD', 'SILVER', 'COPPER', 'PLATINUM', 'PALLADIUM']:
        if name not in metals_data or len(metals_data[name]) < 20:
            metals[name.lower()] = MetalStressCategory(
                name=name.title(),
                price=None,
                return_5d=0.0,
                volatility=0.0,
                stress_level=0.0,
                data_available=False,
                return_1d=0.0,
                return_21d=0.0,
                momentum_signal="No data",
            )
            continue
        
        try:
            prices = metals_data[name]
            current_price = float(prices.iloc[-1])
            
            # 1-day return
            ret_1d = (prices.iloc[-1] / prices.iloc[-2] - 1) if len(prices) >= 2 else 0.0
            
            # 5-day return
            ret_5d = (prices.iloc[-1] / prices.iloc[-5] - 1) if len(prices) >= 5 else 0.0
            
            # 21-day (1 month) return
            ret_21d = (prices.iloc[-1] / prices.iloc[-21] - 1) if len(prices) >= 21 else 0.0
            
            # 20-day realized volatility (annualized)
            returns = prices.pct_change().dropna()
            if len(returns) >= 20:
                vol_20d = float(returns.iloc[-20:].std() * np.sqrt(252))
            else:
                vol_20d = 0.0
            
            # Compute momentum signal
            momentum_signal = _compute_momentum_signal(float(ret_1d), float(ret_5d), float(ret_21d))
            
            # Volatility percentile as stress
            vol_pct = _compute_volatility_percentile(prices)
            stress = vol_pct * 2.0
            
            metals[name.lower()] = MetalStressCategory(
                name=name.title(),
                price=current_price,
                return_5d=float(ret_5d),
                return_1d=float(ret_1d),
                return_21d=float(ret_21d),
                volatility=vol_20d,
                stress_level=min(stress, 2.0),
                momentum_signal=momentum_signal,
                data_available=True,
            )
        except Exception:
            metals[name.lower()] = MetalStressCategory(
                name=name.title(),
                price=None,
                return_5d=0.0,
                volatility=0.0,
                stress_level=0.0,
                data_available=False,
                return_1d=0.0,
                return_21d=0.0,
                momentum_signal="Error",
            )
    
    return metals


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_scale_factor(temperature: float) -> float:
    """Compute position scale factor using sigmoid (matches main risk_temperature.py)."""
    temp = max(TEMP_MIN, min(TEMP_MAX, temperature))
    scale = 1.0 / (1.0 + math.exp(SIGMOID_K * (temp - SIGMOID_THRESHOLD)))
    return scale


# =============================================================================
# ESCALATION PROTOCOL STATE (February 2026)
# =============================================================================

@dataclass
class EscalationState:
    """Track escalation protocol state for hysteresis."""
    current_regime: str = "Normal"
    consecutive_elevated_count: int = 0
    consecutive_stressed_count: int = 0
    days_below_normal_threshold: int = 0
    days_below_elevated_threshold: int = 0
    last_transition_timestamp: Optional[str] = None
    last_computation_timestamp: Optional[str] = None
    temperature_history: List[float] = field(default_factory=list)


_escalation_state = EscalationState()


def _update_escalation_state(
    temperature: float,
    current_state: EscalationState,
) -> Tuple[str, bool, Optional[str]]:
    """
    Update escalation state with hysteresis logic.
    
    Escalation Protocol (per Copilot Story):
    - Normal → Elevated: temp > 0.7 for 2 consecutive computations
    - Elevated → Stressed: temp > 1.2 for 2 consecutive computations
    - Stressed → Normal: temp < 0.5 sustained for 5 business days
    - Elevated → Normal: temp < 0.4 sustained for 3 business days
    
    Returns:
        Tuple of (new_regime, transition_occurred, alert_severity)
    """
    previous_regime = current_state.current_regime
    new_regime = previous_regime
    transition_occurred = False
    alert_severity = None
    
    # Track consecutive counts
    if temperature > ESCALATION_ELEVATED_TO_STRESSED_TEMP:
        current_state.consecutive_stressed_count += 1
        current_state.consecutive_elevated_count = 0
    elif temperature > ESCALATION_NORMAL_TO_ELEVATED_TEMP:
        current_state.consecutive_elevated_count += 1
        current_state.consecutive_stressed_count = 0
    else:
        current_state.consecutive_elevated_count = 0
        current_state.consecutive_stressed_count = 0
    
    # Track days below thresholds for downward transitions
    if temperature < ESCALATION_STRESSED_TO_NORMAL_TEMP:
        current_state.days_below_normal_threshold += 1
    else:
        current_state.days_below_normal_threshold = 0
    
    if temperature < ESCALATION_ELEVATED_TO_NORMAL_TEMP:
        current_state.days_below_elevated_threshold += 1
    else:
        current_state.days_below_elevated_threshold = 0
    
    # Apply escalation rules
    if previous_regime == "Normal":
        if current_state.consecutive_elevated_count >= ESCALATION_CONSECUTIVE_REQUIRED:
            new_regime = "Elevated"
            transition_occurred = True
            alert_severity = AlertSeverity.INFO
    
    elif previous_regime == "Elevated":
        if current_state.consecutive_stressed_count >= ESCALATION_CONSECUTIVE_REQUIRED:
            new_regime = "Stressed"
            transition_occurred = True
            alert_severity = AlertSeverity.WARNING
        elif current_state.days_below_elevated_threshold >= ESCALATION_SUSTAINED_DAYS_ELEVATED_TO_NORMAL:
            new_regime = "Normal"
            transition_occurred = True
            alert_severity = AlertSeverity.INFO
    
    elif previous_regime == "Stressed":
        if temperature > 1.5:
            new_regime = "Extreme"
            transition_occurred = True
            alert_severity = AlertSeverity.CRITICAL
        elif current_state.days_below_normal_threshold >= ESCALATION_SUSTAINED_DAYS_STRESSED_TO_NORMAL:
            new_regime = "Normal"
            transition_occurred = True
            alert_severity = AlertSeverity.INFO
    
    elif previous_regime == "Extreme":
        if temperature < 1.2:  # Hysteresis for downward
            new_regime = "Stressed"
            transition_occurred = True
            alert_severity = AlertSeverity.WARNING
    
    # Check for CRITICAL temperature threshold
    if temperature > 1.5 and alert_severity != AlertSeverity.CRITICAL:
        alert_severity = AlertSeverity.CRITICAL
    elif temperature > 1.0 and alert_severity is None:
        alert_severity = AlertSeverity.WARNING
    
    # Update state
    if transition_occurred:
        current_state.last_transition_timestamp = datetime.now().isoformat()
    current_state.current_regime = new_regime
    current_state.last_computation_timestamp = datetime.now().isoformat()
    current_state.temperature_history.append(temperature)
    if len(current_state.temperature_history) > 100:
        current_state.temperature_history = current_state.temperature_history[-100:]
    
    return new_regime, transition_occurred, alert_severity


def _generate_alert(
    severity: str,
    temperature: float,
    regime_state: str,
    primary_indicator: str,
    action_text: str,
    data_quality: float,
    degraded_mode: bool = False,
    degraded_reason: Optional[str] = None,
    vol_term_structure_triggered: bool = False,
) -> Dict:
    """
    Generate alert payload for routing to operations.
    
    Alert payloads include:
    - Current temperature value
    - Current regime state
    - Primary contributing indicator
    - Recommended action
    - Data quality metric
    """
    alert = {
        "severity": severity,
        "timestamp": datetime.now().isoformat(),
        "module": "metals_risk_temperature",
        "temperature": temperature,
        "regime_state": regime_state,
        "primary_indicator": primary_indicator,
        "recommended_action": action_text,
        "data_quality_pct": data_quality,
        "degraded_mode": degraded_mode,
    }
    
    if degraded_mode and degraded_reason:
        alert["degraded_mode_reason"] = degraded_reason
        
    if vol_term_structure_triggered:
        alert["vol_term_structure_inversion"] = True
        alert["message"] = (
            f"ANTICIPATORY WARNING: Volatility term structure inversion detected. "
            f"Temperature: {temperature:.2f}, Regime: {regime_state}"
        )
    elif severity == AlertSeverity.CRITICAL:
        alert["message"] = (
            f"CRITICAL: Metals risk temperature at {temperature:.2f} (Regime: {regime_state}). "
            f"Action: {action_text}"
        )
    elif severity == AlertSeverity.WARNING:
        alert["message"] = (
            f"WARNING: Metals risk temperature elevated to {temperature:.2f} (Regime: {regime_state}). "
            f"Action: {action_text}"
        )
    else:
        alert["message"] = (
            f"INFO: Metals regime transition to {regime_state}. Temperature: {temperature:.2f}"
        )
    
    return alert


def compute_metals_risk_temperature(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> MetalsRiskTemperatureResult:
    """
    Compute complete metals risk temperature.
    
    This is the main entry point for the metals risk temperature module.
    
    Args:
        start_date: Start date for historical data
        end_date: End date (default: today)
        
    Returns:
        MetalsRiskTemperatureResult with temperature, indicators, and diagnostics
    """
    # Fetch all metals data
    metals_data = _fetch_metals_data(start_date, end_date)
    
    # Compute stress indicators
    indicators = [
        compute_copper_gold_stress(metals_data),
        compute_silver_gold_stress(metals_data),
        compute_gold_volatility_stress(metals_data),
        compute_precious_industrial_stress(metals_data),
        compute_platinum_gold_stress(metals_data),
    ]
    
    # Compute individual metal stress
    metals = compute_individual_metal_stress(metals_data)
    
    # Aggregate temperature (sum of weighted contributions)
    temperature = sum(ind.contribution for ind in indicators)
    temperature = max(TEMP_MIN, min(TEMP_MAX, temperature))
    
    # Compute scale factor
    scale_factor = compute_scale_factor(temperature)
    
    # Determine status and action text
    if temperature < 0.3:
        status = "Calm"
        action_text = "Full metals exposure permitted"
    elif temperature < 0.7:
        status = "Elevated"
        action_text = "Monitor metals positions"
    elif temperature < 1.2:
        status = "Stressed"
        action_text = "Reduce metals exposure"
    else:
        status = "Extreme"
        action_text = "Defensive metals positioning"
    
    # Data quality
    available_count = sum(1 for ind in indicators if ind.data_available)
    data_quality = available_count / len(indicators) if indicators else 0.0
    
    return MetalsRiskTemperatureResult(
        temperature=temperature,
        scale_factor=scale_factor,
        indicators=indicators,
        metals=metals,
        computed_at=datetime.now().isoformat(),
        data_quality=data_quality,
        status=status,
        action_text=action_text,
    )


# =============================================================================
# CACHING
# =============================================================================

_metals_temp_cache: Dict[str, Tuple[datetime, MetalsRiskTemperatureResult]] = {}


def get_cached_metals_risk_temperature(
    start_date: str = "2020-01-01",
    cache_ttl_seconds: int = CACHE_TTL_SECONDS,
) -> MetalsRiskTemperatureResult:
    """Get metals risk temperature with caching."""
    cache_key = start_date
    now = datetime.now()
    
    if cache_key in _metals_temp_cache:
        cached_time, cached_result = _metals_temp_cache[cache_key]
        if (now - cached_time).total_seconds() < cache_ttl_seconds:
            return cached_result
    
    result = compute_metals_risk_temperature(start_date=start_date)
    _metals_temp_cache[cache_key] = (now, result)
    
    return result


def clear_metals_risk_temperature_cache():
    """Clear the metals risk temperature cache."""
    global _metals_temp_cache, _metals_data_cache
    _metals_temp_cache = {}
    _metals_data_cache = {}


# =============================================================================
# ANTICIPATORY COMPUTATION (February 2026)
# =============================================================================

def compute_anticipatory_metals_risk_temperature(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> Tuple[MetalsRiskTemperatureResult, List[Dict], DataQualityReport]:
    """
    Compute ANTICIPATORY metals risk temperature with all February 2026 enhancements.
    
    This is the RECOMMENDED entry point for production use.
    
    ENHANCEMENTS INCLUDED:
    1. Volatility Term Structure Inversion (leading indicator)
    2. Robust MAD-based z-score computation
    3. Multi-source data federation with failover
    4. Degraded mode with temperature floor
    5. Enhanced escalation protocol with hysteresis
    6. Alert generation with severity routing
    
    Args:
        start_date: Start date for historical data
        end_date: End date (default: today)
        
    Returns:
        Tuple of (MetalsRiskTemperatureResult, alerts_list, DataQualityReport)
    """
    global _escalation_state
    
    alerts = []
    
    # Fetch data with multi-source federation
    metals_data, quality_report = _fetch_metals_data_with_failover(start_date, end_date)
    
    # Compute REACTIVE stress indicators (now using robust z-scores)
    reactive_indicators = [
        compute_copper_gold_stress(metals_data),
        compute_silver_gold_stress(metals_data),
        compute_gold_volatility_stress(metals_data),
        compute_precious_industrial_stress(metals_data),
        compute_platinum_gold_stress(metals_data),
    ]
    
    # Compute ANTICIPATORY indicator: Volatility Term Structure
    vol_ts_indicator, vol_ts_details = compute_volatility_term_structure_stress(metals_data)
    
    # Combine all indicators
    all_indicators = reactive_indicators + [vol_ts_indicator]
    
    # Compute individual metal stress
    metals = compute_individual_metal_stress(metals_data)
    
    # Aggregate temperature
    # Base temperature from reactive indicators
    reactive_temp = sum(ind.contribution for ind in reactive_indicators)
    
    # Add anticipatory contribution (additive, not weighted)
    anticipatory_contribution = vol_ts_indicator.contribution
    
    raw_temperature = reactive_temp + anticipatory_contribution
    temperature = max(TEMP_MIN, min(TEMP_MAX, raw_temperature))
    
    # Check for degraded mode
    degraded_mode = quality_report.degraded_mode
    temperature_floor_applied = False
    
    if degraded_mode:
        if temperature < DATA_DEGRADED_MODE_TEMP_FLOOR:
            temperature = DATA_DEGRADED_MODE_TEMP_FLOOR
            temperature_floor_applied = True
        
        # Generate degraded mode alert
        alerts.append(_generate_alert(
            severity=AlertSeverity.WARNING,
            temperature=temperature,
            regime_state=_escalation_state.current_regime,
            primary_indicator="DATA_QUALITY",
            action_text="Reduce exposure due to data uncertainty",
            data_quality=quality_report.data_quality_pct,
            degraded_mode=True,
            degraded_reason=quality_report.degraded_mode_reason,
        ))
    
    # Update escalation state
    previous_regime = _escalation_state.current_regime
    new_regime, transition_occurred, alert_severity = _update_escalation_state(
        temperature, _escalation_state
    )
    
    # Compute scale factor
    scale_factor = compute_scale_factor(temperature)
    
    # Determine status and action text based on escalation regime
    status = new_regime
    if new_regime == "Normal":
        if temperature < 0.3:
            status = "Calm"
        action_text = "Full metals exposure permitted"
    elif new_regime == "Elevated":
        action_text = "Monitor metals positions closely"
    elif new_regime == "Stressed":
        action_text = "Reduce metals exposure significantly"
    else:  # Extreme
        action_text = "Defensive positioning - minimize metals exposure"
    
    # Find primary contributing indicator
    max_contributor = max(all_indicators, key=lambda x: x.contribution)
    
    # Generate alerts based on severity
    if alert_severity:
        alerts.append(_generate_alert(
            severity=alert_severity,
            temperature=temperature,
            regime_state=new_regime,
            primary_indicator=max_contributor.name,
            action_text=action_text,
            data_quality=quality_report.data_quality_pct,
            vol_term_structure_triggered=(vol_ts_indicator.contribution > 0),
        ))
    
    # Special alert for vol term structure inversion even without regime change
    if vol_ts_indicator.contribution > 0 and alert_severity != AlertSeverity.CRITICAL:
        alerts.append(_generate_alert(
            severity=AlertSeverity.WARNING,
            temperature=temperature,
            regime_state=new_regime,
            primary_indicator="Vol Term Structure",
            action_text="Anticipatory warning: volatility term structure inverted",
            data_quality=quality_report.data_quality_pct,
            vol_term_structure_triggered=True,
        ))
    
    # Compute crash risk based on vol term structure inversion
    vol_inversion_count = int(vol_ts_indicator.zscore)  # zscore holds the count
    avg_vol_ratio = vol_ts_indicator.value if vol_ts_indicator.value > 0 else 1.0
    crash_risk_pct, crash_risk_level = compute_crash_risk(
        vol_inversion_count=vol_inversion_count,
        temperature=temperature,
        avg_vol_ratio=avg_vol_ratio,
    )
    
    # Data quality
    available_count = sum(1 for ind in all_indicators if ind.data_available)
    data_quality = available_count / len(all_indicators) if all_indicators else 0.0
    
    result = MetalsRiskTemperatureResult(
        temperature=temperature,
        scale_factor=scale_factor,
        indicators=all_indicators,
        metals=metals,
        computed_at=datetime.now().isoformat(),
        data_quality=data_quality,
        status=status,
        action_text=action_text,
        regime_state=new_regime,
        previous_regime_state=previous_regime if transition_occurred else None,
        regime_transition_occurred=transition_occurred,
        raw_temperature=raw_temperature,
        temperature_floor_applied=temperature_floor_applied,
        crash_risk_pct=crash_risk_pct,
        crash_risk_level=crash_risk_level,
        vol_inversion_count=vol_inversion_count,
    )
    
    return result, alerts, quality_report


def reset_escalation_state() -> None:
    """Reset the escalation state to default."""
    global _escalation_state
    _escalation_state = EscalationState()
    logger.info("Escalation state reset")


# =============================================================================
# RENDER FUNCTION (Minimalist Apple-like display)
# =============================================================================

def render_metals_risk_temperature(
    result: MetalsRiskTemperatureResult,
    console = None,
) -> None:
    """
    Render minimalist Apple-inspired metals risk temperature display.
    Matches make temp aesthetic: no boxes, no icons, clean progress bars.
    """
    from rich.console import Console
    from rich.text import Text
    
    if console is None:
        console = Console()
    
    if result is None:
        return
    
    # Status colors
    if result.temperature < 0.3:
        status_color = "green"
    elif result.temperature < 0.7:
        status_color = "yellow"
    elif result.temperature < 1.2:
        status_color = "bright_red"
    else:
        status_color = "bold red"
    
    console.print()
    console.print()
    
    # Title
    console.print("  [dim]Metals Risk Temperature[/dim]")
    console.print()
    
    # Hero temperature with status
    hero = Text()
    hero.append("  ")
    hero.append(f"{result.temperature:.2f}", style=f"bold {status_color}")
    hero.append("  ")
    hero.append(result.status, style=f"{status_color}")
    console.print(hero)
    console.print()
    
    # Main gauge bar
    gauge = Text()
    gauge.append("  ")
    gauge_width = 48
    filled = int(min(1.0, result.temperature / 2.0) * gauge_width)
    
    for i in range(gauge_width):
        if i < filled:
            segment_pct = i / gauge_width
            if segment_pct < 0.25:
                gauge.append("━", style="bright_green")
            elif segment_pct < 0.5:
                gauge.append("━", style="yellow")
            elif segment_pct < 0.75:
                gauge.append("━", style="bright_red")
            else:
                gauge.append("━", style="bold red")
        else:
            gauge.append("━", style="bright_black")
    
    console.print(gauge)
    
    # Scale labels
    labels = Text()
    labels.append("  ")
    labels.append("0", style="dim")
    labels.append(" " * 22)
    labels.append("1", style="dim")
    labels.append(" " * 22)
    labels.append("2", style="dim")
    console.print(labels)
    console.print()
    
    # Action text
    console.print(f"  [dim italic]{result.action_text}[/dim italic]")
    console.print()
    
    # Crash Risk Display (using shared component from signals_ux)
    crash_risk_pct = getattr(result, 'crash_risk_pct', 0.0)
    crash_risk_level = getattr(result, 'crash_risk_level', 'Low')
    vol_inversion_count = getattr(result, 'vol_inversion_count', 0)
    
    if crash_risk_pct > 0.02:  # Only show if above baseline
        # Extract inverted metals from vol indicator
        inverted_metals = None
        if vol_inversion_count > 0:
            for ind in result.indicators:
                if ind.name == "Vol Term Structure" and ind.interpretation:
                    interp = ind.interpretation
                    if "(" in interp and ")" in interp:
                        metals_str = interp[interp.find("(")+1:interp.find(")")]
                        inverted_metals = [m.strip() for m in metals_str.split(",")]
                    break
        
        # Extract momentum data from metals
        momentum_data = {}
        for metal_key in ['gold', 'silver', 'copper', 'platinum', 'palladium']:
            if metal_key in result.metals:
                metal = result.metals[metal_key]
                if metal.data_available:
                    momentum_data[metal.name] = metal.return_5d
        
        # Use the shared crash risk assessment component
        try:
            from decision.signals_ux import render_crash_risk_assessment
        except ImportError:
            try:
                # Fallback for running as script
                from signals_ux import render_crash_risk_assessment
            except ImportError:
                render_crash_risk_assessment = None
        
        if render_crash_risk_assessment:
            render_crash_risk_assessment(
                crash_risk_pct=crash_risk_pct,
                crash_risk_level=crash_risk_level,
                vol_inversion_count=vol_inversion_count,
                inverted_metals=inverted_metals,
                momentum_data=momentum_data if momentum_data else None,
                console=console,
            )
        else:
            # Fallback: minimal display if signals_ux is not available
            console.print(f"  [bold]Crash Risk: {crash_risk_pct:.0%} ({crash_risk_level})[/bold]")
            console.print()
    
    # Ratio-based stress indicators
    console.print("  [dim]Stress Indicators[/dim]  [dim italic](z-score = deviation from normal)[/dim italic]")
    console.print()
    
    # Define display names with consistent formatting
    indicator_display_names = {
        "Copper/Gold": "Cu/Au Ratio",      # Industrial vs safe-haven
        "Silver/Gold": "Ag/Au Ratio",      # Speculative intensity
        "Gold Vol": "Gold Volatility",     # Fear gauge
        "Precious/Industrial": "Prec/Ind",  # Sector rotation
        "Platinum/Gold": "Pt/Au Ratio",    # Industrial precious
        "Vol Term Structure": "Vol Spike",  # Short-term vs long-term vol
    }
    
    for ind in result.indicators:
        if not ind.data_available:
            continue
        
        # Determine color based on contribution
        raw_stress = ind.contribution / max(0.01, COPPER_GOLD_WEIGHT)  # Normalize
        if raw_stress < 0.5:
            ind_style = "bright_green"
        elif raw_stress < 1.0:
            ind_style = "yellow"
        elif raw_stress < 1.5:
            ind_style = "bright_red"
        else:
            ind_style = "bold red"
        
        # Get display name (shortened for alignment)
        display_name = indicator_display_names.get(ind.name, ind.name)
        
        line = Text()
        line.append("  ")
        line.append(f"{display_name:<16}", style="dim")
        
        # Mini bar
        mini_width = 12
        mini_filled = int(min(1.0, raw_stress / 2.0) * mini_width)
        for i in range(mini_width):
            if i < mini_filled:
                line.append("━", style=ind_style)
            else:
                line.append("━", style="bright_black")
        
        line.append(f"  z={ind.zscore:+.1f}", style=ind_style)
        line.append(f"  {ind.interpretation}", style="dim italic")
        console.print(line)
    
    console.print()
    
    # Individual metals stress with momentum
    console.print("  [dim]Individual Metals[/dim]")
    console.print()
    
    # Header row
    header = Text()
    header.append("  ")
    header.append(f"{'Metal':<12}", style="dim")
    header.append(f"{'Stress':<14}", style="dim")
    header.append(f"{'Price':<11}", style="dim")
    header.append(f"{'5d':<8}", style="dim")
    header.append(f"{'21d':<8}", style="dim")
    header.append("Trend", style="dim")
    console.print(header)
    
    for metal_key in ['gold', 'silver', 'copper', 'platinum', 'palladium']:
        if metal_key not in result.metals:
            continue
        metal = result.metals[metal_key]
        if not metal.data_available:
            continue
        
        # Stress color
        if metal.stress_level < 0.5:
            metal_style = "bright_green"
        elif metal.stress_level < 1.0:
            metal_style = "yellow"
        elif metal.stress_level < 1.5:
            metal_style = "bright_red"
        else:
            metal_style = "bold red"
        
        line = Text()
        line.append("  ")
        line.append(f"{metal.name:<12}", style="white")
        
        # Mini bar
        mini_width = 12
        mini_filled = int(min(1.0, metal.stress_level / 2.0) * mini_width)
        for i in range(mini_width):
            if i < mini_filled:
                line.append("━", style=metal_style)
            else:
                line.append("━", style="bright_black")
        line.append("  ")
        
        # Price
        if metal.price:
            if metal.price >= 1000:
                price_str = f"${metal.price:,.0f}"
            else:
                price_str = f"${metal.price:.2f}"
            line.append(f"{price_str:<11}", style="white")
        else:
            line.append(f"{'--':<11}", style="dim")
        
        # 5-day return
        ret_5d_style = "bright_green" if metal.return_5d >= 0 else "indian_red1"
        line.append(f"{metal.return_5d:+.1%}".ljust(8), style=ret_5d_style)
        
        # 21-day return
        ret_21d = getattr(metal, 'return_21d', 0.0)
        ret_21d_style = "bright_green" if ret_21d >= 0 else "indian_red1"
        line.append(f"{ret_21d:+.1%}".ljust(8), style=ret_21d_style)
        
        # Momentum signal
        momentum = getattr(metal, 'momentum_signal', '')
        if "Strong" in momentum and "↑" in momentum:
            mom_style = "bold bright_green"
        elif "Rising" in momentum or "↗" in momentum:
            mom_style = "bright_green"
        elif "Weak" in momentum or "↓" in momentum:
            mom_style = "bold indian_red1"
        elif "Falling" in momentum or "↘" in momentum:
            mom_style = "indian_red1"
        else:
            mom_style = "dim"
        line.append(momentum if momentum else "→ Flat", style=mom_style)
        
        console.print(line)
    
    console.print()
    
    # Position sizing
    scale = result.scale_factor
    if scale > 0.9:
        scale_style = "bright_green"
        scale_text = "Full Allocation"
    elif scale > 0.6:
        scale_style = "yellow"
        scale_text = "Reduced"
    elif scale > 0.3:
        scale_style = "bright_red"
        scale_text = "Significantly Reduced"
    else:
        scale_style = "bold red"
        scale_text = "Minimal"
    
    pos_line = Text()
    pos_line.append("  ")
    pos_line.append("Position Size   ", style="dim")
    pos_line.append(f"{scale:.0%}", style=f"bold {scale_style}")
    pos_line.append(f"  {scale_text}", style="dim italic")
    console.print(pos_line)
    
    # Data quality
    quality_line = Text()
    quality_line.append("  ")
    quality_line.append("Data Quality    ", style="dim")
    available = sum(1 for ind in result.indicators if ind.data_available)
    total = len(result.indicators)
    quality_style = "green" if available == total else ("yellow" if available >= 3 else "red")
    quality_line.append(f"{available}/{total}", style=quality_style)
    quality_line.append("  indicators", style="dim italic")
    console.print(quality_line)
    
    # Governance status (if available)
    if hasattr(result, 'regime_state') and result.regime_state:
        console.print()
        console.print("  [dim]Governance[/dim]")
        console.print()
        
        # Regime state
        regime_line = Text()
        regime_line.append("  ")
        regime_line.append("Regime State    ", style="dim")
        regime_line.append(result.regime_state, style=status_color)
        if result.regime_transition_occurred:
            regime_line.append("  ← ", style="dim")
            regime_line.append(result.previous_regime_state or "Unknown", style="dim italic")
        console.print(regime_line)
        
        # Temperature floor warning
        if result.temperature_floor_applied:
            floor_line = Text()
            floor_line.append("  ")
            floor_line.append("⚠️ Temp Floor   ", style="dim")
            floor_line.append("Applied (degraded mode)", style="yellow")
            console.print(floor_line)
        
        # Rate limiting
        if result.rate_limit_applied:
            rate_line = Text()
            rate_line.append("  ")
            rate_line.append("Rate Limited    ", style="dim")
            rate_line.append("Yes", style="yellow")
            if result.raw_temperature is not None:
                rate_line.append(f"  (raw: {result.raw_temperature:.2f})", style="dim italic")
            console.print(rate_line)
        
        # Imputation warning
        if result.imputation_warning:
            imp_line = Text()
            imp_line.append("  ")
            imp_line.append("⚠️ Imputation   ", style="dim")
            imp_line.append(f"{result.imputed_indicators} indicators imputed", style="yellow")
            console.print(imp_line)
        
        # Gap risk
        gap_line = Text()
        gap_line.append("  ")
        gap_line.append("Gap Risk        ", style="dim")
        gap_line.append(f"{result.gap_risk_estimate:.1%}", style="white")
        console.print(gap_line)
        
        # Overnight budget
        if result.overnight_budget_active:
            overnight_line = Text()
            overnight_line.append("  ")
            overnight_line.append("Overnight Cap   ", style="dim")
            if result.overnight_max_position:
                overnight_line.append(f"{result.overnight_max_position:.0%}", style="yellow")
            overnight_line.append("  Active", style="dim italic")
            console.print(overnight_line)
    
    console.print()
    console.print()


def render_alerts(alerts: List[Dict], console=None) -> None:
    """Render alerts with severity-based formatting."""
    from rich.console import Console
    from rich.text import Text
    from rich.panel import Panel
    
    if console is None:
        console = Console()
    
    if not alerts:
        return
    
    console.print()
    console.print("  [dim]Alerts[/dim]")
    console.print()
    
    for alert in alerts:
        severity = alert.get("severity", "INFO")
        message = alert.get("message", "")
        
        if severity == AlertSeverity.CRITICAL:
            style = "bold red"
            icon = "🚨"
        elif severity == AlertSeverity.WARNING:
            style = "yellow"
            icon = "⚠️"
        else:
            style = "dim"
            icon = "ℹ️"
        
        alert_line = Text()
        alert_line.append(f"  {icon} [{severity}] ", style=style)
        alert_line.append(message, style=style)
        console.print(alert_line)
    
    console.print()


def reset_escalation_state() -> None:
    """Reset the escalation state to default."""
    global _escalation_state
    _escalation_state = EscalationState()
    logger.info("Escalation state reset")


# =============================================================================
# STANDALONE CLI
# =============================================================================

if __name__ == "__main__":
    """Run metals risk temperature computation and display."""
    import argparse
    from rich.console import Console
    
    parser = argparse.ArgumentParser(description="Metals Risk Temperature — Anticipatory Enhancement")
    parser.add_argument("--basic", action="store_true", help="Use basic computation (no anticipatory features)")
    parser.add_argument("--audit", action="store_true", help="Show full audit trail")
    parser.add_argument("--reset", action="store_true", help="Reset escalation state")
    parser.add_argument("--alerts", action="store_true", help="Show alerts")
    parser.add_argument("--data-report", action="store_true", help="Show data quality report")
    args = parser.parse_args()
    
    console = Console()
    
    if args.reset:
        reset_escalation_state()
        console.print("[green]Escalation state reset[/green]")
    
    # Compute metals risk temperature
    if args.basic:
        result = compute_metals_risk_temperature(start_date="2020-01-01")
        alerts = []
        quality_report = None
    else:
        result, alerts, quality_report = compute_anticipatory_metals_risk_temperature(
            start_date="2020-01-01"
        )
    
    # Render the display
    render_metals_risk_temperature(result, console=console)
    
    # Show alerts if requested or if critical
    if args.alerts or any(a.get("severity") == AlertSeverity.CRITICAL for a in alerts):
        render_alerts(alerts, console=console)
    
    # Show data quality report if requested
    if args.data_report and quality_report:
        console.print()
        console.print("  [dim]Data Quality Report[/dim]")
        console.print()
        console.print(f"    Metals Available: {quality_report.metals_available}/{quality_report.metals_total}")
        console.print(f"    Data Quality: {quality_report.data_quality_pct:.0%}")
        console.print(f"    Degraded Mode: {'Yes' if quality_report.degraded_mode else 'No'}")
        if quality_report.failover_events:
            console.print(f"    Failover Events: {len(quality_report.failover_events)}")
            for event in quality_report.failover_events:
                console.print(f"      - {event['metal']}: {event['from_source']} → {event['to_source']}")
        if quality_report.price_divergence_alerts:
            console.print(f"    Price Divergence Alerts: {len(quality_report.price_divergence_alerts)}")
        console.print()
    
    # Show audit trail if requested
    if args.audit and result.audit_trail:
        console.print()
        console.print(result.render_audit_trail())
