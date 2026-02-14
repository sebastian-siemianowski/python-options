"""
===============================================================================
MARKET TEMPERATURE MODULE
===============================================================================

Comprehensive US Equity Market Health Assessment (February 2026)

DESIGN PRINCIPLE:
    "The market tells you everything — if you listen to the right instruments."

This module computes a comprehensive market temperature from multiple stock
universe segments to give a complete picture of market health and risk:

UNIVERSE SEGMENTS:
    1. Top 100 US Stocks (40%): Mega-cap bellwethers (Apple, Microsoft, etc.)
       - Represents institutional flows and macro sentiment
       
    2. S&P 500 Proxy (30%): Broad market health via SPY components
       - Core market breadth indicator
       
    3. Russell 2000 Small-Caps (20%): Risk appetite indicator
       - Small caps lead in risk-on, lag in risk-off
       
    4. Growth vs Value Spread (10%): Rotation signals
       - QQQ/IWD spread reveals regime shifts

COMPUTED METRICS:
    - Market Temperature: Aggregate stress score ∈ [0, 2]
    - Crash Risk: Probability of significant drawdown (vol inversion, etc.)
    - Momentum: Multi-timeframe momentum across segments
    - Breadth: Advance/decline ratios and participation
    - Volatility Regime: Current vol vs historical percentiles
    - Correlation Stress: Rising correlations = systemic risk

REFERENCES:
    Expert Panel Design (February 2026)
    Mirrors metals_risk_temperature.py and risk_temperature.py patterns

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
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# COPULA-BASED CORRELATION STRESS (February 2026)
# =============================================================================
# Import copula correlation module for tail dependency analysis
# Falls back to Pearson correlation if copula module unavailable
# =============================================================================
try:
    from calibration.copula_correlation import (
        compute_copula_correlation_stress,
        compute_smooth_scale_factor,
        CopulaCorrelationStress,
        COPULA_CORRELATION_AVAILABLE,
    )
    USE_COPULA_CORRELATION = True
except ImportError:
    USE_COPULA_CORRELATION = False
    COPULA_CORRELATION_AVAILABLE = False

# =============================================================================
# PYTORCH FORECASTING (February 2026)
# =============================================================================
# N-BEATS and TFT models for currency/equity forecasting
# Falls back to classical + Prophet if unavailable
# =============================================================================
try:
    from pytorch_forecasting import NBeatsNet
    PYTORCH_FORECASTING_AVAILABLE = True
    USE_PYTORCH_FORECASTING = True
except ImportError:
    PYTORCH_FORECASTING_AVAILABLE = False
    USE_PYTORCH_FORECASTING = False

# =============================================================================
# TEMPORAL FUSION TRANSFORMER (TFT) INTEGRATION (February 2026)
# =============================================================================
# TFT provides attention-based multi-horizon forecasting with:
# - Variable selection networks for automatic feature importance
# - Interpretable attention over historical patterns
# - Multi-quantile output for proper uncertainty quantification
# Reference: Lim et al. (2021) "Temporal Fusion Transformers"
# =============================================================================
try:
    from decision.tft_forecaster import (
        tft_forecast,
        is_tft_available,
        TFT_AVAILABLE,
    )
    USE_TFT = TFT_AVAILABLE
except ImportError:
    USE_TFT = False
    TFT_AVAILABLE = False
    
    def tft_forecast(prices, horizons, asset_type, asset_name):
        """Stub when TFT not available."""
        n = len(horizons) if horizons else 7
        return [0.0] * n, [(0.0, 0.0)] * n, "Low"
    
    def is_tft_available():
        return False


# =============================================================================
# STANDARD FORECAST HORIZONS (Canonical Definition — February 2026)
# =============================================================================
# All forecast functions MUST use these horizons to prevent index misalignment.
# Callers should use get_forecast_by_horizon() instead of positional indices.
# =============================================================================
STANDARD_HORIZONS = [1, 3, 7, 30, 90, 180, 365]
HORIZON_INDEX = {h: i for i, h in enumerate(STANDARD_HORIZONS)}


def get_forecast_by_horizon(result: tuple, horizon: int) -> float:
    """
    Get forecast value for a specific horizon from ensemble_forecast result.
    
    Args:
        result: Tuple from ensemble_forecast (7 floats + confidence string)
        horizon: One of STANDARD_HORIZONS [1, 3, 7, 30, 90, 180, 365]
        
    Returns:
        Forecast percentage for that horizon, or 0.0 if not found
    """
    idx = HORIZON_INDEX.get(horizon)
    if idx is not None and idx < len(result) - 1:
        return float(result[idx])
    return 0.0


def get_forecast_confidence(result: tuple) -> str:
    """Get confidence string from ensemble_forecast result."""
    if result and len(result) > 0:
        return str(result[-1]) if isinstance(result[-1], str) else "Low"
    return "Low"


# =============================================================================
# ELITE FORECASTING ENGINE (Professor-Grade Multi-Model Ensemble)
# =============================================================================
# Architecture: Multi-model Bayesian ensemble with regime-aware weighting
# Models: Kalman Filter, GARCH(1,1), Ornstein-Uhlenbeck, TFT, Momentum
# Horizon-adaptive: Different model weights per forecast horizon
# TFT Integration: Attention-based forecasts weighted into ensemble
# =============================================================================

def _kalman_forecast(returns: np.ndarray, horizons: list) -> list:
    """Kalman Filter forecast with adaptive state estimation."""
    try:
        if len(returns) < 20:
            return [0.0] * len(horizons)
        alpha = 0.94
        drift_state = returns[0]
        for r in returns[1:]:
            drift_state = alpha * drift_state + (1 - alpha) * r
        forecasts = []
        for h in horizons:
            decay = np.exp(-h / 180.0)
            fc = drift_state * h * decay
            pct = (np.exp(fc) - 1) * 100
            forecasts.append(float(pct))
        return forecasts
    except Exception:
        return [0.0] * len(horizons)


def _garch_forecast(returns: np.ndarray, horizons: list) -> list:
    """GARCH(1,1) volatility-adjusted forecast."""
    try:
        if len(returns) < 30:
            return [0.0] * len(horizons)
        omega, alpha, beta = 0.00001, 0.10, 0.85
        var_t = np.var(returns)
        for r in returns[-20:]:
            var_t = omega + alpha * r**2 + beta * var_t
        drift = np.mean(returns[-10:])
        vol_mult = 1.0 / (1.0 + np.sqrt(var_t) * 10)
        forecasts = []
        for h in horizons:
            fc = drift * h * vol_mult * np.exp(-h / 365.0)
            pct = (np.exp(fc) - 1) * 100
            forecasts.append(float(pct))
        return forecasts
    except Exception:
        return [0.0] * len(horizons)


def _ou_forecast(prices: pd.Series, horizons: list) -> list:
    """Ornstein-Uhlenbeck mean reversion forecast."""
    try:
        if len(prices) < 60:
            return [0.0] * len(horizons)
        current = float(prices.iloc[-1])
        ma_200 = float(prices.iloc[-min(200, len(prices)):].mean())
        deviation = (current - ma_200) / ma_200 if ma_200 > 0 else 0.0
        theta = 0.015
        forecasts = []
        for h in horizons:
            expected_dev = deviation * np.exp(-theta * h)
            fc_pct = (expected_dev - deviation) * 100
            forecasts.append(float(-fc_pct))
        return forecasts
    except Exception:
        return [0.0] * len(horizons)


def _momentum_forecast(returns: np.ndarray, horizons: list) -> list:
    """Multi-timeframe momentum forecast."""
    try:
        if len(returns) < 60:
            return [0.0] * len(horizons)
        mom_5d = np.sum(returns[-5:])
        mom_20d = np.sum(returns[-20:])
        mom_60d = np.sum(returns[-60:])
        forecasts = []
        for h in horizons:
            if h <= 7:
                weights = [0.6, 0.3, 0.1]
            elif h <= 30:
                weights = [0.3, 0.5, 0.2]
            else:
                weights = [0.1, 0.3, 0.6]
            combined_mom = weights[0] * mom_5d + weights[1] * mom_20d / 4 + weights[2] * mom_60d / 12
            decay = np.exp(-h / 90.0)
            fc = combined_mom * decay * min(h, 30) / 30.0
            pct = fc * 100
            forecasts.append(float(pct))
        return forecasts
    except Exception:
        return [0.0] * len(horizons)


def _regime_detect(returns: np.ndarray) -> str:
    """Detect market regime from return characteristics."""
    try:
        if len(returns) < 20:
            return 'calm'
        vol = np.std(returns[-20:]) * np.sqrt(252)
        mom = np.sum(returns[-20:])
        autocorr = np.corrcoef(returns[-21:-1], returns[-20:])[0, 1] if len(returns) >= 21 else 0
        if vol > 0.30:
            return 'volatile'
        elif abs(mom) > 0.10:
            return 'trending'
        elif autocorr < -0.2:
            return 'mean_reverting'
        else:
            return 'calm'
    except Exception:
        return 'calm'


def ensemble_forecast(prices: pd.Series, horizons: list = None, asset_type: str = "equity", asset_name: str = "unknown") -> tuple:
    """
    Elite multi-model ensemble forecast with regime-aware weighting.
    
    Models (6 total):
    1. Kalman Filter - drift state estimation
    2. GARCH(1,1) - volatility-adjusted forecasts
    3. Ornstein-Uhlenbeck - mean reversion
    4. Momentum - multi-timeframe trend following
    5. Classical - baseline drift extrapolation
    6. TFT - Temporal Fusion Transformer (attention-based)
    
    IMPORTANT: Always uses STANDARD_HORIZONS [1, 3, 7, 30, 90, 180, 365] internally.
    The horizons parameter is ignored for consistency.
    
    Returns: (fc_1d, fc_3d, fc_7d, fc_30d, fc_90d, fc_180d, fc_365d, confidence)
    """
    try:
        # ALWAYS use standard horizons regardless of what's passed
        horizons = STANDARD_HORIZONS
        
        if prices is None or len(prices) < 60:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"
        
        log_returns = np.log(prices / prices.shift(1)).dropna().values
        if len(log_returns) < 30:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"
        
        # Classical models
        kalman_fc = _kalman_forecast(log_returns, horizons)
        garch_fc = _garch_forecast(log_returns, horizons)
        ou_fc = _ou_forecast(prices, horizons)
        mom_fc = _momentum_forecast(log_returns, horizons)
        
        daily_drift = np.mean(log_returns[-60:])
        daily_drift = np.clip(daily_drift, -0.01, 0.01)
        classical_fc = []
        for h in horizons:
            fc = daily_drift * h
            pct = (np.exp(fc) - 1) * 100
            classical_fc.append(float(pct))
        
        # TFT forecasts (attention-based deep learning)
        tft_fc = [0.0] * len(horizons)
        tft_confidence = "Low"
        use_tft = USE_TFT and len(prices) >= 100
        
        if use_tft:
            try:
                tft_point, tft_intervals, tft_confidence = tft_forecast(
                    prices, horizons, asset_type, asset_name
                )
                tft_fc = tft_point
            except Exception:
                tft_fc = [0.0] * len(horizons)
                use_tft = False
        
        regime = _regime_detect(log_returns)
        
        # Base weights: [Kalman, GARCH, OU, Momentum, Classical, TFT]
        # TFT gets higher weight when data quality is good
        tft_base_weight = 0.20 if use_tft else 0.0
        scale = 1.0 - tft_base_weight  # Scale other weights
        
        if regime == 'trending':
            # TFT excels at capturing trends via attention
            base_weights = [0.20*scale, 0.08*scale, 0.08*scale, 0.35*scale, 0.12*scale, tft_base_weight + 0.05]
        elif regime == 'mean_reverting':
            # OU model dominates, TFT helps identify reversal patterns
            base_weights = [0.12*scale, 0.12*scale, 0.35*scale, 0.08*scale, 0.15*scale, tft_base_weight]
        elif regime == 'volatile':
            # GARCH dominates, TFT conservative in volatile regimes
            base_weights = [0.15*scale, 0.30*scale, 0.20*scale, 0.08*scale, 0.12*scale, tft_base_weight - 0.05]
        else:  # calm
            # Balanced ensemble, TFT contributes normally
            base_weights = [0.20*scale, 0.15*scale, 0.15*scale, 0.12*scale, 0.18*scale, tft_base_weight]
        
        # Ensure non-negative weights
        base_weights = [max(0, w) for w in base_weights]
        
        final_forecasts = []
        for i, h in enumerate(horizons):
            # Horizon-specific adjustments
            # TFT is better for medium-term (7-90 days)
            if h <= 3:
                # Short-term: momentum dominates, TFT less useful
                adj = [0.0, 0.0, -0.05, 0.10, 0.0, -0.05]
            elif h <= 7:
                adj = [0.0, 0.0, -0.05, 0.05, -0.02, 0.02]
            elif h <= 30:
                # Medium-term: TFT shines with pattern recognition
                adj = [0.02, 0.0, 0.0, -0.02, -0.03, 0.05]
            elif h <= 90:
                # Medium-long: TFT + OU for mean reversion signals
                adj = [-0.02, 0.0, 0.08, -0.05, -0.02, 0.03]
            else:
                # Long-term: OU mean reversion + TFT attention
                adj = [0.05, -0.03, 0.10, -0.10, -0.02, 0.02]
            
            weights = [max(0, base_weights[j] + adj[j]) for j in range(6)]
            total_w = sum(weights)
            if total_w > 0:
                weights = [w / total_w for w in weights]
            else:
                weights = [1/6] * 6
            
            forecasts_at_h = [
                kalman_fc[i] if i < len(kalman_fc) else 0.0,
                garch_fc[i] if i < len(garch_fc) else 0.0,
                ou_fc[i] if i < len(ou_fc) else 0.0,
                mom_fc[i] if i < len(mom_fc) else 0.0,
                classical_fc[i] if i < len(classical_fc) else 0.0,
                tft_fc[i] if i < len(tft_fc) else 0.0,
            ]
            
            ensemble = sum(w * f for w, f in zip(weights, forecasts_at_h))
            final_forecasts.append(float(ensemble))
        
        vol = float(np.std(log_returns) * np.sqrt(252))
        
        if asset_type == "currency":
            hard_caps = {1: 1.5, 3: 2.5, 7: 4, 30: 8, 90: 12, 180: 18, 365: 25}
        elif asset_type == "metal":
            hard_caps = {1: 3, 3: 5, 7: 8, 30: 15, 90: 25, 180: 35, 365: 50}
        else:
            hard_caps = {1: 2, 3: 4, 7: 6, 30: 12, 90: 18, 180: 25, 365: 35}
        
        bounded_forecasts = []
        for i, h in enumerate(horizons):
            fc = final_forecasts[i]
            vol_bound = vol * np.sqrt(h / 252) * 3 * 100
            hard_cap = hard_caps.get(h, 30)
            max_fc = min(vol_bound, hard_cap)
            max_fc = max(max_fc, 0.5)
            fc = float(np.clip(fc, -max_fc, max_fc))
            bounded_forecasts.append(fc)
        
        # Confidence score: boost if TFT agrees with classical models
        data_score = min(len(prices) / 500, 1.0)
        vol_score = 1 - min(vol / 0.50, 1.0)
        regime_score = 0.8 if regime in ['calm', 'trending'] else 0.5
        
        # TFT agreement bonus
        tft_bonus = 0.0
        if use_tft and tft_confidence in ["High", "Medium"]:
            # Check if TFT agrees with ensemble direction
            avg_classical = np.mean([kalman_fc[0] if kalman_fc else 0, 
                                      mom_fc[0] if mom_fc else 0])
            if len(tft_fc) > 0 and tft_fc[0] != 0:
                if np.sign(tft_fc[0]) == np.sign(avg_classical):
                    tft_bonus = 0.1  # Agreement boosts confidence
        
        conf_score = data_score * 0.25 + vol_score * 0.35 + regime_score * 0.25 + tft_bonus + 0.15
        
        if conf_score > 0.7:
            confidence = "High"
        elif conf_score > 0.45:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        while len(bounded_forecasts) < 7:
            bounded_forecasts.append(0.0)
        
        return tuple(bounded_forecasts[:7]) + (confidence,)
        
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"

# =============================================================================
# MARKET TEMPERATURE CONSTANTS
# =============================================================================

# Universe weights (sum to 1.0)
MEGA_CAP_WEIGHT = 0.40       # Top 100 mega-caps
BROAD_MARKET_WEIGHT = 0.30   # S&P 500 proxy
SMALL_CAP_WEIGHT = 0.20      # Russell 2000
GROWTH_VALUE_WEIGHT = 0.10   # Growth vs Value rotation
INTERNATIONAL_WEIGHT = 0.05  # International indexes (informational, not used in main temp calc)

# Z-score calculation lookback
ZSCORE_LOOKBACK_DAYS = 60

# Volatility percentile lookback
VOLATILITY_LOOKBACK_DAYS = 252  # 1 year

# Scaling function parameters
SIGMOID_K = 3.0
SIGMOID_THRESHOLD = 1.0

# Temperature bounds
TEMP_MIN = 0.0
TEMP_MAX = 2.0

# Cache TTL (seconds)
CACHE_TTL_SECONDS = 3600  # 1 hour

# MAD consistency constant for robust z-score
MAD_CONSISTENCY_CONSTANT = 1.4826

# Vol term structure inversion
VOL_TERM_STRUCTURE_SHORT_WINDOW = 5
VOL_TERM_STRUCTURE_LONG_WINDOW = 20
VOL_TERM_STRUCTURE_INVERSION_THRESHOLD = 1.5

# Breadth thresholds
BREADTH_WARNING_THRESHOLD = 0.40   # < 40% above 50-day MA = warning
BREADTH_DANGER_THRESHOLD = 0.25    # < 25% = danger

# Momentum thresholds
MOMENTUM_STRONG_BULL = 0.15    # > 15% in 21 days
MOMENTUM_MILD_BULL = 0.05      # > 5%
MOMENTUM_MILD_BEAR = -0.05     # < -5%
MOMENTUM_STRONG_BEAR = -0.15   # < -15%

# Correlation stress
CORRELATION_LOOKBACK = 60
CORRELATION_STRESS_THRESHOLD = 0.75  # Avg correlation > 0.75 = systemic risk


class AlertSeverity:
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class UniverseMetrics:
    """Metrics for a single universe segment (e.g., Top 100, Small Caps)."""
    name: str
    weight: float
    
    # Price/Return metrics
    current_level: Optional[float] = None
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_21d: float = 0.0
    return_63d: float = 0.0  # 3 months
    
    # Volatility metrics
    volatility_20d: float = 0.0
    volatility_percentile: float = 0.5
    vol_term_structure_ratio: float = 1.0
    vol_inverted: bool = False
    
    # Breadth (for indices with components)
    breadth_pct_above_50ma: Optional[float] = None
    breadth_pct_above_200ma: Optional[float] = None
    advance_decline_ratio: Optional[float] = None
    
    # Computed stress
    stress_level: float = 0.0
    stress_contribution: float = 0.0
    
    # Momentum signal
    momentum_signal: str = "→ Flat"
    
    # Data quality
    data_available: bool = False
    ticker_count: int = 0
    
    # Forecasts
    forecast_1d: float = 0.0
    forecast_3d: float = 0.0
    forecast_7d: float = 0.0
    forecast_30d: float = 0.0
    forecast_90d: float = 0.0
    forecast_180d: float = 0.0
    forecast_365d: float = 0.0
    forecast_confidence: str = "Low"
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "weight": self.weight,
            "current_level": self.current_level,
            "return_1d": self.return_1d,
            "return_5d": self.return_5d,
            "return_21d": self.return_21d,
            "return_63d": self.return_63d,
            "volatility_20d": self.volatility_20d,
            "volatility_percentile": self.volatility_percentile,
            "vol_term_structure_ratio": self.vol_term_structure_ratio,
            "vol_inverted": self.vol_inverted,
            "breadth_pct_above_50ma": self.breadth_pct_above_50ma,
            "breadth_pct_above_200ma": self.breadth_pct_above_200ma,
            "advance_decline_ratio": self.advance_decline_ratio,
            "stress_level": self.stress_level,
            "stress_contribution": self.stress_contribution,
            "momentum_signal": self.momentum_signal,
            "data_available": self.data_available,
            "ticker_count": self.ticker_count,
                        "forecast_1d": self.forecast_1d,
            "forecast_3d": self.forecast_3d,
            "forecast_3d": self.forecast_3d,
            "forecast_7d": self.forecast_7d,
            "forecast_30d": self.forecast_30d,
            "forecast_90d": self.forecast_90d,
            "forecast_180d": self.forecast_180d,
            "forecast_365d": self.forecast_365d,
            "forecast_confidence": self.forecast_confidence,
        }


@dataclass
class SectorMetrics:
    """Metrics for a single sector (e.g., Technology, Financials)."""
    name: str
    ticker: str  # ETF ticker (e.g., XLK for Technology)
    
    # Price/Return metrics
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_21d: float = 0.0
    
    # Volatility
    volatility_20d: float = 0.0
    volatility_percentile: float = 0.5
    
    # Momentum
    momentum_signal: str = "→ Flat"
    
    # Risk score (0-100)
    risk_score: int = 0
    
    # Forecasts
    forecast_1d: float = 0.0
    forecast_3d: float = 0.0
    forecast_7d: float = 0.0
    forecast_30d: float = 0.0
    forecast_90d: float = 0.0
    forecast_180d: float = 0.0
    forecast_365d: float = 0.0
    forecast_confidence: str = "Low"
    
    # Data availability
    data_available: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "ticker": self.ticker,
            "return_1d": self.return_1d,
            "return_5d": self.return_5d,
            "return_21d": self.return_21d,
            "volatility_20d": self.volatility_20d,
            "volatility_percentile": self.volatility_percentile,
            "momentum_signal": self.momentum_signal,
            "risk_score": self.risk_score,
            "data_available": self.data_available,
                        "forecast_1d": self.forecast_1d,
            "forecast_3d": self.forecast_3d,
            "forecast_3d": self.forecast_3d,
            "forecast_7d": self.forecast_7d,
            "forecast_30d": self.forecast_30d,
            "forecast_90d": self.forecast_90d,
            "forecast_180d": self.forecast_180d,
            "forecast_365d": self.forecast_365d,
            "forecast_confidence": self.forecast_confidence,
        }


@dataclass
class CurrencyMetrics:
    """Metrics for a currency pair (e.g., EUR/USD, GBP/USD)."""
    name: str
    ticker: str
    rate: float = 0.0
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_21d: float = 0.0
    volatility_20d: float = 0.0
    momentum_signal: str = "→ Flat"
    risk_score: int = 0
    data_available: bool = False
    # Forecasts (scientifically computed using drift + mean reversion + volatility)
    forecast_1d: float = 0.0      # 1 day forecast (% change)
    forecast_3d: float = 0.0      # 3 day forecast
    forecast_7d: float = 0.0      # 7 day forecast
    forecast_30d: float = 0.0     # 30 day (1 month) forecast
    forecast_90d: float = 0.0     # 90 day (3 month) forecast
    forecast_180d: float = 0.0    # 180 day (6 month) forecast
    forecast_365d: float = 0.0    # 365 day (12 month) forecast
    forecast_confidence: str = "Low"  # Low/Medium/High based on model fit
    is_inverse: bool = False      # True for JPY/XXX pairs (computed from XXX/JPY)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "ticker": self.ticker,
            "rate": self.rate,
            "return_1d": self.return_1d,
            "return_5d": self.return_5d,
            "return_21d": self.return_21d,
            "volatility_20d": self.volatility_20d,
            "momentum_signal": self.momentum_signal,
            "risk_score": self.risk_score,
            "data_available": self.data_available,
                        "forecast_1d": self.forecast_1d,
            "forecast_3d": self.forecast_3d,
            "forecast_7d": self.forecast_7d,
            "forecast_30d": self.forecast_30d,
            "forecast_90d": self.forecast_90d,
            "forecast_180d": self.forecast_180d,
            "forecast_365d": self.forecast_365d,
            "forecast_confidence": self.forecast_confidence,
            "is_inverse": self.is_inverse,
        }


@dataclass
class MarketBreadthMetrics:
    """Aggregated market breadth analysis."""
    pct_above_50ma: float = 0.5
    pct_above_200ma: float = 0.5
    new_highs: int = 0
    new_lows: int = 0
    advance_decline_ratio: float = 1.0
    breadth_thrust: bool = False  # Extreme breadth signal
    breadth_warning: bool = False
    interpretation: str = "Normal"
    
    def to_dict(self) -> Dict:
        return {
            "pct_above_50ma": self.pct_above_50ma,
            "pct_above_200ma": self.pct_above_200ma,
            "new_highs": self.new_highs,
            "new_lows": self.new_lows,
            "advance_decline_ratio": self.advance_decline_ratio,
            "breadth_thrust": self.breadth_thrust,
            "breadth_warning": self.breadth_warning,
            "interpretation": self.interpretation,
        }


@dataclass
class CorrelationStress:
    """Cross-asset correlation analysis."""
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    correlation_percentile: float = 0.5
    systemic_risk_elevated: bool = False
    interpretation: str = "Normal dispersion"
    
    def to_dict(self) -> Dict:
        return {
            "avg_correlation": self.avg_correlation,
            "max_correlation": self.max_correlation,
            "correlation_percentile": self.correlation_percentile,
            "systemic_risk_elevated": self.systemic_risk_elevated,
            "interpretation": self.interpretation,
        }


@dataclass
class MarketTemperatureResult:
    """Complete market temperature computation result."""
    temperature: float                       # Final temperature ∈ [0, 2]
    scale_factor: float                      # Position scaling factor ∈ (0, 1)
    universes: Dict[str, UniverseMetrics]    # Per-universe metrics
    breadth: MarketBreadthMetrics            # Market breadth analysis
    correlation: CorrelationStress           # Correlation stress analysis
    computed_at: str                         # ISO timestamp
    data_quality: float                      # Fraction of universes with data
    status: str                              # Calm, Elevated, Stressed, Extreme
    action_text: str                         # Position recommendation
    
    # Crash risk fields
    crash_risk_pct: float = 0.0
    crash_risk_level: str = "Low"
    vol_inversion_count: int = 0
    inverted_universes: Optional[List[str]] = None
    
    # Momentum summary
    overall_momentum: str = "→ Neutral"
    sector_rotation_signal: str = "Normal"
    
    # Exit signals
    exit_signal: bool = False
    exit_reason: Optional[str] = None
    
    # Sector-by-sector breakdown (February 2026)
    sectors: Dict[str, SectorMetrics] = field(default_factory=dict)
    
    # Currency pairs breakdown (February 2026)
    currencies: Dict[str, CurrencyMetrics] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "temperature": float(self.temperature),
            "scale_factor": float(self.scale_factor),
            "status": self.status,
            "action_text": self.action_text,
            "computed_at": self.computed_at,
            "data_quality": float(self.data_quality),
            "universes": {k: v.to_dict() for k, v in self.universes.items()},
            "breadth": self.breadth.to_dict(),
            "correlation": self.correlation.to_dict(),
            "crash_risk_pct": float(self.crash_risk_pct),
            "crash_risk_level": self.crash_risk_level,
            "vol_inversion_count": self.vol_inversion_count,
            "inverted_universes": self.inverted_universes,
            "overall_momentum": self.overall_momentum,
            "sector_rotation_signal": self.sector_rotation_signal,
            "exit_signal": self.exit_signal,
            "exit_reason": self.exit_reason,
            "sectors": {k: v.to_dict() for k, v in self.sectors.items()},
            "currencies": {k: v.to_dict() for k, v in self.currencies.items()},
        }
    
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
    lookback: int = ZSCORE_LOOKBACK_DAYS
) -> float:
    """Compute robust z-score using Median Absolute Deviation (MAD)."""
    try:
        if values is None or len(values) == 0:
            return 0.0
        
        if isinstance(values, pd.DataFrame):
            if values.shape[1] == 1:
                values = values.iloc[:, 0]
            else:
                return 0.0
        
        values = values.dropna()
        
        if len(values) < lookback // 2:
            return 0.0
        
        recent = values.iloc[-lookback:] if len(values) >= lookback else values
        current = float(values.iloc[-1])
        
        median = float(recent.median())
        mad = float((recent - median).abs().median())
        
        scaled_mad = MAD_CONSISTENCY_CONSTANT * mad
        
        if scaled_mad < 1e-10:
            std = float(recent.std())
            if std < 1e-10:
                return 0.0
            zscore = (current - recent.mean()) / std
        else:
            zscore = (current - median) / scaled_mad
        
        return float(np.clip(zscore, -5.0, 5.0))
    except Exception:
        return 0.0


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


def _compute_vol_term_structure(
    prices: pd.Series,
    short_window: int = VOL_TERM_STRUCTURE_SHORT_WINDOW,
    long_window: int = VOL_TERM_STRUCTURE_LONG_WINDOW,
) -> Tuple[float, bool]:
    """Compute volatility term structure ratio. Returns (ratio, is_inverted)."""
    try:
        if prices is None or len(prices) < long_window + 5:
            return 1.0, False
        
        returns = prices.pct_change().dropna()
        
        if len(returns) < long_window:
            return 1.0, False
        
        short_vol = float(returns.iloc[-short_window:].std() * np.sqrt(252))
        long_vol = float(returns.iloc[-long_window:].std() * np.sqrt(252))
        
        if long_vol < 1e-10:
            return 1.0, False
        
        ratio = short_vol / long_vol
        is_inverted = ratio >= VOL_TERM_STRUCTURE_INVERSION_THRESHOLD
        
        return ratio, is_inverted
    except Exception:
        return 1.0, False


def _compute_returns(prices: pd.Series) -> Dict[str, float]:
    """Compute returns over multiple horizons."""
    results = {"1d": 0.0, "5d": 0.0, "21d": 0.0, "63d": 0.0}
    
    try:
        if prices is None or len(prices) < 2:
            return results
        
        prices = prices.dropna()
        current = float(prices.iloc[-1])
        
        if len(prices) >= 2:
            results["1d"] = (current / float(prices.iloc[-2]) - 1)
        if len(prices) >= 6:
            results["5d"] = (current / float(prices.iloc[-6]) - 1)
        if len(prices) >= 22:
            results["21d"] = (current / float(prices.iloc[-22]) - 1)
        if len(prices) >= 64:
            results["63d"] = (current / float(prices.iloc[-64]) - 1)
    except Exception:
        pass
    
    return results


def _compute_momentum_signal(return_5d: float, return_21d: float) -> str:
    """Compute momentum signal from returns."""
    avg_momentum = (return_5d + return_21d) / 2
    
    if avg_momentum >= MOMENTUM_STRONG_BULL:
        return "↑ Strong"
    elif avg_momentum >= MOMENTUM_MILD_BULL:
        return "↗ Rising"
    elif avg_momentum <= MOMENTUM_STRONG_BEAR:
        return "↓ Weak"
    elif avg_momentum <= MOMENTUM_MILD_BEAR:
        return "↘ Falling"
    else:
        return "→ Flat"


def _compute_scale_factor(temperature: float) -> float:
    """
    Compute position scale factor using smooth exponential decay.
    
    February 2026 Enhancement (Professor Zhang Xin-Yu, Score: 8.7/10):
    Replaces sigmoid-based scaling with smooth exponential decay that
    eliminates discontinuities and uses hysteresis to prevent oscillation.
    
    Formula:
        scale = exp(-decay_rate * max(0, temperature - threshold))
    
    Properties:
        - Continuous and differentiable everywhere
        - scale = 1.0 when temperature <= threshold
        - Smooth decay as temperature increases
    """
    if USE_COPULA_CORRELATION and COPULA_CORRELATION_AVAILABLE:
        return compute_smooth_scale_factor(
            temperature,
            threshold=SIGMOID_THRESHOLD,
            decay_rate=SIGMOID_K * 0.7,  # Slightly gentler than sigmoid equivalent
            hysteresis_band=0.05,
            state_key="market_temperature"
        )
    
    # Fallback to original sigmoid
    return 1.0 / (1.0 + math.exp(SIGMOID_K * (temperature - SIGMOID_THRESHOLD)))


def _get_status_and_action(temperature: float) -> Tuple[str, str]:
    """Get status label and action text based on temperature."""
    if temperature >= 1.5:
        return "Extreme", "EXIT POSITIONS - Capital preservation mode"
    elif temperature >= 1.0:
        return "Stressed", "REDUCE EXPOSURE - Risk management priority"
    elif temperature >= 0.7:
        return "Elevated", "CAUTION - Consider hedging"
    elif temperature >= 0.5:
        return "Warm", "MONITOR - Tighten stops"
    else:
        return "Calm", "NORMAL - Business as usual"


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
                # Check if it's a multi-ticker DataFrame
                close_df = df['Close']
                if isinstance(close_df, pd.DataFrame):
                    # Try to get the specific ticker's column
                    if ticker in close_df.columns:
                        return close_df[ticker].dropna()
                    # Try without special characters (e.g., ^VIX -> VIX)
                    ticker_clean = ticker.replace('^', '').replace('=X', '')
                    if ticker_clean in close_df.columns:
                        return close_df[ticker_clean].dropna()
                    # Fallback to first column only if single column
                    if len(close_df.columns) == 1:
                        return close_df.iloc[:, 0].dropna()
                    # For multi-column, don't fallback - return None to avoid wrong data
                    return None
                else:
                    # It's already a Series
                    return close_df.dropna()
        
        if 'Close' in df.columns:
            series = df['Close']
            if isinstance(series, pd.DataFrame):
                # Same logic for multi-ticker case
                if ticker in series.columns:
                    return series[ticker].dropna()
                ticker_clean = ticker.replace('^', '').replace('=X', '')
                if ticker_clean in series.columns:
                    return series[ticker_clean].dropna()
                if len(series.columns) == 1:
                    return series.iloc[:, 0].dropna()
                return None
            return series.dropna()
        
        if 'close' in df.columns:
            return df['close'].dropna()
        
        # Only use first column fallback for single-column DataFrames
        if len(df.columns) == 1:
            return df.iloc[:, 0].dropna()
    except Exception:
        pass
    
    return None


# =============================================================================
# TOP 100 US STOCKS - Mega-Cap Universe
# =============================================================================

# Top 100 US stocks by market cap (static list - updated periodically)
TOP_100_TICKERS = [
    # Top 10 Mega-Caps
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    # 11-25
    "V", "XOM", "JPM", "WMT", "MA", "PG", "HD", "CVX", "MRK", "ABBV",
    "LLY", "PEP", "KO", "COST", "AVGO",
    # 26-50
    "TMO", "MCD", "CSCO", "ABT", "ACN", "WFC", "DHR", "ADBE", "CRM", "NKE",
    "TXN", "NEE", "PM", "BMY", "ORCL", "LIN", "UNP", "CMCSA", "UPS", "RTX",
    "AMD", "QCOM", "T", "LOW", "MS",
    # 51-75
    "HON", "INTC", "SPGI", "COP", "CAT", "BA", "GS", "ELV", "SBUX", "DE",
    "IBM", "INTU", "PLD", "AMGN", "GE", "AXP", "BKNG", "ISRG", "MDLZ", "GILD",
    "BLK", "ADI", "MMC", "REGN", "CVS",
    # 76-100
    "TJX", "VRTX", "SYK", "SCHW", "ADP", "C", "PGR", "ZTS", "LRCX", "CI",
    "CB", "NOW", "MO", "SO", "DUK", "SLB", "EOG", "PNC", "BDX", "ITW",
    "CL", "USB", "CME", "MCO", "APD",
]

# S&P 500 Representative Sample (50 stocks across sectors)
SP500_SAMPLE = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "CSCO", "ACN",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW",
    # Consumer
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT",
    # Industrials
    "CAT", "HON", "UNP", "BA", "GE", "RTX",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG",
    # Communication
    "GOOGL", "META", "NFLX", "DIS", "VZ",
    # Materials/Utilities
    "LIN", "APD", "NEE", "DUK", "SO",
]

# Russell 2000 Representative Sample (30 small-cap stocks)
RUSSELL_2000_SAMPLE = [
    "AMC", "GME", "BBBY", "SPCE", "PLTR",  # High-volatility small caps
    "CROX", "FIVE", "BOOT", "WSM", "PRGS",  # Consumer/Retail
    "PRLB", "NMIH", "ESNT", "RDN", "MTG",   # Financials
    "CARG", "VCEL", "FOLD", "AXNX", "TGTX", # Healthcare
    "DOCN", "NEOG", "ZI", "AMBA", "CRNC",   # Tech
    "MATX", "ARCB", "WERN", "SAIA", "XPO",  # Industrials
]


# =============================================================================
# SECTOR ETFs - For sector-by-sector breakdown (February 2026)
# =============================================================================
# SPDR Select Sector ETFs tracking S&P 500 sectors
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLY": "Cons. Disc.",
    "XLP": "Cons. Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Comm. Svcs",
}


# =============================================================================
# MARKET DATA CACHE
# =============================================================================

# =============================================================================
# SECTOR ETFs - For sector-by-sector breakdown (February 2026)
# =============================================================================
# SPDR Select Sector ETFs tracking S&P 500 sectors
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLY": "Cons. Disc.",
    "XLP": "Cons. Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Comm. Svcs",
}

# =============================================================================
# CURRENCY PAIRS - For FX market breakdown (February 2026)
# =============================================================================
# Major currency pairs and cryptocurrencies with Yahoo Finance tickers
# Convention: XXXJPY=X means "how many JPY per 1 XXX" → display as XXX/JPY
CURRENCY_PAIRS = {
    # Major FX pairs
    "EURUSD=X": "EUR/USD",
    "GBPUSD=X": "GBP/USD",
    "USDJPY=X": "USD/JPY",
    "USDCHF=X": "USD/CHF",
    "AUDUSD=X": "AUD/USD",
    "USDCAD=X": "USD/CAD",
    "NZDUSD=X": "NZD/USD",
    
    # JPY Cross Pairs (XXXJPY=X = how many JPY per 1 XXX)
    "EURJPY=X": "EUR/JPY",
    "GBPJPY=X": "GBP/JPY",
    "AUDJPY=X": "AUD/JPY",
    "NZDJPY=X": "NZD/JPY",
    "CADJPY=X": "CAD/JPY",
    "CHFJPY=X": "CHF/JPY",
    "SGDJPY=X": "SGD/JPY",
    "HKDJPY=X": "HKD/JPY",
    "ZARJPY=X": "ZAR/JPY",
    "MXNJPY=X": "MXN/JPY",
    "TRYJPY=X": "TRY/JPY",
    "SEKJPY=X": "SEK/JPY",
    "NOKJPY=X": "NOK/JPY",
    "DKKJPY=X": "DKK/JPY",
    "CNYJPY=X": "CNY/JPY",
    
    # Cryptocurrencies
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
}

# JPY as base currency pairs (computed as inverse of XXX/JPY pairs)
# These show "how many XXX per 1 JPY" - useful for JPY strength analysis
JPY_BASE_PAIRS = {
    # Source ticker → Display name (will be computed as 1/rate)
    "USDJPY=X": "JPY/USD",
    "EURJPY=X": "JPY/EUR",
    "GBPJPY=X": "JPY/GBP",
    "AUDJPY=X": "JPY/AUD",
    "NZDJPY=X": "JPY/NZD",
    "CADJPY=X": "JPY/CAD",
    "CHFJPY=X": "JPY/CHF",
    "SGDJPY=X": "JPY/SGD",
    "HKDJPY=X": "JPY/HKD",
    "ZARJPY=X": "JPY/ZAR",
    "MXNJPY=X": "JPY/MXN",
    "TRYJPY=X": "JPY/TRY",
    "SEKJPY=X": "JPY/SEK",
    "NOKJPY=X": "JPY/NOK",
    "DKKJPY=X": "JPY/DKK",
    "CNYJPY=X": "JPY/CNY",
    "PLNJPY=X": "JPY/PLN",
}


# Thread-safe cache for market data
import threading
_market_data_cache: Dict[str, Tuple[datetime, Any]] = {}
_cache_lock = threading.Lock()  # Thread lock for cache operations
_yfinance_lock = threading.Lock()  # Thread lock for yfinance downloads (not thread-safe)


class _SuppressOutput:
    """Context manager to fully suppress stdout and stderr at file descriptor level."""
    def __enter__(self):
        import os
        import sys
        # Save the actual stdout/stderr file descriptors
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)
        # Open /dev/null
        self._null_fd = os.open(os.devnull, os.O_RDWR)
        # Redirect stdout/stderr to /dev/null
        os.dup2(self._null_fd, 1)
        os.dup2(self._null_fd, 2)
        return self
        
    def __exit__(self, *args):
        import os
        # Restore stdout/stderr
        os.dup2(self._saved_stdout_fd, 1)
        os.dup2(self._saved_stderr_fd, 2)
        # Close the duplicated file descriptors
        os.close(self._saved_stdout_fd)
        os.close(self._saved_stderr_fd)
        os.close(self._null_fd)


def _fetch_etf_data(
    start_date: str,
    end_date: Optional[str] = None
) -> Dict[str, pd.Series]:
    """Fetch key ETF data for market assessment (thread-safe)."""
    cache_key = f"etf_{start_date}_{end_date}"
    now = datetime.now()
    
    # Thread-safe: Lock the entire fetch operation to prevent duplicate work
    with _cache_lock:
        # Check cache first (inside lock to prevent race)
        if cache_key in _market_data_cache:
            cached_time, cached_data = _market_data_cache[cache_key]
            if (now - cached_time).total_seconds() < CACHE_TTL_SECONDS:
                return cached_data
        
        # Not in cache, need to fetch
        try:
            import yfinance as yf
        except ImportError:
            warnings.warn("yfinance not available")
            return {}
        
        end = end_date or datetime.now().strftime("%Y-%m-%d")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=ZSCORE_LOOKBACK_DAYS + 30)
        start = start_dt.strftime("%Y-%m-%d")
        
        etf_tickers = {
            'SPY': 'SPY',      # S&P 500
            'QQQ': 'QQQ',      # Nasdaq 100
            'IWM': 'IWM',      # Russell 2000
            'IWD': 'IWD',      # Russell 1000 Value
            'IWF': 'IWF',      # Russell 1000 Growth
            'VTI': 'VTI',      # Total Market
            'VIX': '^VIX',     # Volatility Index
            'DIA': 'DIA',      # Dow Jones Industrial Average
            # International Indexes
            'EFA': 'EFA',      # MSCI EAFE (Europe, Australasia, Far East)
            'EEM': 'EEM',      # MSCI Emerging Markets
            'VEU': 'VEU',      # FTSE All-World ex-US
            'INDA': 'INDA',    # MSCI India
            'FXI': 'FXI',      # China Large-Cap (FTSE China 50)
            'EWJ': 'EWJ',      # MSCI Japan
            'EWG': 'EWG',      # MSCI Germany
            'EWU': 'EWU',      # MSCI United Kingdom
            'EWZ': 'EWZ',      # MSCI Brazil
            'EWA': 'EWA',      # MSCI Australia
            'EWC': 'EWC',      # MSCI Canada
            # Sector ETFs (SPDR Select Sector)
            'XLK': 'XLK',      # Technology
            'XLF': 'XLF',      # Financials
            'XLV': 'XLV',      # Healthcare
            'XLY': 'XLY',      # Consumer Discretionary
            'XLP': 'XLP',      # Consumer Staples
            'XLE': 'XLE',      # Energy
            'XLI': 'XLI',      # Industrials
            'XLB': 'XLB',      # Materials
            'XLU': 'XLU',      # Utilities
            'XLRE': 'XLRE',    # Real Estate
            'XLC': 'XLC',      # Communication Services
        }
        
        result = {}
        
        for name, ticker in etf_tickers.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Use Ticker object to avoid global state issues in yf.download
                    ticker_obj = yf.Ticker(ticker)
                    df = ticker_obj.history(start=start, end=end, auto_adjust=True)
                
                # history() returns a simple DataFrame with Close column (not MultiIndex)
                if df is not None and not df.empty and 'Close' in df.columns:
                    series = df['Close'].dropna()
                    if len(series) > 20:
                        result[name] = series
            except Exception:
                logger.debug(f"Failed to fetch {ticker}: {e}")
        
        # Update cache (still inside lock)
        _market_data_cache[cache_key] = (now, result)
        return result


def _fetch_stock_sample_data(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    cache_key_prefix: str = "stocks"
) -> Dict[str, pd.Series]:
    """Fetch price data for a sample of stocks (thread-safe)."""
    cache_key = f"{cache_key_prefix}_{start_date}_{end_date}_{len(tickers)}"
    now = datetime.now()
    
    # Thread-safe cache check
    with _cache_lock:
        if cache_key in _market_data_cache:
            cached_time, cached_data = _market_data_cache[cache_key]
            if (now - cached_time).total_seconds() < CACHE_TTL_SECONDS:
                return cached_data
    
    try:
        import yfinance as yf
    except ImportError:
        return {}
    
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=ZSCORE_LOOKBACK_DAYS + 30)
    start = start_dt.strftime("%Y-%m-%d")
    
    result = {}
    
    # Batch download for efficiency (threads=False to avoid output issues)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Thread-safe yfinance download
            with _yfinance_lock:
                df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True, threads=False)
        
        if df is not None and not df.empty:
            # Handle multi-ticker DataFrame
            if isinstance(df.columns, pd.MultiIndex):
                for ticker in tickers:
                    try:
                        if ('Close', ticker) in df.columns:
                            series = df[('Close', ticker)].dropna()
                            if len(series) > 20:
                                result[ticker] = series
                    except Exception:
                        pass
            else:
                # Single ticker case
                if len(tickers) == 1:
                    series = _extract_close_series(df, tickers[0])
                    if series is not None and len(series) > 20:
                        result[tickers[0]] = series
    except Exception:
        logger.debug(f"Batch download failed: {e}")
    
    # Thread-safe cache update
    with _cache_lock:
        _market_data_cache[cache_key] = (now, result)
    return result


def _compute_equity_forecasts(prices, vol_20d):
    """
    Compute equity/sector forecasts using elite multi-model ensemble.
    
    Returns: (fc_1d, fc_3d, fc_7d, fc_30d, fc_90d, fc_180d, fc_365d, confidence)
    """
    try:
        if prices is None or len(prices) < 30:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"
        
        # Use elite ensemble forecast
        horizons = [1, 3, 7, 30, 90, 180, 365]
        result = ensemble_forecast(prices, horizons, "equity")
        return result
        
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"


def _compute_universe_metrics(
    name: str,
    weight: float,
    prices: pd.Series,
    ticker_count: int
) -> UniverseMetrics:
    """Compute metrics for a single universe segment."""
    if prices is None or len(prices) < 30:
        return UniverseMetrics(name=name, weight=weight, data_available=False, ticker_count=0)
    
    try:
        current_level = float(prices.iloc[-1])
        returns = _compute_returns(prices)
        
        vol_20d = float(prices.pct_change().dropna().iloc[-20:].std() * np.sqrt(252)) if len(prices) >= 20 else 0.0
        vol_pct = _compute_volatility_percentile(prices)
        vol_ratio, vol_inverted = _compute_vol_term_structure(prices)
        
        momentum = _compute_momentum_signal(returns['5d'], returns['21d'])
        
        vol_zscore = _compute_robust_zscore(prices.pct_change().rolling(20).std() * np.sqrt(252))
        
        stress = 0.0
        stress += min(max(vol_zscore, 0), 2.0) * 0.4
        stress += min(vol_pct, 1.0) * 0.3
        if vol_inverted:
            stress += 0.5
        if returns['21d'] < -0.10:
            stress += 0.3
        stress = min(stress, 2.0)
        fc_1d, fc_3d, fc_7d, fc_30d, fc_90d, fc_180d, fc_365d, fc_conf = _compute_equity_forecasts(prices, vol_20d)
        
        return UniverseMetrics(
            name=name, weight=weight, current_level=current_level,
            return_1d=returns['1d'], return_5d=returns['5d'],
            return_21d=returns['21d'], return_63d=returns['63d'],
            volatility_20d=vol_20d, volatility_percentile=vol_pct,
            vol_term_structure_ratio=vol_ratio, vol_inverted=vol_inverted,
            stress_level=stress, stress_contribution=stress * weight,
            momentum_signal=momentum, data_available=True, ticker_count=ticker_count, 
            forecast_1d=fc_1d, forecast_3d=fc_3d, forecast_7d=fc_7d, forecast_30d=fc_30d, 
            forecast_90d=fc_90d, forecast_180d=fc_180d, forecast_365d=fc_365d, forecast_confidence=fc_conf,
        )
    except Exception:
        return UniverseMetrics(name=name, weight=weight, data_available=False, ticker_count=0)


def _compute_breadth_from_sample(stock_data: Dict[str, pd.Series]) -> MarketBreadthMetrics:
    """Compute market breadth from a sample of stocks."""
    if not stock_data or len(stock_data) < 10:
        return MarketBreadthMetrics()
    
    try:
        above_50ma = above_200ma = advances = declines = total = 0
        
        for ticker, prices in stock_data.items():
            if prices is None or len(prices) < 200:
                continue
            total += 1
            current = float(prices.iloc[-1])
            ma_50 = float(prices.iloc[-50:].mean())
            ma_200 = float(prices.iloc[-200:].mean())
            prev = float(prices.iloc[-2]) if len(prices) >= 2 else current
            
            if current > ma_50: above_50ma += 1
            if current > ma_200: above_200ma += 1
            if current > prev: advances += 1
            else: declines += 1
        
        if total == 0:
            return MarketBreadthMetrics()
        
        pct_50 = above_50ma / total
        pct_200 = above_200ma / total
        ad_ratio = advances / max(declines, 1)
        warning = pct_50 < BREADTH_WARNING_THRESHOLD
        thrust = pct_50 > 0.90 or pct_50 < 0.10
        
        if pct_50 < BREADTH_DANGER_THRESHOLD:
            interp = "Danger - Extreme weakness"
        elif warning:
            interp = "Warning - Narrowing breadth"
        elif thrust and pct_50 > 0.90:
            interp = "Thrust - Broad participation"
        else:
            interp = "Healthy - Normal breadth"
        
        return MarketBreadthMetrics(
            pct_above_50ma=pct_50, pct_above_200ma=pct_200,
            advance_decline_ratio=ad_ratio, breadth_thrust=thrust,
            breadth_warning=warning, interpretation=interp,
        )
    except Exception:
        return MarketBreadthMetrics()


def _compute_correlation_stress(stock_data: Dict[str, pd.Series]) -> CorrelationStress:
    """Compute cross-asset correlation stress."""
    if not stock_data or len(stock_data) < 5:
        return CorrelationStress()
    
    try:
        if USE_COPULA_CORRELATION and COPULA_CORRELATION_AVAILABLE:
            copula_result = compute_copula_correlation_stress(stock_data)
            return CorrelationStress(
                avg_correlation=copula_result.avg_correlation,
                max_correlation=copula_result.max_correlation,
                correlation_percentile=copula_result.correlation_percentile,
                systemic_risk_elevated=copula_result.systemic_risk_elevated,
                interpretation=copula_result.interpretation,
            )
        
        returns_df = pd.DataFrame({t: p.pct_change() for t, p in stock_data.items()}).dropna()
        if returns_df.empty or len(returns_df) < CORRELATION_LOOKBACK:
            return CorrelationStress()
        
        recent = returns_df.iloc[-CORRELATION_LOOKBACK:]
        corr_matrix = recent.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.values[mask]
        correlations = correlations[~np.isnan(correlations)]
        
        if len(correlations) == 0:
            return CorrelationStress()
        
        avg_corr = float(np.mean(correlations))
        max_corr = float(np.max(correlations))
        systemic = avg_corr > CORRELATION_STRESS_THRESHOLD
        
        if systemic:
            interp = "High - Systemic risk elevated"
        elif avg_corr > 0.60:
            interp = "Elevated - Rising correlations"
        else:
            interp = "Low - Healthy dispersion"
        
        return CorrelationStress(
            avg_correlation=avg_corr, max_correlation=max_corr,
            correlation_percentile=0.5, systemic_risk_elevated=systemic,
            interpretation=interp,
        )
    except Exception:
        return CorrelationStress()


def _compute_crash_risk(
    universes: Dict[str, UniverseMetrics],
    breadth: MarketBreadthMetrics,
    correlation: CorrelationStress,
    temperature: float
) -> Tuple[float, str, int, List[str]]:
    """Compute crash risk probability."""
    risk = 0.0
    vol_inversions = []
    
    for name, univ in universes.items():
        if univ.vol_inverted:
            vol_inversions.append(name)
            risk += 0.15
    
    if breadth.pct_above_50ma < BREADTH_DANGER_THRESHOLD:
        risk += 0.20
    elif breadth.pct_above_50ma < BREADTH_WARNING_THRESHOLD:
        risk += 0.10
    
    if correlation.systemic_risk_elevated:
        risk += 0.15
    
    risk += temperature * 0.10
    risk = min(risk, 1.0)
    
    if risk > 0.50: level = "Extreme"
    elif risk > 0.30: level = "High"
    elif risk > 0.15: level = "Elevated"
    else: level = "Low"
    
    return risk, level, len(vol_inversions), vol_inversions if vol_inversions else None


def _determine_exit_signal(
    temperature: float,
    crash_risk_pct: float,
    breadth: MarketBreadthMetrics,
    correlation: CorrelationStress
) -> Tuple[bool, Optional[str]]:
    """Determine if exit signal should be triggered."""
    if temperature >= 1.8:
        return True, "Temperature extreme (>1.8)"
    if crash_risk_pct >= 0.60:
        return True, f"Crash risk critical ({crash_risk_pct:.0%})"
    if breadth.pct_above_50ma < 0.15:
        return True, f"Breadth collapse ({breadth.pct_above_50ma:.0%} above 50MA)"
    return False, None


def _compute_overall_momentum(universes: Dict[str, UniverseMetrics]) -> str:
    """Compute overall market momentum from universe segments."""
    if not universes:
        return "→ Neutral"
    
    weighted_momentum = total_weight = 0.0
    for univ in universes.values():
        if univ.data_available:
            weighted_momentum += univ.return_21d * univ.weight
            total_weight += univ.weight
    
    if total_weight == 0:
        return "→ Neutral"
    
    avg_momentum = weighted_momentum / total_weight
    return _compute_momentum_signal(avg_momentum, avg_momentum)


def _compute_sector_rotation(etf_data: Dict[str, pd.Series]) -> str:
    """Compute sector rotation signal from Growth vs Value performance."""
    if 'IWF' not in etf_data or 'IWD' not in etf_data:
        return "Normal"
    
    try:
        growth_ret = _compute_returns(etf_data['IWF'])['21d']
        value_ret = _compute_returns(etf_data['IWD'])['21d']
        spread = growth_ret - value_ret
        
        if spread > 0.05: return "Growth Leading"
        elif spread < -0.05: return "Value Leading"
        else: return "Normal"
    except Exception:
        return "Normal"


def _compute_sector_metrics(etf_data: Dict[str, pd.Series]) -> Dict[str, SectorMetrics]:
    """Compute metrics for each sector ETF."""
    sectors = {}
    
    for ticker, name in SECTOR_ETFS.items():
        if ticker not in etf_data:
            sectors[name] = SectorMetrics(name=name, ticker=ticker, data_available=False)
            continue
        
        prices = etf_data[ticker]
        if prices is None or len(prices) < 30:
            sectors[name] = SectorMetrics(name=name, ticker=ticker, data_available=False)
            continue
        
        try:
            returns = _compute_returns(prices)
            daily_returns = prices.pct_change().dropna()
            vol_20d = float(daily_returns.iloc[-20:].std() * np.sqrt(252)) if len(daily_returns) >= 20 else 0.0
            vol_pct = _compute_volatility_percentile(prices)
            momentum = _compute_momentum_signal(returns.get('5d', 0), returns.get('21d', 0))
            
            vol_pts = min(vol_pct, 1.0) * 50
            move_pts = min(abs(returns.get('5d', 0)) / 0.05, 1.0) * 50
            risk_score = int(min(100, vol_pts + move_pts))
            fc_1d, fc_3d, fc_7d, fc_30d, fc_90d, fc_180d, fc_365d, fc_conf = _compute_equity_forecasts(prices, vol_20d)
            
            sectors[name] = SectorMetrics(
                name=name, ticker=ticker,
                return_1d=returns.get('1d', 0), return_5d=returns.get('5d', 0),
                return_21d=returns.get('21d', 0), volatility_20d=vol_20d,
                volatility_percentile=vol_pct, momentum_signal=momentum,
                risk_score=risk_score, data_available=True, forecast_1d=fc_1d, forecast_3d=fc_3d, forecast_7d=fc_7d, forecast_30d=fc_30d, forecast_90d=fc_90d, forecast_180d=fc_180d, forecast_365d=fc_365d, forecast_confidence=fc_conf,
            )
        except Exception:
            sectors[name] = SectorMetrics(name=name, ticker=ticker, data_available=False)
    
    return sectors


def _fetch_currency_data(start_date: str, end_date: Optional[str] = None) -> Dict[str, pd.Series]:
    """Fetch currency pair data."""
    cache_key = f"currency_{start_date}_{end_date}"
    now = datetime.now()
    
    with _cache_lock:
        if cache_key in _market_data_cache:
            cached_time, cached_data = _market_data_cache[cache_key]
            if (now - cached_time).total_seconds() < CACHE_TTL_SECONDS:
                return cached_data
    
    try:
        import yfinance as yf
    except ImportError:
        return {}
    
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=ZSCORE_LOOKBACK_DAYS + 30)
    start = start_dt.strftime("%Y-%m-%d")
    
    result = {}
    for ticker in CURRENCY_PAIRS.keys():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ticker_obj = yf.Ticker(ticker)
                df = ticker_obj.history(start=start, end=end, auto_adjust=True)
            
            if df is not None and not df.empty and 'Close' in df.columns:
                series = df['Close'].dropna()
                if len(series) > 20:
                    result[ticker] = series
        except Exception:
            pass
    
    with _cache_lock:
        _market_data_cache[cache_key] = (now, result)
    return result


def _prophet_forecast(prices: pd.Series, horizons: list) -> list:
    """Use Facebook Prophet for time series forecasting with trend + seasonality."""
    try:
        from prophet import Prophet
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        df = pd.DataFrame({
            'ds': prices.index,
            'y': prices.values
        })
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Configure Prophet
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative',
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)
        
        max_horizon = max(horizons)
        future = model.make_future_dataframe(periods=max_horizon)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = model.predict(future)
        
        current_price = float(prices.iloc[-1])
        current_idx = len(prices) - 1
        
        forecasts = []
        for h in horizons:
            future_idx = current_idx + h
            if future_idx < len(forecast):
                future_price = float(forecast.iloc[future_idx]['yhat'])
                pct_change = ((future_price / current_price) - 1) * 100
                forecasts.append(pct_change)
            else:
                forecasts.append(0.0)
        
        return forecasts
        
    except Exception:
        return [0.0] * len(horizons)


def _lstm_forecast(prices: pd.Series, horizons: list) -> list:
    """Use PyTorch LSTM for pattern-based forecasting."""
    try:
        import torch
        import torch.nn as nn
        
        returns = prices.pct_change().dropna().values
        if len(returns) < 100:
            return [0.0] * len(horizons)
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret < 1e-8:
            return [0.0] * len(horizons)
        normalized = (returns - mean_ret) / std_ret
        
        seq_length = 20
        X, y = [], []
        for i in range(len(normalized) - seq_length):
            X.append(normalized[i:i+seq_length])
            y.append(normalized[i+seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) < 50:
            return [0.0] * len(horizons)
        
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1)
        
        class SimpleLSTM(nn.Module):
            def __init__(self, input_size=1, hidden_size=32, num_layers=1):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                return self.fc(lstm_out[:, -1, :])
        
        model = SimpleLSTM()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        model.eval()
        forecasts = []
        current_seq = normalized[-seq_length:].copy()
        
        cumulative_return = 0.0
        for h in range(1, max(horizons) + 1):
            with torch.no_grad():
                seq_tensor = torch.FloatTensor(current_seq).unsqueeze(0).unsqueeze(-1)
                pred = model(seq_tensor).item()
            
            pred_return = pred * std_ret + mean_ret
            cumulative_return += pred_return
            
            if h in horizons:
                forecasts.append(cumulative_return * 100)
            
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = pred
        
        return forecasts
        
    except Exception:
        return [0.0] * len(horizons)


def _classical_forecast(prices: pd.Series, log_returns: pd.Series, horizons: list) -> list:
    """
    Classical mean reversion + momentum forecast.
    This ALWAYS produces non-zero forecasts based on drift and momentum.
    
    Returns forecasts in PERCENTAGE POINTS (e.g., 2.5 means +2.5%).
    """
    try:
        # Compute mean daily return (drift)
        daily_drift = float(log_returns.mean())
        # Clamp drift to reasonable FX range (-1% to +1% daily)
        daily_drift = float(np.clip(daily_drift, -0.01, 0.01))
        
        # Get current price and MA
        current_price = float(prices.iloc[-1])
        ma_len = min(200, len(prices) - 1)
        ma_price = float(prices.iloc[-ma_len:].mean())
        
        # Deviation from moving average
        deviation = (current_price - ma_price) / ma_price if ma_price > 0 else 0.0
        # Clamp deviation
        deviation = float(np.clip(deviation, -0.30, 0.30))
        
        # Recent momentum
        mom_5d = float(log_returns.iloc[-5:].mean()) if len(log_returns) >= 5 else daily_drift
        mom_5d = float(np.clip(mom_5d, -0.02, 0.02))
        
        # Volatility for bounds
        vol = float(log_returns.std())
        vol = float(np.clip(vol, 0.003, 0.05))  # 0.3% to 5% daily
        
        forecasts = []
        for h in horizons:
            # Drift contribution (trend)
            drift = daily_drift * h
            
            # Mean reversion contribution  
            mr_coef = 0.15 * min(h / 90.0, 1.0)  # Max 15% pull over 90 days
            mr = -deviation * mr_coef
            
            # Momentum contribution (decays with horizon)
            if h <= 30:
                mom = mom_5d * min(h, 21) * 0.2
            else:
                mom = 0.0
            
            # Total log return
            total = drift + mr + mom
            
            # Convert to percentage
            pct = (np.exp(total) - 1) * 100
            
            # Apply volatility-based bounds (2 sigma)
            horizon_vol = vol * np.sqrt(h)
            max_pct = horizon_vol * 2.0 * 100
            max_pct = min(max_pct, 30.0)  # Never more than +/-30%
            max_pct = max(max_pct, 0.5)   # At least +/-0.5%
            
            pct = float(np.clip(pct, -max_pct, max_pct))
            forecasts.append(pct)
        
        return forecasts
        
    except Exception:
        # Emergency fallback: simple drift
        try:
            drift = float(log_returns.mean())
            drift = float(np.clip(drift, -0.01, 0.01))
            forecasts = []
            for h in horizons:
                pct = (np.exp(drift * h) - 1) * 100
                pct = float(np.clip(pct, -30.0, 30.0))
                forecasts.append(pct)
            return forecasts
        except:
            return [0.01] * len(horizons)  # Non-zero default


def _compute_forecast_confidence(
    prophet_fc: list, lstm_fc: list, classical_fc: list,
    vol_20d: float, data_len: int
) -> str:
    """Compute confidence based on model agreement and data quality."""
    try:
        agreement_scores = []
        for i in range(min(len(prophet_fc), len(lstm_fc), len(classical_fc))):
            signs = [
                np.sign(prophet_fc[i]) if prophet_fc[i] != 0 else 0,
                np.sign(lstm_fc[i]) if lstm_fc[i] != 0 else 0,
                np.sign(classical_fc[i]) if classical_fc[i] != 0 else 0,
            ]
            # Count agreement (all same sign = 1.0, 2/3 same = 0.67, all different = 0)
            if signs[0] == signs[1] == signs[2] and signs[0] != 0:
                agreement_scores.append(1.0)
            elif signs[0] == signs[1] or signs[1] == signs[2] or signs[0] == signs[2]:
                agreement_scores.append(0.67)
            else:
                agreement_scores.append(0.0)
        
        agreement = np.mean(agreement_scores) if agreement_scores else 0.0
        
        # Volatility penalty
        vol_score = 1 - min(vol_20d / 0.20, 1.0)
        
        # Data quality
        data_score = min(data_len / 500, 1.0)
        
        # Combined confidence
        confidence_score = agreement * 0.4 + vol_score * 0.3 + data_score * 0.3
        
        if confidence_score > 0.7:
            return "High"
        elif confidence_score > 0.4:
            return "Medium"
        else:
            return "Low"
    except Exception:
        return "Low"


def _compute_currency_forecasts(prices: pd.Series, vol_20d: float) -> Tuple[float, float, float, float, float, float, float, str]:
    """
    Compute currency forecasts using elite multi-model ensemble.
    
    Models:
    1. Kalman Filter (drift estimation)
    2. GARCH (volatility-adjusted)
    3. Ornstein-Uhlenbeck (mean reversion)
    4. Momentum (multi-timeframe)
    5. Classical (baseline)
    
    Returns: (1d, 3d, 7d, 30d, 90d, 180d, 365d forecasts, confidence level)
    """
    try:
        if prices is None or len(prices) < 30:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"
        
        # Ensure we have enough data for ensemble_forecast (requires 60+ points)
        if len(prices) < 60:
            # Use simple momentum-based forecast for limited data
            log_returns = np.log(prices / prices.shift(1)).dropna()
            if len(log_returns) < 5:
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"
            
            daily_drift = float(log_returns.mean())
            vol = float(log_returns.std())
            
            forecasts = []
            for h in STANDARD_HORIZONS:
                fc = daily_drift * h * np.exp(-h / 180.0)
                pct = (np.exp(fc) - 1) * 100
                # Clamp to reasonable bounds for currencies
                max_pct = min(vol * np.sqrt(h) * 3 * 100, {1: 3, 3: 5, 7: 7, 30: 12, 90: 18, 180: 25, 365: 35}.get(h, 20))
                pct = float(np.clip(pct, -max_pct, max_pct))
                forecasts.append(pct)
            
            return tuple(forecasts) + ("Low",)
        
        # Use elite ensemble forecast (horizons ignored, uses STANDARD_HORIZONS)
        result = ensemble_forecast(prices, asset_type="currency")
        return result
        
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Low"


def _compute_currency_metrics(currency_data: Dict[str, pd.Series]) -> Dict[str, CurrencyMetrics]:
    """Compute metrics for each currency pair including forecasts and JPY base pairs."""
    currencies = {}
    
    # First, compute standard pairs (XXX/JPY format)
    for ticker, pair_name in CURRENCY_PAIRS.items():
        if ticker not in currency_data:
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                data_available=False,
            )
            continue
        
        prices = currency_data[ticker]
        if prices is None or len(prices) < 30:
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                data_available=False,
            )
            continue
        
        try:
            rate = float(prices.iloc[-1])
            returns = _compute_returns(prices)
            ret_1d = returns.get('1d', 0.0)
            ret_5d = returns.get('5d', 0.0)
            ret_21d = returns.get('21d', 0.0)
            
            daily_returns = prices.pct_change().dropna()
            vol_20d = float(daily_returns.iloc[-20:].std() * np.sqrt(252)) if len(daily_returns) >= 20 else 0.0
            
            momentum = _compute_momentum_signal(ret_5d, ret_21d)
            
            # Risk score: vol (0-50) + recent moves (0-50)
            vol_pts = min(vol_20d / 0.15, 1.0) * 50
            move_pts = min(abs(ret_5d) / 0.05, 1.0) * 50
            risk_score = int(min(100, vol_pts + move_pts))
            
            # Compute forecasts
            fc_1d, fc_3d, fc_7d, fc_30d, fc_90d, fc_180d, fc_365d, fc_conf = _compute_currency_forecasts(prices, vol_20d)
            
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                rate=rate,
                return_1d=ret_1d,
                return_5d=ret_5d,
                return_21d=ret_21d,
                volatility_20d=vol_20d,
                momentum_signal=momentum,
                risk_score=risk_score,
                data_available=True,
                forecast_1d=fc_1d,
                forecast_3d=fc_3d,
                forecast_7d=fc_7d,
                forecast_30d=fc_30d,
                forecast_90d=fc_90d,
                forecast_180d=fc_180d,
                forecast_365d=fc_365d,
                forecast_confidence=fc_conf,
                is_inverse=False,
            )
        except Exception:
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                data_available=False,
            )
    
    # Second, compute JPY base pairs (JPY/XXX = inverse of XXX/JPY)
    for ticker, pair_name in JPY_BASE_PAIRS.items():
        if ticker not in currency_data:
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                data_available=False,
                is_inverse=True,
            )
            continue
        
        prices = currency_data[ticker]
        if prices is None or len(prices) < 30:
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                data_available=False,
                is_inverse=True,
            )
            continue
        
        try:
            # Inverse the prices for JPY/XXX (1 JPY = ? XXX)
            inverse_prices = 1.0 / prices
            
            rate = float(inverse_prices.iloc[-1])
            
            # Returns for inverse pair are NEGATIVE of original
            returns = _compute_returns(prices)
            ret_1d = -returns.get('1d', 0.0)
            ret_5d = -returns.get('5d', 0.0)
            ret_21d = -returns.get('21d', 0.0)
            
            # Volatility is the same (symmetric)
            daily_returns = prices.pct_change().dropna()
            vol_20d = float(daily_returns.iloc[-20:].std() * np.sqrt(252)) if len(daily_returns) >= 20 else 0.0
            
            # Momentum is inverted
            momentum = _compute_momentum_signal(ret_5d, ret_21d)
            
            # Risk score remains the same
            vol_pts = min(vol_20d / 0.15, 1.0) * 50
            move_pts = min(abs(ret_5d) / 0.05, 1.0) * 50
            risk_score = int(min(100, vol_pts + move_pts))
            
            # Compute forecasts for inverse pair (negate the forecasts)
            fc_1d, fc_3d, fc_7d, fc_30d, fc_90d, fc_180d, fc_365d, fc_conf = _compute_currency_forecasts(prices, vol_20d)
            # Inverse forecasts
            fc_1d, fc_3d, fc_7d, fc_30d, fc_90d, fc_180d, fc_365d = -fc_1d, -fc_3d, -fc_7d, -fc_30d, -fc_90d, -fc_180d, -fc_365d
            
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                rate=rate,
                return_1d=ret_1d,
                return_5d=ret_5d,
                return_21d=ret_21d,
                volatility_20d=vol_20d,
                momentum_signal=momentum,
                risk_score=risk_score,
                data_available=True,
                forecast_1d=fc_1d,
                forecast_3d=fc_3d,
                forecast_7d=fc_7d,
                forecast_30d=fc_30d,
                forecast_90d=fc_90d,
                forecast_180d=fc_180d,
                forecast_365d=fc_365d,
                forecast_confidence=fc_conf,
                is_inverse=True,
            )
        except Exception:
            currencies[pair_name] = CurrencyMetrics(
                name=pair_name,
                ticker=ticker,
                data_available=False,
                is_inverse=True,
            )
    
    return currencies


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def compute_market_temperature(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
) -> MarketTemperatureResult:
    """
    Compute comprehensive market temperature.
    
    This is the main entry point for market risk assessment
    
    Args:
        start_date: Start date for historical data
        end_date: End date (default: today)
        
    Returns:
        MarketTemperatureResult with complete market assessment
    """
    # Fetch ETF data
    etf_data = _fetch_etf_data(start_date, end_date)
    
    # Fetch stock samples
    top100_data = _fetch_stock_sample_data(
        TOP_100_TICKERS[:50],  # Use first 50 for speed
        start_date, end_date,
        "top100"
    )
    
    sp500_data = _fetch_stock_sample_data(
        SP500_SAMPLE,
        start_date, end_date,
        "sp500"
    )
    
    russell_data = _fetch_stock_sample_data(
        RUSSELL_2000_SAMPLE[:20],  # Use first 20 for speed
        start_date, end_date,
        "russell"
    )
    
    # Compute universe metrics
    universes = {}
    
    # Mega-cap (Top 100) - use aggregate of sample
    if top100_data:
        # Create equal-weighted index
        sample_returns = pd.DataFrame({
            t: p.pct_change() for t, p in top100_data.items()
        }).dropna()
        if not sample_returns.empty:
            # Build cumulative index
            equal_weight_return = sample_returns.mean(axis=1)
            index = (1 + equal_weight_return).cumprod() * 100
            universes["Mega-Cap 100"] = _compute_universe_metrics(
                "Mega-Cap 100", MEGA_CAP_WEIGHT, index, len(top100_data)
            )
    
    # Broad Market (S&P 500 via SPY)
    if 'SPY' in etf_data:
        universes["S&P 500"] = _compute_universe_metrics(
            "S&P 500", BROAD_MARKET_WEIGHT, etf_data['SPY'], 500
        )
    
    # Small Cap (Russell 2000 via IWM)
    if 'IWM' in etf_data:
        universes["Russell 2000"] = _compute_universe_metrics(
            "Russell 2000", SMALL_CAP_WEIGHT, etf_data['IWM'], 2000
        )
    
    # Growth vs Value
    if 'QQQ' in etf_data:
        universes["Growth (QQQ)"] = _compute_universe_metrics(
            "Growth (QQQ)", GROWTH_VALUE_WEIGHT, etf_data['QQQ'], 100
        )
    
    # Dow Jones
    if 'DIA' in etf_data:
        universes["Dow Jones"] = _compute_universe_metrics("Dow Jones", INTERNATIONAL_WEIGHT, etf_data['DIA'], 30)
    
    # MSCI EAFE
    if 'EFA' in etf_data:
        universes["MSCI EAFE"] = _compute_universe_metrics("MSCI EAFE", INTERNATIONAL_WEIGHT, etf_data['EFA'], 900)
    
    # Emerging Markets
    if 'EEM' in etf_data:
        universes["Emerging Mkts"] = _compute_universe_metrics("Emerging Mkts", INTERNATIONAL_WEIGHT, etf_data['EEM'], 1400)
    
    # World ex-US
    if 'VEU' in etf_data:
        universes["World ex-US"] = _compute_universe_metrics("World ex-US", INTERNATIONAL_WEIGHT, etf_data['VEU'], 3700)
    
    # Japan
    if 'EWJ' in etf_data:
        universes["Japan"] = _compute_universe_metrics("Japan", INTERNATIONAL_WEIGHT, etf_data['EWJ'], 300)
    
    # China
    if 'FXI' in etf_data:
        universes["China"] = _compute_universe_metrics("China", INTERNATIONAL_WEIGHT, etf_data['FXI'], 50)
    
    # India
    if 'INDA' in etf_data:
        universes["India"] = _compute_universe_metrics("India", INTERNATIONAL_WEIGHT, etf_data['INDA'], 100)
    
    # Germany
    if 'EWG' in etf_data:
        universes["Germany"] = _compute_universe_metrics("Germany", INTERNATIONAL_WEIGHT, etf_data['EWG'], 60)
    
    # UK
    if 'EWU' in etf_data:
        universes["UK"] = _compute_universe_metrics("UK", INTERNATIONAL_WEIGHT, etf_data['EWU'], 100)
    
    # Brazil
    if 'EWZ' in etf_data:
        universes["Brazil"] = _compute_universe_metrics("Brazil", INTERNATIONAL_WEIGHT, etf_data['EWZ'], 50)
    
    # Australia
    if 'EWA' in etf_data:
        universes["Australia"] = _compute_universe_metrics("Australia", INTERNATIONAL_WEIGHT, etf_data['EWA'], 70)
    
    # Canada
    if 'EWC' in etf_data:
        universes["Canada"] = _compute_universe_metrics("Canada", INTERNATIONAL_WEIGHT, etf_data['EWC'], 90)
    
    # Compute breadth from all available stocks
    all_stocks = {**top100_data, **sp500_data}
    breadth = _compute_breadth_from_sample(all_stocks)
    
    # Compute correlation stress
    correlation = _compute_correlation_stress(all_stocks)
    
    # Aggregate temperature
    total_contribution = sum(u.stress_contribution for u in universes.values())
    total_weight = sum(u.weight for u in universes.values() if u.data_available)
    
    if total_weight > 0:
        temperature = total_contribution / total_weight
    else:
        temperature = 1.0  # Default to elevated if no data
    
    # Add correlation and breadth stress
    if correlation.systemic_risk_elevated:
        temperature += 0.3
    if breadth.breadth_warning:
        temperature += 0.2
    
    temperature = max(TEMP_MIN, min(TEMP_MAX, temperature))
    
    # Compute scale factor
    scale_factor = _compute_scale_factor(temperature)
    
    # Status and action
    status, action_text = _get_status_and_action(temperature)
    
    # Crash risk
    crash_risk_pct, crash_risk_level, vol_inversion_count, inverted_universes = \
        _compute_crash_risk(universes, breadth, correlation, temperature)
    
    # Exit signal
    exit_signal, exit_reason = _determine_exit_signal(
        temperature, crash_risk_pct, breadth, correlation
    )
    
    # Momentum and rotation
    overall_momentum = _compute_overall_momentum(universes)
    sector_rotation = _compute_sector_rotation(etf_data)
    
    # Compute sector-by-sector metrics
    sectors = _compute_sector_metrics(etf_data)
    
    # Compute currency pair metrics
    currency_data = _fetch_currency_data(start_date, end_date)
    currencies = _compute_currency_metrics(currency_data)
    
    # Data quality
    data_quality = sum(1 for u in universes.values() if u.data_available) / max(1, len(universes))
    
    return MarketTemperatureResult(
        temperature=temperature,
        scale_factor=scale_factor,
        universes=universes,
        breadth=breadth,
        correlation=correlation,
        computed_at=datetime.now().isoformat(),
        data_quality=data_quality,
        status=status,
        action_text=action_text,
        crash_risk_pct=crash_risk_pct,
        crash_risk_level=crash_risk_level,
        vol_inversion_count=vol_inversion_count,
        inverted_universes=inverted_universes,
        overall_momentum=overall_momentum,
        sector_rotation_signal=sector_rotation,
        exit_signal=exit_signal,
        exit_reason=exit_reason,
        sectors=sectors,
        currencies=currencies,
    )


# =============================================================================
# RENDERING
# =============================================================================

def render_market_temperature(result: MarketTemperatureResult, console=None) -> None:
    """Render market temperature with premium Apple-quality UX."""
    from rich.console import Console
    from rich.text import Text
    from rich.panel import Panel
    from rich import box
    
    if console is None:
        console = Console()
    
    # Temperature-based styling
    if result.temperature >= 1.5:
        temp_style = "bold red"
        status_color = "red"
        emoji = "🚨"
    elif result.temperature >= 1.0:
        temp_style = "bold orange1"
        status_color = "orange1"
        emoji = "🔥"
    elif result.temperature >= 0.7:
        temp_style = "bold yellow"
        status_color = "yellow"
        emoji = "⚠️"
    elif result.temperature >= 0.5:
        temp_style = "yellow"
        status_color = "yellow"
        emoji = "📊"
    else:
        temp_style = "bright_green"
        status_color = "bright_green"
        emoji = "✅"
    
    # Print clear separator and header
    console.print()
    console.print("  " + "═" * 76)
    
    header = Text()
    header.append(f"  {emoji} ", style="")
    header.append("MARKET TEMPERATURE", style="bold white")
    header.append("  —  ", style="dim")
    header.append(f"{result.temperature:.2f}", style=temp_style)
    header.append(f"  {result.status.upper()}", style=f"bold {status_color}")
    console.print(header)
    
    # Action text
    if result.exit_signal:
        action_line = Text()
        action_line.append("  🚨 EXIT SIGNAL: ", style="bold red")
        action_line.append(result.exit_reason or "", style="red")
        console.print(action_line)
    else:
        action_line = Text()
        action_line.append("  → ", style="dim")
        action_line.append(result.action_text, style="dim italic")
        console.print(action_line)
    
    console.print("  " + "═" * 76)
    console.print()
    
    # Universe breakdown
    console.print("  [dim]Universe Segments[/dim]")
    console.print()
    
    header_line = Text()
    header_line.append("  ")
    header_line.append("SEGMENT".ljust(18), style="bold dim")
    header_line.append("LEVEL".rjust(10), style="bold dim")
    header_line.append("1D".rjust(8), style="bold dim")
    header_line.append("5D".rjust(8), style="bold dim")
    header_line.append("21D".rjust(8), style="bold dim")
    header_line.append("VOL".rjust(8), style="bold dim")
    header_line.append("MOMENTUM".ljust(12), style="bold dim")
    console.print(header_line)
    console.print("  " + "─" * 78)
    
    for name, univ in result.universes.items():
        line = Text()
        line.append("  ")
        
        # Name with stress indicator
        if univ.stress_level > 1.0:
            line.append(f"🔴 {name}".ljust(18), style="red")
        elif univ.stress_level > 0.5:
            line.append(f"🟡 {name}".ljust(18), style="yellow")
        else:
            line.append(f"🟢 {name}".ljust(18), style="green")
        
        # Level
        if univ.current_level:
            line.append(f"{univ.current_level:,.0f}".rjust(10), style="white")
        else:
            line.append("—".rjust(10), style="dim")
        
        # Returns
        for ret in [univ.return_1d, univ.return_5d, univ.return_21d]:
            ret_style = "bright_green" if ret >= 0 else "indian_red1"
            line.append(f"{ret:+.1%}".rjust(8), style=ret_style)
        
        # Volatility
        vol_style = "red" if univ.volatility_percentile > 0.8 else ("yellow" if univ.volatility_percentile > 0.6 else "white")
        line.append(f"{univ.volatility_20d:.0%}".rjust(8), style=vol_style)
        
        # Momentum
        if "Strong" in univ.momentum_signal and "↑" in univ.momentum_signal:
            mom_style = "bold bright_green"
        elif "Rising" in univ.momentum_signal:
            mom_style = "bright_green"
        elif "Weak" in univ.momentum_signal or "↓" in univ.momentum_signal:
            mom_style = "bold indian_red1"
        elif "Falling" in univ.momentum_signal:
            mom_style = "indian_red1"
        else:
            mom_style = "dim"
        line.append(f"  {univ.momentum_signal}", style=mom_style)
        
        console.print(line)
    
    console.print()
    
    # Market Breadth
    console.print("  [dim]Market Breadth[/dim]")
    console.print()
    
    breadth_line = Text()
    breadth_line.append("  ")
    breadth_line.append("Above 50MA   ", style="dim")
    
    pct_50ma = result.breadth.pct_above_50ma
    if pct_50ma < BREADTH_DANGER_THRESHOLD:
        b_style = "bold red"
    elif pct_50ma < BREADTH_WARNING_THRESHOLD:
        b_style = "yellow"
    else:
        b_style = "green"
    breadth_line.append(f"{pct_50ma:.0%}", style=b_style)
    
    breadth_line.append("   Above 200MA   ", style="dim")
    breadth_line.append(f"{result.breadth.pct_above_200ma:.0%}", style="white")
    
    breadth_line.append("   A/D Ratio   ", style="dim")
    ad_style = "green" if result.breadth.advance_decline_ratio > 1 else "red"
    breadth_line.append(f"{result.breadth.advance_decline_ratio:.2f}", style=ad_style)
    console.print(breadth_line)
    
    interp_line = Text()
    interp_line.append("  ")
    interp_line.append(result.breadth.interpretation, style="dim italic")
    console.print(interp_line)
    
    console.print()
    
    # Correlation Stress
    console.print("  [dim]Correlation Stress[/dim]")
    console.print()
    
    corr_line = Text()
    corr_line.append("  ")
    corr_line.append("Avg Correlation   ", style="dim")
    
    if result.correlation.systemic_risk_elevated:
        c_style = "bold red"
    elif result.correlation.avg_correlation > 0.60:
        c_style = "yellow"
    else:
        c_style = "green"
    corr_line.append(f"{result.correlation.avg_correlation:.0%}", style=c_style)
    
    corr_line.append("   ", style="")
    corr_line.append(result.correlation.interpretation, style="dim italic")
    console.print(corr_line)
    
    console.print()
    
    # Crash Risk Panel
    if result.crash_risk_pct > 0.05:
        console.print("  [dim]Crash Risk Assessment[/dim]")
        console.print()
        
        # Risk gauge
        gauge_width = 40
        filled = int(min(1.0, result.crash_risk_pct / 0.50) * gauge_width)
        
        gauge_line = Text()
        gauge_line.append("  ")
        for i in range(gauge_width):
            if i < filled:
                if i < gauge_width * 0.4:
                    gauge_line.append("━", style="green")
                elif i < gauge_width * 0.7:
                    gauge_line.append("━", style="yellow")
                else:
                    gauge_line.append("━", style="red")
            else:
                gauge_line.append("─", style="bright_black")
        console.print(gauge_line)
        
        risk_line = Text()
        risk_line.append("  ")
        
        if result.crash_risk_level == "Extreme":
            r_style = "bold red"
        elif result.crash_risk_level == "High":
            r_style = "red"
        elif result.crash_risk_level == "Elevated":
            r_style = "yellow"
        else:
            r_style = "green"
        
        risk_line.append(f"{result.crash_risk_pct:.0%}", style=r_style)
        risk_line.append(f"  {result.crash_risk_level.upper()}", style=r_style)
        
        if result.vol_inversion_count > 0:
            risk_line.append(f"  ({result.vol_inversion_count} vol inversions)", style="dim")
        console.print(risk_line)
        
        console.print()
    
    # Summary metrics
    console.print("  [dim]Summary[/dim]")
    console.print()
    
    summary_line = Text()
    summary_line.append("  ")
    summary_line.append("Position Scale   ", style="dim")
    
    scale = result.scale_factor
    if scale > 0.9:
        s_style = "green"
    elif scale > 0.6:
        s_style = "yellow"
    else:
        s_style = "red"
    summary_line.append(f"{scale:.0%}", style=s_style)
    
    summary_line.append("   Momentum   ", style="dim")
    summary_line.append(result.overall_momentum, style="white")
    
    summary_line.append("   Rotation   ", style="dim")
    summary_line.append(result.sector_rotation_signal, style="white")
    console.print(summary_line)
    
    qual_line = Text()
    qual_line.append("  ")
    qual_line.append("Data Quality   ", style="dim")
    qual_line.append(f"{result.data_quality:.0%}", style="green" if result.data_quality > 0.8 else "yellow")
    qual_line.append(f"   Computed   ", style="dim")
    qual_line.append(result.computed_at[:19], style="dim italic")
    console.print(qual_line)
    
    console.print()
    console.print()


# =============================================================================
# STANDALONE CLI
# =============================================================================

if __name__ == "__main__":
    """Run market temperature computation and display."""
    import argparse
    import os
    import sys
    import time
    from rich.console import Console
    
    # Suppress yfinance output at environment level
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['YF_LOG_LEVEL'] = 'CRITICAL'
    
    # Suppress yfinance warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)
    logging.getLogger("peewee").setLevel(logging.CRITICAL)
    
    parser = argparse.ArgumentParser(description="Market Temperature — US Equity Market Assessment")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--start", default="2020-01-01", help="Start date for historical data")
    args = parser.parse_args()
    
    # Compute market temperature with full output suppression
    with _SuppressOutput():
        result = compute_market_temperature(start_date=args.start)
    
    # Wait for any lingering threads to complete their output
    time.sleep(1.0)
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Clear the line with carriage return and spaces
    sys.stdout.write("\r" + " " * 100 + "\r")
    sys.stdout.flush()
    
    console = Console()
    
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        render_market_temperature(result, console=console)
