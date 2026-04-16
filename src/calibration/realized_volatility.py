#!/usr/bin/env python3
"""
===============================================================================
REALIZED VOLATILITY ESTIMATORS — Range-Based Volatility for Improved Calibration
===============================================================================

This module implements range-based volatility estimators that are significantly
more efficient than close-to-close EWMA volatility.

MATHEMATICAL FOUNDATION:

The efficiency of a volatility estimator is measured by the ratio of variance
of the estimator to the theoretical minimum (Cramér-Rao bound).

1. CLOSE-TO-CLOSE (Traditional EWMA):
   σ²_CC = Var(log(C_t/C_{t-1}))
   Efficiency: 1.0x (baseline)
   
2. PARKINSON (1980) - Uses High-Low range:
   σ²_P = (1/(4*log(2))) * (log(H/L))²
   Efficiency: 5.2x more efficient than close-to-close
   
3. GARMAN-KLASS (1980) - Uses OHLC:
   σ²_GK = 0.5*(log(H/L))² - (2*log(2)-1)*(log(C/O))²
   Efficiency: 7.4x more efficient than close-to-close
   
4. ROGERS-SATCHELL (1991) - Drift-robust:
   σ²_RS = log(H/C)*log(H/O) + log(L/C)*log(L/O)
   Efficiency: 8.0x, handles non-zero drift
   
5. YANG-ZHANG (2000) - Overnight jump robust:
   σ²_YZ = σ²_overnight + k*σ²_open + (1-k)*σ²_RS
   Efficiency: 14x, handles overnight gaps

WHY THIS MATTERS FOR CALIBRATION:

The Kalman filter observation equation is:
    r_t = μ_t + √(c·σ_t²)·ε_t

If σ_t is poorly estimated (high variance), then:
- c parameter absorbs the noise → biased c estimate
- PIT histogram shows systematic bias
- Hyvärinen score degrades

Range-based estimators provide 5-14x more precise σ_t estimates, directly
improving PIT calibration without adding parameters.

IMPLEMENTATION NOTES:

- All estimators are computed on a rolling window (default 21 days)
- EWMA-style exponential weighting is applied for time-varying estimates
- Fallback to close-to-close when OHLC data is unavailable
- NaN handling for missing data points

===============================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
from enum import Enum
import warnings


# =============================================================================
# CONFIGURATION
# =============================================================================

class VolatilityEstimator(Enum):
    """Available volatility estimation methods."""
    EWMA = "ewma"                    # Traditional close-to-close EWMA
    PARKINSON = "parkinson"          # High-Low range
    GARMAN_KLASS = "garman_klass"    # OHLC (default)
    ROGERS_SATCHELL = "rogers_satchell"  # Drift-robust OHLC
    YANG_ZHANG = "yang_zhang"        # Overnight-jump robust


# Default estimator for production use
DEFAULT_VOLATILITY_ESTIMATOR = VolatilityEstimator.GARMAN_KLASS

# Default span for EWMA-style smoothing
DEFAULT_VOL_SPAN = 21

# Annualization factor (trading days)
ANNUALIZATION_FACTOR = 252

# Minimum variance floor (prevents division by zero)
# Raised from 1e-12 to 1e-8 to limit HAR contamination from stale-price
# days where GK variance = 0 (O=H=L=C). At 1e-12 the HAR rolling mean of
# near-zero GK values produces vol ≈ 1e-6 which causes pathological z-scores.
MIN_VARIANCE = 1e-8


# =============================================================================
# DATA CLASS FOR VOLATILITY RESULT
# =============================================================================

@dataclass
class VolatilityResult:
    """Result of volatility estimation."""
    volatility: np.ndarray          # Time series of volatility estimates
    estimator: VolatilityEstimator  # Method used
    efficiency_vs_cc: float         # Theoretical efficiency vs close-to-close
    annualized: bool                # Whether values are annualized
    span: int                       # Smoothing span used
    
    # Diagnostic fields
    n_missing_ohlc: int = 0         # Number of points with missing OHLC
    fallback_used: bool = False     # Whether fallback to EWMA was needed
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "estimator": self.estimator.value,
            "efficiency_vs_cc": self.efficiency_vs_cc,
            "annualized": self.annualized,
            "span": self.span,
            "n_missing_ohlc": self.n_missing_ohlc,
            "fallback_used": self.fallback_used,
        }


# =============================================================================
# CORE VOLATILITY ESTIMATORS
# =============================================================================

def _parkinson_variance(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """
    Parkinson (1980) range-based variance estimator.
    
    σ²_P = (1/(4*log(2))) * (log(H/L))²
    
    5.2x more efficient than close-to-close.
    """
    log_hl = np.log(high / low)
    variance = (log_hl ** 2) / (4 * np.log(2))
    return variance


def _garman_klass_variance(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """
    Garman-Klass (1980) OHLC variance estimator.
    
    σ²_GK = 0.5*(log(H/L))² - (2*log(2)-1)*(log(C/O))²
    
    7.4x more efficient than close-to-close.
    Uses all OHLC information optimally under assumption of no drift.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    
    variance = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    
    # Ensure non-negative (numerical precision issues can cause negative values)
    variance = np.maximum(variance, MIN_VARIANCE)
    
    return variance


def _rogers_satchell_variance(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:
    """
    Rogers-Satchell (1991) drift-robust variance estimator.
    
    σ²_RS = log(H/C)*log(H/O) + log(L/C)*log(L/O)
    
    8.0x more efficient than close-to-close.
    Handles non-zero drift (trending markets).
    """
    log_hc = np.log(high / close)
    log_ho = np.log(high / open_)
    log_lc = np.log(low / close)
    log_lo = np.log(low / open_)
    
    variance = log_hc * log_ho + log_lc * log_lo
    
    # Ensure non-negative
    variance = np.maximum(variance, MIN_VARIANCE)
    
    return variance


def _yang_zhang_variance(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 21,
) -> np.ndarray:
    """
    Yang-Zhang (2000) overnight-jump robust variance estimator.
    
    σ²_YZ = σ²_overnight + k*σ²_open + (1-k)*σ²_RS
    
    14x more efficient than close-to-close.
    Handles overnight gaps (earnings, news events).
    """
    n = len(close)
    variance = np.full(n, np.nan)
    
    if n < window + 1:
        return variance
    
    # Overnight return: log(O_t / C_{t-1})
    log_overnight = np.zeros(n)
    log_overnight[1:] = np.log(open_[1:] / close[:-1])
    
    # Open-to-close return: log(C_t / O_t)
    log_oc = np.log(close / open_)
    
    # Rogers-Satchell component
    rs_var = _rogers_satchell_variance(open_, high, low, close)
    
    # Compute rolling statistics
    for t in range(window, n):
        # Window slices
        overnight_window = log_overnight[t-window+1:t+1]
        oc_window = log_oc[t-window+1:t+1]
        rs_window = rs_var[t-window+1:t+1]
        
        # Overnight variance
        overnight_mean = np.mean(overnight_window)
        var_overnight = np.mean((overnight_window - overnight_mean) ** 2)
        
        # Open-to-close variance
        oc_mean = np.mean(oc_window)
        var_oc = np.mean((oc_window - oc_mean) ** 2)
        
        # Rogers-Satchell mean
        var_rs = np.mean(rs_window)
        
        # Optimal k parameter (Yang-Zhang, 2000)
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        # Combined estimator
        variance[t] = var_overnight + k * var_oc + (1 - k) * var_rs
    
    # Ensure non-negative
    variance = np.maximum(variance, MIN_VARIANCE)
    
    return variance


def _ewma_variance_cc(returns: np.ndarray, span: int = 21) -> np.ndarray:
    """
    Traditional EWMA close-to-close variance.
    
    This is the baseline estimator for comparison.
    """
    # Convert to pandas for ewm
    ret_series = pd.Series(returns)
    variance = ret_series.ewm(span=span, adjust=False).var().values
    
    # Ensure non-negative
    variance = np.maximum(variance, MIN_VARIANCE)
    
    return variance


# =============================================================================
# EWMA SMOOTHING FOR RANGE-BASED ESTIMATORS
# =============================================================================

def _ewma_smooth(values: np.ndarray, span: int = 21) -> np.ndarray:
    """
    Apply EWMA smoothing to point-in-time variance estimates.
    
    Range-based estimators produce daily variance estimates.
    We smooth these with EWMA to get stable, time-varying volatility.
    """
    series = pd.Series(values)
    smoothed = series.ewm(span=span, adjust=False).mean().values
    return smoothed


# =============================================================================
# MAIN ENTRY POINTS
# =============================================================================

def compute_realized_volatility(
    open_: Optional[np.ndarray] = None,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    close: Optional[np.ndarray] = None,
    returns: Optional[np.ndarray] = None,
    estimator: VolatilityEstimator = DEFAULT_VOLATILITY_ESTIMATOR,
    span: int = DEFAULT_VOL_SPAN,
    annualize: bool = True,
) -> VolatilityResult:
    """
    Compute realized volatility using specified estimator.
    
    Args:
        open_: Open prices (required for GK, RS, YZ)
        high: High prices (required for all range-based)
        low: Low prices (required for all range-based)
        close: Close prices (required for all)
        returns: Log returns (alternative to close for EWMA)
        estimator: Volatility estimation method
        span: EWMA smoothing span
        annualize: Whether to annualize the result
        
    Returns:
        VolatilityResult with time series and metadata
    """
    # Determine which data is available
    has_ohlc = all(x is not None for x in [open_, high, low, close])
    has_hl = all(x is not None for x in [high, low])
    has_close = close is not None
    has_returns = returns is not None
    
    # Efficiency multipliers vs close-to-close
    efficiency_map = {
        VolatilityEstimator.EWMA: 1.0,
        VolatilityEstimator.PARKINSON: 5.2,
        VolatilityEstimator.GARMAN_KLASS: 7.4,
        VolatilityEstimator.ROGERS_SATCHELL: 8.0,
        VolatilityEstimator.YANG_ZHANG: 14.0,
    }
    
    n_missing_ohlc = 0
    fallback_used = False
    actual_estimator = estimator
    
    # Attempt requested estimator, fallback if data unavailable
    if estimator in [VolatilityEstimator.GARMAN_KLASS, 
                     VolatilityEstimator.ROGERS_SATCHELL,
                     VolatilityEstimator.YANG_ZHANG]:
        if not has_ohlc:
            warnings.warn(f"OHLC data required for {estimator.value}, falling back to EWMA")
            actual_estimator = VolatilityEstimator.EWMA
            fallback_used = True
    elif estimator == VolatilityEstimator.PARKINSON:
        if not has_hl:
            warnings.warn("High-Low data required for Parkinson, falling back to EWMA")
            actual_estimator = VolatilityEstimator.EWMA
            fallback_used = True
    
    # Compute variance based on estimator
    if actual_estimator == VolatilityEstimator.EWMA:
        if has_returns:
            variance = _ewma_variance_cc(returns, span)
        elif has_close:
            log_ret = np.log(close[1:] / close[:-1])
            variance = _ewma_variance_cc(log_ret, span)
            # Prepend NaN for first observation
            variance = np.concatenate([[np.nan], variance])
        else:
            raise ValueError("Either returns or close prices required for EWMA")
            
    elif actual_estimator == VolatilityEstimator.PARKINSON:
        daily_var = _parkinson_variance(high, low)
        variance = _ewma_smooth(daily_var, span)
        
    elif actual_estimator == VolatilityEstimator.GARMAN_KLASS:
        daily_var = _garman_klass_variance(open_, high, low, close)
        variance = _ewma_smooth(daily_var, span)
        
        # Count missing OHLC (where any is NaN or invalid)
        valid_mask = (
            np.isfinite(open_) & np.isfinite(high) & 
            np.isfinite(low) & np.isfinite(close) &
            (high >= low) & (high >= open_) & (high >= close) &
            (low <= open_) & (low <= close)
        )
        n_missing_ohlc = int(np.sum(~valid_mask))
        
    elif actual_estimator == VolatilityEstimator.ROGERS_SATCHELL:
        daily_var = _rogers_satchell_variance(open_, high, low, close)
        variance = _ewma_smooth(daily_var, span)
        
    elif actual_estimator == VolatilityEstimator.YANG_ZHANG:
        variance = _yang_zhang_variance(open_, high, low, close, window=span)
    
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    
    # Convert variance to volatility (standard deviation)
    volatility = np.sqrt(np.maximum(variance, MIN_VARIANCE))
    
    # Annualize if requested
    if annualize:
        volatility = volatility * np.sqrt(ANNUALIZATION_FACTOR)
    
    return VolatilityResult(
        volatility=volatility,
        estimator=actual_estimator,
        efficiency_vs_cc=efficiency_map.get(actual_estimator, 1.0),
        annualized=annualize,
        span=span,
        n_missing_ohlc=n_missing_ohlc,
        fallback_used=fallback_used,
    )


def compute_gk_volatility(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    span: int = DEFAULT_VOL_SPAN,
    annualize: bool = False,
) -> np.ndarray:
    """
    Convenience function for Garman-Klass volatility.
    
    This is the recommended replacement for EWMA in tuning and signals.
    
    Args:
        open_: Open prices
        high: High prices
        low: Low prices
        close: Close prices
        span: EWMA smoothing span (default 21)
        annualize: Whether to annualize (default False for Kalman filter use)
        
    Returns:
        Array of volatility estimates (same length as input)
    """
    result = compute_realized_volatility(
        open_=open_,
        high=high,
        low=low,
        close=close,
        estimator=VolatilityEstimator.GARMAN_KLASS,
        span=span,
        annualize=annualize,
    )
    return result.volatility


def compute_volatility_from_df(
    df: pd.DataFrame,
    estimator: VolatilityEstimator = DEFAULT_VOLATILITY_ESTIMATOR,
    span: int = DEFAULT_VOL_SPAN,
    annualize: bool = False,
) -> Tuple[np.ndarray, VolatilityResult]:
    """
    Compute volatility from a DataFrame with OHLC columns.
    
    Automatically detects column names (case-insensitive).
    Falls back to close-to-close if OHLC not available.
    
    Args:
        df: DataFrame with price data
        estimator: Volatility estimation method
        span: EWMA smoothing span
        annualize: Whether to annualize
        
    Returns:
        Tuple of (volatility array, VolatilityResult metadata)
    """
    # Detect column names (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    
    open_ = df[cols['open']].values if 'open' in cols else None
    high = df[cols['high']].values if 'high' in cols else None
    low = df[cols['low']].values if 'low' in cols else None
    close = df[cols['close']].values if 'close' in cols else None
    
    # Also try 'adj close' for close
    if close is None and 'adj close' in cols:
        close = df[cols['adj close']].values
    
    if close is None:
        raise ValueError("No 'Close' or 'Adj Close' column found in DataFrame")
    
    # Compute returns for EWMA fallback
    returns = None
    if estimator == VolatilityEstimator.EWMA:
        returns = np.log(close[1:] / close[:-1])
    
    result = compute_realized_volatility(
        open_=open_,
        high=high,
        low=low,
        close=close,
        returns=returns,
        estimator=estimator,
        span=span,
        annualize=annualize,
    )
    
    return result.volatility, result


# =============================================================================
# HYBRID ESTIMATOR (PRODUCTION RECOMMENDED)
# =============================================================================

def compute_hybrid_volatility(
    open_: Optional[np.ndarray],
    high: Optional[np.ndarray],
    low: Optional[np.ndarray],
    close: np.ndarray,
    span: int = DEFAULT_VOL_SPAN,
    annualize: bool = False,
) -> Tuple[np.ndarray, str]:
    """
    Compute volatility using best available estimator.
    
    Priority:
    1. Garman-Klass (if OHLC available and valid)
    2. Parkinson (if High-Low available)
    3. EWMA close-to-close (fallback)
    
    This is the RECOMMENDED function for production use.
    
    Args:
        open_: Open prices (can be None)
        high: High prices (can be None)
        low: Low prices (can be None)
        close: Close prices (required)
        span: EWMA smoothing span
        annualize: Whether to annualize
        
    Returns:
        Tuple of (volatility array, estimator name used)
    """
    has_ohlc = all(x is not None and len(x) == len(close) 
                   for x in [open_, high, low])
    has_hl = all(x is not None and len(x) == len(close) 
                 for x in [high, low])
    
    if has_ohlc:
        # Validate OHLC relationships
        valid_ohlc = (
            np.all(np.isfinite(open_)) and 
            np.all(np.isfinite(high)) and
            np.all(np.isfinite(low)) and
            np.all(high >= low)
        )
        if valid_ohlc:
            vol = compute_gk_volatility(open_, high, low, close, span, annualize)
            return vol, "GK"
    
    if has_hl:
        result = compute_realized_volatility(
            high=high, low=low, close=close,
            estimator=VolatilityEstimator.PARKINSON,
            span=span, annualize=annualize,
        )
        return result.volatility, "Parkinson"
    
    # Fallback to EWMA
    log_ret = np.log(close[1:] / close[:-1])
    variance = _ewma_variance_cc(log_ret, span)
    vol = np.sqrt(variance)
    if annualize:
        vol = vol * np.sqrt(ANNUALIZATION_FACTOR)
    # Prepend NaN
    vol = np.concatenate([[np.nan], vol])
    return vol, "EWMA"


# =============================================================================
# HAR (HETEROGENEOUS AUTOREGRESSIVE) VOLATILITY - February 2026
# =============================================================================
# Multi-horizon memory for improved crash detection (Corsi 2009)
# σ²_t = w₁·RV_daily + w₂·RV_weekly + w₃·RV_monthly
# Reduces lag during crash onset compared to single-horizon EWMA
# =============================================================================

# Default HAR weights from Corsi (2009) empirical findings
HAR_WEIGHT_DAILY = 0.5
HAR_WEIGHT_WEEKLY = 0.3
HAR_WEIGHT_MONTHLY = 0.2

# HAR horizons (trading days)
HAR_HORIZON_DAILY = 1
HAR_HORIZON_WEEKLY = 5
HAR_HORIZON_MONTHLY = 22


def compute_har_volatility(
    returns: Optional[np.ndarray] = None,
    open_: Optional[np.ndarray] = None,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    close: Optional[np.ndarray] = None,
    w_daily: float = HAR_WEIGHT_DAILY,
    w_weekly: float = HAR_WEIGHT_WEEKLY,
    w_monthly: float = HAR_WEIGHT_MONTHLY,
    use_gk: bool = True,
    annualize: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute HAR (Heterogeneous Autoregressive) volatility with multi-horizon memory.
    
    HAR captures the "rough" nature of volatility by combining:
    - Daily RV: Captures immediate shocks
    - Weekly RV: Medium-term persistence
    - Monthly RV: Long-term baseline
    
    This reduces lag during crash onset compared to single-horizon EWMA.
    
    Formula:
        σ²_t = w_daily × RV_daily(t) + w_weekly × RV_weekly(t) + w_monthly × RV_monthly(t)
    
    Where:
        RV_daily = squared return (or GK variance)
        RV_weekly = 5-day rolling mean of daily RV
        RV_monthly = 22-day rolling mean of daily RV
    
    Args:
        returns: Log returns (used if OHLC not provided)
        open_, high, low, close: OHLC data for Garman-Klass base
        w_daily: Weight for daily component (default 0.5)
        w_weekly: Weight for weekly component (default 0.3)
        w_monthly: Weight for monthly component (default 0.2)
        use_gk: Use Garman-Klass for daily RV if OHLC available
        annualize: Whether to annualize output
        
    Returns:
        Tuple of (volatility array, diagnostics dict)
        
    References:
        Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility"
        Journal of Financial Econometrics, 7(2), 174-196.
    """
    # Normalize weights
    w_sum = w_daily + w_weekly + w_monthly
    w_d = w_daily / w_sum
    w_w = w_weekly / w_sum
    w_m = w_monthly / w_sum
    
    # Compute daily realized variance
    has_ohlc = all(x is not None for x in [open_, high, low, close])
    
    if use_gk and has_ohlc:
        # Garman-Klass daily variance (7.4x efficient)
        rv_daily = _garman_klass_variance(open_, high, low, close)
        estimator_used = "GK"
    elif returns is not None:
        # Squared returns
        rv_daily = returns ** 2
        estimator_used = "SqRet"
    elif close is not None:
        # Compute returns from close
        log_ret = np.log(close[1:] / close[:-1])
        rv_daily = np.concatenate([[np.nan], log_ret ** 2])
        estimator_used = "SqRet"
    else:
        raise ValueError("Either returns or close prices must be provided")
    
    n = len(rv_daily)
    
    # Compute multi-horizon RV components
    rv_daily_clean = np.where(np.isfinite(rv_daily), rv_daily, 0.0)
    
    # Weekly RV: 5-day rolling mean
    rv_weekly = np.full(n, np.nan)
    for t in range(HAR_HORIZON_WEEKLY, n):
        rv_weekly[t] = np.mean(rv_daily_clean[t-HAR_HORIZON_WEEKLY+1:t+1])
    # Fill early values with expanding mean
    for t in range(1, HAR_HORIZON_WEEKLY):
        rv_weekly[t] = np.mean(rv_daily_clean[1:t+1])
    
    # Monthly RV: 22-day rolling mean
    rv_monthly = np.full(n, np.nan)
    for t in range(HAR_HORIZON_MONTHLY, n):
        rv_monthly[t] = np.mean(rv_daily_clean[t-HAR_HORIZON_MONTHLY+1:t+1])
    # Fill early values with expanding mean
    for t in range(1, HAR_HORIZON_MONTHLY):
        rv_monthly[t] = np.mean(rv_daily_clean[1:t+1])
    
    # HAR composite variance
    har_variance = (
        w_d * rv_daily_clean + 
        w_w * np.nan_to_num(rv_weekly, nan=rv_daily_clean) + 
        w_m * np.nan_to_num(rv_monthly, nan=rv_daily_clean)
    )
    
    # Floor to prevent zero variance
    har_variance = np.maximum(har_variance, MIN_VARIANCE)
    
    # Convert to volatility
    har_vol = np.sqrt(har_variance)
    
    if annualize:
        har_vol = har_vol * np.sqrt(ANNUALIZATION_FACTOR)
    
    diagnostics = {
        "estimator": f"HAR-{estimator_used}",
        "weights": {"daily": w_d, "weekly": w_w, "monthly": w_m},
        "rv_daily_mean": float(np.nanmean(rv_daily)),
        "rv_weekly_mean": float(np.nanmean(rv_weekly)),
        "rv_monthly_mean": float(np.nanmean(rv_monthly)),
        "har_vol_mean": float(np.nanmean(har_vol)),
    }
    
    return har_vol, diagnostics


def compute_hybrid_volatility_har(
    open_: Optional[np.ndarray],
    high: Optional[np.ndarray],
    low: Optional[np.ndarray],
    close: np.ndarray,
    use_har: bool = True,
    har_weights: Optional[Tuple[float, float, float]] = None,
    span: int = DEFAULT_VOL_SPAN,
    annualize: bool = False,
) -> Tuple[np.ndarray, str]:
    """
    Compute volatility using HAR (multi-horizon) or standard GK.
    
    This is an enhanced version of compute_hybrid_volatility that supports
    HAR multi-horizon memory for improved crash detection.
    
    Args:
        open_, high, low, close: OHLC data
        use_har: If True, use HAR multi-horizon; if False, use standard GK/EWMA
        har_weights: Optional (w_daily, w_weekly, w_monthly) tuple
        span: EWMA span for non-HAR fallback
        annualize: Whether to annualize
        
    Returns:
        Tuple of (volatility array, estimator name used)
    """
    has_ohlc = all(x is not None and len(x) == len(close) 
                   for x in [open_, high, low])
    
    if use_har:
        if har_weights is not None:
            w_d, w_w, w_m = har_weights
        else:
            w_d, w_w, w_m = HAR_WEIGHT_DAILY, HAR_WEIGHT_WEEKLY, HAR_WEIGHT_MONTHLY
        
        try:
            vol, diag = compute_har_volatility(
                open_=open_ if has_ohlc else None,
                high=high if has_ohlc else None,
                low=low if has_ohlc else None,
                close=close,
                w_daily=w_d,
                w_weekly=w_w,
                w_monthly=w_m,
                use_gk=has_ohlc,
                annualize=annualize,
            )
            return vol, diag.get("estimator", "HAR")
        except Exception:
            # Fall through to standard hybrid
            pass
    
    # Standard hybrid (GK/Parkinson/EWMA)
    return compute_hybrid_volatility(open_, high, low, close, span, annualize)


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def compare_estimators(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    span: int = 21,
) -> Dict[str, Dict]:
    """
    Compare all available volatility estimators on the same data.
    
    Useful for diagnosing which estimator works best for a given asset.
    
    Returns:
        Dictionary mapping estimator name to stats (mean, std, correlation with GK)
    """
    results = {}
    
    # Compute all estimators
    log_ret = np.log(close[1:] / close[:-1])
    
    ewma_vol = np.sqrt(_ewma_variance_cc(log_ret, span))
    ewma_vol = np.concatenate([[np.nan], ewma_vol])
    
    park_var = _parkinson_variance(high, low)
    park_vol = np.sqrt(_ewma_smooth(park_var, span))
    
    gk_var = _garman_klass_variance(open_, high, low, close)
    gk_vol = np.sqrt(_ewma_smooth(gk_var, span))
    
    rs_var = _rogers_satchell_variance(open_, high, low, close)
    rs_vol = np.sqrt(_ewma_smooth(rs_var, span))
    
    yz_var = _yang_zhang_variance(open_, high, low, close, span)
    yz_vol = np.sqrt(np.maximum(yz_var, MIN_VARIANCE))
    
    estimators = {
        "EWMA": ewma_vol,
        "Parkinson": park_vol,
        "Garman-Klass": gk_vol,
        "Rogers-Satchell": rs_vol,
        "Yang-Zhang": yz_vol,
    }
    
    for name, vol in estimators.items():
        valid = np.isfinite(vol)
        if np.sum(valid) < 10:
            continue
            
        results[name] = {
            "mean": float(np.nanmean(vol)),
            "std": float(np.nanstd(vol)),
            "min": float(np.nanmin(vol)),
            "max": float(np.nanmax(vol)),
            "corr_with_gk": float(np.corrcoef(
                vol[valid & np.isfinite(gk_vol)],
                gk_vol[valid & np.isfinite(gk_vol)]
            )[0, 1]) if name != "Garman-Klass" else 1.0,
        }
    
    return results


# =============================================================================
# AUGMENTATION STRING GENERATION
# =============================================================================

def get_volatility_augmentation_code(estimator: VolatilityEstimator) -> str:
    """
    Get short code for volatility estimator for display in model string.
    
    Examples:
        VolatilityEstimator.GARMAN_KLASS -> "GK"
        VolatilityEstimator.EWMA -> "EWMA"
    """
    code_map = {
        VolatilityEstimator.EWMA: "EWMA",
        VolatilityEstimator.PARKINSON: "Park",
        VolatilityEstimator.GARMAN_KLASS: "GK",
        VolatilityEstimator.ROGERS_SATCHELL: "RS",
        VolatilityEstimator.YANG_ZHANG: "YZ",
    }
    return code_map.get(estimator, "?")


# =============================================================================
# GARMAN-KLASS c PRIOR (Story 2.2 — Observation Noise Calibration)
# =============================================================================

# Bounds for the GK-informed c prior to prevent pathological values
GK_C_PRIOR_MIN = 0.3   # Floor: c prior never below 0.3
GK_C_PRIOR_MAX = 5.0   # Cap: c prior never above 5.0
GK_C_PRIOR_FALLBACK = 1.0  # Neutral prior when OHLC unavailable

# Bounds multipliers for L-BFGS-B integration
GK_C_BOUNDS_LOWER_MULT = 0.5  # c_min = 0.5 * c_prior
GK_C_BOUNDS_UPPER_MULT = 2.0  # c_max = 2.0 * c_prior


def gk_c_prior(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    span: int = DEFAULT_VOL_SPAN,
    fallback: float = GK_C_PRIOR_FALLBACK,
) -> float:
    """
    Compute empirical c prior from Garman-Klass / close-to-close variance ratio.

    c_prior = sigma^2_GK / sigma^2_CC

    The GK estimator uses OHLC data (7.4x more efficient than close-to-close).
    When GK variance matches CC variance, c ~ 1.0 (balanced signal/noise).
    When they diverge, the ratio provides an informative starting point for
    the observation noise scalar c in Kalman filter estimation.

    Parameters
    ----------
    open_, high, low, close : np.ndarray
        OHLC price arrays (must be same length, positive values).
    span : int
        EWM smoothing span for variance estimates.
    fallback : float
        Value returned if computation fails (default 1.0 = uninformative).

    Returns
    -------
    c_prior : float
        Empirical prior for c, clipped to [GK_C_PRIOR_MIN, GK_C_PRIOR_MAX].
    """
    try:
        open_ = np.asarray(open_, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)

        # Need at least span + 5 points for stable EWMA
        min_len = min(len(open_), len(high), len(low), len(close))
        if min_len < span + 5:
            return fallback

        # Validate OHLC consistency
        valid = (
            np.isfinite(open_) & np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
            & (open_ > 0) & (high > 0) & (low > 0) & (close > 0)
            & (high >= low)
        )
        if np.sum(valid) < span + 5:
            return fallback

        # Compute GK variance (point-in-time)
        gk_var = _garman_klass_variance(open_, high, low, close)

        # Compute CC variance (close-to-close squared log returns)
        log_ret = np.log(close[1:] / close[:-1])
        cc_var = log_ret ** 2  # Pointwise squared return

        # Align lengths (GK has same length as input, CC is 1 shorter)
        n = min(len(gk_var), len(cc_var))
        gk_var = gk_var[1 : n + 1]  # Drop first (matches CC alignment)
        cc_var = cc_var[:n]

        # EWMA smooth both for stability
        gk_smoothed = _ewma_smooth(gk_var, span=span)
        cc_smoothed = _ewma_smooth(cc_var, span=span)

        # Use median of the ratio for robustness against outliers
        # Only use latter half (after EWMA burn-in)
        burn_in = max(span * 2, 42)
        if n <= burn_in:
            return fallback

        gk_tail = gk_smoothed[burn_in:]
        cc_tail = cc_smoothed[burn_in:]

        # Filter valid ratios
        valid_ratio = (
            np.isfinite(gk_tail) & np.isfinite(cc_tail)
            & (gk_tail > MIN_VARIANCE) & (cc_tail > MIN_VARIANCE)
        )
        if np.sum(valid_ratio) < 10:
            return fallback

        ratio = gk_tail[valid_ratio] / cc_tail[valid_ratio]
        c_prior = float(np.median(ratio))

        # Clip to safe range
        c_prior = float(np.clip(c_prior, GK_C_PRIOR_MIN, GK_C_PRIOR_MAX))

        return c_prior

    except Exception:
        return fallback


def compute_gk_informed_c_bounds(
    c_prior: float,
    lower_mult: float = GK_C_BOUNDS_LOWER_MULT,
    upper_mult: float = GK_C_BOUNDS_UPPER_MULT,
    absolute_min: float = 0.01,
    absolute_max: float = 10.0,
) -> Tuple[float, float]:
    """
    Compute GK-informed c bounds for L-BFGS-B optimization.

    c_min = max(absolute_min, lower_mult * c_prior)
    c_max = min(absolute_max, upper_mult * c_prior)

    Parameters
    ----------
    c_prior : float
        GK-derived c prior from gk_c_prior().
    lower_mult : float
        Lower multiplier (default 0.5).
    upper_mult : float
        Upper multiplier (default 2.0).
    absolute_min : float
        Hard floor for c_min.
    absolute_max : float
        Hard ceiling for c_max.

    Returns
    -------
    (c_min, c_max) : Tuple[float, float]
    """
    c_min = max(absolute_min, lower_mult * c_prior)
    c_max = min(absolute_max, upper_mult * c_prior)

    # Ensure valid range
    if c_min >= c_max:
        c_min = max(absolute_min, 0.5 * c_prior)
        c_max = min(absolute_max, 2.5 * c_prior)

    # Final safety
    if c_min >= c_max:
        c_min, c_max = 0.1, 5.0

    return (c_min, c_max)


# =============================================================================
# STORY 7.1: MULTI-ESTIMATOR VOLATILITY FUSION KERNEL
# =============================================================================
#
# Fuses GK, Yang-Zhang, Parkinson, and EWMA with regime-adaptive weights.
#
#   sigma_fused^2 = sum_k w_k(regime) * sigma_k^2
#
# Regime logic:
#   - CRISIS_JUMP (4):     YZ-heavy (gap-robust)
#   - HIGH_VOL_TREND (1):  YZ + GK blend (gaps + efficiency)
#   - LOW_VOL_TREND (0):   GK-heavy (most efficient)
#   - LOW_VOL_RANGE (2):   Parkinson-heavy (range is informative)
#   - HIGH_VOL_RANGE (3):  GK + Parkinson blend
#
# =============================================================================

# Regime integer codes (from models.regime)
_REGIME_LOW_VOL_TREND = 0
_REGIME_HIGH_VOL_TREND = 1
_REGIME_LOW_VOL_RANGE = 2
_REGIME_HIGH_VOL_RANGE = 3
_REGIME_CRISIS_JUMP = 4

# Regime-dependent weights: [GK, YZ, Parkinson, EWMA]
# Rows ordered by regime integer code 0..4
FUSION_REGIME_WEIGHTS = np.array([
    [0.55, 0.10, 0.20, 0.15],  # 0: LOW_VOL_TREND   -> GK-heavy
    [0.30, 0.40, 0.15, 0.15],  # 1: HIGH_VOL_TREND  -> YZ + GK
    [0.20, 0.10, 0.50, 0.20],  # 2: LOW_VOL_RANGE   -> Parkinson-heavy
    [0.35, 0.20, 0.30, 0.15],  # 3: HIGH_VOL_RANGE  -> GK + Parkinson
    [0.15, 0.55, 0.15, 0.15],  # 4: CRISIS_JUMP     -> YZ-heavy
], dtype=np.float64)

# Default weights when no regime is available
FUSION_DEFAULT_WEIGHTS = np.array([0.40, 0.20, 0.25, 0.15], dtype=np.float64)


@dataclass
class VolFusionResult:
    """Result of multi-estimator volatility fusion."""
    volatility: np.ndarray          # Fused volatility time series
    component_vols: Dict[str, np.ndarray]  # Individual estimator vols
    regime_weights_used: np.ndarray  # (T, 4) weight matrix applied
    method: str = "fusion"


def vol_fusion_kernel(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    returns: np.ndarray,
    regime: Optional[np.ndarray] = None,
    span: int = DEFAULT_VOL_SPAN,
    annualize: bool = False,
) -> VolFusionResult:
    """
    Multi-estimator volatility fusion with regime-adaptive weights.

    Fuses Garman-Klass, Yang-Zhang, Parkinson, and EWMA estimators using
    regime-dependent weighting. Each estimator has different strengths:
      - GK: Most efficient under continuous trading (7.4x)
      - YZ: Robust to overnight gaps (14x, handles earnings/news)
      - Parkinson: Efficient for ranging markets (5.2x, uses H-L)
      - EWMA: Robust baseline, always available (1.0x)

    Parameters
    ----------
    open_, high, low, close : np.ndarray
        OHLC price arrays (same length).
    returns : np.ndarray
        Log returns array (same length as OHLC).
    regime : np.ndarray, optional
        Integer regime labels (0-4). If None, uses default weights.
    span : int
        EWMA smoothing span for each component estimator.
    annualize : bool
        Whether to annualize the output.

    Returns
    -------
    VolFusionResult
        Fused volatility and diagnostics.
    """
    n = len(close)

    # --- Compute each component estimator ---
    # 1. Garman-Klass
    gk_var = _garman_klass_variance(open_, high, low, close)
    gk_var_smooth = _ewma_smooth(gk_var, span)

    # 2. Yang-Zhang (windowed, so has NaN lead-in)
    yz_var = _yang_zhang_variance(open_, high, low, close, window=span)
    # Fill NaN lead-in with GK values
    yz_nan_mask = np.isnan(yz_var)
    yz_var_filled = np.where(yz_nan_mask, gk_var_smooth, yz_var)

    # 3. Parkinson
    pk_var = _parkinson_variance(high, low)
    pk_var_smooth = _ewma_smooth(pk_var, span)

    # 4. EWMA close-to-close
    ewma_var = _ewma_variance_cc(returns, span)

    # Stack: (T, 4) = [GK, YZ, Parkinson, EWMA]
    var_stack = np.column_stack([gk_var_smooth, yz_var_filled,
                                 pk_var_smooth, ewma_var])

    # Replace NaN with MIN_VARIANCE before flooring
    var_stack = np.where(np.isfinite(var_stack), var_stack, MIN_VARIANCE)

    # Floor all variances
    var_stack = np.maximum(var_stack, MIN_VARIANCE)

    # --- Regime-adaptive weights ---
    weights = np.empty((n, 4), dtype=np.float64)
    if regime is not None:
        regime_int = np.asarray(regime, dtype=int)
        for t in range(n):
            r = regime_int[t]
            if 0 <= r <= 4:
                weights[t] = FUSION_REGIME_WEIGHTS[r]
            else:
                weights[t] = FUSION_DEFAULT_WEIGHTS
    else:
        weights[:] = FUSION_DEFAULT_WEIGHTS

    # --- Fuse ---
    fused_var = np.sum(weights * var_stack, axis=1)
    fused_var = np.maximum(fused_var, MIN_VARIANCE)
    fused_vol = np.sqrt(fused_var)

    if annualize:
        fused_vol = fused_vol * np.sqrt(ANNUALIZATION_FACTOR)

    # Component vols for diagnostics
    component_vols = {
        "garman_klass": np.sqrt(np.maximum(gk_var_smooth, MIN_VARIANCE)),
        "yang_zhang": np.sqrt(np.maximum(yz_var_filled, MIN_VARIANCE)),
        "parkinson": np.sqrt(np.maximum(pk_var_smooth, MIN_VARIANCE)),
        "ewma": np.sqrt(np.maximum(ewma_var, MIN_VARIANCE)),
    }

    return VolFusionResult(
        volatility=fused_vol,
        component_vols=component_vols,
        regime_weights_used=weights,
    )


# =============================================================================
# STORY 7.2: HAR-GK HYBRID VOLATILITY WITH ADAPTIVE HORIZON WEIGHTS
# =============================================================================
#
# True HAR-GK: Garman-Klass at EACH horizon (daily, weekly, monthly),
# with OLS-estimated horizon weights from in-sample realized variance.
#
# Unlike standard HAR (Corsi 2009) which uses squared returns at each
# horizon, this computes GK variance rolling means at 1d, 5d, 22d windows.
# The GK base gives 7.4x efficiency gain at each horizon.
#
# OLS regression:
#   RV_target(t+1) = w_d * RV_gk_daily(t) + w_w * RV_gk_weekly(t)
#                   + w_m * RV_gk_monthly(t) + epsilon
#
# Weights are constrained: w >= 0, sum(w) = 1 (convex combination).
# =============================================================================

# HAR-GK constants
HAR_GK_HORIZON_DAILY = 1
HAR_GK_HORIZON_WEEKLY = 5
HAR_GK_HORIZON_MONTHLY = 22

# Default Corsi (2009) weights (used when OLS fails or insufficient data)
HAR_GK_DEFAULT_WEIGHTS = np.array([0.5, 0.3, 0.2], dtype=np.float64)

# Minimum training samples for OLS weight estimation
HAR_GK_MIN_OLS_SAMPLES = 60


@dataclass
class HarGkResult:
    """Result of HAR-GK hybrid volatility estimation."""
    volatility: np.ndarray          # Fused HAR-GK volatility time series
    gk_daily: np.ndarray            # GK variance at daily horizon
    gk_weekly: np.ndarray           # GK variance at weekly horizon (5d rolling)
    gk_monthly: np.ndarray          # GK variance at monthly horizon (22d rolling)
    weights: np.ndarray             # (3,) estimated or default weights [w_d, w_w, w_m]
    weights_method: str             # "ols" or "default"
    efficiency_vs_cc: float         # Theoretical efficiency gain vs close-to-close HAR


def _compute_gk_horizon_components(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Garman-Klass variance at daily, weekly, and monthly horizons.

    Returns rolling means of daily GK variance at each horizon window,
    providing multi-horizon memory with 7.4x efficiency at each level.

    Parameters
    ----------
    open_, high, low, close : np.ndarray
        OHLC price arrays (same length).

    Returns
    -------
    (gk_daily, gk_weekly, gk_monthly) : Tuple of np.ndarray
        GK variance at each horizon. Shape (n,) each.
    """
    n = len(close)
    gk_var = _garman_klass_variance(open_, high, low, close)

    # Daily: point-in-time GK variance
    gk_daily = np.copy(gk_var)

    # Weekly: 5-day rolling mean of daily GK variance
    gk_weekly = np.full(n, np.nan)
    for t in range(HAR_GK_HORIZON_WEEKLY, n):
        gk_weekly[t] = np.mean(gk_var[t - HAR_GK_HORIZON_WEEKLY + 1 : t + 1])
    # Fill early values with expanding mean
    for t in range(1, min(HAR_GK_HORIZON_WEEKLY, n)):
        gk_weekly[t] = np.mean(gk_var[1 : t + 1]) if t > 0 else gk_var[t]

    # Monthly: 22-day rolling mean of daily GK variance
    gk_monthly = np.full(n, np.nan)
    for t in range(HAR_GK_HORIZON_MONTHLY, n):
        gk_monthly[t] = np.mean(gk_var[t - HAR_GK_HORIZON_MONTHLY + 1 : t + 1])
    # Fill early values with expanding mean
    for t in range(1, min(HAR_GK_HORIZON_MONTHLY, n)):
        gk_monthly[t] = np.mean(gk_var[1 : t + 1]) if t > 0 else gk_var[t]

    # Replace NaN with daily value (graceful fallback)
    gk_weekly = np.where(np.isfinite(gk_weekly), gk_weekly, gk_daily)
    gk_monthly = np.where(np.isfinite(gk_monthly), gk_monthly, gk_daily)

    # Floor all components
    gk_daily = np.maximum(gk_daily, MIN_VARIANCE)
    gk_weekly = np.maximum(gk_weekly, MIN_VARIANCE)
    gk_monthly = np.maximum(gk_monthly, MIN_VARIANCE)

    return gk_daily, gk_weekly, gk_monthly


def _estimate_har_weights_ols(
    gk_daily: np.ndarray,
    gk_weekly: np.ndarray,
    gk_monthly: np.ndarray,
    close: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """
    Estimate HAR horizon weights via constrained OLS.

    Regresses next-day realized variance (squared log return) on the three
    GK horizon components. Weights are projected onto the simplex
    (non-negative, sum to 1) for convexity.

    Parameters
    ----------
    gk_daily, gk_weekly, gk_monthly : np.ndarray
        GK variance at each horizon.
    close : np.ndarray
        Close prices for computing realized variance target.

    Returns
    -------
    (weights, success) : Tuple[np.ndarray, bool]
        (3,) weight vector and whether OLS succeeded.
    """
    n = len(close)
    if n < HAR_GK_MIN_OLS_SAMPLES + HAR_GK_HORIZON_MONTHLY + 1:
        return HAR_GK_DEFAULT_WEIGHTS.copy(), False

    # Target: next-day realized variance (squared log return)
    log_ret = np.log(close[1:] / close[:-1])
    rv_target = log_ret ** 2  # length n-1

    # Predictors at time t predict RV at t+1
    # Align: X[t] = [gk_daily[t], gk_weekly[t], gk_monthly[t]], y = rv_target[t] = ret[t+1]^2
    # So X is from index 0..n-2, y is from index 0..n-2
    X = np.column_stack([gk_daily[:-1], gk_weekly[:-1], gk_monthly[:-1]])
    y = rv_target  # length n-1

    # Remove NaN/Inf rows
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if np.sum(valid) < HAR_GK_MIN_OLS_SAMPLES:
        return HAR_GK_DEFAULT_WEIGHTS.copy(), False

    X_valid = X[valid]
    y_valid = y[valid]

    # OLS: w = (X'X)^{-1} X'y (no intercept, weights are proportions)
    try:
        XtX = X_valid.T @ X_valid
        Xty = X_valid.T @ y_valid

        # Add small ridge for numerical stability
        XtX += 1e-10 * np.eye(3)
        w_raw = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return HAR_GK_DEFAULT_WEIGHTS.copy(), False

    # Project onto simplex: non-negative, sum to 1
    w_proj = np.maximum(w_raw, 0.0)
    w_sum = np.sum(w_proj)
    if w_sum < 1e-12:
        return HAR_GK_DEFAULT_WEIGHTS.copy(), False
    w_proj = w_proj / w_sum

    return w_proj, True


def har_gk_hybrid(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    estimate_weights: bool = True,
    weights: Optional[np.ndarray] = None,
    annualize: bool = False,
) -> HarGkResult:
    """
    HAR-GK Hybrid Volatility with Adaptive Horizon Weights.

    Computes Garman-Klass variance at three horizons (daily, weekly, monthly)
    and combines them with either OLS-estimated or fixed weights. This gives
    the multi-horizon memory of HAR (Corsi 2009) with the 7.4x efficiency
    of GK at each horizon, yielding 3-5x efficiency over standard
    close-to-close HAR.

    Formula:
        sigma^2_HAR-GK(t) = w_d * GK_daily(t) + w_w * GK_weekly(t) + w_m * GK_monthly(t)

    Where:
        GK_daily = point-in-time GK variance
        GK_weekly = 5-day rolling mean of GK variance
        GK_monthly = 22-day rolling mean of GK variance

    Parameters
    ----------
    open_, high, low, close : np.ndarray
        OHLC price arrays (same length).
    estimate_weights : bool
        If True, estimate weights via OLS. If False, use default or provided.
    weights : np.ndarray, optional
        (3,) weight vector [w_daily, w_weekly, w_monthly]. Overrides OLS.
    annualize : bool
        Whether to annualize the output.

    Returns
    -------
    HarGkResult
        Volatility time series, horizon components, and weight information.
    """
    n = len(close)

    # Compute GK at each horizon
    gk_daily, gk_weekly, gk_monthly = _compute_gk_horizon_components(
        open_, high, low, close,
    )

    # Determine weights
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = np.maximum(w, 0.0)
        w_sum = np.sum(w)
        if w_sum > 0:
            w = w / w_sum
        else:
            w = HAR_GK_DEFAULT_WEIGHTS.copy()
        method = "explicit"
    elif estimate_weights:
        w, ols_ok = _estimate_har_weights_ols(
            gk_daily, gk_weekly, gk_monthly, close,
        )
        method = "ols" if ols_ok else "default"
    else:
        w = HAR_GK_DEFAULT_WEIGHTS.copy()
        method = "default"

    # Fuse: weighted combination of horizon components
    har_gk_var = w[0] * gk_daily + w[1] * gk_weekly + w[2] * gk_monthly
    har_gk_var = np.maximum(har_gk_var, MIN_VARIANCE)

    # Convert to volatility
    har_gk_vol = np.sqrt(har_gk_var)

    if annualize:
        har_gk_vol = har_gk_vol * np.sqrt(ANNUALIZATION_FACTOR)

    # Theoretical efficiency: GK at each horizon gives ~7.4x vs CC
    # Standard HAR with CC gives ~1x at each horizon
    # HAR-GK gives ~7.4x / 1x = 7.4x over HAR-CC, but after averaging
    # the practical gain is ~3-5x due to correlation between horizons
    efficiency = 5.0  # Conservative practical estimate

    return HarGkResult(
        volatility=har_gk_vol,
        gk_daily=gk_daily,
        gk_weekly=gk_weekly,
        gk_monthly=gk_monthly,
        weights=w,
        weights_method=method,
        efficiency_vs_cc=efficiency,
    )


# =============================================================================
# STORY 7.3: OVERNIGHT GAP DETECTOR AND VOL ADJUSTMENT
# =============================================================================
#
# Detects overnight gaps (|log(O_t/C_{t-1})| > 2*sigma) and inflates
# variance + filter uncertainty on gap days.
#
# Why this matters:
#   - Stocks gap 2-5% on earnings/news (UPST, NVDA, AFRM)
#   - GK and Parkinson estimators ignore overnight component
#   - Without gap detection, Kalman filter treats a 4% gap as gradual drift
#   - This causes: biased mu, too-tight P_t, overconfident signals
#
# On gap days:
#   sigma_t^2 += gap^2 / 4   (gap variance inflation)
#   P_t *= gap_p_inflate       (honest uncertainty increase)
# =============================================================================

# Gap detection configuration
GAP_THRESHOLD_SIGMA = 2.0     # Flag gap if |gap| > 2 * trailing vol
GAP_TRAILING_WINDOW = 20      # 20-day trailing vol for threshold
GAP_VAR_FRACTION = 0.25       # Add gap^2 * this to variance (1/4 of gap variance)
GAP_P_INFLATE_FACTOR = 2.0    # Multiply P_t by this on gap days


@dataclass
class GapDetectionResult:
    """Result of overnight gap detection."""
    is_gap: np.ndarray          # (T,) bool: True if gap detected
    gap_return: np.ndarray      # (T,) float: log(O_t / C_{t-1})
    gap_magnitude: np.ndarray   # (T,) float: |gap_return|
    threshold: np.ndarray       # (T,) float: threshold used at each t
    n_gaps: int                 # Total number of gap days detected
    gap_fraction: float         # Fraction of days with gaps


def detect_overnight_gap(
    open_: np.ndarray,
    close: np.ndarray,
    vol: Optional[np.ndarray] = None,
    threshold_sigma: float = GAP_THRESHOLD_SIGMA,
    trailing_window: int = GAP_TRAILING_WINDOW,
) -> GapDetectionResult:
    """
    Detect overnight price gaps.

    A gap is flagged when |log(O_t / C_{t-1})| > threshold_sigma * sigma_t,
    where sigma_t is the trailing close-to-close volatility.

    Parameters
    ----------
    open_ : np.ndarray
        Open prices, shape (T,).
    close : np.ndarray
        Close prices, shape (T,).
    vol : np.ndarray, optional
        Pre-computed volatility for threshold. If None, computed from
        trailing close-to-close returns.
    threshold_sigma : float
        Number of sigmas for gap threshold (default 2.0).
    trailing_window : int
        Window for trailing vol computation (default 20).

    Returns
    -------
    GapDetectionResult
        Gap flags, magnitudes, and diagnostics.
    """
    n = len(close)

    # Overnight gap return: log(O_t / C_{t-1})
    gap_return = np.zeros(n)
    if n > 1:
        # Protect against zero/negative prices
        safe_close_prev = np.maximum(close[:-1], 1e-10)
        safe_open = np.maximum(open_[1:], 1e-10)
        gap_return[1:] = np.log(safe_open / safe_close_prev)

    gap_magnitude = np.abs(gap_return)

    # Compute trailing volatility for threshold
    if vol is not None:
        trailing_vol = np.copy(vol)
    else:
        # Use trailing standard deviation of close-to-close returns
        log_ret = np.zeros(n)
        if n > 1:
            log_ret[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-10))

        trailing_vol = np.full(n, np.nan)
        for t in range(trailing_window, n):
            window = log_ret[t - trailing_window + 1 : t + 1]
            trailing_vol[t] = np.std(window)
        # Fill early values with expanding std
        for t in range(2, min(trailing_window, n)):
            trailing_vol[t] = np.std(log_ret[1 : t + 1])
        # First two points: use a conservative default
        if n >= 2:
            trailing_vol[0] = 0.01
            trailing_vol[1] = 0.01

    # Ensure trailing_vol is finite and positive
    trailing_vol = np.where(
        np.isfinite(trailing_vol) & (trailing_vol > 0),
        trailing_vol,
        0.01,
    )

    # Threshold: gap must exceed threshold_sigma * trailing_vol
    threshold = threshold_sigma * trailing_vol

    # Detect gaps
    is_gap = gap_magnitude > threshold
    # First point can never be a gap (no prior close)
    is_gap[0] = False

    n_gaps = int(np.sum(is_gap))
    gap_frac = n_gaps / max(n - 1, 1)  # Exclude first point

    return GapDetectionResult(
        is_gap=is_gap,
        gap_return=gap_return,
        gap_magnitude=gap_magnitude,
        threshold=threshold,
        n_gaps=n_gaps,
        gap_fraction=gap_frac,
    )


def adjust_vol_for_gaps(
    variance: np.ndarray,
    gap_result: GapDetectionResult,
    gap_var_fraction: float = GAP_VAR_FRACTION,
) -> np.ndarray:
    """
    Inflate variance on gap days by adding a fraction of the gap variance.

    On gap days:
        sigma_t^2 += gap_var_fraction * gap_return_t^2

    This accounts for the overnight price movement that range-based
    estimators (GK, Parkinson) miss entirely.

    Parameters
    ----------
    variance : np.ndarray
        Input variance time series, shape (T,).
    gap_result : GapDetectionResult
        Output from detect_overnight_gap().
    gap_var_fraction : float
        Fraction of gap^2 to add (default 0.25 = gap^2/4).

    Returns
    -------
    np.ndarray
        Adjusted variance with gap inflation.
    """
    adjusted = np.copy(variance)
    gap_days = gap_result.is_gap
    adjusted[gap_days] += gap_var_fraction * gap_result.gap_return[gap_days] ** 2
    adjusted = np.maximum(adjusted, MIN_VARIANCE)
    return adjusted


def adjust_filter_P_for_gaps(
    P: np.ndarray,
    gap_result: GapDetectionResult,
    inflate_factor: float = GAP_P_INFLATE_FACTOR,
) -> np.ndarray:
    """
    Inflate Kalman filter state uncertainty P on gap days.

    On gap days:
        P_t *= inflate_factor

    This makes the filter honestly uncertain after an overnight gap,
    preventing overconfident signals following earnings/news events.

    Parameters
    ----------
    P : np.ndarray
        Filter state uncertainty, shape (T,).
    gap_result : GapDetectionResult
        Output from detect_overnight_gap().
    inflate_factor : float
        Multiplicative inflation factor (default 2.0).

    Returns
    -------
    np.ndarray
        Adjusted P with gap inflation.
    """
    adjusted = np.copy(P)
    adjusted[gap_result.is_gap] *= inflate_factor
    return adjusted
