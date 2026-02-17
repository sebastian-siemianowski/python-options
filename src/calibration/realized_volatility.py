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
MIN_VARIANCE = 1e-12


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
