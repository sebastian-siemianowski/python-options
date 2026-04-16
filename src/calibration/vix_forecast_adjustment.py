#!/usr/bin/env python3
"""
===============================================================================
VIX-INTEGRATED FORECAST ADJUSTMENT
===============================================================================

Epic 23: VIX-Integrated Forecast Adjustment

VIX is the strongest leading indicator for equity tail risk. This module
provides three layers of VIX integration beyond the existing nu adjustment:

1. VIX-CONDITIONAL DRIFT ADJUSTMENT (Story 23.1)
   High VIX correlates with negative drift (fear). Dampens bullish drift
   expectations when VIX is elevated to avoid extrapolating into selloffs.

2. VIX TERM STRUCTURE for horizon-dependent vol (Story 23.2)
   Interpolates implied vol from VIX (30d) and VIX3M (90d) at any horizon.
   7-day forecasts use 7-day implied vol, not just 30-day VIX.

3. CORRELATION SPIKE DETECTION (Story 23.3)
   When VIX spikes, cross-asset correlations increase ("correlation spike").
   Detects these episodes and inflates portfolio vol accordingly.

REFERENCES:
   Whaley, R. (2000). "The Investor Fear Gauge"
   Drechsler, I. & Yaron, A. (2011). "What's Vol Got to Do with It"
   Longin, F. & Solnik, B. (2001). "Extreme Correlation of International
       Equity Markets"

===============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# VIX drift adjustment (Story 23.1)
VIX_DRIFT_THRESHOLD_LOW = 15.0     # Below this: no adjustment
VIX_DRIFT_THRESHOLD_HIGH = 25.0    # Above this: dampening begins
VIX_DRIFT_EXTREME = 35.0           # Above this: max dampening
VIX_DRIFT_MAX_DAMPENING = 0.70     # Max dampening (70% reduction)
VIX_DRIFT_SLOPE = 0.3              # Dampening slope per VIX unit above threshold
VIX_MEDIAN_DEFAULT = 18.0          # Historical VIX median

# VIX term structure (Story 23.2)
VIX_30D_HORIZON = 30               # VIX measures 30-day implied vol
VIX_90D_HORIZON = 90               # VIX3M measures 90-day implied vol
TRADING_DAYS_PER_YEAR = 252

# Correlation spike detection (Story 23.3)
CORRELATION_SPIKE_THRESHOLD = 0.50  # Avg pairwise corr to flag spike
CORRELATION_LOOKBACK = 21           # Days for rolling correlation
CORRELATION_MIN_ASSETS = 3          # Minimum assets for correlation
VIX_SPIKE_THRESHOLD = 25.0         # VIX level to start checking for spikes


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VIXDriftAdjustmentResult:
    """Result of VIX-conditional drift adjustment."""
    mu_adjusted: np.ndarray       # Adjusted drift (same shape as input)
    mu_original: np.ndarray       # Original drift
    vix_current: float            # VIX value used
    dampening_factor: float       # Multiplicative factor applied (1.0 = no change)
    adjustment_applied: bool      # Whether any adjustment was made
    regime: str                   # "low_fear", "elevated", "extreme"
    
    def to_dict(self) -> Dict:
        return {
            "vix_current": self.vix_current,
            "dampening_factor": self.dampening_factor,
            "adjustment_applied": self.adjustment_applied,
            "regime": self.regime,
        }


@dataclass
class VIXTermStructureResult:
    """Result of VIX term structure interpolation."""
    implied_vol: float            # Interpolated implied vol at target horizon
    horizon_days: int             # Target horizon in days
    vix_30: float                 # 30-day VIX input
    vix_90: float                 # 90-day VIX3M input
    term_structure_state: str     # "contango", "backwardation", or "flat"
    daily_vol: float              # Annualized vol converted to daily
    
    def to_dict(self) -> Dict:
        return {
            "implied_vol": self.implied_vol,
            "horizon_days": self.horizon_days,
            "vix_30": self.vix_30,
            "vix_90": self.vix_90,
            "term_structure_state": self.term_structure_state,
            "daily_vol": self.daily_vol,
        }


@dataclass
class CorrelationSpikeResult:
    """Result of correlation spike detection."""
    is_spike: bool                # True if correlation spike detected
    avg_pairwise_correlation: float  # Average pairwise correlation
    vix_triggered: bool           # Whether VIX level contributed to flag
    portfolio_vol_inflation: float  # Multiplicative vol inflation factor
    n_assets: int                 # Number of assets analyzed
    correlation_matrix: Optional[np.ndarray] = None  # Full pairwise correlation matrix
    
    def to_dict(self) -> Dict:
        return {
            "is_spike": self.is_spike,
            "avg_pairwise_correlation": self.avg_pairwise_correlation,
            "vix_triggered": self.vix_triggered,
            "portfolio_vol_inflation": self.portfolio_vol_inflation,
            "n_assets": self.n_assets,
        }


# =============================================================================
# STORY 23.1: VIX-Conditional Drift Adjustment
# =============================================================================

def vix_drift_adjustment(
    mu_t: np.ndarray,
    vix_current: float,
    vix_median: float = VIX_MEDIAN_DEFAULT,
) -> VIXDriftAdjustmentResult:
    """
    Adjust Kalman drift estimate based on current VIX level.
    
    When VIX is elevated, bullish drift is dampened to avoid extrapolating
    positive trends into fear-driven selloffs.
    
    Rules:
        VIX < 15:  No adjustment (low fear, let signal through)
        VIX 15-25: No adjustment (normal range)
        VIX 25-35: mu_adj = mu * (1 - 0.3 * (VIX - 25) / 25)  [linear dampening]
        VIX > 35:  mu_adj = mu * 0.3  [70% dampening, fear dominates]
    
    Note: Only BULLISH (positive) drift is dampened. Bearish signals pass through
    unchanged since VIX confirms the bearish thesis.
    
    Args:
        mu_t: Drift estimate(s) - scalar or array
        vix_current: Current VIX level
        vix_median: Historical VIX median (for reference, not directly used in formula)
        
    Returns:
        VIXDriftAdjustmentResult with adjusted drift and diagnostics
    """
    mu_t = np.asarray(mu_t, dtype=np.float64)
    original_shape = mu_t.shape
    mu_flat = mu_t.ravel()
    
    if not np.isfinite(vix_current) or vix_current < 0:
        return VIXDriftAdjustmentResult(
            mu_adjusted=mu_t.copy(),
            mu_original=mu_t.copy(),
            vix_current=float(vix_current) if np.isfinite(vix_current) else 0.0,
            dampening_factor=1.0,
            adjustment_applied=False,
            regime="unknown",
        )
    
    # Determine regime and dampening factor
    if vix_current < VIX_DRIFT_THRESHOLD_HIGH:
        # Low or normal VIX: no adjustment
        dampening = 1.0
        regime = "low_fear"
    elif vix_current >= VIX_DRIFT_EXTREME:
        # Extreme VIX: maximum dampening
        dampening = 1.0 - VIX_DRIFT_MAX_DAMPENING
        regime = "extreme"
    else:
        # Elevated VIX: linear dampening
        # dampening = 1 - 0.3 * (VIX - 25) / 25
        vix_excess = vix_current - VIX_DRIFT_THRESHOLD_HIGH
        vix_range = VIX_DRIFT_THRESHOLD_HIGH  # 25
        dampening = 1.0 - VIX_DRIFT_SLOPE * vix_excess / vix_range
        dampening = max(1.0 - VIX_DRIFT_MAX_DAMPENING, dampening)
        regime = "elevated"
    
    # Apply dampening only to positive (bullish) drift
    mu_adjusted = mu_flat.copy()
    positive_mask = mu_flat > 0
    mu_adjusted[positive_mask] = mu_flat[positive_mask] * dampening
    
    mu_adjusted = mu_adjusted.reshape(original_shape)
    
    return VIXDriftAdjustmentResult(
        mu_adjusted=mu_adjusted,
        mu_original=mu_t.copy(),
        vix_current=float(vix_current),
        dampening_factor=float(dampening),
        adjustment_applied=dampening < 1.0,
        regime=regime,
    )


def vix_drift_adjustment_series(
    mu_series: np.ndarray,
    vix_series: np.ndarray,
    vix_median: float = VIX_MEDIAN_DEFAULT,
) -> np.ndarray:
    """
    Apply VIX drift adjustment to a time series of drift estimates.
    
    Each mu_t is adjusted based on the corresponding vix_t.
    
    Args:
        mu_series: (T,) array of drift estimates
        vix_series: (T,) array of VIX values
        vix_median: Historical VIX median
        
    Returns:
        (T,) array of adjusted drifts
    """
    mu_series = np.asarray(mu_series, dtype=np.float64).ravel()
    vix_series = np.asarray(vix_series, dtype=np.float64).ravel()
    
    T = len(mu_series)
    if len(vix_series) != T:
        raise ValueError(f"Length mismatch: mu={T}, vix={len(vix_series)}")
    
    adjusted = np.empty(T, dtype=np.float64)
    for t in range(T):
        result = vix_drift_adjustment(
            np.array([mu_series[t]]),
            vix_series[t],
            vix_median,
        )
        adjusted[t] = result.mu_adjusted[0]
    
    return adjusted


# =============================================================================
# STORY 23.2: VIX Term Structure for Horizon-Dependent Vol
# =============================================================================

def vix_term_structure_vol(
    vix_30: float,
    vix_90: float,
    horizon: int,
) -> VIXTermStructureResult:
    """
    Interpolate implied volatility at arbitrary horizon using VIX term structure.
    
    Uses linear interpolation in variance space (variance scales linearly with time):
        sigma^2(H) = sigma^2(30) + (sigma^2(90) - sigma^2(30)) * (H - 30) / (90 - 30)
    
    For H < 30: extrapolate using the slope (near-term is more responsive to stress)
    For H > 90: extrapolate with dampening (uncertainty about long-term vol)
    
    Term structure states:
        Contango: VIX3M > VIX (normal, vol increasing with horizon)
        Backwardation: VIX3M < VIX (stress, near-term risk highest)
    
    Args:
        vix_30: 30-day VIX (annualized percentage, e.g., 20 for 20%)
        vix_90: 90-day VIX3M (annualized percentage)
        horizon: Target horizon in calendar days
        
    Returns:
        VIXTermStructureResult with interpolated vol and diagnostics
    """
    if not (np.isfinite(vix_30) and np.isfinite(vix_90)):
        return VIXTermStructureResult(
            implied_vol=float(vix_30) if np.isfinite(vix_30) else 20.0,
            horizon_days=horizon,
            vix_30=float(vix_30) if np.isfinite(vix_30) else 20.0,
            vix_90=float(vix_90) if np.isfinite(vix_90) else 20.0,
            term_structure_state="unknown",
            daily_vol=0.0,
        )
    
    if vix_30 <= 0 or vix_90 <= 0:
        vix_30 = max(vix_30, 1.0)
        vix_90 = max(vix_90, 1.0)
    
    # Determine term structure state
    if abs(vix_90 - vix_30) < 0.5:
        state = "flat"
    elif vix_90 > vix_30:
        state = "contango"
    else:
        state = "backwardation"
    
    if horizon <= 0:
        horizon = 1
    
    # Interpolate in VARIANCE-PER-DAY space (not total variance).
    # This directly gives annualized vol at each horizon:
    #   implied_vol = sqrt(vpd * 252) * 100
    #
    # In contango (VIX3M > VIX): vpd increases with horizon -> vol increases
    # In backwardation (VIX3M < VIX): vpd decreases -> near-term vol highest
    
    vpd_30 = (vix_30 / 100.0) ** 2 / TRADING_DAYS_PER_YEAR
    vpd_90 = (vix_90 / 100.0) ** 2 / TRADING_DAYS_PER_YEAR
    
    slope = (vpd_90 - vpd_30) / (VIX_90D_HORIZON - VIX_30D_HORIZON)
    
    if VIX_30D_HORIZON <= horizon <= VIX_90D_HORIZON:
        # Linear interpolation between 30d and 90d
        vpd = vpd_30 + slope * (horizon - VIX_30D_HORIZON)
    elif horizon < VIX_30D_HORIZON:
        # Extrapolate below 30d using same slope
        vpd = vpd_30 + slope * (horizon - VIX_30D_HORIZON)
        # Floor: vpd cannot go below 50% of the lower anchor
        vpd = max(vpd, min(vpd_30, vpd_90) * 0.5)
    else:
        # Extrapolate above 90d with dampened slope
        dampening = 0.7
        vpd = vpd_90 + dampening * slope * (horizon - VIX_90D_HORIZON)
        # Floor: vpd stays at least 50% of lower anchor
        vpd = max(vpd, min(vpd_30, vpd_90) * 0.5)
    
    # Convert variance-per-day to annualized vol (%)
    var_daily_H = max(vpd, 1e-12)
    sigma_annual = np.sqrt(var_daily_H * TRADING_DAYS_PER_YEAR)
    implied_vol = sigma_annual * 100.0
    
    # Daily vol (for Kalman filter use)
    daily_vol = np.sqrt(var_daily_H)
    
    return VIXTermStructureResult(
        implied_vol=float(implied_vol),
        horizon_days=horizon,
        vix_30=float(vix_30),
        vix_90=float(vix_90),
        term_structure_state=state,
        daily_vol=float(daily_vol),
    )


# =============================================================================
# STORY 23.3: Correlation Spike Detection for Portfolio-Level Risk
# =============================================================================

def detect_correlation_spike(
    returns_matrix: np.ndarray,
    vix: float,
    threshold: float = CORRELATION_SPIKE_THRESHOLD,
    lookback: int = CORRELATION_LOOKBACK,
) -> CorrelationSpikeResult:
    """
    Detect correlation spikes: episodes where all assets move together.
    
    During VIX spikes, cross-asset correlations increase dramatically,
    destroying portfolio diversification benefits.
    
    Detection logic:
    1. Compute rolling pairwise correlation over lookback window
    2. If avg pairwise correlation > threshold AND VIX elevated -> SPIKE
    3. Compute portfolio vol inflation factor
    
    Vol inflation during spike:
        inflation = 1 + avg_corr * sqrt(n_assets)
    
    Args:
        returns_matrix: (T, N) matrix of asset returns
        vix: Current VIX level
        threshold: Correlation threshold for spike flag (default: 0.50)
        lookback: Rolling window for correlation (default: 21 days)
        
    Returns:
        CorrelationSpikeResult with spike flag and diagnostics
    """
    returns_matrix = np.asarray(returns_matrix, dtype=np.float64)
    
    if returns_matrix.ndim != 2:
        raise ValueError(f"Expected 2D returns matrix, got {returns_matrix.ndim}D")
    
    T, N = returns_matrix.shape
    
    if N < CORRELATION_MIN_ASSETS:
        return CorrelationSpikeResult(
            is_spike=False,
            avg_pairwise_correlation=0.0,
            vix_triggered=False,
            portfolio_vol_inflation=1.0,
            n_assets=N,
        )
    
    # Use last `lookback` days for rolling correlation
    if T >= lookback:
        window = returns_matrix[-lookback:]
    else:
        window = returns_matrix
    
    # Handle NaN: replace with zero
    window = np.nan_to_num(window, nan=0.0)
    
    # Compute pairwise correlation matrix
    corr_matrix = np.corrcoef(window.T)
    
    # Handle NaN in correlation matrix (e.g., zero-variance columns)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pairwise_corrs = corr_matrix[mask]
    
    if len(pairwise_corrs) == 0:
        avg_corr = 0.0
    else:
        avg_corr = float(np.mean(pairwise_corrs))
    
    # VIX trigger check
    vix_triggered = np.isfinite(vix) and vix >= VIX_SPIKE_THRESHOLD
    
    # Spike detection: high correlation AND elevated VIX
    is_spike = avg_corr > threshold and vix_triggered
    
    # Also flag if correlation alone is very high (even without VIX data)
    if avg_corr > threshold + 0.2:  # Very high correlation regardless of VIX
        is_spike = True
    
    # Portfolio vol inflation factor
    if is_spike:
        # inflation = 1 + avg_corr * sqrt(n_assets)
        inflation = 1.0 + avg_corr * np.sqrt(N)
    else:
        inflation = 1.0
    
    return CorrelationSpikeResult(
        is_spike=is_spike,
        avg_pairwise_correlation=float(avg_corr),
        vix_triggered=vix_triggered,
        portfolio_vol_inflation=float(inflation),
        n_assets=N,
        correlation_matrix=corr_matrix,
    )


def position_size_during_spike(
    base_size: float,
    spike_result: CorrelationSpikeResult,
) -> float:
    """
    Reduce position size during correlation spike.
    
    Position is inversely proportional to the vol inflation factor:
        adjusted_size = base_size / inflation
    
    Args:
        base_size: Normal position size (e.g., 1.0 for 100%)
        spike_result: Result from detect_correlation_spike
        
    Returns:
        Adjusted position size
    """
    if not spike_result.is_spike:
        return base_size
    
    return base_size / spike_result.portfolio_vol_inflation


def rolling_correlation_spike_detection(
    returns_matrix: np.ndarray,
    vix_series: np.ndarray,
    threshold: float = CORRELATION_SPIKE_THRESHOLD,
    lookback: int = CORRELATION_LOOKBACK,
) -> List[CorrelationSpikeResult]:
    """
    Detect correlation spikes over a rolling time series.
    
    Args:
        returns_matrix: (T, N) matrix of asset returns
        vix_series: (T,) VIX values
        threshold: Correlation threshold
        lookback: Rolling window size
        
    Returns:
        List of CorrelationSpikeResult, one per time step from lookback onwards
    """
    returns_matrix = np.asarray(returns_matrix, dtype=np.float64)
    vix_series = np.asarray(vix_series, dtype=np.float64).ravel()
    
    T, N = returns_matrix.shape
    if len(vix_series) != T:
        raise ValueError(f"Length mismatch: returns T={T}, vix={len(vix_series)}")
    
    results = []
    for t in range(lookback, T):
        window = returns_matrix[t - lookback:t]
        vix_t = vix_series[t]
        result = detect_correlation_spike(window, vix_t, threshold, lookback)
        results.append(result)
    
    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    "VIXDriftAdjustmentResult",
    "VIXTermStructureResult",
    "CorrelationSpikeResult",
    # Story 23.1
    "vix_drift_adjustment",
    "vix_drift_adjustment_series",
    # Story 23.2
    "vix_term_structure_vol",
    # Story 23.3
    "detect_correlation_spike",
    "position_size_during_spike",
    "rolling_correlation_spike_detection",
    # Constants
    "VIX_DRIFT_THRESHOLD_LOW",
    "VIX_DRIFT_THRESHOLD_HIGH",
    "VIX_DRIFT_EXTREME",
    "VIX_DRIFT_MAX_DAMPENING",
    "VIX_MEDIAN_DEFAULT",
    "CORRELATION_SPIKE_THRESHOLD",
]
