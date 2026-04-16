"""
Epic 15: Regime-Aware Position Sizing
======================================

Story 15.1: Regime-Specific Position Limits
Story 15.2: Dynamic Leverage via Forecast Confidence
Story 15.3: Volatility-Targeting Overlay

Position sizing that respects regime risk, leverages high-confidence
periods, and maintains consistent portfolio volatility.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Story 15.1: Regime-Specific Position Limits
# ---------------------------------------------------------------------------

# Regime names (must match tune.py / signals.py regime classification)
LOW_VOL_TREND = "LOW_VOL_TREND"
HIGH_VOL_TREND = "HIGH_VOL_TREND"
LOW_VOL_RANGE = "LOW_VOL_RANGE"
HIGH_VOL_RANGE = "HIGH_VOL_RANGE"
CRISIS_JUMP = "CRISIS_JUMP"

# Maximum position fractions per regime
REGIME_LIMITS: Dict[str, float] = {
    LOW_VOL_TREND: 0.80,   # Best conditions
    HIGH_VOL_TREND: 0.50,  # Direction exists but noisy
    LOW_VOL_RANGE: 0.20,   # Weak signals
    HIGH_VOL_RANGE: 0.10,  # Avoid
    CRISIS_JUMP: 0.05,     # Tail risk dominates
}


@dataclass
class RegimePositionResult:
    """Result of regime-specific position limiting."""
    regime: str
    max_fraction: float       # Maximum allowed position fraction
    raw_fraction: float       # Original requested fraction
    limited_fraction: float   # Fraction after regime limit applied
    was_limited: bool         # Whether the limit was binding


def regime_position_limit(
    regime: str,
    raw_fraction: float,
    custom_limits: Optional[Dict[str, float]] = None,
) -> RegimePositionResult:
    """
    Apply regime-specific position limit.

    Parameters
    ----------
    regime : str
        Current regime label.
    raw_fraction : float
        Desired position fraction (can be negative for short).
    custom_limits : dict or None
        Override default limits.

    Returns
    -------
    RegimePositionResult
        Limited position fraction with metadata.
    """
    limits = custom_limits if custom_limits is not None else REGIME_LIMITS
    max_frac = limits.get(regime, 0.20)  # Default to conservative if unknown

    sign = np.sign(raw_fraction) if raw_fraction != 0 else 1.0
    abs_frac = abs(raw_fraction)

    if abs_frac > max_frac:
        limited = sign * max_frac
        was_limited = True
    else:
        limited = raw_fraction
        was_limited = False

    return RegimePositionResult(
        regime=regime,
        max_fraction=max_frac,
        raw_fraction=raw_fraction,
        limited_fraction=float(limited),
        was_limited=was_limited,
    )


def regime_position_limit_array(
    regimes: list,
    fractions: np.ndarray,
    custom_limits: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Apply regime-specific position limits to an array.

    Parameters
    ----------
    regimes : list of str, length T
        Regime labels per timestep.
    fractions : ndarray, shape (T,)
        Raw position fractions.
    custom_limits : dict or None
        Override default limits.

    Returns
    -------
    ndarray, shape (T,)
        Limited position fractions.
    """
    fractions = np.asarray(fractions, dtype=float)
    n = len(fractions)
    limited = np.zeros(n)

    for t in range(n):
        result = regime_position_limit(regimes[t], fractions[t], custom_limits)
        limited[t] = result.limited_fraction

    return limited


# ---------------------------------------------------------------------------
# Story 15.2: Dynamic Leverage via Forecast Confidence
# ---------------------------------------------------------------------------

# Default parameters
DEFAULT_CONF_THRESHOLD = 0.55
DEFAULT_LEVERAGE_K = 2.0
DEFAULT_LEVERAGE_MAX = 1.5


@dataclass
class LeverageResult:
    """Result of dynamic leverage computation."""
    leverage: float           # Applied leverage multiplier
    confidence: float         # Input confidence
    is_leveraged: bool        # Whether leverage > 1.0
    effective_fraction: float # Position fraction after leverage


def dynamic_leverage(
    confidence: float,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    k: float = DEFAULT_LEVERAGE_K,
    L_max: float = DEFAULT_LEVERAGE_MAX,
) -> float:
    """
    Compute dynamic leverage from forecast confidence.

    L = 1 + k * (conf - conf_threshold) when conf > conf_threshold
    L = 1.0 otherwise
    Capped at L_max.

    Parameters
    ----------
    confidence : float
        Forecast confidence in [0, 1].
    conf_threshold : float
        Minimum confidence for leverage.
    k : float
        Leverage sensitivity.
    L_max : float
        Maximum leverage.

    Returns
    -------
    float
        Leverage multiplier >= 1.0.
    """
    if not np.isfinite(confidence):
        return 1.0
    if confidence <= conf_threshold:
        return 1.0
    L = 1.0 + k * (confidence - conf_threshold)
    return float(min(L, L_max))


def apply_dynamic_leverage(
    fraction: float,
    confidence: float,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    k: float = DEFAULT_LEVERAGE_K,
    L_max: float = DEFAULT_LEVERAGE_MAX,
) -> LeverageResult:
    """
    Apply dynamic leverage to a position fraction.

    Parameters
    ----------
    fraction : float
        Base position fraction.
    confidence : float
        Forecast confidence.
    conf_threshold, k, L_max : float
        Leverage parameters.

    Returns
    -------
    LeverageResult
        Leveraged position with metadata.
    """
    lev = dynamic_leverage(confidence, conf_threshold, k, L_max)
    effective = fraction * lev

    return LeverageResult(
        leverage=lev,
        confidence=confidence,
        is_leveraged=lev > 1.0,
        effective_fraction=float(effective),
    )


def dynamic_leverage_array(
    fractions: np.ndarray,
    confidences: np.ndarray,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    k: float = DEFAULT_LEVERAGE_K,
    L_max: float = DEFAULT_LEVERAGE_MAX,
) -> np.ndarray:
    """
    Apply dynamic leverage to position fraction arrays.

    Parameters
    ----------
    fractions : ndarray, shape (T,)
        Base position fractions.
    confidences : ndarray, shape (T,)
        Forecast confidences.

    Returns
    -------
    ndarray, shape (T,)
        Leveraged position fractions.
    """
    fractions = np.asarray(fractions, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    n = len(fractions)

    result = np.zeros(n)
    for t in range(n):
        lev = dynamic_leverage(confidences[t], conf_threshold, k, L_max)
        result[t] = fractions[t] * lev

    return result


# ---------------------------------------------------------------------------
# Story 15.3: Volatility-Targeting Overlay
# ---------------------------------------------------------------------------

# Default volatility target (annualized)
DEFAULT_VOL_TARGET = 0.15
# Minimum weight to avoid complete elimination
MIN_VOL_WEIGHT = 0.05
# Maximum weight to prevent excessive leverage
MAX_VOL_WEIGHT = 3.0


@dataclass
class VolTargetResult:
    """Result of volatility targeting."""
    weight: float              # Volatility-targeting weight
    sigma_asset: float         # Asset's annualized volatility
    sigma_target: float        # Target annualized volatility
    was_capped: bool           # Whether weight was capped at MAX_VOL_WEIGHT
    was_floored: bool          # Whether weight was floored at MIN_VOL_WEIGHT


def vol_target_weight(
    sigma_asset: float,
    sigma_target: float = DEFAULT_VOL_TARGET,
) -> VolTargetResult:
    """
    Compute volatility-targeting weight.

    w = sigma_target / sigma_asset

    Parameters
    ----------
    sigma_asset : float
        Asset's annualized volatility.
    sigma_target : float
        Target annualized volatility (default 15%).

    Returns
    -------
    VolTargetResult
        Weight with metadata.
    """
    if sigma_asset <= 0 or not np.isfinite(sigma_asset):
        return VolTargetResult(
            weight=MIN_VOL_WEIGHT, sigma_asset=sigma_asset,
            sigma_target=sigma_target, was_capped=False, was_floored=True,
        )

    w = sigma_target / sigma_asset

    was_capped = w > MAX_VOL_WEIGHT
    was_floored = w < MIN_VOL_WEIGHT

    w = float(np.clip(w, MIN_VOL_WEIGHT, MAX_VOL_WEIGHT))

    return VolTargetResult(
        weight=w, sigma_asset=sigma_asset,
        sigma_target=sigma_target, was_capped=was_capped, was_floored=was_floored,
    )


def vol_target_weight_array(
    sigma_assets: np.ndarray,
    sigma_target: float = DEFAULT_VOL_TARGET,
) -> np.ndarray:
    """
    Compute volatility-targeting weights for multiple assets.

    Parameters
    ----------
    sigma_assets : ndarray, shape (N,)
        Annualized volatilities per asset.
    sigma_target : float
        Target annualized volatility.

    Returns
    -------
    ndarray, shape (N,)
        Volatility-targeting weights.
    """
    sigma_assets = np.asarray(sigma_assets, dtype=float)
    weights = np.zeros(len(sigma_assets))

    for i in range(len(sigma_assets)):
        result = vol_target_weight(sigma_assets[i], sigma_target)
        weights[i] = result.weight

    return weights


def compute_portfolio_vol(
    returns_matrix: np.ndarray,
    weights: np.ndarray,
    annualization: float = 252.0,
) -> float:
    """
    Compute annualized portfolio volatility.

    Parameters
    ----------
    returns_matrix : ndarray, shape (T, N)
        Returns for N assets.
    weights : ndarray, shape (N,)
        Portfolio weights.
    annualization : float
        Annualization factor.

    Returns
    -------
    float
        Annualized portfolio volatility.
    """
    returns_matrix = np.asarray(returns_matrix, dtype=float)
    weights = np.asarray(weights, dtype=float)

    port_returns = returns_matrix @ weights
    if len(port_returns) < 2:
        return 0.0
    return float(np.std(port_returns, ddof=1) * np.sqrt(annualization))
