"""
Story 8.8: Signal Decay and Time-to-Live (TTL).

Implements exponential decay model:
  strength(t) = strength_0 * exp(-lambda * age)
  half_life = horizon * decay_fraction (e.g., 0.5 * horizon)

Usage:
    from decision.signal_decay import compute_signal_ttl, decay_signal, SignalWithTTL
"""
import math
from dataclasses import dataclass
from typing import Optional


# Configuration
DECAY_FRACTION = 0.5        # Half-life = 50% of horizon
REFRESH_THRESHOLD = 0.3     # Recommend refresh when strength < 30%
EXPIRED_THRESHOLD = 0.1     # Signal considered expired below 10%


@dataclass
class SignalWithTTL:
    """Signal with time-to-live metadata."""
    symbol: str
    forecast_pct: float
    confidence: float
    horizon_days: int
    generation_age_days: float   # How old the signal is
    half_life_days: float        # Decay half-life
    current_strength: float      # Decayed strength [0, 1]
    ttl_remaining_days: float    # Estimated days until expired
    needs_refresh: bool
    is_expired: bool


def compute_half_life(
    horizon_days: int,
    decay_fraction: float = DECAY_FRACTION,
) -> float:
    """
    Compute signal half-life from horizon.
    
    Half-life is a fraction of the forecast horizon.
    Longer horizons have longer half-lives.
    
    Args:
        horizon_days: Forecast horizon in days.
        decay_fraction: Fraction of horizon for half-life.
    
    Returns:
        Half-life in days.
    """
    return max(0.5, horizon_days * decay_fraction)


def compute_decay_rate(half_life: float) -> float:
    """Compute exponential decay rate from half-life."""
    if half_life <= 0:
        return float("inf")
    return math.log(2) / half_life


def compute_signal_strength(
    age_days: float,
    half_life: float,
) -> float:
    """
    Compute current signal strength via exponential decay.
    
    strength(t) = exp(-lambda * t)
    
    Args:
        age_days: Days since signal generation.
        half_life: Decay half-life in days.
    
    Returns:
        Current strength [0, 1].
    """
    if age_days <= 0:
        return 1.0
    
    lam = compute_decay_rate(half_life)
    return math.exp(-lam * age_days)


def compute_ttl_remaining(
    current_strength: float,
    half_life: float,
    threshold: float = EXPIRED_THRESHOLD,
) -> float:
    """
    Estimate remaining time until signal expires.
    
    Args:
        current_strength: Current decayed strength [0, 1].
        half_life: Decay half-life.
        threshold: Expiry threshold.
    
    Returns:
        Days remaining. 0 if already expired.
    """
    if current_strength <= threshold:
        return 0.0
    
    if current_strength >= 1.0:
        # Full strength: TTL = time to reach threshold
        lam = compute_decay_rate(half_life)
        if lam <= 0:
            return float("inf")
        return -math.log(threshold) / lam
    
    # General case: time from current to threshold
    lam = compute_decay_rate(half_life)
    if lam <= 0:
        return float("inf")
    return (math.log(current_strength) - math.log(threshold)) / lam


def decay_signal(
    symbol: str,
    forecast_pct: float,
    confidence: float,
    horizon_days: int,
    age_days: float,
) -> SignalWithTTL:
    """
    Apply decay to a signal and compute TTL.
    
    Args:
        symbol: Asset symbol.
        forecast_pct: Original forecast.
        confidence: Original confidence.
        horizon_days: Forecast horizon.
        age_days: Days since generation.
    
    Returns:
        SignalWithTTL with decayed values.
    """
    hl = compute_half_life(horizon_days)
    strength = compute_signal_strength(age_days, hl)
    ttl = compute_ttl_remaining(strength, hl)
    
    return SignalWithTTL(
        symbol=symbol,
        forecast_pct=forecast_pct * strength,
        confidence=confidence * strength,
        horizon_days=horizon_days,
        generation_age_days=age_days,
        half_life_days=hl,
        current_strength=strength,
        ttl_remaining_days=ttl,
        needs_refresh=strength < REFRESH_THRESHOLD,
        is_expired=strength < EXPIRED_THRESHOLD,
    )


def compute_predictive_power_decay(
    realized_returns: list,
    forecast_returns: list,
    ages: list,
    age_bins: Optional[list] = None,
) -> dict:
    """
    Historical analysis: forecast accuracy by signal age.
    
    Groups signal-realization pairs by age bucket and computes
    hit rate per bucket.
    
    Args:
        realized_returns: Actual returns.
        forecast_returns: Forecasted returns.
        ages: Signal ages in days.
        age_bins: Age bucket edges (default [0,1,3,7,14,30]).
    
    Returns:
        {age_bucket: hit_rate}.
    """
    if age_bins is None:
        age_bins = [0, 1, 3, 7, 14, 30]
    
    result = {}
    for i in range(len(age_bins) - 1):
        lo, hi = age_bins[i], age_bins[i + 1]
        mask = [(lo <= a < hi) for a in ages]
        hits = [
            1 if (r > 0 and f > 0) or (r < 0 and f < 0) else 0
            for r, f, m in zip(realized_returns, forecast_returns, mask) if m
        ]
        
        label = f"{lo}-{hi}d"
        result[label] = sum(hits) / max(len(hits), 1)
    
    return result
