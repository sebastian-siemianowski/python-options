"""
Story 8.6: Volatility Surface Integration.

IV skew ratio, term structure slope, IV rank for forecast uncertainty adjustment.

Usage:
    from decision.vol_surface import (
        compute_skew_ratio,
        compute_term_structure_slope,
        compute_iv_rank,
        adjust_forecast_with_iv,
    )
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class IVContext:
    """Implied volatility context for an asset."""
    symbol: str
    skew_ratio: float          # put IV / call IV (>1 = fear premium)
    term_slope: float          # (long IV - short IV) / short IV
    iv_rank: float             # percentile [0, 1]
    iv_current: float          # current ATM IV
    iv_signal: str             # "HIGH_FEAR", "NORMAL", "COMPLACENT"


def compute_skew_ratio(
    put_iv: float,
    call_iv: float,
) -> float:
    """
    Compute put/call IV skew ratio.
    
    > 1.0: Put premium (fear/hedging demand)
    = 1.0: Symmetric risk pricing
    < 1.0: Call premium (unusual)
    
    Args:
        put_iv: 25-delta put implied volatility.
        call_iv: 25-delta call implied volatility.
    
    Returns:
        Skew ratio.
    """
    if call_iv <= 0:
        return 1.0
    return put_iv / call_iv


def compute_term_structure_slope(
    short_iv: float,
    long_iv: float,
) -> float:
    """
    Term structure slope: (long - short) / short.
    
    Positive: Normal contango (long-term vol > short-term).
    Negative: Backwardation (near-term fear).
    
    Args:
        short_iv: Front-month IV.
        long_iv: Back-month IV (e.g., 3-month).
    
    Returns:
        Slope (fractional).
    """
    if short_iv <= 0:
        return 0.0
    return (long_iv - short_iv) / short_iv


def compute_iv_rank(
    current_iv: float,
    historical_iv: np.ndarray,
) -> float:
    """
    IV rank: percentile of current IV in historical distribution.
    
    0.0 = lowest in lookback, 1.0 = highest.
    
    Args:
        current_iv: Current ATM implied volatility.
        historical_iv: Array of historical IV values.
    
    Returns:
        IV rank [0, 1].
    """
    if len(historical_iv) == 0:
        return 0.5
    
    rank = float(np.mean(historical_iv < current_iv))
    return rank


def compute_iv_context(
    symbol: str,
    put_iv: float,
    call_iv: float,
    short_iv: float,
    long_iv: float,
    current_iv: float,
    historical_iv: np.ndarray,
) -> IVContext:
    """
    Full IV context computation.
    
    Args:
        symbol: Asset symbol.
        put_iv: 25-delta put IV.
        call_iv: 25-delta call IV.
        short_iv: Front-month IV.
        long_iv: Back-month IV.
        current_iv: Current ATM IV.
        historical_iv: Historical IV array.
    
    Returns:
        IVContext with all metrics.
    """
    skew = compute_skew_ratio(put_iv, call_iv)
    slope = compute_term_structure_slope(short_iv, long_iv)
    rank = compute_iv_rank(current_iv, historical_iv)
    
    # Classify signal
    if rank > 0.80 and skew > 1.15:
        signal = "HIGH_FEAR"
    elif rank < 0.20 and skew < 0.95:
        signal = "COMPLACENT"
    else:
        signal = "NORMAL"
    
    return IVContext(
        symbol=symbol,
        skew_ratio=skew,
        term_slope=slope,
        iv_rank=rank,
        iv_current=current_iv,
        iv_signal=signal,
    )


def adjust_forecast_with_iv(
    forecast_pct: float,
    sigma: float,
    confidence: float,
    iv_context: IVContext,
) -> Dict[str, float]:
    """
    Adjust forecast uncertainty using IV context.
    
    HIGH_FEAR: Widen intervals, reduce confidence.
    COMPLACENT: Slightly tighten (but cautious).
    NORMAL: No adjustment.
    
    Args:
        forecast_pct: Forecast return.
        sigma: Forecast standard deviation.
        confidence: Forecast confidence.
        iv_context: IV context.
    
    Returns:
        Dict with adjusted parameters.
    """
    adj_forecast = forecast_pct
    adj_sigma = sigma
    adj_confidence = confidence
    
    if iv_context.iv_signal == "HIGH_FEAR":
        # More uncertainty: widen sigma by skew ratio, reduce confidence
        adj_sigma = sigma * (1.0 + 0.5 * (iv_context.skew_ratio - 1.0))
        adj_confidence = confidence * 0.75
    
    elif iv_context.iv_signal == "COMPLACENT":
        # Slightly tighten, but don't overfit to complacency
        adj_sigma = sigma * 0.95
    
    # IV rank adjustment: high rank -> wider sigma
    if iv_context.iv_rank > 0.70:
        rank_adj = 1.0 + 0.3 * (iv_context.iv_rank - 0.70) / 0.30
        adj_sigma *= rank_adj
    
    return {
        "forecast_pct": adj_forecast,
        "sigma": adj_sigma,
        "confidence": max(0.0, min(1.0, adj_confidence)),
    }
