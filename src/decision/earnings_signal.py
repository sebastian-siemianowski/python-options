"""
Story 8.1: Earnings Event Signal Augmentation.

Detects pre/post earnings windows and adjusts forecast parameters:
  - Pre-earnings (T-3 to T): Reduce confidence, widen intervals
  - Post-earnings (T to T+2): Amplify innovation weighting
  - Historical earnings volatility per asset for calibration

Usage:
    from decision.earnings_signal import (
        detect_earnings_window,
        adjust_for_earnings,
        compute_historical_earnings_vol,
    )
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


# Configuration
PRE_EARNINGS_WINDOW = 3       # days before earnings
POST_EARNINGS_WINDOW = 2      # days after earnings
CONFIDENCE_REDUCTION = 0.4    # reduce confidence by 40% pre-earnings
INTERVAL_WIDEN_FACTOR = 1.8   # widen intervals 80% pre-earnings
POST_AMPLIFICATION = 1.5      # amplify post-earnings innovation weight


@dataclass
class EarningsContext:
    """Earnings event context for a given date."""
    in_pre_earnings: bool
    in_post_earnings: bool
    days_to_earnings: Optional[int]   # negative = past, positive = future
    earnings_date: Optional[str]
    context_label: str                # "PRE_EARNINGS", "POST_EARNINGS", "NORMAL"
    historical_earnings_vol: float    # historical earnings-day vol multiple


def detect_earnings_window(
    current_date_idx: int,
    earnings_date_idx: Optional[int],
    total_days: int,
) -> EarningsContext:
    """
    Detect whether current date is in pre/post earnings window.
    
    Args:
        current_date_idx: Index of current date in time series.
        earnings_date_idx: Index of nearest earnings date (None if no earnings).
        total_days: Total number of days in the series.
    
    Returns:
        EarningsContext with window detection.
    """
    if earnings_date_idx is None:
        return EarningsContext(
            in_pre_earnings=False,
            in_post_earnings=False,
            days_to_earnings=None,
            earnings_date=None,
            context_label="NORMAL",
            historical_earnings_vol=1.0,
        )
    
    days_to = earnings_date_idx - current_date_idx
    
    in_pre = 0 < days_to <= PRE_EARNINGS_WINDOW
    in_post = -POST_EARNINGS_WINDOW <= days_to <= 0
    
    if in_pre:
        label = "PRE_EARNINGS"
    elif in_post:
        label = "POST_EARNINGS"
    else:
        label = "NORMAL"
    
    return EarningsContext(
        in_pre_earnings=in_pre,
        in_post_earnings=in_post,
        days_to_earnings=days_to,
        earnings_date=None,  # populated externally
        context_label=label,
        historical_earnings_vol=1.0,  # populated by compute_historical_earnings_vol
    )


def adjust_for_earnings(
    forecast_pct: float,
    confidence: float,
    sigma: float,
    context: EarningsContext,
) -> Dict[str, float]:
    """
    Adjust forecast parameters based on earnings context.
    
    Pre-earnings: reduce confidence, widen uncertainty.
    Post-earnings: amplify innovation weight (accept new information faster).
    
    Args:
        forecast_pct: Forecast return percentage.
        confidence: Forecast confidence [0, 1].
        sigma: Forecast standard deviation.
        context: Earnings context.
    
    Returns:
        Dict with adjusted forecast_pct, confidence, sigma.
    """
    adj_forecast = forecast_pct
    adj_confidence = confidence
    adj_sigma = sigma
    
    if context.in_pre_earnings:
        # Reduce confidence, widen intervals
        adj_confidence = confidence * (1.0 - CONFIDENCE_REDUCTION)
        adj_sigma = sigma * INTERVAL_WIDEN_FACTOR * context.historical_earnings_vol
    
    elif context.in_post_earnings:
        # Amplify innovation weight (larger moves are more informative)
        # Sigma stays wide but forecast signal is amplified
        adj_forecast = forecast_pct * POST_AMPLIFICATION
        adj_sigma = sigma * context.historical_earnings_vol
    
    return {
        "forecast_pct": adj_forecast,
        "confidence": max(0.0, min(1.0, adj_confidence)),
        "sigma": adj_sigma,
    }


def compute_historical_earnings_vol(
    returns: np.ndarray,
    earnings_indices: List[int],
    window: int = 1,
) -> float:
    """
    Compute historical volatility multiple around earnings dates.
    
    Ratio of earnings-day volatility to normal-day volatility.
    
    Args:
        returns: Full returns array.
        earnings_indices: Indices of past earnings dates.
        window: Days around earnings to include.
    
    Returns:
        Volatility multiple (> 1 means earnings days are more volatile).
    """
    if len(returns) == 0 or len(earnings_indices) == 0:
        return 1.0
    
    n = len(returns)
    earnings_mask = np.zeros(n, dtype=bool)
    
    for idx in earnings_indices:
        for d in range(-window, window + 1):
            pos = idx + d
            if 0 <= pos < n:
                earnings_mask[pos] = True
    
    earnings_returns = returns[earnings_mask]
    normal_returns = returns[~earnings_mask]
    
    if len(earnings_returns) < 2 or len(normal_returns) < 2:
        return 1.0
    
    earnings_vol = np.std(earnings_returns)
    normal_vol = np.std(normal_returns)
    
    if normal_vol < 1e-10:
        return 1.0
    
    return float(earnings_vol / normal_vol)


def find_nearest_earnings(
    current_idx: int,
    earnings_indices: List[int],
    look_ahead: int = 30,
    look_back: int = 5,
) -> Optional[int]:
    """
    Find nearest earnings date within look window.
    
    Prioritizes upcoming earnings (look_ahead) over past (look_back).
    
    Args:
        current_idx: Current index in the series.
        earnings_indices: Sorted list of earnings date indices.
        look_ahead: Max days forward to search.
        look_back: Max days backward to search.
    
    Returns:
        Index of nearest earnings date, or None.
    """
    if not earnings_indices:
        return None
    
    best = None
    best_dist = float("inf")
    
    for e_idx in earnings_indices:
        dist = e_idx - current_idx
        if -look_back <= dist <= look_ahead:
            if abs(dist) < best_dist:
                best_dist = abs(dist)
                best = e_idx
    
    return best
