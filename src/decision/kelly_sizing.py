"""
Story 8.7: Kelly Criterion Position Sizing.

Implements:
  f* = (p*b - q) / b   (Kelly fraction)
  Half-Kelly default, per-asset 10% cap, portfolio normalization.

Usage:
    from decision.kelly_sizing import compute_kelly_fraction, KellyRecommendation
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional


# Configuration
HALF_KELLY_SCALE = 0.5     # Half-Kelly by default
MAX_POSITION_PCT = 0.10    # 10% max per asset
MAX_PORTFOLIO_EXPOSURE = 1.0  # 100% max total exposure
MIN_EDGE = 0.001            # Minimum edge to size > 0


@dataclass
class KellyRecommendation:
    """Kelly position size recommendation."""
    symbol: str
    full_kelly: float        # Full Kelly fraction
    half_kelly: float        # Half-Kelly (default)
    capped_size: float       # After per-asset cap
    p_win: float             # Win probability
    avg_win: float           # Average win magnitude
    avg_loss: float          # Average loss magnitude
    edge: float              # Expected return per unit risk


def compute_kelly_fraction(
    p_win: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Kelly fraction: f* = (p*b - q) / b
    
    where b = avg_win / avg_loss, q = 1 - p.
    
    Args:
        p_win: Probability of winning.
        avg_win: Average win return (positive).
        avg_loss: Average loss return (positive magnitude).
    
    Returns:
        Kelly fraction (can be negative = don't bet).
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    
    b = avg_win / avg_loss
    q = 1.0 - p_win
    
    f = (p_win * b - q) / b
    return float(f)


def compute_kelly_from_quantiles(
    forecast_pct: float,
    sigma: float,
    p_up: Optional[float] = None,
) -> float:
    """
    Kelly fraction from forecast distribution.
    
    Uses normal approximation if p_up not provided.
    
    Args:
        forecast_pct: Expected return.
        sigma: Standard deviation of return.
        p_up: Probability of positive return.
    
    Returns:
        Kelly fraction.
    """
    if sigma <= 0:
        return 0.0
    
    if p_up is None:
        # Normal approximation
        from scipy.stats import norm
        p_up = float(norm.cdf(forecast_pct / sigma))
    
    # Expected win/loss magnitudes
    if p_up > 0 and p_up < 1:
        avg_win = abs(forecast_pct) + 0.5 * sigma
        avg_loss = abs(forecast_pct) + 0.5 * sigma if forecast_pct < 0 else 0.5 * sigma
        avg_loss = max(avg_loss, 1e-8)
        
        return compute_kelly_fraction(p_up, avg_win, avg_loss)
    
    return 0.0


def recommend_position_size(
    symbol: str,
    p_win: float,
    avg_win: float,
    avg_loss: float,
    kelly_scale: float = HALF_KELLY_SCALE,
    max_position: float = MAX_POSITION_PCT,
) -> KellyRecommendation:
    """
    Full position size recommendation with half-Kelly and cap.
    
    Args:
        symbol: Asset symbol.
        p_win: Win probability.
        avg_win: Average win magnitude.
        avg_loss: Average loss magnitude.
        kelly_scale: Fraction of full Kelly (0.5 = half-Kelly).
        max_position: Per-asset position cap.
    
    Returns:
        KellyRecommendation.
    """
    full_k = compute_kelly_fraction(p_win, avg_win, avg_loss)
    half_k = full_k * kelly_scale
    
    # Don't size negative edge
    if half_k < MIN_EDGE:
        half_k = 0.0
    
    capped = min(half_k, max_position)
    
    edge = p_win * avg_win - (1.0 - p_win) * avg_loss
    
    return KellyRecommendation(
        symbol=symbol,
        full_kelly=full_k,
        half_kelly=half_k if full_k >= MIN_EDGE else 0.0,
        capped_size=capped,
        p_win=p_win,
        avg_win=avg_win,
        avg_loss=avg_loss,
        edge=edge,
    )


def normalize_portfolio(
    recommendations: List[KellyRecommendation],
    max_exposure: float = MAX_PORTFOLIO_EXPOSURE,
) -> List[KellyRecommendation]:
    """
    Normalize portfolio positions so total exposure <= max_exposure.
    
    Scales all positions proportionally if over limit.
    
    Args:
        recommendations: List of position recommendations.
        max_exposure: Maximum total exposure.
    
    Returns:
        Normalized recommendations (new objects).
    """
    total = sum(abs(r.capped_size) for r in recommendations)
    
    if total <= max_exposure or total <= 0:
        return recommendations
    
    scale = max_exposure / total
    
    result = []
    for r in recommendations:
        result.append(KellyRecommendation(
            symbol=r.symbol,
            full_kelly=r.full_kelly,
            half_kelly=r.half_kelly,
            capped_size=r.capped_size * scale,
            p_win=r.p_win,
            avg_win=r.avg_win,
            avg_loss=r.avg_loss,
            edge=r.edge,
        ))
    
    return result
