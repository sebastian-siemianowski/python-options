"""
Story 8.5: Sector Rotation Signal Engine.

Relative momentum vs benchmark + sector breadth for rotation signals.

Usage:
    from decision.sector_rotation import (
        compute_sector_momentum,
        compute_sector_breadth,
        generate_rotation_signal,
    )
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List


# Configuration
MOMENTUM_WINDOW = 20        # days for momentum computation
BREADTH_THRESHOLD = 0.6     # fraction of positive signals for "strong"


@dataclass
class SectorSignal:
    """Sector rotation signal."""
    sector: str
    relative_momentum: float    # sector return - benchmark return
    breadth: float              # fraction of assets with positive forecast
    composite_score: float      # combined signal
    recommendation: str         # "OVERWEIGHT", "NEUTRAL", "UNDERWEIGHT"


def compute_sector_momentum(
    sector_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    window: int = MOMENTUM_WINDOW,
) -> float:
    """
    Relative sector momentum vs benchmark.
    
    Args:
        sector_returns: Sector daily returns.
        benchmark_returns: Benchmark (e.g., SPY) daily returns.
        window: Lookback period.
    
    Returns:
        Relative momentum (cumulative excess return over window).
    """
    if len(sector_returns) < window or len(benchmark_returns) < window:
        return 0.0
    
    sec_cum = float(np.sum(sector_returns[-window:]))
    bench_cum = float(np.sum(benchmark_returns[-window:]))
    
    return sec_cum - bench_cum


def compute_sector_breadth(
    asset_forecasts: Dict[str, float],
) -> float:
    """
    Sector breadth: fraction of assets with positive forecast.
    
    Args:
        asset_forecasts: {asset: forecast_pct} for assets in sector.
    
    Returns:
        Breadth [0, 1].
    """
    if not asset_forecasts:
        return 0.5
    
    positive = sum(1 for f in asset_forecasts.values() if f > 0)
    return positive / len(asset_forecasts)


def generate_rotation_signal(
    sector: str,
    sector_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    asset_forecasts: Dict[str, float],
    momentum_weight: float = 0.6,
    breadth_weight: float = 0.4,
) -> SectorSignal:
    """
    Generate composite sector rotation signal.
    
    Args:
        sector: Sector name.
        sector_returns: Sector average daily returns.
        benchmark_returns: Benchmark returns.
        asset_forecasts: {asset: forecast} for sector assets.
        momentum_weight: Weight for relative momentum.
        breadth_weight: Weight for breadth signal.
    
    Returns:
        SectorSignal with recommendation.
    """
    rel_mom = compute_sector_momentum(sector_returns, benchmark_returns)
    breadth = compute_sector_breadth(asset_forecasts)
    
    # Normalize momentum to [-1, 1] range using tanh
    mom_signal = float(np.tanh(rel_mom * 50))  # scale returns to signal
    
    # Breadth signal: 0.5 = neutral, >0.5 positive, <0.5 negative
    breadth_signal = (breadth - 0.5) * 2  # map [0,1] -> [-1,1]
    
    composite = momentum_weight * mom_signal + breadth_weight * breadth_signal
    
    if composite > 0.3:
        rec = "OVERWEIGHT"
    elif composite < -0.3:
        rec = "UNDERWEIGHT"
    else:
        rec = "NEUTRAL"
    
    return SectorSignal(
        sector=sector,
        relative_momentum=rel_mom,
        breadth=breadth,
        composite_score=composite,
        recommendation=rec,
    )


def rank_sectors(signals: List[SectorSignal]) -> List[SectorSignal]:
    """Rank sectors by composite score (descending)."""
    return sorted(signals, key=lambda s: s.composite_score, reverse=True)
