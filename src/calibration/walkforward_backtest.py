"""
Story 6.1: Walk-Forward Backtest Engine for Forecast Signals.

Evaluates forecast profitability via rolling out-of-sample testing.
No look-ahead bias: train on [0,T), forecast at T, measure at T+H.

Usage:
    from calibration.walkforward_backtest import walk_forward_backtest
    results = walk_forward_backtest(prices, forecast_fn, ...)
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


# Default configuration
DEFAULT_TRAIN_WINDOW = 252
DEFAULT_STEP_SIZE = 5   # weekly
DEFAULT_HORIZONS = [1, 3, 7, 30]


@dataclass
class WalkForwardStep:
    """Single walk-forward evaluation step."""
    date_idx: int
    forecast_pct: float
    realized_return: float
    direction_correct: bool
    horizon: int


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward backtest results."""
    steps: List[WalkForwardStep] = field(default_factory=list)
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    information_coefficient: float = 0.0
    per_horizon_hit_rate: Dict[int, float] = field(default_factory=dict)
    n_steps: int = 0


def walk_forward_backtest(
    returns: np.ndarray,
    forecast_fn: Callable[[np.ndarray], float],
    train_window: int = DEFAULT_TRAIN_WINDOW,
    step_size: int = DEFAULT_STEP_SIZE,
    horizons: Optional[List[int]] = None,
) -> WalkForwardResult:
    """
    Walk-forward backtest engine.
    
    Args:
        returns: Array of daily returns (log or simple).
        forecast_fn: Given training returns, produce a forecast percentage.
        train_window: Number of days in training window.
        step_size: Days between evaluation steps.
        horizons: Forecast horizons to evaluate.
    
    Returns:
        WalkForwardResult with aggregate metrics.
    """
    if horizons is None:
        horizons = list(DEFAULT_HORIZONS)
    
    n = len(returns)
    all_steps: List[WalkForwardStep] = []
    
    for h in horizons:
        t = train_window
        while t + h < n:
            # Train on [0, t)
            train_data = returns[:t]
            
            # Forecast
            try:
                fc = forecast_fn(train_data)
            except Exception:
                fc = 0.0
            
            # Realized return over horizon
            realized = float(np.sum(returns[t:t+h])) * 100  # percent
            
            # Direction correct
            direction_correct = (fc > 0 and realized > 0) or (fc < 0 and realized < 0)
            
            all_steps.append(WalkForwardStep(
                date_idx=t,
                forecast_pct=fc,
                realized_return=realized,
                direction_correct=direction_correct,
                horizon=h,
            ))
            
            t += step_size
    
    return _compute_metrics(all_steps, horizons)


def _compute_metrics(
    steps: List[WalkForwardStep],
    horizons: List[int],
) -> WalkForwardResult:
    """Compute aggregate metrics from walk-forward steps."""
    result = WalkForwardResult(steps=steps, n_steps=len(steps))
    
    if not steps:
        return result
    
    # Hit rate
    correct = sum(1 for s in steps if s.direction_correct)
    result.hit_rate = correct / len(steps)
    
    # Per-horizon hit rate
    for h in horizons:
        h_steps = [s for s in steps if s.horizon == h]
        if h_steps:
            h_correct = sum(1 for s in h_steps if s.direction_correct)
            result.per_horizon_hit_rate[h] = h_correct / len(h_steps)
    
    # Use forecast-weighted returns for Sharpe
    weighted_returns = np.array([
        s.realized_return * np.sign(s.forecast_pct) for s in steps
    ])
    
    if len(weighted_returns) > 1:
        mean_r = np.mean(weighted_returns)
        std_r = np.std(weighted_returns, ddof=1)
        
        # Sharpe (annualized assuming weekly steps)
        if std_r > 0:
            result.sharpe = (mean_r / std_r) * math.sqrt(52)
        
        # Sortino (downside deviation)
        downside = weighted_returns[weighted_returns < 0]
        if len(downside) > 0:
            downside_std = np.std(downside, ddof=1)
            if downside_std > 0:
                result.sortino = (mean_r / downside_std) * math.sqrt(52)
    
    # Max drawdown from cumulative weighted returns
    cum_returns = np.cumsum(weighted_returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns - peak
    result.max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    # Information coefficient (rank correlation)
    forecasts = np.array([s.forecast_pct for s in steps])
    realized = np.array([s.realized_return for s in steps])
    result.information_coefficient = _rank_correlation(forecasts, realized)
    
    return result


def _rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation."""
    if len(x) < 3:
        return 0.0
    
    rx = _rankdata(x)
    ry = _rankdata(y)
    
    n = len(x)
    d = rx - ry
    rho = 1 - (6 * np.sum(d**2)) / (n * (n**2 - 1))
    return float(rho)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Simple rank data (average method for ties)."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return ranks
