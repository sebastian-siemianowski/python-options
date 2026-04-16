"""
Story 6.2: Trading Strategy Simulator.

Simulates long-only or long-short trading with realistic costs.
Produces equity curves, PnL, and Sharpe ratios from forecast signals.

Usage:
    from calibration.trading_simulator import simulate_trading
    result = simulate_trading(returns, forecasts, mode="long_only")
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


DEFAULT_COST_BPS = 5.0
LONG_ONLY = "long_only"
LONG_SHORT = "long_short"


@dataclass
class TradeRecord:
    """Single day trade record."""
    day: int
    position: float       # -1 to +1
    pnl: float            # daily PnL (after costs)
    cost: float           # transaction cost for the day
    gross_return: float   # before costs


@dataclass
class SimulationResult:
    """Full simulation output."""
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    total_pnl: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    total_costs: float = 0.0
    avg_turnover: float = 0.0
    n_trades: int = 0


def simulate_trading(
    returns: np.ndarray,
    forecasts: np.ndarray,
    mode: str = LONG_ONLY,
    cost_bps: float = DEFAULT_COST_BPS,
    position_scale: float = 1.0,
) -> SimulationResult:
    """
    Simulate trading strategy from forecasts.
    
    Args:
        returns: Daily returns array.
        forecasts: Forecast array (same length), sign = direction, magnitude = conviction.
        mode: "long_only" or "long_short".
        cost_bps: Transaction cost in basis points per trade.
        position_scale: Scale factor for position sizing.
    
    Returns:
        SimulationResult with equity curve and metrics.
    """
    n = min(len(returns), len(forecasts))
    if n < 2:
        return SimulationResult()
    
    returns = returns[:n]
    forecasts = forecasts[:n]
    cost_frac = cost_bps / 10_000.0
    
    trades: List[TradeRecord] = []
    equity = np.zeros(n)
    prev_position = 0.0
    cumulative = 0.0
    total_turnover = 0.0
    
    for i in range(n):
        # Position from forecast
        raw_position = _compute_position(forecasts[i], mode, position_scale)
        
        # Transaction cost from turnover
        turnover = abs(raw_position - prev_position)
        total_turnover += turnover
        day_cost = turnover * cost_frac
        
        # PnL = position * return - cost
        gross = raw_position * returns[i]
        net = gross - day_cost
        cumulative += net
        
        trades.append(TradeRecord(
            day=i,
            position=raw_position,
            pnl=net,
            cost=day_cost,
            gross_return=gross,
        ))
        
        equity[i] = cumulative
        prev_position = raw_position
    
    return _compute_sim_metrics(trades, equity, total_turnover, n)


def _compute_position(forecast: float, mode: str, scale: float) -> float:
    """Convert forecast to position in [-1, +1]."""
    if mode == LONG_ONLY:
        pos = max(0.0, np.tanh(forecast * scale))
    else:
        pos = np.tanh(forecast * scale)
    return float(pos)


def _compute_sim_metrics(
    trades: List[TradeRecord],
    equity: np.ndarray,
    total_turnover: float,
    n: int,
) -> SimulationResult:
    """Compute simulation metrics."""
    result = SimulationResult(trades=trades, equity_curve=equity)
    
    if not trades:
        return result
    
    pnls = np.array([t.pnl for t in trades])
    result.total_pnl = float(np.sum(pnls))
    result.total_costs = float(sum(t.cost for t in trades))
    result.n_trades = sum(1 for t in trades if abs(t.position) > 0.01)
    result.avg_turnover = total_turnover / n if n > 0 else 0.0
    
    # Hit rate (positive PnL days)
    positive_days = sum(1 for t in trades if t.pnl > 0)
    active_days = sum(1 for t in trades if abs(t.position) > 0.01)
    result.hit_rate = positive_days / active_days if active_days > 0 else 0.0
    
    # Sharpe
    if len(pnls) > 1:
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1)
        if std_pnl > 0:
            result.sharpe = (mean_pnl / std_pnl) * math.sqrt(252)
    
    # Sortino
    downside = pnls[pnls < 0]
    if len(downside) > 0:
        ds_std = np.std(downside, ddof=1)
        if ds_std > 0:
            result.sortino = (np.mean(pnls) / ds_std) * math.sqrt(252)
    
    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    result.max_drawdown = float(np.min(dd))
    
    return result
