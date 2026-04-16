"""
Story 6.8: Transaction Cost Sensitivity Analysis.

Runs backtests across multiple cost levels to find the breakeven cost
where Sharpe = 0. Answers: "How good must execution quality be?"

Usage:
    from calibration.cost_sensitivity import run_cost_sensitivity
    report = run_cost_sensitivity(returns, forecasts)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

from calibration.trading_simulator import simulate_trading


DEFAULT_COST_GRID = [0, 1, 3, 5, 10, 15, 20]


@dataclass
class CostLevel:
    """Result at one cost level."""
    cost_bps: float
    sharpe: float
    total_pnl: float
    total_costs: float


@dataclass
class CostSensitivityReport:
    """Full cost sensitivity analysis."""
    levels: List[CostLevel] = field(default_factory=list)
    breakeven_bps: Optional[float] = None
    avg_turnover: float = 0.0


def run_cost_sensitivity(
    returns: np.ndarray,
    forecasts: np.ndarray,
    cost_grid: Optional[List[float]] = None,
    mode: str = "long_only",
) -> CostSensitivityReport:
    """
    Run backtest at multiple cost levels.
    
    Args:
        returns: Daily returns array.
        forecasts: Forecast array (same length).
        cost_grid: List of cost levels in bps.
        mode: "long_only" or "long_short".
    
    Returns:
        CostSensitivityReport with breakeven analysis.
    """
    if cost_grid is None:
        cost_grid = list(DEFAULT_COST_GRID)
    
    levels: List[CostLevel] = []
    
    for bps in sorted(cost_grid):
        sim = simulate_trading(returns, forecasts, mode=mode, cost_bps=bps)
        levels.append(CostLevel(
            cost_bps=bps,
            sharpe=sim.sharpe,
            total_pnl=sim.total_pnl,
            total_costs=sim.total_costs,
        ))
    
    report = CostSensitivityReport(levels=levels)
    
    if levels:
        report.avg_turnover = simulate_trading(
            returns, forecasts, mode=mode, cost_bps=0
        ).avg_turnover
    
    # Find breakeven via linear interpolation
    report.breakeven_bps = _find_breakeven(levels)
    
    return report


def _find_breakeven(levels: List[CostLevel]) -> Optional[float]:
    """Find cost level where Sharpe crosses zero via interpolation."""
    if len(levels) < 2:
        return None
    
    # Look for sign change in Sharpe
    for i in range(len(levels) - 1):
        s1 = levels[i].sharpe
        s2 = levels[i + 1].sharpe
        
        if s1 > 0 and s2 <= 0:
            # Linear interpolation
            c1 = levels[i].cost_bps
            c2 = levels[i + 1].cost_bps
            if abs(s1 - s2) > 1e-10:
                breakeven = c1 + (c2 - c1) * s1 / (s1 - s2)
                return float(breakeven)
    
    # All positive or all negative
    if all(l.sharpe > 0 for l in levels):
        return None  # Profitable at all cost levels
    if all(l.sharpe <= 0 for l in levels):
        return 0.0  # Not profitable even at zero cost
    
    return None
