"""
Story 6.5: Regime-Specific Profitability Analysis.

Breaks down backtest metrics by market regime to identify where
the system excels and where it struggles.

Usage:
    from calibration.regime_profitability import compute_regime_profitability
    regime_metrics = compute_regime_profitability(daily_pnl, regime_labels)
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List


# Standard regime labels (from MarketRegime IntEnum)
REGIME_NAMES = [
    "LOW_VOL_TREND",
    "HIGH_VOL_TREND",
    "LOW_VOL_RANGE",
    "HIGH_VOL_RANGE",
    "CRISIS_JUMP",
]


@dataclass
class RegimeMetrics:
    """Profitability metrics for a single regime."""
    regime: str
    sharpe: float = 0.0
    sortino: float = 0.0
    hit_rate: float = 0.0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0
    n_days: int = 0
    pct_of_total: float = 0.0


@dataclass
class RegimeProfitabilityReport:
    """Full regime profitability breakdown."""
    per_regime: Dict[str, RegimeMetrics] = field(default_factory=dict)
    best_regime: str = ""
    worst_regime: str = ""
    regime_transitions: int = 0


def compute_regime_profitability(
    daily_pnl: np.ndarray,
    regime_labels: np.ndarray,
    regime_names: List[str] = None,
) -> RegimeProfitabilityReport:
    """
    Compute profitability metrics per regime.
    
    Args:
        daily_pnl: Array of daily PnL values.
        regime_labels: Array of regime indices (integers), same length as daily_pnl.
        regime_names: Optional names for regime indices.
    
    Returns:
        RegimeProfitabilityReport with per-regime metrics.
    """
    if regime_names is None:
        regime_names = list(REGIME_NAMES)
    
    n = min(len(daily_pnl), len(regime_labels))
    if n < 2:
        return RegimeProfitabilityReport()
    
    daily_pnl = np.asarray(daily_pnl[:n], dtype=float)
    regime_labels = np.asarray(regime_labels[:n], dtype=int)
    
    report = RegimeProfitabilityReport()
    
    # Count transitions
    report.regime_transitions = int(np.sum(np.diff(regime_labels) != 0))
    
    unique_regimes = np.unique(regime_labels)
    best_sharpe = -999.0
    worst_sharpe = 999.0
    
    for r in unique_regimes:
        mask = regime_labels == r
        pnl = daily_pnl[mask]
        
        name = regime_names[r] if r < len(regime_names) else f"REGIME_{r}"
        
        metrics = RegimeMetrics(
            regime=name,
            n_days=int(np.sum(mask)),
            pct_of_total=float(np.sum(mask)) / n,
        )
        
        if len(pnl) > 1:
            metrics.avg_pnl = float(np.mean(pnl))
            
            # Hit rate
            metrics.hit_rate = float(np.mean(pnl > 0))
            
            # Sharpe
            std_pnl = np.std(pnl, ddof=1)
            if std_pnl > 0:
                metrics.sharpe = (np.mean(pnl) / std_pnl) * math.sqrt(252)
            
            # Sortino
            downside = pnl[pnl < 0]
            if len(downside) > 0:
                ds_std = np.std(downside, ddof=1)
                if ds_std > 0:
                    metrics.sortino = (np.mean(pnl) / ds_std) * math.sqrt(252)
            
            # Max drawdown
            cum = np.cumsum(pnl)
            peak = np.maximum.accumulate(cum)
            dd = cum - peak
            metrics.max_drawdown = float(np.min(dd)) if len(dd) > 0 else 0.0
        
        report.per_regime[name] = metrics
        
        if metrics.sharpe > best_sharpe:
            best_sharpe = metrics.sharpe
            report.best_regime = name
        if metrics.sharpe < worst_sharpe:
            worst_sharpe = metrics.sharpe
            report.worst_regime = name
    
    return report
