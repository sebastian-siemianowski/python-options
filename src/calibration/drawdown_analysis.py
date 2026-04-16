"""
Story 6.9: Drawdown Analysis and Risk Budgeting.

Detailed drawdown analysis: top-N events with dates, duration, recovery time.
Per-asset risk contribution during drawdowns.

Usage:
    from calibration.drawdown_analysis import analyze_drawdowns
    report = analyze_drawdowns(equity_curve, asset_pnl_dict)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


TOP_N_DRAWDOWNS = 5


@dataclass
class DrawdownEvent:
    """Single drawdown event."""
    start_idx: int
    trough_idx: int
    recovery_idx: Optional[int]  # None if not yet recovered
    magnitude: float              # Negative number
    duration: int                 # Days from start to recovery (or current)
    trough_duration: int          # Days from start to trough


@dataclass
class DrawdownReport:
    """Full drawdown analysis."""
    drawdown_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    top_events: List[DrawdownEvent] = field(default_factory=list)
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    drawdown_durations: List[int] = field(default_factory=list)
    per_asset_risk: Dict[str, float] = field(default_factory=dict)


def analyze_drawdowns(
    equity_curve: np.ndarray,
    asset_pnl: Optional[Dict[str, np.ndarray]] = None,
    top_n: int = TOP_N_DRAWDOWNS,
) -> DrawdownReport:
    """
    Analyze drawdowns from equity curve.
    
    Args:
        equity_curve: Cumulative PnL or equity values.
        asset_pnl: Optional per-asset PnL arrays for risk attribution.
        top_n: Number of top drawdown events to report.
    
    Returns:
        DrawdownReport with events and risk budgets.
    """
    equity = np.asarray(equity_curve, dtype=float)
    n = len(equity)
    
    if n < 2:
        return DrawdownReport()
    
    # Compute drawdown curve
    peak = np.maximum.accumulate(equity)
    dd_curve = equity - peak
    
    report = DrawdownReport(drawdown_curve=dd_curve)
    report.max_drawdown = float(np.min(dd_curve))
    report.avg_drawdown = float(np.mean(dd_curve[dd_curve < 0])) if np.any(dd_curve < 0) else 0.0
    
    # Find drawdown events
    events = _identify_drawdown_events(equity, dd_curve, n)
    
    # Sort by magnitude (most severe first)
    events.sort(key=lambda e: e.magnitude)
    report.top_events = events[:top_n]
    report.drawdown_durations = [e.duration for e in events]
    
    # Per-asset risk contribution during drawdowns
    if asset_pnl:
        report.per_asset_risk = _compute_risk_contribution(
            asset_pnl, dd_curve
        )
    
    return report


def _identify_drawdown_events(
    equity: np.ndarray,
    dd_curve: np.ndarray,
    n: int,
) -> List[DrawdownEvent]:
    """Identify distinct drawdown events."""
    events = []
    in_drawdown = False
    start = 0
    trough_idx = 0
    trough_val = 0.0
    
    for i in range(n):
        if dd_curve[i] < 0:
            if not in_drawdown:
                # Start new drawdown
                in_drawdown = True
                start = i
                trough_idx = i
                trough_val = dd_curve[i]
            else:
                # Update trough
                if dd_curve[i] < trough_val:
                    trough_idx = i
                    trough_val = dd_curve[i]
        else:
            if in_drawdown:
                # Recovery
                events.append(DrawdownEvent(
                    start_idx=start,
                    trough_idx=trough_idx,
                    recovery_idx=i,
                    magnitude=float(trough_val),
                    duration=i - start,
                    trough_duration=trough_idx - start,
                ))
                in_drawdown = False
    
    # Handle ongoing drawdown at end
    if in_drawdown:
        events.append(DrawdownEvent(
            start_idx=start,
            trough_idx=trough_idx,
            recovery_idx=None,
            magnitude=float(trough_val),
            duration=n - start,
            trough_duration=trough_idx - start,
        ))
    
    return events


def _compute_risk_contribution(
    asset_pnl: Dict[str, np.ndarray],
    dd_curve: np.ndarray,
) -> Dict[str, float]:
    """Compute per-asset risk contribution during drawdown periods."""
    dd_mask = dd_curve < 0
    n_dd = int(np.sum(dd_mask))
    
    if n_dd == 0:
        return {}
    
    contributions = {}
    total_loss = 0.0
    
    for symbol, pnl in asset_pnl.items():
        pnl = np.asarray(pnl, dtype=float)
        # Only look at drawdown days (truncate to matching length)
        min_len = min(len(pnl), len(dd_mask))
        mask = dd_mask[:min_len]
        dd_pnl = pnl[:min_len][mask]
        
        loss = float(np.sum(dd_pnl[dd_pnl < 0]))
        contributions[symbol] = loss
        total_loss += loss
    
    # Normalize to fractions
    if total_loss < 0:
        for k in contributions:
            contributions[k] = contributions[k] / total_loss
    
    return dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))
