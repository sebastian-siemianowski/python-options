"""
Story 6.10: Automated Backtest Report with Executive Summary.

Aggregates all backtest diagnostics into a single executive summary
with recommendation: DEPLOY / REVIEW / REJECT.

Usage:
    from calibration.executive_summary import generate_executive_summary
    summary = generate_executive_summary(sharpe=0.7, ...)
"""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List


@dataclass
class ExecutiveSummary:
    """One-page backtest executive summary."""
    # Core metrics
    sharpe: float = 0.0
    sharpe_ci_lower: float = 0.0
    sharpe_ci_upper: float = 0.0
    hit_rate: float = 0.0
    max_drawdown: float = 0.0
    cagr_pct: float = 0.0
    
    # Best/worst
    best_asset: str = ""
    worst_asset: str = ""
    best_regime: str = ""
    worst_regime: str = ""
    
    # Cost analysis
    breakeven_cost_bps: Optional[float] = None
    
    # Significance
    skill_significant: bool = False
    
    # Recommendation
    recommendation: str = "REVIEW"
    reasons: List[str] = field(default_factory=list)


def generate_executive_summary(
    sharpe: float = 0.0,
    sharpe_ci_lower: float = 0.0,
    sharpe_ci_upper: float = 0.0,
    hit_rate: float = 0.0,
    max_drawdown: float = 0.0,
    cagr_pct: float = 0.0,
    best_asset: str = "",
    worst_asset: str = "",
    best_regime: str = "",
    worst_regime: str = "",
    breakeven_cost_bps: Optional[float] = None,
) -> ExecutiveSummary:
    """
    Generate executive summary from aggregate metrics.
    
    Returns:
        ExecutiveSummary with recommendation.
    """
    summary = ExecutiveSummary(
        sharpe=sharpe,
        sharpe_ci_lower=sharpe_ci_lower,
        sharpe_ci_upper=sharpe_ci_upper,
        hit_rate=hit_rate,
        max_drawdown=max_drawdown,
        cagr_pct=cagr_pct,
        best_asset=best_asset,
        worst_asset=worst_asset,
        best_regime=best_regime,
        worst_regime=worst_regime,
        breakeven_cost_bps=breakeven_cost_bps,
    )
    
    # Determine significance
    summary.skill_significant = sharpe_ci_lower > 0
    
    # Determine recommendation
    reasons = []
    reject = False
    review = False
    
    if sharpe < 0.2:
        reject = True
        reasons.append(f"Sharpe {sharpe:.2f} < 0.2")
    elif sharpe < 0.5:
        review = True
        reasons.append(f"Sharpe {sharpe:.2f} < 0.5")
    
    if hit_rate < 0.50:
        reject = True
        reasons.append(f"Hit rate {hit_rate:.1%} < 50%")
    elif hit_rate < 0.55:
        review = True
        reasons.append(f"Hit rate {hit_rate:.1%} < 55%")
    
    if max_drawdown < -0.25:
        reject = True
        reasons.append(f"Max DD {max_drawdown:.1%} < -25%")
    elif max_drawdown < -0.15:
        review = True
        reasons.append(f"Max DD {max_drawdown:.1%} < -15%")
    
    if not summary.skill_significant:
        review = True
        reasons.append("Sharpe not significant at 95% CI")
    
    if reject:
        summary.recommendation = "REJECT"
    elif review:
        summary.recommendation = "REVIEW"
    else:
        summary.recommendation = "DEPLOY"
    
    summary.reasons = reasons
    return summary


def save_executive_summary(summary: ExecutiveSummary, filepath: str) -> None:
    """Save executive summary to JSON."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(asdict(summary), f, indent=2)


def load_executive_summary(filepath: str) -> Optional[ExecutiveSummary]:
    """Load executive summary from JSON."""
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        data = json.load(f)
    return ExecutiveSummary(**data)
