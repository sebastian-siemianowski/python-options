"""
Story 6.4: Profitability Gate.

Automated validation that rejects deploys if forecast quality is below threshold.
Returns exit codes for CI integration.

Usage:
    from calibration.profitability_gate import evaluate_profitability_gate
    verdict = evaluate_profitability_gate(backtest_result, quality_report)
"""
from dataclasses import dataclass
from typing import List, Optional


# Thresholds
SHARPE_GREEN = 0.5
SHARPE_AMBER = 0.2
HIT_RATE_GREEN = 0.55
HIT_RATE_AMBER = 0.50
MAX_DD_GREEN = -0.15
MAX_DD_AMBER = -0.25
IC_GREEN = 0.05
IC_AMBER = 0.02

# Exit codes for CI
EXIT_PASS = 0
EXIT_REVIEW = 1
EXIT_REJECT = 2


@dataclass
class GateVerdict:
    """Profitability gate result."""
    status: str          # "DEPLOY", "REVIEW", "REJECT"
    exit_code: int
    sharpe: float = 0.0
    hit_rate: float = 0.0
    max_drawdown: float = 0.0
    ic: float = 0.0
    reasons: list = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


def evaluate_profitability_gate(
    sharpe: float = 0.0,
    hit_rate: float = 0.0,
    max_drawdown: float = 0.0,
    ic: float = 0.0,
    baseline_sharpe: Optional[float] = None,
) -> GateVerdict:
    """
    Evaluate profitability gate.
    
    Args:
        sharpe: Portfolio Sharpe ratio.
        hit_rate: Forecast hit rate (fraction).
        max_drawdown: Maximum drawdown (negative number).
        ic: Information coefficient.
        baseline_sharpe: If provided, compare against baseline.
    
    Returns:
        GateVerdict with status and exit code.
    """
    reasons: List[str] = []
    reject = False
    review = False
    
    # Sharpe check
    if sharpe >= SHARPE_GREEN:
        pass
    elif sharpe >= SHARPE_AMBER:
        review = True
        reasons.append(f"Sharpe {sharpe:.2f} below green ({SHARPE_GREEN})")
    else:
        reject = True
        reasons.append(f"Sharpe {sharpe:.2f} below amber ({SHARPE_AMBER})")
    
    # Hit rate check
    if hit_rate >= HIT_RATE_GREEN:
        pass
    elif hit_rate >= HIT_RATE_AMBER:
        review = True
        reasons.append(f"Hit rate {hit_rate:.2%} below green ({HIT_RATE_GREEN:.0%})")
    else:
        reject = True
        reasons.append(f"Hit rate {hit_rate:.2%} below amber ({HIT_RATE_AMBER:.0%})")
    
    # Max drawdown check
    if max_drawdown >= MAX_DD_GREEN:
        pass
    elif max_drawdown >= MAX_DD_AMBER:
        review = True
        reasons.append(f"Max DD {max_drawdown:.2%} below green ({MAX_DD_GREEN:.0%})")
    else:
        reject = True
        reasons.append(f"Max DD {max_drawdown:.2%} below amber ({MAX_DD_AMBER:.0%})")
    
    # IC check
    if ic >= IC_GREEN:
        pass
    elif ic >= IC_AMBER:
        review = True
        reasons.append(f"IC {ic:.3f} below green ({IC_GREEN})")
    else:
        reject = True
        reasons.append(f"IC {ic:.3f} below amber ({IC_AMBER})")
    
    # Baseline comparison
    if baseline_sharpe is not None:
        improvement = sharpe - baseline_sharpe
        if improvement < 0:
            review = True
            reasons.append(f"Sharpe regressed vs baseline ({baseline_sharpe:.2f})")
    
    if reject:
        status = "REJECT"
        exit_code = EXIT_REJECT
    elif review:
        status = "REVIEW"
        exit_code = EXIT_REVIEW
    else:
        status = "DEPLOY"
        exit_code = EXIT_PASS
    
    return GateVerdict(
        status=status,
        exit_code=exit_code,
        sharpe=sharpe,
        hit_rate=hit_rate,
        max_drawdown=max_drawdown,
        ic=ic,
        reasons=reasons,
    )
