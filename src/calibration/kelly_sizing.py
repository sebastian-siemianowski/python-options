"""
Epic 13: Kelly Criterion Integration with Calibrated Probabilities
==================================================================

Story 13.1: Full Kelly Sizing from BMA Predictive Distribution
Story 13.2: Risk-Adjusted Kelly with Drawdown Constraint
Story 13.3: Fractional Kelly Auto-Tuning via Utility Maximization

Provides growth-optimal position sizing from BMA forecast distributions,
with drawdown dampening and data-driven Kelly fraction selection.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Story 13.1: Full Kelly Sizing
# ---------------------------------------------------------------------------

# Bounds for Kelly fraction (max 50% of capital in either direction)
KELLY_MAX = 0.5
KELLY_MIN = -0.5
# Default fractional Kelly (half-Kelly)
DEFAULT_KELLY_FRAC = 0.5
# Minimum Kelly fraction to enter a trade
MIN_KELLY_THRESHOLD = 0.02


def kelly_fraction(
    mu: float,
    sigma: float,
    nu: Optional[float] = None,
    kelly_frac: float = DEFAULT_KELLY_FRAC,
) -> float:
    """
    Compute Kelly fraction from BMA predictive distribution.

    Gaussian case:
        f = kelly_frac * mu / sigma^2

    Student-t case (nu > 4):
        f = kelly_frac * mu / sigma^2 * 1/(1 + 6/(nu-4))

    Parameters
    ----------
    mu : float
        Predicted mean return.
    sigma : float
        Predicted standard deviation.
    nu : float or None
        Degrees of freedom for Student-t. None = Gaussian.
    kelly_frac : float
        Fractional Kelly multiplier (default 0.5 = half-Kelly).

    Returns
    -------
    float
        Kelly fraction in [KELLY_MIN, KELLY_MAX].
    """
    if sigma <= 0 or not np.isfinite(sigma):
        return 0.0
    if not np.isfinite(mu):
        return 0.0

    # Base Gaussian Kelly
    f = mu / (sigma ** 2)

    # Student-t kurtosis adjustment
    if nu is not None and np.isfinite(nu) and nu > 4.0:
        kappa_excess = 6.0 / (nu - 4.0)
        f = f / (1.0 + kappa_excess / 6.0)
    elif nu is not None and nu <= 4.0:
        # Very heavy tails: reduce aggressively
        f = f * 0.25

    # Apply fractional Kelly
    f = kelly_frac * f

    # Bound
    return float(np.clip(f, KELLY_MIN, KELLY_MAX))


def kelly_fraction_array(
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: Optional[np.ndarray] = None,
    kelly_frac: float = DEFAULT_KELLY_FRAC,
) -> np.ndarray:
    """
    Vectorized Kelly fraction for arrays of predictions.

    Parameters
    ----------
    mu : ndarray, shape (T,)
        Predicted mean returns.
    sigma : ndarray, shape (T,)
        Predicted standard deviations.
    nu : ndarray or None, shape (T,)
        Degrees of freedom (None = all Gaussian).
    kelly_frac : float
        Fractional Kelly multiplier.

    Returns
    -------
    ndarray, shape (T,)
        Kelly fractions bounded to [KELLY_MIN, KELLY_MAX].
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(mu)

    result = np.zeros(n)
    for i in range(n):
        nu_i = float(nu[i]) if nu is not None else None
        result[i] = kelly_fraction(mu[i], sigma[i], nu_i, kelly_frac)

    return result


def kelly_position_pnl(
    returns: np.ndarray,
    fractions: np.ndarray,
) -> np.ndarray:
    """
    Compute PnL series from Kelly position sizing.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        Realized returns.
    fractions : ndarray, shape (T,)
        Kelly fractions (position sizes).

    Returns
    -------
    ndarray, shape (T,)
        Strategy returns = f_t * r_t.
    """
    returns = np.asarray(returns, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    return fractions * returns


def kelly_hit_rate(
    returns: np.ndarray,
    fractions: np.ndarray,
    f_min: float = MIN_KELLY_THRESHOLD,
) -> float:
    """
    Hit rate when trading only when |f| > f_min.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        Realized returns.
    fractions : ndarray, shape (T,)
        Kelly fractions.
    f_min : float
        Minimum Kelly fraction to trade.

    Returns
    -------
    float
        Hit rate (fraction of profitable trades when trading).
    """
    mask = np.abs(fractions) > f_min
    if mask.sum() == 0:
        return 0.5  # no trades
    pnl = fractions[mask] * returns[mask]
    return float(np.mean(pnl > 0))


def kelly_profit_factor(
    returns: np.ndarray,
    fractions: np.ndarray,
    f_min: float = MIN_KELLY_THRESHOLD,
) -> float:
    """
    Profit factor = gross profits / gross losses.

    Parameters
    ----------
    returns : ndarray
        Realized returns.
    fractions : ndarray
        Kelly fractions.
    f_min : float
        Minimum Kelly fraction to trade.

    Returns
    -------
    float
        Profit factor (> 1.0 means profitable).
    """
    mask = np.abs(fractions) > f_min
    if mask.sum() == 0:
        return 1.0
    pnl = fractions[mask] * returns[mask]
    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = np.abs(np.sum(pnl[pnl < 0]))
    if gross_loss < 1e-15:
        return 10.0  # cap
    return float(gross_profit / gross_loss)


# ---------------------------------------------------------------------------
# Story 13.2: Drawdown-Adjusted Kelly
# ---------------------------------------------------------------------------

# Drawdown thresholds
DD_REDUCE_THRESHOLD = 0.10   # Start reducing at 10% drawdown
DD_FLAT_THRESHOLD = 0.15     # Go flat at 15% drawdown


@dataclass
class DrawdownAdjustedResult:
    """Result of drawdown-adjusted Kelly sizing."""
    f_kelly: float          # Original Kelly fraction
    f_adjusted: float       # Drawdown-adjusted fraction
    current_dd: float       # Current drawdown
    dd_dampener: float      # Dampening factor applied
    is_flat: bool           # Whether position is flat due to drawdown


def drawdown_adjusted_kelly(
    f_kelly: float,
    current_dd: float,
    max_dd: float = DD_FLAT_THRESHOLD,
) -> DrawdownAdjustedResult:
    """
    Scale Kelly fraction by drawdown-dependent dampener.

    When dd > 10%: f_adj = f_kelly * (1 - dd/max_dd)
    When dd > 15%: f_adj = 0 (flat)

    Parameters
    ----------
    f_kelly : float
        Raw Kelly fraction.
    current_dd : float
        Current drawdown (positive value, e.g. 0.12 = 12%).
    max_dd : float
        Maximum allowed drawdown before going flat.

    Returns
    -------
    DrawdownAdjustedResult
        Adjusted Kelly fraction with metadata.
    """
    current_dd = abs(current_dd)

    if current_dd >= max_dd:
        return DrawdownAdjustedResult(
            f_kelly=f_kelly, f_adjusted=0.0,
            current_dd=current_dd, dd_dampener=0.0, is_flat=True,
        )
    elif current_dd >= DD_REDUCE_THRESHOLD:
        dampener = 1.0 - current_dd / max_dd
        f_adj = f_kelly * dampener
        return DrawdownAdjustedResult(
            f_kelly=f_kelly, f_adjusted=float(np.clip(f_adj, KELLY_MIN, KELLY_MAX)),
            current_dd=current_dd, dd_dampener=dampener, is_flat=False,
        )
    else:
        return DrawdownAdjustedResult(
            f_kelly=f_kelly, f_adjusted=f_kelly,
            current_dd=current_dd, dd_dampener=1.0, is_flat=False,
        )


def compute_running_drawdown(equity_curve: np.ndarray) -> np.ndarray:
    """
    Compute running drawdown from equity curve.

    Parameters
    ----------
    equity_curve : ndarray, shape (T,)
        Cumulative equity (e.g. starting at 1.0).

    Returns
    -------
    ndarray, shape (T,)
        Drawdown at each point (positive = underwater).
    """
    equity_curve = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (running_max - equity_curve) / np.maximum(running_max, 1e-15)
    return drawdown


def apply_drawdown_kelly_backtest(
    returns: np.ndarray,
    fractions: np.ndarray,
    max_dd: float = DD_FLAT_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply drawdown-adjusted Kelly sizing in a backtest.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        Realized returns.
    fractions : ndarray, shape (T,)
        Raw Kelly fractions.
    max_dd : float
        Maximum allowed drawdown.

    Returns
    -------
    Tuple of (strategy_returns, adjusted_fractions, drawdowns)
        Each shape (T,).
    """
    returns = np.asarray(returns, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    n = len(returns)

    adjusted = np.zeros(n)
    equity = np.ones(n + 1)
    dd = np.zeros(n)

    for t in range(n):
        # Compute current drawdown from equity
        peak = np.max(equity[:t + 1])
        dd[t] = (peak - equity[t]) / max(peak, 1e-15)

        # Adjust Kelly fraction
        result = drawdown_adjusted_kelly(fractions[t], dd[t], max_dd)
        adjusted[t] = result.f_adjusted

        # Update equity
        equity[t + 1] = equity[t] * (1.0 + adjusted[t] * returns[t])

    strat_returns = adjusted * returns
    return strat_returns, adjusted, dd


def compute_max_drawdown(returns: np.ndarray) -> float:
    """
    Compute maximum drawdown from a returns series.

    Parameters
    ----------
    returns : ndarray
        Strategy returns.

    Returns
    -------
    float
        Maximum drawdown (positive value).
    """
    equity = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(equity)
    dd = (running_max - equity) / np.maximum(running_max, 1e-15)
    return float(np.max(dd)) if len(dd) > 0 else 0.0


def compute_sharpe(returns: np.ndarray, annualization: float = 252.0) -> float:
    """
    Compute annualized Sharpe ratio.

    Parameters
    ----------
    returns : ndarray
        Daily strategy returns.
    annualization : float
        Annualization factor.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0
    std = np.std(returns, ddof=1)
    if std < 1e-15:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(annualization))


# ---------------------------------------------------------------------------
# Story 13.3: Fractional Kelly Auto-Tuning
# ---------------------------------------------------------------------------

# Default grid of Kelly fractions to search
DEFAULT_FRAC_GRID = [0.1, 0.2, 0.3, 0.5]
# Walk-forward parameters
AUTOTUNE_TRAIN_DAYS = 252   # 1 year training
AUTOTUNE_TEST_DAYS = 21     # 1 month test


@dataclass
class KellyAutoTuneResult:
    """Result of Kelly fraction auto-tuning."""
    optimal_frac: float         # Best Kelly fraction from walk-forward
    expected_utility: float     # Expected log utility at optimal fraction
    frac_grid: List[float]      # Grid searched
    utilities: List[float]      # Utility at each grid point
    n_folds: int                # Number of walk-forward folds
    sharpe_at_optimal: float    # Sharpe ratio at optimal fraction


def _log_utility(returns: np.ndarray, frac: float) -> float:
    """
    Compute expected log utility: E[log(1 + f*r)].

    Parameters
    ----------
    returns : ndarray
        Realized returns.
    frac : float
        Kelly fraction.

    Returns
    -------
    float
        Average log utility.
    """
    portfolio_returns = 1.0 + frac * returns
    # Clip to avoid log(0) or log(negative)
    portfolio_returns = np.maximum(portfolio_returns, 1e-10)
    return float(np.mean(np.log(portfolio_returns)))


def auto_tune_kelly_frac(
    returns: np.ndarray,
    forecasts_mu: np.ndarray,
    forecasts_sigma: np.ndarray,
    forecasts_nu: Optional[np.ndarray] = None,
    frac_grid: Optional[List[float]] = None,
    train_days: int = AUTOTUNE_TRAIN_DAYS,
    test_days: int = AUTOTUNE_TEST_DAYS,
) -> KellyAutoTuneResult:
    """
    Data-driven Kelly fraction selection via walk-forward utility maximization.

    For each fold:
      1. Train: compute Kelly fractions with each candidate frac
      2. Test: evaluate log utility E[log(1 + f*r)] on out-of-sample data
    Select the frac with highest average OOS utility.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        Realized returns.
    forecasts_mu : ndarray, shape (T,)
        Predicted mean returns.
    forecasts_sigma : ndarray, shape (T,)
        Predicted standard deviations.
    forecasts_nu : ndarray or None, shape (T,)
        Degrees of freedom (None = Gaussian).
    frac_grid : list of float or None
        Kelly fractions to search (default: [0.1, 0.2, 0.3, 0.5]).
    train_days : int
        Training window size.
    test_days : int
        Test window size.

    Returns
    -------
    KellyAutoTuneResult
        Optimal Kelly fraction with diagnostics.
    """
    if frac_grid is None:
        frac_grid = list(DEFAULT_FRAC_GRID)

    returns = np.asarray(returns, dtype=float)
    forecasts_mu = np.asarray(forecasts_mu, dtype=float)
    forecasts_sigma = np.asarray(forecasts_sigma, dtype=float)
    n = len(returns)

    n_fracs = len(frac_grid)
    fold_utilities = [[] for _ in range(n_fracs)]
    n_folds = 0

    # Walk-forward: rolling train/test windows
    start = train_days
    while start + test_days <= n:
        test_start = start
        test_end = start + test_days

        test_returns = returns[test_start:test_end]
        test_mu = forecasts_mu[test_start:test_end]
        test_sigma = forecasts_sigma[test_start:test_end]
        test_nu = forecasts_nu[test_start:test_end] if forecasts_nu is not None else None

        for j, frac in enumerate(frac_grid):
            # Compute Kelly fractions for this candidate
            fracs = kelly_fraction_array(test_mu, test_sigma, test_nu, kelly_frac=frac)
            # Compute log utility on test data
            util = _log_utility(test_returns, fracs)
            fold_utilities[j].append(util)

        n_folds += 1
        start += test_days

    # Fallback for short data
    if n_folds == 0:
        return KellyAutoTuneResult(
            optimal_frac=0.5,
            expected_utility=0.0,
            frac_grid=frac_grid,
            utilities=[0.0] * n_fracs,
            n_folds=0,
            sharpe_at_optimal=0.0,
        )

    # Average utility per fraction
    avg_utilities = [float(np.mean(u)) for u in fold_utilities]

    # Find optimal
    best_idx = int(np.argmax(avg_utilities))
    optimal_frac = frac_grid[best_idx]

    # Compute Sharpe at optimal fraction
    fracs = kelly_fraction_array(
        forecasts_mu, forecasts_sigma, forecasts_nu, kelly_frac=optimal_frac
    )
    strat_returns = fracs * returns
    sharpe = compute_sharpe(strat_returns)

    return KellyAutoTuneResult(
        optimal_frac=optimal_frac,
        expected_utility=avg_utilities[best_idx],
        frac_grid=frac_grid,
        utilities=avg_utilities,
        n_folds=n_folds,
        sharpe_at_optimal=sharpe,
    )


def _log_utility_vectorized(returns: np.ndarray, fractions: np.ndarray) -> float:
    """
    Compute log utility with position-specific fractions.

    Parameters
    ----------
    returns : ndarray
        Realized returns.
    fractions : ndarray
        Position fractions at each timestep.

    Returns
    -------
    float
        Average log utility.
    """
    portfolio_returns = 1.0 + fractions * returns
    portfolio_returns = np.maximum(portfolio_returns, 1e-10)
    return float(np.mean(np.log(portfolio_returns)))
