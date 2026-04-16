"""
Epic 14: Walk-Forward Backtest with Transaction Costs
=====================================================

Story 14.1: Realistic Transaction Cost Model
Story 14.2: Turnover-Penalized Signal Generation
Story 14.3: Optimal Rebalancing Frequency per Asset Class

Models realistic trading friction (spreads, market impact) and optimizes
signal generation and rebalancing frequency to maximize net profitability.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Story 14.1: Realistic Transaction Cost Model
# ---------------------------------------------------------------------------

# Default spread estimates in basis points
SPREAD_BPS = {
    "large_cap": 2.0,
    "mid_cap": 5.0,
    "small_cap": 15.0,
    "crypto": 10.0,
    "metals": 3.0,
}

# Market impact coefficient
IMPACT_COEFFICIENT = 0.1


@dataclass
class TransactionCostResult:
    """Result of transaction cost computation."""
    spread_cost: float       # Cost from bid-ask spread (dollars)
    impact_cost: float       # Cost from market impact (dollars)
    total_cost: float        # Total round-trip cost (dollars)
    cost_bps: float          # Total cost in basis points of trade value


def transaction_cost(
    price: float,
    shares: float,
    spread_bps: float = 2.0,
    adv: float = 1e6,
    daily_vol: float = 0.02,
) -> TransactionCostResult:
    """
    Compute round-trip transaction cost.

    Parameters
    ----------
    price : float
        Current price per share.
    shares : float
        Number of shares traded (absolute).
    spread_bps : float
        Half-spread in basis points.
    adv : float
        Average daily volume in shares.
    daily_vol : float
        Daily volatility (for market impact).

    Returns
    -------
    TransactionCostResult
        Breakdown of transaction costs.
    """
    shares = abs(shares)
    if price <= 0 or shares <= 0:
        return TransactionCostResult(0.0, 0.0, 0.0, 0.0)

    trade_value = price * shares

    # Spread cost: 2 x half-spread (round trip)
    spread_cost = trade_value * spread_bps / 10000.0 * 2.0

    # Market impact: 0.1 * sigma_daily * sqrt(V / ADV)
    if adv > 0:
        participation = shares / adv
        impact_frac = IMPACT_COEFFICIENT * daily_vol * np.sqrt(participation)
    else:
        impact_frac = 0.0
    impact_cost = trade_value * impact_frac

    total = spread_cost + impact_cost
    cost_bps_val = total / max(trade_value, 1e-15) * 10000.0

    return TransactionCostResult(
        spread_cost=float(spread_cost),
        impact_cost=float(impact_cost),
        total_cost=float(total),
        cost_bps=float(cost_bps_val),
    )


def get_spread_bps(asset_class: str) -> float:
    """
    Get default spread for an asset class.

    Parameters
    ----------
    asset_class : str
        One of: large_cap, mid_cap, small_cap, crypto, metals.

    Returns
    -------
    float
        Half-spread in basis points.
    """
    return SPREAD_BPS.get(asset_class, 5.0)


def compute_cost_adjusted_returns(
    gross_returns: np.ndarray,
    positions: np.ndarray,
    spread_bps: float = 2.0,
) -> np.ndarray:
    """
    Compute net returns after transaction costs.

    Costs are incurred on position changes (turnover).

    Parameters
    ----------
    gross_returns : ndarray, shape (T,)
        Gross strategy returns.
    positions : ndarray, shape (T,)
        Position fractions at each timestep.
    spread_bps : float
        Half-spread in basis points.

    Returns
    -------
    ndarray, shape (T,)
        Net returns after subtracting cost of position changes.
    """
    gross_returns = np.asarray(gross_returns, dtype=float)
    positions = np.asarray(positions, dtype=float)
    n = len(gross_returns)

    net_returns = gross_returns.copy()
    cost_per_unit = 2.0 * spread_bps / 10000.0  # Round-trip spread

    for t in range(1, n):
        turnover = abs(positions[t] - positions[t - 1])
        cost = turnover * cost_per_unit
        net_returns[t] -= cost

    # First period: assume entry from flat
    net_returns[0] -= abs(positions[0]) * cost_per_unit

    return net_returns


# ---------------------------------------------------------------------------
# Story 14.2: Turnover-Penalized Signal Generation
# ---------------------------------------------------------------------------

# Default dead-zone multiplier (trades suppressed if |delta| < multiplier * cost)
DEAD_ZONE_MULTIPLIER = 1.0


def turnover_filter(
    signal: float,
    prev_signal: float,
    cost_threshold: float,
) -> float:
    """
    Suppress signal flip if change is less than cost threshold.

    Parameters
    ----------
    signal : float
        Current raw signal.
    prev_signal : float
        Previous signal.
    cost_threshold : float
        Minimum signal change to justify trading (break-even threshold).

    Returns
    -------
    float
        Filtered signal (either new signal or previous signal if change too small).
    """
    delta = abs(signal - prev_signal)
    if delta < cost_threshold:
        return prev_signal
    return signal


def turnover_filter_array(
    signals: np.ndarray,
    cost_threshold: float,
) -> np.ndarray:
    """
    Apply turnover filter to a signal series.

    Parameters
    ----------
    signals : ndarray, shape (T,)
        Raw signal series.
    cost_threshold : float
        Minimum signal change to justify trading.

    Returns
    -------
    ndarray, shape (T,)
        Filtered signal series with reduced turnover.
    """
    signals = np.asarray(signals, dtype=float)
    n = len(signals)
    if n == 0:
        return signals.copy()

    filtered = np.zeros(n)
    filtered[0] = signals[0]

    for t in range(1, n):
        filtered[t] = turnover_filter(signals[t], filtered[t - 1], cost_threshold)

    return filtered


def compute_break_even_threshold(spread_bps: float, impact_bps: float = 0.0) -> float:
    """
    Compute break-even cost threshold for signal changes.

    A trade must generate expected return > round-trip cost to be worthwhile.

    Parameters
    ----------
    spread_bps : float
        Half-spread in basis points.
    impact_bps : float
        Expected market impact in basis points.

    Returns
    -------
    float
        Break-even threshold (fraction, not bps).
    """
    round_trip_bps = 2.0 * spread_bps + impact_bps
    return round_trip_bps / 10000.0


def compute_turnover(positions: np.ndarray) -> float:
    """
    Compute average daily turnover from position series.

    Parameters
    ----------
    positions : ndarray, shape (T,)
        Position fractions.

    Returns
    -------
    float
        Average daily turnover (sum of |delta_position|).
    """
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 2:
        return 0.0
    deltas = np.abs(np.diff(positions))
    return float(np.mean(deltas))


def compute_turnover_reduction(
    raw_positions: np.ndarray,
    filtered_positions: np.ndarray,
) -> float:
    """
    Compute turnover reduction percentage.

    Parameters
    ----------
    raw_positions : ndarray
        Unfiltered positions.
    filtered_positions : ndarray
        Filtered positions.

    Returns
    -------
    float
        Reduction in turnover (0 to 1).
    """
    raw_to = compute_turnover(raw_positions)
    filt_to = compute_turnover(filtered_positions)
    if raw_to < 1e-15:
        return 0.0
    return float((raw_to - filt_to) / raw_to)


# ---------------------------------------------------------------------------
# Story 14.3: Optimal Rebalancing Frequency
# ---------------------------------------------------------------------------

# Default frequency options (in trading days)
DEFAULT_FREQ_OPTIONS = [1, 3, 5, 10, 21]


@dataclass
class RebalanceResult:
    """Result of optimal rebalancing frequency analysis."""
    optimal_freq: int           # Best rebalancing frequency (days)
    net_sharpe_at_optimal: float  # Net Sharpe at optimal frequency
    freq_options: List[int]     # Frequencies tested
    net_sharpes: List[float]    # Net Sharpe at each frequency
    gross_sharpes: List[float]  # Gross Sharpe at each frequency
    n_folds: int                # Number of walk-forward folds


def _resample_signals(
    signals: np.ndarray,
    freq: int,
) -> np.ndarray:
    """
    Resample signals to a lower frequency (hold signal for freq days).

    Parameters
    ----------
    signals : ndarray, shape (T,)
        Daily signals.
    freq : int
        Rebalancing frequency in days.

    Returns
    -------
    ndarray, shape (T,)
        Resampled signals (held constant between rebalance dates).
    """
    signals = np.asarray(signals, dtype=float)
    n = len(signals)
    resampled = np.zeros(n)

    current_signal = signals[0]
    for t in range(n):
        if t % freq == 0:
            current_signal = signals[t]
        resampled[t] = current_signal

    return resampled


def _compute_net_sharpe(
    returns: np.ndarray,
    signals: np.ndarray,
    spread_bps: float,
    annualization: float = 252.0,
) -> float:
    """
    Compute net Sharpe ratio after transaction costs.

    Parameters
    ----------
    returns : ndarray
        Realized returns.
    signals : ndarray
        Position signals.
    spread_bps : float
        Half-spread in basis points.
    annualization : float
        Annualization factor.

    Returns
    -------
    float
        Net annualized Sharpe ratio.
    """
    gross_pnl = signals * returns
    net_pnl = compute_cost_adjusted_returns(gross_pnl, signals, spread_bps)

    if len(net_pnl) < 2:
        return 0.0
    std = np.std(net_pnl, ddof=1)
    if std < 1e-15:
        return 0.0
    return float(np.mean(net_pnl) / std * np.sqrt(annualization))


def optimal_rebalance_freq(
    returns: np.ndarray,
    signals: np.ndarray,
    spread_bps: float = 5.0,
    freq_options: Optional[List[int]] = None,
    train_days: int = 252,
    test_days: int = 21,
) -> RebalanceResult:
    """
    Find optimal rebalancing frequency via walk-forward net Sharpe maximization.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        Realized returns.
    signals : ndarray, shape (T,)
        Raw daily signals.
    spread_bps : float
        Half-spread in basis points.
    freq_options : list of int or None
        Rebalancing frequencies to test.
    train_days : int
        Training window.
    test_days : int
        Test window.

    Returns
    -------
    RebalanceResult
        Optimal frequency with diagnostics.
    """
    if freq_options is None:
        freq_options = list(DEFAULT_FREQ_OPTIONS)

    returns = np.asarray(returns, dtype=float)
    signals = np.asarray(signals, dtype=float)
    n = len(returns)
    n_freqs = len(freq_options)

    fold_net_sharpes = [[] for _ in range(n_freqs)]
    fold_gross_sharpes = [[] for _ in range(n_freqs)]
    n_folds = 0

    start = train_days
    while start + test_days <= n:
        test_r = returns[start:start + test_days]
        test_s = signals[start:start + test_days]

        for j, freq in enumerate(freq_options):
            resampled = _resample_signals(test_s, freq)
            # Gross Sharpe
            gross_pnl = resampled * test_r
            if len(gross_pnl) > 1 and np.std(gross_pnl, ddof=1) > 1e-15:
                gs = float(np.mean(gross_pnl) / np.std(gross_pnl, ddof=1) * np.sqrt(252))
            else:
                gs = 0.0
            fold_gross_sharpes[j].append(gs)
            # Net Sharpe
            ns = _compute_net_sharpe(test_r, resampled, spread_bps)
            fold_net_sharpes[j].append(ns)

        n_folds += 1
        start += test_days

    if n_folds == 0:
        return RebalanceResult(
            optimal_freq=1,
            net_sharpe_at_optimal=0.0,
            freq_options=freq_options,
            net_sharpes=[0.0] * n_freqs,
            gross_sharpes=[0.0] * n_freqs,
            n_folds=0,
        )

    avg_net = [float(np.mean(s)) for s in fold_net_sharpes]
    avg_gross = [float(np.mean(s)) for s in fold_gross_sharpes]

    best_idx = int(np.argmax(avg_net))

    return RebalanceResult(
        optimal_freq=freq_options[best_idx],
        net_sharpe_at_optimal=avg_net[best_idx],
        freq_options=freq_options,
        net_sharpes=avg_net,
        gross_sharpes=avg_gross,
        n_folds=n_folds,
    )
