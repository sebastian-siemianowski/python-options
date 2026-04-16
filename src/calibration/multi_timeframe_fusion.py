"""
Epic 12: Multi-Timeframe Signal Fusion
========================================

Story 12.1: Adaptive Momentum Horizon Weights via OOS Ranking
Story 12.2: Momentum-Mean Reversion Regime Switch
Story 12.3: Cross-Asset Momentum Confirmation

Adapts momentum horizon weights per asset using rolling cross-validation,
switches between momentum and mean-reversion based on variance ratio,
and confirms signals using cross-asset correlation.
"""
import os
import sys
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# ===================================================================
# Constants
# ===================================================================

# Story 12.1: Adaptive Momentum Weights
DEFAULT_LOOKBACKS = [5, 10, 20, 60]
CV_TRAIN_WINDOW = 252              # 1-year train window
CV_TEST_WINDOW = 21                # 1-month test window
CV_MIN_TRAIN = 126                 # Minimum training samples
WEIGHT_SMOOTHING = 0.3             # Exponential smoothing of weights across folds

# Story 12.2: Momentum-MR Regime Switch
VR_WINDOW = 60                     # Window for variance ratio computation
VR_MOMENTUM_THRESHOLD = 1.2        # VR > 1.2 -> momentum
VR_MEAN_REVERT_THRESHOLD = 0.8     # VR < 0.8 -> mean reversion
VR_Q_PERIOD = 5                    # Multi-day returns period for VR
MOMENTUM_REGIME = "MOMENTUM"
MEAN_REVERT_REGIME = "MEAN_REVERT"
NEUTRAL_REGIME = "NEUTRAL"

# Story 12.3: Cross-Asset Confirmation
CORRELATION_WINDOW = 252           # Trailing correlation estimation window
CONFIRMATION_HIGH = 0.5            # High confirmation threshold
CONFIRMATION_LOW = -0.3            # Low/divergent confirmation threshold
CONFIRMATION_BOOST = 0.15          # 15% confidence boost on high confirmation
CONFIRMATION_PENALTY = 0.20        # 20% confidence reduction on divergent signals
MIN_CORRELATION = 0.1              # Minimum |rho| to include in confirmation


# ===================================================================
# Story 12.1: Adaptive Momentum Horizon Weights
# ===================================================================

@dataclass
class AdaptiveMomentumResult:
    """Result of adaptive momentum horizon weight estimation."""
    weights: np.ndarray              # Optimal weights per lookback, shape (n_lookbacks,)
    lookbacks: List[int]             # Lookback periods
    hit_rates_per_horizon: np.ndarray  # OOS hit rate per lookback
    n_folds: int                     # Number of CV folds used
    best_horizon: int                # Lookback with highest hit rate


def _compute_momentum_signal(returns: np.ndarray, lookback: int) -> np.ndarray:
    """Compute momentum signal as cumulative return over lookback period.

    Args:
        returns: Daily returns, shape (n,)
        lookback: Number of days to look back

    Returns:
        Momentum signal, shape (n,). NaN for first (lookback-1) entries.
    """
    n = len(returns)
    signal = np.full(n, np.nan)
    for t in range(lookback, n):
        signal[t] = np.sum(returns[t - lookback:t])
    return signal


def adaptive_momentum_weights(
    returns: np.ndarray,
    lookbacks: Optional[List[int]] = None,
    cv_window: int = CV_TRAIN_WINDOW,
    test_window: int = CV_TEST_WINDOW,
) -> AdaptiveMomentumResult:
    """Estimate asset-specific momentum horizon weights via rolling OOS ranking.

    Uses rolling 1-year train, 1-month test, ranked by directional accuracy.

    Args:
        returns: Daily returns, shape (n,)
        lookbacks: Lookback periods to test (default: [5, 10, 20, 60])
        cv_window: Training window in days
        test_window: Testing window in days

    Returns:
        AdaptiveMomentumResult with optimal weights
    """
    if lookbacks is None:
        lookbacks = DEFAULT_LOOKBACKS

    n = len(returns)
    n_horizons = len(lookbacks)

    # Compute momentum signals for all horizons
    signals = np.zeros((n, n_horizons))
    for i, lb in enumerate(lookbacks):
        signals[:, i] = _compute_momentum_signal(returns, lb)

    # Rolling CV
    fold_hit_rates = []
    max_lb = max(lookbacks)

    start = max_lb + cv_window
    n_folds = 0

    while start + test_window <= n:
        train_end = start
        train_start = start - cv_window
        test_end = start + test_window

        # Compute hit rates per horizon on test set
        hr = np.zeros(n_horizons)
        for i in range(n_horizons):
            pred_signs = np.sign(signals[train_end:test_end, i])
            actual_signs = np.sign(returns[train_end:test_end])

            valid = ~np.isnan(pred_signs) & (pred_signs != 0)
            if valid.sum() > 0:
                hr[i] = np.mean(pred_signs[valid] == actual_signs[valid])
            else:
                hr[i] = 0.5  # No data -> coin flip

        fold_hit_rates.append(hr)
        n_folds += 1
        start += test_window

    if n_folds == 0:
        # Not enough data for CV
        weights = np.ones(n_horizons) / n_horizons
        return AdaptiveMomentumResult(
            weights=weights,
            lookbacks=lookbacks,
            hit_rates_per_horizon=np.full(n_horizons, 0.5),
            n_folds=0,
            best_horizon=lookbacks[0],
        )

    # Average hit rates across folds (with exponential recency weighting)
    fold_hr = np.array(fold_hit_rates)  # (n_folds, n_horizons)
    decay = np.exp(-WEIGHT_SMOOTHING * np.arange(n_folds)[::-1])
    decay /= decay.sum()
    avg_hr = np.dot(decay, fold_hr)

    # Convert hit rates to weights via softmax
    # Shift for numerical stability
    hr_shifted = avg_hr - avg_hr.max()
    exp_hr = np.exp(hr_shifted * 20)  # Temperature scaling
    weights = exp_hr / exp_hr.sum()

    best_idx = np.argmax(avg_hr)

    return AdaptiveMomentumResult(
        weights=weights,
        lookbacks=lookbacks,
        hit_rates_per_horizon=avg_hr,
        n_folds=n_folds,
        best_horizon=lookbacks[best_idx],
    )


def combine_momentum_signals(
    returns: np.ndarray,
    weights: np.ndarray,
    lookbacks: Optional[List[int]] = None,
) -> np.ndarray:
    """Combine momentum signals using adaptive weights.

    Args:
        returns: Daily returns, shape (n,)
        weights: Weights per lookback, shape (n_lookbacks,)
        lookbacks: Lookback periods (default: [5, 10, 20, 60])

    Returns:
        Combined momentum signal, shape (n,)
    """
    if lookbacks is None:
        lookbacks = DEFAULT_LOOKBACKS

    n = len(returns)
    signals = np.zeros((n, len(lookbacks)))
    for i, lb in enumerate(lookbacks):
        signals[:, i] = _compute_momentum_signal(returns, lb)

    # Weighted combination (handle NaN by replacing with 0 and re-normalizing)
    combined = np.zeros(n)
    for t in range(n):
        valid = ~np.isnan(signals[t])
        if valid.sum() > 0:
            w = weights[valid]
            w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
            combined[t] = np.dot(w, signals[t, valid])
        else:
            combined[t] = 0.0

    return combined


# ===================================================================
# Story 12.2: Momentum-Mean Reversion Regime Switch
# ===================================================================

@dataclass
class MomentumMRRegime:
    """Result of momentum/mean-reversion regime classification."""
    regime: str                    # MOMENTUM, MEAN_REVERT, or NEUTRAL
    variance_ratio: float          # Computed VR statistic
    is_momentum: bool              # True if MOMENTUM regime
    is_mean_reverting: bool        # True if MEAN_REVERT regime


def compute_variance_ratio(
    returns: np.ndarray,
    q: int = VR_Q_PERIOD,
) -> float:
    """Compute Lo-MacKinlay variance ratio VR(q).

    VR(q) = Var(r_q) / (q * Var(r_1))

    VR > 1: positive serial correlation (momentum)
    VR < 1: negative serial correlation (mean reversion)
    VR = 1: random walk

    Args:
        returns: Daily returns, shape (n,)
        q: Multi-day period

    Returns:
        Variance ratio statistic
    """
    n = len(returns)
    if n < q + 1:
        return 1.0  # Default to random walk

    # Daily variance
    var_1 = np.var(returns, ddof=1)
    if var_1 < 1e-20:
        return 1.0

    # q-day returns
    r_q = np.array([np.sum(returns[i:i + q]) for i in range(n - q + 1)])
    var_q = np.var(r_q, ddof=1)

    vr = var_q / (q * var_1)
    return float(vr)


def momentum_mr_regime_indicator(
    returns: np.ndarray,
    vol: Optional[np.ndarray] = None,
    window: int = VR_WINDOW,
    q: int = VR_Q_PERIOD,
) -> MomentumMRRegime:
    """Classify current regime as MOMENTUM, MEAN_REVERT, or NEUTRAL.

    Based on trailing variance ratio test.

    Args:
        returns: Daily returns, shape (n,)
        vol: Optional volatility series (unused currently, reserved)
        window: Trailing window for VR computation
        q: Multi-day period for VR

    Returns:
        MomentumMRRegime
    """
    n = len(returns)
    if n < window:
        trailing = returns
    else:
        trailing = returns[-window:]

    vr = compute_variance_ratio(trailing, q)

    if vr > VR_MOMENTUM_THRESHOLD:
        regime = MOMENTUM_REGIME
    elif vr < VR_MEAN_REVERT_THRESHOLD:
        regime = MEAN_REVERT_REGIME
    else:
        regime = NEUTRAL_REGIME

    return MomentumMRRegime(
        regime=regime,
        variance_ratio=vr,
        is_momentum=(regime == MOMENTUM_REGIME),
        is_mean_reverting=(regime == MEAN_REVERT_REGIME),
    )


def apply_regime_signal_weights(
    momentum_signal: float,
    mr_signal: float,
    regime: MomentumMRRegime,
) -> float:
    """Apply regime-dependent signal weighting.

    In MOMENTUM regime: momentum doubled, MR zeroed.
    In MEAN_REVERT regime: MR doubled, momentum zeroed.
    In NEUTRAL: equal weight.

    Args:
        momentum_signal: Momentum signal strength
        mr_signal: Mean-reversion signal strength
        regime: Current regime classification

    Returns:
        Combined signal
    """
    if regime.is_momentum:
        return 2.0 * momentum_signal + 0.0 * mr_signal
    elif regime.is_mean_reverting:
        return 0.0 * momentum_signal + 2.0 * mr_signal
    else:
        return momentum_signal + mr_signal


def compute_variance_ratio_timeseries(
    returns: np.ndarray,
    window: int = VR_WINDOW,
    q: int = VR_Q_PERIOD,
) -> np.ndarray:
    """Compute rolling variance ratio time series.

    Args:
        returns: Daily returns, shape (n,)
        window: Rolling window
        q: Multi-day period

    Returns:
        VR time series, shape (n,). NaN for first (window-1) entries.
    """
    n = len(returns)
    vr_ts = np.full(n, np.nan)
    for t in range(window, n):
        vr_ts[t] = compute_variance_ratio(returns[t - window:t], q)
    return vr_ts


# ===================================================================
# Story 12.3: Cross-Asset Momentum Confirmation
# ===================================================================

@dataclass
class CrossAssetConfirmation:
    """Result of cross-asset momentum confirmation."""
    confirmation_score: float        # Weighted agreement score (-1 to 1)
    confidence_multiplier: float     # Confidence adjustment multiplier
    n_correlated_assets: int         # Number of assets used for confirmation
    is_confirmed: bool               # Score > CONFIRMATION_HIGH
    is_divergent: bool               # Score < CONFIRMATION_LOW


def estimate_correlation_matrix(
    returns_matrix: np.ndarray,
    window: int = CORRELATION_WINDOW,
) -> np.ndarray:
    """Estimate trailing correlation matrix.

    Args:
        returns_matrix: shape (n, n_assets) - daily returns for each asset
        window: Trailing window

    Returns:
        Correlation matrix, shape (n_assets, n_assets)
    """
    n = returns_matrix.shape[0]
    if n < window:
        data = returns_matrix
    else:
        data = returns_matrix[-window:]

    # Handle NaN by pairwise complete observation
    n_assets = data.shape[1]
    corr = np.eye(n_assets)

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            valid = ~(np.isnan(data[:, i]) | np.isnan(data[:, j]))
            if valid.sum() > 10:
                r_ij = np.corrcoef(data[valid, i], data[valid, j])[0, 1]
                if np.isfinite(r_ij):
                    corr[i, j] = r_ij
                    corr[j, i] = r_ij

    return corr


def cross_asset_confirmation(
    target_idx: int,
    momentum_signals: np.ndarray,
    correlation_matrix: np.ndarray,
    min_correlation: float = MIN_CORRELATION,
) -> CrossAssetConfirmation:
    """Compute cross-asset momentum confirmation score.

    Confirmation = weighted average of correlated assets' momentum
    where weights = rho^2 (squared correlation).

    Args:
        target_idx: Index of target asset in the matrix
        momentum_signals: Current momentum signals per asset, shape (n_assets,)
        correlation_matrix: Correlation matrix, shape (n_assets, n_assets)
        min_correlation: Minimum |rho| to include asset

    Returns:
        CrossAssetConfirmation
    """
    n_assets = len(momentum_signals)
    target_signal = momentum_signals[target_idx]

    # Get correlations with target
    rhos = correlation_matrix[target_idx].copy()
    rhos[target_idx] = 0.0  # Exclude self

    # Filter by minimum correlation
    valid = np.abs(rhos) >= min_correlation
    valid[target_idx] = False

    n_correlated = valid.sum()

    if n_correlated == 0:
        return CrossAssetConfirmation(
            confirmation_score=0.0,
            confidence_multiplier=1.0,
            n_correlated_assets=0,
            is_confirmed=False,
            is_divergent=False,
        )

    # Weights = rho^2 (correlation-squared weighting)
    weights = rhos[valid] ** 2
    weights /= weights.sum()

    # Confirmation: weighted average of other assets' momentum signs
    # Normalized to [-1, 1] via sign of target
    other_signals = momentum_signals[valid]

    if abs(target_signal) < 1e-10:
        # No target signal -> no confirmation
        confirmation = 0.0
    else:
        target_sign = np.sign(target_signal)
        # Agreement: positive if others agree with target direction
        agreement = np.sign(other_signals) * target_sign
        confirmation = float(np.dot(weights, agreement))

    # Confidence multiplier
    if confirmation > CONFIRMATION_HIGH:
        multiplier = 1.0 + CONFIRMATION_BOOST
    elif confirmation < CONFIRMATION_LOW:
        multiplier = 1.0 - CONFIRMATION_PENALTY
    else:
        multiplier = 1.0

    return CrossAssetConfirmation(
        confirmation_score=float(confirmation),
        confidence_multiplier=float(multiplier),
        n_correlated_assets=int(n_correlated),
        is_confirmed=(confirmation > CONFIRMATION_HIGH),
        is_divergent=(confirmation < CONFIRMATION_LOW),
    )
