"""
Epic 25: Rolling Walk-Forward Calibration Engine

Provides honest out-of-sample validation via walk-forward:
1. Walk-forward backtest framework (train/test splits)
2. Expanding window with decay weighting
3. Overfitting detector via IS-OOS divergence

References:
- Tashman (2000): Out-of-sample tests of forecasting accuracy
- Bergmeir & Benitez (2012): On the use of cross-validation for time series
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Generator

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Story 25.1: Walk-forward
DEFAULT_TRAIN_WINDOW = 504   # ~2 years of trading days
DEFAULT_STEP_SIZE = 21       # ~1 month
MIN_TRAIN_WINDOW = 63        # ~3 months minimum
MIN_TEST_SIZE = 1            # At least 1 observation in test

# Story 25.2: Expanding window
DEFAULT_LAMBDA_DECAY = 0.998  # Half-life ~347 days
MIN_LAMBDA = 0.9
MAX_LAMBDA = 1.0

# Story 25.3: Overfitting detection
DEFAULT_OVERFIT_THRESHOLD = 0.25  # 25% relative IS-OOS gap
DEFAULT_HIT_RATE_THRESHOLD = 0.05  # 5% absolute gap
OVERFIT_WEIGHT_PENALTY = 0.5  # Reduce BMA weight by 50%


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardFold:
    """A single walk-forward fold."""
    fold_index: int
    train_start: int
    train_end: int   # exclusive
    test_start: int
    test_end: int    # exclusive
    train_size: int
    test_size: int


@dataclass
class WalkForwardResult:
    """Result of walk-forward backtest."""
    folds: List[WalkForwardFold]
    n_folds: int
    is_metrics: np.ndarray   # In-sample metric per fold
    oos_metrics: np.ndarray  # Out-of-sample metric per fold
    is_oos_gaps: np.ndarray  # (IS - OOS) / max(|OOS|, eps) per fold
    mean_is_metric: float
    mean_oos_metric: float
    mean_gap: float
    train_window: int
    step_size: int
    total_observations: int
    no_leakage_verified: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "mean_is_metric": float(self.mean_is_metric),
            "mean_oos_metric": float(self.mean_oos_metric),
            "mean_gap": float(self.mean_gap),
            "train_window": self.train_window,
            "step_size": self.step_size,
            "total_observations": self.total_observations,
            "no_leakage_verified": self.no_leakage_verified,
        }


@dataclass
class ExpandingWindowResult:
    """Result of expanding window training with decay."""
    train_indices: np.ndarray
    weights: np.ndarray          # Decay weights for each training observation
    effective_sample_size: float  # Sum of squared weights / (sum of weights)^2
    lambda_decay: float
    half_life_days: float
    n_train: int
    current_time: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "effective_sample_size": float(self.effective_sample_size),
            "lambda_decay": float(self.lambda_decay),
            "half_life_days": float(self.half_life_days),
            "n_train": self.n_train,
            "current_time": self.current_time,
        }


@dataclass
class OverfitDetectionResult:
    """Result of overfitting detection."""
    is_overfit: bool
    crps_gap_relative: float     # (IS_CRPS - OOS_CRPS) / max(|OOS_CRPS|, eps)
    hit_rate_gap_absolute: float # IS_hit_rate - OOS_hit_rate
    crps_overfit: bool           # IS CRPS < OOS by > threshold
    hit_rate_overfit: bool       # IS hit rate > OOS by > threshold
    recommended_weight_penalty: float
    severity: str                # "NONE", "MILD", "SEVERE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_overfit": self.is_overfit,
            "crps_gap_relative": float(self.crps_gap_relative),
            "hit_rate_gap_absolute": float(self.hit_rate_gap_absolute),
            "crps_overfit": self.crps_overfit,
            "hit_rate_overfit": self.hit_rate_overfit,
            "recommended_weight_penalty": float(self.recommended_weight_penalty),
            "severity": self.severity,
        }


# ---------------------------------------------------------------------------
# Story 25.1: Walk-Forward Backtest Framework
# ---------------------------------------------------------------------------

def walk_forward_splits(
    n_obs: int,
    train_window: int = DEFAULT_TRAIN_WINDOW,
    step: int = DEFAULT_STEP_SIZE,
) -> List[WalkForwardFold]:
    """
    Generate walk-forward train/test splits.

    For each fold:
      - Train on [train_start, train_end)
      - Test on [test_start, test_end)
      - No overlap: test_start = train_end

    Parameters
    ----------
    n_obs : int
        Total number of observations.
    train_window : int
        Size of training window.
    step : int
        Step size (test window size).

    Returns
    -------
    List[WalkForwardFold]
        List of train/test splits.

    Raises
    ------
    ValueError
        If n_obs is too small for even one fold.
    """
    train_window = max(int(train_window), MIN_TRAIN_WINDOW)
    step = max(int(step), 1)

    if n_obs < train_window + MIN_TEST_SIZE:
        raise ValueError(
            f"Need >= {train_window + MIN_TEST_SIZE} observations, got {n_obs}"
        )

    folds = []
    fold_idx = 0
    train_start = 0

    while train_start + train_window + MIN_TEST_SIZE <= n_obs:
        train_end = train_start + train_window
        test_start = train_end
        test_end = min(test_start + step, n_obs)

        folds.append(WalkForwardFold(
            fold_index=fold_idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_size=train_end - train_start,
            test_size=test_end - test_start,
        ))

        train_start += step
        fold_idx += 1

    return folds


def walk_forward_backtest(
    returns: np.ndarray,
    vol: np.ndarray,
    train_window: int = DEFAULT_TRAIN_WINDOW,
    step: int = DEFAULT_STEP_SIZE,
    metric_fn: Optional[object] = None,
) -> WalkForwardResult:
    """
    Run walk-forward backtest on returns/vol data.

    For each fold, computes IS and OOS metrics using the provided
    metric function. Default metric is RMSE of returns vs mean forecast.

    Parameters
    ----------
    returns : array-like, shape (T,)
        Return series.
    vol : array-like, shape (T,)
        Volatility series.
    train_window : int
        Training window size in days.
    step : int
        Step size (test period) in days.
    metric_fn : callable, optional
        Function(train_returns, train_vol, test_returns, test_vol) -> (is_metric, oos_metric).
        If None, uses simple mean-forecast RMSE.

    Returns
    -------
    WalkForwardResult
        Walk-forward metrics and fold details.
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    vol = np.asarray(vol, dtype=np.float64).ravel()

    n_obs = len(returns)
    if len(vol) != n_obs:
        raise ValueError(f"returns and vol must have same length, got {n_obs} and {len(vol)}")

    folds = walk_forward_splits(n_obs, train_window, step)

    if len(folds) == 0:
        raise ValueError("No walk-forward folds generated. Need more data.")

    is_metrics = np.zeros(len(folds))
    oos_metrics = np.zeros(len(folds))

    for i, fold in enumerate(folds):
        train_ret = returns[fold.train_start:fold.train_end]
        train_vol = vol[fold.train_start:fold.train_end]
        test_ret = returns[fold.test_start:fold.test_end]
        test_vol = vol[fold.test_start:fold.test_end]

        if metric_fn is not None:
            is_m, oos_m = metric_fn(train_ret, train_vol, test_ret, test_vol)
        else:
            # Default: RMSE using training mean as forecast
            mu_hat = np.mean(train_ret)
            is_m = np.sqrt(np.mean((train_ret - mu_hat) ** 2))
            oos_m = np.sqrt(np.mean((test_ret - mu_hat) ** 2))

        is_metrics[i] = is_m
        oos_metrics[i] = oos_m

    # IS-OOS gap: positive means IS is better (lower) = overfitting signal
    # For RMSE-like metrics where lower is better:
    # gap = (OOS - IS) / max(|OOS|, eps) -> positive = overfit
    eps = 1e-10
    is_oos_gaps = (oos_metrics - is_metrics) / np.maximum(np.abs(oos_metrics), eps)

    # Verify no data leakage: each fold's test starts after train ends
    no_leakage = all(
        fold.test_start >= fold.train_end for fold in folds
    )

    return WalkForwardResult(
        folds=folds,
        n_folds=len(folds),
        is_metrics=is_metrics,
        oos_metrics=oos_metrics,
        is_oos_gaps=is_oos_gaps,
        mean_is_metric=float(np.mean(is_metrics)),
        mean_oos_metric=float(np.mean(oos_metrics)),
        mean_gap=float(np.mean(is_oos_gaps)),
        train_window=train_window,
        step_size=step,
        total_observations=n_obs,
        no_leakage_verified=no_leakage,
    )


# ---------------------------------------------------------------------------
# Story 25.2: Expanding Window with Decay Weighting
# ---------------------------------------------------------------------------

def expanding_window_train(
    returns: np.ndarray,
    vol: np.ndarray,
    t: int,
    lambda_decay: float = DEFAULT_LAMBDA_DECAY,
) -> ExpandingWindowResult:
    """
    Expanding window training data with exponential decay weighting.

    Uses all data from [0, t-1] with decay weights lambda^(t-1-s) for
    observation s. More recent data gets higher weight.

    Parameters
    ----------
    returns : array-like, shape (T,)
        Full return series.
    vol : array-like, shape (T,)
        Full volatility series.
    t : int
        Current time index. Train on [0, t-1].
    lambda_decay : float
        Decay factor. Default 0.998 (half-life ~347 days).

    Returns
    -------
    ExpandingWindowResult
        Training indices, weights, effective sample size.

    Raises
    ------
    ValueError
        If t < 1 (no training data).
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    vol = np.asarray(vol, dtype=np.float64).ravel()

    if t < 1:
        raise ValueError(f"Need t >= 1 for training data, got t={t}")

    t = min(t, len(returns))

    # Clamp lambda
    lambda_decay = float(np.clip(lambda_decay, MIN_LAMBDA, MAX_LAMBDA))

    # Training indices: [0, t-1]
    train_indices = np.arange(t)

    # Decay weights: lambda^(t-1-s) for observation s
    # Most recent (s = t-1) gets weight 1.0
    ages = (t - 1) - train_indices  # age of each observation
    weights = lambda_decay ** ages

    # Normalize weights to sum to 1
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        normalized_weights = weights / weight_sum
    else:
        normalized_weights = np.ones(t) / t

    # Effective sample size: (sum w)^2 / sum(w^2)
    ess = weight_sum ** 2 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else t

    # Half-life: lambda^h = 0.5 => h = ln(0.5) / ln(lambda)
    if lambda_decay < 1.0:
        half_life = np.log(0.5) / np.log(lambda_decay)
    else:
        half_life = float('inf')

    return ExpandingWindowResult(
        train_indices=train_indices,
        weights=normalized_weights,
        effective_sample_size=float(ess),
        lambda_decay=lambda_decay,
        half_life_days=float(half_life),
        n_train=t,
        current_time=t,
    )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted mean."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(weights)
    if w_sum == 0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / w_sum)


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted variance (frequency weights)."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    w_sum = np.sum(weights)
    if w_sum == 0:
        return float(np.var(values))
    mu = np.sum(values * weights) / w_sum
    return float(np.sum(weights * (values - mu) ** 2) / w_sum)


# ---------------------------------------------------------------------------
# Story 25.3: Overfitting Detector
# ---------------------------------------------------------------------------

def detect_overfitting(
    is_metrics: np.ndarray,
    oos_metrics: np.ndarray,
    threshold: float = DEFAULT_OVERFIT_THRESHOLD,
    is_hit_rate: Optional[float] = None,
    oos_hit_rate: Optional[float] = None,
    hit_rate_threshold: float = DEFAULT_HIT_RATE_THRESHOLD,
) -> OverfitDetectionResult:
    """
    Detect overfitting via IS-OOS metric divergence.

    Checks two conditions:
    1. CRPS gap: IS CRPS is better (lower) than OOS by > threshold
    2. Hit rate gap: IS hit rate higher than OOS by > hit_rate_threshold

    Parameters
    ----------
    is_metrics : array-like
        In-sample metrics per fold (e.g., CRPS, RMSE). Lower is better.
    oos_metrics : array-like
        Out-of-sample metrics per fold.
    threshold : float
        Relative threshold for CRPS gap. Default 0.25 (25%).
    is_hit_rate : float, optional
        In-sample hit rate.
    oos_hit_rate : float, optional
        Out-of-sample hit rate.
    hit_rate_threshold : float
        Absolute threshold for hit rate gap. Default 0.05 (5%).

    Returns
    -------
    OverfitDetectionResult
        Overfitting flag and diagnostics.
    """
    is_arr = np.asarray(is_metrics, dtype=np.float64).ravel()
    oos_arr = np.asarray(oos_metrics, dtype=np.float64).ravel()

    # Use means across folds
    mean_is = float(np.mean(is_arr))
    mean_oos = float(np.mean(oos_arr))

    # CRPS gap: (OOS - IS) / max(|OOS|, eps)
    # Positive = OOS is worse = overfitting signal
    eps = 1e-10
    crps_gap = (mean_oos - mean_is) / max(abs(mean_oos), eps)
    crps_overfit = crps_gap > threshold

    # Hit rate gap
    if is_hit_rate is not None and oos_hit_rate is not None:
        hr_gap = float(is_hit_rate) - float(oos_hit_rate)
        hr_overfit = hr_gap > hit_rate_threshold
    else:
        hr_gap = 0.0
        hr_overfit = False

    # Overall overfit flag: either condition triggers
    is_overfit = crps_overfit or hr_overfit

    # Severity
    if not is_overfit:
        severity = "NONE"
        penalty = 1.0  # No penalty
    elif crps_gap > 2 * threshold or hr_gap > 2 * hit_rate_threshold:
        severity = "SEVERE"
        penalty = OVERFIT_WEIGHT_PENALTY * 0.5  # 75% reduction
    else:
        severity = "MILD"
        penalty = OVERFIT_WEIGHT_PENALTY  # 50% reduction

    return OverfitDetectionResult(
        is_overfit=is_overfit,
        crps_gap_relative=float(crps_gap),
        hit_rate_gap_absolute=float(hr_gap),
        crps_overfit=crps_overfit,
        hit_rate_overfit=hr_overfit,
        recommended_weight_penalty=penalty,
        severity=severity,
    )


def compute_walk_forward_hit_rate(
    returns: np.ndarray,
    forecasts: np.ndarray,
    train_end: int,
) -> Tuple[float, float]:
    """
    Compute IS and OOS hit rates from walk-forward split.

    Hit rate = fraction of times sign(forecast) == sign(return).

    Parameters
    ----------
    returns : array-like, shape (T,)
        Actual returns.
    forecasts : array-like, shape (T,)
        Forecast returns.
    train_end : int
        Index where training ends and test begins.

    Returns
    -------
    Tuple[float, float]
        (is_hit_rate, oos_hit_rate)
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    forecasts = np.asarray(forecasts, dtype=np.float64).ravel()

    if len(returns) != len(forecasts):
        raise ValueError("returns and forecasts must have same length")

    if train_end <= 0 or train_end >= len(returns):
        raise ValueError(f"train_end must be in (0, {len(returns)}), got {train_end}")

    # IS hit rate
    is_ret = returns[:train_end]
    is_fc = forecasts[:train_end]
    is_correct = np.sign(is_ret) == np.sign(is_fc)
    is_hr = float(np.mean(is_correct)) if len(is_correct) > 0 else 0.5

    # OOS hit rate
    oos_ret = returns[train_end:]
    oos_fc = forecasts[train_end:]
    oos_correct = np.sign(oos_ret) == np.sign(oos_fc)
    oos_hr = float(np.mean(oos_correct)) if len(oos_correct) > 0 else 0.5

    return is_hr, oos_hr
