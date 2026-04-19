"""
Story 4.2 + 4.3: CRPS Stacking Optimizer with Temporal Forgetting
===================================================================

Story 4.2: Convex optimizer that minimizes combined CRPS subject to simplex
constraints to find optimal BMA stacking weights.

Story 4.3: Temporal CRPS stacking with exponential forgetting. Recent model
performance is weighted more heavily via lambda^(T-t) decay, enabling
adaptation to changing market regimes.

Given a T x M matrix of per-observation CRPS scores from M models,
finds w* = argmin_w sum_t CRPS(sum_m w_m * F_{m,t}, r_t) s.t. w >= 0, sum(w) = 1.

In practice, the stacking objective uses the linear pool of CRPS scores:
  combined_crps(w) = sum_t sum_m w_m * crps_{m,t}

This is exact when models contribute additively (Yao et al. 2018).
"""
import os
import sys
import logging
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

logger = logging.getLogger(__name__)


@dataclass
class StackingResult:
    """Result of CRPS stacking optimization."""
    weights: np.ndarray          # M-dim simplex weights
    combined_crps: float         # Weighted mean CRPS
    n_models: int
    n_timesteps: int
    converged: bool
    n_iterations: int
    l1_distance_from_init: float  # L1 distance from initial weights


def crps_stacking_weights(
    crps_matrix: np.ndarray,
    bic_weights: Optional[np.ndarray] = None,
    min_weight: float = 0.0,
) -> StackingResult:
    """
    Find optimal stacking weights minimizing combined CRPS.

    Parameters
    ----------
    crps_matrix : ndarray, shape (T, M)
        Per-observation CRPS for each model. crps_matrix[t, m] = CRPS of model m
        at time t. Lower = better.
    bic_weights : ndarray, shape (M,), optional
        BIC-based weights for warm-starting. If None, uses uniform 1/M.
    min_weight : float
        Minimum weight per model (default 0.0 for full pruning).

    Returns
    -------
    StackingResult
    """
    T, M = crps_matrix.shape

    if M == 0:
        return StackingResult(
            weights=np.array([]),
            combined_crps=float('inf'),
            n_models=0, n_timesteps=T,
            converged=False, n_iterations=0,
            l1_distance_from_init=0.0,
        )

    if M == 1:
        return StackingResult(
            weights=np.array([1.0]),
            combined_crps=float(np.mean(crps_matrix[:, 0])),
            n_models=1, n_timesteps=T,
            converged=True, n_iterations=0,
            l1_distance_from_init=0.0,
        )

    # Warm start
    if bic_weights is not None:
        w0 = np.clip(bic_weights, min_weight, None)
        w0 = w0 / w0.sum()
    else:
        w0 = np.ones(M) / M

    # Normalize CRPS matrix for numerical stability
    crps_scale = np.mean(np.abs(crps_matrix)) + 1e-15

    def objective(w):
        """Mean combined CRPS: (1/T) * sum_t sum_m w_m * crps_{m,t}."""
        return float(np.mean(crps_matrix @ w)) / crps_scale

    def gradient(w):
        """Gradient: (1/T) * sum_t crps_{m,t} for each m."""
        return np.mean(crps_matrix, axis=0) / crps_scale

    # Simplex constraints: w >= min_weight, sum(w) = 1
    bounds = [(min_weight, 1.0)] * M
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    result = minimize(
        objective,
        w0,
        method='SLSQP',
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-12},
    )

    w_opt = result.x
    # Ensure simplex (numerical cleanup)
    w_opt = np.clip(w_opt, min_weight, None)
    w_opt = w_opt / w_opt.sum()

    combined = float(np.mean(crps_matrix @ w_opt))
    l1_dist = float(np.sum(np.abs(w_opt - w0)))

    return StackingResult(
        weights=w_opt,
        combined_crps=combined,
        n_models=M,
        n_timesteps=T,
        converged=result.success,
        n_iterations=result.nit,
        l1_distance_from_init=l1_dist,
    )


def compute_bic_weights(bic_values: np.ndarray) -> np.ndarray:
    """
    Convert BIC values to BMA weights: w_m ~ exp(-0.5 * delta_BIC_m).

    Parameters
    ----------
    bic_values : ndarray, shape (M,)
        BIC for each model (lower = better).

    Returns
    -------
    ndarray, shape (M,)
        Normalized BIC weights on the simplex.
    """
    delta_bic = bic_values - np.min(bic_values)
    log_w = -0.5 * delta_bic
    log_w -= np.max(log_w)  # Shift for numerical stability
    w = np.exp(log_w)
    return w / w.sum()


def combined_crps_score(crps_matrix: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute mean combined CRPS for given weights.

    Parameters
    ----------
    crps_matrix : ndarray, shape (T, M)
    weights : ndarray, shape (M,)

    Returns
    -------
    float : Mean combined CRPS.
    """
    return float(np.mean(crps_matrix @ weights))


# ---------------------------------------------------------------------------
# Story 4.3: Temporal CRPS Stacking with Exponential Forgetting
# ---------------------------------------------------------------------------

@dataclass
class TemporalStackingResult:
    """Result of temporal CRPS stacking optimization."""
    weights: np.ndarray               # M-dim simplex weights (final snapshot)
    weight_path: np.ndarray           # (n_windows, M) weight trajectory
    timestamps: np.ndarray            # (n_windows,) end-index of each window
    combined_crps: float              # Temporally-weighted combined CRPS
    n_models: int
    n_timesteps: int
    lambda_decay: float
    half_life_days: float
    converged: bool
    monthly_l1_turnover: float        # Mean monthly L1 weight change
    weight_shift_speed: Optional[float]  # Days to detect regime shift (None if no shift)


# Default constants
TEMPORAL_LAMBDA_DEFAULT = 0.995
TEMPORAL_MIN_WEIGHT = 0.0
TEMPORAL_WINDOW_STEP = 21           # Re-estimate weights monthly
TEMPORAL_MIN_HISTORY = 63           # Need 3 months before first estimate


def _compute_exponential_weights(T: int, lambda_decay: float) -> np.ndarray:
    """
    Compute normalized exponential decay weights lambda^(T-1-t) for t=0..T-1.

    Most recent observation (t=T-1) gets weight 1, oldest (t=0) gets lambda^(T-1).
    """
    exponents = np.arange(T - 1, -1, -1, dtype=np.float64)  # T-1, T-2, ..., 0
    raw = lambda_decay ** exponents
    return raw / raw.sum()


def temporal_crps_stacking(
    crps_matrix: np.ndarray,
    bic_weights: Optional[np.ndarray] = None,
    lambda_decay: float = TEMPORAL_LAMBDA_DEFAULT,
    min_weight: float = TEMPORAL_MIN_WEIGHT,
    window_step: int = TEMPORAL_WINDOW_STEP,
    min_history: int = TEMPORAL_MIN_HISTORY,
) -> TemporalStackingResult:
    """
    Temporal CRPS stacking with exponential forgetting.

    Periodically re-optimizes model weights using exponentially-weighted CRPS,
    giving more influence to recent observations. This enables the system to
    adapt to changing regimes while maintaining stable, smooth weight transitions.

    Parameters
    ----------
    crps_matrix : ndarray, shape (T, M)
        Per-observation CRPS for each model. crps_matrix[t, m] = CRPS of model m
        at time t. Lower = better.
    bic_weights : ndarray, shape (M,), optional
        BIC-based weights for warm-starting the first window. If None, uniform.
    lambda_decay : float
        Exponential decay factor. lambda=0.995 gives half-life ~138 days.
        Must be in (0, 1].
    min_weight : float
        Minimum weight per model (default 0.0).
    window_step : int
        Number of observations between weight re-estimations (default 21 = monthly).
    min_history : int
        Minimum observations before first weight estimation (default 63 = 3 months).

    Returns
    -------
    TemporalStackingResult
    """
    T, M = crps_matrix.shape

    half_life = -np.log(2) / np.log(lambda_decay) if lambda_decay < 1.0 else float('inf')

    if M == 0:
        return TemporalStackingResult(
            weights=np.array([]),
            weight_path=np.empty((0, 0)),
            timestamps=np.array([], dtype=int),
            combined_crps=float('inf'),
            n_models=0, n_timesteps=T,
            lambda_decay=lambda_decay,
            half_life_days=half_life,
            converged=False,
            monthly_l1_turnover=0.0,
            weight_shift_speed=None,
        )

    if M == 1:
        w = np.array([1.0])
        n_windows = max(1, (T - min_history) // window_step + 1) if T >= min_history else 0
        wp = np.ones((max(n_windows, 1), 1))
        ts = np.array([min(min_history + i * window_step, T) for i in range(max(n_windows, 1))], dtype=int)
        return TemporalStackingResult(
            weights=w,
            weight_path=wp,
            timestamps=ts,
            combined_crps=float(np.mean(crps_matrix[:, 0])),
            n_models=1, n_timesteps=T,
            lambda_decay=lambda_decay,
            half_life_days=half_life,
            converged=True,
            monthly_l1_turnover=0.0,
            weight_shift_speed=None,
        )

    # Initial weights (warm start)
    if bic_weights is not None:
        w_prev = np.clip(bic_weights, min_weight, None)
        w_prev = w_prev / w_prev.sum()
    else:
        w_prev = np.ones(M) / M

    # Collect weight snapshots at each re-estimation point
    weight_snapshots = []
    snapshot_times = []

    t = min_history
    while t <= T:
        end_t = t
        crps_window = crps_matrix[:end_t, :]  # Use all data up to this point
        T_win = crps_window.shape[0]

        # Exponential decay weights for this window
        time_weights = _compute_exponential_weights(T_win, lambda_decay)

        # Weighted CRPS: each observation scaled by its temporal weight
        # weighted_crps[t_i, m] = time_weights[t_i] * crps_window[t_i, m]
        weighted_crps = crps_window * time_weights[:, np.newaxis]

        # Scale for numerical stability
        crps_scale = np.sum(np.abs(weighted_crps)) + 1e-15

        def objective(w, _wc=weighted_crps, _s=crps_scale):
            return float(np.sum(_wc @ w)) / _s

        def gradient(w, _wc=weighted_crps, _s=crps_scale):
            return np.sum(_wc, axis=0) / _s

        bounds = [(min_weight, 1.0)] * M
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        result = minimize(
            objective,
            w_prev,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12},
        )

        w_opt = np.clip(result.x, min_weight, None)
        w_opt = w_opt / w_opt.sum()

        weight_snapshots.append(w_opt.copy())
        snapshot_times.append(end_t)

        # Warm-start next window from current solution
        w_prev = w_opt.copy()

        t += window_step

    if len(weight_snapshots) == 0:
        # Not enough history for even one estimation
        w_final = np.ones(M) / M if bic_weights is None else bic_weights.copy()
        return TemporalStackingResult(
            weights=w_final,
            weight_path=w_final.reshape(1, -1),
            timestamps=np.array([T], dtype=int),
            combined_crps=float(np.mean(crps_matrix @ w_final)),
            n_models=M, n_timesteps=T,
            lambda_decay=lambda_decay,
            half_life_days=half_life,
            converged=False,
            monthly_l1_turnover=0.0,
            weight_shift_speed=None,
        )

    weight_path = np.array(weight_snapshots)  # (n_windows, M)
    timestamps = np.array(snapshot_times, dtype=int)

    # Final weights
    w_final = weight_path[-1]

    # Combined CRPS using final temporally-weighted objective
    time_weights_full = _compute_exponential_weights(T, lambda_decay)
    combined_crps = float(np.sum((crps_matrix @ w_final) * time_weights_full))

    # Monthly L1 turnover: mean L1 change between consecutive snapshots
    if len(weight_path) > 1:
        l1_changes = np.sum(np.abs(np.diff(weight_path, axis=0)), axis=1)
        monthly_l1_turnover = float(np.mean(l1_changes))
    else:
        monthly_l1_turnover = 0.0

    # Weight shift speed: detect if there's a large regime shift
    # Defined as: how many snapshots until weights shift by > 0.10 L1
    weight_shift_speed = _detect_weight_shift_speed(
        weight_path, timestamps, l1_threshold=0.10
    )

    return TemporalStackingResult(
        weights=w_final,
        weight_path=weight_path,
        timestamps=timestamps,
        combined_crps=combined_crps,
        n_models=M, n_timesteps=T,
        lambda_decay=lambda_decay,
        half_life_days=half_life,
        converged=True,
        monthly_l1_turnover=monthly_l1_turnover,
        weight_shift_speed=weight_shift_speed,
    )


def _detect_weight_shift_speed(
    weight_path: np.ndarray,
    timestamps: np.ndarray,
    l1_threshold: float = 0.10,
) -> Optional[float]:
    """
    Detect how quickly weights shift during regime transitions.

    Returns the number of trading days for the largest cumulative L1 shift
    to exceed threshold, or None if no significant shift detected.
    """
    if len(weight_path) < 2:
        return None

    n = len(weight_path)
    best_speed = None

    for start_idx in range(n - 1):
        cumulative_l1 = 0.0
        for end_idx in range(start_idx + 1, n):
            cumulative_l1 = float(np.sum(np.abs(
                weight_path[end_idx] - weight_path[start_idx]
            )))
            if cumulative_l1 >= l1_threshold:
                days = int(timestamps[end_idx] - timestamps[start_idx])
                if best_speed is None or days < best_speed:
                    best_speed = float(days)
                break

    return best_speed
