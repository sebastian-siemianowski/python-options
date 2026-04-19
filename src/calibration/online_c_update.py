"""
===============================================================================
ONLINE c UPDATE via Innovation Variance Monitoring (Tune.md Story 2.3)
===============================================================================

Lightweight adaptive observation noise scalar c that self-corrects between
full re-tuning cycles.

THEORY:
    For a Kalman filter with R_t = c * sigma_t^2:
    - Innovation: v_t = r_t - mu_pred_t
    - Normalized innovation variance ratio: v_t^2 / R_t
    - If calibrated: E[v_t^2 / R_t] = 1

    Update rule:
        c_{t+1} = c_t + eta_t * (v_t^2 / R_t - 1)
        eta_t = max(eta_min, eta_init * decay^t)

    This makes c self-correct:
    - ratio > 1 persistently => c too low => increase
    - ratio < 1 persistently => c too high => decrease

INTEGRATION:
    tune.py (batch MLE) -> c_init
    online_c_update.py  -> c_path[t] (time-varying c)
    signals.py          -> uses latest c_t for live filtering
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Learning rate bounds
ONLINE_C_ETA_INIT = 0.02       # Initial learning rate
ONLINE_C_ETA_MIN = 0.001       # Floor for learning rate
ONLINE_C_ETA_DECAY = 0.998     # Per-step multiplicative decay

# c bounds (safety clipping)
ONLINE_C_MIN = 0.1
ONLINE_C_MAX = 10.0

# Tracking quality thresholds
ONLINE_C_TRACKING_TOLERANCE = 0.10  # 10% tracking tolerance


@dataclass
class OnlineCConfig:
    """Configuration for online c update."""
    eta_init: float = ONLINE_C_ETA_INIT
    eta_min: float = ONLINE_C_ETA_MIN
    eta_decay: float = ONLINE_C_ETA_DECAY
    c_min: float = ONLINE_C_MIN
    c_max: float = ONLINE_C_MAX


@dataclass
class OnlineCResult:
    """Result of online c update."""
    c_path: np.ndarray          # Time-varying c values
    eta_path: np.ndarray        # Learning rate at each step
    c_final: float              # Final c value
    ratio_ema: float            # Final EMA of innovation ratio
    c_init: float               # Initial c (from MLE)
    n_steps: int                # Number of steps processed
    converged: bool             # Whether ratio_ema is near 1.0
    tracking_error: float       # |c_final - c_target| / c_target if target provided


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def compute_innovations(
    returns: np.ndarray,
    mu_filtered: np.ndarray,
    phi: float = 1.0,
) -> np.ndarray:
    """
    Compute one-step-ahead prediction innovations.

    v_t = r_t - mu_pred_t where mu_pred_t = phi * mu_{t-1}

    For t=0, use mu_pred = 0 (no prior state).
    """
    N = len(returns)
    innovations = np.empty(N, dtype=np.float64)

    # mu_pred_0 = phi * mu_init; use mu_filtered shifted by 1
    innovations[0] = returns[0]  # mu_pred_0 ~ 0
    if N > 1:
        # mu_pred_t = phi * mu_filtered_{t-1}
        mu_pred = phi * mu_filtered[:-1]
        innovations[1:] = returns[1:] - mu_pred

    return innovations


def run_online_c_update(
    returns: np.ndarray,
    mu_filtered: np.ndarray,
    vol: np.ndarray,
    c_init: float,
    phi: float = 1.0,
    config: Optional[OnlineCConfig] = None,
    c_target: Optional[float] = None,
) -> OnlineCResult:
    """
    Run online c update on a returns series.

    Parameters
    ----------
    returns : array
        Log returns.
    mu_filtered : array
        Filtered state (mu) from Kalman filter with c_init.
    vol : array
        EWMA volatility (sigma, not sigma^2).
    c_init : float
        Initial c value (from MLE batch tuning).
    phi : float
        AR(1) persistence parameter.
    config : OnlineCConfig, optional
        Configuration. Uses defaults if None.
    c_target : float, optional
        Target c for tracking error computation (e.g., regime-conditional c).

    Returns
    -------
    OnlineCResult
    """
    if config is None:
        config = OnlineCConfig()

    returns = np.asarray(returns, dtype=np.float64).flatten()
    mu_filtered = np.asarray(mu_filtered, dtype=np.float64).flatten()
    vol = np.asarray(vol, dtype=np.float64).flatten()

    N = min(len(returns), len(mu_filtered), len(vol))
    returns = returns[:N]
    mu_filtered = mu_filtered[:N]
    vol = vol[:N]

    # Compute innovations
    innovations = compute_innovations(returns, mu_filtered, phi)

    # Compute vol^2
    vol_sq = vol ** 2

    # Import and run Numba kernel
    from models.numba_kernels import online_c_update_kernel

    c_path, eta_path, ratio_ema = online_c_update_kernel(
        innovations,
        vol_sq,
        float(c_init),
        float(config.eta_init),
        float(config.eta_min),
        float(config.eta_decay),
        float(config.c_min),
        float(config.c_max),
    )

    c_final = float(c_path[-1]) if N > 0 else c_init

    # Check convergence: ratio_ema should be near 1.0
    converged = abs(ratio_ema - 1.0) < 0.15

    # Compute tracking error if target provided
    tracking_error = 0.0
    if c_target is not None and c_target > 0:
        tracking_error = abs(c_final - c_target) / c_target

    return OnlineCResult(
        c_path=c_path,
        eta_path=eta_path,
        c_final=c_final,
        ratio_ema=float(ratio_ema),
        c_init=c_init,
        n_steps=N,
        converged=converged,
        tracking_error=tracking_error,
    )


def evaluate_online_c_tracking(
    c_path: np.ndarray,
    c_target: float,
    window: int = 20,
    tolerance: float = ONLINE_C_TRACKING_TOLERANCE,
) -> dict:
    """
    Evaluate how well online c tracks a target value.

    Parameters
    ----------
    c_path : array
        Time-varying c values from online update.
    c_target : float
        Target c (e.g., regime-conditional c from batch tuning).
    window : int
        Number of observations for convergence check.
    tolerance : float
        Relative tolerance for "tracking within X%".

    Returns
    -------
    dict with:
        - tracks_within_tolerance: bool
        - relative_error_at_end: float
        - convergence_step: int or None (first step within tolerance)
        - mean_c_last_window: float
    """
    c_path = np.asarray(c_path)
    N = len(c_path)

    if N == 0 or c_target <= 0:
        return {
            "tracks_within_tolerance": False,
            "relative_error_at_end": float("inf"),
            "convergence_step": None,
            "mean_c_last_window": 0.0,
        }

    # Mean c over last `window` observations
    tail = c_path[-min(window, N):]
    mean_c_last = float(np.mean(tail))
    rel_error = abs(mean_c_last - c_target) / c_target

    # Find first step where c is within tolerance
    rel_errors = np.abs(c_path - c_target) / c_target
    within_mask = rel_errors < tolerance
    convergence_step = None
    if np.any(within_mask):
        convergence_step = int(np.argmax(within_mask))

    return {
        "tracks_within_tolerance": rel_error < tolerance,
        "relative_error_at_end": rel_error,
        "convergence_step": convergence_step,
        "mean_c_last_window": mean_c_last,
    }


def compute_rolling_hit_rate(
    returns: np.ndarray,
    mu_filtered_baseline: np.ndarray,
    mu_filtered_online: np.ndarray,
    window: int = 60,
) -> Tuple[float, float, float]:
    """
    Compare hit rates (directional accuracy) between baseline and online-c models.

    Parameters
    ----------
    returns : array
        Actual returns.
    mu_filtered_baseline : array
        Filtered mu from static c model.
    mu_filtered_online : array
        Filtered mu from online-c model.
    window : int
        Rolling window size.

    Returns
    -------
    Tuple of (baseline_hit_rate, online_hit_rate, improvement)
    """
    N = min(len(returns), len(mu_filtered_baseline), len(mu_filtered_online))
    if N < window + 1:
        return 0.5, 0.5, 0.0

    returns = np.asarray(returns[:N])
    mu_base = np.asarray(mu_filtered_baseline[:N])
    mu_online = np.asarray(mu_filtered_online[:N])

    # Directional hit: sign(mu_pred) == sign(r_t)
    # mu_pred_t = mu_filtered_{t-1} (one-step ahead)
    actual_sign = np.sign(returns[1:])
    base_pred_sign = np.sign(mu_base[:-1])
    online_pred_sign = np.sign(mu_online[:-1])

    base_hits = (actual_sign == base_pred_sign).astype(float)
    online_hits = (actual_sign == online_pred_sign).astype(float)

    # Rolling mean over last `window`
    if len(base_hits) >= window:
        base_rate = float(np.mean(base_hits[-window:]))
        online_rate = float(np.mean(online_hits[-window:]))
    else:
        base_rate = float(np.mean(base_hits))
        online_rate = float(np.mean(online_hits))

    improvement = online_rate - base_rate
    return base_rate, online_rate, improvement
