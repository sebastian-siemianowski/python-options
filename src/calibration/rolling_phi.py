"""
Story 3.2: Rolling phi Estimation with Structural Break Detection
================================================================

Rolling estimation of the AR(1) drift persistence parameter phi with
automatic CUSUM-based structural break detection and post-break reset.

Architecture:
  1. Slide a window across the return series
  2. Estimate phi via MLE in each window (lightweight Stage-1 only)
  3. Monitor phi_t for CUSUM breaks (|delta_phi| > threshold)
  4. On break: reset phi to asset-class prior and re-estimate with post-break data

Integration:
  - Uses compute_phi_prior() from phi_student_t_unified for reset targets
  - Designed for post-tuning monitoring (not real-time filtering)
"""
import os
import sys
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RollingPhiConfig:
    """Configuration for rolling phi estimation."""
    window: int = 252           # Rolling window size (trading days)
    step: int = 21              # Step size between estimates (trading days)
    phi_min: float = -0.80      # Lower bound for phi
    phi_max: float = 0.99       # Upper bound for phi
    cusum_threshold: float = 0.3  # |delta_phi| threshold for structural break
    cusum_cooldown: int = 63    # Min observations between detected breaks (~3 months)
    reset_to_prior: bool = True  # On break, reset to asset-class prior


@dataclass
class BreakPoint:
    """A detected structural break in phi."""
    index: int                  # Index in the original series
    phi_before: float           # phi estimate before break
    phi_after: float            # phi estimate after break
    delta_phi: float            # Change magnitude
    cusum_value: float          # CUSUM statistic at break


@dataclass
class RollingPhiResult:
    """Result of rolling phi estimation."""
    phi_t: np.ndarray           # phi estimates at each step
    timestamps: np.ndarray      # Indices (into original series) for each phi_t
    breaks: List[BreakPoint]    # Detected structural breaks
    n_windows: int              # Number of windows evaluated
    n_breaks: int               # Number of breaks detected
    phi_mean: float             # Mean phi across all windows
    phi_std: float              # Std of phi across all windows


# ---------------------------------------------------------------------------
# Lightweight phi MLE (window-local, no full Stage-1 pipeline)
# ---------------------------------------------------------------------------

def _estimate_phi_window(returns: np.ndarray, vol: np.ndarray,
                         phi_min: float = -0.80, phi_max: float = 0.99) -> float:
    """
    Estimate phi from a single window of returns via conditional MLE.

    Uses the Kalman filter one-step-ahead prediction errors with a fixed
    q and c (estimated from the window itself) to find the phi that
    maximizes the Gaussian approximate likelihood.

    This is a lightweight estimator for rolling windows -- not the full
    15-stage unified pipeline.
    """
    n = len(returns)
    if n < 30:
        return 0.0

    returns = np.asarray(returns, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)

    # Quick AR(1) OLS estimate as starting point
    r = returns[np.isfinite(returns)]
    if len(r) < 30:
        return 0.0

    mean_r = float(np.mean(r))
    centered = r - mean_r
    var_r = float(np.sum(centered ** 2))
    if var_r < 1e-20:
        return 0.0
    acf1 = float(np.sum(centered[:-1] * centered[1:])) / var_r
    phi_ols = float(np.clip(acf1, phi_min, phi_max))

    # Refine via grid search around OLS estimate using Kalman filter log-likelihood
    try:
        from models.numba_kernels import phi_gaussian_filter_kernel
    except ImportError:
        return phi_ols

    # Estimate q and c from the window
    vol_clean = vol[np.isfinite(vol)]
    if len(vol_clean) < 10:
        return phi_ols

    vol_median = float(np.median(vol_clean))
    q_est = max(1e-8, (vol_median * 0.1) ** 2)
    c_est = 1.0

    best_phi = phi_ols
    best_ll = -1e18

    # Grid: 11 points around OLS estimate
    phi_candidates = np.linspace(
        max(phi_min, phi_ols - 0.3),
        min(phi_max, phi_ols + 0.3),
        11,
    )
    # Always include the OLS estimate and 0
    phi_candidates = np.unique(np.concatenate([phi_candidates, [phi_ols, 0.0]]))

    for phi_try in phi_candidates:
        phi_try = float(np.clip(phi_try, phi_min, phi_max))
        try:
            _, _, ll = phi_gaussian_filter_kernel(returns, vol, q_est, c_est, phi_try)
            if np.isfinite(ll) and ll > best_ll:
                best_ll = ll
                best_phi = phi_try
        except Exception:
            continue

    return float(np.clip(best_phi, phi_min, phi_max))


# ---------------------------------------------------------------------------
# CUSUM Break Detection
# ---------------------------------------------------------------------------

def cusum_phi_breaks(phi_series: np.ndarray, timestamps: np.ndarray,
                     threshold: float = 0.3,
                     cooldown: int = 3) -> List[BreakPoint]:
    """
    Detect structural breaks in a phi time series using CUSUM.

    Monitors the cumulative sum of (phi_t - phi_mean) and flags breaks
    when consecutive phi changes exceed the threshold.

    Parameters
    ----------
    phi_series : array
        Rolling phi estimates.
    timestamps : array
        Corresponding indices in original series.
    threshold : float
        |delta_phi| threshold for break detection.
    cooldown : int
        Minimum number of steps between breaks.

    Returns
    -------
    list of BreakPoint
    """
    n = len(phi_series)
    if n < 3:
        return []

    breaks = []
    last_break_idx = -cooldown - 1

    # Compute running mean of phi
    phi_mean = float(np.mean(phi_series))

    # CUSUM on phi deviations from running mean
    cusum_pos = 0.0
    cusum_neg = 0.0

    for i in range(1, n):
        delta = phi_series[i] - phi_series[i - 1]

        # Page's CUSUM for positive and negative shifts
        cusum_pos = max(0.0, cusum_pos + delta - threshold * 0.3)
        cusum_neg = max(0.0, cusum_neg - delta - threshold * 0.3)

        # Check if either CUSUM exceeds detection threshold
        if (cusum_pos > threshold or cusum_neg > threshold) and (i - last_break_idx) > cooldown:
            bp = BreakPoint(
                index=int(timestamps[i]),
                phi_before=float(phi_series[i - 1]),
                phi_after=float(phi_series[i]),
                delta_phi=float(delta),
                cusum_value=float(max(cusum_pos, cusum_neg)),
            )
            breaks.append(bp)
            last_break_idx = i
            # Reset CUSUM after break
            cusum_pos = 0.0
            cusum_neg = 0.0

    return breaks


# ---------------------------------------------------------------------------
# Main Rolling Estimation
# ---------------------------------------------------------------------------

def rolling_phi_estimate(
    returns: np.ndarray,
    vol: np.ndarray,
    window: int = 252,
    step: int = 21,
    config: Optional[RollingPhiConfig] = None,
    asset_symbol: Optional[str] = None,
) -> RollingPhiResult:
    """
    Compute rolling phi estimates with structural break detection.

    Parameters
    ----------
    returns : array
        Full return series.
    vol : array
        Full volatility series (same length as returns).
    window : int
        Rolling window size in observations.
    step : int
        Step size between phi re-estimates.
    config : RollingPhiConfig, optional
        Configuration overrides.
    asset_symbol : str, optional
        Asset symbol for prior-based resets.

    Returns
    -------
    RollingPhiResult
    """
    if config is None:
        config = RollingPhiConfig(window=window, step=step)

    returns = np.asarray(returns, dtype=np.float64).flatten()
    vol = np.asarray(vol, dtype=np.float64).flatten()
    n = len(returns)

    if n < config.window + config.step:
        # Not enough data for even one rolling estimate
        return RollingPhiResult(
            phi_t=np.array([]),
            timestamps=np.array([], dtype=int),
            breaks=[],
            n_windows=0,
            n_breaks=0,
            phi_mean=0.0,
            phi_std=0.0,
        )

    # Get asset-class prior for resets
    phi_prior = 0.0
    try:
        from models.phi_student_t_unified import compute_phi_prior
        phi_prior, _ = compute_phi_prior(asset_symbol, returns=None)
    except ImportError:
        pass

    # Rolling estimation
    phi_list = []
    ts_list = []

    start_indices = range(0, n - config.window + 1, config.step)
    for start in start_indices:
        end = start + config.window
        r_win = returns[start:end]
        v_win = vol[start:end]

        phi_est = _estimate_phi_window(r_win, v_win, config.phi_min, config.phi_max)
        phi_list.append(phi_est)
        ts_list.append(end - 1)  # Timestamp = end of window

    phi_t = np.array(phi_list, dtype=np.float64)
    timestamps = np.array(ts_list, dtype=int)

    # CUSUM break detection
    cusum_cooldown_steps = max(1, config.cusum_cooldown // config.step)
    breaks = cusum_phi_breaks(
        phi_t, timestamps,
        threshold=config.cusum_threshold,
        cooldown=cusum_cooldown_steps,
    )

    # Post-break reset: re-estimate phi using only post-break data
    if config.reset_to_prior and breaks:
        for bp in breaks:
            # Find the phi_t index closest to the break
            bp_ts_idx = np.searchsorted(timestamps, bp.index)
            if bp_ts_idx < len(phi_t):
                # Re-estimate from break point using post-break data
                post_start = bp.index
                if post_start + 60 <= n:
                    # Enough post-break data: re-estimate
                    r_post = returns[post_start:min(post_start + config.window, n)]
                    v_post = vol[post_start:min(post_start + config.window, n)]
                    phi_reset = _estimate_phi_window(r_post, v_post, config.phi_min, config.phi_max)
                else:
                    # Not enough post-break data: use prior
                    phi_reset = phi_prior

                # Update subsequent phi_t values up to next break
                next_break_ts = n
                for other_bp in breaks:
                    if other_bp.index > bp.index:
                        next_break_ts = min(next_break_ts, other_bp.index)
                        break

                for j in range(bp_ts_idx, len(phi_t)):
                    if timestamps[j] < next_break_ts:
                        # Blend: start at reset value, gradually let window estimates take over
                        weight = min(1.0, (timestamps[j] - bp.index) / config.window)
                        phi_t[j] = (1.0 - weight) * phi_reset + weight * phi_t[j]

    # Summary stats
    phi_mean = float(np.mean(phi_t)) if len(phi_t) > 0 else 0.0
    phi_std = float(np.std(phi_t)) if len(phi_t) > 0 else 0.0

    return RollingPhiResult(
        phi_t=phi_t,
        timestamps=timestamps,
        breaks=breaks,
        n_windows=len(phi_t),
        n_breaks=len(breaks),
        phi_mean=phi_mean,
        phi_std=phi_std,
    )
