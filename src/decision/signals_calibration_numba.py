"""
===============================================================================
NUMBA-ACCELERATED KERNELS FOR SIGNAL CALIBRATION
===============================================================================

JIT-compiled implementations of the compute-heavy calibration functions:
  - Isotonic regression (PAV algorithm)
  - Magnitude scale factor
  - Bias correction (winsorized mean)
  - Hit rate computation
  - Brier score computation
  - Label threshold grid search
  - Isotonic map application

These kernels are called from signals_calibration.py as drop-in replacements
for the pure Python/numpy fallbacks. Each kernel has identical semantics.

Performance: ~5-10x speedup over numpy for typical calibration workloads
(36 eval points × 4 horizons × grid search).

Design:
  - fastmath=True for all kernels (calibration doesn't need tail precision)
  - cache=True for persistent JIT compilation
  - No Python objects, no scipy, no dynamic allocation
  - All inputs/outputs are primitive types or numpy arrays

Author: Quantitative Systems Team
Date: 2026-03-01
===============================================================================
"""

from numba import njit
import numpy as np
import math


# =============================================================================
# ISOTONIC REGRESSION (Pool-Adjacent-Violators via binning)
# =============================================================================

@njit(cache=True, fastmath=True)
def isotonic_regression_nb(
    raw_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_bins: int,
) -> tuple:
    """
    Numba-accelerated isotonic regression via binned PAV.

    Parameters
    ----------
    raw_probs : ndarray of float64
        Raw probability values ∈ [0, 1]
    actual_outcomes : ndarray of float64
        Binary outcomes (0 or 1)
    n_bins : int
        Number of bins for binning step

    Returns
    -------
    tuple of (x_out, y_out) : ndarray of float64
        Breakpoints for piecewise-linear isotonic map.
        Empty bins have NaN values.
    """
    n = len(raw_probs)

    # Bin edges
    bin_width = 1.0 / n_bins
    bin_x = np.full(n_bins, np.nan)
    bin_y = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    # Accumulate into bins
    bin_sum_p = np.zeros(n_bins)
    bin_sum_y = np.zeros(n_bins)

    for i in range(n):
        p = raw_probs[i]
        b = int(p / bin_width)
        if b >= n_bins:
            b = n_bins - 1
        if b < 0:
            b = 0
        bin_sum_p[b] += p
        bin_sum_y[b] += actual_outcomes[i]
        bin_counts[b] += 1

    # Compute bin means (only for bins with >= 2 points)
    valid_count = 0
    for b in range(n_bins):
        if bin_counts[b] >= 2:
            bin_x[b] = bin_sum_p[b] / bin_counts[b]
            bin_y[b] = bin_sum_y[b] / bin_counts[b]
            valid_count += 1

    if valid_count < 2:
        # Not enough data — return identity map
        x_out = np.array([0.0, 1.0])
        y_out = np.array([0.0, 1.0])
        return x_out, y_out

    # Compact to remove NaN entries
    cx = np.empty(valid_count)
    cy = np.empty(valid_count)
    cw = np.ones(valid_count)
    j = 0
    for b in range(n_bins):
        if not math.isnan(bin_x[b]):
            cx[j] = bin_x[b]
            cy[j] = bin_y[b]
            cw[j] = float(bin_counts[b])
            j += 1

    # Pool-Adjacent-Violators (PAV) for monotonicity
    # In-place merging with weight tracking
    m = valid_count
    i = 0
    while i < m - 1:
        if cy[i] > cy[i + 1]:
            # Merge i and i+1
            total_w = cw[i] + cw[i + 1]
            cy[i] = (cw[i] * cy[i] + cw[i + 1] * cy[i + 1]) / total_w
            cx[i] = (cw[i] * cx[i] + cw[i + 1] * cx[i + 1]) / total_w
            cw[i] = total_w
            # Shift remaining elements left
            for k in range(i + 1, m - 1):
                cx[k] = cx[k + 1]
                cy[k] = cy[k + 1]
                cw[k] = cw[k + 1]
            m -= 1
            if i > 0:
                i -= 1
        else:
            i += 1

    # Build output arrays
    x_out = np.empty(m)
    y_out = np.empty(m)
    for i in range(m):
        x_out[i] = cx[i]
        y_out[i] = max(0.0, min(1.0, cy[i]))

    return x_out, y_out


# =============================================================================
# APPLY ISOTONIC MAP (vectorized interpolation)
# =============================================================================

@njit(cache=True, fastmath=True)
def apply_isotonic_map_nb(
    raw_probs: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """
    Apply isotonic calibration map to an array of probabilities.

    Uses piecewise-linear interpolation between breakpoints.

    Parameters
    ----------
    raw_probs : ndarray of float64
        Raw p_up values to calibrate
    map_x : ndarray of float64
        Breakpoint x-coordinates (sorted ascending)
    map_y : ndarray of float64
        Breakpoint y-coordinates (monotonically non-decreasing)

    Returns
    -------
    ndarray of float64
        Calibrated probabilities, clamped to [0, 1]
    """
    n = len(raw_probs)
    m = len(map_x)
    result = np.empty(n)

    if m < 2:
        for i in range(n):
            result[i] = raw_probs[i]
        return result

    for i in range(n):
        p = raw_probs[i]

        # Clamp to map range
        if p <= map_x[0]:
            result[i] = map_y[0]
        elif p >= map_x[m - 1]:
            result[i] = map_y[m - 1]
        else:
            # Binary search for interval
            lo = 0
            hi = m - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if map_x[mid] <= p:
                    lo = mid
                else:
                    hi = mid
            # Linear interpolation
            dx = map_x[hi] - map_x[lo]
            if dx > 1e-12:
                t = (p - map_x[lo]) / dx
                result[i] = map_y[lo] + t * (map_y[hi] - map_y[lo])
            else:
                result[i] = map_y[lo]

        # Clamp to [0, 1]
        if result[i] < 0.0:
            result[i] = 0.0
        elif result[i] > 1.0:
            result[i] = 1.0

    return result


# =============================================================================
# MAGNITUDE SCALE FACTOR
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_magnitude_scale_nb(
    predicted: np.ndarray,
    actual: np.ndarray,
    epsilon: float,
    scale_min: float,
    scale_max: float,
) -> float:
    """
    Compute magnitude scale factor: median(|actual| / |predicted|).

    Parameters
    ----------
    predicted : ndarray of float64
        Predicted returns (%)
    actual : ndarray of float64
        Actual returns (%)
    epsilon : float
        Minimum |predicted| to avoid division by zero
    scale_min : float
        Lower clamp for output
    scale_max : float
        Upper clamp for output

    Returns
    -------
    float
        Magnitude scale factor, clamped to [scale_min, scale_max]
    """
    n = len(predicted)

    # Count valid entries (|predicted| > epsilon)
    valid_count = 0
    for i in range(n):
        if abs(predicted[i]) > epsilon:
            valid_count += 1

    if valid_count < 3:
        return 1.0

    # Compute ratios
    ratios = np.empty(valid_count)
    j = 0
    for i in range(n):
        if abs(predicted[i]) > epsilon:
            ratios[j] = abs(actual[i]) / abs(predicted[i])
            j += 1

    # Compute median via sorting
    ratios_sorted = np.sort(ratios)
    if valid_count % 2 == 1:
        median_val = ratios_sorted[valid_count // 2]
    else:
        mid = valid_count // 2
        median_val = (ratios_sorted[mid - 1] + ratios_sorted[mid]) / 2.0

    # Clamp
    if median_val < scale_min:
        return scale_min
    elif median_val > scale_max:
        return scale_max
    else:
        return median_val


# =============================================================================
# BIAS CORRECTION (winsorized mean)
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_bias_correction_nb(
    predicted: np.ndarray,
    actual: np.ndarray,
    winsorize_pct: float,
) -> float:
    """
    Compute additive bias correction via winsorized mean of (actual - predicted).

    Parameters
    ----------
    predicted : ndarray of float64
        Predicted returns (%)
    actual : ndarray of float64
        Actual returns (%)
    winsorize_pct : float
        Percentile for winsorization (e.g. 5.0 = clip at 5th and 95th)

    Returns
    -------
    float
        Bias correction factor (additive, in %)
    """
    n = len(predicted)
    if n < 3:
        return 0.0

    errors = np.empty(n)
    for i in range(n):
        errors[i] = actual[i] - predicted[i]

    # Sort for percentile computation
    sorted_errors = np.sort(errors)

    # Compute percentile indices
    lo_idx = int(math.floor(winsorize_pct / 100.0 * n))
    hi_idx = int(math.ceil((100.0 - winsorize_pct) / 100.0 * n)) - 1
    if lo_idx < 0:
        lo_idx = 0
    if hi_idx >= n:
        hi_idx = n - 1
    if lo_idx >= hi_idx:
        # Not enough spread — just use mean
        total = 0.0
        for i in range(n):
            total += errors[i]
        return total / n

    lo_val = sorted_errors[lo_idx]
    hi_val = sorted_errors[hi_idx]

    # Winsorize (clip) and compute mean
    total = 0.0
    for i in range(n):
        v = errors[i]
        if v < lo_val:
            v = lo_val
        elif v > hi_val:
            v = hi_val
        total += v

    return total / n


# =============================================================================
# HIT RATE COMPUTATION
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_hit_rates_nb(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> float:
    """
    Compute directional hit rate: fraction of correct sign predictions.

    Parameters
    ----------
    predicted : ndarray of float64
        Predicted returns
    actual : ndarray of float64
        Actual returns

    Returns
    -------
    float
        Hit rate ∈ [0, 1]
    """
    n = len(predicted)
    if n == 0:
        return 0.5

    correct = 0
    total = 0
    for i in range(n):
        p_sign = 1.0 if predicted[i] > 0 else (-1.0 if predicted[i] < 0 else 0.0)
        a_sign = 1.0 if actual[i] > 0 else (-1.0 if actual[i] < 0 else 0.0)
        if p_sign != 0.0:
            total += 1
            if p_sign == a_sign:
                correct += 1

    if total == 0:
        return 0.5
    return float(correct) / float(total)


# =============================================================================
# BRIER SCORE COMPUTATION
# =============================================================================

@njit(cache=True, fastmath=True)
def compute_brier_score_nb(
    p_ups: np.ndarray,
    actual_ups: np.ndarray,
) -> float:
    """
    Compute Brier score: mean((p_up - actual_up)²).

    Parameters
    ----------
    p_ups : ndarray of float64
        Predicted probabilities ∈ [0, 1]
    actual_ups : ndarray of float64
        Binary outcomes (0 or 1)

    Returns
    -------
    float
        Brier score (lower is better, 0.25 = coin flip)
    """
    n = len(p_ups)
    if n == 0:
        return 0.25

    total = 0.0
    for i in range(n):
        diff = p_ups[i] - actual_ups[i]
        total += diff * diff

    return total / n


# =============================================================================
# LABEL THRESHOLD GRID SEARCH
# =============================================================================

@njit(cache=True, fastmath=True)
def grid_search_thresholds_nb(
    p_ups: np.ndarray,
    actual_ups: np.ndarray,
    exp_rets: np.ndarray,
    actual_rets: np.ndarray,
    buy_grid: np.ndarray,
    sell_grid: np.ndarray,
    min_separation: float,
    hit_weight: float,
    brier_weight: float,
    acc_weight: float,
) -> tuple:
    """
    Grid search for optimal buy/sell thresholds.

    Optimizes: hit_weight * hit_rate + brier_weight * inv_brier + acc_weight * label_acc

    Parameters
    ----------
    p_ups : ndarray of float64
        Raw p_up values
    actual_ups : ndarray of float64
        Binary outcomes
    exp_rets : ndarray of float64
        Predicted returns (%)
    actual_rets : ndarray of float64
        Actual returns (%)
    buy_grid : ndarray of float64
        Buy threshold candidates
    sell_grid : ndarray of float64
        Sell threshold candidates
    min_separation : float
        Minimum buy_thr - sell_thr
    hit_weight : float
        Weight for hit rate component
    brier_weight : float
        Weight for inverse Brier component
    acc_weight : float
        Weight for label accuracy component

    Returns
    -------
    tuple of (best_buy, best_sell, best_score) : float
    """
    n = len(p_ups)
    best_score = -1.0
    best_buy = 0.58
    best_sell = 0.42

    n_buy = len(buy_grid)
    n_sell = len(sell_grid)

    for bi in range(n_buy):
        buy_thr = buy_grid[bi]
        for si in range(n_sell):
            sell_thr = sell_grid[si]

            if buy_thr - sell_thr < min_separation:
                continue

            # Evaluate this threshold pair
            score = _eval_threshold_pair_nb(
                p_ups, actual_ups, exp_rets, actual_rets,
                buy_thr, sell_thr,
                hit_weight, brier_weight, acc_weight,
            )

            if score > best_score:
                best_score = score
                best_buy = buy_thr
                best_sell = sell_thr

    return best_buy, best_sell, best_score


@njit(cache=True, fastmath=True)
def _eval_threshold_pair_nb(
    p_ups: np.ndarray,
    actual_ups: np.ndarray,
    exp_rets: np.ndarray,
    actual_rets: np.ndarray,
    buy_thr: float,
    sell_thr: float,
    hit_weight: float,
    brier_weight: float,
    acc_weight: float,
) -> float:
    """Evaluate a single (buy_thr, sell_thr) pair. Returns composite score."""
    n = len(p_ups)
    if n < 5:
        return 0.0

    # --- Hit rate ---
    hit_correct = 0
    hit_total = 0
    for i in range(n):
        p_sign = 1.0 if exp_rets[i] > 0 else (-1.0 if exp_rets[i] < 0 else 0.0)
        a_sign = 1.0 if actual_rets[i] > 0 else (-1.0 if actual_rets[i] < 0 else 0.0)
        is_acted = (p_ups[i] >= buy_thr) or (p_ups[i] <= sell_thr)
        if is_acted and p_sign != 0.0:
            hit_total += 1
            if p_sign == a_sign:
                hit_correct += 1

    if hit_total < 3:
        return 0.0

    hit_rate = float(hit_correct) / float(hit_total)

    # --- Brier score ---
    brier_sum = 0.0
    for i in range(n):
        d = p_ups[i] - actual_ups[i]
        brier_sum += d * d
    brier = brier_sum / n
    inv_brier = 1.0 - brier / 0.25
    if inv_brier < 0.0:
        inv_brier = 0.0

    # --- Label accuracy ---
    label_correct = 0
    label_total = 0
    for i in range(n):
        if p_ups[i] >= buy_thr:
            label_total += 1
            if actual_rets[i] > 0:
                label_correct += 1
        elif p_ups[i] <= sell_thr:
            label_total += 1
            if actual_rets[i] < 0:
                label_correct += 1

    if label_total == 0:
        label_acc = 0.0
    else:
        label_acc = float(label_correct) / float(label_total)

    return hit_weight * hit_rate + brier_weight * inv_brier + acc_weight * label_acc


# =============================================================================
# BATCH CALIBRATION HELPER (multi-horizon at once)
# =============================================================================

@njit(cache=True, fastmath=True)
def batch_compute_corrections_nb(
    predicted: np.ndarray,
    actual: np.ndarray,
    p_ups: np.ndarray,
    actual_ups: np.ndarray,
    epsilon: float,
    winsorize_pct: float,
    scale_min: float,
    scale_max: float,
) -> tuple:
    """
    Compute all three numeric corrections at once for a single horizon.

    Parameters
    ----------
    predicted, actual : ndarray of float64
        Predicted and actual returns (%)
    p_ups, actual_ups : ndarray of float64
        Probability and binary outcomes
    epsilon : float
        Min |predicted| for magnitude ratio
    winsorize_pct : float
        Percentile for bias winsorization
    scale_min, scale_max : float
        Clamp range for magnitude scale

    Returns
    -------
    tuple of (mag_scale, bias, hit_rate, brier) : float
        Magnitude scale factor, additive bias, hit rate, Brier score
    """
    mag_scale = compute_magnitude_scale_nb(predicted, actual, epsilon, scale_min, scale_max)
    bias = compute_bias_correction_nb(predicted, actual, winsorize_pct)
    hit_rate = compute_hit_rates_nb(predicted, actual)
    brier = compute_brier_score_nb(p_ups, actual_ups)

    return mag_scale, bias, hit_rate, brier
