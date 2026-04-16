"""
Epic 26: PIT-Based Online Recalibration

Provides online calibration correction without full re-tuning:
1. Isotonic regression for PIT recalibration
2. Location-scale correction via innovation statistics
3. Adaptive recalibration frequency based on PIT deviation

References:
- Gneiting et al. (2007): Probabilistic forecasts, calibration, and sharpness
- Dawid (1984): Statistical theory of prequential analysis
- Hamill (2001): Interpretation of rank histograms (PIT histograms)
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from scipy.stats import kstest, uniform

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Story 26.1: Isotonic recalibration
DEFAULT_PIT_WINDOW = 126    # ~6 months of trading days
MIN_PIT_VALUES = 20         # Minimum for isotonic fit
N_BINS_ECE = 10             # Bins for Expected Calibration Error

# Story 26.2: Location-scale correction
DEFAULT_CORRECTION_WINDOW = 60  # ~3 months
DEFAULT_EWM_LAMBDA = 0.95       # Exponential decay for location correction
BIAS_THRESHOLD_SE = 0.5         # Max bias in standard errors after correction

# Story 26.3: Adaptive recalibration
KS_THRESHOLD_DAILY = 0.10
KS_THRESHOLD_WEEKLY = 0.05
KS_WINDOW = 60                  # PIT values for KS test


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IsotonicRecalibrationResult:
    """Result of isotonic PIT recalibration."""
    recalibrated_pit: np.ndarray  # g(PIT) values
    mapping_x: np.ndarray         # Input grid for mapping function
    mapping_y: np.ndarray         # Output values: g(x)
    ks_before: float              # KS stat vs U(0,1) before recalibration
    ks_after: float               # KS stat vs U(0,1) after recalibration
    ece_before: float             # ECE before
    ece_after: float              # ECE after
    n_pit_values: int
    window: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ks_before": float(self.ks_before),
            "ks_after": float(self.ks_after),
            "ece_before": float(self.ece_before),
            "ece_after": float(self.ece_after),
            "n_pit_values": self.n_pit_values,
            "window": self.window,
        }


@dataclass
class LocationScaleResult:
    """Result of location-scale correction."""
    delta_mu: float        # Location correction (add to mu)
    scale_sigma: float     # Scale correction (multiply sigma)
    mean_innovation: float # Mean of innovations before correction
    variance_ratio: float  # Var(innovations) / mean(R)
    bias_se: float         # Bias in standard errors: |mean_innovation| / sqrt(mean_R)
    n_observations: int
    window: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta_mu": float(self.delta_mu),
            "scale_sigma": float(self.scale_sigma),
            "mean_innovation": float(self.mean_innovation),
            "variance_ratio": float(self.variance_ratio),
            "bias_se": float(self.bias_se),
            "n_observations": self.n_observations,
            "window": self.window,
        }


@dataclass
class RecalibrationScheduleResult:
    """Result of adaptive recalibration scheduling."""
    frequency: str          # "daily", "weekly", "monthly"
    ks_statistic: float     # KS test statistic on trailing PIT
    ks_pvalue: float        # KS test p-value
    n_pit_values: int
    threshold_daily: float
    threshold_weekly: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frequency": self.frequency,
            "ks_statistic": float(self.ks_statistic),
            "ks_pvalue": float(self.ks_pvalue),
            "n_pit_values": self.n_pit_values,
        }


# ---------------------------------------------------------------------------
# Story 26.1: Isotonic Regression for PIT Recalibration
# ---------------------------------------------------------------------------

def _compute_ece(pit_values: np.ndarray, n_bins: int = N_BINS_ECE) -> float:
    """
    Compute Expected Calibration Error.

    For PIT values from a well-calibrated model, the fraction of PIT < p
    should equal p for all p in [0, 1].

    ECE = mean(|fraction_in_bin - expected_fraction|) over bins.
    """
    pit = np.asarray(pit_values, dtype=np.float64)
    pit = pit[np.isfinite(pit)]
    if len(pit) < 2:
        return 1.0

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = np.sum((pit >= lo) & (pit < hi))
        observed_frac = in_bin / len(pit)
        expected_frac = 1.0 / n_bins
        ece += abs(observed_frac - expected_frac)
    return ece / n_bins


def _isotonic_regression(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Pool Adjacent Violators (PAV) isotonic regression.

    Returns isotonic (non-decreasing) fit of y against x ordering.
    """
    n = len(y)
    if n == 0:
        return np.array([])

    # Sort by x
    order = np.argsort(x)
    y_sorted = y[order].copy()

    # PAV algorithm
    blocks = [[i] for i in range(n)]
    values = [y_sorted[i] for i in range(n)]

    i = 0
    while i < len(values) - 1:
        if values[i] > values[i + 1]:
            # Merge blocks
            merged_block = blocks[i] + blocks[i + 1]
            merged_value = np.mean(y_sorted[merged_block])
            blocks[i] = merged_block
            values[i] = merged_value
            blocks.pop(i + 1)
            values.pop(i + 1)
            # Step back
            if i > 0:
                i -= 1
        else:
            i += 1

    # Reconstruct output
    result = np.zeros(n)
    for block, value in zip(blocks, values):
        for idx in block:
            result[idx] = value

    # Un-sort back to original order
    output = np.zeros(n)
    output[order] = result
    return output


def isotonic_recalibrate(
    pit_values: np.ndarray,
    window: int = DEFAULT_PIT_WINDOW,
) -> IsotonicRecalibrationResult:
    """
    Isotonic regression recalibration of PIT values.

    Fits a monotonic mapping g: [0,1] -> [0,1] such that g(PIT)
    is closer to U(0,1) than the raw PIT values.

    Parameters
    ----------
    pit_values : array-like, shape (T,)
        PIT values (should be in [0, 1]).
    window : int
        Number of most recent PIT values to use for fitting.

    Returns
    -------
    IsotonicRecalibrationResult
        Recalibrated PIT values and diagnostics.
    """
    pit = np.asarray(pit_values, dtype=np.float64).ravel()

    # Filter to valid [0, 1] values
    valid_mask = np.isfinite(pit) & (pit >= 0) & (pit <= 1)
    pit_valid = pit[valid_mask]

    if len(pit_valid) < MIN_PIT_VALUES:
        raise ValueError(
            f"Need >= {MIN_PIT_VALUES} valid PIT values, got {len(pit_valid)}"
        )

    # Use trailing window
    window = max(window, MIN_PIT_VALUES)
    pit_window = pit_valid[-window:]

    # KS and ECE before recalibration
    ks_before, _ = kstest(pit_window, 'uniform')
    ece_before = _compute_ece(pit_window)

    # Build calibration mapping using isotonic regression
    # Target: if PIT were uniform, the quantile function should be identity
    # We fit: sorted(PIT) -> expected_uniform_quantiles
    n = len(pit_window)
    sorted_pit = np.sort(pit_window)
    expected_quantiles = (np.arange(n) + 0.5) / n  # Expected U(0,1) quantiles

    # Isotonic fit: for each PIT value, map to the expected quantile
    # We use PAV on the sorted PIT vs expected quantiles
    # Then interpolate for new PIT values
    iso_fit = _isotonic_regression(sorted_pit, expected_quantiles)

    # Build interpolation grid (sorted, monotonic)
    # Ensure it's sorted and monotonic for interpolation
    grid_x = sorted_pit
    grid_y = iso_fit

    # Make monotonic: take cumulative max
    grid_y = np.maximum.accumulate(grid_y)
    # Clamp to [0, 1]
    grid_y = np.clip(grid_y, 0, 1)

    # Apply mapping to all valid PIT values
    recalibrated = np.interp(pit_valid, grid_x, grid_y)
    recalibrated = np.clip(recalibrated, 0, 1)

    # KS and ECE after recalibration
    recalib_window = recalibrated[-window:]
    ks_after, _ = kstest(recalib_window, 'uniform')
    ece_after = _compute_ece(recalib_window)

    return IsotonicRecalibrationResult(
        recalibrated_pit=recalibrated,
        mapping_x=grid_x,
        mapping_y=grid_y,
        ks_before=float(ks_before),
        ks_after=float(ks_after),
        ece_before=float(ece_before),
        ece_after=float(ece_after),
        n_pit_values=len(pit_valid),
        window=window,
    )


# ---------------------------------------------------------------------------
# Story 26.2: Location-Scale Correction
# ---------------------------------------------------------------------------

def location_scale_correction(
    innovations: np.ndarray,
    R: np.ndarray,
    window: int = DEFAULT_CORRECTION_WINDOW,
    ewm_lambda: float = DEFAULT_EWM_LAMBDA,
) -> LocationScaleResult:
    """
    Online location-scale correction via innovation statistics.

    Location: delta_mu = EWM(innovations, lambda) -- corrects systematic bias.
    Scale: s_sigma = sqrt(Var(innovations) / mean(R)) -- corrects vol estimate.

    Parameters
    ----------
    innovations : array-like, shape (T,)
        Innovation sequence: v_t = r_t - mu_t (observed - predicted).
    R : array-like, shape (T,)
        Predicted variance (observation noise) at each timestep.
    window : int
        Number of recent observations for correction.
    ewm_lambda : float
        Decay factor for exponential weighted mean (0 < lambda < 1).

    Returns
    -------
    LocationScaleResult
        Location and scale corrections.
    """
    innov = np.asarray(innovations, dtype=np.float64).ravel()
    r_arr = np.asarray(R, dtype=np.float64).ravel()

    if len(innov) != len(r_arr):
        raise ValueError(
            f"innovations and R must have same length, got {len(innov)} and {len(r_arr)}"
        )

    # Use trailing window
    window = max(window, 5)
    innov_w = innov[-window:]
    r_w = r_arr[-window:]

    # Filter NaN
    valid = np.isfinite(innov_w) & np.isfinite(r_w) & (r_w > 0)
    innov_w = innov_w[valid]
    r_w = r_w[valid]

    if len(innov_w) < 5:
        return LocationScaleResult(
            delta_mu=0.0,
            scale_sigma=1.0,
            mean_innovation=0.0,
            variance_ratio=1.0,
            bias_se=0.0,
            n_observations=len(innov_w),
            window=window,
        )

    # Location correction: EWM of innovations
    # EWM: s_t = lambda * s_{t-1} + (1 - lambda) * v_t
    ewm_lambda = float(np.clip(ewm_lambda, 0.5, 0.999))
    alpha = 1.0 - ewm_lambda

    ewm_val = innov_w[0]
    for i in range(1, len(innov_w)):
        ewm_val = ewm_lambda * ewm_val + alpha * innov_w[i]

    delta_mu = float(ewm_val)
    mean_innov = float(np.mean(innov_w))

    # Scale correction: sqrt(Var(innovations) / mean(R))
    var_innov = float(np.var(innov_w, ddof=1)) if len(innov_w) > 1 else float(np.mean(r_w))
    mean_r = float(np.mean(r_w))
    variance_ratio = var_innov / max(mean_r, 1e-12)
    scale_sigma = float(np.sqrt(max(variance_ratio, 0.01)))

    # Bias in standard errors
    mean_sigma = float(np.sqrt(max(mean_r, 1e-12)))
    bias_se = abs(mean_innov) / max(mean_sigma, 1e-12)

    return LocationScaleResult(
        delta_mu=delta_mu,
        scale_sigma=scale_sigma,
        mean_innovation=mean_innov,
        variance_ratio=variance_ratio,
        bias_se=float(bias_se),
        n_observations=len(innov_w),
        window=window,
    )


# ---------------------------------------------------------------------------
# Story 26.3: Adaptive Recalibration Frequency
# ---------------------------------------------------------------------------

def recalibration_schedule(
    pit_values: np.ndarray,
    threshold_daily: float = KS_THRESHOLD_DAILY,
    threshold_weekly: float = KS_THRESHOLD_WEEKLY,
    ks_window: int = KS_WINDOW,
) -> RecalibrationScheduleResult:
    """
    Determine recalibration frequency based on PIT deviation rate.

    Uses KS statistic on trailing PIT values against U(0,1).

    Parameters
    ----------
    pit_values : array-like, shape (T,)
        Recent PIT values.
    threshold_daily : float
        KS threshold above which daily recalibration is needed.
    threshold_weekly : float
        KS threshold above which weekly recalibration is needed.
    ks_window : int
        Number of trailing PIT values for KS test.

    Returns
    -------
    RecalibrationScheduleResult
        Recommended recalibration frequency and diagnostics.
    """
    pit = np.asarray(pit_values, dtype=np.float64).ravel()

    # Filter valid
    valid = np.isfinite(pit) & (pit >= 0) & (pit <= 1)
    pit_valid = pit[valid]

    if len(pit_valid) < 5:
        # Too few values: default to daily (conservative)
        return RecalibrationScheduleResult(
            frequency="daily",
            ks_statistic=1.0,
            ks_pvalue=0.0,
            n_pit_values=len(pit_valid),
            threshold_daily=threshold_daily,
            threshold_weekly=threshold_weekly,
        )

    # Use trailing window
    pit_window = pit_valid[-ks_window:]

    # KS test against U(0,1)
    ks_stat, ks_pval = kstest(pit_window, 'uniform')

    # Determine frequency
    if ks_stat > threshold_daily:
        frequency = "daily"
    elif ks_stat > threshold_weekly:
        frequency = "weekly"
    else:
        frequency = "monthly"

    return RecalibrationScheduleResult(
        frequency=frequency,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_pval),
        n_pit_values=len(pit_window),
        threshold_daily=threshold_daily,
        threshold_weekly=threshold_weekly,
    )


def compute_recalibration_stability(
    mapping_current: Tuple[np.ndarray, np.ndarray],
    mapping_previous: Tuple[np.ndarray, np.ndarray],
    n_grid: int = 100,
) -> float:
    """
    Compute sup-norm between two recalibration mappings.

    Parameters
    ----------
    mapping_current : tuple of (x, y) arrays
        Current recalibration mapping.
    mapping_previous : tuple of (x, y) arrays
        Previous recalibration mapping.
    n_grid : int
        Number of grid points for comparison.

    Returns
    -------
    float
        Sup-norm distance between mappings.
    """
    grid = np.linspace(0, 1, n_grid)

    x_curr, y_curr = mapping_current
    x_prev, y_prev = mapping_previous

    # Interpolate both on common grid
    vals_curr = np.interp(grid, x_curr, y_curr)
    vals_prev = np.interp(grid, x_prev, y_prev)

    # Sup-norm
    return float(np.max(np.abs(vals_curr - vals_prev)))
