# Sample-size-aware PIT calibration functions (Elite Fix - February 2026)
#
# These functions provide more informative calibration metrics that scale
# better with sample size. The standard KS test becomes overly sensitive
# with large samples (n>5000), even tiny miscalibration (2-3%) gives p=0.

from __future__ import annotations
import numpy as np
from typing import Dict
from scipy.stats import kstest


def compute_pit_calibration_metrics(
    pit_values: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive PIT calibration metrics.
    
    Returns:
        Dict with:
        - ks_statistic: Standard KS statistic
        - ks_pvalue: Standard KS p-value  
        - max_deviation: Maximum deviation from uniform quantiles
        - mean_deviation: Mean absolute deviation from uniform quantiles
        - practical_calibration: Boolean for practical adequacy (MAD < 0.05)
        - calibration_score: Score from 0-1 (1 = perfect calibration)
    """
    pit_clean = np.asarray(pit_values).flatten()
    pit_clean = pit_clean[np.isfinite(pit_clean)]
    n = len(pit_clean)
    
    if n < 2:
        return {
            "ks_statistic": 1.0,
            "ks_pvalue": 0.0,
            "max_deviation": 1.0,
            "mean_deviation": 1.0,
            "practical_calibration": False,
            "calibration_score": 0.0,
        }
    
    ks_result = kstest(pit_clean, 'uniform')
    ks_stat = float(ks_result.statistic)
    ks_p = float(ks_result.pvalue)
    
    pit_sorted = np.sort(pit_clean)
    uniform_quantiles = np.linspace(0, 1, n)
    
    deviations = np.abs(pit_sorted - uniform_quantiles)
    max_deviation = float(np.max(deviations))
    mean_deviation = float(np.mean(deviations))
    
    practical_calibration = mean_deviation < 0.05
    calibration_score = float(np.exp(-10.0 * mean_deviation))
    
    return {
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "max_deviation": max_deviation,
        "mean_deviation": mean_deviation,
        "practical_calibration": practical_calibration,
        "calibration_score": calibration_score,
    }


def sample_size_adjusted_pit_threshold(n_samples: int) -> float:
    """Get adjusted PIT p-value threshold for sample size."""
    if n_samples <= 1000:
        return 0.05
    log_adjustment = 1.0 + 0.5 * np.log10(n_samples / 1000.0)
    return min(0.05 * log_adjustment, 0.15)


def is_pit_calibrated(
    pit_pvalue: float,
    n_samples: int,
    strict: bool = False
) -> bool:
    """Check if PIT calibration passes using sample-size-adjusted threshold."""
    if strict:
        return pit_pvalue >= 0.05
    threshold = sample_size_adjusted_pit_threshold(n_samples)
    return pit_pvalue >= threshold
