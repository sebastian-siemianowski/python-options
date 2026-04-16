"""
Innovation Sequence Diagnostics for Kalman Filter Health.

Story 9.1: Ljung-Box autocorrelation test on innovations.
Story 9.2: Innovation variance ratio test.
Story 9.3: CUSUM drift detection.

The innovation sequence v_t = r_t - mu_{t|t-1} should be:
  1. Zero mean (unbiased filter)
  2. White noise (filter extracts all signal)
  3. Consistent variance: v_t^2 / R_t ~ 1 (correct observation noise)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


# =============================================================================
# STORY 9.1: LJUNG-BOX AUTOCORRELATION TEST
# =============================================================================
#
# If innovations are autocorrelated, the filter is missing systematic signal.
# Ljung-Box Q-test at lags {1, 5, 10, 20} detects this.
#
# Q = n(n+2) * sum_{k=1}^{K} r_k^2 / (n-k)
# where r_k = sample autocorrelation at lag k
# Under H0 (white noise): Q ~ chi2(K)
# =============================================================================

DEFAULT_LAGS = [1, 5, 10, 20]
LB_PVALUE_THRESHOLD = 0.01  # p-value below this flags MISSPECIFIED


@dataclass
class LjungBoxResult:
    """Result of Ljung-Box test on innovations."""
    q_stats: dict               # {lag: Q-statistic}
    p_values: dict              # {lag: p-value}
    is_misspecified: bool       # True if any p-value < threshold
    flagged_lags: list          # Lags where p-value < threshold
    n_obs: int                  # Number of observations


def innovation_ljung_box(
    innovations: np.ndarray,
    R: np.ndarray,
    lags: Sequence[int] = DEFAULT_LAGS,
    threshold: float = LB_PVALUE_THRESHOLD,
) -> LjungBoxResult:
    """
    Ljung-Box Q-test on standardized innovations.

    Standardizes innovations by sqrt(R) then computes the Ljung-Box
    Q-statistic at each specified lag.

    Parameters
    ----------
    innovations : np.ndarray
        Innovation sequence v_t = r_t - mu_{t|t-1}.
    R : np.ndarray
        Observation noise variance (one per time step).
    lags : sequence of int
        Lags to test (default [1, 5, 10, 20]).
    threshold : float
        p-value threshold for MISSPECIFIED flag.

    Returns
    -------
    LjungBoxResult
        Q-statistics, p-values, and misspecification flag.
    """
    from scipy.stats import chi2

    innovations = np.asarray(innovations, dtype=float)
    R = np.asarray(R, dtype=float)

    n = len(innovations)
    if n < 3:
        return LjungBoxResult(
            q_stats={}, p_values={}, is_misspecified=False,
            flagged_lags=[], n_obs=n,
        )

    # Standardize innovations
    safe_R = np.maximum(R, 1e-20)
    std_innov = innovations / np.sqrt(safe_R)

    # Remove mean for robustness
    std_innov = std_innov - np.mean(std_innov)

    # Compute sample autocorrelations
    var_innov = np.var(std_innov, ddof=0)
    if var_innov < 1e-20:
        # Constant innovations -> no autocorrelation
        q_stats = {lag: 0.0 for lag in lags}
        p_values = {lag: 1.0 for lag in lags}
        return LjungBoxResult(
            q_stats=q_stats, p_values=p_values, is_misspecified=False,
            flagged_lags=[], n_obs=n,
        )

    q_stats = {}
    p_values = {}
    flagged_lags = []

    for max_lag in lags:
        if max_lag >= n:
            q_stats[max_lag] = 0.0
            p_values[max_lag] = 1.0
            continue

        # Compute Q-stat up to max_lag
        q_val = 0.0
        for k in range(1, max_lag + 1):
            r_k = np.sum(std_innov[k:] * std_innov[:-k]) / (n * var_innov)
            q_val += (r_k ** 2) / (n - k)
        q_val *= n * (n + 2)

        # p-value from chi2(max_lag)
        p_val = 1.0 - chi2.cdf(q_val, df=max_lag)

        q_stats[max_lag] = float(q_val)
        p_values[max_lag] = float(p_val)

        if p_val < threshold:
            flagged_lags.append(max_lag)

    return LjungBoxResult(
        q_stats=q_stats,
        p_values=p_values,
        is_misspecified=len(flagged_lags) > 0,
        flagged_lags=flagged_lags,
        n_obs=n,
    )


# =============================================================================
# STORY 9.2: INNOVATION VARIANCE RATIO TEST
# =============================================================================
#
# The variance ratio VR_t = Var(v_t) / mean(R_t) should be near 1.0 if the
# observation noise c is correctly calibrated.
#
#   VR > 1.5: innovations are too volatile -> c is too small (intervals tight)
#   VR < 0.7: innovations are too quiet   -> c is too large (intervals wide)
#
# Online correction: c_new = c_old * VR^0.5 (square-root dampening)
# =============================================================================

VR_UPPER_THRESHOLD = 1.5   # VR above this -> c too small
VR_LOWER_THRESHOLD = 0.7   # VR below this -> c too large
VR_DEFAULT_WINDOW = 60     # Rolling window size
VR_DAMPENING_POWER = 0.5   # Square-root dampening for c correction


@dataclass
class VarianceRatioResult:
    """Result of innovation variance ratio test."""
    rolling_vr: np.ndarray      # Rolling VR time series
    current_vr: float           # Most recent VR value
    c_correction: float         # Multiplicative correction for c
    is_c_too_small: bool        # VR > upper threshold
    is_c_too_large: bool        # VR < lower threshold
    needs_correction: bool      # True if c_correction != 1.0
    n_obs: int                  # Number of observations
    window: int                 # Window used


def innovation_variance_ratio(
    innovations: np.ndarray,
    R: np.ndarray,
    window: int = VR_DEFAULT_WINDOW,
) -> VarianceRatioResult:
    """
    Rolling variance ratio test on innovations.

    Computes VR_t = Var(v_{t-w:t}^2) / mean(R_{t-w:t}) in a rolling window.
    If VR deviates from 1.0, the observation noise c needs correction.

    Parameters
    ----------
    innovations : np.ndarray
        Innovation sequence v_t = r_t - mu_{t|t-1}.
    R : np.ndarray
        Observation noise variance (one per time step).
    window : int
        Rolling window size (default 60).

    Returns
    -------
    VarianceRatioResult
        Rolling VR, current VR, c correction factor, and flags.
    """
    innovations = np.asarray(innovations, dtype=float)
    R = np.asarray(R, dtype=float)
    n = len(innovations)

    if n < window:
        # Not enough data: assume correctly calibrated
        rolling_vr = np.ones(n)
        return VarianceRatioResult(
            rolling_vr=rolling_vr,
            current_vr=1.0,
            c_correction=1.0,
            is_c_too_small=False,
            is_c_too_large=False,
            needs_correction=False,
            n_obs=n,
            window=window,
        )

    # Compute rolling variance of innovations and rolling mean of R
    innov_sq = innovations ** 2
    rolling_vr = np.ones(n)

    for t in range(window, n):
        window_innov_var = np.mean(innov_sq[t - window:t])
        window_R_mean = np.mean(R[t - window:t])
        if window_R_mean > 1e-20:
            rolling_vr[t] = window_innov_var / window_R_mean
        else:
            rolling_vr[t] = 1.0

    # Fill the initial part with the first computed value
    if n > window:
        rolling_vr[:window] = rolling_vr[window]

    # Current VR is the last value
    current_vr = float(rolling_vr[-1])

    # Determine correction
    is_c_too_small = current_vr > VR_UPPER_THRESHOLD
    is_c_too_large = current_vr < VR_LOWER_THRESHOLD
    needs_correction = is_c_too_small or is_c_too_large

    if needs_correction:
        c_correction = float(current_vr ** VR_DAMPENING_POWER)
    else:
        c_correction = 1.0

    return VarianceRatioResult(
        rolling_vr=rolling_vr,
        current_vr=current_vr,
        c_correction=c_correction,
        is_c_too_small=is_c_too_small,
        is_c_too_large=is_c_too_large,
        needs_correction=needs_correction,
        n_obs=n,
        window=window,
    )


# =============================================================================
# STORY 9.3: CUSUM DRIFT DETECTION
# =============================================================================
#
# CUSUM (Cumulative Sum) chart on standardized innovations detects persistent
# drift shifts. When the true drift moves beyond the filter's tracking ability,
# cumulated standardized innovations will trend away from zero.
#
# Two-sided CUSUM:
#   S_t^+ = max(0, S_{t-1}^+ + z_t - k)    (detects positive drift)
#   S_t^- = max(0, S_{t-1}^- - z_t - k)    (detects negative drift)
#
# where z_t = v_t / sqrt(R_t) is the standardized innovation,
# k = 0.5 (reference value for 1-sigma shift detection),
# h = threshold (alarm when S_t > h).
#
# ARL = 500 under H0 achieved with h ~= 4.0 and k = 0.5.
# =============================================================================

CUSUM_THRESHOLD = 4.0       # Alarm threshold h
CUSUM_REFERENCE = 0.5       # Reference value k (detects 1-sigma shifts)
CUSUM_Q_MULTIPLIER = 10.0   # q increase on alarm
CUSUM_ALARM_DURATION = 5    # Days of increased q after alarm


@dataclass
class CUSUMResult:
    """Result of CUSUM drift detection on innovations."""
    cusum_pos: np.ndarray       # Positive CUSUM path S_t^+
    cusum_neg: np.ndarray       # Negative CUSUM path S_t^-
    alarm_times: list           # Time indices where alarm triggered
    alarm_directions: list      # +1 for positive drift, -1 for negative drift
    n_alarms: int               # Total number of alarms
    has_alarm: bool             # True if any alarm occurred
    q_multiplier: float         # Recommended q multiplier on alarm
    alarm_duration: int         # Days to apply q_multiplier
    n_obs: int                  # Number of observations
    threshold: float            # Threshold used


def innovation_cusum(
    innovations: np.ndarray,
    R: np.ndarray,
    threshold: float = CUSUM_THRESHOLD,
    reference: float = CUSUM_REFERENCE,
) -> CUSUMResult:
    """
    Two-sided CUSUM chart on standardized innovations for drift detection.

    Detects persistent drift shifts in the Kalman filter's state.
    When an alarm fires, q should be increased by q_multiplier for
    alarm_duration days to allow rapid state adaptation.

    Parameters
    ----------
    innovations : np.ndarray
        Innovation sequence v_t = r_t - mu_{t|t-1}.
    R : np.ndarray
        Observation noise variance (one per time step).
    threshold : float
        CUSUM alarm threshold h (default 4.0, ARL ~500 under H0).
    reference : float
        Reference value k for shift detection (default 0.5 for 1-sigma).

    Returns
    -------
    CUSUMResult
        CUSUM paths, alarm times, and q correction parameters.
    """
    innovations = np.asarray(innovations, dtype=float)
    R = np.asarray(R, dtype=float)
    n = len(innovations)

    if n < 2:
        return CUSUMResult(
            cusum_pos=np.zeros(n),
            cusum_neg=np.zeros(n),
            alarm_times=[],
            alarm_directions=[],
            n_alarms=0,
            has_alarm=False,
            q_multiplier=CUSUM_Q_MULTIPLIER,
            alarm_duration=CUSUM_ALARM_DURATION,
            n_obs=n,
            threshold=threshold,
        )

    # Standardize innovations
    safe_R = np.maximum(R, 1e-20)
    z = innovations / np.sqrt(safe_R)

    # Two-sided CUSUM
    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    alarm_times = []
    alarm_directions = []

    # Cooldown to prevent rapid re-alarming
    cooldown = 0

    for t in range(1, n):
        cusum_pos[t] = max(0.0, cusum_pos[t - 1] + z[t] - reference)
        cusum_neg[t] = max(0.0, cusum_neg[t - 1] - z[t] - reference)

        if cooldown > 0:
            cooldown -= 1
            continue

        if cusum_pos[t] > threshold:
            alarm_times.append(t)
            alarm_directions.append(1)
            cusum_pos[t] = 0.0  # Reset after alarm
            cooldown = CUSUM_ALARM_DURATION

        elif cusum_neg[t] > threshold:
            alarm_times.append(t)
            alarm_directions.append(-1)
            cusum_neg[t] = 0.0  # Reset after alarm
            cooldown = CUSUM_ALARM_DURATION

    return CUSUMResult(
        cusum_pos=cusum_pos,
        cusum_neg=cusum_neg,
        alarm_times=alarm_times,
        alarm_directions=alarm_directions,
        n_alarms=len(alarm_times),
        has_alarm=len(alarm_times) > 0,
        q_multiplier=CUSUM_Q_MULTIPLIER,
        alarm_duration=CUSUM_ALARM_DURATION,
        n_obs=n,
        threshold=threshold,
    )
