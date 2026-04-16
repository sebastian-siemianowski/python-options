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
# Page's CUSUM (Cumulative Sum) control chart on standardized innovations
# detects persistent drift shifts in the Kalman filter's state estimate.
#
# Theory (Page 1954, Hawkins & Olwell 1998):
# -----------------------------------------
# When the true drift mu moves by delta from the filter's tracking state,
# standardized innovations z_t = v_t / sqrt(R_t) shift from N(0,1) to
# N(delta, 1). The two-sided CUSUM detects both positive and negative shifts.
#
# Two-sided CUSUM recursion:
#   S_t^+ = max(0, S_{t-1}^+ + z_t - k)    (detects positive drift)
#   S_t^- = max(0, S_{t-1}^- - z_t - k)    (detects negative drift)
#   Alarm if S_t^+ > h  or  S_t^- > h
#
# where k = reference value (optimal for detecting shift delta: k = delta/2),
#       h = threshold calibrated for target ARL under H0.
#
# For k = 0.5 (optimal for 1-sigma shift), h = 5.0 yields ARL ~500 under
# two-sided monitoring (Siegmund 1985). This is approximately one false alarm
# per 2 trading years.
#
# After alarm: CUSUM resets to zero and q is increased by a multiplier for
# a fixed number of days to enable fast state adaptation. The q schedule
# uses exponential decay back to baseline for smooth transition.
#
# Key properties:
#   - ARL_0 = 500 under H0 (false alarm rate ~0.2% per day)
#   - Detection delay for 1-sigma shift: < 15 days (median ~8-10)
#   - Direction-aware: alarm_directions tells which way drift shifted
#   - q correction: 10x boost with exponential decay over 5 days
# =============================================================================

CUSUM_THRESHOLD = 5.0       # Alarm threshold h (ARL ~500 for two-sided k=0.5)
CUSUM_REFERENCE = 0.5       # Reference value k (optimal for 1-sigma shift)
CUSUM_Q_MULTIPLIER = 10.0   # Peak q multiplier on alarm
CUSUM_ALARM_DURATION = 5    # Days of increased q after alarm
CUSUM_Q_DECAY_RATE = 0.5    # Exponential decay rate for q schedule
CUSUM_COOLDOWN_DAYS = 5     # Cooldown between alarms (prevent rapid re-alarm)
CUSUM_MIN_OBS = 2           # Minimum observations for CUSUM


@dataclass
class CUSUMResult:
    """Result of CUSUM drift detection on standardized innovations.

    The CUSUM chart monitors z_t = v_t / sqrt(R_t) for persistent drift.
    When S_t^+ or S_t^- exceeds the threshold h, an alarm is raised.

    Attributes
    ----------
    cusum_pos : np.ndarray
        Positive CUSUM path S_t^+, detects upward drift.
    cusum_neg : np.ndarray
        Negative CUSUM path S_t^-, detects downward drift.
    alarm_times : list
        Time indices where alarms fired.
    alarm_directions : list
        +1 for positive drift alarm, -1 for negative drift alarm.
    alarm_magnitudes : list
        CUSUM value at alarm time (indicates severity).
    n_alarms : int
        Total number of alarms.
    has_alarm : bool
        True if any alarm occurred.
    q_multiplier : float
        Peak q multiplier to apply on alarm.
    alarm_duration : int
        Number of days for q correction window.
    n_obs : int
        Number of observations processed.
    threshold : float
        Threshold h used for this run.
    reference : float
        Reference value k used for this run.
    estimated_arl : float
        Estimated ARL based on theoretical approximation (Siegmund 1985).
    max_cusum : float
        Maximum CUSUM value observed across both paths.
    """
    cusum_pos: np.ndarray
    cusum_neg: np.ndarray
    alarm_times: list
    alarm_directions: list
    alarm_magnitudes: list
    n_alarms: int
    has_alarm: bool
    q_multiplier: float
    alarm_duration: int
    n_obs: int
    threshold: float
    reference: float
    estimated_arl: float
    max_cusum: float


def _cusum_inner_loop(
    z: np.ndarray,
    h: float,
    k: float,
    cooldown_days: int,
) -> tuple:
    """
    Pure-Python CUSUM inner loop.

    Runs the two-sided Page's CUSUM recursion with cooldown and
    reset-on-alarm. Separated for clarity and potential Numba
    acceleration if profiling shows this is a bottleneck.

    Parameters
    ----------
    z : np.ndarray
        Standardized innovations z_t = v_t / sqrt(R_t).
    h : float
        Alarm threshold.
    k : float
        Reference value.
    cooldown_days : int
        Cooldown period after alarm before next alarm can fire.

    Returns
    -------
    tuple of (cusum_pos, cusum_neg, alarm_times, alarm_directions,
              alarm_magnitudes)
    """
    n = len(z)
    cusum_pos = np.zeros(n, dtype=np.float64)
    cusum_neg = np.zeros(n, dtype=np.float64)
    alarm_times = []
    alarm_directions = []
    alarm_magnitudes = []
    cooldown = 0

    for t in range(1, n):
        # CUSUM recursion (Page 1954)
        cusum_pos[t] = max(0.0, cusum_pos[t - 1] + z[t] - k)
        cusum_neg[t] = max(0.0, cusum_neg[t - 1] - z[t] - k)

        if cooldown > 0:
            cooldown -= 1
            continue

        # Check positive alarm first (arbitrary tie-break)
        if cusum_pos[t] > h:
            alarm_times.append(t)
            alarm_directions.append(1)
            alarm_magnitudes.append(float(cusum_pos[t]))
            cusum_pos[t] = 0.0  # Reset on alarm
            cooldown = cooldown_days

        elif cusum_neg[t] > h:
            alarm_times.append(t)
            alarm_directions.append(-1)
            alarm_magnitudes.append(float(cusum_neg[t]))
            cusum_neg[t] = 0.0  # Reset on alarm
            cooldown = cooldown_days

    return cusum_pos, cusum_neg, alarm_times, alarm_directions, alarm_magnitudes


def _estimate_arl_siegmund(h: float, k: float) -> float:
    """
    Theoretical ARL approximation using Siegmund (1985) formula.

    For one-sided CUSUM with reference k and threshold h:
        ARL_0 ~ exp(2*k*h + 1.166) / (2*k^2)

    For two-sided, the ARL is approximately halved (either side can alarm):
        ARL_two_sided ~ ARL_one_sided / 2

    Parameters
    ----------
    h : float
        Alarm threshold.
    k : float
        Reference value.

    Returns
    -------
    float
        Estimated ARL under H0.
    """
    if k <= 0 or h <= 0:
        return float('inf')
    arl_one_sided = math.exp(2.0 * k * h + 1.166) / (2.0 * k * k)
    return arl_one_sided / 2.0


def innovation_cusum(
    innovations: np.ndarray,
    R: np.ndarray,
    threshold: float = CUSUM_THRESHOLD,
    reference: float = CUSUM_REFERENCE,
) -> CUSUMResult:
    """
    Two-sided Page's CUSUM chart on standardized innovations.

    Detects persistent drift shifts in the Kalman filter's state estimate.
    Implements the classical CUSUM with reset-on-alarm and cooldown to
    prevent alarm clustering.

    When an alarm fires:
      1. CUSUM path resets to zero (allows detection of next shift)
      2. Alarm direction (+1/-1) identifies drift direction
      3. Alarm magnitude indicates severity
      4. q should be boosted via apply_cusum_q_correction() for fast adaptation

    Statistical guarantees:
      - With h=5.0, k=0.5: ARL_0 ~ 500 (false alarm ~1 per 2 years)
      - Detection delay for 1-sigma shift: median ~8-10 days
      - Detection delay for 2-sigma shift: median ~3-4 days

    Parameters
    ----------
    innovations : np.ndarray
        Innovation sequence v_t = r_t - mu_{t|t-1}.
    R : np.ndarray
        Observation noise variance R_t (one per time step).
    threshold : float
        CUSUM alarm threshold h (default 5.0, ARL ~500).
    reference : float
        Reference value k for shift detection (default 0.5 for 1-sigma).

    Returns
    -------
    CUSUMResult
        CUSUM paths, alarm times/directions/magnitudes, and q correction
        parameters for integration with the Kalman filter.
    """
    innovations = np.asarray(innovations, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    n = len(innovations)

    # Edge case: too few observations
    if n < CUSUM_MIN_OBS:
        return CUSUMResult(
            cusum_pos=np.zeros(n),
            cusum_neg=np.zeros(n),
            alarm_times=[],
            alarm_directions=[],
            alarm_magnitudes=[],
            n_alarms=0,
            has_alarm=False,
            q_multiplier=CUSUM_Q_MULTIPLIER,
            alarm_duration=CUSUM_ALARM_DURATION,
            n_obs=n,
            threshold=threshold,
            reference=reference,
            estimated_arl=_estimate_arl_siegmund(threshold, reference),
            max_cusum=0.0,
        )

    # Standardize innovations: z_t = v_t / sqrt(R_t)
    safe_R = np.maximum(R, 1e-20)
    z = innovations / np.sqrt(safe_R)

    # Run CUSUM inner loop
    cusum_pos, cusum_neg, alarm_times, alarm_directions, alarm_magnitudes = (
        _cusum_inner_loop(z, threshold, reference, CUSUM_COOLDOWN_DAYS)
    )

    max_cusum = max(
        float(np.max(cusum_pos)) if n > 0 else 0.0,
        float(np.max(cusum_neg)) if n > 0 else 0.0,
    )

    return CUSUMResult(
        cusum_pos=cusum_pos,
        cusum_neg=cusum_neg,
        alarm_times=alarm_times,
        alarm_directions=alarm_directions,
        alarm_magnitudes=alarm_magnitudes,
        n_alarms=len(alarm_times),
        has_alarm=len(alarm_times) > 0,
        q_multiplier=CUSUM_Q_MULTIPLIER,
        alarm_duration=CUSUM_ALARM_DURATION,
        n_obs=n,
        threshold=threshold,
        reference=reference,
        estimated_arl=_estimate_arl_siegmund(threshold, reference),
        max_cusum=max_cusum,
    )


def apply_cusum_q_correction(
    cusum_result: CUSUMResult,
    q_base: float,
    T: int,
    decay_rate: float = CUSUM_Q_DECAY_RATE,
) -> np.ndarray:
    """
    Generate a time-varying q_t array based on CUSUM alarms.

    At each alarm time, q is boosted to q_base * q_multiplier and then
    decays exponentially back to q_base over alarm_duration days:

        q_t = q_base * (1 + (multiplier - 1) * exp(-decay_rate * days_since_alarm))

    This provides smooth transition back to baseline rather than an
    abrupt drop, which helps the Kalman filter stabilize.

    Parameters
    ----------
    cusum_result : CUSUMResult
        Output from innovation_cusum().
    q_base : float
        Baseline process noise variance.
    T : int
        Length of the q_t array to generate.
    decay_rate : float
        Exponential decay rate (default 0.5, half-life ~1.4 days).

    Returns
    -------
    np.ndarray
        Array of length T with time-varying q_t values.
    """
    q_t = np.full(T, q_base, dtype=np.float64)

    if not cusum_result.has_alarm:
        return q_t

    multiplier = cusum_result.q_multiplier
    duration = cusum_result.alarm_duration

    for alarm_time in cusum_result.alarm_times:
        for d in range(duration):
            idx = alarm_time + d
            if idx >= T:
                break
            # Exponential decay: peak at alarm, decays toward baseline
            boost = (multiplier - 1.0) * math.exp(-decay_rate * d)
            q_candidate = q_base * (1.0 + boost)
            # Take max in case multiple alarms overlap
            if q_candidate > q_t[idx]:
                q_t[idx] = q_candidate

    return q_t


def calibrate_cusum_threshold(
    k: float = CUSUM_REFERENCE,
    target_arl: float = 500.0,
    n_sim: int = 2000,
    sim_length: int = 5000,
    seed: int = 42,
) -> float:
    """
    Monte Carlo calibration of CUSUM threshold for target ARL.

    Simulates n_sim white-noise series of length sim_length and finds
    the threshold h that yields the desired Average Run Length under H0.

    Uses bisection on [1, 20] to find h such that the empirical ARL
    matches target_arl.

    Parameters
    ----------
    k : float
        Reference value (default 0.5).
    target_arl : float
        Target ARL under H0 (default 500).
    n_sim : int
        Number of Monte Carlo simulations (default 2000).
    sim_length : int
        Length of each simulated series (default 5000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    float
        Calibrated threshold h.
    """
    rng = np.random.default_rng(seed)

    # Pre-generate all random samples
    all_z = rng.standard_normal((n_sim, sim_length))

    def _empirical_arl(h: float) -> float:
        """Compute empirical ARL for threshold h."""
        total_rl = 0.0
        n_alarms = 0
        for i in range(n_sim):
            z = all_z[i]
            s_pos = 0.0
            s_neg = 0.0
            run_start = 0
            for t in range(sim_length):
                s_pos = max(0.0, s_pos + z[t] - k)
                s_neg = max(0.0, s_neg - z[t] - k)
                if s_pos > h or s_neg > h:
                    total_rl += (t - run_start + 1)
                    n_alarms += 1
                    # Reset and continue (to get more run-lengths)
                    s_pos = 0.0
                    s_neg = 0.0
                    run_start = t + 1
        if n_alarms == 0:
            return float('inf')
        return total_rl / n_alarms

    # Bisection search
    h_low, h_high = 1.0, 20.0
    for _ in range(30):  # ~30 iterations gives precision < 0.001
        h_mid = (h_low + h_high) / 2.0
        arl_mid = _empirical_arl(h_mid)
        if arl_mid < target_arl:
            h_low = h_mid
        else:
            h_high = h_mid
    return (h_low + h_high) / 2.0


def compute_cusum_diagnostics(
    innovations: np.ndarray,
    R: np.ndarray,
    threshold: float = CUSUM_THRESHOLD,
    reference: float = CUSUM_REFERENCE,
) -> dict:
    """
    Comprehensive CUSUM diagnostics for a single asset's innovation stream.

    Returns a dictionary of diagnostic metrics including CUSUM result,
    alarm statistics, estimated ARL, and q correction schedule.

    Parameters
    ----------
    innovations : np.ndarray
        Innovation sequence.
    R : np.ndarray
        Observation noise variance.
    threshold : float
        CUSUM threshold.
    reference : float
        CUSUM reference value.

    Returns
    -------
    dict
        Diagnostic metrics:
        - 'cusum_result': CUSUMResult object
        - 'q_schedule': np.ndarray of time-varying q (with q_base=1.0)
        - 'alarm_density': alarms per 252 observations (annualized)
        - 'max_cusum_pos': peak positive CUSUM
        - 'max_cusum_neg': peak negative CUSUM
        - 'estimated_arl': theoretical ARL from Siegmund formula
        - 'pct_time_in_alarm': fraction of time under alarm correction
    """
    result = innovation_cusum(innovations, R, threshold, reference)
    n = len(innovations)

    # Generate q schedule with unit base
    q_schedule = apply_cusum_q_correction(result, q_base=1.0, T=n)

    # Annualized alarm density
    alarm_density = (result.n_alarms / max(n, 1)) * 252.0

    # Fraction of time under alarm correction
    pct_in_alarm = float(np.sum(q_schedule > 1.0)) / max(n, 1)

    return {
        'cusum_result': result,
        'q_schedule': q_schedule,
        'alarm_density': alarm_density,
        'max_cusum_pos': float(np.max(result.cusum_pos)) if n > 0 else 0.0,
        'max_cusum_neg': float(np.max(result.cusum_neg)) if n > 0 else 0.0,
        'estimated_arl': result.estimated_arl,
        'pct_time_in_alarm': pct_in_alarm,
    }
