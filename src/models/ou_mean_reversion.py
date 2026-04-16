"""
Epic 20: Mean Reversion Enhancement -- OU Parameter Accuracy
=============================================================

Story 20.1: Multi-Scale Kappa Estimation
Story 20.2: Adaptive Equilibrium with Change-Point Detection (PELT)
Story 20.3: Kappa-Dependent Position Timing (mr_signal_strength)

Implements robust OU mean-reversion parameter estimation by pooling
kappa estimates across multiple observation frequencies and detecting
structural breaks in the equilibrium level.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# Constants
# =====================================================================

# Default observation frequencies (in trading days)
DEFAULT_FREQUENCIES: List[int] = [1, 5, 22]

# Minimum observations per frequency for a valid kappa estimate
MIN_OBS_PER_FREQ: int = 30

# Kappa bounds (continuous-time OU)
KAPPA_MIN: float = 0.001
KAPPA_MAX: float = 0.50

# PELT defaults
PELT_MIN_SEGMENT: int = 22          # Minimum segment length (1 month)
PELT_MAX_CHANGEPOINTS: int = 5      # Safety cap per 2 years
PELT_BIC_PENALTY_FACTOR: float = 2.0  # BIC multiplier for conservatism

# MR signal strength thresholds
MR_STRONG_THRESHOLD: float = 2.0    # |z| > 2 => strong MR signal
MR_WEAK_THRESHOLD: float = 0.5      # |z| < 0.5 => no MR signal
MR_KELLY_STRONG: float = 0.80       # Trade at 80% of Kelly when strong
MR_KELLY_WEAK: float = 0.0          # No trade when weak


# =====================================================================
# Story 20.1: Multi-Scale Kappa Estimation
# =====================================================================

@dataclass
class KappaEstimate:
    """Single-frequency kappa estimate."""
    frequency: int
    kappa: float
    se: float
    n_obs: int
    rho: float             # AR(1) coefficient at this frequency
    r_squared: float       # Regression R^2


@dataclass
class MultiScaleKappaResult:
    """Result from multi_scale_kappa()."""
    kappa_pooled: float
    se_pooled: float
    half_life_days: float
    per_frequency: List[KappaEstimate]
    cv: float              # Coefficient of variation across frequencies
    n_frequencies_used: int


def _estimate_kappa_at_frequency(
    log_prices: np.ndarray,
    freq: int,
) -> Optional[KappaEstimate]:
    """
    Estimate kappa at a given observation frequency using OLS.

    Model: Delta X_t = -kappa * X_t * dt + noise
    where X_t = log_price_t - mean(log_prices) (demeaned).

    At frequency f: subsample every f-th observation, then
    regress (X_{t+f} - X_t) on X_t to get rho_f = exp(-kappa * f).
    """
    if len(log_prices) < freq * MIN_OBS_PER_FREQ:
        return None

    # Subsample at frequency f
    subsampled = log_prices[::freq]
    n = len(subsampled)
    if n < MIN_OBS_PER_FREQ:
        return None

    # Demean
    mu_level = np.mean(subsampled)
    x = subsampled - mu_level

    # AR(1): x_{t+1} = rho * x_t + epsilon
    x_lag = x[:-1]
    x_lead = x[1:]

    if len(x_lag) < 10:
        return None

    # OLS: rho = sum(x_lag * x_lead) / sum(x_lag^2)
    ss_xx = np.sum(x_lag ** 2)
    if ss_xx < 1e-30:
        return None

    rho = np.sum(x_lag * x_lead) / ss_xx

    # Residuals and R^2
    residuals = x_lead - rho * x_lag
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((x_lead - np.mean(x_lead)) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-30) if ss_tot > 1e-30 else 0.0

    # Convert to continuous-time kappa
    # rho = exp(-kappa * dt), dt = freq (in days)
    # kappa = -ln(rho) / dt
    if rho <= 0.0 or rho >= 1.0:
        # rho outside (0, 1) => no valid MR at this frequency
        # rho <= 0 means explosive or oscillatory
        if rho <= 0.0:
            return None
        # rho >= 1 means unit root or explosive
        return None

    kappa_f = -math.log(rho) / freq

    # Standard error via delta method:
    # Var(rho) = (1 - rho^2) / n
    # Var(kappa) = (1 / (rho * freq))^2 * Var(rho)
    n_eff = len(x_lag)
    var_rho = (1.0 - rho ** 2) / max(n_eff, 1)
    dkappa_drho = 1.0 / (rho * freq)
    se_kappa = abs(dkappa_drho) * math.sqrt(max(var_rho, 0.0))

    # Clamp
    kappa_f = max(KAPPA_MIN, min(KAPPA_MAX, kappa_f))
    se_kappa = max(se_kappa, 1e-6)

    return KappaEstimate(
        frequency=freq,
        kappa=kappa_f,
        se=se_kappa,
        n_obs=n_eff,
        rho=rho,
        r_squared=max(0.0, min(1.0, r_squared)),
    )


def multi_scale_kappa(
    prices: np.ndarray,
    frequencies: Optional[List[int]] = None,
) -> MultiScaleKappaResult:
    """
    Estimate kappa at multiple observation frequencies and pool via
    inverse-variance weighting.

    Parameters
    ----------
    prices : array-like
        Price series (not log prices).
    frequencies : list of int
        Observation frequencies in trading days (default: [1, 5, 22]).

    Returns
    -------
    MultiScaleKappaResult
        Pooled kappa estimate with per-frequency details.
    """
    if frequencies is None:
        frequencies = list(DEFAULT_FREQUENCIES)

    prices = np.asarray(prices, dtype=np.float64)
    if len(prices) < 2:
        return MultiScaleKappaResult(
            kappa_pooled=0.05,
            se_pooled=1.0,
            half_life_days=math.log(2) / 0.05,
            per_frequency=[],
            cv=1.0,
            n_frequencies_used=0,
        )

    # Work in log space
    log_prices = np.log(np.maximum(prices, 1e-10))

    estimates: List[KappaEstimate] = []
    for freq in frequencies:
        est = _estimate_kappa_at_frequency(log_prices, freq)
        if est is not None:
            estimates.append(est)

    if len(estimates) == 0:
        return MultiScaleKappaResult(
            kappa_pooled=0.05,
            se_pooled=1.0,
            half_life_days=math.log(2) / 0.05,
            per_frequency=[],
            cv=1.0,
            n_frequencies_used=0,
        )

    # Inverse-variance weighting
    weights = np.array([1.0 / (e.se ** 2) for e in estimates])
    kappas = np.array([e.kappa for e in estimates])

    w_sum = np.sum(weights)
    kappa_pooled = np.sum(weights * kappas) / w_sum
    se_pooled = 1.0 / math.sqrt(w_sum)

    # Clamp
    kappa_pooled = max(KAPPA_MIN, min(KAPPA_MAX, kappa_pooled))
    se_pooled = max(se_pooled, 1e-6)

    # Coefficient of variation
    if len(estimates) > 1:
        kappa_std = np.std(kappas)
        cv = kappa_std / max(abs(kappa_pooled), 1e-10)
    else:
        cv = 0.0

    half_life = math.log(2) / max(kappa_pooled, 1e-10)

    return MultiScaleKappaResult(
        kappa_pooled=kappa_pooled,
        se_pooled=se_pooled,
        half_life_days=half_life,
        per_frequency=estimates,
        cv=cv,
        n_frequencies_used=len(estimates),
    )


# =====================================================================
# Story 20.2: Adaptive Equilibrium with Change-Point Detection (PELT)
# =====================================================================

@dataclass
class ChangePointResult:
    """Result from detect_equilibrium_shift()."""
    change_points: List[int]           # Indices of change points
    n_segments: int
    segment_means: List[float]         # Mean of each segment
    segment_lengths: List[int]
    penalty: float                     # Penalty value used
    total_cost: float                  # Total segmentation cost


def _segment_cost(data: np.ndarray, start: int, end: int) -> float:
    """
    Cost of a segment [start, end) under a Gaussian model.
    Cost = n * log(variance) + n * log(2 * pi) + n
    Simplified: n * log(variance) for comparison.
    """
    seg = data[start:end]
    n = len(seg)
    if n < 2:
        return 0.0
    var = np.var(seg, ddof=0)
    if var < 1e-30:
        return 0.0
    return n * math.log(var)


def detect_equilibrium_shift(
    smoothed_mu: np.ndarray,
    penalty: str = "bic",
    min_segment: Optional[int] = None,
    max_changepoints: Optional[int] = None,
) -> ChangePointResult:
    """
    PELT (Pruned Exact Linear Time) change-point detection on a
    smoothed level estimate.

    Parameters
    ----------
    smoothed_mu : array-like
        Smoothed level estimates (e.g., from Kalman smoother).
    penalty : str
        Penalty type: 'bic' or 'manual'.
    min_segment : int, optional
        Minimum segment length. Default: PELT_MIN_SEGMENT.
    max_changepoints : int, optional
        Maximum number of change points. Default: PELT_MAX_CHANGEPOINTS.

    Returns
    -------
    ChangePointResult
    """
    smoothed_mu = np.asarray(smoothed_mu, dtype=np.float64)
    n = len(smoothed_mu)

    if min_segment is None:
        min_segment = PELT_MIN_SEGMENT
    if max_changepoints is None:
        max_changepoints = PELT_MAX_CHANGEPOINTS

    # Compute penalty value
    if penalty == "bic":
        # BIC penalty: log(n) per change point (with conservatism factor)
        pen_val = PELT_BIC_PENALTY_FACTOR * math.log(max(n, 2))
    else:
        pen_val = PELT_BIC_PENALTY_FACTOR * math.log(max(n, 2))

    if n < 2 * min_segment:
        # Too short for any segmentation
        return ChangePointResult(
            change_points=[],
            n_segments=1,
            segment_means=[float(np.mean(smoothed_mu))] if n > 0 else [0.0],
            segment_lengths=[n],
            penalty=pen_val,
            total_cost=_segment_cost(smoothed_mu, 0, n),
        )

    # PELT algorithm: dynamic programming with pruning
    # F[t] = minimum cost of segmenting data[0:t]
    # last_change[t] = last change point before t
    F = np.full(n + 1, np.inf)
    F[0] = -pen_val  # offset so first segment doesn't pay penalty
    last_change = np.zeros(n + 1, dtype=np.int64)
    admissible = {0}  # pruned set of candidate change points

    for t in range(min_segment, n + 1):
        new_admissible = set()
        best_f = np.inf
        best_s = 0

        for s in admissible:
            seg_len = t - s
            if seg_len < min_segment:
                new_admissible.add(s)
                continue

            cost = F[s] + _segment_cost(smoothed_mu, s, t) + pen_val
            if cost < best_f:
                best_f = cost
                best_s = s

            # PELT pruning: keep s if it could still be optimal
            if cost <= F[t] + pen_val if F[t] < np.inf else True:
                new_admissible.add(s)

        F[t] = best_f
        last_change[t] = best_s
        new_admissible.add(t)
        admissible = new_admissible

    # Backtrack to find change points
    cps: List[int] = []
    pos = n
    while pos > 0:
        cp = int(last_change[pos])
        if cp > 0:
            cps.append(cp)
        pos = cp

    cps.sort()

    # Enforce max_changepoints: keep the most significant ones
    if len(cps) > max_changepoints:
        # Rank by significance: difference in segment means
        significances = []
        boundaries = [0] + cps + [n]
        for i, cp in enumerate(cps):
            left_mean = np.mean(smoothed_mu[boundaries[i]:cp])
            right_mean = np.mean(smoothed_mu[cp:boundaries[i + 2]])
            significances.append(abs(right_mean - left_mean))
        # Keep top max_changepoints
        ranked = sorted(range(len(cps)), key=lambda i: significances[i], reverse=True)
        kept = sorted(ranked[:max_changepoints])
        cps = [cps[i] for i in kept]

    # Compute segment statistics
    boundaries = [0] + cps + [n]
    segment_means = []
    segment_lengths = []
    for i in range(len(boundaries) - 1):
        seg = smoothed_mu[boundaries[i]:boundaries[i + 1]]
        segment_means.append(float(np.mean(seg)) if len(seg) > 0 else 0.0)
        segment_lengths.append(len(seg))

    return ChangePointResult(
        change_points=cps,
        n_segments=len(segment_means),
        segment_means=segment_means,
        segment_lengths=segment_lengths,
        penalty=pen_val,
        total_cost=float(F[n]),
    )


def adaptive_equilibrium(
    smoothed_mu: np.ndarray,
    change_points: List[int],
) -> np.ndarray:
    """
    Compute adaptive equilibrium that resets at change points.

    After each change point, the equilibrium becomes the mean of the
    post-change-point segment.

    Parameters
    ----------
    smoothed_mu : array-like
        Smoothed level estimates.
    change_points : list of int
        Detected change point locations.

    Returns
    -------
    np.ndarray
        Adaptive equilibrium at each time step.
    """
    smoothed_mu = np.asarray(smoothed_mu, dtype=np.float64)
    n = len(smoothed_mu)
    equilibrium = np.empty(n, dtype=np.float64)

    boundaries = [0] + sorted(change_points) + [n]
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        seg_mean = np.mean(smoothed_mu[start:end]) if end > start else 0.0
        equilibrium[start:end] = seg_mean

    return equilibrium


# =====================================================================
# Story 20.3: Kappa-Dependent Position Timing
# =====================================================================

@dataclass
class MRSignalResult:
    """Result from mr_signal_strength()."""
    z: float               # Normalized MR signal: kappa * (X - mu) / sigma
    strength: str          # 'strong', 'moderate', 'weak', 'none'
    kelly_fraction: float  # Fraction of Kelly to use
    direction: int         # -1 (short / revert down) or +1 (long / revert up)
    kappa: float
    distance: float        # |X - mu|


def mr_signal_strength(
    price: float,
    equilibrium: float,
    kappa: float,
    sigma: float,
) -> MRSignalResult:
    """
    Compute mean-reversion signal strength.

    z = kappa * (X - mu) / sigma

    - |z| > 2.0: strong MR signal (trade at 80% of Kelly)
    - |z| in [0.5, 2.0]: moderate (linear interpolation)
    - |z| < 0.5: no MR signal (near equilibrium)

    Parameters
    ----------
    price : float
        Current price (or log price).
    equilibrium : float
        Current equilibrium estimate.
    kappa : float
        Mean-reversion speed.
    sigma : float
        Current volatility.

    Returns
    -------
    MRSignalResult
    """
    sigma = max(sigma, 1e-10)
    kappa = max(kappa, 0.0)

    distance = price - equilibrium
    z = kappa * distance / sigma

    abs_z = abs(z)

    if abs_z >= MR_STRONG_THRESHOLD:
        strength = "strong"
        kelly_fraction = MR_KELLY_STRONG
    elif abs_z >= MR_WEAK_THRESHOLD:
        strength = "moderate"
        # Linear interpolation between weak and strong
        t = (abs_z - MR_WEAK_THRESHOLD) / (MR_STRONG_THRESHOLD - MR_WEAK_THRESHOLD)
        kelly_fraction = MR_KELLY_WEAK + t * (MR_KELLY_STRONG - MR_KELLY_WEAK)
    else:
        strength = "none"
        kelly_fraction = MR_KELLY_WEAK

    # Direction: if price > equilibrium, expect reversion down => short (-1)
    # if price < equilibrium, expect reversion up => long (+1)
    if abs(distance) < 1e-15:
        direction = 0
    else:
        direction = -1 if distance > 0 else 1

    return MRSignalResult(
        z=float(z),
        strength=strength,
        kelly_fraction=float(kelly_fraction),
        direction=direction,
        kappa=float(kappa),
        distance=float(abs(distance)),
    )


def mr_signal_strength_array(
    prices: np.ndarray,
    equilibrium: np.ndarray,
    kappa: float,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized MR signal strength for arrays.

    Returns
    -------
    z : np.ndarray
        Normalized MR signal at each time step.
    kelly_fraction : np.ndarray
        Kelly fraction at each time step.
    direction : np.ndarray
        Direction at each time step (-1, 0, +1).
    """
    prices = np.asarray(prices, dtype=np.float64)
    equilibrium = np.asarray(equilibrium, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = np.maximum(sigma, 1e-10)

    distance = prices - equilibrium
    z = kappa * distance / sigma
    abs_z = np.abs(z)

    # Kelly fraction: piecewise linear
    kelly = np.zeros_like(z)
    # Strong: |z| >= 2.0
    strong = abs_z >= MR_STRONG_THRESHOLD
    kelly[strong] = MR_KELLY_STRONG

    # Moderate: 0.5 <= |z| < 2.0
    moderate = (abs_z >= MR_WEAK_THRESHOLD) & (~strong)
    t = (abs_z[moderate] - MR_WEAK_THRESHOLD) / (MR_STRONG_THRESHOLD - MR_WEAK_THRESHOLD)
    kelly[moderate] = MR_KELLY_WEAK + t * (MR_KELLY_STRONG - MR_KELLY_WEAK)

    # Direction: -1 if above equilibrium (expect reversion down), +1 if below
    direction = np.where(distance > 1e-15, -1, np.where(distance < -1e-15, 1, 0))

    return z, kelly, direction.astype(np.float64)
