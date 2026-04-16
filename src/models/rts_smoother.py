"""
Epic 21: Rauch-Tung-Striebel Smoother for Improved Retrospective State
========================================================================

Story 21.1: Numba-Compiled RTS Backward Pass
Story 21.2: Smoothed-State Parameter Re-Estimation (EM Cycle)
Story 21.3: Smoothed Innovation Diagnostics

Implements the RTS smoother for scalar (1D) Kalman state, EM parameter
re-estimation from smoothed states, and diagnostic tools for smoothed
innovations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# =====================================================================
# Constants
# =====================================================================

# EM algorithm
EM_MAX_ITER: int = 20
EM_DEFAULT_ITER: int = 5
EM_CONVERGENCE_TOL: float = 1e-6

# Smoother numerical stability
SMOOTHER_P_FLOOR: float = 1e-12
SMOOTHER_Q_FLOOR: float = 1e-15

# Diagnostics
LJUNG_BOX_DEFAULT_LAGS: int = 10
CUSUM_THRESHOLD: float = 4.0


# =====================================================================
# Story 21.1: RTS Smoother Backward Pass
# =====================================================================

@dataclass
class RTSSmootherResult:
    """Result from rts_smoother_backward()."""
    mu_smooth: np.ndarray      # Smoothed state means
    P_smooth: np.ndarray       # Smoothed state variances
    G: np.ndarray              # Smoother gains


def rts_smoother_backward(
    mu_filt: np.ndarray,
    P_filt: np.ndarray,
    mu_pred: np.ndarray,
    P_pred: np.ndarray,
    phi: float,
    q: float,
) -> RTSSmootherResult:
    """
    Rauch-Tung-Striebel smoother backward pass for scalar Kalman filter.

    Given filtered estimates (mu_filt, P_filt) and one-step predictions
    (mu_pred, P_pred) from the forward filter, compute smoothed estimates
    that incorporate future observations.

    The RTS equations (scalar case):
        G_t = P_filt[t] * phi / P_pred[t+1]
        mu_smooth[t] = mu_filt[t] + G_t * (mu_smooth[t+1] - mu_pred[t+1])
        P_smooth[t] = P_filt[t] + G_t^2 * (P_smooth[t+1] - P_pred[t+1])

    Parameters
    ----------
    mu_filt : np.ndarray, shape (T,)
        Filtered state means from forward pass.
    P_filt : np.ndarray, shape (T,)
        Filtered state variances from forward pass.
    mu_pred : np.ndarray, shape (T,)
        Predicted state means: mu_pred[t] = phi * mu_filt[t-1].
    P_pred : np.ndarray, shape (T,)
        Predicted state variances: P_pred[t] = phi^2 * P_filt[t-1] + q.
    phi : float
        State transition coefficient.
    q : float
        Process noise variance.

    Returns
    -------
    RTSSmootherResult
    """
    mu_filt = np.asarray(mu_filt, dtype=np.float64)
    P_filt = np.asarray(P_filt, dtype=np.float64)
    mu_pred = np.asarray(mu_pred, dtype=np.float64)
    P_pred = np.asarray(P_pred, dtype=np.float64)

    T = len(mu_filt)
    if T == 0:
        return RTSSmootherResult(
            mu_smooth=np.array([], dtype=np.float64),
            P_smooth=np.array([], dtype=np.float64),
            G=np.array([], dtype=np.float64),
        )

    mu_s = np.empty(T, dtype=np.float64)
    P_s = np.empty(T, dtype=np.float64)
    G = np.zeros(T, dtype=np.float64)

    # Initialize: last time step, smoothed = filtered
    mu_s[T - 1] = mu_filt[T - 1]
    P_s[T - 1] = P_filt[T - 1]

    # Backward pass
    for t in range(T - 2, -1, -1):
        P_pred_next = P_pred[t + 1] if t + 1 < len(P_pred) else phi ** 2 * P_filt[t] + q
        P_pred_next = max(P_pred_next, SMOOTHER_P_FLOOR)

        G_t = P_filt[t] * phi / P_pred_next
        G[t] = G_t

        mu_pred_next = mu_pred[t + 1] if t + 1 < len(mu_pred) else phi * mu_filt[t]

        mu_s[t] = mu_filt[t] + G_t * (mu_s[t + 1] - mu_pred_next)
        P_s[t] = P_filt[t] + G_t ** 2 * (P_s[t + 1] - P_pred_next)

        # Ensure P_smooth >= 0
        P_s[t] = max(P_s[t], 0.0)

    return RTSSmootherResult(
        mu_smooth=mu_s,
        P_smooth=P_s,
        G=G,
    )


def forward_filter_gaussian(
    returns: np.ndarray,
    vol: np.ndarray,
    phi: float,
    q: float,
    c: float,
    mu_0: float = 0.0,
    P_0: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Simple Gaussian Kalman filter forward pass (scalar).

    State: mu_t = phi * mu_{t-1} + w_t, w_t ~ N(0, q)
    Obs:   r_t = mu_t + e_t, e_t ~ N(0, R_t), R_t = c * vol_t^2

    Returns
    -------
    mu_filt, P_filt, mu_pred, P_pred, log_lik
    """
    returns = np.asarray(returns, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)
    T = len(returns)

    mu_filt = np.empty(T, dtype=np.float64)
    P_filt = np.empty(T, dtype=np.float64)
    mu_pred = np.empty(T, dtype=np.float64)
    P_pred = np.empty(T, dtype=np.float64)

    log_lik = 0.0
    mu_prev = mu_0
    P_prev = P_0

    for t in range(T):
        # Predict
        mu_p = phi * mu_prev
        P_p = phi ** 2 * P_prev + q

        mu_pred[t] = mu_p
        P_pred[t] = P_p

        # Observation noise
        R_t = max(c * vol[t] ** 2, SMOOTHER_Q_FLOOR)

        # Innovation
        v_t = returns[t] - mu_p
        S_t = P_p + R_t

        # Kalman gain
        K_t = P_p / max(S_t, SMOOTHER_P_FLOOR)

        # Update
        mu_filt[t] = mu_p + K_t * v_t
        P_filt[t] = (1.0 - K_t) * P_p

        # Log-likelihood
        log_lik += -0.5 * (math.log(2 * math.pi * S_t) + v_t ** 2 / S_t)

        mu_prev = mu_filt[t]
        P_prev = P_filt[t]

    return mu_filt, P_filt, mu_pred, P_pred, log_lik


# =====================================================================
# Story 21.2: Smoothed-State Parameter Re-Estimation (EM Cycle)
# =====================================================================

@dataclass
class EMResult:
    """Result from em_parameter_update()."""
    q_star: float
    c_star: float
    phi_star: float
    log_likelihoods: List[float]
    n_iter: int
    converged: bool


def em_parameter_update(
    returns: np.ndarray,
    vol: np.ndarray,
    mu_smooth: np.ndarray,
    P_smooth: np.ndarray,
    phi_current: float = 1.0,
) -> Tuple[float, float, float]:
    """
    M-step: update (q, c, phi) from smoothed states.

    q* = (1/T) * sum(P_smooth[t] + (mu_smooth[t] - phi * mu_smooth[t-1])^2)
    c* = (1/T) * sum((r_t - mu_smooth[t])^2 / vol_t^2)
    phi* = sum(mu_smooth[t] * mu_smooth[t-1]) / sum(mu_smooth[t-1]^2)

    Parameters
    ----------
    returns : array of returns
    vol : array of volatilities
    mu_smooth : smoothed state means
    P_smooth : smoothed state variances
    phi_current : current phi estimate

    Returns
    -------
    (q_star, c_star, phi_star)
    """
    returns = np.asarray(returns, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)
    mu_smooth = np.asarray(mu_smooth, dtype=np.float64)
    P_smooth = np.asarray(P_smooth, dtype=np.float64)
    T = len(returns)

    if T < 2:
        return 1e-6, 1.0, phi_current

    # phi update: phi* = sum(mu_s[t] * mu_s[t-1]) / sum(mu_s[t-1]^2)
    num = np.sum(mu_smooth[1:] * mu_smooth[:-1])
    den = np.sum(mu_smooth[:-1] ** 2)
    phi_star = num / max(den, 1e-30)
    phi_star = max(0.5, min(1.0, phi_star))  # Clamp to [0.5, 1.0]

    # q update: q* = (1/T) * sum(P_s[t] + (mu_s[t] - phi * mu_s[t-1])^2)
    state_errors = mu_smooth[1:] - phi_star * mu_smooth[:-1]
    q_star = np.mean(P_smooth[1:] + state_errors ** 2)
    q_star = max(q_star, SMOOTHER_Q_FLOOR)

    # c update: c* = mean((r_t - mu_s[t])^2 / vol_t^2)
    vol_sq = np.maximum(vol ** 2, 1e-20)
    obs_errors = (returns - mu_smooth) ** 2
    c_star = np.mean(obs_errors / vol_sq)
    c_star = max(c_star, 0.01)
    c_star = min(c_star, 10.0)

    return float(q_star), float(c_star), float(phi_star)


def em_fit(
    returns: np.ndarray,
    vol: np.ndarray,
    phi_init: float = 1.0,
    q_init: float = 1e-6,
    c_init: float = 1.0,
    max_iter: int = EM_DEFAULT_ITER,
    tol: float = EM_CONVERGENCE_TOL,
) -> EMResult:
    """
    Full EM algorithm: alternate E-step (filter + smooth) and M-step.

    Parameters
    ----------
    returns : array of returns
    vol : array of volatilities
    phi_init, q_init, c_init : initial parameters
    max_iter : maximum EM iterations
    tol : convergence tolerance (relative change in log-likelihood)

    Returns
    -------
    EMResult
    """
    returns = np.asarray(returns, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)

    phi = phi_init
    q = q_init
    c = c_init

    log_liks: List[float] = []
    converged = False

    for i in range(max_iter):
        # E-step: forward filter + backward smoother
        mu_filt, P_filt, mu_pred, P_pred, log_lik = forward_filter_gaussian(
            returns, vol, phi, q, c,
        )
        log_liks.append(log_lik)

        rts = rts_smoother_backward(mu_filt, P_filt, mu_pred, P_pred, phi, q)

        # M-step: update parameters
        q_new, c_new, phi_new = em_parameter_update(
            returns, vol, rts.mu_smooth, rts.P_smooth, phi,
        )

        # Check convergence
        if i > 0:
            rel_change = abs(log_lik - log_liks[-2]) / max(abs(log_lik), 1.0)
            if rel_change < tol:
                converged = True
                phi, q, c = phi_new, q_new, c_new
                break

        phi, q, c = phi_new, q_new, c_new

    return EMResult(
        q_star=q,
        c_star=c,
        phi_star=phi,
        log_likelihoods=log_liks,
        n_iter=len(log_liks),
        converged=converged,
    )


# =====================================================================
# Story 21.3: Smoothed Innovation Diagnostics
# =====================================================================

@dataclass
class InnovationDiagnostics:
    """Result from smoothed_innovations()."""
    innovations: np.ndarray
    acf: np.ndarray                # Autocorrelation function
    ljung_box_stat: float
    ljung_box_pvalue: float
    cusum: np.ndarray              # Cumulative sum of standardized innovations
    cusum_max: float
    cusum_breach: bool             # Did CUSUM exceed threshold?
    cusum_breach_idx: Optional[int]  # Index of first breach


def smoothed_innovations(
    returns: np.ndarray,
    mu_smooth: np.ndarray,
    vol: Optional[np.ndarray] = None,
    lags: int = LJUNG_BOX_DEFAULT_LAGS,
) -> InnovationDiagnostics:
    """
    Compute innovations from smoothed states and run diagnostics.

    v_t^s = r_t - mu_smooth[t]

    Diagnostics:
    - Autocorrelation function (ACF) up to `lags`
    - Ljung-Box test for residual autocorrelation
    - CUSUM test for drift detection

    Parameters
    ----------
    returns : array of returns
    mu_smooth : smoothed state means
    vol : optional volatilities for standardization
    lags : number of lags for ACF/Ljung-Box

    Returns
    -------
    InnovationDiagnostics
    """
    returns = np.asarray(returns, dtype=np.float64)
    mu_smooth = np.asarray(mu_smooth, dtype=np.float64)
    T = len(returns)

    innovations = returns - mu_smooth

    # Standardize if vol provided
    if vol is not None:
        vol = np.asarray(vol, dtype=np.float64)
        std_innovations = innovations / np.maximum(vol, 1e-10)
    else:
        std_innov_std = np.std(innovations)
        std_innovations = innovations / max(std_innov_std, 1e-10)

    # ACF computation
    n_lags = min(lags, T - 1)
    acf = np.zeros(n_lags + 1)
    acf[0] = 1.0
    innov_centered = innovations - np.mean(innovations)
    var_innov = np.var(innovations)
    if var_innov > 1e-30:
        for k in range(1, n_lags + 1):
            acf[k] = np.mean(innov_centered[k:] * innov_centered[:-k]) / var_innov

    # Ljung-Box statistic
    # Q = T * (T+2) * sum(acf[k]^2 / (T-k)) for k=1..m
    lb_stat = 0.0
    for k in range(1, n_lags + 1):
        lb_stat += acf[k] ** 2 / max(T - k, 1)
    lb_stat *= T * (T + 2)

    # P-value: Q ~ chi-squared(m)
    # Use simple approximation: if lb_stat < critical value, not significant
    # chi2(m) mean = m, so p-value approximate via CDF
    # For simplicity, use scipy-free normal approximation for large m
    # chi2(m) ~ N(m, 2m) for large m
    lb_mean = n_lags
    lb_std = math.sqrt(max(2 * n_lags, 1))
    z_lb = (lb_stat - lb_mean) / max(lb_std, 1e-10)
    # Approximate p-value using standard normal (upper tail)
    lb_pvalue = 0.5 * math.erfc(z_lb / math.sqrt(2))
    lb_pvalue = max(0.0, min(1.0, lb_pvalue))

    # CUSUM test
    cusum = np.cumsum(std_innovations)
    cusum_max = float(np.max(np.abs(cusum)))
    cusum_threshold = CUSUM_THRESHOLD * math.sqrt(T)

    cusum_breach = cusum_max > cusum_threshold
    cusum_breach_idx = None
    if cusum_breach:
        abs_cusum = np.abs(cusum)
        breach_mask = abs_cusum > cusum_threshold
        if np.any(breach_mask):
            cusum_breach_idx = int(np.argmax(breach_mask))

    return InnovationDiagnostics(
        innovations=innovations,
        acf=acf,
        ljung_box_stat=float(lb_stat),
        ljung_box_pvalue=float(lb_pvalue),
        cusum=cusum,
        cusum_max=cusum_max,
        cusum_breach=cusum_breach,
        cusum_breach_idx=cusum_breach_idx,
    )


def compare_innovations(
    returns: np.ndarray,
    mu_filtered: np.ndarray,
    mu_smoothed: np.ndarray,
    vol: Optional[np.ndarray] = None,
    lags: int = LJUNG_BOX_DEFAULT_LAGS,
) -> Dict[str, InnovationDiagnostics]:
    """
    Compare filtered vs smoothed innovations diagnostics.

    Returns
    -------
    dict with keys 'filtered' and 'smoothed', each containing InnovationDiagnostics.
    """
    filt_diag = smoothed_innovations(returns, mu_filtered, vol, lags)
    smooth_diag = smoothed_innovations(returns, mu_smoothed, vol, lags)
    return {"filtered": filt_diag, "smoothed": smooth_diag}
