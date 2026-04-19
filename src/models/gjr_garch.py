"""
GJR-GARCH Innovation Volatility for Kalman Observation Noise
=============================================================

Epic 16: Three stories for time-varying observation noise R_t.

Story 16.1: Post-Filter GJR-GARCH(1,1) on Innovation Sequence
Story 16.2: Iterated Filter-GARCH Cycle (2-Pass Estimation)
Story 16.3: GARCH Forecast Variance for Multi-Horizon Signals

The GJR-GARCH(1,1) model on Kalman innovations:

    h_t = omega + alpha * v_{t-1}^2 + gamma * v_{t-1}^2 * I(v_{t-1}<0) + beta * h_{t-1}

Stationarity constraint: alpha + 0.5*gamma + beta < 1

Using R_t = c * sigma_t^2 * h_t / h_bar (normalized) tightens observation
noise during calm periods and widens after shocks.
"""

import os
import sys
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GJR-GARCH defaults
DEFAULT_OMEGA = 1e-6
DEFAULT_ALPHA = 0.05
DEFAULT_GAMMA = 0.05
DEFAULT_BETA = 0.90

# Stationarity: alpha + 0.5*gamma + beta < 1
STATIONARITY_MARGIN = 0.999

# Optimization bounds
OMEGA_MIN, OMEGA_MAX = 1e-10, 1e-2
ALPHA_MIN, ALPHA_MAX = 1e-6, 0.50
GAMMA_MIN, GAMMA_MAX = 0.0, 0.50
BETA_MIN, BETA_MAX = 0.01, 0.999

# Iterated filter-GARCH
DEFAULT_N_ITER = 3
CONVERGENCE_TOL = 0.1  # nats

# Forecast
STANDARD_HORIZONS = [1, 3, 7, 30]


# ---------------------------------------------------------------------------
# Result Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GARCHFitResult:
    """Result from GJR-GARCH(1,1) MLE fit on innovations."""
    omega: float
    alpha: float
    gamma: float
    beta: float
    log_likelihood: float
    h_series: np.ndarray          # Conditional variances h_t
    unconditional_var: float      # h_bar = omega / (1 - alpha - 0.5*gamma - beta)
    persistence: float            # alpha + 0.5*gamma + beta
    leverage_effect: bool         # gamma > 0
    converged: bool


@dataclass
class IteratedFilterGARCHResult:
    """Result from iterated filter-GARCH cycle."""
    mu_filtered: np.ndarray
    P_filtered: np.ndarray
    mu_pred: np.ndarray
    S_pred: np.ndarray
    log_likelihood: float
    garch_fit: GARCHFitResult
    n_iterations: int
    ll_history: list              # Log-likelihood per iteration
    converged: bool
    bic_improvement: float        # BIC improvement from GARCH adjustment


@dataclass
class GARCHForecastResult:
    """Result from GARCH variance forecast."""
    horizon: int
    forecast_var: float           # E[h_{T+H}]
    unconditional_var: float      # h_bar (long-run variance)
    current_var: float            # h_T
    mean_reversion_factor: float  # (alpha + beta)^{H-1}


# ---------------------------------------------------------------------------
# Story 16.1: Post-Filter GJR-GARCH on Innovation Sequence
# ---------------------------------------------------------------------------

def _gjr_garch_loglik(
    params: np.ndarray,
    innovations: np.ndarray,
) -> float:
    """
    Negative log-likelihood for GJR-GARCH(1,1).

    Parameters
    ----------
    params : array [omega, alpha, gamma, beta]
    innovations : array of Kalman filter innovations v_t

    Returns
    -------
    Negative log-likelihood (for minimization).
    """
    omega, alpha, gamma, beta = params

    # Stationarity check
    persistence = alpha + 0.5 * gamma + beta
    if persistence >= STATIONARITY_MARGIN or omega <= 0:
        return 1e12

    n = len(innovations)
    if n < 3:
        return 1e12

    # Unconditional variance as initial h
    h_bar = omega / max(1.0 - persistence, 1e-8)
    h = max(h_bar, 1e-12)

    nll = 0.0
    for t in range(n):
        v = innovations[t]
        v2 = v * v

        # Clamp h for numerical stability
        if h < 1e-12:
            h = 1e-12

        # Gaussian log-likelihood contribution: -0.5 * (log(2*pi*h) + v^2/h)
        nll += 0.5 * (math.log(2.0 * math.pi * h) + v2 / h)

        # GJR-GARCH recursion for next h
        neg_indicator = 1.0 if v < 0.0 else 0.0
        h = omega + alpha * v2 + gamma * v2 * neg_indicator + beta * h

    return nll


def _gjr_garch_h_series(
    params: np.ndarray,
    innovations: np.ndarray,
) -> np.ndarray:
    """Compute conditional variance series h_t given GARCH parameters."""
    omega, alpha, gamma, beta = params
    n = len(innovations)

    persistence = alpha + 0.5 * gamma + beta
    h_bar = omega / max(1.0 - persistence, 1e-8)
    h = max(h_bar, 1e-12)

    h_series = np.empty(n)
    for t in range(n):
        h_series[t] = h
        v = innovations[t]
        v2 = v * v
        neg_indicator = 1.0 if v < 0.0 else 0.0
        h = omega + alpha * v2 + gamma * v2 * neg_indicator + beta * h
        if h < 1e-12:
            h = 1e-12

    return h_series


def fit_gjr_garch_innovations(
    innovations: np.ndarray,
    R: Optional[np.ndarray] = None,
) -> GARCHFitResult:
    """
    Fit GJR-GARCH(1,1) to Kalman filter innovation sequence.

    Story 16.1: Post-Filter GJR-GARCH on Innovation Sequence.

    The innovation sequence v_t = r_t - mu_{t|t-1} exhibits volatility
    clustering. This function fits:

        h_t = omega + alpha * v_{t-1}^2 + gamma * v_{t-1}^2 * I(v_{t-1}<0) + beta * h_{t-1}

    with stationarity constraint alpha + 0.5*gamma + beta < 1.

    Parameters
    ----------
    innovations : np.ndarray
        Kalman filter innovation sequence v_t = r_t - mu_pred_t.
    R : np.ndarray, optional
        Observation noise R_t from filter. If provided, innovations are
        standardized by sqrt(R_t) before fitting.

    Returns
    -------
    GARCHFitResult
        Fitted parameters, conditional variance series, diagnostics.
    """
    from scipy.optimize import minimize

    innovations = np.asarray(innovations, dtype=np.float64).ravel()

    # Optionally standardize by observation noise
    if R is not None:
        R = np.asarray(R, dtype=np.float64).ravel()
        R_safe = np.maximum(R, 1e-12)
        std_innovations = innovations / np.sqrt(R_safe)
    else:
        std_innovations = innovations.copy()

    # Remove NaN/Inf
    valid = np.isfinite(std_innovations)
    std_innovations = std_innovations[valid]

    n = len(std_innovations)
    if n < 10:
        # Not enough data -- return defaults
        return GARCHFitResult(
            omega=DEFAULT_OMEGA, alpha=DEFAULT_ALPHA,
            gamma=DEFAULT_GAMMA, beta=DEFAULT_BETA,
            log_likelihood=-1e6,
            h_series=np.ones(len(innovations)) * 1e-4,
            unconditional_var=1e-4,
            persistence=DEFAULT_ALPHA + 0.5 * DEFAULT_GAMMA + DEFAULT_BETA,
            leverage_effect=True,
            converged=False,
        )

    # Sample variance for initialization
    sample_var = max(np.var(std_innovations), 1e-10)

    # Multi-start optimization with different initializations
    best_nll = 1e12
    best_params = np.array([DEFAULT_OMEGA, DEFAULT_ALPHA, DEFAULT_GAMMA, DEFAULT_BETA])
    converged = False

    starts = [
        np.array([sample_var * 0.01, 0.05, 0.05, 0.88]),
        np.array([sample_var * 0.05, 0.08, 0.10, 0.80]),
        np.array([sample_var * 0.001, 0.03, 0.02, 0.93]),
    ]

    bounds = [
        (OMEGA_MIN, OMEGA_MAX),
        (ALPHA_MIN, ALPHA_MAX),
        (GAMMA_MIN, GAMMA_MAX),
        (BETA_MIN, BETA_MAX),
    ]

    # Stationarity constraint: alpha + 0.5*gamma + beta < STATIONARITY_MARGIN
    constraints = [{
        'type': 'ineq',
        'fun': lambda p: STATIONARITY_MARGIN - (p[1] + 0.5 * p[2] + p[3]),
    }]

    for x0 in starts:
        try:
            result = minimize(
                _gjr_garch_loglik,
                x0,
                args=(std_innovations,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10},
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x.copy()
                converged = result.success
        except Exception:
            continue

    omega, alpha, gamma_val, beta = best_params

    # Compute persistence and unconditional variance
    persistence = alpha + 0.5 * gamma_val + beta
    if persistence >= 1.0:
        # Force stationarity
        scale = 0.98 / persistence
        alpha *= scale
        gamma_val *= scale
        beta *= scale
        persistence = alpha + 0.5 * gamma_val + beta

    h_bar = omega / max(1.0 - persistence, 1e-8)

    # Compute h series on original innovations
    if R is not None:
        h_series = _gjr_garch_h_series(best_params, std_innovations)
        # Pad back to original length if needed
        full_h = np.ones(len(innovations)) * h_bar
        full_h[valid] = h_series
        h_series = full_h
    else:
        h_series = _gjr_garch_h_series(best_params, std_innovations)
        full_h = np.ones(len(innovations)) * h_bar
        full_h[valid] = h_series
        h_series = full_h

    log_likelihood = -best_nll

    return GARCHFitResult(
        omega=float(omega),
        alpha=float(alpha),
        gamma=float(gamma_val),
        beta=float(beta),
        log_likelihood=float(log_likelihood),
        h_series=h_series,
        unconditional_var=float(h_bar),
        persistence=float(persistence),
        leverage_effect=bool(gamma_val > 1e-6),
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Story 16.2: Iterated Filter-GARCH Cycle (2-Pass Estimation)
# ---------------------------------------------------------------------------

def _run_kalman_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    R_scale: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run phi-Gaussian Kalman filter, optionally with GARCH-scaled R_t.

    Parameters
    ----------
    returns : array of asset returns
    vol : array of EWMA/GK volatility
    q : process noise
    c : observation noise coefficient
    phi : AR(1) coefficient
    R_scale : optional array of GARCH h_t / h_bar scaling factors

    Returns
    -------
    (mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, log_likelihood)
    """
    n = len(returns)
    q_val = float(q)
    c_val = float(c)
    phi_val = float(np.clip(phi, -0.999, 0.999))

    mu = 0.0
    P = 1e-4
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    mu_pred_arr = np.zeros(n)
    S_pred_arr = np.zeros(n)
    log_likelihood = 0.0

    for t in range(n):
        # Prediction
        mu_pred = phi_val * mu
        P_pred = (phi_val ** 2) * P + q_val

        vol_t = float(vol[t])
        R = c_val * (vol_t ** 2)

        # Apply GARCH scaling if provided
        if R_scale is not None:
            R *= float(R_scale[t])

        S = P_pred + R
        if S <= 1e-12:
            S = 1e-12

        mu_pred_arr[t] = mu_pred
        S_pred_arr[t] = S

        r_val = float(returns[t])
        innovation = r_val - mu_pred

        K = P_pred / S
        mu = mu_pred + K * innovation
        P = (1.0 - K) * P_pred
        P = max(P, 1e-12)

        mu_filtered[t] = mu
        P_filtered[t] = P

        log_likelihood += -0.5 * (math.log(2.0 * math.pi * S) + (innovation ** 2) / S)

    return mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, log_likelihood


def iterated_filter_garch(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float = 0.0,
    n_iter: int = DEFAULT_N_ITER,
) -> IteratedFilterGARCHResult:
    """
    Iterated Filter-GARCH Cycle (2-Pass Estimation).

    Story 16.2: The chicken-egg problem -- filter innovations depend on R_t
    which depends on innovations. Solve by iterating:

        Pass 1: Run filter with static R_t -> get innovations -> fit GARCH
        Pass 2: Re-run filter with GARCH-adjusted R_t -> get innovations -> refit GARCH
        ...until convergence (|delta_ll| < 0.1 nats)

    Parameters
    ----------
    returns : array of asset returns
    vol : array of EWMA/GK volatility
    q : process noise
    c : observation noise coefficient
    phi : AR(1) coefficient
    nu : degrees of freedom (unused for Gaussian, reserved for Student-t)
    n_iter : maximum iterations (typically converges in 2-3)

    Returns
    -------
    IteratedFilterGARCHResult with final filter outputs and GARCH fit.
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    vol = np.asarray(vol, dtype=np.float64).ravel()
    n = len(returns)

    ll_history = []
    garch_fit = None
    R_scale = None

    # Compute BIC for initial static filter
    mu_f0, P_f0, mu_p0, S_p0, ll0 = _run_kalman_filter(returns, vol, q, c, phi)
    k_base = 3  # q, c, phi
    bic_base = -2.0 * ll0 + k_base * math.log(n)

    for iteration in range(n_iter):
        # Step 1: Run filter (first pass static, subsequent with GARCH R)
        mu_f, P_f, mu_p, S_p, ll = _run_kalman_filter(
            returns, vol, q, c, phi, R_scale=R_scale,
        )
        ll_history.append(ll)

        # Step 2: Extract innovations
        innovations = returns - mu_p

        # Step 3: Fit GJR-GARCH on innovations
        R_obs = np.array([c * v**2 for v in vol])
        garch_fit = fit_gjr_garch_innovations(innovations, R=R_obs)

        # Step 4: Compute R_scale = h_t / h_bar for next iteration
        h_bar = max(garch_fit.unconditional_var, 1e-12)
        R_scale = garch_fit.h_series / h_bar
        # Clamp scaling to prevent extreme values
        R_scale = np.clip(R_scale, 0.1, 10.0)

        # Check convergence (after at least 2 iterations)
        if iteration > 0:
            delta_ll = abs(ll_history[-1] - ll_history[-2])
            if delta_ll < CONVERGENCE_TOL:
                break

    # Final pass with converged GARCH scaling
    mu_f, P_f, mu_p, S_p, ll_final = _run_kalman_filter(
        returns, vol, q, c, phi, R_scale=R_scale,
    )

    # BIC improvement
    k_garch = k_base + 4  # +4 GARCH params
    bic_garch = -2.0 * ll_final + k_garch * math.log(n)
    bic_improvement = bic_base - bic_garch  # Positive = GARCH better

    n_actual = len(ll_history)
    did_converge = (n_actual < n_iter) or (
        n_actual >= 2 and abs(ll_history[-1] - ll_history[-2]) < CONVERGENCE_TOL
    )

    return IteratedFilterGARCHResult(
        mu_filtered=mu_f,
        P_filtered=P_f,
        mu_pred=mu_p,
        S_pred=S_p,
        log_likelihood=ll_final,
        garch_fit=garch_fit,
        n_iterations=n_actual,
        ll_history=ll_history,
        converged=did_converge,
        bic_improvement=float(bic_improvement),
    )


# ---------------------------------------------------------------------------
# Story 16.3: GARCH Forecast Variance for Multi-Horizon Signals
# ---------------------------------------------------------------------------

def garch_variance_forecast(
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    h_T: float,
    horizon: int,
) -> GARCHForecastResult:
    """
    GARCH-based variance forecast at horizon H.

    Story 16.3: Uses the known GARCH recursion for multi-horizon forecasts:

        E[h_{T+H}] = h_bar + (alpha + 0.5*gamma + beta)^{H-1} * (h_T - h_bar)

    At H=1: captures recent vol regime.
    At H=30: reverts toward unconditional variance (as theory requires).

    For GJR-GARCH, the persistence parameter is alpha + 0.5*gamma + beta
    (accounting for the asymmetric leverage under the assumption that
    P(v<0) = 0.5 for the Gaussian case).

    Parameters
    ----------
    omega : GARCH intercept
    alpha : ARCH coefficient
    gamma : GJR leverage coefficient
    beta : GARCH coefficient
    h_T : current conditional variance
    horizon : forecast horizon H (days)

    Returns
    -------
    GARCHForecastResult with forecast variance and diagnostics.
    """
    # Persistence (expected value of (alpha + gamma*I + beta) under Gaussian)
    persistence = alpha + 0.5 * gamma + beta

    # Unconditional variance
    if persistence >= 1.0:
        h_bar = h_T  # Non-stationary: no mean reversion target
    else:
        h_bar = omega / max(1.0 - persistence, 1e-10)

    # Clamp h_T
    h_T = max(float(h_T), 1e-12)

    # Multi-step forecast: E[h_{T+H}] = h_bar + persistence^{H-1} * (h_T - h_bar)
    H = max(int(horizon), 1)

    if H == 1:
        # One-step: direct GARCH recursion
        # Assume E[v_T^2] = h_T and E[I(v<0)] = 0.5
        forecast_var = omega + (alpha + 0.5 * gamma) * h_T + beta * h_T
    else:
        # Multi-step: geometric decay toward h_bar
        mr_factor = persistence ** (H - 1)
        forecast_var = h_bar + mr_factor * (h_T - h_bar)

    # Ensure non-negative
    forecast_var = max(forecast_var, 1e-12)

    mr_factor = persistence ** max(H - 1, 0)

    return GARCHForecastResult(
        horizon=H,
        forecast_var=float(forecast_var),
        unconditional_var=float(h_bar),
        current_var=float(h_T),
        mean_reversion_factor=float(mr_factor),
    )


def garch_variance_forecast_multi(
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    h_T: float,
    horizons: Optional[list] = None,
) -> list:
    """
    GARCH variance forecasts at multiple horizons.

    Parameters
    ----------
    omega, alpha, gamma, beta : GARCH parameters
    h_T : current conditional variance
    horizons : list of horizons (default: [1, 3, 7, 30])

    Returns
    -------
    List of GARCHForecastResult, one per horizon.
    """
    if horizons is None:
        horizons = STANDARD_HORIZONS

    return [
        garch_variance_forecast(omega, alpha, gamma, beta, h_T, h)
        for h in horizons
    ]


def compute_garch_adjusted_R(
    vol: np.ndarray,
    c: float,
    garch_fit: GARCHFitResult,
) -> np.ndarray:
    """
    Compute GARCH-adjusted observation noise R_t.

    R_t = c * sigma_t^2 * (h_t / h_bar)

    This tightens R during calm innovation periods and widens after shocks.

    Parameters
    ----------
    vol : EWMA/GK volatility array
    c : observation noise coefficient
    garch_fit : fitted GJR-GARCH result

    Returns
    -------
    R_adjusted : array of GARCH-adjusted observation noise
    """
    vol = np.asarray(vol, dtype=np.float64)
    R_base = c * vol ** 2

    h_bar = max(garch_fit.unconditional_var, 1e-12)
    h_ratio = garch_fit.h_series / h_bar
    h_ratio = np.clip(h_ratio, 0.1, 10.0)

    return R_base * h_ratio
