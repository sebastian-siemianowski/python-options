"""
Continuous nu Estimation for Student-t Models.

Story 8.1: Golden-section profile likelihood refinement of nu.
Story 8.2: Regime-conditional nu estimates.
Story 8.3: VIX-conditional tail thickness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar


# =============================================================================
# STORY 8.1: CONTINUOUS NU REFINEMENT VIA PROFILE LIKELIHOOD
# =============================================================================
#
# The BMA grid (nu in {3, 4, 8, 20}) is coarse.  A stock with true nu=5.7
# loses ~5-15 BIC nats by being forced to nu=4 or nu=8.
#
# Golden-section search on the profile log-likelihood (fixing q, c, phi)
# finds nu* in [2.1, 50] with tolerance < 0.1 in ~15 function evaluations.
#
# BIC improvement = 2 * (ll_refined - ll_grid)  [same k, so k*log(n) cancels]
# =============================================================================

NU_MIN = 2.1
NU_MAX = 50.0
NU_SEARCH_MARGIN_LO = 2.0    # Search from nu_grid_best - 2
NU_SEARCH_MARGIN_HI = 4.0    # Search to nu_grid_best + 4
NU_GOLDEN_XTOL = 0.1         # Tolerance on nu (< 0.1 required)
NU_GOLDEN_MAXITER = 30        # Max iterations (15 typical for convergence)


@dataclass
class NuRefinementResult:
    """Result of continuous nu refinement."""
    nu_refined: float            # Optimized nu (continuous, in [2.1, 50])
    nu_grid: float               # Input grid nu for comparison
    ll_refined: float            # Log-likelihood at nu_refined
    ll_grid: float               # Log-likelihood at nu_grid
    bic_improvement: float       # BIC(grid) - BIC(refined) = 2*(ll_refined - ll_grid)
    converged: bool              # Whether optimizer converged
    n_evaluations: int           # Number of likelihood evaluations


FilterFunc = Callable[[np.ndarray, np.ndarray, float, float, float, float],
                      Tuple[np.ndarray, np.ndarray, float]]


def _student_t_log_likelihood(
    returns: np.ndarray,
    vol: np.ndarray,
    nu: float,
) -> float:
    """
    Compute Student-t log-likelihood for standardized returns.

    Used as a lightweight filter function when no Kalman filter is available.
    Standardizes returns by vol and evaluates the t-distribution density.

    Parameters
    ----------
    returns : np.ndarray
        Return series.
    vol : np.ndarray
        Volatility series (must be positive).
    nu : float
        Degrees of freedom.

    Returns
    -------
    float
        Total log-likelihood.
    """
    from scipy.special import gammaln

    n = len(returns)
    if n == 0 or nu <= 2.0:
        return -1e12

    # Standardize
    safe_vol = np.maximum(vol, 1e-10)
    z = returns / safe_vol

    # Student-t log-density: log t(z | nu)
    # = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*log(nu*pi) - ((nu+1)/2)*log(1 + z^2/nu)
    # Minus log(vol) for the change of variables
    const = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * math.log(nu * math.pi)
    )
    ll_per = const - ((nu + 1.0) / 2.0) * np.log(1.0 + z ** 2 / nu) - np.log(safe_vol)

    return float(np.sum(ll_per))


def refine_nu_continuous(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu_grid_best: float,
    filter_func: Optional[FilterFunc] = None,
    nu_lo: float = NU_MIN,
    nu_hi: float = NU_MAX,
    xtol: float = NU_GOLDEN_XTOL,
) -> NuRefinementResult:
    """
    Refine nu via golden-section search on profile log-likelihood.

    Fixes (q, c, phi) at their optimized values and searches for the nu
    that maximizes the filter log-likelihood over a neighborhood of the
    grid optimum.

    The search range is [nu_grid_best - 2, nu_grid_best + 4], clipped to
    [nu_lo, nu_hi].  Golden-section search (Brent's method) converges in
    ~15 evaluations with tolerance < 0.1.

    Parameters
    ----------
    returns : np.ndarray
        Return series.
    vol : np.ndarray
        Volatility series.
    q : float
        Process noise.
    c : float
        Observation noise scaling.
    phi : float
        State transition parameter.
    nu_grid_best : float
        Best nu from discrete grid search (starting point).
    filter_func : callable, optional
        Callable(returns, vol, q, c, phi, nu) -> (mu_arr, P_arr, log_likelihood).
        If None, uses a direct Student-t log-likelihood on standardized returns.
    nu_lo, nu_hi : float
        Absolute bounds for nu search.
    xtol : float
        Tolerance on nu (default 0.1, satisfying delta_nu < 0.1).

    Returns
    -------
    NuRefinementResult
        Refined nu, log-likelihoods, BIC improvement, and diagnostics.
    """
    n_evals = 0

    # Default filter: direct Student-t log-likelihood
    if filter_func is None:
        def _default_filter(ret, v, q_, c_, phi_, nu_):
            ll = _student_t_log_likelihood(ret, v, nu_)
            return np.zeros(len(ret)), np.zeros(len(ret)), ll
        filter_func = _default_filter

    # Compute log-likelihood at grid point
    try:
        _, _, ll_grid = filter_func(returns, vol, q, c, phi, nu_grid_best)
        if not math.isfinite(ll_grid):
            ll_grid = -1e12
    except Exception:
        ll_grid = -1e12

    # Search range
    search_lo = max(nu_lo, nu_grid_best - NU_SEARCH_MARGIN_LO)
    search_hi = min(nu_hi, nu_grid_best + NU_SEARCH_MARGIN_HI)
    if search_lo >= search_hi:
        search_lo = max(nu_lo, nu_grid_best - 1.0)
        search_hi = min(nu_hi, nu_grid_best + 2.0)
    if search_lo >= search_hi:
        # Cannot search, return grid value
        return NuRefinementResult(
            nu_refined=nu_grid_best,
            nu_grid=nu_grid_best,
            ll_refined=ll_grid,
            ll_grid=ll_grid,
            bic_improvement=0.0,
            converged=False,
            n_evaluations=1,
        )

    def neg_ll(nu_val):
        nonlocal n_evals
        n_evals += 1
        if nu_val <= 2.0 or nu_val > 50.0:
            return 1e12
        try:
            _, _, ll = filter_func(returns, vol, q, c, phi, nu_val)
            return -ll if math.isfinite(ll) else 1e12
        except Exception:
            return 1e12

    result = minimize_scalar(
        neg_ll,
        bounds=(search_lo, search_hi),
        method='bounded',
        options={'xatol': xtol, 'maxiter': NU_GOLDEN_MAXITER},
    )

    actually_converged = bool(result.success and result.fun < 1e11)
    if actually_converged:
        nu_refined = float(result.x)
        ll_refined = float(-result.fun)
    else:
        nu_refined = nu_grid_best
        ll_refined = ll_grid

    # BIC improvement: since k is the same for grid and refined,
    # BIC(grid) - BIC(refined) = (-2*ll_grid + k*log(n)) - (-2*ll_refined + k*log(n))
    #                           = 2 * (ll_refined - ll_grid)
    bic_improvement = 2.0 * (ll_refined - ll_grid)

    return NuRefinementResult(
        nu_refined=nu_refined,
        nu_grid=nu_grid_best,
        ll_refined=ll_refined,
        ll_grid=ll_grid,
        bic_improvement=bic_improvement,
        converged=actually_converged,
        n_evaluations=n_evals,
    )


# =============================================================================
# STORY 8.2: REGIME-CONDITIONAL NU ESTIMATION
# =============================================================================
#
# Different regimes have different tail behavior:
#   - CRISIS (regime 4): heavy tails (low nu)
#   - LOW_VOL_TREND (regime 0): light tails (high nu)
#
# For each regime with >= MIN_REGIME_SAMPLES observations, we run
# golden-section refinement on the regime subset.  Regimes with too
# few samples borrow the global nu estimate.
# =============================================================================

MIN_REGIME_SAMPLES = 50  # Minimum samples for regime-specific nu estimation

# Regime labels (consistent with tune.py and signals.py)
REGIME_LOW_VOL_TREND = 0
REGIME_HIGH_VOL_TREND = 1
REGIME_LOW_VOL_RANGE = 2
REGIME_HIGH_VOL_RANGE = 3
REGIME_CRISIS_JUMP = 4


@dataclass
class RegimeNuResult:
    """Result of regime-conditional nu estimation."""
    regime_nus: Dict[int, float]        # {regime_id: nu_refined}
    global_nu: float                     # Fallback global nu
    regime_counts: Dict[int, int]        # {regime_id: n_samples}
    regime_ll: Dict[int, float]          # {regime_id: log_likelihood}
    borrowed_regimes: list               # Regimes that borrowed from global
    total_bic_improvement: float         # vs single global nu


def regime_nu_estimates(
    returns: np.ndarray,
    vol: np.ndarray,
    regime_labels: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu_global: float = 8.0,
    filter_func: Optional[FilterFunc] = None,
    min_samples: int = MIN_REGIME_SAMPLES,
) -> RegimeNuResult:
    """
    Estimate regime-specific nu values via profile likelihood.

    For each regime with sufficient samples, refines nu independently.
    Regimes with < min_samples observations borrow the global estimate.

    Parameters
    ----------
    returns : np.ndarray
        Full return series.
    vol : np.ndarray
        Full volatility series.
    regime_labels : np.ndarray
        Integer regime labels (0-4) for each observation.
    q, c, phi : float
        Fixed filter parameters.
    nu_global : float
        Global nu estimate (starting point and fallback).
    filter_func : callable, optional
        Callable(returns, vol, q, c, phi, nu) -> (mu, P, ll).
    min_samples : int
        Minimum samples for separate regime estimation.

    Returns
    -------
    RegimeNuResult
        Per-regime nu estimates and diagnostics.
    """
    n = len(returns)
    regime_labels = np.asarray(regime_labels, dtype=int)

    # Count samples per regime
    unique_regimes = np.unique(regime_labels)
    regime_counts = {}
    for r in unique_regimes:
        regime_counts[int(r)] = int(np.sum(regime_labels == r))

    # Compute global nu first (used as fallback)
    global_result = refine_nu_continuous(
        returns, vol, q, c, phi, nu_global,
        filter_func=filter_func,
    )
    global_nu = global_result.nu_refined

    # Per-regime estimation
    regime_nus = {}
    regime_ll = {}
    borrowed_regimes = []

    # Compute global log-likelihood for BIC comparison
    ll_global_total = 0.0

    for r in unique_regimes:
        r_int = int(r)
        mask = regime_labels == r
        r_returns = returns[mask]
        r_vol = vol[mask]
        n_r = len(r_returns)

        if n_r < min_samples:
            # Borrow from global
            regime_nus[r_int] = global_nu
            borrowed_regimes.append(r_int)
            # Compute ll at global nu for this regime
            if filter_func is not None:
                try:
                    _, _, ll = filter_func(r_returns, r_vol, q, c, phi, global_nu)
                    regime_ll[r_int] = float(ll) if math.isfinite(ll) else 0.0
                except Exception:
                    regime_ll[r_int] = 0.0
            else:
                regime_ll[r_int] = _student_t_log_likelihood(r_returns, r_vol, global_nu)
            ll_global_total += regime_ll[r_int]
        else:
            # Refine nu for this regime
            result = refine_nu_continuous(
                r_returns, r_vol, q, c, phi, nu_global,
                filter_func=filter_func,
            )
            regime_nus[r_int] = result.nu_refined
            regime_ll[r_int] = result.ll_refined
            # Also compute ll at global for comparison
            if filter_func is not None:
                try:
                    _, _, ll_g = filter_func(r_returns, r_vol, q, c, phi, global_nu)
                    ll_global_total += float(ll_g) if math.isfinite(ll_g) else 0.0
                except Exception:
                    ll_global_total += 0.0
            else:
                ll_global_total += _student_t_log_likelihood(r_returns, r_vol, global_nu)

    # Total BIC improvement: sum of per-regime ll vs global nu ll
    ll_regime_total = sum(regime_ll.values())
    # Extra parameters: one nu per non-borrowed regime minus 1 global
    n_extra_params = len([r for r in unique_regimes if int(r) not in borrowed_regimes]) - 1
    # BIC improvement = 2*(ll_regime - ll_global) - n_extra_params * log(n)
    total_bic_improvement = 2.0 * (ll_regime_total - ll_global_total) - n_extra_params * math.log(n)

    return RegimeNuResult(
        regime_nus=regime_nus,
        global_nu=global_nu,
        regime_counts=regime_counts,
        regime_ll=regime_ll,
        borrowed_regimes=borrowed_regimes,
        total_bic_improvement=total_bic_improvement,
    )


# =============================================================================
# STORY 8.3: VIX-CONDITIONAL NU (TAIL THICKNESS COUPLING)
# =============================================================================
#
# VIX is a leading indicator for tail fattening across equities.
# When VIX spikes, individual asset tails fatten before asset-level vol
# confirms it. This function adjusts nu downward proportionally to
# VIX excess above its historical median.
#
# Formula:
#   nu_t = max(nu_min, nu_base - kappa * max(0, VIX - VIX_median))
#
# When VIX < VIX_median: nu_t = nu_base (no unnecessary fattening)
# When VIX > 30 (median=18): nu drops by kappa * 12 >= 2
# =============================================================================

VIX_KAPPA_DEFAULT = 0.17       # Drop per VIX point above median
VIX_MEDIAN_DEFAULT = 18.0      # Historical VIX median
NU_FLOOR = 2.1                 # Absolute minimum nu (must be > 2 for finite variance)


def vix_conditional_nu(
    nu_base: float,
    vix_current: float,
    vix_median: float = VIX_MEDIAN_DEFAULT,
    kappa: float = VIX_KAPPA_DEFAULT,
    nu_min: float = NU_FLOOR,
) -> float:
    """
    Compute VIX-conditional tail thickness parameter.

    When VIX is elevated above its median, nu decreases (heavier tails).
    When VIX is at or below the median, nu_base is returned unchanged.

    Formula:
        nu_t = max(nu_min, nu_base - kappa * max(0, VIX - VIX_median))

    Parameters
    ----------
    nu_base : float
        Base degrees of freedom (from model fitting).
    vix_current : float
        Current VIX level.
    vix_median : float
        Historical VIX median (default 18.0).
    kappa : float
        Sensitivity: nu reduction per VIX point above median.
    nu_min : float
        Floor for nu (must be > 2 for finite variance).

    Returns
    -------
    float
        Adjusted nu_t >= nu_min.
    """
    vix_excess = max(0.0, vix_current - vix_median)
    nu_t = nu_base - kappa * vix_excess
    return max(nu_min, nu_t)
