"""
Volatility fitting: GARCH/GJR-GARCH MLE, OU parameter estimation, cache I/O.

Extracted from tune.py (Story 2.3).
"""
import os
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from tuning.tune_modules.config import *  # noqa: F401,F403
from tuning.tune_modules.utilities import *  # noqa: F401,F403


__all__ = [
    # GARCH constants
    "GARCH_ALPHA_MIN",
    "GARCH_ALPHA_MAX",
    "GARCH_BETA_MIN",
    "GARCH_BETA_MAX",
    "GARCH_OMEGA_MIN",
    "GARCH_PERSISTENCE_MAX",
    "GARCH_STARTS",
    # OU constants
    "OU_HALF_LIFE_MIN_TUNE",
    "OU_HALF_LIFE_MAX_TUNE",
    # Functions
    "gjr_garch_log_likelihood",
    "garch_log_likelihood",
    "fit_garch_mle",
    "fit_ou_params",
    "load_cache",
    "load_single_asset_cache",
    "save_cache_json",
]


# =============================================================================
# Story 3.3: GARCH(1,1) MLE Parameter Fitting
# =============================================================================

GARCH_ALPHA_MIN = 1e-6
GARCH_ALPHA_MAX = 0.50
GARCH_BETA_MIN = 1e-6
GARCH_BETA_MAX = 0.999
GARCH_OMEGA_MIN = 1e-12
GARCH_PERSISTENCE_MAX = 0.999

# -------------------------------------------------------------------------
# Multi-Start GARCH (Story 1.6)
# -------------------------------------------------------------------------
# Five (alpha, beta) starting points covering the typical GARCH parameter
# space.  The optimizer runs from each start and selects the highest LL.
# -------------------------------------------------------------------------
GARCH_STARTS = [
    (0.03, 0.93),  # Low reactivity, high persistence
    (0.08, 0.88),  # Standard (legacy)
    (0.15, 0.80),  # High reactivity
    (0.05, 0.90),  # Medium
    (0.10, 0.85),  # Medium-high reactivity
]


def gjr_garch_log_likelihood(params, returns, barrier_lambda=5.0):
    """
    GJR-GARCH(1,1) negative log-likelihood (Glosten-Jagannathan-Runkle 1993).

    h_t = omega + alpha * e_{t-1}^2 + gamma * e_{t-1}^2 * I(e_{t-1}<0) + beta * h_{t-1}

    params: [omega, alpha, beta, gamma]
    returns: array of mean-adjusted returns
    barrier_lambda: log-barrier penalty weight for stationarity (Story 3.1)

    The leverage term gamma captures asymmetric volatility:
      - Negative returns increase variance by (alpha + gamma) * e^2
      - Positive returns increase variance by alpha * e^2
    """
    omega, alpha, beta, gamma = params
    T = len(returns)

    # Effective persistence including half the leverage effect
    eff_pers = alpha + gamma * 0.5 + beta

    # Story 3.1: log-barrier stationarity penalty
    # -lambda * log(1 - persistence) -> +inf as persistence -> 1
    if eff_pers >= 1.0:
        return 1e15
    barrier_penalty = -barrier_lambda * np.log(max(1.0 - eff_pers, 1e-15))

    if eff_pers < 1.0:
        sigma2_init = omega / (1.0 - eff_pers)
    else:
        sigma2_init = np.var(returns)

    sigma2 = max(sigma2_init, 1e-20)
    total_ll = 0.0

    for t in range(T):
        r = returns[t]
        if sigma2 < 1e-20:
            sigma2 = 1e-20
        total_ll += -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + r * r / sigma2)
        leverage = gamma * r * r * (1.0 if r < 0.0 else 0.0)
        sigma2 = omega + alpha * r * r + leverage + beta * sigma2

    return -total_ll + barrier_penalty  # Negative for minimization


def garch_log_likelihood(params, returns):
    """
    GARCH(1,1) negative log-likelihood (backward compatible wrapper).

    Delegates to GJR-GARCH with gamma=0.
    """
    omega, alpha, beta = params
    return gjr_garch_log_likelihood([omega, alpha, beta, 0.0], returns)


def fit_garch_mle(returns, max_iter=200):
    """
    Fit GJR-GARCH(1,1) via multi-start MLE (Story 1.6).

    Runs SLSQP from 5 starting points covering the (alpha, beta) space,
    selects the solution with highest log-likelihood.  The GJR leverage
    term gamma captures asymmetric volatility.

    Returns dict with:
      - omega, alpha, beta, gamma (GJR leverage)
      - persistence (alpha + gamma/2 + beta)
      - long_run_vol (annualized)
      - converged: bool
      - se_alpha, se_beta, se_gamma (standard errors via Hessian)
      - best_start: (alpha0, beta0) of the winning start
      - n_starts_tried: number of starts attempted
    """
    from scipy.optimize import minimize

    returns = np.asarray(returns, dtype=np.float64)
    returns = returns - np.mean(returns)
    sample_var = float(np.var(returns))

    if sample_var < 1e-20 or len(returns) < 30:
        return None

    bounds = [
        (GARCH_OMEGA_MIN, sample_var * 10),   # omega
        (GARCH_ALPHA_MIN, GARCH_ALPHA_MAX),    # alpha
        (GARCH_BETA_MIN, GARCH_BETA_MAX),      # beta
        (0.0, 0.30),                           # gamma (leverage)
    ]

    # Stationarity: alpha + gamma/2 + beta < 0.999
    constraints = [{
        'type': 'ineq',
        'fun': lambda p: GARCH_PERSISTENCE_MAX - (p[1] + p[3] * 0.5 + p[2]),
    }]

    best_result = None
    best_ll = float('inf')  # minimising neg-LL
    best_start = None
    n_tried = 0

    for alpha0, beta0 in GARCH_STARTS:
        omega0 = sample_var * max(1.0 - alpha0 - beta0, 0.01)
        x0 = [max(omega0, 1e-10), alpha0, beta0, 0.02]  # gamma0=0.02

        try:
            result = minimize(
                gjr_garch_log_likelihood,
                x0,
                args=(returns,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_iter, 'ftol': 1e-10},
            )
            n_tried += 1
            if result.fun < best_ll:
                best_ll = result.fun
                best_result = result
                best_start = (alpha0, beta0)
        except Exception:
            n_tried += 1
            continue

    if best_result is None:
        return None

    omega, alpha, beta, gamma = best_result.x
    persistence = alpha + gamma * 0.5 + beta

    if persistence >= 1.0:
        return None

    long_run_var = omega / (1.0 - persistence) if persistence < 1 else sample_var
    long_run_vol = float(np.sqrt(max(long_run_var, 0.0) * 252))

    # Standard errors via numerical Hessian (central finite differences)
    se_alpha = float('inf')
    se_beta = float('inf')
    se_gamma = float('inf')
    try:
        h = 1e-5
        x_opt = best_result.x
        n_p = len(x_opt)
        hessian = np.zeros((n_p, n_p))
        f0 = best_result.fun
        for i in range(n_p):
            for j in range(i, n_p):
                xpp = x_opt.copy()
                xpn = x_opt.copy()
                xnp = x_opt.copy()
                xnn = x_opt.copy()
                xpp[i] += h; xpp[j] += h
                xpn[i] += h; xpn[j] -= h
                xnp[i] -= h; xnp[j] += h
                xnn[i] -= h; xnn[j] -= h
                hessian[i, j] = (
                    gjr_garch_log_likelihood(xpp, returns)
                    - gjr_garch_log_likelihood(xpn, returns)
                    - gjr_garch_log_likelihood(xnp, returns)
                    + gjr_garch_log_likelihood(xnn, returns)
                ) / (4.0 * h * h)
                hessian[j, i] = hessian[i, j]
        # Hessian of neg-LL -> information matrix
        # SE = sqrt(diag(inv(Hessian)))
        eigvals = np.linalg.eigvalsh(hessian)
        if np.all(eigvals > 1e-12):
            cov = np.linalg.inv(hessian)
            diag = np.diag(cov)
            if diag[1] > 0:
                se_alpha = float(np.sqrt(diag[1]))
            if diag[2] > 0:
                se_beta = float(np.sqrt(diag[2]))
            if diag[3] > 0:
                se_gamma = float(np.sqrt(diag[3]))
    except Exception:
        pass

    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "garch_leverage": float(gamma),  # alias for downstream
        "persistence": float(persistence),
        "long_run_vol": long_run_vol,
        "converged": bool(best_result.success),
        "se_alpha": se_alpha,
        "se_beta": se_beta,
        "se_gamma": se_gamma,
        "best_start": best_start,
        "n_starts_tried": n_tried,
    }


# =============================================================================
# Story 3.4: OU Parameter Estimation in Tuning Pipeline
# =============================================================================

OU_HALF_LIFE_MIN_TUNE = 5
OU_HALF_LIFE_MAX_TUNE = 252


def fit_ou_params(prices):
    """
    Story 3.4: Estimate Ornstein-Uhlenbeck parameters from prices.
    
    AR(1) regression: log_price_t = phi * log_price_{t-1} + c + eps
    kappa = -log(phi) (daily frequency, dt=1)
    
    Returns dict with:
      - kappa: mean reversion speed
      - theta: long-run level (current EWMA of prices)
      - sigma_ou: residual volatility
      - half_life_days: ln(2)/kappa
    """
    prices_arr = np.asarray(prices, dtype=np.float64)
    if len(prices_arr) < 60:
        return None
    
    log_p = np.log(prices_arr[prices_arr > 0])
    if len(log_p) < 60:
        return None
    
    # AR(1) regression: y_t = phi * y_{t-1}
    y = log_p[1:]
    x = log_p[:-1]
    
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    
    num = float(np.sum((x - x_mean) * (y - y_mean)))
    den = float(np.sum((x - x_mean) ** 2))
    
    if den < 1e-20:
        return None
    
    phi = num / den
    phi = max(phi, 0.001)
    phi = min(phi, 0.9999)
    
    kappa = -np.log(phi)
    kappa = np.clip(kappa, np.log(2) / OU_HALF_LIFE_MAX_TUNE, np.log(2) / OU_HALF_LIFE_MIN_TUNE)
    
    half_life = np.log(2) / kappa
    
    # Theta from EWMA with span = half_life
    span = max(int(half_life), 5)
    ew_alpha = 2.0 / (span + 1)
    theta = float(prices_arr[0])
    for p in prices_arr[1:]:
        theta = ew_alpha * float(p) + (1 - ew_alpha) * theta
    
    # Residual vol
    residuals = y - phi * x
    sigma_ou = float(np.std(residuals))
    
    return {
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma_ou": sigma_ou,
        "half_life_days": float(half_life),
    }


# =============================================================================
# CACHE MANAGEMENT - Per-Asset Architecture
# =============================================================================
# Cache is now stored in individual files per asset under:
#   src/data/tune/{SYMBOL}.json
# 
# This enables:
#   - Git-friendly storage (small individual files)
#   - Parallel-safe tuning (no file lock contention)
#   - Incremental updates (re-tune one asset without touching others)
#
# Legacy single-file cache is supported for backward compatibility during migration.
# =============================================================================

# Import per-asset cache module
try:
    from tuning.kalman_cache import (
        load_tuned_params as _load_per_asset,
        save_tuned_params as _save_per_asset,
        load_full_cache as _load_full_cache,
        list_cached_symbols,
        get_cache_stats as get_kalman_cache_stats,
        TUNE_CACHE_DIR,
    )
    PER_ASSET_CACHE_AVAILABLE = True
except ImportError:
    PER_ASSET_CACHE_AVAILABLE = False


def load_cache(cache_json: str) -> Dict[str, Dict]:
    """
    Load existing cache from per-asset files or legacy single JSON file.
    
    The cache_json parameter is kept for backward compatibility but is ignored
    when per-asset cache is available. It falls back to the legacy behavior
    if the per-asset module is not found.
    
    Args:
        cache_json: Path to legacy cache file (used as fallback)
        
    Returns:
        Dict mapping symbol -> params for all cached assets
    """
    # Try per-asset cache first
    if PER_ASSET_CACHE_AVAILABLE:
        cache = _load_full_cache()
        if cache:
            return cache
    
    # Fallback to legacy single-file cache
    # Note: cache_json might be a directory path (from retune), so check isfile()
    if cache_json and os.path.isfile(cache_json):
        try:
            with open(cache_json, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            return {}
    return {}


def load_single_asset_cache(symbol: str, cache_json: str = None) -> Optional[Dict]:
    """
    Load cached parameters for a single asset.
    
    This is more efficient than load_cache() when you only need one asset.
    
    Args:
        symbol: Asset symbol
        cache_json: Path to cache directory or legacy cache file (used as fallback)
        
    Returns:
        Dict with tuned parameters or None if not cached
    """
    if PER_ASSET_CACHE_AVAILABLE:
        return _load_per_asset(symbol)
    
    # Fallback to legacy cache (only if it's a file, not a directory)
    if cache_json and os.path.isfile(cache_json):
        try:
            with open(cache_json, 'r') as f:
                cache = json.load(f)
            return cache.get(symbol)
        except Exception:
            pass
    return None


def save_cache_json(cache: Dict[str, Dict], cache_json: str) -> None:
    """
    Persist cache to per-asset files.
    
    Each asset is saved to its own JSON file in the cache directory.
    The cache_json path is used as the cache directory.
    
    Args:
        cache: Dict mapping symbol -> params
        cache_json: Path to cache directory
    """
    import numpy as np
    
    class NumpyEncoder(json.JSONEncoder):
        """Custom JSON encoder that handles numpy types."""
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return super().default(obj)
    
    # Use per-asset cache
    if PER_ASSET_CACHE_AVAILABLE:
        saved_count = 0
        for symbol, params in cache.items():
            try:
                _save_per_asset(symbol, params)
                saved_count += 1
            except Exception as e:
                print(f"Warning: Failed to save {symbol} to per-asset cache: {e}")
        return
    
    # Fallback to legacy single-file cache (only if per-asset not available)
    # Note: This should never run in normal operation since PER_ASSET_CACHE_AVAILABLE=True
    # Skip if cache_json is a directory (new architecture)
    if os.path.isdir(cache_json):
        print(f"Warning: Legacy cache save skipped - {cache_json} is a directory")
        return
        
    os.makedirs(os.path.dirname(cache_json) if os.path.dirname(cache_json) else '.', exist_ok=True)
    json_temp = cache_json + '.tmp'
    try:
        with open(json_temp, 'w') as f:
            json.dump(cache, f, indent=2, cls=NumpyEncoder)
        os.replace(json_temp, cache_json)
    finally:
        # Clean up temp file if it exists
        if os.path.exists(json_temp):
            try:
                os.remove(json_temp)
            except Exception:
                pass


# =============================================================================
# DRIFT MODEL CLASSES — NOW IN src/models/
# =============================================================================
# The following model classes have been moved to separate files for modularity:
#
#   GaussianDriftModel     → src/models/gaussian.py
#   PhiGaussianDriftModel  → src/models/phi_gaussian.py
#   PhiStudentTDriftModel  → src/models/phi_student_t.py
#
# They are imported at the top of this file from the models package.
#
# Each model class implements:
#   - filter() / filter_phi(): Run Kalman filter with model-specific dynamics
#   - optimize_params() / optimize_params_fixed_nu(): Joint parameter optimization
#   - pit_ks(): PIT/KS calibration test
#
# This refactoring:
#   - Preserves 100% backward compatibility via compatibility wrappers below
#   - Enables easier testing of individual model classes
#   - Allows extension without modifying this file
#   - Reduces tune.py file size by ~1400 lines
# =============================================================================


# Compatibility wrappers to preserve existing API surface

