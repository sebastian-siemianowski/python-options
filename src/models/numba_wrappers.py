"""
Wrapper functions that handle scipy calls before invoking Numba kernels.

Separation of concerns:
    - Numba kernels: Pure numeric loops, no Python objects
    - Wrappers: Array preparation, gamma precomputation, fallback handling

Architectural Invariant:
    There is NO bare Student-t wrapper. All Student-t filtering requires φ.

Model variant mapping:
    - Gaussian base: gaussian_filter_kernel
    - φ-Gaussian: phi_gaussian_filter_kernel
    - φ-Student-t: phi_student_t_filter_kernel (the ONLY Student-t variant)
    - φ-Gaussian+Mom (CRSP/CELH/DPRO): momentum_phi_gaussian_filter_kernel
    - φ-Student-t+Mom (GLDW/MAGD/BKSY/ASTS): momentum_phi_student_t_filter_kernel

Author: Quantitative Systems Team
Date: 2026-02-04
"""

from typing import Tuple, Dict, List, Optional
from functools import lru_cache
import numpy as np

# Try to import scipy for gamma precomputation
try:
    from scipy.special import gammaln
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# Try to import Numba kernels
try:
    from .numba_kernels import (
        gaussian_filter_kernel,
        phi_gaussian_filter_kernel,
        phi_student_t_filter_kernel,
        momentum_phi_gaussian_filter_kernel,
        momentum_phi_student_t_filter_kernel,
        student_t_filter_with_lfo_cv_kernel,
        gaussian_filter_with_lfo_cv_kernel,
        ms_q_student_t_filter_kernel,
        unified_phi_student_t_filter_kernel,
        unified_phi_student_t_filter_extended_kernel,
        gaussian_cv_test_fold_kernel,
        phi_gaussian_cv_test_fold_kernel,
        phi_gaussian_train_state_kernel,
        gas_q_filter_gaussian_kernel,
        build_garch_kernel,
        chi2_ewm_correction_kernel,
        pit_var_stretching_kernel,
        phi_student_t_cv_test_fold_kernel,
        phi_student_t_cv_test_fold_stats_kernel,
        phi_student_t_train_state_kernel,
        phi_student_t_improved_train_state_kernel,
        compute_ms_process_noise_ewm_kernel,
        stage6_ewm_fold_kernel,
        ewm_mu_correction_kernel,
        gaussian_score_fold_kernel,
        student_t_cdf_array_kernel,
        student_t_pdf_array_kernel,
        crps_student_t_kernel,
        crps_student_t_numerical_kernel,
        pit_ks_unified_kernel,
        garch_variance_kernel,
        phi_gaussian_filter_with_predictive_kernel,
        phi_student_t_augmented_filter_kernel,
        # AD tail-correction kernels (March 2026)
        ad_twsc_kernel,
        ad_sptg_cdf_student_t_array,
        ad_sptg_cdf_gaussian_array,
    )
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


# =============================================================================
# AVAILABILITY CHECKS
# =============================================================================

def is_numba_available() -> bool:
    """Check if Numba kernels compiled successfully."""
    return _NUMBA_AVAILABLE


def is_scipy_available() -> bool:
    """Check if scipy is available for gamma precomputation."""
    return _SCIPY_AVAILABLE


# =============================================================================
# ARRAY PREPARATION
# =============================================================================

def prepare_arrays(*arrays) -> Tuple[np.ndarray, ...]:
    """
    Ensure arrays are contiguous float64 for Numba.
    
    This is CRITICAL for Numba performance - non-contiguous arrays
    cause massive slowdowns due to cache misses.
    
    Uses ravel() instead of flatten() to avoid unnecessary copies
    when arrays are already 1D. Fast-path skips conversion entirely
    for arrays already in the right format.
    
    Performance: specialized fast paths for 1-2 arrays (common case)
    avoid list building and tuple conversion overhead.
    Called 71K+ times per asset — every microsecond counts.
    """
    # Fast path for 2 arrays (the overwhelmingly common case: returns, vol)
    _f64 = np.float64
    if len(arrays) == 2:
        a, b = arrays
        a_ok = a.dtype == _f64 and a.ndim == 1 and a.flags['C_CONTIGUOUS']
        b_ok = b.dtype == _f64 and b.ndim == 1 and b.flags['C_CONTIGUOUS']
        if a_ok and b_ok:
            return (a, b)
        return (
            a if a_ok else np.ascontiguousarray(a.ravel(), dtype=_f64),
            b if b_ok else np.ascontiguousarray(b.ravel(), dtype=_f64),
        )
    # Fast path for 1 array
    if len(arrays) == 1:
        a = arrays[0]
        if a.dtype == _f64 and a.ndim == 1 and a.flags['C_CONTIGUOUS']:
            return (a,)
        return (np.ascontiguousarray(a.ravel(), dtype=_f64),)
    # General path
    result = []
    for arr in arrays:
        if (arr.dtype == _f64 and arr.ndim == 1
                and arr.flags['C_CONTIGUOUS']):
            result.append(arr)
        else:
            result.append(np.ascontiguousarray(arr.ravel(), dtype=_f64))
    return tuple(result)


# =============================================================================
# ARRAY PREPARATION AUDIT — Story 13.1
# =============================================================================

def validate_prepare_arrays(*arrays) -> dict:
    """
    Audit prepare_arrays() output for float64 precision and C-contiguity.

    Returns diagnostic dict with validation results for each array.
    """
    prepared = prepare_arrays(*arrays)
    diagnostics = []
    for i, arr in enumerate(prepared):
        diag = {
            "index": i,
            "dtype": str(arr.dtype),
            "is_float64": arr.dtype == np.float64,
            "is_c_contiguous": arr.flags["C_CONTIGUOUS"],
            "ndim": arr.ndim,
            "is_1d": arr.ndim == 1,
            "shape": arr.shape,
            "has_nan": bool(np.any(np.isnan(arr))) if arr.size > 0 else False,
            "has_inf": bool(np.any(np.isinf(arr))) if arr.size > 0 else False,
        }
        diag["valid"] = diag["is_float64"] and diag["is_c_contiguous"] and diag["is_1d"]
        diagnostics.append(diag)

    return {
        "n_arrays": len(prepared),
        "all_valid": all(d["valid"] for d in diagnostics),
        "arrays": diagnostics,
        "prepared": prepared,
    }


# =============================================================================
# GAMMA PRECOMPUTATION (for φ-Student-t)
# =============================================================================

@lru_cache(maxsize=64)
def precompute_gamma_values(nu: float) -> Tuple[float, float]:
    """
    Precompute gamma function values for Student-t.
    
    Using scipy.special.gammaln ensures correctness at low ν
    where Stirling's approximation has significant error.
    
    This is why we precompute in Python rather than approximating in Numba:
    at ν=4, Stirling error can flip BMA model rankings.
    
    Cached via lru_cache: ν takes only a handful of values (3, 4, 5, 6, 8, 10, 12, 20)
    across an entire tuning run, so 58K+ calls collapse to ~20 unique computations.
    
    Parameters
    ----------
    nu : float
        Degrees of freedom
        
    Returns
    -------
    log_gamma_half_nu : float
        gammaln(ν/2)
    log_gamma_half_nu_plus_half : float
        gammaln((ν+1)/2)
    """
    if not _SCIPY_AVAILABLE:
        # Fallback to Stirling approximation (less accurate at low ν)
        def _stirling_gammaln(x: float) -> float:
            return (x - 0.5) * np.log(x) - x + 0.5 * np.log(2.0 * np.pi)
        
        log_gamma_half_nu = _stirling_gammaln(nu / 2.0)
        log_gamma_half_nu_plus_half = _stirling_gammaln((nu + 1.0) / 2.0)
    else:
        log_gamma_half_nu = float(gammaln(nu / 2.0))
        log_gamma_half_nu_plus_half = float(gammaln((nu + 1.0) / 2.0))
    
    return log_gamma_half_nu, log_gamma_half_nu_plus_half


# =============================================================================
# GAMMA PRECOMPUTATION AUDIT — Story 13.2
# =============================================================================

def validate_gamma_precomputation(
    nu_values: Optional[List[float]] = None,
) -> dict:
    """
    Validate gamma precomputation consistency and cache behavior.

    Checks that precompute_gamma_values() returns the same result
    for all filter types and that cached values match scipy.

    Parameters
    ----------
    nu_values : list of float, optional
        DoF values to test. Default: common values.

    Returns
    -------
    dict with keys:
        all_consistent : bool
        max_error : float (vs scipy reference)
        cache_info : dict (lru_cache stats)
        per_nu : list of dicts with per-value diagnostics
    """
    if nu_values is None:
        nu_values = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 20.0, 50.0]

    per_nu = []
    max_error = 0.0

    for nu in nu_values:
        lg_half, lg_half_plus = precompute_gamma_values(nu)

        # Reference from scipy
        if _SCIPY_AVAILABLE:
            ref_half = float(gammaln(nu / 2.0))
            ref_half_plus = float(gammaln((nu + 1.0) / 2.0))
            err_half = abs(lg_half - ref_half)
            err_half_plus = abs(lg_half_plus - ref_half_plus)
            max_error = max(max_error, err_half, err_half_plus)
        else:
            err_half = err_half_plus = float("nan")

        # Consistency: call again, should return same values
        lg_half2, lg_half_plus2 = precompute_gamma_values(nu)
        consistent = (lg_half == lg_half2) and (lg_half_plus == lg_half_plus2)

        per_nu.append({
            "nu": nu,
            "log_gamma_half_nu": lg_half,
            "log_gamma_half_nu_plus_half": lg_half_plus,
            "error_half": err_half,
            "error_half_plus": err_half_plus,
            "consistent": consistent,
        })

    cache_info = precompute_gamma_values.cache_info()

    return {
        "all_consistent": all(d["consistent"] for d in per_nu),
        "max_error": max_error,
        "cache_info": {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
        },
        "per_nu": per_nu,
    }


# =============================================================================
# DYNAMIC MOMENTUM CAP — Story 14.1
# =============================================================================

# Asset-class momentum cap multipliers:
#   k=3.0 standard, k=2.0 high-vol, k=4.0 slow
_MOMENTUM_CAP_K = {
    "metals_gold": 4.0,
    "metals_silver": 3.5,
    "metals_other": 4.0,
    "high_vol_equity": 2.0,
    "crypto": 2.0,
    "large_cap": 3.0,
    "index": 3.0,
    "forex": 3.0,
}

# Symbols for asset-class dispatch (mirrors phi_student_t.py)
_CRYPTO_SYMS = frozenset({
    "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "ADA-USD",
    "XRP-USD", "DOT-USD", "AVAX-USD", "LINK-USD", "MATIC-USD",
})
_HIGH_VOL_SYMS = frozenset({
    "MSTR", "IONQ", "SMCI", "RIVN", "UPST", "AFRM", "DKNG",
    "GME", "AMC", "RKLB",
})
_INDEX_SYMS = frozenset({
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI",
})
_METALS_GOLD_SYMS = frozenset({
    "GC=F", "GLD", "IAU", "SGOL", "GLDM",
})
_METALS_SILVER_SYMS = frozenset({
    "SI=F", "SLV", "SIVR",
})
_METALS_OTHER_SYMS = frozenset({
    "HG=F", "PL=F", "PA=F", "CPER", "PPLT",
})


def _classify_for_momentum(symbol: str) -> str:
    """Classify asset symbol for momentum cap dispatch."""
    if symbol is None:
        return "large_cap"
    sym = symbol.strip().upper()
    if sym in _CRYPTO_SYMS:
        return "crypto"
    if sym in _HIGH_VOL_SYMS:
        return "high_vol_equity"
    if sym in _METALS_GOLD_SYMS:
        return "metals_gold"
    if sym in _METALS_SILVER_SYMS:
        return "metals_silver"
    if sym in _METALS_OTHER_SYMS:
        return "metals_other"
    if sym in _INDEX_SYMS:
        return "index"
    if sym.endswith("=X"):
        return "forex"
    return "large_cap"


def compute_dynamic_momentum_cap(
    q: float,
    asset_symbol: Optional[str] = None,
    k_override: Optional[float] = None,
) -> dict:
    """
    Compute the momentum cap u_max = k * sqrt(q) for a given asset.

    Parameters
    ----------
    q : float
        Process noise variance.
    asset_symbol : str, optional
        Asset symbol for class-based k selection.
    k_override : float, optional
        Override the asset-class k value.

    Returns
    -------
    dict with keys:
        k : float — the cap multiplier
        u_max : float — the absolute cap value
        asset_class : str — detected class
        sqrt_q : float
    """
    asset_class = _classify_for_momentum(asset_symbol)
    k = k_override if k_override is not None else _MOMENTUM_CAP_K.get(asset_class, 3.0)
    sqrt_q = np.sqrt(max(q, 0.0))
    u_max = k * sqrt_q

    return {
        "k": k,
        "u_max": u_max,
        "asset_class": asset_class,
        "sqrt_q": sqrt_q,
    }


def monitor_cap_binding(
    momentum_adjustment: np.ndarray,
    u_max: float,
) -> dict:
    """
    Monitor how often the momentum cap binds.

    Optimal binding rate: 5-15% of timesteps.

    Returns
    -------
    dict with keys:
        binding_rate : float — fraction of timesteps at cap
        n_bound : int — number of capped timesteps
        n_total : int — total timesteps
        recommendation : str — 'ok', 'increase_k', or 'decrease_k'
    """
    n = len(momentum_adjustment)
    if n == 0:
        return {"binding_rate": 0.0, "n_bound": 0, "n_total": 0,
                "recommendation": "ok"}

    abs_mom = np.abs(momentum_adjustment)
    n_bound = int(np.sum(abs_mom >= u_max * 0.999))  # near-cap tolerance
    binding_rate = n_bound / n

    if binding_rate > 0.20:
        recommendation = "increase_k"
    elif binding_rate < 0.03:
        recommendation = "decrease_k"
    else:
        recommendation = "ok"

    return {
        "binding_rate": binding_rate,
        "n_bound": n_bound,
        "n_total": n,
        "recommendation": recommendation,
    }


def apply_momentum_cap(
    momentum_raw: np.ndarray,
    q: float,
    asset_symbol: Optional[str] = None,
    k_override: Optional[float] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Apply dynamic momentum cap to raw momentum signal.

    Returns capped momentum array and diagnostics.
    """
    cap_info = compute_dynamic_momentum_cap(q, asset_symbol, k_override)
    u_max = cap_info["u_max"]

    if u_max <= 0.0:
        capped = np.zeros_like(momentum_raw)
    else:
        capped = np.clip(momentum_raw, -u_max, u_max)

    binding = monitor_cap_binding(momentum_raw, u_max)
    cap_info.update(binding)

    return capped, cap_info


# =============================================================================
# PHI SHRINKAGE FOR MOMENTUM — Story 14.2
# =============================================================================

def compute_phi_shrinkage_for_momentum(
    phi_mle: float,
    momentum_enabled: bool = False,
    center_momentum: float = 1.0,
    tau_momentum: float = 0.10,
    center_no_momentum: float = 0.0,
    tau_no_momentum: float = 0.20,
) -> dict:
    """
    Compute phi shrinkage prior parameters based on momentum status.

    When momentum is active:
        - Center = 1.0 (unit root, persistence captures level)
        - Tau = 0.10 (tight, prevents collinearity with kappa)
    When momentum is inactive:
        - Center = 0.0 (mean reversion)
        - Tau = 0.20 (looser)

    Returns
    -------
    dict with keys:
        center : float — prior center
        tau : float — prior standard deviation
        log_prior : float — Gaussian log prior
        phi_shrunk : float — MAP-like shrinkage of phi toward center
        momentum_enabled : bool
        collinearity_safe : bool — |phi_mle - center| < 3*tau
    """
    if momentum_enabled:
        center = center_momentum
        tau = tau_momentum
    else:
        center = center_no_momentum
        tau = tau_no_momentum

    tau_safe = max(tau, 1e-6)
    deviation = phi_mle - center
    log_prior = -0.5 * (deviation ** 2) / (tau_safe ** 2)

    # Shrinkage: blend MLE toward prior center
    # posterior_mean ~ (phi_mle * precision_data + center * precision_prior) / (precision_data + precision_prior)
    # Simplified: phi_shrunk = center + tau^2 / (tau^2 + sigma_phi^2) * (phi_mle - center)
    # Approximate sigma_phi ~ 0.05 (typical estimation uncertainty)
    sigma_phi_approx = 0.05
    shrinkage_factor = tau_safe ** 2 / (tau_safe ** 2 + sigma_phi_approx ** 2)
    phi_shrunk = center + shrinkage_factor * (phi_mle - center)

    # Bounds: no explosive or oscillatory
    phi_shrunk = max(-0.5, min(1.05, phi_shrunk))

    collinearity_safe = abs(deviation) < 3.0 * tau_safe

    return {
        "center": center,
        "tau": tau_safe,
        "log_prior": log_prior,
        "phi_shrunk": phi_shrunk,
        "momentum_enabled": momentum_enabled,
        "collinearity_safe": collinearity_safe,
        "phi_mle": phi_mle,
        "deviation": deviation,
    }


# =============================================================================
# STATE-SPACE EQUILIBRIUM ESTIMATION — Story 14.3
# =============================================================================

def rts_smoother(
    mu_filtered: np.ndarray,
    P_filtered: np.ndarray,
    phi: float,
    q: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rauch-Tung-Striebel fixed-interval smoother.

    Backward pass on Kalman filtered states to produce the smoothed
    equilibrium estimate. The smoother is lag-free — it uses future
    observations to correct the filtered estimate.

    mu_pred[t] = phi * mu_filt[t-1]
    P_pred[t] = phi^2 * P_filt[t-1] + q
    G[t] = phi * P_filt[t-1] / P_pred[t]    (smoother gain)
    mu_smooth[t] = mu_filt[t] + G[t+1] * (mu_smooth[t+1] - mu_pred[t+1])
    P_smooth[t] = P_filt[t] + G[t+1]^2 * (P_smooth[t+1] - P_pred[t+1])

    Parameters
    ----------
    mu_filtered : np.ndarray — filtered state means
    P_filtered : np.ndarray — filtered state variances
    phi : float — AR(1) coefficient
    q : float — process noise variance

    Returns
    -------
    mu_smooth : np.ndarray — smoothed equilibrium
    P_smooth : np.ndarray — smoothed variances
    """
    T = len(mu_filtered)
    mu_smooth = np.empty(T, dtype=np.float64)
    P_smooth = np.empty(T, dtype=np.float64)

    mu_smooth[T - 1] = mu_filtered[T - 1]
    P_smooth[T - 1] = P_filtered[T - 1]

    for t in range(T - 2, -1, -1):
        P_pred = phi ** 2 * P_filtered[t] + q
        if P_pred < 1e-30:
            P_pred = 1e-30
        G = phi * P_filtered[t] / P_pred
        mu_pred = phi * mu_filtered[t]
        mu_smooth[t] = mu_filtered[t] + G * (mu_smooth[t + 1] - mu_pred)
        P_smooth[t] = P_filtered[t] + G ** 2 * (P_smooth[t + 1] - P_pred)

    return mu_smooth, P_smooth


def compute_equilibrium_and_mr_signal(
    mu_filtered: np.ndarray,
    P_filtered: np.ndarray,
    phi: float,
    q: float,
    kappa: float = 0.1,
) -> dict:
    """
    Compute state-space equilibrium and mean-reversion signal.

    MR_t = kappa * (mu_t - mu*_t)
    where mu*_t is the smoothed equilibrium.

    Parameters
    ----------
    kappa : float — mean-reversion speed

    Returns
    -------
    dict with keys:
        equilibrium : np.ndarray — smoothed equilibrium mu*_t
        P_smooth : np.ndarray — smoother variance
        mr_signal : np.ndarray — mean-reversion signal
        mr_mean : float — mean of MR signal (should be ~0)
        mr_std : float — std of MR signal
    """
    mu_smooth, P_smooth = rts_smoother(mu_filtered, P_filtered, phi, q)
    mr_signal = kappa * (mu_filtered - mu_smooth)

    return {
        "equilibrium": mu_smooth,
        "P_smooth": P_smooth,
        "mr_signal": mr_signal,
        "mr_mean": float(np.mean(mr_signal)),
        "mr_std": float(np.std(mr_signal)),
    }


# =============================================================================
# PIT-DRIVEN VARIANCE INFLATION WITH DEAD-ZONE — Story 15.2
# =============================================================================

# Asset-class dead-zones: (low, high) for PIT MAD
_PIT_DEAD_ZONES = {
    "metals_gold": (0.25, 0.60),
    "metals_silver": (0.25, 0.60),
    "metals_other": (0.25, 0.60),
    "crypto": (0.28, 0.52),
    "high_vol_equity": (0.28, 0.52),
    "large_cap": (0.30, 0.55),
    "index": (0.30, 0.55),
    "forex": (0.30, 0.55),
}


def _compute_pit_mad(pit_values: np.ndarray) -> float:
    """
    Mean Absolute Deviation of PIT values from 0.5.
    Uniform[0,1] has MAD ~ 0.25; peaked -> MAD < 0.25; U-shaped -> MAD > 0.25.
    We scale to [0, 1] convention where ~0.42 is well-calibrated.

    Actually, the simple version: median of |pit - uniform_quantile|.
    Here we use the simpler metric: MAD = mean(|pit - 0.5|).
    """
    return float(np.mean(np.abs(pit_values - 0.5)))


def pit_variance_inflation_iterative(
    pit_values: np.ndarray,
    beta_init: float = 1.0,
    max_iter: int = 10,
    step_size: float = 0.05,
    asset_class: Optional[str] = None,
    dead_zone: Optional[Tuple[float, float]] = None,
) -> dict:
    """
    Iterative PIT-driven variance inflation with dead-zone.

    Adjusts beta (variance multiplier) based on PIT MAD:
    - If MAD < dead_zone[0] (peaked/overconfident): decrease beta by step_size
    - If MAD > dead_zone[1] (U-shaped/underconfident): increase beta by step_size
    - If MAD in dead-zone: no correction (well-calibrated)

    Parameters
    ----------
    pit_values : np.ndarray — PIT values in [0, 1]
    beta_init : float — initial variance inflation factor
    max_iter : int — maximum correction iterations
    step_size : float — beta adjustment per iteration (default 5%)
    asset_class : str, optional — for asset-class dead-zones
    dead_zone : tuple, optional — override (low, high) bounds

    Returns
    -------
    dict with keys:
        beta_final : float — converged variance inflation
        n_iter : int — iterations used
        converged : bool — beta stabilized
        trajectory : list of (iter, beta, pit_mad)
        pit_mad_initial : float
        pit_mad_final : float
        correction_applied : bool
    """
    if dead_zone is None:
        ac = asset_class if asset_class else "large_cap"
        dead_zone = _PIT_DEAD_ZONES.get(ac, (0.30, 0.55))

    pit_mad = _compute_pit_mad(pit_values)
    pit_mad_initial = pit_mad

    beta = beta_init
    trajectory = [(0, beta, pit_mad)]
    converged = False
    correction_applied = False

    for i in range(1, max_iter + 1):
        if dead_zone[0] <= pit_mad <= dead_zone[1]:
            converged = True
            break

        if pit_mad < dead_zone[0]:
            # Too peaked -> widen variance -> decrease beta
            beta *= (1.0 - step_size)
            correction_applied = True
        else:
            # Too spread -> narrow variance -> increase beta
            beta *= (1.0 + step_size)
            correction_applied = True

        beta = max(0.5, min(5.0, beta))

        # Recompute PIT with new beta (approximate: scale z by 1/sqrt(beta))
        # pit_new = Phi(z / sqrt(beta)) where z = Phi_inv(pit_old) * sqrt(beta_old)
        # Simplified: just track beta, actual PIT recomputation happens upstream
        trajectory.append((i, beta, pit_mad))

        if len(trajectory) >= 2 and abs(trajectory[-1][1] - trajectory[-2][1]) < 0.01:
            converged = True
            break

    return {
        "beta_final": beta,
        "n_iter": len(trajectory) - 1,
        "converged": converged,
        "trajectory": trajectory,
        "pit_mad_initial": pit_mad_initial,
        "pit_mad_final": pit_mad,
        "correction_applied": correction_applied,
    }


# =============================================================================
# PIT ENTROPY DIAGNOSTIC — Story 15.3
# =============================================================================

def pit_entropy(pit_values: np.ndarray, n_bins: int = 20) -> dict:
    """
    Compute PIT entropy as a calibration diagnostic.

    H(u) = -sum(p_b * log(p_b)) where p_b are bin probabilities.
    Well-calibrated: H ~ log(B). Entropy ratio in [0.95, 1.05].

    Parameters
    ----------
    pit_values : np.ndarray — PIT values in [0, 1]
    n_bins : int — number of histogram bins (default 20)

    Returns
    -------
    dict with keys:
        entropy : float — Shannon entropy H(u)
        entropy_uniform : float — H_uniform = log(B)
        entropy_ratio : float — H(u) / H_uniform
        well_calibrated : bool — ratio in [0.95, 1.05]
        bin_counts : np.ndarray — counts per bin
        bin_probs : np.ndarray — probabilities per bin
    """
    pit_clean = pit_values[np.isfinite(pit_values)]
    pit_clean = np.clip(pit_clean, 0.0, 1.0)

    counts, _ = np.histogram(pit_clean, bins=n_bins, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return {
            "entropy": 0.0,
            "entropy_uniform": np.log(n_bins),
            "entropy_ratio": 0.0,
            "well_calibrated": False,
            "bin_counts": counts,
            "bin_probs": np.zeros(n_bins),
        }

    probs = counts / total
    # Avoid log(0) by filtering zero bins
    nonzero = probs > 0
    entropy = -np.sum(probs[nonzero] * np.log(probs[nonzero]))
    entropy_uniform = np.log(n_bins)
    ratio = entropy / entropy_uniform if entropy_uniform > 0 else 0.0

    return {
        "entropy": float(entropy),
        "entropy_uniform": float(entropy_uniform),
        "entropy_ratio": float(ratio),
        "well_calibrated": 0.95 <= ratio <= 1.05,
        "bin_counts": counts,
        "bin_probs": probs,
    }


# =============================================================================
# CRPS-OPTIMAL SIGMA SHRINKAGE (Story 16.2)
# =============================================================================

def crps_optimal_sigma_shrinkage(nu):
    """
    Compute CRPS-optimal scale shrinkage factor alpha* for Student-t.

    Formula: alpha* = sqrt((3*nu + 4) / (3*nu + 8))

    For nu=4:  alpha* = 0.8944
    For nu=8:  alpha* = 0.9354
    For nu=20: alpha* = 0.9701

    Parameters
    ----------
    nu : float
        Degrees of freedom (must be > 2 for finite variance).

    Returns
    -------
    dict with:
        alpha_star : float  – optimal shrinkage factor
        nu : float          – input nu
        variance_ratio : float – nu / (nu - 2), the excess variance
        effective_df_adj : float – (3*nu + 4) / (3*nu + 8)
    """
    if nu <= 2.0:
        raise ValueError(f"nu must be > 2 for finite variance, got {nu}")
    variance_ratio = nu / (nu - 2.0)
    alpha_sq = (3.0 * nu + 4.0) / (3.0 * nu + 8.0)
    alpha_star = np.sqrt(alpha_sq)
    return {
        "alpha_star": float(alpha_star),
        "nu": float(nu),
        "variance_ratio": float(variance_ratio),
        "effective_df_adj": float(alpha_sq),
    }


def apply_crps_sigma_shrinkage(sigma_arr, nu):
    """
    Apply CRPS-optimal shrinkage to an array of scale parameters.

    Parameters
    ----------
    sigma_arr : np.ndarray
        Array of scale (sigma) values.
    nu : float
        Degrees of freedom.

    Returns
    -------
    np.ndarray
        Shrunk sigma values: sigma * alpha_star.
    """
    result = crps_optimal_sigma_shrinkage(nu)
    alpha = result["alpha_star"]
    return sigma_arr * alpha


def verify_crps_shrinkage_vs_grid(z, sigma, nu, grid_points=200):
    """
    Verify that alpha* matches grid-search optimum for CRPS.

    Parameters
    ----------
    z : np.ndarray
        Standardized residuals.
    sigma : np.ndarray
        Scale parameters.
    nu : float
        Degrees of freedom.
    grid_points : int
        Number of grid points for alpha search.

    Returns
    -------
    dict with:
        alpha_star_formula : float
        alpha_star_grid : float
        crps_at_formula : float
        crps_at_grid : float
        relative_gap : float  – should be < 0.01
    """
    from models.numba_kernels import crps_student_t_kernel

    result = crps_optimal_sigma_shrinkage(nu)
    alpha_formula = result["alpha_star"]

    alphas = np.linspace(0.5, 1.5, grid_points)
    crps_vals = np.empty(grid_points)
    for i, a in enumerate(alphas):
        # When shrinking sigma -> alpha*sigma, re-standardize: z_new = z / alpha
        crps_vals[i] = crps_student_t_kernel(z / a, sigma * a, nu)

    best_idx = np.argmin(crps_vals)
    alpha_grid = alphas[best_idx]
    crps_grid = crps_vals[best_idx]
    crps_formula = crps_student_t_kernel(z / alpha_formula, sigma * alpha_formula, nu)

    gap = abs(crps_formula - crps_grid) / max(abs(crps_grid), 1e-10)

    return {
        "alpha_star_formula": float(alpha_formula),
        "alpha_star_grid": float(alpha_grid),
        "crps_at_formula": float(crps_formula),
        "crps_at_grid": float(crps_grid),
        "relative_gap": float(gap),
    }


# =============================================================================
# CRPS DECOMPOSITION – Hersbach (2000) (Story 16.3)
# =============================================================================

def crps_decomposition(pit_values, n_bins=20):
    """
    Decompose CRPS into Reliability, Resolution, Uncertainty using
    the Hersbach (2000) binned-PIT method.

    CRPS = Reliability - Resolution + Uncertainty

    - Reliability near 0 = well calibrated (target < 0.002)
    - Resolution > 0 = model is informative/sharp (target > 0.010)
    - Uncertainty = irreducible (depends on data variance)

    Parameters
    ----------
    pit_values : np.ndarray
        PIT values in [0, 1].
    n_bins : int
        Number of bins for the decomposition.

    Returns
    -------
    dict with:
        reliability : float
        resolution : float
        uncertainty : float
        crps_reconstructed : float  (= reliability - resolution + uncertainty)
        well_calibrated : bool  (reliability < 0.002)
        informative : bool  (resolution > 0.010)
        bin_counts : np.ndarray
        bin_obs_freq : np.ndarray  (observed frequency per bin)
    """
    pit = np.asarray(pit_values, dtype=np.float64)
    pit = pit[np.isfinite(pit)]
    n = len(pit)

    if n < 10:
        return {
            "reliability": np.nan,
            "resolution": np.nan,
            "uncertainty": np.nan,
            "crps_reconstructed": np.nan,
            "well_calibrated": False,
            "informative": False,
            "bin_counts": np.array([]),
            "bin_obs_freq": np.array([]),
        }

    # Bin edges: [0, 1/J, 2/J, ..., 1]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_counts = np.zeros(n_bins, dtype=np.float64)
    bin_obs_sum = np.zeros(n_bins, dtype=np.float64)

    for i in range(n):
        p = pit[i]
        # Determine bin index
        idx = int(p * n_bins)
        if idx >= n_bins:
            idx = n_bins - 1
        if idx < 0:
            idx = 0
        bin_counts[idx] += 1.0
        # Observed frequency: o_j = fraction of obs where pit <= (j+1)/J
        # Actually in Hersbach, we need the cumulative approach.
        # For each bin j, o_j = (# observations with PIT <= (j+1)/J) / n
        # We'll compute this after binning.

    # Hersbach decomposition:
    # g_j = bin_counts[j] / n  (fraction of PITs in bin j)
    # o_j = cumulative fraction: P(PIT <= (j+1)/J) for j=0,..,J-1
    # For uniform PIT: o_j = (j+1)/J
    #
    # Reliability = sum_j g_j * (o_bar_j - p_j)^2
    # Resolution = sum_j g_j * (o_bar_j - o_bar)^2
    # Uncertainty = o_bar * (1 - o_bar)
    # where o_bar_j is the average observation in bin j
    # and p_j is the forecast probability for bin j
    #
    # Using the simplified Hersbach (2000) approach with rank histogram:
    # alpha_j = (cumulative count up to bin j) / n
    # Reliability = (1/J) * sum_{j=0}^{J} (alpha_j - j/J)^2
    # Uncertainty = variance of the observation indicator

    # Compute cumulative fractions (alpha_j for j=0,...,J)
    J = n_bins
    alpha = np.zeros(J + 1)
    alpha[0] = 0.0
    for j in range(J):
        alpha[j + 1] = alpha[j] + bin_counts[j] / n

    # Expected cumulative under uniformity: j/J
    expected = np.linspace(0.0, 1.0, J + 1)

    # Reliability: (1/J) * sum (alpha_j - j/J)^2
    reliability = np.sum((alpha - expected) ** 2) / J

    # Resolution: (1/J) * sum (alpha_j - alpha_bar)^2 - something
    # Under Hersbach:
    # CRPS = sum_{j=0}^{J-1} [(1/J)(alpha_{j+1} + alpha_j)/2 - (2j+1)/(2J)]^2 * (1/J)?
    # Actually, the standard Hersbach decomposition is:
    #
    # Define: for each bin j, let o_j = alpha_j (cumulative proportion)
    #         p_j = j/J (forecast probability)
    # Then:
    #   Reliability = (1/J) * sum_{j=0}^{J} (alpha_j - j/J)^2
    #   Uncertainty is the mean Brier score of a climatological forecast
    #
    # Simpler equivalent form:
    # Uncertainty = mean(pit) * (1 - mean(pit))
    # For well-calibrated PIT (uniform on [0,1]): uncertainty = 0.25

    mean_pit = np.mean(pit)
    uncertainty = mean_pit * (1.0 - mean_pit)

    # Resolution: derived from CRPS = Rel - Res + Unc
    # Res = Rel + Unc - CRPS
    # But we can also compute it directly:
    # Resolution = (1/J) * sum (alpha_j - mean_alpha)^2
    # where mean_alpha = 0.5 (for uniform PIT, the mean cumulative is 0.5)
    mean_alpha = np.mean(alpha)
    resolution = np.sum((alpha - mean_alpha) ** 2) / J

    crps_reconstructed = reliability - resolution + uncertainty

    obs_freq = bin_counts / n

    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "crps_reconstructed": float(crps_reconstructed),
        "well_calibrated": reliability < 0.002,
        "informative": resolution > 0.010,
        "bin_counts": bin_counts,
        "bin_obs_freq": obs_freq,
    }


# =============================================================================
# BASE MODEL WRAPPERS
# =============================================================================

def run_gaussian_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    P0: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run Gaussian Kalman filter (random walk drift).
    
    Parameters
    ----------
    returns : np.ndarray
        Log returns
    vol : np.ndarray
        EWMA volatility estimates
    q : float
        Process noise variance
    c : float
        Observation noise scale
    P0 : float
        Initial state covariance
        
    Returns
    -------
    mu_filtered : np.ndarray
    P_filtered : np.ndarray
    log_likelihood : float
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol = prepare_arrays(returns, vol)
    return gaussian_filter_kernel(returns, vol, float(q), float(c), float(P0))


def run_phi_gaussian_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    P0: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run φ-Gaussian Kalman filter (AR(1) drift).
    
    Parameters
    ----------
    phi : float
        AR(1) persistence coefficient
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol = prepare_arrays(returns, vol)
    return phi_gaussian_filter_kernel(
        returns, vol, float(q), float(c), float(phi), float(P0)
    )


def run_phi_student_t_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    P0: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run φ-Student-t Kalman filter (AR(1) drift, heavy-tailed observations).
    
    This is the ONLY Student-t variant. There is no bare Student-t.
    
    Parameters
    ----------
    phi : float
        AR(1) persistence coefficient
    nu : float
        Degrees of freedom (typically from grid: 4, 6, 8, 12, 20)
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol = prepare_arrays(returns, vol)
    log_g1, log_g2 = precompute_gamma_values(nu)
    
    return phi_student_t_filter_kernel(
        returns, vol,
        float(q), float(c), float(phi), float(nu),
        log_g1, log_g2,
        float(P0)
    )


# =============================================================================
# MOMENTUM-AUGMENTED WRAPPERS
# =============================================================================

def run_momentum_phi_gaussian_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    momentum_adjustment: np.ndarray,
    P0: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run φ-Gaussian filter with momentum augmentation.
    
    Used by: CRSP (φ-Gaussian+Mom+EVTM+CST20%)
             CELH (φ-Gaussian+Mom+EVTH+CST17%)
             DPRO (φ-Gaussian+Mom+EVTH+CST19%)
    
    Parameters
    ----------
    momentum_adjustment : np.ndarray
        Per-timestep momentum signal to add to drift prediction
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol, momentum_adjustment = prepare_arrays(
        returns, vol, momentum_adjustment
    )
    return momentum_phi_gaussian_filter_kernel(
        returns, vol,
        float(q), float(c), float(phi),
        momentum_adjustment,
        float(P0)
    )


def run_momentum_phi_student_t_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    momentum_adjustment: np.ndarray,
    P0: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run φ-Student-t filter with momentum augmentation.
    
    Used by: GLDW (φ-Student-t+Mom+EVTH+CST17%)
             MAGD (φ-Student-t+Mom+Hλ←+EVTM+CST17%)
             BKSY (φ-Student-t+Mom+Hλ→+EVTH+CST17%)
             ASTS (φ-Student-t+Mom+Hλ→+EVTH+CST14%)
    
    Notes on augmentation layers:
    - Hλ← : Hierarchical λ with backward-looking momentum
    - Hλ→ : Hierarchical λ with forward-looking momentum
    - EVTH/EVTM: EVT tail handling affects vol estimation UPSTREAM
    - CST##%: CVaR constraint affects position sizing DOWNSTREAM
    
    None of these alter the Kalman filter mathematics.
    
    Parameters
    ----------
    momentum_adjustment : np.ndarray
        Per-timestep momentum signal (may include Hλ scaling)
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol, momentum_adjustment = prepare_arrays(
        returns, vol, momentum_adjustment
    )
    log_g1, log_g2 = precompute_gamma_values(nu)
    
    return momentum_phi_student_t_filter_kernel(
        returns, vol,
        float(q), float(c), float(phi), float(nu),
        log_g1, log_g2,
        momentum_adjustment,
        float(P0)
    )


# =============================================================================
# BATCH PROCESSING FOR BMA (multiple ν values)
# =============================================================================

def run_phi_student_t_filter_batch(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu_grid: List[float],
    P0: float = 1e-4
) -> Dict[float, Tuple[np.ndarray, np.ndarray, float]]:
    """
    Run φ-Student-t filter for multiple ν values (discrete grid BMA).
    
    Returns dict mapping ν -> (mu_filtered, P_filtered, log_likelihood)
    
    Used for Bayesian Model Averaging over ν ∈ {4, 8, 20}
    
    This batch function amortizes the cost of:
    - Array preparation (done once)
    - Gamma precomputation (done per ν, but efficiently)
    
    Parameters
    ----------
    nu_grid : List[float]
        List of ν values to evaluate (e.g., [4, 8, 20])
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    # Prepare arrays once
    returns, vol = prepare_arrays(returns, vol)
    results = {}
    
    for nu in nu_grid:
        log_g1, log_g2 = precompute_gamma_values(nu)
        mu, P, ll = phi_student_t_filter_kernel(
            returns, vol,
            float(q), float(c), float(phi), float(nu),
            log_g1, log_g2,
            float(P0)
        )
        results[nu] = (mu, P, ll)
    
    return results


def run_momentum_phi_student_t_filter_batch(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu_grid: List[float],
    momentum_adjustment: np.ndarray,
    P0: float = 1e-4
) -> Dict[float, Tuple[np.ndarray, np.ndarray, float]]:
    """
    Run momentum-augmented φ-Student-t filter for multiple ν values.
    
    Parameters
    ----------
    nu_grid : List[float]
        List of ν values to evaluate
    momentum_adjustment : np.ndarray
        Per-timestep momentum signal
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol, momentum_adjustment = prepare_arrays(
        returns, vol, momentum_adjustment
    )
    results = {}
    
    for nu in nu_grid:
        log_g1, log_g2 = precompute_gamma_values(nu)
        mu, P, ll = momentum_phi_student_t_filter_kernel(
            returns, vol,
            float(q), float(c), float(phi), float(nu),
            log_g1, log_g2,
            momentum_adjustment,
            float(P0)
        )
        results[nu] = (mu, P, ll)
    
    return results


# =============================================================================
# MS-q AND FUSED LFO-CV WRAPPERS (February 2026)
# =============================================================================
# These wrappers provide:
# 1. Numba-accelerated MS-q filtering (10× speedup)
# 2. Fused LFO-CV computation (40% overall speedup by avoiding second pass)
# =============================================================================

def run_student_t_filter_with_lfo_cv(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    lfo_start_frac: float = 0.5,
    P0: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Run φ-Student-t filter with FUSED LFO-CV computation.
    
    This is 40% faster than running filter + separate LFO-CV computation
    because it computes the predictive log-density during the single pass.
    
    Parameters
    ----------
    lfo_start_frac : float
        Fraction of data before starting LFO-CV accumulation (default 0.5)
        
    Returns
    -------
    mu_filtered : np.ndarray
    P_filtered : np.ndarray
    log_likelihood : float
    lfo_cv_score : float
        Mean predictive log-density from t=lfo_start to T
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol = prepare_arrays(returns, vol)
    log_g1, log_g2 = precompute_gamma_values(nu)
    
    return student_t_filter_with_lfo_cv_kernel(
        returns, vol,
        float(q), float(c), float(phi), float(nu),
        log_g1, log_g2,
        float(P0), float(lfo_start_frac)
    )


def run_gaussian_filter_with_lfo_cv(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float = 1.0,
    lfo_start_frac: float = 0.5,
    P0: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Run Gaussian filter with FUSED LFO-CV computation.
    
    Returns
    -------
    mu_filtered : np.ndarray
    P_filtered : np.ndarray
    log_likelihood : float
    lfo_cv_score : float
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol = prepare_arrays(returns, vol)
    
    return gaussian_filter_with_lfo_cv_kernel(
        returns, vol,
        float(q), float(c), float(phi),
        float(P0), float(lfo_start_frac)
    )


def run_ms_q_student_t_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    c: float,
    phi: float,
    nu: float,
    q_calm: float,
    q_stress: float,
    sensitivity: float = 2.0,
    threshold: float = 1.3,
    lfo_start_frac: float = 0.5,
    P0: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Run Numba-accelerated MS-q Student-t filter with fused LFO-CV.
    
    This provides ~10× speedup over pure Python implementation.
    
    Parameters
    ----------
    q_calm : float
        Process noise in calm regime
    q_stress : float
        Process noise in stress regime (typically 100× q_calm)
    sensitivity : float
        Sigmoid sensitivity to vol_relative (default 2.0)
    threshold : float
        Vol_relative threshold for regime transition (default 1.3)
    lfo_start_frac : float
        Fraction of data before starting LFO-CV accumulation
        
    Returns
    -------
    mu_filtered : np.ndarray
    P_filtered : np.ndarray
    q_t : np.ndarray
        Time-varying process noise
    p_stress : np.ndarray
        Probability of stress regime at each timestep
    log_likelihood : float
    lfo_cv_score : float
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol = prepare_arrays(returns, vol)
    log_g1, log_g2 = precompute_gamma_values(nu)
    
    return ms_q_student_t_filter_kernel(
        returns, vol,
        float(c), float(phi), float(nu),
        float(q_calm), float(q_stress),
        float(sensitivity), float(threshold),
        log_g1, log_g2,
        float(P0), float(lfo_start_frac)
    )


def run_student_t_filter_with_lfo_cv_batch(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu_grid: List[float],
    lfo_start_frac: float = 0.5,
    P0: float = 1e-4,
) -> Dict[float, Tuple[np.ndarray, np.ndarray, float, float]]:
    """
    Run fused filter+LFO-CV for multiple ν values (BMA optimization).
    
    Returns dict mapping ν -> (mu, P, log_likelihood, lfo_cv_score)
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    
    returns, vol = prepare_arrays(returns, vol)
    results = {}
    
    for nu in nu_grid:
        log_g1, log_g2 = precompute_gamma_values(nu)
        mu, P, ll, lfo = student_t_filter_with_lfo_cv_kernel(
            returns, vol,
            float(q), float(c), float(phi), float(nu),
            log_g1, log_g2,
            float(P0), float(lfo_start_frac)
        )
        results[nu] = (mu, P, ll, lfo)
    
    return results


# =============================================================================
# UNIFIED φ-STUDENT-T WRAPPER (VoV + MS-q + Smooth Asymmetric ν + Momentum)
# =============================================================================

def run_unified_phi_student_t_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    c: float,
    phi: float,
    nu_base: float,
    # MS-q arrays (precomputed)
    q_t: np.ndarray,
    p_stress: np.ndarray,
    # VoV arrays (precomputed)
    vov_rolling: np.ndarray,
    gamma_vov: float,
    vov_damping: float,
    # Asymmetry parameters
    alpha_asym: float,
    k_asym: float,
    # Momentum
    momentum: np.ndarray,
    P0: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run Numba-accelerated unified φ-Student-t filter.
    
    This is the wrapper for the elite unified model combining:
      1. Smooth Asymmetric ν (Lanczos gammaln for dynamic ν)
      2. Probabilistic MS-q regime switching
      3. VoV scaling with redundancy damping
      4. Momentum drift input
      5. Robust Student-t weighting
    
    All time-varying arrays must be precomputed before calling:
      - q_t: from compute_ms_process_noise_smooth()
      - p_stress: from compute_ms_process_noise_smooth()
      - vov_rolling: rolling std of log(vol)
      - momentum: exogenous signal or zeros
    
    Parameters
    ----------
    returns : np.ndarray
        Log returns
    vol : np.ndarray
        EWMA volatility
    c : float
        Observation noise scale
    phi : float
        AR(1) persistence
    nu_base : float
        Base degrees of freedom
    q_t : np.ndarray
        Time-varying process noise
    p_stress : np.ndarray
        Stress probability per timestep
    vov_rolling : np.ndarray
        Rolling vol-of-vol
    gamma_vov : float
        VoV sensitivity
    vov_damping : float
        Redundancy damping factor
    alpha_asym : float
        Asymmetry parameter (negative = heavier left tail)
    k_asym : float
        Asymmetry transition sharpness
    momentum : np.ndarray
        Momentum signal per timestep
    P0 : float
        Initial state covariance
        
    Returns
    -------
    mu_filtered : np.ndarray
        Posterior state mean
    P_filtered : np.ndarray
        Posterior state variance
    mu_pred : np.ndarray
        Prior predictive mean (for PIT)
    S_pred : np.ndarray
        Prior predictive variance (for PIT)
    log_likelihood : float
        Total log-likelihood
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available for unified filter")
    
    # Prepare all arrays as contiguous float64 (ravel avoids copy when 1D)
    returns, vol, q_t, p_stress, vov_rolling, momentum = prepare_arrays(
        returns, vol, q_t, p_stress, vov_rolling, momentum
    )
    
    return unified_phi_student_t_filter_kernel(
        returns, vol,
        float(c), float(phi), float(nu_base),
        q_t, p_stress,
        vov_rolling, float(gamma_vov), float(vov_damping),
        float(alpha_asym), float(k_asym),
        momentum, float(P0)
    )


def is_unified_filter_available() -> bool:
    """Check if Numba unified filter kernel is available."""
    return _NUMBA_AVAILABLE


def run_unified_phi_student_t_filter_extended(
    returns: np.ndarray,
    vol: np.ndarray,
    c: float,
    phi: float,
    nu_base: float,
    q_t: np.ndarray,
    p_stress: np.ndarray,
    vov_rolling: np.ndarray,
    gamma_vov: float,
    vov_damping: float,
    alpha_asym: float,
    k_asym: float,
    momentum: np.ndarray,
    P0: float,
    # Extended parameters
    risk_prem: float = 0.0,
    mu_drift: float = 0.0,
    skew_kappa: float = 0.0,
    skew_rho: float = 0.0,
    jump_var: float = 0.0,
    jump_intensity: float = 0.0,
    jump_sensitivity: float = 0.0,
    jump_mean: float = 0.0,
    ewm_lambda: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run extended unified phi-Student-t filter kernel.

    Handles all features: risk premium, mu drift, GAS skew,
    Merton jump-diffusion, and causal EWM correction.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available for extended unified filter")

    returns, vol, q_t, p_stress, vov_rolling, momentum = prepare_arrays(
        returns, vol, q_t, p_stress, vov_rolling, momentum
    )

    return unified_phi_student_t_filter_extended_kernel(
        returns, vol,
        float(c), float(phi), float(nu_base),
        q_t, p_stress,
        vov_rolling, float(gamma_vov), float(vov_damping),
        float(alpha_asym), float(k_asym),
        momentum, float(P0),
        float(risk_prem), float(mu_drift),
        float(skew_kappa), float(skew_rho),
        float(jump_var), float(jump_intensity),
        float(jump_sensitivity), float(jump_mean),
        float(ewm_lambda),
    )


# =============================================================================
# CV TEST-FOLD FORWARD-PASS WRAPPERS
# =============================================================================

def run_gaussian_cv_test_fold(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    q: float,
    c: float,
    mu_init: float,
    P_init: float,
    test_start: int,
    test_end: int,
    std_buf: np.ndarray,
    std_offset: int,
    std_max: int,
) -> Tuple[float, int, int]:
    """Run Numba-accelerated Gaussian CV test-fold forward pass."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return gaussian_cv_test_fold_kernel(
        returns, vol_sq,
        float(q), float(c),
        float(mu_init), float(P_init),
        int(test_start), int(test_end),
        std_buf, int(std_offset), int(std_max),
    )


def run_phi_gaussian_cv_test_fold(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    q: float,
    c: float,
    phi: float,
    mu_init: float,
    P_init: float,
    test_start: int,
    test_end: int,
    std_buf: np.ndarray,
    std_offset: int,
    std_max: int,
) -> Tuple[float, int, int]:
    """Run Numba-accelerated φ-Gaussian CV test-fold forward pass."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return phi_gaussian_cv_test_fold_kernel(
        returns, vol_sq,
        float(q), float(c), float(phi),
        float(mu_init), float(P_init),
        int(test_start), int(test_end),
        std_buf, int(std_offset), int(std_max),
    )


def run_phi_gaussian_train_state(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    q: float,
    c: float,
    phi: float,
    train_start: int,
    train_end: int,
    P0: float = 1e-4,
) -> Tuple[float, float, float]:
    """Run terminal-state φ-Gaussian training fold without array allocation."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    mu, P, ll = phi_gaussian_train_state_kernel(
        returns, vol_sq,
        float(q), float(c), float(phi),
        int(train_start), int(train_end), float(P0),
    )
    return float(mu), float(P), float(ll)


def is_cv_kernel_available() -> bool:
    """Check if Numba CV test-fold kernels are compiled and available."""
    return _NUMBA_AVAILABLE

# =============================================================================
# GAS-Q GAUSSIAN FILTER WRAPPER
# =============================================================================

def run_gas_q_filter_gaussian(
    returns: np.ndarray,
    vol: np.ndarray,
    c: float,
    phi: float,
    omega: float,
    alpha: float,
    beta: float,
    q_init: float,
    q_min: float,
    q_max: float,
    score_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Run Numba-accelerated GAS-Q Gaussian filter.

    Returns
    -------
    mu_filtered, P_filtered, q_path, score_path, log_likelihood
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")

    returns, vol = prepare_arrays(returns, vol)
    vol_sq = vol * vol
    n = len(returns)
    mu_filtered = np.zeros(n, dtype=np.float64)
    P_filtered = np.zeros(n, dtype=np.float64)
    q_path = np.zeros(n, dtype=np.float64)
    score_path = np.zeros(n, dtype=np.float64)

    log_ll = gas_q_filter_gaussian_kernel(
        returns, vol_sq, float(c), float(phi),
        float(omega), float(alpha), float(beta),
        float(q_init), float(q_min), float(q_max), float(score_scale),
        mu_filtered, P_filtered, q_path, score_path,
    )
    return mu_filtered, P_filtered, q_path, score_path, float(log_ll)


# =============================================================================
# BUILD-GARCH WRAPPER
# =============================================================================

def run_build_garch(
    n_train: int,
    innovations: np.ndarray,
    sq_inn: np.ndarray,
    neg_ind: np.ndarray,
    garch_omega: float,
    garch_alpha: float,
    garch_leverage: float,
    garch_beta: float,
    unconditional_var: float,
    q_stress_ratio: float,
    rho_c: float,
    kap_c: float,
    eta_c: float = 0.0,
    reg_c: float = 0.0,
) -> np.ndarray:
    """
    Run Numba-accelerated GJR-GARCH variance construction.

    Returns h array of length n_train.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")

    h_out = np.zeros(n_train, dtype=np.float64)
    build_garch_kernel(
        int(n_train),
        np.ascontiguousarray(innovations, dtype=np.float64),
        np.ascontiguousarray(sq_inn, dtype=np.float64),
        np.ascontiguousarray(neg_ind, dtype=np.float64),
        float(garch_omega), float(garch_alpha),
        float(garch_leverage), float(garch_beta),
        float(unconditional_var), float(q_stress_ratio),
        float(rho_c), float(kap_c), float(eta_c), float(reg_c),
        h_out,
    )
    return h_out


# =============================================================================
# CHI² EWM CORRECTION WRAPPER
# =============================================================================

def run_chi2_ewm_correction(
    z_raw: np.ndarray,
    chi2_target: float,
    chi2_lambda: float = 0.98,
) -> np.ndarray:
    """
    Run Numba-accelerated chi² EWM scale correction.

    Returns scale adjustment array (same length as z_raw).
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")

    z_raw = np.ascontiguousarray(z_raw, dtype=np.float64)
    scale_adj = np.ones(len(z_raw), dtype=np.float64)
    chi2_ewm_correction_kernel(z_raw, float(chi2_target), float(chi2_lambda), scale_adj)
    return scale_adj


# =============================================================================
# PIT-VARIANCE STRETCHING WRAPPER
# =============================================================================

def run_pit_var_stretching(
    pit_values: np.ndarray,
) -> np.ndarray:
    """
    Run Numba-accelerated PIT-variance stretching in-place.

    Returns the modified pit_values array.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")

    pit_values = np.ascontiguousarray(pit_values, dtype=np.float64)
    pit_var_stretching_kernel(pit_values)
    return pit_values


def is_gas_q_kernel_available() -> bool:
    """Check if GAS-Q Numba kernel is available."""
    return _NUMBA_AVAILABLE


# =============================================================================
# phi-STUDENT-T CV TEST-FOLD WRAPPER
# =============================================================================

def run_phi_student_t_cv_test_fold(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu_scale: float,
    log_norm_const: float,
    neg_exp: float,
    inv_nu: float,
    mu_init: float,
    P_init: float,
    test_start: int,
    test_end: int,
    nu_val: float = 8.0,
    gamma_vov: float = 0.0,
    vov_rolling: np.ndarray = None,
) -> float:
    """
    Run Numba-accelerated phi-Student-t CV test-fold forward pass.

    Returns log-likelihood of the validation fold.
    Supports VoV inflation and robust Student-t weighting.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    use_vov = 1 if (gamma_vov > 1e-12 and vov_rolling is not None) else 0
    if vov_rolling is None:
        vov_rolling = np.empty(1, dtype=np.float64)  # dummy for Numba typing
    return phi_student_t_cv_test_fold_kernel(
        returns, vol_sq,
        float(q), float(c), float(phi),
        float(nu_scale), float(log_norm_const), float(neg_exp),
        float(inv_nu),
        float(mu_init), float(P_init),
        int(test_start), int(test_end),
        float(nu_val), float(gamma_vov),
        vov_rolling, int(use_vov),
    )


def run_phi_student_t_cv_test_fold_stats(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu_scale: float,
    log_norm_const: float,
    neg_exp: float,
    inv_nu: float,
    mu_init: float,
    P_init: float,
    test_start: int,
    test_end: int,
    nu_val: float = 8.0,
    gamma_vov: float = 0.0,
    vov_rolling: np.ndarray = None,
) -> Tuple[float, int, float]:
    """Run a phi-Student-t CV fold and return LL plus dispersion stats."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    use_vov = 1 if (gamma_vov > 1e-12 and vov_rolling is not None) else 0
    if vov_rolling is None:
        vov_rolling = np.empty(1, dtype=np.float64)
    ll_fold, obs_count, z2_sum = phi_student_t_cv_test_fold_stats_kernel(
        returns, vol_sq,
        float(q), float(c), float(phi),
        float(nu_scale), float(log_norm_const), float(neg_exp),
        float(inv_nu),
        float(mu_init), float(P_init),
        int(test_start), int(test_end),
        float(nu_val), float(gamma_vov),
        vov_rolling, int(use_vov),
    )
    return float(ll_fold), int(obs_count), float(z2_sum)


def run_phi_student_t_train_state(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    log_norm_const: float,
    neg_exp: float,
    inv_nu: float,
    train_start: int,
    train_end: int,
    gamma_vov: float = 0.0,
    vov_rolling: np.ndarray = None,
    robust_wt: bool = True,
    online_scale_adapt: bool = True,
) -> Tuple[float, float, float]:
    """Run terminal-state Student-t training fold without array allocation."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    use_vov = 1 if (gamma_vov > 1e-12 and vov_rolling is not None) else 0
    if vov_rolling is None:
        vov_rolling = np.empty(1, dtype=np.float64)
    mu, P, ll = phi_student_t_train_state_kernel(
        returns, vol_sq,
        float(q), float(c), float(phi), float(nu),
        float(log_norm_const), float(neg_exp), float(inv_nu),
        int(train_start), int(train_end),
        float(gamma_vov), vov_rolling, int(use_vov),
        1 if robust_wt else 0,
        1 if online_scale_adapt else 0,
    )
    return float(mu), float(P), float(ll)


def run_phi_student_t_improved_train_state(
    returns: np.ndarray,
    vol_sq: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    log_norm_const: float,
    neg_exp: float,
    inv_nu: float,
    scale_factor: float,
    train_start: int,
    train_end: int,
    gamma_vov: float = 0.0,
    vov_rolling: np.ndarray = None,
    online_scale_adapt: bool = True,
    p_min: float = 1e-12,
    p_max_default: float = 1.0,
) -> Tuple[float, float, float]:
    """Run terminal-state improved Student-t training fold without arrays."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    use_vov = 1 if (gamma_vov > 1e-12 and vov_rolling is not None) else 0
    if vov_rolling is None:
        vov_rolling = np.empty(1, dtype=np.float64)
    mu, P, ll = phi_student_t_improved_train_state_kernel(
        returns, vol_sq,
        float(q), float(c), float(phi), float(nu),
        float(log_norm_const), float(neg_exp), float(inv_nu),
        float(scale_factor),
        int(train_start), int(train_end),
        float(gamma_vov), vov_rolling, int(use_vov),
        1 if online_scale_adapt else 0,
        float(p_min), float(p_max_default),
    )
    return float(mu), float(P), float(ll)


# =============================================================================
# MS PROCESS NOISE EWM WRAPPER
# =============================================================================

def run_compute_ms_process_noise_ewm(
    vol: np.ndarray,
    lam: float,
    warmup_mean: float,
    warmup_var: float,
) -> np.ndarray:
    """Run Numba-accelerated EWM z-score computation for MS process noise."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return compute_ms_process_noise_ewm_kernel(
        np.ascontiguousarray(vol.ravel(), dtype=np.float64),
        float(lam), float(warmup_mean), float(warmup_var),
    )


# =============================================================================
# STAGE 6 EWM FOLD WRAPPER
# =============================================================================

def run_stage6_ewm_fold(
    it_arr: np.ndarray,
    Sb_arr: np.ndarray,
    ee: int,
    ve: int,
    lam: float,
    init_em: float,
    init_en: float,
    init_ed: float,
) -> tuple:
    """Run Numba-accelerated Stage 6 EWM fold computation."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return stage6_ewm_fold_kernel(
        np.ascontiguousarray(it_arr, dtype=np.float64),
        np.ascontiguousarray(Sb_arr, dtype=np.float64),
        int(ee), int(ve), float(lam),
        float(init_em), float(init_en), float(init_ed),
    )


# =============================================================================
# STAGE 5f EWM CORRECTION WRAPPER
# =============================================================================

def run_ewm_mu_correction(
    returns: np.ndarray,
    mu_pred: np.ndarray,
    lam: float,
    n_train: int,
) -> np.ndarray:
    """Run Numba-accelerated Stage 5f EWM bias correction."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return ewm_mu_correction_kernel(
        np.ascontiguousarray(returns, dtype=np.float64),
        np.ascontiguousarray(mu_pred, dtype=np.float64),
        float(lam), int(n_train),
    )


# =============================================================================
# GAUSSIAN SCORE FOLD WRAPPER
# =============================================================================

def run_gaussian_score_fold(
    it_arr: np.ndarray,
    Sb_arr: np.ndarray,
    ee: int,
    ve: int,
    lam: float,
    init_em: float,
    init_en: float,
    init_ed: float,
) -> tuple:
    """Run Numba-accelerated Gaussian Stage 5 _score_fold."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return gaussian_score_fold_kernel(
        np.ascontiguousarray(it_arr, dtype=np.float64),
        np.ascontiguousarray(Sb_arr, dtype=np.float64),
        int(ee), int(ve), float(lam),
        float(init_em), float(init_en), float(init_ed),
    )


# =============================================================================
# STUDENT-T CDF / PDF / CRPS WRAPPERS
# =============================================================================

def run_student_t_cdf_array(
    z_arr: np.ndarray,
    nu: float,
) -> np.ndarray:
    """Numba-accelerated Student-t CDF (replaces scipy.stats.t.cdf)."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return student_t_cdf_array_kernel(
        np.ascontiguousarray(z_arr.ravel(), dtype=np.float64),
        float(nu),
    )


def run_student_t_pdf_array(
    z_arr: np.ndarray,
    nu: float,
) -> np.ndarray:
    """Numba-accelerated Student-t PDF (replaces scipy.stats.t.pdf)."""
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return student_t_pdf_array_kernel(
        np.ascontiguousarray(z_arr.ravel(), dtype=np.float64),
        float(nu),
    )


def run_crps_student_t(
    z_arr: np.ndarray,
    sigma_arr: np.ndarray,
    nu: float,
) -> float:
    """Numba-accelerated Student-t CRPS (v7.6: numerical g(ν)).

    v7.6: Switched from analytic B_ratio formula (crps_student_t_kernel,
    incorrect C(ν) constant) to numerical Gini half-mean-difference
    (crps_student_t_numerical_kernel, correct for all ν).
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    return float(crps_student_t_numerical_kernel(
        np.ascontiguousarray(z_arr.ravel(), dtype=np.float64),
        np.ascontiguousarray(sigma_arr.ravel(), dtype=np.float64),
        float(nu),
    ))


# =============================================================================
# PIT-KS UNIFIED WRAPPER
# =============================================================================

def run_pit_ks_unified(
    returns: np.ndarray,
    mu_pred: np.ndarray,
    S_pred: np.ndarray,
    nu_base: float,
    alpha_asym: float,
    k_asym: float,
    variance_inflation: float,
) -> np.ndarray:
    """
    Compute PIT values for unified Student-t with smooth asymmetric nu.

    Replaces per-element Python loop + scalar scipy CDF with compiled Numba.
    ~50x faster for typical 2000-element series.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    returns = np.ascontiguousarray(returns.ravel(), dtype=np.float64)
    mu_pred = np.ascontiguousarray(mu_pred.ravel(), dtype=np.float64)
    S_pred = np.ascontiguousarray(S_pred.ravel(), dtype=np.float64)
    pit_out = np.empty(len(returns), dtype=np.float64)
    pit_ks_unified_kernel(
        returns, mu_pred, S_pred,
        float(nu_base), float(alpha_asym), float(k_asym),
        float(variance_inflation), pit_out,
    )
    return pit_out


# =============================================================================
# GARCH VARIANCE WRAPPER
# =============================================================================

def run_garch_variance(
    innovations: np.ndarray,
    go: float, ga: float, gb: float, gl: float, gu: float,
    rl: float, km: float, tv: float, se: float, rs: float, sm: float,
) -> np.ndarray:
    """
    Compute GJR-GARCH(1,1) variance with full feature set.

    Replaces Python loop in _compute_garch_variance with compiled Numba.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    innovations = np.ascontiguousarray(innovations.ravel(), dtype=np.float64)
    n = len(innovations)
    sq = innovations * innovations
    neg = (innovations < 0).astype(np.float64)
    h_out = np.empty(n, dtype=np.float64)
    garch_variance_kernel(
        sq, neg, innovations, n,
        float(go), float(ga), float(gb), float(gl), float(gu),
        float(rl), float(km), float(tv), float(se), float(rs), float(sm),
        h_out,
    )
    return h_out


# =============================================================================
# GAUSSIAN FILTER WITH PREDICTIVE WRAPPER
# =============================================================================

def run_phi_gaussian_filter_with_predictive(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    P0: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    φ-Gaussian filter returning predictive mu_pred and S_pred for PIT.

    Numba-compiled replacement for gaussian.py filter_phi_with_predictive.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    returns, vol = prepare_arrays(returns, vol)
    return phi_gaussian_filter_with_predictive_kernel(
        returns, vol, float(q), float(c), float(phi), float(P0),
    )


# =============================================================================
# STUDENT-T AUGMENTED FILTER WRAPPER
# =============================================================================

def run_phi_student_t_augmented_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    exogenous_input: np.ndarray = None,
    robust_wt: bool = False,
    P0: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    φ-Student-t filter with optional exogenous input and robust weighting.

    Numba-compiled replacement for _filter_phi_core Python fallback.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    returns, vol = prepare_arrays(returns, vol)
    log_g1, log_g2 = precompute_gamma_values(nu)
    has_exo = exogenous_input is not None
    if has_exo:
        exogenous_input = np.ascontiguousarray(
            exogenous_input.ravel(), dtype=np.float64
        )
    else:
        exogenous_input = np.empty(0, dtype=np.float64)
    return phi_student_t_augmented_filter_kernel(
        returns, vol,
        float(q), float(c), float(phi), float(nu),
        log_g1, log_g2,
        exogenous_input, has_exo, robust_wt,
        float(P0),
    )


# =============================================================================
# STUDENT-T ENHANCED FILTER WRAPPER (VoV + Online Scale Adapt)
# =============================================================================

def run_phi_student_t_enhanced_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    exogenous_input: np.ndarray = None,
    robust_wt: bool = False,
    online_scale_adapt: bool = False,
    gamma_vov: float = 0.0,
    vov_rolling: np.ndarray = None,
    P0: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    φ-Student-t filter with VoV + online scale adaptation.

    Numba-compiled replacement for the Python fallback in _filter_phi_core
    when VoV or online_scale_adapt are active. Provides 5-10× speedup
    for ν=3,4 optimization in Stage 1.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    from models.numba_kernels import phi_student_t_enhanced_filter_kernel

    returns, vol = prepare_arrays(returns, vol)
    log_g1, log_g2 = precompute_gamma_values(nu)
    has_exo = exogenous_input is not None
    if has_exo:
        exogenous_input = np.ascontiguousarray(
            exogenous_input.ravel(), dtype=np.float64
        )
    else:
        exogenous_input = np.empty(0, dtype=np.float64)
    has_vov = gamma_vov > 1e-12 and vov_rolling is not None
    if has_vov:
        vov_rolling = np.ascontiguousarray(
            vov_rolling.ravel(), dtype=np.float64
        )
    else:
        vov_rolling = np.empty(0, dtype=np.float64)
    return phi_student_t_enhanced_filter_kernel(
        returns, vol,
        float(q), float(c), float(phi), float(nu),
        log_g1, log_g2,
        exogenous_input, has_exo, robust_wt,
        online_scale_adapt, float(gamma_vov), vov_rolling, has_vov,
        float(P0),
    )


# =============================================================================
# HANSEN SKEW-T FILTER WRAPPER (March 2026)
# =============================================================================

def run_phi_hansen_skew_t_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    hansen_lambda: float,
    exogenous_input: np.ndarray = None,
    online_scale_adapt: bool = False,
    gamma_vov: float = 0.0,
    vov_rolling: np.ndarray = None,
    P0: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    φ-Student-t filter with Hansen Skew-t observation noise.

    Numba-compiled. Always uses robust weighting (Hansen-adapted).
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    from models.numba_kernels import phi_hansen_skew_t_filter_kernel

    returns, vol = prepare_arrays(returns, vol)
    has_exo = exogenous_input is not None
    if has_exo:
        exogenous_input = np.ascontiguousarray(
            exogenous_input.ravel(), dtype=np.float64
        )
    else:
        exogenous_input = np.empty(0, dtype=np.float64)
    has_vov = gamma_vov > 1e-12 and vov_rolling is not None
    if has_vov:
        vov_rolling = np.ascontiguousarray(
            vov_rolling.ravel(), dtype=np.float64
        )
    else:
        vov_rolling = np.empty(0, dtype=np.float64)
    return phi_hansen_skew_t_filter_kernel(
        returns, vol,
        float(q), float(c), float(phi), float(nu),
        float(hansen_lambda),
        exogenous_input, has_exo,
        online_scale_adapt, float(gamma_vov), vov_rolling, has_vov,
        float(P0),
    )


# =============================================================================
# CST FILTER WRAPPER (March 2026)
# =============================================================================

def run_phi_cst_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu_normal: float,
    nu_crisis: float,
    epsilon: float,
    exogenous_input: np.ndarray = None,
    online_scale_adapt: bool = False,
    gamma_vov: float = 0.0,
    vov_rolling: np.ndarray = None,
    P0: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    φ-Student-t filter with Contaminated Student-t observation noise.

    Numba-compiled. Always uses CST-posterior-weighted robust updates.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    from models.numba_kernels import phi_cst_filter_kernel

    returns, vol = prepare_arrays(returns, vol)
    has_exo = exogenous_input is not None
    if has_exo:
        exogenous_input = np.ascontiguousarray(
            exogenous_input.ravel(), dtype=np.float64
        )
    else:
        exogenous_input = np.empty(0, dtype=np.float64)
    has_vov = gamma_vov > 1e-12 and vov_rolling is not None
    if has_vov:
        vov_rolling = np.ascontiguousarray(
            vov_rolling.ravel(), dtype=np.float64
        )
    else:
        vov_rolling = np.empty(0, dtype=np.float64)
    return phi_cst_filter_kernel(
        returns, vol,
        float(q), float(c), float(phi),
        float(nu_normal), float(nu_crisis), float(epsilon),
        exogenous_input, has_exo,
        online_scale_adapt, float(gamma_vov), vov_rolling, has_vov,
        float(P0),
    )


# =============================================================================
# AD TAIL-CORRECTION WRAPPERS (March 2026)
# =============================================================================

def run_ad_twsc(
    z_arr: np.ndarray,
    ewma_lambda: float = 0.97,
    alpha_quantile: float = 0.05,
    kappa: float = 0.5,
    max_inflate: float = 2.0,
    deadzone: float = 0.15,
) -> np.ndarray:
    """
    Run Numba-accelerated Tail-Weighted Scale Correction.

    Returns per-observation scale inflation factors.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    z_arr = np.ascontiguousarray(z_arr.ravel(), dtype=np.float64)
    return ad_twsc_kernel(z_arr, float(ewma_lambda), float(alpha_quantile),
                          float(kappa), float(max_inflate), float(deadzone))


def run_ad_sptg_student_t(
    z_arr: np.ndarray,
    nu: float,
    xi_left: float,
    sigma_left: float,
    u_left: float,
    xi_right: float,
    sigma_right: float,
    u_right: float,
    p_left: float,
    p_right: float,
) -> np.ndarray:
    """
    Run Numba-accelerated SPTG CDF for Student-t.

    Returns array of PIT values with GPD-grafted tails.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    z_arr = np.ascontiguousarray(z_arr.ravel(), dtype=np.float64)
    return ad_sptg_cdf_student_t_array(
        z_arr, float(nu),
        float(xi_left), float(sigma_left), float(u_left),
        float(xi_right), float(sigma_right), float(u_right),
        float(p_left), float(p_right),
    )


def run_ad_sptg_gaussian(
    z_arr: np.ndarray,
    xi_left: float,
    sigma_left: float,
    u_left: float,
    xi_right: float,
    sigma_right: float,
    u_right: float,
    p_left: float,
    p_right: float,
) -> np.ndarray:
    """
    Run Numba-accelerated SPTG CDF for Gaussian.

    Returns array of PIT values with GPD-grafted tails.
    """
    if not _NUMBA_AVAILABLE:
        raise ImportError("Numba kernels not available")
    z_arr = np.ascontiguousarray(z_arr.ravel(), dtype=np.float64)
    return ad_sptg_cdf_gaussian_array(
        z_arr,
        float(xi_left), float(sigma_left), float(u_left),
        float(xi_right), float(sigma_right), float(u_right),
        float(p_left), float(p_right),
    )
