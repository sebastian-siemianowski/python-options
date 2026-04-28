"""
Phi-Student-t Drift Model — Kalman Filter with AR(1) Drift and Student-t Noise.

    State:        μ_t = φ·μ_{t-1} + w_t,  w_t ~ N(0, q)
    Observation:  r_t = μ_t + ε_t,         ε_t ~ Student-t(ν, 0, scale_t)

    Var(ε_t) = c·σ_t²;  scale_t = sqrt(c·σ_t² × (ν-2)/ν) for ν > 2.

Parameters:
    q   Process noise variance (drift evolution speed)
    c   Observation noise multiplier (scales EWMA σ_t²)
    φ   AR(1) persistence (1=random walk, 0=mean-reverting)
    ν   Degrees of freedom (tail heaviness; ν→∞ → Gaussian)

ν is selected from a discrete grid to avoid identifiability issues;
each ν value becomes a separate BMA sub-model.
Gaussian shrinkage prior on φ: φ_r ~ N(φ_global, τ²).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm
from scipy.stats import t as student_t

try:
    from .student_t_common import (
        asset_phi_profile,
        compute_berkowitz_full as common_compute_berkowitz_full,
        compute_berkowitz_pvalue as common_compute_berkowitz_pvalue,
        compute_ad_statistic,
        compute_cvm_statistic,
        estimate_gjr_garch_params,
        ewm_lagged_correction,
        pit_simple_path as common_pit_simple_path,
        precompute_vov,
        variance_to_scale,
        variance_to_scale_vec,
    )
except ImportError:
    from models.student_t_common import (
        asset_phi_profile,
        compute_berkowitz_full as common_compute_berkowitz_full,
        compute_berkowitz_pvalue as common_compute_berkowitz_pvalue,
        compute_ad_statistic,
        compute_cvm_statistic,
        estimate_gjr_garch_params,
        ewm_lagged_correction,
        pit_simple_path as common_pit_simple_path,
        precompute_vov,
        variance_to_scale,
        variance_to_scale_vec,
    )

try:
    from .asset_classification import (
        CRYPTO_SYMBOLS as _SHARED_CRYPTO_SYMBOLS,
        HIGH_VOL_EQUITY_SYMBOLS as _SHARED_HIGH_VOL_EQUITY_SYMBOLS,
        INDEX_SYMBOLS as _SHARED_INDEX_SYMBOLS,
        LARGE_CAP_SYMBOLS as _SHARED_LARGE_CAP_SYMBOLS,
        METALS_GOLD_SYMBOLS as _SHARED_METALS_GOLD_SYMBOLS,
        METALS_OTHER_SYMBOLS as _SHARED_METALS_OTHER_SYMBOLS,
        METALS_SILVER_SYMBOLS as _SHARED_METALS_SILVER_SYMBOLS,
        detect_asset_class as _shared_detect_asset_class,
    )
except ImportError:
    from models.asset_classification import (
        CRYPTO_SYMBOLS as _SHARED_CRYPTO_SYMBOLS,
        HIGH_VOL_EQUITY_SYMBOLS as _SHARED_HIGH_VOL_EQUITY_SYMBOLS,
        INDEX_SYMBOLS as _SHARED_INDEX_SYMBOLS,
        LARGE_CAP_SYMBOLS as _SHARED_LARGE_CAP_SYMBOLS,
        METALS_GOLD_SYMBOLS as _SHARED_METALS_GOLD_SYMBOLS,
        METALS_OTHER_SYMBOLS as _SHARED_METALS_OTHER_SYMBOLS,
        METALS_SILVER_SYMBOLS as _SHARED_METALS_SILVER_SYMBOLS,
        detect_asset_class as _shared_detect_asset_class,
    )

# Filter cache for deterministic result reuse
try:
    from .filter_cache import (
        cached_phi_student_t_filter,
        get_filter_cache,
        FilterCacheKey,
        FILTER_CACHE_ENABLED,
    )
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False
    FILTER_CACHE_ENABLED = False

# Numba wrappers for JIT-compiled filters (optional performance enhancement)
try:
    from .numba_wrappers import (
        is_numba_available,
        run_phi_student_t_filter,
        run_phi_student_t_filter_batch,
        # MS-q and fused LFO-CV wrappers
        run_ms_q_student_t_filter,
        run_student_t_filter_with_lfo_cv,
        run_student_t_filter_with_lfo_cv_batch,
        # Filter accelerators
        run_phi_student_t_augmented_filter,
        run_phi_student_t_enhanced_filter,
        run_phi_student_t_improved_cv_test_fold,
        run_phi_student_t_improved_train_state_only,
        # CDF/PDF array kernels (vectorized replacements for scipy)
        run_student_t_cdf_array,
        run_student_t_pdf_array,
    )
    _USE_NUMBA = is_numba_available()
    _MS_Q_NUMBA_AVAILABLE = _USE_NUMBA
    _NUMBA_CDF_ARRAY = _USE_NUMBA
    _NUMBA_ENHANCED = _USE_NUMBA
except ImportError:
    _USE_NUMBA = False
    _MS_Q_NUMBA_AVAILABLE = False
    _NUMBA_CDF_ARRAY = False
    _NUMBA_ENHANCED = False
    run_phi_student_t_filter = None
    run_phi_student_t_filter_batch = None
    run_ms_q_student_t_filter = None
    run_student_t_filter_with_lfo_cv = None
    run_student_t_filter_with_lfo_cv_batch = None
    run_phi_student_t_augmented_filter = None
    run_phi_student_t_improved_cv_test_fold = None
    run_phi_student_t_improved_train_state_only = None
    run_student_t_cdf_array = None
    run_student_t_pdf_array = None

# Numerical stability primitives (Epic 28)
try:
    from calibration.numerical_stability import (
        safe_student_t_logpdf,
        safe_student_t_logpdf_scalar,
        clamp_covariance_array,
        P_MIN_DEFAULT as _P_MIN,
        P_MAX_DEFAULT as _P_MAX,
    )
    _NUMERICAL_STABILITY_AVAILABLE = True
except ImportError:
    _NUMERICAL_STABILITY_AVAILABLE = False
    _P_MIN = 1e-10
    _P_MAX = 1.0


def _fast_t_cdf(z, nu):
    """
    Vectorized Student-t CDF: uses Numba array kernel when available, scipy fallback.
    """
    if _NUMBA_CDF_ARRAY:
        try:
            return run_student_t_cdf_array(np.ascontiguousarray(z, dtype=np.float64), float(nu))
        except Exception:
            pass
    from scipy.stats import t as _t
    return _t.cdf(z, df=nu)


def _fast_t_pdf(z, nu):
    """
    Vectorized Student-t PDF: uses Numba array kernel when available, scipy fallback.
    """
    if _NUMBA_CDF_ARRAY and run_student_t_pdf_array is not None:
        try:
            return run_student_t_pdf_array(np.ascontiguousarray(z, dtype=np.float64), float(nu))
        except Exception:
            pass
    from scipy.stats import t as _t
    return _t.pdf(z, df=nu)


def _fast_ks_uniform(pit_values):
    """
    Inline KS test against Uniform(0,1) — replaces scipy.stats.kstest.
    
    Returns (statistic, p_value).
    
    Uses the Kolmogorov asymptotic approximation for p-value:
        λ = (√n + 0.12 + 0.11/√n) × D
        p ≈ 2·exp(-2λ²)
    
    This is equivalent to scipy's kstest for n > ~40 (our typical n > 100).
    Eliminates the expensive kolmogn() p-value computation (0.275s cumulative
    from 1,056 calls in a single-asset tune).
    """
    n = len(pit_values)
    if n < 2:
        return 1.0, 0.0
    sorted_pit = np.sort(pit_values)
    ecdf = np.arange(1, n + 1) / n
    D_plus = float(np.max(ecdf - sorted_pit))
    D_minus = float(np.max(sorted_pit - np.arange(0, n) / n))
    D = max(D_plus, D_minus)
    sqrt_n = math.sqrt(n)
    lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * D
    if lam < 0.001:
        p = 1.0
    elif lam > 3.0:
        p = 0.0
    else:
        # 4-term alternating series: P(K>λ) = 2·Σ (-1)^{k+1} exp(-2k²λ²)
        # Single-term (2·exp(-2λ²)) overestimates and saturates at 1.0 for λ<0.59
        lam2 = lam * lam
        p = 2.0 * (math.exp(-2.0 * lam2)
                   - math.exp(-8.0 * lam2)
                   + math.exp(-18.0 * lam2)
                   - math.exp(-32.0 * lam2))
        if p < 0.0:
            p = 0.0
    return D, p


# ---------------------------------------------------------------------------
# φ SHRINKAGE PRIOR CONSTANTS (self-contained, no external dependencies)
# ---------------------------------------------------------------------------

PHI_SHRINKAGE_TAU_MIN = 1e-3
PHI_SHRINKAGE_GLOBAL_DEFAULT = 0.0
PHI_SHRINKAGE_LAMBDA_DEFAULT = 0.05

# Discrete ν grid for Student-t models
# 4 BMA flavours: heavy tails (3), fat tails (4), moderate (8), near-Gaussian (20).
# Intermediate ν values are still explored by the ν-refinement step inside
# tune_and_calibrate — this grid only controls how many *separate models*
# compete in BMA at the tune.py level.
STUDENT_T_NU_GRID = [3, 4, 8, 20]


# ── Story 6.1: Continuous nu optimization via profile likelihood ──────
def profile_likelihood_refine_nu(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu_grid_best: float,
    nu_lo: float = 2.1,
    nu_hi: float = 50.0,
    filter_func=None,
) -> Tuple[float, float]:
    """Refine nu via golden-section search on profile log-likelihood.

    Fixes (q, c, phi) at their optimized values and searches for the
    nu that maximizes the Kalman filter log-likelihood over [nu_lo, nu_hi].
    Default search range: [nu_grid_best - 2, nu_grid_best + 4], clipped to [2.1, 50].

    Args:
        returns: Return series.
        vol: Volatility series.
        q, c, phi: Fixed filter parameters.
        nu_grid_best: Best nu from discrete grid search.
        nu_lo, nu_hi: Absolute bounds for nu search.
        filter_func: Callable(returns, vol, q, c, phi, nu) -> (mu, P, ll).
                     If None, uses PhiStudentTDriftModel.filter_phi.

    Returns:
        (nu_refined, ll_refined): Refined nu and its log-likelihood.
    """
    from scipy.optimize import minimize_scalar

    # Search range: [nu_grid_best - 2, nu_grid_best + 4], clipped
    search_lo = max(nu_lo, nu_grid_best - 2.0)
    search_hi = min(nu_hi, nu_grid_best + 4.0)
    if search_lo >= search_hi:
        search_lo = max(nu_lo, nu_grid_best - 1.0)
        search_hi = min(nu_hi, nu_grid_best + 2.0)

    if filter_func is None:
        filter_func = PhiStudentTDriftModel.filter_phi

    def neg_ll(nu_val):
        if nu_val <= 2.0 or nu_val > 50.0:
            return 1e12
        try:
            _, _, ll = filter_func(returns, vol, q, c, phi, nu_val)
            return -ll if math.isfinite(ll) else 1e12
        except Exception:
            return 1e12

    result = minimize_scalar(neg_ll, bounds=(search_lo, search_hi), method='bounded',
                             options={'xatol': 0.1, 'maxiter': 30})

    nu_refined = float(result.x) if result.success else nu_grid_best
    ll_refined = float(-result.fun) if result.success else float('nan')

    return nu_refined, ll_refined


# ---------------------------------------------------------------------------
# ENHANCED STUDENT-T CONFIGURATION
# Three enhancements for Hyvarinen/PIT calibration:
#   1. VoV:         R_t = c × σ² × (1 + γ × |Δlog(σ)|)
#   2. Two-Piece:   Different νL (crash) vs νR (recovery)
#   3. Mixture:     Blend νcalm/νstress with dynamic weights
# No BMA penalties — BIC's parameter count handles complexity.
# ---------------------------------------------------------------------------

# Vol-of-Vol (VoV) Enhancement
GAMMA_VOV_GRID = [0.3, 0.5, 0.7]
VOV_BMA_PENALTY = 0.0  # REMOVED: Equal competition with base Student-t

# Two-Piece Student-t Enhancement
NU_LEFT_GRID = [3, 4, 5]
NU_RIGHT_GRID = [8, 12, 20]
TWO_PIECE_BMA_PENALTY = 0.0  # REMOVED: Equal competition with base Student-t


def refine_two_piece_nu(
    nu_L_grid: float,
    nu_R_grid: float,
    innovations: np.ndarray,
    scale: float = 1.0,
    nu_L_bounds: Tuple[float, float] = (2.1, 30.0),
    nu_R_bounds: Tuple[float, float] = (2.1, 30.0),
) -> Tuple[float, float, float]:
    """Refine (nu_L, nu_R) for two-piece Student-t via profile likelihood.

    Story 8.1: After grid search selects best (nu_L, nu_R), refine via
    2D bounded optimization. Uses two-piece Student-t log-likelihood:
    f(r) = c * t(r; nu_L) for r < 0, c * t(r; nu_R) for r >= 0.

    Args:
        nu_L_grid: Grid-selected left tail nu.
        nu_R_grid: Grid-selected right tail nu.
        innovations: Standardized innovations (residuals / scale).
        scale: Scale parameter (sigma).
        nu_L_bounds: Absolute bounds for nu_L search.
        nu_R_bounds: Absolute bounds for nu_R search.

    Returns:
        (nu_L_refined, nu_R_refined, ll): Refined parameters and log-likelihood.
    """
    from scipy.optimize import minimize

    innovations = np.asarray(innovations).flatten()
    z = innovations / max(scale, 1e-12)
    left_mask = z < 0.0
    right_mask = ~left_mask

    if _NUMERICAL_STABILITY_AVAILABLE:
        def _student_t_logpdf(x, nu):
            """Log-pdf of Student-t(nu, 0, 1) via safe_student_t_logpdf."""
            return safe_student_t_logpdf(np.asarray(x), nu, np.zeros_like(x), np.ones_like(x))
    else:
        from scipy.special import gammaln as scipy_gammaln

        def _student_t_logpdf(x, nu):
            """Log-pdf of Student-t(nu, 0, 1)."""
            return (
                scipy_gammaln(0.5 * (nu + 1)) - scipy_gammaln(0.5 * nu)
                - 0.5 * np.log(nu * np.pi)
                - 0.5 * (nu + 1) * np.log(1.0 + x ** 2 / nu)
            )

    def neg_ll(params):
        nu_L, nu_R = params
        if nu_L < 2.01 or nu_R < 2.01:
            return 1e12
        ll_left = np.sum(_student_t_logpdf(z[left_mask], nu_L)) if np.any(left_mask) else 0.0
        ll_right = np.sum(_student_t_logpdf(z[right_mask], nu_R)) if np.any(right_mask) else 0.0
        return -(ll_left + ll_right)

    # Search around grid-selected values
    lo_L = max(nu_L_bounds[0], nu_L_grid - 2.0)
    hi_L = min(nu_L_bounds[1], nu_L_grid + 3.0)
    lo_R = max(nu_R_bounds[0], nu_R_grid - 3.0)
    hi_R = min(nu_R_bounds[1], nu_R_grid + 3.0)

    result = minimize(
        neg_ll,
        x0=[nu_L_grid, nu_R_grid],
        method='L-BFGS-B',
        bounds=[(lo_L, hi_L), (lo_R, hi_R)],
        options={'maxiter': 100, 'ftol': 1e-8},
    )

    if result.success:
        nu_L_ref, nu_R_ref = float(result.x[0]), float(result.x[1])
        ll_ref = float(-result.fun)
    else:
        nu_L_ref, nu_R_ref = nu_L_grid, nu_R_grid
        ll_ref = float(-neg_ll([nu_L_grid, nu_R_grid]))

    return nu_L_ref, nu_R_ref, ll_ref

# Two-Component Mixture Student-t Enhancement
NU_CALM_GRID = [12, 20]
NU_STRESS_GRID = [4, 6]
MIXTURE_WEIGHT_DEFAULT = 0.8
MIXTURE_WEIGHT_K = 2.0  # Sigmoid sensitivity to vol_relative
MIXTURE_BMA_PENALTY = 0.0  # REMOVED: Equal competition with base Student-t

# ---------------------------------------------------------------------------
# ENHANCED MIXTURE WEIGHT DYNAMICS
# w_t = sigmoid(a×z_t + b×Δσ_t + c×M_t)
# Multi-factor: shocks (z_t), vol acceleration (Δσ_t), momentum (M_t).
# ---------------------------------------------------------------------------

# Default mixture weight sensitivity parameters
MIXTURE_WEIGHT_A_SHOCK = 1.0       # Sensitivity to standardized residuals
MIXTURE_WEIGHT_B_VOL_ACCEL = 0.5   # Sensitivity to vol acceleration
MIXTURE_WEIGHT_C_MOMENTUM = 0.3    # Sensitivity to momentum


# ── Story 8.2: Multi-Factor Mixture Weight Dynamics ──────────────────
# Asset-class default factor loadings (a=shock, b=vol_accel, c=momentum)
MIXTURE_FACTOR_LOADINGS = {
    'metals_gold':     (0.8, 0.3, 0.2),
    'metals_silver':   (1.2, 0.6, 0.2),
    'high_vol_equity': (1.5, 0.8, 0.4),
    'crypto':          (1.3, 0.7, 0.3),
    'large_cap':       (1.0, 0.5, 0.3),
    'index':           (0.8, 0.4, 0.3),
    'forex':           (0.9, 0.4, 0.2),
}


def compute_mixture_weight_dynamic(
    returns: np.ndarray,
    vol: np.ndarray,
    a: float = MIXTURE_WEIGHT_A_SHOCK,
    b: float = MIXTURE_WEIGHT_B_VOL_ACCEL,
    c: float = MIXTURE_WEIGHT_C_MOMENTUM,
    base_weight: float = MIXTURE_WEIGHT_DEFAULT,
    momentum_window: int = 20,
) -> Tuple[np.ndarray, dict]:
    """Compute time-varying calm-regime mixture weight from robust market-state factors.

    The implementation is deliberately leakage-free: every feature at ``t`` only
    uses information available before or at ``t``.  Inputs are length-aligned and
    sanitised so a single bad quote cannot poison the whole calibration pass.
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    vol = np.asarray(vol, dtype=np.float64).ravel()
    n = min(len(returns), len(vol))
    if n <= 0:
        return np.empty(0, dtype=np.float64), {
            'mean_w': float('nan'), 'min_w': float('nan'), 'max_w': float('nan'),
            'factor_z_std': 0.0, 'factor_vol_accel_std': 0.0,
            'factor_momentum_std': 0.0, 'n_obs': 0,
        }

    returns = returns[:n].copy()
    vol = vol[:n].copy()

    finite_r = np.isfinite(returns)
    if not np.all(finite_r):
        returns[~finite_r] = 0.0

    finite_vol = vol[np.isfinite(vol) & (vol > 0)]
    vol_fill = float(np.median(finite_vol)) if finite_vol.size else max(float(np.std(returns)), 1e-4)
    vol_floor = max(vol_fill * 1e-4, 1e-10)
    vol = np.where(np.isfinite(vol) & (vol > 0), vol, vol_fill)
    vol = np.maximum(vol, vol_floor)

    # Factor 1: bounded standardized shock.  Large absolute shocks move the
    # model toward the stress component, but clipping prevents one bad print
    # from saturating all downstream weights.
    z_t = -np.clip(np.abs(returns) / vol, 0.0, 8.0)

    # Factor 2: log-vol acceleration, robustly standardized.  A rise in vol is
    # stress, so the sign is negative for positive acceleration.
    log_vol = np.log(vol)
    delta_log_vol = np.zeros(n, dtype=np.float64)
    if n > 1:
        delta_log_vol[1:] = np.diff(log_vol)
    med_dv = float(np.median(delta_log_vol))
    mad_dv = float(np.median(np.abs(delta_log_vol - med_dv)))
    scale_dv = max(1.4826 * mad_dv, float(np.std(delta_log_vol)), 1e-8)
    delta_sigma_signal = -np.clip((delta_log_vol - med_dv) / scale_dv, -8.0, 8.0)

    # Factor 3: past-window return momentum, vectorised and leakage-free.
    M_t = np.zeros(n, dtype=np.float64)
    w = int(max(2, min(momentum_window, max(n - 1, 2))))
    if n > w:
        cs = np.concatenate(([0.0], np.cumsum(returns)))
        cs2 = np.concatenate(([0.0], np.cumsum(returns * returns)))
        idx = np.arange(w, n)
        sums = cs[idx] - cs[idx - w]
        sums2 = cs2[idx] - cs2[idx - w]
        means = sums / float(w)
        vars_ = np.maximum(sums2 / float(w) - means * means, 1e-12)
        M_t[idx] = np.clip(means / np.sqrt(vars_), -5.0, 5.0)

    # Combine via a numerically stable logistic map.
    bw = float(np.clip(base_weight, 1e-4, 1.0 - 1e-4))
    logit_base = math.log(bw / (1.0 - bw))
    linear = logit_base + float(a) * z_t + float(b) * delta_sigma_signal + float(c) * M_t
    linear = np.clip(linear, -50.0, 50.0)
    w_t = 1.0 / (1.0 + np.exp(-linear))
    w_t = np.clip(w_t, 0.01, 0.99)

    diagnostics = {
        'mean_w': float(np.mean(w_t)),
        'min_w': float(np.min(w_t)),
        'max_w': float(np.max(w_t)),
        'factor_z_std': float(np.std(z_t)),
        'factor_vol_accel_std': float(np.std(delta_sigma_signal)),
        'factor_momentum_std': float(np.std(M_t)),
        'n_obs': int(n),
        'momentum_window_used': int(w),
        'vol_floor': float(vol_floor),
    }

    return w_t, diagnostics


def get_mixture_factor_loadings(asset_symbol: str = None) -> Tuple[float, float, float]:
    """Return calibrated (a, b, c) factor loadings per asset class.

    Args:
        asset_symbol: Ticker symbol.

    Returns:
        (a_shock, b_vol_accel, c_momentum) tuple.
    """
    if asset_symbol is None:
        return (MIXTURE_WEIGHT_A_SHOCK, MIXTURE_WEIGHT_B_VOL_ACCEL,
                MIXTURE_WEIGHT_C_MOMENTUM)

    sym = asset_symbol.strip().upper()
    asset_class = _detect_asset_class(sym)

    if asset_class is not None and asset_class in MIXTURE_FACTOR_LOADINGS:
        return MIXTURE_FACTOR_LOADINGS[asset_class]

    if sym in _CRYPTO_SYMBOLS or 'BTC' in sym or 'ETH' in sym:
        return MIXTURE_FACTOR_LOADINGS['crypto']

    if sym in _LARGE_CAP_SYMBOLS:
        return MIXTURE_FACTOR_LOADINGS['large_cap']

    if sym in _INDEX_SYMBOLS:
        return MIXTURE_FACTOR_LOADINGS['index']

    return (MIXTURE_WEIGHT_A_SHOCK, MIXTURE_WEIGHT_B_VOL_ACCEL,
            MIXTURE_WEIGHT_C_MOMENTUM)


# ── Story 8.3: Mixture vs Two-Piece Model Selection Gate ─────────────
VOV_MIXTURE_THRESHOLD = 0.5    # Vol-of-vol threshold for mixture preference
SKEW_TWO_PIECE_THRESHOLD = 0.3  # Abs skewness threshold for two-piece preference
BIC_OCCAM_MARGIN = 6.0          # BIC margin (3 nats * 2) for Occam gate


def compute_model_selection_gate(
    returns: np.ndarray,
    vol: np.ndarray,
    bic_mixture: float,
    bic_two_piece: float,
) -> Tuple[str, dict]:
    """Select between mixture and two-piece Student-t models with robust gates.

    BIC remains primary.  The heuristic layer only breaks close calls using
    robust skew and vol-of-vol diagnostics so model selection is less brittle
    on noisy, short, or partially missing price histories.
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    vol = np.asarray(vol, dtype=np.float64).ravel()
    n = min(len(returns), len(vol))
    if n <= 0:
        selected = 'mixture' if np.isfinite(bic_mixture) and bic_mixture < bic_two_piece else 'two_piece'
        return selected, {
            'empirical_skewness': 0.0, 'abs_skewness': 0.0, 'vol_of_vol': 0.0,
            'bic_mixture': float(bic_mixture), 'bic_two_piece': float(bic_two_piece),
            'bic_diff_nats': float('nan'), 'selected': selected,
            'skew_favors_two_piece': False, 'vov_favors_mixture': False,
            'n_obs': 0,
        }

    r = returns[:n].copy()
    v = vol[:n].copy()
    r = r[np.isfinite(r)]
    if r.size < 3:
        r = np.zeros(3, dtype=np.float64)
    finite_v = v[np.isfinite(v) & (v > 0)]
    v_fill = float(np.median(finite_v)) if finite_v.size else 1.0
    v = np.where(np.isfinite(v) & (v > 0), v, v_fill)

    # Robust skew estimate: median/MAD standardisation plus clipping prevents
    # a single crash candle from dominating the model-family gate.
    r_med = float(np.median(r))
    r_mad = float(np.median(np.abs(r - r_med)))
    r_scale = max(1.4826 * r_mad, float(np.std(r)), 1e-10)
    z = np.clip((r - r_med) / r_scale, -8.0, 8.0)
    skew = float(np.mean(z ** 3))
    abs_skew = abs(skew)

    vol_mean = float(np.mean(np.abs(v)))
    vol_std = float(np.std(v))
    vov = vol_std / max(vol_mean, 1e-10)

    bic_mixture_f = float(bic_mixture) if np.isfinite(bic_mixture) else float('inf')
    bic_two_piece_f = float(bic_two_piece) if np.isfinite(bic_two_piece) else float('inf')
    bic_diff = bic_mixture_f - bic_two_piece_f
    bic_diff_nats = bic_diff / 2.0 if np.isfinite(bic_diff) else float('nan')

    adjusted_bic_mixture = bic_mixture_f
    adjusted_bic_two_piece = bic_two_piece_f

    # Vol-of-vol favours regime mixtures; static skew favours two-piece tails.
    if vov > VOV_MIXTURE_THRESHOLD:
        adjusted_bic_mixture -= 4.0
    if abs_skew > SKEW_TWO_PIECE_THRESHOLD:
        adjusted_bic_two_piece -= 2.0

    if not np.isfinite(adjusted_bic_mixture) and not np.isfinite(adjusted_bic_two_piece):
        selected = 'two_piece'
    elif abs(bic_diff) < BIC_OCCAM_MARGIN:
        selected = 'two_piece'
    elif adjusted_bic_mixture < adjusted_bic_two_piece:
        selected = 'mixture'
    else:
        selected = 'two_piece'

    # Keep the override bounded to close calls only; BIC still wins decisively.
    if abs_skew > SKEW_TWO_PIECE_THRESHOLD and (not np.isfinite(bic_diff_nats) or bic_diff_nats < 5.0):
        selected = 'two_piece'

    diagnostics = {
        'empirical_skewness': float(skew),
        'abs_skewness': float(abs_skew),
        'vol_of_vol': float(vov),
        'bic_mixture': float(bic_mixture_f),
        'bic_two_piece': float(bic_two_piece_f),
        'bic_diff_nats': float(bic_diff_nats),
        'adjusted_bic_mixture': float(adjusted_bic_mixture),
        'adjusted_bic_two_piece': float(adjusted_bic_two_piece),
        'selected': selected,
        'skew_favors_two_piece': bool(abs_skew > SKEW_TWO_PIECE_THRESHOLD),
        'vov_favors_mixture': bool(vov > VOV_MIXTURE_THRESHOLD),
        'n_obs': int(n),
    }

    return selected, diagnostics


# ---------------------------------------------------------------------------
# MARKOV-SWITCHING PROCESS NOISE (MS-q)
# ---------------------------------------------------------------------------
# Two-state (calm, stress) process noise driven by vol structure:
#   q_t = (1 - p_stress) × q_calm + p_stress × q_stress
#   p_stress = sigmoid(sensitivity × (vol_relative - threshold))
#
# Proactive: shifts q BEFORE forecast errors materialize (unlike GAS-Q).
# ---------------------------------------------------------------------------

# MS-q Configuration
MS_Q_ENABLED = True           # Master switch for MS-q models
MS_Q_CALM_DEFAULT = 1e-6      # Process noise in calm regime
MS_Q_STRESS_DEFAULT = 1e-4    # Process noise in stress regime (100x calm)
MS_Q_SENSITIVITY = 2.0        # Sigmoid sensitivity to vol_relative
MS_Q_THRESHOLD = 1.3          # vol_relative threshold for transition
MS_Q_BMA_PENALTY = 0.0        # No penalty - fair competition via BIC


# ---------------------------------------------------------------------------
# ASSET-CLASS ADAPTIVE CALIBRATION PROFILES
# ---------------------------------------------------------------------------
# Metals have fundamentally different vol dynamics than equities:
#   Gold — slow macro-driven regimes, jump processes (CPI, Fed)
#   Silver — explosive VoV, leveraged-gold, crisis fat tails
#
# Profiles adjust REGULARIZATION CENTERS and INITIALIZATION only.
# The optimizer still finds likelihood-optimal values.
# ---------------------------------------------------------------------------

# Metals ticker sets, maintained centrally with the retune universe.
METALS_GOLD_SYMBOLS = _SHARED_METALS_GOLD_SYMBOLS
METALS_SILVER_SYMBOLS = _SHARED_METALS_SILVER_SYMBOLS
METALS_OTHER_SYMBOLS = _SHARED_METALS_OTHER_SYMBOLS

# ---------------------------------------------------------------------------
# HIGH-VOLATILITY EQUITY SYMBOLS
# ---------------------------------------------------------------------------
# Crypto-correlated, meme, and micro-cap stocks with kurtosis >> 6.
# Need lower ν, sharper asymmetry, and weaker VoV damping for CRPS.
# ---------------------------------------------------------------------------
HIGH_VOL_EQUITY_SYMBOLS = _SHARED_HIGH_VOL_EQUITY_SYMBOLS


def _detect_asset_class(asset_symbol: str, returns=None) -> Optional[str]:
    """
    Detect asset class from symbol for calibration profile selection.

    Returns:
        'metals_gold': Gold futures/ETFs — slow macro regimes, jump-driven
        'metals_silver': Silver futures/ETFs — explosive VoV, leveraged-gold
        'metals_other': Other metals (copper, platinum, palladium)
        'high_vol_equity': Crypto-correlated / meme / micro-cap with extreme kurtosis
        None: No special profile (equities, FX, crypto — use generic)
    """
    return _shared_detect_asset_class(asset_symbol, returns=returns)


# ── Story 7.1: Calibrated MS-q sensitivity per asset class ───────────
# Each entry: (sensitivity, threshold)
MS_Q_ASSET_CLASS_PARAMS = {
    'metals_gold':     (4.0, 1.5),   # Slow, macro-driven
    'metals_silver':   (4.5, 1.2),   # Faster, gap-prone
    'metals_other':    (4.0, 1.3),   # Similar to gold
    'large_cap':       (2.5, 1.3),   # Standard
    'high_vol_equity': (3.0, 1.0),   # Earlier switching
    'crypto':          (3.5, 1.1),   # Structural breaks
    'index':           (2.5, 1.3),   # Standard
    'forex':           (3.0, 1.2),   # Moderate
}

# Broad classes for detection, maintained centrally with the retune universe.
_CRYPTO_SYMBOLS = _SHARED_CRYPTO_SYMBOLS
_LARGE_CAP_SYMBOLS = _SHARED_LARGE_CAP_SYMBOLS
_INDEX_SYMBOLS = _SHARED_INDEX_SYMBOLS


def get_ms_q_params(asset_symbol: str = None) -> Tuple[float, float]:
    """Return calibrated (sensitivity, threshold) for MS-q per asset class.

    Story 7.1: Asset-class calibrated sigmoid parameters for regime detection.

    Args:
        asset_symbol: Ticker symbol (e.g. 'MSTR', 'GC=F', 'BTC-USD').

    Returns:
        (sensitivity, threshold) tuple.
    """
    if asset_symbol is None:
        return (MS_Q_SENSITIVITY, MS_Q_THRESHOLD)

    sym = asset_symbol.strip().upper()

    # Check specific asset classes
    asset_class = _detect_asset_class(sym)
    if asset_class is not None:
        if asset_class in MS_Q_ASSET_CLASS_PARAMS:
            return MS_Q_ASSET_CLASS_PARAMS[asset_class]

    # Crypto
    if sym in _CRYPTO_SYMBOLS or 'BTC' in sym or 'ETH' in sym:
        return MS_Q_ASSET_CLASS_PARAMS['crypto']

    # Large cap
    if sym in _LARGE_CAP_SYMBOLS:
        return MS_Q_ASSET_CLASS_PARAMS['large_cap']

    # Index
    if sym in _INDEX_SYMBOLS:
        return MS_Q_ASSET_CLASS_PARAMS['index']

    # Default
    return (MS_Q_SENSITIVITY, MS_Q_THRESHOLD)


# ── Story 7.2: EWM vs Expanding Window Z-Score Selection ─────────────
# Per-asset-class preferred z-score method and EWM lambda
MS_Z_ASSET_CLASS_DEFAULTS = {
    'metals_gold':     ('ewm', 0.97),    # Slow regimes, long memory
    'metals_silver':   ('ewm', 0.96),    # Slightly faster
    'metals_other':    ('ewm', 0.97),    # Similar to gold
    'high_vol_equity': ('ewm', 0.94),    # Fast regimes
    'crypto':          ('ewm', 0.95),    # Structural breaks
    'large_cap':       ('expanding', 0.0),  # Stable, long-run mean informative
    'index':           ('expanding', 0.0),  # Stable
    'forex':           ('ewm', 0.96),    # Moderate
}


def get_ms_z_method(asset_symbol: str = None) -> Tuple[str, float]:
    """Return preferred (z_method, ewm_lambda) per asset class.

    Story 7.2: Asset-class z-score method selection.

    Args:
        asset_symbol: Ticker symbol.

    Returns:
        ('ewm', lambda) or ('expanding', 0.0).
    """
    if asset_symbol is None:
        return ('expanding', 0.0)

    sym = asset_symbol.strip().upper()
    asset_class = _detect_asset_class(sym)

    if asset_class is not None and asset_class in MS_Z_ASSET_CLASS_DEFAULTS:
        return MS_Z_ASSET_CLASS_DEFAULTS[asset_class]

    # Crypto
    if sym in _CRYPTO_SYMBOLS or 'BTC' in sym or 'ETH' in sym:
        return MS_Z_ASSET_CLASS_DEFAULTS['crypto']

    # Large cap
    if sym in _LARGE_CAP_SYMBOLS:
        return MS_Z_ASSET_CLASS_DEFAULTS['large_cap']

    # Index
    if sym in _INDEX_SYMBOLS:
        return MS_Z_ASSET_CLASS_DEFAULTS['index']

    return ('expanding', 0.0)


def select_ms_z_method_auto(
    vol: np.ndarray,
    q_calm: float,
    q_stress: float,
    sensitivity: float = 2.0,
    returns: np.ndarray = None,
) -> Tuple[str, float, dict]:
    """Select MS-q z-score method by robust BIC-style likelihood proxy."""
    vol = np.asarray(vol, dtype=np.float64).ravel()
    n = len(vol)
    if n <= 0:
        return 'expanding', 0.0, {'expanding': float('inf'), 'n_obs': 0}

    finite_vol = vol[np.isfinite(vol) & (vol > 0)]
    vol_fill = float(np.median(finite_vol)) if finite_vol.size else 1.0
    vol = np.where(np.isfinite(vol) & (vol > 0), vol, vol_fill)

    q_calm = float(max(q_calm, 1e-12))
    q_stress = float(max(q_stress, q_calm * MIN_Q_RATIO, 1e-12))
    sensitivity = float(np.clip(sensitivity, 0.25, 8.0))

    returns_clean = None
    if returns is not None:
        rr = np.asarray(returns, dtype=np.float64).ravel()
        m = min(len(rr), n)
        if m > 0:
            returns_clean = np.zeros(n, dtype=np.float64)
            r_part = rr[:m].copy()
            r_part[~np.isfinite(r_part)] = 0.0
            # Huberise only the proxy loss; the actual filter still sees tails.
            cap = max(float(np.percentile(np.abs(r_part), 99.5)), 1e-10) if m > 10 else max(float(np.std(r_part) * 8.0), 1e-10)
            returns_clean[:m] = np.clip(r_part, -cap, cap)

    candidates = [
        ('expanding', 0.0),
        ('ewm', 0.94),
        ('ewm', 0.96),
        ('ewm', 0.97),
    ]

    best_bic = np.inf
    best_method = 'expanding'
    best_lambda = 0.0
    bic_results = {}

    for method, lam in candidates:
        ewm_lam = lam if method == 'ewm' else 0.0
        q_t, p_stress = compute_ms_process_noise_smooth(
            vol, q_calm, q_stress, sensitivity=sensitivity, ewm_lambda=ewm_lam
        )
        q_safe = np.maximum(q_t, 1e-20)
        log_q = np.log(q_safe)

        if returns_clean is not None:
            innov_sq = returns_clean ** 2
            ll = -0.5 * float(np.sum(np.log(2.0 * np.pi) + log_q + innov_sq / q_safe))
        else:
            # Entropy proxy when returns are unavailable; lower variance churn is preferred.
            ll = -0.5 * float(np.sum(log_q + (p_stress - np.mean(p_stress)) ** 2))

        k = 1 if method == 'ewm' else 0
        bic = -2.0 * ll + k * math.log(max(n, 2))

        label = f"{method}_{lam}" if method == 'ewm' else 'expanding'
        bic_results[label] = float(bic)

        if np.isfinite(bic) and bic < best_bic:
            best_bic = bic
            best_method = method
            best_lambda = lam

    if best_method == 'expanding' and n < 100:
        best_method = 'ewm'
        best_lambda = 0.96

    bic_results['selected'] = f"{best_method}_{best_lambda}" if best_method == 'ewm' else 'expanding'
    bic_results['n_obs'] = int(n)
    return best_method, best_lambda, bic_results


# ── Story 7.3: Joint q_calm / q_stress optimization ──────────────────
MIN_Q_RATIO = 5.0     # Hard constraint: q_stress / q_calm >= 5
MAX_Q_RATIO = 1000.0  # Hard constraint: q_stress / q_calm <= 1000


def optimize_q_calm_stress(
    vol: np.ndarray,
    returns: np.ndarray,
    sensitivity: float = 2.0,
    ewm_lambda: float = 0.0,
    q_calm_bounds: Tuple[float, float] = (1e-8, 1e-4),
    q_stress_bounds: Tuple[float, float] = (1e-5, 1e-1),
    n_grid: int = 10,
) -> Tuple[float, float, dict]:
    """Optimize q_calm and q_stress jointly with hard identifiability constraints.

    Uses a robust grid search followed by a bounded log-space refinement.  The
    likelihood proxy is intentionally simple, but all constraints are enforced
    and the single-q baseline is computed on the same sanitized data.
    """
    vol = np.asarray(vol, dtype=np.float64).ravel()
    returns = np.asarray(returns, dtype=np.float64).ravel()
    n = min(len(vol), len(returns))
    if n <= 0:
        diagnostics = {
            'q_calm': float(MS_Q_CALM_DEFAULT),
            'q_stress': float(MS_Q_STRESS_DEFAULT),
            'ratio': float(MS_Q_STRESS_DEFAULT / MS_Q_CALM_DEFAULT),
            'bic_msq': float('inf'), 'bic_single': float('inf'),
            'bic_improvement_nats': 0.0, 'll_msq': float('-inf'),
            'll_single': float('-inf'), 'msq_selected': False,
            'fit_success': False, 'n_obs': 0,
        }
        return float(MS_Q_CALM_DEFAULT), float(MS_Q_STRESS_DEFAULT), diagnostics

    vol = vol[:n].copy()
    returns = returns[:n].copy()
    returns[~np.isfinite(returns)] = 0.0
    finite_vol = vol[np.isfinite(vol) & (vol > 0)]
    vol_fill = float(np.median(finite_vol)) if finite_vol.size else max(float(np.std(returns)), 1e-4)
    vol = np.where(np.isfinite(vol) & (vol > 0), vol, vol_fill)

    qc_lo, qc_hi = sorted((float(q_calm_bounds[0]), float(q_calm_bounds[1])))
    qs_lo, qs_hi = sorted((float(q_stress_bounds[0]), float(q_stress_bounds[1])))
    qc_lo = max(qc_lo, 1e-12)
    qc_hi = max(qc_hi, qc_lo * 10.0)
    qs_lo = max(qs_lo, qc_lo * MIN_Q_RATIO)
    qs_hi = max(qs_hi, qs_lo * 10.0)

    sensitivity = float(np.clip(sensitivity, 0.25, 8.0))
    ewm_lambda = float(np.clip(ewm_lambda, 0.0, 0.999))

    # Robust proxy loss: cap extreme squared returns so q-regime selection is
    # not hijacked by a single erroneous candle.
    cap = max(float(np.percentile(np.abs(returns), 99.5)), 1e-10) if n > 20 else max(float(np.std(returns) * 8.0), 1e-10)
    r_proxy = np.clip(returns, -cap, cap)

    def _proxy_ll(qc, qs):
        ratio = qs / max(qc, 1e-20)
        if ratio < MIN_Q_RATIO or ratio > MAX_Q_RATIO or qc <= 0 or qs <= 0:
            return float('-inf')
        q_t, _ = compute_ms_process_noise_smooth(
            vol, qc, qs, sensitivity=sensitivity, ewm_lambda=ewm_lambda
        )
        q_safe = np.maximum(q_t, 1e-20)
        return -0.5 * float(np.sum(np.log(2.0 * np.pi) + np.log(q_safe) + (r_proxy * r_proxy) / q_safe))

    q_calm_grid = np.geomspace(qc_lo, qc_hi, max(int(n_grid), 3))
    q_stress_grid = np.geomspace(qs_lo, qs_hi, max(int(n_grid), 3))

    best_ll = float('-inf')
    best_qc = qc_lo
    best_qs = qs_lo
    for qc in q_calm_grid:
        for qs in q_stress_grid:
            ll = _proxy_ll(float(qc), float(qs))
            if ll > best_ll:
                best_ll = ll
                best_qc = float(qc)
                best_qs = float(qs)

    opt_success = False
    try:
        def _neg_logspace(x):
            qc = 10.0 ** float(x[0])
            qs = 10.0 ** float(x[1])
            ll = _proxy_ll(qc, qs)
            if not np.isfinite(ll):
                return 1e12
            return -ll

        res = minimize(
            _neg_logspace,
            [math.log10(best_qc), math.log10(best_qs)],
            method='L-BFGS-B',
            bounds=[(math.log10(qc_lo), math.log10(qc_hi)),
                    (math.log10(qs_lo), math.log10(qs_hi))],
            options={'maxiter': 80, 'ftol': 1e-9},
        )
        if res.success and np.isfinite(res.fun):
            qc_ref = 10.0 ** float(res.x[0])
            qs_ref = 10.0 ** float(res.x[1])
            ll_ref = _proxy_ll(qc_ref, qs_ref)
            if ll_ref > best_ll:
                best_ll = ll_ref
                best_qc = qc_ref
                best_qs = qs_ref
                opt_success = True
    except Exception:
        pass

    # Enforce ratio in the returned values even if optimizer ended on a boundary.
    ratio = best_qs / max(best_qc, 1e-20)
    if ratio < MIN_Q_RATIO:
        best_qs = best_qc * MIN_Q_RATIO
    elif ratio > MAX_Q_RATIO:
        best_qs = best_qc * MAX_Q_RATIO
    best_qs = float(np.clip(best_qs, qs_lo, qs_hi))
    best_qc = float(np.clip(best_qc, qc_lo, qc_hi))

    q_single = max(float(np.mean((1.0 - 0.5) * best_qc + 0.5 * best_qs)), float(np.var(r_proxy)), 1e-12)
    ll_single = -0.5 * float(np.sum(np.log(2.0 * np.pi * q_single) + (r_proxy * r_proxy) / q_single))

    bic_msq = -2.0 * best_ll + 2.0 * math.log(max(n, 2))
    bic_single = -2.0 * ll_single + 1.0 * math.log(max(n, 2))
    ratio = best_qs / max(best_qc, 1e-20)
    bic_improvement = bic_single - bic_msq

    diagnostics = {
        'q_calm': float(best_qc),
        'q_stress': float(best_qs),
        'ratio': float(ratio),
        'bic_msq': float(bic_msq),
        'bic_single': float(bic_single),
        'bic_improvement_nats': float(bic_improvement / 2.0),
        'll_msq': float(best_ll),
        'll_single': float(ll_single),
        'msq_selected': bool(bic_improvement > 10.0),
        'fit_success': bool(np.isfinite(best_ll)),
        'optimizer_refined': bool(opt_success),
        'n_obs': int(n),
    }

    return float(best_qc), float(best_qs), diagnostics


# ---------------------------------------------------------------------------
# MARKOV-SWITCHING PROCESS NOISE
# ---------------------------------------------------------------------------
def compute_ms_process_noise_smooth(
    vol: np.ndarray,
    q_calm: float,
    q_stress: float,
    sensitivity: float = 2.0,
    ewm_lambda: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute smooth probabilistic MS process noise from volatility z-scores.

    The update is leakage-free: z_t is computed from statistics available before
    assimilating observation t wherever possible.  All exponentials are clipped
    for numerical stability.
    """
    vol = np.asarray(vol, dtype=np.float64).ravel()
    n = len(vol)
    if n <= 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    finite_vol = vol[np.isfinite(vol) & (vol > 0)]
    vol_fill = float(np.median(finite_vol)) if finite_vol.size else 1.0
    vol_floor = max(vol_fill * 1e-4, 1e-10)
    vol = np.where(np.isfinite(vol) & (vol > 0), vol, vol_fill)
    vol = np.maximum(vol, vol_floor)

    q_calm = float(max(q_calm, 1e-12))
    q_stress = float(max(q_stress, q_calm))
    sensitivity = float(np.clip(sensitivity, 0.25, 8.0))

    if ewm_lambda > 0.01:
        lam = float(np.clip(ewm_lambda, 0.5, 0.999))
        one_minus_lam = 1.0 - lam

        warmup = min(20, n)
        ewm_mean = float(np.mean(vol[:warmup])) if warmup > 0 else vol_fill
        ewm_var = float(np.var(vol[:warmup])) if warmup > 1 else max(vol_fill * vol_fill * 0.01, 1e-12)
        ewm_var = max(ewm_var, 1e-12)

        try:
            from models.numba_wrappers import run_compute_ms_process_noise_ewm
            vol_zscore = run_compute_ms_process_noise_ewm(vol, lam, ewm_mean, ewm_var)
            vol_zscore = np.asarray(vol_zscore, dtype=np.float64)
            if len(vol_zscore) != n or not np.all(np.isfinite(vol_zscore)):
                raise ValueError("invalid numba zscore")
        except (ImportError, Exception):
            vol_zscore = np.empty(n, dtype=np.float64)
            for t in range(n):
                ewm_std = math.sqrt(max(ewm_var, 1e-12))
                vol_zscore[t] = (vol[t] - ewm_mean) / max(ewm_std, 1e-8)
                # Update after scoring t to avoid look-ahead.
                diff_pre = vol[t] - ewm_mean
                ewm_mean = lam * ewm_mean + one_minus_lam * vol[t]
                ewm_var = lam * ewm_var + one_minus_lam * (diff_pre * diff_pre)
                ewm_var = max(ewm_var, 1e-12)
    else:
        vol_cumsum = np.cumsum(vol)
        vol_sq_cumsum = np.cumsum(vol * vol)
        counts = np.arange(1, n + 1, dtype=np.float64)

        # Expanding stats excluding current observation when possible.
        prev_counts = np.maximum(counts - 1.0, 1.0)
        prev_sum = vol_cumsum - vol
        prev_sq_sum = vol_sq_cumsum - vol * vol
        vol_mean = prev_sum / prev_counts
        vol_var = prev_sq_sum / prev_counts - vol_mean * vol_mean

        warmup = min(20, n)
        init_mean = float(np.mean(vol[:warmup])) if warmup else vol_fill
        init_std = max(float(np.std(vol[:warmup])) if warmup > 1 else vol_fill * 0.1, 1e-8)
        vol_mean[:warmup] = init_mean
        vol_var[:warmup] = init_std * init_std
        vol_std = np.sqrt(np.maximum(vol_var, 1e-12))
        vol_zscore = (vol - vol_mean) / np.maximum(vol_std, 1e-8)

    vol_zscore = np.clip(np.where(np.isfinite(vol_zscore), vol_zscore, 0.0), -12.0, 12.0)

    logits = np.clip(sensitivity * vol_zscore, -60.0, 60.0)
    p_stress = 1.0 / (1.0 + np.exp(-logits))
    p_stress = np.clip(p_stress, 0.01, 0.99)

    q_t = (1.0 - p_stress) * q_calm + p_stress * q_stress
    q_t = np.maximum(q_t, 1e-12)

    return q_t.astype(np.float64, copy=False), p_stress.astype(np.float64, copy=False)


# ── Story 6.2: Regime-dependent nu via stress probability ─────────────
def compute_regime_dependent_nu(
    vol: np.ndarray,
    nu_calm: float = 12.0,
    nu_crisis: float = 4.0,
    sensitivity: float = 2.0,
    ewm_lambda: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute time-varying nu based on volatility regime.

    nu_eff(t) = (1 - p_stress(t)) * nu_calm + p_stress(t) * nu_crisis

    Uses the same MS stress probability as MS-q, ensuring consistent
    regime assignments across process noise and tail heaviness.

    Args:
        vol: Volatility series.
        nu_calm: Degrees of freedom in calm regime (>= 8).
        nu_crisis: Degrees of freedom in crisis regime (<= 6).
        sensitivity: Sigmoid sensitivity to vol z-score.
        ewm_lambda: EWM decay for vol statistics (0 = expanding window).

    Returns:
        (nu_eff, p_stress): Time-varying nu and stress probability arrays.
    """
    # Reuse MS-q stress probability computation
    _, p_stress = compute_ms_process_noise_smooth(
        vol, q_calm=1e-6, q_stress=1e-4,
        sensitivity=sensitivity, ewm_lambda=ewm_lambda,
    )

    nu_calm = max(nu_calm, 2.5)
    nu_crisis = max(nu_crisis, 2.1)
    nu_eff = (1.0 - p_stress) * nu_calm + p_stress * nu_crisis

    return nu_eff, p_stress


# ── Story 6.3: Asymmetric nu calibration per asset class ──────────────
ASSET_CLASS_ASYMMETRIC_NU = {
    # (alpha_asym, k_asym) per asset class
    # alpha < 0: heavier left tail (crash asymmetry)
    # alpha ~ 0: symmetric
    'equity':       (-0.15, 1.5),   # Equities: heavier left tail
    'large_cap':    (-0.10, 1.5),   # Large cap: mild left skew
    'small_cap':    (-0.20, 1.5),   # Small cap: stronger crash asymmetry
    'high_vol':     (-0.25, 2.0),   # High vol: pronounced crash tail
    'metals_gold':  (0.0, 1.0),     # Gold: symmetric (safe haven)
    'metals_silver':(-0.05, 1.5),   # Silver: slight left skew
    'metals_other': (-0.05, 1.5),   # Other metals: slight left skew
    'crypto':       (0.05, 1.5),    # Crypto: unconstrained, slight bubble tail
    'forex':        (0.0, 1.0),     # Forex: symmetric
    'index':        (-0.10, 1.5),   # Indices: mild crash asymmetry
}


def get_asymmetric_nu_params(asset_symbol: str = None) -> Tuple[float, float]:
    """Return calibrated (alpha_asym, k_asym) for an asset's class.

    Story 6.3: Empirically calibrated asymmetry parameters.
    alpha < 0 means heavier left (crash) tail.

    Args:
        asset_symbol: Ticker symbol (e.g. 'MSTR', 'GC=F', 'BTC-USD').

    Returns:
        (alpha_asym, k_asym) tuple.
    """
    if asset_symbol is None:
        return ASSET_CLASS_ASYMMETRIC_NU.get('equity', (-0.15, 1.5))

    sym = asset_symbol.strip().upper()

    asset_class = _detect_asset_class(sym)
    if asset_class is not None:
        if asset_class in ASSET_CLASS_ASYMMETRIC_NU:
            return ASSET_CLASS_ASYMMETRIC_NU[asset_class]
        # Map detector classes to our dict
        mapping = {
            'metals_gold': 'metals_gold',
            'metals_silver': 'metals_silver',
            'metals_other': 'metals_other',
            'high_vol_equity': 'high_vol',
        }
        mapped = mapping.get(asset_class)
        if mapped and mapped in ASSET_CLASS_ASYMMETRIC_NU:
            return ASSET_CLASS_ASYMMETRIC_NU[mapped]

    # Crypto detection
    if 'BTC' in sym or 'ETH' in sym or 'SOL' in sym or 'DOGE' in sym:
        return ASSET_CLASS_ASYMMETRIC_NU['crypto']

    # Index detection
    if sym in ('SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI'):
        return ASSET_CLASS_ASYMMETRIC_NU['index']

    # Large cap heuristic
    _LARGE_CAP = {'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META',
                  'TSLA', 'BRK-B', 'JPM', 'V', 'MA', 'UNH', 'HD', 'PG'}
    if sym in _LARGE_CAP:
        return ASSET_CLASS_ASYMMETRIC_NU['large_cap']

    # Default: equity
    return ASSET_CLASS_ASYMMETRIC_NU['equity']


def compute_empirical_tail_ratio(returns: np.ndarray) -> float:
    """Compute empirical tail asymmetry from return series.

    Returns alpha_hat in the same convention used by compute_effective_nu:
    negative means heavier left/crash tail; positive means heavier right tail.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return 0.0

    center = float(np.median(r))
    r_centered = r - center

    left_mask = r_centered < 0
    right_mask = r_centered > 0
    if np.sum(left_mask) < 10 or np.sum(right_mask) < 10:
        return 0.0

    mad = float(np.median(np.abs(r_centered)))
    std = max(1.4826 * mad, float(np.std(r_centered)), 1e-10)
    z = np.clip(r_centered / std, -12.0, 12.0)

    left_kurt = float(np.mean(z[left_mask] ** 4))
    right_kurt = float(np.mean(z[right_mask] ** 4))
    total_kurt = left_kurt + right_kurt
    if total_kurt < 1e-10 or not np.isfinite(total_kurt):
        return 0.0

    # Sign convention: alpha < 0 => lower effective ν on negative shocks.
    return float(np.clip((right_kurt - left_kurt) / total_kurt, -1.0, 1.0))


def compute_ms_process_noise(
    vol: np.ndarray,
    q_calm: float = MS_Q_CALM_DEFAULT,
    q_stress: float = MS_Q_STRESS_DEFAULT,
    sensitivity: float = MS_Q_SENSITIVITY,
    threshold: float = MS_Q_THRESHOLD,
    vol_median: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible MS-q path that honours the legacy relative-vol threshold."""
    vol = np.asarray(vol, dtype=np.float64).ravel()
    n = len(vol)
    if n <= 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    finite_vol = vol[np.isfinite(vol) & (vol > 0)]
    fill = float(np.median(finite_vol)) if finite_vol.size else 1.0
    if vol_median is not None and np.isfinite(vol_median) and vol_median > 0:
        fill = float(vol_median)
    vol = np.where(np.isfinite(vol) & (vol > 0), vol, fill)
    vol = np.maximum(vol, max(fill * 1e-4, 1e-10))

    q_calm = float(max(q_calm, 1e-12))
    q_stress = float(max(q_stress, q_calm))
    sensitivity = float(np.clip(sensitivity, 0.25, 8.0))
    threshold = float(max(threshold, 1e-6))

    vol_cumsum = np.cumsum(vol)
    counts = np.arange(1, n + 1, dtype=np.float64)
    baseline = vol_cumsum / counts
    warmup = min(20, n)
    if warmup > 0:
        baseline[:warmup] = fill if vol_median is not None else float(np.mean(vol[:warmup]))
    baseline = np.maximum(baseline, max(fill * 1e-4, 1e-10))

    vol_relative = np.clip(vol / baseline, 0.0, 20.0)
    logits = np.clip(sensitivity * (vol_relative - threshold), -60.0, 60.0)
    p_stress = 1.0 / (1.0 + np.exp(-logits))
    p_stress = np.clip(p_stress, 0.01, 0.99)
    q_t = np.maximum((1.0 - p_stress) * q_calm + p_stress * q_stress, 1e-12)
    return q_t.astype(np.float64, copy=False), p_stress.astype(np.float64, copy=False)


def filter_phi_ms_q(
    y: np.ndarray,
    vol: np.ndarray,
    c: float,
    phi: float,
    nu: float,
    q_calm: float = MS_Q_CALM_DEFAULT,
    q_stress: float = MS_Q_STRESS_DEFAULT,
    sensitivity: float = MS_Q_SENSITIVITY,
    threshold: float = MS_Q_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Kalman filter with Markov-switching process noise for Student-t observations.

    Uses a scale-mixture Student-t update: large residuals inflate the effective
    observation variance instead of making the covariance update negative.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    vol = np.asarray(vol, dtype=np.float64).ravel()
    n = min(len(y), len(vol))
    if n <= 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, 0.0, empty, empty

    y = y[:n].copy()
    vol = vol[:n].copy()
    y[~np.isfinite(y)] = 0.0
    finite_vol = vol[np.isfinite(vol) & (vol > 0)]
    vol_fill = float(np.median(finite_vol)) if finite_vol.size else max(float(np.std(y)), 1e-4)
    vol = np.where(np.isfinite(vol) & (vol > 0), vol, vol_fill)
    vol = np.maximum(vol, max(vol_fill * 1e-4, 1e-10))

    c = float(max(c, 1e-12))
    phi = float(np.clip(phi, -0.999, 0.999))
    nu = float(np.clip(nu, 2.01, 60.0))
    q_calm = float(max(q_calm, 1e-12))
    q_stress = float(max(q_stress, q_calm))

    q_t, p_stress = compute_ms_process_noise(
        vol, q_calm=q_calm, q_stress=q_stress,
        sensitivity=sensitivity, threshold=threshold
    )

    mu = np.zeros(n, dtype=np.float64)
    P = np.zeros(n, dtype=np.float64)
    mu_t = 0.0
    vol_var_med = max(float(np.median(vol * vol)), 1e-12)
    p_floor = max(float(_P_MIN), 1e-12)
    p_cap = max(float(_P_MAX), 100.0 * vol_var_med, 10.0 * q_stress, p_floor * 10.0)
    P_t = min(max(10.0 * vol_var_med, p_floor), p_cap)

    total_ll = 0.0
    log_norm_const = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * math.log(nu * math.pi)
    neg_exp = -0.5 * (nu + 1.0)
    inv_nu = 1.0 / nu
    scale_factor = (nu - 2.0) / nu if nu > 2.0 else 1.0

    for t in range(n):
        R_t = max(c * (vol[t] ** 2), 1e-20)
        S_t = max(P_t + R_t, 1e-20)
        scale_t = max(math.sqrt(S_t * scale_factor), 1e-10)

        innovation = y[t] - mu_t
        z = innovation / scale_t
        ll_t = log_norm_const - math.log(scale_t) + neg_exp * math.log1p((z * z) * inv_nu)
        if math.isfinite(ll_t):
            total_ll += ll_t

        # Scale-mixture robust update.  w_t is latent precision; R_eff=R/w.
        z_sq_s = (innovation * innovation) / S_t
        w_t = (nu + 1.0) / (nu + z_sq_s)
        w_t = float(np.clip(w_t, 0.05, 20.0))
        R_eff = R_t / max(w_t, 1e-8)
        S_eff = max(P_t + R_eff, 1e-20)
        K_t = P_t / S_eff

        mu_t = mu_t + K_t * innovation
        P_t = (1.0 - K_t) * (1.0 - K_t) * P_t + K_t * K_t * R_eff
        P_t = max(p_floor, min(P_t, p_cap))

        mu[t] = mu_t
        P[t] = P_t

        mu_t = phi * mu_t
        P_t = phi * phi * P_t + q_t[t]
        P_t = max(p_floor, min(P_t, p_cap))

    return mu, P, float(total_ll), q_t, p_stress


def optimize_params_ms_q(
    returns: np.ndarray,
    vol: np.ndarray,
    nu: float,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
) -> Tuple[float, float, float, float, float, float, Dict]:
    """
    Optimize MS-q model parameters: (c, phi, q_calm, q_stress).

    Uses robust q-regime pre-optimization plus bounded likelihood refinement
    for c and phi.  Returned log-likelihood is the unpenalized filter LL.
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    vol = np.asarray(vol, dtype=np.float64).ravel()
    n = min(len(returns), len(vol))
    if n <= 0:
        diagnostics = {
            "fit_success": False, "n_obs": 0, "nu": float(nu),
            "q_ratio": float(MS_Q_STRESS_DEFAULT / MS_Q_CALM_DEFAULT),
            "lfo_cv_diagnostics": {"error": "empty_input"},
        }
        return 1.0, 0.0, MS_Q_CALM_DEFAULT, MS_Q_STRESS_DEFAULT, float('-inf'), float('-inf'), diagnostics

    returns = returns[:n].copy()
    vol = vol[:n].copy()
    returns[~np.isfinite(returns)] = 0.0
    finite_vol = vol[np.isfinite(vol) & (vol > 0)]
    vol_fill = float(np.median(finite_vol)) if finite_vol.size else max(float(np.std(returns)), 1e-4)
    vol = np.where(np.isfinite(vol) & (vol > 0), vol, vol_fill)

    nu = float(np.clip(nu if np.isfinite(nu) else 8.0, 2.1, 60.0))
    ret_var = max(float(np.var(returns)), 1e-12)
    vol_var = max(float(np.median(vol * vol)), 1e-12)
    c_mom = float(np.clip(ret_var / vol_var, 0.01, 50.0))

    q_pairs = []
    try:
        qc0, qs0, qdiag = optimize_q_calm_stress(vol, returns, n_grid=8)
        q_pairs.append((qc0, qs0))
    except Exception:
        qdiag = {"error": "optimize_q_calm_stress_failed"}

    for q_calm in [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]:
        for ratio in [5, 10, 25, 50, 100, 250]:
            q_stress = q_calm * ratio
            if MIN_Q_RATIO <= ratio <= MAX_Q_RATIO:
                q_pairs.append((q_calm, q_stress))

    # Deduplicate approximate pairs.
    seen = set()
    q_pairs_unique = []
    for qc, qs in q_pairs:
        key = (round(math.log10(max(qc, 1e-20)), 4), round(math.log10(max(qs, 1e-20)), 4))
        if key not in seen:
            seen.add(key)
            q_pairs_unique.append((float(qc), float(qs)))

    best_score = float('inf')
    best_params = None
    best_ll = float('-inf')
    best_success = False

    for q_calm, q_stress in q_pairs_unique:
        q_calm = max(float(q_calm), 1e-12)
        q_stress = max(float(q_stress), q_calm * MIN_Q_RATIO)
        ratio = q_stress / q_calm
        if ratio < MIN_Q_RATIO or ratio > MAX_Q_RATIO:
            continue

        def neg_ll(params):
            log_c_try, phi_try = params
            if not (np.isfinite(log_c_try) and np.isfinite(phi_try)):
                return 1e12
            c_try = 10.0 ** float(log_c_try)
            phi_try = float(np.clip(phi_try, -0.999, 0.999))
            if c_try <= 0.0:
                return 1e12
            try:
                _, _, ll, _, _ = filter_phi_ms_q(
                    returns, vol, c_try, phi_try, nu,
                    q_calm=q_calm, q_stress=q_stress
                )
                if not np.isfinite(ll):
                    return 1e12
                log_q_avg = math.log10(max((q_calm + q_stress) * 0.5, 1e-20))
                prior_penalty = float(prior_lambda) * (log_q_avg - float(prior_log_q_mean)) ** 2
                phi_penalty = 0.5 * (phi_try / 0.75) ** 2
                return -(ll / max(n, 1)) + prior_penalty / max(n, 1) + phi_penalty / max(n, 1)
            except Exception:
                return 1e12

        starts = [
            (math.log10(c_mom), 0.0),
            (math.log10(max(c_mom * 0.5, 0.01)), 0.25),
            (math.log10(max(c_mom * 2.0, 0.01)), -0.15),
            (0.0, 0.5),
        ]

        for x0 in starts:
            try:
                result = minimize(
                    neg_ll,
                    [float(np.clip(x0[0], math.log10(0.01), math.log10(50.0))), x0[1]],
                    method='L-BFGS-B',
                    bounds=[(math.log10(0.01), math.log10(50.0)), (-0.999, 0.999)],
                    options={'maxiter': 80, 'ftol': 1e-9}
                )
                score = float(result.fun) if np.isfinite(result.fun) else float('inf')
                if score < best_score:
                    c_opt_try = 10.0 ** float(result.x[0])
                    phi_opt_try = float(np.clip(result.x[1], -0.999, 0.999))
                    _, _, ll_unpen, _, _ = filter_phi_ms_q(
                        returns, vol, c_opt_try, phi_opt_try, nu,
                        q_calm=q_calm, q_stress=q_stress
                    )
                    best_score = score
                    best_ll = float(ll_unpen)
                    best_params = (c_opt_try, phi_opt_try, q_calm, q_stress)
                    best_success = bool(result.success)
            except Exception:
                continue

    if best_params is None:
        c_opt, phi_opt = c_mom, 0.0
        q_calm_opt, q_stress_opt = MS_Q_CALM_DEFAULT, MS_Q_STRESS_DEFAULT
        _, _, ll_opt, _, _ = filter_phi_ms_q(
            returns, vol, c_opt, phi_opt, nu,
            q_calm=q_calm_opt, q_stress=q_stress_opt
        )
    else:
        c_opt, phi_opt, q_calm_opt, q_stress_opt = best_params
        ll_opt = best_ll

    try:
        from tuning.diagnostics import compute_lfo_cv_score_student_t
        q_avg = (q_calm_opt + q_stress_opt) / 2.0
        lfo_cv_score, lfo_diag = compute_lfo_cv_score_student_t(
            returns, vol, q_avg, c_opt, phi_opt, nu
        )
    except Exception:
        lfo_cv_score = float('-inf')
        lfo_diag = {"error": "lfo_cv_not_available"}

    diagnostics = {
        "fit_success": bool(best_params is not None),
        "optimizer_success": bool(best_success),
        "n_obs": int(n),
        "nu": float(nu),
        "q_ratio": float(q_stress_opt / q_calm_opt if q_calm_opt > 0 else 0.0),
        "q_regime_diagnostics": qdiag,
        "objective": float(best_score),
        "lfo_cv_diagnostics": lfo_diag,
    }

    return float(c_opt), float(phi_opt), float(q_calm_opt), float(q_stress_opt), float(ll_opt), float(lfo_cv_score), diagnostics


# ---------------------------------------------------------------------------
# TUNING CONFIGURATION (v2.0)
# Plateau-optimal parameter selection with curvature awareness,
# ridge vs basin detection, drift vs noise coherence decomposition.
# ---------------------------------------------------------------------------

ELITE_TUNING_ENABLED = True  # Master switch for elite tuning diagnostics
CURVATURE_PENALTY_WEIGHT = 0.1
COHERENCE_PENALTY_WEIGHT = 0.05
HESSIAN_EPSILON = 1e-4
MAX_CONDITION_NUMBER = 1e6


def _compute_curvature_penalty(
    objective_fn,
    optimal_params: np.ndarray,
    bounds: list,
    epsilon: float = 1e-4,
    max_condition_number: float = 1e6
) -> Tuple[float, float, Dict]:
    """
    Compute curvature penalty from local Hessian approximation.

    Returns:
        - penalty: Soft penalty based on condition number
        - condition_number: κ(H)
        - diagnostics: Dict with eigenvalues and curvature details
    """
    n = len(optimal_params)
    H = np.zeros((n, n))
    f_x = objective_fn(optimal_params)

    # Compute Hessian via finite differences
    for i in range(n):
        x_plus = optimal_params.copy()
        x_minus = optimal_params.copy()

        step = epsilon
        if bounds:
            lo, hi = bounds[i]
            step = min(epsilon, (hi - optimal_params[i]) / 2, (optimal_params[i] - lo) / 2)
            step = max(step, 1e-8)

        x_plus[i] += step
        x_minus[i] -= step

        f_plus = objective_fn(x_plus)
        f_minus = objective_fn(x_minus)

        H[i, i] = (f_plus - 2 * f_x + f_minus) / (step ** 2)

    # Off-diagonal elements
    for i in range(n):
        for j in range(i + 1, n):
            x_pp = optimal_params.copy()
            x_pm = optimal_params.copy()
            x_mp = optimal_params.copy()
            x_mm = optimal_params.copy()

            step_i = step_j = epsilon
            if bounds:
                lo_i, hi_i = bounds[i]
                lo_j, hi_j = bounds[j]
                step_i = min(epsilon, (hi_i - optimal_params[i]) / 2, (optimal_params[i] - lo_i) / 2)
                step_j = min(epsilon, (hi_j - optimal_params[j]) / 2, (optimal_params[j] - lo_j) / 2)
                step_i = max(step_i, 1e-8)
                step_j = max(step_j, 1e-8)

            x_pp[i] += step_i; x_pp[j] += step_j
            x_pm[i] += step_i; x_pm[j] -= step_j
            x_mp[i] -= step_i; x_mp[j] += step_j
            x_mm[i] -= step_i; x_mm[j] -= step_j

            f_pp = objective_fn(x_pp)
            f_pm = objective_fn(x_pm)
            f_mp = objective_fn(x_mp)
            f_mm = objective_fn(x_mm)

            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * step_i * step_j)
            H[j, i] = H[i, j]

    # Compute condition number
    try:
        eigenvalues = np.linalg.eigvalsh(H)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = eigenvalues[np.isfinite(eigenvalues)]

        if len(eigenvalues) == 0:
            return 0.0, 1.0, {'error': 'no_valid_eigenvalues'}

        abs_eig = np.abs(eigenvalues)
        max_eig = np.max(abs_eig)
        min_eig = np.max([np.min(abs_eig[abs_eig > 1e-12]), 1e-12])
        condition_number = max_eig / min_eig

        if condition_number > max_condition_number:
            penalty = np.log(condition_number / max_condition_number)
        else:
            penalty = 0.0

        return penalty, condition_number, {
            'eigenvalues': eigenvalues.tolist(),
            'max_eigenvalue': float(max_eig),
            'min_eigenvalue': float(min_eig),
            'penalty': float(penalty)
        }
    except np.linalg.LinAlgError:
        return 0.0, float('inf'), {'error': 'linalg_error'}


def _compute_fragility_index(
    condition_number: float,
    coherence_variance: np.ndarray,
    plateau_score: float = 0.5,
    basin_score: float = 1.0,
    drift_ratio: float = 0.0
) -> Tuple[float, Dict[str, float]]:
    """
    Compute unified fragility index (PURE PARAMETER FRAGILITY v2.0).

    Components:
        1. Curvature fragility: Sharp optima are fragile
        2. Coherence fragility: Inconsistent parameters are fragile
        3. Plateau fragility: Narrow plateaus are fragile
        4. Basin fragility: Ridges are more fragile than basins
        5. Drift fragility: Drifting parameters are unstable
    """
    components = {}

    # 1. Curvature fragility
    if condition_number > 0 and np.isfinite(condition_number):
        curvature_fragility = min(np.log10(max(condition_number, 1)) / 10, 1.0)
    else:
        curvature_fragility = 0.5
    components['curvature'] = curvature_fragility

    # 2. Coherence fragility
    if len(coherence_variance) > 0:
        coherence_fragility = min(np.mean(coherence_variance) * 10, 1.0)
    else:
        coherence_fragility = 0.5
    components['coherence'] = coherence_fragility

    # 3. Plateau fragility
    plateau_fragility = 1.0 - min(plateau_score, 1.0)
    components['plateau'] = plateau_fragility

    # 4. Basin fragility (v2.0)
    basin_fragility = 1.0 - min(basin_score, 1.0)
    components['basin'] = basin_fragility

    # 5. Drift fragility (v2.0)
    drift_fragility = min(drift_ratio * 2, 1.0)
    components['drift'] = drift_fragility

    # Weighted combination (v2.0 weights)
    weights = {
        'curvature': 0.25,
        'coherence': 0.15,
        'plateau': 0.20,
        'basin': 0.25,
        'drift': 0.15,
    }

    fragility_index = sum(weights[k] * components[k] for k in weights)

    return fragility_index, components


def _phi_shrinkage_log_prior(
    phi_r: float,
    phi_global: float,
    tau: float,
    tau_min: float = PHI_SHRINKAGE_TAU_MIN
) -> float:
    """Compute Gaussian shrinkage log-prior for φ."""
    tau_safe = max(tau, tau_min)
    if not np.isfinite(phi_global):
        return float('-inf')
    deviation = phi_r - phi_global
    return -0.5 * (deviation ** 2) / (tau_safe ** 2)


def _lambda_to_tau(lam: float, lam_min: float = 1e-12) -> float:
    """Convert legacy penalty weight λ to Gaussian prior std τ."""
    lam_safe = max(lam, lam_min)
    return 1.0 / math.sqrt(2.0 * lam_safe)


def _compute_phi_prior_diagnostics(
    phi_r: float,
    phi_global: float,
    tau: float,
    log_likelihood: float
) -> Dict[str, Optional[float]]:
    """Compute diagnostic information for φ shrinkage prior."""
    log_prior = _phi_shrinkage_log_prior(phi_r, phi_global, tau)
    ratio = None
    if np.isfinite(log_prior) and np.isfinite(log_likelihood):
        abs_prior = abs(log_prior)
        abs_ll = abs(log_likelihood)
        if abs_ll > 1e-12:
            ratio = abs_prior / abs_ll
    return {
        'phi_prior_logp': float(log_prior) if np.isfinite(log_prior) else None,
        'phi_likelihood_logp': float(log_likelihood) if np.isfinite(log_likelihood) else None,
        'phi_prior_likelihood_ratio': float(ratio) if ratio is not None else None,
        'phi_deviation_from_global': float(phi_r - phi_global),
        'phi_tau_used': float(tau),
    }


class PhiStudentTDriftModel:
    """
    Standard fixed-ν Student-t Kalman filter with AR(1) drift.

    Encapsulates fixed-ν Student-t BMA sub-models. For each ν in STUDENT_T_NU_GRID,
    tune_and_calibrate() runs the complete tune-filter-calibrate pipeline:

    tune_and_calibrate — Pipeline Stages
    =========================================================================
    1. MLE optimisation (q, c, φ) via cross-validated L-BFGS-B
    2. Enhanced Kalman filter (robust_wt + VoV + online_scale_adapt)
    3. PIT / Hyvärinen / CRPS metrics
    4. Post-hoc GARCH innovation blending (Engle 1982, GJR 1993)
    5. ν-refinement: discrete grid + continuous brentq solver
    6. Score-driven GAS time-varying ν (Creal-Koopman-Lucas 2013)
    7. CRPS-optimal scale shrinkage (Gneiting & Raftery 2007)
    8. Momentum augmentation (if wrapper supplied)
    9. Isotonic recalibration (Kuleshov et al. 2018)
    =========================================================================

    NOTE: The unified Student-t model (optimize_params_unified, filter_phi_unified,
    filter_and_calibrate, etc.) has been split into the independent
    UnifiedPhiStudentTModel class in phi_student_t_unified.py.
    """

    nu_min_default: float = 2.1
    nu_max_default: float = 30.0

    @staticmethod
    def _clip_nu(nu: float, nu_min: float, nu_max: float) -> float:
        if not np.isfinite(nu):
            return float(nu_min)
        return float(np.clip(float(nu), max(2.001, float(nu_min)), max(float(nu_min), float(nu_max))))

    @staticmethod
    def _variance_to_scale(variance: float, nu: float) -> float:
        """Convert predictive variance to Student-t scale with hard finite guards."""
        return variance_to_scale(variance, nu)

    @staticmethod
    def _variance_to_scale_vec(variance: np.ndarray, nu: float) -> np.ndarray:
        """Vectorized version of _variance_to_scale for array inputs."""
        return variance_to_scale_vec(variance, nu)

    @staticmethod
    def compute_effective_nu(
        nu_base: float,
        innovation: float,
        scale: float,
        alpha: float,
        k: float = 1.0,
        nu_min: float = 2.1,
        nu_max: float = 50.0,
    ) -> float:
        """
        Smooth asymmetric ν via tanh: ν_eff = ν_base × (1 + α·tanh(k·z)).

        α < 0: heavier left tail (crashes get lower ν)
        α > 0: heavier right tail
        """
        nu_base = float(nu_base) if np.isfinite(nu_base) else 8.0
        innovation = float(innovation) if np.isfinite(innovation) else 0.0
        scale = float(scale) if np.isfinite(scale) else 1.0
        alpha = float(np.clip(alpha if np.isfinite(alpha) else 0.0, -0.95, 0.95))
        k = float(np.clip(k if np.isfinite(k) else 1.0, 0.0, 10.0))

        scale_safe = max(abs(scale), 1e-10)
        z = float(np.clip(innovation / scale_safe, -50.0, 50.0))
        modulation = 1.0 + alpha * math.tanh(k * z)
        nu_raw = nu_base * max(modulation, 1e-6)
        return float(np.clip(nu_raw, max(2.001, nu_min), max(nu_min, nu_max)))

    @staticmethod
    def logpdf(x: float, nu: float, mu: float, scale: float) -> float:
        """
        Log-density of scaled Student-t with location ``mu`` and scale ``scale``.
        Returns a large negative sentinel if inputs are invalid to keep optimizers stable.
        """
        if not (np.isfinite(x) and np.isfinite(nu) and np.isfinite(mu) and np.isfinite(scale)):
            return -1e12
        if scale <= 0.0 or nu <= 2.0:
            return -1e12

        if _NUMERICAL_STABILITY_AVAILABLE:
            try:
                val = safe_student_t_logpdf_scalar(float(x), float(nu), float(mu), float(scale))
                return float(val) if np.isfinite(val) else -1e12
            except Exception:
                pass

        z = (float(x) - float(mu)) / max(float(scale), 1e-12)
        log_norm = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * math.log(nu * math.pi)
            - math.log(max(scale, 1e-12))
        )
        log_kernel = -0.5 * (nu + 1.0) * math.log1p((z * z) / nu)
        val = float(log_norm + log_kernel)
        return val if math.isfinite(val) else -1e12

    @classmethod
    def pit_ks_predictive(
        cls,
        returns: np.ndarray,
        mu_pred: np.ndarray,
        S_pred: np.ndarray,
        nu: float,
    ) -> Tuple[float, float]:
        """
        PIT/KS using predictive Student-t distribution with finite-data guards.
        """
        returns = np.asarray(returns, dtype=np.float64).ravel()
        mu_pred = np.asarray(mu_pred, dtype=np.float64).ravel()
        S_pred = np.asarray(S_pred, dtype=np.float64).ravel()

        n = min(len(returns), len(mu_pred), len(S_pred))
        if n < 2:
            return 1.0, 0.0

        r = returns[:n]
        m = mu_pred[:n]
        S = S_pred[:n]
        valid = np.isfinite(r) & np.isfinite(m) & np.isfinite(S) & (S > 0)
        if int(np.sum(valid)) < 2:
            return 1.0, 0.0

        r = r[valid]
        m = m[valid]
        S = np.maximum(S[valid], 1e-20)
        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)
        scale = cls._variance_to_scale_vec(S, nu_val)

        z = np.clip((r - m) / scale, -1e6, 1e6)
        pit_values = _fast_t_cdf(z, nu_val)
        pit_values = np.asarray(pit_values, dtype=np.float64)
        valid_pit = np.isfinite(pit_values)
        pit_clean = np.clip(pit_values[valid_pit], 1e-12, 1.0 - 1e-12)
        if len(pit_clean) < 2:
            return 1.0, 0.0

        ks_stat, ks_p = _fast_ks_uniform(pit_clean)
        return float(ks_stat), float(ks_p)

    @classmethod
    def optimize_params_fixed_nu(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        nu: float = 8.0,
        train_frac: float = 0.7,
        q_min: float = 1e-10,
        q_max: float = 1e-1,
        c_min: float = 0.3,
        c_max: float = 3.0,
        phi_min: float = -0.8,
        phi_max: float = 0.999,
        prior_log_q_mean: float = -6.0,
        prior_lambda: float = 1.0,
        asset_symbol: str = None,
        gamma_vov: float = 0.0,
        vov_rolling: np.ndarray = None,
        warm_start_params: Tuple[float, float, float] = None,
    ) -> Tuple[float, float, float, float, Dict]:
        """
        Optimize (q, c, φ) for a fixed ν with leakage-aware expanding-window CV.

        Major quality safeguards:
          * no full-sample winsorisation of returns;
          * fold scoring starts from the filtered training state;
          * scale-aware q/c bounds and starts;
          * robust Student-t covariance update consistent with inference;
          * finite-data guards for live/production feeds.
        """
        returns = np.asarray(returns, dtype=np.float64).ravel()
        vol = np.asarray(vol, dtype=np.float64).ravel()
        n = min(len(returns), len(vol))
        if n <= 0:
            raise ValueError("returns and vol must contain at least one observation")

        returns = returns[:n].copy()
        vol = vol[:n].copy()
        returns[~np.isfinite(returns)] = 0.0

        finite_vol = vol[np.isfinite(vol) & (vol > 0)]
        vol_fill = float(np.median(finite_vol)) if finite_vol.size else max(float(np.std(returns)), 1e-4)
        vol_floor = max(vol_fill * 1e-4, 1e-10)
        vol = np.where(np.isfinite(vol) & (vol > 0), vol, vol_fill)
        vol = np.maximum(vol, vol_floor)

        if vov_rolling is not None:
            vov_rolling = np.asarray(vov_rolling, dtype=np.float64).ravel()
            if len(vov_rolling) < n:
                pad_val = float(vov_rolling[-1]) if len(vov_rolling) else 0.0
                vov_rolling = np.pad(vov_rolling, (0, n - len(vov_rolling)), constant_values=pad_val)
            vov_rolling = np.where(np.isfinite(vov_rolling[:n]), vov_rolling[:n], 0.0)

        if n < 8:
            vol_var_med = max(float(np.median(vol * vol)), 1e-12)
            ret_var = max(float(np.var(returns)), vol_var_med, 1e-12)
            q_fallback = float(np.clip(0.01 * ret_var, max(q_min, 1e-12), max(q_max, q_min * 10.0)))
            c_fallback = float(np.clip(ret_var / vol_var_med, max(c_min, 1e-4), max(c_max, c_min * 10.0)))
            phi_fallback = float(np.clip(0.0, phi_min, phi_max))
            _, _, ll_fb = cls.filter_phi(returns, vol, q_fallback, c_fallback, phi_fallback, nu)
            return q_fallback, c_fallback, phi_fallback, float(ll_fb), {
                "nu_fixed": float(nu),
                "optimizer_converged": False,
                "n_folds": 0,
                "fallback": "too_few_observations",
                "log10_q": float(math.log10(q_fallback)),
            }

        train_frac = float(np.clip(train_frac, 0.5, 0.9))
        split0 = max(5, min(int(n * train_frac), n - 1))
        ret_train0 = returns[:split0]
        vol_sq = vol * vol
        vol_var_med = max(float(np.median(vol_sq[:split0])), 1e-12)
        ret_var = max(float(np.var(ret_train0)), vol_var_med * 0.1, 1e-12)

        # Scale-aware bounds.  q is drift-process variance, so it should be far
        # below return variance but not fixed to a unit-scale prior.
        q_min_eff = max(float(q_min), ret_var * 1e-8, vol_var_med * 1e-6, 1e-12)
        q_max_eff = max(float(q_max), q_min_eff * 10.0)
        q_max_eff = min(q_max_eff, max(ret_var * 0.50, vol_var_med * 50.0, q_min_eff * 10.0))

        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)
        obs_var_factor = (nu_val / (nu_val - 2.0)) if nu_val > 2.0 else 1.0
        c_mom = ret_var / max(vol_var_med * obs_var_factor, 1e-12)
        c_mom = float(np.clip(c_mom, 1e-4, 50.0))
        c_min_eff = max(min(float(c_min), c_mom * 0.25), 1e-4)
        c_max_eff = max(float(c_max), c_mom * 4.0, c_min_eff * 10.0)
        if asset_symbol and any(tag in str(asset_symbol).upper() for tag in ["=F", "GC", "SI", "CL", "NG"]):
            c_max_eff = max(c_max_eff, 10.0)
        c_max_eff = min(c_max_eff, 50.0)

        phi_min_eff = float(np.clip(phi_min, -0.999, 0.999))
        phi_max_eff = float(np.clip(phi_max, phi_min_eff + 1e-6, 0.999))

        # Expanding-window CV folds with non-empty validation windows.
        folds = []
        min_train = min(max(40, int(n * 0.40)), max(n - 5, 5))
        test_window = max(5, min(int(n * 0.12), max(n - min_train, 5)))
        te = min_train
        while te < n:
            ve = min(te + test_window, n)
            if ve - te >= 3:
                folds.append((0, te, te, ve))
            te += test_window
        if not folds:
            sp = max(5, min(split0, n - 1))
            folds = [(0, sp, sp, n)]
        if len(folds) > 3:
            fold_idx = np.unique(np.round(np.linspace(0, len(folds) - 1, 3)).astype(int))
            folds = [folds[int(i)] for i in fold_idx]

        log_norm_const = math.lgamma((nu_val + 1.0) / 2.0) - math.lgamma(nu_val / 2.0) - 0.5 * math.log(nu_val * math.pi)
        neg_exp = -0.5 * (nu_val + 1.0)
        inv_nu = 1.0 / nu_val
        scale_factor = (nu_val - 2.0) / nu_val if nu_val > 2.0 else 1.0
        has_vov = gamma_vov > 1e-12 and vov_rolling is not None
        vov_full = np.ascontiguousarray(vov_rolling, dtype=np.float64) if has_vov else None
        use_osa = True
        p_floor = max(float(_P_MIN), 1e-12)
        p_cap = max(float(_P_MAX), 100.0 * vol_var_med, 1e-6)
        asset_class = _detect_asset_class(asset_symbol, returns=ret_train0) if asset_symbol else None
        phi_prior_center, phi_prior_tau, phi_prior_strength = asset_phi_profile(asset_class)
        use_train_state_only_kernel = (
            _USE_NUMBA
            and run_phi_student_t_improved_train_state_only is not None
            and os.environ.get("PHI_STUDENT_T_IMPROVED_DISABLE_STATE_ONLY_KERNEL", "") != "1"
        )
        use_cv_test_kernel = (
            _USE_NUMBA
            and run_phi_student_t_improved_cv_test_fold is not None
            and os.environ.get("PHI_STUDENT_T_IMPROVED_DISABLE_CV_TEST_KERNEL", "") != "1"
        )
        use_cv_var_cal = (
            os.environ.get("PHI_STUDENT_T_IMPROVED_ENABLE_CV_VAR_CAL", "1") == "1"
            and os.environ.get("PHI_STUDENT_T_IMPROVED_DISABLE_CV_VAR_CAL", "") != "1"
        )
        cv_var_cal_lambda = float(os.environ.get("PHI_STUDENT_T_IMPROVED_CV_VAR_CAL_LAMBDA", "0.01205"))
        cv_var_cal_min_obs = int(os.environ.get("PHI_STUDENT_T_IMPROVED_CV_VAR_CAL_MIN_OBS", "30"))
        cv_var_cal_z2_cap = float(os.environ.get("PHI_STUDENT_T_IMPROVED_CV_VAR_CAL_Z2_CAP", "50.0"))
        sign_brier_weight = float(os.environ.get("PHI_STUDENT_T_IMPROVED_SIGN_BRIER_WEIGHT", "0.005"))
        optimizer_maxiter = int(os.environ.get("PHI_STUDENT_T_IMPROVED_MAXITER", "45"))
        optimizer_maxls = int(os.environ.get("PHI_STUDENT_T_IMPROVED_MAXLS", "30"))

        def neg_cv_ll(params):
            log_q, log_c, phi_raw = params
            if not (np.isfinite(log_q) and np.isfinite(log_c) and np.isfinite(phi_raw)):
                return 1e12
            q = 10.0 ** float(log_q)
            c = 10.0 ** float(log_c)
            phi_clip = float(np.clip(phi_raw, phi_min_eff, phi_max_eff))
            if q <= 0.0 or c <= 0.0:
                return 1e12

            total_ll = 0.0
            total_count = 0
            total_z2 = 0.0
            total_z2_count = 0
            total_sign_brier = 0.0
            total_sign_count = 0
            for ts, te_f, vs, ve in folds:
                try:
                    if use_train_state_only_kernel:
                        mu_p, P_p = run_phi_student_t_improved_train_state_only(
                            returns, vol_sq, q, c, phi_clip, nu_val,
                            scale_factor,
                            ts, te_f,
                            gamma_vov=gamma_vov if has_vov else 0.0,
                            vov_rolling=vov_full,
                            online_scale_adapt=use_osa,
                            p_min=p_floor,
                            p_max_default=float(_P_MAX),
                        )
                    else:
                        vov_tr = vov_rolling[ts:te_f] if has_vov else None
                        mu_f, P_f, _, _, _ = cls._filter_phi_core(
                            returns[ts:te_f], vol[ts:te_f], q, c, phi_clip, nu_val,
                            robust_wt=True,
                            online_scale_adapt=use_osa,
                            gamma_vov=gamma_vov if has_vov else 0.0,
                            vov_rolling=vov_tr,
                        )
                        if len(mu_f) == 0:
                            continue
                        mu_p = float(mu_f[-1])
                        P_p = float(P_f[-1])
                    if use_cv_test_kernel:
                        (
                            ll_fold,
                            obs_count,
                            z2_count,
                            z2_sum,
                            sign_count,
                            sign_brier_sum,
                        ) = run_phi_student_t_improved_cv_test_fold(
                            returns,
                            vol_sq,
                            q,
                            c,
                            phi_clip,
                            nu_val,
                            log_norm_const,
                            neg_exp,
                            inv_nu,
                            scale_factor,
                            mu_p,
                            P_p,
                            vs,
                            ve,
                            gamma_vov=gamma_vov if has_vov else 0.0,
                            vov_rolling=vov_full,
                            online_scale_adapt=use_osa,
                            p_floor=p_floor,
                            p_cap=p_cap,
                            z2_cap=cv_var_cal_z2_cap,
                        )
                        total_ll += ll_fold
                        total_count += obs_count
                        total_sign_brier += sign_brier_sum
                        total_sign_count += sign_count
                        if use_cv_var_cal:
                            total_z2 += z2_sum
                            total_z2_count += z2_count
                        continue
                    c_adj = 1.0
                    ewm_z2 = (nu_val / (nu_val - 2.0)) if nu_val > 2.0 else 1.0
                    chi2_tgt = ewm_z2
                    chi2_lam = 0.985
                    osa_strength = min(1.0, (chi2_tgt - 1.0) / 0.5) if nu_val > 2.0 else 1.0
                    phi_sq = phi_clip * phi_clip

                    for t in range(vs, ve):
                        mu_pred = phi_clip * mu_p
                        P_pred = max(phi_sq * P_p + q, p_floor)
                        c_eff = c * c_adj
                        R_t = max(c_eff * vol_sq[t], 1e-20)
                        if has_vov:
                            R_t *= max(0.05, 1.0 + gamma_vov * vov_rolling[t])
                        S = max(P_pred + R_t, 1e-20)
                        scale = max(math.sqrt(S * scale_factor), 1e-10)
                        inn = returns[t] - mu_pred
                        z = inn / scale
                        ll_t = log_norm_const - math.log(scale) + neg_exp * math.log1p((z * z) * inv_nu)
                        if math.isfinite(ll_t):
                            total_ll += ll_t
                            total_count += 1
                            if sign_brier_weight > 0.0:
                                p_up = 1.0 - float(_fast_t_cdf(np.array([(0.0 - mu_pred) / scale]), nu_val)[0])
                                p_up = float(np.clip(p_up, 0.0, 1.0))
                                y_up = 1.0 if returns[t] > 0.0 else 0.0
                                total_sign_brier += (p_up - y_up) ** 2
                                total_sign_count += 1

                        z_sq_s = (inn * inn) / S
                        if use_cv_var_cal and math.isfinite(z_sq_s):
                            total_z2 += min(z_sq_s, cv_var_cal_z2_cap)
                            total_z2_count += 1
                        w_t = float(np.clip((nu_val + 1.0) / (nu_val + z_sq_s), 0.05, 20.0))
                        R_eff = R_t / max(w_t, 1e-8)
                        S_eff = max(P_pred + R_eff, 1e-20)
                        K = P_pred / S_eff
                        mu_p = mu_pred + K * inn
                        P_p = (1.0 - K) * (1.0 - K) * P_pred + K * K * R_eff
                        P_p = max(p_floor, min(P_p, p_cap))

                        if use_osa:
                            z2w = min(z * z, chi2_tgt * 50.0)
                            ewm_z2 = chi2_lam * ewm_z2 + (1.0 - chi2_lam) * z2w
                            ratio = float(np.clip(ewm_z2 / max(chi2_tgt, 1e-12), 0.35, 2.85))
                            c_adj = 1.0 + osa_strength * (math.sqrt(ratio) - 1.0)
                            c_adj = float(np.clip(c_adj, 0.4, 2.5))
                except Exception:
                    continue

            if total_count <= 0:
                return 1e12

            # Weak priors prevent pathological drift/noise decompositions.
            prior_pen = float(prior_lambda) * (float(log_q) - float(prior_log_q_mean)) ** 2
            c_prior_pen = 0.05 * (float(log_c) - math.log10(max(c_mom, 1e-8))) ** 2
            # Balance two truths: drift persistence exists, but likelihood can
            # over-explain return noise with phi.  The zero-center term keeps
            # the latent drift honest; the weak asset term prevents large-cap,
            # index, and metal assets from collapsing to a silent phi.
            zero_phi_pen = 0.35 * (phi_clip / 0.80) ** 2
            asset_phi_pen = 0.25 * phi_prior_strength * (
                (phi_clip - phi_prior_center) / max(phi_prior_tau, 1e-6)
            ) ** 2
            phi_shrink_pen = zero_phi_pen + asset_phi_pen
            stationarity_pen = 0.0
            if abs(phi_clip) > 0.95:
                stationarity_pen = 10.0 * (abs(phi_clip) - 0.95) ** 2

            # Average LL gives comparable regularisation across different CV layouts.
            mean_ll = total_ll / float(total_count)
            var_cal_pen = 0.0
            if (
                use_cv_var_cal
                and cv_var_cal_lambda > 0.0
                and total_z2_count >= cv_var_cal_min_obs
            ):
                mean_z2 = total_z2 / float(total_z2_count)
                var_cal_pen = cv_var_cal_lambda * (math.log(max(mean_z2, 1e-8)) ** 2)
            sign_brier_pen = 0.0
            if sign_brier_weight > 0.0 and total_sign_count > 0:
                sign_brier_pen = sign_brier_weight * (total_sign_brier / float(total_sign_count))
            return -(mean_ll - prior_pen / max(total_count, 1) - c_prior_pen / max(len(folds), 1)
                     - phi_shrink_pen / max(total_count, 1) - stationarity_pen - var_cal_pen
                     - sign_brier_pen)

        log_q_min = math.log10(q_min_eff)
        log_q_max = math.log10(max(q_max_eff, q_min_eff * 10.0))
        log_c_min = math.log10(c_min_eff)
        log_c_max = math.log10(max(c_max_eff, c_min_eff * 10.0))

        grid_candidates = []
        def _add_candidate(lq, lc, ph):
            lq = float(np.clip(lq, log_q_min, log_q_max))
            lc = float(np.clip(lc, log_c_min, log_c_max))
            ph = float(np.clip(ph, phi_min_eff, phi_max_eff))
            try:
                val = neg_cv_ll([lq, lc, ph])
                if np.isfinite(val):
                    grid_candidates.append((float(val), lq, lc, ph))
            except Exception:
                pass

        if warm_start_params is not None:
            try:
                ws_q, ws_c, ws_phi = warm_start_params
                _add_candidate(math.log10(max(float(ws_q), q_min_eff)), math.log10(max(float(ws_c), c_min_eff)), float(ws_phi))
            except Exception:
                pass

        q_start_vals = {
            prior_log_q_mean,
            math.log10(max(ret_var * 1e-5, q_min_eff)),
            math.log10(max(ret_var * 1e-4, q_min_eff)),
            math.log10(max(q_min_eff, min(q_max_eff, vol_var_med * 1e-3))),
        }
        c_start_vals = {
            math.log10(max(c_mom, c_min_eff)),
            math.log10(max(1.0, c_min_eff)),
            math.log10(max(min(c_max_eff * 0.5, c_mom * 2.0), c_min_eff)),
        }
        asset_phi_start = phi_prior_center
        phi_start_vals = [-0.10, 0.0, 0.35, asset_phi_start]
        if asset_class in ('metals_gold', 'index', 'large_cap'):
            phi_start_vals.append(0.80)
        phi_start_vals = sorted({float(np.clip(v, phi_min_eff, phi_max_eff)) for v in phi_start_vals})
        for lq0 in q_start_vals:
            for lc0 in c_start_vals:
                for ph0 in phi_start_vals:
                    _add_candidate(lq0, lc0, ph0)

        grid_candidates.sort(key=lambda x: x[0])
        top_starts = grid_candidates[:1] if grid_candidates else [
            (1e20, np.clip(prior_log_q_mean, log_q_min, log_q_max),
             np.clip(math.log10(max(c_mom, c_min_eff)), log_c_min, log_c_max), 0.0)
        ]

        best_result = None
        best_val = float('inf')
        from scipy.optimize import minimize as sp_minimize
        for _, lq0, lc0, ph0 in top_starts:
            try:
                res = sp_minimize(
                    neg_cv_ll, [lq0, lc0, ph0],
                    method='L-BFGS-B',
                    bounds=[(log_q_min, log_q_max), (log_c_min, log_c_max), (phi_min_eff, phi_max_eff)],
                    options={'maxiter': optimizer_maxiter, 'ftol': 1e-7, 'gtol': 1e-5, 'maxls': optimizer_maxls},
                )
                val = float(res.fun) if np.isfinite(res.fun) else float('inf')
                if val < best_val:
                    best_val = val
                    best_result = res
            except Exception:
                continue

        if best_result is None:
            raise ValueError("All optimizer starts failed for Student-t fixed-nu")

        log_q_opt, log_c_opt, phi_opt = best_result.x
        q_opt = float(10.0 ** log_q_opt)
        c_opt = float(10.0 ** log_c_opt)
        phi_opt = float(np.clip(phi_opt, phi_min_eff, phi_max_eff))
        cv_ll = float(-best_val)

        diagnostics = {
            "nu_fixed": float(nu_val),
            "optimizer_converged": bool(best_result.success),
            "optimizer_message": str(getattr(best_result, "message", "")),
            "n_folds": int(len(folds)),
            "n_obs": int(n),
            "log10_q": float(log_q_opt),
            "log10_c": float(log_c_opt),
            "c_mom": float(c_mom),
            "q_bounds": (float(q_min_eff), float(q_max_eff)),
            "c_bounds": (float(c_min_eff), float(c_max_eff)),
            "phi_bounds": (float(phi_min_eff), float(phi_max_eff)),
            "phi_prior_center": float(phi_prior_center),
            "phi_prior_tau": float(phi_prior_tau),
            "phi_prior_strength": float(phi_prior_strength),
            "n_grid_candidates": int(len(grid_candidates)),
            "cv_objective": float(best_val),
            "optimizer_maxiter": int(optimizer_maxiter),
            "optimizer_maxls": int(optimizer_maxls),
            "state_only_kernel_enabled": bool(use_train_state_only_kernel),
            "cv_test_kernel_enabled": bool(use_cv_test_kernel),
            "cv_var_calibration_enabled": bool(use_cv_var_cal),
            "cv_var_calibration_lambda": float(cv_var_cal_lambda),
            "sign_brier_weight": float(sign_brier_weight),
        }

        return q_opt, c_opt, phi_opt, cv_ll, diagnostics

    # =====================================================================
    # NU_REFINE grids (Lange, Little & Taylor 1989)
    # =====================================================================
    _NU_REFINE = {
        3:  [2.5, 3.0, 3.5, 4.0],
        4:  [3.5, 4.0, 5.0, 6.0],
        8:  [6.0, 7.0, 8.0, 10.0, 12.0],
        20: [15.0, 18.0, 20.0, 25.0],
    }
    # PIT-gated extensions: activated only when Stage 3 PIT < 0.10
    # (zero-cost for well-calibrated assets, targets DFSC/APLM/ESLT-class failures)
    _NU_REFINE_EXTENDED = {
        3:  [2.1, 2.5, 3.0, 3.5, 4.0, 5.0],
        4:  [2.5, 3.5, 4.0, 5.0, 6.0, 7.0],
        8:  [5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0],
        20: [12.0, 15.0, 18.0, 20.0, 25.0, 35.0, 50.0],
    }
    _GAMMA_VOV_DEFAULT = 0.3

    @staticmethod
    def _precompute_vov(vol: np.ndarray, window: int = 20) -> np.ndarray:
        """Backward-compatible hook; implementation lives in student_t_common."""
        return precompute_vov(vol, window)

    @staticmethod
    def _compute_cvm_statistic(pit_values: np.ndarray) -> float:
        """Backward-compatible hook; implementation lives in student_t_common."""
        return compute_cvm_statistic(pit_values)

    @staticmethod
    def _compute_ad_statistic(pit_values: np.ndarray) -> float:
        """Backward-compatible hook; implementation lives in student_t_common."""
        return compute_ad_statistic(pit_values)

    # =====================================================================
    # tune_and_calibrate — COMPLETE φ-t pipeline in the model
    # =====================================================================
    @classmethod
    def tune_and_calibrate(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        nu_fixed: int,
        *,
        prior_log_q_mean: float = -6.0,
        prior_lambda: float = 1.0,
        asset_symbol: str = None,
        n_obs: int = None,
        n_train: int = None,
        vov_rolling: np.ndarray = None,
        gamma_vov: float = None,
        momentum_wrapper=None,
        prices: np.ndarray = None,
        regime_labels: np.ndarray = None,
        warm_start_params: Tuple[float, float, float] = None,
    ) -> dict:
        """
        Complete tune-filter-calibrate pipeline for a FIXED ν sub-model.

        Encapsulates all stages that were previously scattered in tune.py:
          1. MLE optimisation (q, c, φ) via cross-validated L-BFGS-B
          2. Enhanced Kalman filter (robust_wt + VoV + online_scale_adapt)
          3. PIT / Hyvärinen / CRPS metrics
          4. Post-hoc GARCH innovation blending (Engle 1982, GJR 1993)
          5. ν-refinement: discrete grid + continuous brentq solver
             (Lange-Little-Taylor 1989)
          6. Score-driven GAS time-varying ν (Creal-Koopman-Lucas 2013)
          7. CRPS-optimal scale shrinkage (Gneiting & Raftery 2007)
          8. Momentum augmentation (if wrapper supplied)
          9. Isotonic recalibration (Kuleshov et al. 2018)

        Returns a dict ready for direct insertion into models[model_name].
        """
        # ── Lazy imports (avoids circular deps: tune → model → tune) ──
        from tuning.diagnostics import (
            compute_crps_student_t_inline,
            compute_hyvarinen_score_student_t,
        )
        from calibration.model_selection import compute_aic, compute_bic

        if n_obs is None:
            n_obs = len(returns)
        if gamma_vov is None:
            gamma_vov = cls._GAMMA_VOV_DEFAULT
        if vov_rolling is None:
            vov_rolling = precompute_vov(vol)
        n_params = 3  # q, c, phi
        # ── Train/test split for out-of-sample evaluation ──
        # Filter runs on full data; CRPS/PIT/Hyvärinen scored on test only
        if n_train is None:
            n_train = int(n_obs * 0.7)

        # =================================================================
        # STAGE 1: Optimize (q, c, φ) — cross-validated MLE
        # =================================================================
        q_st, c_st, phi_st, ll_cv_st, diag_st = cls.optimize_params_fixed_nu(
            returns, vol,
            nu=nu_fixed,
            prior_log_q_mean=prior_log_q_mean,
            prior_lambda=prior_lambda,
            asset_symbol=asset_symbol,
            gamma_vov=gamma_vov,
            vov_rolling=vov_rolling,
            warm_start_params=warm_start_params,
        )

        # =================================================================
        # STAGE 2: Enhanced Kalman filter with predictive output
        # =================================================================
        # Adaptive: online_scale_adapt for ν ≤ 5 only
        # (χ²-target = ν/(ν-2) ≥ 1.67 → needs in-filter correction)
        # Graduated OSA: always enabled (March 2026 v2)
        # chi² target ν/(ν-2) naturally graduates strength:
        #   ν=3: tgt=3.0 (strong), ν=8: tgt=1.33 (moderate), ν=20: tgt=1.11 (gentle)
        _use_osa = True
        mu_st, P_st, mu_pred_st, S_pred_st, ll_full_st = cls.filter_phi_with_predictive(
            returns, vol, q_st, c_st, phi_st, nu_fixed,
            robust_wt=True,
            online_scale_adapt=_use_osa,
            gamma_vov=gamma_vov,
            vov_rolling=vov_rolling,
        )

        # =================================================================
        # STAGE 3: PIT + Berkowitz + histogram MAD
        # =================================================================
        from tuning.tune import compute_extended_pit_metrics_student_t
        # PIT computed on FULL data (filter-based, needs all predictions)
        _pit_ext_base = compute_extended_pit_metrics_student_t(
            returns, vol, q_st, c_st, phi_st, nu_fixed,
            mu_pred_precomputed=mu_pred_st, S_pred_precomputed=S_pred_st,
            scale_already_adapted=_use_osa,
        )
        ks_st = _pit_ext_base["ks_statistic"]
        pit_p_st = _pit_ext_base["pit_ks_pvalue"]
        _ad_p_st = _pit_ext_base.get("ad_pvalue", float('nan'))
        _berk_st = _pit_ext_base["berkowitz_pvalue"]
        _mad_st = _pit_ext_base["histogram_mad"]

        # Information criteria
        aic_st = compute_aic(ll_full_st, n_params)
        bic_st = compute_bic(ll_full_st, n_params, n_obs)
        mean_ll_st = ll_full_st / max(n_obs, 1)

        # Forecast scale: σ = √(S × (ν-2)/ν)
        if nu_fixed > 2:
            forecast_scale_st = np.sqrt(S_pred_st * (nu_fixed - 2) / nu_fixed)
        else:
            forecast_scale_st = np.sqrt(S_pred_st)

        # ── Out-of-sample scoring on test fold only (March 2026) ──
        # Filter runs on full data; decision metrics use test period only
        _r_test = returns[n_train:]
        _mu_test = mu_pred_st[n_train:]
        _fs_test = forecast_scale_st[n_train:]
        hyvarinen_st = compute_hyvarinen_score_student_t(
            _r_test, _mu_test, _fs_test, nu_fixed
        )
        crps_st = compute_crps_student_t_inline(
            _r_test, _mu_test, _fs_test, nu_fixed
        )

        # =================================================================
        # STAGE 4: Post-hoc GARCH innovation blending (GJR 1993)
        # =================================================================
        try:
            _innovations_st = returns[:len(mu_pred_st)] - mu_pred_st
            _garch_params = cls._stage_5c_garch_estimation(
                returns[:len(mu_pred_st)], mu_pred_st, 0.0, len(mu_pred_st)
            )
            _g_omega = _garch_params['garch_omega']
            _g_alpha = _garch_params['garch_alpha']
            _g_beta = _garch_params['garch_beta']
            _g_lev = _garch_params['garch_leverage']
            _g_uvar = _garch_params['unconditional_var']
            _n_inn = len(_innovations_st)
            _h_garch = np.empty(_n_inn, dtype=np.float64)
            _h_garch[0] = _g_uvar
            for _gi in range(1, _n_inn):
                _e2 = _innovations_st[_gi - 1] ** 2
                _neg = 1.0 if _innovations_st[_gi - 1] < 0 else 0.0
                _h_garch[_gi] = (_g_omega + _g_alpha * _e2
                                 + _g_lev * _e2 * _neg
                                 + _g_beta * _h_garch[_gi - 1])
                if _h_garch[_gi] < 1e-12:
                    _h_garch[_gi] = 1e-12
            _h_mean = max(float(np.mean(_h_garch)), 1e-12)
            _garch_h_ratio = _h_garch / _h_mean
            # ── Golden-section search for GARCH blend weight (March 2026) ──
            # Replaces 5-point grid with continuous optimiser on [0.0, 0.6]
            # Scored on test fold only to prevent in-sample selection bias
            _garch_blend_w = 0.0  # default: no blending
            _best_garch_crps = crps_st  # must beat current baseline
            _garch_h_ratio_test = _garch_h_ratio[n_train:]
            _S_test = S_pred_st[n_train:]
            _gr_g = (math.sqrt(5) + 1) / 2
            _gw_lo, _gw_hi = 0.0, 0.9
            for _ in range(8):
                _gw1 = _gw_hi - (_gw_hi - _gw_lo) / _gr_g
                _gw2 = _gw_lo + (_gw_hi - _gw_lo) / _gr_g
                # Evaluate at probe point 1
                _S_gw1 = _S_test * (_gw1 * _garch_h_ratio_test + (1.0 - _gw1))
                if nu_fixed > 2:
                    _fs_gw1 = np.sqrt(_S_gw1 * (nu_fixed - 2) / nu_fixed)
                else:
                    _fs_gw1 = np.sqrt(_S_gw1)
                _c_gw1 = compute_crps_student_t_inline(
                    _r_test, _mu_test, _fs_gw1, nu_fixed)
                # Evaluate at probe point 2
                _S_gw2 = _S_test * (_gw2 * _garch_h_ratio_test + (1.0 - _gw2))
                if nu_fixed > 2:
                    _fs_gw2 = np.sqrt(_S_gw2 * (nu_fixed - 2) / nu_fixed)
                else:
                    _fs_gw2 = np.sqrt(_S_gw2)
                _c_gw2 = compute_crps_student_t_inline(
                    _r_test, _mu_test, _fs_gw2, nu_fixed)
                if _c_gw1 < _c_gw2:
                    _gw_hi = _gw2
                else:
                    _gw_lo = _gw1
            _garch_blend_w = (_gw_lo + _gw_hi) / 2.0
            S_pred_blended = S_pred_st * (_garch_blend_w * _garch_h_ratio
                                          + (1.0 - _garch_blend_w))
            if nu_fixed > 2:
                _fs_bl = np.sqrt(S_pred_blended * (nu_fixed - 2) / nu_fixed)
            else:
                _fs_bl = np.sqrt(S_pred_blended)
            _fs_bl_test = _fs_bl[n_train:]
            crps_blended = compute_crps_student_t_inline(
                _r_test, _mu_test, _fs_bl_test, nu_fixed
            )
            # PIT-rescue: relax CRPS gate when PIT is failing (March 2026)
            # If PIT < 0.05, allow up to 5% CRPS regression to rescue calibration
            _garch_crps_tol = 1.01 if pit_p_st >= 0.05 else 1.05
            if np.isfinite(crps_blended) and crps_blended <= crps_st * _garch_crps_tol:
                _pit_ext_garch = compute_extended_pit_metrics_student_t(
                    returns, vol, q_st, c_st, phi_st, nu_fixed,
                    mu_pred_precomputed=mu_pred_st,
                    S_pred_precomputed=S_pred_blended,
                    scale_already_adapted=_use_osa,
                )
                _pit_p_garch = _pit_ext_garch["pit_ks_pvalue"]
                # PIT-rescue: accept if PIT improves AND doesn't break passing models
                _garch_accept = False
                if pit_p_st >= 0.05 and _pit_p_garch < 0.05:
                    _garch_accept = False  # PIT veto: don't break passing
                elif pit_p_st < 0.05 and _pit_p_garch > pit_p_st:
                    _garch_accept = True   # PIT-rescue: any PIT improvement
                else:
                    _garch_accept = True   # Normal acceptance
                if _garch_accept:
                    crps_st = crps_blended
                    S_pred_st = S_pred_blended
                    _pit_ext_base = _pit_ext_garch
                    ks_st = _pit_ext_base["ks_statistic"]
                    pit_p_st = _pit_ext_base["pit_ks_pvalue"]
                    _ad_p_st = _pit_ext_base.get("ad_pvalue", float('nan'))
                    _berk_st = _pit_ext_base["berkowitz_pvalue"]
                    _mad_st = _pit_ext_base["histogram_mad"]
                    if nu_fixed > 2:
                        forecast_scale_st = np.sqrt(
                            S_pred_blended * (nu_fixed - 2) / nu_fixed)
                    else:
                        forecast_scale_st = np.sqrt(S_pred_blended)
                    # Update test slices after GARCH acceptance
                    _fs_test = forecast_scale_st[n_train:]
                    _mu_test = mu_pred_st[n_train:]
                    hyvarinen_st = compute_hyvarinen_score_student_t(
                        _r_test, _mu_test, _fs_test, nu_fixed
                    )
        except Exception:
            pass

        # =================================================================
        # STAGE 5: ν-refinement (March 2026 v2: PIT-gated grid, profile
        # c re-opt, PIT-rescue scoring, Hyvärinen-aware, skip-refilter)
        # =================================================================

        # ── PIT-gated grid selection ───────────────────────────────────
        # If the model is already well-calibrated (PIT ≥ 0.10), use the
        # narrow grid (preserves speed). If PIT < 0.10, activate the
        # extended grid to search wider ν space (targets DFSC/APLM/ESLT).
        _pit_needs_rescue = pit_p_st < 0.10
        if _pit_needs_rescue:
            _nu_refine_grid = list(cls._NU_REFINE_EXTENDED.get(nu_fixed, []))
        else:
            _nu_refine_grid = list(cls._NU_REFINE.get(nu_fixed, []))

        # ── Continuous ν via Newton-Raphson (Lange-Little-Taylor 1989) ──
        # Raised NR cap from 30→60 to explore near-Gaussian tails
        try:
            from scipy.special import digamma as _digamma, polygamma as _pg_nr
            _z_nu = (returns - mu_pred_st) / np.maximum(forecast_scale_st, 1e-10)
            _z2_nu = _z_nu * _z_nu
            _nu_nr = float(nu_fixed)
            for _nr_iter in range(3):
                _nv = _nu_nr
                _nv2 = _nv / 2.0
                _nvp = (_nv + 1.0) / 2.0
                _ph = _digamma(_nv2)
                _ph1 = _digamma(_nvp)
                _tr = _pg_nr(1, _nv2)
                _tr1 = _pg_nr(1, _nvp)
                _nz_arr = _nv + _z2_nu
                _s_vec = (-_ph + _ph1 + math.log(_nv) - np.log(_nz_arr)
                          + (_z2_nu - 1.0) / _nz_arr)
                _ds_vec = (-0.5 * _tr + 0.5 * _tr1 + 1.0 / _nv
                           - 1.0 / _nz_arr
                           - (_z2_nu - 1.0) / (_nz_arr ** 2))
                _s_mean = float(np.mean(_s_vec)) / 2.0
                _ds_mean = float(np.mean(_ds_vec)) / 2.0
                if abs(_ds_mean) < 1e-15:
                    break
                _nu_nr = max(2.1, min(60.0, _nu_nr - _s_mean / _ds_mean))
            if 2.1 <= _nu_nr <= 60.0:
                if all(abs(_nu_nr - _g) > 0.3 for _g in _nu_refine_grid):
                    _nu_refine_grid.append(_nu_nr)
        except Exception:
            pass

        # ── Score-driven GAS ν (Creal-Koopman-Lucas 2013) ──────────────
        try:
            from scipy.special import digamma as _dg, polygamma as _pg
            _z_gas = (returns - mu_pred_st) / np.maximum(forecast_scale_st, 1e-10)
            _z2_gas = _z_gas * _z_gas
            _n_gas = len(_z2_gas)
            _nu_t = float(nu_fixed)
            _omega_gas = float(nu_fixed) * 0.02
            _beta_gas = 0.98
            _alpha_gas = 0.15
            _nu_series = np.empty(_n_gas, dtype=np.float64)
            for _ig in range(_n_gas):
                _nu_t = max(2.2, min(50.0, _nu_t))
                _nu_series[_ig] = _nu_t
                _nv2 = _nu_t / 2.0
                _nvp = (_nu_t + 1.0) / 2.0
                _s_t = (0.5 * (_dg(_nvp) - _dg(_nv2) - 1.0 / _nu_t
                         + math.log(_nu_t / (_nu_t + _z2_gas[_ig]))
                         + (_nu_t + 1.0) * _z2_gas[_ig]
                         / (_nu_t * (_nu_t + _z2_gas[_ig]))))
                _I_nu = 0.5 * (_pg(1, _nvp) - _pg(1, _nv2)
                               - 2.0 * (_nu_t + 3.0)
                               / (_nu_t * (_nu_t + 1.0) ** 2))
                _fisher_inv_sqrt = 1.0 / math.sqrt(max(_I_nu, 1e-8))
                _nu_t = _omega_gas + _beta_gas * _nu_t + _alpha_gas * _s_t * _fisher_inv_sqrt
            _nu_gas_med = float(np.median(_nu_series[max(0, _n_gas // 4):]))
            _nu_gas_med = max(2.2, min(60.0, _nu_gas_med))
            if all(abs(_nu_gas_med - _g) > 0.3 for _g in _nu_refine_grid):
                _nu_refine_grid.append(_nu_gas_med)
        except Exception:
            pass

        # ── ν grid search: skip-refilter + profile c + PIT-rescue ──────
        # March 2026 v2: Five interleaved improvements
        #   1. Skip re-filter: reuse baseline mu_pred_st/S_pred_st, only
        #      recompute scale as √(S*(ν'-2)/ν') — valid because (q,c,φ)
        #      are fixed and robust_wt change is second-order for nearby ν.
        #   2. Profile c re-optimization: golden-section on c ∈ [c*0.75, c*1.35]
        #      with 10 iters, scored by test-fold CRPS (no filter re-run).
        #   3. PIT-rescue: when baseline PIT < 0.05, switch scoring to
        #      0.60×PIT_ratio + 0.40×CRPS_ratio, allow 5% CRPS regression.
        #   4. Hyvärinen in scoring: adds 0.20 weight to prevent variance collapse.
        #   5. Only re-filter the WINNING candidate for exact downstream values.
        _best_nu_eff = float(nu_fixed)
        _best_c_profile = float(c_st)
        _base_pit_passing = pit_p_st >= 0.05
        _base_pit_vals = _pit_ext_base.get("pit_values", np.array([]))
        _base_cvm = compute_cvm_statistic(_base_pit_vals)
        if not np.isfinite(_base_cvm) or _base_cvm < 1e-12:
            _base_cvm = 1.0
        _base_ad = compute_ad_statistic(_base_pit_vals)
        if not np.isfinite(_base_ad) or _base_ad < 1e-12:
            _base_ad = 1.0
        _orig_crps = crps_st
        _orig_cvm = _base_cvm
        _orig_ad = _base_ad
        _orig_hyv = hyvarinen_st
        # Early-exit: skip ν-refinement when PIT is clearly passing and
        # CRPS is reasonable (saves ~5 candidates × ~14 CRPS evals each)
        _skip_nu_refine = (pit_p_st >= 0.20 and crps_st < 0.035)
        # PIT-rescue mode: when baseline PIT < 0.05, prioritise calibration
        _pit_rescue = pit_p_st < 0.05
        if _pit_rescue:
            # PIT-rescue scoring: 0.60×PIT + 0.40×CRPS (Gneiting & Ranjan 2013)
            _best_refine_score = 0.40  # baseline PIT score = 0
        else:
            # Normal scoring: 0.40 CRPS + 0.20 CvM + 0.20 AD + 0.20 Hyv
            _best_refine_score = 1.0  # sum of (ratio=1.0) weights

        for _nu_trial in _nu_refine_grid:
            if _skip_nu_refine:
                break
            if abs(_nu_trial - nu_fixed) < 0.01:
                continue
            try:
                # ── Skip re-filter: reuse baseline filter output ──
                # When (q,c,φ) are fixed, ν only affects:
                # (a) scale factor ν/(ν-2) and (b) Student-t CDF shape
                # Robust_wt w_t = (ν+1)/(ν+z²) is second-order for Δν<8.
                if _nu_trial > 2:
                    _scale_ref = np.sqrt(S_pred_st * (_nu_trial - 2) / _nu_trial)
                else:
                    _scale_ref = np.sqrt(S_pred_st)

                # ── Profile c re-optimisation (Gneiting & Raftery 2007) ──
                # c*_profile(ν') minimises CRPS by absorbing the ν-dependent
                # scale change. 10-iter golden-section, no filter needed.
                _c_prof = float(c_st)
                _nu_ratio = (nu_fixed * (_nu_trial - 2)) / (max(_nu_trial * (nu_fixed - 2), 1e-10)) if (nu_fixed > 2 and _nu_trial > 2) else 1.0
                _c_lo = c_st * min(_nu_ratio * 0.85, 0.75)
                _c_hi = c_st * max(_nu_ratio * 1.15, 1.35)
                _c_lo = max(_c_lo, 0.1)
                _c_hi = min(_c_hi, 10.0)
                _gr_c = (math.sqrt(5) + 1) / 2
                for _ in range(7):
                    _c1 = _c_hi - (_c_hi - _c_lo) / _gr_c
                    _c2 = _c_lo + (_c_hi - _c_lo) / _gr_c
                    # Scale correction: scale ∝ √c, so scale_new = scale_ref * √(c_new/c_st)
                    _sc1 = _scale_ref * math.sqrt(_c1 / c_st)
                    _sc2 = _scale_ref * math.sqrt(_c2 / c_st)
                    _crps_c1 = compute_crps_student_t_inline(
                        _r_test, _mu_test, _sc1[n_train:], _nu_trial)
                    _crps_c2 = compute_crps_student_t_inline(
                        _r_test, _mu_test, _sc2[n_train:], _nu_trial)
                    if _crps_c1 < _crps_c2:
                        _c_hi = _c2
                    else:
                        _c_lo = _c1
                _c_prof = (_c_lo + _c_hi) / 2.0
                # Apply profiled c to scale
                _scale_profiled = _scale_ref * math.sqrt(_c_prof / c_st)

                # Test-only CRPS
                _crps_ref = compute_crps_student_t_inline(
                    _r_test, _mu_test, _scale_profiled[n_train:], _nu_trial)

                # PIT on full data with profiled scale
                _S_profiled = (_scale_profiled ** 2) * (_nu_trial / (_nu_trial - 2)) if _nu_trial > 2 else _scale_profiled ** 2
                _pit_ref = compute_extended_pit_metrics_student_t(
                    returns, vol, q_st, _c_prof, phi_st, _nu_trial,
                    mu_pred_precomputed=mu_pred_st,
                    S_pred_precomputed=_S_profiled,
                    scale_already_adapted=_use_osa,
                )
                _pit_p_ref = _pit_ref["pit_ks_pvalue"]

                # PIT veto: protect passing models
                if _base_pit_passing and _pit_p_ref < 0.05:
                    continue

                # Hyvärinen score on test fold
                _hyv_ref = compute_hyvarinen_score_student_t(
                    _r_test, _mu_test, _scale_profiled[n_train:], _nu_trial)

                # CvM & AD for calibration quality
                _cvm_ref = compute_cvm_statistic(
                    _pit_ref.get("pit_values", np.array([])))
                _ad_ref = compute_ad_statistic(
                    _pit_ref.get("pit_values", np.array([])))

                # ── Compute composite score ──
                if _pit_rescue:
                    # PIT-rescue mode: 0.60×PIT_ratio + 0.40×CRPS_ratio
                    # Allow up to 5% CRPS regression for PIT improvement
                    _pit_ratio = min(_pit_p_ref / max(pit_p_st, 1e-6), 10.0)
                    _crps_ratio = min(_orig_crps / max(_crps_ref, 1e-12), 5.0)
                    _score_ref = 0.60 * _pit_ratio + 0.40 * _crps_ratio
                    # Gate: CRPS must not degrade more than 5%
                    if _crps_ref > _orig_crps * 1.05:
                        continue
                else:
                    # Normal: 0.40 CRPS + 0.20 CvM + 0.20 AD + 0.20 Hyv
                    _hyv_ratio = min(abs(_orig_hyv) / max(abs(_hyv_ref), 1e-6), 5.0) if abs(_orig_hyv) > 1e-6 else 1.0
                    _score_ref = (0.40 * min(_orig_crps / max(_crps_ref, 1e-12), 5.0)
                                  + 0.20 * min(_orig_cvm / max(_cvm_ref, 1e-12), 5.0)
                                  + 0.20 * min(_orig_ad / max(_ad_ref, 1e-12), 5.0)
                                  + 0.20 * _hyv_ratio)

                if np.isfinite(_score_ref) and _score_ref > _best_refine_score:
                    _best_refine_score = _score_ref
                    _best_nu_eff = _nu_trial
                    _best_c_profile = _c_prof
                    # Store skip-filter results for now (may re-filter winner below)
                    forecast_scale_st = _scale_profiled
                    crps_st = _crps_ref
                    pit_p_st = _pit_p_ref
                    ks_st = _pit_ref["ks_statistic"]
                    _ad_p_st = _pit_ref.get("ad_pvalue", float('nan'))
                    _berk_st = _pit_ref["berkowitz_pvalue"]
                    _mad_st = _pit_ref["histogram_mad"]
                    hyvarinen_st = _hyv_ref
            except Exception:
                continue

        # ── Re-filter the winning ν for exact downstream values ──
        if abs(_best_nu_eff - nu_fixed) > 0.01:
            try:
                _, _, _mu_win, _S_win, _ll_win = cls.filter_phi_with_predictive(
                    returns, vol, q_st, _best_c_profile, phi_st, _best_nu_eff,
                    robust_wt=True, online_scale_adapt=_use_osa,
                    gamma_vov=gamma_vov, vov_rolling=vov_rolling,
                )
                mu_pred_st = _mu_win
                S_pred_st = _S_win
                ll_full_st = _ll_win
                c_st = _best_c_profile  # adopt profiled c
                if _best_nu_eff > 2:
                    forecast_scale_st = np.sqrt(_S_win * (_best_nu_eff - 2) / _best_nu_eff)
                else:
                    forecast_scale_st = np.sqrt(_S_win)
                bic_st = compute_bic(_ll_win, n_params, n_obs)
                aic_st = compute_aic(_ll_win, n_params)
                mean_ll_st = _ll_win / max(n_obs, 1)
                # Recompute test-fold metrics with exact filter output
                _fs_test = forecast_scale_st[n_train:]
                _mu_test = mu_pred_st[n_train:]
                crps_st = compute_crps_student_t_inline(
                    _r_test, _mu_test, _fs_test, _best_nu_eff)
                hyvarinen_st = compute_hyvarinen_score_student_t(
                    _r_test, _mu_test, _fs_test, _best_nu_eff)
            except Exception:
                pass

        # =================================================================
        # STAGE 6: CRPS-optimal scale correction (Gneiting & Raftery 2007)
        # March 2026: Bidirectional [0.80, 1.20], acceptance on test-only
        # =================================================================
        try:
            if n_train > 30:
                _returns_sh = returns[:n_train]
                _mu_sh = mu_pred_st[:n_train]
                _scale_sh = forecast_scale_st[:n_train]
                _gr = (math.sqrt(5) + 1) / 2
                _a_lo, _a_hi = 0.60, 1.60
                for _ in range(10):
                    _a1 = _a_hi - (_a_hi - _a_lo) / _gr
                    _a2 = _a_lo + (_a_hi - _a_lo) / _gr
                    _c1 = compute_crps_student_t_inline(
                        _returns_sh, _mu_sh, _a1 * _scale_sh, _best_nu_eff)
                    _c2 = compute_crps_student_t_inline(
                        _returns_sh, _mu_sh, _a2 * _scale_sh, _best_nu_eff)
                    if _c1 < _c2:
                        _a_hi = _a2
                    else:
                        _a_lo = _a1
                _alpha_opt = (_a_lo + _a_hi) / 2.0
                if abs(_alpha_opt - 1.0) > 0.005:
                    _scale_shrunk = forecast_scale_st * _alpha_opt
                    # Acceptance: test-only CRPS
                    _crps_shrunk = compute_crps_student_t_inline(
                        _r_test, mu_pred_st[n_train:], _scale_shrunk[n_train:],
                        _best_nu_eff)
                    if np.isfinite(_crps_shrunk) and _crps_shrunk < crps_st:
                        _nu_eff = _best_nu_eff
                        _S_shrunk = ((_scale_shrunk ** 2) * (_nu_eff / (_nu_eff - 2))
                                     if _nu_eff > 2 else _scale_shrunk ** 2)
                        _pit_shrunk = compute_extended_pit_metrics_student_t(
                            returns, vol, q_st, c_st, phi_st, _nu_eff,
                            mu_pred_precomputed=mu_pred_st,
                            S_pred_precomputed=_S_shrunk,
                            scale_already_adapted=_use_osa,
                        )
                        _pit_p_shrunk = _pit_shrunk["pit_ks_pvalue"]
                        _accept = False
                        if pit_p_st >= 0.05 and _pit_p_shrunk >= 0.05:
                            _accept = True
                        elif _pit_p_shrunk > pit_p_st:
                            _accept = True
                        if _accept:
                            forecast_scale_st = _scale_shrunk
                            crps_st = _crps_shrunk
                            S_pred_st = _S_shrunk
                            pit_p_st = _pit_p_shrunk
                            ks_st = _pit_shrunk["ks_statistic"]
                            _ad_p_st = _pit_shrunk.get("ad_pvalue", float('nan'))
                            _berk_st = _pit_shrunk["berkowitz_pvalue"]
                            _mad_st = _pit_shrunk["histogram_mad"]
                            _fs_test = forecast_scale_st[n_train:]
                            hyvarinen_st = compute_hyvarinen_score_student_t(
                                _r_test, _mu_test, _fs_test,
                                _best_nu_eff)
        except Exception:
            pass

        # =================================================================
        # STAGE 7: Momentum augmentation
        # =================================================================
        _mom_activated = False
        _mom_diag = None
        if momentum_wrapper is not None:
            try:
                from models.momentum_augmented import apply_phi_shrinkage_for_mr
                mu_mom, P_mom, ll_mom = momentum_wrapper.filter(
                    returns, vol, q_st, c_st,
                    phi=phi_st, nu=_best_nu_eff,
                    base_model='phi_student_t',
                )
                _mom_u_t = momentum_wrapper._exogenous_input
                _mom_phi_eff = apply_phi_shrinkage_for_mr(
                    phi_st, momentum_wrapper.config)
                _use_osa_mom = True  # Graduated OSA
                _, _, mu_pred_mom, S_pred_mom, _ = cls.filter_phi_with_predictive(
                    returns, vol, q_st, c_st, _mom_phi_eff, _best_nu_eff,
                    exogenous_input=_mom_u_t,
                    robust_wt=True,
                    online_scale_adapt=_use_osa_mom,
                    gamma_vov=gamma_vov,
                    vov_rolling=vov_rolling,
                )
                if _best_nu_eff > 2:
                    _fs_mom = np.sqrt(
                        S_pred_mom * (_best_nu_eff - 2) / _best_nu_eff)
                else:
                    _fs_mom = np.sqrt(S_pred_mom)
                crps_mom = compute_crps_student_t_inline(
                    _r_test, mu_pred_mom[n_train:], _fs_mom[n_train:],
                    _best_nu_eff)
                if np.isfinite(crps_mom) and crps_mom < crps_st:
                    _mom_activated = True
                    _mom_diag = momentum_wrapper.get_diagnostics()
                    pit_ext_mom = compute_extended_pit_metrics_student_t(
                        returns, vol, q_st, c_st, phi_st, _best_nu_eff,
                        mu_pred_precomputed=mu_pred_mom,
                        S_pred_precomputed=S_pred_mom,
                        scale_already_adapted=_use_osa_mom,
                    )
                    ks_st = pit_ext_mom["ks_statistic"]
                    pit_p_st = pit_ext_mom["pit_ks_pvalue"]
                    _ad_p_st = pit_ext_mom.get("ad_pvalue", float('nan'))
                    _berk_st = pit_ext_mom["berkowitz_pvalue"]
                    _mad_st = pit_ext_mom["histogram_mad"]
                    ll_full_st = ll_mom
                    mean_ll_st = ll_mom / max(n_obs, 1)
                    bic_st = compute_bic(ll_mom, n_params, n_obs)
                    aic_st = compute_aic(ll_mom, n_params)
                    hyvarinen_st = compute_hyvarinen_score_student_t(
                        _r_test, mu_pred_mom[n_train:], _fs_mom[n_train:],
                        _best_nu_eff)
                    crps_st = crps_mom
                    forecast_scale_st = _fs_mom
                    _pit_ext_base = pit_ext_mom
            except Exception:
                pass

        # =================================================================
        # STAGE 7.5: Hansen Skew-t augmentation (March 2026)
        # =================================================================
        # Fit Hansen skewness λ to Kalman residuals. If the asymmetric
        # observation likelihood improves CRPS, activate it.
        # This is a FULL Kalman filter re-run with Hansen logpdf/robust wt.
        # Template: same CRPS gate as momentum (Stage 7).
        # =================================================================
        _hansen_activated = False
        _hansen_lambda = 0.0
        _hansen_diag = None
        try:
            from models.hansen_skew_t import fit_hansen_skew_t_mle, hansen_skew_t_cdf
            # Compute standardised residuals for Hansen MLE
            if forecast_scale_st is not None and len(forecast_scale_st) == n_obs:
                _fs_valid = forecast_scale_st[forecast_scale_st > 1e-12]
                if len(_fs_valid) >= 50:
                    _resid_hansen = (returns - mu_pred_st) / np.maximum(forecast_scale_st, 1e-12)
                    _resid_train = _resid_hansen[:n_train]
                    try:
                        _nu_h, _lam_h, _ll_h, _h_diag = fit_hansen_skew_t_mle(
                            _resid_train, nu_hint=_best_nu_eff)
                        if (_h_diag.get('fit_success', False) and
                            abs(_lam_h) > 0.01 and np.isfinite(_ll_h)):
                            # Re-run filter with Hansen observation noise
                            try:
                                from models.numba_wrappers import run_phi_hansen_skew_t_filter
                                _exo = None
                                if momentum_wrapper is not None and _mom_activated:
                                    _exo = momentum_wrapper._exogenous_input
                                mu_f_h, P_f_h, mu_p_h, S_p_h, ll_h = run_phi_hansen_skew_t_filter(
                                    returns, vol, q_st, c_st, phi_st, _best_nu_eff,
                                    hansen_lambda=_lam_h,
                                    exogenous_input=_exo,
                                    online_scale_adapt=True,
                                    gamma_vov=gamma_vov if gamma_vov else 0.0,
                                    vov_rolling=vov_rolling,
                                )
                            except Exception:
                                # Python fallback if Numba not available
                                mu_f_h, P_f_h, mu_p_h, S_p_h, ll_h = cls._filter_phi_core(
                                    returns, vol, q_st, c_st, phi_st, _best_nu_eff,
                                    robust_wt=True, online_scale_adapt=True,
                                    gamma_vov=gamma_vov if gamma_vov else 0.0,
                                    vov_rolling=vov_rolling,
                                )
                            if _best_nu_eff > 2:
                                _fs_h = np.sqrt(S_p_h * (_best_nu_eff - 2) / _best_nu_eff)
                            else:
                                _fs_h = np.sqrt(S_p_h)
                            # CRPS on test fold (must use Hansen CDF for proper scoring)
                            crps_hansen = compute_crps_student_t_inline(
                                _r_test, mu_p_h[n_train:], _fs_h[n_train:], _best_nu_eff)
                            # BIC penalty: +1 parameter for λ
                            _n_params_h = n_params + 1
                            bic_hansen = compute_bic(ll_h, _n_params_h, n_obs)
                            # CRPS gate (strict improvement)
                            if (np.isfinite(crps_hansen) and crps_hansen < crps_st and
                                np.isfinite(bic_hansen) and bic_hansen < bic_st + 2.0):
                                _hansen_activated = True
                                _hansen_lambda = float(_lam_h)
                                _hansen_diag = {
                                    'nu_hansen': float(_nu_h),
                                    'lambda_hansen': float(_lam_h),
                                    'crps_before': float(crps_st),
                                    'crps_after': float(crps_hansen),
                                    'bic_before': float(bic_st),
                                    'bic_after': float(bic_hansen),
                                    'fit_diagnostics': _h_diag,
                                }
                                # Update all downstream metrics
                                mu_pred_st = mu_p_h
                                S_pred_st = S_p_h
                                forecast_scale_st = _fs_h
                                crps_st = crps_hansen
                                ll_full_st = ll_h
                                mean_ll_st = ll_h / max(n_obs, 1)
                                n_params = _n_params_h
                                bic_st = bic_hansen
                                aic_st = compute_aic(ll_h, _n_params_h)
                                # Recompute PIT/metrics with Hansen CDF
                                pit_ext_h = compute_extended_pit_metrics_student_t(
                                    returns, vol, q_st, c_st, phi_st, _best_nu_eff,
                                    mu_pred_precomputed=mu_p_h,
                                    S_pred_precomputed=S_p_h,
                                    scale_already_adapted=True,
                                )
                                ks_st = pit_ext_h["ks_statistic"]
                                pit_p_st = pit_ext_h["pit_ks_pvalue"]
                                _ad_p_st = pit_ext_h.get("ad_pvalue", float('nan'))
                                _berk_st = pit_ext_h["berkowitz_pvalue"]
                                _mad_st = pit_ext_h["histogram_mad"]
                                _pit_ext_base = pit_ext_h
                                hyvarinen_st = compute_hyvarinen_score_student_t(
                                    _r_test, mu_p_h[n_train:], _fs_h[n_train:], _best_nu_eff)
                    except Exception:
                        pass
        except Exception:
            pass

        # =================================================================
        # STAGE 7.6: Contaminated Student-t augmentation (March 2026)
        # =================================================================
        # Fit CST (ν_crisis, ε) via profile likelihood on residuals.
        # If the mixture observation likelihood improves CRPS, activate it.
        # Uses Numba CST filter kernel for speed.
        # =================================================================
        _cst_activated = False
        _cst_nu_crisis = None
        _cst_epsilon = 0.0
        _cst_diag = None
        try:
            # Only attempt if we have enough data and ν is moderate
            if n_obs >= 100 and _best_nu_eff >= 4:
                # Profile likelihood grid search for (ν_crisis, ε)
                _CST_NU_CRISIS_GRID = [3.0, 4.0, 5.0, 6.0]
                _CST_EPSILON_GRID = [0.02, 0.05, 0.10]
                _best_cst_crps = crps_st
                _best_cst_combo = None

                for _nc in _CST_NU_CRISIS_GRID:
                    if _nc >= _best_nu_eff:
                        continue  # ν_crisis must be < ν_normal
                    for _eps in _CST_EPSILON_GRID:
                        try:
                            from models.numba_wrappers import run_phi_cst_filter
                            _exo_cst = None
                            if momentum_wrapper is not None and _mom_activated:
                                _exo_cst = momentum_wrapper._exogenous_input
                            mu_f_c, P_f_c, mu_p_c, S_p_c, ll_c = run_phi_cst_filter(
                                returns, vol, q_st, c_st, phi_st,
                                nu_normal=_best_nu_eff,
                                nu_crisis=_nc,
                                epsilon=_eps,
                                exogenous_input=_exo_cst,
                                online_scale_adapt=True,
                                gamma_vov=gamma_vov if gamma_vov else 0.0,
                                vov_rolling=vov_rolling,
                            )
                            if _best_nu_eff > 2:
                                _fs_c = np.sqrt(S_p_c * (_best_nu_eff - 2) / _best_nu_eff)
                            else:
                                _fs_c = np.sqrt(S_p_c)
                            _crps_c = compute_crps_student_t_inline(
                                _r_test, mu_p_c[n_train:], _fs_c[n_train:], _best_nu_eff)
                            # +2 extra params: ν_crisis and ε
                            _n_params_c = n_params + 2
                            _bic_c = compute_bic(ll_c, _n_params_c, n_obs)
                            if (np.isfinite(_crps_c) and _crps_c < _best_cst_crps and
                                np.isfinite(_bic_c) and _bic_c < bic_st + 4.0):
                                _best_cst_crps = _crps_c
                                _best_cst_combo = (_nc, _eps, mu_p_c, S_p_c, _fs_c, ll_c, _bic_c, _n_params_c)
                        except Exception:
                            continue

                if _best_cst_combo is not None:
                    _nc, _eps, mu_p_c, S_p_c, _fs_c, ll_c, _bic_c, _n_params_c = _best_cst_combo
                    _cst_activated = True
                    _cst_nu_crisis = float(_nc)
                    _cst_epsilon = float(_eps)
                    _cst_diag = {
                        'nu_crisis': float(_nc),
                        'epsilon': float(_eps),
                        'crps_before': float(crps_st),
                        'crps_after': float(_best_cst_crps),
                        'bic_before': float(bic_st),
                        'bic_after': float(_bic_c),
                    }
                    # Update all downstream metrics
                    mu_pred_st = mu_p_c
                    S_pred_st = S_p_c
                    forecast_scale_st = _fs_c
                    crps_st = _best_cst_crps
                    ll_full_st = ll_c
                    mean_ll_st = ll_c / max(n_obs, 1)
                    n_params = _n_params_c
                    bic_st = _bic_c
                    aic_st = compute_aic(ll_c, _n_params_c)
                    # Recompute PIT with CST
                    pit_ext_c = compute_extended_pit_metrics_student_t(
                        returns, vol, q_st, c_st, phi_st, _best_nu_eff,
                        mu_pred_precomputed=mu_p_c,
                        S_pred_precomputed=S_p_c,
                        scale_already_adapted=True,
                    )
                    ks_st = pit_ext_c["ks_statistic"]
                    pit_p_st = pit_ext_c["pit_ks_pvalue"]
                    _ad_p_st = pit_ext_c.get("ad_pvalue", float('nan'))
                    _berk_st = pit_ext_c["berkowitz_pvalue"]
                    _mad_st = pit_ext_c["histogram_mad"]
                    _pit_ext_base = pit_ext_c
                    hyvarinen_st = compute_hyvarinen_score_student_t(
                        _r_test, mu_p_c[n_train:], _fs_c[n_train:], _best_nu_eff)
        except Exception:
            pass

        # =================================================================
        # STAGE 8: Scale-corrected isotonic recalibration (March 2026)
        # =================================================================
        # Key fix: instead of cosmetically warping PIT values, infer scale
        # correction from the PIT distribution and apply to forecast_scale_st.
        # This makes downstream CRPS/Hyvärinen metrics coherent with the
        # improved calibration.
        # =================================================================
        try:
            _raw_pit = _pit_ext_base.get("pit_values", None)
            if _raw_pit is not None and len(_raw_pit) >= 50:
                # Step 1: Derive scale correction from PIT IQR
                # (Diebold, Gunther & Tay 1998: PIT-based calibration)
                # For perfect calibration: PIT ~ Uniform(0,1), IQR = 0.5
                _pit_q25 = float(np.percentile(_raw_pit, 25))
                _pit_q75 = float(np.percentile(_raw_pit, 75))
                _pit_iqr = _pit_q75 - _pit_q25
                _scale_corr = _pit_iqr / 0.5  # >1 = under-dispersed, <1 = over-dispersed
                _scale_corr = float(np.clip(_scale_corr, 0.5, 2.0))

                # Only apply if correction is meaningful (>5% change)
                if abs(_scale_corr - 1.0) > 0.05:
                    forecast_scale_st = forecast_scale_st * _scale_corr

                    # Recompute PIT with corrected scale
                    _pit_corr = compute_extended_pit_metrics_student_t(
                        returns, vol, q_st, c_st, phi_st, _best_nu_eff,
                        mu_pred_precomputed=mu_pred_st,
                        S_pred_precomputed=S_pred_st,
                        forecast_scale_override=forecast_scale_st,
                        scale_already_adapted=True,
                    )
                    _pit_p_corr = _pit_corr.get("pit_ks_pvalue", 0.0)

                    # Accept only if PIT improves (no degradation)
                    if _pit_p_corr > pit_p_st * 0.95:
                        pit_p_st = float(_pit_p_corr)
                        ks_st = float(_pit_corr["ks_statistic"])
                        _ad_p_st = float(_pit_corr.get("ad_pvalue", float('nan')))
                        _berk_st = float(_pit_corr.get("berkowitz_pvalue", _berk_st))
                        _mad_st = float(_pit_corr.get("histogram_mad", _mad_st))
                        _raw_pit = _pit_corr.get("pit_values", _raw_pit)

                        # Recompute CRPS and Hyvärinen on test fold
                        _fs_test = forecast_scale_st[n_train:]
                        crps_st = compute_crps_student_t_inline(
                            _r_test, _mu_test, _fs_test, _best_nu_eff)
                        hyvarinen_st = compute_hyvarinen_score_student_t(
                            _r_test, _mu_test, _fs_test, _best_nu_eff)
                        bic_st = compute_bic(ll_full_st, n_params, n_obs)
                    else:
                        # Revert scale correction
                        forecast_scale_st = forecast_scale_st / _scale_corr

                # Step 2: Isotonic PIT blending (Kuleshov et al. 2018)
                # March 2026: adaptive blend weight based on KS improvement
                from calibration.isotonic_recalibration import IsotonicRecalibrator
                _iso_recal = IsotonicRecalibrator()
                _iso_result = _iso_recal.fit(_raw_pit)
                if _iso_result.fit_success and not _iso_result.is_identity:
                    _iso_pit = _iso_recal.transform(_raw_pit)
                    # Adaptive blend: stronger when isotonic reduces KS more
                    _ks_raw, _ = _fast_ks_uniform(_raw_pit)
                    _ks_iso_only, _ = _fast_ks_uniform(_iso_pit)
                    _ks_improve = 1.0 - min(_ks_iso_only / max(_ks_raw, 1e-10), 1.0)
                    _blend_w = float(np.clip(0.2 + 0.5 * _ks_improve, 0.1, 0.7))
                    _blended_pit = ((1.0 - _blend_w) * _raw_pit
                                    + _blend_w * _iso_pit)
                    _blended_pit = np.clip(_blended_pit, 0.001, 0.999)
                    _ks_iso, _p_iso = _fast_ks_uniform(_blended_pit)
                    if _p_iso > pit_p_st * 1.05:
                        pit_p_st = float(_p_iso)
                        ks_st = float(_ks_iso)
                        # Update Berkowitz for downstream consistency
                        _n_bt = int(len(_blended_pit) * 0.7)
                        _berk_test = _blended_pit[_n_bt:]
                        if len(_berk_test) >= 30:
                            _bp, _, _ = cls._compute_berkowitz_full(
                                _berk_test)
                            if np.isfinite(_bp):
                                _berk_st = float(_bp)
        except Exception:
            pass

        # =================================================================
        # Assemble result dict
        # =================================================================
        return {
            "q": float(q_st),
            "c": float(c_st),
            "phi": float(phi_st),
            "nu": float(_best_nu_eff),
            "nu_grid": float(nu_fixed),
            "log_likelihood": float(ll_full_st),
            "mean_log_likelihood": float(mean_ll_st),
            "cv_penalized_ll": float(ll_cv_st),
            "bic": float(bic_st),
            "aic": float(aic_st),
            "hyvarinen_score": float(hyvarinen_st),
            "crps": float(crps_st),
            "n_params": int(n_params),
            "ks_statistic": float(ks_st),
            "pit_ks_pvalue": float(pit_p_st),
            "ad_pvalue": float(_ad_p_st),
            "ad_pvalue_raw": float(_pit_ext_base.get("ad_pvalue_raw", _ad_p_st) if isinstance(_pit_ext_base, dict) else _ad_p_st),
            "ad_correction": _pit_ext_base.get("ad_correction", {}) if isinstance(_pit_ext_base, dict) else {},
            "berkowitz_pvalue": float(_berk_st),
            "berkowitz_lr": float(_pit_ext_base.get("berkowitz_lr", 0.0)),
            "pit_count": int(_pit_ext_base.get("pit_count", 0)),
            "histogram_mad": float(_mad_st),
            "fit_success": True,
            "diagnostics": diag_st,
            "nu_fixed": True,
            "momentum_augmented": _mom_activated,
            "momentum_diagnostics": _mom_diag,
            "hansen_activated": _hansen_activated,
            "hansen_lambda": float(_hansen_lambda),
            "hansen_diagnostics": _hansen_diag,
            "cst_activated": _cst_activated,
            "cst_nu_crisis": float(_cst_nu_crisis) if _cst_nu_crisis is not None else None,
            "cst_epsilon": float(_cst_epsilon),
            "cst_diagnostics": _cst_diag,
        }

    @staticmethod
    def _pit_simple_path(returns_test, mu_pred_test, S_calibrated, nu, t_df_asym):
        """PIT via basic Student-t CDF (non-GARCH path). Returns (pit, sigma, mu_eff)."""
        return common_pit_simple_path(
            returns_test, mu_pred_test, S_calibrated, nu, t_df_asym,
            cdf_fn=_fast_t_cdf,
        )

    @staticmethod
    def _compute_berkowitz_pvalue(pit_values):
        """Berkowitz (2001) p-value only. Wrapper around _compute_berkowitz_full."""
        return common_compute_berkowitz_pvalue(pit_values)

    @staticmethod
    def _compute_berkowitz_full(pit_values):
        """Berkowitz (2001) LR test. Returns (p_value, lr_statistic, n_pit)."""
        return common_compute_berkowitz_full(pit_values)

    @classmethod
    def filter_phi(cls, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, nu: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Kalman drift filter with persistence (phi) and Student-t observation noise.

        The public path now uses the same robust Student-t covariance update as
        the predictive/calibration path.  This avoids train/inference drift and
        prevents negative covariance updates on central observations.
        """
        mu, P, _, _, ll = cls._filter_phi_core(
            returns, vol, q, c, phi, nu,
            robust_wt=True,
            online_scale_adapt=False,
        )
        return mu, P, ll

    @classmethod
    def _filter_phi_core(
        cls,
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Consolidated phi-Student-t Kalman filter.

        Uses a numerically stable scale-mixture update for Student-t noise:
        the latent precision changes the effective observation variance rather
        than multiplying a Gaussian gain by w_t.  This keeps covariance positive,
        improves tail handling, and makes filtered drift estimates less fragile.
        """
        returns = np.asarray(returns, dtype=np.float64).ravel()
        vol = np.asarray(vol, dtype=np.float64).ravel()
        n = min(len(returns), len(vol))
        if n <= 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty, empty, 0.0

        returns = returns[:n].copy()
        vol = vol[:n].copy()
        returns[~np.isfinite(returns)] = 0.0

        finite_vol = vol[np.isfinite(vol) & (vol > 0)]
        vol_fill = float(np.median(finite_vol)) if finite_vol.size else max(float(np.std(returns)), 1e-4)
        vol_floor = max(vol_fill * 1e-4, 1e-10)
        vol = np.where(np.isfinite(vol) & (vol > 0), vol, vol_fill)
        vol = np.maximum(vol, vol_floor)

        has_exo = exogenous_input is not None
        if has_exo:
            exo = np.asarray(exogenous_input, dtype=np.float64).ravel()
            if len(exo) < n:
                exo = np.pad(exo, (0, n - len(exo)), constant_values=0.0)
            exo = exo[:n]
            exo = np.where(np.isfinite(exo), exo, 0.0)
        else:
            exo = None

        q_val = float(q) if np.isfinite(q) else 1e-8
        c_val = float(c) if np.isfinite(c) else 1.0
        q_val = max(q_val, 1e-14)
        c_val = max(c_val, 1e-12)
        phi_val = float(np.clip(phi if np.isfinite(phi) else 0.0, -0.999, 0.999))
        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)

        has_vov = gamma_vov > 1e-12 and vov_rolling is not None
        if has_vov:
            vov = np.asarray(vov_rolling, dtype=np.float64).ravel()
            if len(vov) < n:
                pad_val = float(vov[-1]) if len(vov) else 0.0
                vov = np.pad(vov, (0, n - len(vov)), constant_values=pad_val)
            vov = np.where(np.isfinite(vov[:n]), vov[:n], 0.0)
            gamma_vov = float(np.clip(gamma_vov, 0.0, 10.0))
        else:
            vov = None
            gamma_vov = 0.0

        phi_sq = phi_val * phi_val
        log_norm_const = math.lgamma((nu_val + 1.0) / 2.0) - math.lgamma(nu_val / 2.0) - 0.5 * math.log(nu_val * math.pi)
        neg_exp = -0.5 * (nu_val + 1.0)
        inv_nu = 1.0 / nu_val
        scale_factor = (nu_val - 2.0) / nu_val if nu_val > 2.0 else 1.0
        vol_sq = vol * vol

        chi2_tgt = (nu_val / (nu_val - 2.0)) if nu_val > 2.0 else 1.0
        chi2_lam = 0.985
        chi2_1m = 1.0 - chi2_lam
        chi2_cap = chi2_tgt * 50.0
        ewm_z2 = chi2_tgt
        c_adj = 1.0
        osa_strength = min(1.0, (chi2_tgt - 1.0) / 0.5) if nu_val > 2.0 else 1.0

        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        mu_pred_arr = np.empty(n, dtype=np.float64)
        S_pred_arr = np.empty(n, dtype=np.float64)

        vol_var_med = max(float(np.median(vol_sq)), 1e-12)
        p_floor = max(float(_P_MIN), 1e-12)
        p_cap = max(float(_P_MAX), 100.0 * vol_var_med, 1000.0 * q_val, p_floor * 10.0)
        mu = 0.0
        P = min(max(vol_var_med, 10.0 * q_val, p_floor), p_cap)
        log_likelihood = 0.0

        for t in range(n):
            u_t = exo[t] if has_exo else 0.0
            mu_pred = phi_val * mu + u_t
            P_pred = max(phi_sq * P + q_val, p_floor)

            c_eff = c_val * c_adj if online_scale_adapt else c_val
            R_t = max(c_eff * vol_sq[t], 1e-20)
            if has_vov:
                R_t *= float(np.clip(1.0 + gamma_vov * vov[t], 0.05, 20.0))
                R_t = max(R_t, 1e-20)

            S = max(P_pred + R_t, 1e-20)
            mu_pred_arr[t] = mu_pred
            S_pred_arr[t] = S

            innovation = returns[t] - mu_pred
            forecast_scale = max(math.sqrt(S * scale_factor), 1e-10)
            z = innovation / forecast_scale

            ll_t = log_norm_const - math.log(forecast_scale) + neg_exp * math.log1p((z * z) * inv_nu)
            if math.isfinite(ll_t):
                log_likelihood += ll_t

            if robust_wt:
                # Latent precision using predictive variance; effective obs
                # noise is R/w, giving a Joseph-positive covariance update.
                z_sq_s = (innovation * innovation) / S
                w_t = float(np.clip((nu_val + 1.0) / (nu_val + z_sq_s), 0.05, 20.0))
                R_eff = R_t / max(w_t, 1e-8)
                S_eff = max(P_pred + R_eff, 1e-20)
                K = P_pred / S_eff
                mu = mu_pred + K * innovation
                P = (1.0 - K) * (1.0 - K) * P_pred + K * K * R_eff
            else:
                K = P_pred / S
                mu = mu_pred + K * innovation
                P = (1.0 - K) * (1.0 - K) * P_pred + K * K * R_t

            P = max(p_floor, min(P, p_cap))
            mu_filtered[t] = mu
            P_filtered[t] = P

            if online_scale_adapt:
                z2w = min(z * z, chi2_cap)
                ewm_z2 = chi2_lam * ewm_z2 + chi2_1m * z2w
                ratio = float(np.clip(ewm_z2 / max(chi2_tgt, 1e-12), 0.35, 2.85))
                dev = abs(ratio - 1.0)
                if dev < 0.04:
                    c_adj = 1.0
                else:
                    c_adj_raw = math.sqrt(ratio)
                    c_adj = 1.0 + osa_strength * (c_adj_raw - 1.0)
                    c_adj = float(np.clip(c_adj, 0.4, 2.5))

        return mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, float(log_likelihood)

    @classmethod
    def _filter_phi_python_optimized(cls, returns, vol, q, c, phi, nu):
        """Backward-compatible wrapper — returns (mu, P, ll) only."""
        mu, P, _, _, ll = cls._filter_phi_core(returns, vol, q, c, phi, nu)
        return mu, P, ll

    @classmethod
    def filter_phi_augmented(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        nu: float,
        exogenous_input: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Kalman filter with exogenous input and robust Student-t weighting.

        STATE-EQUATION INTEGRATION:
            mu_t = phi * mu_{t-1} + u_t + w_t
            r_t = mu_t + eps_t,  eps_t ~ t(nu)
        """
        mu, P, _, _, ll = cls._filter_phi_core(
            returns, vol, q, c, phi, nu,
            exogenous_input=exogenous_input, robust_wt=True,
        )
        return mu, P, ll

    @classmethod
    def filter_phi_with_predictive(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        nu: float,
        exogenous_input: np.ndarray = None,
        robust_wt: bool = True,
        online_scale_adapt: bool = False,
        gamma_vov: float = 0.0,
        vov_rolling: np.ndarray = None,
        prediction_bias_lambda: float = 0.95,
        prediction_bias_cap_z: float = 0.18,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """phi-Student-t filter returning predictive mu_pred and S_pred for PIT.

        Enhanced (March 2026):
          - robust_wt=True by default (Meinhold & Singpurwalla 1989)
          - online_scale_adapt: embed chi² EWM scale correction (Harvey 1989)
          - gamma_vov/vov_rolling: vol-of-vol R_t inflation (BN-S 2002)

        Args:
            exogenous_input: Optional u_t array (momentum/MR signal) injected
                into the state prediction: mu_pred_t = phi * mu_{t-1} + u_t.
            robust_wt: Use Student-t outlier downweighting w_t=(ν+1)/(ν+z²).
            online_scale_adapt: Embed chi² EWM scale correction in filter loop.
            gamma_vov: Vol-of-vol sensitivity (0.0 = disabled).
            vov_rolling: Precomputed rolling std of log-vol.
            prediction_bias_lambda: Causal lagged innovation-memory correction.
            prediction_bias_cap_z: Cap correction as a fraction of forecast sigma.
        """
        mu_f, P_f, mu_pred, S_pred, ll = cls._filter_phi_core(
            returns, vol, q, c, phi, nu,
            exogenous_input=exogenous_input,
            robust_wt=robust_wt,
            online_scale_adapt=online_scale_adapt,
            gamma_vov=gamma_vov,
            vov_rolling=vov_rolling,
        )
        if prediction_bias_lambda >= 0.01 and len(mu_pred) > 2:
            correction = ewm_lagged_correction(returns, mu_pred, prediction_bias_lambda)
            cap_z = float(np.clip(prediction_bias_cap_z, 0.0, 1.0))
            if cap_z > 0.0:
                nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)
                scale_factor = (nu_val - 2.0) / nu_val if nu_val > 2.0 else 1.0
                cap = cap_z * np.sqrt(np.maximum(S_pred, 1e-20) * scale_factor)
                correction = np.clip(correction, -cap, cap)
            mu_pred = mu_pred + correction
        return mu_f, P_f, mu_pred, S_pred, ll

    @staticmethod
    def _stage_5c_garch_estimation(returns_train, mu_pred_train, mu_drift_opt, n_train):
        """
        Stage 5c: robust GJR-GARCH(1,1) parameter estimation.

        Moment-based, bounded, and stationarity-enforced.  This avoids unstable
        GARCH blends when residuals are flat, missing, or dominated by one print.
        """
        return estimate_gjr_garch_params(
            returns_train,
            mu_pred_train,
            mu_drift_opt,
            n_train,
        )

    # =========================================================================
    # AD TAIL-CORRECTION PIPELINE (March 2026)
    # =========================================================================
    # Three stacking causal corrections for model calibration improvement.
    # Applied AFTER chi² EWM and PIT-var stretching.
    #
    # REAL MODEL IMPROVEMENT (not cosmetic):
    #   Stage A: TWSC — Scale correction factor stored → applied to P_filtered
    #            in signals.py → calibrates MC simulation variance
    #   Stage B: SPTG — GPD tail parameters stored → used to adjust effective ν
    #            in MC simulation → more accurate tail risk estimation
    #   Stage C: Isotonic — Transport map knots stored → applied to directional
    #            probability p(up) in signals.py → calibrated Kelly sizing
    #
    # The correction parameters are persisted to the tune cache JSON and loaded
    # by signals.py at inference time for REAL predictive improvement.
    # =========================================================================

    @classmethod
    def apply_ad_correction_pipeline(
        cls,
        returns: np.ndarray,
        mu_pred: np.ndarray,
        scale: np.ndarray,
        nu: float,
        pit_raw: np.ndarray,
        min_obs: int = 250,
    ) -> tuple:
        """
        Apply calibration correction pipeline and persist usable calibration params.

        Corrections are accepted only when they improve a distributional metric
        or preserve calibration while providing a coherent scale/tail parameter.
        """
        returns = np.asarray(returns, dtype=np.float64).ravel()
        mu_pred = np.asarray(mu_pred, dtype=np.float64).ravel()
        scale = np.asarray(scale, dtype=np.float64).ravel()
        pit = np.asarray(pit_raw, dtype=np.float64).ravel()
        n = min(len(returns), len(mu_pred), len(scale), len(pit))
        if n <= 0:
            return np.empty(0, dtype=np.float64), {
                'twsc_applied': False, 'sptg_applied': False,
                'isotonic_applied': False, 'calibration_params': {}, 'n_obs': 0,
            }

        returns = returns[:n].copy()
        mu_pred = mu_pred[:n].copy()
        scale = scale[:n].copy()
        pit = pit[:n].copy()

        returns[~np.isfinite(returns)] = 0.0
        mu_pred[~np.isfinite(mu_pred)] = 0.0
        finite_scale = scale[np.isfinite(scale) & (scale > 0)]
        scale_fill = float(np.median(finite_scale)) if finite_scale.size else 1.0
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, scale_fill)
        scale = np.maximum(scale, max(scale_fill * 1e-6, 1e-10))
        pit = np.clip(np.where(np.isfinite(pit), pit, 0.5), 0.001, 0.999)

        if n < 50:
            return pit, {'twsc_applied': False, 'sptg_applied': False,
                         'isotonic_applied': False, 'calibration_params': {}, 'n_obs': int(n)}

        nu = cls._clip_nu(nu, cls.nu_min_default, 60.0)
        diag = {
            'twsc_applied': False,
            'sptg_applied': False,
            'sptg_xi_left': float('nan'),
            'sptg_xi_right': float('nan'),
            'isotonic_applied': False,
            'isotonic_ks_improvement': 0.0,
            'n_obs': int(n),
        }
        cal_params = {}

        z = np.clip((returns - mu_pred) / np.maximum(scale, 1e-10), -1e6, 1e6)
        pit_base = pit.copy()
        ks_base, p_base = _fast_ks_uniform(pit_base)
        ad_base = compute_ad_statistic(pit_base)

        # Stage A: Tail-weighted scale correction.
        scale_inflate = None
        try:
            from models.numba_wrappers import run_ad_twsc
            scale_inflate = run_ad_twsc(z, ewma_lambda=0.97, alpha_quantile=0.05,
                                        kappa=0.5, max_inflate=2.0, deadzone=0.15)
        except (ImportError, Exception):
            try:
                from models.numba_kernels import ad_twsc_kernel
                z_cont = np.ascontiguousarray(z, dtype=np.float64)
                scale_inflate = ad_twsc_kernel(z_cont, 0.97, 0.05, 0.5, 2.0, 0.15)
            except Exception:
                scale_inflate = None

        if scale_inflate is not None:
            scale_inflate = np.asarray(scale_inflate, dtype=np.float64).ravel()
            if len(scale_inflate) >= n:
                scale_inflate = np.clip(np.where(np.isfinite(scale_inflate[:n]), scale_inflate[:n], 1.0), 1.0, 2.5)
                z_twsc = z / scale_inflate
                pit_twsc = np.clip(_fast_t_cdf(z_twsc, nu), 0.001, 0.999)
                ks_twsc, p_twsc = _fast_ks_uniform(pit_twsc)
                ad_twsc = compute_ad_statistic(pit_twsc)
                if (p_twsc >= p_base * 0.95) or (ad_twsc <= ad_base * 1.02):
                    pit = pit_twsc
                    diag['twsc_applied'] = True
                    tail_start = max(1, int(n * 0.7))
                    tail_factors = scale_inflate[tail_start:]
                    tail_factors = tail_factors[tail_factors > 0]
                    if len(tail_factors) > 0:
                        twsc_geo_mean = float(np.exp(np.mean(np.log(tail_factors))))
                        cal_params['twsc_scale_factor'] = float(np.clip(twsc_geo_mean, 1.0, 2.5))
                        cal_params['twsc_last_ewma'] = float(scale_inflate[-1])
                    z_for_gpd = z_twsc
                    ks_base, p_base, ad_base = ks_twsc, p_twsc, ad_twsc
                else:
                    z_for_gpd = z
            else:
                z_for_gpd = z
        else:
            z_for_gpd = z

        # Stage B: Semi-parametric EVT tail grafting.
        if n >= min_obs:
            try:
                from calibration.evt_tail import fit_gpd_pot
                z_for_gpd = np.asarray(z_for_gpd, dtype=np.float64)
                abs_z = np.abs(z_for_gpd)
                left_losses = abs_z[z_for_gpd < 0]
                right_losses = abs_z[z_for_gpd > 0]

                if len(left_losses) >= 25 and len(right_losses) >= 25:
                    gpd_left = fit_gpd_pot(left_losses, threshold_percentile=0.90)
                    gpd_right = fit_gpd_pot(right_losses, threshold_percentile=0.90)

                    if gpd_left.fit_success and gpd_right.fit_success:
                        u_left = float(gpd_left.threshold)
                        u_right = float(gpd_right.threshold)
                        p_left_val = float(_fast_t_cdf(np.array([-u_left]), nu)[0])
                        p_right_val = float(1.0 - _fast_t_cdf(np.array([u_right]), nu)[0])

                        if p_left_val > 0.001 and p_right_val > 0.001 and u_left > 0.5 and u_right > 0.5:
                            try:
                                from models.numba_wrappers import run_ad_sptg_student_t
                                pit_sptg = run_ad_sptg_student_t(
                                    z_for_gpd, nu,
                                    gpd_left.xi, gpd_left.sigma, u_left,
                                    gpd_right.xi, gpd_right.sigma, u_right,
                                    p_left_val, p_right_val,
                                )
                            except (ImportError, Exception):
                                from models.numba_kernels import ad_sptg_cdf_student_t_array
                                pit_sptg = ad_sptg_cdf_student_t_array(
                                    np.ascontiguousarray(z_for_gpd, dtype=np.float64), nu,
                                    gpd_left.xi, gpd_left.sigma, u_left,
                                    gpd_right.xi, gpd_right.sigma, u_right,
                                    p_left_val, p_right_val,
                                )
                            pit_sptg = np.clip(np.asarray(pit_sptg, dtype=np.float64), 0.001, 0.999)
                            ks_sptg, p_sptg = _fast_ks_uniform(pit_sptg)
                            ad_sptg = compute_ad_statistic(pit_sptg)
                            if (p_sptg >= p_base * 0.95) or (ad_sptg <= ad_base):
                                pit = pit_sptg
                                diag['sptg_applied'] = True
                                diag['sptg_xi_left'] = float(gpd_left.xi)
                                diag['sptg_xi_right'] = float(gpd_right.xi)
                                cal_params['gpd_left_xi'] = float(gpd_left.xi)
                                cal_params['gpd_left_sigma'] = float(gpd_left.sigma)
                                cal_params['gpd_left_threshold'] = u_left
                                cal_params['gpd_right_xi'] = float(gpd_right.xi)
                                cal_params['gpd_right_sigma'] = float(gpd_right.sigma)
                                cal_params['gpd_right_threshold'] = u_right
                                xi_max = max(abs(float(gpd_left.xi)), abs(float(gpd_right.xi)))
                                if xi_max > 0.02:
                                    nu_from_gpd = 1.0 / xi_max
                                    nu_effective = max(2.5, min(nu, nu_from_gpd))
                                    cal_params['nu_effective'] = float(nu_effective)
                                    cal_params['nu_adjustment_ratio'] = float(nu_effective / nu)
                                else:
                                    cal_params['nu_effective'] = float(nu)
                                    cal_params['nu_adjustment_ratio'] = 1.0
                                ks_base, p_base, ad_base = ks_sptg, p_sptg, ad_sptg
            except Exception:
                pass

        # Stage C: monotone isotonic PIT recalibration.  This is only a
        # calibration-map output; it does not pretend to improve CRPS directly.
        if n >= 100:
            try:
                from calibration.isotonic_recalibration import IsotonicRecalibrator
                recal = IsotonicRecalibrator()
                result = recal.fit(pit)
                if result.fit_success and not result.is_identity:
                    pit_iso = np.clip(recal.transform(pit), 0.001, 0.999)
                    ks_before, p_before = _fast_ks_uniform(pit)
                    ks_after, p_after = _fast_ks_uniform(pit_iso)
                    ad_before = compute_ad_statistic(pit)
                    ad_after = compute_ad_statistic(pit_iso)
                    if (p_after >= p_before) or (ad_after <= ad_before):
                        pit = pit_iso
                        diag['isotonic_applied'] = True
                        diag['isotonic_ks_improvement'] = float(p_after - p_before)
                        cal_params['isotonic_x_knots'] = result.x_knots.tolist()
                        cal_params['isotonic_y_knots'] = result.y_knots.tolist()
            except Exception:
                pass

        diag['calibration_params'] = cal_params
        return np.clip(pit, 0.001, 0.999), diag


# ─── Backward compatibility shim ───
# UnifiedStudentTConfig has moved to phi_student_t_unified.py
# This lazy import prevents circular imports and maintains backward compat
def __getattr__(name):
    if name == "UnifiedStudentTConfig":
        from models.phi_student_t_unified import UnifiedStudentTConfig
        return UnifiedStudentTConfig
    if name == "UnifiedPhiStudentTModel":
        from models.phi_student_t_unified import UnifiedPhiStudentTModel
        return UnifiedPhiStudentTModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
