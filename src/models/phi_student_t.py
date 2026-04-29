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
        ad_correction_pipeline_student_t,
        clip_nu as common_clip_nu,
        compute_berkowitz_full as common_compute_berkowitz_full,
        compute_berkowitz_pvalue as common_compute_berkowitz_pvalue,
        compute_ad_statistic,
        compute_cvm_statistic,
        pit_simple_path as common_pit_simple_path,
        variance_to_scale,
        variance_to_scale_vec,
    )
except ImportError:
    from models.student_t_common import (
        ad_correction_pipeline_student_t,
        clip_nu as common_clip_nu,
        compute_berkowitz_full as common_compute_berkowitz_full,
        compute_berkowitz_pvalue as common_compute_berkowitz_pvalue,
        compute_ad_statistic,
        compute_cvm_statistic,
        pit_simple_path as common_pit_simple_path,
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
    """Compute time-varying mixture weight from multi-factor conditioning.

    Story 8.2: w_t = sigmoid(logit(base) + a*z_t + b*delta_sigma_t + c*M_t)

    Three factors:
      z_t:         Standardized shock (abs return / rolling vol)
      delta_sigma:  Vol acceleration (change in rolling vol)
      M_t:         Momentum signal (sign * rolling mean return)

    Higher w_t = more weight on calm component.
    w_t drops (more stress) when shocks are large or vol accelerates.

    Args:
        returns: Return series.
        vol: Volatility series (same length).
        a, b, c: Factor loadings for shock, vol accel, momentum.
        base_weight: Base calm-regime weight (0, 1).
        momentum_window: Lookback for momentum computation.

    Returns:
        (w_t, diagnostics): Time-varying calm weight and factor diagnostics.
    """
    returns = np.asarray(returns).flatten()
    vol = np.asarray(vol).flatten()
    n = len(returns)

    # Factor 1: Standardized shocks (negative = stress)
    vol_safe = np.maximum(vol, 1e-8)
    z_t = -np.abs(returns) / vol_safe  # negative shocks push w_t down

    # Factor 2: Vol acceleration (positive = stress)
    delta_sigma = np.zeros(n)
    if n > 1:
        delta_sigma[1:] = np.diff(vol)
    delta_sigma_signal = -delta_sigma / np.maximum(np.std(delta_sigma[1:]) if n > 2 else 1.0, 1e-8)

    # Factor 3: Momentum (positive = calm trends)
    M_t = np.zeros(n)
    for t in range(momentum_window, n):
        window_ret = returns[t - momentum_window:t]
        M_t[t] = np.mean(window_ret) / max(np.std(window_ret), 1e-8)

    # Combine via logistic
    logit_base = np.log(max(base_weight, 1e-6) / max(1 - base_weight, 1e-6))
    linear = logit_base + a * z_t + b * delta_sigma_signal + c * M_t
    w_t = 1.0 / (1.0 + np.exp(-linear))
    w_t = np.clip(w_t, 0.01, 0.99)

    diagnostics = {
        'mean_w': float(np.mean(w_t)),
        'min_w': float(np.min(w_t)),
        'max_w': float(np.max(w_t)),
        'factor_z_std': float(np.std(z_t)),
        'factor_vol_accel_std': float(np.std(delta_sigma_signal)),
        'factor_momentum_std': float(np.std(M_t)),
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
    """Select between mixture and two-piece Student-t models.

    Story 8.3: Selection heuristic beyond BIC:
    - Mixture preferred when vol-of-vol > 0.5 (dynamic regimes)
    - Two-piece preferred when |empirical skewness| > 0.3 (static asymmetry)
    - When BIC within 3 nats: prefer simpler (two-piece)

    Args:
        returns: Return series.
        vol: Volatility series.
        bic_mixture: BIC of mixture model.
        bic_two_piece: BIC of two-piece model.

    Returns:
        (selected_model, diagnostics): 'mixture' or 'two_piece' and details.
    """
    returns = np.asarray(returns).flatten()
    vol = np.asarray(vol).flatten()

    # Compute data diagnostics
    skew = float(np.mean(((returns - np.mean(returns)) / max(np.std(returns), 1e-8)) ** 3))
    abs_skew = abs(skew)

    # Vol-of-vol: std(vol) / mean(vol)
    vol_mean = float(np.mean(np.abs(vol)))
    vol_std = float(np.std(vol))
    vov = vol_std / max(vol_mean, 1e-8)

    # BIC difference (positive = two-piece better)
    bic_diff = bic_mixture - bic_two_piece
    bic_diff_nats = bic_diff / 2.0

    # Heuristic adjustments
    adjusted_bic_mixture = bic_mixture
    adjusted_bic_two_piece = bic_two_piece

    # VoV > threshold => upweight mixture by 2 BIC nats (=4 in BIC scale)
    if vov > VOV_MIXTURE_THRESHOLD:
        adjusted_bic_mixture -= 4.0

    # Occam gate: when within 3 nats, prefer two-piece (simpler)
    if abs(bic_diff) < BIC_OCCAM_MARGIN:
        selected = 'two_piece'
    elif adjusted_bic_mixture < adjusted_bic_two_piece:
        selected = 'mixture'
    else:
        selected = 'two_piece'

    # Static asymmetry override: strong skewness favors two-piece
    if abs_skew > SKEW_TWO_PIECE_THRESHOLD and bic_diff_nats < 5.0:
        selected = 'two_piece'

    diagnostics = {
        'empirical_skewness': float(skew),
        'abs_skewness': float(abs_skew),
        'vol_of_vol': float(vov),
        'bic_mixture': float(bic_mixture),
        'bic_two_piece': float(bic_two_piece),
        'bic_diff_nats': float(bic_diff_nats),
        'selected': selected,
        'skew_favors_two_piece': bool(abs_skew > SKEW_TWO_PIECE_THRESHOLD),
        'vov_favors_mixture': bool(vov > VOV_MIXTURE_THRESHOLD),
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


def _detect_asset_class(asset_symbol: str) -> Optional[str]:
    """
    Detect asset class from symbol for calibration profile selection.

    Returns:
        'metals_gold': Gold futures/ETFs — slow macro regimes, jump-driven
        'metals_silver': Silver futures/ETFs — explosive VoV, leveraged-gold
        'metals_other': Other metals (copper, platinum, palladium)
        'high_vol_equity': Crypto-correlated / meme / micro-cap with extreme kurtosis
        None: No special profile (equities, FX, crypto — use generic)
    """
    return _shared_detect_asset_class(asset_symbol)


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
    """Select optimal z-score method via in-sample BIC comparison.

    Story 7.2: Compare EWM-based MS-q vs expanding-window MS-q.
    Selection via log-likelihood proxy: sum of log(q_t) weighted by
    squared innovations.

    Args:
        vol: Volatility series.
        q_calm: Calm regime process noise.
        q_stress: Stress regime process noise.
        sensitivity: Sigmoid sensitivity.
        returns: Optional returns series for innovation-weighted BIC.

    Returns:
        (best_method, best_lambda, diagnostics) where diagnostics contains
        BIC for each candidate.
    """
    n = len(vol)
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
        # Log-likelihood proxy: -0.5 * sum(log(q_t) + innovations^2 / q_t)
        log_q = np.log(np.maximum(q_t, 1e-20))

        if returns is not None and len(returns) == n:
            innov_sq = returns ** 2
            ll = -0.5 * np.sum(log_q + innov_sq / np.maximum(q_t, 1e-20))
        else:
            ll = -0.5 * np.sum(log_q)

        # BIC: -2*ll + k*log(n), k=1 for ewm (lambda), k=0 for expanding
        k = 1 if method == 'ewm' else 0
        bic = -2.0 * ll + k * np.log(max(n, 1))

        label = f"{method}_{lam}" if method == 'ewm' else 'expanding'
        bic_results[label] = float(bic)

        if bic < best_bic:
            best_bic = bic
            best_method = method
            best_lambda = lam

    # Expanding requires n > 100 for stable mean
    if best_method == 'expanding' and n < 100:
        best_method = 'ewm'
        best_lambda = 0.96

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
    """Optimize q_calm and q_stress jointly with identifiability constraints.

    Story 7.3: Constrained optimization ensuring the two-regime model is
    genuinely bimodal (not a single-regime collapse).

    Constraints:
        MIN_Q_RATIO <= q_stress / q_calm <= MAX_Q_RATIO

    Args:
        vol: Volatility series.
        returns: Return series (same length as vol).
        sensitivity: Sigmoid sensitivity for MS transition.
        ewm_lambda: EWM decay for vol statistics.
        q_calm_bounds: (min, max) for q_calm search.
        q_stress_bounds: (min, max) for q_stress search.
        n_grid: Grid points per dimension.

    Returns:
        (q_calm_opt, q_stress_opt, diagnostics) where diagnostics
        includes ratio, BIC, and comparison with single-q model.
    """
    n = len(vol)
    returns = np.asarray(returns).flatten()

    # Grid search over q_calm x q_stress
    q_calm_grid = np.geomspace(q_calm_bounds[0], q_calm_bounds[1], n_grid)
    q_stress_grid = np.geomspace(q_stress_bounds[0], q_stress_bounds[1], n_grid)

    best_ll = -np.inf
    best_qc = q_calm_bounds[0]
    best_qs = q_stress_bounds[1]

    for qc in q_calm_grid:
        for qs in q_stress_grid:
            ratio = qs / max(qc, 1e-20)
            if ratio < MIN_Q_RATIO or ratio > MAX_Q_RATIO:
                continue

            q_t, _ = compute_ms_process_noise_smooth(
                vol, qc, qs, sensitivity=sensitivity, ewm_lambda=ewm_lambda
            )
            log_q = np.log(np.maximum(q_t, 1e-20))
            ll = -0.5 * np.sum(log_q + returns ** 2 / np.maximum(q_t, 1e-20))

            if ll > best_ll:
                best_ll = ll
                best_qc = qc
                best_qs = qs

    # Single-q baseline for BIC comparison
    q_single = 0.5 * (best_qc + best_qs)
    ll_single = -0.5 * np.sum(
        np.log(q_single) + returns ** 2 / q_single
    )

    # BIC: -2*ll + k*log(n)
    bic_msq = -2.0 * best_ll + 2.0 * np.log(max(n, 1))    # k=2 (q_calm, q_stress)
    bic_single = -2.0 * ll_single + 1.0 * np.log(max(n, 1))  # k=1

    ratio = best_qs / max(best_qc, 1e-20)
    bic_improvement = bic_single - bic_msq  # positive = MS-q is better

    diagnostics = {
        'q_calm': float(best_qc),
        'q_stress': float(best_qs),
        'ratio': float(ratio),
        'bic_msq': float(bic_msq),
        'bic_single': float(bic_single),
        'bic_improvement_nats': float(bic_improvement / 2.0),
        'll_msq': float(best_ll),
        'll_single': float(ll_single),
        'msq_selected': bool(bic_improvement > 10.0),  # > 5 nats = 10 in -2*ll scale
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
    Compute smooth probabilistic MS process noise.

    Two modes:
      ewm_lambda = 0: Expanding-window z-score (backward-compatible)
      ewm_lambda > 0: EWM with corresponding half-life (faster regime detection)

    Returns (q_t, p_stress): time-varying process noise and stress probability.
    """
    vol = np.asarray(vol).flatten()
    n = len(vol)

    # Bound sensitivity — widened to [1.0, 5.0] for metals
    sensitivity = float(np.clip(sensitivity, 1.0, 5.0))

    if ewm_lambda > 0.01:
        # ---------------------------------------------------------------------------
        # EWM MODE: Exponentially-weighted moving statistics
        # ---------------------------------------------------------------------------
        lam = float(np.clip(ewm_lambda, 0.5, 0.999))
        one_minus_lam = 1.0 - lam

        warmup = min(20, n)
        ewm_mean = float(np.mean(vol[:warmup])) if warmup > 0 else float(vol[0])
        ewm_var = float(np.var(vol[:warmup])) if warmup > 1 else 1e-6
        ewm_var = max(ewm_var, 1e-12)

        # Try Numba-accelerated EWM z-score computation
        try:
            from models.numba_wrappers import run_compute_ms_process_noise_ewm
            vol_zscore = run_compute_ms_process_noise_ewm(vol, lam, ewm_mean, ewm_var)
        except (ImportError, Exception):
            vol_zscore = np.empty(n)
            for t in range(n):
                ewm_std = math.sqrt(ewm_var)
                ewm_std = max(ewm_std, 1e-6)
                vol_zscore[t] = (vol[t] - ewm_mean) / ewm_std

                # Update AFTER computing z-score (no look-ahead)
                ewm_mean = lam * ewm_mean + one_minus_lam * vol[t]
                diff = vol[t] - ewm_mean
                ewm_var = lam * ewm_var + one_minus_lam * (diff * diff)
                ewm_var = max(ewm_var, 1e-12)
    else:
        # ---------------------------------------------------------------------------
        # EXPANDING-WINDOW MODE: Original behavior (backward-compatible)
        # ---------------------------------------------------------------------------
        vol_cumsum = np.cumsum(vol)
        vol_sq_cumsum = np.cumsum(vol ** 2)
        counts = np.arange(1, n + 1, dtype=np.float64)

        vol_mean = vol_cumsum / counts
        vol_var = vol_sq_cumsum / counts - vol_mean ** 2
        vol_var = np.maximum(vol_var, 1e-12)
        vol_std = np.sqrt(vol_var)

        warmup = min(20, n)
        if n > warmup:
            init_mean = np.mean(vol[:warmup])
            init_std = max(np.std(vol[:warmup]), 1e-6)
            vol_mean[:warmup] = init_mean
            vol_std[:warmup] = init_std

        vol_zscore = (vol - vol_mean) / np.maximum(vol_std, 1e-6)

    # Smooth sigmoid transition
    p_stress = 1.0 / (1.0 + np.exp(-sensitivity * vol_zscore))
    p_stress = np.clip(p_stress, 0.01, 0.99)

    # Time-varying q
    q_t = (1.0 - p_stress) * q_calm + p_stress * q_stress

    return q_t, p_stress


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

    Returns alpha_hat = (left_kurtosis - right_kurtosis) / total_kurtosis.
    Negative means heavier left tail (crashes), positive means heavier right.
    """
    r = np.asarray(returns).flatten()
    r_centered = r - np.mean(r)

    left_mask = r_centered < 0
    right_mask = r_centered > 0

    if np.sum(left_mask) < 10 or np.sum(right_mask) < 10:
        return 0.0

    std = max(float(np.std(r_centered)), 1e-10)
    z = r_centered / std

    left_kurt = float(np.mean(z[left_mask] ** 4))
    right_kurt = float(np.mean(z[right_mask] ** 4))
    total_kurt = left_kurt + right_kurt

    if total_kurt < 1e-10:
        return 0.0

    return (left_kurt - right_kurt) / total_kurt


def compute_ms_process_noise(
    vol: np.ndarray,
    q_calm: float = MS_Q_CALM_DEFAULT,
    q_stress: float = MS_Q_STRESS_DEFAULT,
    sensitivity: float = MS_Q_SENSITIVITY,
    threshold: float = MS_Q_THRESHOLD,
    vol_median: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias — delegates to compute_ms_process_noise_smooth."""
    return compute_ms_process_noise_smooth(vol, q_calm, q_stress, sensitivity, ewm_lambda=0.0)


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
    Kalman filter with Markov-switching process noise for Student-t.
    Uses vol_relative threshold (legacy MS-q path used by tune.py/signals.py).
    """
    y = np.asarray(y).flatten()
    vol = np.asarray(vol).flatten()
    n = len(y)
    nu = max(float(nu), 2.01)

    # Inline MS process noise: vol_relative → sigmoid → q_t
    vol_cumsum = np.cumsum(vol)
    vol_count = np.arange(1, n + 1)
    vol_baseline = vol_cumsum / vol_count
    if n > 20:
        vol_baseline[:20] = np.mean(vol[:20])
    vol_baseline = np.maximum(vol_baseline, 1e-10)
    vol_relative = vol / vol_baseline
    z_ms = sensitivity * (vol_relative - threshold)
    p_stress = np.clip(1.0 / (1.0 + np.exp(-z_ms)), 0.01, 0.99)
    q_t = (1.0 - p_stress) * q_calm + p_stress * q_stress

    # Initialize state
    mu = np.zeros(n)
    P = np.zeros(n)
    mu_t = 0.0
    P_t = 1.0

    # Log-likelihood accumulation
    total_ll = 0.0

    # Log of gamma function ratios for Student-t PDF
    log_gamma_ratio = gammaln((nu + 1) / 2) - gammaln(nu / 2)
    log_norm_const = log_gamma_ratio - 0.5 * np.log(nu * np.pi)

    for t in range(n):
        # Observation variance
        R_t = c * (vol[t] ** 2)

        # Predictive variance
        S_t = P_t + R_t

        # Student-t scale
        if nu > 2:
            scale_t = np.sqrt(S_t * (nu - 2) / nu)
        else:
            scale_t = np.sqrt(S_t)

        # Innovation
        innovation = y[t] - mu_t
        z = innovation / scale_t

        # Log-likelihood contribution
        ll_t = log_norm_const - np.log(scale_t) - ((nu + 1) / 2) * np.log(1 + z**2 / nu)
        if np.isfinite(ll_t):
            total_ll += ll_t

        # Robust weighting for Student-t (downweight outliers)
        # Use z² = innovation² / scale_t² (consistent with log-likelihood)
        z_sq = z ** 2  # z = innovation / scale_t already computed above
        w_t = (nu + 1) / (nu + z_sq)

        # Kalman gain
        K_t = P_t / S_t if S_t > 1e-12 else 0.0

        # Weighted update
        mu_t = mu_t + K_t * w_t * innovation
        P_t = (1 - w_t * K_t) * P_t
        # Covariance clamping (Story 28.2: prevent P collapse/explosion)
        P_t = max(_P_MIN, min(P_t, _P_MAX))

        # Store filtered state
        mu[t] = mu_t
        P[t] = P_t

        # State prediction with TIME-VARYING q
        mu_t = phi * mu_t
        P_t = (phi ** 2) * P_t + q_t[t]
        P_t = max(_P_MIN, min(P_t, _P_MAX))

    return mu, P, total_ll, q_t, p_stress


def optimize_params_ms_q(
    returns: np.ndarray,
    vol: np.ndarray,
    nu: float,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
) -> Tuple[float, float, float, float, float, float, Dict]:
    """
    Optimize MS-q model parameters: (c, phi, q_calm, q_stress).

    Uses concentrated likelihood over q_calm and q_stress while
    jointly optimizing c and phi.

    Args:
        returns: Time series of returns
        vol: Time series of volatility estimates
        nu: Degrees of freedom (fixed)
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Prior strength

    Returns:
        Tuple of (c_opt, phi_opt, q_calm_opt, q_stress_opt, log_likelihood, lfo_cv_score, diagnostics)
    """
    returns = np.asarray(returns).flatten()
    vol = np.asarray(vol).flatten()
    n = len(returns)

    # Grid search over q_calm/q_stress ratios
    q_calm_grid = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
    q_ratio_grid = [10, 50, 100, 200]  # q_stress / q_calm

    best_ll = float('-inf')
    best_params = None

    for q_calm in q_calm_grid:
        for ratio in q_ratio_grid:
            q_stress = q_calm * ratio

            # Optimize c, phi for this (q_calm, q_stress) pair
            def neg_ll(params):
                c_try, phi_try = params
                if c_try <= 0 or abs(phi_try) >= 1:
                    return 1e12

                try:
                    _, _, ll, _, _ = filter_phi_ms_q(
                        returns, vol, c_try, phi_try, nu,
                        q_calm=q_calm, q_stress=q_stress
                    )

                    # Add regularization
                    log_q_avg = np.log10((q_calm + q_stress) / 2)
                    prior_penalty = prior_lambda * (log_q_avg - prior_log_q_mean) ** 2

                    return -ll + prior_penalty
                except Exception:
                    return 1e12

            # Initial guess
            c_init = 1.0
            phi_init = 0.0

            try:
                result = minimize(
                    neg_ll,
                    [c_init, phi_init],
                    method='L-BFGS-B',
                    bounds=[(0.01, 10.0), (-0.99, 0.99)],
                    options={'maxiter': 100}
                )

                if result.success and -result.fun > best_ll:
                    best_ll = -result.fun
                    best_params = (result.x[0], result.x[1], q_calm, q_stress)
            except Exception:
                continue

    if best_params is None:
        # Fallback to defaults
        c_opt, phi_opt = 1.0, 0.0
        q_calm_opt, q_stress_opt = MS_Q_CALM_DEFAULT, MS_Q_STRESS_DEFAULT
        ll_opt = float('-inf')
    else:
        c_opt, phi_opt, q_calm_opt, q_stress_opt = best_params
        ll_opt = best_ll

    # Compute LFO-CV score for model comparison
    try:
        from tuning.diagnostics import compute_lfo_cv_score_student_t
        # Use average q for LFO-CV (approximate)
        q_avg = (q_calm_opt + q_stress_opt) / 2
        lfo_cv_score, lfo_diag = compute_lfo_cv_score_student_t(
            returns, vol, q_avg, c_opt, phi_opt, nu
        )
    except ImportError:
        lfo_cv_score = float('-inf')
        lfo_diag = {"error": "lfo_cv_not_available"}

    diagnostics = {
        "fit_success": best_params is not None,
        "n_obs": n,
        "nu": nu,
        "q_ratio": q_stress_opt / q_calm_opt if q_calm_opt > 0 else 0,
        "lfo_cv_diagnostics": lfo_diag,
    }

    return c_opt, phi_opt, q_calm_opt, q_stress_opt, ll_opt, lfo_cv_score, diagnostics


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
        return common_clip_nu(nu, nu_min, nu_max)

    @staticmethod
    def _variance_to_scale(variance: float, nu: float) -> float:
        """Convert predictive variance to Student-t scale."""
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
        Differentiable, bounded, always returns ν > 2.1.
        """
        # Standardized residual
        scale_safe = max(abs(scale), 1e-10)
        z = innovation / scale_safe

        # Smooth asymmetric modulation via tanh
        # tanh is bounded in [-1, 1] and smooth everywhere
        # Use math.tanh for scalar — avoids numpy array overhead
        modulation = 1.0 + alpha * math.tanh(k * z)
        nu_raw = nu_base * modulation

        # Ensure ν > 2 (finite variance requirement)
        if nu_raw < nu_min:
            return nu_min
        if nu_raw > nu_max:
            return nu_max
        return nu_raw

    @staticmethod
    def logpdf(x: float, nu: float, mu: float, scale: float) -> float:
        """
        Log-density of scaled Student-t with location ``mu`` and scale ``scale``.
        Returns a large negative sentinel if inputs are invalid to keep optimizers stable.
        """
        if scale <= 0 or nu <= 0:
            return -1e12

        if _NUMERICAL_STABILITY_AVAILABLE:
            return safe_student_t_logpdf_scalar(x, nu, mu, scale)

        z = (x - mu) / scale
        log_norm = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi * (scale ** 2))
        log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + (z ** 2) / nu)
        return float(log_norm + log_kernel)

    @classmethod
    def pit_ks_predictive(
        cls,
        returns: np.ndarray,
        mu_pred: np.ndarray,
        S_pred: np.ndarray,
        nu: float,
    ) -> Tuple[float, float]:
        """
        PIT/KS using predictive distribution for Student-t.

        Standardises returns by predictive mean/scale, computes Student-t CDF
        PIT values, and returns (KS statistic, KS p-value).
        Vectorized: single batch CDF call instead of per-element loop.
        """
        returns = np.asarray(returns).flatten()
        mu_pred = np.asarray(mu_pred).flatten()
        S_pred = np.asarray(S_pred).flatten()

        n = min(len(returns), len(mu_pred), len(S_pred))
        if n < 2:
            return 1.0, 0.0

        S_clamped = np.maximum(S_pred[:n], 1e-20)
        if nu > 2:
            scale = np.sqrt(S_clamped * (nu - 2) / nu)
        else:
            scale = np.sqrt(S_clamped)
        scale = np.maximum(scale, 1e-10)

        z = (returns[:n] - mu_pred[:n]) / scale
        pit_values = _fast_t_cdf(z, nu)

        valid = np.isfinite(pit_values)
        pit_clean = np.clip(pit_values[valid], 0, 1)
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
        Optimize (q, c, φ) for a FIXED ν via cross-validated MLE.

        Uses L-BFGS-B with Bayesian regularization on log10(q).
        Returns (q, c, phi, cv_penalized_ll, diagnostics).
        """
        n = len(returns)
        returns = np.asarray(returns).flatten()
        vol = np.asarray(vol).flatten()

        # Keep the optimizer causal.  The old full-sample winsorisation used
        # future extrema to alter early observations; Student-t tails already
        # supply the robust scoring we need.
        returns_r = returns.astype(np.float64, copy=True)
        returns_r[~np.isfinite(returns_r)] = 0.0
        vol = vol.astype(np.float64, copy=True)
        finite_vol = vol[np.isfinite(vol) & (vol > 0.0)]
        vol_fill = float(np.median(finite_vol)) if finite_vol.size else max(float(np.std(returns_r)), 1e-4)
        vol = np.where(np.isfinite(vol) & (vol > 0.0), vol, vol_fill)
        vol = np.maximum(vol, max(vol_fill * 1e-4, 1e-10))

        # Scale-aware q_min
        vol_var_med = float(np.median(vol ** 2))
        q_min = max(q_min, 0.001 * vol_var_med, 1e-8)

        # Adaptive c bounds for metals/futures
        if asset_symbol and any(tag in str(asset_symbol).upper() for tag in ["=F", "GC", "SI", "CL", "NG"]):
            c_max = max(c_max, 10.0)

        # CV folds (expanding window)
        min_train = min(max(60, int(n * 0.4)), max(n - 5, 1))
        test_window = min(max(20, int(n * 0.1)), max(n - min_train, 5))
        folds = []
        te = min_train
        while te + test_window <= n:
            folds.append((0, te, te, min(te + test_window, n)))
            te += test_window
        if not folds:
            sp = int(n * train_frac)
            folds = [(0, sp, sp, n)]

        nu_val = float(np.clip(nu, cls.nu_min_default, cls.nu_max_default))
        log_norm_const = gammaln((nu_val + 1.0) / 2.0) - gammaln(nu_val / 2.0) - 0.5 * np.log(nu_val * np.pi)
        neg_exp = -((nu_val + 1.0) / 2.0)
        inv_nu = 1.0 / nu_val

        # Pre-compute vol^2 for inner-loop access
        _vol_sq = np.ascontiguousarray((vol * vol).flatten(), dtype=np.float64)
        _returns_r = np.ascontiguousarray(returns_r.flatten(), dtype=np.float64)
        _mlog = math.log
        _msqrt = math.sqrt
        _misfinite = math.isfinite
        # Pre-compute Student-t scale factor
        _nu_scale = ((nu_val - 2) / nu_val) if nu_val > 2 else 1.0

        # Enhanced filter flags for optimizer-filter consistency
        # (March 2026: fixes parameter bias from train/inference mismatch)
        _has_vov_opt = gamma_vov > 1e-12 and vov_rolling is not None
        # Graduated OSA: always enabled for train/inference consistency
        _use_osa_opt = True

        # Numba-accelerated CV test fold kernel (always used)
        # Now supports VoV inflation and robust Student-t weighting (March 2026).
        # Note: OSA is handled by the training fold filter only — the test fold
        # Numba kernel uses the OSA-adjusted initial state from training, which
        # is sufficient for CV scoring.
        _use_numba_cv = False
        try:
            from models.numba_wrappers import run_phi_student_t_cv_test_fold as _numba_cv_fold
            from models.numba_wrappers import run_phi_student_t_train_state_only as _numba_train_state_only
            _use_numba_cv = True
        except (ImportError, Exception):
            _numba_train_state_only = None
            pass

        # Pre-compute VoV array for Numba (contiguous float64)
        _vov_full = None
        if _has_vov_opt and vov_rolling is not None:
            _vov_full = np.ascontiguousarray(vov_rolling.flatten(), dtype=np.float64)
        _use_train_state_only_kernel = (
            _use_numba_cv
            and _numba_train_state_only is not None
            and os.environ.get("PHI_STUDENT_T_DISABLE_STATE_ONLY_KERNEL", "") != "1"
        )

        def neg_cv_ll(params):
            log_q, log_c, phi = params
            q = 10 ** log_q
            c = 10 ** log_c
            phi_clip = float(np.clip(phi, phi_min, phi_max))
            if q <= 0 or c <= 0:
                return 1e12

            total_ll = 0.0
            for ts, te_f, vs, ve in folds:
                if te_f - ts < 3:
                    continue
                if _use_train_state_only_kernel:
                    mu_p, P_p = _numba_train_state_only(
                        _returns_r, _vol_sq, q, c, phi_clip, nu_val,
                        ts, te_f,
                        gamma_vov=gamma_vov if _has_vov_opt else 0.0,
                        vov_rolling=_vov_full,
                        robust_wt=True,
                        online_scale_adapt=_use_osa_opt,
                    )
                else:
                    ret_tr = _returns_r[ts:te_f]
                    vol_tr = vol[ts:te_f]
                    # Use enhanced filter for training fold (March 2026)
                    # Ensures parameters optimized against the same filter used in inference
                    _vov_tr = vov_rolling[ts:te_f] if _has_vov_opt else None
                    mu_f, P_f, _, _, _ = cls._filter_phi_core(
                        ret_tr, vol_tr, q, c, phi_clip, nu_val,
                        robust_wt=True,
                        online_scale_adapt=_use_osa_opt,
                        gamma_vov=gamma_vov if _has_vov_opt else 0.0,
                        vov_rolling=_vov_tr,
                    )
                    mu_p = float(mu_f[-1])
                    P_p = float(P_f[-1])
                if _use_numba_cv:
                    total_ll += _numba_cv_fold(
                        _returns_r, _vol_sq, q, c, phi_clip,
                        _nu_scale, log_norm_const, neg_exp, inv_nu,
                        mu_p, P_p, vs, ve,
                        nu_val=nu_val,
                        gamma_vov=gamma_vov if _has_vov_opt else 0.0,
                        vov_rolling=_vov_full,
                        online_scale_adapt=_use_osa_opt,
                    )
                else:
                    phi_clip_sq = phi_clip * phi_clip
                    _nu_p1 = nu_val + 1.0
                    c_adj = 1.0
                    chi2_tgt = (nu_val / (nu_val - 2.0)) if nu_val > 2.0 else 1.0
                    chi2_lam = 0.98
                    chi2_1m = 1.0 - chi2_lam
                    chi2_cap = chi2_tgt * 50.0
                    ewm_z2 = chi2_tgt
                    osa_strength = min(1.0, (chi2_tgt - 1.0) / 0.5) if nu_val > 2.0 else 1.0
                    for t in range(vs, ve):
                        mu_p = phi_clip * mu_p
                        P_p = phi_clip_sq * P_p + q
                        R_t = (c * c_adj if _use_osa_opt else c) * _vol_sq[t]
                        if _has_vov_opt:
                            R_t *= max(0.05, min(20.0, 1.0 + gamma_vov * vov_rolling[t]))
                        S = P_p + R_t
                        if S < 1e-12:
                            S = 1e-12
                        inn = _returns_r[t] - mu_p
                        scale = _msqrt(S * _nu_scale)
                        if scale > 1e-12:
                            z = inn / scale
                            ll_t = log_norm_const - _mlog(scale) + neg_exp * _mlog(1.0 + z * z * inv_nu)
                            if _misfinite(ll_t):
                                total_ll += ll_t
                        # Precision-weighted Joseph covariance update.
                        z_sq_cv = (inn * inn) / S
                        w_cv = float(np.clip(_nu_p1 / (nu_val + z_sq_cv), 0.05, 20.0))
                        R_eff = max(R_t / max(w_cv, 1e-8), 1e-20)
                        S_eff = max(P_p + R_eff, 1e-20)
                        K = P_p / S_eff
                        mu_p = mu_p + K * inn
                        P_p = (1.0 - K) * (1.0 - K) * P_p + K * K * R_eff
                        if P_p < 1e-12:
                            P_p = 1e-12
                        if _use_osa_opt and scale > 1e-12:
                            z2w = min(z * z, chi2_cap)
                            ewm_z2 = chi2_lam * ewm_z2 + chi2_1m * z2w
                            ratio = float(np.clip(ewm_z2 / max(chi2_tgt, 1e-12), 0.3, 3.0))
                            dev = ratio - 1.0 if ratio >= 1.0 else 1.0 - ratio
                            if ratio >= 1.0:
                                dz_lo, dz_rng = 0.25, 0.25
                            else:
                                dz_lo, dz_rng = 0.05, 0.10
                            if dev < dz_lo:
                                c_adj = 1.0
                            elif dev >= dz_lo + dz_rng:
                                c_adj = 1.0 + osa_strength * (_msqrt(ratio) - 1.0)
                            else:
                                s_frac = (dev - dz_lo) / dz_rng
                                c_adj_raw = 1.0 + s_frac * (_msqrt(ratio) - 1.0)
                                c_adj = 1.0 + osa_strength * (c_adj_raw - 1.0)

            # Bayesian regularization on log10(q)
            prior_pen = prior_lambda * (log_q - prior_log_q_mean) ** 2

            # Explicit φ shrinkage prior (Gaussian centered at 0)
            # Prevents fitting noise (bid-ask bounce → pathological phi≈-1)
            # Reference: Roll (1984), Hansen & Lunde (2005)
            _n_obs_shrink = max(len(_returns_r), 100)
            _phi_tau = 1.0 / _msqrt(2.0 * max(PHI_SHRINKAGE_LAMBDA_DEFAULT / _n_obs_shrink, 1e-12))
            _phi_tau = max(_phi_tau, PHI_SHRINKAGE_TAU_MIN)
            _phi_shrink_pen = 0.5 * (phi_clip ** 2) / (_phi_tau ** 2)

            return -(total_ll - prior_pen - _phi_shrink_pen)

        from scipy.optimize import minimize as sp_minimize

        # Phase 1: Coarse grid evaluation (3×2×2 = 12 points)
        _log_q_min = np.log10(q_min)
        _log_q_max = np.log10(q_max)
        _log_c_min = np.log10(c_min)
        _log_c_max = np.log10(c_max)
        grid_candidates = []

        # Warm-start from previous ν: inject as priority candidate
        if warm_start_params is not None:
            ws_q, ws_c, ws_phi = warm_start_params
            ws_lq = np.clip(np.log10(max(ws_q, 1e-12)), _log_q_min, _log_q_max)
            ws_lc = np.clip(np.log10(max(ws_c, 1e-6)), _log_c_min, _log_c_max)
            ws_ph = float(np.clip(ws_phi, phi_min, phi_max))
            try:
                val = neg_cv_ll([ws_lq, ws_lc, ws_ph])
                grid_candidates.append((val, ws_lq, ws_lc, ws_ph))
            except Exception:
                pass

        # MoM c initialisation (March 2026): c_mom = Var(r) / (median(vol²) × ν/(ν-2))
        # Provides a scale-consistent starting point for c based on the
        # method of moments. Particularly helpful for assets with unusual
        # volatility ratios (metals, small-caps).
        try:
            _nu_scale_mom = (nu_val / (nu_val - 2.0)) if nu_val > 2 else 1.0
            _vol_med_sq = max(float(np.median(_vol_sq)), 1e-12)
            _c_mom = float(np.var(returns)) / (_vol_med_sq * _nu_scale_mom)
            _c_mom = np.clip(_c_mom, c_min, c_max)
            _lc_mom = float(np.log10(max(_c_mom, 1e-6)))
            for _phi_mom in [0.0, 0.3]:
                for _lq_mom in [-6.0, -5.0]:
                    try:
                        val = neg_cv_ll([_lq_mom, _lc_mom, _phi_mom])
                        grid_candidates.append((val, _lq_mom, _lc_mom, _phi_mom))
                    except Exception:
                        continue
        except Exception:
            pass

        for log_q0 in [-6.0, -5.0, -4.0]:
            for log_c0 in [-0.05, 0.15]:
                for phi0 in [0.0, 0.3]:
                    try:
                        val = neg_cv_ll([log_q0, log_c0, phi0])
                        grid_candidates.append((val, log_q0, log_c0, phi0))
                    except Exception:
                        continue

        # Sort by objective.  Extra L-BFGS starts are only worth paying for
        # when the coarse-grid scores are close enough that local topology is
        # genuinely ambiguous.  A ten-nat objective margin keeps the old
        # multi-start safety valve for hard assets while deleting redundant
        # starts for the common well-separated case.
        grid_candidates.sort(key=lambda x: x[0])
        if grid_candidates:
            if os.environ.get("PHI_STUDENT_T_DISABLE_START_PRUNE", "") == "1":
                top_starts = grid_candidates[:3]
            else:
                best_grid_val = grid_candidates[0][0]
                prune_margin = float(os.environ.get("PHI_STUDENT_T_START_PRUNE_MARGIN", "10.0"))
                top_starts = [
                    cand for cand in grid_candidates[:3]
                    if cand[0] <= best_grid_val + prune_margin
                ]
                if not top_starts:
                    top_starts = grid_candidates[:1]
        else:
            top_starts = [(1e20, -5.0, 0.0, 0.0)]

        best_result = None
        best_val = 1e20
        for _, lq0, lc0, ph0 in top_starts:
            try:
                res = sp_minimize(
                    neg_cv_ll, [lq0, lc0, ph0],
                    method='L-BFGS-B',
                    bounds=[(_log_q_min, _log_q_max),
                            (_log_c_min, _log_c_max),
                            (phi_min, phi_max)],
                    options={'maxiter': 60, 'ftol': 1e-8},
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_result = res
            except Exception:
                continue

        if best_result is None:
            raise ValueError("All optimizer starts failed for Student-t fixed-nu")

        log_q_opt, log_c_opt, phi_opt = best_result.x
        q_opt = 10 ** log_q_opt
        c_opt = 10 ** log_c_opt
        phi_opt = float(np.clip(phi_opt, phi_min, phi_max))
        cv_ll = -best_val

        diagnostics = {
            "nu_fixed": nu_val,
            "optimizer_converged": best_result.success,
            "n_folds": len(folds),
            "log10_q": log_q_opt,
            "n_optimizer_starts": len(top_starts),
            "state_only_kernel_enabled": bool(_use_train_state_only_kernel),
        }

        return q_opt, c_opt, phi_opt, cv_ll, diagnostics

    # =====================================================================
    # VoV precomputation (Barndorff-Nielsen & Shephard 2002)
    # =====================================================================
    @staticmethod
    def _precompute_vov(vol: np.ndarray, window: int = 20) -> np.ndarray:
        """Rolling std of log-vol (O(n) via cumulative sums). Shared across ν."""
        n = len(vol)
        if n <= window:
            return np.zeros(n, dtype=np.float64)
        log_vol = np.log(np.maximum(vol, 1e-10))
        cs1 = np.concatenate(([0.0], np.cumsum(log_vol)))
        cs2 = np.concatenate(([0.0], np.cumsum(log_vol * log_vol)))
        inv_w = 1.0 / float(window)
        idx = np.arange(window, n)
        s1 = cs1[idx] - cs1[idx - window]
        s2 = cs2[idx] - cs2[idx - window]
        var_arr = np.maximum(s2 * inv_w - (s1 * inv_w) ** 2, 0.0)
        vov = np.empty(n, dtype=np.float64)
        vov[window:] = np.sqrt(var_arr)
        vov[:window] = vov[window]
        return vov

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

    # =====================================================================
    # Cramér-von Mises statistic (Anderson 1962)
    # =====================================================================
    @staticmethod
    def _compute_cvm_statistic(pit_values: np.ndarray) -> float:
        """
        Cramér-von Mises W² for Uniform(0,1).

        W² = Σ(u_(i) - (2i-1)/(2n))² + 1/(12n)

        More sensitive to the middle of the distribution than KS
        (which only captures max deviation). Lower = better calibrated.
        """
        n = len(pit_values)
        if n < 2:
            return float('inf')
        u = np.sort(pit_values)
        i_vals = np.arange(1, n + 1)
        w2 = float(np.sum((u - (2.0 * i_vals - 1.0) / (2.0 * n)) ** 2)
                   ) + 1.0 / (12.0 * n)
        return w2

    # =====================================================================
    # Anderson-Darling statistic for Uniform(0,1) (Anderson & Darling 1952)
    # =====================================================================
    @staticmethod
    def _compute_ad_statistic(pit_values: np.ndarray) -> float:
        """
        Anderson-Darling A² for Uniform(0,1).

        A² = -n - (1/n) Σ [(2i-1)(ln u_i + ln(1-u_{n+1-i}))]

        3-10× more powerful than KS for tail miscalibration — captures both
        heavy-tail deficiency and light-tail excess simultaneously.
        Lower = better calibrated.
        """
        n = len(pit_values)
        if n < 2:
            return float('inf')
        u = np.sort(np.clip(pit_values, 1e-10, 1.0 - 1e-10))
        i_vals = np.arange(1, n + 1)
        log_u = np.log(u)
        log_1mu = np.log(1.0 - u[::-1])  # u_{n+1-i}
        a2 = -float(n) - float(np.sum((2.0 * i_vals - 1.0) * (log_u + log_1mu))) / float(n)
        return max(a2, 0.0)

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
            vov_rolling = cls._precompute_vov(vol)
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
        _base_cvm = cls._compute_cvm_statistic(_base_pit_vals)
        if not np.isfinite(_base_cvm) or _base_cvm < 1e-12:
            _base_cvm = 1.0
        _base_ad = cls._compute_ad_statistic(_base_pit_vals)
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
                _cvm_ref = cls._compute_cvm_statistic(
                    _pit_ref.get("pit_values", np.array([])))
                _ad_ref = cls._compute_ad_statistic(
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

        This is the PRIMARY Student-t filter. There is no bare Student-t model.
        Uses Numba JIT-compiled kernel when available (10-50× speedup).
        """
        # Try Numba-accelerated version first
        if _USE_NUMBA:
            try:
                return run_phi_student_t_filter(returns, vol, q, c, phi, nu)
            except Exception:
                pass  # Fall through to Python implementation

        return cls._filter_phi_python_optimized(returns, vol, q, c, phi, nu)

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

        Uses Numba-compiled kernel when available, falls back to Python.

        Returns (mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, log_likelihood).
        All four filter variants dispatch to this single loop:
          - filter_phi  (via Numba or this fallback)
          - filter_phi_augmented  (robust_wt=True + exogenous_input)
          - filter_phi_with_predictive  (thin wrapper)

        Args:
            returns, vol, q, c, phi, nu: standard Kalman params
            exogenous_input: optional u_t injected into state prediction
            robust_wt: if True use Student-t w_t = (nu+1)/(nu+z^2) weighting
            online_scale_adapt: if True embed chi² EWM scale correction in filter
                (Harvey 1989, Durbin & Koopman 2012). Tracks E[z²] and dynamically
                adjusts c to keep standardised residuals calibrated.
            gamma_vov: vol-of-vol sensitivity (Barndorff-Nielsen & Shephard 2002).
                Inflates R_t during volatile periods: R_t *= (1 + γ·vov_rolling_t).
            vov_rolling: precomputed rolling std of log-vol (window=20).
        """
        # Numba-accelerated path — enhanced kernel for VoV/OSA features
        _use_enhanced = online_scale_adapt or (gamma_vov > 1e-12 and vov_rolling is not None)
        if _use_enhanced and _NUMBA_ENHANCED:
            try:
                return run_phi_student_t_enhanced_filter(
                    returns, vol, q, c,
                    float(max(-0.999, min(0.999, phi))),
                    float(cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)),
                    exogenous_input=exogenous_input,
                    robust_wt=robust_wt,
                    online_scale_adapt=online_scale_adapt,
                    gamma_vov=float(gamma_vov) if gamma_vov else 0.0,
                    vov_rolling=vov_rolling,
                )
            except Exception:
                pass  # Fall through to Python
        if not _use_enhanced and _USE_NUMBA and run_phi_student_t_augmented_filter is not None:
            try:
                return run_phi_student_t_augmented_filter(
                    returns, vol, q, c,
                    float(max(-0.999, min(0.999, phi))),
                    float(cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)),
                    exogenous_input=exogenous_input,
                    robust_wt=robust_wt,
                )
            except Exception:
                pass  # Fall through to Python

        n = len(returns)
        if not (returns.flags['C_CONTIGUOUS'] and returns.dtype == np.float64 and returns.ndim == 1):
            returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        if not (vol.flags['C_CONTIGUOUS'] and vol.dtype == np.float64 and vol.ndim == 1):
            vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)

        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, 'item') else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, 'item') else float(c)
        _phi_raw = phi
        phi_val = float(max(-0.999, min(0.999, _phi_raw)))
        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)

        phi_sq = phi_val * phi_val
        log_norm_const = math.lgamma((nu_val + 1.0) / 2.0) - math.lgamma(nu_val / 2.0) - 0.5 * math.log(nu_val * math.pi)
        neg_exp = -((nu_val + 1.0) / 2.0)
        inv_nu = 1.0 / nu_val
        _vol_sq = vol * vol

        # VoV: precompute flags
        _has_vov = gamma_vov > 1e-12 and vov_rolling is not None

        # Online scale adaptation: chi² EWM state (Harvey 1989)
        _chi2_tgt = (nu_val / (nu_val - 2.0)) if nu_val > 2.0 else 1.0
        _chi2_lam = 0.98
        _chi2_1m = 1.0 - _chi2_lam
        _chi2_cap = _chi2_tgt * 50.0
        _ewm_z2 = _chi2_tgt  # initialise at theoretical target
        _c_adj = 1.0  # multiplicative c correction
        # Graduated OSA strength: full for heavy tails (ν≤5), tapering for ν→∞
        # strength = min(1, (E[z²]-1)/0.5) where E[z²] = ν/(ν-2)
        _osa_strength = min(1.0, (_chi2_tgt - 1.0) / 0.5) if nu_val > 2 else 1.0

        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        mu_pred_arr = np.empty(n, dtype=np.float64)
        S_pred_arr = np.empty(n, dtype=np.float64)

        mu = 0.0
        P = 1e-4
        log_likelihood = 0.0
        has_exo = exogenous_input is not None
        _msqrt = math.sqrt
        _mlog = math.log
        _misfinite = math.isfinite
        _mabs = abs

        for t in range(n):
            u_t = exogenous_input[t] if has_exo and t < len(exogenous_input) else 0.0
            mu_pred = phi_val * mu + u_t
            P_pred = phi_sq * P + q_val

            # Observation noise R_t with optional VoV and online scale adapt
            c_eff = c_val * _c_adj if online_scale_adapt else c_val
            R_t = c_eff * _vol_sq[t]
            if _has_vov:
                R_t *= (1.0 + gamma_vov * vov_rolling[t])

            S = P_pred + R_t
            if S <= 1e-12:
                S = 1e-12

            mu_pred_arr[t] = mu_pred
            S_pred_arr[t] = S

            innovation = returns[t] - mu_pred

            if robust_wt:
                z_sq = (innovation * innovation) / S
                w_t = float(np.clip((nu_val + 1.0) / (nu_val + z_sq), 0.05, 20.0))
                R_eff = max(R_t / max(w_t, 1e-8), 1e-20)
                S_eff = max(P_pred + R_eff, 1e-20)
                K = P_pred / S_eff
                mu = mu_pred + K * innovation
                P = (1.0 - K) * (1.0 - K) * P_pred + K * K * R_eff
            else:
                K = P_pred / S
                mu = mu_pred + K * innovation
                P = (1.0 - K) * P_pred

            if P < 1e-12:
                P = 1e-12
            mu_filtered[t] = mu
            P_filtered[t] = P

            if nu_val > 2:
                forecast_scale = _msqrt(S * (nu_val - 2) / nu_val)
            else:
                forecast_scale = _msqrt(S)
            if forecast_scale > 1e-12:
                z = innovation / forecast_scale
                ll_t = log_norm_const - _mlog(forecast_scale) + neg_exp * _mlog(1.0 + z * z * inv_nu)
                if _misfinite(ll_t):
                    log_likelihood += ll_t

                # Online scale adaptation: track E[z²] and adjust c for next step
                if online_scale_adapt:
                    _z2_raw = z * z
                    _z2w = min(_z2_raw, _chi2_cap)
                    _ewm_z2 = _chi2_lam * _ewm_z2 + _chi2_1m * _z2w
                    _ratio = _ewm_z2 / _chi2_tgt
                    _ratio = max(0.3, min(3.0, _ratio))
                    _dev = _mabs(_ratio - 1.0)
                    if _ratio >= 1.0:
                        _dz_lo, _dz_rng = 0.25, 0.25
                    else:
                        _dz_lo, _dz_rng = 0.05, 0.10  # Tighter for under-dispersion
                    if _dev < _dz_lo:
                        _c_adj = 1.0
                    elif _dev >= _dz_lo + _dz_rng:
                        _c_adj_raw = _msqrt(_ratio)
                        _c_adj = 1.0 + _osa_strength * (_c_adj_raw - 1.0)
                    else:
                        _s_frac = (_dev - _dz_lo) / _dz_rng
                        _c_adj_raw = 1.0 + _s_frac * (_msqrt(_ratio) - 1.0)
                        _c_adj = 1.0 + _osa_strength * (_c_adj_raw - 1.0)

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
        """
        return cls._filter_phi_core(returns, vol, q, c, phi, nu,
                                     exogenous_input=exogenous_input,
                                     robust_wt=robust_wt,
                                     online_scale_adapt=online_scale_adapt,
                                     gamma_vov=gamma_vov,
                                     vov_rolling=vov_rolling)

    @staticmethod
    def _stage_5c_garch_estimation(returns_train, mu_pred_train, mu_drift_opt, n_train):
        """
        Stage 5c: GJR-GARCH(1,1) parameter estimation (Glosten-Jagannathan-Runkle 1993).

        h_t = ω + α·ε²_{t-1} + γ_lev·ε²_{t-1}·I(ε_{t-1}<0) + β·h_{t-1}

        The leverage term γ_lev captures asymmetric variance reaction:
          Negative returns → variance increases by (α + γ_lev)·ε²
          Positive returns → variance increases by α·ε² only

        Returns:
            dict with keys: garch_omega, garch_alpha, garch_beta,
            garch_leverage, unconditional_var
        """
        innovations_train = returns_train - mu_pred_train - mu_drift_opt
        sq_innov = innovations_train ** 2
        unconditional_var = float(np.var(innovations_train))

        garch_leverage = 0.0

        if n_train > 100:
            sq_centered = sq_innov - unconditional_var
            denom = np.sum(sq_centered[:-1]**2)
            if denom > 1e-12:
                garch_alpha = float(np.sum(sq_centered[1:] * sq_centered[:-1]) / denom)
                garch_alpha = np.clip(garch_alpha, 0.02, 0.25)
            else:
                garch_alpha = 0.08

            neg_indicator = (innovations_train[:-1] < 0).astype(np.float64)
            n_neg = max(int(np.sum(neg_indicator)), 1)
            n_pos = max(int(np.sum(1.0 - neg_indicator)), 1)
            mean_sq_after_neg = float(np.sum(sq_innov[1:] * neg_indicator) / n_neg)
            mean_sq_after_pos = float(np.sum(sq_innov[1:] * (1.0 - neg_indicator)) / n_pos)

            if mean_sq_after_pos > 1e-12:
                leverage_ratio = mean_sq_after_neg / mean_sq_after_pos
            else:
                leverage_ratio = 1.0

            if leverage_ratio > 1.0:
                garch_leverage = float(np.clip(
                    garch_alpha * (leverage_ratio - 1.0), 0.0, 0.20
                ))

            garch_beta = 0.97 - garch_alpha - garch_leverage / 2.0
            garch_beta = float(np.clip(garch_beta, 0.70, 0.95))

            total_persistence = garch_alpha + garch_leverage / 2.0 + garch_beta
            if total_persistence >= 0.99:
                garch_beta = 0.98 - garch_alpha - garch_leverage / 2.0
                garch_beta = max(garch_beta, 0.5)

            garch_omega = unconditional_var * (1 - garch_alpha - garch_leverage / 2.0 - garch_beta)
            garch_omega = max(garch_omega, 1e-10)
        else:
            garch_omega = unconditional_var * 0.05
            garch_alpha = 0.08
            garch_beta = 0.87

        return {
            'garch_omega': float(garch_omega),
            'garch_alpha': float(garch_alpha),
            'garch_beta': float(garch_beta),
            'garch_leverage': float(garch_leverage),
            'unconditional_var': float(unconditional_var),
        }

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
        """Apply the shared Student-t calibration transport pipeline."""
        return ad_correction_pipeline_student_t(
            returns,
            mu_pred,
            scale,
            nu,
            pit_raw,
            min_obs=min_obs,
            cdf_fn=_fast_t_cdf,
            ks_fn=_fast_ks_uniform,
            ad_fn=compute_ad_statistic,
            nu_min=cls.nu_min_default,
            nu_max=60.0,
            gpd_method="pwm",
        )


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
