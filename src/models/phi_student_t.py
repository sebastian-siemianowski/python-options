"""
===============================================================================
PHI-STUDENT-T DRIFT MODEL — Kalman Filter with AR(1) Drift and Student-t Noise
===============================================================================

Implements a state-space model with AR(1) drift and heavy-tailed observation noise:

    State equation:    μ_t = φ·μ_{t-1} + w_t,  w_t ~ N(0, q)
    Observation:       r_t = μ_t + ε_t,         ε_t ~ Student-t(ν, 0, √(c)·σ_t)

Parameters:
    q:   Process noise variance (drift evolution uncertainty)
    c:   Observation noise scale (multiplier on EWMA variance)
    φ:   AR(1) persistence coefficient (φ=1 is random walk, φ=0 is mean reversion)
    ν:   Degrees of freedom (controls tail heaviness; ν→∞ approaches Gaussian)

DISCRETE ν GRID:
    Instead of continuously optimizing ν (which causes identifiability issues),
    we use a discrete grid: ν ∈ {4, 6, 8, 12, 20}
    Each ν value becomes a separate sub-model in Bayesian Model Averaging.

The model includes an explicit Gaussian shrinkage prior on φ:
    φ_r ~ N(φ_global, τ²)
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import kstest
from scipy.stats import norm
from scipy.stats import t as student_t

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
    )
    _USE_NUMBA = is_numba_available()
except ImportError:
    _USE_NUMBA = False
    run_phi_student_t_filter = None
    run_phi_student_t_filter_batch = None


# =============================================================================
# φ SHRINKAGE PRIOR CONSTANTS (self-contained, no external dependencies)
# =============================================================================

PHI_SHRINKAGE_TAU_MIN = 1e-3
PHI_SHRINKAGE_GLOBAL_DEFAULT = 0.0
PHI_SHRINKAGE_LAMBDA_DEFAULT = 0.05

# Discrete ν grid for Student-t models
STUDENT_T_NU_GRID = [4, 6, 8, 12, 20]


# =============================================================================
# ENHANCED STUDENT-T CONFIGURATION (February 2026)
# =============================================================================
# Three enhancements to improve Hyvarinen/PIT calibration:
#   1. Vol-of-Vol (VoV): R_t = c × σ² × (1 + γ × |Δlog(σ)|)
#   2. Two-Piece: Different νL (crash) vs νR (recovery) tails
#   3. Two-Component Mixture: Blend νcalm and νstress with dynamic weights
# =============================================================================

# Vol-of-Vol (VoV) Enhancement
GAMMA_VOV_GRID = [0.3, 0.5, 0.7]
VOV_BMA_PENALTY = 1.0

# Two-Piece Student-t Enhancement
NU_LEFT_GRID = [3, 4, 5]
NU_RIGHT_GRID = [8, 12, 20]
TWO_PIECE_BMA_PENALTY = 3.0

# Two-Component Mixture Student-t Enhancement
NU_CALM_GRID = [12, 20]
NU_STRESS_GRID = [4, 6]
MIXTURE_WEIGHT_DEFAULT = 0.8
MIXTURE_WEIGHT_K = 2.0  # Sigmoid sensitivity to vol_relative
MIXTURE_BMA_PENALTY = 4.0


# =============================================================================
# ELITE TUNING CONFIGURATION (v2.0 - February 2026)
# =============================================================================
# Plateau-optimal parameter selection with:
# - Directional curvature awareness (φ-q coupling more dangerous)
# - Ridge vs basin detection
# - Drift vs noise decomposition in coherence
# =============================================================================

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
    """Encapsulates Student-t heavy-tail logic so drift model behavior stays modular."""

    nu_min_default: float = 2.1
    nu_max_default: float = 30.0

    @staticmethod
    def _clip_nu(nu: float, nu_min: float, nu_max: float) -> float:
        return float(np.clip(float(nu), nu_min, nu_max))

    @staticmethod
    def logpdf(x: float, nu: float, mu: float, scale: float) -> float:
        """
        Log-density of scaled Student-t with location ``mu`` and scale ``scale``.
        Returns a large negative sentinel if inputs are invalid to keep optimizers stable.
        """
        if scale <= 0 or nu <= 0:
            return -1e12

        z = (x - mu) / scale
        log_norm = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi * (scale ** 2))
        log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + (z ** 2) / nu)
        return float(log_norm + log_kernel)

    @staticmethod
    def logpdf_two_piece(x: float, nu_left: float, nu_right: float, mu: float, scale: float) -> float:
        """
        Log-density of Two-Piece Student-t with different ν for left/right tails.
        
        Two-Piece Student-t (Fernández & Steel inspired):
            p(x) ∝ t(x; νL) if x < μ
            p(x) ∝ t(x; νR) if x ≥ μ
        
        This allows crash tails (νL small) to be heavier than recovery tails (νR larger).
        
        Args:
            x: Observation value
            nu_left: Degrees of freedom for x < μ (crash tail)
            nu_right: Degrees of freedom for x ≥ μ (recovery tail)
            mu: Location parameter
            scale: Scale parameter
            
        Returns:
            Log-density value
        """
        if scale <= 0 or nu_left <= 0 or nu_right <= 0:
            return -1e12
        
        z = (x - mu) / scale
        
        # Choose ν based on sign of innovation
        if z < 0:
            nu = nu_left
        else:
            nu = nu_right
        
        # Standard Student-t log-density with chosen ν
        log_norm = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi * (scale ** 2))
        log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + (z ** 2) / nu)
        
        # Normalization correction for two-piece (approximate - assumes similar scales)
        # True normalization: 2 / (1/σL + 1/σR) but for same scale, just log(2) - log(2) = 0
        return float(log_norm + log_kernel)

    @staticmethod
    def logpdf_mixture(x: float, nu_calm: float, nu_stress: float, w_calm: float, mu: float, scale: float) -> float:
        """
        Log-density of Two-Component Student-t mixture.
        
        Mixture model:
            p(x) = w_calm × t(x; νcalm) + (1 - w_calm) × t(x; νstress)
        
        This captures two curvature regimes in the central body:
            - Calm regime: lighter tails (νcalm > νstress)
            - Stress regime: heavier tails (νstress < νcalm)
        
        Args:
            x: Observation value
            nu_calm: Degrees of freedom for calm component (lighter tails)
            nu_stress: Degrees of freedom for stress component (heavier tails)
            w_calm: Weight on calm component ∈ [0, 1]
            mu: Location parameter
            scale: Scale parameter
            
        Returns:
            Log-density value
        """
        if scale <= 0 or nu_calm <= 0 or nu_stress <= 0:
            return -1e12
        
        w_calm = float(np.clip(w_calm, 0.001, 0.999))
        z = (x - mu) / scale
        
        # Compute both component log-densities
        def _t_logpdf(nu):
            log_norm = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi * (scale ** 2))
            log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + (z ** 2) / nu)
            return log_norm + log_kernel
        
        ll_calm = _t_logpdf(nu_calm)
        ll_stress = _t_logpdf(nu_stress)
        
        # Log-sum-exp for numerical stability
        # log(w1*exp(ll1) + w2*exp(ll2)) = ll_max + log(w1*exp(ll1-ll_max) + w2*exp(ll2-ll_max))
        ll_max = max(ll_calm, ll_stress)
        log_mix = ll_max + np.log(
            w_calm * np.exp(ll_calm - ll_max) + 
            (1 - w_calm) * np.exp(ll_stress - ll_max)
        )
        
        return float(log_mix) if np.isfinite(log_mix) else -1e12

    @classmethod
    def filter(cls, returns: np.ndarray, vol: np.ndarray, q: float, c: float, nu: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Kalman drift filter with Student-t observation noise (no AR persistence)."""
        n = len(returns)
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, 'item') else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, 'item') else float(c)
        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)

        mu = 0.0
        P = 1e-4
        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
        log_likelihood = 0.0

        for t in range(n):
            mu_pred = float(mu)
            P_pred = float(P) + q_val

            vol_t = vol[t]
            vol_scalar = float(vol_t) if np.ndim(vol_t) == 0 else float(vol_t.item())
            obs_scale = np.sqrt(c_val) * vol_scalar

            nu_adjust = min(nu_val / (nu_val + 3.0), 1.0)
            R = c_val * (vol_scalar ** 2)
            K = nu_adjust * P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0

            ret_t = returns[t]
            r_val = float(ret_t) if np.ndim(ret_t) == 0 else float(ret_t.item())
            innovation = r_val - mu_pred

            mu = float(mu_pred + K * innovation)
            P = float((1.0 - K) * P_pred)

            mu_filtered[t] = mu
            P_filtered[t] = P

            forecast_scale = np.sqrt(P_pred + R)
            if forecast_scale > 1e-12:
                ll_t = cls.logpdf(r_val, nu_val, mu_pred, forecast_scale)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t

        return mu_filtered, P_filtered, log_likelihood

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
    def _filter_phi_python_optimized(cls, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, nu: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Optimized pure Python φ-Student-t filter with reduced overhead.
        
        Performance optimizations (February 2026):
        - Pre-compute constants outside the loop (log_norm_const, phi_sq, nu_adjust, inv_nu)
        - Pre-compute R array once (c * vol**2)
        - Use np.empty instead of np.zeros
        - Inline logpdf calculation to avoid function call overhead
        - Ensure contiguous array access
        """
        n = len(returns)
        
        # Convert to contiguous float64 arrays once
        returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)
        
        # Extract scalar values once
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))
        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)
        
        # Pre-compute constants (computed once, used n times)
        phi_sq = phi_val * phi_val
        nu_adjust = min(nu_val / (nu_val + 3.0), 1.0)
        
        # Pre-compute log-pdf constants (avoids gammaln call in loop)
        log_norm_const = gammaln((nu_val + 1.0) / 2.0) - gammaln(nu_val / 2.0) - 0.5 * np.log(nu_val * np.pi)
        neg_exp = -((nu_val + 1.0) / 2.0)
        inv_nu = 1.0 / nu_val
        
        # Pre-compute R values (vectorized)
        R = c_val * (vol * vol)
        
        # Allocate output arrays
        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        
        # State initialization
        mu = 0.0
        P = 1e-4
        log_likelihood = 0.0
        
        # Main filter loop (optimized)
        for t in range(n):
            # Prediction step
            mu_pred = phi_val * mu
            P_pred = phi_sq * P + q_val
            
            # Observation update
            S = P_pred + R[t]
            if S <= 1e-12:
                S = 1e-12
            
            innovation = returns[t] - mu_pred
            K = nu_adjust * P_pred / S
            
            # State update
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12
            
            # Store filtered values
            mu_filtered[t] = mu
            P_filtered[t] = P
            
            # Inlined log-pdf calculation (avoids function call + gammaln per step)
            forecast_scale = np.sqrt(S)
            if forecast_scale > 1e-12:
                z = innovation / forecast_scale
                ll_t = log_norm_const - np.log(forecast_scale) + neg_exp * np.log(1.0 + z * z * inv_nu)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t
        
        return mu_filtered, P_filtered, float(log_likelihood)
    
    @classmethod
    def _filter_phi_python(cls, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, nu: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Pure Python implementation of φ-Student-t filter (for fallback and testing)."""
        # Delegate to optimized version
        return cls._filter_phi_python_optimized(returns, vol, q, c, phi, nu)

    @classmethod
    def _filter_phi_with_trajectory(cls, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, nu: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Pure Python φ-Student-t filter with per-timestep likelihood trajectory.
        
        Enables fold-aware CV likelihood slicing without re-execution.
        Returns (mu_filtered, P_filtered, total_log_likelihood, loglik_trajectory).
        """
        n = len(returns)
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))
        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)

        mu = 0.0
        P = 1e-4
        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
        loglik_trajectory = np.zeros(n)
        log_likelihood = 0.0

        for t in range(n):
            mu_pred = phi_val * mu
            P_pred = (phi_val ** 2) * P + q_val

            vol_t = vol[t]
            vol_scalar = float(vol_t) if np.ndim(vol_t) == 0 else float(vol_t.item())
            R = c_val * (vol_scalar ** 2)

            ret_t = returns[t]
            r_val = float(ret_t) if np.ndim(ret_t) == 0 else float(ret_t.item())
            innovation = r_val - mu_pred

            S = P_pred + R
            if S <= 1e-12:
                S = 1e-12
            nu_adjust = min(nu_val / (nu_val + 3.0), 1.0)
            K = nu_adjust * P_pred / S

            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12

            mu_filtered[t] = mu
            P_filtered[t] = P

            forecast_scale = np.sqrt(S)
            ll_t = 0.0
            if forecast_scale > 1e-12:
                ll_t = cls.logpdf(r_val, nu_val, mu_pred, forecast_scale)
                if not np.isfinite(ll_t):
                    ll_t = 0.0
            
            loglik_trajectory[t] = ll_t
            log_likelihood += ll_t

        return mu_filtered, P_filtered, float(log_likelihood), loglik_trajectory

    @classmethod
    def filter_with_trajectory(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        nu: float,
        regime_id: str = "global",
        use_cache: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Kalman filter with per-timestep likelihood trajectory.
        
        Enables fold-aware CV likelihood slicing without re-execution.
        Uses deterministic result cache when available.
        
        Returns (mu_filtered, P_filtered, total_log_likelihood, loglik_trajectory).
        """
        if use_cache and _CACHE_AVAILABLE and FILTER_CACHE_ENABLED:
            return cached_phi_student_t_filter(
                returns, vol, q, c, phi, nu,
                filter_fn=cls._filter_phi_with_trajectory,
                regime_id=regime_id
            )
        return cls._filter_phi_with_trajectory(returns, vol, q, c, phi, nu)

    @staticmethod
    def pit_ks(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float, nu: float) -> Tuple[float, float]:
        """PIT/KS for Student-t forecasts with parameter uncertainty included.
        
        Uses the Student-t distribution CDF for the PIT transformation, which is
        more appropriate for heavy-tailed return distributions.
        """
        returns_flat = np.asarray(returns).flatten()
        mu_flat = np.asarray(mu_filtered).flatten()
        vol_flat = np.asarray(vol).flatten()
        P_flat = np.asarray(P_filtered).flatten()

        # Compute forecast scale with numerical floor
        forecast_var = c * (vol_flat ** 2) + P_flat
        forecast_scale = np.sqrt(np.maximum(forecast_var, 1e-20))
        forecast_scale = np.where(forecast_scale < 1e-10, 1e-10, forecast_scale)
        
        standardized = (returns_flat - mu_flat) / forecast_scale
        
        # Handle any remaining NaN/Inf values
        valid_mask = np.isfinite(standardized)
        if not np.any(valid_mask):
            return 1.0, 0.0
        
        standardized_clean = standardized[valid_mask]
        
        # Ensure nu is valid for Student-t (must be > 0)
        nu_safe = max(nu, 2.01)
        pit_values = student_t.cdf(standardized_clean, df=nu_safe)
        
        if len(pit_values) < 2:
            return 1.0, 0.0
            
        ks_result = kstest(pit_values, 'uniform')
        return float(ks_result.statistic), float(ks_result.pvalue)

    # =========================================================================
    # ENHANCED FILTER METHODS (February 2026)
    # =========================================================================
    
    @classmethod
    def filter_phi_vov(
        cls, 
        returns: np.ndarray, 
        vol: np.ndarray, 
        q: float, 
        c: float, 
        phi: float, 
        nu: float,
        gamma_vov: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        φ-Student-t filter with Volatility-of-Volatility observation noise correction.
        
        Standard model: R_t = c × σ_t²
        VoV model:      R_t = c × σ_t² × (1 + γ × |Δlog(σ_t)|)
        
        When volatility changes rapidly, the EWMA vol lags true vol, so we need
        a larger multiplier. When γ=0, this reduces to standard filter_phi.
        
        Args:
            returns: Return series
            vol: EWMA volatility series
            q: Process noise variance
            c: Base observation noise scale
            phi: AR(1) persistence
            nu: Degrees of freedom
            gamma_vov: Vol-of-vol sensitivity (0 = disabled)
            
        Returns:
            (mu_filtered, P_filtered, log_likelihood)
        """
        n = len(returns)
        
        # Convert to contiguous arrays
        returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)
        
        # Extract scalar values
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))
        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)
        gamma = float(gamma_vov) if np.ndim(gamma_vov) == 0 else float(gamma_vov.item())
        
        # Pre-compute vol changes: |Δlog(σ_t)|
        log_vol = np.log(np.maximum(vol, 1e-10))
        vol_changes = np.zeros(n)
        vol_changes[1:] = np.abs(np.diff(log_vol))
        
        # Pre-compute VoV-adjusted c multipliers: 1 + γ × |Δlog(σ)|
        vov_mult = 1.0 + gamma * vol_changes
        
        # Pre-compute constants
        phi_sq = phi_val * phi_val
        nu_adjust = min(nu_val / (nu_val + 3.0), 1.0)
        log_norm_const = gammaln((nu_val + 1.0) / 2.0) - gammaln(nu_val / 2.0) - 0.5 * np.log(nu_val * np.pi)
        neg_exp = -((nu_val + 1.0) / 2.0)
        inv_nu = 1.0 / nu_val
        
        # Pre-compute base R values (will be scaled by vov_mult)
        R_base = c_val * (vol * vol)
        
        # Allocate output arrays
        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        
        # State initialization
        mu = 0.0
        P = 1e-4
        log_likelihood = 0.0
        
        # Main filter loop with VoV correction
        for t in range(n):
            # Prediction step
            mu_pred = phi_val * mu
            P_pred = phi_sq * P + q_val
            
            # VoV-adjusted observation variance
            R = R_base[t] * vov_mult[t]
            
            # Observation update
            S = P_pred + R
            if S <= 1e-12:
                S = 1e-12
            
            innovation = returns[t] - mu_pred
            K = nu_adjust * P_pred / S
            
            # State update
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12
            
            # Store filtered values
            mu_filtered[t] = mu
            P_filtered[t] = P
            
            # Inlined log-pdf calculation
            forecast_scale = np.sqrt(S)
            if forecast_scale > 1e-12:
                z = innovation / forecast_scale
                ll_t = log_norm_const - np.log(forecast_scale) + neg_exp * np.log(1.0 + z * z * inv_nu)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t
        
        return mu_filtered, P_filtered, float(log_likelihood)

    @classmethod
    def filter_phi_two_piece(
        cls, 
        returns: np.ndarray, 
        vol: np.ndarray, 
        q: float, 
        c: float, 
        phi: float, 
        nu_left: float,
        nu_right: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        φ-Student-t filter with Two-Piece asymmetric tail behavior.
        
        Uses different degrees of freedom for negative vs positive innovations:
            - νL (nu_left): For negative returns (crash tail) - typically smaller = heavier
            - νR (nu_right): For positive returns (recovery tail) - typically larger = lighter
        
        This captures the empirical asymmetry where crashes are more extreme than rallies.
        
        Args:
            returns: Return series
            vol: EWMA volatility series
            q: Process noise variance
            c: Observation noise scale
            phi: AR(1) persistence
            nu_left: Degrees of freedom for negative innovations (crash)
            nu_right: Degrees of freedom for positive innovations (recovery)
            
        Returns:
            (mu_filtered, P_filtered, log_likelihood)
        """
        n = len(returns)
        
        # Convert to contiguous arrays
        returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)
        
        # Extract scalar values
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))
        nu_L = cls._clip_nu(nu_left, cls.nu_min_default, cls.nu_max_default)
        nu_R = cls._clip_nu(nu_right, cls.nu_min_default, cls.nu_max_default)
        
        # Pre-compute constants
        phi_sq = phi_val * phi_val
        
        # Pre-compute log-pdf constants for both ν values
        log_norm_L = gammaln((nu_L + 1.0) / 2.0) - gammaln(nu_L / 2.0) - 0.5 * np.log(nu_L * np.pi)
        neg_exp_L = -((nu_L + 1.0) / 2.0)
        inv_nu_L = 1.0 / nu_L
        nu_adjust_L = min(nu_L / (nu_L + 3.0), 1.0)
        
        log_norm_R = gammaln((nu_R + 1.0) / 2.0) - gammaln(nu_R / 2.0) - 0.5 * np.log(nu_R * np.pi)
        neg_exp_R = -((nu_R + 1.0) / 2.0)
        inv_nu_R = 1.0 / nu_R
        nu_adjust_R = min(nu_R / (nu_R + 3.0), 1.0)
        
        # Pre-compute R values
        R = c_val * (vol * vol)
        
        # Allocate output arrays
        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        
        # State initialization
        mu = 0.0
        P = 1e-4
        log_likelihood = 0.0
        
        # Main filter loop with two-piece likelihood
        for t in range(n):
            # Prediction step
            mu_pred = phi_val * mu
            P_pred = phi_sq * P + q_val
            
            # Observation update
            S = P_pred + R[t]
            if S <= 1e-12:
                S = 1e-12
            
            innovation = returns[t] - mu_pred
            
            # Choose ν based on innovation sign
            if innovation < 0:
                # Crash tail (left)
                nu_adjust = nu_adjust_L
                log_norm = log_norm_L
                neg_exp = neg_exp_L
                inv_nu = inv_nu_L
            else:
                # Recovery tail (right)
                nu_adjust = nu_adjust_R
                log_norm = log_norm_R
                neg_exp = neg_exp_R
                inv_nu = inv_nu_R
            
            K = nu_adjust * P_pred / S
            
            # State update
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12
            
            # Store filtered values
            mu_filtered[t] = mu
            P_filtered[t] = P
            
            # Inlined log-pdf calculation with chosen ν
            forecast_scale = np.sqrt(S)
            if forecast_scale > 1e-12:
                z = innovation / forecast_scale
                ll_t = log_norm - np.log(forecast_scale) + neg_exp * np.log(1.0 + z * z * inv_nu)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t
        
        return mu_filtered, P_filtered, float(log_likelihood)

    @classmethod
    def filter_phi_mixture(
        cls, 
        returns: np.ndarray, 
        vol: np.ndarray, 
        q: float, 
        c: float, 
        phi: float, 
        nu_calm: float,
        nu_stress: float,
        w_base: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        φ-Student-t filter with Two-Component mixture body.
        
        Mixture model with dynamic weights:
            p(r_t) = w_t × t(νcalm) + (1 - w_t) × t(νstress)
        
        Weight dynamics (higher vol → more stress weight):
            w_t = sigmoid(w_base - k × vol_relative_t)
        
        where vol_relative = σ_t / median(σ), k = 2.0
        
        This captures two curvature regimes:
            - Calm: lighter tails (νcalm > νstress)
            - Stress: heavier tails (νstress < νcalm)
        
        Args:
            returns: Return series
            vol: EWMA volatility series
            q: Process noise variance
            c: Observation noise scale
            phi: AR(1) persistence
            nu_calm: Degrees of freedom for calm regime (lighter)
            nu_stress: Degrees of freedom for stress regime (heavier)
            w_base: Base weight parameter (default 0.5)
            
        Returns:
            (mu_filtered, P_filtered, log_likelihood)
        """
        n = len(returns)
        
        # Convert to contiguous arrays
        returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)
        
        # Extract scalar values
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))
        nu_C = cls._clip_nu(nu_calm, cls.nu_min_default, cls.nu_max_default)
        nu_S = cls._clip_nu(nu_stress, cls.nu_min_default, cls.nu_max_default)
        w_b = float(w_base) if np.ndim(w_base) == 0 else float(w_base.item())
        
        # Pre-compute constants
        phi_sq = phi_val * phi_val
        k = MIXTURE_WEIGHT_K  # Sensitivity to vol_relative
        
        # Pre-compute vol_relative and dynamic weights
        vol_median = np.median(vol)
        if vol_median < 1e-10:
            vol_median = 1e-10
        vol_relative = vol / vol_median
        
        # w_t = sigmoid(w_base - k * vol_relative)
        # Higher vol → lower w_calm → more stress weight
        exponent = np.clip(-(w_b - k * vol_relative), -50, 50)
        w_calm = 1.0 / (1.0 + np.exp(exponent))
        
        # Pre-compute log-pdf constants for both ν values
        log_norm_C = gammaln((nu_C + 1.0) / 2.0) - gammaln(nu_C / 2.0) - 0.5 * np.log(nu_C * np.pi)
        neg_exp_C = -((nu_C + 1.0) / 2.0)
        inv_nu_C = 1.0 / nu_C
        
        log_norm_S = gammaln((nu_S + 1.0) / 2.0) - gammaln(nu_S / 2.0) - 0.5 * np.log(nu_S * np.pi)
        neg_exp_S = -((nu_S + 1.0) / 2.0)
        inv_nu_S = 1.0 / nu_S
        
        # Use average ν for Kalman gain (weighted by w_calm)
        # This is an approximation; exact would require mixture state estimation
        nu_avg = w_b * nu_C + (1 - w_b) * nu_S  # Use base weight for stability
        nu_adjust = min(nu_avg / (nu_avg + 3.0), 1.0)
        
        # Pre-compute R values
        R = c_val * (vol * vol)
        
        # Allocate output arrays
        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        
        # State initialization
        mu = 0.0
        P = 1e-4
        log_likelihood = 0.0
        
        # Main filter loop with mixture likelihood
        for t in range(n):
            # Prediction step
            mu_pred = phi_val * mu
            P_pred = phi_sq * P + q_val
            
            # Observation update
            S = P_pred + R[t]
            if S <= 1e-12:
                S = 1e-12
            
            innovation = returns[t] - mu_pred
            K = nu_adjust * P_pred / S
            
            # State update
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12
            
            # Store filtered values
            mu_filtered[t] = mu
            P_filtered[t] = P
            
            # Mixture log-pdf calculation
            forecast_scale = np.sqrt(S)
            if forecast_scale > 1e-12:
                z = innovation / forecast_scale
                
                # Calm component log-pdf
                ll_C = log_norm_C - np.log(forecast_scale) + neg_exp_C * np.log(1.0 + z * z * inv_nu_C)
                
                # Stress component log-pdf
                ll_S = log_norm_S - np.log(forecast_scale) + neg_exp_S * np.log(1.0 + z * z * inv_nu_S)
                
                # Mixture log-pdf via log-sum-exp
                w_t = w_calm[t]
                ll_max = max(ll_C, ll_S)
                ll_mix = ll_max + np.log(
                    w_t * np.exp(ll_C - ll_max) + 
                    (1.0 - w_t) * np.exp(ll_S - ll_max)
                )
                
                if np.isfinite(ll_mix):
                    log_likelihood += ll_mix
        
        return mu_filtered, P_filtered, float(log_likelihood)

    @staticmethod
    def pit_ks_two_piece(
        returns: np.ndarray, 
        mu_filtered: np.ndarray, 
        vol: np.ndarray, 
        P_filtered: np.ndarray, 
        c: float, 
        nu_left: float,
        nu_right: float
    ) -> Tuple[float, float]:
        """PIT/KS calibration test for Two-Piece Student-t model."""
        returns_flat = np.asarray(returns).flatten()
        mu_flat = np.asarray(mu_filtered).flatten()
        vol_flat = np.asarray(vol).flatten()
        P_flat = np.asarray(P_filtered).flatten()
        
        forecast_var = c * (vol_flat ** 2) + P_flat
        forecast_scale = np.sqrt(np.maximum(forecast_var, 1e-20))
        forecast_scale = np.where(forecast_scale < 1e-10, 1e-10, forecast_scale)
        
        standardized = (returns_flat - mu_flat) / forecast_scale
        
        valid_mask = np.isfinite(standardized)
        if not np.any(valid_mask):
            return 1.0, 0.0
        
        standardized_clean = standardized[valid_mask]
        
        # Two-piece CDF: use νL for z<0, νR for z>=0
        pit_values = np.empty(len(standardized_clean))
        nu_L_safe = max(nu_left, 2.01)
        nu_R_safe = max(nu_right, 2.01)
        
        neg_mask = standardized_clean < 0
        pos_mask = ~neg_mask
        
        # For z < 0: CDF with νL, scaled to [0, 0.5]
        if np.any(neg_mask):
            pit_values[neg_mask] = 0.5 * student_t.cdf(standardized_clean[neg_mask], df=nu_L_safe) / student_t.cdf(0, df=nu_L_safe)
        
        # For z >= 0: CDF with νR, mapped to [0.5, 1]
        if np.any(pos_mask):
            pit_values[pos_mask] = 0.5 + 0.5 * (student_t.cdf(standardized_clean[pos_mask], df=nu_R_safe) - 0.5) / 0.5
        
        if len(pit_values) < 2:
            return 1.0, 0.0
        
        ks_result = kstest(pit_values, 'uniform')
        return float(ks_result.statistic), float(ks_result.pvalue)

    @staticmethod
    def pit_ks_vov(
        returns: np.ndarray, 
        mu_filtered: np.ndarray, 
        vol: np.ndarray, 
        P_filtered: np.ndarray, 
        c: float, 
        nu: float,
        gamma_vov: float
    ) -> Tuple[float, float]:
        """PIT/KS calibration test for Vol-of-Vol corrected model."""
        returns_flat = np.asarray(returns).flatten()
        mu_flat = np.asarray(mu_filtered).flatten()
        vol_flat = np.asarray(vol).flatten()
        P_flat = np.asarray(P_filtered).flatten()
        
        # Compute VoV-adjusted forecast variance
        log_vol = np.log(np.maximum(vol_flat, 1e-10))
        vol_changes = np.zeros(len(vol_flat))
        vol_changes[1:] = np.abs(np.diff(log_vol))
        vov_mult = 1.0 + gamma_vov * vol_changes
        
        forecast_var = c * (vol_flat ** 2) * vov_mult + P_flat
        forecast_scale = np.sqrt(np.maximum(forecast_var, 1e-20))
        forecast_scale = np.where(forecast_scale < 1e-10, 1e-10, forecast_scale)
        
        standardized = (returns_flat - mu_flat) / forecast_scale
        
        valid_mask = np.isfinite(standardized)
        if not np.any(valid_mask):
            return 1.0, 0.0
        
        standardized_clean = standardized[valid_mask]
        nu_safe = max(nu, 2.01)
        pit_values = student_t.cdf(standardized_clean, df=nu_safe)
        
        if len(pit_values) < 2:
            return 1.0, 0.0
        
        ks_result = kstest(pit_values, 'uniform')
        return float(ks_result.statistic), float(ks_result.pvalue)

    @staticmethod
    def compute_pit_ks_pvalue(returns: np.ndarray, mu_filtered: np.ndarray, vol: np.ndarray, P_filtered: np.ndarray, c: float = 1.0) -> Tuple[float, float]:
        """PIT/KS for Gaussian forecasts including parameter uncertainty.
        
        This is a Gaussian version used for comparison purposes.
        """
        returns_flat = np.asarray(returns).flatten()
        mu_flat = np.asarray(mu_filtered).flatten()
        vol_flat = np.asarray(vol).flatten()
        P_flat = np.asarray(P_filtered).flatten()

        forecast_var = c * (vol_flat ** 2) + P_flat
        forecast_std = np.sqrt(np.maximum(forecast_var, 1e-20))
        forecast_std = np.where(forecast_std < 1e-10, 1e-10, forecast_std)
        
        standardized = (returns_flat - mu_flat) / forecast_std
        
        valid_mask = np.isfinite(standardized)
        if not np.any(valid_mask):
            return 1.0, 0.0
        
        standardized_clean = standardized[valid_mask]
        pit_values = norm.cdf(standardized_clean)
        
        if len(pit_values) < 2:
            return 1.0, 0.0
            
        ks_result = kstest(pit_values, 'uniform')
        return float(ks_result.statistic), float(ks_result.pvalue)

    @classmethod
    def filter_phi_batch(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        nu_grid: List[float] = None
    ) -> Dict[float, Tuple[np.ndarray, np.ndarray, float]]:
        """
        Run φ-Student-t filter for multiple ν values (discrete grid BMA).
        
        Significantly faster than calling filter_phi() in a loop because:
        - Arrays are prepared once
        - Gamma values are precomputed efficiently per ν
        
        Parameters
        ----------
        nu_grid : List[float], optional
            List of ν values to evaluate. Default: [4, 6, 8, 12, 20]
        
        Returns
        -------
        results : Dict[float, Tuple[np.ndarray, np.ndarray, float]]
            Dict mapping ν -> (mu_filtered, P_filtered, log_likelihood)
        """
        if nu_grid is None:
            nu_grid = STUDENT_T_NU_GRID
        
        # Try Numba batch version
        if _USE_NUMBA:
            try:
                return run_phi_student_t_filter_batch(returns, vol, q, c, phi, nu_grid)
            except Exception:
                pass  # Fall through to Python implementation
        
        # Python fallback
        results = {}
        for nu in nu_grid:
            results[nu] = cls._filter_phi_python(returns, vol, q, c, phi, nu)
        return results

    @staticmethod
    def optimize_params(
        returns: np.ndarray,
        vol: np.ndarray,
        train_frac: float = 0.7,
        q_min: float = 1e-10,
        q_max: float = 1e-1,
        c_min: float = 0.3,
        c_max: float = 3.0,
        phi_min: float = -0.999,
        phi_max: float = 0.999,
        nu_min: float = 2.1,
        nu_max: float = 30.0,
        prior_log_q_mean: float = -6.0,
        prior_lambda: float = 1.0
    ) -> Tuple[float, float, float, float, float, Dict]:
        """Jointly optimize (q, c, φ, ν) for the φ-Student-t drift model via CV MLE."""
        n = len(returns)
        ret_p005 = np.percentile(returns, 0.5)
        ret_p995 = np.percentile(returns, 99.5)
        returns_robust = np.clip(returns, ret_p005, ret_p995)

        vol_mean = float(np.mean(vol))
        vol_std = float(np.std(vol))
        vol_cv = vol_std / vol_mean if vol_mean > 0 else 0.0
        ret_std = float(np.std(returns_robust))
        ret_mean = float(np.mean(returns_robust))
        rv_ratio = abs(ret_mean) / ret_std if ret_std > 0 else 0.0

        if vol_cv > 0.5 or rv_ratio > 0.15:
            adaptive_prior_mean = prior_log_q_mean + 0.5
            adaptive_lambda = prior_lambda * 0.5
        elif vol_cv < 0.2 and rv_ratio < 0.05:
            adaptive_prior_mean = prior_log_q_mean - 0.3
            adaptive_lambda = prior_lambda * 1.5
        else:
            adaptive_prior_mean = prior_log_q_mean
            adaptive_lambda = prior_lambda

        min_train = min(max(60, int(n * 0.4)), max(n - 5, 1))
        test_window = min(max(20, int(n * 0.1)), max(n - min_train, 5))
        fold_splits = []
        train_end = min_train
        while train_end + test_window <= n:
            test_end = min(train_end + test_window, n)
            if test_end - train_end >= 20:
                fold_splits.append((0, train_end, train_end, test_end))
            train_end += test_window
        if not fold_splits:
            split_idx = int(n * train_frac)
            fold_splits = [(0, split_idx, split_idx, n)]

        def neg_pen_ll(params: np.ndarray) -> float:
            log_q, log_c, phi, log_nu = params
            q = 10 ** log_q
            c = 10 ** log_c
            phi_clip = float(np.clip(phi, phi_min, phi_max))
            nu = 10 ** log_nu
            if q <= 0 or c <= 0 or not np.isfinite(q) or not np.isfinite(c) or nu < nu_min or nu > nu_max:
                return 1e12
            total_ll_oos = 0.0
            total_obs = 0
            all_standardized = []
            for tr_start, tr_end, te_start, te_end in fold_splits:
                try:
                    ret_train = returns_robust[tr_start:tr_end]
                    vol_train = vol[tr_start:tr_end]
                    if len(ret_train) < 3:
                        continue
                    mu_filt_train, P_filt_train, _ = PhiStudentTDriftModel.filter_phi(ret_train, vol_train, q, c, phi_clip, nu)
                    mu_pred = float(mu_filt_train[-1])
                    P_pred = float(P_filt_train[-1])
                    ll_fold = 0.0
                    for t in range(te_start, te_end):
                        mu_pred = phi_clip * mu_pred
                        P_pred = (phi_clip ** 2) * P_pred + q
                        ret_t = float(returns_robust[t]) if np.ndim(returns_robust[t]) == 0 else float(returns_robust[t].item())
                        vol_t = float(vol[t]) if np.ndim(vol[t]) == 0 else float(vol[t].item())
                        R = c * (vol_t ** 2)
                        innovation = ret_t - mu_pred
                        forecast_var = P_pred + R

                        if forecast_var > 1e-12:
                            forecast_std = np.sqrt(forecast_var)
                            ll_contrib = PhiStudentTDriftModel.logpdf(ret_t, nu, mu_pred, forecast_std)
                            ll_fold += ll_contrib
                            if len(all_standardized) < 1000:
                                all_standardized.append(float(innovation / forecast_std))

                        nu_adjust = min(nu / (nu + 3.0), 1.0)
                        K = nu_adjust * P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0
                        mu_pred = mu_pred + K * innovation
                        P_pred = (1.0 - K) * P_pred

                    total_ll_oos += ll_fold
                    total_obs += (te_end - te_start)

                except Exception:
                    continue
            if total_obs == 0:
                return 1e12
            avg_ll = total_ll_oos / max(total_obs, 1)
            calibration_penalty = 0.0
            if len(all_standardized) >= 30:
                try:
                    pit_values = student_t.cdf(all_standardized, df=nu)
                    ks_result = kstest(pit_values, 'uniform')
                    ks_stat = float(ks_result.statistic)
                    if ks_stat > 0.05:
                        calibration_penalty = -50.0 * ((ks_stat - 0.05) ** 2)
                        if ks_stat > 0.10:
                            calibration_penalty -= 100.0 * (ks_stat - 0.10)
                        if ks_stat > 0.15:
                            calibration_penalty -= 200.0 * (ks_stat - 0.15)
                except Exception:
                    pass
            prior_scale = 1.0 / max(total_obs, 100)
            log_prior_q = -adaptive_lambda * prior_scale * (log_q - adaptive_prior_mean) ** 2
            log_c_target = np.log10(0.9)
            log_prior_c = -0.1 * prior_scale * (log_c - log_c_target) ** 2
            
            # Explicit φ shrinkage prior
            phi_lambda_effective = PHI_SHRINKAGE_LAMBDA_DEFAULT * prior_scale
            phi_tau = _lambda_to_tau(phi_lambda_effective)
            log_prior_phi = _phi_shrinkage_log_prior(
                phi_r=phi_clip,
                phi_global=PHI_SHRINKAGE_GLOBAL_DEFAULT,
                tau=phi_tau
            )
            log_prior_nu = -0.05 * prior_scale * (log_nu - np.log10(6.0)) ** 2

            penalized_ll = avg_ll + log_prior_q + log_prior_c + log_prior_phi + log_prior_nu + calibration_penalty
            return -penalized_ll if np.isfinite(penalized_ll) else 1e12

        log_q_min = np.log10(q_min)
        log_q_max = np.log10(q_max)
        log_c_min = np.log10(c_min)
        log_c_max = np.log10(c_max)
        log_nu_min = np.log10(nu_min)
        log_nu_max = np.log10(nu_max)

        grid_best = (adaptive_prior_mean, np.log10(0.9), 0.0, np.log10(6.0))
        best_neg = float('inf')
        for lq in np.linspace(log_q_min, log_q_max, 4):
            for lc in np.linspace(log_c_min, log_c_max, 3):
                for lp in np.linspace(phi_min, phi_max, 5):
                    for ln in np.linspace(log_nu_min, log_nu_max, 3):
                        val = neg_pen_ll(np.array([lq, lc, lp, ln]))
                        if val < best_neg:
                            best_neg = val
                            grid_best = (lq, lc, lp, ln)
        bounds = [(log_q_min, log_q_max), (log_c_min, log_c_max), (phi_min, phi_max), (log_nu_min, log_nu_max)]
        start_points = [np.array(grid_best), np.array([adaptive_prior_mean, np.log10(0.9), 0.0, np.log10(6.0)])]
        best_res = None
        best_fun = float('inf')
        for x0 in start_points:
            try:
                res = minimize(neg_pen_ll, x0=x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 120, 'ftol': 1e-6})
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_res = res
            except Exception:
                continue

        if best_res is not None and best_res.success:
            lq_opt, lc_opt, phi_opt, ln_opt = best_res.x
            q_opt = 10 ** lq_opt
            c_opt = 10 ** lc_opt
            phi_opt = float(np.clip(phi_opt, phi_min, phi_max))
            nu_opt = 10 ** ln_opt
            ll_opt = -best_res.fun
        else:
            lq_opt, lc_opt, phi_opt, ln_opt = grid_best
            q_opt = 10 ** lq_opt
            c_opt = 10 ** lc_opt
            phi_opt = float(np.clip(phi_opt, phi_min, phi_max))
            nu_opt = 10 ** ln_opt
            ll_opt = -best_neg

        # Compute φ shrinkage prior diagnostics for auditability
        n_obs_approx = len(returns)
        prior_scale_diag = 1.0 / max(n_obs_approx, 100)
        phi_lambda_eff_diag = PHI_SHRINKAGE_LAMBDA_DEFAULT * prior_scale_diag
        phi_tau_diag = _lambda_to_tau(phi_lambda_eff_diag)
        phi_prior_diag = _compute_phi_prior_diagnostics(
            phi_r=phi_opt,
            phi_global=PHI_SHRINKAGE_GLOBAL_DEFAULT,
            tau=phi_tau_diag,
            log_likelihood=ll_opt
        )

        diagnostics = {
            'grid_best_q': float(10 ** grid_best[0]),
            'grid_best_c': float(10 ** grid_best[1]),
            'grid_best_phi': float(grid_best[2]),
            'grid_best_nu': float(10 ** grid_best[3]),
            'refined_best_q': float(q_opt),
            'refined_best_c': float(c_opt),
            'refined_best_phi': float(phi_opt),
            'refined_best_nu': float(nu_opt),
            'prior_applied': adaptive_lambda > 0,
            'prior_log_q_mean': float(adaptive_prior_mean),
            'prior_lambda': float(adaptive_lambda),
            'vol_cv': float(vol_cv),
            'rv_ratio': float(rv_ratio),
            'n_folds': int(len(fold_splits)),
            'optimization_successful': best_res is not None and (best_res.success if best_res else False),
            **phi_prior_diag,
        }

        return q_opt, c_opt, phi_opt, nu_opt, ll_opt, diagnostics

    @staticmethod
    def optimize_params_fixed_nu(
        returns: np.ndarray,
        vol: np.ndarray,
        nu: float,
        train_frac: float = 0.7,
        q_min: float = 1e-10,
        q_max: float = 1e-1,
        c_min: float = 0.3,
        c_max: float = 3.0,
        phi_min: float = -0.999,
        phi_max: float = 0.999,
        prior_log_q_mean: float = -6.0,
        prior_lambda: float = 1.0
    ) -> Tuple[float, float, float, float, Dict]:
        """
        Optimize (q, c, φ) for the φ-Student-t drift model with FIXED ν.
        
        This method is part of the discrete ν grid approach:
        - ν is held fixed (passed as argument, not optimized)
        - Only q, c, φ are optimized via CV MLE
        - Each ν value becomes a separate sub-model in BMA
        """
        n = len(returns)
        ret_p005 = np.percentile(returns, 0.5)
        ret_p995 = np.percentile(returns, 99.5)
        returns_robust = np.clip(returns, ret_p005, ret_p995)

        vol_mean = float(np.mean(vol))
        vol_std = float(np.std(vol))
        vol_cv = vol_std / vol_mean if vol_mean > 0 else 0.0
        ret_std = float(np.std(returns_robust))
        ret_mean = float(np.mean(returns_robust))
        rv_ratio = abs(ret_mean) / ret_std if ret_std > 0 else 0.0

        if vol_cv > 0.5 or rv_ratio > 0.15:
            adaptive_prior_mean = prior_log_q_mean + 0.5
            adaptive_lambda = prior_lambda * 0.5
        elif vol_cv < 0.2 and rv_ratio < 0.05:
            adaptive_prior_mean = prior_log_q_mean - 0.3
            adaptive_lambda = prior_lambda * 1.5
        else:
            adaptive_prior_mean = prior_log_q_mean
            adaptive_lambda = prior_lambda

        min_train = min(max(60, int(n * 0.4)), max(n - 5, 1))
        test_window = min(max(20, int(n * 0.1)), max(n - min_train, 5))
        fold_splits = []
        train_end = min_train
        while train_end + test_window <= n:
            test_end = min(train_end + test_window, n)
            if test_end - train_end >= 20:
                fold_splits.append((0, train_end, train_end, test_end))
            train_end += test_window
        if not fold_splits:
            split_idx = int(n * train_frac)
            fold_splits = [(0, split_idx, split_idx, n)]

        nu_fixed = float(nu)

        def neg_pen_ll(params: np.ndarray) -> float:
            log_q, log_c, phi = params
            q = 10 ** log_q
            c = 10 ** log_c
            phi_clip = float(np.clip(phi, phi_min, phi_max))
            
            if q <= 0 or c <= 0 or not np.isfinite(q) or not np.isfinite(c):
                return 1e12
            
            total_ll_oos = 0.0
            total_obs = 0
            all_standardized = []
            
            for tr_start, tr_end, te_start, te_end in fold_splits:
                try:
                    ret_train = returns_robust[tr_start:tr_end]
                    vol_train = vol[tr_start:tr_end]
                    if len(ret_train) < 3:
                        continue
                    
                    mu_filt_train, P_filt_train, _ = PhiStudentTDriftModel.filter_phi(
                        ret_train, vol_train, q, c, phi_clip, nu_fixed
                    )
                    mu_pred = float(mu_filt_train[-1])
                    P_pred = float(P_filt_train[-1])
                    ll_fold = 0.0
                    
                    for t in range(te_start, te_end):
                        mu_pred = phi_clip * mu_pred
                        P_pred = (phi_clip ** 2) * P_pred + q
                        ret_t = float(returns_robust[t]) if np.ndim(returns_robust[t]) == 0 else float(returns_robust[t].item())
                        vol_t = float(vol[t]) if np.ndim(vol[t]) == 0 else float(vol[t].item())
                        R = c * (vol_t ** 2)
                        innovation = ret_t - mu_pred
                        forecast_var = P_pred + R

                        if forecast_var > 1e-12:
                            forecast_std = np.sqrt(forecast_var)
                            ll_contrib = PhiStudentTDriftModel.logpdf(ret_t, nu_fixed, mu_pred, forecast_std)
                            ll_fold += ll_contrib
                            if len(all_standardized) < 1000:
                                all_standardized.append(float(innovation / forecast_std))

                        nu_adjust = min(nu_fixed / (nu_fixed + 3.0), 1.0)
                        K = nu_adjust * P_pred / (P_pred + R) if (P_pred + R) > 1e-12 else 0.0
                        mu_pred = mu_pred + K * innovation
                        P_pred = (1.0 - K) * P_pred

                    total_ll_oos += ll_fold
                    total_obs += (te_end - te_start)

                except Exception:
                    continue
            
            if total_obs == 0:
                return 1e12
            
            avg_ll = total_ll_oos / max(total_obs, 1)
            
            calibration_penalty = 0.0
            if len(all_standardized) >= 30:
                try:
                    pit_values = student_t.cdf(all_standardized, df=nu_fixed)
                    ks_result = kstest(pit_values, 'uniform')
                    ks_stat = float(ks_result.statistic)
                    if ks_stat > 0.05:
                        calibration_penalty = -50.0 * ((ks_stat - 0.05) ** 2)
                        if ks_stat > 0.10:
                            calibration_penalty -= 100.0 * (ks_stat - 0.10)
                        if ks_stat > 0.15:
                            calibration_penalty -= 200.0 * (ks_stat - 0.15)
                except Exception:
                    pass
            
            prior_scale = 1.0 / max(total_obs, 100)
            log_prior_q = -adaptive_lambda * prior_scale * (log_q - adaptive_prior_mean) ** 2
            log_c_target = np.log10(0.9)
            log_prior_c = -0.1 * prior_scale * (log_c - log_c_target) ** 2
            
            phi_lambda_effective = PHI_SHRINKAGE_LAMBDA_DEFAULT * prior_scale
            phi_tau = _lambda_to_tau(phi_lambda_effective)
            log_prior_phi = _phi_shrinkage_log_prior(
                phi_r=phi_clip,
                phi_global=PHI_SHRINKAGE_GLOBAL_DEFAULT,
                tau=phi_tau
            )

            penalized_ll = avg_ll + log_prior_q + log_prior_c + log_prior_phi + calibration_penalty
            return -penalized_ll if np.isfinite(penalized_ll) else 1e12

        log_q_min = np.log10(q_min)
        log_q_max = np.log10(q_max)
        log_c_min = np.log10(c_min)
        log_c_max = np.log10(c_max)

        # Optimized grid search with parallel evaluation (February 2026)
        # Use coarser grid (3x2x3 = 18) with parallel execution
        lq_grid = np.linspace(log_q_min, log_q_max, 3)
        lc_grid = np.linspace(log_c_min, log_c_max, 2)
        lp_grid = np.array([phi_min, 0.0, phi_max * 0.5])
        
        # Generate all grid points
        grid_points = [(lq, lc, lp) 
                       for lq in lq_grid 
                       for lc in lc_grid 
                       for lp in lp_grid]
        
        def _eval_point(point):
            lq, lc, lp = point
            val = neg_pen_ll(np.array([lq, lc, lp]))
            return val, point
        
        grid_best = (adaptive_prior_mean, np.log10(0.9), 0.0)
        best_neg = float('inf')
        
        # Use 4 threads for parallel evaluation
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(_eval_point, grid_points))
            for val, point in results:
                if val < best_neg:
                    best_neg = val
                    grid_best = point
        except Exception:
            # Fallback to sequential if parallel fails
            for point in grid_points:
                val = neg_pen_ll(np.array(point))
                if val < best_neg:
                    best_neg = val
                    grid_best = point
        
        bounds = [(log_q_min, log_q_max), (log_c_min, log_c_max), (phi_min, phi_max)]
        start_points = [
            np.array(grid_best),
            np.array([adaptive_prior_mean, np.log10(0.9), 0.0])
        ]
        best_res = None
        best_fun = float('inf')
        for x0 in start_points:
            try:
                res = minimize(
                    neg_pen_ll, x0=x0, method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 120, 'ftol': 1e-6}
                )
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_res = res
            except Exception:
                continue

        if best_res is not None and best_res.success:
            lq_opt, lc_opt, phi_opt = best_res.x
            q_opt = 10 ** lq_opt
            c_opt = 10 ** lc_opt
            phi_opt = float(np.clip(phi_opt, phi_min, phi_max))
            ll_opt = -best_res.fun
        else:
            lq_opt, lc_opt, phi_opt = grid_best
            q_opt = 10 ** lq_opt
            c_opt = 10 ** lc_opt
            phi_opt = float(np.clip(phi_opt, phi_min, phi_max))
            ll_opt = -best_neg

        # =====================================================================
        # ELITE TUNING: Curvature and Fragility Analysis (February 2026)
        # =====================================================================
        elite_diagnostics = {}
        if ELITE_TUNING_ENABLED:
            optimal_params = np.array([lq_opt, lc_opt, phi_opt])
            param_ranges = np.array([log_q_max - log_q_min, log_c_max - log_c_min, phi_max - phi_min])
            
            # Compute curvature penalty (prefer flat regions)
            try:
                curvature_penalty, condition_number, curv_diag = _compute_curvature_penalty(
                    neg_pen_ll, optimal_params, bounds, HESSIAN_EPSILON, MAX_CONDITION_NUMBER
                )
                elite_diagnostics['curvature'] = curv_diag
                elite_diagnostics['condition_number'] = float(condition_number)
            except Exception:
                curvature_penalty = 0.0
                condition_number = 1.0
                elite_diagnostics['curvature'] = {'error': 'computation_failed'}
            
            # Compute coherence (fold-to-fold stability)
            # Note: Would need fold_optimal_params from per-fold optimization
            # For now, we provide placeholder indicating single-optimum result
            elite_diagnostics['coherence'] = {
                'coherence_penalty': 0.0,
                'n_folds_with_separate_optima': 0,
                'note': 'coherence computed at tune.py level with per-fold optima'
            }
            
            # Compute fragility index
            try:
                fragility_index, frag_components = _compute_fragility_index(
                    condition_number, np.array([]), 0.0
                )
                elite_diagnostics['fragility_index'] = float(fragility_index)
                elite_diagnostics['fragility_components'] = frag_components
                elite_diagnostics['fragility_warning'] = fragility_index > 0.5
            except Exception:
                elite_diagnostics['fragility_index'] = 0.5
                elite_diagnostics['fragility_warning'] = False
        
        n_obs_approx = len(returns)
        prior_scale_diag = 1.0 / max(n_obs_approx, 100)
        phi_lambda_eff_diag = PHI_SHRINKAGE_LAMBDA_DEFAULT * prior_scale_diag
        phi_tau_diag = _lambda_to_tau(phi_lambda_eff_diag)
        phi_prior_diag = _compute_phi_prior_diagnostics(
            phi_r=phi_opt,
            phi_global=PHI_SHRINKAGE_GLOBAL_DEFAULT,
            tau=phi_tau_diag,
            log_likelihood=ll_opt
        )

        diagnostics = {
            'nu_fixed': float(nu_fixed),
            'grid_best_q': float(10 ** grid_best[0]),
            'grid_best_c': float(10 ** grid_best[1]),
            'grid_best_phi': float(grid_best[2]),
            'refined_best_q': float(q_opt),
            'refined_best_c': float(c_opt),
            'refined_best_phi': float(phi_opt),
            'prior_applied': adaptive_lambda > 0,
            'prior_log_q_mean': float(adaptive_prior_mean),
            'prior_lambda': float(adaptive_lambda),
            'vol_cv': float(vol_cv),
            'rv_ratio': float(rv_ratio),
            'n_folds': int(len(fold_splits)),
            'optimization_successful': best_res is not None and (best_res.success if best_res else False),
            'elite_tuning_enabled': ELITE_TUNING_ENABLED,
            'elite_diagnostics': elite_diagnostics if ELITE_TUNING_ENABLED else None,
            **phi_prior_diag,
        }

        return q_opt, c_opt, phi_opt, ll_opt, diagnostics
