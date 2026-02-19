"""
===============================================================================
PHI-STUDENT-T DRIFT MODEL — Kalman Filter with AR(1) Drift and Student-t Noise
===============================================================================

Implements a state-space model with AR(1) drift and heavy-tailed observation noise:

    State equation:    μ_t = φ·μ_{t-1} + w_t,  w_t ~ N(0, q)
    Observation:       r_t = μ_t + ε_t,         ε_t ~ Student-t(ν, 0, scale_t)
    
    Where:
        - Observation variance: Var(ε_t) = c·σ_t²
        - Student-t scale: scale_t = sqrt(c·σ_t² × (ν-2)/ν) for ν > 2
        
    CRITICAL: The c parameter scales the observation VARIANCE, not the scale.
    For Student-t with ν degrees of freedom: Var = scale² × ν/(ν-2).
    So scale = sqrt(Var × (ν-2)/ν) = sqrt(c·σ_t² × (ν-2)/ν).

Parameters:
    q:   Process noise variance (drift evolution uncertainty)
    c:   Observation noise variance multiplier (scales EWMA variance σ_t²)
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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

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
        # MS-q and fused LFO-CV wrappers (February 2026)
        run_ms_q_student_t_filter,
        run_student_t_filter_with_lfo_cv,
        run_student_t_filter_with_lfo_cv_batch,
        # Unified filter (February 2026)
        run_unified_phi_student_t_filter,
        is_unified_filter_available,
    )
    _USE_NUMBA = is_numba_available()
    _MS_Q_NUMBA_AVAILABLE = _USE_NUMBA
    _UNIFIED_NUMBA_AVAILABLE = is_unified_filter_available() if is_numba_available() else False
except ImportError:
    _USE_NUMBA = False
    _MS_Q_NUMBA_AVAILABLE = False
    _UNIFIED_NUMBA_AVAILABLE = False
    run_phi_student_t_filter = None
    run_phi_student_t_filter_batch = None
    run_ms_q_student_t_filter = None
    run_student_t_filter_with_lfo_cv = None
    run_student_t_filter_with_lfo_cv_batch = None
    run_unified_phi_student_t_filter = None


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
#
# NOTE: No BMA penalties - all models compete fairly on equal footing.
# Standard Student-t has no penalty, so enhanced variants should not either.
# Model complexity is already penalized by BIC's parameter count.
# =============================================================================

# Vol-of-Vol (VoV) Enhancement
GAMMA_VOV_GRID = [0.3, 0.5, 0.7]
VOV_BMA_PENALTY = 0.0  # REMOVED: Equal competition with base Student-t

# Two-Piece Student-t Enhancement
NU_LEFT_GRID = [3, 4, 5]
NU_RIGHT_GRID = [8, 12, 20]
TWO_PIECE_BMA_PENALTY = 0.0  # REMOVED: Equal competition with base Student-t

# Two-Component Mixture Student-t Enhancement
NU_CALM_GRID = [12, 20]
NU_STRESS_GRID = [4, 6]
MIXTURE_WEIGHT_DEFAULT = 0.8
MIXTURE_WEIGHT_K = 2.0  # Sigmoid sensitivity to vol_relative
MIXTURE_BMA_PENALTY = 0.0  # REMOVED: Equal competition with base Student-t

# =============================================================================
# ENHANCED MIXTURE WEIGHT DYNAMICS (February 2026 - Expert Panel)
# =============================================================================
# Upgraded from reactive (vol-only) to multi-factor conditioning:
#   w_t = sigmoid(a × z_t + b × Δσ_t + c × M_t)
# Where:
#   z_t = standardized residuals (shock detection)
#   Δσ_t = vol acceleration (regime change detection)  
#   M_t = momentum (trend structure)
# =============================================================================

# Default mixture weight sensitivity parameters
MIXTURE_WEIGHT_A_SHOCK = 1.0       # Sensitivity to standardized residuals
MIXTURE_WEIGHT_B_VOL_ACCEL = 0.5   # Sensitivity to vol acceleration
MIXTURE_WEIGHT_C_MOMENTUM = 0.3    # Sensitivity to momentum


# =============================================================================
# MARKOV-SWITCHING PROCESS NOISE (MS-q) — February 2026
# =============================================================================
# Proactive regime-switching q based on volatility structure.
# Unlike GAS-Q (reactive to errors), MS-q shifts BEFORE errors materialize.
#
# PROBLEM WITH STATIC q:
#   Static process noise doesn't adapt to market regime changes.
#   During regime transitions, forecast errors spike before q catches up.
#
# PROBLEM WITH GAS-Q:
#   GAS-Q is reactive — it responds AFTER seeing large errors.
#   But regime changes are predictable from volatility structure.
#
# SOLUTION (MS-q):
#   Markov-Switching Process Noise with 2 states (calm, stress):
#       q_t = (1 - p_stress_t) × q_calm + p_stress_t × q_stress
#   where:
#       p_stress_t = sigmoid(sensitivity × (vol_relative_t - threshold))
#
# The transition probability is PROACTIVE — it shifts q BEFORE errors materialize.
#
# THEORETICAL BASIS:
# - Hamilton (1989) Markov-switching models
# - Regime-switching variance in financial econometrics
# - Consistent with Contaminated Student-t philosophy for nu
# =============================================================================

# MS-q Configuration
MS_Q_ENABLED = True           # Master switch for MS-q models
MS_Q_CALM_DEFAULT = 1e-6      # Process noise in calm regime
MS_Q_STRESS_DEFAULT = 1e-4    # Process noise in stress regime (100x calm)
MS_Q_SENSITIVITY = 2.0        # Sigmoid sensitivity to vol_relative
MS_Q_THRESHOLD = 1.3          # vol_relative threshold for transition
MS_Q_BMA_PENALTY = 0.0        # No penalty - fair competition via BIC


# =============================================================================
# UNIFIED STUDENT-T CONFIGURATION (February 2026 - Elite Architecture)
# =============================================================================
# Consolidates 48+ model variants into single adaptive architecture:
#   - Smooth asymmetric ν (replaces Two-Piece)
#   - Probabilistic MS-q (replaces threshold-based)
#   - VoV with redundancy damping
#   - Momentum integration
#   - State collapse regularization
# =============================================================================

@dataclass
class UnifiedStudentTConfig:
    """
    Configuration for Unified Elite Student-T Model.
    
    Combines ALL enhancements into single coherent architecture:
      1. Smooth Asymmetric ν: tanh-modulated tail heaviness
      2. Probabilistic MS-q: sigmoid regime switching
      3. Adaptive VoV: with redundancy damping
      4. Momentum: exogenous drift input
      5. State regularization: prevents φ→1/q→0 collapse
    """
    
    # Core parameters
    q: float = 1e-6
    c: float = 1.0
    phi: float = 0.0
    nu_base: float = 8.0
    
    # Smooth asymmetric ν: ν_eff = ν_base × (1 + α × tanh(k × z))
    # α < 0: heavier left tail (crashes), α > 0: heavier right tail
    alpha_asym: float = 0.0      # [-0.3, 0.3], asymmetry magnitude
    k_asym: float = 1.0          # [0.5, 2.0], transition sharpness
    
    # Probabilistic MS-q: p_stress = sigmoid(sensitivity × vol_zscore)
    q_calm: float = None         # Defaults to q if None
    q_stress_ratio: float = 10.0 # q_stress = q_calm × ratio
    ms_sensitivity: float = 2.0  # [1.0, 3.0], regime sensitivity
    
    # VoV with MS-q redundancy damping
    gamma_vov: float = 0.3       # [0, 1.0], VoV sensitivity
    vov_damping: float = 0.3     # [0, 0.5], reduce VoV when MS-q active
    vov_window: int = 20         # Rolling window for VoV
    
    # Momentum/exogenous input
    exogenous_input: np.ndarray = field(default=None, repr=False)
    
    # Data-driven bounds (auto-computed)
    c_min: float = 0.01
    c_max: float = 10.0
    q_min: float = 1e-8
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.q_calm is None:
            self.q_calm = self.q
        # Clip parameters to valid ranges
        self.alpha_asym = float(np.clip(self.alpha_asym, -0.3, 0.3))
        self.k_asym = float(np.clip(self.k_asym, 0.5, 2.0))
        self.ms_sensitivity = float(np.clip(self.ms_sensitivity, 1.0, 3.0))
        self.gamma_vov = float(np.clip(self.gamma_vov, 0.0, 1.0))
        self.vov_damping = float(np.clip(self.vov_damping, 0.0, 0.5))
    
    @property
    def q_stress(self) -> float:
        """Compute stress regime process noise."""
        return self.q_calm * self.q_stress_ratio
    
    @classmethod
    def auto_configure(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        nu_base: float = 8.0,
    ) -> 'UnifiedStudentTConfig':
        """
        Auto-configure from data characteristics.
        
        Uses robust statistics (MAD) for c bounds and
        data-driven initialization for asymmetry and VoV.
        """
        returns = np.asarray(returns).flatten()
        vol = np.asarray(vol).flatten()
        
        # Filter valid data
        valid_mask = np.isfinite(returns) & np.isfinite(vol) & (vol > 0)
        returns_clean = returns[valid_mask]
        vol_clean = vol[valid_mask]
        
        if len(returns_clean) < 30:
            # Not enough data, use defaults
            return cls(nu_base=nu_base)
        
        # Data-driven c bounds using robust MAD scale
        returns_mad = float(np.median(np.abs(returns_clean - np.median(returns_clean))))
        vol_median = float(np.median(vol_clean))
        
        if vol_median > 1e-10 and returns_mad > 0:
            # c_target such that c × vol² ≈ returns_variance
            returns_scale = returns_mad / 0.6745  # MAD to σ conversion
            c_target = (returns_scale / vol_median) ** 2
            c_target = float(np.clip(c_target, 0.01, 50.0))
            c_min = max(0.1 * c_target, 0.001)
            c_max = min(10.0 * c_target, 100.0)
        else:
            c_target = 1.0
            c_min = 0.01
            c_max = 10.0
        
        # VoV from realized vol-of-vol
        log_vol = np.log(np.maximum(vol_clean, 1e-10))
        if len(log_vol) > 1:
            vol_cv = float(np.std(np.diff(log_vol)))
            gamma_vov = 0.3 * float(np.clip(vol_cv / 0.02, 0, 1)) if vol_cv > 0.01 else 0.0
        else:
            gamma_vov = 0.0
        
        # Asymmetry from skewness (α = -0.1 × skewness)
        if len(returns_clean) > 30:
            ret_std = float(np.std(returns_clean))
            if ret_std > 1e-10:
                ret_centered = returns_clean - np.mean(returns_clean)
                skewness = float(np.mean((ret_centered / ret_std) ** 3))
                alpha_asym = -0.1 * float(np.clip(skewness, -3, 3))
            else:
                alpha_asym = 0.0
        else:
            alpha_asym = 0.0
        
        # MS-q: enable stronger switching if vol is volatile
        if len(log_vol) > 1:
            vol_cv = float(np.std(np.diff(log_vol)))
            q_stress_ratio = 10.0 if vol_cv > 0.02 else 5.0
        else:
            q_stress_ratio = 5.0
        
        # Scale-aware q_min
        q_min = max(1e-8, 0.001 * vol_median ** 2)
        
        return cls(
            nu_base=nu_base,
            c=c_target,
            c_min=c_min,
            c_max=c_max,
            q_min=q_min,
            gamma_vov=gamma_vov,
            alpha_asym=alpha_asym,
            q_stress_ratio=q_stress_ratio,
        )
    
    @classmethod
    def from_legacy(
        cls,
        q: float,
        c: float,
        phi: float,
        nu: float,
        **kwargs,
    ) -> 'UnifiedStudentTConfig':
        """Convert legacy tuned params to unified config with conservative defaults."""
        return cls(
            q=q,
            c=c,
            phi=phi,
            nu_base=nu,
            alpha_asym=0.0,      # Conservative: no asymmetry
            gamma_vov=0.3,       # Moderate VoV
            ms_sensitivity=2.0,  # Default sensitivity
            q_stress_ratio=10.0, # Standard stress ratio
            **kwargs,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'q': self.q,
            'c': self.c,
            'phi': self.phi,
            'nu_base': self.nu_base,
            'alpha_asym': self.alpha_asym,
            'k_asym': self.k_asym,
            'q_calm': self.q_calm,
            'q_stress_ratio': self.q_stress_ratio,
            'ms_sensitivity': self.ms_sensitivity,
            'gamma_vov': self.gamma_vov,
            'vov_damping': self.vov_damping,
            'c_min': self.c_min,
            'c_max': self.c_max,
            'q_min': self.q_min,
        }


def compute_ms_process_noise_smooth(
    vol: np.ndarray,
    q_calm: float,
    q_stress: float,
    sensitivity: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute smooth probabilistic MS process noise.
    
    Uses expanding-window z-score (no lookahead) with sigmoid transition.
    Sensitivity bounded to [1.0, 3.0] to prevent quasi-step behavior.
    
    This replaces the threshold-based MS-q with a fully smooth,
    differentiable regime switching mechanism.
    
    Args:
        vol: Time series of volatility estimates
        q_calm: Process noise for calm regime
        q_stress: Process noise for stress regime
        sensitivity: Sigmoid sensitivity (bounded to [1.0, 3.0])
        
    Returns:
        Tuple of (q_t, p_stress):
        - q_t: Time-varying process noise array
        - p_stress: Probability of stress regime array
    """
    vol = np.asarray(vol).flatten()
    n = len(vol)
    
    # Bound sensitivity to prevent quasi-step behavior
    sensitivity = float(np.clip(sensitivity, 1.0, 3.0))
    
    # Expanding-window statistics (no lookahead)
    vol_cumsum = np.cumsum(vol)
    vol_sq_cumsum = np.cumsum(vol ** 2)
    counts = np.arange(1, n + 1, dtype=np.float64)
    
    vol_mean = vol_cumsum / counts
    vol_var = vol_sq_cumsum / counts - vol_mean ** 2
    vol_var = np.maximum(vol_var, 1e-12)  # Prevent negative variance
    vol_std = np.sqrt(vol_var)
    
    # Warm-up: use first 20 obs for initial estimates
    warmup = min(20, n)
    if n > warmup:
        init_mean = np.mean(vol[:warmup])
        init_std = max(np.std(vol[:warmup]), 1e-6)
        vol_mean[:warmup] = init_mean
        vol_std[:warmup] = init_std
    
    # Z-score
    vol_zscore = (vol - vol_mean) / np.maximum(vol_std, 1e-6)
    
    # Smooth sigmoid transition
    p_stress = 1.0 / (1.0 + np.exp(-sensitivity * vol_zscore))
    p_stress = np.clip(p_stress, 0.01, 0.99)
    
    # Time-varying q
    q_t = (1.0 - p_stress) * q_calm + p_stress * q_stress
    
    return q_t, p_stress


def compute_ms_process_noise(
    vol: np.ndarray,
    q_calm: float = MS_Q_CALM_DEFAULT,
    q_stress: float = MS_Q_STRESS_DEFAULT,
    sensitivity: float = MS_Q_SENSITIVITY,
    threshold: float = MS_Q_THRESHOLD,
    vol_median: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Markov-switching process noise based on volatility regime.
    
    The key insight: volatility structure PREDICTS regime changes.
    When vol rises above median, we proactively increase q BEFORE
    forecast errors materialize.
    
    Args:
        vol: Time series of volatility estimates
        q_calm: Process noise for calm regime
        q_stress: Process noise for stress regime
        sensitivity: Sigmoid sensitivity to vol_relative
        threshold: vol_relative threshold for regime transition
        vol_median: Median volatility (computed if None)
        
    Returns:
        Tuple of (q_t, p_stress_t):
        - q_t: Time-varying process noise array
        - p_stress_t: Probability of stress regime array
    """
    vol = np.asarray(vol).flatten()
    n = len(vol)
    
    # Compute expanding median for normalization (no future leakage)
    if vol_median is None:
        vol_cumsum = np.cumsum(vol)
        vol_count = np.arange(1, n + 1)
        # Use expanding mean as proxy for median (faster, similar result)
        vol_baseline = vol_cumsum / vol_count
        # Warm-up: use first 20 observations for initial baseline
        if n > 20:
            vol_baseline[:20] = np.mean(vol[:20])
    else:
        vol_baseline = np.full(n, vol_median)
    
    # Prevent division by zero
    vol_baseline = np.maximum(vol_baseline, 1e-10)
    
    # Compute vol_relative
    vol_relative = vol / vol_baseline
    
    # Compute stress probability via sigmoid
    # p_stress = sigmoid(sensitivity × (vol_relative - threshold))
    z = sensitivity * (vol_relative - threshold)
    p_stress = 1.0 / (1.0 + np.exp(-z))
    
    # Clip to [0.01, 0.99] for numerical stability
    p_stress = np.clip(p_stress, 0.01, 0.99)
    
    # Compute time-varying q
    q_t = (1.0 - p_stress) * q_calm + p_stress * q_stress
    
    return q_t, p_stress


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
    
    Unlike static q or reactive GAS-q, MS-q adapts PROACTIVELY
    based on volatility regime before errors materialize.
    
    Args:
        y: Observations (returns)
        vol: Volatility estimates
        c: Observation noise scale
        phi: AR(1) coefficient
        nu: Degrees of freedom for Student-t
        q_calm: Process noise in calm regime
        q_stress: Process noise in stress regime
        sensitivity: Sigmoid sensitivity
        threshold: Vol_relative threshold
        
    Returns:
        Tuple of (mu_filtered, P_filtered, log_likelihood, q_t, p_stress_t)
    """
    y = np.asarray(y).flatten()
    vol = np.asarray(vol).flatten()
    n = len(y)
    nu = max(float(nu), 2.01)
    
    # Compute time-varying q
    q_t, p_stress = compute_ms_process_noise(
        vol, q_calm, q_stress, sensitivity, threshold
    )
    
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
        # FIX: Use z² = innovation² / scale_t² (consistent with log-likelihood)
        z_sq = z ** 2  # z = innovation / scale_t already computed above
        w_t = (nu + 1) / (nu + z_sq)
        
        # Kalman gain
        K_t = P_t / S_t if S_t > 1e-12 else 0.0
        
        # Weighted update
        mu_t = mu_t + K_t * w_t * innovation
        P_t = (1 - w_t * K_t) * P_t
        
        # Store filtered state
        mu[t] = mu_t
        P[t] = P_t
        
        # State prediction with TIME-VARYING q
        mu_t = phi * mu_t
        P_t = (phi ** 2) * P_t + q_t[t]
    
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
    def _variance_to_scale(variance: float, nu: float) -> float:
        """
        Convert predictive variance to Student-t scale parameter.
        
        For Student-t with ν degrees of freedom:
            Var(X) = scale² × ν/(ν-2)  when ν > 2
            
        So:
            scale = sqrt(Var × (ν-2)/ν)
            
        This is critical for correct PIT calibration. Using sqrt(Var) directly
        inflates the scale by sqrt(ν/(ν-2)), causing standardized residuals
        to be too small and PIT values to concentrate around 0.5.
        
        Args:
            variance: Predictive variance (P_pred + R)
            nu: Degrees of freedom
            
        Returns:
            Student-t scale parameter
        """
        if variance <= 1e-20:
            return 1e-10
        if nu > 2:
            return np.sqrt(variance * (nu - 2) / nu)
        else:
            # For ν ≤ 2, variance is infinite; use variance as proxy for scale²
            return np.sqrt(variance)

    @staticmethod
    def _variance_to_scale_vec(variance: np.ndarray, nu: float) -> np.ndarray:
        """Vectorized version of _variance_to_scale for array inputs."""
        variance_safe = np.maximum(variance, 1e-20)
        if nu > 2:
            scale = np.sqrt(variance_safe * (nu - 2) / nu)
        else:
            scale = np.sqrt(variance_safe)
        return np.where(scale < 1e-10, 1e-10, scale)

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
        Compute smooth asymmetric effective ν using tanh modulation.
        
        Formula:
            ν_eff = ν_base × (1 + α × tanh(k × z))
            
        where z = innovation / scale is the standardized residual.
        
        Properties:
            - Differentiable everywhere (smooth, no discontinuities)
            - Bounded: ν_eff ∈ [ν_base×0.7, ν_base×1.3] for |α|≤0.3
            - α < 0: heavier left tail (crashes get lower ν)
            - α > 0: heavier right tail (recoveries get lower ν)
            - CRITICAL: Always returns ν_eff > 2.1 to ensure variance defined
        
        This replaces the hard Two-Piece switch with smooth, differentiable
        asymmetry that doesn't create optimizer instability.
        
        Args:
            nu_base: Base degrees of freedom
            innovation: y_t - μ_pred (residual)
            scale: Predictive standard deviation sqrt(S)
            alpha: Asymmetry parameter in [-0.3, 0.3]
            k: Transition sharpness in [0.5, 2.0]
            nu_min: Minimum ν (must be > 2 for finite variance)
            nu_max: Maximum ν
            
        Returns:
            Effective degrees of freedom ν_eff
        """
        # Standardized residual
        scale_safe = max(abs(scale), 1e-10)
        z = innovation / scale_safe
        
        # Smooth asymmetric modulation via tanh
        # tanh is bounded in [-1, 1] and smooth everywhere
        modulation = 1.0 + alpha * np.tanh(k * z)
        nu_raw = nu_base * modulation
        
        # CRITICAL: Ensure ν > 2 (finite variance requirement)
        return float(np.clip(nu_raw, nu_min, nu_max))

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
            # FIX: Convert variance S to Student-t scale: scale = sqrt(S × (ν-2)/ν)
            if nu_val > 2:
                forecast_scale = np.sqrt(S * (nu_val - 2) / nu_val)
            else:
                forecast_scale = np.sqrt(S)
            if forecast_scale > 1e-12:
                z = innovation / forecast_scale
                ll_t = log_norm_const - np.log(forecast_scale) + neg_exp * np.log(1.0 + z * z * inv_nu)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t
        
        return mu_filtered, P_filtered, float(log_likelihood)
    
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
        Kalman filter with exogenous input in state equation.
        
        STATE-EQUATION INTEGRATION (Elite Upgrade - February 2026):
            μ_t = φ × μ_{t-1} + u_t + w_t
            r_t = μ_t + ε_t,  ε_t ~ t(ν)
        
        Preserves probabilistic coherence — likelihood computed correctly.
        
        EXPERT VALIDATED:
        - mu_pred includes u_t (coherent likelihood)
        - scale_t uses variance parameterization (consistent with rest of codebase)
        - gammaln imported at module level (Expert #2)
        
        Args:
            returns: Array of returns
            vol: Array of volatility estimates
            q: Process noise variance
            c: Observation noise scale
            phi: AR(1) persistence
            nu: Degrees of freedom
            exogenous_input: Array of u_t values (α×MOM - β×MR)
            
        Returns:
            Tuple of (mu_filtered, P_filtered, log_likelihood)
        """
        n = len(returns)
        
        # Convert to contiguous float64 arrays
        returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)
        
        # Extract scalar values
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))
        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)
        
        # Pre-compute constants
        phi_sq = phi_val * phi_val
        nu_adjust = min(nu_val / (nu_val + 3.0), 1.0)
        
        # Pre-compute log-pdf constants (gammaln imported at module level - Expert #2)
        log_norm_const = gammaln((nu_val + 1.0) / 2.0) - gammaln(nu_val / 2.0) - 0.5 * np.log(nu_val * np.pi)
        neg_exp = -((nu_val + 1.0) / 2.0)
        inv_nu = 1.0 / nu_val
        
        # Pre-compute R values
        R = c_val * (vol * vol)
        
        # Allocate output arrays
        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        
        # State initialization
        mu = 0.0
        P = 1e-4
        log_likelihood = 0.0
        
        # Main filter loop with exogenous input
        for t in range(n):
            # Exogenous input (KEY: injected into state equation)
            u_t = exogenous_input[t] if exogenous_input is not None and t < len(exogenous_input) else 0.0
            
            # Prediction step INCLUDES exogenous input (Expert #1: coherent)
            mu_pred = phi_val * mu + u_t
            P_pred = phi_sq * P + q_val
            
            # Observation update
            S = P_pred + R[t]
            if S <= 1e-12:
                S = 1e-12
            
            innovation = returns[t] - mu_pred
            K = nu_adjust * P_pred / S
            
            # State update with robust Student-t weighting
            z_sq = (innovation ** 2) / S
            w_t = (nu_val + 1.0) / (nu_val + z_sq)
            
            mu = mu_pred + K * w_t * innovation
            P = (1.0 - w_t * K) * P_pred
            if P < 1e-12:
                P = 1e-12
            
            # Store filtered values
            mu_filtered[t] = mu
            P_filtered[t] = P
            
            # Log-likelihood (coherent with u_t - Expert #1)
            # FIX: Convert variance S to Student-t scale
            if nu_val > 2:
                forecast_scale = np.sqrt(S * (nu_val - 2) / nu_val)
            else:
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
    def filter_phi_with_predictive(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        nu: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        ELITE FIX: φ-Student-t filter returning PREDICTIVE values for proper PIT.
        
        CRITICAL BUG IN EXISTING pit_ks:
        The existing code computes PIT using POSTERIOR values (mu_filtered, P_filtered),
        but PIT requires PRIOR PREDICTIVE values (mu_pred, S_pred = P_pred + R).
        
        Using posterior values makes residuals look artificially concentrated because
        the posterior has already "seen" the observation y_t.
        
        This method returns both filtered AND predictive values:
            mu_pred[t] = φ × μ_{t-1}     (BEFORE seeing y_t)
            S_pred[t] = P_pred + R_t      (BEFORE seeing y_t)
        
        For proper PIT:
            z_t = (y_t - mu_pred[t]) / scale_pred[t]
            PIT_t = F_ν(z_t)
        
        Where scale_pred = sqrt(S_pred × (ν-2)/ν) for Student-t.
        
        Args:
            returns: Return observations
            vol: Volatility estimates
            q: Process noise variance
            c: Observation noise scale  
            phi: AR(1) persistence
            nu: Degrees of freedom
            
        Returns:
            Tuple of:
            - mu_filtered: Posterior mean (after update)
            - P_filtered: Posterior variance (after update)
            - mu_pred: Prior predictive mean (BEFORE seeing y_t)
            - S_pred: Prior predictive variance (BEFORE seeing y_t) = P_pred + R
            - log_likelihood: Total log-likelihood
        """
        n = len(returns)
        
        # Convert to contiguous float64 arrays
        returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)
        
        # Extract scalar values
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))
        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)
        
        # Pre-compute constants
        phi_sq = phi_val * phi_val
        nu_adjust = min(nu_val / (nu_val + 3.0), 1.0)
        
        # Pre-compute log-pdf constants
        log_norm_const = gammaln((nu_val + 1.0) / 2.0) - gammaln(nu_val / 2.0) - 0.5 * np.log(nu_val * np.pi)
        neg_exp = -((nu_val + 1.0) / 2.0)
        inv_nu = 1.0 / nu_val
        
        # Pre-compute R values
        R = c_val * (vol * vol)
        
        # Allocate output arrays
        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        mu_pred_arr = np.empty(n, dtype=np.float64)  # PREDICTIVE mean
        S_pred_arr = np.empty(n, dtype=np.float64)   # PREDICTIVE variance
        
        # State initialization
        mu = 0.0
        P = 1e-4
        log_likelihood = 0.0
        
        # Main filter loop
        for t in range(n):
            # =========================================================
            # PREDICTION STEP (BEFORE seeing y_t)
            # =========================================================
            mu_pred = phi_val * mu
            P_pred = phi_sq * P + q_val
            
            # Predictive variance = state uncertainty + observation noise
            S = P_pred + R[t]
            if S <= 1e-12:
                S = 1e-12
            
            # STORE PREDICTIVE VALUES (critical for proper PIT!)
            mu_pred_arr[t] = mu_pred
            S_pred_arr[t] = S
            
            # =========================================================
            # UPDATE STEP (AFTER seeing y_t)
            # =========================================================
            innovation = returns[t] - mu_pred
            K = nu_adjust * P_pred / S
            
            # State update
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12
            
            # Store filtered (posterior) values
            mu_filtered[t] = mu
            P_filtered[t] = P
            
            # Log-likelihood contribution
            # FIX: Convert variance S to Student-t scale
            if nu_val > 2:
                forecast_scale = np.sqrt(S * (nu_val - 2) / nu_val)
            else:
                forecast_scale = np.sqrt(S)
            if forecast_scale > 1e-12:
                z = innovation / forecast_scale
                ll_t = log_norm_const - np.log(forecast_scale) + neg_exp * np.log(1.0 + z * z * inv_nu)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t
        
        return mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, float(log_likelihood)

    @staticmethod
    def pit_ks_predictive(
        returns: np.ndarray,
        mu_pred: np.ndarray,
        S_pred: np.ndarray,
        nu: float,
    ) -> Tuple[float, float]:
        """
        ELITE FIX: Proper PIT/KS using PREDICTIVE distribution.
        
        This is the correct PIT calculation:
            z_t = (y_t - μ_pred_t) / scale_t
            PIT_t = F_ν(z_t)
        
        Where:
            μ_pred_t = φ × μ_{t-1}  (predictive mean BEFORE y_t)
            S_pred_t = P_pred + R_t  (predictive variance BEFORE y_t)
            scale_t = sqrt(S_pred × (ν-2)/ν)  (Student-t scale parameter)
        
        The key insight: PIT transforms y_t through the CDF of its PRIOR
        predictive distribution (before y_t is observed). Using posterior
        values (after the update) is mathematically incorrect and causes
        the systematic miscalibration seen in the diagnostics.
        
        Args:
            returns: Observed returns
            mu_pred: Predictive means (from filter_phi_with_predictive)
            S_pred: Predictive variances (from filter_phi_with_predictive)
            nu: Degrees of freedom
            
        Returns:
            Tuple of (KS statistic, KS p-value)
        """
        returns_flat = np.asarray(returns).flatten()
        mu_pred_flat = np.asarray(mu_pred).flatten()
        S_pred_flat = np.asarray(S_pred).flatten()
        
        # Ensure nu is valid
        nu_safe = max(nu, 2.01)
        
        # Compute Student-t scale from predictive variance
        # For Student-t: Var = scale² × ν/(ν-2)
        # So: scale = sqrt(Var × (ν-2)/ν)
        if nu_safe > 2:
            scale_factor = (nu_safe - 2) / nu_safe
        else:
            scale_factor = 1.0
        
        forecast_scale = np.sqrt(np.maximum(S_pred_flat * scale_factor, 1e-20))
        forecast_scale = np.where(forecast_scale < 1e-10, 1e-10, forecast_scale)
        
        # Standardize using predictive mean and scale
        standardized = (returns_flat - mu_pred_flat) / forecast_scale
        
        # Handle NaN/Inf
        valid_mask = np.isfinite(standardized)
        if not np.any(valid_mask):
            return 1.0, 0.0
        
        standardized_clean = standardized[valid_mask]
        
        # Compute PIT values
        pit_values = student_t.cdf(standardized_clean, df=nu_safe)
        
        if len(pit_values) < 2:
            return 1.0, 0.0
        
        # KS test against uniform
        ks_result = kstest(pit_values, 'uniform')
        return float(ks_result.statistic), float(ks_result.pvalue)

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

            # FIX: Convert variance S to Student-t scale
            if nu_val > 2:
                forecast_scale = np.sqrt(S * (nu_val - 2) / nu_val)
            else:
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
        
        FIX #3: Correct scale parameterization for Student-t CDF.
        scipy.stats.t.cdf expects standardized = (y - μ) / scale
        where Var = scale² × ν/(ν-2), so scale = sqrt(Var × (ν-2)/ν)
        """
        returns_flat = np.asarray(returns).flatten()
        mu_flat = np.asarray(mu_filtered).flatten()
        vol_flat = np.asarray(vol).flatten()
        P_flat = np.asarray(P_filtered).flatten()

        # Ensure nu is valid for Student-t (must be > 2 for finite variance)
        nu_safe = max(nu, 2.01)
        
        # FIX #3: Compute correct Student-t scale (not variance!)
        # For Student-t, Var = scale² × ν/(ν-2)
        # So scale = sqrt(Var × (ν-2)/ν)
        forecast_var = c * (vol_flat ** 2) + P_flat
        
        if nu_safe > 2:
            # Correct scale: sqrt(variance × (ν-2)/ν)
            forecast_scale = np.sqrt(np.maximum(forecast_var * (nu_safe - 2) / nu_safe, 1e-20))
        else:
            forecast_scale = np.sqrt(np.maximum(forecast_var, 1e-20))
        
        forecast_scale = np.where(forecast_scale < 1e-10, 1e-10, forecast_scale)
        
        standardized = (returns_flat - mu_flat) / forecast_scale
        
        # Handle any remaining NaN/Inf values
        valid_mask = np.isfinite(standardized)
        if not np.any(valid_mask):
            return 1.0, 0.0
        
        standardized_clean = standardized[valid_mask]
        
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
            # FIX: Convert variance S to Student-t scale
            if nu_val > 2:
                forecast_scale = np.sqrt(S * (nu_val - 2) / nu_val)
            else:
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
            # FIX: Convert variance S to Student-t scale using chosen inv_nu
            # inv_nu = 1/nu, so (nu-2)/nu = 1 - 2*inv_nu
            scale_factor = 1.0 - 2.0 * inv_nu
            if scale_factor > 0:
                forecast_scale = np.sqrt(S * scale_factor)
            else:
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
            # FIX: Each mixture component needs its own scale conversion
            # For component with ν_C: scale_C = sqrt(S × (ν_C-2)/ν_C)
            # For component with ν_S: scale_S = sqrt(S × (ν_S-2)/ν_S)
            scale_factor_C = max(1.0 - 2.0 * inv_nu_C, 0.01)  # (ν_C-2)/ν_C
            scale_factor_S = max(1.0 - 2.0 * inv_nu_S, 0.01)  # (ν_S-2)/ν_S
            forecast_scale_C = np.sqrt(S * scale_factor_C)
            forecast_scale_S = np.sqrt(S * scale_factor_S)
            
            if forecast_scale_C > 1e-12 and forecast_scale_S > 1e-12:
                z_C = innovation / forecast_scale_C
                z_S = innovation / forecast_scale_S
                
                # Calm component log-pdf with its own scale
                ll_C = log_norm_C - np.log(forecast_scale_C) + neg_exp_C * np.log(1.0 + z_C * z_C * inv_nu_C)
                
                # Stress component log-pdf with its own scale
                ll_S = log_norm_S - np.log(forecast_scale_S) + neg_exp_S * np.log(1.0 + z_S * z_S * inv_nu_S)
                
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

    @classmethod
    def filter_phi_mixture_enhanced(
        cls, 
        returns: np.ndarray, 
        vol: np.ndarray, 
        q: float, 
        c: float, 
        phi: float, 
        nu_calm: float,
        nu_stress: float,
        w_base: float = 0.5,
        a_shock: float = MIXTURE_WEIGHT_A_SHOCK,
        b_vol_accel: float = MIXTURE_WEIGHT_B_VOL_ACCEL,
        c_momentum: float = MIXTURE_WEIGHT_C_MOMENTUM,
        momentum_window: int = 21
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        φ-Student-t filter with ENHANCED multi-factor mixture weight dynamics.
        
        Expert Panel Enhancement (February 2026):
        Instead of reactive (vol-only) weighting, uses multi-factor conditioning:
        
            w_t = sigmoid(a × z_t + b × Δσ_t + c × M_t)
        
        Where:
            z_t = standardized residual (shock detection)
            Δσ_t = vol acceleration (regime change detection)
            M_t = normalized momentum (trend structure)
        
        This makes the mixture respond to:
            - Shocks (extreme residuals)
            - Volatility expansion/contraction
            - Trend structure
        
        Args:
            returns: Return series
            vol: EWMA/GK volatility series
            q: Process noise variance
            c: Observation noise scale
            phi: AR(1) persistence
            nu_calm: Degrees of freedom for calm regime
            nu_stress: Degrees of freedom for stress regime
            w_base: Base weight parameter (sigmoid intercept)
            a_shock: Sensitivity to standardized residuals
            b_vol_accel: Sensitivity to vol acceleration
            c_momentum: Sensitivity to momentum
            momentum_window: Window for momentum calculation
            
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
        
        # =====================================================================
        # MULTI-FACTOR WEIGHT COMPONENTS
        # =====================================================================
        
        # 1. Vol acceleration: Δlog(σ_t) normalized
        log_vol = np.log(np.maximum(vol, 1e-10))
        vol_accel = np.zeros(n)
        vol_accel[1:] = np.diff(log_vol)
        # Normalize by rolling std
        vol_accel_std = np.std(vol_accel[vol_accel != 0]) if np.any(vol_accel != 0) else 1.0
        vol_accel_z = vol_accel / max(vol_accel_std, 1e-6)
        
        # 2. Momentum: Rolling mean return normalized
        momentum = np.zeros(n)
        ret_std = np.std(returns) if np.std(returns) > 0 else 1.0
        for t in range(momentum_window, n):
            momentum[t] = np.mean(returns[t-momentum_window:t]) / ret_std
        # Fill early values
        for t in range(1, min(momentum_window, n)):
            momentum[t] = np.mean(returns[:t]) / ret_std
        
        # 3. Standardized residuals (computed during filtering, initialized to 0)
        # Will be updated in the loop based on previous residual
        z_prev = 0.0
        
        # Pre-compute log-pdf constants for both ν values
        log_norm_C = gammaln((nu_C + 1.0) / 2.0) - gammaln(nu_C / 2.0) - 0.5 * np.log(nu_C * np.pi)
        neg_exp_C = -((nu_C + 1.0) / 2.0)
        inv_nu_C = 1.0 / nu_C
        
        log_norm_S = gammaln((nu_S + 1.0) / 2.0) - gammaln(nu_S / 2.0) - 0.5 * np.log(nu_S * np.pi)
        neg_exp_S = -((nu_S + 1.0) / 2.0)
        inv_nu_S = 1.0 / nu_S
        
        # Use average ν for Kalman gain
        nu_avg = w_b * nu_C + (1 - w_b) * nu_S
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
        
        # Main filter loop with enhanced mixture likelihood
        for t in range(n):
            # Prediction step
            mu_pred = phi_val * mu
            P_pred = phi_sq * P + q_val
            
            # Observation update
            S = P_pred + R[t]
            if S <= 1e-12:
                S = 1e-12
            
            innovation = returns[t] - mu_pred
            
            # FIX: Convert variance to Student-t scale for each component
            scale_factor_C = max(1.0 - 2.0 * inv_nu_C, 0.01)  # (ν_C-2)/ν_C
            scale_factor_S = max(1.0 - 2.0 * inv_nu_S, 0.01)  # (ν_S-2)/ν_S
            forecast_scale_C = np.sqrt(S * scale_factor_C)
            forecast_scale_S = np.sqrt(S * scale_factor_S)
            # For z_t used in mixture weight, use average scale factor
            avg_scale_factor = (scale_factor_C + scale_factor_S) / 2
            forecast_scale_avg = np.sqrt(S * avg_scale_factor)
            
            # Current standardized residual (for mixture weight calculation)
            z_t = innovation / forecast_scale_avg if forecast_scale_avg > 1e-12 else 0.0
            
            # =====================================================================
            # ENHANCED MIXTURE WEIGHT CALCULATION
            # w_t = sigmoid(a × z_prev + b × Δσ_t + c × M_t - w_base)
            # Higher values → more CALM weight (lower stress)
            # Shocks (high |z|), vol expansion, negative momentum → stress
            # =====================================================================
            composite_signal = (
                -a_shock * abs(z_prev) +          # Shocks reduce calm weight
                -b_vol_accel * vol_accel_z[t] +   # Vol expansion reduces calm weight
                c_momentum * momentum[t] +         # Positive momentum increases calm
                w_b                                # Base intercept
            )
            # Sigmoid transformation
            exponent = np.clip(-composite_signal, -50, 50)
            w_calm_t = 1.0 / (1.0 + np.exp(exponent))
            
            # Update z_prev for next iteration
            z_prev = z_t
            
            K = nu_adjust * P_pred / S
            
            # State update
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12
            
            # Store filtered values
            mu_filtered[t] = mu
            P_filtered[t] = P
            
            # Mixture log-pdf calculation with component-specific scales
            if forecast_scale_C > 1e-12 and forecast_scale_S > 1e-12:
                z_C = innovation / forecast_scale_C
                z_S = innovation / forecast_scale_S
                
                # Calm component log-pdf with its own scale
                ll_C = log_norm_C - np.log(forecast_scale_C) + neg_exp_C * np.log(1.0 + z_C * z_C * inv_nu_C)
                
                # Stress component log-pdf with its own scale
                ll_S = log_norm_S - np.log(forecast_scale_S) + neg_exp_S * np.log(1.0 + z_S * z_S * inv_nu_S)
                
                # Mixture log-pdf via log-sum-exp
                ll_max = max(ll_C, ll_S)
                ll_mix = ll_max + np.log(
                    w_calm_t * np.exp(ll_C - ll_max) + 
                    (1.0 - w_calm_t) * np.exp(ll_S - ll_max)
                )
                
                if np.isfinite(ll_mix):
                    log_likelihood += ll_mix

        return mu_filtered, P_filtered, float(log_likelihood)

    # =========================================================================
    # UNIFIED STUDENT-T FILTER (February 2026 - Elite Architecture)
    # =========================================================================
    # Consolidates 48+ model variants into single adaptive architecture.
    # Combines: VoV, Smooth Asymmetric ν, Probabilistic MS-q, Momentum
    # =========================================================================
    
    @classmethod
    def filter_phi_unified(
        cls, 
        returns: np.ndarray, 
        vol: np.ndarray, 
        config: 'UnifiedStudentTConfig',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        UNIFIED Elite φ-Student-t filter combining ALL enhancements.
        
        This is the canonical production filter that merges:
          1. Smooth Asymmetric ν: tanh-modulated tail heaviness (differentiable)
          2. Probabilistic MS-q: sigmoid regime switching (no hard thresholds)
          3. Adaptive VoV: with MS-q redundancy damping
          4. Momentum: exogenous drift input
          5. Robust Student-t weighting: outlier downweighting
        
        Uses Numba-accelerated kernel when available (~10x speedup).
        Falls back to pure Python implementation when Numba not available.
        
        All enhancements operate smoothly and are fully differentiable,
        enabling stable optimization without gradient discontinuities.
        
        Args:
            returns: Return series
            vol: EWMA/GK volatility series
            config: UnifiedStudentTConfig instance with all parameters
            
        Returns:
            Tuple of (mu_filtered, P_filtered, mu_pred, S_pred, log_likelihood):
            - mu_filtered: Posterior state mean
            - P_filtered: Posterior state variance
            - mu_pred: Prior predictive mean (for PIT)
            - S_pred: Prior predictive variance (for PIT)
            - log_likelihood: Total log-likelihood
        """
        n = len(returns)
        
        # Convert to contiguous arrays
        returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)
        
        # Extract config values with safety
        q_base = float(config.q)
        c_val = float(config.c)
        phi_val = float(np.clip(config.phi, -0.999, 0.999))
        nu_base = float(config.nu_base)
        alpha = float(config.alpha_asym)
        k_asym = float(config.k_asym)
        gamma_vov = float(config.gamma_vov)
        damping = float(config.vov_damping)
        vov_window = int(config.vov_window)
        
        # MS-q setup
        q_calm = float(config.q_calm) if config.q_calm is not None else q_base
        q_stress = q_calm * float(config.q_stress_ratio)
        ms_enabled = abs(q_stress - q_calm) > 1e-12
        
        # Compute MS-q time series using smooth probabilistic function
        if ms_enabled:
            q_t, p_stress = compute_ms_process_noise_smooth(
                vol, q_calm, q_stress, config.ms_sensitivity
            )
        else:
            q_t = np.full(n, q_base)
            p_stress = np.zeros(n)
        
        # Compute VoV rolling
        log_vol = np.log(np.maximum(vol, 1e-10))
        vov_rolling = np.zeros(n)
        for t in range(vov_window, n):
            vov_rolling[t] = np.std(log_vol[t-vov_window:t])
        if n > vov_window:
            vov_rolling[:vov_window] = vov_rolling[vov_window]
        
        # Prepare momentum array
        if config.exogenous_input is not None:
            momentum = np.ascontiguousarray(
                config.exogenous_input[:n].flatten(), dtype=np.float64
            )
            if len(momentum) < n:
                # Pad with zeros
                momentum = np.pad(momentum, (0, n - len(momentum)), mode='constant')
        else:
            momentum = np.zeros(n, dtype=np.float64)
        
        # =====================================================================
        # Try Numba-accelerated kernel (10x speedup)
        # =====================================================================
        if _UNIFIED_NUMBA_AVAILABLE and run_unified_phi_student_t_filter is not None:
            try:
                return run_unified_phi_student_t_filter(
                    returns, vol,
                    c_val, phi_val, nu_base,
                    q_t, p_stress,
                    vov_rolling, gamma_vov, damping,
                    alpha, k_asym,
                    momentum, 1e-4  # P0
                )
            except Exception:
                # Fall through to Python implementation
                pass
        
        # =====================================================================
        # Pure Python fallback implementation
        # =====================================================================
        phi_sq = phi_val * phi_val
        R_base = c_val * (vol ** 2)
        
        # Allocate output arrays
        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        mu_pred_arr = np.empty(n, dtype=np.float64)
        S_pred_arr = np.empty(n, dtype=np.float64)
        
        # State initialization
        mu = 0.0
        P = 1e-4
        log_likelihood = 0.0
        
        # Main filter loop with ALL enhancements
        for t in range(n):
            # === PREDICTION STEP ===
            # Momentum: inject exogenous input if provided
            u_t = 0.0
            if config.exogenous_input is not None and t < len(config.exogenous_input):
                u_t = float(config.exogenous_input[t])
            
            # MS-q: time-varying process noise
            q_t_val = q_t[t]
            
            # State prediction with momentum
            mu_pred = phi_val * mu + u_t
            P_pred = phi_sq * P + q_t_val
            
            # VoV with redundancy damping (reduce VoV when MS-q active)
            vov_effective = gamma_vov * (1.0 - damping * p_stress[t])
            R = R_base[t] * (1.0 + vov_effective * vov_rolling[t])
            
            # Predictive variance
            S = P_pred + R
            if S <= 1e-12:
                S = 1e-12
            
            # Store predictive values (for proper PIT computation)
            mu_pred_arr[t] = mu_pred
            S_pred_arr[t] = S
            
            # Innovation
            innovation = returns[t] - mu_pred
            
            # === UPDATE STEP ===
            # Smooth asymmetric ν (replaces hard Two-Piece switch)
            scale = np.sqrt(S)
            nu_eff = cls.compute_effective_nu(nu_base, innovation, scale, alpha, k_asym)
            
            # ν-adjusted Kalman gain
            nu_adjust = min(nu_eff / (nu_eff + 3.0), 1.0)
            K = nu_adjust * P_pred / S
            
            # Robust Student-t weighting (downweight outliers)
            z_sq = (innovation ** 2) / S
            w_t = (nu_eff + 1.0) / (nu_eff + z_sq)
            
            # State update with robust weighting
            mu = mu_pred + K * w_t * innovation
            P = (1.0 - w_t * K) * P_pred
            if P < 1e-12:
                P = 1e-12
            
            # Store filtered values
            mu_filtered[t] = mu
            P_filtered[t] = P
            
            # === LOG-LIKELIHOOD ===
            # Convert variance S to Student-t scale (CRITICAL for correct PIT)
            scale_factor = max((nu_eff - 2.0) / nu_eff, 0.01)
            forecast_scale = np.sqrt(S * scale_factor)
            
            if forecast_scale > 1e-12:
                z = innovation / forecast_scale
                inv_nu = 1.0 / nu_eff
                log_norm = (gammaln((nu_eff + 1.0) / 2.0) - gammaln(nu_eff / 2.0) 
                           - 0.5 * np.log(nu_eff * np.pi))
                neg_exp = -((nu_eff + 1.0) / 2.0)
                
                ll_t = log_norm - np.log(forecast_scale) + neg_exp * np.log(1.0 + z * z * inv_nu)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t
        
        return mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, float(log_likelihood)

    @classmethod
    def optimize_params_unified(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        nu_base: float = 8.0,
        train_frac: float = 0.7,
        asset_symbol: str = None,
    ) -> Tuple['UnifiedStudentTConfig', Dict]:
        """
        Staged optimization for unified model.
        
        Optimization is performed in stages to avoid feature interaction instability:
          Stage 1: Base parameters (q, c, φ) with state regularization
          Stage 2: VoV gamma (freeze base)
          Stage 3: MS-q sensitivity (freeze base + VoV)
          Stage 4: Asymmetry alpha (freeze all, fine-tune)
        
        Includes:
          - Data-driven bounds (MAD-based c, scale-aware q_min)
          - State regularization (prevents φ→1/q→0 collapse)
          - Parameter regularization (prevents overfitting)
          - Hessian condition check with graceful degradation
        
        Args:
            returns: Return series
            vol: EWMA/GK volatility series
            nu_base: Base degrees of freedom (from discrete grid)
            train_frac: Fraction for train/test split
            asset_symbol: Asset symbol for logging
            
        Returns:
            Tuple of (UnifiedStudentTConfig, diagnostics_dict)
        """
        returns = np.asarray(returns).flatten()
        vol = np.asarray(vol).flatten()
        
        # Auto-configure initial bounds from data
        config = UnifiedStudentTConfig.auto_configure(returns, vol, nu_base)
        
        n = len(returns)
        n_train = int(n * train_frac)
        returns_train = returns[:n_train]
        vol_train = vol[:n_train]
        
        # =====================================================================
        # STAGE 1: Base parameters (q, c, φ) with state regularization
        # =====================================================================
        def neg_ll_base(params):
            log_q, c, phi = params
            q = 10 ** log_q
            
            # Create config with base params only (disable enhancements)
            cfg = UnifiedStudentTConfig(
                q=q, c=c, phi=phi, nu_base=nu_base,
                alpha_asym=0.0,      # No asymmetry
                gamma_vov=0.0,       # No VoV
                q_stress_ratio=1.0,  # No MS-q
            )
            
            try:
                _, _, _, _, ll = cls.filter_phi_unified(returns_train, vol_train, cfg)
            except Exception:
                return 1e10
            
            if not np.isfinite(ll):
                return 1e10
            
            # State regularization: smooth quadratic penalty for collapse
            phi_pen = max(0.0, abs(phi) - 0.95)
            q_pen = max(0.0, -7.0 - log_q)
            state_reg = 50.0 * (phi_pen ** 2 + q_pen ** 2)
            
            # Extra penalty for joint collapse (φ→1 AND q→0)
            if phi_pen > 0 and q_pen > 0:
                state_reg += 30.0 * phi_pen * q_pen
            
            return -ll / n_train + state_reg
        
        # Initial guess
        log_q_init = np.log10(max(config.q_min * 10, 1e-7))
        c_init = (config.c_min + config.c_max) / 2
        x0_base = [log_q_init, c_init, 0.0]
        
        bounds_base = [
            (np.log10(config.q_min), -2),   # log₁₀(q)
            (config.c_min, config.c_max),    # c
            (-0.99, 0.99),                   # φ
        ]
        
        try:
            result_base = minimize(
                neg_ll_base, x0_base, bounds=bounds_base, 
                method='L-BFGS-B', options={'maxiter': 200}
            )
            stage1_success = result_base.success
        except Exception as e:
            stage1_success = False
            result_base = None
        
        if not stage1_success:
            # Graceful degradation: return auto-configured defaults
            return config, {
                "stage": 0, 
                "success": False, 
                "error": "Stage 1 optimization failed",
                "degraded": True,
            }
        
        log_q_opt, c_opt, phi_opt = result_base.x
        q_opt = 10 ** log_q_opt
        
        # =====================================================================
        # STAGE 2: VoV gamma (freeze base parameters)
        # =====================================================================
        def neg_ll_vov(gamma_arr):
            gamma = gamma_arr[0]
            cfg = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_base,
                alpha_asym=0.0,
                gamma_vov=gamma,
                q_stress_ratio=1.0,
            )
            try:
                _, _, _, _, ll = cls.filter_phi_unified(returns_train, vol_train, cfg)
            except Exception:
                return 1e10
            
            if not np.isfinite(ll):
                return 1e10
            
            # Regularization: keep gamma near prior (0.3)
            reg = 10.0 * (gamma - 0.3) ** 2
            return -ll / n_train + reg
        
        try:
            result_vov = minimize(
                neg_ll_vov, [config.gamma_vov], 
                bounds=[(0.0, 1.0)], method='L-BFGS-B'
            )
            gamma_opt = result_vov.x[0] if result_vov.success else config.gamma_vov
        except Exception:
            gamma_opt = config.gamma_vov
        
        # =====================================================================
        # STAGE 3: MS-q sensitivity (freeze base + VoV)
        # =====================================================================
        def neg_ll_msq(sens_arr):
            sens = sens_arr[0]
            cfg = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_base,
                alpha_asym=0.0,
                gamma_vov=gamma_opt,
                ms_sensitivity=sens,
                q_stress_ratio=10.0,
            )
            try:
                _, _, _, _, ll = cls.filter_phi_unified(returns_train, vol_train, cfg)
            except Exception:
                return 1e10
            
            if not np.isfinite(ll):
                return 1e10
            
            # Regularization: keep sensitivity near 2.0
            reg = 10.0 * (sens - 2.0) ** 2
            return -ll / n_train + reg
        
        try:
            result_msq = minimize(
                neg_ll_msq, [2.0], 
                bounds=[(1.0, 3.0)], method='L-BFGS-B'
            )
            sens_opt = result_msq.x[0] if result_msq.success else 2.0
        except Exception:
            sens_opt = 2.0
        
        # =====================================================================
        # STAGE 4: Asymmetry alpha (freeze all, fine-tune)
        # =====================================================================
        def neg_ll_asym(alpha_arr):
            alpha = alpha_arr[0]
            cfg = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_base,
                alpha_asym=alpha,
                gamma_vov=gamma_opt,
                ms_sensitivity=sens_opt,
                q_stress_ratio=10.0,
            )
            try:
                _, _, _, _, ll = cls.filter_phi_unified(returns_train, vol_train, cfg)
            except Exception:
                return 1e10
            
            if not np.isfinite(ll):
                return 1e10
            
            # Regularization: keep alpha small (penalize strong asymmetry)
            reg = 10.0 * alpha ** 2
            return -ll / n_train + reg
        
        try:
            result_asym = minimize(
                neg_ll_asym, [config.alpha_asym], 
                bounds=[(-0.3, 0.3)], method='L-BFGS-B'
            )
            alpha_opt = result_asym.x[0] if result_asym.success else config.alpha_asym
        except Exception:
            alpha_opt = config.alpha_asym
        
        # =====================================================================
        # HESSIAN CONDITION CHECK (graceful degradation)
        # =====================================================================
        try:
            if hasattr(result_base, 'hess_inv'):
                hess_inv = result_base.hess_inv
                if hasattr(hess_inv, 'todense'):
                    hess_inv = hess_inv.todense()
                cond_num = float(np.linalg.cond(hess_inv))
            else:
                cond_num = 1.0
        except Exception:
            cond_num = 1e10
        
        degraded = False
        if cond_num > 1e6:
            # Graceful degradation: disable advanced features
            gamma_opt = 0.0
            sens_opt = 2.0
            alpha_opt = 0.0
            degraded = True
        
        # =====================================================================
        # BUILD FINAL CONFIG
        # =====================================================================
        final_config = UnifiedStudentTConfig(
            q=q_opt,
            c=c_opt,
            phi=phi_opt,
            nu_base=nu_base,
            alpha_asym=alpha_opt,
            gamma_vov=gamma_opt,
            ms_sensitivity=sens_opt,
            q_stress_ratio=10.0,
            vov_damping=0.3,
            q_min=config.q_min,
            c_min=config.c_min,
            c_max=config.c_max,
        )
        
        diagnostics = {
            "stage": 4,
            "success": True,
            "degraded": degraded,
            "hessian_cond": cond_num,
            "q": float(q_opt),
            "c": float(c_opt),
            "phi": float(phi_opt),
            "log10_q": float(log_q_opt),
            "gamma_vov": float(gamma_opt),
            "ms_sensitivity": float(sens_opt),
            "alpha_asym": float(alpha_opt),
            "c_bounds": (float(config.c_min), float(config.c_max)),
            "q_min": float(config.q_min),
        }
        
        return final_config, diagnostics

    @classmethod
    def pit_ks_unified(
        cls,
        returns: np.ndarray,
        mu_pred: np.ndarray,
        S_pred: np.ndarray,
        config: 'UnifiedStudentTConfig',
    ) -> Tuple[float, float, Dict]:
        """
        PIT/KS calibration for unified model using PREDICTIVE distribution.
        
        Uses predictive values (mu_pred, S_pred) rather than posterior,
        which is the correct approach for proper PIT calibration.
        
        Args:
            returns: Return series
            mu_pred: Predictive means from filter
            S_pred: Predictive variances from filter
            config: UnifiedStudentTConfig instance
            
        Returns:
            Tuple of (ks_statistic, ks_pvalue, metrics_dict)
        """
        returns = np.asarray(returns).flatten()
        mu_pred = np.asarray(mu_pred).flatten()
        S_pred = np.asarray(S_pred).flatten()
        
        n = len(returns)
        pit_values = np.empty(n)
        
        nu_base = config.nu_base
        alpha = config.alpha_asym
        k_asym = config.k_asym
        
        for t in range(n):
            innovation = returns[t] - mu_pred[t]
            scale = np.sqrt(max(S_pred[t], 1e-12))
            
            # Effective ν (smooth asymmetric)
            nu_eff = cls.compute_effective_nu(nu_base, innovation, scale, alpha, k_asym)
            
            # Student-t scale (variance to scale conversion)
            if nu_eff > 2:
                t_scale = np.sqrt(S_pred[t] * (nu_eff - 2) / nu_eff)
            else:
                t_scale = scale
            
            t_scale = max(t_scale, 1e-10)
            
            # PIT value via Student-t CDF
            pit_values[t] = student_t.cdf(innovation, df=nu_eff, loc=0, scale=t_scale)
        
        # Clean and compute KS
        valid = np.isfinite(pit_values)
        pit_clean = np.clip(pit_values[valid], 0, 1)
        
        if len(pit_clean) < 20:
            return 1.0, 0.0, {"n_samples": len(pit_clean), "calibrated": False}
        
        ks_result = kstest(pit_clean, 'uniform')
        
        # Histogram MAD for practical calibration grading
        hist, _ = np.histogram(pit_clean, bins=10, range=(0, 1))
        hist_freq = hist / len(pit_clean)
        hist_mad = float(np.mean(np.abs(hist_freq - 0.1)))
        
        # Calibration grade (A/B/C/F)
        if hist_mad < 0.02:
            grade = "A"
        elif hist_mad < 0.05:
            grade = "B"
        elif hist_mad < 0.10:
            grade = "C"
        else:
            grade = "F"
        
        metrics = {
            "n_samples": len(pit_clean),
            "ks_statistic": float(ks_result.statistic),
            "ks_pvalue": float(ks_result.pvalue),
            "histogram_mad": hist_mad,
            "calibration_grade": grade,
            "calibrated": hist_mad < 0.05,
        }
        
        return float(ks_result.statistic), float(ks_result.pvalue), metrics


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
            
            # ================================================================
            # ELITE FIX 3: Smooth quadratic φ-q regularization
            # ================================================================
            # Prevents deterministic state evolution collapse (φ→1 AND q→0)
            # Strong penalty (500×) because this is the ROOT CAUSE of PIT failures
            # ================================================================
            phi_near_one_penalty = max(0.0, abs(phi_clip) - 0.95) ** 2
            q_very_small_penalty = max(0.0, -7.0 - log_q) ** 2
            state_regularization = -500.0 * (phi_near_one_penalty + q_very_small_penalty)

            penalized_ll = avg_ll + log_prior_q + log_prior_c + log_prior_phi + log_prior_nu + calibration_penalty + state_regularization
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
    def get_c_bounds_for_asset(
        asset_symbol: str = None,
        vol_median: float = None,
        returns_std: float = None,
        returns_mad: float = None,
    ) -> Tuple[float, float]:
        """
        Return (c_min, c_max) bounds adaptive to asset class and data scale.
        
        ELITE FIX (February 2026): Data-driven c bounds using robust MAD scale.
        
        The key insight: c should be such that R = c × vol² ≈ observed return variance.
        When c is constrained to wrong bounds, the model can't fit properly,
        causing systematic PIT miscalibration.
        
        Primary method: Data-driven from returns/vol ratio using robust MAD scale.
        Fallback: Asset class detection by symbol.
        
        Args:
            asset_symbol: Ticker symbol for asset class detection
            vol_median: Median volatility for scale-based detection
            returns_std: Standard deviation of returns
            returns_mad: Median Absolute Deviation of returns (robust scale)
            
        Returns:
            Tuple of (c_min, c_max) bounds
        """
        # =================================================================
        # PRIMARY: Data-driven c bounds from returns/vol ratio
        # =================================================================
        # c_target = (returns_scale / vol_median)² such that R ≈ Var(returns)
        # Use MAD for robustness (MAD ≈ 0.6745 × σ for Gaussian)
        
        if vol_median is not None and vol_median > 1e-10:
            # Prefer robust MAD scale, fallback to std
            if returns_mad is not None and returns_mad > 0:
                # Convert MAD to σ-equivalent: σ ≈ MAD / 0.6745
                returns_scale = returns_mad / 0.6745
            elif returns_std is not None and returns_std > 0:
                returns_scale = returns_std
            else:
                returns_scale = None
            
            if returns_scale is not None:
                # Target c such that c × vol² ≈ returns_variance
                c_target = (returns_scale / vol_median) ** 2
                
                # Cap c_target to prevent explosion in micro-vol regimes
                c_target = min(c_target, 50.0)
                
                # Wide bounds around target (0.1× to 10×)
                c_min = max(0.1 * c_target, 0.001)
                c_max = min(10.0 * c_target, 20.0)
                
                # Ensure reasonable spread
                if c_max < 2 * c_min:
                    c_max = 2 * c_min
                
                return (c_min, c_max)
        
        # =================================================================
        # FALLBACK: Asset class detection by symbol
        # =================================================================
        if asset_symbol:
            symbol_upper = asset_symbol.upper()
            
            # FX pairs: much smaller returns, need smaller c
            if '=X' in symbol_upper:
                return (0.005, 0.5)
            
            # JPY crosses (USDJPY, EURJPY, etc.)
            if symbol_upper.endswith('JPY') or symbol_upper.startswith('JPY'):
                return (0.005, 0.5)
            
            # Other FX-like patterns
            if any(ccy in symbol_upper for ccy in ['USD', 'EUR', 'GBP', 'CHF', 'AUD', 'NZD', 'CAD']):
                if len(symbol_upper) <= 8 and not symbol_upper.startswith('^'):
                    if 'USD' in symbol_upper and len(symbol_upper) == 6:
                        return (0.005, 0.5)
            
            # Crypto: very high volatility, need larger c
            if symbol_upper in ['BTC-USD', 'ETH-USD', 'BTC', 'ETH']:
                return (0.5, 10.0)
            if 'BITCOIN' in symbol_upper or 'CRYPTO' in symbol_upper:
                return (0.5, 10.0)
            
            # Indices: typically smooth
            if symbol_upper.startswith('^'):
                return (0.1, 3.0)
        
        # Detection by scale (fallback)
        if returns_std is not None and returns_std < 0.005:
            return (0.005, 0.5)
        
        if vol_median is not None and vol_median < 0.01:
            return (0.005, 0.5)
        
        # Default: equities
        return (0.2, 5.0)

    @staticmethod
    def optimize_params_fixed_nu(
        returns: np.ndarray,
        vol: np.ndarray,
        nu: float,
        train_frac: float = 0.7,
        q_min: float = None,  # ELITE FIX: Now auto-computed if None
        q_max: float = 1e-1,
        c_min: float = None,  # FIX #4: Now auto-detected if None
        c_max: float = None,  # FIX #4: Now auto-detected if None
        phi_min: float = -0.999,
        phi_max: float = 0.999,
        prior_log_q_mean: float = -6.0,
        prior_lambda: float = 1.0,
        asset_symbol: str = None,  # FIX #4: Asset symbol for c-bounds detection
    ) -> Tuple[float, float, float, float, Dict]:
        """
        Optimize (q, c, φ) for the φ-Student-t drift model with FIXED ν.
        
        This method is part of the discrete ν grid approach:
        - ν is held fixed (passed as argument, not optimized)
        - Only q, c, φ are optimized via CV MLE
        - Each ν value becomes a separate sub-model in BMA
        
        FIX #4 (February 2026): Asset-class adaptive c bounds.
        If c_min/c_max are None, they are auto-detected based on:
        - Asset symbol (FX, crypto, indices, equities)
        - Returns scale (std of returns)
        - Volatility scale (median of vol)
        
        ELITE FIX (February 2026): Scale-aware q_min.
        If q_min is None, it is auto-computed based on observation variance
        to prevent deterministic state collapse (φ→1, q→0).
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
        
        # Compute robust MAD scale for c bounds
        ret_mad = float(np.median(np.abs(returns_robust - np.median(returns_robust))))
        vol_median = float(np.median(vol))
        vol_var_median = vol_median ** 2
        
        # ================================================================
        # ELITE FIX 2: Scale-aware q_min to prevent deterministic state collapse
        # ================================================================
        # q_min should be ~0.1% of observation noise scale
        # This prevents φ→1, q→10^-12 collapse that causes overconfident forecasts
        if q_min is None:
            q_min = max(1e-10, 0.001 * vol_var_median)
            # Hard floor at 1e-8 to prevent deterministic state
            q_min = max(q_min, 1e-8)
        
        # FIX #4: Auto-detect c bounds based on asset class (now with MAD)
        if c_min is None or c_max is None:
            c_min_auto, c_max_auto = PhiStudentTDriftModel.get_c_bounds_for_asset(
                asset_symbol=asset_symbol,
                vol_median=vol_median,
                returns_std=ret_std,
                returns_mad=ret_mad,  # ELITE FIX: Pass MAD for robust scale
            )
            c_min = c_min if c_min is not None else c_min_auto
            c_max = c_max if c_max is not None else c_max_auto

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
                            # FIX: Convert variance to Student-t scale
                            # For Student-t: Var = scale² × ν/(ν-2), so scale = sqrt(Var × (ν-2)/ν)
                            if nu_fixed > 2:
                                forecast_scale = np.sqrt(forecast_var * (nu_fixed - 2) / nu_fixed)
                            else:
                                forecast_scale = np.sqrt(forecast_var)
                            ll_contrib = PhiStudentTDriftModel.logpdf(ret_t, nu_fixed, mu_pred, forecast_scale)
                            ll_fold += ll_contrib
                            if len(all_standardized) < 1000:
                                all_standardized.append(float(innovation / forecast_scale))

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
            
            # ================================================================
            # ELITE FIX 3: Smooth quadratic φ-q regularization
            # ================================================================
            # Prevents deterministic state evolution collapse (φ→1 AND q→0)
            # When |φ| > 0.95 AND log₁₀(q) < -7, state becomes deterministic
            # → Predictive variance collapses → Overconfident forecasts → PIT failure
            #
            # Uses smooth quadratic penalties (not multiplicative) for optimizer stability
            # Strong penalty (500×) because this is the ROOT CAUSE of PIT failures
            # ================================================================
            phi_near_one_penalty = max(0.0, abs(phi_clip) - 0.95) ** 2
            q_very_small_penalty = max(0.0, -7.0 - log_q) ** 2
            state_regularization = -500.0 * (phi_near_one_penalty + q_very_small_penalty)

            penalized_ll = avg_ll + log_prior_q + log_prior_c + log_prior_phi + calibration_penalty + state_regularization
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
