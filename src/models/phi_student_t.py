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
    
    # ELITE CALIBRATION FIX (February 2026): Variance inflation
    # Multiplies S_pred by β to ensure predictive variance ≈ returns variance
    # β < 1: model was over-estimating variance (rare)
    # β > 1: model was under-estimating variance (common for q→0 collapse)
    variance_inflation: float = 1.0  # [0.5, 5.0], optimized for PIT uniformity
    
    # ELITE CALIBRATION FIX (February 2026): Mean drift correction
    # Equities have positive risk premium (~15-25% annualized) that the zero-mean
    # Kalman filter doesn't capture. This causes systematic positive innovations
    # and right-skewed PIT histogram.
    # mu_drift = mean(returns - mu_pred) on training data
    mu_drift: float = 0.0  # Mean bias correction for PIT
    
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
        # FIX (Feb 2026): More sensitive formula for gamma variation across assets
        log_vol = np.log(np.maximum(vol_clean, 1e-10))
        if len(log_vol) > 1:
            vol_cv = float(np.std(np.diff(log_vol)))
            # Linear scaling: vol_cv typical range [0.01, 0.05] -> gamma [0.15, 0.75]
            # Formula: gamma = 15 * vol_cv, clipped to [0.0, 1.0]
            gamma_vov = float(np.clip(15.0 * vol_cv, 0.0, 1.0)) if vol_cv > 0.005 else 0.0
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
        
        # =====================================================================
        # ELITE CALIBRATION FIX (February 2026): Scale-aware q_min
        # =====================================================================
        # Previous: q_min = max(1e-8, 0.001 * vol_median²) was too aggressive
        # This caused log₁₀(q) → -7.0 collapse for low-vol assets (COST, SO, IBM, FX)
        # 
        # Mathematical insight: For proper PIT calibration, process noise q must
        # contribute meaningfully to predictive variance S = P_pred + R
        # 
        # If q → 0, then in steady state P_pred → q/(1-φ²) → 0
        # This makes S ≈ R = c×σ², and any mismatch between EWMA σ and true vol
        # causes systematic under-estimation of predictive variance → U-shaped PIT
        #
        # Fix: Set q_min so that q contributes at least 5% of observation variance
        obs_var = vol_median ** 2
        ret_var = float(np.var(returns_clean)) if len(returns_clean) > 30 else obs_var
        
        # q_min should be:
        # 1. At least 1e-6 (absolute floor - prevents numerical issues)
        # 2. At least 5% of observation variance (ensures q contributes to S)
        # 3. At least 2% of return variance (ensures calibration feasibility)
        q_min = max(1e-6, 0.05 * obs_var, 0.02 * ret_var)
        
        # Compute calibration-aware initial q from excess return variance
        # If ret_var >> c×vol², q should absorb ~50% of the difference
        excess_var = max(0.0, ret_var - c_target * obs_var)
        q_init = max(q_min, 0.5 * excess_var) if excess_var > 0 else q_min * 10
        
        return cls(
            nu_base=nu_base,
            q=q_init,  # Initialize with calibration-aware value
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


def compute_optimal_variance_inflation(
    returns: np.ndarray,
    mu_pred: np.ndarray,
    S_pred: np.ndarray,
    nu: float,
    beta_min: float = 0.1,
    beta_max: float = 5.0,
) -> float:
    """
    ELITE CALIBRATION FIX (February 2026): Find optimal variance scaling beta.
    
    FIX: Use ANALYTICAL SOLUTION based on variance ratio, not histogram MAD.
    
    The core insight: For calibrated PIT, we need:
        Var(innovations) = mean(beta * S_pred)
    
    Therefore:
        beta_optimal = Var(innovations) / mean(S_pred)
    
    Args:
        returns: Observed returns
        mu_pred: Predictive means from filter
        S_pred: Predictive variances from filter
        nu: Degrees of freedom for Student-t
        beta_min: Minimum variance inflation
        beta_max: Maximum variance inflation
        
    Returns:
        Optimal beta that calibrates variance
    """
    returns = np.asarray(returns).flatten()
    mu_pred = np.asarray(mu_pred).flatten()
    S_pred = np.asarray(S_pred).flatten()
    
    # Compute actual innovation variance
    innovations = returns - mu_pred
    actual_var = float(np.var(innovations))
    
    # Mean predicted variance
    mean_S = float(np.mean(S_pred))
    
    # Analytical solution: beta = actual_var / mean_S
    if mean_S > 1e-12:
        beta_analytical = actual_var / mean_S
    else:
        beta_analytical = 1.0
    
    # Clip to valid range
    beta_opt = float(np.clip(beta_analytical, beta_min, beta_max))
    
    return beta_opt


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
        
        # Pre-compute log-pdf constants (gammaln imported at module level - Expert #2)
        log_norm_const = gammaln((nu_val + 1.0) / 2.0) - gammaln(nu_val / 2.0) - 0.5 * np.log(nu_val * np.pi)
        neg_exp = -((nu_val + 1.0) / 2.0)
        inv_nu = 1.0 / nu_val
        
        # Pre-compute R values
        R = c_val * (vol * vol)
        
        # Allocate output arrays
        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        mu_pred_arr = np.empty(n, dtype=np.float64)
        S_pred_arr = np.empty(n, dtype=np.float64)
        
        # State initialization
        mu = 0.0
        P = 1e-4
        log_likelihood = 0.0
        
        # Main filter loop
        for t in range(n):
            # === PREDICTION ===
            # Momentum input
            u_t = 0.0
            if config.exogenous_input is not None and t < len(config.exogenous_input):
                u_t = float(config.exogenous_input[t])
            
            # MS-q: time-varying process noise
            q_t_val = q_t[t]
            
            # State prediction
            mu_pred = phi_val * mu + u_t
            P_pred = phi_sq * P + q_t_val
            
            # VoV with redundancy damping
            vov_effective = gamma_vov * (1.0 - damping * p_stress[t])
            R = R_base[t] * (1.0 + vov_effective * vov_rolling[t])
            
            # Predictive variance
            S = P_pred + R
            if S <= 1e-12:
                S = 1e-12
            
            # Store predictive values (CRITICAL for proper PIT)
            mu_pred_arr[t] = mu_pred
            S_pred_arr[t] = S
            
            # Innovation
            innovation = returns[t] - mu_pred
            
            # === UPDATE ===
            # Smooth asymmetric ν
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
            # Convert variance to Student-t scale using effective ν
            if nu_eff > 2:
                scale_factor = (nu_eff - 2) / nu_eff
            else:
                scale_factor = 0.5  # Fallback for ν ≤ 2
            forecast_scale = np.sqrt(S * scale_factor)
            
            if forecast_scale > 1e-12:
                z = innovation / forecast_scale
                # Recompute log-norm for effective ν
                log_norm_eff = gammaln((nu_eff + 1.0) / 2.0) - gammaln(nu_eff / 2.0) - 0.5 * np.log(nu_eff * np.pi)
                neg_exp_eff = -((nu_eff + 1.0) / 2.0)
                inv_nu_eff = 1.0 / nu_eff
                
                ll_t = log_norm_eff - np.log(forecast_scale) + neg_exp_eff * np.log(1.0 + z * z * inv_nu_eff)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t
        
        return mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, float(log_likelihood)

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
        
        ELITE FIX (February 2026): Applies variance_inflation β to S_pred
        to calibrate predictive variance for uniform PIT distribution.
        
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
        nu_base = config.nu_base
        alpha = config.alpha_asym
        k_asym = config.k_asym
        
        # ELITE calibration parameters
        variance_inflation = getattr(config, 'variance_inflation', 1.0)
        mu_drift = getattr(config, 'mu_drift', 0.0)
        
        # =========================================================================
        # ELITE CALIBRATION PIPELINE (International Quant Literature - Feb 2026)
        # =========================================================================
        # V3 WAVELET-ENHANCED: Combines DTCWT (UK), Asymmetric GAS (Renaissance),
        # Wavelet Nu (Chinese), Hansen Skew-t (German), Beta Calibration (MIT)
        # =========================================================================
        elite_enabled = True
        ks_raw_pvalue = None
        elite_diag = {}
        
        try:
            from .elite_pit_diagnostics import (
                compute_elite_calibrated_pit,
                compute_elite_calibrated_pit_v2,
                compute_elite_calibrated_pit_v3,
                compute_berkowitz_lr_test,
                compute_pit_autocorrelation,
            )
            
            if elite_enabled and n >= 100:
                # Use V3 elite wavelet-enhanced calibration pipeline
                pit_calibrated, ks_pvalue_calib, elite_diag = compute_elite_calibrated_pit_v3(
                    returns=returns,
                    mu_pred=mu_pred,
                    S_pred=S_pred,
                    nu=nu_base,
                    variance_inflation=variance_inflation,
                    mu_drift=mu_drift,
                    use_wavelet_vol=True,       # DTCWT multi-scale (UK/Cambridge)
                    use_asymmetric_gas=True,    # Renaissance leverage effect
                    use_wavelet_nu=True,        # Chinese realized kurtosis
                    use_beta_calibration=True,  # MIT ensemble
                    use_dynamic_skew=True,      # German Hansen skew-t
                    train_frac=0.7,
                )
                pit_clean = pit_calibrated
                ks_pvalue = ks_pvalue_calib
                ks_stat = float(kstest(pit_clean, 'uniform').statistic)
                ks_raw_pvalue = elite_diag.get('ks_pvalue_raw', ks_pvalue)
            else:
                raise ValueError("Use fallback")
                
        except (ImportError, ValueError, Exception):
            # Fallback to basic implementation
            pit_values = np.empty(n)
            
            for t in range(n):
                innovation = returns[t] - mu_pred[t] - mu_drift
                S_calibrated = S_pred[t] * variance_inflation
                
                if nu_base > 2:
                    t_scale_base = np.sqrt(max(S_calibrated, 1e-12) * (nu_base - 2) / nu_base)
                else:
                    t_scale_base = np.sqrt(max(S_calibrated, 1e-12))
                scale = t_scale_base
                
                nu_eff = cls.compute_effective_nu(nu_base, innovation, scale, alpha, k_asym)
                
                if nu_eff > 2:
                    t_scale = np.sqrt(S_calibrated * (nu_eff - 2) / nu_eff)
                else:
                    t_scale = scale
                t_scale = max(t_scale, 1e-10)
                
                pit_values[t] = student_t.cdf(innovation, df=nu_eff, loc=0, scale=t_scale)
            
            valid = np.isfinite(pit_values)
            pit_clean = np.clip(pit_values[valid], 0, 1)
            ks_result = kstest(pit_clean, 'uniform')
            ks_stat = float(ks_result.statistic)
            ks_pvalue = float(ks_result.pvalue)
            ks_raw_pvalue = ks_pvalue
        
        if len(pit_clean) < 20:
            return 1.0, 0.0, {"n_samples": len(pit_clean), "calibrated": False}
        
        # Histogram MAD for practical calibration grading
        hist, _ = np.histogram(pit_clean, bins=10, range=(0, 1))
        hist_freq = hist / len(pit_clean)
        hist_mad = float(np.mean(np.abs(hist_freq - 0.1)))
        
        # ELITE: Berkowitz LR test (institutional VaR standard)
        try:
            from .elite_pit_diagnostics import compute_berkowitz_lr_test, compute_pit_autocorrelation
            berkowitz_lr, berkowitz_p, berk_diag = compute_berkowitz_lr_test(pit_clean)
            pit_acf = compute_pit_autocorrelation(pit_clean)
        except ImportError:
            berkowitz_lr, berkowitz_p = float('nan'), float('nan')
            pit_acf = {}
        
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
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "ks_pvalue_raw": ks_raw_pvalue,
            "ks_improvement": float(ks_pvalue - ks_raw_pvalue) if ks_raw_pvalue else 0.0,
            "histogram_mad": hist_mad,
            "calibration_grade": grade,
            "calibrated": hist_mad < 0.05,
            # ELITE diagnostics
            "berkowitz_lr": float(berkowitz_lr) if np.isfinite(berkowitz_lr) else None,
            "berkowitz_pvalue": float(berkowitz_p) if np.isfinite(berkowitz_p) else None,
            "pit_autocorr_lag1": pit_acf.get('autocorrelations', {}).get('lag_1'),
            "ljung_box_pvalue": pit_acf.get('ljung_box_pvalue'),
            "has_dynamic_misspec": pit_acf.get('has_autocorrelation', False),
            "gas_volatility_enabled": elite_diag.get('gas_enabled', False),
            "isotonic_calibration_enabled": elite_diag.get('isotonic_enabled', False),
        }
        
        return ks_stat, ks_pvalue, metrics


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
            scale_factor = max(1.0 - 2.0 * inv_nu, 0.01)  # (ν-2)/ν
            forecast_scale = np.sqrt(S * scale_factor)
            
            if forecast_scale > 1e-12:
                z = innovation / forecast_scale
                ll_t = log_norm - np.log(forecast_scale) + neg_exp * np.log(1.0 + z * z * inv_nu)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t
        
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
            z_t = standardized residuals (shock detection)
            Δσ_t = vol acceleration (regime change detection)  
            M_t = momentum (trend structure)
        
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
            w_base: Base weight parameter (default 0.5)
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
            K = nu_adjust * P_pred / S
            
            # State update
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12
            
            # Store filtered values
            mu_filtered[t] = mu
            P_filtered[t] = P
            
            # Compute component-specific scales BEFORE using them
            scale_factor_C = max(1.0 - 2.0 * inv_nu_C, 0.01)  # (ν_C-2)/ν_C
            scale_factor_S = max(1.0 - 2.0 * inv_nu_S, 0.01)  # (ν_S-2)/ν_S
            forecast_scale_C = np.sqrt(S * scale_factor_C)
            forecast_scale_S = np.sqrt(S * scale_factor_S)
            
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
                    w_calm[t] * np.exp(ll_C - ll_max) + 
                    (1.0 - w_calm[t]) * np.exp(ll_S - ll_max)
                )
                
                if np.isfinite(ll_mix):
                    log_likelihood += ll_mix

        return mu_filtered, P_filtered, float(log_likelihood)

    @classmethod
    def filter_phi_unified(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        config: 'UnifiedStudentTConfig',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        UNIFIED Elite φ-Student-t filter combining ALL enhancements.
        
        =========================================================================
        ELITE CALIBRATION ENGINE (February 2026)
        =========================================================================
        
        Integrates:
          1. Smooth Asymmetric ν: tanh-modulated tail heaviness
          2. Probabilistic MS-q: sigmoid regime switching
          3. Adaptive VoV: with MS-q redundancy damping
          4. Momentum: exogenous drift input
          5. Robust Student-t weighting: outlier downweighting
        
        CRITICAL FOR PIT CALIBRATION:
        Returns PREDICTIVE values (mu_pred, S_pred) for proper PIT computation.
        Using posterior values causes artificial concentration around 0.5.
        
        Args:
            returns: Return series
            vol: EWMA/GK volatility series
            config: UnifiedStudentTConfig with all parameters
            
        Returns:
            Tuple of (mu_filtered, P_filtered, mu_pred, S_pred, log_likelihood)
            - mu_pred, S_pred are PRIOR PREDICTIVE (before seeing y_t)
        """
        n = len(returns)
        
        # Convert to contiguous arrays
        returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)
        
        # Extract config values
        q_base = float(config.q)
        c_val = float(config.c)
        phi_val = float(np.clip(config.phi, -0.999, 0.999))
        nu_base = float(config.nu_base)
        alpha = float(config.alpha_asym)
        k_asym = float(config.k_asym)
        gamma_vov = float(config.gamma_vov)
        damping = float(config.vov_damping)
        
        # MS-q setup
        q_calm = float(config.q_calm) if config.q_calm is not None else q_base
        q_stress = q_calm * float(config.q_stress_ratio)
        ms_enabled = abs(q_stress - q_calm) > 1e-12
        
        # Compute MS-q time series
        if ms_enabled:
            q_t, p_stress = compute_ms_process_noise_smooth(
                vol, q_calm, q_stress, config.ms_sensitivity
            )
        else:
            q_t = np.full(n, q_base)
            p_stress = np.zeros(n)
        
        # Compute VoV rolling (20-day window)
        log_vol = np.log(np.maximum(vol, 1e-10))
        vov_rolling = np.zeros(n)
        window = config.vov_window
        for t in range(window, n):
            vov_rolling[t] = np.std(log_vol[t-window:t])
        if n > window:
            vov_rolling[:window] = vov_rolling[window] if n > window else 0.0
        
        # Try Numba-accelerated version first
        if _UNIFIED_NUMBA_AVAILABLE:
            try:
                momentum = config.exogenous_input if config.exogenous_input is not None else np.zeros(n)
                momentum = np.ascontiguousarray(momentum.flatten(), dtype=np.float64)
                if len(momentum) < n:
                    momentum = np.concatenate([momentum, np.zeros(n - len(momentum))])
                
                return run_unified_phi_student_t_filter(
                    returns, vol, c_val, phi_val, nu_base,
                    q_t, p_stress,
                    vov_rolling, gamma_vov, damping,
                    alpha, k_asym,
                    momentum, 1e-4
                )
            except Exception:
                pass  # Fall through to Python implementation
        
        # Python implementation
        phi_sq = phi_val * phi_val
        
        # Pre-compute base R
        R_base = c_val * (vol ** 2)
        
        # Pre-compute log-pdf constants
        log_norm_const = gammaln((nu_base + 1.0) / 2.0) - gammaln(nu_base / 2.0) - 0.5 * np.log(nu_base * np.pi)
        neg_exp = -((nu_base + 1.0) / 2.0)
        inv_nu = 1.0 / nu_base
        
        # Allocate output arrays
        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        mu_pred_arr = np.empty(n, dtype=np.float64)
        S_pred_arr = np.empty(n, dtype=np.float64)
        
        # State initialization
        mu = 0.0
        P = 1e-4
        log_likelihood = 0.0
        
        # Main filter loop
        for t in range(n):
            # === PREDICTION ===
            # Momentum input
            u_t = 0.0
            if config.exogenous_input is not None and t < len(config.exogenous_input):
                u_t = float(config.exogenous_input[t])
            
            # MS-q: time-varying process noise
            q_t_val = q_t[t]
            
            # State prediction
            mu_pred = phi_val * mu + u_t
            P_pred = phi_sq * P + q_t_val
            
            # VoV with redundancy damping
            vov_effective = gamma_vov * (1.0 - damping * p_stress[t])
            R = R_base[t] * (1.0 + vov_effective * vov_rolling[t])
            
            # Predictive variance
            S = P_pred + R
            if S <= 1e-12:
                S = 1e-12
            
            # Store predictive values (CRITICAL for proper PIT)
            mu_pred_arr[t] = mu_pred
            S_pred_arr[t] = S
            
            # Innovation
            innovation = returns[t] - mu_pred
            
            # === UPDATE ===
            # Smooth asymmetric ν
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
            # Convert variance to Student-t scale using effective ν
            if nu_eff > 2:
                scale_factor = (nu_eff - 2) / nu_eff
            else:
                scale_factor = 0.5  # Fallback for ν ≤ 2
            forecast_scale = np.sqrt(S * scale_factor)
            
            if forecast_scale > 1e-12:
                z = innovation / forecast_scale
                # Recompute log-norm for effective ν
                log_norm_eff = gammaln((nu_eff + 1.0) / 2.0) - gammaln(nu_eff / 2.0) - 0.5 * np.log(nu_eff * np.pi)
                neg_exp_eff = -((nu_eff + 1.0) / 2.0)
                inv_nu_eff = 1.0 / nu_eff
                
                ll_t = log_norm_eff - np.log(forecast_scale) + neg_exp_eff * np.log(1.0 + z * z * inv_nu_eff)
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
            
            # FIX: Reduced reg
            reg = 1.0 * (gamma - config.gamma_vov) ** 2
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
            
            # FIX: Reduced regularization centered on data-driven prior
            reg = 1.0 * (alpha - config.alpha_asym) ** 2
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
        elif cond_num < 1e-6:
            # Warning: very flat region, optimization may be unstable
            pass
        
        # =====================================================================
        # STAGE 5: ELITE VARIANCE, MEAN & NU CALIBRATION (February 2026)
        # =====================================================================
        # Compute:
        #   1. Optimal nu: Select nu from grid that maximizes KS p-value
        #   2. variance_inflation β: E[innovation²] / E[S_pred]
        #   3. mu_drift: mean(innovation) - accounts for equity risk premium
        # All computed on TRAINING DATA ONLY to prevent information leakage
        
        # Grid search for optimal nu
        NU_GRID = [4, 6, 8, 10, 12]
        best_nu = nu_base
        best_ks_p = 0.0
        best_beta = 1.0
        best_mu_drift = 0.0
        
        for test_nu in NU_GRID:
            try:
                temp_config = UnifiedStudentTConfig(
                    q=q_opt, c=c_opt, phi=phi_opt, nu_base=float(test_nu),
                    alpha_asym=alpha_opt, gamma_vov=gamma_opt,
                    ms_sensitivity=sens_opt, q_stress_ratio=10.0,
                    vov_damping=0.3, variance_inflation=1.0,
                )
                
                # Use training data only
                _, _, mu_pred_train, S_pred_train, _ = cls.filter_phi_unified(
                    returns_train, vol_train, temp_config
                )
                
                # Compute innovations on training data
                innovations_train = returns_train - mu_pred_train
                
                # Variance inflation for this nu
                test_beta = compute_optimal_variance_inflation(
                    returns_train, mu_pred_train, S_pred_train, float(test_nu)
                )
                test_beta = float(np.clip(test_beta, 0.5, 2.0))
                
                # Mean drift correction
                test_mu_drift = float(np.mean(innovations_train))
                
                # Compute PIT with this (nu, beta, mu_drift)
                from scipy.stats import t as student_t, kstest
                pit_values = []
                for t in range(len(returns_train)):
                    inn = innovations_train[t] - test_mu_drift
                    S_cal = S_pred_train[t] * test_beta
                    if test_nu > 2:
                        t_scale = np.sqrt(S_cal * (test_nu - 2) / test_nu)
                    else:
                        t_scale = np.sqrt(S_cal)
                    t_scale = max(t_scale, 1e-10)
                    pit_values.append(student_t.cdf(inn, df=test_nu, loc=0, scale=t_scale))
                
                _, ks_p = kstest(pit_values, 'uniform')
                
                if ks_p > best_ks_p:
                    best_ks_p = ks_p
                    best_nu = float(test_nu)
                    best_beta = test_beta
                    best_mu_drift = test_mu_drift
                    
            except Exception:
                continue
        
        # Use best found values
        nu_opt = best_nu
        beta_opt = best_beta
        mu_drift_opt = best_mu_drift
        
        # =====================================================================
        # BUILD FINAL CONFIG
        # =====================================================================
        # ELITE FIX: Use optimal nu from Stage 5 grid search, not input nu_base
        final_config = UnifiedStudentTConfig(
            q=q_opt,
            c=c_opt,
            phi=phi_opt,
            nu_base=nu_opt,  # ELITE FIX: Use optimized nu from grid search
            alpha_asym=alpha_opt,
            gamma_vov=gamma_opt,
            ms_sensitivity=sens_opt,
            q_stress_ratio=10.0,
            vov_damping=0.3,
            variance_inflation=beta_opt,
            mu_drift=mu_drift_opt,  # ELITE FIX: mean drift correction
            q_min=config.q_min,
            c_min=config.c_min,
            c_max=config.c_max,
        )
        
        diagnostics = {
            "stage": 5,
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
            "variance_inflation": float(beta_opt),
            "mu_drift": float(mu_drift_opt),  # ELITE FIX
            "c_bounds": (float(config.c_min), float(config.c_max)),
            "q_min": float(config.q_min),
        }
        
        return final_config, diagnostics
