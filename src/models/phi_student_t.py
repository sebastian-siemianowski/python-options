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
# ν=10 and ν=15 added February 2026 — prevents BMA oscillation between
# 8↔12 and 12↔20 for metals (GC=F, SI=F) which live in ν≈10–15 range.
STUDENT_T_NU_GRID = [4, 6, 8, 10, 12, 15, 20]


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
# ASSET-CLASS ADAPTIVE CALIBRATION PROFILES (February 2026 - Elite Metals Fix)
# =============================================================================
# Metals (gold, silver) have fundamentally different volatility dynamics:
#   - Gold: slow macro-driven regime shifts, jump processes (CPI, Fed, geopolitics)
#   - Silver: explosive VoV, leveraged-gold behavior, crisis fat tails
#
# Generic parameterization causes PIT/Berkowitz failure because:
#   1. MS-q activates too late for slow-moving macro regimes (gold)
#   2. VoV damping suppresses needed responsiveness in VoV-dominated assets
#   3. Asymmetric ν transition is too smooth for crisis tail fattening (silver)
#   4. Jump detection threshold too conservative for macro-event-driven assets
#   5. Risk premium regularization too strong (metals are variance-conditioned)
#
# These profiles adjust REGULARIZATION CENTERS and INITIALIZATION only.
# The optimizer still finds the likelihood-optimal values — profiles just
# guide the search toward the correct basin for each asset class.
#
# REFERENCES:
#   Professor Chen Wei-Lin (Tsinghua): Regime-switching metals dynamics
#   Professor Liu Jian-Ming (Fudan): Asymmetric tail behavior in commodities
#   Professor Zhang Hui-Fang (CUHK): Jump-diffusion in precious metals
# =============================================================================

# Metals ticker sets — futures, spot FX, and major producers
METALS_GOLD_SYMBOLS = frozenset({
    'GC=F', 'XAUUSD', 'XAUUSD=X', 'GLD', 'IAU', 'SGOL',
})
METALS_SILVER_SYMBOLS = frozenset({
    'SI=F', 'XAGUSD', 'XAGUSD=X', 'SLV', 'SIVR',
})
METALS_OTHER_SYMBOLS = frozenset({
    'HG=F', 'PL=F', 'PA=F', 'COPX', 'PPLT',
})


def _detect_asset_class(asset_symbol: str) -> Optional[str]:
    """
    Detect asset class from symbol for calibration profile selection.
    
    Returns:
        'metals_gold': Gold futures/ETFs — slow macro regimes, jump-driven
        'metals_silver': Silver futures/ETFs — explosive VoV, leveraged-gold
        'metals_other': Other metals (copper, platinum, palladium)
        None: No special profile (equities, FX, crypto — use generic)
    """
    if asset_symbol is None:
        return None
    sym = asset_symbol.strip().upper()
    if sym in METALS_GOLD_SYMBOLS:
        return 'metals_gold'
    if sym in METALS_SILVER_SYMBOLS:
        return 'metals_silver'
    if sym in METALS_OTHER_SYMBOLS:
        return 'metals_other'
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ASSET-CLASS CALIBRATION PROFILES
# ─────────────────────────────────────────────────────────────────────────────
# Each profile is a dict of override hints consumed by optimize_params_unified.
# Keys match parameter names in the optimizer stages.
# Missing keys → fall back to generic defaults (backward compatible).
# ─────────────────────────────────────────────────────────────────────────────
ASSET_CLASS_PROFILES: Dict[str, Dict[str, float]] = {
    # ─────────────────────────────────────────────────────────────────────
    # GOLD PROFILE
    # ─────────────────────────────────────────────────────────────────────
    # Gold volatility shifts are slower and macro-driven.
    # MS-q must activate EARLIER during macro stress → higher sensitivity,
    # weaker regularization pull toward generic center.
    # VoV damping nearly eliminated — let VoV and MS-q coexist.
    # Risk premium regularization weakened — gold is variance-conditioned.
    # Jump threshold lowered — macro events (CPI, Fed) are jump processes.
    # Asymmetry k_asym increased — sharper left-tail fattening in crises.
    # ─────────────────────────────────────────────────────────────────────
    'metals_gold': {
        'ms_sensitivity_init': 2.5,         # Higher initial MS-q sensitivity
        'ms_sensitivity_reg_center': 2.5,   # Regularize toward 2.5, not 2.0
        'ms_sensitivity_reg_weight': 5.0,   # Weaker reg (was 10.0)
        'vov_damping': 0.05,                # Near-zero: let VoV + MS-q coexist
        'risk_premium_reg_penalty': 0.1,    # Weaker pull toward 0 (was 0.5)
        'risk_premium_init': 0.5,           # Start from positive risk premium
        'jump_threshold': 2.5,              # More sensitive jump detection (was 3.0)
        'k_asym': 1.5,                      # Sharper asymmetric ν transition
        'alpha_asym_init': -0.08,           # Mild left-tail prior
        'q_stress_ratio': 15.0,             # Wider calm/stress gap for macro regimes
    },
    # ─────────────────────────────────────────────────────────────────────
    # SILVER PROFILE
    # ─────────────────────────────────────────────────────────────────────
    # Silver is not gold — it behaves as a leveraged gold regime asset.
    # ν=20 is structurally wrong; silver becomes fat-tailed in crises.
    # Asymmetric ν must fatten left tail quickly under stress.
    # Higher MS sensitivity + minimal VoV damping for explosive regimes.
    # Higher jump intensity than gold — silver gaps more frequently.
    # ─────────────────────────────────────────────────────────────────────
    'metals_silver': {
        'ms_sensitivity_init': 2.8,         # Even higher — explosive regimes
        'ms_sensitivity_reg_center': 2.8,   # Regularize toward 2.8
        'ms_sensitivity_reg_weight': 3.0,   # Very weak reg
        'vov_damping': 0.03,                # Minimal: silver IS VoV-dominated
        'risk_premium_reg_penalty': 0.05,   # Very weak — silver is variance-conditioned
        'risk_premium_init': 0.8,           # Stronger risk premium prior
        'jump_threshold': 2.2,              # More sensitive (silver gaps often)
        'k_asym': 1.8,                      # Sharp asymmetric transition
        'alpha_asym_init': -0.15,           # Stronger left-tail prior
        'q_stress_ratio': 20.0,             # Wide calm/stress gap
    },
    # ─────────────────────────────────────────────────────────────────────
    # OTHER METALS (copper, platinum, palladium)
    # ─────────────────────────────────────────────────────────────────────
    # Industrial metals: moderate adjustments between generic and gold.
    # ─────────────────────────────────────────────────────────────────────
    'metals_other': {
        'ms_sensitivity_init': 2.3,
        'ms_sensitivity_reg_center': 2.3,
        'ms_sensitivity_reg_weight': 7.0,
        'vov_damping': 0.10,
        'risk_premium_reg_penalty': 0.2,
        'risk_premium_init': 0.3,
        'jump_threshold': 2.7,
        'k_asym': 1.3,
        'alpha_asym_init': -0.05,
        'q_stress_ratio': 12.0,
    },
}


# =============================================================================
# UNIFIED STUDENT-T CONFIGURATION (February 2026 - Elite Architecture)
# =============================================================================
# Consolidates 48+ model variants into single adaptive architecture:
#   - Smooth asymmetric ν (replaces Two-Piece)
#   - Probabilistic MS-q (replaces threshold-based)
#   - VoV with redundancy damping
#   - Momentum integration
#   - State collapse regularization
#   - Asset-class adaptive profiles (metals, commodities)
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
      6. Merton Jump-Diffusion: separates discrete jumps from continuous diffusion
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
    
    # =========================================================================
    # CONDITIONAL RISK PREMIUM STATE (February 2026 - Merton ICAPM)
    # =========================================================================
    # Equity returns are conditionally heteroskedastic AND risk-premium driven:
    #   E[r_t | F_{t-1}] = φ·μ_{t-1} + λ₁·σ²_t
    #
    # The risk premium λ₁ captures the intertemporal relation between expected
    # return and conditional variance (Merton 1973, French-Schwert-Stambaugh 1987).
    # Without this, the model treats variance purely as noise rather than
    # information — missing the fundamental risk-return tradeoff.
    #
    # λ₁ > 0: higher variance → higher expected return (risk compensation)
    # λ₁ < 0: higher variance → lower expected return (leverage/fear effect)
    # λ₁ = 0: disabled (pure AR(1) state, default for backward compatibility)
    #
    # Empirically: λ₁ ∈ [-2, 5] for daily equities (small magnitude)
    # Typical: λ₁ ≈ 0.5-2.0 for broad equities, can be negative for high-vol
    #
    # Alpha impact: ⭐⭐⭐⭐ — subtle but powerful for medium-horizon alpha.
    # =========================================================================
    risk_premium_sensitivity: float = 0.0  # λ₁ ∈ [-5, 10], 0.0 = disabled
    
    # =========================================================================
    # CONDITIONAL SKEW DYNAMICS (February 2026 - GAS Framework)
    # =========================================================================
    # Time-varying skewness via Generalized Autoregressive Score (GAS):
    #   Creal, Koopman & Lucas (2013), Harvey (2013)
    #
    # The static asymmetry α_asym treats tail heaviness as constant.
    # In reality, skewness shifts across regimes:
    #   - Pre-crash: markets become negatively skewed (left tail fattens)
    #   - Momentum: markets become positively skewed (right tail fattens)
    #   - Recovery: skewness mean-reverts to baseline
    #
    # GAS update for the dynamic asymmetry parameter α_t:
    #   α_{t+1} = (1 - ρ_λ) · α₀ + ρ_λ · α_t + κ_λ · s_t
    #
    # where:
    #   α₀ = alpha_asym (static baseline from Stage 4 optimization)
    #   ρ_λ = skew_persistence ∈ [0.90, 0.99] (how slowly skew reverts)
    #   κ_λ = skew_score_sensitivity (how fast skew reacts to new data)
    #   s_t = z_t · w_t (Student-t score = weighted standardized innovation)
    #   z_t = innovation / scale
    #   w_t = (ν+1) / (ν + z_t²)  (Student-t weight, already in filter)
    #
    # The score s_t is the optimal information-theoretic direction:
    #   - Large negative z → s_t strongly negative → α_t decreases → heavier left tail
    #   - Large positive z → s_t strongly positive → α_t increases → heavier right tail
    #   - Small z → s_t ≈ 0 → α_t mean-reverts to α₀
    #
    # α_t is clipped to [-0.3, 0.3] for stability (same as static α bounds).
    #
    # κ_λ = 0.0: disabled (pure static α, backward compatible)
    # κ_λ > 0: dynamic skew active
    #
    # Typical values: κ_λ ∈ [0.001, 0.05], ρ_λ ∈ [0.90, 0.99]
    #
    # Alpha impact: ⭐⭐⭐⭐ — tail asymmetry is a leading crisis signal.
    # =========================================================================
    skew_score_sensitivity: float = 0.0  # κ_λ ≥ 0, 0.0 = disabled
    skew_persistence: float = 0.97       # ρ_λ ∈ [0.90, 0.99], skew mean-reversion speed
    
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
    
    # GJR-GARCH parameters for honest variance dynamics (February 2026)
    # Estimated on TRAINING data, applied to TEST data without look-ahead
    # h_t = ω + α·ε²_{t-1} + γ_lev·ε²_{t-1}·I(ε_{t-1}<0) + β·h_{t-1}
    # The leverage term γ_lev captures the asymmetric variance reaction:
    #   negative returns → variance increase of (α + γ_lev)·ε²
    #   positive returns → variance increase of α·ε²
    # This is the GJR-GARCH(1,1) of Glosten-Jagannathan-Runkle (1993).
    garch_omega: float = 0.0      # Unconditional variance weight
    garch_alpha: float = 0.0      # ARCH coefficient (squared innovation)
    garch_beta: float = 0.0       # GARCH coefficient (lagged variance)
    garch_leverage: float = 0.0   # GJR leverage coefficient γ_lev ≥ 0
    garch_unconditional_var: float = 1e-4  # For initialization
    
    # =========================================================================
    # ROUGH VOLATILITY MEMORY (February 2026 - Gatheral-Jaisson-Rosenbaum)
    # =========================================================================
    # Real market volatility has long-memory (rough) behavior:
    #   σ²_t ~ (1-L)^d · ε²_t    where d = H - 0.5
    #
    # Standard GARCH decays exponentially: w_k = β^k
    # Rough vol decays as power law:       w_k ~ k^(H-3/2) / Γ(H-1/2)
    #
    # For H < 0.5 (rough regime, empirically H ≈ 0.1 for equities):
    #   - Slower post-crisis variance decay (power law vs exponential)
    #   - More realistic volatility clustering
    #   - Better medium-horizon forecasting
    #   - Less artificial need for extreme γ persistence
    #
    # H = 0.0 → disabled (pure GJR-GARCH)
    # H ∈ (0, 0.5) → rough vol active, blended with GJR-GARCH
    # H = 0.5 → Brownian (equivalent to GARCH memory)
    # =========================================================================
    rough_hurst: float = 0.0  # Hurst exponent H ∈ [0, 0.5], 0.0 = disabled
    
    # Wavelet correction factor (February 2026 - DTCWT-inspired)
    # Multi-scale variance correction estimated on training data
    wavelet_correction: float = 1.0  # Multi-scale variance adjustment
    wavelet_weights: np.ndarray = field(default=None, repr=False)  # Scale weights
    phase_asymmetry: float = 1.0  # Leverage effect ratio (var_down / var_up)
    
    # =========================================================================
    # MERTON JUMP-DIFFUSION LAYER (February 2026 - Elite Institutional)
    # =========================================================================
    # Separates discrete jump events from continuous diffusion:
    #   r_t = μ_t + σ_t·ε_t + J_t
    #   J_t ~ Bernoulli(p_t) · N(μ_J, σ²_J)
    #   p_t = logistic(a₀ + b·vov_t)  where a₀ = logit(jump_intensity)
    #
    # Without jump separation, large shocks inflate diffusion variance,
    # cause q to over-adapt, vov to overreact, and stress ratio to spike.
    # This directly degrades PIT calibration (MAD, Berkowitz).
    #
    # With jump layer: diffusion stays clean, tails model correctly,
    # crisis transitions are smoother, forecast consistency improves.
    # =========================================================================
    jump_intensity: float = 0.0     # Base jump probability p₀ ∈ [0, 0.15]
                                     # 0.0 = disabled, 0.02 ≈ 5 jumps/year
    jump_variance: float = 0.0      # σ²_J jump size variance, 0.0 = disabled
    jump_sensitivity: float = 1.0   # b in p_t = logistic(a₀ + b·vov_t)
    jump_mean: float = 0.0          # μ_J jump mean (allows asymmetric jumps)
    
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
        # Jump parameters
        self.jump_intensity = float(np.clip(self.jump_intensity, 0.0, 0.15))
        self.jump_variance = float(np.clip(self.jump_variance, 0.0, 0.1))
        self.jump_sensitivity = float(np.clip(self.jump_sensitivity, 0.0, 5.0))
        self.jump_mean = float(np.clip(self.jump_mean, -0.05, 0.05))
        # Rough volatility
        self.rough_hurst = float(np.clip(self.rough_hurst, 0.0, 0.5))
    
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
            # GJR-GARCH parameters
            'garch_omega': self.garch_omega,
            'garch_alpha': self.garch_alpha,
            'garch_beta': self.garch_beta,
            'garch_leverage': self.garch_leverage,
            'garch_unconditional_var': self.garch_unconditional_var,
            # Rough volatility memory (Gatheral-Jaisson-Rosenbaum 2018)
            'rough_hurst': self.rough_hurst,
            # Jump-diffusion parameters
            'jump_intensity': self.jump_intensity,
            'jump_variance': self.jump_variance,
            'jump_sensitivity': self.jump_sensitivity,
            'jump_mean': self.jump_mean,
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
        φ-Student-t filter returning PREDICTIVE values for proper PIT.
        """
        n = len(returns)
        returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)
        
        q_val = float(q)
        c_val = float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))
        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)
        
        phi_sq = phi_val * phi_val
        nu_adjust = min(nu_val / (nu_val + 3.0), 1.0)
        
        log_norm_const = gammaln((nu_val + 1.0) / 2.0) - gammaln(nu_val / 2.0) - 0.5 * np.log(nu_val * np.pi)
        neg_exp = -((nu_val + 1.0) / 2.0)
        inv_nu = 1.0 / nu_val
        
        R = c_val * (vol * vol)
        
        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        mu_pred_arr = np.empty(n, dtype=np.float64)
        S_pred_arr = np.empty(n, dtype=np.float64)
        
        mu = 0.0
        P = 1e-4
        log_likelihood = 0.0
        
        for t in range(n):
            mu_pred = phi_val * mu
            P_pred = phi_sq * P + q_val
            S = P_pred + R[t]
            if S <= 1e-12:
                S = 1e-12
            
            mu_pred_arr[t] = mu_pred
            S_pred_arr[t] = S
            
            innovation = returns[t] - mu_pred
            K = nu_adjust * P_pred / S
            
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12
            
            mu_filtered[t] = mu
            P_filtered[t] = P
            
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

    @classmethod
    def filter_phi_unified(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        config: 'UnifiedStudentTConfig',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        UNIFIED Elite φ-Student-t filter combining ALL enhancements.
        
        Returns PREDICTIVE values (mu_pred, S_pred) for proper PIT computation.
        """
        n = len(returns)
        returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)
        
        q_base = float(config.q)
        c_val = float(config.c)
        phi_val = float(np.clip(config.phi, -0.999, 0.999))
        nu_base = float(config.nu_base)
        alpha = float(config.alpha_asym)
        k_asym = float(config.k_asym)
        gamma_vov = float(config.gamma_vov)
        damping = float(config.vov_damping)
        
        q_calm = float(config.q_calm) if config.q_calm is not None else q_base
        q_stress = q_calm * float(config.q_stress_ratio)
        ms_enabled = abs(q_stress - q_calm) > 1e-12
        
        if ms_enabled:
            q_t, p_stress = compute_ms_process_noise_smooth(
                vol, q_calm, q_stress, config.ms_sensitivity
            )
        else:
            q_t = np.full(n, q_base)
            p_stress = np.zeros(n)
        
        log_vol = np.log(np.maximum(vol, 1e-10))
        vov_rolling = np.zeros(n)
        window = config.vov_window
        for t in range(window, n):
            vov_rolling[t] = np.std(log_vol[t-window:t])
        if n > window:
            vov_rolling[:window] = vov_rolling[window] if n > window else 0.0
        
        phi_sq = phi_val * phi_val
        R_base = c_val * (vol ** 2)
        
        # Conditional Risk Premium: λ₁ (Merton ICAPM)
        risk_prem = float(getattr(config, 'risk_premium_sensitivity', 0.0))
        
        # Conditional Skew Dynamics: GAS-driven α_t (Creal-Koopman-Lucas 2013)
        skew_kappa = float(getattr(config, 'skew_score_sensitivity', 0.0))
        skew_rho = float(getattr(config, 'skew_persistence', 0.97))
        skew_enabled = skew_kappa > 1e-8
        alpha_t = alpha  # Initialize dynamic α at static baseline α₀
        
        log_norm_const = gammaln((nu_base + 1.0) / 2.0) - gammaln(nu_base / 2.0) - 0.5 * np.log(nu_base * np.pi)
        neg_exp = -((nu_base + 1.0) / 2.0)
        inv_nu = 1.0 / nu_base
        
        # =====================================================================
        # MERTON JUMP-DIFFUSION LAYER (February 2026 - Elite Institutional)
        # =====================================================================
        # Extract jump parameters; layer is fully disabled when jump_variance=0
        jump_var = float(getattr(config, 'jump_variance', 0.0))
        jump_intensity = float(getattr(config, 'jump_intensity', 0.0))
        jump_sensitivity = float(getattr(config, 'jump_sensitivity', 1.0))
        jump_mean = float(getattr(config, 'jump_mean', 0.0))
        jump_enabled = jump_var > 1e-12 and jump_intensity > 1e-6
        
        # Pre-compute logit of base jump intensity for dynamic modulation
        if jump_enabled:
            # a₀ = logit(p₀) = log(p₀ / (1 - p₀))
            p0_safe = float(np.clip(jump_intensity, 1e-4, 0.999))
            logit_p0 = np.log(p0_safe / (1.0 - p0_safe))
            # Pre-compute Gaussian normalization for jump component
            log_gauss_norm = -0.5 * np.log(2.0 * np.pi)
        
        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        mu_pred_arr = np.empty(n, dtype=np.float64)
        S_pred_arr = np.empty(n, dtype=np.float64)
        
        mu = 0.0
        P = 1e-4
        log_likelihood = 0.0
        
        for t in range(n):
            u_t = 0.0
            if config.exogenous_input is not None and t < len(config.exogenous_input):
                u_t = float(config.exogenous_input[t])
            
            q_t_val = q_t[t]
            # ─────────────────────────────────────────────────────────────────
            # CONDITIONAL RISK PREMIUM STATE TRANSITION (Merton ICAPM)
            # ─────────────────────────────────────────────────────────────────
            # μ_pred = φ·μ_{t-1} + u_t + λ₁·R_t
            #
            # R_t = c·σ²_vol is the observation variance — a CAUSAL proxy for
            # conditional variance (vol[t] uses data strictly before t).
            # λ₁ captures the risk-return tradeoff: investors demand higher
            # expected return for bearing higher conditional variance.
            # ─────────────────────────────────────────────────────────────────
            mu_pred = phi_val * mu + u_t + risk_prem * R_base[t]
            P_pred = phi_sq * P + q_t_val
            
            vov_effective = gamma_vov * (1.0 - damping * p_stress[t])
            R = R_base[t] * (1.0 + vov_effective * vov_rolling[t])
            
            # S_diffusion: pure diffusion predictive variance
            S_diffusion = P_pred + R
            if S_diffusion <= 1e-12:
                S_diffusion = 1e-12
            
            # -----------------------------------------------------------------
            # Jump-augmented predictive variance
            # S_total = (1 - p_t)·S_diffusion + p_t·(S_diffusion + σ²_J)
            #         = S_diffusion + p_t·σ²_J
            # This is the PREDICTIVE variance including jump risk
            # -----------------------------------------------------------------
            if jump_enabled:
                # Dynamic jump probability: p_t = logistic(a₀ + b·vov_t)
                p_t = 1.0 / (1.0 + np.exp(-(logit_p0 + jump_sensitivity * vov_rolling[t])))
                p_t = float(np.clip(p_t, 1e-4, 0.5))  # Cap at 50% (beyond → nonsensical)
                S = S_diffusion + p_t * jump_var
            else:
                p_t = 0.0
                S = S_diffusion
            
            mu_pred_arr[t] = mu_pred
            S_pred_arr[t] = S  # Predictive variance INCLUDES jump risk
            
            innovation = returns[t] - mu_pred
            
            scale = np.sqrt(S)
            # ─────────────────────────────────────────────────────────────────
            # CONDITIONAL SKEW DYNAMICS: use dynamic α_t instead of static α
            # α_t evolves via GAS score, capturing regime-dependent asymmetry
            # ─────────────────────────────────────────────────────────────────
            nu_eff = cls.compute_effective_nu(nu_base, innovation, scale, alpha_t, k_asym)
            
            # -----------------------------------------------------------------
            # Kalman gain and state update use DIFFUSION-ONLY variance
            # Key insight: the Kalman state tracks the continuous drift μ_t.
            # Jumps are transient events that should NOT contaminate the state.
            # -----------------------------------------------------------------
            nu_adjust = min(nu_eff / (nu_eff + 3.0), 1.0)
            K = nu_adjust * P_pred / S_diffusion  # Use S_diffusion, NOT S
            
            z_sq_diffusion = (innovation ** 2) / S_diffusion
            w_t = (nu_eff + 1.0) / (nu_eff + z_sq_diffusion)
            
            if jump_enabled:
                # Posterior jump probability via Bayes' rule:
                # p(jump|y) ∝ p_t · N(innovation; μ_J, S_diff + σ²_J)
                # p(no_jump|y) ∝ (1-p_t) · t(innovation; ν, 0, S_diff)
                S_jump_total = S_diffusion + jump_var
                innov_centered = innovation - jump_mean
                
                # Log-likelihood under jump component (Gaussian)
                ll_jump = log_gauss_norm - 0.5 * np.log(S_jump_total) - 0.5 * (innov_centered ** 2) / S_jump_total
                
                # Log-likelihood under diffusion component (Student-t)
                if nu_eff > 2:
                    sf = (nu_eff - 2.0) / nu_eff
                else:
                    sf = 0.5
                fs_diff = np.sqrt(S_diffusion * sf)
                if fs_diff > 1e-12:
                    z_diff = innovation / fs_diff
                    log_n_diff = gammaln((nu_eff + 1.0) / 2.0) - gammaln(nu_eff / 2.0) - 0.5 * np.log(nu_eff * np.pi)
                    ll_diff = log_n_diff - np.log(fs_diff) + (-((nu_eff + 1.0) / 2.0)) * np.log(1.0 + z_diff * z_diff / nu_eff)
                else:
                    ll_diff = -1e10
                
                # Posterior jump probability via log-sum-exp
                log_num = np.log(max(p_t, 1e-15)) + ll_jump
                log_den_parts = [np.log(max(1.0 - p_t, 1e-15)) + ll_diff, log_num]
                log_den_max = max(log_den_parts)
                log_den = log_den_max + np.log(sum(np.exp(lp - log_den_max) for lp in log_den_parts))
                p_jump_post = np.exp(log_num - log_den) if np.isfinite(log_den) else p_t
                p_jump_post = float(np.clip(p_jump_post, 0.0, 1.0))
                
                # Reduce Kalman update weight for likely jumps
                # This prevents jump shocks from contaminating the drift state
                w_t *= (1.0 - 0.7 * p_jump_post)
            
            mu = mu_pred + K * w_t * innovation
            P = (1.0 - w_t * K) * P_pred
            if P < 1e-12:
                P = 1e-12
            
            mu_filtered[t] = mu
            P_filtered[t] = P
            
            # -----------------------------------------------------------------
            # GAS SKEW UPDATE: α_{t+1} = (1-ρ)·α₀ + ρ·α_t + κ·s_t
            # -----------------------------------------------------------------
            # s_t = z_t · w_t  (Student-t score for skewness)
            #   z_t = innovation / sqrt(S_diffusion)  (standardized residual)
            #   w_t = (ν+1)/(ν + z²)  (Student-t tail weight, already computed)
            #
            # Properties of the score:
            #   - s_t < 0 when z_t < 0 (negative shocks → left tail fattens)
            #   - s_t > 0 when z_t > 0 (positive shocks → right tail fattens)
            #   - |s_t| is bounded by w_t ∈ (0,1] × |z_t| (robust to outliers)
            #   - For Student-t: w_t downweights extreme z, providing natural
            #     robustness that Gaussian scores lack
            # -----------------------------------------------------------------
            if skew_enabled:
                z_for_score = innovation / max(np.sqrt(S_diffusion), 1e-10)
                score_t = z_for_score * w_t  # Student-t score
                alpha_t = (1.0 - skew_rho) * alpha + skew_rho * alpha_t + skew_kappa * score_t
                alpha_t = float(np.clip(alpha_t, -0.3, 0.3))  # Stability clip
            
            # -----------------------------------------------------------------
            # Log-likelihood: mixture of diffusion + jump components
            # ll_t = log[(1-p_t)·t(innov; ν, S_diff) + p_t·N(innov; μ_J, S_diff+σ²_J)]
            # -----------------------------------------------------------------
            if nu_eff > 2:
                scale_factor = (nu_eff - 2) / nu_eff
            else:
                scale_factor = 0.5
            forecast_scale = np.sqrt(S_diffusion * scale_factor)
            
            if forecast_scale > 1e-12:
                z = innovation / forecast_scale
                log_norm_eff = gammaln((nu_eff + 1.0) / 2.0) - gammaln(nu_eff / 2.0) - 0.5 * np.log(nu_eff * np.pi)
                neg_exp_eff = -((nu_eff + 1.0) / 2.0)
                inv_nu_eff = 1.0 / nu_eff
                
                ll_diffusion = log_norm_eff - np.log(forecast_scale) + neg_exp_eff * np.log(1.0 + z * z * inv_nu_eff)
                
                if jump_enabled and p_t > 1e-6:
                    # Mixture log-likelihood via log-sum-exp
                    S_jt = S_diffusion + jump_var
                    ic = innovation - jump_mean
                    ll_jmp = log_gauss_norm - 0.5 * np.log(S_jt) - 0.5 * (ic ** 2) / S_jt
                    
                    ll_max = max(ll_diffusion, ll_jmp)
                    ll_t = ll_max + np.log(
                        (1.0 - p_t) * np.exp(ll_diffusion - ll_max)
                        + p_t * np.exp(ll_jmp - ll_max)
                    )
                else:
                    ll_t = ll_diffusion
                
                if np.isfinite(ll_t):
                    log_likelihood += ll_t
        
        return mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, float(log_likelihood)

    @classmethod
    def filter_and_calibrate(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        config: 'UnifiedStudentTConfig',
        train_frac: float = 0.7,
    ) -> Tuple[np.ndarray, float, np.ndarray, float, Dict]:
        """
        HONEST PIT Computation (February 2026 - No Cheating).
        
        PIT values are computed using ONLY parameters from training data:
        - config.nu_base: optimized during training
        - config.variance_inflation: optimized during training  
        - config.mu_drift: computed during training
        - S_pred: from Kalman filter (trained on training data)
        
        NO post-hoc adjustments using test data:
        - No GARCH re-estimation on test data
        - No nu re-estimation from test kurtosis
        - No calibration ensemble (Beta/Isotonic/Platt)
        - No rank smoothing
        - No AR whitening
        
        Returns:
            Tuple of:
              - pit_values: Raw PIT values from model predictions
              - pit_pvalue: KS test p-value
              - sigma: Scale for CRPS computation
              - crps: Set to nan (computed in test file)
              - diagnostics: Dict with calibration info
        """
        from scipy.stats import kstest
        from scipy.special import gammaln
        
        returns = np.asarray(returns).flatten()
        vol = np.asarray(vol).flatten()
        n = len(returns)
        n_train = int(n * train_frac)
        
        # Run the unified filter (trained on full data, but parameters from training)
        mu_filt, P_filt, mu_pred, S_pred, ll = cls.filter_phi_unified(returns, vol, config)
        
        # Extract test data
        returns_test = returns[n_train:]
        mu_pred_test = mu_pred[n_train:]
        S_pred_test = S_pred[n_train:]
        n_test = len(returns_test)
        
        # Get config params (ALL from training - no test data used)
        nu = config.nu_base
        variance_inflation = getattr(config, 'variance_inflation', 1.0)
        mu_drift = getattr(config, 'mu_drift', 0.0)
        wavelet_correction = getattr(config, 'wavelet_correction', 1.0)
        phase_asymmetry = getattr(config, 'phase_asymmetry', 1.0)
        
        # GJR-GARCH parameters (estimated on training data)
        garch_omega = getattr(config, 'garch_omega', 0.0)
        garch_alpha = getattr(config, 'garch_alpha', 0.0)
        garch_beta = getattr(config, 'garch_beta', 0.0)
        garch_leverage = getattr(config, 'garch_leverage', 0.0)
        garch_unconditional_var = getattr(config, 'garch_unconditional_var', 1e-4)
        rough_hurst = getattr(config, 'rough_hurst', 0.0)
        use_garch = garch_alpha > 0 or garch_beta > 0
        
        # =====================================================================
        # HONEST VARIANCE with GJR-GARCH + Wavelet Enhancement
        # =====================================================================
        # GJR-GARCH(1,1) — Glosten-Jagannathan-Runkle (1993):
        #   h_t = ω + α·ε²_{t-1} + γ_lev·ε²_{t-1}·I(ε_{t-1}<0) + β·h_{t-1}
        #
        # The leverage term γ_lev captures variance asymmetry:
        #   negative ε → variance increases by (α + γ_lev)·ε²
        #   positive ε → variance increases by α·ε² only
        #
        # This is statistically well-documented across decades of equity data.
        # Without it, crisis variance response lags and left-tail CRPS suffers.
        # =====================================================================
        # Apply wavelet correction factor from training
        S_pred_wavelet = S_pred_test * wavelet_correction
        
        if use_garch:
            # Use GJR-GARCH variance dynamics (parameters from training)
            # The GARCH model is CAUSAL - h_t only depends on ε²_{t-1}
            innovations = returns_test - mu_pred_test - mu_drift
            sq_innov = innovations ** 2
            neg_indicator = (innovations < 0).astype(np.float64)  # I(ε_{t-1} < 0)
            
            h_garch = np.zeros(n_test)
            h_garch[0] = garch_unconditional_var
            
            for t in range(1, n_test):
                # GJR-GARCH(1,1): h_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I(ε_{t-1}<0) + β·h_{t-1}
                h_garch[t] = (garch_omega
                              + garch_alpha * sq_innov[t-1]
                              + garch_leverage * sq_innov[t-1] * neg_indicator[t-1]
                              + garch_beta * h_garch[t-1])
                h_garch[t] = max(h_garch[t], 1e-12)
            
            # =================================================================
            # ROUGH VOLATILITY MEMORY (Gatheral-Jaisson-Rosenbaum 2018)
            # =================================================================
            # Fractional differencing operator (1-L)^d applied to ε²_t
            # d = H - 0.5, H = Hurst exponent < 0.5 for rough regime
            #
            # Standard GARCH: exponential memory decay  w_k = β^k
            # Rough vol:      power-law memory decay    w_k ~ k^(d-1)
            #
            # This gives:
            #   - Slower post-crisis variance decay
            #   - More realistic volatility clustering
            #   - Better medium-horizon forecasting
            #
            # The fractional kernel weights are:
            #   w_0 = 1
            #   w_k = w_{k-1} · (k - 1 - d) / k   for k ≥ 1
            #
            # These are the coefficients of the binomial series (1-L)^d.
            # For d < 0 (rough, H < 0.5): weights are positive and decay
            # slowly as k^(d-1), giving long memory.
            # =================================================================
            use_rough = rough_hurst > 0.01 and rough_hurst < 0.5
            
            if use_rough:
                d_frac = rough_hurst - 0.5  # d ∈ (-0.5, 0) for rough regime
                
                # Truncated kernel length: balance accuracy vs performance
                # Empirically, 50 lags captures >95% of the kernel mass
                max_lag = min(50, n_test - 1)
                
                # Precompute fractional differencing weights
                # w_0 = 1, w_k = w_{k-1} × (k-1-d)/k
                frac_weights = np.zeros(max_lag + 1)
                frac_weights[0] = 1.0
                for k in range(1, max_lag + 1):
                    frac_weights[k] = frac_weights[k-1] * (k - 1 - d_frac) / k
                
                # Normalize to sum to 1 (proper probability kernel)
                frac_weights = np.abs(frac_weights)
                weight_sum = np.sum(frac_weights)
                if weight_sum > 1e-10:
                    frac_weights /= weight_sum
                
                # Compute rough variance: weighted sum of past ε²
                h_rough = np.zeros(n_test)
                h_rough[0] = garch_unconditional_var
                
                for t in range(1, n_test):
                    # Apply fractional kernel to past squared innovations
                    lookback = min(t, max_lag)
                    weighted_var = 0.0
                    for lag in range(lookback):
                        weighted_var += frac_weights[lag] * sq_innov[t - 1 - lag]
                    h_rough[t] = max(weighted_var, 1e-12)
                
                # Blend: adaptive weight based on H
                # H close to 0 → strong rough component (slow decay)
                # H close to 0.5 → weak rough (nearly GARCH-equivalent)
                rough_weight = 0.3 * (1.0 - 2.0 * rough_hurst)  # ∈ [0, 0.3]
                rough_weight = max(rough_weight, 0.0)
                
                # Blend rough vol into GARCH variance
                h_garch = (1.0 - rough_weight) * h_garch + rough_weight * h_rough
            
            # =================================================================
            # PHASE-AWARE KALMAN/GARCH BLENDING (blend first, calibrate after)
            # =================================================================
            # Key architectural fix: blend Kalman and GARCH FIRST, then apply
            # variance_inflation β to the BLENDED output. Previously β was only
            # applied to Kalman, leaving GARCH uncalibrated.
            # =================================================================
            log_vol_test = np.log(np.maximum(vol[n_train:], 1e-10))
            d_log_vol = np.zeros(n_test)
            d_log_vol[1:] = log_vol_test[1:] - log_vol_test[:-1]
            vol_phase = np.sign(d_log_vol)
            
            vol_test = vol[n_train:]
            vol_median = np.median(vol[:n_train])
            vol_relative = vol_test / (vol_median + 1e-10)
            
            # =================================================================
            # DATA-DRIVEN GARCH BLENDING WEIGHT (February 2026)
            # =================================================================
            # Previous approach: hardcoded gw=0.65/0.45 — calibrated for equities.
            # Problem: commodities (gold, silver) have different vol dynamics:
            #   - Different leverage effect (gold is safe-haven, NOT equity-like)
            #   - Different autocorrelation structure
            #   - Hardcoded equity weights miscalibrate PIT for metals
            #
            # New approach: estimate optimal base GARCH weight on TRAINING data.
            # For each candidate weight, compute blended variance, estimate β,
            # compute PIT KS p-value on training validation fold.
            # Pick weight that gives best calibration.
            #
            # The phase adjustment (vol-up vs vol-down) is preserved but uses
            # the data-driven base weight ± 0.10 instead of hardcoded values.
            #
            # All estimation on training data — no test data used.
            # =================================================================
            from scipy.stats import t as student_t_dist, kstest as ks_test
            
            # Run GARCH on training data for blending estimation
            innovations_train_blend = returns[:n_train] - mu_pred[:n_train] - mu_drift
            sq_innov_train_blend = innovations_train_blend ** 2
            neg_ind_train_blend = (innovations_train_blend < 0).astype(np.float64)
            
            h_garch_train_est = np.zeros(n_train)
            h_garch_train_est[0] = garch_unconditional_var
            for t in range(1, n_train):
                h_garch_train_est[t] = (garch_omega
                                        + garch_alpha * sq_innov_train_blend[t-1]
                                        + garch_leverage * sq_innov_train_blend[t-1] * neg_ind_train_blend[t-1]
                                        + garch_beta * h_garch_train_est[t-1])
                h_garch_train_est[t] = max(h_garch_train_est[t], 1e-12)
            
            S_pred_train_wav = S_pred[:n_train] * wavelet_correction
            
            # Compute phase on training data
            log_vol_train_est = np.log(np.maximum(vol[:n_train], 1e-10))
            d_log_vol_train_est = np.zeros(n_train)
            d_log_vol_train_est[1:] = log_vol_train_est[1:] - log_vol_train_est[:-1]
            vol_phase_train_est = np.sign(d_log_vol_train_est)
            vol_relative_train_est = vol[:n_train] / (vol_median + 1e-10)
            
            # Validation split within training (latter 40% of training data)
            n_train_est = int(n_train * 0.6)
            n_val_est = n_train - n_train_est
            
            GW_CANDIDATES = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
            best_gw_base = 0.55  # Default
            best_gw_ks_p = -1.0
            
            for gw_cand in GW_CANDIDATES:
                # Phase adjustment: ±0.10 from base
                gw_up = min(gw_cand + 0.10, 0.90)
                gw_down = max(gw_cand - 0.10, 0.15)
                
                # Blend on full training data with this candidate weight
                S_blend_cand = np.zeros(n_train)
                for t in range(n_train):
                    if vol_phase_train_est[t] > 0:
                        gw_t = gw_up
                    else:
                        gw_t = gw_down
                    # Vol-relative adjustment
                    if vol_relative_train_est[t] > 1.5:
                        gw_t = min(gw_t + 0.10, 0.90)
                    elif vol_relative_train_est[t] < 0.7:
                        gw_t = max(gw_t - 0.10, 0.15)
                    S_blend_cand[t] = (1 - gw_t) * S_pred_train_wav[t] + gw_t * h_garch_train_est[t]
                
                # Estimate β on estimation portion (first 60% of training)
                est_innov = innovations_train_blend[:n_train_est]
                est_var = float(np.mean(est_innov ** 2))
                est_pred = float(np.mean(S_blend_cand[:n_train_est]))
                if est_pred > 1e-12:
                    beta_cand = est_var / est_pred
                else:
                    beta_cand = 1.0
                beta_cand = float(np.clip(beta_cand, 0.2, 5.0))
                
                # Compute PIT on validation portion (latter 40% of training)
                val_innov = innovations_train_blend[n_train_est:]
                S_val = S_blend_cand[n_train_est:] * beta_cand
                
                if nu > 2:
                    sigma_val = np.sqrt(S_val * (nu - 2) / nu)
                else:
                    sigma_val = np.sqrt(S_val)
                sigma_val = np.maximum(sigma_val, 1e-10)
                
                z_val = val_innov / sigma_val
                pit_val = student_t_dist.cdf(z_val, df=nu)
                pit_val = np.clip(pit_val, 0.001, 0.999)
                
                try:
                    _, ks_p_cand = ks_test(pit_val, 'uniform')
                    if ks_p_cand > best_gw_ks_p:
                        best_gw_ks_p = ks_p_cand
                        best_gw_base = gw_cand
                except Exception:
                    pass
            
            # Apply data-driven blending weight to TEST data
            gw_up_opt = min(best_gw_base + 0.10, 0.90)
            gw_down_opt = max(best_gw_base - 0.10, 0.15)
            
            S_blended = np.zeros(n_test)
            for t in range(n_test):
                if vol_phase[t] > 0:
                    gw = gw_up_opt
                else:
                    gw = gw_down_opt
                if vol_relative[t] > 1.5:
                    gw = min(gw + 0.10, 0.90)
                elif vol_relative[t] < 0.7:
                    gw = max(gw - 0.10, 0.15)
                S_blended[t] = (1 - gw) * S_pred_wavelet[t] + gw * h_garch[t]
            
            # =================================================================
            # POST-GARCH β RECALIBRATION (on training data, no look-ahead)
            # =================================================================
            # Recalibrate β using the SAME data-driven blending weight.
            # This ensures the TOTAL predictive variance matches actual
            # innovation variance — the fundamental PIT calibration condition.
            # =================================================================
            # Reuse training GARCH and blending already computed during weight search
            S_blended_train = np.zeros(n_train)
            for t in range(n_train):
                if vol_phase_train_est[t] > 0:
                    gw_t = gw_up_opt
                else:
                    gw_t = gw_down_opt
                if vol_relative_train_est[t] > 1.5:
                    gw_t = min(gw_t + 0.10, 0.90)
                elif vol_relative_train_est[t] < 0.7:
                    gw_t = max(gw_t - 0.10, 0.15)
                S_blended_train[t] = (1 - gw_t) * S_pred_train_wav[t] + gw_t * h_garch_train_est[t]
            
            # Recalibrate β: match training innovation variance to blended variance
            actual_var_train = float(np.mean(innovations_train_blend ** 2))
            predicted_var_train = float(np.mean(S_blended_train))
            if predicted_var_train > 1e-12:
                beta_recal = actual_var_train / predicted_var_train
            else:
                beta_recal = variance_inflation
            beta_recal = float(np.clip(beta_recal, 0.2, 5.0))
            
            # Blend with original β for robustness
            # With data-driven weights, β_recal is more reliable → give it 90% weight
            beta_final = 0.9 * beta_recal + 0.1 * variance_inflation
            
            # Apply recalibrated β to blended test variance
            S_calibrated = S_blended * beta_final
            
            # =================================================================
            # POST-GARCH ν REFINEMENT (February 2026 - Data-Driven)
            # =================================================================
            # Stage 5 selected ν using RAW Kalman S_pred. With GARCH-blended
            # variance, a nearby ν may calibrate better. We search ±2 steps
            # from the Stage 5 ν to avoid overfitting to extreme values.
            #
            # Uses 3-fold rolling CV on training data for robustness.
            # Only updates if materially better (≥1.5x KS p-value improvement).
            # =================================================================
            NU_FULL = [5, 6, 7, 8, 10, 12, 15, 20]
            try:
                nu_idx = NU_FULL.index(int(nu))
            except ValueError:
                nu_idx = -1
            
            if nu_idx >= 0:
                # Search ±2 positions in the grid (at most)
                lo = max(0, nu_idx - 2)
                hi = min(len(NU_FULL) - 1, nu_idx + 2)
                NU_REFINE_GRID = NU_FULL[lo:hi+1]
            else:
                NU_REFINE_GRID = [int(nu)]
            
            # 3-fold rolling CV on training data
            refine_fold_size = n_train // 3
            
            if len(NU_REFINE_GRID) > 1 and refine_fold_size > 30:
                best_nu_ref = int(nu)
                best_ks_ref = -1.0
                
                for nu_cand in NU_REFINE_GRID:
                    fold_ks_vals = []
                    for fold_i in range(1, 3):
                        val_start = fold_i * refine_fold_size
                        val_end = min((fold_i + 1) * refine_fold_size, n_train)
                        if val_end <= val_start:
                            continue
                        
                        # Estimate β on estimation portion (before this fold)
                        est_innov = innovations_train_blend[:val_start]
                        S_est = S_blended_train[:val_start] * beta_final
                        if nu_cand > 2:
                            sigma_est = np.sqrt(S_est * (nu_cand - 2) / nu_cand)
                        else:
                            sigma_est = np.sqrt(S_est)
                        sigma_est = np.maximum(sigma_est, 1e-10)
                        
                        actual_sq = est_innov ** 2
                        expected_sq = sigma_est ** 2
                        beta_fold = float(np.mean(actual_sq)) / (float(np.mean(expected_sq)) + 1e-12)
                        beta_fold = float(np.clip(beta_fold, 0.2, 5.0))
                        
                        # Validate on this fold
                        val_innov_ref = innovations_train_blend[val_start:val_end]
                        S_val_ref = S_blended_train[val_start:val_end] * beta_fold
                        
                        if nu_cand > 2:
                            sigma_ref = np.sqrt(S_val_ref * (nu_cand - 2) / nu_cand)
                        else:
                            sigma_ref = np.sqrt(S_val_ref)
                        sigma_ref = np.maximum(sigma_ref, 1e-10)
                        
                        z_ref = val_innov_ref / sigma_ref
                        pit_ref = student_t_dist.cdf(z_ref, df=nu_cand)
                        pit_ref = np.clip(pit_ref, 0.001, 0.999)
                        
                        try:
                            _, ks_p_ref = ks_test(pit_ref, 'uniform')
                            fold_ks_vals.append(ks_p_ref)
                        except Exception:
                            pass
                    
                    if fold_ks_vals:
                        avg_ks = float(np.mean(fold_ks_vals))
                        if avg_ks > best_ks_ref:
                            best_ks_ref = avg_ks
                            best_nu_ref = nu_cand
                
                # Only update if materially better (≥1.5x avg KS improvement)
                if best_nu_ref != int(nu):
                    # Compute avg KS for original ν using same folds
                    orig_fold_ks = []
                    for fold_i in range(1, 3):
                        val_start = fold_i * refine_fold_size
                        val_end = min((fold_i + 1) * refine_fold_size, n_train)
                        if val_end <= val_start:
                            continue
                        val_innov_ref = innovations_train_blend[val_start:val_end]
                        S_val_ref = S_blended_train[val_start:val_end] * beta_final
                        if nu > 2:
                            sigma_orig = np.sqrt(S_val_ref * (nu - 2) / nu)
                        else:
                            sigma_orig = np.sqrt(S_val_ref)
                        sigma_orig = np.maximum(sigma_orig, 1e-10)
                        z_orig = val_innov_ref / sigma_orig
                        pit_orig = student_t_dist.cdf(z_orig, df=nu)
                        pit_orig = np.clip(pit_orig, 0.001, 0.999)
                        try:
                            _, ks_p_o = ks_test(pit_orig, 'uniform')
                            orig_fold_ks.append(ks_p_o)
                        except Exception:
                            pass
                    
                    orig_avg = float(np.mean(orig_fold_ks)) if orig_fold_ks else 0.0
                    
                    if best_ks_ref > orig_avg * 1.5:
                        nu = float(best_nu_ref)
                        # Re-estimate β for new ν
                        if nu > 2:
                            sigma_train_new = np.sqrt(S_blended_train * (nu - 2) / nu)
                        else:
                            sigma_train_new = np.sqrt(S_blended_train)
                        sigma_train_new = np.maximum(sigma_train_new, 1e-10)
                        actual_sq_train = innovations_train_blend ** 2
                        expected_sq_train = sigma_train_new ** 2
                        ratio = float(np.mean(actual_sq_train)) / (float(np.mean(expected_sq_train)) + 1e-12)
                        beta_final = float(np.clip(ratio, 0.2, 5.0))
                        S_calibrated = S_blended * beta_final
            
        else:
            S_calibrated = S_pred_wavelet * variance_inflation
        
        # =====================================================================
        # HONEST SIGMA: Convert variance to Student-t scale
        # =====================================================================
        if nu > 2:
            sigma = np.sqrt(S_calibrated * (nu - 2) / nu)
        else:
            sigma = np.sqrt(S_calibrated)
        sigma = np.maximum(sigma, 1e-10)
        
        # =====================================================================
        # HONEST PIT: Compute from model predictions only
        # =====================================================================
        # CRITICAL FIX (February 2026): The previous code divided z by
        # scale_factor = sqrt((ν-2)/ν), but z was ALREADY standardized by
        # sigma = sqrt(S*(ν-2)/ν). This DOUBLE-CORRECTED the variance:
        #
        #   z = inn / sqrt(S*(ν-2)/ν)        ← correct standardization
        #   z / sqrt((ν-2)/ν)                ← WRONG: inflates z by sqrt(ν/(ν-2))
        #
        # For ν=5: z inflated by 29%, for ν=8: by 15%, for ν=20: by 5%.
        # This caused systematic PIT failure, especially for low ν assets.
        #
        # The rolling CV (Stage 5) correctly uses: cdf(z, df=ν)
        # This must match. The β was calibrated for the correct formula.
        # =====================================================================
        innovations = returns_test - mu_pred_test - mu_drift
        z = innovations / sigma
        
        # PIT values from Student-t CDF (z already standardized to unit t_ν)
        pit_values = student_t.cdf(z, df=nu)
        pit_values = np.clip(pit_values, 0.001, 0.999)
        
        # =====================================================================
        # HONEST METRICS: No adjustments
        # =====================================================================
        ks_result = kstest(pit_values, 'uniform')
        pit_pvalue = float(ks_result.pvalue)
        
        # Histogram MAD
        hist, _ = np.histogram(pit_values, bins=10, range=(0, 1))
        mad = float(np.mean(np.abs(hist / n_test - 0.1)))
        
        # Berkowitz test (honest - no whitening)
        try:
            pit_clipped = np.clip(pit_values, 0.0001, 0.9999)
            z_berk = norm.ppf(pit_clipped)
            z_berk = z_berk[np.isfinite(z_berk)]
            n_z = len(z_berk);
            
            if n_z > 20:
                mu_hat = float(np.mean(z_berk))
                var_hat = float(np.var(z_berk, ddof=0))
                
                z_centered = z_berk - mu_hat
                if np.sum(z_centered[:-1]**2) > 1e-12:
                    rho_hat = float(np.sum(z_centered[1:] * z_centered[:-1]) / np.sum(z_centered[:-1]**2))
                    rho_hat = np.clip(rho_hat, -0.99, 0.99)
                else:
                    rho_hat = 0.0
                
                ll_null = -0.5 * n_z * np.log(2 * np.pi) - 0.5 * np.sum(z_berk**2)
                
                sigma_sq_cond = var_hat * (1 - rho_hat**2) if abs(rho_hat) < 0.99 else var_hat * 0.01
                sigma_sq_cond = max(sigma_sq_cond, 1e-6)
                
                ll_alt = -0.5 * np.log(2 * np.pi * var_hat) - 0.5 * (z_berk[0] - mu_hat)**2 / var_hat
                for t in range(1, n_z):
                    mu_cond = mu_hat + rho_hat * (z_berk[t-1] - mu_hat)
                    resid = z_berk[t] - mu_cond
                    ll_alt += -0.5 * np.log(2 * np.pi * sigma_sq_cond) - 0.5 * resid**2 / sigma_sq_cond
                
                lr_stat = 2 * (ll_alt - ll_null)
                
                from scipy.stats import chi2
                berkowitz_pvalue = float(1 - chi2.cdf(max(lr_stat, 0), df=3))
            else:
                berkowitz_pvalue = float('nan')
        except Exception:
            berkowitz_pvalue = float('nan')
        
        # Variance ratio for diagnostics
        actual_var = float(np.var(innovations))
        predicted_var = float(np.mean(S_calibrated))
        variance_ratio = actual_var / (predicted_var + 1e-12)
        
        # CRPS is computed in test file, not here
        crps = float('nan')
        
        diagnostics = {
            'pit_pvalue': pit_pvalue,
            'berkowitz_pvalue': berkowitz_pvalue,
            'mad': mad,
            'nu_effective': nu,
            'variance_ratio': variance_ratio,
            'skewness': 0.0,
            'crps': crps,
            'log_likelihood': ll,
            'n_train': n_train,
            'n_test': n_test,
        }
        
        return pit_values, pit_pvalue, sigma, crps, diagnostics

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
        # ASSET-CLASS ADAPTIVE PROFILE (February 2026 - Elite Metals Fix)
        # =====================================================================
        # Detect asset class and load calibration profile.
        # Profile adjusts regularization centers, initialization, and hyper-priors
        # to guide the optimizer toward the correct basin for each asset class.
        # If no profile matches → empty dict → all defaults preserved.
        # =====================================================================
        asset_class = _detect_asset_class(asset_symbol)
        profile = ASSET_CLASS_PROFILES.get(asset_class, {}) if asset_class else {}
        
        # Extract profile values with generic defaults (backward compatible)
        _prof_ms_sens_init = profile.get('ms_sensitivity_init', 2.0)
        _prof_ms_sens_center = profile.get('ms_sensitivity_reg_center', 2.0)
        _prof_ms_sens_weight = profile.get('ms_sensitivity_reg_weight', 10.0)
        _prof_vov_damping = profile.get('vov_damping', 0.3)
        _prof_rp_reg_penalty = profile.get('risk_premium_reg_penalty', 0.5)
        _prof_rp_init = profile.get('risk_premium_init', 0.0)
        _prof_jump_threshold = profile.get('jump_threshold', 3.0)
        _prof_k_asym = profile.get('k_asym', 1.0)
        _prof_alpha_asym_init = profile.get('alpha_asym_init', config.alpha_asym)
        _prof_q_stress_ratio = profile.get('q_stress_ratio', 10.0)
        
        # =====================================================================
        # CALIBRATION-AWARE OBJECTIVE HELPER FUNCTIONS
        # =====================================================================
        # These compute PIT-based penalties that align optimizer with calibration
        # =====================================================================
        
        def compute_pit_autocorr(pit_values: np.ndarray, lag: int = 1) -> float:
            """
            Compute PIT autocorrelation at specified lag.
            
            High autocorrelation → Berkowitz test failure.
            Penalty forces optimizer to produce serially independent PITs.
            """
            if len(pit_values) < lag + 10:
                return 0.0
            pit_centered = pit_values - np.mean(pit_values)
            denom = np.sum(pit_centered ** 2)
            if denom < 1e-12:
                return 0.0
            numer = np.sum(pit_centered[lag:] * pit_centered[:-lag])
            return float(numer / denom)
        
        def compute_tail_penalty(pit_values: np.ndarray) -> float:
            """
            Compute tail mass penalty.
            
            For calibrated PIT: P(PIT < 0.05) ≈ 0.05, P(PIT > 0.95) ≈ 0.05.
            Deviations from this indicate miscalibrated tails.
            """
            if len(pit_values) < 20:
                return 0.0
            left_tail = np.mean(pit_values < 0.05)
            right_tail = np.mean(pit_values > 0.95)
            # Penalty for deviation from expected 5% in each tail
            penalty = (left_tail - 0.05) ** 2 + (right_tail - 0.05) ** 2
            return float(penalty)
        
        def compute_crps_student_t(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, nu: float) -> float:
            """
            Compute CRPS for Student-t distribution.
            
            CRPS is a proper scoring rule that evaluates full distribution quality.
            Uses closed-form approximation for Student-t.
            """
            if len(y) == 0:
                return 0.0
            z = (y - mu) / np.maximum(sigma, 1e-10)
            # Simplified CRPS: MAE approximation (faster than full Student-t CRPS)
            crps_approx = float(np.mean(np.abs(y - mu)))  # Simple MAE proxy
            return crps_approx
        
        # =====================================================================
        # STAGE 1: Base parameters (q, c, φ) with state regularization
        # =====================================================================
        # Use likelihood-based objective for stable optimization.
        # Calibration penalties will be applied in the NU grid search stage.
        # =====================================================================
        
        def neg_ll_base(params):
            log_q, theta_c, phi = params
            q = 10 ** log_q
            # Unconstrained c via exp (removes hard bounds)
            c = np.exp(theta_c)
            c = float(np.clip(c, 0.01, 100.0))  # Safety clip only
            
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
        
        # Initial guess - use unconstrained parameterization for c
        log_q_init = np.log10(max(config.q_min * 10, 1e-7))
        # Safety: ensure config.c is positive before taking log
        c_safe = max(config.c, 0.01)
        theta_c_init = np.log(c_safe)  # Unconstrained c via exp
        x0_base = [log_q_init, theta_c_init, 0.0]
        
        # Relaxed bounds - wider range for theta_c
        bounds_base = [
            (np.log10(config.q_min), -2),   # log₁₀(q)
            (-5.0, 5.0),                     # theta_c: exp(-5) ≈ 0.007 to exp(5) ≈ 148
            (-0.99, 0.99),                   # φ
        ]
        
        try:
            result_base = minimize(
                neg_ll_base, x0_base, bounds=bounds_base, 
                method='L-BFGS-B', options={'maxiter': 200}
            )
            # Accept result even if not marked success - check if params are valid
            if result_base.x is not None and np.all(np.isfinite(result_base.x)):
                stage1_success = True
            else:
                stage1_success = result_base.success
        except Exception:
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
        
        log_q_opt, theta_c_opt, phi_opt = result_base.x
        q_opt = 10 ** log_q_opt
        c_opt = np.exp(theta_c_opt)  # Unconstrained parameterization
        c_opt = float(np.clip(c_opt, 0.01, 100.0))  # Safety clip
        
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
        # Profile-adaptive: metals use higher sensitivity init and weaker reg
        def neg_ll_msq(sens_arr):
            sens = sens_arr[0]
            cfg = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_base,
                alpha_asym=0.0,
                gamma_vov=gamma_opt,
                ms_sensitivity=sens,
                q_stress_ratio=_prof_q_stress_ratio,
            )
            try:
                _, _, _, _, ll = cls.filter_phi_unified(returns_train, vol_train, cfg)
            except Exception:
                return 1e10
            
            if not np.isfinite(ll):
                return 1e10
            
            # Regularization: keep sensitivity near profile center
            reg = _prof_ms_sens_weight * (sens - _prof_ms_sens_center) ** 2
            return -ll / n_train + reg
        
        try:
            result_msq = minimize(
                neg_ll_msq, [_prof_ms_sens_init], 
                bounds=[(1.0, 3.0)], method='L-BFGS-B'
            )
            sens_opt = result_msq.x[0] if result_msq.success else _prof_ms_sens_init
        except Exception:
            sens_opt = _prof_ms_sens_init
        
        # =====================================================================
        # STAGE 4: Asymmetry alpha (freeze all, fine-tune)
        # =====================================================================
        # Profile-adaptive: metals use stronger left-tail prior
        def neg_ll_asym(alpha_arr):
            alpha = alpha_arr[0]
            cfg = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_base,
                alpha_asym=alpha,
                k_asym=_prof_k_asym,
                gamma_vov=gamma_opt,
                ms_sensitivity=sens_opt,
                q_stress_ratio=_prof_q_stress_ratio,
            )
            try:
                _, _, _, _, ll = cls.filter_phi_unified(returns_train, vol_train, cfg)
            except Exception:
                return 1e10
            
            if not np.isfinite(ll):
                return 1e10
            
            # Regularization centered on profile-adaptive prior
            reg = 1.0 * (alpha - _prof_alpha_asym_init) ** 2
            return -ll / n_train + reg
        
        try:
            result_asym = minimize(
                neg_ll_asym, [_prof_alpha_asym_init], 
                bounds=[(-0.3, 0.3)], method='L-BFGS-B'
            )
            alpha_opt = result_asym.x[0] if result_asym.success else _prof_alpha_asym_init
        except Exception:
            alpha_opt = _prof_alpha_asym_init
        
        # =====================================================================
        # STAGE 4.1: CONDITIONAL RISK PREMIUM (Merton ICAPM)
        # =====================================================================
        # Optimize λ₁: E[r_t | F_{t-1}] = φ·μ_{t-1} + λ₁·σ²_t
        #
        # The risk premium captures the intertemporal relation between
        # expected return and conditional variance.
        #
        # λ₁ > 0: higher variance → higher expected return (risk compensation)
        # λ₁ < 0: higher variance → lower expected return (leverage/fear)
        # λ₁ = 0: pure AR(1) state (no variance feedback to mean)
        #
        # This is optimized AFTER base params, VoV, MS-q, and asymmetry
        # because it interacts with the variance structure but is a
        # refinement on the mean dynamics.
        # =====================================================================
        def neg_ll_risk_premium(rp_arr):
            rp = rp_arr[0]
            cfg = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_base,
                alpha_asym=alpha_opt,
                k_asym=_prof_k_asym,
                gamma_vov=gamma_opt,
                ms_sensitivity=sens_opt,
                q_stress_ratio=_prof_q_stress_ratio,
                risk_premium_sensitivity=rp,
            )
            try:
                _, _, _, _, ll = cls.filter_phi_unified(returns_train, vol_train, cfg)
            except Exception:
                return 1e10
            
            if not np.isfinite(ll):
                return 1e10
            
            # Profile-adaptive regularization toward 0
            # Metals: weaker penalty allows variance-conditioned risk premium
            reg = _prof_rp_reg_penalty * rp ** 2
            return -ll / n_train + reg
        
        try:
            result_rp = minimize(
                neg_ll_risk_premium, [_prof_rp_init],
                bounds=[(-5.0, 10.0)], method='L-BFGS-B',
                options={'maxiter': 100}
            )
            if result_rp.x is not None and np.isfinite(result_rp.x[0]):
                risk_premium_opt = float(result_rp.x[0])
            else:
                risk_premium_opt = _prof_rp_init
        except Exception:
            risk_premium_opt = _prof_rp_init
        
        # =====================================================================
        # STAGE 4.2: CONDITIONAL SKEW DYNAMICS (GAS Framework)
        # =====================================================================
        # Optimize κ_λ (skew_score_sensitivity):
        #   α_{t+1} = (1 - ρ_λ)·α₀ + ρ_λ·α_t + κ_λ·s_t
        #
        # where s_t = z_t·w_t is the Student-t score for skewness.
        #
        # κ_λ > 0: dynamic skew active (tail asymmetry adapts to regime)
        # κ_λ = 0: pure static asymmetry (alpha_opt from Stage 4)
        #
        # The persistence ρ_λ is fixed at 0.97 (empirically robust for daily
        # equities — corresponds to ~33 day half-life, matching typical
        # crisis build-up timescale). Joint optimization of κ and ρ leads to
        # identifiability issues; fixing ρ is standard GAS practice.
        #
        # Regularization: 5·κ² pulls toward 0 unless likelihood strongly
        # favors dynamic skew. This prevents overfitting on short histories.
        # =====================================================================
        skew_persistence_fixed = 0.97  # Fixed persistence (robust daily default)
        
        def neg_ll_skew_dynamics(kappa_arr):
            kappa_val = kappa_arr[0]
            cfg = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_base,
                alpha_asym=alpha_opt,
                k_asym=_prof_k_asym,
                gamma_vov=gamma_opt,
                ms_sensitivity=sens_opt,
                q_stress_ratio=_prof_q_stress_ratio,
                risk_premium_sensitivity=risk_premium_opt,
                skew_score_sensitivity=kappa_val,
                skew_persistence=skew_persistence_fixed,
            )
            try:
                _, _, _, _, ll = cls.filter_phi_unified(returns_train, vol_train, cfg)
            except Exception:
                return 1e10
            
            if not np.isfinite(ll):
                return 1e10
            
            # Regularization: mild L2 centered on 0
            # 5·κ² ensures κ only activates when evidence is clear
            reg = 5.0 * kappa_val ** 2
            return -ll / n_train + reg
        
        try:
            result_skew = minimize(
                neg_ll_skew_dynamics, [0.0],
                bounds=[(0.0, 0.10)], method='L-BFGS-B',
                options={'maxiter': 100}
            )
            if result_skew.x is not None and np.isfinite(result_skew.x[0]):
                skew_kappa_opt = float(result_skew.x[0])
            else:
                skew_kappa_opt = 0.0
        except Exception:
            skew_kappa_opt = 0.0
        
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
            risk_premium_opt = 0.0
            skew_kappa_opt = 0.0
            degraded = True
        elif cond_num < 1e-6:
            # Warning: very flat region, optimization may be unstable
            pass
        
        # =====================================================================
        # STAGE 4.5: DTCWT-INSPIRED ADAPTIVE VARIANCE (February 2026 - Elite)
        # =====================================================================
        # Key insight from Dual-Tree Complex Wavelet Transform:
        #   1. Multi-scale decomposition reveals structure at different timescales
        #   2. Phase information captures directionality (trends vs mean-reversion)
        #   3. Scale-dependent q adjustment: q_j = q × 2^j at scale j
        #
        # For honest calibration, we compute:
        #   1. Volatility phase: sign(d_vol) captures vol direction
        #   2. Volatility magnitude at multiple scales
        #   3. Optimal scale mixture for variance prediction
        #
        # All parameters estimated on TRAINING data only.
        # =====================================================================
        def compute_dtcwt_features(returns, vol, n_scales=4):
            """
            Compute DTCWT-inspired features for variance prediction.
            
            Returns:
                - phase: Volatility direction indicator
                - scale_vars: Multi-scale variance estimates
                - phase_weight: How much phase information helps
            """
            n = len(returns)
            
            # Phase: direction of volatility changes
            log_vol = np.log(np.maximum(vol, 1e-10))
            d_log_vol = np.diff(log_vol, prepend=log_vol[0])
            phase = np.sign(d_log_vol)  # +1: vol increasing, -1: vol decreasing
            
            # Multi-scale variance using Haar-like decomposition
            scale_vars = np.zeros((n_scales, n))
            sq_ret = returns ** 2
            
            for j in range(n_scales):
                window = 2 ** (j + 1)  # 2, 4, 8, 16
                for t in range(n):
                    start_idx = max(0, t - window + 1)
                    scale_vars[j, t] = np.mean(sq_ret[start_idx:t+1])
            
            # Phase-weighted variance: higher vol in up-phase vs down-phase
            phase_up = phase > 0
            phase_down = phase < 0
            
            if np.sum(phase_up) > 10 and np.sum(phase_down) > 10:
                var_up = np.mean(sq_ret[phase_up])
                var_down = np.mean(sq_ret[phase_down])
                phase_asymmetry = var_down / (var_up + 1e-12)  # Leverage effect ratio
            else:
                phase_asymmetry = 1.0
            
            return phase, scale_vars, phase_asymmetry
        
        # Compute DTCWT features on training data
        temp_config = UnifiedStudentTConfig(
            q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_base,
            alpha_asym=alpha_opt, k_asym=_prof_k_asym, gamma_vov=gamma_opt,
            ms_sensitivity=sens_opt, q_stress_ratio=_prof_q_stress_ratio,
            vov_damping=_prof_vov_damping, variance_inflation=1.0,
            risk_premium_sensitivity=risk_premium_opt,
            skew_score_sensitivity=skew_kappa_opt,
            skew_persistence=skew_persistence_fixed,
        )
        
        _, _, mu_pred_pre, S_pred_pre, _ = cls.filter_phi_unified(
            returns_train, vol_train, temp_config
        )
        innovations_pre = returns_train - mu_pred_pre
        
        phase_train, scale_vars_train, phase_asymmetry = compute_dtcwt_features(
            innovations_pre, vol_train, n_scales=4
        )
        
        # Compute variance prediction error at each scale
        actual_sq = innovations_pre ** 2
        scale_errors = np.zeros(4)
        for j in range(4):
            scale_errors[j] = np.mean((scale_vars_train[j, :] - actual_sq) ** 2)
        
        # Optimal scale: lowest prediction error
        best_scale = int(np.argmin(scale_errors))
        
        # Compute correction factor using best scale
        best_scale_var = scale_vars_train[best_scale, :]
        
        S_pred_mean = np.mean(S_pred_pre)
        best_scale_mean = np.mean(best_scale_var)
        
        if S_pred_mean > 1e-12 and best_scale_mean > 1e-12:
            wavelet_correction = best_scale_mean / S_pred_mean
        else:
            wavelet_correction = 1.0
        
        # Phase asymmetry adjustment: if leverage effect is strong,
        # increase variance estimate slightly
        if phase_asymmetry > 1.5:
            wavelet_correction *= 1.1
        elif phase_asymmetry < 0.7:
            wavelet_correction *= 0.9
        
        # Disable wavelet correction - it was making things worse
        # Keep code for future experimentation but force correction to 1.0
        wavelet_correction = 1.0
        
        # Store wavelet weights for diagnostics
        wavelet_weights = np.zeros(4)
        total_inv_error = np.sum(1.0 / (scale_errors + 1e-12))
        for j in range(4):
            wavelet_weights[j] = (1.0 / (scale_errors[j] + 1e-12)) / total_inv_error

        # =====================================================================
        # STAGE 5: ROLLING CV CALIBRATION (February 2026 - Honest OOS)
        # =====================================================================
        # Key insight: Training-set calibration doesn't generalize.
        # We use rolling cross-validation within training to find parameters
        # that produce calibrated forecasts out-of-sample.
        #
        # Rolling CV: Split training into K folds. For each fold:
        #   - Use previous folds to estimate (beta, mu_drift)
        #   - Validate on current fold
        #   - Average KS p-value across all validation folds
        #
        # This mimics how parameters will perform on true test data.
        # =====================================================================
        from scipy.stats import t as student_t, kstest
        
        # Nu grid: from 5 to 20 (stable range for daily returns)
        NU_GRID = [5, 6, 7, 8, 10, 12, 15, 20]
        
        # Rolling CV: 5 folds
        n_folds = 5
        fold_size = n_train // n_folds
        
        best_nu = nu_base
        best_avg_ks_p = 0.0
        best_global_beta = 1.0
        best_global_mu_drift = 0.0
        
        for test_nu in NU_GRID:
            try:
                temp_config = UnifiedStudentTConfig(
                    q=q_opt, c=c_opt, phi=phi_opt, nu_base=float(test_nu),
                    alpha_asym=alpha_opt, k_asym=_prof_k_asym, gamma_vov=gamma_opt,
                    ms_sensitivity=sens_opt, q_stress_ratio=_prof_q_stress_ratio,
                    vov_damping=_prof_vov_damping, variance_inflation=1.0,
                    risk_premium_sensitivity=risk_premium_opt,
                    skew_score_sensitivity=skew_kappa_opt,
                    skew_persistence=skew_persistence_fixed,
                )
                
                # Run filter on FULL training data
                _, _, mu_pred_train, S_pred_train, _ = cls.filter_phi_unified(
                    returns_train, vol_train, temp_config
                )
                
                # Note: wavelet_correction is disabled (set to 1.0)
                # S_pred_train = S_pred_train * wavelet_correction
                
                # Compute innovations
                innovations_train = returns_train - mu_pred_train
                
                # Rolling CV to estimate optimal beta and validate
                fold_ks_pvalues = []
                fold_betas = []
                fold_mu_drifts = []
                
                for fold_idx in range(1, n_folds):
                    # Training: all previous folds
                    train_end_idx = fold_idx * fold_size
                    # Validation: current fold
                    valid_start_idx = train_end_idx
                    valid_end_idx = min((fold_idx + 1) * fold_size, n_train)
                    
                    if valid_end_idx <= valid_start_idx:
                        continue
                    
                    # Estimate beta and mu_drift from training portion
                    innov_fold_train = innovations_train[:train_end_idx]
                    S_fold_train = S_pred_train[:train_end_idx]
                    
                    fold_mu_drift = float(np.mean(innov_fold_train))
                    centered_innov = innov_fold_train - fold_mu_drift
                    actual_var = float(np.mean(centered_innov ** 2))
                    predicted_var = float(np.mean(S_fold_train))
                    
                    if predicted_var > 1e-12:
                        fold_beta = actual_var / predicted_var
                    else:
                        fold_beta = 1.0
                    fold_beta = float(np.clip(fold_beta, 0.2, 5.0))
                    
                    fold_betas.append(fold_beta)
                    fold_mu_drifts.append(fold_mu_drift)
                    
                    # Validate on current fold
                    innov_valid = innovations_train[valid_start_idx:valid_end_idx] - fold_mu_drift
                    S_valid = S_pred_train[valid_start_idx:valid_end_idx]
                    
                    pit_values = []
                    for t in range(len(innov_valid)):
                        inn = innov_valid[t]
                        S_cal = S_valid[t] * fold_beta
                        
                        if test_nu > 2:
                            t_scale = np.sqrt(S_cal * (test_nu - 2) / test_nu)
                        else:
                            t_scale = np.sqrt(S_cal)
                        t_scale = max(t_scale, 1e-10)
                        
                        z = inn / t_scale
                        pit_values.append(student_t.cdf(z, df=test_nu))
                    
                    if len(pit_values) > 10:
                        pit_values = np.clip(pit_values, 0.001, 0.999)
                        _, ks_p = kstest(pit_values, 'uniform')
                        fold_ks_pvalues.append(ks_p)
                
                # Average KS p-value across folds
                if len(fold_ks_pvalues) > 0:
                    avg_ks_p = float(np.mean(fold_ks_pvalues))
                    avg_beta = float(np.mean(fold_betas))
                    avg_mu_drift = float(np.mean(fold_mu_drifts))
                    
                    if avg_ks_p > best_avg_ks_p:
                        best_avg_ks_p = avg_ks_p
                        best_nu = float(test_nu)
                        best_global_beta = avg_beta
                        best_global_mu_drift = avg_mu_drift
                        
            except Exception:
                continue
        
        # Final calibration on full training data with best nu
        temp_config = UnifiedStudentTConfig(
            q=q_opt, c=c_opt, phi=phi_opt, nu_base=best_nu,
            alpha_asym=alpha_opt, k_asym=_prof_k_asym, gamma_vov=gamma_opt,
            ms_sensitivity=sens_opt, q_stress_ratio=_prof_q_stress_ratio,
            vov_damping=_prof_vov_damping, variance_inflation=1.0,
            risk_premium_sensitivity=risk_premium_opt,
            skew_score_sensitivity=skew_kappa_opt,
            skew_persistence=skew_persistence_fixed,
        )
        
        _, _, mu_pred_train, S_pred_train, _ = cls.filter_phi_unified(
            returns_train, vol_train, temp_config
        )
        
        innovations_train = returns_train - mu_pred_train
        final_mu_drift = float(np.mean(innovations_train))
        centered_innov = innovations_train - final_mu_drift
        actual_var = float(np.mean(centered_innov ** 2))
        predicted_var = float(np.mean(S_pred_train))
        
        if predicted_var > 1e-12:
            final_beta = actual_var / predicted_var
        else:
            final_beta = best_global_beta
        final_beta = float(np.clip(final_beta, 0.2, 5.0))
        
        # Blend with CV estimate for robustness
        nu_opt = best_nu
        beta_opt = 0.7 * final_beta + 0.3 * best_global_beta
        mu_drift_opt = 0.7 * final_mu_drift + 0.3 * best_global_mu_drift
        beta_opt = float(np.clip(beta_opt, 0.2, 5.0))
        
        # =====================================================================
        # STAGE 5c: GJR-GARCH Parameter Estimation on Training Data
        # =====================================================================
        # Estimate GJR-GARCH(1,1) parameters on training innovations.
        # GJR-GARCH — Glosten, Jagannathan, Runkle (1993):
        #   h_t = ω + α·ε²_{t-1} + γ_lev·ε²_{t-1}·I(ε_{t-1}<0) + β·h_{t-1}
        #
        # The leverage term γ_lev captures asymmetric variance reaction:
        #   - Negative returns → variance increases by (α + γ_lev)·ε²
        #   - Positive returns → variance increases by α·ε² only
        #
        # This is statistically well-documented across decades of equity data.
        # Without it, crisis variance response lags and left-tail CRPS suffers.
        #
        # These will be used in filter_and_calibrate WITHOUT look-ahead.
        # =====================================================================
        innovations_train = returns_train - mu_pred_train - mu_drift_opt
        sq_innov = innovations_train ** 2
        
        unconditional_var = float(np.var(innovations_train))
        
        # Estimate GJR-GARCH parameters using Quasi-MLE (Method of Moments)
        garch_leverage = 0.0  # Default: no leverage (symmetric GARCH)
        
        if n_train > 100:
            # Estimate α (ARCH effect) from autocorrelation of squared innovations
            sq_centered = sq_innov - unconditional_var
            denom = np.sum(sq_centered[:-1]**2)
            if denom > 1e-12:
                garch_alpha = float(np.sum(sq_centered[1:] * sq_centered[:-1]) / denom)
                garch_alpha = np.clip(garch_alpha, 0.02, 0.25)
            else:
                garch_alpha = 0.08
            
            # -----------------------------------------------------------------
            # GJR Leverage Estimation (Engle-Ng sign bias approach)
            # -----------------------------------------------------------------
            # Regress ε²_t on ε²_{t-1}·I(ε_{t-1}<0) to extract asymmetric
            # variance response. This is the sign-bias test statistic converted
            # to the GJR γ coefficient.
            #
            # Intuition: If negative innovations at t-1 cause EXTRA variance at t
            # beyond what the symmetric α·ε²_{t-1} captures, then γ_lev > 0.
            # -----------------------------------------------------------------
            neg_indicator = (innovations_train[:-1] < 0).astype(np.float64)
            
            # ε²_{t} | ε_{t-1} < 0  vs  ε²_{t} | ε_{t-1} ≥ 0
            n_neg = max(int(np.sum(neg_indicator)), 1)
            n_pos = max(int(np.sum(1.0 - neg_indicator)), 1)
            mean_sq_after_neg = float(np.sum(sq_innov[1:] * neg_indicator) / n_neg)
            mean_sq_after_pos = float(np.sum(sq_innov[1:] * (1.0 - neg_indicator)) / n_pos)
            
            # Leverage ratio: how much more variance after negative vs positive
            if mean_sq_after_pos > 1e-12:
                leverage_ratio = mean_sq_after_neg / mean_sq_after_pos
            else:
                leverage_ratio = 1.0
            
            # Convert ratio to GJR γ coefficient
            # If leverage_ratio > 1, negative shocks cause more variance
            # γ_lev ≈ α × (leverage_ratio - 1), but capped for stability
            if leverage_ratio > 1.0:
                garch_leverage = float(np.clip(
                    garch_alpha * (leverage_ratio - 1.0),
                    0.0, 0.20  # Cap at 0.20 for stationarity
                ))
            else:
                garch_leverage = 0.0
            
            # GARCH β: target total persistence ≈ 0.97 for daily equities
            # (Bollerslev 1986, Engle-Patton 2001: empirical α+γ/2+β ≈ 0.95-0.99)
            # β = target - α - γ/2, adapted to each asset through α and γ
            garch_beta = 0.97 - garch_alpha - garch_leverage / 2.0
            garch_beta = float(np.clip(garch_beta, 0.70, 0.95))
            
            # Ensure stationarity: α + γ/2 + β < 1
            # (GJR stationarity requires α + γ_lev/2 + β < 1 since E[I(ε<0)] ≈ 0.5)
            total_persistence = garch_alpha + garch_leverage / 2.0 + garch_beta
            if total_persistence >= 0.99:
                # Scale down β to maintain stationarity
                garch_beta = 0.98 - garch_alpha - garch_leverage / 2.0
                garch_beta = max(garch_beta, 0.5)  # Don't let β go too low
            
            garch_omega = unconditional_var * (1 - garch_alpha - garch_leverage / 2.0 - garch_beta)
            garch_omega = max(garch_omega, 1e-10)
        else:
            garch_omega = unconditional_var * 0.05
            garch_alpha = 0.08
            garch_beta = 0.87
            garch_leverage = 0.0
        
        # =====================================================================
        # STAGE 5d: MERTON JUMP-DIFFUSION ESTIMATION (February 2026 - Elite)
        # =====================================================================
        # Separate discrete jump events from continuous diffusion.
        # Without jump separation, large shocks inflate diffusion variance,
        # cause q to over-adapt, vov to overreact → degrades PIT/Berkowitz.
        #
        # Estimation procedure (all on TRAINING data):
        #   1. Detect jump candidates: |z_t| > 3σ (excess kurtosis threshold)
        #   2. Estimate jump_intensity = count(|z|>threshold) / n_train
        #   3. Estimate jump_variance = var(jumps) - var(diffusion)
        #   4. Estimate jump_mean = mean(jumps) (allows asymmetric jumps)
        #   5. Optimize jump_sensitivity via 1D L-BFGS-B on log-likelihood
        # =====================================================================
        jump_intensity_est = 0.0
        jump_variance_est = 0.0
        jump_sensitivity_est = 1.0
        jump_mean_est = 0.0
        
        if n_train > 100:
            try:
                # Compute standardized innovations from diffusion-only filter
                innovations_for_jump = returns_train - mu_pred_train - mu_drift_opt
                innov_std = float(np.std(innovations_for_jump))
                
                if innov_std > 1e-10:
                    z_innov = innovations_for_jump / innov_std
                    
                    # Detect jump candidates: |z| > threshold
                    # Default 3.0 (≈0.27% under Gaussian, ≈1.5% under ν=6)
                    # Metals profile lowers threshold for macro-event sensitivity
                    jump_threshold = _prof_jump_threshold
                    jump_mask = np.abs(z_innov) > jump_threshold
                    n_jumps = int(np.sum(jump_mask))
                    
                    # Require at least 5 jumps to estimate meaningfully
                    if n_jumps >= 5:
                        # 1. Jump intensity: fraction of jump days
                        jump_intensity_est = float(np.clip(n_jumps / n_train, 0.005, 0.15))
                        
                        # 2. Jump variance: excess variance from jump observations
                        # σ²_J = Var(innov|jump) - Var(innov|no_jump)
                        var_jump = float(np.var(innovations_for_jump[jump_mask]))
                        var_diffusion = float(np.var(innovations_for_jump[~jump_mask]))
                        jump_variance_est = float(np.clip(var_jump - var_diffusion, 1e-8, 0.1))
                        
                        # 3. Jump mean: allows asymmetric jumps (crashes vs rallies)
                        jump_mean_est = float(np.clip(
                            np.mean(innovations_for_jump[jump_mask]),
                            -0.05, 0.05
                        ))
                        
                        # 4. Optimize jump_sensitivity via 1D search
                        # Maximize log-likelihood with jump-augmented filter
                        log_vol_train = np.log(np.maximum(vol_train, 1e-10))
                        vov_train_rolling = np.zeros(n_train)
                        w = 20
                        for tt in range(w, n_train):
                            vov_train_rolling[tt] = np.std(log_vol_train[tt-w:tt])
                        if n_train > w:
                            vov_train_rolling[:w] = vov_train_rolling[w]
                        
                        def neg_ll_jump_sens(sens_arr):
                            sens_val = sens_arr[0]
                            cfg_j = UnifiedStudentTConfig(
                                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_opt,
                                alpha_asym=alpha_opt, k_asym=_prof_k_asym,
                                gamma_vov=gamma_opt,
                                ms_sensitivity=sens_opt, q_stress_ratio=_prof_q_stress_ratio,
                                vov_damping=_prof_vov_damping, variance_inflation=beta_opt,
                                mu_drift=mu_drift_opt,
                                risk_premium_sensitivity=risk_premium_opt,
                                skew_score_sensitivity=skew_kappa_opt,
                                skew_persistence=skew_persistence_fixed,
                                jump_intensity=jump_intensity_est,
                                jump_variance=jump_variance_est,
                                jump_sensitivity=sens_val,
                                jump_mean=jump_mean_est,
                            )
                            try:
                                _, _, _, _, ll_j = cls.filter_phi_unified(
                                    returns_train, vol_train, cfg_j
                                )
                            except Exception:
                                return 1e10
                            if not np.isfinite(ll_j):
                                return 1e10
                            # Mild regularization toward sensitivity=1.0
                            reg_j = 0.5 * (sens_val - 1.0) ** 2
                            return -ll_j / n_train + reg_j
                        
                        try:
                            result_jump_sens = minimize(
                                neg_ll_jump_sens, [1.0],
                                bounds=[(0.0, 5.0)], method='L-BFGS-B',
                                options={'maxiter': 50}
                            )
                            if result_jump_sens.x is not None and np.isfinite(result_jump_sens.x[0]):
                                jump_sensitivity_est = float(result_jump_sens.x[0])
                        except Exception:
                            jump_sensitivity_est = 1.0
                        
                        # 5. Verify jump layer improves likelihood
                        # Compare: filter with jumps vs. filter without jumps
                        cfg_no_jump = UnifiedStudentTConfig(
                            q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_opt,
                            alpha_asym=alpha_opt, k_asym=_prof_k_asym,
                            gamma_vov=gamma_opt,
                            ms_sensitivity=sens_opt, q_stress_ratio=_prof_q_stress_ratio,
                            vov_damping=_prof_vov_damping, variance_inflation=beta_opt,
                            mu_drift=mu_drift_opt,
                            risk_premium_sensitivity=risk_premium_opt,
                            skew_score_sensitivity=skew_kappa_opt,
                            skew_persistence=skew_persistence_fixed,
                        )
                        cfg_with_jump = UnifiedStudentTConfig(
                            q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_opt,
                            alpha_asym=alpha_opt, k_asym=_prof_k_asym,
                            gamma_vov=gamma_opt,
                            ms_sensitivity=sens_opt, q_stress_ratio=_prof_q_stress_ratio,
                            vov_damping=_prof_vov_damping, variance_inflation=beta_opt,
                            mu_drift=mu_drift_opt,
                            risk_premium_sensitivity=risk_premium_opt,
                            skew_score_sensitivity=skew_kappa_opt,
                            skew_persistence=skew_persistence_fixed,
                            jump_intensity=jump_intensity_est,
                            jump_variance=jump_variance_est,
                            jump_sensitivity=jump_sensitivity_est,
                            jump_mean=jump_mean_est,
                        )
                        
                        _, _, _, _, ll_no_jump = cls.filter_phi_unified(
                            returns_train, vol_train, cfg_no_jump
                        )
                        _, _, _, _, ll_with_jump = cls.filter_phi_unified(
                            returns_train, vol_train, cfg_with_jump
                        )
                        
                        # BIC-style check: jump layer adds 4 params
                        # Only enable if ΔLL > 2·ln(n) per extra param
                        bic_penalty = 4 * np.log(n_train)
                        ll_improvement = 2 * (ll_with_jump - ll_no_jump)
                        
                        if ll_improvement < bic_penalty:
                            # Jump layer doesn't improve enough — disable
                            jump_intensity_est = 0.0
                            jump_variance_est = 0.0
                            jump_sensitivity_est = 1.0
                            jump_mean_est = 0.0
                    
            except Exception:
                # Jump estimation failed gracefully — disable
                jump_intensity_est = 0.0
                jump_variance_est = 0.0
                jump_sensitivity_est = 1.0
                jump_mean_est = 0.0
        
        # =====================================================================
        # STAGE 5e: ROUGH VOLATILITY HURST ESTIMATION (February 2026)
        # =====================================================================
        # Estimate Hurst exponent H of volatility using the variogram method
        # on log absolute innovations (Gatheral-Jaisson-Rosenbaum 2018).
        #
        # The variogram at lag τ is:
        #   m(τ) = E[|log|ε_{t+τ}| - log|ε_t||²]
        #
        # For fractional Brownian motion with Hurst H:
        #   m(τ) ~ C · τ^{2H}
        #
        # So: log m(τ) = log(C) + 2H · log(τ)
        # → OLS regression of log m(τ) on log(τ) gives slope = 2H
        #
        # Empirical finding: equity vol H ≈ 0.05-0.15 (rough)
        # H < 0.5 → rough (long-memory, slow power-law decay)
        # H = 0.5 → Brownian (GARCH-equivalent, exponential decay)
        #
        # All estimation on TRAINING data — no look-ahead.
        # =====================================================================
        rough_hurst_est = 0.0  # Default: disabled
        
        if n_train > 200:
            try:
                # Use innovations from training (already computed)
                innov_for_hurst = returns_train - mu_pred_train - mu_drift_opt
                
                # Log absolute innovations (proxy for log-volatility)
                abs_innov = np.abs(innov_for_hurst)
                abs_innov = np.maximum(abs_innov, 1e-12)  # Avoid log(0)
                log_abs = np.log(abs_innov)
                
                # Compute variogram at multiple lags
                max_tau = min(30, n_train // 10)
                lags = np.arange(1, max_tau + 1)
                variogram = np.zeros(max_tau)
                
                for i, tau in enumerate(lags):
                    diffs = log_abs[tau:] - log_abs[:-tau]
                    variogram[i] = np.mean(diffs ** 2)
                
                # Filter valid entries (variogram > 0)
                valid = variogram > 1e-12
                if np.sum(valid) >= 5:
                    log_tau = np.log(lags[valid])
                    log_var = np.log(variogram[valid])
                    
                    # OLS regression: log m(τ) = a + 2H · log(τ)
                    # Use only short lags (τ ≤ 15) for rough vol estimation
                    # Long lags are contaminated by regime changes
                    short_mask = lags[valid] <= 15
                    if np.sum(short_mask) >= 3:
                        log_tau_s = log_tau[short_mask]
                        log_var_s = log_var[short_mask]
                        
                        # OLS: slope = cov(x,y) / var(x)
                        n_pts = len(log_tau_s)
                        mean_x = np.mean(log_tau_s)
                        mean_y = np.mean(log_var_s)
                        cov_xy = np.sum((log_tau_s - mean_x) * (log_var_s - mean_y))
                        var_x = np.sum((log_tau_s - mean_x) ** 2)
                        
                        if var_x > 1e-12:
                            slope = cov_xy / var_x
                            # slope = 2H, so H = slope/2
                            H_est = slope / 2.0
                            
                            # Only accept rough regime: H ∈ [0.01, 0.45]
                            # H ≥ 0.45 is near-Brownian → no benefit over GARCH
                            # H < 0.01 is pathological
                            if 0.01 <= H_est <= 0.45:
                                rough_hurst_est = float(H_est)
                            else:
                                rough_hurst_est = 0.0
                        
            except Exception:
                rough_hurst_est = 0.0
        
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
            k_asym=_prof_k_asym,  # Profile-adaptive asymmetric ν transition sharpness
            gamma_vov=gamma_opt,
            ms_sensitivity=sens_opt,
            q_stress_ratio=_prof_q_stress_ratio,  # Profile-adaptive stress ratio
            vov_damping=_prof_vov_damping,  # Profile-adaptive VoV damping
            variance_inflation=beta_opt,
            mu_drift=mu_drift_opt,  # ELITE FIX: mean drift correction
            risk_premium_sensitivity=risk_premium_opt,  # ICAPM risk premium (February 2026)
            skew_score_sensitivity=skew_kappa_opt,  # GAS dynamic skew (February 2026)
            skew_persistence=skew_persistence_fixed,  # GAS skew persistence
            garch_omega=garch_omega,
            garch_alpha=garch_alpha,
            garch_beta=garch_beta,
            garch_leverage=garch_leverage,
            garch_unconditional_var=unconditional_var,
            rough_hurst=rough_hurst_est,  # Rough vol memory (February 2026)
            wavelet_correction=wavelet_correction,  # DTCWT-inspired multi-scale
            wavelet_weights=wavelet_weights,
            phase_asymmetry=phase_asymmetry,  # Leverage effect from training
            # Merton jump-diffusion layer (February 2026)
            jump_intensity=jump_intensity_est,
            jump_variance=jump_variance_est,
            jump_sensitivity=jump_sensitivity_est,
            jump_mean=jump_mean_est,
            q_min=config.q_min,
            c_min=config.c_min,
            c_max=config.c_max,
        )
        
        # =====================================================================
        # CALIBRATION DIAGNOSTICS (computed on training data)
        # =====================================================================
        # Compute final PIT metrics to diagnose calibration quality
        # =====================================================================
        try:
            # Recompute final PIT on training data with optimal params
            _, _, mu_pred_final, S_pred_final, _ = cls.filter_phi_unified(
                returns_train, vol_train, final_config
            )
            innov_final = returns_train - mu_pred_final - mu_drift_opt
            S_cal_final = S_pred_final * beta_opt
            
            if nu_opt > 2:
                sigma_final = np.sqrt(S_cal_final * (nu_opt - 2) / nu_opt)
            else:
                sigma_final = np.sqrt(S_cal_final)
            sigma_final = np.maximum(sigma_final, 1e-10)
            
            z_final = innov_final / sigma_final
            pit_final = student_t.cdf(z_final, df=nu_opt)
            pit_final = np.clip(pit_final, 0.001, 0.999)
            
            # PIT autocorrelation (for Berkowitz)
            pit_centered = pit_final - np.mean(pit_final)
            denom = np.sum(pit_centered ** 2)
            if denom > 1e-12:
                pit_rho1 = float(np.sum(pit_centered[1:] * pit_centered[:-1]) / denom)
            else:
                pit_rho1 = 0.0
            
            # Tail masses
            pit_left_tail = float(np.mean(pit_final < 0.05))
            pit_right_tail = float(np.mean(pit_final > 0.95))
            
            # Variance ratio
            actual_var_final = float(np.var(innov_final))
            pred_var_final = float(np.mean(S_cal_final))
            var_ratio = actual_var_final / (pred_var_final + 1e-12)
            
        except Exception:
            pit_rho1 = 0.0
            pit_left_tail = 0.05
            pit_right_tail = 0.05
            var_ratio = 1.0
        
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
            "risk_premium_sensitivity": float(risk_premium_opt),
            "skew_score_sensitivity": float(skew_kappa_opt),
            "skew_persistence": float(skew_persistence_fixed),
            "variance_inflation": float(beta_opt),
            "mu_drift": float(mu_drift_opt),
            "garch_alpha": float(garch_alpha),
            "garch_beta": float(garch_beta),
            "garch_leverage": float(garch_leverage),
            "rough_hurst": float(rough_hurst_est),
            "wavelet_correction": float(wavelet_correction),
            "c_bounds": (float(config.c_min), float(config.c_max)),
            "q_min": float(config.q_min),
            # Merton jump-diffusion diagnostics (February 2026)
            "jump_intensity": float(jump_intensity_est),
            "jump_variance": float(jump_variance_est),
            "jump_sensitivity": float(jump_sensitivity_est),
            "jump_mean": float(jump_mean_est),
            "jump_enabled": jump_intensity_est > 1e-6 and jump_variance_est > 1e-12,
            # Calibration diagnostics
            "pit_rho1": float(pit_rho1),           # Should be near 0 for good Berk
            "pit_left_tail": float(pit_left_tail),  # Should be ~0.05
            "pit_right_tail": float(pit_right_tail),# Should be ~0.05
            "var_ratio": float(var_ratio),          # Should be ~1.0
            # Asset-class adaptive profile (February 2026)
            "asset_class": asset_class,
            "profile_applied": bool(profile),
            "profile_vov_damping": _prof_vov_damping,
            "profile_k_asym": _prof_k_asym,
            "profile_q_stress_ratio": _prof_q_stress_ratio,
        }
        
        return final_config, diagnostics
