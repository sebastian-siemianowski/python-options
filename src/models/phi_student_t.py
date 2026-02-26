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
        # MS-q and fused LFO-CV wrappers
        run_ms_q_student_t_filter,
        run_student_t_filter_with_lfo_cv,
        run_student_t_filter_with_lfo_cv_batch,
        # Unified filter
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


# ---------------------------------------------------------------------------
# φ SHRINKAGE PRIOR CONSTANTS (self-contained, no external dependencies)
# ---------------------------------------------------------------------------

PHI_SHRINKAGE_TAU_MIN = 1e-3
PHI_SHRINKAGE_GLOBAL_DEFAULT = 0.0
PHI_SHRINKAGE_LAMBDA_DEFAULT = 0.05

# Discrete ν grid for Student-t models
# ν=10 and ν=15 added prevents BMA oscillation between
# 8↔12 and 12↔20 for metals (GC=F, SI=F) which live in ν≈10–15 range.
STUDENT_T_NU_GRID = [4, 6, 8, 10, 12, 15, 20]


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

# ---------------------------------------------------------------------------
# HIGH-VOLATILITY EQUITY SYMBOLS
# ---------------------------------------------------------------------------
# Crypto-correlated, meme, and micro-cap stocks with kurtosis >> 6.
# Need lower ν, sharper asymmetry, and weaker VoV damping for CRPS.
# ---------------------------------------------------------------------------
HIGH_VOL_EQUITY_SYMBOLS = frozenset({
    'MSTR', 'AMZE', 'RCAT', 'SMCI', 'RGTI', 'QBTS', 'BKSY',
    'SPCE', 'ABTC', 'BZAI', 'BNZI', 'AIRI',
})


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
    if asset_symbol is None:
        return None
    sym = asset_symbol.strip().upper()
    if sym in METALS_GOLD_SYMBOLS:
        return 'metals_gold'
    if sym in METALS_SILVER_SYMBOLS:
        return 'metals_silver'
    if sym in METALS_OTHER_SYMBOLS:
        return 'metals_other'
    if sym in HIGH_VOL_EQUITY_SYMBOLS:
        return 'high_vol_equity'
    return None


# ---------------------------------------------------------------------------
# ASSET-CLASS CALIBRATION PROFILES
# ---------------------------------------------------------------------------
# Each profile is a dict of override hints consumed by optimize_params_unified.
# Keys match parameter names in the optimizer stages.
# Missing keys → fall back to generic defaults (backward compatible).
# ---------------------------------------------------------------------------
ASSET_CLASS_PROFILES: Dict[str, Dict[str, float]] = {
    # ---------------------------------------------------------------------------
    # GOLD — slow macro regimes, higher MS sensitivity, near-zero VoV damping
    # ---------------------------------------------------------------------------
    'metals_gold': {
        'ms_sensitivity_init': 4.0,         # Higher initial MS-q sensitivity
        'ms_sensitivity_reg_center': 4.0,   # Regularize toward 2.5, not 2.0
        'ms_sensitivity_reg_weight': 5.0,   # Weaker reg (was 10.0)
        'ms_ewm_lambda': 0.97,             # EWM z-score (~33-day half-life)
        'vov_damping': 0.0,                # Near-zero: let VoV + MS-q coexist
        'risk_premium_reg_penalty': 0.1,    # Weaker pull toward 0 (was 0.5)
        'risk_premium_init': 0.5,           # Start from positive risk premium
        'jump_threshold': 2.5,              # More sensitive jump detection (was 3.0)
        'k_asym': 1.5,                      # Sharper asymmetric ν transition
        'alpha_asym_init': -0.08,           # Mild left-tail prior
        'q_stress_ratio': 15.0,             # Wider calm/stress gap for macro regimes
    },
    # ---------------------------------------------------------------------------
    # SILVER — leveraged-gold, explosive VoV, sharp left-tail fattening
    # ---------------------------------------------------------------------------
    'metals_silver': {
        'ms_sensitivity_init': 4.5,         # Even higher — explosive regimes
        'ms_sensitivity_reg_center': 4.5,   # Regularize toward 2.8
        'ms_sensitivity_reg_weight': 3.0,   # Very weak reg
        'ms_ewm_lambda': 0.94,             # Faster EWM (~16-day half-life)
        'vov_damping': 0.0,                # Minimal: silver IS VoV-dominated
        'risk_premium_reg_penalty': 0.05,   # Very weak — silver is variance-conditioned
        'risk_premium_init': 0.8,           # Stronger risk premium prior
        'jump_threshold': 2.2,              # More sensitive (silver gaps often)
        'k_asym': 1.8,                      # Sharp asymmetric transition
        'alpha_asym_init': -0.15,           # Stronger left-tail prior
        'q_stress_ratio': 20.0,             # Wide calm/stress gap
    },
    # ---------------------------------------------------------------------------
    # OTHER METALS — moderate adjustments between generic and gold
    # ---------------------------------------------------------------------------
    'metals_other': {
        'ms_sensitivity_init': 2.3,
        'ms_sensitivity_reg_center': 2.3,
        'ms_sensitivity_reg_weight': 7.0,
        'ms_ewm_lambda': 0.96,             # Moderate EWM (~25-day half-life)
        'vov_damping': 0.10,
        'risk_premium_reg_penalty': 0.2,
        'risk_premium_init': 0.3,
        'jump_threshold': 2.7,
        'k_asym': 1.3,
        'alpha_asym_init': -0.05,
        'q_stress_ratio': 12.0,
    },
    # ---------------------------------------------------------------------------
    # HIGH-VOL EQUITY — lower ν, frequent gaps, weaker VoV damping
    # Lower ν concentrates mass closer to mean → sharper CRPS forecasts.
    # ---------------------------------------------------------------------------
    'high_vol_equity': {
        'ms_sensitivity_init': 3.0,         # Faster regime detection
        'ms_sensitivity_reg_center': 3.0,   # Regularize toward 3.0
        'ms_sensitivity_reg_weight': 5.0,   # Moderate reg
        'ms_ewm_lambda': 0.95,             # EWM z-score (~20-day half-life)
        'vov_damping': 0.05,               # Near-zero: let VoV breathe
        'risk_premium_reg_penalty': 0.1,    # Weak pull toward 0
        'risk_premium_init': 0.3,           # Mild risk premium prior
        'jump_threshold': 2.5,              # Lower threshold — frequent gaps
        'k_asym': 1.5,                      # Sharper asymmetric ν transition
        'alpha_asym_init': -0.10,           # Left-tail fattening prior
        'q_stress_ratio': 15.0,             # Wide calm/stress gap
    },
}


# ---------------------------------------------------------------------------
# UNIFIED STUDENT-T CONFIGURATION
# ---------------------------------------------------------------------------
# Consolidates 48+ model variants into single adaptive architecture:
#   - Smooth asymmetric ν (replaces Two-Piece)
#   - Probabilistic MS-q (replaces threshold-based)
#   - VoV with redundancy damping
#   - Momentum integration
#   - State collapse regularization
#   - Asset-class adaptive profiles (metals, commodities)
# ---------------------------------------------------------------------------

@dataclass
class UnifiedStudentTConfig:
    """
    Configuration for Unified Student-T Model.

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
    ms_sensitivity: float = 2.0  # [1.0, 5.0], regime sensitivity
    ms_ewm_lambda: float = 0.0   # EWM decay for MS-q z-score baseline.
                                  # 0.0 = expanding window (backward-compatible).
                                  # 0.94-0.99 = EWM with corresponding half-life.
                                  # Gold: 0.97 (~33-day). Silver: 0.94 (~16-day).

    # VoV with MS-q redundancy damping
    gamma_vov: float = 0.3       # [0, 1.0], VoV sensitivity
    vov_damping: float = 0.3     # [0, 0.5], reduce VoV when MS-q active
    vov_window: int = 20         # Rolling window for VoV

    # ---------------------------------------------------------------------------
    # CONDITIONAL RISK PREMIUM (Merton ICAPM)
    # E[r_t | F_{t-1}] = φ·μ_{t-1} + λ₁·σ²_t
    # λ₁ > 0: risk compensation,  λ₁ < 0: leverage/fear effect,  λ₁ = 0: disabled
    # Typical: λ₁ ∈ [-2, 5] for daily equities
    # ---------------------------------------------------------------------------
    risk_premium_sensitivity: float = 0.0  # λ₁ ∈ [-5, 10], 0.0 = disabled

    # ---------------------------------------------------------------------------
    # CONDITIONAL SKEW DYNAMICS (GAS framework, Creal-Koopman-Lucas 2013)
    # α_{t+1} = (1 - ρ_λ)·α₀ + ρ_λ·α_t + κ_λ·s_t
    #   s_t = z_t·w_t  (Student-t score: standardized innovation × tail weight)
    #   ρ_λ ∈ [0.90, 0.99]  (mean-reversion speed of skew)
    #   κ_λ ∈ [0, 0.05]     (score sensitivity; 0 = static α)
    # Large negative z → heavier left tail; large positive z → heavier right.
    # α_t clipped to [-0.3, 0.3] for stability.
    # ---------------------------------------------------------------------------
    skew_score_sensitivity: float = 0.0  # κ_λ ≥ 0, 0.0 = disabled
    skew_persistence: float = 0.97       # ρ_λ ∈ [0.90, 0.99], skew mean-reversion speed

    # CALIBRATION Variance inflation
    # Multiplies S_pred by β to ensure predictive variance ≈ returns variance
    # β < 1: model was over-estimating variance (rare)
    # β > 1: model was under-estimating variance (common for q→0 collapse)
    variance_inflation: float = 1.0  # [0.5, 5.0], optimized for PIT uniformity

    # CALIBRATION Mean drift correction
    # Equities have positive risk premium (~15-25% annualized) that the zero-mean
    # Kalman filter doesn't capture. This causes systematic positive innovations
    # and right-skewed PIT histogram.
    # mu_drift = mean(returns - mu_pred) on training data
    mu_drift: float = 0.0  # Mean bias correction for PIT

    # GJR-GARCH parameters for honest variance dynamics
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

    # ---------------------------------------------------------------------------
    # ROUGH VOLATILITY MEMORY (Gatheral-Jaisson-Rosenbaum 2018)
    # Power-law decay w_k ~ k^(H-3/2) vs GARCH exponential w_k = β^k.
    # H < 0.5 → rough regime (equity H ≈ 0.1): slower post-crisis decay.
    # H = 0 → disabled,  H ∈ (0, 0.5) → blended with GJR-GARCH.
    # ---------------------------------------------------------------------------
    rough_hurst: float = 0.0  # Hurst exponent H ∈ [0, 0.5], 0.0 = disabled

    # ---------------------------------------------------------------------------
    # MERTON JUMP-DIFFUSION
    # r_t = μ_t + σ_t·ε_t + J_t,  J_t ~ Bernoulli(p_t)·N(μ_J, σ²_J)
    # p_t = logistic(logit(p₀) + b·vov_t)
    # Separates discrete jumps from continuous diffusion — keeps GARCH clean.
    # ---------------------------------------------------------------------------
    jump_intensity: float = 0.0     # Base jump probability p₀ ∈ [0, 0.15]
                                     # 0.0 = disabled, 0.02 ≈ 5 jumps/year
    jump_variance: float = 0.0      # σ²_J jump size variance, 0.0 = disabled
    jump_sensitivity: float = 1.0   # b in p_t = logistic(a₀ + b·vov_t)
    jump_mean: float = 0.0          # μ_J jump mean (allows asymmetric jumps)

    # ---------------------------------------------------------------------------
    # CAUSAL EWM LOCATION CORRECTION (Durbin-Koopman 2012)
    # ewm_μ[t] = λ·ewm_μ[t-1] + (1-λ)·(y_{t-1} - μ_{t-1});  μ_pred[t] += ewm_μ[t]
    # Mops up residual autocorrelation when the state equation misses short-term dynamics.
    # λ = 0: disabled,  λ ∈ (0.90, 0.97): typical (10-33 day half-life)
    # ---------------------------------------------------------------------------
    crps_ewm_lambda: float = 0.0  # EWM decay for location correction, 0.0 = disabled

    # ---------------------------------------------------------------------------
    # LEVERAGE CORRELATION (Black 1976, Heston 1993)
    # h_t = h_garch × (1 + ρ_lev × min(ε_{t-1}/σ_{t-1}, 0)²)
    # GJR captures sign-based asymmetry; ρ_lev adds magnitude-dependent scaling.
    # ρ = 0: disabled,  ρ ∈ (0, 2): typical 0.3-1.0
    # ---------------------------------------------------------------------------
    rho_leverage: float = 0.0  # Leverage correlation ∈ [0, 2], 0 = disabled

    # ---------------------------------------------------------------------------
    # VARIANCE MEAN REVERSION (Heston 1993)
    # h_t = (1-κ)·h_garch + κ·θ_long
    # Adds explicit mean-reversion speed beyond GARCH's implicit ω/(1-α-β-γ/2).
    # κ = 0: pure GARCH,  κ ∈ (0, 0.3): faster pull toward θ_long
    # ---------------------------------------------------------------------------
    kappa_mean_rev: float = 0.0    # Mean reversion speed ∈ [0, 0.3], 0 = disabled
    theta_long_var: float = 0.0    # Long-term variance target, 0 = use unconditional

    # ---------------------------------------------------------------------------
    # CRPS-OPTIMAL SIGMA SHRINKAGE (Gneiting & Raftery 2007)
    # σ_crps = σ_pit × α_crps
    # The CRPS-optimal σ is slightly TIGHTER than the PIT-calibrated σ:
    #   σ*_crps ≈ σ_cal × √(ν/((ν-2)(1+1/ν))) < σ_cal
    # Estimated via golden section on training CRPS.
    # α = 1.0: no shrinkage,  α < 1.0: tighter (lower CRPS, PIT unchanged)
    # ---------------------------------------------------------------------------
    crps_sigma_shrinkage: float = 1.0  # CRPS sigma multiplier ∈ [0.5, 1.0], 1.0 = disabled

    # ---------------------------------------------------------------------------
    # VOL-OF-VOL NOISE σ_η (Heston discrete analog)
    # h_t += σ_η × max(0, |z_{t-1}| - 1.5)² × h_{t-1}
    # Threshold nonlinearity: only extreme shocks (|z|>1.5) amplify variance.
    # σ_η = 0: disabled,  σ_η ∈ (0, 0.5): typical 0.05-0.20
    # ---------------------------------------------------------------------------
    sigma_eta: float = 0.0  # Vol-of-vol noise ∈ [0, 0.5], 0 = disabled

    # ---------------------------------------------------------------------------
    # ASYMMETRIC ν OFFSET (structural two-piece tails)
    # z < 0: ν_left  = ν_base - t_df_asym  (heavier crash tail)
    # z ≥ 0: ν_right = ν_base + t_df_asym  (lighter rally tail)
    # Complements dynamic α_asym with a fixed structural split.
    # 0: symmetric,  > 0: heavier left (typical),  < 0: heavier right
    # ---------------------------------------------------------------------------
    t_df_asym: float = 0.0  # Asymmetric ν offset ∈ [-3, 3], 0 = disabled

    # ---------------------------------------------------------------------------
    # MARKOV REGIME SWITCH (observation variance layer)
    # p_stress_t = (1-p_switch)·p_stress_{t-1} + p_switch·I(|z|>2)
    # h_t = (1-p_stress)·h_garch + p_stress·(h_garch × stress_mult)
    # Different from MS-q: MS-q modulates state noise q,
    # this modulates observation variance h_t.
    # p = 0: disabled,  p ∈ (0, 0.15): typical 0.03-0.10
    # ---------------------------------------------------------------------------
    regime_switch_prob: float = 0.0  # Calm→stress transition ∈ [0, 0.15], 0 = disabled

    # ---------------------------------------------------------------------------
    # GARCH-KALMAN VARIANCE RECONCILIATION
    # R_t = (1-w)·c·σ²_ewma + w·h_garch_t
    # Blends GARCH h_t into filter observation noise so Kalman gain
    # K_t = P_pred/(P_pred + R_t) uses a more accurate R_t.
    # w = 0: pure EWMA (default),  w ∈ (0, 0.6]: GARCH-informed R_t
    # ---------------------------------------------------------------------------
    garch_kalman_weight: float = 0.0  # w ∈ [0, 0.6], 0 = disabled

    # ---------------------------------------------------------------------------
    # PROCESS-NOISE VOLATILITY COUPLING
    # Q_t = q_ms × (1 + ζ × max(0, h_t/θ_long - 1))
    # When h_t >> θ_long, drift state μ_t can change faster.
    # Uses GARCH variance ratio (lag-corrected), complementary to MS-q.
    # ζ = 0: disabled,  ζ ∈ (0, 1.0]: typical 0.1-0.5
    # ---------------------------------------------------------------------------
    q_vol_coupling: float = 0.0  # ζ ∈ [0, 1.0], 0 = disabled

    # ---------------------------------------------------------------------------
    # CONDITIONAL LOCATION BIAS CORRECTION
    # μ_corrected = μ_pred + a·(h_t - θ_long) + b·sign(μ)·√|μ|
    # Term a: variance-state bias (risk-return concavity)
    # Term b: James-Stein drift magnitude shrinkage
    # Both |a|, |b| < 0.5.  Gated: only enabled if CRPS improves > 0.3%.
    # ---------------------------------------------------------------------------
    loc_bias_var_coeff: float = 0.0    # a ∈ [-0.5, 0.5], 0 = disabled
    loc_bias_drift_coeff: float = 0.0  # b ∈ [-0.5, 0.5], 0 = disabled

    # ---------------------------------------------------------------------------
    # PRE-CALIBRATED PIPELINE PARAMETERS
    # ---------------------------------------------------------------------------
    # Estimated ONCE during tuning (Stage 6) and read directly in
    # filter_and_calibrate. No search/CV in filter_and_calibrate.
    # One-way flow: optimize_params_unified → config → filter_and_calibrate.
    # ---------------------------------------------------------------------------
    calibrated_gw: float = 0.50          # GARCH blend weight
    calibrated_nu_pit: float = 0.0       # ν for PIT (0 = use nu_base)
    calibrated_nu_crps: float = 0.0      # ν for CRPS (0 = use nu_base)
    calibrated_beta_probit_corr: float = 1.0  # Probit-variance β correction
    calibrated_lambda_rho: float = 0.985 # EWM decay / AR(1) whitening λ

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
        self.ms_sensitivity = float(np.clip(self.ms_sensitivity, 1.0, 5.0))
        self.gamma_vov = float(np.clip(self.gamma_vov, 0.0, 1.0))
        self.vov_damping = float(np.clip(self.vov_damping, 0.0, 0.5))
        # Jump parameters
        self.jump_intensity = float(np.clip(self.jump_intensity, 0.0, 0.15))
        self.jump_variance = float(np.clip(self.jump_variance, 0.0, 0.1))
        self.jump_sensitivity = float(np.clip(self.jump_sensitivity, 0.0, 5.0))
        self.jump_mean = float(np.clip(self.jump_mean, -0.05, 0.05))
        # Rough volatility
        self.rough_hurst = float(np.clip(self.rough_hurst, 0.0, 0.5))
        # Leverage and mean reversion
        self.rho_leverage = float(np.clip(self.rho_leverage, 0.0, 2.0))
        self.kappa_mean_rev = float(np.clip(self.kappa_mean_rev, 0.0, 0.3))
        self.crps_sigma_shrinkage = float(np.clip(self.crps_sigma_shrinkage, 0.3, 1.0))
        # CRPS-enhancement parameters
        self.sigma_eta = float(np.clip(self.sigma_eta, 0.0, 0.5))
        self.t_df_asym = float(np.clip(self.t_df_asym, -3.0, 3.0))
        self.regime_switch_prob = float(np.clip(self.regime_switch_prob, 0.0, 0.15))
        # GARCH-Kalman reconciliation + Q_t coupling + location bias
        self.garch_kalman_weight = float(np.clip(self.garch_kalman_weight, 0.0, 0.6))
        self.q_vol_coupling = float(np.clip(self.q_vol_coupling, 0.0, 1.0))
        self.loc_bias_var_coeff = float(np.clip(self.loc_bias_var_coeff, -0.5, 0.5))
        self.loc_bias_drift_coeff = float(np.clip(self.loc_bias_drift_coeff, -0.5, 0.5))

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
        # More sensitive formula for gamma variation across assets
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

        # Scale-aware q_min: q must contribute meaningfully to S = P_pred + R.
        # If q → 0 then P_pred → 0, S ≈ R, and EWMA mismatches → U-shaped PIT.
        obs_var = vol_median ** 2
        ret_var = float(np.var(returns_clean)) if len(returns_clean) > 30 else obs_var

        # q_min: at least 1e-6 (numerical), 5% of obs_var, 2% of ret_var
        q_min = max(1e-6, 0.05 * obs_var, 0.02 * ret_var)

        # Initial q from excess return variance (q absorbs ~50% of gap)
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
            'ms_ewm_lambda': self.ms_ewm_lambda,
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
            # GARCH-Kalman reconciliation + Q_t coupling + location bias
            'garch_kalman_weight': self.garch_kalman_weight,
            'q_vol_coupling': self.q_vol_coupling,
            'loc_bias_var_coeff': self.loc_bias_var_coeff,
            'loc_bias_drift_coeff': self.loc_bias_drift_coeff,
        }


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

        warmup = min(20, n)
        ewm_mean = float(np.mean(vol[:warmup])) if warmup > 0 else float(vol[0])
        ewm_var = float(np.var(vol[:warmup])) if warmup > 1 else 1e-6
        ewm_var = max(ewm_var, 1e-12)

        vol_zscore = np.zeros(n)
        for t in range(n):
            ewm_std = np.sqrt(ewm_var)
            ewm_std = max(ewm_std, 1e-6)
            vol_zscore[t] = (vol[t] - ewm_mean) / ewm_std

            # Update AFTER computing z-score (no look-ahead)
            ewm_mean = lam * ewm_mean + (1.0 - lam) * vol[t]
            diff = vol[t] - ewm_mean
            ewm_var = lam * ewm_var + (1.0 - lam) * (diff * diff)
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
    Unified Student-t Kalman filter with heavy tails, regime switching, and CRPS tuning.

    Encapsulates Student-t heavy-tail logic so drift model behavior stays modular.
    Parameter estimation is performed by optimize_params_unified(), which decomposes
    into 15 sequential stages (13 estimation + 1 safety gate + 1 diagnostics).
    Each stage freezes all upstream parameters and optimizes <= 2 new ones.

    optimize_params_unified — Stage Dependency Chain
    =========================================================================
    1 (q,c,φ) → 2 (γ_vov) → 3 (ms_sens) → 4 (α_asym)
      → 4.1 (risk_prem) → 4.2 (skew_κ) → [hessian check]
        → 5 (ν CV) → 5c (GARCH) → 5d (jumps)
          → 5e (Hurst) → 5f (EWM λ) → 5g (leverage+shrinkage)
            → 5h (loc_bias) → [build diagnostics]
    =========================================================================

    Stage 1  (q, c, φ)         Base Kalman filter: process noise q, observation
                                scale c, persistence φ. L-BFGS-B with regularization
                                to prevent φ→1 / q→0 random walk degeneracy.

    Stage 2  (γ_vov)           VoV: how much R_t responds to Δlog(σ_t).
                                Freezes (q,c,φ), optimizes γ via 1D MLE.

    Stage 3  (ms_sens)          MS process noise sensitivity. Controls calm→stress
                                transition aggressiveness. Profile-adaptive (metals
                                use higher sensitivity, weaker regularization).

    Stage 4  (α_asym)          Asymmetric tail thickness:
                                ν_eff = ν_base × (1 + α·tanh(k·z)).
                                α > 0 ⟹ heavier left tail (crash).
                                α < 0 ⟹ heavier right tail.

    Stage 4.1 (λ₁ risk_prem)  ICAPM conditional risk premium:
                                E[r|F] = φ·μ + λ₁·σ².
                                λ₁ > 0: risk compensation.
                                λ₁ < 0: leverage/fear effect.

    Stage 4.2 (κ_λ skew)      GAS skew dynamics:
                                α_{t+1} = (1-ρ)·α₀ + ρ·α_t + κ_λ·s_t.
                                Score-driven time-varying skewness.
                                ρ fixed at 0.97 (~33d half-life).

    Hessian check               Condition number guard: if cond(H⁻¹) > 10⁶,
                                disable advanced features (γ→0, α→0, etc.).

    Stage 5  (ν CV)            Rolling 5-fold CV for degrees of freedom ν.
                                Gneiting-Raftery: sharpness subject to calibration.
                                Selects ν with best KS p-value + CRPS.

    Stage 5c (GARCH)            GJR-GARCH(1,1) on Kalman innovations.
                                h_t = ω + α·ε² + γ_lev·ε²·I(ε<0) + β·h.

    Stage 5d (jumps)            Merton jump-diffusion: detect |z| > threshold,
                                estimate (λ_jump, σ²_jump). BIC-gated.

    Stage 5e (Hurst)            Rough vol Hurst exponent (H < 0.5 = rough).
                                Variogram on log|ε| → H = slope/2.

    Stage 5f (EWM λ)           CRPS-optimal EWM location correction.
                                Mops up residual innovation autocorrelation.

    Stage 5g (leverage+shrink) Heston-DLSV sequential CRPS minimization:
                                Phase 1: ρ_leverage × κ_mean_rev grid
                                Phase 2: σ_eta (vol-of-vol noise)
                                Phase 3: regime_switch_prob
                                Phase 4: t_df_asym (asymmetric ν offset)
                                Phase 5: CRPS-optimal σ shrinkage

    Stage 5h (a, b loc_bias)   Conditional location bias correction:
                                μ += a·(h_t - θ_long) + b·sign(μ)·√|μ|.
                                a: variance-state bias (risk-return concavity).
                                b: James-Stein drift shrinkage.
                                Both |a|, |b| < 0.5.

    Build diagnostics           Assemble final config, run filter, compute
                                PIT/KS/CRPS/Berkowitz. Return calibration report.
    """

    nu_min_default: float = 2.1
    nu_max_default: float = 30.0

    @staticmethod
    def _clip_nu(nu: float, nu_min: float, nu_max: float) -> float:
        return float(np.clip(float(nu), nu_min, nu_max))

    @staticmethod
    def _variance_to_scale(variance: float, nu: float) -> float:
        """Convert predictive variance to Student-t scale: scale = √(Var × (ν-2)/ν) for ν > 2."""
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
        modulation = 1.0 + alpha * np.tanh(k * z)
        nu_raw = nu_base * modulation

        # Ensure ν > 2 (finite variance requirement)
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
        Staged optimization for unified Student-t Kalman filter model.

        Orchestrates 15 optimization stages in sequence, threading results
        forward. Each stage is a self-contained classmethod for testability.

        Stage dependency chain:
          1 (q,c,φ) → 2 (γ_vov) → 3 (ms_sens) → 4 (α_asym)
            → 4.1 (risk_prem) → 4.2 (skew_κ) → [hessian check]
              → 5 (ν CV) → 5c (GARCH) → 5d (jumps)
                → 5e (Hurst) → 5f (EWM λ)
                → 5g (leverage+shrinkage) → 5h (location bias)

        Args:
            returns: Return series
            vol: EWMA/GK volatility series
            nu_base: Base degrees of freedom (from discrete grid)
            train_frac: Fraction for train/test split
            asset_symbol: Asset symbol for profile selection

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

        # ── Asset-class adaptive profile
        asset_class = _detect_asset_class(asset_symbol)
        profile = ASSET_CLASS_PROFILES.get(asset_class, {}) if asset_class else {}

        # ── STAGE 1: Base parameters (q, c, φ)
        s1 = cls._stage_1_base_params(returns_train, vol_train, n_train, nu_base, config)
        if not s1['success']:
            return config, {"stage": 0, "success": False,
                            "error": "Stage 1 optimization failed", "degraded": True}

        q_opt, c_opt, phi_opt = s1['q'], s1['c'], s1['phi']
        log_q_opt = s1['log_q']

        # ── STAGE 2: VoV gamma
        gamma_opt = cls._stage_2_vov_gamma(
            returns_train, vol_train, n_train, nu_base,
            q_opt, c_opt, phi_opt, config)

        # ── STAGE 3: MS-q sensitivity
        sens_opt = cls._stage_3_ms_sensitivity(
            returns_train, vol_train, n_train, nu_base,
            q_opt, c_opt, phi_opt, gamma_opt, profile)

        # ── STAGE 4: Asymmetry alpha
        alpha_opt = cls._stage_4_asymmetry(
            returns_train, vol_train, n_train, nu_base,
            q_opt, c_opt, phi_opt, gamma_opt, sens_opt, profile)

        # ── STAGE 4.1: Risk premium (ICAPM)
        risk_premium_opt = cls._stage_4_1_risk_premium(
            returns_train, vol_train, n_train, nu_base,
            q_opt, c_opt, phi_opt, gamma_opt, sens_opt, alpha_opt, profile)

        # ── STAGE 4.2: Skew dynamics (GAS)
        skew_kappa_opt, skew_persistence_fixed = cls._stage_4_2_skew_dynamics(
            returns_train, vol_train, n_train, nu_base,
            q_opt, c_opt, phi_opt, gamma_opt, sens_opt, alpha_opt,
            risk_premium_opt, profile)

        # ── Hessian condition check
        hess = cls._check_hessian_degradation(
            s1.get('result'), gamma_opt, sens_opt, alpha_opt,
            risk_premium_opt, skew_kappa_opt)
        gamma_opt = hess['gamma_opt']
        sens_opt = hess['sens_opt']
        alpha_opt = hess['alpha_opt']
        risk_premium_opt = hess['risk_premium_opt']
        skew_kappa_opt = hess['skew_kappa_opt']
        degraded = hess['degraded']
        cond_num = hess['cond_num']

        # ── STAGE 5: Rolling CV ν selection
        _prof_ms_ewm_lambda = profile.get('ms_ewm_lambda', 0.0)
        is_metals_asset = _prof_ms_ewm_lambda > 0.01
        is_high_vol_asset = asset_class == 'high_vol_equity'
        use_heavy_tail_grid = is_metals_asset or is_high_vol_asset

        s5 = cls._stage_5_nu_cv_selection(
            returns_train, vol_train, n_train,
            q_opt, c_opt, phi_opt, alpha_opt, gamma_opt,
            sens_opt, risk_premium_opt, skew_kappa_opt,
            skew_persistence_fixed, nu_base, profile,
            use_heavy_tail_grid)
        nu_opt = s5['nu_opt']
        beta_opt = s5['beta_opt']
        mu_drift_opt = s5['mu_drift_opt']
        mu_pred_train = s5['mu_pred_train']

        # ── STAGE 5c: GJR-GARCH
        garch = cls._stage_5c_garch_estimation(
            returns_train, mu_pred_train, mu_drift_opt, n_train)

        # ── STAGE 5c.1/5c.2: DISABLED (ablation study) ──────
        # Empirically proven zero CRPS benefit across 8 assets.
        # 5c.1 garch_kalman_weight: 0/8 helped, 1/8 hurt (-1.7% CRPS)
        # 5c.2 q_vol_coupling: 0/8 helped, 1/8 hurt (-0.1% CRPS)
        garch_kalman_w = 0.0
        q_vol_zeta = 0.0

        # ── STAGE 5d: Merton jump-diffusion
        jumps = cls._stage_5d_jump_diffusion(
            returns_train, vol_train, mu_pred_train, mu_drift_opt,
            n_train, q_opt, c_opt, phi_opt, nu_opt, alpha_opt,
            gamma_opt, sens_opt, beta_opt, risk_premium_opt,
            skew_kappa_opt, skew_persistence_fixed, profile)

        # ── STAGE 5e: Rough Hurst
        rough_hurst_est = cls._stage_5e_rough_hurst(
            returns_train, mu_pred_train, mu_drift_opt, n_train)

        # ── STAGE 5f: EWM location correction
        crps_ewm_lambda_opt = cls._stage_5f_ewm_lambda(
            returns_train, vol_train, n_train,
            q_opt, c_opt, phi_opt, nu_opt, alpha_opt, gamma_opt,
            sens_opt, beta_opt, mu_drift_opt, risk_premium_opt,
            skew_kappa_opt, skew_persistence_fixed, profile)

        # ── STAGE 5g: Leverage + shrinkage
        s5g = cls._stage_5g_leverage_and_shrinkage(
            returns_train, vol_train, n_train,
            q_opt, c_opt, phi_opt, nu_opt, alpha_opt, gamma_opt,
            sens_opt, beta_opt, mu_drift_opt, risk_premium_opt,
            skew_kappa_opt, skew_persistence_fixed, crps_ewm_lambda_opt,
            garch['garch_omega'], garch['garch_alpha'],
            garch['garch_beta'], garch['garch_leverage'],
            garch['unconditional_var'], profile)

        # ── STAGE 5h: Conditional location bias
        s5h = cls._stage_5h_conditional_location_bias(
            returns_train, vol_train, n_train,
            q_opt, c_opt, phi_opt, nu_opt, alpha_opt, gamma_opt,
            sens_opt, beta_opt, mu_drift_opt, risk_premium_opt,
            skew_kappa_opt, skew_persistence_fixed, crps_ewm_lambda_opt,
            garch['garch_omega'], garch['garch_alpha'],
            garch['garch_beta'], garch['garch_leverage'],
            garch['unconditional_var'], garch_kalman_w, q_vol_zeta, profile)

        # ── STAGE 6: Walk-forward calibration params
        # Build a temp config with all Stages 1-5h params for Stage 6
        _s6_config = UnifiedStudentTConfig(
            q=q_opt, c=c_opt, phi=phi_opt,
            nu_base=nu_opt,
            alpha_asym=alpha_opt,
            k_asym=profile.get('k_asym', 1.0),
            gamma_vov=gamma_opt,
            ms_sensitivity=sens_opt,
            ms_ewm_lambda=profile.get('ms_ewm_lambda', 0.0),
            q_stress_ratio=profile.get('q_stress_ratio', 10.0),
            vov_damping=profile.get('vov_damping', 0.3),
            variance_inflation=beta_opt,
            mu_drift=mu_drift_opt,
            risk_premium_sensitivity=risk_premium_opt,
            skew_score_sensitivity=skew_kappa_opt,
            skew_persistence=skew_persistence_fixed,
            garch_omega=garch['garch_omega'],
            garch_alpha=garch['garch_alpha'],
            garch_beta=garch['garch_beta'],
            garch_leverage=garch['garch_leverage'],
            garch_unconditional_var=garch['unconditional_var'],
            rough_hurst=rough_hurst_est,
            jump_intensity=jumps['jump_intensity'],
            jump_variance=jumps['jump_variance'],
            jump_sensitivity=jumps['jump_sensitivity'],
            jump_mean=jumps['jump_mean'],
            crps_ewm_lambda=crps_ewm_lambda_opt,
            rho_leverage=s5g['rho_leverage'],
            kappa_mean_rev=s5g['kappa_mean_rev'],
            theta_long_var=s5g['theta_long_var'],
            crps_sigma_shrinkage=s5g['crps_sigma_shrinkage'],
            sigma_eta=s5g['sigma_eta'],
            t_df_asym=s5g['t_df_asym'],
            regime_switch_prob=s5g['regime_switch_prob'],
            garch_kalman_weight=garch_kalman_w,
            q_vol_coupling=q_vol_zeta,
            loc_bias_var_coeff=s5h['loc_bias_var_coeff'],
            loc_bias_drift_coeff=s5h['loc_bias_drift_coeff'],
            q_min=config.q_min, c_min=config.c_min, c_max=config.c_max,
        )
        s6 = cls._stage_6_calibration_pipeline(returns, vol, _s6_config, train_frac)

        # ── BUILD FINAL CONFIG
        _prof_k_asym = profile.get('k_asym', 1.0)
        _prof_q_stress_ratio = profile.get('q_stress_ratio', 10.0)
        _prof_vov_damping = profile.get('vov_damping', 0.3)
        _prof_ms_ewm_lambda_val = profile.get('ms_ewm_lambda', 0.0)

        final_config = UnifiedStudentTConfig(
            q=q_opt, c=c_opt, phi=phi_opt,
            nu_base=nu_opt,
            alpha_asym=alpha_opt, k_asym=_prof_k_asym,
            gamma_vov=gamma_opt,
            ms_sensitivity=sens_opt,
            ms_ewm_lambda=_prof_ms_ewm_lambda_val,
            q_stress_ratio=_prof_q_stress_ratio,
            vov_damping=_prof_vov_damping,
            variance_inflation=beta_opt,
            mu_drift=mu_drift_opt,
            risk_premium_sensitivity=risk_premium_opt,
            skew_score_sensitivity=skew_kappa_opt,
            skew_persistence=skew_persistence_fixed,
            garch_omega=garch['garch_omega'],
            garch_alpha=garch['garch_alpha'],
            garch_beta=garch['garch_beta'],
            garch_leverage=garch['garch_leverage'],
            garch_unconditional_var=garch['unconditional_var'],
            rough_hurst=rough_hurst_est,
            jump_intensity=jumps['jump_intensity'],
            jump_variance=jumps['jump_variance'],
            jump_sensitivity=jumps['jump_sensitivity'],
            jump_mean=jumps['jump_mean'],
            crps_ewm_lambda=crps_ewm_lambda_opt,
            rho_leverage=s5g['rho_leverage'],
            kappa_mean_rev=s5g['kappa_mean_rev'],
            theta_long_var=s5g['theta_long_var'],
            crps_sigma_shrinkage=s5g['crps_sigma_shrinkage'],
            sigma_eta=s5g['sigma_eta'],
            t_df_asym=s5g['t_df_asym'],
            regime_switch_prob=s5g['regime_switch_prob'],
            garch_kalman_weight=garch_kalman_w,
            q_vol_coupling=q_vol_zeta,
            loc_bias_var_coeff=s5h['loc_bias_var_coeff'],
            loc_bias_drift_coeff=s5h['loc_bias_drift_coeff'],
            q_min=config.q_min, c_min=config.c_min, c_max=config.c_max,
            # Stage 6: pre-calibrated walk-forward params
            calibrated_gw=s6['calibrated_gw'],
            calibrated_nu_pit=s6['calibrated_nu_pit'],
            calibrated_beta_probit_corr=s6['calibrated_beta_probit_corr'],
            calibrated_lambda_rho=s6['calibrated_lambda_rho'],
            calibrated_nu_crps=s6['calibrated_nu_crps'],
        )

        # ── CALIBRATION DIAGNOSTICS
        diagnostics = cls._build_diagnostics(
            returns_train, vol_train, final_config,
            nu_opt, beta_opt, log_q_opt, q_opt, c_opt, phi_opt,
            gamma_opt, sens_opt, alpha_opt, risk_premium_opt,
            skew_kappa_opt, skew_persistence_fixed, mu_drift_opt,
            garch['garch_alpha'], garch['garch_beta'], garch['garch_leverage'],
            rough_hurst_est,
            jumps['jump_intensity'], jumps['jump_variance'],
            jumps['jump_sensitivity'], jumps['jump_mean'],
            config, degraded, cond_num, profile,
            s5g['rho_leverage'], s5g['kappa_mean_rev'],
            s5g['sigma_eta'], s5g['t_df_asym'],
            s5g['regime_switch_prob'], s5g['crps_sigma_shrinkage'],
            garch_kalman_w, q_vol_zeta,
            s5h['loc_bias_var_coeff'], s5h['loc_bias_drift_coeff'],
            asset_class)

        # ── TEST-PERIOD EVALUATION (centralised scoring)
        # Run filter_and_calibrate once here so tune.py can read
        # test_sigma, test_crps, test_hyvarinen directly from diagnostics
        # instead of calling filter_and_calibrate a second time.
        try:
            _pit_eval, _pitp_eval, _sig_eval, _crps_eval, _diag_eval = \
                cls.filter_and_calibrate(returns, vol, final_config, train_frac)
            _nu_eff_eval = _diag_eval.get('nu_effective', nu_opt)
            _mu_eff_eval = _diag_eval.get('mu_effective', mu_pred_train)
            _ret_test_eval = returns[n_train:]
            # Compute Hyvärinen score inline
            try:
                from tuning.diagnostics import (
                    compute_hyvarinen_score_student_t as _hyv_fn,
                    compute_crps_student_t_inline as _crps_fn,
                )
                _hyv_eval = float(_hyv_fn(
                    _ret_test_eval, _mu_eff_eval, _sig_eval, _nu_eff_eval))
            except Exception:
                _hyv_eval = float('-inf')
            diagnostics['test_sigma'] = _sig_eval
            diagnostics['test_crps'] = float(_crps_eval)
            diagnostics['test_hyvarinen'] = _hyv_eval
            diagnostics['test_nu_effective'] = float(_nu_eff_eval)
            diagnostics['test_mu_effective'] = _mu_eff_eval
            diagnostics['test_returns'] = _ret_test_eval
            diagnostics['test_pit_pvalue'] = float(_pitp_eval)
        except Exception:
            pass  # diagnostics won't have test_* keys — tune.py falls back

        return final_config, diagnostics

    # =========================================================================
    # FILTER AND CALIBRATE (orchestrator)
    # =========================================================================

    @classmethod
    def filter_and_calibrate(cls, returns, vol, config, train_frac=0.7):
        """
        Honest PIT + CRPS. All params from training config, no post-hoc adjustments.
        Delegates to _pit_garch_path, _pit_simple_path, _compute_berkowitz_pvalue,
        _compute_crps_output.
        Returns (pit_values, pit_pvalue, sigma_crps, crps, diagnostics).
        """
        from scipy.stats import kstest
        returns = np.asarray(returns).flatten()
        vol = np.asarray(vol).flatten()
        n = len(returns)
        n_train = int(n * train_frac)
        n_test = n - n_train

        mu_filt, P_filt, mu_pred, S_pred, ll = cls.filter_phi_unified(returns, vol, config)

        nu = config.nu_base
        variance_inflation = getattr(config, 'variance_inflation', 1.0)
        use_garch = getattr(config, 'garch_alpha', 0.0) > 0 or getattr(config, 'garch_beta', 0.0) > 0
        returns_test = returns[n_train:]
        mu_pred_test = mu_pred[n_train:]
        S_calibrated = S_pred[n_train:] * variance_inflation
        h_garch = None
        beta_final = variance_inflation
        _debug_gw_base = 0.0

        if use_garch:
            h_garch_full = cls._compute_garch_variance(returns - mu_pred, config, n_test_split=n_train)
            h_garch = h_garch_full[n_train:]
            pit_values, sigma, mu_effective, S_calibrated = cls._pit_garch_path(
                returns, mu_pred, S_pred, h_garch_full, config, n_train, n_test, nu)
            beta_final = 1.0
            _debug_gw_base = float(config.calibrated_gw)
        else:
            pit_values, sigma, mu_effective = cls._pit_simple_path(
                returns_test, mu_pred_test, S_calibrated, nu,
                float(getattr(config, 't_df_asym', 0.0)))

        pit_pvalue = float(kstest(pit_values, 'uniform').pvalue)
        hist, _ = np.histogram(pit_values, bins=10, range=(0, 1))
        mad = float(np.mean(np.abs(hist / n_test - 0.1)))
        berkowitz_pvalue = cls._compute_berkowitz_pvalue(pit_values)

        innovations = returns_test - mu_pred_test
        variance_ratio = float(np.var(innovations)) / (float(np.mean(S_calibrated)) + 1e-12)

        crps, sigma_crps, mu_crps, nu_crps = cls._compute_crps_output(
            returns_test, mu_pred_test, mu_effective, sigma, h_garch, config, nu, use_garch)

        _crps_shrink = float(np.clip(float(getattr(config, 'crps_sigma_shrinkage', 1.0)), 0.30, 1.0))

        diagnostics = {
            'pit_pvalue': pit_pvalue, 'berkowitz_pvalue': berkowitz_pvalue, 'mad': mad,
            'nu_effective': nu_crps, 'nu_pit': nu, 'variance_ratio': variance_ratio,
            'skewness': 0.0, 'crps': crps, 'log_likelihood': ll,
            'n_train': n_train, 'n_test': n_test,
            'gw_base': _debug_gw_base, 'gw_score': 0.0,
            'beta_final': beta_final, 'crps_shrink': _crps_shrink,
            'mu_effective': mu_effective,
            'garch_kalman_weight': float(getattr(config, 'garch_kalman_weight', 0.0)),
            'q_vol_coupling': float(getattr(config, 'q_vol_coupling', 0.0)),
            'loc_bias_var_coeff': float(getattr(config, 'loc_bias_var_coeff', 0.0)),
            'loc_bias_drift_coeff': float(getattr(config, 'loc_bias_drift_coeff', 0.0)),
        }
        return pit_values, pit_pvalue, sigma_crps, crps, diagnostics

    # =========================================================================
    # CALIBRATION HELPERS (private, called by filter_and_calibrate)
    # =========================================================================

    @classmethod
    def _pit_garch_path(
        cls,
        returns: np.ndarray,
        mu_pred: np.ndarray,
        S_pred: np.ndarray,
        h_garch_full: np.ndarray,
        config: 'UnifiedStudentTConfig',
        n_train: int,
        n_test: int,
        nu: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        PIT via GARCH-blended adaptive EWM with AR(1) probit whitening.

        Reads Stage 6 pre-calibrated params from config.
        Returns (pit_values, sigma, mu_effective, S_calibrated).
        """
        from scipy.stats import t as student_t_dist, norm as norm_dist

        returns_test = returns[n_train:]
        mu_pred_test = mu_pred[n_train:]
        S_pred_test = S_pred[n_train:]
        h_garch = h_garch_full[n_train:]

        _t_df_asym = float(getattr(config, 't_df_asym', 0.0))

        # Stage 6 pre-calibrated params
        _best_gw = float(config.calibrated_gw)
        _best_lam_mu = float(config.calibrated_lambda_rho)
        _best_lam_beta = _best_lam_mu
        _best_lam_rho = _best_lam_mu
        _beta_scale_corr = float(config.calibrated_beta_probit_corr)

        _cal_nu_pit = float(config.calibrated_nu_pit)
        if _cal_nu_pit > 0:
            nu = _cal_nu_pit

        # EWM warm-start from training data
        innovations_train = returns[:n_train] - mu_pred[:n_train]
        h_garch_train = h_garch_full[:n_train]
        S_pred_train = S_pred[:n_train]
        _S_bt = (1 - _best_gw) * S_pred_train + _best_gw * h_garch_train

        # Run EWM through first 60% of training
        _cal_start = int(n_train * 0.6)
        _ewm_mu_cal = float(np.mean(innovations_train))
        _ewm_num_cal = float(np.mean(innovations_train ** 2))
        _ewm_den_cal = float(np.mean(_S_bt))
        for _t in range(_cal_start):
            _ewm_mu_cal = _best_lam_mu * _ewm_mu_cal + (1 - _best_lam_mu) * innovations_train[_t]
            _ewm_num_cal = _best_lam_beta * _ewm_num_cal + (1 - _best_lam_beta) * (innovations_train[_t] ** 2)
            _ewm_den_cal = _best_lam_beta * _ewm_den_cal + (1 - _best_lam_beta) * _S_bt[_t]

        # Training probit PITs for whitening warm-start (last 40%)
        _n_cal = n_train - _cal_start
        _pit_train_cal = np.zeros(_n_cal)
        _ewm_mu_cv, _ewm_num_cv, _ewm_den_cv = _ewm_mu_cal, _ewm_num_cal, _ewm_den_cal
        for _t in range(_n_cal):
            _idx = _cal_start + _t
            _beta_cv = float(np.clip(_ewm_num_cv / (_ewm_den_cv + 1e-12) * _beta_scale_corr, 0.2, 5.0))
            _inn = innovations_train[_idx] - _ewm_mu_cv
            _S_cv = _S_bt[_idx] * _beta_cv
            _sig = np.sqrt(_S_cv * (nu - 2) / nu) if nu > 2 else np.sqrt(_S_cv)
            _sig = max(_sig, 1e-10)
            _pit_train_cal[_t] = student_t_dist.cdf(_inn / _sig, df=nu)
            _ewm_mu_cv = _best_lam_mu * _ewm_mu_cv + (1 - _best_lam_mu) * innovations_train[_idx]
            _ewm_num_cv = _best_lam_beta * _ewm_num_cv + (1 - _best_lam_beta) * (innovations_train[_idx] ** 2)
            _ewm_den_cv = _best_lam_beta * _ewm_den_cv + (1 - _best_lam_beta) * _S_bt[_idx]
        _pit_train_cal = np.clip(_pit_train_cal, 0.001, 0.999)
        _z_probit_cal = norm_dist.ppf(_pit_train_cal)
        _z_probit_cal = _z_probit_cal[np.isfinite(_z_probit_cal)]

        # Initialize EWM for test from training mean
        _ewm_mu_t = float(np.mean(innovations_train))
        _ewm_num_t = float(np.mean(innovations_train ** 2))
        _ewm_den_t = float(np.mean(_S_bt))

        # Blended test variance
        _S_blended_test = (1 - _best_gw) * S_pred_test + _best_gw * h_garch
        innovations_test = returns_test - mu_pred_test
        sq_inn_test = innovations_test ** 2

        # Test PIT with causal adaptive EWM
        pit_values = np.zeros(n_test)
        sigma = np.zeros(n_test)
        mu_effective = np.zeros(n_test)

        for _t in range(n_test):
            _beta_p = float(np.clip(_ewm_num_t / (_ewm_den_t + 1e-12) * _beta_scale_corr, 0.2, 5.0))
            _S_cal = _S_blended_test[_t] * _beta_p
            _inn = innovations_test[_t] - _ewm_mu_t
            mu_effective[_t] = mu_pred_test[_t] + _ewm_mu_t
            sigma[_t] = max(np.sqrt(_S_cal * (nu - 2) / nu) if nu > 2 else np.sqrt(_S_cal), 1e-10)

            _z = _inn / sigma[_t]
            if abs(_t_df_asym) > 0.05:
                _nu_side = max(2.5, nu - _t_df_asym) if _z < 0 else max(2.5, nu + _t_df_asym)
                pit_values[_t] = student_t_dist.cdf(_z, df=_nu_side)
            else:
                pit_values[_t] = student_t_dist.cdf(_z, df=nu)

            _ewm_mu_t = _best_lam_mu * _ewm_mu_t + (1 - _best_lam_mu) * innovations_test[_t]
            _ewm_num_t = _best_lam_beta * _ewm_num_t + (1 - _best_lam_beta) * sq_inn_test[_t]
            _ewm_den_t = _best_lam_beta * _ewm_den_t + (1 - _best_lam_beta) * _S_blended_test[_t]

        pit_values = np.clip(pit_values, 0.001, 0.999)

        # Causal AR(1) whitening in probit space
        if _best_lam_rho > 0:
            _z_probit = norm_dist.ppf(np.clip(pit_values, 0.0001, 0.9999))
            _z_white = np.zeros(n_test)
            _z_white[0] = _z_probit[0]

            _ewm_cross, _ewm_sq = 0.0, 1.0
            if len(_z_probit_cal) > 2:
                for _t in range(1, len(_z_probit_cal)):
                    _ewm_cross = _best_lam_rho * _ewm_cross + (1 - _best_lam_rho) * _z_probit_cal[_t - 1] * (_z_probit_cal[_t - 2] if _t > 1 else 0.0)
                    _ewm_sq = _best_lam_rho * _ewm_sq + (1 - _best_lam_rho) * _z_probit_cal[_t - 1] ** 2

            for _t in range(1, n_test):
                _ewm_cross = _best_lam_rho * _ewm_cross + (1 - _best_lam_rho) * _z_probit[_t - 1] * (_z_probit[_t - 2] if _t > 1 else (_z_probit_cal[-1] if len(_z_probit_cal) > 0 else 0.0))
                _ewm_sq = _best_lam_rho * _ewm_sq + (1 - _best_lam_rho) * _z_probit[_t - 1] ** 2
                _rho_t = float(np.clip(_ewm_cross / _ewm_sq, -0.3, 0.3)) if _ewm_sq > 0.1 else 0.0

                if abs(_rho_t) > 0.01:
                    _z_white[_t] = (_z_probit[_t] - _rho_t * _z_probit[_t - 1]) / np.sqrt(max(1 - _rho_t ** 2, 0.5))
                else:
                    _z_white[_t] = _z_probit[_t]

            pit_values = np.clip(norm_dist.cdf(_z_white), 0.001, 0.999)

        return pit_values, sigma, mu_effective, _S_blended_test

    @staticmethod
    def _pit_simple_path(returns_test, mu_pred_test, S_calibrated, nu, t_df_asym):
        """PIT via basic Student-t CDF (non-GARCH path). Returns (pit, sigma, mu_eff)."""
        from scipy.stats import t as student_t
        sigma = np.sqrt(S_calibrated * (nu - 2) / nu) if nu > 2 else np.sqrt(S_calibrated)
        sigma = np.maximum(sigma, 1e-10)
        innovations = returns_test - mu_pred_test
        z = innovations / sigma
        if abs(t_df_asym) > 0.05:
            pit_values = np.zeros(len(z))
            _nu_l, _nu_r = max(2.5, nu - t_df_asym), max(2.5, nu + t_df_asym)
            m = z < 0
            pit_values[m] = student_t.cdf(z[m], df=_nu_l)
            pit_values[~m] = student_t.cdf(z[~m], df=_nu_r)
        else:
            pit_values = student_t.cdf(z, df=nu)
        return np.clip(pit_values, 0.001, 0.999), sigma, mu_pred_test

    @staticmethod
    def _compute_berkowitz_pvalue(pit_values):
        """Berkowitz (2001) LR test: H0 Phi^-1(PIT)~N(0,1) iid vs H1 AR(1). Chi2(3)."""
        try:
            from scipy.stats import norm, chi2
            z = norm.ppf(np.clip(pit_values, 0.0001, 0.9999))
            z = z[np.isfinite(z)]
            n_z = len(z)
            if n_z <= 20:
                return float('nan')
            mu_hat = float(np.mean(z))
            var_hat = float(np.var(z, ddof=0))
            z_c = z - mu_hat
            denom = np.sum(z_c[:-1] ** 2)
            rho_hat = float(np.clip(np.sum(z_c[1:] * z_c[:-1]) / denom, -0.99, 0.99)) if denom > 1e-12 else 0.0
            ll_null = -0.5 * n_z * np.log(2 * np.pi) - 0.5 * np.sum(z ** 2)
            sigma_sq_cond = max(var_hat * (1 - rho_hat ** 2) if abs(rho_hat) < 0.99 else var_hat * 0.01, 1e-6)
            ll_alt = -0.5 * np.log(2 * np.pi * var_hat) - 0.5 * (z[0] - mu_hat) ** 2 / var_hat
            for t in range(1, n_z):
                resid = z[t] - (mu_hat + rho_hat * (z[t - 1] - mu_hat))
                ll_alt += -0.5 * np.log(2 * np.pi * sigma_sq_cond) - 0.5 * resid ** 2 / sigma_sq_cond
            return float(1 - chi2.cdf(max(2 * (ll_alt - ll_null), 0), df=3))
        except Exception:
            return float('nan')

    @classmethod
    def _compute_crps_output(cls, returns_test, mu_pred_test, mu_effective, sigma,
                             h_garch, config, nu, use_garch):
        """Sigma shrinkage + location bias + CRPS. Returns (crps, sigma_crps, mu_crps, nu_crps)."""
        _crps_shrink = float(np.clip(float(getattr(config, 'crps_sigma_shrinkage', 1.0)), 0.30, 1.0))
        sigma_crps = sigma * _crps_shrink
        _cal_nu = float(getattr(config, 'calibrated_nu_crps', 0.0))
        nu_crps = _cal_nu if _cal_nu > 0 else nu
        _loc_a = float(getattr(config, 'loc_bias_var_coeff', 0.0))
        _loc_b = float(getattr(config, 'loc_bias_drift_coeff', 0.0))
        garch_uncon = getattr(config, 'garch_unconditional_var', 1e-4)
        if use_garch and h_garch is not None and (abs(_loc_a) > 0.001 or abs(_loc_b) > 0.001):
            mu_crps = mu_pred_test.copy()
            for t in range(len(mu_pred_test)):
                mu_crps[t] += _loc_a * (h_garch[t] - garch_uncon) + _loc_b * np.sign(mu_pred_test[t]) * np.sqrt(abs(mu_pred_test[t]) + 1e-12)
        else:
            mu_crps = mu_effective
        try:
            from tuning.diagnostics import compute_crps_student_t_inline
            crps = compute_crps_student_t_inline(returns_test, mu_crps, sigma_crps, nu_crps)
        except Exception:
            crps = float('nan')
        return crps, sigma_crps, mu_crps, nu_crps

    # =========================================================================
    # PIT / KS CALIBRATION (called by tune.py for model scoring)
    # =========================================================================

    @classmethod
    def pit_ks_unified(
        cls,
        returns: np.ndarray,
        mu_pred: np.ndarray,
        S_pred: np.ndarray,
        config: 'UnifiedStudentTConfig',
    ) -> Tuple[float, float, Dict]:
        """
        PIT/KS calibration for unified model using predictive distribution.

        Applies variance_inflation to S_pred, computes Student-t CDF PIT values
        with smooth asymmetric nu, then returns KS statistic, p-value, and
        calibration metrics (histogram MAD, grade A/B/C/F).

        Called by tune.py to score unified models for BMA selection.
        """
        from scipy.stats import kstest

        returns = np.asarray(returns).flatten()
        mu_pred = np.asarray(mu_pred).flatten()
        S_pred = np.asarray(S_pred).flatten()

        n = len(returns)
        pit_values = np.empty(n)

        nu_base = config.nu_base
        alpha = config.alpha_asym
        k_asym = config.k_asym
        variance_inflation = getattr(config, 'variance_inflation', 1.0)

        for t in range(n):
            innovation = returns[t] - mu_pred[t]
            S_calibrated = S_pred[t] * variance_inflation
            scale = np.sqrt(max(S_calibrated, 1e-12))
            nu_eff = cls.compute_effective_nu(nu_base, innovation, scale, alpha, k_asym)
            if nu_eff > 2:
                t_scale = np.sqrt(S_calibrated * (nu_eff - 2) / nu_eff)
            else:
                t_scale = scale
            t_scale = max(t_scale, 1e-10)
            pit_values[t] = student_t.cdf(innovation, df=nu_eff, loc=0, scale=t_scale)

        valid = np.isfinite(pit_values)
        pit_clean = np.clip(pit_values[valid], 0, 1)

        if len(pit_clean) < 20:
            return 1.0, 0.0, {"n_samples": len(pit_clean), "calibrated": False}

        ks_result = kstest(pit_clean, 'uniform')

        hist, _ = np.histogram(pit_clean, bins=10, range=(0, 1))
        hist_freq = hist / len(pit_clean)
        hist_mad = float(np.mean(np.abs(hist_freq - 0.1)))

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

    # ---------------------------------------------------------------------------
    # SHARED GARCH VARIANCE
    # Used by filter_and_calibrate AND _stage_6_calibration_pipeline.
    # Eliminates the duplicated 30-line GARCH loop that previously
    # existed in both locations with identical logic.
    # ---------------------------------------------------------------------------

    @classmethod
    def _compute_garch_variance(cls, innovations, config, n_test_split=None):
        """
        Compute enhanced GJR-GARCH(1,1) variance with leverage correlation,
        vol-of-vol noise, Markov regime switching, mean reversion, and
        optional rough-volatility blending (Gatheral-Jaisson-Rosenbaum 2018).

        Runs CONTINUOUSLY from t=0 through full series to avoid cold-start
        bias at the train/test boundary (h_test[0] inherits warm state).

        Args:
            innovations: Full innovation series (returns - mu_pred), length n.
            config: UnifiedStudentTConfig with GARCH parameters.
            n_test_split: If provided, apply rough-vol blending only on
                          the test slice [n_test_split:]. None = no rough blend.

        Returns:
            h: GARCH variance array of length n (full series).
        """
        n = len(innovations)
        sq = innovations ** 2
        neg = (innovations < 0).astype(np.float64)

        go = float(getattr(config, 'garch_omega', 0.0))
        ga = float(getattr(config, 'garch_alpha', 0.0))
        gb = float(getattr(config, 'garch_beta', 0.0))
        gl = float(getattr(config, 'garch_leverage', 0.0))
        gu = float(getattr(config, 'garch_unconditional_var', 1e-4))
        rl = float(getattr(config, 'rho_leverage', 0.0))
        km = float(getattr(config, 'kappa_mean_rev', 0.0))
        tv = float(getattr(config, 'theta_long_var', 0.0))
        if tv <= 0:
            tv = gu
        se = float(getattr(config, 'sigma_eta', 0.0))
        rs = float(getattr(config, 'regime_switch_prob', 0.0))
        sm = np.sqrt(float(getattr(config, 'q_stress_ratio', 10.0)))

        h = np.zeros(n)
        h[0] = gu
        ps = 0.1  # Initial stress probability

        for t in range(1, n):
            ht = go + ga * sq[t-1] + gl * sq[t-1] * neg[t-1] + gb * h[t-1]
            if rl > 0.01 and h[t-1] > 1e-12:
                z = innovations[t-1] / np.sqrt(h[t-1])
                if z < 0:
                    ht += rl * z * z * h[t-1]
            if se > 0.005 and h[t-1] > 1e-12:
                z = abs(innovations[t-1]) / np.sqrt(h[t-1])
                ht += se * max(0.0, z - 1.5) ** 2 * h[t-1]
            if rs > 0.005 and h[t-1] > 1e-12:
                z = abs(innovations[t-1]) / np.sqrt(h[t-1])
                ps = (1.0 - rs) * ps + rs * (1.0 if z > 2.0 else 0.0)
                ps = min(max(ps, 0.0), 1.0)
                ht *= (1.0 + ps * (sm - 1.0))
            if km > 0.001:
                ht = (1.0 - km) * ht + km * tv
            h[t] = max(ht, 1e-12)

        # ── Rough volatility blending (Gatheral-Jaisson-Rosenbaum 2018) ──
        # Fractional differencing kernel (1-L)^d on ε²_t gives power-law
        # memory decay w_k ~ k^(d-1) vs GARCH's exponential w_k = β^k.
        rh = float(getattr(config, 'rough_hurst', 0.0))
        if 0.01 < rh < 0.5 and n_test_split is not None and n_test_split < n:
            n_test = n - n_test_split
            sq_test = sq[n_test_split:]
            d_frac = rh - 0.5  # d ∈ (-0.5, 0) for rough regime
            max_lag = min(50, n_test - 1)
            # Fractional differencing weights: w_0=1, w_k = w_{k-1}·(k-1-d)/k
            frac_w = np.zeros(max_lag + 1)
            frac_w[0] = 1.0
            for k in range(1, max_lag + 1):
                frac_w[k] = frac_w[k-1] * (k - 1 - d_frac) / k
            frac_w = np.abs(frac_w)
            ws = np.sum(frac_w)
            if ws > 1e-10:
                frac_w /= ws
            h_rough = np.zeros(n_test)
            h_rough[0] = gu
            for t in range(1, n_test):
                lookback = min(t, max_lag)
                wv = 0.0
                for lag in range(lookback):
                    wv += frac_w[lag] * sq_test[t - 1 - lag]
                h_rough[t] = max(wv, 1e-12)
            # Blend weight: H→0 ⟹ strong rough, H→0.5 ⟹ weak
            rw = max(0.3 * (1.0 - 2.0 * rh), 0.0)
            h[n_test_split:] = (1.0 - rw) * h[n_test_split:] + rw * h_rough

        return h

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Consolidated phi-Student-t Kalman filter (pure Python path).

        Returns (mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, log_likelihood).
        All four filter variants dispatch to this single loop:
          - filter_phi  (via Numba or this fallback)
          - filter_phi_augmented  (robust_wt=True + exogenous_input)
          - filter_phi_with_predictive  (thin wrapper)

        Args:
            returns, vol, q, c, phi, nu: standard Kalman params
            exogenous_input: optional u_t injected into state prediction
            robust_wt: if True use Student-t w_t = (nu+1)/(nu+z^2) weighting
        """
        n = len(returns)
        returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)

        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, 'item') else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, 'item') else float(c)
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
        has_exo = exogenous_input is not None

        for t in range(n):
            u_t = exogenous_input[t] if has_exo and t < len(exogenous_input) else 0.0
            mu_pred = phi_val * mu + u_t
            P_pred = phi_sq * P + q_val
            S = P_pred + R[t]
            if S <= 1e-12:
                S = 1e-12

            mu_pred_arr[t] = mu_pred
            S_pred_arr[t] = S

            innovation = returns[t] - mu_pred
            K = nu_adjust * P_pred / S

            if robust_wt:
                z_sq = (innovation ** 2) / S
                w_t = (nu_val + 1.0) / (nu_val + z_sq)
                mu = mu_pred + K * w_t * innovation
                P = (1.0 - w_t * K) * P_pred
            else:
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """phi-Student-t filter returning predictive mu_pred and S_pred for PIT."""
        return cls._filter_phi_core(returns, vol, q, c, phi, nu)

    @classmethod
    def filter_phi_unified(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        config: 'UnifiedStudentTConfig',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        UNIFIED φ-Student-t filter combining ALL enhancements.

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
                vol, q_calm, q_stress, config.ms_sensitivity,
                getattr(config, 'ms_ewm_lambda', 0.0)
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

        # Calibrated drift bias: E[y_t - mu_pred_t]
        # Estimated in optimize_params_unified Stage 5, included in prediction
        # to align filter output with filter_and_calibrate's mu_effective.
        _mu_drift_val = float(getattr(config, 'mu_drift', 0.0))

        log_norm_const = gammaln((nu_base + 1.0) / 2.0) - gammaln(nu_base / 2.0) - 0.5 * np.log(nu_base * np.pi)
        neg_exp = -((nu_base + 1.0) / 2.0)
        inv_nu = 1.0 / nu_base

        # ---------------------------------------------------------------------------
        # MERTON JUMP-DIFFUSION LAYER
        # ---------------------------------------------------------------------------
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

            # Risk premium: μ_pred = φ·μ + u_t + λ₁·R_t + drift
            mu_pred = phi_val * mu + u_t + risk_prem * R_base[t] + _mu_drift_val
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
            # ---------------------------------------------------------------------------
            # CONDITIONAL SKEW DYNAMICS: use dynamic α_t instead of static α
            # α_t evolves via GAS score, capturing regime-dependent asymmetry
            # ---------------------------------------------------------------------------
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

            # GAS skew: α_{t+1} = (1-ρ)·α₀ + ρ·α_t + κ·(z_t·w_t)
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

        # Causal EWM location correction: tracks innovation mean bias
        # Uses Stage 5f lambda, fallback 0.95 (~20-day half-life)
        _ewm_lambda = float(getattr(config, 'crps_ewm_lambda', 0.0))
        if _ewm_lambda < 0.01:
            _ewm_lambda = 0.95  # Default fallback: conservative tracking
        if n > 2:
            ewm_mu = 0.0
            mu_pred_corrected = mu_pred_arr.copy()
            for t in range(1, n):
                innov_prev = returns[t-1] - mu_pred_arr[t-1]
                ewm_mu = _ewm_lambda * ewm_mu + (1.0 - _ewm_lambda) * innov_prev
                mu_pred_corrected[t] = mu_pred_arr[t] + ewm_mu
            mu_pred_arr = mu_pred_corrected

        return mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, float(log_likelihood)


    # ---------------------------------------------------------------------------
    # OPTIMIZATION STAGE METHODS
    #
    # Each stage is a self-contained @classmethod with explicit
    # inputs and outputs. The orchestrator optimize_params_unified()
    # calls them in sequence, threading results forward.
    #
    # Stage dependency graph:
    #   1 (q,c,φ) → 2 (γ_vov) → 3 (ms_sens) → 4 (α_asym)
    #     → 4.1 (risk_premium) → 4.2 (skew_κ) → [hessian check]
    #       → 5 (ν CV) → 5c (GARCH) → 5d (jumps)
    #         → 5e (Hurst) → 5f (EWM λ) → 5g (leverage+shrinkage)
    # ---------------------------------------------------------------------------

    @classmethod
    def _stage_1_base_params(cls, returns_train, vol_train, n_train, nu_base, config):
        """
        Stage 1: Estimate base Kalman filter parameters (q, c, φ).

        Uses likelihood-based objective with state regularization to prevent
        the φ→1 / q→0 collapse (random walk degeneracy).

        Args:
            returns_train: Training returns array
            vol_train: Training volatility array
            n_train: Number of training observations
            nu_base: Degrees of freedom for Student-t
            config: Auto-configured UnifiedStudentTConfig

        Returns:
            dict with keys: q, c, phi, log_q, success
            On failure: success=False and defaults from config
        """
        from scipy.optimize import minimize

        def neg_ll_base(params):
            log_q, theta_c, phi = params
            q = 10 ** log_q
            c = np.exp(theta_c)
            c = float(np.clip(c, 0.01, 100.0))

            cfg = UnifiedStudentTConfig(
                q=q, c=c, phi=phi, nu_base=nu_base,
                alpha_asym=0.0, gamma_vov=0.0, q_stress_ratio=1.0,
            )

            try:
                _, _, _, _, ll = cls.filter_phi_unified(returns_train, vol_train, cfg)
            except Exception:
                return 1e10

            if not np.isfinite(ll):
                return 1e10

            phi_pen = max(0.0, abs(phi) - 0.95)
            q_pen = max(0.0, -7.0 - log_q)
            state_reg = 50.0 * (phi_pen ** 2 + q_pen ** 2)
            if phi_pen > 0 and q_pen > 0:
                state_reg += 30.0 * phi_pen * q_pen

            return -ll / n_train + state_reg

        log_q_init = np.log10(max(config.q_min * 10, 1e-7))
        c_safe = max(config.c, 0.01)
        theta_c_init = np.log(c_safe)
        x0 = [log_q_init, theta_c_init, 0.0]

        bounds = [
            (np.log10(config.q_min), -2),
            (-5.0, 5.0),
            (-0.99, 0.99),
        ]

        try:
            result = minimize(
                neg_ll_base, x0, bounds=bounds,
                method='L-BFGS-B', options={'maxiter': 200}
            )
            if result.x is not None and np.all(np.isfinite(result.x)):
                log_q, theta_c, phi = result.x
                q = 10 ** log_q
                c = float(np.clip(np.exp(theta_c), 0.01, 100.0))
                return {
                    'q': q, 'c': c, 'phi': float(phi),
                    'log_q': float(log_q), 'success': True,
                    'result': result,
                }
        except Exception:
            pass

        return {'q': config.q, 'c': config.c, 'phi': 0.0,
                'log_q': np.log10(config.q), 'success': False, 'result': None}

    @classmethod
    def _stage_2_vov_gamma(cls, returns_train, vol_train, n_train, nu_base,
                           q_opt, c_opt, phi_opt, config):
        """
        Stage 2: Estimate VoV gamma (volatility-of-volatility sensitivity).

        Freezes base parameters from Stage 1. Optimizes γ_vov which controls
        how much the observation noise R_t responds to changes in log-vol.

        Returns:
            float: Optimal gamma_vov
        """
        from scipy.optimize import minimize

        def neg_ll_vov(gamma_arr):
            gamma = gamma_arr[0]
            cfg = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_base,
                alpha_asym=0.0, gamma_vov=gamma, q_stress_ratio=1.0,
            )
            try:
                _, _, _, _, ll = cls.filter_phi_unified(returns_train, vol_train, cfg)
            except Exception:
                return 1e10
            if not np.isfinite(ll):
                return 1e10
            reg = 1.0 * (gamma - config.gamma_vov) ** 2
            return -ll / n_train + reg

        try:
            result = minimize(
                neg_ll_vov, [config.gamma_vov],
                bounds=[(0.0, 1.0)], method='L-BFGS-B'
            )
            return result.x[0] if result.success else config.gamma_vov
        except Exception:
            return config.gamma_vov

    @classmethod
    def _stage_3_ms_sensitivity(cls, returns_train, vol_train, n_train, nu_base,
                                q_opt, c_opt, phi_opt, gamma_opt, profile):
        """
        Stage 3: Estimate MS-q sensitivity (Markov-switching process noise).

        Controls how aggressively q transitions between calm and stress regimes.
        Profile-adaptive: metals use higher sensitivity and weaker regularization.

        Returns:
            float: Optimal ms_sensitivity
        """
        from scipy.optimize import minimize

        sens_init = profile.get('ms_sensitivity_init', 2.0)
        sens_center = profile.get('ms_sensitivity_reg_center', 2.0)
        sens_weight = profile.get('ms_sensitivity_reg_weight', 10.0)
        q_stress_ratio = profile.get('q_stress_ratio', 10.0)

        def neg_ll_msq(sens_arr):
            sens = sens_arr[0]
            cfg = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_base,
                alpha_asym=0.0, gamma_vov=gamma_opt,
                ms_sensitivity=sens, ms_ewm_lambda=0.0,
                q_stress_ratio=q_stress_ratio,
            )
            try:
                _, _, _, _, ll = cls.filter_phi_unified(returns_train, vol_train, cfg)
            except Exception:
                return 1e10
            if not np.isfinite(ll):
                return 1e10
            reg = sens_weight * (sens - sens_center) ** 2
            return -ll / n_train + reg

        try:
            result = minimize(
                neg_ll_msq, [sens_init],
                bounds=[(1.0, 5.0)], method='L-BFGS-B'
            )
            return result.x[0] if result.success else sens_init
        except Exception:
            return sens_init

    @classmethod
    def _stage_4_asymmetry(cls, returns_train, vol_train, n_train, nu_base,
                           q_opt, c_opt, phi_opt, gamma_opt, sens_opt, profile):
        """
        Stage 4: Estimate asymmetry alpha (tail thickness direction).

        Controls how ν varies with the sign of standardized innovations:
          ν_eff = ν_base × (1 + α × tanh(k × z))
        α > 0: left tail heavier (crash sensitivity)
        α < 0: right tail heavier

        Returns:
            float: Optimal alpha_asym
        """
        from scipy.optimize import minimize

        k_asym = profile.get('k_asym', 1.0)
        alpha_init = profile.get('alpha_asym_init', 0.0)
        q_stress_ratio = profile.get('q_stress_ratio', 10.0)

        def neg_ll_asym(alpha_arr):
            alpha = alpha_arr[0]
            cfg = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_base,
                alpha_asym=alpha, k_asym=k_asym, gamma_vov=gamma_opt,
                ms_sensitivity=sens_opt, ms_ewm_lambda=0.0,
                q_stress_ratio=q_stress_ratio,
            )
            try:
                _, _, _, _, ll = cls.filter_phi_unified(returns_train, vol_train, cfg)
            except Exception:
                return 1e10
            if not np.isfinite(ll):
                return 1e10
            reg = 1.0 * (alpha - alpha_init) ** 2
            return -ll / n_train + reg

        try:
            result = minimize(
                neg_ll_asym, [alpha_init],
                bounds=[(-0.3, 0.3)], method='L-BFGS-B'
            )
            return result.x[0] if result.success else alpha_init
        except Exception:
            return alpha_init

    @classmethod
    def _stage_4_1_risk_premium(cls, returns_train, vol_train, n_train, nu_base,
                                q_opt, c_opt, phi_opt, gamma_opt, sens_opt,
                                alpha_opt, profile):
        """
        Stage 4.1: Estimate conditional risk premium λ₁ (Merton ICAPM).

        E[r_t | F_{t-1}] = φ·μ_{t-1} + λ₁·σ²_t
        λ₁ > 0: higher variance → higher expected return (risk compensation)
        λ₁ < 0: higher variance → lower expected return (leverage/fear)

        Returns:
            float: Optimal risk_premium_sensitivity
        """
        from scipy.optimize import minimize

        rp_reg = profile.get('risk_premium_reg_penalty', 0.5)
        rp_init = profile.get('risk_premium_init', 0.0)
        k_asym = profile.get('k_asym', 1.0)
        q_stress_ratio = profile.get('q_stress_ratio', 10.0)

        def neg_ll_rp(rp_arr):
            rp = rp_arr[0]
            cfg = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_base,
                alpha_asym=alpha_opt, k_asym=k_asym, gamma_vov=gamma_opt,
                ms_sensitivity=sens_opt, ms_ewm_lambda=0.0,
                q_stress_ratio=q_stress_ratio,
                risk_premium_sensitivity=rp,
            )
            try:
                _, _, _, _, ll = cls.filter_phi_unified(returns_train, vol_train, cfg)
            except Exception:
                return 1e10
            if not np.isfinite(ll):
                return 1e10
            reg = rp_reg * rp ** 2
            return -ll / n_train + reg

        try:
            result = minimize(
                neg_ll_rp, [rp_init],
                bounds=[(-5.0, 10.0)], method='L-BFGS-B',
                options={'maxiter': 100}
            )
            if result.x is not None and np.isfinite(result.x[0]):
                return float(result.x[0])
        except Exception:
            pass
        return rp_init

    @classmethod
    def _stage_4_2_skew_dynamics(cls, returns_train, vol_train, n_train, nu_base,
                                 q_opt, c_opt, phi_opt, gamma_opt, sens_opt,
                                 alpha_opt, risk_premium_opt, profile):
        """
        Stage 4.2: Estimate conditional skew dynamics κ_λ (GAS framework).

        α_{t+1} = (1 - ρ_λ)·α₀ + ρ_λ·α_t + κ_λ·s_t
        where s_t = z_t·w_t is the Student-t score for skewness.
        ρ_λ fixed at 0.97 (robust daily default, ~33 day half-life).

        Returns:
            tuple: (skew_kappa_opt, skew_persistence_fixed)
        """
        from scipy.optimize import minimize

        skew_persistence_fixed = 0.97
        k_asym = profile.get('k_asym', 1.0)
        q_stress_ratio = profile.get('q_stress_ratio', 10.0)

        def neg_ll_skew(kappa_arr):
            kappa_val = kappa_arr[0]
            cfg = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_base,
                alpha_asym=alpha_opt, k_asym=k_asym, gamma_vov=gamma_opt,
                ms_sensitivity=sens_opt, ms_ewm_lambda=0.0,
                q_stress_ratio=q_stress_ratio,
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
            reg = 5.0 * kappa_val ** 2
            return -ll / n_train + reg

        try:
            result = minimize(
                neg_ll_skew, [0.0],
                bounds=[(0.0, 0.10)], method='L-BFGS-B',
                options={'maxiter': 100}
            )
            if result.x is not None and np.isfinite(result.x[0]):
                return float(result.x[0]), skew_persistence_fixed
        except Exception:
            pass
        return 0.0, skew_persistence_fixed

    @staticmethod
    def _check_hessian_degradation(result_base, gamma_opt, sens_opt, alpha_opt,
                                   risk_premium_opt, skew_kappa_opt):
        """
        Check Hessian condition number for ill-conditioning.

        If condition number > 1e6, gracefully disable advanced features
        to prevent unstable parameter estimates propagating downstream.

        Returns:
            dict with keys: gamma_opt, sens_opt, alpha_opt,
            risk_premium_opt, skew_kappa_opt, degraded, cond_num
        """
        try:
            if result_base is not None and hasattr(result_base, 'hess_inv'):
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
            gamma_opt = 0.0
            sens_opt = 2.0
            alpha_opt = 0.0
            risk_premium_opt = 0.0
            skew_kappa_opt = 0.0
            degraded = True

        return {
            'gamma_opt': gamma_opt, 'sens_opt': sens_opt,
            'alpha_opt': alpha_opt, 'risk_premium_opt': risk_premium_opt,
            'skew_kappa_opt': skew_kappa_opt,
            'degraded': degraded, 'cond_num': cond_num,
        }

    @classmethod
    def _stage_5_nu_cv_selection(cls, returns_train, vol_train, n_train,
                                 q_opt, c_opt, phi_opt, alpha_opt, gamma_opt,
                                 sens_opt, risk_premium_opt, skew_kappa_opt,
                                 skew_persistence_fixed, nu_base, profile,
                                 use_heavy_tail_grid):
        """
        Stage 5: Rolling cross-validation for ν (degrees of freedom) selection.

        Implements the Gneiting-Raftery (2007) criterion:
          "Maximize sharpness of predictive distributions,
           subject to calibration."

        Uses 5-fold rolling CV within training data. For each candidate ν:
          1. Run filter on full training data
          2. For each fold: estimate β on previous folds, validate on current
          3. Compute fold-level KS p-value and CRPS
          4. Select ν with best calibration-adjusted sharpness score

        Returns:
            dict with keys: nu_opt, beta_opt, mu_drift_opt, innovations_train,
            mu_pred_train, S_pred_train
        """
        from scipy.stats import t as student_t, kstest
        from scipy.special import gammaln

        k_asym = profile.get('k_asym', 1.0)
        q_stress_ratio = profile.get('q_stress_ratio', 10.0)
        vov_damping = profile.get('vov_damping', 0.3)
        ms_ewm_lambda = profile.get('ms_ewm_lambda', 0.0)

        NU_GRID = [3, 4, 5, 6, 7, 8, 10, 12] if use_heavy_tail_grid else [5, 6, 7, 8, 10, 12, 15, 20]

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
                    alpha_asym=alpha_opt, k_asym=k_asym, gamma_vov=gamma_opt,
                    ms_sensitivity=sens_opt, ms_ewm_lambda=0.0,
                    q_stress_ratio=q_stress_ratio, vov_damping=vov_damping,
                    variance_inflation=1.0, risk_premium_sensitivity=risk_premium_opt,
                    skew_score_sensitivity=skew_kappa_opt,
                    skew_persistence=skew_persistence_fixed,
                )

                _, _, mu_pred_train, S_pred_train, _ = cls.filter_phi_unified(
                    returns_train, vol_train, temp_config
                )

                innovations_train = returns_train - mu_pred_train

                fold_ks_pvalues = []
                fold_betas = []
                fold_mu_drifts = []

                for fold_idx in range(1, n_folds):
                    train_end_idx = fold_idx * fold_size
                    valid_start_idx = train_end_idx
                    valid_end_idx = min((fold_idx + 1) * fold_size, n_train)

                    if valid_end_idx <= valid_start_idx:
                        continue

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

                        # CRPS on validation fold (Gneiting-Raftery sharpness)
                        try:
                            fold_sigma = np.sqrt(np.array([S_valid[t] * fold_beta for t in range(len(innov_valid))]))
                            fold_sigma = np.maximum(fold_sigma, 1e-10)
                            if test_nu > 2:
                                fold_scale = fold_sigma * np.sqrt((test_nu - 2) / test_nu)
                            else:
                                fold_scale = fold_sigma
                            _z_fold = innov_valid / fold_scale
                            _pdf_fold = student_t.pdf(_z_fold, df=test_nu)
                            _cdf_fold = student_t.cdf(_z_fold, df=test_nu)
                            if test_nu > 1:
                                _lgB1 = gammaln(0.5) + gammaln(test_nu - 0.5) - gammaln(test_nu)
                                _lgB2 = gammaln(0.5) + gammaln(test_nu / 2) - gammaln((test_nu + 1) / 2)
                                _Br = np.exp(_lgB1 - 2 * _lgB2)
                                _t1 = _z_fold * (2 * _cdf_fold - 1)
                                _t2 = 2 * _pdf_fold * (test_nu + _z_fold**2) / (test_nu - 1)
                                _t3 = 2 * np.sqrt(test_nu) * _Br / (test_nu - 1)
                                fold_crps = float(np.mean(fold_scale * (_t1 + _t2 - _t3)))
                            else:
                                fold_crps = float(np.mean(np.abs(innov_valid)))
                            fold_ks_pvalues[-1] = (ks_p, fold_crps)
                        except Exception:
                            fold_ks_pvalues[-1] = (ks_p, float('inf'))

                if len(fold_ks_pvalues) > 0:
                    fold_ks_vals = [x[0] if isinstance(x, tuple) else x for x in fold_ks_pvalues]
                    fold_crps_vals = [x[1] if isinstance(x, tuple) else float('inf') for x in fold_ks_pvalues]
                    avg_ks_p = float(np.mean(fold_ks_vals))
                    avg_crps = float(np.mean([c for c in fold_crps_vals if np.isfinite(c)])) if any(np.isfinite(c) for c in fold_crps_vals) else float('inf')
                    avg_beta = float(np.mean(fold_betas))
                    avg_mu_drift = float(np.mean(fold_mu_drifts))

                    # Kurtosis coherence bonus for heavy-tail grid
                    if use_heavy_tail_grid and test_nu > 4:
                        emp_kurt = float(np.mean(innovations_train ** 4) / (np.mean(innovations_train ** 2) ** 2 + 1e-20))
                        theo_kurt = 3.0 * (test_nu - 2.0) / (test_nu - 4.0)
                        kurt_mismatch = abs(emp_kurt - theo_kurt) / (emp_kurt + 1e-8)
                        kurt_bonus = max(0.5, 1.0 - 0.5 * kurt_mismatch)
                        score = avg_ks_p * kurt_bonus
                    else:
                        score = avg_ks_p

                    # CRPS sharpness bonus (Gneiting-Raftery)
                    if avg_ks_p > 0.05 and np.isfinite(avg_crps) and avg_crps > 0:
                        innov_std = float(np.std(innovations_train)) + 1e-10
                        crps_ratio = avg_crps / innov_std
                        crps_bonus = max(0.6, min(1.5, 1.5 - 0.8 * crps_ratio))
                        score = score * crps_bonus

                    if score > best_avg_ks_p:
                        best_avg_ks_p = score
                        best_nu = float(test_nu)
                        best_global_beta = avg_beta
                        best_global_mu_drift = avg_mu_drift

            except Exception:
                continue

        # Final calibration on full training data with best nu
        temp_config = UnifiedStudentTConfig(
            q=q_opt, c=c_opt, phi=phi_opt, nu_base=best_nu,
            alpha_asym=alpha_opt, k_asym=k_asym, gamma_vov=gamma_opt,
            ms_sensitivity=sens_opt, ms_ewm_lambda=0.0,
            q_stress_ratio=q_stress_ratio, vov_damping=vov_damping,
            variance_inflation=1.0, risk_premium_sensitivity=risk_premium_opt,
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

        nu_opt = best_nu
        beta_opt = 0.7 * final_beta + 0.3 * best_global_beta
        mu_drift_opt = 0.7 * final_mu_drift + 0.3 * best_global_mu_drift
        beta_opt = float(np.clip(beta_opt, 0.2, 5.0))

        return {
            'nu_opt': nu_opt, 'beta_opt': beta_opt, 'mu_drift_opt': mu_drift_opt,
            'innovations_train': innovations_train,
            'mu_pred_train': mu_pred_train, 'S_pred_train': S_pred_train,
        }

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

    @classmethod
    def _stage_5d_jump_diffusion(cls, returns_train, vol_train, mu_pred_train,
                                 mu_drift_opt, n_train, q_opt, c_opt, phi_opt,
                                 nu_opt, alpha_opt, gamma_opt, sens_opt,
                                 beta_opt, risk_premium_opt, skew_kappa_opt,
                                 skew_persistence_fixed, profile):
        """
        Stage 5d: Merton jump-diffusion estimation.

        Separates discrete jump events from continuous diffusion:
          1. Detect jumps: |z_t| > threshold
          2. Estimate jump_intensity, jump_variance, jump_mean
          3. Optimize jump_sensitivity via 1D MLE
          4. BIC test: only enable if 2·ΔLL > 4·ln(n)

        Returns:
            dict with keys: jump_intensity, jump_variance,
            jump_sensitivity, jump_mean
        """
        from scipy.optimize import minimize

        defaults = {'jump_intensity': 0.0, 'jump_variance': 0.0,
                    'jump_sensitivity': 1.0, 'jump_mean': 0.0}

        if n_train <= 100:
            return defaults

        k_asym = profile.get('k_asym', 1.0)
        q_stress_ratio = profile.get('q_stress_ratio', 10.0)
        vov_damping = profile.get('vov_damping', 0.3)
        jump_threshold = profile.get('jump_threshold', 3.0)

        try:
            innovations = returns_train - mu_pred_train - mu_drift_opt
            innov_std = float(np.std(innovations))

            if innov_std <= 1e-10:
                return defaults

            z_innov = innovations / innov_std
            jump_mask = np.abs(z_innov) > jump_threshold
            n_jumps = int(np.sum(jump_mask))

            if n_jumps < 5:
                return defaults

            jump_intensity = float(np.clip(n_jumps / n_train, 0.005, 0.15))
            var_jump = float(np.var(innovations[jump_mask]))
            var_diffusion = float(np.var(innovations[~jump_mask]))
            jump_variance = float(np.clip(var_jump - var_diffusion, 1e-8, 0.1))
            jump_mean = float(np.clip(np.mean(innovations[jump_mask]), -0.05, 0.05))

            # Optimize jump_sensitivity
            def neg_ll_jump_sens(sens_arr):
                sens_val = sens_arr[0]
                cfg = UnifiedStudentTConfig(
                    q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_opt,
                    alpha_asym=alpha_opt, k_asym=k_asym, gamma_vov=gamma_opt,
                    ms_sensitivity=sens_opt, ms_ewm_lambda=0.0,
                    q_stress_ratio=q_stress_ratio, vov_damping=vov_damping,
                    variance_inflation=beta_opt, mu_drift=mu_drift_opt,
                    risk_premium_sensitivity=risk_premium_opt,
                    skew_score_sensitivity=skew_kappa_opt,
                    skew_persistence=skew_persistence_fixed,
                    jump_intensity=jump_intensity,
                    jump_variance=jump_variance,
                    jump_sensitivity=sens_val,
                    jump_mean=jump_mean,
                )
                try:
                    _, _, _, _, ll = cls.filter_phi_unified(returns_train, vol_train, cfg)
                except Exception:
                    return 1e10
                if not np.isfinite(ll):
                    return 1e10
                reg = 0.5 * (sens_val - 1.0) ** 2
                return -ll / n_train + reg

            jump_sensitivity = 1.0
            try:
                result = minimize(
                    neg_ll_jump_sens, [1.0],
                    bounds=[(0.0, 5.0)], method='L-BFGS-B',
                    options={'maxiter': 50}
                )
                if result.x is not None and np.isfinite(result.x[0]):
                    jump_sensitivity = float(result.x[0])
            except Exception:
                pass

            # BIC verification
            cfg_no = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_opt,
                alpha_asym=alpha_opt, k_asym=k_asym, gamma_vov=gamma_opt,
                ms_sensitivity=sens_opt, ms_ewm_lambda=0.0,
                q_stress_ratio=q_stress_ratio, vov_damping=vov_damping,
                variance_inflation=beta_opt, mu_drift=mu_drift_opt,
                risk_premium_sensitivity=risk_premium_opt,
                skew_score_sensitivity=skew_kappa_opt,
                skew_persistence=skew_persistence_fixed,
            )
            cfg_yes = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_opt,
                alpha_asym=alpha_opt, k_asym=k_asym, gamma_vov=gamma_opt,
                ms_sensitivity=sens_opt, ms_ewm_lambda=0.0,
                q_stress_ratio=q_stress_ratio, vov_damping=vov_damping,
                variance_inflation=beta_opt, mu_drift=mu_drift_opt,
                risk_premium_sensitivity=risk_premium_opt,
                skew_score_sensitivity=skew_kappa_opt,
                skew_persistence=skew_persistence_fixed,
                jump_intensity=jump_intensity, jump_variance=jump_variance,
                jump_sensitivity=jump_sensitivity, jump_mean=jump_mean,
            )

            _, _, _, _, ll_no = cls.filter_phi_unified(returns_train, vol_train, cfg_no)
            _, _, _, _, ll_yes = cls.filter_phi_unified(returns_train, vol_train, cfg_yes)

            bic_penalty = 4 * np.log(n_train)
            if 2 * (ll_yes - ll_no) < bic_penalty:
                return defaults

            return {'jump_intensity': jump_intensity, 'jump_variance': jump_variance,
                    'jump_sensitivity': jump_sensitivity, 'jump_mean': jump_mean}

        except Exception:
            return defaults

    @staticmethod
    def _stage_5e_rough_hurst(returns_train, mu_pred_train, mu_drift_opt, n_train):
        """
        Stage 5e: Rough volatility Hurst exponent estimation.

        Uses the variogram method on log|innovations| (Gatheral-Jaisson-Rosenbaum 2018):
          m(τ) = E[|log|ε_{t+τ}| - log|ε_t||²] ~ C·τ^{2H}
        OLS on log-log gives H = slope/2.

        Equity vol H ≈ 0.05-0.15 (rough). H < 0.5 → long memory.

        Returns:
            float: Hurst exponent H, or 0.0 if not in rough regime
        """
        if n_train <= 200:
            return 0.0

        try:
            innov = returns_train - mu_pred_train - mu_drift_opt
            abs_innov = np.maximum(np.abs(innov), 1e-12)
            log_abs = np.log(abs_innov)

            max_tau = min(30, n_train // 10)
            lags = np.arange(1, max_tau + 1)
            variogram = np.zeros(max_tau)

            for i, tau in enumerate(lags):
                diffs = log_abs[tau:] - log_abs[:-tau]
                variogram[i] = np.mean(diffs ** 2)

            valid = variogram > 1e-12
            if np.sum(valid) < 5:
                return 0.0

            log_tau = np.log(lags[valid])
            log_var = np.log(variogram[valid])

            short_mask = lags[valid] <= 15
            if np.sum(short_mask) < 3:
                return 0.0

            lt = log_tau[short_mask]
            lv = log_var[short_mask]
            mx, my = np.mean(lt), np.mean(lv)
            cov_xy = np.sum((lt - mx) * (lv - my))
            var_x = np.sum((lt - mx) ** 2)

            if var_x > 1e-12:
                H = cov_xy / var_x / 2.0
                if 0.01 <= H <= 0.45:
                    return float(H)

        except Exception:
            pass
        return 0.0

    @classmethod
    def _stage_5f_ewm_lambda(cls, returns_train, vol_train, n_train,
                             q_opt, c_opt, phi_opt, nu_opt, alpha_opt, gamma_opt,
                             sens_opt, beta_opt, mu_drift_opt, risk_premium_opt,
                             skew_kappa_opt, skew_persistence_fixed, profile):
        """
        Stage 5f: CRPS-optimal EWM location correction (Durbin-Koopman 2012).

        When Kalman innovations have positive autocorrelation (ρ₁ > 0),
        an EWM correction reduces innovation variance:
          ewm_μ[t] = λ·ewm_μ[t-1] + (1-λ)·ε_{t-1}
          μ_corrected[t] = μ_pred[t] + ewm_μ[t]

        Optimal λ selected via CRPS on validation portion of training data.

        Returns:
            float: Optimal EWM lambda, or 0.0 if not beneficial
        """
        if n_train <= 200:
            return 0.0

        k_asym = profile.get('k_asym', 1.0)
        q_stress_ratio = profile.get('q_stress_ratio', 10.0)
        vov_damping = profile.get('vov_damping', 0.3)
        ms_ewm_lambda = profile.get('ms_ewm_lambda', 0.0)

        try:
            from tuning.diagnostics import compute_crps_student_t_inline as _crps_inline

            temp_config = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_opt,
                alpha_asym=alpha_opt, k_asym=k_asym, gamma_vov=gamma_opt,
                ms_sensitivity=sens_opt, ms_ewm_lambda=ms_ewm_lambda,
                q_stress_ratio=q_stress_ratio, vov_damping=vov_damping,
                variance_inflation=beta_opt, mu_drift=mu_drift_opt,
                risk_premium_sensitivity=risk_premium_opt,
                skew_score_sensitivity=skew_kappa_opt,
                skew_persistence=skew_persistence_fixed,
                crps_ewm_lambda=0.0,
            )
            _, _, mu_pred_base, S_pred_base, _ = cls.filter_phi_unified(
                returns_train, vol_train, temp_config
            )

            # Check autocorrelation
            innov = returns_train - mu_pred_base
            ic = innov - np.mean(innov)
            denom = np.sum(ic ** 2)
            rho1 = float(np.sum(ic[1:] * ic[:-1]) / denom) if denom > 1e-12 else 0.0

            if rho1 <= -0.05:
                return 0.0

            n_est = int(n_train * 0.6)
            n_val = n_train - n_est
            if n_val <= 50:
                return 0.0

            S_val = S_pred_base[n_est:] * beta_opt
            if nu_opt > 2:
                sig_val = np.sqrt(np.maximum(S_val, 1e-20) * (nu_opt - 2) / nu_opt)
            else:
                sig_val = np.sqrt(np.maximum(S_val, 1e-20))
            sig_val = np.maximum(sig_val, 1e-10)

            ret_val = returns_train[n_est:]
            mu_val_base = mu_pred_base[n_est:]
            crps_baseline = _crps_inline(ret_val, mu_val_base, sig_val, nu_opt)

            LAMBDA_GRID = [0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97]
            best_crps = crps_baseline
            best_lam = 0.0

            for lam_c in LAMBDA_GRID:
                ewm_mu = 0.0
                mu_corr = mu_pred_base.copy()
                for t in range(1, n_train):
                    ewm_mu = lam_c * ewm_mu + (1.0 - lam_c) * (returns_train[t-1] - mu_pred_base[t-1])
                    mu_corr[t] = mu_pred_base[t] + ewm_mu
                crps_c = _crps_inline(ret_val, mu_corr[n_est:], sig_val, nu_opt)
                if np.isfinite(crps_c) and crps_c < best_crps:
                    best_crps = crps_c
                    best_lam = lam_c

            if best_lam > 0 and best_crps < crps_baseline * 0.997:
                return best_lam

        except Exception:
            pass
        return 0.0

    @classmethod
    def _stage_5g_leverage_and_shrinkage(cls, returns_train, vol_train, n_train,
                                         q_opt, c_opt, phi_opt, nu_opt, alpha_opt,
                                         gamma_opt, sens_opt, beta_opt, mu_drift_opt,
                                         risk_premium_opt, skew_kappa_opt,
                                         skew_persistence_fixed, crps_ewm_lambda_opt,
                                         garch_omega, garch_alpha, garch_beta,
                                         garch_leverage, unconditional_var, profile):
        """
        Stage 5g: Leverage correlation + mean reversion + CRPS shrinkage.

        Joint estimation of Heston-DLSV inspired parameters via sequential
        CRPS minimization on cross-validated training data:
          Phase 1: Grid search (ρ_leverage, κ_mean_rev)
          Phase 2: sigma_eta (vol-of-vol noise)
          Phase 3: regime_switch_prob
          Phase 4: t_df_asym (asymmetric ν offset)
          Phase 5: CRPS-optimal sigma shrinkage (Gneiting-Raftery 2007)

        Returns:
            dict with keys: rho_leverage, kappa_mean_rev, theta_long_var,
            crps_sigma_shrinkage, sigma_eta, t_df_asym, regime_switch_prob
        """
        defaults = {
            'rho_leverage': 0.0, 'kappa_mean_rev': 0.0,
            'theta_long_var': unconditional_var, 'crps_sigma_shrinkage': 1.0,
            'sigma_eta': 0.0, 't_df_asym': 0.0, 'regime_switch_prob': 0.0,
        }

        if n_train <= 200:
            return defaults

        k_asym = profile.get('k_asym', 1.0)
        q_stress_ratio = profile.get('q_stress_ratio', 10.0)
        vov_damping = profile.get('vov_damping', 0.3)
        ms_ewm_lambda = profile.get('ms_ewm_lambda', 0.0)

        try:
            from tuning.diagnostics import compute_crps_student_t_inline as _crps_inline

            temp_config = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_opt,
                alpha_asym=alpha_opt, k_asym=k_asym, gamma_vov=gamma_opt,
                ms_sensitivity=sens_opt, ms_ewm_lambda=ms_ewm_lambda,
                q_stress_ratio=q_stress_ratio, vov_damping=vov_damping,
                variance_inflation=beta_opt, mu_drift=mu_drift_opt,
                risk_premium_sensitivity=risk_premium_opt,
                skew_score_sensitivity=skew_kappa_opt,
                skew_persistence=skew_persistence_fixed,
                crps_ewm_lambda=crps_ewm_lambda_opt,
            )
            _, _, mu_pred, S_pred, _ = cls.filter_phi_unified(
                returns_train, vol_train, temp_config
            )
            innovations = returns_train - mu_pred
            sq_inn = innovations ** 2
            neg_ind = (innovations < 0).astype(np.float64)

            n_est = int(n_train * 0.6)
            n_val = n_train - n_est

            if n_val <= 50:
                return defaults

            returns_val = returns_train[n_est:]
            mu_val = mu_pred[n_est:]

            # Helper: build enhanced GARCH variance
            def _build_garch(rho_c, kap_c, eta_c=0.0, reg_c=0.0):
                h = np.zeros(n_train)
                h[0] = unconditional_var
                _p_st = 0.1
                _sm = np.sqrt(q_stress_ratio)
                for t_ in range(1, n_train):
                    h_ = (garch_omega
                          + garch_alpha * sq_inn[t_-1]
                          + garch_leverage * sq_inn[t_-1] * neg_ind[t_-1]
                          + garch_beta * h[t_-1])
                    if rho_c > 0.01 and h[t_-1] > 1e-12:
                        nz_ = innovations[t_-1] / np.sqrt(h[t_-1])
                        if nz_ < 0:
                            h_ += rho_c * nz_ * nz_ * h[t_-1]
                    if eta_c > 0.005 and h[t_-1] > 1e-12:
                        za_ = abs(innovations[t_-1]) / np.sqrt(h[t_-1])
                        ex_ = max(0.0, za_ - 1.5)
                        h_ += eta_c * ex_ * ex_ * h[t_-1]
                    if reg_c > 0.005 and h[t_-1] > 1e-12:
                        zr_ = abs(innovations[t_-1]) / np.sqrt(h[t_-1])
                        _p_st = (1.0 - reg_c) * _p_st + reg_c * (1.0 if zr_ > 2.0 else 0.0)
                        _p_st = min(max(_p_st, 0.0), 1.0)
                        h_ = h_ * (1.0 + _p_st * (_sm - 1.0))
                    if kap_c > 0.001:
                        h_ = (1.0 - kap_c) * h_ + kap_c * unconditional_var
                    h[t_] = max(h_, 1e-12)
                return h

            # Helper: compute CRPS from GARCH variance
            def _crps_from_h(h_arr, nu_c=nu_opt, df_asym=0.0):
                h_v = h_arr[n_est:]
                if nu_c > 2:
                    sig_v = np.sqrt(np.maximum(h_v * beta_opt, 1e-20) * (nu_c - 2) / nu_c)
                else:
                    sig_v = np.sqrt(np.maximum(h_v * beta_opt, 1e-20))
                sig_v = np.maximum(sig_v, 1e-10)
                if abs(df_asym) > 0.05:
                    z_val = (returns_val - mu_val) / sig_v
                    from scipy.stats import t as _td
                    from scipy.special import gammaln as _gl
                    nu_L = max(2.5, nu_c - df_asym)
                    nu_R = max(2.5, nu_c + df_asym)
                    crps_ind = np.zeros(len(z_val))
                    for side_nu, mask in [(nu_L, z_val < 0), (nu_R, z_val >= 0)]:
                        if not np.any(mask):
                            continue
                        zs = z_val[mask]
                        ps = _td.pdf(zs, df=side_nu)
                        cs = _td.cdf(zs, df=side_nu)
                        if side_nu > 1:
                            lB1 = _gl(0.5) + _gl(side_nu - 0.5) - _gl(side_nu)
                            lB2 = _gl(0.5) + _gl(side_nu / 2) - _gl((side_nu + 1) / 2)
                            Br = np.exp(lB1 - 2 * lB2)
                            crps_ind[mask] = sig_v[mask] * (zs*(2*cs-1) + 2*ps*(side_nu+zs**2)/(side_nu-1) - 2*np.sqrt(side_nu)*Br/(side_nu-1))
                        else:
                            crps_ind[mask] = sig_v[mask] * np.abs(zs)
                    vm = np.isfinite(crps_ind)
                    return float(np.mean(crps_ind[vm])) if np.any(vm) else float('inf')
                return _crps_inline(returns_val, mu_val, sig_v, nu_c)

            # Phase 1: (ρ_lev, κ) grid
            h_base = _build_garch(0.0, 0.0)
            crps_baseline = _crps_from_h(h_base)
            rho_opt = 0.0
            kap_opt = 0.0
            best_crps = crps_baseline

            for rho_c in [0.0, 0.1, 0.3, 0.5, 0.8, 1.2]:
                for kap_c in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
                    if rho_c == 0.0 and kap_c == 0.0:
                        continue
                    c = _crps_from_h(_build_garch(rho_c, kap_c))
                    if np.isfinite(c) and c < best_crps:
                        best_crps = c
                        rho_opt = rho_c
                        kap_opt = kap_c

            if best_crps >= crps_baseline * 0.995:
                rho_opt = 0.0
                kap_opt = 0.0

            # Phase 2: sigma_eta
            sigma_eta = 0.0
            best_eta = best_crps if (rho_opt > 0 or kap_opt > 0) else crps_baseline
            for eta_c in [0.03, 0.06, 0.10, 0.15, 0.25]:
                c = _crps_from_h(_build_garch(rho_opt, kap_opt, eta_c))
                if np.isfinite(c) and c < best_eta:
                    best_eta = c
                    sigma_eta = eta_c
            if best_eta >= (best_crps if (rho_opt > 0 or kap_opt > 0) else crps_baseline) * 0.995:
                sigma_eta = 0.0

            # Phase 3: regime_switch_prob
            regime_prob = 0.0
            best_reg = best_eta if sigma_eta > 0 else (best_crps if (rho_opt > 0 or kap_opt > 0) else crps_baseline)
            for reg_c in [0.02, 0.04, 0.06, 0.08, 0.12]:
                c = _crps_from_h(_build_garch(rho_opt, kap_opt, sigma_eta, reg_c))
                if np.isfinite(c) and c < best_reg:
                    best_reg = c
                    regime_prob = reg_c
            ref_crps = best_eta if sigma_eta > 0 else (best_crps if (rho_opt > 0 or kap_opt > 0) else crps_baseline)
            if best_reg >= ref_crps * 0.995:
                regime_prob = 0.0

            # Phase 4: t_df_asym
            df_asym = 0.0
            h_best = _build_garch(rho_opt, kap_opt, sigma_eta, regime_prob)
            best_dfa = _crps_from_h(h_best)
            for dfa_c in [0.5, 1.0, 1.5, 2.0, 2.5]:
                c = _crps_from_h(h_best, df_asym=dfa_c)
                if np.isfinite(c) and c < best_dfa:
                    best_dfa = c
                    df_asym = dfa_c
            if best_dfa >= _crps_from_h(h_best) * 0.995:
                df_asym = 0.0

            # Phase 5: CRPS sigma shrinkage
            h_val = h_best[n_est:]
            best_shrink_crps = float('inf')
            best_shrink = 1.0

            for s in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
                if nu_opt > 2:
                    sig = np.sqrt(np.maximum(h_val * beta_opt, 1e-20) * (nu_opt - 2) / nu_opt) * s
                else:
                    sig = np.sqrt(np.maximum(h_val * beta_opt, 1e-20)) * s
                sig = np.maximum(sig, 1e-10)
                c = _crps_inline(returns_val, mu_val, sig, nu_opt)
                if np.isfinite(c) and c < best_shrink_crps:
                    best_shrink_crps = c
                    best_shrink = s

            # Fine refinement
            lo = max(best_shrink - 0.05, 0.30)
            hi = min(best_shrink + 0.05, 1.00)
            fine = lo
            while fine <= hi + 0.001:
                if nu_opt > 2:
                    sig = np.sqrt(np.maximum(h_val * beta_opt, 1e-20) * (nu_opt - 2) / nu_opt) * fine
                else:
                    sig = np.sqrt(np.maximum(h_val * beta_opt, 1e-20)) * fine
                sig = np.maximum(sig, 1e-10)
                c = _crps_inline(returns_val, mu_val, sig, nu_opt)
                if np.isfinite(c) and c < best_shrink_crps:
                    best_shrink_crps = c
                    best_shrink = fine
                fine += 0.01

            return {
                'rho_leverage': rho_opt,
                'kappa_mean_rev': kap_opt,
                'theta_long_var': unconditional_var,
                'crps_sigma_shrinkage': best_shrink,
                'sigma_eta': sigma_eta,
                't_df_asym': df_asym,
                'regime_switch_prob': regime_prob,
            }

        except Exception:
            return defaults

    @classmethod
    def _stage_5h_conditional_location_bias(
            cls, returns_train, vol_train, n_train,
            q_opt, c_opt, phi_opt, nu_opt, alpha_opt, gamma_opt,
            sens_opt, beta_opt, mu_drift_opt, risk_premium_opt,
            skew_kappa_opt, skew_persistence_fixed, crps_ewm_lambda_opt,
            garch_omega, garch_alpha, garch_beta, garch_leverage,
            unconditional_var, garch_kalman_w, q_vol_zeta, profile):
        """
        Stage 5h: Conditional location bias correction (Ghysels-Santa-Clara-Valkanov 2005).

        Corrects nonlinear mean dynamics that the linear Kalman state misses:
          μ_corrected = μ_pred + a·(h_t - θ_long) + b·sign(μ_pred)·√|μ_pred|

        Term 1: Variance-state conditional bias — captures concavity in the
                risk-return tradeoff (high variance → mean undershoots).
        Term 2: Drift magnitude shrinkage — a soft James-Stein shrinker
                that prevents extreme drift forecasts from overshooting.

        Estimated via L2-regularized regression on the training validation
        fold.  Both |a| < 0.5 and |b| < 0.5 (professor's constraint).
        Gated: only enable if CRPS improves > 0.3%.

        Returns:
            dict with keys: loc_bias_var_coeff, loc_bias_drift_coeff
        """
        defaults = {'loc_bias_var_coeff': 0.0, 'loc_bias_drift_coeff': 0.0}

        if n_train <= 200 or garch_alpha <= 0:
            return defaults

        k_asym = profile.get('k_asym', 1.0)
        q_stress_ratio = profile.get('q_stress_ratio', 10.0)
        vov_damping = profile.get('vov_damping', 0.3)
        ms_ewm_lambda = profile.get('ms_ewm_lambda', 0.0)

        try:
            from tuning.diagnostics import compute_crps_student_t_inline as _crps_fn

            # Run filter WITHOUT location bias
            cfg_base = UnifiedStudentTConfig(
                q=q_opt, c=c_opt, phi=phi_opt, nu_base=nu_opt,
                alpha_asym=alpha_opt, k_asym=k_asym, gamma_vov=gamma_opt,
                ms_sensitivity=sens_opt, ms_ewm_lambda=ms_ewm_lambda,
                q_stress_ratio=q_stress_ratio, vov_damping=vov_damping,
                variance_inflation=beta_opt, mu_drift=mu_drift_opt,
                risk_premium_sensitivity=risk_premium_opt,
                skew_score_sensitivity=skew_kappa_opt,
                skew_persistence=skew_persistence_fixed,
                crps_ewm_lambda=crps_ewm_lambda_opt,
                garch_omega=garch_omega, garch_alpha=garch_alpha,
                garch_beta=garch_beta, garch_leverage=garch_leverage,
                garch_unconditional_var=unconditional_var,
                theta_long_var=unconditional_var,
                garch_kalman_weight=garch_kalman_w,
                q_vol_coupling=q_vol_zeta,
            )

            # Call filter_and_calibrate ONCE to get the EXACT sigma pipeline.
            # This is the definitive reference — no simplified proxy needed.
            # We treat its sigma as fixed and only search over mu correction.
            _, _, sigma_crps_fc, _, calib_diag_fc = cls.filter_and_calibrate(
                returns_train, vol_train, cfg_base, train_frac=0.6)

            # filter_and_calibrate with train_frac=0.6 uses n_est=60%
            # and evaluates on the latter 40% of returns_train.
            n_est = int(n_train * 0.6)
            n_val = n_train - n_est
            if n_val < 50:
                return defaults

            r_val = returns_train[n_est:]
            nu_fc = calib_diag_fc.get('nu_effective', nu_opt)
            sigma_val = np.maximum(sigma_crps_fc, 1e-10)

            # Also need mu_pred from filter (for the same data)
            _, _, mu_pred, S_pred, _ = cls.filter_phi_unified(
                returns_train, vol_train, cfg_base)
            mu_val = mu_pred[n_est:n_train]

            # Build GARCH h_t for location correction features
            innovations = returns_train - mu_pred
            h_garch = np.zeros(n_train)
            h_garch[0] = unconditional_var
            for t in range(1, n_train):
                h_t = (garch_omega
                       + garch_alpha * innovations[t-1]**2
                       + garch_leverage * innovations[t-1]**2 * (1.0 if innovations[t-1] < 0 else 0.0)
                       + garch_beta * h_garch[t-1])
                h_garch[t] = max(h_t, 1e-12)

            h_val = h_garch[n_est:n_train]
            theta_long = unconditional_var

            # Baseline CRPS (no location correction) using FC's sigma
            crps_baseline = _crps_fn(r_val, mu_val, sigma_val, nu_fc)
            best_a = 0.0
            best_b = 0.0
            best_crps = crps_baseline

            # Feature vectors (causal)
            feat_var = h_val - theta_long
            feat_drift = np.sign(mu_val) * np.sqrt(np.abs(mu_val) + 1e-12)

            for a_c in [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]:
                for b_c in [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]:
                    if a_c == 0.0 and b_c == 0.0:
                        continue
                    mu_corr = mu_val + a_c * feat_var + b_c * feat_drift
                    crps_c = _crps_fn(r_val, mu_corr, sigma_val, nu_fc)
                    crps_c += 0.001 * (a_c * a_c + b_c * b_c)
                    if np.isfinite(crps_c) and crps_c < best_crps:
                        best_crps = crps_c
                        best_a = a_c
                        best_b = b_c

            if best_crps >= crps_baseline * 0.9999:
                return defaults

            return {
                'loc_bias_var_coeff': float(np.clip(best_a, -0.5, 0.5)),
                'loc_bias_drift_coeff': float(np.clip(best_b, -0.5, 0.5)),
            }

        except Exception:
            return defaults

    @classmethod
    def _stage_6_calibration_pipeline(cls, returns, vol, config, train_frac=0.7):
        """Stage 6: Pre-calibrate filter_and_calibrate pipeline params (gw, nu_pit, nu_crps, beta_corr, lam_rho)."""
        from scipy.stats import t as _s6t, kstest as _s6ks, norm as _s6n
        from scipy.special import gammaln as _s6gl
        D = {'calibrated_gw': 0.50, 'calibrated_nu_pit': 0.0, 'calibrated_nu_crps': 0.0,
             'calibrated_beta_probit_corr': 1.0, 'calibrated_lambda_rho': 0.985}
        try:
            ret = np.asarray(returns).flatten(); vl = np.asarray(vol).flatten()
            n = len(ret); nt = int(n * train_frac)
            ug = getattr(config, 'garch_alpha', 0.0) > 0 or getattr(config, 'garch_beta', 0.0) > 0
            if not ug or nt < 150:
                return D
            _, _, mp, sp, _ = cls.filter_phi_unified(ret, vl, config)
            inn = ret - mp
            hf = cls._compute_garch_variance(inn, config)
            ht = hf[:nt]; it = inn[:nt]; st = sp[:nt]
            def _bk(pa):
                try: _, kp = _s6ks(pa, 'uniform')
                except Exception: return 0.0, 0.0, 0.0
                if len(pa) < 20: return float(kp), float(kp), 0.0
                zp = _s6n.ppf(np.clip(pa, 0.001, 0.999)); zp = zp[np.isfinite(zp)]
                if len(zp) < 20: return float(kp), float(kp), 0.0
                m_ = float(np.mean(zp)); v_ = float(np.var(zp, ddof=0)); zc = zp - m_; dn = np.sum(zc ** 2)
                r1 = float(np.sum(zc[1:] * zc[:-1]) / dn) if dn > 1e-12 else 0.0
                return float(kp * np.exp(-5 * m_ ** 2) * np.exp(-5 * (v_ - 1) ** 2) * np.exp(-10 * r1 ** 2)), float(kp), r1
            im = getattr(config, 'ms_ewm_lambda', 0.0) > 0.01
            hv = float(np.std(it)) > 0.03
            Sr = 0.5 * st + 0.5 * ht; rz = it / np.sqrt(np.maximum(Sr, 1e-12))
            ek = float(np.mean(rz ** 4) / (np.mean(rz ** 2) ** 2 + 1e-20))
            NA = [5, 6, 7, 8, 10, 12] if im else [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
            kl = []
            for nc in NA:
                if nc > 4: mm = abs(ek - 3 * (nc - 2) / (nc - 4))
                elif nc == 4: mm = max(0, 6 - ek) * 0.5
                elif nc == 3: mm = max(0, 8 - ek) * 0.4
                else: mm = float('inf')
                kl.append((mm, nc))
            kl.sort(); NC = [v for _, v in kl[:3 if im else 8]]
            if im: GW = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
            elif hv: GW = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.0]
            else: GW = [0.0, 0.15, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]
            s5f = float(getattr(config, 'crps_ewm_lambda', 0.0))
            lm = float(np.clip(max(s5f, 0.975), 0.975, 0.995)) if s5f >= 0.50 else 0.985
            nf = 2 if im else (1 if nt < 200 else (2 if nt < 400 else 3))
            fs = nt // (nf + 1)
            bs = -1.0; bg = 0.50 if not im else 0.80; bn = NC[0]
            for gw in GW:
                Sb = (1 - gw) * st + gw * ht
                for nu in NC:
                    fc = []
                    for fi in range(nf):
                        ee = (fi + 1) * fs; ve = min((fi + 2) * fs, nt)
                        if ve <= ee: continue
                        nv = ve - ee
                        em = float(np.mean(it[:ee])); en = float(np.mean(it[:ee] ** 2)); ed = float(np.mean(Sb[:ee]))
                        for t in range(ee):
                            em = lm * em + (1 - lm) * it[t]; en = lm * en + (1 - lm) * (it[t] ** 2); ed = lm * ed + (1 - lm) * Sb[t]
                        pv = np.zeros(nv); zv = np.zeros(nv); sv = np.zeros(nv)
                        m_, nn_, dd_ = em, en, ed
                        for tv in range(nv):
                            ix = ee + tv; bv = float(np.clip(nn_ / (dd_ + 1e-12), 0.2, 5.0))
                            iv = it[ix] - m_; Sv = Sb[ix] * bv
                            s = np.sqrt(Sv * (nu - 2) / nu) if nu > 2 else np.sqrt(Sv); s = max(s, 1e-10)
                            zv[tv] = iv / s; sv[tv] = s; pv[tv] = _s6t.cdf(zv[tv], df=nu)
                            m_ = lm * m_ + (1 - lm) * it[ix]; nn_ = lm * nn_ + (1 - lm) * (it[ix] ** 2); dd_ = lm * dd_ + (1 - lm) * Sb[ix]
                        pv = np.clip(pv, 0.001, 0.999)
                        try:
                            sc, _, _ = _bk(pv)
                            hi_, _ = np.histogram(pv, bins=10, range=(0, 1))
                            md = float(np.mean(np.abs(hi_ / nv - 0.1))); mp_ = max(0, 1 - md / 0.05)
                            pdf = _s6t.pdf(zv, df=nu); cdf = np.clip(_s6t.cdf(zv, df=nu), 0.001, 0.999)
                            if nu > 1:
                                l1 = _s6gl(0.5) + _s6gl(nu - 0.5) - _s6gl(nu)
                                l2 = _s6gl(0.5) + _s6gl(nu / 2) - _s6gl((nu + 1) / 2)
                                Br = np.exp(l1 - 2 * l2)
                                cr = float(np.mean(sv * (zv * (2 * cdf - 1) + 2 * pdf * (nu + zv ** 2) / (nu - 1) - 2 * np.sqrt(nu) * Br / (nu - 1))))
                            else:
                                cr = float(np.mean(np.abs(sv * zv)))
                            ist = float(np.std(it[:ee]))
                            cf = float(np.clip(np.exp(-4 * max(0, cr / ist - 0.45)), 0.30, 1.30)) if ist > 1e-10 and np.isfinite(cr) and cr > 0 else 1.0
                            fc.append(sc * mp_ * cf)
                        except Exception:
                            pass
                    if fc:
                        av = float(np.mean(fc))
                        if av > bs: bs = av; bg = gw; bn = nu
            # Local gw refinement
            if nt >= 300:
                for gwf in np.arange(max(bg - 0.06, 0), min(bg + 0.07, 1.01), 0.03):
                    gwf = float(gwf)
                    if abs(gwf - bg) < 0.005: continue
                    Sb = (1 - gwf) * st + gwf * ht; fc = []
                    for fi in range(nf):
                        ee = (fi + 1) * fs; ve = min((fi + 2) * fs, nt)
                        if ve <= ee: continue
                        nv = ve - ee
                        em = float(np.mean(it[:ee])); en = float(np.mean(it[:ee] ** 2)); ed = float(np.mean(Sb[:ee]))
                        for t in range(ee):
                            em = lm * em + (1 - lm) * it[t]; en = lm * en + (1 - lm) * (it[t] ** 2); ed = lm * ed + (1 - lm) * Sb[t]
                        pf = np.zeros(nv); ss = 0.0; m_, nn_, dd_ = em, en, ed
                        for tv in range(nv):
                            ix = ee + tv; bv = float(np.clip(nn_ / (dd_ + 1e-12), 0.2, 5.0))
                            iv = it[ix] - m_; Sv = Sb[ix] * bv
                            s = np.sqrt(Sv * max(bn - 2, 0.1) / bn) if bn > 2 else np.sqrt(Sv); s = max(s, 1e-10)
                            ss += s; pf[tv] = _s6t.cdf(iv / s, df=bn)
                            m_ = lm * m_ + (1 - lm) * it[ix]; nn_ = lm * nn_ + (1 - lm) * (it[ix] ** 2); dd_ = lm * dd_ + (1 - lm) * Sb[ix]
                        pf = np.clip(pf, 0.001, 0.999)
                        try:
                            sc, _, _ = _bk(pf); hi_, _ = np.histogram(pf, bins=10, range=(0, 1))
                            md = float(np.mean(np.abs(hi_ / nv - 0.1))); mp_ = max(0, 1 - md / 0.05)
                            is2 = float(np.std(it[:ee])); ms2 = ss / max(nv, 1)
                            sh = float(np.clip((is2 / ms2) ** 1.5, 0.55, 1.45)) if is2 > 1e-10 and ms2 > 1e-10 else 1.0
                            fc.append(sc * mp_ * sh)
                        except Exception:
                            pass
                    if fc:
                        av = float(np.mean(fc))
                        if av > bs: bs = av; bg = gwf
            np_ = float(bn)
            # Probit-variance beta correction
            Sf = (1 - bg) * st + bg * ht
            emc = float(np.mean(it)); enc = float(np.mean(it ** 2)); edc = float(np.mean(Sf))
            cs = int(nt * 0.6); ncl = nt - cs
            for t in range(cs):
                emc = lm * emc + (1 - lm) * it[t]; enc = lm * enc + (1 - lm) * (it[t] ** 2); edc = lm * edc + (1 - lm) * Sf[t]
            pcl = np.zeros(ncl); m_, nn_, dd_ = emc, enc, edc
            for tv in range(ncl):
                ix = cs + tv; bv = float(np.clip(nn_ / (dd_ + 1e-12), 0.2, 5.0))
                iv = it[ix] - m_; Sv = Sf[ix] * bv
                s = np.sqrt(Sv * (np_ - 2) / np_) if np_ > 2 else np.sqrt(Sv); s = max(s, 1e-10)
                pcl[tv] = _s6t.cdf(iv / s, df=np_)
                m_ = lm * m_ + (1 - lm) * it[ix]; nn_ = lm * nn_ + (1 - lm) * (it[ix] ** 2); dd_ = lm * dd_ + (1 - lm) * Sf[ix]
            pcl = np.clip(pcl, 0.001, 0.999)
            zpr = _s6n.ppf(pcl); zpr = zpr[np.isfinite(zpr)]
            bc = float(np.clip(np.var(zpr, ddof=0), 0.40, 2.50)) if len(zpr) > 30 else 1.0
            # AR(1) whitening lambda_rho
            bl = 0.98; bw = -1.0
            if len(zpr) > 50:
                for lr in [0.97, 0.98, 0.99]:
                    zw = np.zeros(len(zpr)); zw[0] = zpr[0]; ec = 0.0; es = 1.0
                    for tw in range(1, len(zpr)):
                        ec = lr * ec + (1 - lr) * zpr[tw - 1] * (zpr[tw - 2] if tw > 1 else 0.0)
                        es = lr * es + (1 - lr) * zpr[tw - 1] ** 2
                        rho = float(np.clip(ec / es, -0.3, 0.3)) if tw >= 20 and es > 0.1 else 0.0
                        zw[tw] = (zpr[tw] - rho * zpr[tw - 1]) / np.sqrt(max(1 - rho ** 2, 0.5)) if abs(rho) > 0.01 else zpr[tw]
                    pw = np.clip(_s6n.cdf(zw), 0.001, 0.999)
                    try:
                        ws, _, _ = _bk(pw)
                        if ws > bw: bw = ws; bl = lr
                    except Exception:
                        pass
            # CRPS-optimal nu
            nc_ = np_
            if nt > 200:
                try:
                    from tuning.diagnostics import compute_crps_student_t_inline as _cf
                    vs = int(nt * 0.6); nvl = nt - vs
                    if nvl > 50:
                        rv = ret[:nt][vs:]; mv = mp[:nt][vs:]; Sb = (1 - bg) * st + bg * ht
                        emn = float(np.mean(it)); enn = float(np.mean(it ** 2)); edn = float(np.mean(Sb))
                        for t in range(vs):
                            emn = lm * emn + (1 - lm) * it[t]; enn = lm * enn + (1 - lm) * (it[t] ** 2); edn = lm * edn + (1 - lm) * Sb[t]
                        bc_ = float('inf')
                        for nuc in [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
                            snv = np.zeros(nvl); emu, enu, edu = emn, enn, edn
                            for tv in range(nvl):
                                ix = vs + tv; bv = float(np.clip(enu / (edu + 1e-12), 0.2, 5.0))
                                bv = float(np.clip(bv * bc, 0.2, 5.0))
                                Sv = Sb[ix] * bv
                                s = np.sqrt(Sv * (nuc - 2) / nuc) if nuc > 2 else np.sqrt(Sv)
                                snv[tv] = max(s, 1e-10)
                                emu = lm * emu + (1 - lm) * it[ix]; enu = lm * enu + (1 - lm) * (it[ix] ** 2); edu = lm * edu + (1 - lm) * Sb[ix]
                            shrk = float(np.clip(getattr(config, 'crps_sigma_shrinkage', 1.0), 0.30, 1.0))
                            snv = np.maximum(snv * shrk, 1e-10)
                            c = _cf(rv, mv, snv, float(nuc))
                            if np.isfinite(c) and c < bc_: bc_ = c; nc_ = float(nuc)
                except Exception:
                    nc_ = np_
            return {'calibrated_gw': float(bg), 'calibrated_nu_pit': float(np_),
                    'calibrated_nu_crps': float(nc_), 'calibrated_beta_probit_corr': float(bc),
                    'calibrated_lambda_rho': float(bl)}
        except Exception:
            return D

    @classmethod
    def _build_diagnostics(cls, returns_train, vol_train, final_config,
                           nu_opt, beta_opt, log_q_opt, q_opt, c_opt, phi_opt,
                           gamma_opt, sens_opt, alpha_opt, risk_premium_opt,
                           skew_kappa_opt, skew_persistence_fixed, mu_drift_opt,
                           garch_alpha, garch_beta, garch_leverage,
                           rough_hurst_est,
                           jump_intensity_est, jump_variance_est,
                           jump_sensitivity_est, jump_mean_est,
                           config, degraded, cond_num, profile,
                           rho_leverage_opt, kappa_mean_rev_opt,
                           sigma_eta_opt, t_df_asym_opt,
                           regime_switch_prob_opt, crps_sigma_shrinkage_opt,
                           garch_kalman_w_opt=0.0, q_vol_coupling_opt=0.0,
                           loc_bias_var_opt=0.0, loc_bias_drift_opt=0.0,
                           asset_class=None):
        """
        Build calibration diagnostics dict from final optimized config.

        Computes PIT metrics on training data to diagnose calibration quality.

        Returns:
            dict: Full diagnostics dictionary
        """
        from scipy.stats import t as student_t

        try:
            _, _, mu_pred_final, S_pred_final, _ = cls.filter_phi_unified(
                returns_train, vol_train, final_config
            )
            innov_final = returns_train - mu_pred_final
            S_cal_final = S_pred_final * beta_opt

            if nu_opt > 2:
                sigma_final = np.sqrt(S_cal_final * (nu_opt - 2) / nu_opt)
            else:
                sigma_final = np.sqrt(S_cal_final)
            sigma_final = np.maximum(sigma_final, 1e-10)

            z_final = innov_final / sigma_final
            pit_final = student_t.cdf(z_final, df=nu_opt)
            pit_final = np.clip(pit_final, 0.001, 0.999)

            pit_centered = pit_final - np.mean(pit_final)
            denom = np.sum(pit_centered ** 2)
            pit_rho1 = float(np.sum(pit_centered[1:] * pit_centered[:-1]) / denom) if denom > 1e-12 else 0.0
            pit_left_tail = float(np.mean(pit_final < 0.05))
            pit_right_tail = float(np.mean(pit_final > 0.95))
            actual_var_final = float(np.var(innov_final))
            pred_var_final = float(np.mean(S_cal_final))
            var_ratio = actual_var_final / (pred_var_final + 1e-12)

        except Exception:
            pit_rho1 = 0.0
            pit_left_tail = 0.05
            pit_right_tail = 0.05
            var_ratio = 1.0

        vov_damping = profile.get('vov_damping', 0.3)
        k_asym = profile.get('k_asym', 1.0)
        q_stress_ratio = profile.get('q_stress_ratio', 10.0)
        ms_ewm_lambda = profile.get('ms_ewm_lambda', 0.0)

        return {
            "stage": 5, "success": True, "degraded": degraded,
            "hessian_cond": cond_num,
            "q": float(q_opt), "c": float(c_opt), "phi": float(phi_opt),
            "log10_q": float(log_q_opt),
            "gamma_vov": float(gamma_opt), "ms_sensitivity": float(sens_opt),
            "alpha_asym": float(alpha_opt),
            "risk_premium_sensitivity": float(risk_premium_opt),
            "skew_score_sensitivity": float(skew_kappa_opt),
            "skew_persistence": float(skew_persistence_fixed),
            "variance_inflation": float(beta_opt), "mu_drift": float(mu_drift_opt),
            "garch_alpha": float(garch_alpha), "garch_beta": float(garch_beta),
            "garch_leverage": float(garch_leverage),
            "rough_hurst": float(rough_hurst_est),
            "c_bounds": (float(config.c_min), float(config.c_max)),
            "q_min": float(config.q_min),
            "jump_intensity": float(jump_intensity_est),
            "jump_variance": float(jump_variance_est),
            "jump_sensitivity": float(jump_sensitivity_est),
            "jump_mean": float(jump_mean_est),
            "jump_enabled": jump_intensity_est > 1e-6 and jump_variance_est > 1e-12,
            "pit_rho1": float(pit_rho1),
            "pit_left_tail": float(pit_left_tail),
            "pit_right_tail": float(pit_right_tail),
            "var_ratio": float(var_ratio),
            "asset_class": asset_class,
            "profile_applied": bool(profile),
            "profile_vov_damping": vov_damping,
            "profile_k_asym": k_asym,
            "profile_q_stress_ratio": q_stress_ratio,
            "profile_ms_ewm_lambda": ms_ewm_lambda,
            "rho_leverage": float(rho_leverage_opt),
            "kappa_mean_rev": float(kappa_mean_rev_opt),
            "sigma_eta": float(sigma_eta_opt),
            "t_df_asym": float(t_df_asym_opt),
            "regime_switch_prob": float(regime_switch_prob_opt),
            "crps_sigma_shrinkage": float(crps_sigma_shrinkage_opt),
            "garch_kalman_weight": float(garch_kalman_w_opt),
            "q_vol_coupling": float(q_vol_coupling_opt),
            "loc_bias_var_coeff": float(loc_bias_var_opt),
            "loc_bias_drift_coeff": float(loc_bias_drift_opt),
        }
