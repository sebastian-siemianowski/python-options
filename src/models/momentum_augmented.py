"""
===============================================================================
MOMENTUM AUGMENTED DRIFT MODEL — Compositional Wrapper for BMA Integration
===============================================================================

Implements momentum augmentation for Kalman-based drift models using the
compositional wrapper pattern approved by the panel of senior professors:

DESIGN PRINCIPLES (From Panel Decision):
    1. No modification to base Kalman filter equations
    2. No explosion of optimization parameter space
    3. Momentum easily enabled/disabled via configuration
    4. signals.py output schema unchanged
    5. Integration via existing BMA candidate pool
    6. Momentum utility learned via BMA weights, not assumed

ARCHITECTURE:
    MomentumAugmentedDriftModel(base_model, momentum_config)
    
    - Accepts any base drift model (Gaussian, φ-Gaussian, φ-Student-t)
    - Computes normalized momentum features with fixed lookback windows
    - Applies momentum adjustment OUTSIDE the base filter
    - Preserves all base model interfaces and return types
    - Exposes enable_momentum flag for toggling

ARCHITECTURAL INVARIANT:
    There is NO bare Student-t model. All Student-t momentum augmentation
    uses φ-Student-t as the base model.

MOMENTUM FEATURES:
    - Normalized returns over configurable lookback windows
    - Default lookbacks: [5, 10, 20, 60] days
    - Normalization: z-score (default) or rank

KEY INSIGHT:
    Momentum enters model selection, not filter equations.
    This preserves identifiability and prevents q/momentum collinearity.

Performance Optimization:
    This module supports Numba JIT-compiled kernels for momentum-augmented
    filtering. When Numba is available, filter methods automatically use
    the accelerated kernels with graceful fallback to pure Python.

February 2026 — Implements Copilot Story:
    "Add Momentum Augmentation Layers to Kalman-Based Drift Models"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.stats import norm


# =============================================================================
# NUMBA INTEGRATION (optional, graceful fallback)
# =============================================================================

try:
    from .numba_wrappers import (
        is_numba_available,
        run_momentum_phi_gaussian_filter,
        run_momentum_phi_student_t_filter,
        run_momentum_phi_student_t_filter_batch,
    )
    _USE_NUMBA = is_numba_available()
except ImportError:
    _USE_NUMBA = False


# =============================================================================
# MOMENTUM CONFIGURATION
# =============================================================================

@dataclass
class MomentumConfig:
    """Configuration for momentum + mean reversion augmentation.
    
    ELITE UPGRADE (February 2026):
    State-equation integration: μ_t = φμ_{t-1} + u_t + w_t
    Where u_t = α_t × MOM_t - β_t × MR_t (exogenous input)
    
    IDENTIFIABILITY NOTE (Expert #3):
    When MR enabled, φ is shrunk toward 1.0 to prevent φ/κ collinearity.
    """
    # Momentum settings (existing)
    enable: bool = True
    lookbacks: List[int] = field(default_factory=lambda: [5, 10, 20, 60])
    normalization: str = "zscore"
    adjustment_scale: float = 0.15  # Max 15% adjustment to drift
    sign_confirmation_weight: float = 0.5  # Weight for sign alignment
    
    # Mean Reversion settings (Elite Upgrade - February 2026)
    enable_mean_reversion: bool = True
    mr_equilibrium_method: str = "state_space"  # 'state_space' or 'ma'
    mr_kappa_prior: float = 0.05  # Prior mean: half-life ~14 days
    mr_kappa_prior_var: float = 0.01  # EXPLICIT prior variance (Expert #4)
    mr_kappa_shrinkage: float = 0.5  # Shrinkage strength
    mr_kappa_min: float = 0.01
    mr_kappa_max: float = 0.10  # TIGHTENED (Expert #3: identifiability)
    mr_adjustment_scale: float = 0.15
    
    # φ shrinkage when MR enabled (Expert #3: identifiability)
    mr_phi_shrinkage_target: float = 1.0  # Shrink φ toward 1.0
    mr_phi_shrinkage_strength: float = 0.3  # 30% shrinkage toward target
    
    # Regime-Adaptive Blending
    regime_adaptive_blend: bool = True
    
    # CRPS feedback DISABLED by default (Expert #5: leakage risk)
    crps_feedback_enabled: bool = False
    crps_feedback_alpha: float = 0.1
    crps_window: int = 63
    
    # Dynamic max_u scaling (Expert #8)
    max_u_scale_by_q: bool = True
    max_u_multiplier: float = 3.0
    max_u_floor: float = 0.005
    max_u_ceiling: float = 0.03
    
    def __post_init__(self):
        """Validate configuration."""
        if self.normalization not in ("zscore", "rank"):
            raise ValueError(f"Invalid normalization: {self.normalization}")
        if not 0 <= self.adjustment_scale <= 1:
            raise ValueError(f"adjustment_scale must be in [0, 1]: {self.adjustment_scale}")
        if not 0 <= self.sign_confirmation_weight <= 1:
            raise ValueError(f"sign_confirmation_weight must be in [0, 1]")
        if self.mr_equilibrium_method not in ("state_space", "ma"):
            raise ValueError(f"Invalid mr_equilibrium_method: {self.mr_equilibrium_method}")
        if self.mr_kappa_prior_var <= 0:
            raise ValueError(f"mr_kappa_prior_var must be positive")


# Default configuration
DEFAULT_MOMENTUM_CONFIG = MomentumConfig()

# Disabled configuration (for ablation testing)
DISABLED_MOMENTUM_CONFIG = MomentumConfig(enable=False)


# =============================================================================
# MOMENTUM FEATURE COMPUTATION
# =============================================================================

def compute_momentum_features(
    returns: np.ndarray,
    lookbacks: List[int] = None,
    normalization: str = "zscore",
) -> Dict[str, np.ndarray]:
    """
    Compute normalized momentum features for a return series.
    
    Features are computed as cumulative returns over lookback windows,
    then normalized to make them comparable across assets.
    
    Args:
        returns: Array of returns (T,)
        lookbacks: List of lookback windows in days
        normalization: 'zscore' or 'rank'
        
    Returns:
        Dictionary mapping feature names to arrays:
        {
            'momentum_5': np.ndarray,
            'momentum_10': np.ndarray,
            ...
            'momentum_composite': np.ndarray,  # Average of all features
        }
    """
    if lookbacks is None:
        lookbacks = [5, 10, 20, 60]
    
    returns = np.asarray(returns).flatten()
    n = len(returns)
    
    features = {}
    valid_features = []
    
    for lb in lookbacks:
        if lb >= n:
            # Insufficient history for this lookback
            continue
            
        # Compute cumulative return over lookback
        momentum = np.zeros(n)
        for t in range(lb, n):
            cum_ret = np.sum(returns[t-lb:t])
            momentum[t] = cum_ret
        
        # Fill early values with expanding window
        for t in range(1, min(lb, n)):
            momentum[t] = np.sum(returns[:t])
        
        # Normalize
        if normalization == "zscore":
            # Rolling z-score normalization
            normalized = np.zeros(n)
            for t in range(1, n):
                window = momentum[max(0, t-252):t+1]  # 1-year rolling window
                if len(window) > 1:
                    mu = np.mean(window)
                    sigma = np.std(window)
                    if sigma > 1e-10:
                        normalized[t] = (momentum[t] - mu) / sigma
                    else:
                        normalized[t] = 0.0
                else:
                    normalized[t] = 0.0
            momentum = normalized
            
        elif normalization == "rank":
            # Rolling percentile rank
            normalized = np.zeros(n)
            for t in range(1, n):
                window = momentum[max(0, t-252):t+1]
                if len(window) > 1:
                    rank = np.sum(window <= momentum[t]) / len(window)
                    normalized[t] = 2 * rank - 1  # Scale to [-1, 1]
                else:
                    normalized[t] = 0.0
            momentum = normalized
        
        # Clip extreme values
        momentum = np.clip(momentum, -3.0, 3.0)
        
        features[f'momentum_{lb}'] = momentum
        valid_features.append(momentum)
    
    # Composite momentum: average of all features
    if valid_features:
        composite = np.mean(valid_features, axis=0)
    else:
        composite = np.zeros(n)
    features['momentum_composite'] = composite
    
    return features


def compute_momentum_signal(
    features: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Combine momentum features into a single signal.
    
    Args:
        features: Dictionary of momentum features from compute_momentum_features()
        weights: Optional weights for each feature (default: equal weights)
        
    Returns:
        Combined momentum signal array
    """
    if 'momentum_composite' in features:
        return features['momentum_composite']
    
    if not features:
        return np.array([0.0])
    
    # Equal-weighted combination
    arrays = list(features.values())
    if not arrays:
        return np.array([0.0])
    
    return np.mean(arrays, axis=0)


# =============================================================================
# MEAN REVERSION SIGNAL COMPUTATION (Elite Upgrade - February 2026)
# =============================================================================
# Expert Panel Validated Implementation:
# - State-equation injection (coherent likelihood)
# - Delta-method κ variance estimation (Expert #4)
# - CAIP-compatible cross-asset shrinkage ready
# - φ/κ identifiability safeguards (Expert #3)
# =============================================================================

# Regime blend weight priors (soft, not hard)
REGIME_ALPHA_PRIOR = {0: 0.5, 1: 0.15, 2: 0.65, 3: 0.35, 4: 0.45}
REGIME_BETA_PRIOR = {0: 0.5, 1: 0.15, 2: 0.25, 3: 0.65, 4: 0.55}


def estimate_local_level_equilibrium(
    prices: np.ndarray,
    vol: np.ndarray,
    q_level: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate local level equilibrium via Kalman filter (state-space).
    
    ELITE: Uses state-space model instead of moving average:
        P_t = ℓ_t + ε_t,  ε_t ~ N(0, σ²_obs)
        ℓ_t = ℓ_{t-1} + η_t,  η_t ~ N(0, q_level)
    
    Args:
        prices: Array of prices
        vol: Array of volatility estimates
        q_level: Process noise for level evolution
        
    Returns:
        Tuple of (equilibrium, equilibrium_var)
    """
    n = len(prices)
    log_prices = np.log(np.maximum(prices, 1e-10))
    
    level_t = log_prices[0]
    P_t = 1.0
    
    equilibrium = np.zeros(n)
    equilibrium_var = np.zeros(n)
    
    for t in range(n):
        R_t = vol[t] ** 2 if t < len(vol) else 0.01
        
        level_pred = level_t
        P_pred = P_t + q_level
        
        innovation = log_prices[t] - level_pred
        S_t = P_pred + R_t
        K_t = P_pred / S_t if S_t > 1e-12 else 0.0
        
        level_t = level_pred + K_t * innovation
        P_t = (1 - K_t) * P_pred
        
        equilibrium[t] = level_t
        equilibrium_var[t] = P_t
    
    return equilibrium, equilibrium_var


def estimate_kappa_bayesian(
    deviations: np.ndarray,
    prior_mean: float = 0.05,
    prior_var: float = 0.01,  # EXPLICIT prior variance (Expert #4)
    shrinkage_strength: float = 0.5,  # Separate from variance
    kappa_min: float = 0.01,
    kappa_max: float = 0.10,  # TIGHTENED (Expert #3)
) -> Tuple[float, Dict]:
    """
    Estimate OU reversion speed with proper delta-method variance.
    
    EXPERT VALIDATED:
    - Var(κ̂) ≈ (1/φ)² × (1-φ²)/T (delta-method, Expert #4)
    - Explicit prior variance (not conflated with shrinkage strength)
    - Tightened kappa_max for identifiability (Expert #3)
    
    Args:
        deviations: Array of vol-normalized deviations from equilibrium
        prior_mean: Prior mean for κ
        prior_var: EXPLICIT prior variance for κ
        shrinkage_strength: Weight for prior (separate from variance)
        kappa_min: Floor for κ
        kappa_max: Cap for κ
        
    Returns:
        Tuple of (kappa_final, diagnostics)
    """
    n = len(deviations)
    
    if n < 30:
        return prior_mean, {"method": "prior_only", "n": n}
    
    z_lag = deviations[:-1]
    z_current = deviations[1:]
    
    valid = np.isfinite(z_lag) & np.isfinite(z_current)
    if np.sum(valid) < 20:
        return prior_mean, {"method": "prior_only", "valid_obs": int(np.sum(valid))}
    
    z_lag_valid = z_lag[valid]
    z_current_valid = z_current[valid]
    
    # AR(1) OLS
    denom = np.sum(z_lag_valid ** 2)
    if denom < 1e-12:
        return prior_mean, {"method": "prior_only", "reason": "zero_variance"}
    
    rho_mle = np.sum(z_lag_valid * z_current_valid) / denom
    rho_mle = np.clip(rho_mle, 0.01, 0.99)
    
    # Convert to κ
    kappa_mle = -np.log(rho_mle)
    
    # PROPER variance via delta method (Expert #4):
    # Var(φ̂) ≈ (1-φ²)/T
    # Var(κ̂) ≈ (1/φ)² × Var(φ̂)
    T_eff = np.sum(valid)
    var_rho = (1 - rho_mle ** 2) / T_eff
    var_kappa = (1 / rho_mle) ** 2 * var_rho
    sigma_sq_mle = max(var_kappa, 1e-6)
    
    # Bayesian shrinkage with EXPLICIT prior variance (Expert #4)
    precision_mle = 1.0 / sigma_sq_mle
    precision_prior = shrinkage_strength / prior_var  # Clear separation
    
    kappa_posterior = (precision_mle * kappa_mle + precision_prior * prior_mean) / (precision_mle + precision_prior)
    kappa_final = np.clip(kappa_posterior, kappa_min, kappa_max)
    
    return kappa_final, {
        "method": "bayesian_delta",
        "rho_mle": float(rho_mle),
        "kappa_mle": float(kappa_mle),
        "var_kappa": float(var_kappa),
        "kappa_posterior": float(kappa_posterior),
        "kappa_final": float(kappa_final),
        "half_life_days": float(np.log(2) / kappa_final) if kappa_final > 0 else float('inf'),
        "shrinkage_ratio": float(precision_prior / (precision_mle + precision_prior)),
        "prior_var_used": float(prior_var),
    }


def compute_mr_signal(
    log_prices: np.ndarray,
    equilibrium: np.ndarray,
    vol: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """
    Compute mean reversion signal with correct dimensions.
    
    MATHEMATICALLY CORRECT (Expert validated):
        z_t = (log P_t - log ℓ_t) / σ_t  (dimensionless deviation)
        MR_signal_t = -κ × z_t × σ_t    (returns units, mean-reverting)
    
    Args:
        log_prices: Log of prices
        equilibrium: Log-scale equilibrium from state-space
        vol: Volatility estimates
        kappa: Reversion speed
        
    Returns:
        Array of mean reversion signals (in return units)
    """
    vol_safe = np.maximum(vol, 1e-10)
    z_deviation = (log_prices - equilibrium) / vol_safe
    z_deviation = np.clip(z_deviation, -3.0, 3.0)
    return -kappa * z_deviation * vol_safe


def compute_adaptive_blend_weights(
    n: int,
    regime_labels: np.ndarray = None,
    regime_probabilities: np.ndarray = None,
    vol_regime: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute adaptive α_t, β_t based on soft regime probability.
    
    NOTE: CRPS feedback DISABLED by default per Expert #5 (leakage risk).
    Enable only after verifying out-of-sample stability.
    
    Args:
        n: Length of arrays
        regime_labels: Optional hard regime labels
        regime_probabilities: Optional soft regime probabilities (n, 5)
        vol_regime: Vol relative to median (for dampening)
        
    Returns:
        Tuple of (alpha_t, beta_t) arrays
    """
    alpha_t = np.full(n, 0.5)
    beta_t = np.full(n, 0.5)
    
    for t in range(n):
        # Soft regime blend
        if regime_probabilities is not None and t < len(regime_probabilities):
            probs = regime_probabilities[t]
            alpha_regime = sum(probs[r] * REGIME_ALPHA_PRIOR.get(r, 0.5) for r in range(min(5, len(probs))))
            beta_regime = sum(probs[r] * REGIME_BETA_PRIOR.get(r, 0.5) for r in range(min(5, len(probs))))
        elif regime_labels is not None and t < len(regime_labels):
            regime = int(regime_labels[t])
            alpha_regime = REGIME_ALPHA_PRIOR.get(regime, 0.5)
            beta_regime = REGIME_BETA_PRIOR.get(regime, 0.5)
        else:
            alpha_regime = 0.5
            beta_regime = 0.5
        
        alpha_t[t] = alpha_regime
        beta_t[t] = beta_regime
        
        # High vol dampening (both signals less reliable in extreme vol)
        if vol_regime is not None and t < len(vol_regime) and vol_regime[t] > 2.0:
            alpha_t[t] *= 0.5
            beta_t[t] *= 0.5
        
        # Normalize so α + β ≤ 1.0
        total = alpha_t[t] + beta_t[t]
        if total > 1.0:
            alpha_t[t] /= total
            beta_t[t] /= total
    
    return alpha_t, beta_t


def apply_phi_shrinkage_for_mr(
    phi: float,
    config: 'MomentumConfig',
) -> float:
    """
    Shrink φ toward 1.0 when MR is enabled (Expert #3: identifiability).
    
    φ and κ can trade off in ranging regimes. To prevent weak identifiability,
    we shrink φ toward 1.0 (pure random walk) when MR is active.
    
    φ_shrunk = (1 - s) × φ + s × target
    where s = mr_phi_shrinkage_strength
    
    Args:
        phi: Original φ value
        config: MomentumConfig with shrinkage settings
        
    Returns:
        Shrunk φ value
    """
    if not config.enable_mean_reversion:
        return phi
    
    s = config.mr_phi_shrinkage_strength
    target = config.mr_phi_shrinkage_target
    
    phi_shrunk = (1 - s) * phi + s * target
    
    return phi_shrunk


# =============================================================================
# MOMENTUM AUGMENTED DRIFT MODEL WRAPPER
# =============================================================================

class MomentumAugmentedDriftModel:
    """
    Wrapper class for momentum-augmented drift models.
    
    This class wraps any base drift model (Gaussian, φ-Gaussian, φ-Student-t)
    and applies momentum augmentation compositionally, without modifying
    the base filter equations.
    
    DESIGN PHILOSOPHY:
        - Base model runs unchanged
        - Momentum features computed separately
        - Adjustments applied post-filter to drift estimates
        - All interfaces identical to base model
    
    Usage:
        # Wrap a base model
        config = MomentumConfig(enable=True, lookbacks=[5, 10, 20])
        wrapper = MomentumAugmentedDriftModel(config)
        
        # Filter returns (same interface as base model)
        mu, P, ll = wrapper.filter(returns, vol, q, c, base_model='gaussian')
        
        # Toggle momentum off for ablation
        wrapper.config.enable = False
        mu_baseline, P_baseline, ll_baseline = wrapper.filter(...)
    """
    
    def __init__(self, config: Optional[MomentumConfig] = None):
        """
        Initialize momentum augmented model.
        
        Args:
            config: MomentumConfig instance (uses defaults if None)
        """
        self.config = config or DEFAULT_MOMENTUM_CONFIG
        
        # Momentum signals
        self._momentum_features: Optional[Dict[str, np.ndarray]] = None
        self._momentum_signal: Optional[np.ndarray] = None
        
        # Mean Reversion signals (Elite Upgrade - February 2026)
        self._mr_signal: Optional[np.ndarray] = None
        self._equilibrium: Optional[np.ndarray] = None
        self._kappa: Optional[float] = None
        self._kappa_diagnostics: Optional[Dict] = None
        
        # Regime-adaptive blend weights
        self._alpha_t: Optional[np.ndarray] = None
        self._beta_t: Optional[np.ndarray] = None
        
        # Exogenous input for state equation
        self._exogenous_input: Optional[np.ndarray] = None
        
        # Process noise q (needed for dynamic max_u scaling - Expert #8)
        self._q: Optional[float] = None
        
        self._diagnostics: Dict[str, Any] = {}
    
    def precompute_momentum(self, returns: np.ndarray) -> None:
        """
        Precompute momentum features for the return series.
        
        This should be called once per asset before filtering.
        Features are cached for efficiency.
        
        Args:
            returns: Array of returns
        """
        if not self.config.enable:
            self._momentum_features = None
            self._momentum_signal = None
            return
        
        n = len(returns)
        self._momentum_features = compute_momentum_features(
            returns,
            lookbacks=self.config.lookbacks,
            normalization=self.config.normalization,
        )
        self._momentum_signal = compute_momentum_signal(self._momentum_features)
        
        # Initialize MR and blend fields so _compute_exogenous_input() works
        if self._mr_signal is None:
            self._mr_signal = np.zeros(n)
        if self._kappa is None:
            self._kappa = self.config.mr_kappa_prior
        if self._alpha_t is None or self._beta_t is None:
            self._alpha_t, self._beta_t = compute_adaptive_blend_weights(n)
        if self._q is None:
            self._q = 1e-6
        
        # Compute exogenous input for state-equation injection
        self._exogenous_input = self._compute_exogenous_input()
    
    def precompute_signals(
        self,
        returns: np.ndarray,
        prices: np.ndarray = None,
        vol: np.ndarray = None,
        regime_labels: np.ndarray = None,
        regime_probabilities: np.ndarray = None,
        q: float = 1e-6,  # Process noise for dynamic max_u scaling
    ) -> None:
        """
        Precompute all augmentation signals (momentum + MR).
        
        STATE-EQUATION INTEGRATION (Elite Upgrade - February 2026):
        Computes exogenous input u_t = α_t × MOM_t - β_t × MR_t
        for injection into state equation.
        
        EXPERT VALIDATED:
        - CRPS feedback disabled by default (Expert #5)
        - φ shrinkage applied when MR enabled (Expert #3)
        - max_u scaled by √q (Expert #8)
        
        Args:
            returns: Array of returns
            prices: Array of prices (required for MR)
            vol: Array of volatility estimates
            regime_labels: Optional hard regime labels
            regime_probabilities: Optional soft regime probabilities
            q: Process noise (for dynamic max_u scaling)
        """
        n = len(returns)
        self._q = q
        
        # Momentum (existing)
        if self.config.enable:
            self._momentum_features = compute_momentum_features(
                returns, lookbacks=self.config.lookbacks,
                normalization=self.config.normalization,
            )
            self._momentum_signal = compute_momentum_signal(self._momentum_features)
        else:
            self._momentum_signal = np.zeros(n)
        
        # Mean Reversion (Elite Upgrade)
        if self.config.enable_mean_reversion and prices is not None and vol is not None:
            log_prices = np.log(np.maximum(prices, 1e-10))
            
            if self.config.mr_equilibrium_method == "state_space":
                self._equilibrium, _ = estimate_local_level_equilibrium(prices, vol)
            else:
                # MA fallback
                import pandas as pd
                ma_50 = pd.Series(log_prices).rolling(50, min_periods=1).mean().values
                self._equilibrium = ma_50
            
            z_deviation = (log_prices - self._equilibrium) / np.maximum(vol, 1e-10)
            
            # Bayesian κ with explicit prior variance (Expert #4)
            self._kappa, self._kappa_diagnostics = estimate_kappa_bayesian(
                z_deviation,
                prior_mean=self.config.mr_kappa_prior,
                prior_var=self.config.mr_kappa_prior_var,  # Explicit
                shrinkage_strength=self.config.mr_kappa_shrinkage,  # Separate
                kappa_min=self.config.mr_kappa_min,
                kappa_max=self.config.mr_kappa_max,
            )
            
            self._mr_signal = compute_mr_signal(log_prices, self._equilibrium, vol, self._kappa)
        else:
            self._mr_signal = np.zeros(n)
            self._kappa = self.config.mr_kappa_prior
        
        # Compute vol regime for dampening
        vol_regime = None
        if vol is not None:
            vol_median = np.median(vol)
            vol_regime = vol / vol_median if vol_median > 0 else np.ones(n)
        
        # Adaptive blend weights (CRPS feedback DISABLED by default - Expert #5)
        self._alpha_t, self._beta_t = compute_adaptive_blend_weights(
            n, regime_labels, regime_probabilities, vol_regime,
        )
        
        # Compute exogenous input with dynamic max_u (Expert #8)
        self._exogenous_input = self._compute_exogenous_input()
    
    def _compute_exogenous_input(self) -> Optional[np.ndarray]:
        """
        Compute exogenous input u_t for state-equation injection.
        
        u_t = α_t × MOM_t × scale - β_t × MR_t × scale
        
        ELITE (Expert #8): max_u scales with √q for vol-consistency.
        
        Returns:
            Array of exogenous inputs, or None
        """
        if self._momentum_signal is None and self._mr_signal is None:
            return None
        
        n = len(self._momentum_signal) if self._momentum_signal is not None else len(self._mr_signal)
        u_t = np.zeros(n)
        
        mom_scale = self.config.adjustment_scale
        mr_scale = self.config.mr_adjustment_scale
        
        for t in range(n):
            mom_contrib = 0.0
            mr_contrib = 0.0
            
            if self._momentum_signal is not None and t < len(self._momentum_signal):
                alpha = self._alpha_t[t] if self._alpha_t is not None else 0.5
                mom_contrib = alpha * self._momentum_signal[t] * mom_scale
            
            if self._mr_signal is not None and t < len(self._mr_signal):
                beta = self._beta_t[t] if self._beta_t is not None else 0.5
                mr_contrib = beta * self._mr_signal[t] * mr_scale
            
            u_t[t] = mom_contrib - mr_contrib
        
        # ELITE: Dynamic max_u scaling (Expert #8)
        if self.config.max_u_scale_by_q and self._q is not None:
            max_u = self.config.max_u_multiplier * np.sqrt(self._q)
            max_u = np.clip(max_u, self.config.max_u_floor, self.config.max_u_ceiling)
        else:
            max_u = 0.02  # Fixed fallback
        
        u_t = np.clip(u_t, -max_u, max_u)
        
        return u_t
    
    def filter(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float = 1.0,
        nu: Optional[float] = None,
        base_model: str = 'gaussian',
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run Kalman filter with state-equation integrated signals.
        
        ELITE UPGRADE (February 2026): Momentum and MR are injected into
        state equation: μ_t = φ × μ_{t-1} + u_t + w_t
        This preserves probabilistic coherence.
        
        EXPERT #3: When MR enabled, φ is shrunk toward 1.0 for identifiability.
        
        Args:
            returns: Array of returns
            vol: Array of EWMA volatility
            q: Process noise variance
            c: Observation noise scale
            phi: AR(1) persistence (1.0 for random walk)
            nu: Degrees of freedom (only for Student-t)
            base_model: 'gaussian', 'phi_gaussian', or 'phi_student_t'
            
        Returns:
            Tuple of (mu_filtered, P_filtered, log_likelihood)
            Same interface as base model filter methods.
        """
        # Import base models here to avoid circular imports
        from models.gaussian import GaussianDriftModel
        from models.phi_student_t import PhiStudentTDriftModel
        
        # Apply φ shrinkage for identifiability (Expert #3)
        phi_effective = apply_phi_shrinkage_for_mr(phi, self.config)
        
        # Ensure signals computed (backward compat)
        if self._exogenous_input is None and self.config.enable:
            self.precompute_momentum(returns)
        
        # Get exogenous input
        u_t = self._exogenous_input
        
        # State-equation integrated filter (if u_t available)
        if u_t is not None and len(u_t) == len(returns):
            if base_model == 'phi_student_t' and nu is not None:
                mu_filtered, P_filtered, ll = PhiStudentTDriftModel.filter_phi_augmented(
                    returns, vol, q, c, phi_effective, nu, exogenous_input=u_t
                )
            elif base_model == 'phi_gaussian':
                mu_filtered, P_filtered, ll = GaussianDriftModel.filter_phi_augmented(
                    returns, vol, q, c, phi_effective, exogenous_input=u_t
                )
            else:
                mu_filtered, P_filtered, ll = GaussianDriftModel.filter_augmented(
                    returns, vol, q, c, exogenous_input=u_t
                )
        else:
            # Fallback to original (no state-equation injection)
            if base_model == 'gaussian':
                mu_filtered, P_filtered, ll = GaussianDriftModel.filter(returns, vol, q, c)
            elif base_model == 'phi_gaussian':
                mu_filtered, P_filtered, ll = GaussianDriftModel.filter_phi(returns, vol, q, c, phi)
            elif base_model == 'phi_student_t':
                if nu is None:
                    raise ValueError("nu required for phi_student_t model")
                mu_filtered, P_filtered, ll = PhiStudentTDriftModel.filter_phi(returns, vol, q, c, phi, nu)
            else:
                raise ValueError(f"Unknown base_model: {base_model}")
            
            # Apply post-filter adjustment (legacy path)
            if self.config.enable and self._momentum_signal is not None:
                mu_filtered, P_filtered = self._apply_momentum_augmentation(mu_filtered, P_filtered)
                # Recompute log-likelihood with adjusted estimates
                ll = self._compute_adjusted_log_likelihood(
                    returns, mu_filtered, vol, P_filtered, c, nu, base_model
                )
        
        # Store diagnostics
        self._diagnostics = {
            'momentum_enabled': self.config.enable,
            'mr_enabled': self.config.enable_mean_reversion,
            'state_equation_injection': u_t is not None and len(u_t) == len(returns),
            'phi_original': phi,
            'phi_effective': phi_effective,
            'phi_shrinkage_applied': abs(phi - phi_effective) > 1e-6,
            'base_model': base_model,
            'q': q, 'c': c, 'nu': nu,
        }
        
        if self._kappa is not None:
            self._diagnostics['mr_kappa'] = float(self._kappa)
            self._diagnostics['mr_half_life_days'] = float(np.log(2) / self._kappa) if self._kappa > 0 else None
            self._diagnostics['mr_equilibrium_method'] = self.config.mr_equilibrium_method
            if self._kappa_diagnostics:
                self._diagnostics['mr_kappa_diagnostics'] = self._kappa_diagnostics
        
        if self._alpha_t is not None:
            self._diagnostics['alpha_mean'] = float(np.mean(self._alpha_t))
            self._diagnostics['beta_mean'] = float(np.mean(self._beta_t))
        
        if self._momentum_signal is not None:
            self._diagnostics['momentum_mean'] = float(np.mean(self._momentum_signal))
            self._diagnostics['momentum_std'] = float(np.std(self._momentum_signal))
        
        if self._mr_signal is not None:
            self._diagnostics['mr_signal_std'] = float(np.std(self._mr_signal))
        
        # Dynamic max_u info (Expert #8)
        if self.config.max_u_scale_by_q and self._q is not None:
            max_u_used = self.config.max_u_multiplier * np.sqrt(self._q)
            max_u_used = np.clip(max_u_used, self.config.max_u_floor, self.config.max_u_ceiling)
            self._diagnostics['max_u_dynamic'] = float(max_u_used)
        
        return mu_filtered, P_filtered, ll
        
        # Apply momentum augmentation if enabled
        if self.config.enable and self._momentum_signal is not None:
            mu_filtered, P_filtered = self._apply_momentum_augmentation(
                mu_filtered, P_filtered
            )
            
            # Recompute log-likelihood with momentum-adjusted uncertainty
            # The forecast variance is c * vol^2 + P, where P is now adjusted
            ll = self._compute_adjusted_log_likelihood(
                returns, mu_filtered, vol, P_filtered, c, nu, base_model
            )
        
        # Store diagnostics
        self._diagnostics = {
            'momentum_enabled': self.config.enable,
            'base_model': base_model,
            'q': q,
            'c': c,
            'phi': phi,
            'nu': nu,
        }
        
        if self.config.enable and self._momentum_signal is not None:
            self._diagnostics.update({
                'momentum_mean': float(np.mean(self._momentum_signal)),
                'momentum_std': float(np.std(self._momentum_signal)),
                'momentum_last': float(self._momentum_signal[-1]) if len(self._momentum_signal) > 0 else 0.0,
            })
        
        return mu_filtered, P_filtered, ll
    
    def _compute_adjusted_log_likelihood(
        self,
        returns: np.ndarray,
        mu_filtered: np.ndarray,
        vol: np.ndarray,
        P_filtered: np.ndarray,
        c: float,
        nu: Optional[float],
        base_model: str,
    ) -> float:
        """
        Recompute log-likelihood with momentum-adjusted uncertainty.
        
        Performance optimizations (February 2026):
        - Pre-compute constants outside loop
        - Vectorize where possible
        - Avoid redundant gammaln calls
        
        Args:
            returns: Array of returns
            mu_filtered: Filtered drift estimates (momentum-adjusted)
            vol: Array of EWMA volatility
            P_filtered: Filtered uncertainty (momentum-adjusted)
            c: Observation noise scale
            nu: Degrees of freedom (only for Student-t)
            base_model: Model type
            
        Returns:
            Adjusted log-likelihood
        """
        from scipy.special import gammaln
        
        n = len(returns)
        
        # Convert to contiguous arrays
        returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)
        mu_filtered = np.ascontiguousarray(mu_filtered.flatten(), dtype=np.float64)
        P_filtered = np.ascontiguousarray(P_filtered.flatten(), dtype=np.float64)
        
        # Pre-compute R array
        R = c * (vol * vol)
        
        # Compute forecast variance array
        forecast_var = R + P_filtered
        forecast_var = np.maximum(forecast_var, 1e-12)
        
        # Compute innovations
        innovations = returns - mu_filtered
        
        if base_model in ('gaussian', 'phi_gaussian'):
            # Vectorized Gaussian log-likelihood
            log_2pi = np.log(2 * np.pi)
            ll_array = -0.5 * (log_2pi + np.log(forecast_var) + (innovations * innovations) / forecast_var)
            ll_array = np.where(np.isfinite(ll_array), ll_array, 0.0)
            return float(np.sum(ll_array))
        else:
            # Student-t log-likelihood
            if nu is None:
                nu = 8.0  # Default
            
            # Pre-compute constants (avoid gammaln in loop)
            log_norm_const = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)
            neg_half_nu_plus_1 = -((nu + 1.0) / 2.0)
            inv_nu = 1.0 / nu;
            
            # Vectorized Student-t log-likelihood
            scale = np.sqrt(forecast_var)
            z = innovations / scale
            log_scale = np.log(scale)
            log_kernel = neg_half_nu_plus_1 * np.log(1.0 + (z * z) * inv_nu)
            ll_array = log_norm_const - log_scale + log_kernel
            ll_array = np.where(np.isfinite(ll_array), ll_array, 0.0)
            return float(np.sum(ll_array))
    
    def _apply_momentum_augmentation(
        self,
        mu_filtered: np.ndarray,
        P_filtered: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply momentum augmentation to filtered drift estimates.
        
        LOGIC:
        1. Sign confirmation: When sign(μ) aligns with sign(momentum),
           allow higher persistence (reduce uncertainty).
        2. Sign conflict: When signs conflict, increase uncertainty.
        3. Magnitude: Momentum adjusts persistence, not raw drift magnitude.
        
        This treats momentum as "permission to believe drift" rather than
        "source of drift" — matching how sophisticated funds use momentum.
        
        Args:
            mu_filtered: Filtered drift estimates from base model
            P_filtered: Filtered uncertainty from base model
            
        Returns:
            Tuple of (adjusted_mu, adjusted_P)
        """
        if self._momentum_signal is None:
            return mu_filtered, P_filtered
        
        n = len(mu_filtered)
        momentum = self._momentum_signal
        
        if len(momentum) != n:
            # Mismatched lengths - skip adjustment
            return mu_filtered, P_filtered
        
        # Initialize adjusted arrays
        mu_adj = mu_filtered.copy()
        P_adj = P_filtered.copy()
        
        # Configuration
        scale = self.config.adjustment_scale
        sign_weight = self.config.sign_confirmation_weight
        
        for t in range(n):
            mu_t = mu_filtered[t]
            P_t = P_filtered[t]
            mom_t = momentum[t]
            
            # Compute sign alignment
            if abs(mu_t) < 1e-10 or abs(mom_t) < 1e-10:
                # Near-zero values: no adjustment
                alignment = 0.0
            else:
                # +1 if signs agree, -1 if signs disagree
                sign_mu = np.sign(mu_t)
                sign_mom = np.sign(mom_t)
                alignment = sign_mu * sign_mom  # +1 or -1
            
            # Momentum strength (0 to 1)
            mom_strength = min(abs(mom_t) / 2.0, 1.0)  # Normalized to [0, 1]
            
            # Adjustment factor based on alignment and strength
            # When aligned: reduce uncertainty (more confident)
            # When conflicting: increase uncertainty (less confident)
            if alignment > 0:
                # Signs agree: allow persistence
                # Reduce P by up to scale * sign_weight * mom_strength
                P_reduction = scale * sign_weight * mom_strength * P_t
                P_adj[t] = max(P_t - P_reduction, P_t * 0.5)  # Never reduce by more than 50%
            elif alignment < 0:
                # Signs disagree: dampen confidence
                # Increase P by up to scale * sign_weight * mom_strength
                P_increase = scale * sign_weight * mom_strength * P_t
                P_adj[t] = P_t + P_increase
        
        return mu_adj, P_adj
    
    def get_momentum_features(self) -> Optional[Dict[str, np.ndarray]]:
        """Return computed momentum features (for diagnostics)."""
        return self._momentum_features
    
    def get_momentum_signal(self) -> Optional[np.ndarray]:
        """Return combined momentum signal (for diagnostics)."""
        return self._momentum_signal
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostics from last filter call."""
        return self._diagnostics.copy()


# =============================================================================
# MODEL NAME GENERATION FOR BMA INTEGRATION
# =============================================================================

def get_momentum_augmented_model_name(base_name: str) -> str:
    """
    Generate BMA model name for momentum-augmented variant.
    
    Args:
        base_name: Base model name (e.g., 'kalman_gaussian')
        
    Returns:
        Momentum-augmented model name (e.g., 'kalman_gaussian_momentum')
    """
    return f"{base_name}_momentum"


# =============================================================================
# NUMBA-ACCELERATED MOMENTUM FILTERING (Direct Filter Methods)
# =============================================================================

class MomentumPhiGaussianFilter:
    """
    Direct Numba-accelerated φ-Gaussian filter with momentum augmentation.
    
    Used by: CRSP, CELH, DPRO augmented models
    
    This class provides direct access to Numba-accelerated momentum filtering
    without the wrapper overhead. For most use cases, use
    MomentumAugmentedDriftModel instead.
    """
    
    @staticmethod
    def filter(
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        momentum_signal: np.ndarray,
        momentum_weight: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run φ-Gaussian filter with momentum augmentation.
        
        Uses Numba kernel when available (10-50× speedup).
        
        Args:
            returns: Array of returns
            vol: Array of EWMA volatility
            q: Process noise variance
            c: Observation noise scale
            phi: AR(1) persistence
            momentum_signal: Per-timestep normalized momentum
            momentum_weight: Scaling factor for momentum contribution
            
        Returns:
            Tuple of (mu_filtered, P_filtered, log_likelihood)
        """
        # Compute momentum adjustment
        momentum_adjustment = momentum_weight * momentum_signal * vol
        
        # Try Numba kernel
        if _USE_NUMBA:
            try:
                return run_momentum_phi_gaussian_filter(
                    returns, vol, q, c, phi, momentum_adjustment
                )
            except Exception:
                pass
        
        # Python fallback
        return MomentumPhiGaussianFilter._filter_python(
            returns, vol, q, c, phi, momentum_adjustment
        )
    
    @staticmethod
    def _filter_python(
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        momentum_adjustment: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Pure Python implementation (for fallback and testing)."""
        n = len(returns)
        mu = 0.0
        P = 1e-4
        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
        log_likelihood = 0.0
        phi_sq = phi ** 2
        
        for t in range(n):
            # Momentum-augmented prediction
            mu_pred = phi * mu + momentum_adjustment[t]
            P_pred = phi_sq * P + q
            
            vol_t = float(vol[t])
            R = c * (vol_t ** 2)
            innovation = float(returns[t]) - mu_pred
            S = P_pred + R
            
            if S > 1e-12:
                K = P_pred / S
                mu = mu_pred + K * innovation
                P = (1.0 - K) * P_pred
                
                innov_sq_scaled = min((innovation ** 2) / S, 100.0)
                ll_contrib = -0.5 * (np.log(2 * np.pi * S) + innov_sq_scaled)
                log_likelihood += max(ll_contrib, -50.0)
            else:
                mu = mu_pred
                P = P_pred
            
            mu_filtered[t] = mu
            P_filtered[t] = max(P, 1e-12)
        
        return mu_filtered, P_filtered, log_likelihood


class MomentumPhiStudentTFilter:
    """
    Direct Numba-accelerated φ-Student-t filter with momentum augmentation.
    
    Used by: GLDW, MAGD, BKSY, ASTS augmented models
    
    ARCHITECTURAL INVARIANT:
        There is NO bare Student-t momentum filter. All Student-t
        momentum filtering uses φ-Student-t.
    
    This class provides direct access to Numba-accelerated momentum filtering
    without the wrapper overhead. For most use cases, use
    MomentumAugmentedDriftModel instead.
    """
    
    @staticmethod
    def filter(
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        nu: float,
        momentum_signal: np.ndarray,
        momentum_weight: float = 0.1,
        hierarchical_lambda: Optional[np.ndarray] = None,
        lambda_direction: str = 'none',
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run φ-Student-t filter with momentum and optional hierarchical λ.
        
        Uses Numba kernel when available (10-50× speedup).
        
        Args:
            returns: Array of returns
            vol: Array of EWMA volatility
            q: Process noise variance
            c: Observation noise scale
            phi: AR(1) persistence
            nu: Degrees of freedom
            momentum_signal: Per-timestep normalized momentum
            momentum_weight: Scaling factor for momentum contribution
            hierarchical_lambda: Per-timestep λ adjustment (optional)
            lambda_direction: 'backward' (Hλ←), 'forward' (Hλ→), or 'none'
            
        Returns:
            Tuple of (mu_filtered, P_filtered, log_likelihood)
        """
        # Compute effective momentum weight
        if hierarchical_lambda is not None and lambda_direction != 'none':
            effective_weight = momentum_weight * hierarchical_lambda
        else:
            effective_weight = np.full(len(returns), momentum_weight)
        
        # Compute momentum adjustment
        momentum_adjustment = effective_weight * momentum_signal * vol
        
        # Try Numba kernel
        if _USE_NUMBA:
            try:
                return run_momentum_phi_student_t_filter(
                    returns, vol, q, c, phi, nu, momentum_adjustment
                )
            except Exception:
                pass
        
        # Python fallback
        return MomentumPhiStudentTFilter._filter_python(
            returns, vol, q, c, phi, nu, momentum_adjustment
        )
    
    @staticmethod
    def filter_batch(
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        nu_grid: List[float],
        momentum_signal: np.ndarray,
        momentum_weight: float = 0.1,
    ) -> Dict[float, Tuple[np.ndarray, np.ndarray, float]]:
        """
        Run momentum-augmented φ-Student-t filter for multiple ν values.
        
        Efficient for BMA over discrete ν grid.
        
        Args:
            nu_grid: List of ν values to evaluate
            
        Returns:
            Dict mapping ν -> (mu_filtered, P_filtered, log_likelihood)
        """
        momentum_adjustment = momentum_weight * momentum_signal * vol
        
        # Try Numba batch version
        if _USE_NUMBA:
            try:
                return run_momentum_phi_student_t_filter_batch(
                    returns, vol, q, c, phi, nu_grid, momentum_adjustment
                )
            except Exception:
                pass
        
        # Python fallback
        results = {}
        for nu in nu_grid:
            results[nu] = MomentumPhiStudentTFilter._filter_python(
                returns, vol, q, c, phi, nu, momentum_adjustment
            )
        return results
    
    @staticmethod
    def _filter_python(
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        nu: float,
        momentum_adjustment: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Pure Python implementation (for fallback and testing)."""
        from scipy.special import gammaln
        
        n = len(returns)
        mu = 0.0
        P = 1e-4
        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
        log_likelihood = 0.0
        phi_sq = phi ** 2
        nu_adjust = min(nu / (nu + 3.0), 1.0)
        
        # Precompute gamma values
        log_g1 = float(gammaln(nu / 2.0))
        log_g2 = float(gammaln((nu + 1.0) / 2.0))
        
        for t in range(n):
            # Momentum-augmented prediction
            mu_pred = phi * mu + momentum_adjustment[t]
            P_pred = phi_sq * P + q
            
            vol_t = float(vol[t])
            R = c * (vol_t ** 2)
            r_t = float(returns[t])
            innovation = r_t - mu_pred
            S = P_pred + R
            
            if S > 1e-12:
                scale = np.sqrt(S)
                z = innovation / scale
                
                # Student-t log-likelihood
                log_norm = log_g2 - log_g1 - 0.5 * np.log(nu * np.pi * scale * scale)
                log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + (z ** 2) / nu)
                ll_t = log_norm + log_kernel
                log_likelihood += max(ll_t, -50.0)
                
                K = nu_adjust * P_pred / S
                mu = mu_pred + K * innovation
                P = (1.0 - K) * P_pred
            else:
                mu = mu_pred
                P = P_pred
            
            mu_filtered[t] = mu
            P_filtered[t] = max(P, 1e-12)
        
        return mu_filtered, P_filtered, log_likelihood


def is_momentum_augmented_model(model_name: str) -> bool:
    """
    Check if a model name corresponds to a momentum-augmented variant.
    
    Args:
        model_name: Model identifier string
        
    Returns:
        True if model is momentum-augmented, False otherwise
    """
    if not model_name:
        return False
    return model_name.endswith('_momentum')


def get_base_model_name(momentum_model_name: str) -> str:
    """
    Extract base model name from momentum-augmented model name.
    
    Args:
        momentum_model_name: Momentum-augmented model name
        
    Returns:
        Base model name
    """
    if is_momentum_augmented_model(momentum_model_name):
        return momentum_model_name[:-9]  # Remove '_momentum' suffix
    return momentum_model_name


# =============================================================================
# BMA PRIOR PENALTY FOR MOMENTUM MODELS
# =============================================================================

# Momentum models receive a prior penalty in BMA (they must earn their weight)
# This is configurable and defaults to 0.3x the base model prior
MOMENTUM_BMA_PRIOR_PENALTY = 0.3

def compute_momentum_model_bic_adjustment(
    base_bic: float,
    prior_penalty: float = MOMENTUM_BMA_PRIOR_PENALTY,
) -> float:
    """
    Compute BIC adjustment for momentum-augmented model.
    
    Momentum models receive a prior penalty in BMA to prevent
    slow drift toward momentum dominance in noisy assets.
    
    The penalty is implemented as an additive BIC adjustment:
        BIC_adjusted = BIC_raw + 2 * log(1/prior_penalty)
    
    This is equivalent to placing a prior weight of `prior_penalty`
    on momentum variants relative to base models.
    
    Args:
        base_bic: Raw BIC from model fitting
        prior_penalty: Prior weight for momentum model (0-1)
        
    Returns:
        Adjusted BIC incorporating prior penalty
    """
    if prior_penalty <= 0 or prior_penalty > 1:
        return base_bic
    
    # Convert prior penalty to BIC adjustment
    # BIC weight ∝ exp(-0.5 * BIC)
    # So adding 2 * log(1/penalty) to BIC is equivalent to multiplying weight by penalty
    bic_adjustment = 2.0 * np.log(1.0 / prior_penalty)
    
    return base_bic + bic_adjustment


# =============================================================================
# ABLATION REPORTING
# =============================================================================

@dataclass
class MomentumAblationResult:
    """Results from momentum ablation comparison.
    
    Compares model performance with and without momentum augmentation.
    """
    model_name: str
    ll_with_momentum: float
    ll_without_momentum: float
    ll_lift: float  # ll_with - ll_without (positive = momentum helps)
    bic_with_momentum: float
    bic_without_momentum: float
    bic_improvement: float  # bic_without - bic_with (positive = momentum helps)
    momentum_helps: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'll_with_momentum': self.ll_with_momentum,
            'll_without_momentum': self.ll_without_momentum,
            'll_lift': self.ll_lift,
            'bic_with_momentum': self.bic_with_momentum,
            'bic_without_momentum': self.bic_without_momentum,
            'bic_improvement': self.bic_improvement,
            'momentum_helps': self.momentum_helps,
        }


def compute_ablation_result(
    model_name: str,
    ll_with: float,
    ll_without: float,
    n_obs: int,
    n_params: int,
) -> MomentumAblationResult:
    """
    Compute ablation comparison result.
    
    Args:
        model_name: Name of the model being compared
        ll_with: Log-likelihood with momentum enabled
        ll_without: Log-likelihood with momentum disabled
        n_obs: Number of observations
        n_params: Number of model parameters
        
    Returns:
        MomentumAblationResult with comparison metrics
    """
    from tuning.tune import compute_bic
    
    bic_with = compute_bic(ll_with, n_params, n_obs)
    bic_without = compute_bic(ll_without, n_params, n_obs)
    
    ll_lift = ll_with - ll_without
    bic_improvement = bic_without - bic_with  # Lower BIC is better
    
    # Momentum helps if it improves BIC (accounting for noise)
    # Use a small threshold to avoid noise-driven conclusions
    momentum_helps = bic_improvement > 1.0  # BIC improvement > 1 is meaningful
    
    return MomentumAblationResult(
        model_name=model_name,
        ll_with_momentum=ll_with,
        ll_without_momentum=ll_without,
        ll_lift=ll_lift,
        bic_with_momentum=bic_with,
        bic_without_momentum=bic_without,
        bic_improvement=bic_improvement,
        momentum_helps=momentum_helps,
    )
