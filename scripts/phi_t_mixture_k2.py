#!/usr/bin/env python3
"""
===============================================================================
DEPRECATED: K=2 MIXTURE OF SYMMETRIC φ-t MODELS
===============================================================================

** THIS MODULE IS NO LONGER USED IN PRODUCTION **

Empirical evaluation showed 0% selection rate (206 attempts, 0 selections).
The model was removed from the tuning pipeline because:
  1. Returns are fat-tailed unimodal, not bimodal
  2. HMM regime-switching + Student-t already captures regime heterogeneity
  3. The GMM approach duplicates regime separation without state transitions

This file is kept for:
  - Historical reference
  - Backward compatibility with cached results that may contain mixture data
  - Research experimentation (not production)

For production calibration, use:
  - HMM regime detection (regime_used, regime_source)
  - Student-t tail modeling (nu_hat)
  - Adaptive ν refinement (adaptive_nu_refinement.py)
  - Generalized Hyperbolic for skewed assets (gh_distribution.py)

See: docs/CALIBRATION_SOLUTIONS_ANALYSIS.md for decision rationale.
===============================================================================

-------------------------------------------------------------------------------
ORIGINAL DOCUMENTATION (HISTORICAL)
-------------------------------------------------------------------------------

This module implements an identifiability-safe K=2 mixture of symmetric φ-t
models to address PIT calibration failures.

DESIGN PRINCIPLES:
    • Asymmetry emerges from geometry (σ dispersion), not parameters
    • φ models persistence (shared across components)
    • ν models tail thickness (shared across components, from existing grid)
    • Mixture models regime coexistence (calm vs stress)

SCOPE (HARD LIMITS):
    • K = 2 only (no K=3, no automatic expansion)
    • Shared φ across components
    • Shared ν across components (from existing STUDENT_T_NU_GRID)
    • Only σ differs between components
    • Static weights per fit (rolling refit allowed, no intraperiod dynamics)

MODEL DEFINITION:
    rₜ = φ·rₜ₋₁ + εₜ  (AR(1) return dynamics)
    εₜ ~ w·T(0, σ_A·volₜ, ν) + (1-w)·T(0, σ_B·volₜ, ν)  (mixture innovations)

    Where:
        • φ is shared (AR(1) coefficient)
        • ν ∈ existing ν grid
        • σ_B ≥ 1.5 × σ_A (hard constraint)
        • w ∈ [0.1, 0.9] (prevents degenerate solutions)
    
    CRITICAL: Likelihood and PIT are computed on INNOVATIONS εₜ = rₜ - φ·rₜ₋₁,
    NOT on raw returns or Kalman-filtered drift.

INTERPRETATION:
    • Component A = calm regime (lower volatility)
    • Component B = stress/tail regime (higher volatility)

NON-GOALS (EXPLICITLY OUT OF SCOPE):
    ❌ No asymmetric φ parameters
    ❌ No time-varying weights
    ❌ No HMMs or regime switching logic
    ❌ No direct optimization of KS p-values
    ❌ No K > 2 mixtures

===============================================================================
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.special import gammaln
from scipy.stats import kstest, t as student_t


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PhiTMixtureK2Config:
    """
    Configuration for K=2 mixture of symmetric φ-t models.
    
    All parameters have conservative defaults that prioritize stability
    over aggressive calibration improvement.
    """
    # Feature toggle
    enabled: bool = True
    
    # Mixture weight bounds [prevents degenerate single-component solutions]
    min_weight: float = 0.1
    max_weight: float = 0.9
    
    # Minimum σ ratio between components [prevents component collapse]
    sigma_ratio_min: float = 1.5
    sigma_ratio_max: float = 5.0
    
    # Entropy regularization [DISABLED: was incorrectly encouraging w=0.5]
    # For calm/stress regime model, we expect w_calm ≈ 0.7-0.8, not 0.5
    # Setting to 0 disables the entropy term entirely
    entropy_penalty: float = 0.0
    
    # BIC threshold [mixture must beat single model by this margin]
    bic_threshold: float = 0.0
    
    # Optimization settings
    max_iterations: int = 500
    convergence_tol: float = 1e-6
    
    # φ bounds (inherited from existing system)
    phi_min: float = -0.999
    phi_max: float = 0.999
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'PhiTMixtureK2Config':
        """Create config from dictionary (e.g., from YAML)."""
        return cls(**{k: v for k, v in config.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config to dictionary."""
        return {
            'enabled': self.enabled,
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'sigma_ratio_min': self.sigma_ratio_min,
            'sigma_ratio_max': self.sigma_ratio_max,
            'entropy_penalty': self.entropy_penalty,
            'bic_threshold': self.bic_threshold,
            'max_iterations': self.max_iterations,
            'convergence_tol': self.convergence_tol,
            'phi_min': self.phi_min,
            'phi_max': self.phi_max,
        }


# Default global configuration
DEFAULT_MIXTURE_CONFIG = PhiTMixtureK2Config()


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class PhiTMixtureK2Result:
    """
    Fit result for K=2 φ-t mixture.
    
    This dataclass stores all parameters and diagnostics from a successful
    mixture model fit.
    """
    # Model parameters
    phi: float                  # Shared AR(1) drift persistence
    nu: float                   # Shared degrees of freedom (from grid)
    sigma_a: float             # Calm regime scale
    sigma_b: float             # Stress regime scale
    weight: float              # Weight on component A (calm)
    
    # Likelihood and model selection
    log_likelihood: float
    bic: float
    aic: float
    
    # Calibration diagnostics
    ks_statistic: float
    pit_ks_pvalue: float
    
    # Optimization diagnostics
    optimization_success: bool
    n_iterations: int
    convergence_message: str
    
    # Metadata
    n_obs: int
    n_params: int = 4  # phi, sigma_a, sigma_ratio, weight (nu is fixed from grid)
    model_type: str = "phi_t_mixture_k2"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def sigma_ratio(self) -> float:
        """Ratio of stress to calm scale."""
        return self.sigma_b / self.sigma_a if self.sigma_a > 0 else np.inf
    
    @property
    def weight_b(self) -> float:
        """Weight on stress component."""
        return 1.0 - self.weight
    
    def to_dict(self) -> Dict[str, Any]:
        """Export result to dictionary for JSON serialization."""
        return {
            'phi': float(self.phi),
            'nu': float(self.nu),
            'sigma_a': float(self.sigma_a),
            'sigma_b': float(self.sigma_b),
            'weight': float(self.weight),
            'weight_b': float(self.weight_b),
            'sigma_ratio': float(self.sigma_ratio),
            'log_likelihood': float(self.log_likelihood),
            'bic': float(self.bic),
            'aic': float(self.aic),
            'ks_statistic': float(self.ks_statistic),
            'pit_ks_pvalue': float(self.pit_ks_pvalue),
            'optimization_success': bool(self.optimization_success),
            'n_iterations': int(self.n_iterations),
            'convergence_message': str(self.convergence_message),
            'n_obs': int(self.n_obs),
            'n_params': int(self.n_params),
            'model_type': str(self.model_type),
            'timestamp': str(self.timestamp),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhiTMixtureK2Result':
        """Create result from dictionary."""
        return cls(
            phi=data['phi'],
            nu=data['nu'],
            sigma_a=data['sigma_a'],
            sigma_b=data['sigma_b'],
            weight=data['weight'],
            log_likelihood=data['log_likelihood'],
            bic=data['bic'],
            aic=data.get('aic', data['bic']),  # Backward compat
            ks_statistic=data.get('ks_statistic', 0.0),
            pit_ks_pvalue=data.get('pit_ks_pvalue', 1.0),
            optimization_success=data.get('optimization_success', True),
            n_iterations=data.get('n_iterations', 0),
            convergence_message=data.get('convergence_message', ''),
            n_obs=data.get('n_obs', 0),
            n_params=data.get('n_params', 5),
            model_type=data.get('model_type', 'phi_t_mixture_k2'),
            timestamp=data.get('timestamp', ''),
        )


# =============================================================================
# MAIN MODEL CLASS
# =============================================================================

class PhiTMixtureK2:
    """
    K=2 mixture of symmetric φ-t models with shared φ and ν.
    
    Asymmetry emerges from σ dispersion, not parameter asymmetry.
    
    This implementation:
    1. Uses joint MLE (not free EM) for stability
    2. Warm-starts from single φ-t fit
    3. Enforces identifiability constraints
    4. Computes PIT using full mixture CDF
    
    Example usage:
        config = PhiTMixtureK2Config(enabled=True, sigma_ratio_min=1.5)
        mixer = PhiTMixtureK2(config)
        
        # Warm-start from single model
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=8.0,  # From grid search
            phi_init=0.1,
            sigma_init=0.02
        )
        
        if result is not None:
            pit = mixer.compute_pit(returns, vol, result)
            print(f"KS p-value: {result.pit_ks_pvalue:.4f}")
    """
    
    def __init__(self, config: Optional[PhiTMixtureK2Config] = None):
        """
        Initialize mixture model with configuration.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or DEFAULT_MIXTURE_CONFIG
    
    # =========================================================================
    # CORE LIKELIHOOD FUNCTIONS
    # =========================================================================
    
    @staticmethod
    def _student_t_logpdf(
        x: np.ndarray, 
        mu: np.ndarray,
        sigma: float, 
        nu: float
    ) -> np.ndarray:
        """
        Log-density of Student-t with location mu and scale sigma.
        
        Vectorized for efficiency.
        
        Args:
            x: Observations (n,)
            mu: Location parameters (n,)
            sigma: Scale parameter (scalar > 0)
            nu: Degrees of freedom (scalar > 2)
            
        Returns:
            Log-densities (n,)
        """
        if sigma <= 0 or nu <= 0:
            return np.full_like(x, -1e12)
        
        z = (x - mu) / sigma
        
        log_norm = (
            gammaln((nu + 1.0) / 2.0) 
            - gammaln(nu / 2.0) 
            - 0.5 * np.log(nu * np.pi) 
            - np.log(sigma)
        )
        log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + (z ** 2) / nu)
        
        return log_norm + log_kernel
    
    @staticmethod
    def _compute_ar1_innovations(returns: np.ndarray, phi: float) -> np.ndarray:
        """
        Compute AR(1) innovations: εₜ = rₜ - φ·rₜ₋₁
        
        This is the CORRECT way to compute innovations for an AR(1) model.
        The first observation is dropped (no valid innovation for t=0).
        
        Model: rₜ = φ·rₜ₋₁ + εₜ
        Innovation: εₜ = rₜ - φ·rₜ₋₁
        
        Args:
            returns: Log returns (n,)
            phi: AR(1) persistence coefficient
            
        Returns:
            Innovations εₜ (n-1,) - first observation dropped
        """
        phi_val = float(np.clip(phi, -0.999, 0.999))
        # εₜ = rₜ - φ·rₜ₋₁ for t = 1, 2, ..., n-1
        innovations = returns[1:] - phi_val * returns[:-1]
        return innovations
    
    def _mixture_negative_log_likelihood(
        self,
        params: np.ndarray,
        returns: np.ndarray,
        vol: np.ndarray,
        nu: float
    ) -> float:
        """
        Negative log-likelihood for K=2 mixture using AR(1) INNOVATIONS.
        
        CORRECT MODEL:
            rₜ = φ·rₜ₋₁ + εₜ
            εₜ ~ w·T(0, σ_A·volₜ, ν) + (1-w)·T(0, σ_B·volₜ, ν)
        
        The likelihood is computed on INNOVATIONS, not raw returns.
        
        Parameters are in transformed space:
            params[0]: phi (AR(1) coefficient)
            params[1]: log(sigma_a)
            params[2]: log(sigma_ratio) where sigma_b = sigma_a * exp(log_ratio)
            params[3]: logit(weight) where weight = sigmoid(logit_w)
        
        Args:
            params: Transformed parameters [phi, log_sigma_a, log_sigma_ratio, logit_w]
            returns: Log returns (n,)
            vol: Volatility estimates (n,)
            nu: Degrees of freedom (fixed)
            
        Returns:
            Negative log-likelihood (scalar)
        """
        phi, log_sigma_a, log_sigma_ratio, logit_w = params
        
        # Transform parameters
        sigma_a = np.exp(log_sigma_a)
        sigma_b = sigma_a * np.exp(log_sigma_ratio)
        w = 1.0 / (1.0 + np.exp(-logit_w))  # Sigmoid
        
        # Validate
        if sigma_a <= 0 or sigma_b <= 0 or not (0 < w < 1):
            return 1e12
        
        # CRITICAL: Compute AR(1) innovations, NOT Kalman filtered drift
        # εₜ = rₜ - φ·rₜ₋₁
        innovations = self._compute_ar1_innovations(returns, phi)
        
        # Align volatility with innovations (drop first observation)
        aligned_vol = vol[1:]
        
        # Effective scales for each component
        effective_sigma_a = sigma_a * aligned_vol
        effective_sigma_b = sigma_b * aligned_vol
        
        n_innovations = len(innovations)
        
        # Vectorized log-pdf computation on INNOVATIONS
        # For Student-t: log p(ε | 0, σ, ν) 
        z_a = innovations / effective_sigma_a
        z_b = innovations / effective_sigma_b
        
        # Log-density of standardized Student-t
        log_norm_a = (
            gammaln((nu + 1.0) / 2.0) 
            - gammaln(nu / 2.0) 
            - 0.5 * np.log(nu * np.pi) 
            - np.log(effective_sigma_a)
        )
        log_kernel_a = -((nu + 1.0) / 2.0) * np.log(1.0 + (z_a ** 2) / nu)
        log_p_a = log_norm_a + log_kernel_a
        
        log_norm_b = (
            gammaln((nu + 1.0) / 2.0) 
            - gammaln(nu / 2.0) 
            - 0.5 * np.log(nu * np.pi) 
            - np.log(effective_sigma_b)
        )
        log_kernel_b = -((nu + 1.0) / 2.0) * np.log(1.0 + (z_b ** 2) / nu)
        log_p_b = log_norm_b + log_kernel_b
        
        # Handle numerical issues
        log_p_a = np.clip(log_p_a, -1e10, 0)
        log_p_b = np.clip(log_p_b, -1e10, 0)
        
        # Log-sum-exp for mixture: log(w·p_a + (1-w)·p_b)
        log_w = np.log(w + 1e-15)
        log_1mw = np.log(1.0 - w + 1e-15)
        
        log_mixture = np.logaddexp(log_w + log_p_a, log_1mw + log_p_b)
        
        # Total negative log-likelihood
        nll = -np.sum(log_mixture)
        
        # Entropy regularization on weights (encourages balanced mixture)
        entropy = -(w * np.log(w + 1e-10) + (1.0 - w) * np.log(1.0 - w + 1e-10))
        nll -= self.config.entropy_penalty * n_innovations * entropy
        
        # Check for NaN/Inf
        if not np.isfinite(nll):
            return 1e12
        
        return float(nll)
    
    # =========================================================================
    # FITTING
    # =========================================================================
    
    def fit(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        nu: float,
        phi_init: float,
        sigma_init: float,
        q_init: float = 1e-6
    ) -> Optional[PhiTMixtureK2Result]:
        """
        Fit K=2 mixture given fixed ν (from grid search).
        
        CORRECT MODEL:
            rₜ = φ·rₜ₋₁ + εₜ
            εₜ ~ w·T(0, σ_A·volₜ, ν) + (1-w)·T(0, σ_B·volₜ, ν)
        
        Likelihood is computed on n-1 INNOVATIONS (first observation dropped).
        
        Warm-starts from single φ-t fit parameters.
        
        Args:
            returns: Log returns (n,)
            vol: Volatility estimates (n,)
            nu: Degrees of freedom (fixed, from STUDENT_T_NU_GRID)
            phi_init: Initial φ from single model fit
            sigma_init: Initial σ from single model fit (used for σ_A)
            q_init: Not used (kept for API compatibility)
            
        Returns:
            PhiTMixtureK2Result if optimization succeeds, None otherwise
        """
        n = len(returns)
        
        if n < 50:  # Minimum observations for mixture
            return None
        
        # Ensure arrays are proper numpy arrays
        returns = np.asarray(returns).flatten()
        vol = np.asarray(vol).flatten()
        
        # Validate inputs
        if len(returns) != len(vol):
            return None
        if not np.all(np.isfinite(returns)):
            returns = returns[np.isfinite(returns)]
            vol = vol[:len(returns)]
        
        # Initial parameters in transformed space
        # sigma_a = sigma_init, sigma_b = ratio * sigma_a, w = 0.5
        init_params = np.array([
            float(np.clip(phi_init, self.config.phi_min, self.config.phi_max)),
            np.log(max(sigma_init, 1e-6)),
            np.log(self.config.sigma_ratio_min),  # Start at minimum ratio
            0.0  # logit(0.5) = 0
        ])
        
        # Bounds in transformed space
        # logit bounds for w ∈ [min_weight, max_weight]
        logit_min = np.log(self.config.min_weight / (1.0 - self.config.min_weight))
        logit_max = np.log(self.config.max_weight / (1.0 - self.config.max_weight))
        
        bounds = Bounds(
            lb=[self.config.phi_min, -12.0, np.log(self.config.sigma_ratio_min), logit_min],
            ub=[self.config.phi_max, 2.0, np.log(self.config.sigma_ratio_max), logit_max]
        )
        
        # Optimize
        try:
            result = minimize(
                self._mixture_negative_log_likelihood,
                init_params,
                args=(returns, vol, nu),
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': self.config.max_iterations,
                    'ftol': self.config.convergence_tol,
                }
            )
        except Exception as e:
            return None
        
        if not result.success and result.fun > 1e10:
            return None
        
        # Extract parameters
        phi, log_sigma_a, log_sigma_ratio, logit_w = result.x
        sigma_a = np.exp(log_sigma_a)
        sigma_b = sigma_a * np.exp(log_sigma_ratio)
        w = 1.0 / (1.0 + np.exp(-logit_w))
        
        # Compute final likelihood (without entropy penalty for BIC)
        log_lik = -self._mixture_negative_log_likelihood(
            result.x, returns, vol, nu
        )
        # Add back entropy penalty that was subtracted
        entropy = -(w * np.log(w + 1e-10) + (1.0 - w) * np.log(1.0 - w + 1e-10))
        log_lik -= self.config.entropy_penalty * n * entropy
        
        # Model selection criteria
        # 5 effective parameters: phi, sigma_a, sigma_ratio, weight, (nu is fixed)
        k = 5
        bic = k * np.log(n) - 2.0 * log_lik
        aic = 2.0 * k - 2.0 * log_lik
        
        # Compute PIT calibration
        pit_values = self._compute_pit_values(returns, vol, phi, sigma_a, sigma_b, w, nu)
        ks_stat, ks_pvalue = self._compute_ks_test(pit_values)
        
        return PhiTMixtureK2Result(
            phi=float(phi),
            nu=float(nu),
            sigma_a=float(sigma_a),
            sigma_b=float(sigma_b),
            weight=float(w),
            log_likelihood=float(log_lik),
            bic=float(bic),
            aic=float(aic),
            ks_statistic=float(ks_stat),
            pit_ks_pvalue=float(ks_pvalue),
            optimization_success=result.success,
            n_iterations=result.nit,
            convergence_message=result.message if hasattr(result, 'message') else '',
            n_obs=n,
        )
    
    # =========================================================================
    # CALIBRATION DIAGNOSTICS
    # =========================================================================
    
    def _compute_pit_values(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        phi: float,
        sigma_a: float,
        sigma_b: float,
        weight: float,
        nu: float
    ) -> np.ndarray:
        """
        Compute PIT values using mixture CDF on AR(1) INNOVATIONS.
        
        CORRECT PIT COMPUTATION:
            Model: rₜ = φ·rₜ₋₁ + εₜ
            Innovation: εₜ = rₜ - φ·rₜ₋₁
            PIT: Uₜ = F_ε(εₜ) where F_ε is the mixture CDF
        
        If model is well-calibrated, Uₜ ~ Uniform(0,1).
        
        Args:
            returns: Log returns (n,)
            vol: Volatility estimates (n,)
            phi: AR(1) persistence
            sigma_a: Calm regime scale
            sigma_b: Stress regime scale
            weight: Weight on calm component
            nu: Degrees of freedom
            
        Returns:
            PIT values (n-1,) - first observation dropped
        """
        # CRITICAL: Compute AR(1) innovations, NOT Kalman filtered drift
        # εₜ = rₜ - φ·rₜ₋₁
        innovations = self._compute_ar1_innovations(returns, phi)
        
        # Align volatility (drop first observation)
        aligned_vol = vol[1:]
        
        n_innovations = len(innovations)
        pit_values = np.zeros(n_innovations)
        
        for t in range(n_innovations):
            # Effective scale for each component
            effective_sigma_a = sigma_a * aligned_vol[t]
            effective_sigma_b = sigma_b * aligned_vol[t]
            
            # CORRECT: Standardize INNOVATIONS, not returns
            # z = εₜ / σₜ
            z_a = innovations[t] / effective_sigma_a
            z_b = innovations[t] / effective_sigma_b
            
            # CDF of standardized Student-t evaluated at z
            cdf_a = student_t.cdf(z_a, df=nu)
            cdf_b = student_t.cdf(z_b, df=nu)
            
            # Mixture CDF: F(ε) = w·F_A(ε) + (1-w)·F_B(ε)
            pit_values[t] = weight * cdf_a + (1.0 - weight) * cdf_b
        
        return pit_values
    
    def _compute_ks_test(self, pit_values: np.ndarray) -> Tuple[float, float]:
        """
        Compute KS test statistic and p-value against uniform distribution.
        
        Args:
            pit_values: PIT values (n,)
            
        Returns:
            (ks_statistic, p_value)
        """
        # Filter valid values
        valid = pit_values[np.isfinite(pit_values)]
        valid = valid[(valid >= 0) & (valid <= 1)]
        
        if len(valid) < 10:
            return 1.0, 0.0
        
        ks_result = kstest(valid, 'uniform')
        return float(ks_result.statistic), float(ks_result.pvalue)
    
    def compute_pit(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        result: PhiTMixtureK2Result
    ) -> np.ndarray:
        """
        Compute PIT values using fitted mixture model.
        
        Public interface for PIT computation.
        
        Args:
            returns: Log returns (n,)
            vol: Volatility estimates (n,)
            result: Fitted mixture result
            
        Returns:
            PIT values (n,)
        """
        return self._compute_pit_values(
            returns=np.asarray(returns).flatten(),
            vol=np.asarray(vol).flatten(),
            phi=result.phi,
            sigma_a=result.sigma_a,
            sigma_b=result.sigma_b,
            weight=result.weight,
            nu=result.nu
        )
    
    def compute_pit_with_kalman_drift(
        self,
        returns: np.ndarray,
        mu_filtered: np.ndarray,
        vol: np.ndarray,
        P_filtered: np.ndarray,
        c: float,
        result: PhiTMixtureK2Result
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute PIT values using Kalman-filtered drift (for comparison with single model).
        
        This method uses the SAME innovation structure as the single Kalman model:
            innovation_t = r_t - μ_t  (Kalman drift)
            forecast_var = c * vol_t² + P_t
        
        But applies mixture scales to the Student-t CDF:
            PIT = w * T_ν(z / σ_A) + (1-w) * T_ν(z / σ_B)
        
        Where z = innovation / sqrt(forecast_var) and σ_A, σ_B are relative scale factors.
        
        This enables apples-to-apples comparison with the single model PIT.
        
        Args:
            returns: Log returns (n,)
            mu_filtered: Kalman-filtered drift estimates (n,)
            vol: Volatility estimates (n,)
            P_filtered: Posterior variance from Kalman filter (n,)
            c: Observation noise multiplier
            result: Fitted mixture result
            
        Returns:
            Tuple of (pit_values, ks_statistic, ks_pvalue)
        """
        returns = np.asarray(returns).flatten()
        mu_filtered = np.asarray(mu_filtered).flatten()
        vol = np.asarray(vol).flatten()
        P_filtered = np.asarray(P_filtered).flatten()
        
        n = len(returns)
        
        # Compute forecast variance (same as single model)
        forecast_var = c * (vol ** 2) + P_filtered
        forecast_std = np.sqrt(np.maximum(forecast_var, 1e-20))
        
        # Compute innovations using Kalman drift
        innovations = returns - mu_filtered
        
        # Standardized innovations
        z = innovations / forecast_std
        
        # For the mixture, we scale the standardized innovations by the relative
        # sigma factors. The mixture says the true scale is either σ_A or σ_B
        # times the base scale, so we need to adjust the standardized values.
        #
        # If the mixture scale is σ_A, then the properly standardized value is:
        #   z_A = innovation / (σ_A * forecast_std_base)
        #       = z / σ_A  (if z = innovation / forecast_std_base)
        #
        # But we need to be careful about what forecast_std represents.
        # In the mixture model, we're saying:
        #   ε ~ w * T(0, σ_A * vol, ν) + (1-w) * T(0, σ_B * vol, ν)
        #
        # So the scale factors σ_A, σ_B are multipliers on vol, not on the full forecast_std.
        # 
        # For proper comparison, we compute:
        #   z_A = innovation / (σ_A * vol)
        #   z_B = innovation / (σ_B * vol)
        
        scale_a = result.sigma_a * vol
        scale_b = result.sigma_b * vol
        
        # Ensure numerical stability
        scale_a = np.maximum(scale_a, 1e-10)
        scale_b = np.maximum(scale_b, 1e-10)
        
        z_a = innovations / scale_a
        z_b = innovations / scale_b
        
        # Compute mixture CDF
        cdf_a = student_t.cdf(z_a, df=result.nu)
        cdf_b = student_t.cdf(z_b, df=result.nu)
        
        pit_values = result.weight * cdf_a + (1.0 - result.weight) * cdf_b
        
        # KS test
        valid = pit_values[np.isfinite(pit_values)]
        valid = valid[(valid >= 0) & (valid <= 1)]
        
        if len(valid) < 10:
            return pit_values, 1.0, 0.0
        
        ks_result = kstest(valid, 'uniform')
        
        return pit_values, float(ks_result.statistic), float(ks_result.pvalue)


# =============================================================================
# MODEL SELECTION: SINGLE vs MIXTURE
# =============================================================================

def should_use_mixture(
    single_bic: float,
    mixture_result: Optional[PhiTMixtureK2Result],
    config: PhiTMixtureK2Config = DEFAULT_MIXTURE_CONFIG
) -> bool:
    """
    Decide whether to use mixture model based on BIC comparison.
    
    The mixture must improve BIC by at least the threshold to be selected.
    This prevents overfitting when the data doesn't justify the extra complexity.
    
    Args:
        single_bic: BIC of single φ-t model
        mixture_result: Fitted mixture result (may be None if fit failed)
        config: Configuration with BIC threshold
        
    Returns:
        True if mixture should be used, False otherwise
    """
    if mixture_result is None:
        return False
    
    if not config.enabled:
        return False
    
    # BIC improvement (lower is better)
    bic_improvement = single_bic - mixture_result.bic
    
    return bic_improvement > config.bic_threshold


def fit_and_select(
    returns: np.ndarray,
    vol: np.ndarray,
    nu: float,
    phi_single: float,
    sigma_single: float,
    bic_single: float,
    config: Optional[PhiTMixtureK2Config] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Fit mixture and select between single and mixture model.
    
    This is the main entry point for integrating mixture models into the
    existing tuning pipeline.
    
    Args:
        returns: Log returns (n,)
        vol: Volatility estimates (n,)
        nu: Degrees of freedom (from grid)
        phi_single: φ from single model fit
        sigma_single: σ from single model fit
        bic_single: BIC of single model
        config: Mixture configuration (uses defaults if None)
        
    Returns:
        (model_type, result_dict) where model_type is 'single' or 'mixture'
    """
    config = config or DEFAULT_MIXTURE_CONFIG
    
    if not config.enabled:
        return 'single', {}
    
    mixer = PhiTMixtureK2(config)
    mixture_result = mixer.fit(
        returns=returns,
        vol=vol,
        nu=nu,
        phi_init=phi_single,
        sigma_init=sigma_single
    )
    
    if should_use_mixture(bic_single, mixture_result, config):
        return 'mixture', mixture_result.to_dict()
    
    return 'single', {}


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_mixture_result(result: PhiTMixtureK2Result, config: PhiTMixtureK2Config) -> List[str]:
    """
    Validate mixture result against configuration constraints.
    
    Returns list of validation errors (empty if valid).
    """
    errors = []
    
    # Check sigma ratio
    if result.sigma_ratio < config.sigma_ratio_min:
        errors.append(f"σ_B/σ_A = {result.sigma_ratio:.2f} < min {config.sigma_ratio_min}")
    
    if result.sigma_ratio > config.sigma_ratio_max:
        errors.append(f"σ_B/σ_A = {result.sigma_ratio:.2f} > max {config.sigma_ratio_max}")
    
    # Check weight bounds
    if result.weight < config.min_weight:
        errors.append(f"weight = {result.weight:.3f} < min {config.min_weight}")
    
    if result.weight > config.max_weight:
        errors.append(f"weight = {result.weight:.3f} > max {config.max_weight}")
    
    # Check phi bounds
    if result.phi < config.phi_min or result.phi > config.phi_max:
        errors.append(f"φ = {result.phi:.3f} outside [{config.phi_min}, {config.phi_max}]")
    
    # Check optimization success
    if not result.optimization_success:
        errors.append(f"Optimization did not converge: {result.convergence_message}")
    
    return errors


# =============================================================================
# DIAGNOSTIC REPORTING
# =============================================================================

def summarize_mixture_improvement(
    single_bic: float,
    single_pit_pvalue: float,
    mixture_result: Optional[PhiTMixtureK2Result]
) -> Dict[str, Any]:
    """
    Summarize the improvement from using mixture model.
    
    Args:
        single_bic: BIC of single model
        single_pit_pvalue: PIT KS p-value of single model
        mixture_result: Fitted mixture result
        
    Returns:
        Dictionary with improvement metrics
    """
    if mixture_result is None:
        return {
            'mixture_available': False,
            'bic_improvement': 0.0,
            'pit_improvement': 0.0,
            'recommendation': 'single'
        }
    
    bic_improvement = single_bic - mixture_result.bic
    pit_improvement = mixture_result.pit_ks_pvalue - single_pit_pvalue
    
    # Determine recommendation
    if bic_improvement > 0 and mixture_result.pit_ks_pvalue > 0.05:
        recommendation = 'mixture'
    elif single_pit_pvalue > 0.05:
        recommendation = 'single'  # Single model is already well-calibrated
    elif bic_improvement > 0:
        recommendation = 'mixture'  # Mixture improves BIC even if still miscalibrated
    else:
        recommendation = 'single'
    
    return {
        'mixture_available': True,
        'bic_improvement': float(bic_improvement),
        'bic_improvement_pct': float(100 * bic_improvement / abs(single_bic)) if single_bic != 0 else 0.0,
        'pit_improvement': float(pit_improvement),
        'single_pit_pvalue': float(single_pit_pvalue),
        'mixture_pit_pvalue': float(mixture_result.pit_ks_pvalue),
        'single_calibrated': single_pit_pvalue >= 0.05,
        'mixture_calibrated': mixture_result.pit_ks_pvalue >= 0.05,
        'sigma_ratio': float(mixture_result.sigma_ratio),
        'calm_weight': float(mixture_result.weight),
        'stress_weight': float(mixture_result.weight_b),
        'recommendation': recommendation
    }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'PhiTMixtureK2Config',
    'PhiTMixtureK2Result',
    'PhiTMixtureK2',
    'DEFAULT_MIXTURE_CONFIG',
    'should_use_mixture',
    'fit_and_select',
    'validate_mixture_result',
    'summarize_mixture_improvement',
]
