#!/usr/bin/env python3
"""
===============================================================================
GENERALIZED HYPERBOLIC (GH) DISTRIBUTION FOR CALIBRATION IMPROVEMENT
===============================================================================

This module implements the Generalized Hyperbolic distribution as a fallback
model for assets that fail PIT calibration with standard φ-t models.

GH DISTRIBUTION:
    The GH distribution is a flexible 5-parameter family that includes:
    - Student-t (λ = -ν/2, α → 0, β = 0)
    - Normal-Inverse-Gaussian (λ = -1/2)
    - Variance-Gamma (δ → 0)
    - Hyperbolic (λ = 1)

    Key advantage: β parameter captures SKEWNESS that Student-t cannot.

PARAMETERS:
    λ (lambda) - Tail behavior index (shape)
    α (alpha)  - Tail decay rate (must be > |β|)
    β (beta)   - Skewness parameter (β > 0 = right skew, β < 0 = left skew)
    δ (delta)  - Scale parameter (δ > 0)
    μ (mu)     - Location parameter

USAGE:
    This model is ONLY attempted when:
    1. Standard φ-t model fails PIT calibration (p < 0.05)
    2. K=2 mixture model fails or is not selected
    3. Adaptive ν refinement fails to improve calibration

    Selection criterion:
    - GH is selected if it improves PIT p-value AND BIC is within threshold

INTEGRATION:
    - tune_q_mle.py: Fits GH after other escalation methods fail
    - tune_pretty.py: Displays GH selection statistics
    - fx_pln_jpy_signals.py: Uses GH CDF for probability computation

===============================================================================
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize, differential_evolution
from scipy.stats import kstest
from scipy.special import kv as bessel_kv  # Modified Bessel function of second kind
import warnings


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class GHModelConfig:
    """Configuration for Generalized Hyperbolic model fitting."""
    
    # Enable/disable GH fallback
    enabled: bool = True
    
    # Only attempt GH when PIT p-value is below this threshold
    pit_threshold: float = 0.05
    
    # BIC penalty threshold: GH must beat current model by this margin
    # Set to negative to allow GH even if BIC is slightly worse (for calibration)
    bic_threshold: float = -10.0  # Allow up to 10 BIC worse if PIT improves
    
    # PIT improvement required to select GH (ratio of new/old p-value)
    pit_improvement_factor: float = 2.0  # Must at least double p-value
    
    # Parameter bounds for optimization
    lambda_bounds: Tuple[float, float] = (-10.0, 10.0)
    alpha_bounds: Tuple[float, float] = (0.01, 50.0)
    beta_bounds: Tuple[float, float] = (-10.0, 10.0)
    delta_bounds: Tuple[float, float] = (0.001, 10.0)
    
    # Optimization settings
    max_iter: int = 500
    use_global_optimization: bool = True  # Use differential evolution
    
    # Regularization to prevent extreme parameters
    regularization_strength: float = 0.01
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config to dictionary."""
        return {
            'enabled': self.enabled,
            'pit_threshold': self.pit_threshold,
            'bic_threshold': self.bic_threshold,
            'pit_improvement_factor': self.pit_improvement_factor,
            'lambda_bounds': list(self.lambda_bounds),
            'alpha_bounds': list(self.alpha_bounds),
            'beta_bounds': list(self.beta_bounds),
            'delta_bounds': list(self.delta_bounds),
            'max_iter': self.max_iter,
            'use_global_optimization': self.use_global_optimization,
        }


# Default configuration
DEFAULT_GH_CONFIG = GHModelConfig()


# =============================================================================
# GH DISTRIBUTION FUNCTIONS
# =============================================================================

def gh_log_pdf(x: np.ndarray, lam: float, alpha: float, beta: float, 
               delta: float, mu: float = 0.0) -> np.ndarray:
    """
    Compute log-PDF of the Generalized Hyperbolic distribution.
    
    The GH density is:
        f(x) = C * (δ² + (x-μ)²)^((λ-1/2)/2) * K_{λ-1/2}(α*√(δ² + (x-μ)²)) * exp(β(x-μ))
    
    where:
        C = (α²-β²)^(λ/2) / (√(2π) * α^(λ-1/2) * δ^λ * K_λ(δ*√(α²-β²)))
        K_ν(x) = modified Bessel function of the second kind
    
    Args:
        x: Data points
        lam: Shape parameter (lambda)
        alpha: Tail decay rate (must be > |beta|)
        beta: Skewness parameter
        delta: Scale parameter (> 0)
        mu: Location parameter
        
    Returns:
        Log-density values
    """
    # Ensure numerical stability
    alpha = max(alpha, abs(beta) + 1e-6)
    delta = max(delta, 1e-6)
    
    # Centered data
    z = x - mu
    
    # Compute gamma = sqrt(alpha^2 - beta^2)
    gamma_sq = alpha**2 - beta**2
    if gamma_sq <= 0:
        return np.full_like(x, -np.inf)
    gamma = np.sqrt(gamma_sq)
    
    # Compute q(z) = sqrt(delta^2 + z^2)
    q = np.sqrt(delta**2 + z**2)
    
    # Log normalization constant
    # C = (gamma)^lambda / (sqrt(2*pi) * alpha^(lambda-0.5) * delta^lambda * K_lambda(delta*gamma))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            bessel_norm = bessel_kv(lam, delta * gamma)
            if bessel_norm <= 0 or not np.isfinite(bessel_norm):
                bessel_norm = 1e-300
        except:
            bessel_norm = 1e-300
    
    log_C = (lam * np.log(gamma) 
             - 0.5 * np.log(2 * np.pi) 
             - (lam - 0.5) * np.log(alpha) 
             - lam * np.log(delta) 
             - np.log(bessel_norm))
    
    # Log kernel
    # (q)^(lambda - 0.5) * K_{lambda-0.5}(alpha * q) * exp(beta * z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            bessel_kernel = bessel_kv(lam - 0.5, alpha * q)
            bessel_kernel = np.where(bessel_kernel > 0, bessel_kernel, 1e-300)
            bessel_kernel = np.where(np.isfinite(bessel_kernel), bessel_kernel, 1e-300)
        except:
            bessel_kernel = np.full_like(q, 1e-300)
    
    log_kernel = ((lam - 0.5) * np.log(q) 
                  + np.log(bessel_kernel) 
                  + beta * z)
    
    log_pdf = log_C + log_kernel
    
    # Handle numerical issues
    log_pdf = np.where(np.isfinite(log_pdf), log_pdf, -1e10)
    
    return log_pdf


def gh_cdf(x: np.ndarray, lam: float, alpha: float, beta: float,
           delta: float, mu: float = 0.0, n_points: int = 1000) -> np.ndarray:
    """
    Compute CDF of the Generalized Hyperbolic distribution via numerical integration.
    
    Uses trapezoidal integration from -inf to x.
    
    Args:
        x: Points at which to evaluate CDF
        lam, alpha, beta, delta, mu: GH parameters
        n_points: Number of integration points
        
    Returns:
        CDF values
    """
    # Determine integration range
    # GH tails decay exponentially, so we can truncate
    x_min = mu - 10 * delta
    x_max = mu + 10 * delta
    
    # Create integration grid
    grid = np.linspace(x_min, x_max, n_points)
    dx = grid[1] - grid[0]
    
    # Compute PDF on grid
    log_pdf = gh_log_pdf(grid, lam, alpha, beta, delta, mu)
    pdf = np.exp(log_pdf)
    
    # Normalize (numerical integration may not sum to 1)
    pdf_sum = np.trapz(pdf, grid)
    if pdf_sum > 0:
        pdf = pdf / pdf_sum
    
    # Compute CDF via cumulative sum
    cdf_grid = np.cumsum(pdf) * dx
    cdf_grid = cdf_grid / cdf_grid[-1]  # Ensure CDF ends at 1
    
    # Interpolate to requested points
    cdf_values = np.interp(x, grid, cdf_grid)
    
    # Clamp to [0, 1]
    cdf_values = np.clip(cdf_values, 0.0, 1.0)
    
    return cdf_values


# =============================================================================
# GH MODEL FITTING
# =============================================================================

@dataclass
class GHModelResult:
    """Result of GH model fitting."""
    
    # Fitted parameters
    lam: float      # Lambda (shape)
    alpha: float    # Tail decay
    beta: float     # Skewness
    delta: float    # Scale
    mu: float       # Location (usually 0 for standardized residuals)
    
    # Fit quality metrics
    log_likelihood: float
    bic: float
    aic: float
    n_obs: int
    
    # Calibration metrics
    pit_ks_pvalue: float
    ks_statistic: float
    
    # Fields with default values must come last
    n_params: int = 5
    
    # Comparison with previous model
    bic_improvement: float = 0.0
    pit_improvement_ratio: float = 1.0
    
    # Interpretation
    skewness_direction: str = "symmetric"  # "left", "right", or "symmetric"
    tail_behavior: str = "medium"  # "light", "medium", "heavy"
    
    @property
    def is_calibrated(self) -> bool:
        """Check if model passes calibration."""
        return self.pit_ks_pvalue >= 0.05
    
    def to_dict(self) -> Dict[str, Any]:
        """Export result to dictionary."""
        return {
            'model_type': 'generalized_hyperbolic',
            'parameters': {
                'lambda': float(self.lam),
                'alpha': float(self.alpha),
                'beta': float(self.beta),
                'delta': float(self.delta),
                'mu': float(self.mu),
            },
            'log_likelihood': float(self.log_likelihood),
            'bic': float(self.bic),
            'aic': float(self.aic),
            'n_obs': int(self.n_obs),
            'n_params': int(self.n_params),
            'pit_ks_pvalue': float(self.pit_ks_pvalue),
            'ks_statistic': float(self.ks_statistic),
            'bic_improvement': float(self.bic_improvement),
            'pit_improvement_ratio': float(self.pit_improvement_ratio),
            'skewness_direction': self.skewness_direction,
            'tail_behavior': self.tail_behavior,
            'is_calibrated': self.is_calibrated,
        }


class GHModel:
    """
    Generalized Hyperbolic model for calibration improvement.
    
    This model is used as a fallback when standard φ-t models fail
    PIT calibration. It captures skewness that Student-t cannot.
    
    Usage:
        model = GHModel(config)
        result = model.fit(standardized_residuals)
        if result.is_calibrated:
            # Use GH model
    """
    
    def __init__(self, config: Optional[GHModelConfig] = None):
        """Initialize GH model with configuration."""
        self.config = config or DEFAULT_GH_CONFIG
    
    def _neg_log_likelihood(self, params: np.ndarray, z: np.ndarray) -> float:
        """
        Negative log-likelihood for optimization.
        
        Args:
            params: [lambda, alpha, beta, delta]
            z: Standardized residuals
            
        Returns:
            Negative log-likelihood (for minimization)
        """
        lam, alpha, beta, delta = params
        
        # Constraint: alpha > |beta|
        if alpha <= abs(beta):
            return 1e10
        
        # Constraint: delta > 0
        if delta <= 0:
            return 1e10
        
        # Compute log-likelihood
        log_pdf = gh_log_pdf(z, lam, alpha, beta, delta, mu=0.0)
        ll = np.sum(log_pdf)
        
        # Regularization to prevent extreme parameters
        reg = self.config.regularization_strength * (
            lam**2 + (alpha - 1)**2 + beta**2 + (delta - 1)**2
        )
        
        # Return negative (for minimization)
        nll = -ll + reg
        
        if not np.isfinite(nll):
            return 1e10
        
        return nll
    
    def _compute_pit(self, z: np.ndarray, lam: float, alpha: float, 
                     beta: float, delta: float) -> Tuple[np.ndarray, float, float]:
        """
        Compute PIT values and KS test for fitted GH model.
        
        Args:
            z: Standardized residuals
            lam, alpha, beta, delta: GH parameters
            
        Returns:
            (pit_values, ks_statistic, ks_pvalue)
        """
        # Compute CDF at each point
        pit_values = gh_cdf(z, lam, alpha, beta, delta, mu=0.0)
        
        # KS test against uniform
        ks_result = kstest(pit_values, 'uniform')
        
        return pit_values, ks_result.statistic, ks_result.pvalue
    
    def _interpret_parameters(self, lam: float, alpha: float, beta: float, 
                              delta: float) -> Tuple[str, str]:
        """
        Interpret GH parameters for human readability.
        
        Returns:
            (skewness_direction, tail_behavior)
        """
        # Skewness direction based on beta
        if beta > 0.5:
            skewness = "right"
        elif beta < -0.5:
            skewness = "left"
        else:
            skewness = "symmetric"
        
        # Tail behavior based on lambda and alpha
        # Lower alpha = heavier tails
        # More negative lambda = heavier tails
        if alpha < 1.0 or lam < -2:
            tails = "heavy"
        elif alpha > 5.0 and lam > 0:
            tails = "light"
        else:
            tails = "medium"
        
        return skewness, tails
    
    def fit(self, z: np.ndarray, 
            single_bic: Optional[float] = None,
            single_pit_pvalue: Optional[float] = None) -> Optional[GHModelResult]:
        """
        Fit GH distribution to standardized residuals.
        
        Args:
            z: Standardized residuals (should be roughly mean 0, variance ~1)
            single_bic: BIC of the single (non-GH) model for comparison
            single_pit_pvalue: PIT p-value of single model for comparison
            
        Returns:
            GHModelResult if fitting succeeds, None otherwise
        """
        n = len(z)
        if n < 50:
            return None  # Not enough data
        
        # Remove any NaN/Inf
        z = z[np.isfinite(z)]
        n = len(z)
        if n < 50:
            return None
        
        # Initial parameter estimates based on data moments
        z_mean = np.mean(z)
        z_std = np.std(z)
        z_skew = np.mean(((z - z_mean) / z_std) ** 3) if z_std > 0 else 0
        z_kurt = np.mean(((z - z_mean) / z_std) ** 4) - 3 if z_std > 0 else 0
        
        # Initial guesses
        lam_init = -0.5  # Start near NIG
        alpha_init = 1.0 / max(z_std, 0.1)
        beta_init = np.clip(z_skew * 0.3, -5, 5)  # Skewness hint
        delta_init = z_std
        
        x0 = np.array([lam_init, alpha_init, beta_init, delta_init])
        
        # Bounds
        bounds = [
            self.config.lambda_bounds,
            self.config.alpha_bounds,
            self.config.beta_bounds,
            self.config.delta_bounds,
        ]
        
        try:
            if self.config.use_global_optimization:
                # Use differential evolution for global optimization
                result = differential_evolution(
                    self._neg_log_likelihood,
                    bounds=bounds,
                    args=(z,),
                    maxiter=self.config.max_iter,
                    seed=42,
                    polish=True,  # Refine with L-BFGS-B
                    workers=1,
                )
            else:
                # Local optimization only
                result = minimize(
                    self._neg_log_likelihood,
                    x0,
                    args=(z,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': self.config.max_iter}
                )
            
            if not result.success and result.fun > 1e9:
                return None
            
            lam, alpha, beta, delta = result.x
            
            # Ensure constraint
            alpha = max(alpha, abs(beta) + 0.01)
            
        except Exception as e:
            return None
        
        # Compute final log-likelihood
        log_pdf = gh_log_pdf(z, lam, alpha, beta, delta, mu=0.0)
        log_likelihood = np.sum(log_pdf)
        
        if not np.isfinite(log_likelihood):
            return None
        
        # Compute BIC and AIC
        k = 5  # Number of parameters (lambda, alpha, beta, delta, mu=0 fixed)
        bic = -2 * log_likelihood + k * np.log(n)
        aic = -2 * log_likelihood + 2 * k
        
        # Compute PIT
        pit_values, ks_stat, ks_pvalue = self._compute_pit(z, lam, alpha, beta, delta)
        
        # Comparison metrics
        bic_improvement = (single_bic - bic) if single_bic is not None else 0.0
        pit_improvement_ratio = (ks_pvalue / single_pit_pvalue) if (single_pit_pvalue and single_pit_pvalue > 0) else 1.0
        
        # Interpret parameters
        skewness_dir, tail_behavior = self._interpret_parameters(lam, alpha, beta, delta)
        
        return GHModelResult(
            lam=lam,
            alpha=alpha,
            beta=beta,
            delta=delta,
            mu=0.0,
            log_likelihood=log_likelihood,
            bic=bic,
            aic=aic,
            n_obs=n,
            pit_ks_pvalue=ks_pvalue,
            ks_statistic=ks_stat,
            bic_improvement=bic_improvement,
            pit_improvement_ratio=pit_improvement_ratio,
            skewness_direction=skewness_dir,
            tail_behavior=tail_behavior,
        )


# =============================================================================
# HELPER FUNCTIONS FOR INTEGRATION
# =============================================================================

def should_attempt_gh(
    pit_ks_pvalue: float,
    mixture_attempted: bool,
    mixture_selected: bool,
    nu_refinement_attempted: bool,
    nu_refinement_improved: bool,
    config: Optional[GHModelConfig] = None
) -> bool:
    """
    Determine if GH model should be attempted.
    
    GH is attempted when:
    1. Current model fails PIT calibration
    2. Previous escalation methods have been tried and failed
    
    Args:
        pit_ks_pvalue: Current PIT p-value
        mixture_attempted: DEPRECATED - K=2 mixture removed (always False for new runs)
        mixture_selected: DEPRECATED - K=2 mixture removed (always False for new runs)
        nu_refinement_attempted: Whether ν refinement was attempted
        nu_refinement_improved: Whether ν refinement improved calibration
        config: GH model configuration
        
    Returns:
        True if GH should be attempted
        
    Note:
        K=2 mixture was removed after empirical evaluation (206 attempts, 0 selections).
        The mixture_attempted/mixture_selected params are kept for backward compatibility.
    """
    config = config or DEFAULT_GH_CONFIG
    
    if not config.enabled:
        return False
    
    # Only attempt if calibration fails
    if pit_ks_pvalue >= config.pit_threshold:
        return False
    
    # K=2 mixture check removed - feature empirically falsified
    # Legacy cached results may still have mixture_selected=True
    if mixture_selected:
        # Mixture was selected but still failing - try GH
        return True
    
    if nu_refinement_improved and pit_ks_pvalue >= 0.01:
        # ν refinement helped significantly - don't need GH
        return False
    
    # Default: attempt GH for any remaining calibration failure
    return True


def should_select_gh(
    gh_result: GHModelResult,
    single_pit_pvalue: float,
    config: Optional[GHModelConfig] = None
) -> bool:
    """
    Determine if GH model should be selected over current model.
    
    Selection criteria:
    1. GH must improve PIT p-value by required factor
    2. BIC penalty must be within threshold
    
    Args:
        gh_result: Result from GH model fitting
        single_pit_pvalue: PIT p-value of current best model
        config: GH model configuration
        
    Returns:
        True if GH should be selected
    """
    config = config or DEFAULT_GH_CONFIG
    
    # Must improve PIT
    if gh_result.pit_ks_pvalue <= single_pit_pvalue:
        return False
    
    # Must meet improvement threshold
    if gh_result.pit_improvement_ratio < config.pit_improvement_factor:
        # Exception: if GH achieves calibration and single doesn't
        if gh_result.is_calibrated and single_pit_pvalue < 0.05:
            return True
        return False
    
    # BIC must be within threshold (negative threshold allows slightly worse BIC)
    if gh_result.bic_improvement < config.bic_threshold:
        # Exception: if GH achieves calibration
        if gh_result.is_calibrated:
            return True
        return False
    
    return True


def compute_gh_probability(
    x: float,
    lam: float,
    alpha: float, 
    beta: float,
    delta: float,
    mu: float = 0.0
) -> float:
    """
    Compute P(X > 0) for GH distribution.
    
    Used in signal generation for probability of positive return.
    
    Args:
        x: Threshold (usually 0 for P(return > 0))
        lam, alpha, beta, delta, mu: GH parameters
        
    Returns:
        Probability P(X > x)
    """
    cdf_at_x = gh_cdf(np.array([x]), lam, alpha, beta, delta, mu)[0]
    return 1.0 - cdf_at_x


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'GHModelConfig',
    'DEFAULT_GH_CONFIG',
    'GHModelResult',
    'GHModel',
    'gh_log_pdf',
    'gh_cdf',
    'should_attempt_gh',
    'should_select_gh',
    'compute_gh_probability',
]
