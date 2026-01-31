"""
===============================================================================
EVT TAIL MODULE — Extreme Value Theory for Conditional Tail Expectation
===============================================================================

Implements the Peaks Over Threshold (POT) method with Generalized Pareto 
Distribution (GPD) fitting for robust expected loss estimation.

THEORETICAL FOUNDATION:
    The Pickands–Balkema–de Haan theorem states that for a broad class of
    distributions F, the conditional excess distribution converges to GPD:
    
        lim_{u→x_F} P(X - u ≤ y | X > u) = G_{ξ,σ}(y)
    
    Where G_{ξ,σ} is the Generalized Pareto Distribution with:
        - ξ (xi/shape): Tail index (ξ > 0: heavy, ξ = 0: exponential, ξ < 0: bounded)
        - σ (sigma/scale): Scale parameter
    
CONDITIONAL TAIL EXPECTATION (CTE):
    For exceedances over threshold u, the expected loss is:
    
        E[X | X > u] = u + σ/(1-ξ)  for ξ < 1
    
    This provides principled extrapolation beyond observed data.

RELATIONSHIP TO STUDENT-T:
    Student-t with ν degrees of freedom has ξ = 1/ν:
        - ν = 4  →  ξ = 0.25 (heavy tails)
        - ν = 10 →  ξ = 0.10 (moderate tails)
        - ν → ∞  →  ξ = 0    (Gaussian-like)
    
    This provides consistency checks between existing ν estimates and GPD ξ.

INTEGRATION:
    This module is used by signal.py to compute EVT-corrected expected loss
    for position sizing, replacing the naive empirical mean of losses.

REFERENCES:
    Pickands, J. (1975). "Statistical Inference Using Extreme Order Statistics"
    Balkema, A.A. & de Haan, L. (1974). "Residual Life Time at Great Age"
    McNeil, A.J. & Frey, R. (2000). "Estimation of Tail-Related Risk Measures"

===============================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.optimize import minimize


# =============================================================================
# EVT CONSTANTS
# =============================================================================

# Threshold selection
EVT_THRESHOLD_PERCENTILE_DEFAULT = 0.90  # 90th percentile (10% exceedances)
EVT_THRESHOLD_PERCENTILE_MIN = 0.80
EVT_THRESHOLD_PERCENTILE_MAX = 0.98

# Minimum exceedances for reliable GPD fitting
EVT_MIN_EXCEEDANCES = 30  # Reduced from 50 for MC sample compatibility
EVT_RECOMMENDED_EXCEEDANCES = 100

# Shape parameter bounds
EVT_XI_MIN = -0.5  # Bounded tails (short-tailed)
EVT_XI_MAX = 1.0   # Fréchet-type (ξ ≥ 1 has infinite mean)
EVT_XI_DEFAULT = 0.1  # Moderate heavy tails

# Fallback multiplier when EVT fails
EVT_FALLBACK_MULTIPLIER = 1.5

# Conservative adjustment for small samples
EVT_SMALL_SAMPLE_ADJUSTMENT = 1.2


@dataclass
class GPDFitResult:
    """
    Results from GPD fitting via Peaks Over Threshold.
    
    Attributes:
        xi: Shape parameter (tail index)
        sigma: Scale parameter
        threshold: Threshold u for exceedances
        n_exceedances: Number of observations above threshold
        cte: Conditional Tail Expectation E[X | X > u]
        fit_success: Whether fitting succeeded
        method: Fitting method used ('mle', 'hill', 'pwm')
        diagnostics: Additional diagnostic information
    """
    xi: float
    sigma: float
    threshold: float
    n_exceedances: int
    cte: float
    fit_success: bool
    method: str
    diagnostics: Dict
    
    @property
    def is_heavy_tailed(self) -> bool:
        """Check if distribution has heavy tails (ξ > 0)."""
        return self.xi > 0.05
    
    @property
    def has_finite_mean(self) -> bool:
        """Check if tail has finite mean (ξ < 1)."""
        return self.xi < 1.0
    
    @property
    def implied_student_t_nu(self) -> Optional[float]:
        """Convert ξ to equivalent Student-t ν (for comparison)."""
        if self.xi > 0.01:
            return 1.0 / self.xi
        return None  # Gaussian-like
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "xi": float(self.xi),
            "sigma": float(self.sigma),
            "threshold": float(self.threshold),
            "n_exceedances": int(self.n_exceedances),
            "cte": float(self.cte),
            "fit_success": self.fit_success,
            "method": self.method,
            "is_heavy_tailed": self.is_heavy_tailed,
            "has_finite_mean": self.has_finite_mean,
            "implied_student_t_nu": self.implied_student_t_nu,
            "diagnostics": self.diagnostics,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'GPDFitResult':
        """Deserialize from dictionary."""
        return cls(
            xi=float(d["xi"]),
            sigma=float(d["sigma"]),
            threshold=float(d["threshold"]),
            n_exceedances=int(d["n_exceedances"]),
            cte=float(d["cte"]),
            fit_success=d["fit_success"],
            method=d.get("method", "unknown"),
            diagnostics=d.get("diagnostics", {}),
        )


def gpd_pdf(x: np.ndarray, xi: float, sigma: float) -> np.ndarray:
    """
    GPD probability density function.
    
    Args:
        x: Exceedances (must be > 0)
        xi: Shape parameter
        sigma: Scale parameter (> 0)
        
    Returns:
        PDF values
    """
    x = np.asarray(x)
    if sigma <= 0:
        return np.zeros_like(x)
    
    if abs(xi) < 1e-10:
        # Exponential case (ξ → 0)
        return (1.0 / sigma) * np.exp(-x / sigma)
    else:
        # General GPD
        z = x / sigma
        term = 1.0 + xi * z
        
        # Handle domain: 1 + ξz > 0
        valid = term > 0
        result = np.zeros_like(x, dtype=float)
        
        if np.any(valid):
            result[valid] = (1.0 / sigma) * np.power(term[valid], -(1.0 / xi + 1.0))
        
        return result


def gpd_cdf(x: np.ndarray, xi: float, sigma: float) -> np.ndarray:
    """
    GPD cumulative distribution function.
    
    Args:
        x: Exceedances (must be > 0)
        xi: Shape parameter
        sigma: Scale parameter (> 0)
        
    Returns:
        CDF values in [0, 1]
    """
    x = np.asarray(x)
    if sigma <= 0:
        return np.ones_like(x)
    
    if abs(xi) < 1e-10:
        # Exponential case
        return 1.0 - np.exp(-x / sigma)
    else:
        z = x / sigma
        term = 1.0 + xi * z
        
        valid = term > 0
        result = np.ones_like(x, dtype=float)
        
        if np.any(valid):
            result[valid] = 1.0 - np.power(term[valid], -1.0 / xi)
        
        return np.clip(result, 0.0, 1.0)


def gpd_quantile(p: np.ndarray, xi: float, sigma: float) -> np.ndarray:
    """
    GPD quantile function (inverse CDF).
    
    Args:
        p: Probability values in [0, 1)
        xi: Shape parameter
        sigma: Scale parameter (> 0)
        
    Returns:
        Quantile values
    """
    p = np.asarray(p)
    
    if sigma <= 0:
        return np.full_like(p, np.nan)
    
    if abs(xi) < 1e-10:
        # Exponential case
        return -sigma * np.log(1.0 - p)
    else:
        return (sigma / xi) * (np.power(1.0 - p, -xi) - 1.0)


def compute_cte_gpd(threshold: float, xi: float, sigma: float) -> float:
    """
    Compute Conditional Tail Expectation (CTE) using GPD parameters.
    
    CTE = E[X | X > u] = u + σ/(1-ξ)  for ξ < 1
    
    This is the expected value of losses exceeding the threshold.
    
    Args:
        threshold: Threshold u
        xi: GPD shape parameter
        sigma: GPD scale parameter
        
    Returns:
        Conditional tail expectation
    """
    if xi >= 1.0:
        # Infinite mean case - return conservative estimate
        warnings.warn(f"GPD xi={xi:.3f} >= 1 implies infinite mean, using fallback")
        return threshold * EVT_FALLBACK_MULTIPLIER * 2.0
    
    if sigma <= 0:
        return threshold * EVT_FALLBACK_MULTIPLIER
    
    # CTE formula: E[X | X > u] = u + mean_excess
    # Mean excess function for GPD: σ/(1-ξ)
    mean_excess = sigma / (1.0 - xi)
    
    cte = threshold + mean_excess
    
    return float(cte)


def _gpd_log_likelihood(params: np.ndarray, exceedances: np.ndarray) -> float:
    """
    Negative log-likelihood for GPD (for minimization).
    
    Args:
        params: [xi, log_sigma] (log-scale for sigma to ensure positivity)
        exceedances: Array of exceedances above threshold
        
    Returns:
        Negative log-likelihood
    """
    xi, log_sigma = params
    sigma = np.exp(log_sigma)
    
    if sigma <= 0:
        return 1e10
    
    n = len(exceedances)
    
    if abs(xi) < 1e-10:
        # Exponential case
        ll = -n * log_sigma - np.sum(exceedances) / sigma
    else:
        z = exceedances / sigma
        term = 1.0 + xi * z
        
        # Check domain constraint
        if np.any(term <= 0):
            return 1e10
        
        ll = -n * log_sigma - (1.0 / xi + 1.0) * np.sum(np.log(term))
    
    return -ll  # Return negative for minimization


def fit_gpd_mle(
    exceedances: np.ndarray,
    xi_init: float = 0.1,
    sigma_init: Optional[float] = None
) -> Tuple[float, float, bool, Dict]:
    """
    Fit GPD parameters using Maximum Likelihood Estimation.
    
    Args:
        exceedances: Array of exceedances (values above threshold)
        xi_init: Initial shape parameter guess
        sigma_init: Initial scale parameter guess (defaults to sample mean)
        
    Returns:
        Tuple of (xi, sigma, success, diagnostics)
    """
    exceedances = np.asarray(exceedances).flatten()
    exceedances = exceedances[exceedances > 0]  # Must be positive
    
    n = len(exceedances)
    
    if n < EVT_MIN_EXCEEDANCES:
        return EVT_XI_DEFAULT, np.mean(exceedances), False, {
            "error": "insufficient_exceedances",
            "n_exceedances": n,
            "min_required": EVT_MIN_EXCEEDANCES,
        }
    
    # Initial estimates
    if sigma_init is None:
        sigma_init = float(np.mean(exceedances))
    
    # Bounds: xi ∈ [xi_min, xi_max], sigma > 0
    x0 = [xi_init, np.log(sigma_init)]
    
    bounds = [
        (EVT_XI_MIN, EVT_XI_MAX),
        (np.log(1e-10), np.log(1e10)),
    ]
    
    try:
        result = minimize(
            _gpd_log_likelihood,
            x0=x0,
            args=(exceedances,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if result.success or result.fun < 1e9:
            xi_hat = float(result.x[0])
            sigma_hat = float(np.exp(result.x[1]))
            
            # Compute standard errors from Hessian (if available)
            se_xi = None
            se_sigma = None
            if hasattr(result, 'hess_inv') and result.hess_inv is not None:
                try:
                    hess_diag = np.diag(result.hess_inv.todense() if hasattr(result.hess_inv, 'todense') else result.hess_inv)
                    se_xi = np.sqrt(max(hess_diag[0], 0))
                    se_sigma = sigma_hat * np.sqrt(max(hess_diag[1], 0))  # Delta method
                except Exception:
                    pass
            
            return xi_hat, sigma_hat, True, {
                "n_iterations": result.nit if hasattr(result, 'nit') else None,
                "log_likelihood": -result.fun,
                "se_xi": se_xi,
                "se_sigma": se_sigma,
            }
        else:
            return EVT_XI_DEFAULT, np.mean(exceedances), False, {
                "error": "optimization_failed",
                "message": result.message if hasattr(result, 'message') else "unknown",
            }
            
    except Exception as e:
        return EVT_XI_DEFAULT, np.mean(exceedances), False, {
            "error": "exception",
            "message": str(e),
        }


def fit_gpd_hill(exceedances: np.ndarray) -> Tuple[float, float, bool, Dict]:
    """
    Fit GPD shape parameter using Hill estimator (more robust for small samples).
    
    The Hill estimator for the tail index α = 1/ξ:
        α_hill = (1/k) Σ_{i=1}^{k} log(X_{(n-i+1)} / X_{(n-k)})
    
    Where X_{(i)} are order statistics and k is the number of upper order statistics.
    
    Args:
        exceedances: Array of exceedances
        
    Returns:
        Tuple of (xi, sigma, success, diagnostics)
    """
    exceedances = np.asarray(exceedances).flatten()
    exceedances = exceedances[exceedances > 0]
    
    n = len(exceedances)
    
    if n < EVT_MIN_EXCEEDANCES:
        return EVT_XI_DEFAULT, np.mean(exceedances), False, {
            "error": "insufficient_exceedances",
            "n_exceedances": n,
        }
    
    # Sort in descending order
    sorted_exc = np.sort(exceedances)[::-1]
    
    # Number of order statistics to use (typically sqrt(n) or n/4)
    k = max(int(np.sqrt(n)), EVT_MIN_EXCEEDANCES // 2)
    k = min(k, n - 1)
    
    # Hill estimator
    log_ratios = np.log(sorted_exc[:k] / sorted_exc[k])
    hill_alpha = np.mean(log_ratios)
    
    # Convert to GPD shape: ξ = 1/α
    if hill_alpha > 0.01:
        xi_hill = 1.0 / hill_alpha
    else:
        xi_hill = 0.0  # Exponential-like
    
    # Clip to valid range
    xi_hill = float(np.clip(xi_hill, EVT_XI_MIN, EVT_XI_MAX))
    
    # Estimate sigma using method of moments
    # For GPD: E[X] = σ/(1-ξ), so σ = E[X]*(1-ξ)
    mean_exc = np.mean(exceedances)
    if xi_hill < 1.0:
        sigma_hill = mean_exc * (1.0 - xi_hill)
    else:
        sigma_hill = mean_exc
    
    return xi_hill, sigma_hill, True, {
        "method": "hill",
        "k_order_stats": k,
        "hill_alpha": float(hill_alpha),
    }


def fit_gpd_pwm(exceedances: np.ndarray) -> Tuple[float, float, bool, Dict]:
    """
    Fit GPD using Probability Weighted Moments (robust alternative to MLE).
    
    Args:
        exceedances: Array of exceedances
        
    Returns:
        Tuple of (xi, sigma, success, diagnostics)
    """
    exceedances = np.asarray(exceedances).flatten()
    exceedances = exceedances[exceedances > 0]
    
    n = len(exceedances)
    
    if n < EVT_MIN_EXCEEDANCES:
        return EVT_XI_DEFAULT, np.mean(exceedances), False, {
            "error": "insufficient_exceedances",
        }
    
    # Sort in ascending order
    sorted_exc = np.sort(exceedances)
    
    # Compute probability-weighted moments
    # β_0 = E[X], β_1 = E[X * F(X)]
    ranks = np.arange(1, n + 1)
    weights = (ranks - 0.35) / n  # Plotting positions
    
    b0 = np.mean(sorted_exc)
    b1 = np.mean(sorted_exc * weights)
    
    # PWM estimators for GPD
    # ξ = 2 - b0/(b0 - 2*b1)
    # σ = 2*b0*b1 / (b0 - 2*b1)
    
    denom = b0 - 2 * b1
    
    if abs(denom) < 1e-10:
        # Exponential case
        xi_pwm = 0.0
        sigma_pwm = b0
    else:
        xi_pwm = 2.0 - b0 / denom
        sigma_pwm = 2.0 * b0 * b1 / denom
    
    # Clip to valid range
    xi_pwm = float(np.clip(xi_pwm, EVT_XI_MIN, EVT_XI_MAX))
    sigma_pwm = float(max(sigma_pwm, 1e-10))
    
    return xi_pwm, sigma_pwm, True, {
        "method": "pwm",
        "b0": float(b0),
        "b1": float(b1),
    }


def select_threshold(
    data: np.ndarray,
    percentile: float = EVT_THRESHOLD_PERCENTILE_DEFAULT
) -> Tuple[float, int]:
    """
    Select POT threshold using percentile method.
    
    Args:
        data: Full data array (losses, should be positive)
        percentile: Percentile for threshold (0.90 = 10% exceedances)
        
    Returns:
        Tuple of (threshold, n_exceedances)
    """
    data = np.asarray(data).flatten()
    data = np.abs(data)  # Work with absolute values
    data = data[np.isfinite(data)]
    
    if len(data) < EVT_MIN_EXCEEDANCES:
        return 0.0, 0
    
    # Clip percentile to valid range
    percentile = np.clip(percentile, EVT_THRESHOLD_PERCENTILE_MIN, EVT_THRESHOLD_PERCENTILE_MAX)
    
    threshold = float(np.percentile(data, percentile * 100))
    n_exceedances = int(np.sum(data > threshold))
    
    return threshold, n_exceedances


def fit_gpd_pot(
    losses: np.ndarray,
    threshold_percentile: float = EVT_THRESHOLD_PERCENTILE_DEFAULT,
    method: str = 'auto'
) -> GPDFitResult:
    """
    Fit GPD using Peaks Over Threshold method.
    
    This is the main entry point for EVT fitting on loss data.
    
    Args:
        losses: Array of losses (positive values representing losses)
        threshold_percentile: Percentile for threshold selection (0.90 = 10% exceedances)
        method: Fitting method ('mle', 'hill', 'pwm', 'auto')
        
    Returns:
        GPDFitResult with fitted parameters and CTE
    """
    losses = np.asarray(losses).flatten()
    losses = np.abs(losses)  # Ensure positive
    losses = losses[np.isfinite(losses)]
    
    # Handle insufficient data
    if len(losses) < EVT_MIN_EXCEEDANCES:
        # Return fallback result
        mean_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
        return GPDFitResult(
            xi=EVT_XI_DEFAULT,
            sigma=mean_loss,
            threshold=0.0,
            n_exceedances=len(losses),
            cte=mean_loss * EVT_FALLBACK_MULTIPLIER,
            fit_success=False,
            method='fallback',
            diagnostics={"error": "insufficient_data", "n_obs": len(losses)},
        )
    
    # Select threshold
    threshold, n_exceedances = select_threshold(losses, threshold_percentile)
    
    if n_exceedances < EVT_MIN_EXCEEDANCES:
        # Adjust threshold to get more exceedances
        new_percentile = 1.0 - (EVT_MIN_EXCEEDANCES / len(losses))
        new_percentile = max(new_percentile, EVT_THRESHOLD_PERCENTILE_MIN)
        threshold, n_exceedances = select_threshold(losses, new_percentile)
    
    if n_exceedances < 10:
        # Still insufficient - return fallback
        mean_loss = float(np.mean(losses))
        return GPDFitResult(
            xi=EVT_XI_DEFAULT,
            sigma=mean_loss,
            threshold=threshold,
            n_exceedances=n_exceedances,
            cte=mean_loss * EVT_FALLBACK_MULTIPLIER,
            fit_success=False,
            method='fallback',
            diagnostics={"error": "insufficient_exceedances", "n_exceedances": n_exceedances},
        )
    
    # Extract exceedances (excess over threshold)
    exceedances = losses[losses > threshold] - threshold
    
    # Fit GPD
    if method == 'auto':
        # Try MLE first, fall back to Hill if it fails
        xi_mle, sigma_mle, success_mle, diag_mle = fit_gpd_mle(exceedances)
        
        if success_mle and EVT_XI_MIN < xi_mle < EVT_XI_MAX:
            xi, sigma, success, diag = xi_mle, sigma_mle, success_mle, diag_mle
            fit_method = 'mle'
        else:
            # Fall back to Hill estimator
            xi, sigma, success, diag = fit_gpd_hill(exceedances)
            fit_method = 'hill'
            if not success:
                # Final fallback to PWM
                xi, sigma, success, diag = fit_gpd_pwm(exceedances)
                fit_method = 'pwm'
    elif method == 'mle':
        xi, sigma, success, diag = fit_gpd_mle(exceedances)
        fit_method = 'mle'
    elif method == 'hill':
        xi, sigma, success, diag = fit_gpd_hill(exceedances)
        fit_method = 'hill'
    elif method == 'pwm':
        xi, sigma, success, diag = fit_gpd_pwm(exceedances)
        fit_method = 'pwm'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute CTE
    cte = compute_cte_gpd(threshold, xi, sigma)
    
    # Apply small-sample adjustment if needed
    if n_exceedances < EVT_RECOMMENDED_EXCEEDANCES:
        adjustment = EVT_SMALL_SAMPLE_ADJUSTMENT * (1.0 + (EVT_RECOMMENDED_EXCEEDANCES - n_exceedances) / EVT_RECOMMENDED_EXCEEDANCES * 0.2)
        cte = cte * adjustment
        diag["small_sample_adjustment"] = adjustment
    
    return GPDFitResult(
        xi=xi,
        sigma=sigma,
        threshold=threshold,
        n_exceedances=n_exceedances,
        cte=cte,
        fit_success=success,
        method=fit_method,
        diagnostics=diag,
    )


def compute_evt_expected_loss(
    r_samples: np.ndarray,
    threshold_percentile: float = EVT_THRESHOLD_PERCENTILE_DEFAULT,
    fallback_multiplier: float = EVT_FALLBACK_MULTIPLIER
) -> Tuple[float, float, GPDFitResult]:
    """
    Compute EVT-corrected expected loss for position sizing.
    
    This function replaces the naive empirical mean of losses with
    GPD-based Conditional Tail Expectation.
    
    Args:
        r_samples: Monte Carlo return samples from BMA
        threshold_percentile: Percentile for POT threshold
        fallback_multiplier: Multiplier for empirical loss if EVT fails
        
    Returns:
        Tuple of (evt_expected_loss, empirical_expected_loss, gpd_result)
    """
    r_samples = np.asarray(r_samples).flatten()
    r_samples = r_samples[np.isfinite(r_samples)]
    
    # Compute empirical expected loss (current method)
    losses = -r_samples[r_samples < 0]  # Convert to positive losses
    empirical_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    
    if len(losses) < EVT_MIN_EXCEEDANCES:
        # Not enough data for EVT - use fallback
        evt_loss = empirical_loss * fallback_multiplier
        gpd_result = GPDFitResult(
            xi=EVT_XI_DEFAULT,
            sigma=empirical_loss,
            threshold=0.0,
            n_exceedances=len(losses),
            cte=evt_loss,
            fit_success=False,
            method='fallback_insufficient',
            diagnostics={"n_losses": len(losses)},
        )
        return evt_loss, empirical_loss, gpd_result
    
    # Fit GPD to losses
    gpd_result = fit_gpd_pot(losses, threshold_percentile)
    
    if gpd_result.fit_success:
        # Use GPD-based CTE, but ensure it's at least as large as empirical
        # (EVT should be more conservative, not less)
        evt_loss = max(gpd_result.cte, empirical_loss)
    else:
        # Fallback: use empirical with multiplier
        evt_loss = empirical_loss * fallback_multiplier
    
    return evt_loss, empirical_loss, gpd_result


def compute_evt_var(
    r_samples: np.ndarray,
    alpha: float = 0.05,
    threshold_percentile: float = EVT_THRESHOLD_PERCENTILE_DEFAULT
) -> Tuple[float, float, GPDFitResult]:
    """
    Compute EVT-based Value at Risk.
    
    Args:
        r_samples: Monte Carlo return samples
        alpha: Tail probability (0.05 = 5% VaR)
        threshold_percentile: Percentile for POT threshold
        
    Returns:
        Tuple of (evt_var, empirical_var, gpd_result)
    """
    r_samples = np.asarray(r_samples).flatten()
    r_samples = r_samples[np.isfinite(r_samples)]
    
    # Empirical VaR
    empirical_var = float(np.percentile(-r_samples, (1 - alpha) * 100))
    
    losses = -r_samples[r_samples < 0]
    
    if len(losses) < EVT_MIN_EXCEEDANCES:
        return empirical_var, empirical_var, GPDFitResult(
            xi=EVT_XI_DEFAULT, sigma=np.mean(losses) if len(losses) > 0 else 0.0,
            threshold=0.0, n_exceedances=len(losses),
            cte=empirical_var, fit_success=False, method='fallback',
            diagnostics={},
        )
    
    # Fit GPD
    gpd_result = fit_gpd_pot(losses, threshold_percentile)
    
    if gpd_result.fit_success:
        # EVT VaR: u + σ/ξ * ((n/k * α)^(-ξ) - 1)
        # where n = total, k = exceedances
        n_total = len(r_samples)
        k = gpd_result.n_exceedances
        xi = gpd_result.xi
        sigma = gpd_result.sigma
        u = gpd_result.threshold
        
        if abs(xi) > 0.01:
            prob_exceed = k / n_total
            evt_var = u + (sigma / xi) * (np.power(prob_exceed / alpha, xi) - 1)
        else:
            # Exponential case
            evt_var = u - sigma * np.log(alpha * n_total / k)
        
        evt_var = max(float(evt_var), empirical_var)
    else:
        evt_var = empirical_var * EVT_FALLBACK_MULTIPLIER
    
    return evt_var, empirical_var, gpd_result


# =============================================================================
# CONSISTENCY CHECK WITH STUDENT-T
# =============================================================================

def check_student_t_consistency(
    nu_student_t: Optional[float],
    xi_gpd: float,
    tolerance: float = 0.3
) -> Dict:
    """
    Check consistency between Student-t ν and GPD ξ estimates.
    
    For Student-t: ξ = 1/ν
    
    Args:
        nu_student_t: Estimated Student-t degrees of freedom
        xi_gpd: Estimated GPD shape parameter
        tolerance: Relative tolerance for consistency check
        
    Returns:
        Dict with consistency metrics
    """
    result = {
        "nu_student_t": nu_student_t,
        "xi_gpd": xi_gpd,
        "consistent": None,
        "implied_nu_from_gpd": None,
        "relative_difference": None,
    }
    
    if nu_student_t is None or xi_gpd is None:
        result["consistent"] = None
        return result
    
    if xi_gpd > 0.01:
        implied_nu = 1.0 / xi_gpd
        result["implied_nu_from_gpd"] = implied_nu
        
        rel_diff = abs(nu_student_t - implied_nu) / max(nu_student_t, implied_nu)
        result["relative_difference"] = rel_diff
        result["consistent"] = rel_diff < tolerance
    else:
        # ξ ≈ 0 implies very light tails, inconsistent with finite ν
        result["consistent"] = nu_student_t > 30  # Gaussian-like
    
    return result
