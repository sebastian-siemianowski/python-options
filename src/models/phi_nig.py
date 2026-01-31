"""
===============================================================================
PHI-NIG DRIFT MODEL — Kalman Filter with AR(1) Drift and NIG Observation Noise
===============================================================================

Implements a state-space model with AR(1) drift and Normal-Inverse Gaussian
(NIG) observation noise:

    State equation:    μ_t = φ·μ_{t-1} + w_t,  w_t ~ N(0, q)
    Observation:       r_t = μ_t + ε_t,         ε_t ~ NIG(α, β, δ, 0)

The NIG distribution is a four-parameter family that captures:
    α:  Tail heaviness (α → ∞ approaches Gaussian)
    β:  Asymmetry (-α < β < α; β=0 is symmetric)
    δ:  Scale parameter (δ > 0)
    μ:  Location parameter (set to 0 for innovation noise)

NIG PROPERTIES:
    - Mean: μ + δβ/√(α² - β²)
    - Variance: δα²/(α² - β²)^(3/2)
    - Skewness: 3β / (α√δ · (α² - β²)^(1/4))
    - Excess Kurtosis: 3(1 + 4β²/α²) / (δ√(α² - β²))

KEY ADVANTAGES OVER STUDENT-T:
    1. Captures ASYMMETRY via β parameter
    2. Semi-heavy tails (Gaussian < NIG < Cauchy)
    3. Closed-form density and CDF
    4. Infinitely divisible (compatible with Lévy processes)

BMA INTEGRATION:
    NIG participates as a candidate model in Bayesian Model Averaging.
    It competes with Gaussian, Student-t, and Skew-t distributions.
    If data doesn't support NIG's extra complexity, BIC penalizes it
    and model weight collapses toward simpler alternatives.

CORE PRINCIPLE:
    "Heavy tails and asymmetry are hypotheses, not certainties."

PARAMETERIZATION NOTE:
    scipy.stats.norminvgauss uses (a, b) parameterization where:
        a = α·δ (shape/tail parameter)
        b = β·δ (asymmetry parameter)
    We work internally with (α, β, δ) for interpretability, then convert.

===============================================================================
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kstest
from scipy.stats import norm
from scipy.stats import norminvgauss
from scipy.special import kv  # Modified Bessel function of second kind


# =============================================================================
# NIG PARAMETER CONSTRAINTS AND DEFAULTS
# =============================================================================

# Minimum tail parameter (α > 0, larger = lighter tails)
NIG_ALPHA_MIN = 0.5
NIG_ALPHA_MAX = 50.0
NIG_ALPHA_DEFAULT = 2.0

# Asymmetry parameter (-α < β < α; β=0 is symmetric)
NIG_BETA_DEFAULT = 0.0

# Scale parameter (δ > 0)
NIG_DELTA_MIN = 0.001
NIG_DELTA_MAX = 1.0
NIG_DELTA_DEFAULT = 0.01

# φ shrinkage prior (shared with other models)
PHI_SHRINKAGE_TAU_MIN = 1e-3
PHI_SHRINKAGE_GLOBAL_DEFAULT = 0.0
PHI_SHRINKAGE_LAMBDA_DEFAULT = 0.05


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


def _convert_to_scipy_params(alpha: float, beta: float, delta: float) -> Tuple[float, float, float, float]:
    """
    Convert (α, β, δ) to scipy.stats.norminvgauss parameters (a, b, loc, scale).
    
    scipy uses: NIG(a, b, loc, scale) where:
        a = α·δ (dimensionless shape)
        b = β·δ (dimensionless asymmetry)
        loc = location
        scale = 1 (we handle scaling separately)
    
    The actual scipy parameterization for density f(x; a, b, loc, scale) is:
        f(x) = (a/π) * δ * K_1(a*q) / q * exp(δ*γ + β*(x-loc))
        where q = √(δ² + (x-loc)²), γ = √(α² - β²)
    
    For simplicity, we use scale=δ and a=α*δ, b=β*δ
    """
    # Ensure valid parameter range
    alpha = max(alpha, NIG_ALPHA_MIN)
    delta = max(delta, NIG_DELTA_MIN)
    # β must satisfy |β| < α
    beta = np.clip(beta, -alpha + 0.01, alpha - 0.01)
    
    # scipy parameterization
    a = alpha * delta
    b = beta * delta
    
    return a, b, 0.0, delta  # loc=0, scale=delta


class PhiNIGDriftModel:
    """
    Encapsulates Normal-Inverse Gaussian (NIG) observation noise for Kalman drift modeling.
    
    NIG captures both:
    - Heavy tails via α parameter (smaller α = heavier tails)
    - Asymmetry via β parameter (β < 0 = left-skewed, β > 0 = right-skewed)
    
    BMA INTEGRATION:
    This model competes with Gaussian, Student-t, and Skew-t in the BMA ensemble.
    BIC penalizes the extra parameters if they don't improve fit.
    """

    alpha_min_default: float = NIG_ALPHA_MIN
    alpha_max_default: float = NIG_ALPHA_MAX
    delta_min_default: float = NIG_DELTA_MIN
    delta_max_default: float = NIG_DELTA_MAX

    @staticmethod
    def _validate_params(alpha: float, beta: float, delta: float) -> Tuple[float, float, float]:
        """Validate and clip NIG parameters to valid range."""
        alpha = float(np.clip(alpha, NIG_ALPHA_MIN, NIG_ALPHA_MAX))
        delta = float(np.clip(delta, NIG_DELTA_MIN, NIG_DELTA_MAX))
        # |β| < α is required for NIG
        max_beta = alpha - 0.01
        beta = float(np.clip(beta, -max_beta, max_beta))
        return alpha, beta, delta

    @classmethod
    def logpdf(cls, x: float, alpha: float, beta: float, delta: float, mu: float = 0.0) -> float:
        """
        Log-density of NIG distribution at point x.
        
        The NIG density is:
            f(x) = (α·δ/π) · K_1(α·q) / q · exp(δ·γ + β·(x-μ))
        
        Where:
            q = √(δ² + (x-μ)²)
            γ = √(α² - β²)
            K_1 = modified Bessel function of second kind
        
        Args:
            x: Observation value
            alpha: Tail parameter (α > 0)
            beta: Asymmetry parameter (|β| < α)
            delta: Scale parameter (δ > 0)
            mu: Location parameter
            
        Returns:
            Log-density value
        """
        alpha, beta, delta = cls._validate_params(alpha, beta, delta)
        
        try:
            # Use scipy's NIG implementation
            a, b, loc, scale = _convert_to_scipy_params(alpha, beta, delta)
            # Adjust x for location
            z = (x - mu)
            logp = norminvgauss.logpdf(z / scale, a, b, loc=0, scale=1)
            # Adjust for scale (Jacobian)
            logp -= np.log(scale)
            
            if np.isfinite(logp):
                return float(logp)
            else:
                return -1e12
        except Exception:
            return -1e12

    @classmethod
    def pdf(cls, x: float, alpha: float, beta: float, delta: float, mu: float = 0.0) -> float:
        """PDF of NIG distribution."""
        logp = cls.logpdf(x, alpha, beta, delta, mu)
        return float(np.exp(logp))

    @classmethod
    def cdf(cls, x: float, alpha: float, beta: float, delta: float, mu: float = 0.0) -> float:
        """
        CDF of NIG distribution.
        
        Args:
            x: Observation value
            alpha, beta, delta: NIG parameters
            mu: Location parameter
            
        Returns:
            CDF value in [0, 1]
        """
        alpha, beta, delta = cls._validate_params(alpha, beta, delta)
        
        try:
            a, b, loc, scale = _convert_to_scipy_params(alpha, beta, delta)
            z = (x - mu)
            cdf_val = norminvgauss.cdf(z / scale, a, b, loc=0, scale=1)
            return float(np.clip(cdf_val, 0.0, 1.0))
        except Exception:
            return 0.5

    @classmethod
    def sample(
        cls,
        alpha: float,
        beta: float,
        delta: float,
        mu: float = 0.0,
        size: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Sample from NIG distribution.
        
        Uses scipy's norminvgauss random variate generation.
        
        Args:
            alpha, beta, delta: NIG parameters
            mu: Location parameter
            size: Number of samples
            rng: Random generator (optional)
            
        Returns:
            Array of samples
        """
        alpha, beta, delta = cls._validate_params(alpha, beta, delta)
        
        if rng is None:
            rng = np.random.default_rng()
        
        try:
            a, b, loc, scale = _convert_to_scipy_params(alpha, beta, delta)
            # Generate standard NIG samples
            samples = norminvgauss.rvs(a, b, loc=0, scale=1, size=size, random_state=rng)
            # Scale and shift
            return mu + scale * samples
        except Exception:
            # Fallback to Gaussian if NIG sampling fails
            return rng.normal(loc=mu, scale=delta, size=size)

    @classmethod
    def fit_mle(
        cls,
        data: np.ndarray,
        alpha_init: float = NIG_ALPHA_DEFAULT,
        beta_init: float = NIG_BETA_DEFAULT,
        delta_init: Optional[float] = None
    ) -> Tuple[float, float, float, float, Dict]:
        """
        Fit NIG parameters via Maximum Likelihood Estimation.
        
        Args:
            data: Array of observations (assumed zero-mean for innovation noise)
            alpha_init: Initial guess for α
            beta_init: Initial guess for β
            delta_init: Initial guess for δ (defaults to sample std)
            
        Returns:
            Tuple of (alpha, beta, delta, log_likelihood, diagnostics)
        """
        data = np.asarray(data).flatten()
        data = data[np.isfinite(data)]
        n = len(data)
        
        if n < 30:
            return NIG_ALPHA_DEFAULT, NIG_BETA_DEFAULT, NIG_DELTA_DEFAULT, float('-inf'), {
                "fit_success": False,
                "error": "insufficient_data"
            }
        
        # Initialize delta from sample std if not provided
        if delta_init is None:
            delta_init = float(np.std(data))
            delta_init = max(delta_init, NIG_DELTA_MIN)
        
        def neg_ll(params):
            alpha, beta, delta = params
            # Validate parameters
            if alpha <= NIG_ALPHA_MIN or delta <= 0 or abs(beta) >= alpha:
                return 1e12
            
            try:
                a, b, loc, scale = _convert_to_scipy_params(alpha, beta, delta)
                ll = np.sum(norminvgauss.logpdf(data / scale, a, b, loc=0, scale=1))
                ll -= n * np.log(scale)  # Jacobian adjustment
                
                if np.isfinite(ll):
                    return -ll
                else:
                    return 1e12
            except Exception:
                return 1e12
        
        # Multi-start optimization
        best_result = None
        best_obj = float('inf')
        
        # Initial points with varying asymmetry
        initial_points = [
            (alpha_init, 0.0, delta_init),           # Symmetric
            (alpha_init, 0.3 * alpha_init, delta_init),  # Right-skewed
            (alpha_init, -0.3 * alpha_init, delta_init), # Left-skewed
            (1.5, 0.0, delta_init),                  # Heavier tails
            (5.0, 0.0, delta_init),                  # Lighter tails
        ]
        
        bounds = [
            (NIG_ALPHA_MIN, NIG_ALPHA_MAX),
            (-NIG_ALPHA_MAX + 0.1, NIG_ALPHA_MAX - 0.1),  # β bounds (will be constrained)
            (NIG_DELTA_MIN, NIG_DELTA_MAX),
        ]
        
        for x0 in initial_points:
            try:
                # Constraint: |β| < α
                constraints = [
                    {'type': 'ineq', 'fun': lambda p: p[0] - abs(p[1]) - 0.01}
                ]
                
                result = minimize(
                    neg_ll,
                    x0=np.array(x0),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200}
                )
                
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_result = result
            except Exception:
                continue
        
        if best_result is None or not np.isfinite(best_obj):
            return NIG_ALPHA_DEFAULT, NIG_BETA_DEFAULT, delta_init, float('-inf'), {
                "fit_success": False,
                "error": "optimization_failed"
            }
        
        alpha_opt, beta_opt, delta_opt = best_result.x
        alpha_opt, beta_opt, delta_opt = cls._validate_params(alpha_opt, beta_opt, delta_opt)
        ll = -best_obj
        
        # Compute diagnostics
        gamma_opt = np.sqrt(alpha_opt**2 - beta_opt**2)
        mean_theoretical = delta_opt * beta_opt / gamma_opt if gamma_opt > 0 else 0
        var_theoretical = delta_opt * alpha_opt**2 / (gamma_opt**3) if gamma_opt > 0 else delta_opt**2
        
        diagnostics = {
            "fit_success": True,
            "n_obs": n,
            "alpha": float(alpha_opt),
            "beta": float(beta_opt),
            "delta": float(delta_opt),
            "gamma": float(gamma_opt),
            "theoretical_mean": float(mean_theoretical),
            "theoretical_var": float(var_theoretical),
            "theoretical_std": float(np.sqrt(var_theoretical)),
            "sample_mean": float(np.mean(data)),
            "sample_std": float(np.std(data)),
            "asymmetry_ratio": float(beta_opt / alpha_opt) if alpha_opt > 0 else 0,
        }
        
        return float(alpha_opt), float(beta_opt), float(delta_opt), float(ll), diagnostics

    @classmethod
    def filter_phi(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        alpha: float,
        beta: float,
        delta: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Kalman drift filter with AR(1) persistence and NIG observation noise.
        
        The Kalman update equations are the same as Student-t (moment-matching
        approximation), but likelihood is computed with NIG density.
        
        Args:
            returns: Array of returns
            vol: Array of EWMA volatility
            q: Process noise variance
            c: Observation noise scale multiplier
            phi: AR(1) persistence
            alpha, beta, delta: NIG parameters for observation noise
            
        Returns:
            Tuple of (mu_filtered, P_filtered, log_likelihood)
        """
        n = len(returns)
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))
        alpha, beta, delta = cls._validate_params(alpha, beta, delta)

        mu = 0.0
        P = 1e-4
        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
        log_likelihood = 0.0

        # Compute NIG variance for Kalman gain adjustment
        gamma_nig = np.sqrt(max(alpha**2 - beta**2, 1e-10))
        nig_var = delta * alpha**2 / (gamma_nig**3) if gamma_nig > 0 else delta**2

        for t in range(n):
            mu_pred = phi_val * mu
            P_pred = (phi_val ** 2) * P + q_val

            vol_t = vol[t]
            vol_scalar = float(vol_t) if np.ndim(vol_t) == 0 else float(vol_t.item())
            
            # Observation variance: c * vol² scaled by NIG variance
            R = c_val * (vol_scalar ** 2) * (nig_var / (delta**2)) if delta > 0 else c_val * (vol_scalar ** 2)

            ret_t = returns[t]
            r_val = float(ret_t) if np.ndim(ret_t) == 0 else float(ret_t.item())
            innovation = r_val - mu_pred

            S = P_pred + R
            if S <= 1e-12:
                S = 1e-12
            
            # Standard Kalman gain (no heavy-tail adjustment for NIG)
            K = P_pred / S

            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12

            mu_filtered[t] = mu
            P_filtered[t] = P

            # Compute log-likelihood using NIG density
            # Scale delta by c * vol for time-varying observation noise
            delta_t = delta * np.sqrt(c_val) * vol_scalar
            if delta_t > 1e-12:
                ll_t = cls.logpdf(r_val, alpha, beta, delta_t, mu=mu_pred)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t

        return mu_filtered, P_filtered, float(log_likelihood)

    @classmethod
    def pit_ks(
        cls,
        returns: np.ndarray,
        mu_filtered: np.ndarray,
        vol: np.ndarray,
        P_filtered: np.ndarray,
        c: float,
        alpha: float,
        beta: float,
        delta: float
    ) -> Tuple[float, float]:
        """
        PIT/KS calibration test for NIG forecasts.
        
        Transforms observations to uniform[0,1] using NIG CDF,
        then tests uniformity via Kolmogorov-Smirnov.
        
        Args:
            returns: Array of returns
            mu_filtered: Filtered drift estimates
            vol: EWMA volatility
            P_filtered: Filtered variance
            c: Observation noise scale
            alpha, beta, delta: NIG parameters
            
        Returns:
            Tuple of (KS statistic, p-value)
        """
        returns_flat = np.asarray(returns).flatten()
        mu_flat = np.asarray(mu_filtered).flatten()
        vol_flat = np.asarray(vol).flatten()
        
        alpha, beta, delta = cls._validate_params(alpha, beta, delta)

        # Compute PIT values using NIG CDF
        pit_values = []
        for i in range(len(returns_flat)):
            if not (np.isfinite(returns_flat[i]) and np.isfinite(mu_flat[i])):
                continue
            
            # Time-varying scale
            delta_t = delta * np.sqrt(c) * vol_flat[i]
            if delta_t <= 1e-12:
                continue
            
            pit = cls.cdf(returns_flat[i], alpha, beta, delta_t, mu=mu_flat[i])
            if np.isfinite(pit):
                pit_values.append(pit)

        pit_values = np.array(pit_values)

        if len(pit_values) < 2:
            return 1.0, 0.0

        ks_result = kstest(pit_values, 'uniform')
        return float(ks_result.statistic), float(ks_result.pvalue)

    @classmethod
    def optimize_params_fixed_nig(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        alpha: float,
        beta: float,
        delta: float,
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
        Optimize (q, c, φ) for fixed NIG parameters via cross-validated MLE.
        
        This follows the same pattern as phi_student_t.optimize_params_fixed_nu.
        
        Args:
            returns: Array of returns
            vol: Array of volatility
            alpha, beta, delta: Fixed NIG parameters
            train_frac: Fraction for training
            q_min, q_max: Bounds for q
            c_min, c_max: Bounds for c
            phi_min, phi_max: Bounds for phi
            prior_log_q_mean: Prior mean for log10(q)
            prior_lambda: Regularization strength
            
        Returns:
            Tuple of (q_opt, c_opt, phi_opt, cv_ll, diagnostics)
        """
        n = len(returns)
        
        # Robust winsorization
        ret_p005 = np.percentile(returns, 0.5)
        ret_p995 = np.percentile(returns, 99.5)
        returns_robust = np.clip(returns, ret_p005, ret_p995)

        # Train/validation split
        n_train = int(n * train_frac)
        ret_train = returns_robust[:n_train]
        vol_train = vol[:n_train]
        ret_val = returns_robust[n_train:]
        vol_val = vol[n_train:]

        tau = _lambda_to_tau(prior_lambda * 0.05)

        def neg_cv_ll(params):
            log_q, c, phi = params
            q = 10.0 ** log_q

            # Regularization on q
            reg = prior_lambda * (log_q - prior_log_q_mean) ** 2

            # φ shrinkage prior
            phi_prior = _phi_shrinkage_log_prior(phi, PHI_SHRINKAGE_GLOBAL_DEFAULT, tau)

            try:
                # Fit on training data
                _, _, ll_train = cls.filter_phi(ret_train, vol_train, q, c, phi, alpha, beta, delta)
                # Validate on holdout
                _, _, ll_val = cls.filter_phi(ret_val, vol_val, q, c, phi, alpha, beta, delta)

                # CV objective: penalized validation likelihood
                cv_objective = -ll_val + reg - phi_prior
                return float(cv_objective) if np.isfinite(cv_objective) else 1e12
            except Exception:
                return 1e12

        # Multi-start optimization
        best_result = None
        best_obj = float('inf')

        initial_points = [
            (-6.0, 1.0, 0.9),
            (-5.0, 0.8, 0.8),
            (-7.0, 1.2, 0.95),
            (-6.5, 1.0, 0.7),
            (-5.5, 0.9, 0.85),
        ]

        bounds = [
            (np.log10(q_min), np.log10(q_max)),
            (c_min, c_max),
            (phi_min, phi_max),
        ]

        for x0 in initial_points:
            try:
                result = minimize(
                    neg_cv_ll,
                    x0=np.array(x0),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is None:
            # Fallback to defaults
            return 1e-6, 1.0, 0.9, float('-inf'), {
                "fit_success": False,
                "error": "optimization_failed"
            }

        log_q_opt, c_opt, phi_opt = best_result.x
        q_opt = 10.0 ** log_q_opt

        # Compute final log-likelihood on full data
        _, _, ll_full = cls.filter_phi(returns_robust, vol, q_opt, c_opt, phi_opt, alpha, beta, delta)

        diagnostics = {
            "fit_success": True,
            "cv_objective": float(best_obj),
            "n_train": int(n_train),
            "n_val": int(n - n_train),
            "alpha": float(alpha),
            "beta": float(beta),
            "delta": float(delta),
        }

        return float(q_opt), float(c_opt), float(phi_opt), float(-best_obj), diagnostics

    @classmethod
    def estimate_nig_from_returns(
        cls,
        returns: np.ndarray,
        min_obs: int = 100
    ) -> Tuple[float, float, float, Dict]:
        """
        Estimate NIG parameters from historical returns.
        
        Uses MLE with multiple starting points to find best fit.
        
        Args:
            returns: Array of returns
            min_obs: Minimum observations required
            
        Returns:
            Tuple of (alpha, beta, delta, diagnostics)
        """
        n = len(returns)
        if n < min_obs:
            return NIG_ALPHA_DEFAULT, NIG_BETA_DEFAULT, NIG_DELTA_DEFAULT, {
                "method": "default_insufficient_data",
                "n": n
            }
        
        # Standardize returns for numerical stability
        ret_mean = np.mean(returns)
        ret_std = np.std(returns)
        if ret_std < 1e-10:
            return NIG_ALPHA_DEFAULT, NIG_BETA_DEFAULT, NIG_DELTA_DEFAULT, {
                "method": "default_zero_variance"
            }
        
        ret_standardized = (returns - ret_mean) / ret_std
        
        # Fit NIG to standardized returns
        alpha, beta, delta, ll, fit_diag = cls.fit_mle(ret_standardized)
        
        # Scale delta back to original units
        delta_scaled = delta * ret_std
        
        return alpha, beta, delta_scaled, {
            "method": "mle",
            "standardized_fit": fit_diag,
            "scale_factor": ret_std,
            "location_adjustment": ret_mean,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS FOR BMA INTEGRATION
# =============================================================================

def get_nig_model_name(alpha: float, beta: float) -> str:
    """
    Generate model name for BMA ensemble.
    
    Format: phi_nig_alpha_{α}_beta_{β}
    Example: phi_nig_alpha_2p0_beta_m0p3
    """
    alpha_str = f"{alpha:.1f}".replace(".", "p").replace("-", "m")
    beta_str = f"{beta:.1f}".replace(".", "p").replace("-", "m")
    return f"phi_nig_alpha_{alpha_str}_beta_{beta_str}"


def parse_nig_model_name(model_name: str) -> Optional[Tuple[float, float]]:
    """
    Parse α and β from model name.
    
    Returns (alpha, beta) or None if not a NIG model.
    """
    if not model_name.startswith("phi_nig_alpha_"):
        return None
    
    try:
        parts = model_name.split("_")
        alpha_idx = parts.index("alpha") + 1
        beta_idx = parts.index("beta") + 1
        
        alpha_str = parts[alpha_idx].replace("p", ".").replace("m", "-")
        beta_str = parts[beta_idx].replace("p", ".").replace("m", "-")
        
        return (float(alpha_str), float(beta_str))
    except (ValueError, IndexError):
        return None


def is_nig_model(model_name: str) -> bool:
    """Check if model name is a NIG variant."""
    return model_name.startswith("phi_nig_")


# =============================================================================
# DISCRETE GRID FOR BMA
# =============================================================================
# Use discrete grids for α and β to avoid continuous optimization instability.
# Each (α, β) combination becomes a separate sub-model in BMA.
# =============================================================================

# Tail heaviness grid (smaller α = heavier tails)
NIG_ALPHA_GRID = [1.5, 2.5, 4.0, 8.0]

# Asymmetry grid (β/α ratio for interpretability)
# Actual β = ratio * α
NIG_BETA_RATIO_GRID = [-0.3, 0.0, 0.3]  # Left-skewed, symmetric, right-skewed
