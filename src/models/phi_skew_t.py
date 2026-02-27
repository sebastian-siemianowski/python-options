"""
===============================================================================
PHI-SKEW-T DRIFT MODEL — Kalman Filter with AR(1) Drift and Skewed Student-t Noise
===============================================================================

Implements a state-space model with AR(1) drift and asymmetric heavy-tailed 
observation noise using the Fernández-Steel skew-t distribution:

    State equation:    μ_t = φ·μ_{t-1} + w_t,  w_t ~ N(0, q)
    Observation:       r_t = μ_t + ε_t,        ε_t ~ SkewT(ν, γ, 0, √(c)·σ_t)

Parameters:
    q:   Process noise variance (drift evolution uncertainty)
    c:   Observation noise scale (multiplier on EWMA variance)
    φ:   AR(1) persistence coefficient (φ=1 is random walk, φ=0 is mean reversion)
    ν:   Degrees of freedom (controls tail heaviness; ν→∞ approaches Gaussian)
    γ:   Skewness parameter (γ=1 is symmetric, γ<1 is left-skewed, γ>1 is right-skewed)

FERNÁNDEZ-STEEL PARAMETERIZATION:
    The skew-t density is defined via the transformation:
        f(z|ν,γ) = (2/(γ + 1/γ)) * [f_t(z/γ|ν) * I(z≥0) + f_t(γ*z|ν) * I(z<0)]
    
    Where f_t is the standard Student-t density with ν degrees of freedom.
    
    Properties:
        - γ = 1: Symmetric Student-t (reduces to existing phi_student_t)
        - γ < 1: Left-skewed (heavier left tail) — typical during stress
        - γ > 1: Right-skewed (heavier right tail) — risk in euphoria

BMA INTEGRATION:
    φ-Skew-t participates as a candidate model in Bayesian Model Averaging.
    The BMA framework allows skewness to be introduced ONLY when supported by data.
    If data does not support skewness, model weight collapses toward symmetric alternatives.

CORE PRINCIPLE:
    "Skewness is a hypothesis, not a certainty."

===============================================================================
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import kstest
from scipy.stats import norm
from scipy.stats import t as student_t


# =============================================================================
# φ SHRINKAGE PRIOR CONSTANTS (shared with phi_student_t)
# =============================================================================

PHI_SHRINKAGE_TAU_MIN = 1e-3
PHI_SHRINKAGE_GLOBAL_DEFAULT = 0.0
PHI_SHRINKAGE_LAMBDA_DEFAULT = 0.05

# Discrete ν grid for Skew-t models (same as Student-t for BMA compatibility)
SKEW_T_NU_GRID = [4, 8, 20]

# Discrete γ grid for skewness (Fernández-Steel parameterization)
# γ = 1 is symmetric, γ < 1 is left-skewed, γ > 1 is right-skewed
# Bounded grid prevents extreme asymmetry that destabilizes estimation
SKEW_T_GAMMA_GRID = [0.7, 0.85, 1.0, 1.15, 1.3]

# Default skewness bounds
GAMMA_MIN = 0.5
GAMMA_MAX = 2.0
GAMMA_DEFAULT = 1.0  # Symmetric


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


class PhiSkewTDriftModel:
    """
    Encapsulates Skew-t heavy-tail and asymmetry logic for Kalman drift modeling.
    
    Implements the Fernández-Steel skew-t distribution which:
    - Reduces to symmetric Student-t when γ=1
    - Captures left skewness (crash risk) when γ<1
    - Captures right skewness (euphoria) when γ>1
    
    BMA INTEGRATION:
    This model competes with Gaussian and symmetric Student-t in the BMA ensemble.
    Skewness is introduced only when the data provides sufficient evidence.
    """

    nu_min_default: float = 2.1
    nu_max_default: float = 30.0
    gamma_min_default: float = GAMMA_MIN
    gamma_max_default: float = GAMMA_MAX

    @staticmethod
    def _clip_nu(nu: float, nu_min: float, nu_max: float) -> float:
        return float(np.clip(float(nu), nu_min, nu_max))

    @staticmethod
    def _clip_gamma(gamma: float, gamma_min: float, gamma_max: float) -> float:
        return float(np.clip(float(gamma), gamma_min, gamma_max))

    @staticmethod
    def logpdf_standard_t(z: float, nu: float) -> float:
        """
        Log-density of standard Student-t (location=0, scale=1).
        """
        if nu <= 0:
            return -1e12
        log_norm = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)
        log_kernel = -((nu + 1.0) / 2.0) * np.log(1.0 + (z ** 2) / nu)
        return float(log_norm + log_kernel)

    @classmethod
    def logpdf_skew_t(cls, x: float, nu: float, gamma: float, mu: float, scale: float) -> float:
        """
        Log-density of Fernández-Steel skewed Student-t distribution.
        
        The density is:
            f(z|ν,γ) = (2/(γ + 1/γ)) * [f_t(z/γ|ν) if z≥0 else f_t(γ*z|ν)]
        
        Where z = (x - μ) / scale is the standardized value.
        
        Args:
            x: Observation value
            nu: Degrees of freedom (tail heaviness)
            gamma: Skewness parameter (1 = symmetric, <1 = left-skewed, >1 = right-skewed)
            mu: Location parameter
            scale: Scale parameter
            
        Returns:
            Log-density value
        """
        if scale <= 0 or nu <= 0 or gamma <= 0:
            return -1e12

        # Standardize
        z = (x - mu) / scale

        # Fernández-Steel transformation
        log_normalizer = np.log(2.0) - np.log(gamma + 1.0 / gamma) - np.log(scale)

        if z >= 0:
            # Right tail: transform by 1/γ
            z_transformed = z / gamma
        else:
            # Left tail: transform by γ
            z_transformed = z * gamma

        log_kernel = cls.logpdf_standard_t(z_transformed, nu)

        return float(log_normalizer + log_kernel)

    @classmethod
    def cdf_skew_t(cls, x: float, nu: float, gamma: float, mu: float = 0.0, scale: float = 1.0) -> float:
        """
        CDF of Fernández-Steel skewed Student-t distribution.
        
        Derived from the transformation:
            F(z|ν,γ) = 2/(γ+1/γ) * [γ*F_t(z/γ|ν) if z≥0 else (1/γ)*F_t(γ*z|ν)]
        
        Where F_t is the standard Student-t CDF.
        
        Args:
            x: Observation value
            nu: Degrees of freedom
            gamma: Skewness parameter
            mu: Location (default 0)
            scale: Scale (default 1)
            
        Returns:
            CDF value in [0, 1]
        """
        if scale <= 0 or nu <= 0 or gamma <= 0:
            return 0.5  # Undefined, return neutral
        
        z = (x - mu) / scale
        c = gamma + 1.0 / gamma  # Normalizing constant
        
        if z >= 0:
            # F_t(z/γ) for the right tail
            cdf_t = student_t.cdf(z / gamma, df=nu)
            # The full CDF for z ≥ 0
            return float((1.0 / c) + (2.0 * gamma / c) * (cdf_t - 0.5))
        else:
            # F_t(γ*z) for the left tail
            cdf_t = student_t.cdf(gamma * z, df=nu)
            # The full CDF for z < 0
            return float((2.0 / (gamma * c)) * cdf_t)

    @classmethod
    def sample_skew_t(
        cls,
        nu: float,
        gamma: float,
        mu: float,
        scale: float,
        size: int = 1,
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Sample from Fernández-Steel skewed Student-t distribution.
        
        Uses the stochastic representation:
            X = μ + scale * Z_skew
        
        Where Z_skew is generated via the inverse CDF method applied to
        standard Student-t samples with Fernández-Steel transformation.
        
        Args:
            nu: Degrees of freedom
            gamma: Skewness parameter
            mu: Location
            scale: Scale
            size: Number of samples
            rng: Random generator (optional)
            
        Returns:
            Array of samples
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Sample from standard Student-t
        t_samples = rng.standard_t(df=nu, size=size)
        
        # Apply Fernández-Steel transformation
        # For z < 0: multiply by γ
        # For z ≥ 0: divide by γ
        skew_samples = np.where(
            t_samples >= 0,
            t_samples / gamma,
            t_samples * gamma
        )
        
        # Location-scale transformation
        return mu + scale * skew_samples

    @classmethod
    def filter_phi(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        nu: float,
        gamma: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Kalman drift filter with AR(1) persistence and Skew-t observation noise.
        
        This extends the φ-Student-t filter by allowing asymmetric observation noise.
        The Kalman update uses the same logic but likelihood is computed with skew-t.
        
        Args:
            returns: Array of returns
            vol: Array of EWMA volatility
            q: Process noise variance
            c: Observation noise scale
            phi: AR(1) persistence
            nu: Degrees of freedom
            gamma: Skewness parameter
            
        Returns:
            Tuple of (mu_filtered, P_filtered, log_likelihood)
        """
        n = len(returns)
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))
        nu_val = cls._clip_nu(nu, cls.nu_min_default, cls.nu_max_default)
        gamma_val = cls._clip_gamma(gamma, cls.gamma_min_default, cls.gamma_max_default)

        mu = 0.0
        P = 1e-4
        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
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
            
            # Adjust Kalman gain for heavy tails (same as Student-t)
            nu_adjust = min(nu_val / (nu_val + 3.0), 1.0)
            K = nu_adjust * P_pred / S

            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12

            mu_filtered[t] = mu
            P_filtered[t] = P

            # Compute log-likelihood using skew-t density
            forecast_scale = np.sqrt(S)
            if forecast_scale > 1e-12:
                ll_t = cls.logpdf_skew_t(r_val, nu_val, gamma_val, mu_pred, forecast_scale)
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
        nu: float,
        gamma: float
    ) -> Tuple[float, float]:
        """
        PIT/KS calibration test for Skew-t forecasts.
        
        Transforms observations to uniform[0,1] using skew-t CDF,
        then tests uniformity via Kolmogorov-Smirnov.
        
        Args:
            returns: Array of returns
            mu_filtered: Filtered drift estimates
            vol: EWMA volatility
            P_filtered: Filtered variance
            c: Observation noise scale
            nu: Degrees of freedom
            gamma: Skewness parameter
            
        Returns:
            Tuple of (KS statistic, p-value)
        """
        returns_flat = np.asarray(returns).flatten()
        mu_flat = np.asarray(mu_filtered).flatten()
        vol_flat = np.asarray(vol).flatten()
        P_flat = np.asarray(P_filtered).flatten()

        # Compute forecast scale
        forecast_var = c * (vol_flat ** 2) + P_flat
        forecast_scale = np.sqrt(np.maximum(forecast_var, 1e-20))
        forecast_scale = np.where(forecast_scale < 1e-10, 1e-10, forecast_scale)

        # Compute PIT values using skew-t CDF
        pit_values = []
        for i in range(len(returns_flat)):
            if np.isfinite(returns_flat[i]) and np.isfinite(mu_flat[i]) and forecast_scale[i] > 0:
                pit = cls.cdf_skew_t(
                    returns_flat[i],
                    nu,
                    gamma,
                    mu=mu_flat[i],
                    scale=forecast_scale[i]
                )
                if np.isfinite(pit):
                    pit_values.append(pit)

        pit_values = np.array(pit_values)

        if len(pit_values) < 2:
            return 1.0, 0.0

        ks_result = kstest(pit_values, 'uniform')
        return float(ks_result.statistic), float(ks_result.pvalue)

    @classmethod
    def optimize_params_fixed_nu_gamma(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        nu: float,
        gamma: float,
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
        Optimize (q, c, φ) for fixed ν and γ via cross-validated MLE.
        
        This follows the same pattern as phi_student_t.optimize_params_fixed_nu
        but with skew-t likelihood.
        
        Args:
            returns: Array of returns
            vol: Array of volatility
            nu: Fixed degrees of freedom
            gamma: Fixed skewness parameter
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

        vol_mean = float(np.mean(vol))
        
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
                _, _, ll_train = cls.filter_phi(ret_train, vol_train, q, c, phi, nu, gamma)
                # Validate on holdout
                _, _, ll_val = cls.filter_phi(ret_val, vol_val, q, c, phi, nu, gamma)

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
        _, _, ll_full = cls.filter_phi(returns_robust, vol, q_opt, c_opt, phi_opt, nu, gamma)

        diagnostics = {
            "fit_success": True,
            "cv_objective": float(best_obj),
            "n_train": int(n_train),
            "n_val": int(n - n_train),
            "nu_fixed": float(nu),
            "gamma_fixed": float(gamma),
        }

        return float(q_opt), float(c_opt), float(phi_opt), float(-best_obj), diagnostics

    @classmethod
    def estimate_gamma_from_returns(
        cls,
        returns: np.ndarray,
        min_obs: int = 100
    ) -> Tuple[float, Dict]:
        """
        Estimate skewness parameter γ from historical returns.
        
        Uses method of moments: compare empirical skewness to theoretical
        skewness under Fernández-Steel parameterization.
        
        For the skew-t with γ ≠ 1:
            E[skewness] ∝ (γ - 1/γ) / (γ + 1/γ)^(3/2)
        
        This mapping is monotone, allowing inversion from sample skewness.
        
        Args:
            returns: Array of returns
            min_obs: Minimum observations required
            
        Returns:
            Tuple of (gamma_estimate, diagnostics)
        """
        n = len(returns)
        if n < min_obs:
            return GAMMA_DEFAULT, {"method": "default_insufficient_data", "n": n}

        # Compute sample skewness (robust: trimmed)
        ret_trimmed = np.sort(returns)[int(0.05 * n):int(0.95 * n)]
        if len(ret_trimmed) < 20:
            return GAMMA_DEFAULT, {"method": "default_trimming_failed", "n": n}

        mean_r = np.mean(ret_trimmed)
        std_r = np.std(ret_trimmed)
        if std_r < 1e-10:
            return GAMMA_DEFAULT, {"method": "default_zero_variance"}

        skew_sample = np.mean(((ret_trimmed - mean_r) / std_r) ** 3)

        # Map sample skewness to γ
        # Approximate inverse: γ ≈ 1 + κ * skew, bounded
        kappa = 0.3  # Sensitivity coefficient (tuned empirically)
        gamma_est = 1.0 + kappa * skew_sample
        gamma_est = float(np.clip(gamma_est, GAMMA_MIN, GAMMA_MAX))

        return gamma_est, {
            "method": "moment_matching",
            "sample_skewness": float(skew_sample),
            "kappa": kappa,
            "n_trimmed": len(ret_trimmed),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS FOR BMA INTEGRATION
# =============================================================================

def compute_bic_skew_t(log_likelihood: float, n_params: int, n_obs: int) -> float:
    """Compute BIC for skew-t model. 4 params: q, c, φ, (ν and γ are fixed per sub-model)."""
    if n_obs <= 0:
        return float('inf')
    return -2.0 * log_likelihood + n_params * np.log(n_obs)


def compute_aic_skew_t(log_likelihood: float, n_params: int) -> float:
    """Compute AIC for skew-t model."""
    return -2.0 * log_likelihood + 2.0 * n_params


def get_skew_t_model_name(nu: float, gamma: float) -> str:
    """
    Generate model name for BMA ensemble.
    
    Format: phi_skew_t_nu_{nu}_gamma_{gamma}
    Example: phi_skew_t_nu_6_gamma_0.85
    """
    # Round gamma to avoid floating point issues
    gamma_str = f"{gamma:.2f}".replace(".", "p")
    return f"phi_skew_t_nu_{int(nu)}_gamma_{gamma_str}"


def parse_skew_t_model_name(model_name: str) -> Optional[Tuple[float, float]]:
    """
    Parse ν and γ from model name.
    
    Returns (nu, gamma) or None if not a skew-t model.
    """
    if not model_name.startswith("phi_skew_t_nu_"):
        return None
    
    try:
        # phi_skew_t_nu_6_gamma_0p85
        parts = model_name.split("_")
        nu_idx = parts.index("nu") + 1
        gamma_idx = parts.index("gamma") + 1
        
        nu = float(parts[nu_idx])
        gamma_str = parts[gamma_idx].replace("p", ".")
        gamma = float(gamma_str)
        
        return (nu, gamma)
    except (ValueError, IndexError):
        return None


def is_skew_t_model(model_name: str) -> bool:
    """Check if model name is a skew-t variant."""
    return model_name.startswith("phi_skew_t_")
