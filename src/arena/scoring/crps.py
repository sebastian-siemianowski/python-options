"""
===============================================================================
CRPS — Continuous Ranked Probability Score
===============================================================================

The CRPS is a strictly proper scoring rule that measures the quality of
probabilistic forecasts. It rewards both:
- Calibration (reliability): forecasted probabilities match observed frequencies
- Sharpness (confidence): narrow predictive distributions when correct

For a predictive CDF F and observation y:
    CRPS(F, y) = ∫ (F(x) - 1{x ≥ y})² dx

Lower CRPS is better. CRPS = 0 means perfect forecast.

Key Properties:
1. Strictly proper: minimized when forecast = true distribution
2. Same units as the variable being forecast
3. Closed-form for location-scale families (Gaussian, Student-t)

Decomposition:
    CRPS = Reliability - Resolution + Uncertainty
    
    Reliability: measures calibration (0 = perfect calibration)
    Sharpness: measures confidence (lower = sharper forecasts)

Reference: Gneiting & Raftery (2007), Gneiting et al. (2005)

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from scipy.special import gamma, gammaln
from scipy.stats import norm, t as student_t


@dataclass
class CRPSResult:
    """
    Result of CRPS computation.
    
    Attributes:
        crps: The CRPS value (lower is better)
        reliability: Calibration component (0 = perfect)
        sharpness: Confidence component (lower = sharper)
        n_observations: Number of observations used
        distribution: Distribution type used
    """
    crps: float
    reliability: float
    sharpness: float
    n_observations: int
    distribution: str
    
    @property
    def skill_score(self) -> float:
        """
        CRPS Skill Score relative to climatological forecast.
        
        CRPSS = 1 - CRPS / CRPS_ref
        
        Positive = better than reference, negative = worse.
        """
        # Use sharpness as proxy for climatological CRPS
        if self.sharpness > 0:
            return 1.0 - self.crps / self.sharpness
        return 0.0


def compute_crps_gaussian(
    observations: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> CRPSResult:
    """
    Compute CRPS for Gaussian predictive distributions.
    
    Closed-form formula:
        CRPS(N(μ,σ²), y) = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
        
    where z = (y - μ) / σ, Φ is standard normal CDF, φ is PDF.
    
    Args:
        observations: Actual observed values (n,)
        mu: Predicted means (n,)
        sigma: Predicted standard deviations (n,)
        
    Returns:
        CRPSResult with CRPS and decomposition
    """
    observations = np.asarray(observations).flatten()
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    
    # Ensure positive sigma
    sigma = np.maximum(sigma, 1e-10)
    
    n = len(observations)
    
    # Standardized residual
    z = (observations - mu) / sigma
    
    # Standard normal PDF and CDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # CRPS for each observation (closed-form for Gaussian)
    # CRPS = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
    crps_individual = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / np.sqrt(np.pi))
    
    # Average CRPS
    crps_mean = np.mean(crps_individual)
    
    # Decomposition
    # Reliability: deviation from perfect calibration
    # For well-calibrated forecasts, PIT values should be uniform
    pit_values = Phi_z  # PIT for Gaussian
    reliability = _compute_reliability(pit_values)
    
    # Sharpness: average predictive standard deviation
    sharpness = np.mean(sigma)
    
    return CRPSResult(
        crps=float(crps_mean),
        reliability=float(reliability),
        sharpness=float(sharpness),
        n_observations=n,
        distribution="gaussian",
    )


def compute_crps_student_t(
    observations: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    nu: float,
) -> CRPSResult:
    """
    Compute CRPS for Student-t predictive distributions.
    
    Uses the closed-form expression from Gneiting & Raftery (2007):
    
    CRPS(t_ν(μ,σ), y) = σ * [z*(2T_ν(z)-1) + 2t_ν(z)*(ν+z²)/(ν-1) 
                          - 2√ν * B(1/2, ν-1/2) / ((ν-1)*B(1/2, ν/2)²)]
    
    where z = (y-μ)/σ, T_ν is Student-t CDF, t_ν is PDF, B is beta function.
    
    Args:
        observations: Actual observed values (n,)
        mu: Predicted means (n,)
        sigma: Predicted scale parameters (n,)
        nu: Degrees of freedom (must be > 1 for finite CRPS)
        
    Returns:
        CRPSResult with CRPS and decomposition
    """
    observations = np.asarray(observations).flatten()
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    
    # Ensure positive sigma and valid nu
    sigma = np.maximum(sigma, 1e-10)
    nu = max(nu, 1.01)  # CRPS requires ν > 1
    
    n = len(observations)
    
    # Standardized residual
    z = (observations - mu) / sigma
    
    # Student-t PDF and CDF
    t_dist = student_t(df=nu)
    pdf_z = t_dist.pdf(z)
    cdf_z = t_dist.cdf(z)
    
    # Closed-form CRPS for Student-t
    # Using formula from Gneiting & Raftery (2007)
    if nu > 1:
        # Beta function ratios using log-gamma for numerical stability
        log_B_half_nu_minus_half = gammaln(0.5) + gammaln(nu - 0.5) - gammaln(nu)
        log_B_half_nu_half = gammaln(0.5) + gammaln(nu / 2) - gammaln((nu + 1) / 2)
        
        B_ratio = np.exp(log_B_half_nu_minus_half - 2 * log_B_half_nu_half)
        
        # CRPS components
        term1 = z * (2 * cdf_z - 1)
        term2 = 2 * pdf_z * (nu + z**2) / (nu - 1)
        term3 = 2 * np.sqrt(nu) * B_ratio / (nu - 1)
        
        crps_individual = sigma * (term1 + term2 - term3)
    else:
        # Fallback to numerical integration for ν ≤ 1
        crps_individual = np.array([
            _crps_numerical(y, m, s, nu) 
            for y, m, s in zip(observations, mu, sigma)
        ])
    
    # Average CRPS
    crps_mean = np.mean(crps_individual)
    
    # Decomposition
    pit_values = cdf_z
    reliability = _compute_reliability(pit_values)
    sharpness = np.mean(sigma) * np.sqrt(nu / (nu - 2)) if nu > 2 else np.mean(sigma)
    
    return CRPSResult(
        crps=float(crps_mean),
        reliability=float(reliability),
        sharpness=float(sharpness),
        n_observations=n,
        distribution=f"student_t_nu_{nu:.0f}",
    )


def compute_crps_empirical(
    observations: np.ndarray,
    forecast_samples: np.ndarray,
) -> CRPSResult:
    """
    Compute CRPS using empirical (sample-based) predictive distribution.
    
    For M samples x₁,...,xₘ from predictive distribution:
        CRPS = (1/M) Σᵢ |xᵢ - y| - (1/2M²) Σᵢⱼ |xᵢ - xⱼ|
    
    This is useful when no closed-form exists (mixtures, complex models).
    
    Args:
        observations: Actual observed values (n,)
        forecast_samples: Samples from predictive distribution (n, M)
        
    Returns:
        CRPSResult with CRPS and decomposition
    """
    observations = np.asarray(observations).flatten()
    forecast_samples = np.asarray(forecast_samples)
    
    if forecast_samples.ndim == 1:
        forecast_samples = forecast_samples.reshape(-1, 1)
    
    n, M = forecast_samples.shape
    
    crps_values = np.zeros(n)
    
    for i in range(n):
        y = observations[i]
        samples = forecast_samples[i, :]
        
        # First term: average absolute error
        term1 = np.mean(np.abs(samples - y))
        
        # Second term: average pairwise distance (divided by 2)
        # Efficient computation using sorted samples
        sorted_samples = np.sort(samples)
        term2 = np.mean(np.abs(np.subtract.outer(samples, samples))) / 2
        
        crps_values[i] = term1 - term2
    
    crps_mean = np.mean(crps_values)
    
    # Empirical PIT
    pit_values = np.array([
        np.mean(forecast_samples[i, :] <= observations[i])
        for i in range(n)
    ])
    reliability = _compute_reliability(pit_values)
    
    # Sharpness: average predictive spread
    sharpness = np.mean(np.std(forecast_samples, axis=1))
    
    return CRPSResult(
        crps=float(crps_mean),
        reliability=float(reliability),
        sharpness=float(sharpness),
        n_observations=n,
        distribution="empirical",
    )


def decompose_crps(
    observations: np.ndarray,
    predicted_cdf_values: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, float]:
    """
    Decompose CRPS into reliability, resolution, and uncertainty.
    
    Following Hersbach (2000):
        CRPS = Reliability - Resolution + Uncertainty
    
    Args:
        observations: Actual observed values
        predicted_cdf_values: F(y) for each observation (PIT values)
        n_bins: Number of bins for decomposition
        
    Returns:
        (reliability, resolution, uncertainty)
    """
    n = len(observations)
    pit = np.asarray(predicted_cdf_values).flatten()
    
    # Bin edges for probability bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    reliability = 0.0
    resolution = 0.0
    
    # Overall frequency of y > F⁻¹(p) for each probability level
    overall_freq = np.mean(pit)
    
    for k in range(n_bins):
        p_low, p_high = bin_edges[k], bin_edges[k + 1]
        p_mid = (p_low + p_high) / 2
        
        # Observations in this probability bin
        in_bin = (pit >= p_low) & (pit < p_high)
        n_k = np.sum(in_bin)
        
        if n_k > 0:
            # Observed frequency in bin
            o_k = np.mean(pit[in_bin])
            
            # Reliability: weighted squared deviation from diagonal
            reliability += n_k * (o_k - p_mid) ** 2
            
            # Resolution: weighted squared deviation from climatology
            resolution += n_k * (o_k - overall_freq) ** 2
    
    reliability /= n
    resolution /= n
    
    # Uncertainty: variance of observations (climatological)
    uncertainty = overall_freq * (1 - overall_freq)
    
    return reliability, resolution, uncertainty


def _compute_reliability(pit_values: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute reliability component from PIT values.
    
    For well-calibrated forecasts, PIT values should be uniform on [0,1].
    Reliability measures deviation from uniformity.
    
    Args:
        pit_values: Probability Integral Transform values
        n_bins: Number of bins for histogram
        
    Returns:
        Reliability score (0 = perfect calibration)
    """
    pit = np.asarray(pit_values).flatten()
    n = len(pit)
    
    if n < n_bins:
        return 0.0
    
    # Expected count per bin for uniform distribution
    expected = n / n_bins
    
    # Observed counts
    hist, _ = np.histogram(pit, bins=n_bins, range=(0, 1))
    
    # Reliability as normalized chi-squared statistic
    reliability = np.sum((hist - expected) ** 2 / expected) / n_bins
    
    return float(reliability)


def _crps_numerical(y: float, mu: float, sigma: float, nu: float) -> float:
    """
    Compute CRPS via numerical integration (fallback for edge cases).
    
    CRPS = ∫ (F(x) - 1{x ≥ y})² dx
    """
    from scipy.integrate import quad
    
    t_dist = student_t(df=nu, loc=mu, scale=sigma)
    
    def integrand(x):
        F_x = t_dist.cdf(x)
        indicator = 1.0 if x >= y else 0.0
        return (F_x - indicator) ** 2
    
    # Integrate from -∞ to +∞ (use practical bounds)
    lower = mu - 10 * sigma
    upper = mu + 10 * sigma
    
    result, _ = quad(integrand, lower, upper, limit=100)
    return result
