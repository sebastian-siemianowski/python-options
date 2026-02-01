"""
===============================================================================
CONTAMINATED STUDENT-T MIXTURE MODEL
===============================================================================

Implements the Regime-Indexed Contaminated Student-t distribution as recommended
by the Expert Panel for capturing distinct fat-tail behavior in normal versus
stressed market regimes.

MATHEMATICAL MODEL:
    p(r) = (1 - ε) × t(r; μ, σ, ν_normal) + ε × t(r; μ, σ, ν_crisis)

Where:
    - ν_normal: Degrees of freedom for typical market conditions (lighter tails)
    - ν_crisis: Degrees of freedom for extreme events (heavier tails, ν_crisis < ν_normal)
    - ε: Contamination probability (tied to vol_regime or drawdown)

CORE PRINCIPLE:
    "5% of the time we're in crisis mode with ν=4, 95% of time we're normal with ν=12"
    This provides intuitive risk management interpretation while maintaining
    statistical rigor.

INTEGRATION:
    - Tuning: Profile likelihood estimation of (ν_normal, ν_crisis, ε)
    - Signals: MC sampling with probability ε from crisis component
    - Fallback: If ε → 0 or ν_crisis → ν_normal, collapses to single Student-t

REFERENCES:
    Tukey, J.W. (1960). "A Survey of Sampling from Contaminated Distributions"
    Lange, K.L., Little, R.J.A., Taylor, J.M.G. (1989). "Robust Statistical Modeling 
        Using the t Distribution"

===============================================================================
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
from scipy.special import gammaln


# =============================================================================
# CONTAMINATED STUDENT-T CONSTANTS
# =============================================================================

# Degrees of freedom bounds
CST_NU_MIN = 2.5           # Minimum ν for finite variance
CST_NU_MAX = 100.0         # Above this, effectively Gaussian
CST_NU_NORMAL_DEFAULT = 12.0   # Default for calm periods
CST_NU_CRISIS_DEFAULT = 4.0    # Default for crisis periods (heavy tails)

# Contamination probability bounds
CST_EPSILON_MIN = 0.01     # Minimum crisis probability (1%)
CST_EPSILON_MAX = 0.30     # Maximum crisis probability (30%)
CST_EPSILON_DEFAULT = 0.05 # Default 5% contamination

# Grid for ν optimization
CST_NU_GRID_NORMAL = [6, 8, 10, 12, 15, 20, 30, 50]
CST_NU_GRID_CRISIS = [3, 4, 5, 6, 8]

# Minimum observations for reliable fitting
CST_MIN_OBS = 100
CST_MIN_CRISIS_OBS = 20    # Minimum observations in crisis regime for ν_crisis estimation


@dataclass
class ContaminatedStudentTParams:
    """
    Parameters for the Contaminated Student-t Mixture Model.
    
    Attributes:
        nu_normal: Degrees of freedom for normal market conditions
        nu_crisis: Degrees of freedom for crisis conditions (heavier tails)
        epsilon: Contamination probability (probability of crisis component)
        epsilon_source: How epsilon was determined ('vol_regime', 'drawdown', 'mle', 'default')
    """
    nu_normal: float
    nu_crisis: float
    epsilon: float
    epsilon_source: str = "default"
    
    def __post_init__(self):
        # Validate parameters
        if self.nu_normal <= 2:
            raise ValueError(f"nu_normal must be > 2 for finite variance, got {self.nu_normal}")
        if self.nu_crisis <= 2:
            raise ValueError(f"nu_crisis must be > 2 for finite variance, got {self.nu_crisis}")
        if not 0 <= self.epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")
        # Crisis should have heavier tails (smaller ν)
        if self.nu_crisis > self.nu_normal:
            warnings.warn(f"nu_crisis ({self.nu_crisis}) > nu_normal ({self.nu_normal}); "
                         f"typically crisis has heavier tails")
    
    @property
    def is_degenerate(self) -> bool:
        """Check if mixture is effectively single-component."""
        return (
            self.epsilon < 0.001 or 
            self.epsilon > 0.999 or
            abs(self.nu_crisis - self.nu_normal) < 0.5
        )
    
    @property
    def effective_nu(self) -> float:
        """Compute effective ν as weighted harmonic mean."""
        # Harmonic mean better represents tail behavior than arithmetic
        return 1.0 / ((1 - self.epsilon) / self.nu_normal + self.epsilon / self.nu_crisis)
    
    @property
    def tail_heaviness_ratio(self) -> float:
        """Ratio of crisis to normal tail heaviness (> 1 means crisis has heavier tails)."""
        return self.nu_normal / self.nu_crisis if self.nu_crisis > 0 else 1.0
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "nu_normal": float(self.nu_normal),
            "nu_crisis": float(self.nu_crisis),
            "epsilon": float(self.epsilon),
            "epsilon_source": self.epsilon_source,
            "effective_nu": float(self.effective_nu),
            "is_degenerate": self.is_degenerate,
            "tail_heaviness_ratio": float(self.tail_heaviness_ratio),
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ContaminatedStudentTParams':
        """Deserialize from dictionary."""
        return cls(
            nu_normal=float(d["nu_normal"]),
            nu_crisis=float(d["nu_crisis"]),
            epsilon=float(d["epsilon"]),
            epsilon_source=d.get("epsilon_source", "unknown"),
        )
    
    @classmethod
    def default(cls) -> 'ContaminatedStudentTParams':
        """Return default parameters."""
        return cls(
            nu_normal=CST_NU_NORMAL_DEFAULT,
            nu_crisis=CST_NU_CRISIS_DEFAULT,
            epsilon=CST_EPSILON_DEFAULT,
            epsilon_source="default",
        )


def student_t_logpdf(x: np.ndarray, nu: float, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """
    Log-PDF of Student-t distribution (location-scale parameterization).
    
    Args:
        x: Observations
        nu: Degrees of freedom
        mu: Location parameter
        sigma: Scale parameter
        
    Returns:
        Log-PDF values
    """
    if sigma <= 0 or nu <= 0:
        return np.full_like(x, -np.inf, dtype=float)
    
    z = (x - mu) / sigma
    
    log_norm = (
        gammaln((nu + 1) / 2) - 
        gammaln(nu / 2) - 
        0.5 * np.log(nu * np.pi) - 
        np.log(sigma)
    )
    
    log_kernel = -((nu + 1) / 2) * np.log(1 + z**2 / nu)
    
    return log_norm + log_kernel


def contaminated_student_t_logpdf(
    x: np.ndarray,
    nu_normal: float,
    nu_crisis: float,
    epsilon: float,
    mu: float = 0.0,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Log-PDF of the Contaminated Student-t Mixture.
    
    p(x) = (1-ε) × t(x; μ, σ, ν_normal) + ε × t(x; μ, σ, ν_crisis)
    
    Uses log-sum-exp trick for numerical stability.
    
    Args:
        x: Observations
        nu_normal: Degrees of freedom for normal component
        nu_crisis: Degrees of freedom for crisis component
        epsilon: Contamination probability
        mu: Location parameter (shared)
        sigma: Scale parameter (shared)
        
    Returns:
        Log-PDF values
    """
    x = np.asarray(x)
    
    # Handle edge cases
    if epsilon < 1e-10:
        return student_t_logpdf(x, nu_normal, mu, sigma)
    if epsilon > 1 - 1e-10:
        return student_t_logpdf(x, nu_crisis, mu, sigma)
    
    # Log probabilities of components
    log_p_normal = np.log(1 - epsilon) + student_t_logpdf(x, nu_normal, mu, sigma)
    log_p_crisis = np.log(epsilon) + student_t_logpdf(x, nu_crisis, mu, sigma)
    
    # Log-sum-exp for numerical stability
    max_log = np.maximum(log_p_normal, log_p_crisis)
    log_sum = max_log + np.log(
        np.exp(log_p_normal - max_log) + np.exp(log_p_crisis - max_log)
    )
    
    return log_sum


def contaminated_student_t_pdf(
    x: np.ndarray,
    nu_normal: float,
    nu_crisis: float,
    epsilon: float,
    mu: float = 0.0,
    sigma: float = 1.0
) -> np.ndarray:
    """
    PDF of the Contaminated Student-t Mixture.
    """
    return np.exp(contaminated_student_t_logpdf(x, nu_normal, nu_crisis, epsilon, mu, sigma))


def contaminated_student_t_rvs(
    size: Union[int, Tuple[int, ...]],
    nu_normal: float,
    nu_crisis: float,
    epsilon: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    random_state: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generate random samples from the Contaminated Student-t Mixture.
    
    For each sample:
        - With probability ε: draw from t(ν_crisis)
        - With probability (1-ε): draw from t(ν_normal)
    
    Args:
        size: Output shape
        nu_normal: Degrees of freedom for normal component
        nu_crisis: Degrees of freedom for crisis component
        epsilon: Contamination probability
        mu: Location parameter
        sigma: Scale parameter
        random_state: NumPy random generator
        
    Returns:
        Random samples from the mixture
    """
    if random_state is None:
        random_state = np.random.default_rng()
    
    # Determine sample size
    if isinstance(size, int):
        n = size
    else:
        n = np.prod(size)
    
    # Sample component indicators
    is_crisis = random_state.random(n) < epsilon
    n_crisis = np.sum(is_crisis)
    n_normal = n - n_crisis
    
    # Sample from each component
    samples = np.empty(n, dtype=float)
    
    if n_normal > 0:
        samples[~is_crisis] = random_state.standard_t(df=nu_normal, size=n_normal)
    if n_crisis > 0:
        samples[is_crisis] = random_state.standard_t(df=nu_crisis, size=n_crisis)
    
    # Scale and shift
    samples = mu + sigma * samples
    
    # Reshape if necessary
    if isinstance(size, tuple):
        samples = samples.reshape(size)
    
    return samples


def compute_crisis_probability_from_vol(
    current_vol: float,
    vol_history: np.ndarray,
    high_vol_threshold_percentile: float = 0.80
) -> float:
    """
    Compute contamination probability ε from volatility regime.
    
    Args:
        current_vol: Current volatility estimate
        vol_history: Historical volatility series
        high_vol_threshold_percentile: Percentile defining high vol regime
        
    Returns:
        Contamination probability ε ∈ [CST_EPSILON_MIN, CST_EPSILON_MAX]
    """
    vol_history = np.asarray(vol_history)
    vol_history = vol_history[np.isfinite(vol_history)]
    
    if len(vol_history) < 20:
        return CST_EPSILON_DEFAULT
    
    # Compute threshold for high volatility
    threshold = np.percentile(vol_history, high_vol_threshold_percentile * 100)
    
    # Historical fraction in high-vol regime
    historical_high_vol_frac = np.mean(vol_history > threshold)
    
    # Current regime indicator (soft)
    if current_vol > threshold:
        # In high vol regime: increase ε
        epsilon = CST_EPSILON_DEFAULT + 0.5 * (CST_EPSILON_MAX - CST_EPSILON_DEFAULT)
    elif current_vol > np.median(vol_history):
        # Elevated vol: moderate ε
        epsilon = CST_EPSILON_DEFAULT + 0.2 * (CST_EPSILON_MAX - CST_EPSILON_DEFAULT)
    else:
        # Low vol: base ε
        epsilon = CST_EPSILON_DEFAULT
    
    # Blend with historical high-vol frequency
    epsilon = 0.7 * epsilon + 0.3 * historical_high_vol_frac
    
    return float(np.clip(epsilon, CST_EPSILON_MIN, CST_EPSILON_MAX))


def compute_crisis_probability_from_drawdown(
    current_drawdown: float,
    drawdown_history: np.ndarray,
    crisis_drawdown_threshold: float = 0.10
) -> float:
    """
    Compute contamination probability ε from drawdown.
    
    Args:
        current_drawdown: Current drawdown (positive value, e.g., 0.05 = 5%)
        drawdown_history: Historical drawdown series
        crisis_drawdown_threshold: Threshold for crisis (default 10%)
        
    Returns:
        Contamination probability ε ∈ [CST_EPSILON_MIN, CST_EPSILON_MAX]
    """
    current_drawdown = abs(current_drawdown)
    
    if current_drawdown > crisis_drawdown_threshold:
        # Deep drawdown: high crisis probability
        epsilon = CST_EPSILON_DEFAULT + 0.7 * (CST_EPSILON_MAX - CST_EPSILON_DEFAULT)
    elif current_drawdown > crisis_drawdown_threshold / 2:
        # Moderate drawdown
        epsilon = CST_EPSILON_DEFAULT + 0.3 * (CST_EPSILON_MAX - CST_EPSILON_DEFAULT)
    else:
        epsilon = CST_EPSILON_DEFAULT
    
    return float(np.clip(epsilon, CST_EPSILON_MIN, CST_EPSILON_MAX))


def fit_contaminated_student_t_profile(
    returns: np.ndarray,
    vol_regime_labels: Optional[np.ndarray] = None,
    nu_grid_normal: Optional[list] = None,
    nu_grid_crisis: Optional[list] = None,
    epsilon_grid: Optional[list] = None,
) -> Tuple[ContaminatedStudentTParams, Dict]:
    """
    Fit Contaminated Student-t using profile likelihood over grids.
    
    Profile likelihood approach:
    1. For each (ν_normal, ν_crisis, ε) in grid
    2. Compute log-likelihood
    3. Select parameters maximizing log-likelihood
    4. Apply constraint: ν_crisis ≤ ν_normal
    
    Args:
        returns: Return series (standardized or raw)
        vol_regime_labels: Optional binary labels (1 = high vol/crisis, 0 = normal)
        nu_grid_normal: Grid of ν values for normal component
        nu_grid_crisis: Grid of ν values for crisis component
        epsilon_grid: Grid of ε values
        
    Returns:
        Tuple of (fitted_params, diagnostics)
    """
    returns = np.asarray(returns).flatten()
    returns = returns[np.isfinite(returns)]
    
    n = len(returns)
    
    if n < CST_MIN_OBS:
        # Insufficient data - return defaults
        return ContaminatedStudentTParams.default(), {
            "fit_success": False,
            "error": "insufficient_data",
            "n_obs": n,
        }
    
    # Standardize returns for fitting
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma < 1e-10:
        return ContaminatedStudentTParams.default(), {
            "fit_success": False,
            "error": "zero_variance",
        }
    z = (returns - mu) / sigma
    
    # Set default grids
    if nu_grid_normal is None:
        nu_grid_normal = CST_NU_GRID_NORMAL
    if nu_grid_crisis is None:
        nu_grid_crisis = CST_NU_GRID_CRISIS
    if epsilon_grid is None:
        epsilon_grid = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    
    # Estimate epsilon from vol regime if available
    epsilon_from_regime = None
    if vol_regime_labels is not None:
        vol_regime_labels = np.asarray(vol_regime_labels).flatten()
        if len(vol_regime_labels) == n:
            crisis_frac = np.mean(vol_regime_labels == 1)
            epsilon_from_regime = float(np.clip(crisis_frac, CST_EPSILON_MIN, CST_EPSILON_MAX))
    
    # Profile likelihood search
    best_ll = -np.inf
    best_params = None
    all_results = []
    
    for nu_n in nu_grid_normal:
        for nu_c in nu_grid_crisis:
            # Enforce constraint: crisis has heavier tails (smaller ν)
            if nu_c > nu_n:
                continue
            
            for eps in epsilon_grid:
                # Compute log-likelihood
                ll = np.sum(contaminated_student_t_logpdf(
                    z, nu_normal=nu_n, nu_crisis=nu_c, epsilon=eps
                ))
                
                all_results.append({
                    "nu_normal": nu_n,
                    "nu_crisis": nu_c,
                    "epsilon": eps,
                    "log_likelihood": ll,
                })
                
                if ll > best_ll:
                    best_ll = ll
                    best_params = (nu_n, nu_c, eps)
    
    if best_params is None:
        return ContaminatedStudentTParams.default(), {
            "fit_success": False,
            "error": "optimization_failed",
        }
    
    nu_normal_hat, nu_crisis_hat, epsilon_hat = best_params
    
    # If we have regime-based epsilon, blend with MLE
    if epsilon_from_regime is not None:
        epsilon_final = 0.5 * epsilon_hat + 0.5 * epsilon_from_regime
        epsilon_source = "blended_mle_regime"
    else:
        epsilon_final = epsilon_hat
        epsilon_source = "mle"
    
    # Compute comparison metrics
    # Single Student-t log-likelihood for comparison
    single_nu_candidates = [4, 6, 8, 10, 12, 15, 20]
    best_single_ll = -np.inf
    best_single_nu = 10
    for nu in single_nu_candidates:
        ll_single = np.sum(student_t_logpdf(z, nu))
        if ll_single > best_single_ll:
            best_single_ll = ll_single
            best_single_nu = nu
    
    # Compute BIC
    k_mixture = 3  # nu_normal, nu_crisis, epsilon
    k_single = 1   # just nu
    bic_mixture = -2 * best_ll + k_mixture * np.log(n)
    bic_single = -2 * best_single_ll + k_single * np.log(n)
    
    # AIC
    aic_mixture = -2 * best_ll + 2 * k_mixture
    aic_single = -2 * best_single_ll + 2 * k_single
    
    params = ContaminatedStudentTParams(
        nu_normal=nu_normal_hat,
        nu_crisis=nu_crisis_hat,
        epsilon=epsilon_final,
        epsilon_source=epsilon_source,
    )
    
    diagnostics = {
        "fit_success": True,
        "n_obs": n,
        "log_likelihood": float(best_ll),
        "bic": float(bic_mixture),
        "aic": float(aic_mixture),
        "best_single_nu": best_single_nu,
        "single_log_likelihood": float(best_single_ll),
        "single_bic": float(bic_single),
        "single_aic": float(aic_single),
        "delta_bic": float(bic_mixture - bic_single),  # Negative = mixture better
        "delta_aic": float(aic_mixture - aic_single),
        "mixture_preferred": bic_mixture < bic_single,
        "epsilon_from_regime": epsilon_from_regime,
        "n_grid_points_searched": len(all_results),
    }
    
    return params, diagnostics


def compute_contaminated_pit(
    returns: np.ndarray,
    params: ContaminatedStudentTParams,
    mu: float = 0.0,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Compute Probability Integral Transform for contaminated Student-t.
    
    For mixture: F(x) = (1-ε) × F_normal(x) + ε × F_crisis(x)
    
    Args:
        returns: Observed returns
        params: Contaminated Student-t parameters
        mu: Location
        sigma: Scale
        
    Returns:
        PIT values (should be uniform if model is well-calibrated)
    """
    returns = np.asarray(returns)
    z = (returns - mu) / sigma
    
    # CDF of each component
    cdf_normal = stats.t.cdf(z, df=params.nu_normal)
    cdf_crisis = stats.t.cdf(z, df=params.nu_crisis)
    
    # Mixture CDF
    pit = (1 - params.epsilon) * cdf_normal + params.epsilon * cdf_crisis
    
    return pit


def compare_contaminated_vs_single(
    returns: np.ndarray,
    params: ContaminatedStudentTParams,
    single_nu: float
) -> Dict:
    """
    Compare contaminated mixture against single Student-t.
    
    Args:
        returns: Return series
        params: Fitted contaminated parameters
        single_nu: Single Student-t degrees of freedom
        
    Returns:
        Comparison metrics
    """
    returns = np.asarray(returns).flatten()
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    
    # Standardize
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    z = (returns - mu) / sigma
    
    # Log-likelihoods
    ll_mixture = np.sum(contaminated_student_t_logpdf(
        z, params.nu_normal, params.nu_crisis, params.epsilon
    ))
    ll_single = np.sum(student_t_logpdf(z, single_nu))
    
    # Information criteria
    k_mixture = 3
    k_single = 1
    
    bic_mixture = -2 * ll_mixture + k_mixture * np.log(n)
    bic_single = -2 * ll_single + k_single * np.log(n)
    
    aic_mixture = -2 * ll_mixture + 2 * k_mixture
    aic_single = -2 * ll_single + 2 * k_single
    
    # Likelihood ratio test
    lr_stat = 2 * (ll_mixture - ll_single)
    # df = k_mixture - k_single = 2
    lr_pvalue = 1 - stats.chi2.cdf(lr_stat, df=2) if lr_stat > 0 else 1.0
    
    return {
        "ll_mixture": float(ll_mixture),
        "ll_single": float(ll_single),
        "bic_mixture": float(bic_mixture),
        "bic_single": float(bic_single),
        "aic_mixture": float(aic_mixture),
        "aic_single": float(aic_single),
        "delta_bic": float(bic_mixture - bic_single),
        "delta_aic": float(aic_mixture - aic_single),
        "lr_statistic": float(lr_stat),
        "lr_pvalue": float(lr_pvalue),
        "mixture_preferred_bic": bic_mixture < bic_single,
        "mixture_preferred_aic": aic_mixture < aic_single,
        "mixture_significant_lr": lr_pvalue < 0.05,
    }
