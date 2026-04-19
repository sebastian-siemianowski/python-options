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
from scipy.optimize import minimize_scalar, minimize, brentq
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
    single_nu_candidates = [4, 8, 20]
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


# =============================================================================
# EPIC 18 CONSTANTS
# =============================================================================

# EM convergence
EM_DEFAULT_ITER = 50
EM_TOL = 0.01  # nats

# Minimum observations for EM
EM_MIN_OBS = 30

# Jump threshold
JUMP_THRESHOLD = 0.5

# Q inflation on jumps
JUMP_Q_MULTIPLIER = 10.0

# Epsilon bounds for EM (tighter than profile)
EM_EPSILON_MIN = 0.02
EM_EPSILON_MAX = 0.20

# Nu grids for EM M-step
EM_NU_NORMAL_GRID = np.array([4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0])
EM_NU_CRISIS_GRID = np.array([2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0])

# Nu bounds for EM
EM_NU_NORMAL_MIN = 4.0
EM_NU_NORMAL_MAX = 30.0
EM_NU_CRISIS_MIN = 2.5
EM_NU_CRISIS_MAX = 8.0


# =============================================================================
# EPIC 18: Story 18.1 -- EM CST Fit
# =============================================================================

@dataclass(frozen=True)
class CSTFitResult:
    """Result of EM-based CST fitting.

    Attributes
    ----------
    epsilon : float
        Contamination probability (fraction from crisis component).
    nu_normal : float
        Degrees of freedom for the normal component.
    nu_crisis : float
        Degrees of freedom for the crisis component (nu_crisis < nu_normal).
    log_likelihood : float
        Final log-likelihood of the mixture model.
    bic : float
        BIC of the fitted model (3 parameters).
    n_iter : int
        Number of EM iterations run.
    converged : bool
        Whether the EM converged within tolerance.
    responsibilities : np.ndarray
        Final posterior responsibilities gamma_t = P(crisis | r_t).
    """
    epsilon: float
    nu_normal: float
    nu_crisis: float
    log_likelihood: float
    bic: float
    n_iter: int
    converged: bool
    responsibilities: np.ndarray


def _em_compute_responsibilities(z: np.ndarray, epsilon: float,
                                  nu_normal: float, nu_crisis: float) -> np.ndarray:
    """E-step: compute gamma_t = P(crisis | z_t) via log-sum-exp."""
    log_p_normal = _em_student_t_logpdf(z, nu_normal) + np.log(
        max(1.0 - epsilon, 1e-12)
    )
    log_p_crisis = _em_student_t_logpdf(z, nu_crisis) + np.log(
        max(epsilon, 1e-12)
    )
    log_max = np.maximum(log_p_normal, log_p_crisis)
    log_denom = log_max + np.log(
        np.exp(log_p_normal - log_max) + np.exp(log_p_crisis - log_max)
    )
    gamma = np.exp(log_p_crisis - log_denom)
    return np.clip(gamma, 1e-10, 1.0 - 1e-10)


def _em_student_t_logpdf(z: np.ndarray, nu: float) -> np.ndarray:
    """Standardized Student-t log-pdf for EM internal use."""
    half_nu = 0.5 * nu
    half_nup1 = 0.5 * (nu + 1.0)
    log_norm = (
        gammaln(half_nup1)
        - gammaln(half_nu)
        - 0.5 * np.log(nu * np.pi)
    )
    log_kernel = -half_nup1 * np.log1p(z ** 2 / nu)
    return log_norm + log_kernel


def _em_weighted_nu_mle(z: np.ndarray, weights: np.ndarray,
                         nu_grid: np.ndarray) -> float:
    """M-step: find nu maximizing weighted log-likelihood on a grid."""
    best_nu = float(nu_grid[0])
    best_ll = -np.inf
    for nu in nu_grid:
        logpdf = _em_student_t_logpdf(z, float(nu))
        wll = float(np.sum(weights * logpdf))
        if wll > best_ll:
            best_ll = wll
            best_nu = float(nu)
    return best_nu


def _em_mixture_loglik(z: np.ndarray, epsilon: float,
                        nu_normal: float, nu_crisis: float) -> float:
    """Compute total log-likelihood of the CST mixture."""
    log_p_normal = _em_student_t_logpdf(z, nu_normal) + np.log(
        max(1.0 - epsilon, 1e-12)
    )
    log_p_crisis = _em_student_t_logpdf(z, nu_crisis) + np.log(
        max(epsilon, 1e-12)
    )
    log_max = np.maximum(log_p_normal, log_p_crisis)
    log_sum = log_max + np.log(
        np.exp(log_p_normal - log_max) + np.exp(log_p_crisis - log_max)
    )
    return float(np.sum(log_sum))


def em_cst_fit(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    n_iter: int = EM_DEFAULT_ITER,
    epsilon_init: float = 0.10,
    nu_normal_init: float = 10.0,
    nu_crisis_init: float = 4.0,
) -> CSTFitResult:
    """Fit a Contaminated Student-t via EM algorithm.

    Parameters
    ----------
    returns : array
        Log returns.
    vol : array
        EWMA volatility estimates (same length as returns).
    q : float
        Kalman process noise.
    c : float
        Kalman observation noise multiplier.
    phi : float
        Kalman AR(1) coefficient.
    n_iter : int
        Maximum EM iterations.
    epsilon_init : float
        Initial contamination probability.
    nu_normal_init : float
        Initial normal component nu.
    nu_crisis_init : float
        Initial crisis component nu.

    Returns
    -------
    CSTFitResult
        Fitted parameters and diagnostics.
    """
    returns = np.asarray(returns, dtype=np.float64)
    vol = np.asarray(vol, dtype=np.float64)

    # Filter NaN / inf
    valid = np.isfinite(returns) & np.isfinite(vol) & (vol > 1e-12)
    r_clean = returns[valid]
    v_clean = vol[valid]
    n = len(r_clean)

    if n < EM_MIN_OBS:
        return CSTFitResult(
            epsilon=epsilon_init,
            nu_normal=nu_normal_init,
            nu_crisis=nu_crisis_init,
            log_likelihood=-1e12,
            bic=1e12,
            n_iter=0,
            converged=False,
            responsibilities=np.full(len(returns), epsilon_init),
        )

    # Compute standardized innovations: z_t = r_t / (c * vol_t)
    scale = c * v_clean
    scale = np.maximum(scale, 1e-12)
    z = r_clean / scale

    # Initialize
    epsilon = float(np.clip(epsilon_init, EM_EPSILON_MIN, EM_EPSILON_MAX))
    nu_normal = float(np.clip(nu_normal_init, EM_NU_NORMAL_MIN, EM_NU_NORMAL_MAX))
    nu_crisis = float(np.clip(nu_crisis_init, EM_NU_CRISIS_MIN, EM_NU_CRISIS_MAX))

    # Ensure nu_crisis < nu_normal
    if nu_crisis >= nu_normal:
        nu_crisis = max(EM_NU_CRISIS_MIN, nu_normal - 2.0)

    prev_ll = -np.inf
    converged = False
    gamma = None
    final_iter = 0

    for iteration in range(n_iter):
        # E-step
        gamma = _em_compute_responsibilities(z, epsilon, nu_normal, nu_crisis)

        # M-step: update epsilon
        epsilon = float(np.clip(np.mean(gamma), EM_EPSILON_MIN, EM_EPSILON_MAX))

        # M-step: update nu_normal (normal weights = 1 - gamma)
        w_normal = 1.0 - gamma
        nu_normal = float(np.clip(
            _em_weighted_nu_mle(z, w_normal, EM_NU_NORMAL_GRID),
            EM_NU_NORMAL_MIN, EM_NU_NORMAL_MAX,
        ))

        # M-step: update nu_crisis (crisis weights = gamma)
        nu_crisis = float(np.clip(
            _em_weighted_nu_mle(z, gamma, EM_NU_CRISIS_GRID),
            EM_NU_CRISIS_MIN, EM_NU_CRISIS_MAX,
        ))

        # Enforce nu_crisis < nu_normal
        if nu_crisis >= nu_normal:
            nu_crisis = max(EM_NU_CRISIS_MIN, nu_normal - 2.0)

        # Check convergence
        ll = _em_mixture_loglik(z, epsilon, nu_normal, nu_crisis)
        final_iter = iteration + 1
        if abs(ll - prev_ll) < EM_TOL:
            converged = True
            prev_ll = ll
            break
        prev_ll = ll

    # Final log-likelihood
    final_ll = _em_mixture_loglik(z, epsilon, nu_normal, nu_crisis)

    # BIC: 3 parameters (epsilon, nu_normal, nu_crisis)
    bic = -2.0 * final_ll + 3.0 * np.log(n)

    # Map responsibilities back to original array
    full_gamma = np.full(len(returns), epsilon)
    full_gamma[valid] = gamma if gamma is not None else epsilon

    return CSTFitResult(
        epsilon=epsilon,
        nu_normal=nu_normal,
        nu_crisis=nu_crisis,
        log_likelihood=float(final_ll),
        bic=float(bic),
        n_iter=final_iter,
        converged=converged,
        responsibilities=full_gamma,
    )


# =============================================================================
# EPIC 18: Story 18.2 -- Online Jump Detection via CST Responsibilities
# =============================================================================

@dataclass(frozen=True)
class JumpProbabilityResult:
    """Result of single-observation jump probability.

    Attributes
    ----------
    gamma : float
        Posterior probability of crisis component: P(crisis | r_t).
    is_jump : bool
        Whether gamma > JUMP_THRESHOLD.
    q_inflation : float
        Multiplicative factor for process noise: 1 + JUMP_Q_MULTIPLIER * gamma.
    log_p_normal : float
        Log-likelihood under normal component.
    log_p_crisis : float
        Log-likelihood under crisis component.
    """
    gamma: float
    is_jump: bool
    q_inflation: float
    log_p_normal: float
    log_p_crisis: float


def cst_jump_probability(
    r_t: float,
    mu_t: float,
    sigma_t: float,
    epsilon: float,
    nu_normal: float,
    nu_crisis: float,
) -> JumpProbabilityResult:
    """Compute posterior jump probability for a single observation.

    Parameters
    ----------
    r_t : float
        Observed return at time t.
    mu_t : float
        Kalman filter predicted mean at time t.
    sigma_t : float
        Predicted standard deviation at time t.
    epsilon : float
        Contamination probability from CST fit.
    nu_normal : float
        Normal component degrees of freedom.
    nu_crisis : float
        Crisis component degrees of freedom.

    Returns
    -------
    JumpProbabilityResult
        Jump detection result with gamma, flag, and q inflation.
    """
    if not np.isfinite(r_t) or not np.isfinite(mu_t):
        return JumpProbabilityResult(
            gamma=epsilon,
            is_jump=False,
            q_inflation=1.0 + JUMP_Q_MULTIPLIER * epsilon,
            log_p_normal=-1e6,
            log_p_crisis=-1e6,
        )

    sigma_safe = max(abs(sigma_t), 1e-12)
    z = np.array([(r_t - mu_t) / sigma_safe])

    lp_normal = float(_em_student_t_logpdf(z, nu_normal)[0])
    lp_crisis = float(_em_student_t_logpdf(z, nu_crisis)[0])

    log_joint_normal = lp_normal + np.log(max(1.0 - epsilon, 1e-12))
    log_joint_crisis = lp_crisis + np.log(max(epsilon, 1e-12))

    log_max = max(log_joint_normal, log_joint_crisis)
    log_denom = log_max + np.log(
        np.exp(log_joint_normal - log_max)
        + np.exp(log_joint_crisis - log_max)
    )

    gamma = float(np.exp(log_joint_crisis - log_denom))
    gamma = max(0.0, min(1.0, gamma))

    is_jump = gamma > JUMP_THRESHOLD
    q_inflation = 1.0 + JUMP_Q_MULTIPLIER * gamma

    return JumpProbabilityResult(
        gamma=gamma,
        is_jump=is_jump,
        q_inflation=q_inflation,
        log_p_normal=lp_normal,
        log_p_crisis=lp_crisis,
    )


def cst_jump_probability_array(
    returns: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    epsilon: float,
    nu_normal: float,
    nu_crisis: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized jump probability for arrays.

    Returns
    -------
    gamma : array
        Posterior crisis probabilities.
    q_inflation : array
        Process noise inflation factors.
    """
    returns = np.asarray(returns, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    sigma_safe = np.maximum(np.abs(sigma), 1e-12)
    z = (returns - mu) / sigma_safe

    lp_normal = _em_student_t_logpdf(z, nu_normal)
    lp_crisis = _em_student_t_logpdf(z, nu_crisis)

    log_joint_normal = lp_normal + np.log(max(1.0 - epsilon, 1e-12))
    log_joint_crisis = lp_crisis + np.log(max(epsilon, 1e-12))

    log_max = np.maximum(log_joint_normal, log_joint_crisis)
    log_denom = log_max + np.log(
        np.exp(log_joint_normal - log_max) + np.exp(log_joint_crisis - log_max)
    )

    gamma = np.clip(np.exp(log_joint_crisis - log_denom), 0.0, 1.0)
    nan_mask = ~np.isfinite(gamma)
    gamma[nan_mask] = epsilon

    q_inflation = 1.0 + JUMP_Q_MULTIPLIER * gamma

    return gamma, q_inflation


# =============================================================================
# EPIC 18: Story 18.3 -- CST-Adjusted Forecast Intervals
# =============================================================================

@dataclass(frozen=True)
class CSTPredictionInterval:
    """CST prediction interval result.

    Attributes
    ----------
    q_lo : float
        Lower quantile.
    q_hi : float
        Upper quantile.
    width : float
        Interval width (q_hi - q_lo).
    width_pure_t : float
        Width from pure Student-t (for comparison).
    width_ratio : float
        CST width / pure-t width (should be >= 1).
    """
    q_lo: float
    q_hi: float
    width: float
    width_pure_t: float
    width_ratio: float


def _cst_cdf_em(x: float, mu: float, sigma: float,
                 epsilon: float, nu_normal: float, nu_crisis: float) -> float:
    """CDF of the CST mixture at point x (for bisection)."""
    sigma_safe = max(abs(sigma), 1e-12)
    z = (x - mu) / sigma_safe
    cdf_normal = float(stats.t.cdf(z, df=nu_normal))
    cdf_crisis = float(stats.t.cdf(z, df=nu_crisis))
    return (1.0 - epsilon) * cdf_normal + epsilon * cdf_crisis


def _cst_quantile_em(p: float, mu: float, sigma: float,
                      epsilon: float, nu_normal: float, nu_crisis: float) -> float:
    """Quantile of the CST mixture via Brent's method on the CDF."""
    sigma_safe = max(abs(sigma), 1e-12)

    # Initial bracket from crisis-component quantiles (wider tails)
    z_lo = float(stats.t.ppf(max(p * 0.1, 1e-8), df=nu_crisis))
    z_hi = float(stats.t.ppf(min(1.0 - (1.0 - p) * 0.1, 1.0 - 1e-8), df=nu_crisis))

    x_lo = mu + z_lo * sigma_safe * 3.0
    x_hi = mu + z_hi * sigma_safe * 3.0

    # Expand bracket until it contains the root
    for _ in range(15):
        f_lo = _cst_cdf_em(x_lo, mu, sigma_safe, epsilon, nu_normal, nu_crisis) - p
        f_hi = _cst_cdf_em(x_hi, mu, sigma_safe, epsilon, nu_normal, nu_crisis) - p
        if f_lo * f_hi < 0:
            break
        spread = x_hi - x_lo
        x_lo -= 0.5 * spread
        x_hi += 0.5 * spread
    else:
        return mu + float(stats.t.ppf(p, df=nu_normal)) * sigma_safe

    try:
        result = brentq(
            lambda x: _cst_cdf_em(x, mu, sigma_safe, epsilon, nu_normal, nu_crisis) - p,
            x_lo, x_hi,
            xtol=1e-8, maxiter=100,
        )
        return float(result)
    except (ValueError, RuntimeError):
        return mu + float(stats.t.ppf(p, df=nu_normal)) * sigma_safe


def cst_prediction_interval(
    mu: float,
    sigma: float,
    epsilon: float,
    nu_normal: float,
    nu_crisis: float,
    alpha: float = 0.05,
) -> CSTPredictionInterval:
    """Compute CST-adjusted prediction interval.

    Parameters
    ----------
    mu : float
        Predicted mean.
    sigma : float
        Predicted standard deviation.
    epsilon : float
        Contamination probability.
    nu_normal : float
        Normal component degrees of freedom.
    nu_crisis : float
        Crisis component degrees of freedom.
    alpha : float
        Significance level (default 0.05 for 95% PI).

    Returns
    -------
    CSTPredictionInterval
        Prediction interval with width comparison to pure Student-t.
    """
    sigma_safe = max(abs(sigma), 1e-12)

    q_lo = _cst_quantile_em(alpha / 2.0, mu, sigma_safe,
                             epsilon, nu_normal, nu_crisis)
    q_hi = _cst_quantile_em(1.0 - alpha / 2.0, mu, sigma_safe,
                             epsilon, nu_normal, nu_crisis)

    width = q_hi - q_lo

    # Pure Student-t comparison (normal component)
    z_lo_pure = float(stats.t.ppf(alpha / 2.0, df=nu_normal))
    z_hi_pure = float(stats.t.ppf(1.0 - alpha / 2.0, df=nu_normal))
    width_pure = (z_hi_pure - z_lo_pure) * sigma_safe

    width_ratio = width / max(width_pure, 1e-12)

    return CSTPredictionInterval(
        q_lo=q_lo,
        q_hi=q_hi,
        width=width,
        width_pure_t=width_pure,
        width_ratio=width_ratio,
    )


def cst_prediction_interval_array(
    mu: np.ndarray,
    sigma: np.ndarray,
    epsilon: float,
    nu_normal: float,
    nu_crisis: float,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized prediction intervals.

    Returns
    -------
    q_lo : array
        Lower bounds.
    q_hi : array
        Upper bounds.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = len(mu)

    q_lo = np.empty(n)
    q_hi = np.empty(n)

    for i in range(n):
        result = cst_prediction_interval(
            float(mu[i]), float(sigma[i]),
            epsilon, nu_normal, nu_crisis, alpha,
        )
        q_lo[i] = result.q_lo
        q_hi[i] = result.q_hi

    return q_lo, q_hi
