"""
Epic 27: Forecast Performance Attribution

Decomposes forecast error into diagnostic sources:
1. Drift attribution: direction errors vs magnitude errors
2. Volatility attribution: coverage analysis of prediction intervals
3. BMA attribution: leave-one-model-out CRPS contribution

References:
- Gneiting & Raftery (2007): Strictly proper scoring rules
- Hersbach (2000): Decomposition of the CRPS
- Madigan & Raftery (1994): Model expansion and BMA
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Story 27.1: Drift attribution
DIRECTION_FAILURE_THRESHOLD = 0.45   # >45% wrong-sign = worse than coin flip
MAGNITUDE_FAILURE_MULTIPLIER = 2.0   # MAE > 2x median = overestimating moves
MIN_OBSERVATIONS = 20                # Minimum for attribution

# Story 27.2: Volatility attribution
DEFAULT_ALPHA = 0.10                 # 90% prediction interval
COVERAGE_UNDER_THRESHOLD = 0.85      # <85% at 90% PI = vol underestimate
COVERAGE_OVER_THRESHOLD = 0.95       # >95% at 90% PI = vol overestimate
ROLLING_COVERAGE_WINDOW = 60         # 60-day rolling window

# Story 27.3: BMA attribution
CRPS_IMPROVEMENT_THRESHOLD = 0.001   # Min CRPS improvement for model removal


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DriftAttributionResult:
    """Result of drift error attribution."""
    direction_error: float        # Fraction of wrong-sign predictions
    magnitude_error: float        # MAE of signed forecast when direction correct
    direction_flag: str           # "" or "DIRECTIONAL_FAILURE"
    magnitude_flag: str           # "" or "MAGNITUDE_FAILURE"
    correct_direction_count: int  # Number of correct-direction predictions
    wrong_direction_count: int    # Number of wrong-direction predictions
    zero_return_count: int        # Number of zero returns (excluded)
    median_magnitude_error: float # Median of |r - mu| when direction correct
    n_observations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction_error": float(self.direction_error),
            "magnitude_error": float(self.magnitude_error),
            "direction_flag": self.direction_flag,
            "magnitude_flag": self.magnitude_flag,
            "correct_direction_count": self.correct_direction_count,
            "wrong_direction_count": self.wrong_direction_count,
            "zero_return_count": self.zero_return_count,
            "median_magnitude_error": float(self.median_magnitude_error),
            "n_observations": self.n_observations,
        }


@dataclass
class VolatilityAttributionResult:
    """Result of volatility attribution via coverage analysis."""
    coverage: float               # Fraction of returns within PI
    alpha: float                  # PI level (e.g. 0.10 for 90%)
    volatility_flag: str          # "" or "VOL_UNDERESTIMATE" or "VOL_OVERESTIMATE"
    crps_reliability: float       # Reliability component of CRPS
    crps_sharpness: float         # Sharpness component of CRPS
    rolling_coverage: np.ndarray  # Rolling coverage over windows
    rolling_coverage_mean: float  # Mean rolling coverage
    rolling_coverage_std: float   # Std of rolling coverage
    n_observations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage": float(self.coverage),
            "alpha": float(self.alpha),
            "volatility_flag": self.volatility_flag,
            "crps_reliability": float(self.crps_reliability),
            "crps_sharpness": float(self.crps_sharpness),
            "rolling_coverage_mean": float(self.rolling_coverage_mean),
            "rolling_coverage_std": float(self.rolling_coverage_std),
            "n_observations": self.n_observations,
        }


@dataclass
class BMAAttributionResult:
    """Result of BMA leave-one-model-out attribution."""
    model_contributions: Dict[str, float]  # model_name -> CRPS contribution
    combined_crps: float                    # Full ensemble CRPS
    best_removal_model: str                 # Model whose removal improves CRPS most
    best_removal_improvement: float         # CRPS improvement from removing it
    harmful_models: List[str]               # Models that worsen combined forecast
    beneficial_models: List[str]            # Models that improve combined forecast
    n_models: int
    n_observations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_contributions": {k: float(v) for k, v in self.model_contributions.items()},
            "combined_crps": float(self.combined_crps),
            "best_removal_model": self.best_removal_model,
            "best_removal_improvement": float(self.best_removal_improvement),
            "harmful_models": self.harmful_models,
            "beneficial_models": self.beneficial_models,
            "n_models": self.n_models,
            "n_observations": self.n_observations,
        }


# ---------------------------------------------------------------------------
# Story 27.1: Drift Attribution via State Error Decomposition
# ---------------------------------------------------------------------------

def drift_attribution(
    returns: np.ndarray,
    mu_forecast: np.ndarray,
    sigma_forecast: np.ndarray,
) -> DriftAttributionResult:
    """
    Decompose forecast error into direction and magnitude components.

    Direction error: fraction of wrong-sign predictions.
    Magnitude error: MAE of signed forecast when direction is correct.

    Parameters
    ----------
    returns : array-like, shape (T,)
        Realized returns.
    mu_forecast : array-like, shape (T,)
        Predicted mean (drift estimate).
    sigma_forecast : array-like, shape (T,)
        Predicted standard deviation (for normalization context).

    Returns
    -------
    DriftAttributionResult
        Direction and magnitude error decomposition.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    mu = np.asarray(mu_forecast, dtype=np.float64).ravel()
    sigma = np.asarray(sigma_forecast, dtype=np.float64).ravel()

    if len(r) != len(mu) or len(r) != len(sigma):
        raise ValueError(
            f"Length mismatch: returns={len(r)}, mu={len(mu)}, sigma={len(sigma)}"
        )

    # Filter valid observations
    valid = np.isfinite(r) & np.isfinite(mu) & np.isfinite(sigma)
    r = r[valid]
    mu = mu[valid]
    sigma = sigma[valid]

    if len(r) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Need >= {MIN_OBSERVATIONS} valid observations, got {len(r)}"
        )

    # Identify non-zero returns (zero returns are ambiguous for direction)
    nonzero_mask = r != 0.0
    zero_count = int(np.sum(~nonzero_mask))

    r_nz = r[nonzero_mask]
    mu_nz = mu[nonzero_mask]

    # Direction analysis: sign agreement
    if len(r_nz) == 0:
        return DriftAttributionResult(
            direction_error=0.5,
            magnitude_error=0.0,
            direction_flag="DIRECTIONAL_FAILURE",
            magnitude_flag="",
            correct_direction_count=0,
            wrong_direction_count=0,
            zero_return_count=zero_count,
            median_magnitude_error=0.0,
            n_observations=len(r),
        )

    sign_r = np.sign(r_nz)
    sign_mu = np.sign(mu_nz)

    # Correct direction: signs match (or mu == 0 counts as wrong)
    correct_mask = sign_r == sign_mu
    correct_count = int(np.sum(correct_mask))
    wrong_count = int(np.sum(~correct_mask))

    direction_error = float(wrong_count / len(r_nz))

    # Magnitude error: MAE when direction is correct
    if correct_count > 0:
        mag_errors = np.abs(r_nz[correct_mask] - mu_nz[correct_mask])
        magnitude_error = float(np.mean(mag_errors))
        median_mag_error = float(np.median(mag_errors))
    else:
        magnitude_error = float(np.mean(np.abs(r_nz - mu_nz)))
        median_mag_error = magnitude_error

    # Flags
    direction_flag = "DIRECTIONAL_FAILURE" if direction_error > DIRECTION_FAILURE_THRESHOLD else ""

    magnitude_flag = ""
    if median_mag_error > 0:
        if magnitude_error > MAGNITUDE_FAILURE_MULTIPLIER * median_mag_error:
            magnitude_flag = "MAGNITUDE_FAILURE"

    return DriftAttributionResult(
        direction_error=direction_error,
        magnitude_error=magnitude_error,
        direction_flag=direction_flag,
        magnitude_flag=magnitude_flag,
        correct_direction_count=correct_count,
        wrong_direction_count=wrong_count,
        zero_return_count=zero_count,
        median_magnitude_error=median_mag_error,
        n_observations=len(r),
    )


# ---------------------------------------------------------------------------
# Story 27.2: Volatility Attribution via Coverage Analysis
# ---------------------------------------------------------------------------

def _gaussian_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    CRPS for Gaussian predictive distribution.

    CRPS(N(mu, sigma^2), y) = sigma * [z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
    where z = (y - mu) / sigma, Phi = CDF, phi = PDF of standard normal.
    """
    from scipy.stats import norm

    sigma_safe = np.maximum(sigma, 1e-12)
    z = (y - mu) / sigma_safe

    crps = sigma_safe * (
        z * (2 * norm.cdf(z) - 1)
        + 2 * norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return crps


def volatility_attribution(
    returns: np.ndarray,
    mu_forecast: np.ndarray,
    sigma_forecast: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    rolling_window: int = ROLLING_COVERAGE_WINDOW,
) -> VolatilityAttributionResult:
    """
    Volatility attribution via prediction interval coverage analysis.

    Checks whether prediction intervals are too wide or too narrow,
    and decomposes CRPS into reliability and sharpness components.

    Parameters
    ----------
    returns : array-like, shape (T,)
        Realized returns.
    mu_forecast : array-like, shape (T,)
        Predicted mean.
    sigma_forecast : array-like, shape (T,)
        Predicted standard deviation.
    alpha : float
        Significance level for prediction interval (default 0.10 = 90% PI).
    rolling_window : int
        Window size for rolling coverage.

    Returns
    -------
    VolatilityAttributionResult
        Coverage metrics and CRPS decomposition.
    """
    from scipy.stats import norm

    r = np.asarray(returns, dtype=np.float64).ravel()
    mu = np.asarray(mu_forecast, dtype=np.float64).ravel()
    sigma = np.asarray(sigma_forecast, dtype=np.float64).ravel()

    if len(r) != len(mu) or len(r) != len(sigma):
        raise ValueError(
            f"Length mismatch: returns={len(r)}, mu={len(mu)}, sigma={len(sigma)}"
        )

    valid = np.isfinite(r) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
    r = r[valid]
    mu = mu[valid]
    sigma = sigma[valid]

    if len(r) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Need >= {MIN_OBSERVATIONS} valid observations, got {len(r)}"
        )

    # Prediction interval bounds
    z_crit = norm.ppf(1 - alpha / 2)
    lower = mu - z_crit * sigma
    upper = mu + z_crit * sigma

    # Coverage: fraction of returns within PI
    in_interval = (r >= lower) & (r <= upper)
    coverage = float(np.mean(in_interval))

    # Volatility flag
    expected_coverage = 1.0 - alpha
    if coverage < COVERAGE_UNDER_THRESHOLD:
        volatility_flag = "VOL_UNDERESTIMATE"
    elif coverage > COVERAGE_OVER_THRESHOLD:
        volatility_flag = "VOL_OVERESTIMATE"
    else:
        volatility_flag = ""

    # CRPS decomposition: reliability + sharpness
    crps_values = _gaussian_crps(mu, sigma, r)
    total_crps = float(np.mean(crps_values))

    # Reliability: how well-calibrated the probabilistic forecast is
    # Computed via PIT histogram approach
    pit = norm.cdf(r, loc=mu, scale=sigma)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        obs_frac = np.mean((pit >= lo) & (pit < hi))
        expected_frac = 1.0 / n_bins
        reliability += (obs_frac - expected_frac) ** 2
    reliability = float(reliability / n_bins)

    # Sharpness: average width of prediction intervals (smaller = sharper)
    sharpness = float(np.mean(2 * z_crit * sigma))

    # Rolling coverage
    n = len(r)
    if n >= rolling_window:
        rolling_cov = np.array([
            np.mean(in_interval[i:i + rolling_window])
            for i in range(n - rolling_window + 1)
        ])
    else:
        rolling_cov = np.array([coverage])

    return VolatilityAttributionResult(
        coverage=coverage,
        alpha=alpha,
        volatility_flag=volatility_flag,
        crps_reliability=reliability,
        crps_sharpness=sharpness,
        rolling_coverage=rolling_cov,
        rolling_coverage_mean=float(np.mean(rolling_cov)),
        rolling_coverage_std=float(np.std(rolling_cov)),
        n_observations=len(r),
    )


# ---------------------------------------------------------------------------
# Story 27.3: BMA Weight Attribution via Leave-One-Model-Out
# ---------------------------------------------------------------------------

def _compute_ensemble_crps(
    model_forecasts: Dict[str, Tuple[np.ndarray, np.ndarray]],
    weights: Dict[str, float],
    returns: np.ndarray,
    exclude_model: Optional[str] = None,
) -> float:
    """
    Compute CRPS of weighted Gaussian mixture ensemble.

    For a Gaussian mixture, the combined predictive:
    mu_combined = sum(w_i * mu_i), sigma_combined = sqrt(sum(w_i * (sigma_i^2 + mu_i^2)) - mu_combined^2)

    Parameters
    ----------
    model_forecasts : dict
        {model_name: (mu_array, sigma_array)}
    weights : dict
        {model_name: weight}
    returns : array
        Realized returns.
    exclude_model : str, optional
        Model to exclude (for leave-one-out).

    Returns
    -------
    float
        Mean CRPS of the ensemble.
    """
    # Filter models
    active_models = {k: v for k, v in model_forecasts.items() if k != exclude_model}
    active_weights = {k: v for k, v in weights.items() if k != exclude_model and k in active_models}

    if not active_weights:
        return float('inf')

    # Renormalize weights
    w_sum = sum(active_weights.values())
    if w_sum <= 0:
        return float('inf')
    norm_weights = {k: v / w_sum for k, v in active_weights.items()}

    n = len(returns)

    # Compute combined forecast (moment matching)
    mu_combined = np.zeros(n)
    var_combined = np.zeros(n)

    for model_name, w in norm_weights.items():
        mu_i, sigma_i = active_models[model_name]
        mu_combined += w * mu_i
        var_combined += w * (sigma_i ** 2 + mu_i ** 2)

    var_combined -= mu_combined ** 2
    var_combined = np.maximum(var_combined, 1e-16)
    sigma_combined = np.sqrt(var_combined)

    # CRPS
    crps_values = _gaussian_crps(mu_combined, sigma_combined, returns)
    return float(np.mean(crps_values))


def bma_attribution(
    model_forecasts: Dict[str, Tuple[np.ndarray, np.ndarray]],
    weights: Dict[str, float],
    returns: np.ndarray,
) -> BMAAttributionResult:
    """
    Leave-one-model-out BMA attribution.

    For each model, computes CRPS with and without it.
    Positive contribution = model improves ensemble (keep).
    Negative contribution = model worsens ensemble (investigate).

    Parameters
    ----------
    model_forecasts : dict
        {model_name: (mu_array, sigma_array)} where each is shape (T,).
    weights : dict
        {model_name: float} BMA weights (should sum to ~1.0).
    returns : array-like, shape (T,)
        Realized returns.

    Returns
    -------
    BMAAttributionResult
        Per-model CRPS contribution and diagnostics.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()

    # Validate inputs
    model_names = sorted(model_forecasts.keys())
    if len(model_names) < 2:
        raise ValueError("Need >= 2 models for attribution")

    for name in model_names:
        mu_i, sigma_i = model_forecasts[name]
        mu_i = np.asarray(mu_i, dtype=np.float64).ravel()
        sigma_i = np.asarray(sigma_i, dtype=np.float64).ravel()
        if len(mu_i) != len(r) or len(sigma_i) != len(r):
            raise ValueError(
                f"Model '{name}' forecast length {len(mu_i)} != returns length {len(r)}"
            )
        model_forecasts[name] = (mu_i, sigma_i)

    if len(r) < MIN_OBSERVATIONS:
        raise ValueError(
            f"Need >= {MIN_OBSERVATIONS} valid observations, got {len(r)}"
        )

    # Ensure weights exist for all models
    for name in model_names:
        if name not in weights:
            weights[name] = 0.0

    # Full ensemble CRPS
    combined_crps = _compute_ensemble_crps(model_forecasts, weights, r)

    # Leave-one-out CRPS for each model
    contributions = {}
    for name in model_names:
        loo_crps = _compute_ensemble_crps(model_forecasts, weights, r, exclude_model=name)
        # Contribution = CRPS_without - CRPS_with
        # Positive = removing model worsens CRPS (model is beneficial)
        # Negative = removing model improves CRPS (model is harmful)
        contributions[name] = float(loo_crps - combined_crps)

    # Identify beneficial and harmful models
    beneficial = [name for name, c in contributions.items() if c > 0]
    harmful = [name for name, c in contributions.items() if c < 0]

    # Best removal: model whose removal improves CRPS the most
    if harmful:
        best_removal = min(contributions, key=contributions.get)
        best_improvement = -contributions[best_removal]  # positive value
    else:
        best_removal = min(contributions, key=contributions.get)
        best_improvement = max(0.0, -contributions[best_removal])

    return BMAAttributionResult(
        model_contributions=contributions,
        combined_crps=combined_crps,
        best_removal_model=best_removal,
        best_removal_improvement=best_improvement,
        harmful_models=harmful,
        beneficial_models=beneficial,
        n_models=len(model_names),
        n_observations=len(r),
    )
