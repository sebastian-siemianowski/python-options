#!/usr/bin/env python3
"""
===============================================================================
FACTOR-AUGMENTED KALMAN FILTER -- Cross-Asset Intelligence
===============================================================================

Epic 22: Factor-Augmented Kalman Filter

Individual assets don't live in isolation. Returns share common factors
(market, size, value, momentum). This module provides:

1. MARKET FACTOR EXTRACTION via PCA on innovations cross-section
   r_t^(i) = beta_i' F_t + alpha_t^(i) + eps_t^(i)
   
   Common factor estimated from many assets (low noise), idiosyncratic
   component has lower variance (easier to estimate).

2. FACTOR-ADJUSTED OBSERVATION NOISE
   R_t^adj = c * sigma_t^2 * (1 - R^2_factor)
   
   Separates systematic from idiosyncratic risk. High-beta stocks get
   larger R^2 reduction; uncorrelated assets are barely affected.

3. GRANGER CAUSALITY for lead-lag signal propagation
   Tests whether lagged leader returns improve follower forecasts.
   Incorporates leader signal: mu_t^follower += gamma * r_{t-k}^leader
   gamma estimated via rolling OLS with ridge regularization.

REFERENCES:
   Stock, J. & Watson, M. (2002). "Forecasting Using Principal Components"
   Granger, C. (1969). "Investigating Causal Relations by Econometric Models"
   Bai, J. & Ng, S. (2002). "Determining the Number of Factors"

===============================================================================
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# PCA / Factor extraction
DEFAULT_N_FACTORS = 3
MIN_ASSETS_FOR_PCA = 5
MIN_OBS_FOR_PCA = 60
LOADING_STABILITY_THRESHOLD = 0.90  # Minimum correlation for "stable" loadings

# Factor-adjusted R
FACTOR_R2_FLOOR = 0.0     # Minimum R^2 (no negative)
FACTOR_R2_CAP = 0.95      # Maximum R^2 (prevent zero R)
R_ADJUSTMENT_MIN = 0.05   # Never reduce R by more than 95%

# Granger causality
GRANGER_MAX_LAG_DEFAULT = 5
GRANGER_MIN_OBS = 30
GRANGER_SIGNIFICANCE = 0.05

# Ridge regression for gamma estimation
RIDGE_LAMBDA_DEFAULT = 0.01
ROLLING_OLS_WINDOW = 252  # 1 year rolling window
ROLLING_OLS_MIN_OBS = 60


# =============================================================================
# DATA CLASSES -- Story 22.1
# =============================================================================

@dataclass
class FactorExtractionResult:
    """Result of PCA-based market factor extraction."""
    # Core outputs
    loadings: np.ndarray        # (n_assets, n_factors) -- factor loadings
    scores: np.ndarray          # (T, n_factors) -- factor scores (time series)
    explained_variance_ratio: np.ndarray  # (n_factors,) -- variance explained per factor
    cumulative_variance: float  # Total variance explained by all factors
    
    # Diagnostics
    n_factors: int
    n_assets: int
    n_obs: int
    eigenvalues: np.ndarray     # All eigenvalues (for scree plot)
    mean_returns: np.ndarray    # (n_assets,) -- mean used for centering
    std_returns: np.ndarray     # (n_assets,) -- std used for standardization
    
    def factor1_variance_share(self) -> float:
        """Fraction of variance explained by first factor (market factor)."""
        if len(self.explained_variance_ratio) == 0:
            return 0.0
        return float(self.explained_variance_ratio[0])
    
    def factor2_variance_share(self) -> float:
        """Fraction of variance explained by second factor."""
        if len(self.explained_variance_ratio) < 2:
            return 0.0
        return float(self.explained_variance_ratio[1])
    
    def to_dict(self) -> Dict:
        return {
            "n_factors": self.n_factors,
            "n_assets": self.n_assets,
            "n_obs": self.n_obs,
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "cumulative_variance": self.cumulative_variance,
            "factor1_share": self.factor1_variance_share(),
            "factor2_share": self.factor2_variance_share(),
        }


@dataclass
class LoadingStabilityResult:
    """Result of factor loading stability check."""
    correlations: np.ndarray       # (n_factors,) -- correlation between consecutive estimates
    is_stable: np.ndarray          # (n_factors,) -- boolean, True if > threshold
    overall_stable: bool           # True if all factors are stable
    n_windows: int                 # Number of rolling windows compared
    threshold: float               # Stability threshold used


# =============================================================================
# DATA CLASSES -- Story 22.2
# =============================================================================

@dataclass
class FactorAdjustedRResult:
    """Result of factor-adjusted observation noise computation."""
    R_adjusted: np.ndarray    # (T,) -- adjusted observation noise
    R_original: np.ndarray    # (T,) -- original R = c * sigma_t^2
    factor_R2: float          # R^2 from factor regression
    reduction_ratio: float    # Average R_adj / R_orig
    
    def to_dict(self) -> Dict:
        return {
            "factor_R2": self.factor_R2,
            "reduction_ratio": self.reduction_ratio,
            "mean_R_adjusted": float(np.mean(self.R_adjusted)),
            "mean_R_original": float(np.mean(self.R_original)),
        }


@dataclass
class AssetFactorR2Result:
    """R^2 of an asset's returns explained by common factors."""
    r_squared: float
    beta_loadings: np.ndarray   # (n_factors,) -- regression coefficients
    residual_std: float         # Std of idiosyncratic component
    f_statistic: float
    f_pvalue: float
    n_obs: int


# =============================================================================
# DATA CLASSES -- Story 22.3
# =============================================================================

@dataclass
class GrangerTestResult:
    """Result of Granger causality test."""
    # Core test results
    p_value: float              # P-value of F-test (null: no Granger causality)
    optimal_lag: int            # Lag with strongest signal
    f_statistic: float          # F-statistic at optimal lag
    is_significant: bool        # p_value < significance level
    
    # Per-lag results
    lag_pvalues: np.ndarray     # (max_lag,) -- p-value at each lag
    lag_fstats: np.ndarray      # (max_lag,) -- F-stat at each lag
    
    # Diagnostics
    n_obs: int
    max_lag_tested: int
    
    def to_dict(self) -> Dict:
        return {
            "p_value": self.p_value,
            "optimal_lag": self.optimal_lag,
            "f_statistic": self.f_statistic,
            "is_significant": self.is_significant,
            "n_obs": self.n_obs,
            "max_lag_tested": self.max_lag_tested,
        }


@dataclass
class LeaderFollowerResult:
    """Result of leader-follower signal propagation."""
    gamma: float                # Signal propagation coefficient
    gamma_se: float             # Standard error of gamma
    optimal_lag: int            # Lag at which leader predicts follower
    granger_result: GrangerTestResult  # Underlying Granger test
    
    # Rolling gamma estimates
    gamma_series: Optional[np.ndarray] = None  # (T,) -- time-varying gamma
    
    # Impact metrics
    hit_rate_improvement: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "gamma": self.gamma,
            "gamma_se": self.gamma_se,
            "optimal_lag": self.optimal_lag,
            "granger_significant": self.granger_result.is_significant,
            "granger_pvalue": self.granger_result.p_value,
            "hit_rate_improvement": self.hit_rate_improvement,
        }


# =============================================================================
# STORY 22.1: Market Factor Extraction via PCA
# =============================================================================

def extract_market_factors(
    innovations_matrix: np.ndarray,
    n_factors: int = DEFAULT_N_FACTORS,
    standardize: bool = True,
) -> FactorExtractionResult:
    """
    Extract common market factors via PCA on innovations cross-section.
    
    The first factor typically captures the "market factor" (>30% variance),
    the second captures size/sector effects (>10% variance).
    
    Args:
        innovations_matrix: (T, N) matrix of Kalman innovations or returns
                           T = time steps, N = assets
        n_factors: Number of factors to extract (default: 3)
        standardize: Whether to standardize columns before PCA
        
    Returns:
        FactorExtractionResult with loadings, scores, and diagnostics
        
    Raises:
        ValueError: If matrix dimensions are insufficient
    """
    innovations_matrix = np.asarray(innovations_matrix, dtype=np.float64)
    
    if innovations_matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {innovations_matrix.ndim}D")
    
    T, N = innovations_matrix.shape
    
    if N < MIN_ASSETS_FOR_PCA:
        raise ValueError(
            f"Need at least {MIN_ASSETS_FOR_PCA} assets for PCA, got {N}"
        )
    if T < MIN_OBS_FOR_PCA:
        raise ValueError(
            f"Need at least {MIN_OBS_FOR_PCA} observations for PCA, got {T}"
        )
    
    n_factors = min(n_factors, N, T)
    
    # Handle NaNs: replace with column mean
    col_means = np.nanmean(innovations_matrix, axis=0)
    for j in range(N):
        mask = np.isnan(innovations_matrix[:, j])
        if np.any(mask):
            innovations_matrix[mask, j] = col_means[j]
    
    # Center (and optionally standardize)
    mean_returns = np.mean(innovations_matrix, axis=0)
    centered = innovations_matrix - mean_returns
    
    if standardize:
        std_returns = np.std(innovations_matrix, axis=0, ddof=1)
        # Protect against zero-variance columns
        std_returns = np.where(std_returns < 1e-12, 1.0, std_returns)
        centered = centered / std_returns
    else:
        std_returns = np.ones(N)
    
    # Compute covariance matrix
    cov_matrix = np.dot(centered.T, centered) / (T - 1)
    
    # Eigen decomposition (symmetric matrix -> use eigh for stability)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Clip negative eigenvalues (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    
    # Extract top factors
    loadings = eigenvectors[:, :n_factors]  # (N, n_factors)
    
    # Compute scores = centered_data @ loadings
    scores = centered @ loadings  # (T, n_factors)
    
    # Explained variance ratios
    total_variance = np.sum(eigenvalues)
    if total_variance < 1e-12:
        explained_variance_ratio = np.zeros(n_factors)
    else:
        explained_variance_ratio = eigenvalues[:n_factors] / total_variance
    
    cumulative_variance = float(np.sum(explained_variance_ratio))
    
    return FactorExtractionResult(
        loadings=loadings,
        scores=scores,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_variance=cumulative_variance,
        n_factors=n_factors,
        n_assets=N,
        n_obs=T,
        eigenvalues=eigenvalues,
        mean_returns=mean_returns,
        std_returns=std_returns,
    )


def rolling_factor_extraction(
    innovations_matrix: np.ndarray,
    window_size: int = 252,
    step_size: int = 21,
    n_factors: int = DEFAULT_N_FACTORS,
) -> List[FactorExtractionResult]:
    """
    Extract factors on rolling windows for stability analysis.
    
    Args:
        innovations_matrix: (T, N) matrix
        window_size: Rolling window size (default: 252 = 1 year)
        step_size: Step between windows (default: 21 = 1 month)
        n_factors: Number of factors
        
    Returns:
        List of FactorExtractionResult, one per window
    """
    innovations_matrix = np.asarray(innovations_matrix, dtype=np.float64)
    T, N = innovations_matrix.shape
    
    results = []
    start = 0
    while start + window_size <= T:
        window = innovations_matrix[start:start + window_size]
        try:
            result = extract_market_factors(window, n_factors=n_factors)
            results.append(result)
        except ValueError:
            pass
        start += step_size
    
    return results


def check_loading_stability(
    rolling_results: List[FactorExtractionResult],
    threshold: float = LOADING_STABILITY_THRESHOLD,
) -> LoadingStabilityResult:
    """
    Check stability of factor loadings across consecutive rolling windows.
    
    Loadings are considered stable if correlation between consecutive
    monthly estimates exceeds the threshold (default: 0.90).
    
    Note: Eigenvectors can flip sign between windows. We compare
    |correlation| to handle sign ambiguity.
    
    Args:
        rolling_results: List of FactorExtractionResult from rolling extraction
        threshold: Minimum correlation for stability
        
    Returns:
        LoadingStabilityResult with per-factor stability measures
    """
    if len(rolling_results) < 2:
        n_factors = rolling_results[0].n_factors if rolling_results else 0
        return LoadingStabilityResult(
            correlations=np.ones(n_factors),
            is_stable=np.ones(n_factors, dtype=bool),
            overall_stable=True,
            n_windows=len(rolling_results),
            threshold=threshold,
        )
    
    n_factors = rolling_results[0].n_factors
    all_correlations = []
    
    for i in range(1, len(rolling_results)):
        prev_loadings = rolling_results[i - 1].loadings
        curr_loadings = rolling_results[i].loadings
        
        # Ensure same shape
        n_f = min(prev_loadings.shape[1], curr_loadings.shape[1], n_factors)
        
        corrs = np.zeros(n_f)
        for f in range(n_f):
            # Use absolute correlation to handle sign flips
            r = np.corrcoef(prev_loadings[:, f], curr_loadings[:, f])[0, 1]
            corrs[f] = abs(r) if np.isfinite(r) else 0.0
        
        all_correlations.append(corrs)
    
    # Average correlations across windows
    avg_correlations = np.mean(all_correlations, axis=0)
    is_stable = avg_correlations >= threshold
    
    return LoadingStabilityResult(
        correlations=avg_correlations,
        is_stable=is_stable,
        overall_stable=bool(np.all(is_stable)),
        n_windows=len(rolling_results),
        threshold=threshold,
    )


def compute_asset_factor_r2(
    asset_returns: np.ndarray,
    factor_scores: np.ndarray,
) -> AssetFactorR2Result:
    """
    Compute R^2 of a single asset's returns explained by common factors.
    
    Runs OLS: r_t^(i) = beta' @ F_t + epsilon_t
    
    Args:
        asset_returns: (T,) returns of one asset
        factor_scores: (T, n_factors) factor scores from PCA
        
    Returns:
        AssetFactorR2Result with R^2 and diagnostics
    """
    asset_returns = np.asarray(asset_returns, dtype=np.float64).ravel()
    factor_scores = np.asarray(factor_scores, dtype=np.float64)
    
    if factor_scores.ndim == 1:
        factor_scores = factor_scores.reshape(-1, 1)
    
    T = len(asset_returns)
    n_factors = factor_scores.shape[1]
    
    if T != factor_scores.shape[0]:
        raise ValueError(
            f"Length mismatch: asset_returns={T}, factor_scores={factor_scores.shape[0]}"
        )
    
    # Remove NaN rows
    valid = np.isfinite(asset_returns) & np.all(np.isfinite(factor_scores), axis=1)
    y = asset_returns[valid]
    X = factor_scores[valid]
    n = len(y)
    
    if n < n_factors + 2:
        return AssetFactorR2Result(
            r_squared=0.0,
            beta_loadings=np.zeros(n_factors),
            residual_std=float(np.std(y)) if len(y) > 0 else 0.0,
            f_statistic=0.0,
            f_pvalue=1.0,
            n_obs=n,
        )
    
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(n), X])
    
    # OLS via normal equations: beta = (X'X)^{-1} X'y
    try:
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return AssetFactorR2Result(
            r_squared=0.0,
            beta_loadings=np.zeros(n_factors),
            residual_std=float(np.std(y)),
            f_statistic=0.0,
            f_pvalue=1.0,
            n_obs=n,
        )
    
    # Fitted values and residuals
    y_hat = X_with_intercept @ beta
    residuals = y - y_hat
    
    # R^2
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    r_squared = np.clip(r_squared, 0.0, 1.0)
    
    # F-statistic
    df_model = n_factors
    df_resid = n - n_factors - 1
    if df_resid > 0 and ss_res > 1e-12:
        ss_model = ss_tot - ss_res
        ms_model = ss_model / df_model
        ms_resid = ss_res / df_resid
        f_stat = ms_model / ms_resid if ms_resid > 1e-12 else 0.0
        f_pvalue = 1.0 - sp_stats.f.cdf(f_stat, df_model, df_resid)
    else:
        f_stat = 0.0
        f_pvalue = 1.0
    
    residual_std = float(np.std(residuals, ddof=1)) if n > 1 else 0.0
    
    return AssetFactorR2Result(
        r_squared=float(r_squared),
        beta_loadings=beta[1:],  # Exclude intercept
        residual_std=residual_std,
        f_statistic=float(f_stat),
        f_pvalue=float(f_pvalue),
        n_obs=n,
    )


# =============================================================================
# STORY 22.2: Factor-Adjusted Innovation Variance
# =============================================================================

def factor_adjusted_R(
    sigma_t: np.ndarray,
    c: float,
    factor_R2: float,
) -> FactorAdjustedRResult:
    """
    Compute factor-adjusted observation noise for Kalman filter.
    
    Formula:
        R_t^adj = c * sigma_t^2 * (1 - R^2_factor)
    
    When the factor R^2 is high (e.g., 0.4 for NVDA), 40% of return
    variance is explained by common factors. The idiosyncratic observation
    noise is only 60% of the original.
    
    Args:
        sigma_t: (T,) volatility time series (EWMA or realized)
        c: Observation noise scaling constant from Kalman
        factor_R2: Fraction of return variance explained by factors [0, 1]
        
    Returns:
        FactorAdjustedRResult with adjusted R and diagnostics
        
    Raises:
        ValueError: If factor_R2 not in [0, 1] or c <= 0
    """
    sigma_t = np.asarray(sigma_t, dtype=np.float64).ravel()
    
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    if not (0.0 <= factor_R2 <= 1.0):
        raise ValueError(f"factor_R2 must be in [0, 1], got {factor_R2}")
    
    # Clip R^2 to prevent zero or negative R
    effective_R2 = np.clip(factor_R2, FACTOR_R2_FLOOR, FACTOR_R2_CAP)
    
    # Original observation noise
    R_original = c * sigma_t ** 2
    
    # Adjusted: remove factor-explained component
    adjustment = 1.0 - effective_R2
    adjustment = max(adjustment, R_ADJUSTMENT_MIN)
    R_adjusted = R_original * adjustment
    
    reduction_ratio = float(np.mean(R_adjusted / np.where(R_original > 1e-20, R_original, 1.0)))
    
    return FactorAdjustedRResult(
        R_adjusted=R_adjusted,
        R_original=R_original,
        factor_R2=factor_R2,
        reduction_ratio=reduction_ratio,
    )


# =============================================================================
# STORY 22.3: Cross-Asset Signal Propagation via Granger Causality
# =============================================================================

def granger_test(
    leader_returns: np.ndarray,
    follower_returns: np.ndarray,
    max_lag: int = GRANGER_MAX_LAG_DEFAULT,
    significance: float = GRANGER_SIGNIFICANCE,
) -> GrangerTestResult:
    """
    Test Granger causality: does leader predict follower beyond self-history?
    
    For each lag k = 1, ..., max_lag, compare:
        Restricted:   y_t = a_0 + sum_{j=1}^{k} a_j y_{t-j}
        Unrestricted: y_t = a_0 + sum_{j=1}^{k} a_j y_{t-j} + sum_{j=1}^{k} b_j x_{t-j}
    
    F-test on whether b_j = 0 for all j.
    
    IMPORTANT: Only LAGGED leader returns are used (no forward leakage).
    
    Args:
        leader_returns: (T,) returns of potential leader asset
        follower_returns: (T,) returns of potential follower asset
        max_lag: Maximum lag to test (default: 5)
        significance: P-value threshold for significance
        
    Returns:
        GrangerTestResult with p-values per lag and optimal lag
        
    Raises:
        ValueError: If arrays are too short or different lengths
    """
    leader = np.asarray(leader_returns, dtype=np.float64).ravel()
    follower = np.asarray(follower_returns, dtype=np.float64).ravel()
    
    if len(leader) != len(follower):
        raise ValueError(
            f"Leader and follower must have same length: {len(leader)} vs {len(follower)}"
        )
    
    T = len(follower)
    if T < max_lag + GRANGER_MIN_OBS:
        raise ValueError(
            f"Need at least {max_lag + GRANGER_MIN_OBS} observations, got {T}"
        )
    
    lag_pvalues = np.ones(max_lag)
    lag_fstats = np.zeros(max_lag)
    
    for k in range(1, max_lag + 1):
        # Build lagged matrices
        n_usable = T - k
        y = follower[k:]
        
        # Restricted model: follower lags only
        X_restricted = np.ones((n_usable, k + 1))  # intercept + k lags
        for j in range(1, k + 1):
            X_restricted[:, j] = follower[k - j:T - j]
        
        # Unrestricted model: follower lags + leader lags
        X_unrestricted = np.ones((n_usable, 2 * k + 1))  # intercept + k follower + k leader
        for j in range(1, k + 1):
            X_unrestricted[:, j] = follower[k - j:T - j]
            X_unrestricted[:, k + j] = leader[k - j:T - j]
        
        # Fit both models via OLS
        try:
            _, res_r, _, _ = np.linalg.lstsq(X_restricted, y, rcond=None)
            _, res_u, _, _ = np.linalg.lstsq(X_unrestricted, y, rcond=None)
            
            # If residuals not returned (perfect fit), compute manually
            beta_r = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_unrestricted, y, rcond=None)[0]
            
            ss_r = np.sum((y - X_restricted @ beta_r) ** 2)
            ss_u = np.sum((y - X_unrestricted @ beta_u) ** 2)
            
            # F-test
            df_diff = k  # Number of additional parameters
            df_resid = n_usable - 2 * k - 1
            
            if df_resid > 0 and ss_u > 1e-20:
                f_stat = ((ss_r - ss_u) / df_diff) / (ss_u / df_resid)
                f_stat = max(f_stat, 0.0)
                p_value = 1.0 - sp_stats.f.cdf(f_stat, df_diff, df_resid)
            else:
                f_stat = 0.0
                p_value = 1.0
                
        except (np.linalg.LinAlgError, ValueError):
            f_stat = 0.0
            p_value = 1.0
        
        lag_pvalues[k - 1] = p_value
        lag_fstats[k - 1] = f_stat
    
    # Optimal lag: lowest p-value
    optimal_idx = np.argmin(lag_pvalues)
    optimal_lag = optimal_idx + 1
    best_pvalue = float(lag_pvalues[optimal_idx])
    best_fstat = float(lag_fstats[optimal_idx])
    
    return GrangerTestResult(
        p_value=best_pvalue,
        optimal_lag=optimal_lag,
        f_statistic=best_fstat,
        is_significant=best_pvalue < significance,
        lag_pvalues=lag_pvalues,
        lag_fstats=lag_fstats,
        n_obs=T,
        max_lag_tested=max_lag,
    )


def estimate_leader_gamma(
    leader_returns: np.ndarray,
    follower_returns: np.ndarray,
    lag: int,
    ridge_lambda: float = RIDGE_LAMBDA_DEFAULT,
    rolling_window: Optional[int] = None,
) -> Tuple[float, float, Optional[np.ndarray]]:
    """
    Estimate signal propagation coefficient gamma via ridge regression.
    
    Model: follower_t = alpha + gamma * leader_{t-lag} + epsilon_t
    
    Ridge regularization prevents overfitting on small samples:
        gamma = (X'X + lambda*I)^{-1} X'y
    
    Args:
        leader_returns: (T,) leader asset returns
        follower_returns: (T,) follower asset returns
        lag: Lag from Granger test (only lagged values used)
        ridge_lambda: Ridge regularization strength
        rolling_window: If set, compute rolling gamma estimates
        
    Returns:
        Tuple of (gamma, gamma_se, gamma_series or None)
    """
    leader = np.asarray(leader_returns, dtype=np.float64).ravel()
    follower = np.asarray(follower_returns, dtype=np.float64).ravel()
    
    T = len(follower)
    if lag < 1:
        raise ValueError(f"Lag must be >= 1, got {lag}")
    if T <= lag:
        return 0.0, float('inf'), None
    
    # Align: follower[lag:] ~ leader[:T-lag]
    y = follower[lag:]
    X = np.column_stack([np.ones(T - lag), leader[:T - lag]])
    
    # Remove NaN/inf
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y_clean = y[valid]
    X_clean = X[valid]
    n = len(y_clean)
    
    if n < 3:
        return 0.0, float('inf'), None
    
    # Ridge regression: beta = (X'X + lambda*I)^{-1} X'y
    XtX = X_clean.T @ X_clean
    penalty = ridge_lambda * np.eye(XtX.shape[0])
    penalty[0, 0] = 0.0  # Don't penalize intercept
    
    try:
        beta = np.linalg.solve(XtX + penalty, X_clean.T @ y_clean)
    except np.linalg.LinAlgError:
        return 0.0, float('inf'), None
    
    gamma = float(beta[1])
    
    # Standard error estimate
    residuals = y_clean - X_clean @ beta
    sigma2 = np.sum(residuals ** 2) / max(n - 2, 1)
    try:
        cov_matrix = sigma2 * np.linalg.inv(XtX + penalty)
        gamma_se = float(np.sqrt(max(cov_matrix[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        gamma_se = float('inf')
    
    # Rolling estimates if requested
    gamma_series = None
    if rolling_window is not None and rolling_window >= ROLLING_OLS_MIN_OBS:
        gamma_series = _rolling_ridge_gamma(
            leader[:T - lag], y, rolling_window, ridge_lambda
        )
    
    return gamma, gamma_se, gamma_series


def _rolling_ridge_gamma(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
    ridge_lambda: float,
) -> np.ndarray:
    """
    Compute rolling ridge regression gamma estimates.
    
    Args:
        x: (T,) leader returns (already lagged)
        y: (T,) follower returns (aligned)
        window: Rolling window size
        ridge_lambda: Ridge penalty
        
    Returns:
        (T,) array of gamma estimates (NaN for insufficient data)
    """
    T = len(y)
    gamma_series = np.full(T, np.nan)
    
    for t in range(window, T):
        y_win = y[t - window:t]
        x_win = x[t - window:t]
        
        valid = np.isfinite(y_win) & np.isfinite(x_win)
        if np.sum(valid) < ROLLING_OLS_MIN_OBS:
            continue
        
        y_v = y_win[valid]
        X_v = np.column_stack([np.ones(np.sum(valid)), x_win[valid]])
        
        XtX = X_v.T @ X_v
        penalty = ridge_lambda * np.eye(2)
        penalty[0, 0] = 0.0
        
        try:
            beta = np.linalg.solve(XtX + penalty, X_v.T @ y_v)
            gamma_series[t] = beta[1]
        except np.linalg.LinAlgError:
            continue
    
    return gamma_series


def leader_follower_signal(
    leader_returns: np.ndarray,
    follower_returns: np.ndarray,
    max_lag: int = GRANGER_MAX_LAG_DEFAULT,
    ridge_lambda: float = RIDGE_LAMBDA_DEFAULT,
    rolling_window: Optional[int] = ROLLING_OLS_WINDOW,
) -> LeaderFollowerResult:
    """
    Full pipeline: Granger test + gamma estimation for leader-follower pair.
    
    Combines:
    1. Granger causality test to find optimal lag
    2. Ridge regression to estimate signal propagation gamma
    3. Optional rolling gamma for time-varying signal
    
    Usage for incorporating into Kalman:
        mu_t^follower += gamma * r_{t-k}^leader
    
    Args:
        leader_returns: (T,) leader asset returns (e.g., BTC)
        follower_returns: (T,) follower asset returns (e.g., MSTR)
        max_lag: Maximum lag for Granger test
        ridge_lambda: Ridge regularization for gamma
        rolling_window: Window for rolling gamma (None to skip)
        
    Returns:
        LeaderFollowerResult with gamma, lag, and Granger diagnostics
    """
    # Step 1: Granger causality test
    granger_result = granger_test(leader_returns, follower_returns, max_lag)
    
    # Step 2: Estimate gamma at optimal lag
    gamma, gamma_se, gamma_series = estimate_leader_gamma(
        leader_returns,
        follower_returns,
        lag=granger_result.optimal_lag,
        ridge_lambda=ridge_lambda,
        rolling_window=rolling_window,
    )
    
    # Step 3: Compute hit rate improvement
    hit_improvement = _compute_hit_rate_improvement(
        leader_returns,
        follower_returns,
        lag=granger_result.optimal_lag,
        gamma=gamma,
    )
    
    return LeaderFollowerResult(
        gamma=gamma,
        gamma_se=gamma_se,
        optimal_lag=granger_result.optimal_lag,
        granger_result=granger_result,
        gamma_series=gamma_series,
        hit_rate_improvement=hit_improvement,
    )


def _compute_hit_rate_improvement(
    leader: np.ndarray,
    follower: np.ndarray,
    lag: int,
    gamma: float,
) -> float:
    """
    Compute hit rate improvement from leader signal on follower direction.
    
    Compares:
    - Baseline: predict sign(follower_t) = sign(mean(follower))
    - With leader: predict sign(follower_t) based on gamma * leader_{t-lag}
    
    Returns improvement as fraction (e.g., 0.03 = 3% improvement).
    """
    T = len(follower)
    if T <= lag or lag < 1:
        return 0.0
    
    follower_aligned = follower[lag:]
    leader_lagged = leader[:T - lag]
    
    valid = np.isfinite(follower_aligned) & np.isfinite(leader_lagged)
    if np.sum(valid) < 20:
        return 0.0
    
    f = follower_aligned[valid]
    l = leader_lagged[valid]
    
    # Baseline: predict positive (majority class)
    baseline_hit = max(np.mean(f > 0), np.mean(f <= 0))
    
    # Leader-augmented: predict sign(gamma * leader)
    predictions = gamma * l
    leader_hit = np.mean(np.sign(predictions) == np.sign(f))
    
    return float(leader_hit - baseline_hit)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    "FactorExtractionResult",
    "LoadingStabilityResult",
    "FactorAdjustedRResult",
    "AssetFactorR2Result",
    "GrangerTestResult",
    "LeaderFollowerResult",
    # Story 22.1: Factor extraction
    "extract_market_factors",
    "rolling_factor_extraction",
    "check_loading_stability",
    "compute_asset_factor_r2",
    # Story 22.2: Factor-adjusted R
    "factor_adjusted_R",
    # Story 22.3: Granger causality
    "granger_test",
    "estimate_leader_gamma",
    "leader_follower_signal",
    # Constants
    "DEFAULT_N_FACTORS",
    "MIN_ASSETS_FOR_PCA",
    "LOADING_STABILITY_THRESHOLD",
    "GRANGER_MAX_LAG_DEFAULT",
    "GRANGER_SIGNIFICANCE",
    "RIDGE_LAMBDA_DEFAULT",
    "ROLLING_OLS_WINDOW",
]
