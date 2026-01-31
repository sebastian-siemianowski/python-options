#!/usr/bin/env python3
"""
ewma_covariance.py

Exponentially Weighted Moving Average (EWMA) covariance matrix estimation.
Implements RiskMetrics methodology with λ ≈ 0.94 for daily data.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def compute_ewma_covariance(
    returns: pd.DataFrame,
    lambda_decay: float = 0.94,
    min_periods: int = 100,
    apply_shrinkage: bool = True,
    shrinkage_target: str = "constant_correlation",
) -> Dict[str, pd.DataFrame]:
    """
    Compute EWMA covariance matrix for multi-asset returns with optional Ledoit-Wolf shrinkage.
    
    Model:
        Σ_t = λ * Σ_{t-1} + (1-λ) * r_t * r_t^T
    
    Where:
        - Σ_t: covariance matrix at time t
        - λ: decay parameter (0.94 = RiskMetrics standard for daily data)
        - r_t: vector of returns at time t
    
    Optional Ledoit-Wolf shrinkage (Priority 3):
        Σ̂ = δ·F + (1-δ)·Σ
    
    Where:
        - F: structured target (constant correlation, identity, or single-factor)
        - δ: optimal shrinkage intensity (data-driven)
    
    This gives:
        ✔ Rolling correlation that adapts to regime changes
        ✔ Regime-responsive covariance (reacts to volatility shifts)
        ✔ Stable estimates (exponential weighting reduces noise)
        ✔ Computationally efficient (recursive update)
        ✔ Improved out-of-sample robustness (via shrinkage)
        ✔ Better numerical stability (via shrinkage)
    
    Args:
        returns: DataFrame with returns (index=dates, columns=assets)
        lambda_decay: EWMA decay parameter (default 0.94)
        min_periods: Minimum observations required for stable estimate
        apply_shrinkage: If True, apply Ledoit-Wolf shrinkage to final covariance (default True)
        shrinkage_target: Target structure for shrinkage ('constant_correlation', 'identity', 'single_factor')
        
    Returns:
        Dictionary with:
            - covariance_matrix: Latest (n×n) covariance matrix (shrunk if apply_shrinkage=True)
            - correlation_matrix: Latest (n×n) correlation matrix
            - covariance_series: Time series of covariance matrices
            - volatilities: Time series of volatilities per asset
            - correlations: Time series of selected correlations
            - shrinkage_applied: Boolean indicating if shrinkage was applied
            - shrinkage_intensity: Optimal shrinkage intensity δ (if applied)
            - unshrunk_covariance_matrix: Original EWMA covariance before shrinkage (if applied)
    """
    # Input validation
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    
    if returns.empty or len(returns) < min_periods:
        raise ValueError(f"Insufficient data: {len(returns)} observations (need >={min_periods})")
    
    if not (0.0 < lambda_decay < 1.0):
        raise ValueError(f"lambda_decay must be in (0, 1), got {lambda_decay}")
    
    # Clean data
    returns_clean = returns.dropna()
    if len(returns_clean) < min_periods:
        raise ValueError(f"After dropping NaNs: {len(returns_clean)} observations (need >={min_periods})")
    
    # Extract values and dimensions
    R = returns_clean.values  # (T, n)
    T, n = R.shape
    asset_names = returns_clean.columns.tolist()
    dates = returns_clean.index
    
    # Initialize covariance with sample covariance over first min_periods
    if min_periods >= T:
        min_periods = max(50, T // 2)
    
    R_init = R[:min_periods, :]
    Sigma = np.cov(R_init, rowvar=False, ddof=1)  # (n, n)
    
    # Storage for time series
    covariance_series = []
    volatility_series = []
    
    # EWMA recursion
    for t in range(min_periods, T):
        r_t = R[t, :].reshape(-1, 1)  # (n, 1)
        
        # Update: Σ_t = λ * Σ_{t-1} + (1-λ) * r_t * r_t^T
        Sigma = lambda_decay * Sigma + (1.0 - lambda_decay) * (r_t @ r_t.T)
        
        # Store
        covariance_series.append(Sigma.copy())
        volatility_series.append(np.sqrt(np.diag(Sigma)))
    
    # Latest covariance and correlation
    covariance_matrix = Sigma
    
    # Compute correlation from covariance
    # Corr = D^{-1/2} * Cov * D^{-1/2}  where D = diag(Cov)
    vols = np.sqrt(np.diag(covariance_matrix))
    vols_inv = 1.0 / np.maximum(vols, 1e-12)
    D_inv_sqrt = np.diag(vols_inv)
    correlation_matrix = D_inv_sqrt @ covariance_matrix @ D_inv_sqrt
    
    # Ensure correlation is exactly 1 on diagonal (numerical stability)
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Build time series DataFrames
    covariance_dates = dates[min_periods:]
    
    # Volatilities time series
    vol_df = pd.DataFrame(
        volatility_series,
        index=covariance_dates,
        columns=asset_names
    )
    
    # Annualized volatilities (252 trading days)
    annualized_vols = vol_df.iloc[-1] * np.sqrt(252)
    
    # Extract correlation time series for all pairs
    correlation_pairs = {}
    for i in range(n):
        for j in range(i+1, n):
            pair_name = f"{asset_names[i]}_vs_{asset_names[j]}"
            corr_series = []
            for Sigma_t in covariance_series:
                vols_t = np.sqrt(np.diag(Sigma_t))
                corr_ij = Sigma_t[i, j] / (vols_t[i] * vols_t[j])
                corr_series.append(corr_ij)
            correlation_pairs[pair_name] = pd.Series(corr_series, index=covariance_dates)
    
    # Priority 3: Apply Ledoit-Wolf shrinkage for improved out-of-sample robustness
    shrinkage_intensity = 0.0
    unshrunk_covariance = None
    shrinkage_applied = False
    
    if apply_shrinkage:
        try:
            # Import shrinkage function
            from decision.portfolio_utils import shrink_covariance_ledoit_wolf
            
            # Store original EWMA covariance before shrinkage
            unshrunk_covariance = covariance_matrix.copy()
            
            # Apply Ledoit-Wolf shrinkage
            shrinkage_result = shrink_covariance_ledoit_wolf(
                sample_cov=covariance_matrix,
                returns=returns_clean.values,
                shrinkage_target=shrinkage_target
            )
            
            # Use shrunk covariance as final estimate
            covariance_matrix = shrinkage_result['shrunk_cov']
            shrinkage_intensity = shrinkage_result['shrinkage_intensity']
            shrinkage_applied = True
            
            # Recompute correlation matrix from shrunk covariance
            vols_shrunk = np.sqrt(np.diag(covariance_matrix))
            vols_inv_shrunk = 1.0 / np.maximum(vols_shrunk, 1e-12)
            D_inv_sqrt_shrunk = np.diag(vols_inv_shrunk)
            correlation_matrix = D_inv_sqrt_shrunk @ covariance_matrix @ D_inv_sqrt_shrunk
            np.fill_diagonal(correlation_matrix, 1.0)
            
        except Exception as e:
            # Fallback: use unshrunk EWMA covariance if shrinkage fails
            shrinkage_applied = False
            shrinkage_intensity = 0.0
            # covariance_matrix already set to EWMA result above
    
    # Build result dictionary
    result = {
        "covariance_matrix": covariance_matrix,
        "correlation_matrix": correlation_matrix,
        "covariance_series": covariance_series,
        "volatilities": vol_df,
        "annualized_volatilities": annualized_vols,
        "correlations": correlation_pairs,
        "asset_names": asset_names,
        "n_assets": n,
        "n_observations": T,
        "lambda_decay": lambda_decay,
        "shrinkage_applied": shrinkage_applied,
        "shrinkage_intensity": shrinkage_intensity,
    }
    
    # Add unshrunk covariance if shrinkage was applied (for comparison)
    if shrinkage_applied and unshrunk_covariance is not None:
        result["unshrunk_covariance_matrix"] = unshrunk_covariance
    
    return result


def compute_portfolio_volatility(
    weights: np.ndarray,
    covariance_matrix: np.ndarray,
) -> float:
    """
    Compute portfolio volatility given weights and covariance matrix.
    
    σ_p = sqrt(w^T * Σ * w)
    
    Args:
        weights: Portfolio weights (n,)
        covariance_matrix: Covariance matrix (n, n)
        
    Returns:
        Portfolio volatility (standard deviation)
    """
    w = np.asarray(weights, dtype=float).ravel()
    cov = np.asarray(covariance_matrix, dtype=float)
    
    variance = w @ cov @ w
    return float(np.sqrt(max(variance, 0.0)))


def compute_rolling_correlation(
    returns1: pd.Series,
    returns2: pd.Series,
    lambda_decay: float = 0.94,
    min_periods: int = 100,
) -> pd.Series:
    """
    Compute EWMA rolling correlation between two return series.
    
    Args:
        returns1: First return series
        returns2: Second return series
        lambda_decay: EWMA decay parameter
        min_periods: Minimum observations for stable estimate
        
    Returns:
        Time series of correlations
    """
    # Combine into DataFrame and use main function
    df = pd.concat([returns1, returns2], axis=1, join='inner').dropna()
    if len(df) < min_periods:
        raise ValueError(f"Insufficient overlapping data: {len(df)} observations")
    
    df.columns = ['asset1', 'asset2']
    
    result = compute_ewma_covariance(df, lambda_decay=lambda_decay, min_periods=min_periods)
    
    # Extract correlation series
    pair_key = 'asset1_vs_asset2'
    return result['correlations'][pair_key]
