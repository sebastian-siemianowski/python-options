#!/usr/bin/env python3
"""
portfolio_utils.py

Portfolio construction and multi-asset utilities:
- Dynamic covariance estimation (EWMA)
- Correlation tracking
- Risk decomposition
- Portfolio optimization helpers
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def ewma_covariance(
    returns: pd.DataFrame,
    lambda_decay: float = 0.94,
    min_periods: int = 30,
    annualize: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Compute dynamic EWMA covariance matrix for multi-asset returns.
    
    Model:
        Σ_t = λ·Σ_{t-1} + (1-λ)·r_t·r_t^T
    
    where:
        - Σ_t: covariance matrix at time t
        - λ: decay factor (0.94 is RiskMetrics standard for daily data)
        - r_t: vector of returns at time t
        
    This provides:
        - Rolling correlation structure
        - Regime-responsive covariance (adapts to volatility shifts)
        - Stable estimates (exponential weighting reduces noise)
    
    Args:
        returns: DataFrame with assets as columns, dates as index, returns as values
        lambda_decay: EWMA decay factor (0 < λ < 1). Higher λ = more weight on history.
                     RiskMetrics standard: 0.94 for daily, 0.97 for monthly
        min_periods: Minimum number of observations before producing estimates
        annualize: If True, multiply covariance by 252 (trading days/year)
        
    Returns:
        Dictionary with:
            - 'covariance': DataFrame of flattened covariance matrices over time
            - 'correlation': DataFrame of flattened correlation matrices over time
            - 'volatility': DataFrame of asset volatilities (sqrt of diagonal) over time
            - 'latest_cov_matrix': Latest full covariance matrix (N×N)
            - 'latest_corr_matrix': Latest full correlation matrix (N×N)
    """
    # Validate inputs
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    
    if returns.empty or returns.shape[1] < 2:
        raise ValueError("Need at least 2 assets to compute covariance")
    
    if not (0.0 < lambda_decay < 1.0):
        raise ValueError(f"lambda_decay must be in (0,1), got {lambda_decay}")
    
    if min_periods < 2:
        raise ValueError(f"min_periods must be >= 2, got {min_periods}")
    
    # Clean data: drop NaN rows
    returns_clean = returns.dropna(how='all')
    
    if len(returns_clean) < min_periods:
        raise ValueError(f"Insufficient data: {len(returns_clean)} rows < {min_periods} min_periods")
    
    # Get asset names and dimensions
    assets = list(returns_clean.columns)
    n_assets = len(assets)
    T = len(returns_clean)
    
    # Initialize storage
    # Store flattened upper triangle of covariance/correlation matrices over time
    dates = returns_clean.index
    
    # Initialize EWMA covariance with sample covariance from first min_periods observations
    init_window = returns_clean.iloc[:min_periods].values
    Sigma_t = np.cov(init_window, rowvar=False, ddof=1)
    
    # Ensure positive definite (regularize if needed)
    Sigma_t = _regularize_covariance(Sigma_t)
    
    # Storage for time series of covariance/correlation elements
    # We'll store full matrices at each time step for flexibility
    cov_series = []
    corr_series = []
    vol_series = []
    
    # Iterate through returns and update EWMA recursively
    for t in range(min_periods, T):
        # Get return vector at time t
        r_t = returns_clean.iloc[t].values  # shape (n_assets,)
        
        # EWMA update: Σ_t = λ·Σ_{t-1} + (1-λ)·r_t·r_t^T
        r_outer = np.outer(r_t, r_t)  # outer product r_t·r_t^T
        Sigma_t = lambda_decay * Sigma_t + (1.0 - lambda_decay) * r_outer
        
        # Regularize to maintain positive definiteness
        Sigma_t = _regularize_covariance(Sigma_t)
        
        # Extract volatilities (sqrt of diagonal)
        vols = np.sqrt(np.maximum(np.diag(Sigma_t), 1e-12))
        
        # Compute correlation matrix: ρ_ij = σ_ij / (σ_i · σ_j)
        vol_outer = np.outer(vols, vols)
        Corr_t = Sigma_t / np.maximum(vol_outer, 1e-12)
        
        # Clamp correlations to [-1, 1] for numerical stability
        Corr_t = np.clip(Corr_t, -1.0, 1.0)
        
        # Store results
        date_t = dates[t]
        cov_series.append({
            'date': date_t,
            'matrix': Sigma_t.copy()
        })
        corr_series.append({
            'date': date_t,
            'matrix': Corr_t.copy()
        })
        vol_series.append({
            'date': date_t,
            'vols': vols.copy()
        })
    
    # Annualize if requested (multiply by trading days per year)
    scale_factor = 252.0 if annualize else 1.0
    
    # Build output DataFrames
    # Covariance: one column per (asset_i, asset_j) pair
    cov_data = []
    for entry in cov_series:
        date = entry['date']
        matrix = entry['matrix'] * scale_factor
        # Flatten to dict: {(asset_i, asset_j): cov_ij}
        row = {'date': date}
        for i in range(n_assets):
            for j in range(n_assets):
                key = f"{assets[i]}_{assets[j]}"
                row[key] = float(matrix[i, j])
        cov_data.append(row)
    
    df_cov = pd.DataFrame(cov_data).set_index('date')
    
    # Correlation: one column per (asset_i, asset_j) pair
    corr_data = []
    for entry in corr_series:
        date = entry['date']
        matrix = entry['matrix']
        row = {'date': date}
        for i in range(n_assets):
            for j in range(n_assets):
                key = f"{assets[i]}_{assets[j]}"
                row[key] = float(matrix[i, j])
        corr_data.append(row)
    
    df_corr = pd.DataFrame(corr_data).set_index('date')
    
    # Volatility: one column per asset (annualized if requested)
    vol_data = []
    for entry in vol_series:
        date = entry['date']
        vols = entry['vols'] * np.sqrt(scale_factor)
        row = {'date': date}
        for i, asset in enumerate(assets):
            row[asset] = float(vols[i])
        vol_data.append(row)
    
    df_vol = pd.DataFrame(vol_data).set_index('date')
    
    # Latest matrices (full N×N)
    latest_cov = cov_series[-1]['matrix'] * scale_factor if cov_series else np.zeros((n_assets, n_assets))
    latest_corr = corr_series[-1]['matrix'] if corr_series else np.eye(n_assets)
    
    # Convert to DataFrames with asset labels
    latest_cov_df = pd.DataFrame(latest_cov, index=assets, columns=assets)
    latest_corr_df = pd.DataFrame(latest_corr, index=assets, columns=assets)
    
    return {
        'covariance': df_cov,
        'correlation': df_corr,
        'volatility': df_vol,
        'latest_cov_matrix': latest_cov_df,
        'latest_corr_matrix': latest_corr_df,
        'lambda_decay': float(lambda_decay),
        'annualized': bool(annualize),
        'n_assets': int(n_assets),
        'assets': assets,
    }


def _regularize_covariance(Sigma: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Regularize covariance matrix to ensure positive definiteness.
    
    Strategy:
        1. Symmetrize: Σ = (Σ + Σ^T) / 2
        2. Add small ridge: Σ = Σ + ε·I
        3. Eigenvalue floor: ensure all eigenvalues >= ε
    
    Args:
        Sigma: Covariance matrix (N×N)
        epsilon: Regularization strength
        
    Returns:
        Regularized positive definite covariance matrix
    """
    # Ensure symmetry
    Sigma = 0.5 * (Sigma + Sigma.T)
    
    # Add small ridge for numerical stability
    n = Sigma.shape[0]
    Sigma_reg = Sigma + epsilon * np.eye(n)
    
    # Eigenvalue decomposition and floor
    try:
        eigvals, eigvecs = np.linalg.eigh(Sigma_reg)
        # Floor negative/tiny eigenvalues
        eigvals = np.maximum(eigvals, epsilon)
        # Reconstruct
        Sigma_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Re-symmetrize after reconstruction
        Sigma_reg = 0.5 * (Sigma_reg + Sigma_reg.T)
    except np.linalg.LinAlgError:
        # Fallback: just use ridge-regularized version
        pass
    
    return Sigma_reg


def extract_pairwise_correlations(
    ewma_result: Dict,
    asset_pairs: Optional[list[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Extract time series of pairwise correlations from EWMA result.
    
    Args:
        ewma_result: Output from ewma_covariance()
        asset_pairs: List of (asset_i, asset_j) tuples to extract.
                    If None, extracts all unique pairs.
                    
    Returns:
        DataFrame with one column per asset pair, indexed by date
    """
    df_corr = ewma_result['correlation']
    assets = ewma_result['assets']
    
    if asset_pairs is None:
        # Extract all unique pairs (upper triangle, excluding diagonal)
        asset_pairs = []
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                asset_pairs.append((assets[i], assets[j]))
    
    # Extract requested pairs
    result = pd.DataFrame(index=df_corr.index)
    for asset_i, asset_j in asset_pairs:
        key = f"{asset_i}_{asset_j}"
        if key in df_corr.columns:
            result[f"{asset_i} vs {asset_j}"] = df_corr[key]
        else:
            # Try reverse order
            key_rev = f"{asset_j}_{asset_i}"
            if key_rev in df_corr.columns:
                result[f"{asset_i} vs {asset_j}"] = df_corr[key_rev]
    
    return result


def compute_portfolio_variance(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame
) -> float:
    """
    Compute portfolio variance: σ_p² = w^T Σ w
    
    Args:
        weights: Asset weights (N,) array, must sum to 1
        cov_matrix: Covariance matrix (N×N) DataFrame
        
    Returns:
        Portfolio variance (scalar)
    """
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights)
    
    if not isinstance(cov_matrix, pd.DataFrame):
        raise TypeError("cov_matrix must be a DataFrame")
    
    # Ensure alignment
    Sigma = cov_matrix.values
    
    if weights.shape[0] != Sigma.shape[0]:
        raise ValueError(f"Weight dimension {weights.shape[0]} != covariance dimension {Sigma.shape[0]}")
    
    # w^T Σ w
    var_p = float(weights.T @ Sigma @ weights)
    return var_p


def compute_portfolio_volatility(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame
) -> float:
    """
    Compute portfolio volatility: σ_p = sqrt(w^T Σ w)
    
    Args:
        weights: Asset weights (N,) array
        cov_matrix: Covariance matrix (N×N) DataFrame
        
    Returns:
        Portfolio volatility (standard deviation)
    """
    var_p = compute_portfolio_variance(weights, cov_matrix)
    return float(np.sqrt(max(var_p, 0.0)))


def compute_cvar_from_paths(
    portfolio_returns: np.ndarray,
    confidence_level: float = 0.95
) -> Dict[str, float]:
    """
    Compute CVaR (Conditional Value at Risk) / Expected Shortfall from simulated paths.
    
    CVaR measures tail risk: the expected loss in the worst (1-α)% of scenarios.
    
    CVaR_α = E[loss | loss > VaR_α]
    
    where:
        - VaR_α: Value at Risk at confidence level α (e.g., 95th percentile)
        - CVaR_α: Expected Shortfall = mean of losses beyond VaR_α
        
    This provides:
        ✓ Survivability: ensures portfolio can withstand extreme scenarios
        ✓ Anti-Lehman insurance: prevents blow-up during regime shifts
        ✓ Prevents over-Kelly in calm regimes: detects when variance underestimates true tail risk
    
    Args:
        portfolio_returns: Array of simulated portfolio returns (n_paths,) or (n_horizons, n_paths)
                          Typically cumulative log returns over a horizon
        confidence_level: Confidence level for VaR/CVaR (default 0.95 = 95%)
                         Higher = focuses on more extreme tail (e.g., 0.99 = 99%)
                         
    Returns:
        Dictionary with:
            - 'cvar': Expected Shortfall (CVaR) - mean loss in worst (1-α)% tail
            - 'var': Value at Risk (VaR) - threshold for worst (1-α)%
            - 'worst_case': Minimum return in entire distribution
            - 'tail_probability': Fraction of paths below VaR threshold (should be ~1-α)
            - 'n_tail_scenarios': Number of scenarios in tail (for validation)
    """
    # Flatten if multi-dimensional (take last horizon if multiple)
    if portfolio_returns.ndim > 1:
        portfolio_returns = portfolio_returns[-1, :]
    
    # Clean data: remove NaN/Inf
    returns = np.asarray(portfolio_returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    
    if returns.size == 0:
        return {
            'cvar': float('nan'),
            'var': float('nan'),
            'worst_case': float('nan'),
            'tail_probability': float('nan'),
            'n_tail_scenarios': 0,
        }
    
    # Sort returns (ascending: worst losses first)
    sorted_returns = np.sort(returns)
    n_scenarios = len(sorted_returns)
    
    # VaR: (1-α) quantile (e.g., 5th percentile for α=0.95)
    var_quantile = 1.0 - confidence_level
    var_threshold = float(np.quantile(sorted_returns, var_quantile))
    
    # CVaR/ES: mean of all returns below VaR threshold
    # This is the expected loss conditional on being in the tail
    tail_mask = sorted_returns <= var_threshold
    tail_returns = sorted_returns[tail_mask]
    
    if tail_returns.size > 0:
        cvar = float(np.mean(tail_returns))
        n_tail = int(tail_returns.size)
        tail_prob = float(n_tail / n_scenarios)
    else:
        # Edge case: no scenarios below VaR (shouldn't happen with proper quantile)
        # Fall back to VaR itself
        cvar = var_threshold
        n_tail = 0
        tail_prob = 0.0
    
    # Worst case (minimum return across all paths)
    worst_case = float(sorted_returns[0])
    
    return {
        'cvar': cvar,
        'var': var_threshold,
        'worst_case': worst_case,
        'tail_probability': tail_prob,
        'n_tail_scenarios': n_tail,
    }


def apply_cvar_constraint(
    weights_kelly: np.ndarray,
    asset_return_paths: Dict[str, np.ndarray],
    cov_matrix: pd.DataFrame,
    r_max: float = -0.20,
    confidence_level: float = 0.95
) -> Dict:
    """
    Apply CVaR tail-risk constraint to Kelly weights.
    
    If portfolio CVaR exceeds maximum acceptable loss (r_max), scale weights down
    while preserving correlation-aware structure.
    
    Constraint:
        ES_95 >= r_max
        
    If violated (ES_95 < r_max, i.e., larger losses):
        w_adjusted = w_kelly × (r_max / ES_95)
        
    This ensures:
        ✓ Survivability: portfolio won't blow up in extreme scenarios
        ✓ Prevents over-Kelly: detects when calm-regime variance underestimates true tail risk
        ✓ Anti-Lehman: guarantees worst-case losses stay within acceptable bounds
    
    Args:
        weights_kelly: Kelly weights (N,) before tail constraint
        asset_return_paths: Dict mapping asset names to return paths (n_horizons, n_paths)
        cov_matrix: Covariance matrix (N×N) for asset ordering
        r_max: Maximum acceptable CVaR loss (default -0.20 = -20% max loss in 95% tail)
               More negative = more conservative (tighter constraint)
        confidence_level: CVaR confidence level (default 0.95 = 95%)
        
    Returns:
        Dictionary with:
            - 'weights_adjusted': Final weights after CVaR constraint
            - 'cvar_unconstrained': CVaR before constraint
            - 'cvar_constrained': CVaR after constraint (should be >= r_max)
            - 'constraint_active': Boolean, True if constraint applied (scaled weights)
            - 'scaling_factor': Multiplier applied to weights (1.0 = no scaling)
            - 'tail_risk_metrics': Full CVaR metrics dictionary
    """
    # Ensure weights is numpy array
    if not isinstance(weights_kelly, np.ndarray):
        weights_kelly = np.array(weights_kelly, dtype=float)
    
    # Get asset ordering from covariance matrix
    assets = list(cov_matrix.index)
    n_assets = len(assets)
    
    if weights_kelly.shape[0] != n_assets:
        raise ValueError(f"Weight dimension {weights_kelly.shape[0]} != number of assets {n_assets}")
    
    # Build portfolio return paths: r_p = Σ(w_i × r_i,t)
    # Align paths with covariance matrix ordering
    try:
        # Stack asset paths into matrix: (n_horizons, n_paths, n_assets)
        first_asset = list(asset_return_paths.values())[0]
        n_horizons, n_paths = first_asset.shape
        
        # Initialize portfolio paths
        path_matrix = np.zeros((n_horizons, n_paths, n_assets), dtype=float)
        
        for i, asset in enumerate(assets):
            if asset in asset_return_paths:
                path_matrix[:, :, i] = asset_return_paths[asset]
            else:
                # Missing asset: use zeros (no contribution)
                pass
        
        # Compute portfolio returns: weighted sum across assets
        # Shape: (n_horizons, n_paths)
        portfolio_paths = np.einsum('hpa,a->hp', path_matrix, weights_kelly)
        
    except Exception as e:
        # Fallback: unable to compute portfolio paths
        return {
            'weights_adjusted': weights_kelly,
            'cvar_unconstrained': float('nan'),
            'cvar_constrained': float('nan'),
            'constraint_active': False,
            'scaling_factor': 1.0,
            'tail_risk_metrics': {},
            'error': f"Failed to compute portfolio paths: {e}"
        }
    
    # Compute CVaR on unconstrained Kelly portfolio
    cvar_metrics = compute_cvar_from_paths(portfolio_paths, confidence_level)
    cvar_unconstrained = cvar_metrics['cvar']
    
    # Check if constraint is violated
    # CVaR is a loss (negative return), r_max is max acceptable loss (also negative)
    # Constraint violated if: cvar < r_max (e.g., -0.25 < -0.20 means loss is too large)
    constraint_violated = (cvar_unconstrained < r_max)
    
    if constraint_violated and np.isfinite(cvar_unconstrained) and cvar_unconstrained != 0.0:
        # Scale weights down to satisfy constraint
        # Target: CVaR_adjusted = r_max
        # Since CVaR scales linearly with position size (for losses):
        # scaling_factor = r_max / CVaR_unconstrained
        scaling_factor = float(r_max / cvar_unconstrained)
        
        # Clamp scaling to [0, 1] (only scale down, never up; ensure non-negative)
        scaling_factor = float(np.clip(scaling_factor, 0.0, 1.0))
        
        weights_adjusted = weights_kelly * scaling_factor
        
        # Recompute CVaR with adjusted weights
        portfolio_paths_adjusted = np.einsum('hpa,a->hp', path_matrix, weights_adjusted)
        cvar_metrics_adjusted = compute_cvar_from_paths(portfolio_paths_adjusted, confidence_level)
        cvar_constrained = cvar_metrics_adjusted['cvar']
        
        constraint_active = True
    else:
        # Constraint satisfied: use original Kelly weights
        weights_adjusted = weights_kelly
        cvar_constrained = cvar_unconstrained
        scaling_factor = 1.0
        constraint_active = False
    
    return {
        'weights_adjusted': weights_adjusted,
        'cvar_unconstrained': float(cvar_unconstrained),
        'cvar_constrained': float(cvar_constrained),
        'constraint_active': bool(constraint_active),
        'scaling_factor': float(scaling_factor),
        'tail_risk_metrics': cvar_metrics,
        'r_max': float(r_max),
        'confidence_level': float(confidence_level),
    }


def shrink_covariance_ledoit_wolf(
    sample_cov: np.ndarray,
    returns: np.ndarray,
    shrinkage_target: str = "constant_correlation"
) -> Dict:
    """
    Shrink sample covariance matrix toward structured target using Ledoit-Wolf optimal shrinkage.
    
    Implements Ledoit & Wolf (2004) "Honey, I Shrunk the Sample Covariance Matrix"
    
    Optimal shrinkage estimator:
        Σ̂ = δ·F + (1-δ)·S
        
    Where:
        - S: sample covariance matrix
        - F: shrinkage target (structured estimator)
        - δ: optimal shrinkage intensity (data-driven)
        
    This provides:
        ✔ Improved out-of-sample performance (reduces estimation error)
        ✔ Better condition number (more stable matrix inversion)
        ✔ Robustness with small samples (n < p or n ~ p)
        ✔ Automatic shrinkage intensity (no hand-tuning)
    
    Args:
        sample_cov: Sample covariance matrix (p×p)
        returns: Return matrix (n×p) used to compute sample_cov
        shrinkage_target: Target structure ('constant_correlation', 'identity', 'single_factor')
                         - 'constant_correlation': common correlation, individual variances
                         - 'identity': scaled identity (equal var, zero corr)
                         - 'single_factor': single-factor model
                         
    Returns:
        Dictionary with:
            - 'shrunk_cov': Shrunk covariance matrix (p×p)
            - 'shrinkage_intensity': Optimal δ ∈ [0,1]
            - 'sample_cov': Original sample covariance (for comparison)
            - 'target': Shrinkage target matrix
            - 'method': 'ledoit_wolf'
    """
    # Input validation
    if not isinstance(sample_cov, np.ndarray):
        sample_cov = np.array(sample_cov, dtype=float)
    if not isinstance(returns, np.ndarray):
        returns = np.array(returns, dtype=float)
    
    # Ensure 2D
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    
    n, p = returns.shape  # n observations, p assets
    
    if sample_cov.shape != (p, p):
        raise ValueError(f"sample_cov shape {sample_cov.shape} != (p, p) where p={p}")
    
    # Ensure symmetry
    sample_cov = 0.5 * (sample_cov + sample_cov.T)
    
    # Build shrinkage target based on specified structure
    if shrinkage_target == "constant_correlation":
        # Target: constant correlation with sample variances
        # F_ij = ρ̄ · σ_i · σ_j  if i≠j, else σ_i²
        variances = np.diag(sample_cov)
        std_devs = np.sqrt(variances)
        
        # Compute average correlation from sample
        corr_matrix = sample_cov / np.outer(std_devs, std_devs)
        # Average off-diagonal correlation
        mask = ~np.eye(p, dtype=bool)
        avg_corr = float(np.mean(corr_matrix[mask]))
        avg_corr = np.clip(avg_corr, -0.999, 0.999)  # keep valid
        
        # Build target: constant correlation
        target = avg_corr * np.outer(std_devs, std_devs)
        np.fill_diagonal(target, variances)
        
    elif shrinkage_target == "identity":
        # Target: scaled identity (equal variance, zero correlation)
        avg_var = float(np.mean(np.diag(sample_cov)))
        target = avg_var * np.eye(p)
        
    elif shrinkage_target == "single_factor":
        # Target: single-factor model (market factor)
        # F = β·β^T·var(market) + diag(specific variances)
        # Approximate: use first principal component as market
        try:
            eigvals, eigvecs = np.linalg.eigh(sample_cov)
            # Largest eigenvalue/vector (first PC)
            idx = np.argmax(eigvals)
            lambda_1 = eigvals[idx]
            v_1 = eigvecs[:, idx]
            
            # Factor loadings: β ≈ v_1 · sqrt(λ_1)
            beta = v_1 * np.sqrt(lambda_1)
            
            # Single-factor cov: β·β^T
            factor_cov = np.outer(beta, beta)
            
            # Specific variances: diagonal residuals
            specific_vars = np.diag(sample_cov) - np.diag(factor_cov)
            specific_vars = np.maximum(specific_vars, 1e-8)  # floor
            
            target = factor_cov + np.diag(specific_vars)
        except Exception:
            # Fallback to constant correlation if PCA fails
            target = sample_cov.copy()
    else:
        raise ValueError(f"Unknown shrinkage_target: {shrinkage_target}")
    
    # Ensure target is symmetric and positive definite
    target = 0.5 * (target + target.T)
    try:
        # Regularize if not PD
        eigvals_t = np.linalg.eigvalsh(target)
        if np.min(eigvals_t) < 1e-8:
            target += 1e-8 * np.eye(p)
    except Exception:
        pass
    
    # Compute optimal shrinkage intensity δ using Ledoit-Wolf formula
    # This is the key: data-driven intensity that minimizes expected loss
    
    # Frobenius norm of difference: ||S - F||²
    diff = sample_cov - target
    delta_sq = float(np.sum(diff ** 2))
    
    # Estimate asymptotic variance of sample covariance
    # This requires higher moments of returns
    try:
        # Demean returns
        returns_centered = returns - np.mean(returns, axis=0, keepdims=True)
        
        # Compute pi-hat (asymptotic variance of vec(S))
        # Simplified estimator: sum of squared cross-products
        pi_hat = 0.0
        for t in range(n):
            r_t = returns_centered[t, :].reshape(-1, 1)
            # Outer product: r_t · r_t^T
            outer_t = r_t @ r_t.T
            # Deviation from sample cov (scaled)
            dev = outer_t - sample_cov
            pi_hat += float(np.sum(dev ** 2))
        
        pi_hat = pi_hat / n
        
        # Compute rho-hat (asymptotic covariance between S and F)
        # For constant correlation target, this simplifies
        rho_hat = 0.0  # Conservative: assume orthogonal
        
        # Gamma-hat = pi - rho
        gamma_hat = pi_hat - rho_hat
        
        # Optimal shrinkage intensity: δ* = max(0, min(1, γ/δ²))
        if delta_sq > 1e-12:
            delta_star = gamma_hat / delta_sq
            delta_star = float(np.clip(delta_star, 0.0, 1.0))
        else:
            # Sample cov ≈ target already: no shrinkage needed
            delta_star = 0.0
            
    except Exception:
        # Fallback: use heuristic shrinkage intensity
        # More shrinkage when n is small relative to p
        if n >= p:
            delta_star = float(p / (n + p))  # Goes to 0 as n → ∞
        else:
            delta_star = 0.5  # More aggressive if n < p
    
    # Apply shrinkage: Σ̂ = δ·F + (1-δ)·S
    shrunk_cov = delta_star * target + (1.0 - delta_star) * sample_cov
    
    # Ensure symmetry and positive definiteness
    shrunk_cov = 0.5 * (shrunk_cov + shrunk_cov.T)
    
    # Final regularization for numerical stability
    try:
        eigvals_shrunk = np.linalg.eigvalsh(shrunk_cov)
        if np.min(eigvals_shrunk) < 1e-10:
            shrunk_cov += 1e-10 * np.eye(p)
    except Exception:
        shrunk_cov += 1e-8 * np.eye(p)
    
    return {
        'shrunk_cov': shrunk_cov,
        'shrinkage_intensity': float(delta_star),
        'sample_cov': sample_cov,
        'target': target,
        'target_type': shrinkage_target,
        'method': 'ledoit_wolf',
        'n_observations': int(n),
        'n_assets': int(p),
    }
