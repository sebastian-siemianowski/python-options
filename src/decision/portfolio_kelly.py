#!/usr/bin/env python3
"""
portfolio_kelly.py

Portfolio optimization using Kelly criterion with EWMA covariance.
Implements fractional Kelly weights: w = (1/2) * Σ^(-1) * μ
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_kelly_weights(
    mu_vec: np.ndarray,
    cov_matrix: np.ndarray,
    risk_fraction: float = 0.5,
    regularization: float = 1e-8,
    max_weight: float = 0.40,
    min_weight: float = -0.20,
) -> Dict[str, np.ndarray]:
    """
    Compute portfolio Kelly weights using fractional Kelly criterion.
    
    Fractional Kelly formula:
        w = (risk_fraction) * Σ^(-1) * μ
    
    Where:
        - μ: vector of expected log returns (n×1)
        - Σ: covariance matrix of returns (n×n)
        - risk_fraction: Kelly fraction (0.5 = half-Kelly for robustness)
    
    This yields:
        ✔ Capital-efficient allocation (maximizes log wealth growth rate)
        ✔ Correlation-aware sizing (diversification bonus captured)
        ✔ Blow-up risk reduced (via fractional Kelly)
        ✔ Leverage naturally emerges when correlations are low
    
    Args:
        mu_vec: Expected returns vector (n,) or (n,1)
        cov_matrix: Covariance matrix (n,n)
        risk_fraction: Kelly fraction (default 0.5 = half-Kelly)
        regularization: Ridge regularization for numerical stability
        max_weight: Maximum weight per asset (clamp upper bound)
        min_weight: Minimum weight per asset (clamp lower bound, can be negative for shorts)
        
    Returns:
        Dictionary with:
            - weights_raw: Raw Kelly weights before normalization
            - weights_normalized: Weights normalized to sum to risk_fraction
            - weights_clamped: Clamped weights respecting position limits
            - leverage: Sum of absolute weights (>1 implies leverage)
            - diversification_ratio: Portfolio vol / weighted avg individual vols
    """
    # Input validation and reshaping
    mu = np.asarray(mu_vec, dtype=float).ravel()
    cov = np.asarray(cov_matrix, dtype=float)
    
    n = len(mu)
    if cov.shape != (n, n):
        raise ValueError(f"Dimension mismatch: mu has {n} assets but cov is {cov.shape}")
    
    # Check for NaNs/Infs
    if not np.all(np.isfinite(mu)):
        raise ValueError("mu_vec contains NaN or Inf")
    if not np.all(np.isfinite(cov)):
        raise ValueError("cov_matrix contains NaN or Inf")
    
    # Ensure covariance matrix is symmetric
    cov = 0.5 * (cov + cov.T)
    
    # Regularization for numerical stability (ridge)
    # Add small diagonal term to ensure positive definiteness
    cov_reg = cov + regularization * np.eye(n)
    
    # Compute inverse via Cholesky (more stable than direct inverse)
    try:
        L = np.linalg.cholesky(cov_reg)
        cov_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse if Cholesky fails
        try:
            cov_inv = np.linalg.pinv(cov_reg)
        except Exception:
            # Last resort: return equal weights
            weights_raw = np.ones(n) / n
            return {
                "weights_raw": weights_raw,
                "weights_normalized": weights_raw * risk_fraction,
                "weights_clamped": np.clip(weights_raw * risk_fraction, min_weight, max_weight),
                "leverage": float(np.sum(np.abs(weights_raw * risk_fraction))),
                "diversification_ratio": 1.0,
                "method": "fallback_equal_weight",
            }
    
    # Fractional Kelly weights: w = (risk_fraction) * Σ^(-1) * μ
    weights_raw = risk_fraction * (cov_inv @ mu)
    
    # Normalize to risk budget if sum exceeds it (but allow <1 if conservative)
    weight_sum = np.sum(np.abs(weights_raw))
    if weight_sum > 1.0:
        # Normalize down to avoid over-leverage
        weights_normalized = weights_raw / weight_sum
    else:
        # Keep raw weights if already conservative
        weights_normalized = weights_raw.copy()
    
    # Clamp individual positions to limits
    weights_clamped = np.clip(weights_normalized, min_weight, max_weight)
    
    # Renormalize after clamping to maintain budget (optional)
    # Comment out if you prefer to respect clamps strictly
    # clamped_sum = np.sum(np.abs(weights_clamped))
    # if clamped_sum > 0:
    #     weights_clamped = weights_clamped / clamped_sum * risk_fraction
    
    # Portfolio statistics
    leverage = float(np.sum(np.abs(weights_clamped)))
    
    # Diversification ratio: portfolio vol / weighted avg individual vols
    try:
        portfolio_variance = weights_clamped @ cov @ weights_clamped
        portfolio_vol = float(np.sqrt(max(portfolio_variance, 0.0)))
        
        individual_vols = np.sqrt(np.diag(cov))
        weighted_avg_vol = float(np.sum(np.abs(weights_clamped) * individual_vols))
        
        if weighted_avg_vol > 1e-12:
            diversification_ratio = portfolio_vol / weighted_avg_vol
        else:
            diversification_ratio = 1.0
    except Exception:
        diversification_ratio = 1.0
    
    return {
        "weights_raw": weights_raw,
        "weights_normalized": weights_normalized,
        "weights_clamped": weights_clamped,
        "leverage": leverage,
        "diversification_ratio": float(diversification_ratio),
        "method": "fractional_kelly",
    }


def compute_portfolio_statistics(
    weights: np.ndarray,
    mu_vec: np.ndarray,
    cov_matrix: np.ndarray,
) -> Dict[str, float]:
    """
    Compute portfolio-level statistics given weights.
    
    Args:
        weights: Portfolio weights (n,)
        mu_vec: Expected returns (n,)
        cov_matrix: Covariance matrix (n,n)
        
    Returns:
        Dictionary with portfolio statistics
    """
    w = np.asarray(weights, dtype=float).ravel()
    mu = np.asarray(mu_vec, dtype=float).ravel()
    cov = np.asarray(cov_matrix, dtype=float)
    
    # Portfolio expected return
    portfolio_return = float(w @ mu)
    
    # Portfolio variance and volatility
    portfolio_variance = float(w @ cov @ w)
    portfolio_vol = float(np.sqrt(max(portfolio_variance, 0.0)))
    
    # Sharpe ratio (assuming log returns, so no risk-free rate adjustment here)
    sharpe = float(portfolio_return / portfolio_vol) if portfolio_vol > 1e-12 else 0.0
    
    # Individual asset contributions to portfolio variance
    marginal_contributions = cov @ w
    contributions = w * marginal_contributions
    contribution_pct = contributions / portfolio_variance if portfolio_variance > 1e-12 else np.zeros_like(w)
    
    return {
        "expected_return": portfolio_return,
        "volatility": portfolio_vol,
        "variance": portfolio_variance,
        "sharpe_ratio": sharpe,
        "marginal_var_contributions": marginal_contributions,
        "var_contributions": contributions,
        "var_contribution_pct": contribution_pct,
    }


def build_multi_asset_portfolio(
    asset_returns: Dict[str, pd.Series],
    asset_expected_returns: Dict[str, float],
    horizon_days: int = 21,
    ewma_lambda: float = 0.94,
    risk_fraction: float = 0.5,
    asset_return_paths: Optional[Dict[str, np.ndarray]] = None,
    apply_cvar: bool = True,
    r_max: float = -0.20,
    cvar_confidence: float = 0.95,
) -> Dict:
    """
    Build Kelly-optimal portfolio from multiple assets with CVaR tail-risk constraint.
    
    Step 3 Enhancement: Adds tail-aware risk constraint using Expected Shortfall.
    
    Process:
        1. Compute EWMA covariance matrix (correlation-aware)
        2. Compute Kelly weights: w = (1/2) × Σ⁻¹ × μ
        3. Apply CVaR constraint if asset_return_paths provided:
           - Compute portfolio CVaR from simulated paths
           - Scale weights if ES₉₅ < r_max (tail risk too large)
           - Preserve correlation structure during scaling
    
    Args:
        asset_returns: Dictionary mapping asset names to return series
        asset_expected_returns: Dictionary mapping asset names to expected returns
        horizon_days: Forecast horizon in trading days
        ewma_lambda: EWMA decay parameter for covariance
        risk_fraction: Kelly fraction (0.5 = half-Kelly)
        asset_return_paths: Optional dict of simulated return paths (n_horizons, n_paths) per asset
                           Required for CVaR constraint. If None, constraint skipped.
        apply_cvar: Whether to apply CVaR constraint (default True if paths provided)
        r_max: Maximum acceptable CVaR loss (default -0.20 = -20% max tail loss)
        cvar_confidence: CVaR confidence level (default 0.95 = 95%)
        
    Returns:
        Dictionary with portfolio weights, statistics, and CVaR metrics
    """
    # Import here to avoid circular dependencies
    import sys
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from tuning.ewma_covariance import compute_ewma_covariance
    
    # Align return series
    asset_names = sorted(asset_returns.keys())
    if len(asset_names) < 2:
        raise ValueError("Need at least 2 assets for portfolio construction")
    
    # Build returns matrix
    returns_list = []
    aligned_names = []
    for name in asset_names:
        ret = asset_returns[name]
        if ret is not None and not ret.empty:
            returns_list.append(ret)
            aligned_names.append(name)
    
    if len(returns_list) < 2:
        raise ValueError("Need at least 2 assets with valid returns")
    
    # Concatenate and align
    returns_df = pd.concat(returns_list, axis=1, join='inner').dropna()
    returns_df.columns = aligned_names
    
    if len(returns_df) < 100:
        raise ValueError(f"Insufficient history: {len(returns_df)} observations (need >=100)")
    
    # Compute EWMA covariance
    ewma_result = compute_ewma_covariance(returns_df, lambda_decay=ewma_lambda)
    cov_matrix = ewma_result["covariance_matrix"]  # Latest covariance
    
    # Build expected returns vector (scale to horizon)
    mu_vec = np.array([
        asset_expected_returns.get(name, 0.0) * horizon_days
        for name in aligned_names
    ], dtype=float)
    
    # Compute Kelly weights
    kelly_result = compute_kelly_weights(
        mu_vec=mu_vec,
        cov_matrix=cov_matrix,
        risk_fraction=risk_fraction,
    )
    
    # Step 3: Apply CVaR tail-risk constraint if asset paths provided
    cvar_result = None
    final_weights = kelly_result["weights_clamped"]
    
    if apply_cvar and asset_return_paths is not None and len(asset_return_paths) > 0:
        try:
            # Import CVaR constraint function
            from decision.portfolio_utils import apply_cvar_constraint
            
            # Convert covariance matrix to DataFrame for apply_cvar_constraint
            cov_df = pd.DataFrame(cov_matrix, index=aligned_names, columns=aligned_names)
            
            # Apply CVaR constraint
            cvar_result = apply_cvar_constraint(
                weights_kelly=kelly_result["weights_clamped"],
                asset_return_paths=asset_return_paths,
                cov_matrix=cov_df,
                r_max=r_max,
                confidence_level=cvar_confidence,
            )
            
            # Use adjusted weights if constraint was applied
            final_weights = cvar_result["weights_adjusted"]
            
        except Exception as e:
            # If CVaR constraint fails, fall back to unconstrained Kelly weights
            # and log the error in the result
            cvar_result = {
                "error": f"CVaR constraint failed: {e}",
                "weights_adjusted": kelly_result["weights_clamped"],
                "constraint_active": False,
            }
    
    # Compute portfolio statistics with final weights (after CVaR constraint if applied)
    portfolio_stats = compute_portfolio_statistics(
        weights=final_weights,
        mu_vec=mu_vec,
        cov_matrix=cov_matrix,
    )
    
    result = {
        "asset_names": aligned_names,
        "weights_raw": kelly_result["weights_raw"],
        "weights_normalized": kelly_result["weights_normalized"],
        "weights_clamped": kelly_result["weights_clamped"],
        "weights_final": final_weights,  # Final weights after CVaR constraint
        "leverage": kelly_result["leverage"],
        "diversification_ratio": kelly_result["diversification_ratio"],
        "expected_returns": mu_vec,
        "covariance_matrix": cov_matrix,
        "correlation_matrix": ewma_result["correlation_matrix"],
        "portfolio_stats": portfolio_stats,
        "horizon_days": horizon_days,
        "risk_fraction": risk_fraction,
    }
    
    # Add CVaR constraint results if available
    if cvar_result is not None:
        result["cvar_constraint"] = cvar_result
    
    return result
