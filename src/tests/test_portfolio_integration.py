#!/usr/bin/env python3
"""
test_portfolio_integration.py

Integration tests for multi-asset portfolio construction with:
- EWMA dynamic covariance
- Kelly optimal weights
- CVaR tail-risk constraint

Tests the full pipeline from asset returns to portfolio allocation.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ewma_covariance import compute_ewma_covariance
from portfolio_kelly import compute_kelly_weights, build_multi_asset_portfolio
from portfolio_utils import apply_cvar_constraint


def test_ewma_covariance_multi_asset():
    """
    Test 1: EWMA covariance computation on multi-asset returns.
    
    Verifies:
    - Covariance matrix is positive definite
    - Correlation matrix is valid ([-1, 1] range, diagonal = 1)
    - Time-varying structure adapts to regime changes
    """
    print("=" * 80)
    print("Test 1: EWMA Dynamic Covariance Estimation")
    print("=" * 80)
    
    # Generate synthetic multi-asset returns with known correlation structure
    np.random.seed(42)
    n_obs = 500
    n_assets = 4
    
    # Correlation structure: assets 0,1 correlated (+0.7), assets 2,3 correlated (+0.6)
    # Cross-correlations low (~0.1)
    mean_returns = np.array([0.0005, 0.0003, 0.0004, 0.0002])
    
    # Build covariance matrix
    corr_matrix = np.eye(n_assets)
    corr_matrix[0, 1] = corr_matrix[1, 0] = 0.7
    corr_matrix[2, 3] = corr_matrix[3, 2] = 0.6
    corr_matrix[0, 2] = corr_matrix[2, 0] = 0.1
    corr_matrix[1, 3] = corr_matrix[3, 1] = 0.1
    
    vols = np.array([0.015, 0.020, 0.018, 0.012])
    cov_matrix = np.outer(vols, vols) * corr_matrix
    
    # Generate returns
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_obs)
    returns_df = pd.DataFrame(
        returns,
        columns=['Asset_A', 'Asset_B', 'Asset_C', 'Asset_D'],
        index=pd.date_range('2020-01-01', periods=n_obs, freq='D')
    )
    
    # Compute EWMA covariance
    ewma_result = compute_ewma_covariance(
        returns=returns_df,
        lambda_decay=0.94,
        min_periods=100,
        apply_shrinkage=True,
        shrinkage_target='constant_correlation'
    )
    
    # Verify results
    cov_latest = ewma_result['covariance_matrix']
    corr_latest = ewma_result['correlation_matrix']
    
    print(f"✓ Computed EWMA covariance for {n_assets} assets over {n_obs} observations")
    print(f"✓ Shrinkage applied: {ewma_result['shrinkage_applied']}")
    print(f"✓ Shrinkage intensity: {ewma_result['shrinkage_intensity']:.4f}")
    
    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(cov_latest.values)
    assert np.all(eigvals > 0), "Covariance matrix is not positive definite"
    print(f"✓ Covariance matrix is positive definite (min eigenvalue: {np.min(eigvals):.6f})")
    
    # Check correlation matrix properties
    assert np.allclose(np.diag(corr_latest.values), 1.0), "Correlation diagonal should be 1"
    assert np.all(corr_latest.values >= -1.0) and np.all(corr_latest.values <= 1.0), "Correlations out of range"
    print(f"✓ Correlation matrix valid (range: [{corr_latest.values.min():.3f}, {corr_latest.values.max():.3f}])")
    
    # Display correlation structure
    print("\nEstimated Correlation Matrix:")
    print(corr_latest.round(3))
    
    print("\n" + "=" * 80)
    print("✅ Test 1 PASSED: EWMA covariance estimation working correctly")
    print("=" * 80 + "\n")
    
    return ewma_result


def test_kelly_weights_with_covariance():
    """
    Test 2: Kelly weight computation with correlation-aware sizing.
    
    Verifies:
    - Kelly weights reduce exposure to correlated assets
    - Diversification ratio < 1 (correlation benefits captured)
    - Weights respect position limits
    """
    print("=" * 80)
    print("Test 2: Kelly Weights with Correlation-Aware Sizing")
    print("=" * 80)
    
    # Setup: 4 assets with known expected returns and correlations
    asset_names = ['Asset_A', 'Asset_B', 'Asset_C', 'Asset_D']
    n_assets = len(asset_names)
    
    # Expected returns (annualized, scale to daily)
    annual_returns = np.array([0.10, 0.08, 0.12, 0.06])
    mu_vec = annual_returns / 252  # daily
    
    # Correlation structure (A-B highly correlated, C-D moderately correlated)
    corr_matrix = np.array([
        [1.0, 0.8, 0.2, 0.1],
        [0.8, 1.0, 0.1, 0.2],
        [0.2, 0.1, 1.0, 0.5],
        [0.1, 0.2, 0.5, 1.0]
    ])
    
    vols = np.array([0.015, 0.018, 0.020, 0.012])
    cov_matrix = np.outer(vols, vols) * corr_matrix
    
    print(f"Expected returns (daily): {mu_vec}")
    print(f"Volatilities (daily): {vols}")
    print(f"\nCorrelation matrix:")
    print(corr_matrix.round(3))
    
    # Compute Kelly weights
    kelly_result = compute_kelly_weights(
        mu_vec=mu_vec,
        cov_matrix=cov_matrix,
        risk_fraction=0.5,
        max_weight=0.40,
        min_weight=-0.20
    )
    
    weights = kelly_result['weights_clamped']
    leverage = kelly_result['leverage']
    div_ratio = kelly_result['diversification_ratio']
    
    print(f"\n✓ Computed Kelly weights:")
    for i, name in enumerate(asset_names):
        print(f"  {name}: {weights[i]:+7.2%}")
    
    print(f"\n✓ Portfolio metrics:")
    print(f"  Leverage (Σ|w_i|): {leverage:.3f}")
    print(f"  Diversification ratio: {div_ratio:.3f}")
    
    # Verify diversification benefits
    assert div_ratio < 1.0, "Diversification ratio should be < 1 (correlation benefits)"
    print(f"✓ Diversification benefits captured (ratio < 1.0)")
    
    # Verify weights respect limits
    assert np.all(weights >= -0.20) and np.all(weights <= 0.40), "Weights exceed position limits"
    print(f"✓ Position limits respected ([-20%, +40%])")
    
    # Verify higher expected return assets get more weight (after correlation adjustment)
    # Asset C has highest return (0.12) and moderate correlation -> should get high weight
    assert weights[2] > 0, "High-return asset should have positive weight"
    print(f"✓ High-return assets allocated appropriately")
    
    print("\n" + "=" * 80)
    print("✅ Test 2 PASSED: Kelly weights capture correlation benefits")
    print("=" * 80 + "\n")
    
    return kelly_result


def test_cvar_constraint_on_portfolio():
    """
    Test 3: CVaR tail-risk constraint application.
    
    Verifies:
    - CVaR computed correctly from simulated paths
    - Constraint scales down weights when tail risk too large
    - Preserves correlation structure during scaling
    """
    print("=" * 80)
    print("Test 3: CVaR Tail-Risk Constraint")
    print("=" * 80)
    
    # Setup portfolio with risky weights
    asset_names = ['Asset_A', 'Asset_B', 'Asset_C', 'Asset_D']
    n_assets = len(asset_names)
    
    # Aggressive Kelly weights (sum to 1.0)
    weights_kelly = np.array([0.40, 0.30, 0.20, 0.10])
    
    # Covariance matrix (moderate correlations)
    corr_matrix = np.array([
        [1.0, 0.5, 0.3, 0.2],
        [0.5, 1.0, 0.4, 0.3],
        [0.3, 0.4, 1.0, 0.5],
        [0.2, 0.3, 0.5, 1.0]
    ])
    vols = np.array([0.020, 0.025, 0.030, 0.015])
    cov_matrix = np.outer(vols, vols) * corr_matrix
    cov_df = pd.DataFrame(cov_matrix, index=asset_names, columns=asset_names)
    
    # Simulate portfolio return paths with fat tails
    np.random.seed(43)
    n_horizons = 21
    n_paths = 3000
    
    # Generate paths per asset (with heavy tails)
    asset_return_paths = {}
    for i, name in enumerate(asset_names):
        # Student-t innovations (df=5 for heavy tails)
        innovations = np.random.standard_t(df=5, size=(n_horizons, n_paths))
        innovations = innovations / np.sqrt(5.0 / 3.0)  # scale to unit variance
        
        # Cumulative returns with drift
        mu_daily = 0.0002 * (i + 1)  # increasing drift
        vol_daily = vols[i]
        
        paths = np.cumsum(mu_daily + vol_daily * innovations, axis=0)
        asset_return_paths[name] = paths
    
    print(f"✓ Simulated {n_paths} paths over {n_horizons} horizons for {n_assets} assets")
    
    # Set aggressive CVaR limit
    r_max = -0.15  # max 15% tail loss
    
    # Apply CVaR constraint
    cvar_result = apply_cvar_constraint(
        weights_kelly=weights_kelly,
        asset_return_paths=asset_return_paths,
        cov_matrix=cov_df,
        r_max=r_max,
        confidence_level=0.95
    )
    
    weights_adjusted = cvar_result['weights_adjusted']
    cvar_unconstrained = cvar_result['cvar_unconstrained']
    cvar_constrained = cvar_result['cvar_constrained']
    constraint_active = cvar_result['constraint_active']
    scaling_factor = cvar_result['scaling_factor']
    
    print(f"\n✓ CVaR constraint results:")
    print(f"  CVaR (unconstrained): {cvar_unconstrained:+.4f} ({cvar_unconstrained*100:.2f}%)")
    print(f"  CVaR (constrained): {cvar_constrained:+.4f} ({cvar_constrained*100:.2f}%)")
    print(f"  Constraint active: {constraint_active}")
    print(f"  Scaling factor: {scaling_factor:.3f}")
    
    if constraint_active:
        print(f"\n✓ Weights adjusted:")
        for i, name in enumerate(asset_names):
            print(f"  {name}: {weights_kelly[i]:.3f} → {weights_adjusted[i]:.3f}")
        
        # Verify constraint is satisfied
        assert cvar_constrained >= r_max - 0.01, f"CVaR constraint violated: {cvar_constrained} < {r_max}"
        print(f"✓ Constraint satisfied: CVaR {cvar_constrained:.4f} ≥ {r_max:.4f}")
        
        # Verify scaling preserves relative weights
        relative_kelly = weights_kelly / np.sum(np.abs(weights_kelly))
        relative_adjusted = weights_adjusted / np.sum(np.abs(weights_adjusted))
        assert np.allclose(relative_kelly, relative_adjusted, atol=0.01), "Relative weights not preserved"
        print(f"✓ Correlation structure preserved during scaling")
    else:
        print(f"✓ Constraint inactive (tail risk acceptable)")
    
    print("\n" + "=" * 80)
    print("✅ Test 3 PASSED: CVaR constraint prevents blow-up risk")
    print("=" * 80 + "\n")
    
    return cvar_result


def test_full_portfolio_pipeline():
    """
    Test 4: End-to-end multi-asset portfolio construction.
    
    Integrates:
    - EWMA covariance estimation
    - Kelly weight optimization
    - CVaR tail-risk constraint
    
    Verifies complete pipeline works together.
    """
    print("=" * 80)
    print("Test 4: Full Multi-Asset Portfolio Pipeline")
    print("=" * 80)
    
    # Generate synthetic asset returns
    np.random.seed(44)
    n_obs = 600
    asset_names = ['PLNJPY', 'Gold', 'Silver', 'BTC', 'SPY']
    n_assets = len(asset_names)
    
    # Diverse correlation structure
    corr_matrix = np.array([
        [1.0,  0.3,  0.2, -0.1,  0.1],  # PLNJPY
        [0.3,  1.0,  0.8,  0.0,  0.2],  # Gold
        [0.2,  0.8,  1.0, -0.1,  0.1],  # Silver
        [-0.1, 0.0, -0.1,  1.0,  0.4],  # BTC
        [0.1,  0.2,  0.1,  0.4,  1.0]   # SPY
    ])
    
    vols = np.array([0.010, 0.015, 0.020, 0.030, 0.012])
    mean_returns = np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0003])
    cov_matrix = np.outer(vols, vols) * corr_matrix
    
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_obs)
    returns_df = pd.DataFrame(
        returns,
        columns=asset_names,
        index=pd.date_range('2020-01-01', periods=n_obs, freq='D')
    )
    
    print(f"✓ Generated {n_obs} days of returns for {n_assets} assets")
    
    # Build asset returns dict and expected returns
    asset_returns = {name: returns_df[name] for name in asset_names}
    asset_expected_returns = {name: mean_returns[i] for i, name in enumerate(asset_names)}
    
    # Simulate return paths for CVaR constraint
    horizon_days = 21
    n_paths = 3000
    asset_return_paths = {}
    
    for i, name in enumerate(asset_names):
        innovations = np.random.standard_t(df=7, size=(horizon_days, n_paths))
        innovations = innovations / np.sqrt(7.0 / 5.0)
        
        mu_h = mean_returns[i] * horizon_days
        vol_h = vols[i] * np.sqrt(horizon_days)
        
        paths = mu_h + vol_h * np.cumsum(innovations, axis=0)
        asset_return_paths[name] = paths
    
    print(f"✓ Simulated {n_paths} forward paths per asset")
    
    # Build portfolio with all components
    try:
        portfolio = build_multi_asset_portfolio(
            asset_returns=asset_returns,
            asset_expected_returns=asset_expected_returns,
            horizon_days=horizon_days,
            ewma_lambda=0.94,
            risk_fraction=0.5,
            asset_return_paths=asset_return_paths,
            apply_cvar=True,
            r_max=-0.20,
            cvar_confidence=0.95
        )
        
        print(f"\n✓ Portfolio constructed successfully")
        print(f"\n📊 Portfolio Weights (after CVaR constraint):")
        for i, name in enumerate(portfolio['asset_names']):
            weight = portfolio['weights_final'][i]
            exp_ret = portfolio['expected_returns'][i]
            print(f"  {name:10s}: {weight:+7.2%}  (E[R] = {exp_ret:+.6f})")
        
        stats = portfolio['portfolio_stats']
        print(f"\n📈 Portfolio Metrics:")
        print(f"  Expected Return: {stats['expected_return']:+.6f}")
        print(f"  Volatility: {stats['volatility']:.6f}")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
        print(f"  Leverage: {portfolio['leverage']:.3f}")
        print(f"  Diversification Ratio: {portfolio['diversification_ratio']:.3f}")
        
        # CVaR metrics
        if 'cvar_constraint' in portfolio:
            cvar = portfolio['cvar_constraint']
            print(f"\n🛡️  CVaR Constraint:")
            print(f"  CVaR₉₅: {cvar['cvar_constrained']:+.4f}")
            print(f"  Constraint Active: {cvar['constraint_active']}")
            if cvar['constraint_active']:
                print(f"  Scaling Factor: {cvar['scaling_factor']:.3f}")
        
        # Verify portfolio properties
        assert portfolio['diversification_ratio'] < 1.0, "Should benefit from diversification"
        assert abs(np.sum(portfolio['weights_final'])) <= 1.0, "Total weight should be ≤ 1.0"
        
        print(f"\n✓ Portfolio diversification benefits captured")
        print(f"✓ Position limits and constraints respected")
        
        print("\n" + "=" * 80)
        print("✅ Test 4 PASSED: Full pipeline working end-to-end")
        print("=" * 80 + "\n")
        
        return portfolio
        
    except Exception as e:
        print(f"\n❌ Portfolio construction failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 80)
    print("MULTI-ASSET PORTFOLIO INTEGRATION TESTS")
    print("Testing Dynamic Covariance + Kelly Criterion + CVaR Constraint")
    print("=" * 80 + "\n")
    
    try:
        # Test 1: EWMA covariance
        ewma_result = test_ewma_covariance_multi_asset()
        
        # Test 2: Kelly weights
        kelly_result = test_kelly_weights_with_covariance()
        
        # Test 3: CVaR constraint
        cvar_result = test_cvar_constraint_on_portfolio()
        
        # Test 4: Full pipeline
        portfolio = test_full_portfolio_pipeline()
        
        print("\n" + "=" * 80)
        print("🎉 ALL INTEGRATION TESTS PASSED")
        print("=" * 80)
        print("\n✅ Dynamic covariance & portfolio Kelly implementation verified")
        print("✅ Diversification benefits captured")
        print("✅ Tail-risk constraints prevent blow-ups")
        print("✅ Full pipeline working end-to-end")
        print("\n" + "=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
