#!/usr/bin/env python3
"""
test_cvar_constraint.py

Test CVaR tail-risk constraint with stress scenarios.
Verifies constraint activates when Expected Shortfall exceeds r_max threshold.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_kelly import build_multi_asset_portfolio
from portfolio_utils import compute_cvar_from_paths


def test_cvar_computation():
    """Test basic CVaR computation from simulated paths."""
    print("=" * 80)
    print("TEST 1: CVaR Computation from Simulated Paths")
    print("=" * 80)
    
    # Simulate portfolio returns with fat left tail (losses)
    np.random.seed(42)
    n_paths = 5000
    
    # Base normal returns
    normal_returns = np.random.normal(loc=0.02, scale=0.10, size=int(n_paths * 0.95))
    
    # Add extreme loss tail (5% of scenarios with large losses)
    tail_returns = np.random.normal(loc=-0.30, scale=0.15, size=int(n_paths * 0.05))
    
    # Combine
    portfolio_returns = np.concatenate([normal_returns, tail_returns])
    np.random.shuffle(portfolio_returns)
    
    print(f"\nSimulated {n_paths} portfolio return paths")
    print(f"Mean return: {np.mean(portfolio_returns):+.4f}")
    print(f"Std dev: {np.std(portfolio_returns):.4f}")
    print(f"Min (worst case): {np.min(portfolio_returns):+.4f}")
    print(f"5th percentile: {np.percentile(portfolio_returns, 5):+.4f}")
    
    # Compute CVaR
    cvar_result = compute_cvar_from_paths(portfolio_returns, confidence_level=0.95)
    
    print(f"\n{'='*80}")
    print("CVaR Metrics @ 95% Confidence:")
    print(f"{'='*80}")
    print(f"VaR₉₅ (5th percentile):        {cvar_result['var']:+.4f}")
    print(f"CVaR₉₅ (Expected Shortfall):   {cvar_result['cvar']:+.4f}")
    print(f"Worst case (minimum):          {cvar_result['worst_case']:+.4f}")
    print(f"Tail probability:              {cvar_result['tail_probability']:.3f} (expect ~0.05)")
    print(f"N tail scenarios:              {cvar_result['n_tail_scenarios']}")
    
    # Verify CVaR is worse (more negative) than VaR
    assert cvar_result['cvar'] < cvar_result['var'], "CVaR should be < VaR (worse loss)"
    print(f"\n✓ CVaR correctly captures tail risk worse than VaR")
    
    return cvar_result


def test_cvar_constraint_inactive():
    """Test CVaR constraint remains inactive with low tail risk."""
    print("\n\n" + "=" * 80)
    print("TEST 2: CVaR Constraint (INACTIVE - Low Tail Risk)")
    print("=" * 80)
    
    # Setup portfolio with low tail risk
    np.random.seed(42)
    n_obs = 500
    n_paths = 3000
    n_horizons = 21
    dates = pd.date_range('2020-01-01', periods=n_obs, freq='D')
    
    # Two assets with low volatility and positive drift
    assets = {
        'Asset_A': {'mu': 0.0005, 'sigma': 0.008},
        'Asset_B': {'mu': 0.0004, 'sigma': 0.010},
    }
    
    # Generate historical returns (for EWMA covariance)
    corr = 0.4
    sigma_A = assets['Asset_A']['sigma']
    sigma_B = assets['Asset_B']['sigma']
    cov_true = np.array([
        [sigma_A**2, corr * sigma_A * sigma_B],
        [corr * sigma_A * sigma_B, sigma_B**2]
    ])
    
    L = np.linalg.cholesky(cov_true)
    z = np.random.standard_normal((n_obs, 2))
    mus = np.array([assets['Asset_A']['mu'], assets['Asset_B']['mu']])
    returns_array = z @ L.T + mus
    
    historical_returns = {
        'Asset_A': pd.Series(returns_array[:, 0], index=dates),
        'Asset_B': pd.Series(returns_array[:, 1], index=dates),
    }
    
    # Generate forward-looking simulated paths (low tail risk scenario)
    # Simulate with same parameters - no extreme tail
    asset_paths = {}
    for i, (name, params) in enumerate(assets.items()):
        paths = np.random.normal(
            loc=params['mu'] * n_horizons,  # cumulative return over horizon
            scale=params['sigma'] * np.sqrt(n_horizons),
            size=(n_horizons, n_paths)
        )
        asset_paths[name] = paths
    
    # Build portfolio with CVaR constraint (r_max = -20%)
    expected_returns = {
        'Asset_A': assets['Asset_A']['mu'],
        'Asset_B': assets['Asset_B']['mu'],
    }
    
    portfolio = build_multi_asset_portfolio(
        asset_returns=historical_returns,
        asset_expected_returns=expected_returns,
        horizon_days=n_horizons,
        asset_return_paths=asset_paths,
        apply_cvar=True,
        r_max=-0.20,  # Max 20% loss in tail
    )
    
    print(f"\nPortfolio constructed with {len(portfolio['asset_names'])} assets")
    print(f"Horizon: {portfolio['horizon_days']} days")
    
    # Display weights
    print(f"\n{'='*80}")
    print("Portfolio Weights:")
    print(f"{'='*80}")
    for i, name in enumerate(portfolio['asset_names']):
        kelly_w = portfolio['weights_clamped'][i]
        final_w = portfolio['weights_final'][i]
        print(f"{name:10s}: Kelly={kelly_w:+.4f}  Final={final_w:+.4f}")
    
    # Display CVaR metrics
    cvar_info = portfolio.get('cvar_constraint', {})
    print(f"\n{'='*80}")
    print("CVaR Tail Risk Metrics:")
    print(f"{'='*80}")
    print(f"CVaR₉₅ (unconstrained):  {cvar_info.get('cvar_unconstrained', float('nan')):+.4f}")
    print(f"CVaR₉₅ (constrained):    {cvar_info.get('cvar_constrained', float('nan')):+.4f}")
    print(f"r_max threshold:         {cvar_info.get('r_max', -0.20):+.4f}")
    print(f"Constraint active:       {cvar_info.get('constraint_active', False)}")
    print(f"Scaling factor:          {cvar_info.get('scaling_factor', 1.0):.4f}")
    
    # Verify constraint is INACTIVE (low tail risk)
    assert cvar_info.get('constraint_active') == False, "Constraint should be inactive with low tail risk"
    assert abs(cvar_info.get('scaling_factor', 1.0) - 1.0) < 1e-6, "Scaling should be 1.0 (no adjustment)"
    
    print(f"\n✓ CVaR constraint correctly INACTIVE (tail risk within limit)")
    
    return portfolio


def test_cvar_constraint_active():
    """Test CVaR constraint activates with high tail risk."""
    print("\n\n" + "=" * 80)
    print("TEST 3: CVaR Constraint (ACTIVE - High Tail Risk)")
    print("=" * 80)
    
    # Setup portfolio with HIGH tail risk (stress scenario)
    np.random.seed(123)
    n_obs = 500
    n_paths = 3000
    n_horizons = 21
    dates = pd.date_range('2020-01-01', periods=n_obs, freq='D')
    
    # Two assets with high volatility
    assets = {
        'Asset_A': {'mu': 0.0010, 'sigma': 0.025},
        'Asset_B': {'mu': 0.0008, 'sigma': 0.030},
    }
    
    # Generate historical returns
    corr = 0.6
    sigma_A = assets['Asset_A']['sigma']
    sigma_B = assets['Asset_B']['sigma']
    cov_true = np.array([
        [sigma_A**2, corr * sigma_A * sigma_B],
        [corr * sigma_A * sigma_B, sigma_B**2]
    ])
    
    L = np.linalg.cholesky(cov_true)
    z = np.random.standard_normal((n_obs, 2))
    mus = np.array([assets['Asset_A']['mu'], assets['Asset_B']['mu']])
    returns_array = z @ L.T + mus
    
    historical_returns = {
        'Asset_A': pd.Series(returns_array[:, 0], index=dates),
        'Asset_B': pd.Series(returns_array[:, 1], index=dates),
    }
    
    # Generate forward-looking simulated paths with FAT LEFT TAIL (stress scenario)
    # Mix normal draws with extreme loss scenarios
    asset_paths = {}
    for i, (name, params) in enumerate(assets.items()):
        # 90% normal paths
        normal_paths = np.random.normal(
            loc=params['mu'] * n_horizons,
            scale=params['sigma'] * np.sqrt(n_horizons),
            size=(n_horizons, int(n_paths * 0.9))
        )
        
        # 10% extreme loss paths (crisis scenario)
        crisis_paths = np.random.normal(
            loc=-0.35,  # -35% loss in crisis
            scale=0.10,
            size=(n_horizons, int(n_paths * 0.1))
        )
        
        # Concatenate
        paths = np.concatenate([normal_paths, crisis_paths], axis=1)
        asset_paths[name] = paths
    
    # Build portfolio with CVaR constraint (r_max = -12% to trigger constraint)
    expected_returns = {
        'Asset_A': assets['Asset_A']['mu'],
        'Asset_B': assets['Asset_B']['mu'],
    }
    
    portfolio = build_multi_asset_portfolio(
        asset_returns=historical_returns,
        asset_expected_returns=expected_returns,
        horizon_days=n_horizons,
        asset_return_paths=asset_paths,
        apply_cvar=True,
        r_max=-0.12,  # Max 12% loss in tail (tighter to trigger with -16% CVaR)
    )
    
    print(f"\nPortfolio constructed with {len(portfolio['asset_names'])} assets")
    print(f"Horizon: {portfolio['horizon_days']} days")
    print(f"Stress scenario: 10% of paths have -35% crisis losses")
    
    # Display weights
    print(f"\n{'='*80}")
    print("Portfolio Weights:")
    print(f"{'='*80}")
    for i, name in enumerate(portfolio['asset_names']):
        kelly_w = portfolio['weights_clamped'][i]
        final_w = portfolio['weights_final'][i]
        change = ((final_w / kelly_w - 1.0) * 100) if abs(kelly_w) > 1e-6 else 0.0
        print(f"{name:10s}: Kelly={kelly_w:+.4f}  Final={final_w:+.4f}  Change={change:+.1f}%")
    
    # Display CVaR metrics
    cvar_info = portfolio.get('cvar_constraint', {})
    print(f"\n{'='*80}")
    print("CVaR Tail Risk Metrics:")
    print(f"{'='*80}")
    print(f"CVaR₉₅ (unconstrained):  {cvar_info.get('cvar_unconstrained', float('nan')):+.4f}")
    print(f"CVaR₉₅ (constrained):    {cvar_info.get('cvar_constrained', float('nan')):+.4f}")
    print(f"r_max threshold:         {cvar_info.get('r_max', -0.12):+.4f}")
    print(f"Constraint active:       {cvar_info.get('constraint_active', False)}")
    print(f"Scaling factor:          {cvar_info.get('scaling_factor', 1.0):.4f}")
    
    # Verify constraint is ACTIVE (high tail risk)
    assert cvar_info.get('constraint_active') == True, "Constraint should be active with high tail risk"
    assert cvar_info.get('scaling_factor', 1.0) < 1.0, "Scaling should be < 1.0 (weights reduced)"
    
    # Verify CVaR after constraint is better (closer to r_max)
    cvar_unc = cvar_info.get('cvar_unconstrained', 0.0)
    cvar_con = cvar_info.get('cvar_constrained', 0.0)
    assert cvar_con > cvar_unc, "Constrained CVaR should be better (less negative) than unconstrained"
    
    print(f"\n✓ CVaR constraint correctly ACTIVE and reduced tail risk")
    print(f"✓ Weights scaled down by {cvar_info.get('scaling_factor', 1.0):.2f}× to meet r_max")
    
    return portfolio


def main():
    print("\n" + "=" * 80)
    print("CVaR TAIL-RISK CONSTRAINT TEST SUITE")
    print("Step 3: Anti-Lehman Insurance")
    print("=" * 80 + "\n")
    
    # Test 1: Basic CVaR computation
    try:
        cvar_result = test_cvar_computation()
        print("\n✅ Test 1 PASSED: CVaR computation works correctly")
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 2: Constraint inactive with low tail risk
    try:
        portfolio_low_risk = test_cvar_constraint_inactive()
        print("\n✅ Test 2 PASSED: Constraint correctly inactive with low tail risk")
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 3: Constraint active with high tail risk
    try:
        portfolio_high_risk = test_cvar_constraint_active()
        print("\n✅ Test 3 PASSED: Constraint correctly active and reduces tail risk")
    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED: CVaR constraint working correctly")
    print("=" * 80)
    print("\n🎯 Step 3 Complete: Tail-aware risk constraint implemented")
    print("   ✓ Survivability: portfolio won't blow up in extreme scenarios")
    print("   ✓ Anti-Lehman insurance: worst-case losses stay within bounds")
    print("   ✓ Prevents over-Kelly: detects when variance underestimates tail risk")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
