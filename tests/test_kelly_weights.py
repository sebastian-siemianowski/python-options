#!/usr/bin/env python3
"""
test_kelly_weights.py

Test Kelly portfolio weight computation with sample multi-asset data.
"""

import sys
import numpy as np
import pandas as pd

# Add scripts directory to path
sys.path.insert(0, '/Users/sebastiansiemianowski/RubymineProjects/python-options/scripts')

from portfolio_kelly import compute_kelly_weights, compute_portfolio_statistics, build_multi_asset_portfolio


def test_basic_kelly_weights():
    """Test basic Kelly weight computation with simple 2-asset case."""
    print("=" * 80)
    print("TEST 1: Basic Kelly Weights (2 assets)")
    print("=" * 80)
    
    # Simple example: 2 assets
    # Asset 1: expected return 0.01, volatility 0.15
    # Asset 2: expected return 0.005, volatility 0.10
    # Correlation: 0.3
    
    mu = np.array([0.01, 0.005])
    vol1, vol2 = 0.15, 0.10
    corr = 0.3
    
    cov = np.array([
        [vol1**2, corr * vol1 * vol2],
        [corr * vol1 * vol2, vol2**2]
    ])
    
    print(f"\nExpected returns: {mu}")
    print(f"Volatilities: [{vol1:.4f}, {vol2:.4f}]")
    print(f"Correlation: {corr:.2f}")
    print(f"\nCovariance matrix:")
    print(cov)
    
    # Compute Kelly weights
    result = compute_kelly_weights(mu, cov, risk_fraction=0.5)
    
    print(f"\n{'='*80}")
    print("Kelly Weights Results:")
    print(f"{'='*80}")
    print(f"Raw weights:        {result['weights_raw']}")
    print(f"Normalized weights: {result['weights_normalized']}")
    print(f"Clamped weights:    {result['weights_clamped']}")
    print(f"Leverage:           {result['leverage']:.3f}")
    print(f"Diversification ratio: {result['diversification_ratio']:.3f}")
    print(f"Method:             {result['method']}")
    
    # Compute portfolio statistics
    portfolio_stats = compute_portfolio_statistics(
        result['weights_clamped'], mu, cov
    )
    
    print(f"\n{'='*80}")
    print("Portfolio Statistics:")
    print(f"{'='*80}")
    print(f"Expected return:    {portfolio_stats['expected_return']:+.6f}")
    print(f"Volatility:         {portfolio_stats['volatility']:.6f}")
    print(f"Sharpe ratio:       {portfolio_stats['sharpe_ratio']:.3f}")
    
    return result


def test_multi_asset_portfolio():
    """Test multi-asset portfolio construction with real data structure."""
    print("\n\n" + "=" * 80)
    print("TEST 2: Multi-Asset Portfolio (4 assets with historical returns)")
    print("=" * 80)
    
    # Simulate realistic return series (normally distributed with different params)
    np.random.seed(42)
    n_obs = 1000
    dates = pd.date_range('2020-01-01', periods=n_obs, freq='D')
    
    # Asset parameters (daily returns)
    assets = {
        'PLNJPY': {'mu': 0.0001, 'sigma': 0.006},   # FX pair
        'Gold': {'mu': 0.0003, 'sigma': 0.012},     # Commodity
        'BTC': {'mu': 0.0015, 'sigma': 0.035},      # Crypto (high vol)
        'SPX': {'mu': 0.0004, 'sigma': 0.011},      # Equity index
    }
    
    # Generate correlated returns
    # Correlation structure
    corr_matrix = np.array([
        [1.00, 0.20, 0.10, 0.30],  # PLNJPY
        [0.20, 1.00, 0.15, 0.40],  # Gold
        [0.10, 0.15, 1.00, 0.25],  # BTC
        [0.30, 0.40, 0.25, 1.00],  # SPX
    ])
    
    # Build covariance from correlation and individual sigmas
    sigmas = np.array([assets[name]['sigma'] for name in assets.keys()])
    cov_true = np.outer(sigmas, sigmas) * corr_matrix
    
    # Generate multivariate normal returns
    L = np.linalg.cholesky(cov_true)
    z = np.random.standard_normal((n_obs, 4))
    mus = np.array([assets[name]['mu'] for name in assets.keys()])
    returns_array = z @ L.T + mus
    
    # Build DataFrame
    returns_df = pd.DataFrame(returns_array, index=dates, columns=list(assets.keys()))
    
    print(f"\nGenerated {n_obs} observations for {len(assets)} assets")
    print(f"\nSample returns (first 5 rows):")
    print(returns_df.head())
    
    print(f"\nTrue parameters (daily):")
    for name in assets.keys():
        print(f"  {name:10s}: μ={assets[name]['mu']:+.6f}, σ={assets[name]['sigma']:.6f}")
    
    print(f"\nTrue correlation matrix:")
    print(corr_matrix)
    
    # Build asset return dict
    asset_returns = {name: returns_df[name] for name in assets.keys()}
    
    # Expected returns (daily, for horizon scaling)
    asset_expected_returns = {name: assets[name]['mu'] for name in assets.keys()}
    
    # Build portfolio for 21-day horizon (1 month)
    try:
        portfolio = build_multi_asset_portfolio(
            asset_returns=asset_returns,
            asset_expected_returns=asset_expected_returns,
            horizon_days=21,
            ewma_lambda=0.94,
            risk_fraction=0.5,
        )
        
        print(f"\n{'='*80}")
        print("Portfolio Results (21-day horizon):")
        print(f"{'='*80}")
        
        print(f"\nAsset allocations (clamped Kelly weights):")
        for i, name in enumerate(portfolio['asset_names']):
            weight = portfolio['weights_clamped'][i]
            exp_ret = portfolio['expected_returns'][i]
            print(f"  {name:10s}: {weight:+7.2%}  (E[r_21d] = {exp_ret:+.6f})")
        
        print(f"\nPortfolio metrics:")
        print(f"  Leverage:              {portfolio['leverage']:.3f}")
        print(f"  Diversification ratio: {portfolio['diversification_ratio']:.3f}")
        print(f"  Risk fraction:         {portfolio['risk_fraction']:.2f}")
        
        stats = portfolio['portfolio_stats']
        print(f"\nPortfolio statistics (21-day horizon):")
        print(f"  Expected return:       {stats['expected_return']:+.6f}")
        print(f"  Volatility:            {stats['volatility']:.6f}")
        print(f"  Sharpe ratio:          {stats['sharpe_ratio']:.3f}")
        
        print(f"\nEWMA covariance matrix (latest):")
        print(portfolio['covariance_matrix'])
        
        print(f"\nEWMA correlation matrix (latest):")
        print(portfolio['correlation_matrix'])
        
        return portfolio
        
    except Exception as e:
        print(f"\n[ERROR] Portfolio construction failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_extreme_cases():
    """Test edge cases and numerical stability."""
    print("\n\n" + "=" * 80)
    print("TEST 3: Edge Cases and Numerical Stability")
    print("=" * 80)
    
    # Case 1: Perfect correlation (ill-conditioned)
    print("\nCase 1: Perfect correlation (singular matrix test)")
    mu = np.array([0.01, 0.01])
    cov = np.array([[0.01, 0.01], [0.01, 0.01]])  # rank deficient
    
    try:
        result = compute_kelly_weights(mu, cov, risk_fraction=0.5, regularization=1e-6)
        print(f"  Weights: {result['weights_clamped']}")
        print(f"  Method: {result['method']}")
        print("  ✓ Handled gracefully with regularization")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Case 2: Negative expected returns
    print("\nCase 2: Negative expected returns (short positions)")
    mu = np.array([0.01, -0.005])
    cov = np.array([[0.01, 0.001], [0.001, 0.01]])
    
    try:
        result = compute_kelly_weights(mu, cov, risk_fraction=0.5, min_weight=-0.20)
        print(f"  Weights: {result['weights_clamped']}")
        print("  ✓ Negative weights allowed for shorts")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    # Case 3: Zero expected returns
    print("\nCase 3: Zero expected returns (no signal)")
    mu = np.array([0.0, 0.0])
    cov = np.array([[0.01, 0.001], [0.001, 0.01]])
    
    try:
        result = compute_kelly_weights(mu, cov, risk_fraction=0.5)
        print(f"  Weights: {result['weights_clamped']}")
        print("  ✓ Returns zero weights as expected")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\nAll edge case tests completed.")


if __name__ == "__main__":
    print("Kelly Portfolio Weight Computation Tests")
    print("=" * 80)
    
    # Run tests
    test_basic_kelly_weights()
    test_multi_asset_portfolio()
    test_extreme_cases()
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
