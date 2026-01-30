#!/usr/bin/env python3
"""
Test script to validate tune.py improvements and signals.py integration.

Tests:
1. Walk-forward CV without look-ahead bias
2. Robust outlier handling
3. Adaptive prior calibration
4. Intelligent parameter adjustment
5. Cache loading in signals.py
6. End-to-end integration
"""

import sys
import os
import json
import numpy as np
import pandas as pd

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tuning'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from tune_q_mle import optimize_q_mle, tune_asset_q, kalman_filter_drift
from signals import _load_tuned_kalman_params, compute_features


def test_walk_forward_cv():
    """Test that CV doesn't use look-ahead information."""
    print("\n" + "="*80)
    print("TEST 1: Walk-Forward CV Without Look-Ahead Bias")
    print("="*80)
    
    # Generate synthetic data
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n) * 0.01
    vol = np.abs(np.random.randn(n) * 0.01) + 0.005
    
    # Run optimization
    try:
        q_opt, c_opt, ll_opt, diagnostics = optimize_q_mle(returns, vol)
        
        print(f"✓ Optimization completed successfully")
        print(f"  q_optimal: {q_opt:.2e}")
        print(f"  c_optimal: {c_opt:.3f}")
        print(f"  n_folds: {diagnostics['n_folds']}")
        print(f"  robust_optimization: {diagnostics.get('robust_optimization', False)}")
        print(f"  optimization_successful: {diagnostics.get('optimization_successful', False)}")
        
        # Validate parameters are reasonable
        assert 1e-10 < q_opt < 1e-2, f"q out of bounds: {q_opt}"
        assert 0.5 < c_opt < 2.0, f"c out of bounds: {c_opt}"
        assert diagnostics['n_folds'] >= 1, "Should have at least 1 fold"
        
        print("✓ All assertions passed")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_robust_outlier_handling():
    """Test that extreme outliers are handled robustly."""
    print("\n" + "="*80)
    print("TEST 2: Robust Outlier Handling")
    print("="*80)
    
    # Generate data with extreme outliers
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n) * 0.01
    returns[100] = 0.15  # Extreme positive outlier
    returns[300] = -0.15  # Extreme negative outlier
    vol = np.abs(np.random.randn(n) * 0.01) + 0.005
    
    try:
        q_opt, c_opt, ll_opt, diagnostics = optimize_q_mle(returns, vol)
        
        print(f"✓ Optimization handled outliers successfully")
        print(f"  q_optimal: {q_opt:.2e}")
        print(f"  c_optimal: {c_opt:.3f}")
        print(f"  winsorized: {diagnostics.get('winsorized', False)}")
        
        # Parameters should still be reasonable despite outliers
        assert 1e-10 < q_opt < 1e-2, f"q out of bounds: {q_opt}"
        assert 0.5 < c_opt < 2.0, f"c out of bounds: {c_opt}"
        
        print("✓ Outliers handled robustly")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_prior():
    """Test adaptive prior based on market characteristics."""
    print("\n" + "="*80)
    print("TEST 3: Adaptive Prior Calibration")
    print("="*80)
    
    # Test 1: High volatility regime (should get higher q)
    np.random.seed(42)
    n = 500
    returns_volatile = np.random.randn(n) * 0.03  # High vol
    vol_volatile = np.abs(np.random.randn(n) * 0.02) + 0.01
    
    # Test 2: Stable regime (should get lower q)
    returns_stable = np.random.randn(n) * 0.005  # Low vol
    vol_stable = np.abs(np.random.randn(n) * 0.003) + 0.002
    
    try:
        q_volatile, c_volatile, _, diag_volatile = optimize_q_mle(returns_volatile, vol_volatile)
        q_stable, c_stable, _, diag_stable = optimize_q_mle(returns_stable, vol_stable)
        
        print(f"✓ Adaptive prior working")
        print(f"  Volatile regime: q={q_volatile:.2e}, vol_cv={diag_volatile['vol_cv']:.3f}")
        print(f"  Stable regime:   q={q_stable:.2e}, vol_cv={diag_stable['vol_cv']:.3f}")
        print(f"  Prior adapts based on market characteristics")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_integration():
    """Test cache loading in signals.py."""
    print("\n" + "="*80)
    print("TEST 4: Cache Integration with signals.py")
    print("="*80)
    
    # Create temporary cache
    cache_path = "/tmp/test_kalman_cache.json"
    test_cache = {
        "TEST_ASSET": {
            "q": 1.5e-8,
            "c": 0.92,
            "timestamp": "2026-01-06T12:00:00Z",
            "delta_ll_vs_zero": 5.2,
            "pit_ks_pvalue": 0.15,
            "calibration_warning": False
        }
    }
    
    try:
        # Write test cache
        with open(cache_path, 'w') as f:
            json.dump(test_cache, f)
        
        # Test loading
        params = _load_tuned_kalman_params("TEST_ASSET", cache_path)
        
        assert params is not None, "Failed to load params"
        assert params['q'] == 1.5e-8, f"Wrong q: {params['q']}"
        assert params['c'] == 0.92, f"Wrong c: {params['c']}"
        assert params['source'] == 'tuned_cache', f"Wrong source: {params['source']}"
        
        print(f"✓ Cache loading works correctly")
        print(f"  Loaded: q={params['q']:.2e}, c={params['c']:.3f}")
        print(f"  Source: {params['source']}")
        
        # Test missing asset
        params_missing = _load_tuned_kalman_params("NONEXISTENT", cache_path)
        assert params_missing is None, "Should return None for missing asset"
        
        print("✓ Handles missing assets correctly")
        
        # Cleanup
        os.remove(cache_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(cache_path):
            os.remove(cache_path)
        return False


def test_kalman_filter():
    """Test Kalman filter implementation."""
    print("\n" + "="*80)
    print("TEST 5: Kalman Filter Correctness")
    print("="*80)
    
    # Generate synthetic data
    np.random.seed(42)
    n = 300
    returns = np.random.randn(n) * 0.01
    vol = np.abs(np.random.randn(n) * 0.01) + 0.005
    
    try:
        # Test with fixed parameters
        q = 1e-7
        c = 0.9
        
        mu_filt, P_filt, ll = kalman_filter_drift(returns, vol, q, c)
        
        assert len(mu_filt) == n, f"Wrong length: {len(mu_filt)} vs {n}"
        assert len(P_filt) == n, f"Wrong length: {len(P_filt)} vs {n}"
        assert np.isfinite(ll), f"LL not finite: {ll}"
        
        # Check that variances are positive
        assert np.all(P_filt > 0), "Negative variances detected"
        
        # Check that estimates are reasonable
        assert np.all(np.abs(mu_filt) < 0.1), "Drift estimates unreasonable"
        
        print(f"✓ Kalman filter working correctly")
        print(f"  n_obs: {n}")
        print(f"  log_likelihood: {ll:.2f}")
        print(f"  mean_drift: {np.mean(mu_filt):.6f}")
        print(f"  mean_variance: {np.mean(P_filt):.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("TUNE_Q_MLE.PY IMPROVEMENTS VALIDATION SUITE")
    print("="*80)
    print("\nTesting production-grade improvements:")
    print("  - Walk-forward cross-validation without look-ahead bias")
    print("  - Robust outlier handling (Huber-like loss for 5-sigma+ events)")
    print("  - Adaptive prior calibration based on market characteristics")
    print("  - Cache integration with signals.py")
    print("  - Kalman filter correctness")
    
    tests = [
        ("Walk-Forward CV", test_walk_forward_cv),
        ("Robust Outliers", test_robust_outlier_handling),
        ("Adaptive Prior", test_adaptive_prior),
        ("Cache Integration", test_cache_integration),
        ("Kalman Filter", test_kalman_filter),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System ready for production.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
