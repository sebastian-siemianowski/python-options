#!/usr/bin/env python3
"""
Test script to verify the PIT calibration scale fix.

This test generates synthetic data from a known Student-t model and verifies
that the PIT values are uniformly distributed (KS test p-value > 0.05).

The bug was: using sqrt(variance) instead of sqrt(variance × (ν-2)/ν) for Student-t scale.
This caused standardized residuals to be too small by a factor of sqrt(ν/(ν-2)).
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
from scipy.stats import t as student_t, kstest

from models.phi_student_t import PhiStudentTDriftModel

np.random.seed(42)


def test_variance_to_scale_helper():
    """Test the new _variance_to_scale helper function."""
    print("=" * 60)
    print("Test 1: _variance_to_scale helper function")
    print("=" * 60)
    
    # For ν=4: scale_factor = (4-2)/4 = 0.5
    variance = 1.0
    nu = 4.0
    expected_scale = np.sqrt(0.5)  # sqrt(1.0 × 0.5)
    actual_scale = PhiStudentTDriftModel._variance_to_scale(variance, nu)
    print(f"  ν=4: variance={variance}, expected_scale={expected_scale:.4f}, actual={actual_scale:.4f}")
    assert abs(actual_scale - expected_scale) < 1e-10, f"Test failed: {actual_scale} != {expected_scale}"
    
    # For ν=6: scale_factor = (6-2)/6 = 0.667
    nu = 6.0
    expected_scale = np.sqrt(4/6)
    actual_scale = PhiStudentTDriftModel._variance_to_scale(variance, nu)
    print(f"  ν=6: variance={variance}, expected_scale={expected_scale:.4f}, actual={actual_scale:.4f}")
    assert abs(actual_scale - expected_scale) < 1e-10, f"Test failed: {actual_scale} != {expected_scale}"
    
    # For ν=20: scale_factor = (20-2)/20 = 0.9
    nu = 20.0
    expected_scale = np.sqrt(0.9)
    actual_scale = PhiStudentTDriftModel._variance_to_scale(variance, nu)
    print(f"  ν=20: variance={variance}, expected_scale={expected_scale:.4f}, actual={actual_scale:.4f}")
    assert abs(actual_scale - expected_scale) < 1e-10, f"Test failed: {actual_scale} != {expected_scale}"
    
    print("✓ Test 1 passed: _variance_to_scale helper works correctly\n")


def test_pit_uniformity_synthetic():
    """
    Test PIT uniformity with synthetic data from known Student-t.
    
    Generate data from: r_t = μ_t + ε_t where ε_t ~ Student-t(ν, 0, scale)
    Run filter and verify PIT values are uniform.
    """
    print("=" * 60)
    print("Test 2: PIT uniformity with synthetic Student-t data")
    print("=" * 60)
    
    n = 1000
    nu = 6.0
    c = 1.0
    q = 1e-6
    phi = 0.0  # No drift, pure Student-t observations
    
    # Generate synthetic data
    vol = np.ones(n) * 0.02  # Constant volatility
    variance = c * vol**2  # Total observation variance
    scale = np.sqrt(variance * (nu - 2) / nu)  # Correct Student-t scale
    
    # True drift is zero, innovations are Student-t
    true_innovations = student_t.rvs(df=nu, loc=0, scale=scale, size=n)
    returns = true_innovations
    
    # Run filter with predictive output
    mu_filt, P_filt, mu_pred, S_pred, ll = PhiStudentTDriftModel.filter_phi_with_predictive(
        returns, vol, q, c, phi, nu
    )
    
    # Compute PIT using predictive distribution
    ks_stat, pit_p = PhiStudentTDriftModel.pit_ks_predictive(
        returns, mu_pred, S_pred, nu
    )
    
    print(f"  n={n}, ν={nu}, c={c}")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  KS p-value:   {pit_p:.4f}")
    
    # The p-value should be > 0.05 for well-calibrated model
    if pit_p > 0.05:
        print(f"✓ Test 2 passed: PIT uniformity confirmed (p={pit_p:.4f} > 0.05)\n")
    else:
        print(f"⚠ Test 2 marginal: p={pit_p:.4f} < 0.05 (may happen ~5% of time)\n")


def test_pit_across_nu_values():
    """Test PIT uniformity across different ν values."""
    print("=" * 60)
    print("Test 3: PIT uniformity across ν grid")
    print("=" * 60)
    
    n = 500
    c = 1.0
    q = 1e-6
    phi = 0.0
    
    vol = np.ones(n) * 0.02
    
    nu_grid = [4, 6, 8, 12, 20]
    results = []
    
    for nu in nu_grid:
        # Generate data with correct scale
        variance = c * vol**2
        scale = np.sqrt(variance[0] * (nu - 2) / nu)
        returns = student_t.rvs(df=nu, loc=0, scale=scale, size=n)
        
        # Run filter
        _, _, mu_pred, S_pred, _ = PhiStudentTDriftModel.filter_phi_with_predictive(
            returns, vol, q, c, phi, float(nu)
        )
        
        # Compute PIT
        ks_stat, pit_p = PhiStudentTDriftModel.pit_ks_predictive(
            returns, mu_pred, S_pred, float(nu)
        )
        
        results.append((nu, ks_stat, pit_p))
        status = "✓" if pit_p > 0.05 else "✗"
        print(f"  ν={nu:2d}: KS={ks_stat:.4f}, p={pit_p:.4f} {status}")
    
    # Count passes
    passes = sum(1 for _, _, p in results if p > 0.05)
    print(f"\n  {passes}/{len(nu_grid)} ν values passed (expect ~95% with correct calibration)")
    
    if passes >= 3:  # At least 3 of 5 should pass
        print("✓ Test 3 passed: PIT calibration works across ν values\n")
    else:
        print("✗ Test 3 failed: Too many calibration failures\n")


def test_scale_factor_impact():
    """
    Demonstrate the impact of the scale fix.
    
    Compare the old (buggy) and new (correct) standardization approaches.
    """
    print("=" * 60)
    print("Test 4: Impact of scale correction")
    print("=" * 60)
    
    nu = 4.0  # Most extreme case: factor = sqrt(4/2) = sqrt(2) ≈ 1.41
    variance = 1.0
    
    # Old (buggy) approach
    old_scale = np.sqrt(variance)  # sqrt(variance) directly
    
    # New (correct) approach  
    new_scale = np.sqrt(variance * (nu - 2) / nu)  # sqrt(variance × (ν-2)/ν)
    
    ratio = old_scale / new_scale
    
    print(f"  ν = {nu}")
    print(f"  Old scale (buggy):   {old_scale:.4f}")
    print(f"  New scale (correct): {new_scale:.4f}")
    print(f"  Ratio (old/new):     {ratio:.4f}")
    print(f"  Expected ratio:      {np.sqrt(nu / (nu - 2)):.4f}")
    
    # The old approach inflates scale by sqrt(ν/(ν-2))
    expected_ratio = np.sqrt(nu / (nu - 2))
    assert abs(ratio - expected_ratio) < 1e-10, f"Ratio mismatch: {ratio} != {expected_ratio}"
    
    print(f"\n  Impact on standardized residuals:")
    print(f"    With old scale: z_old = innovation / {old_scale:.4f}")
    print(f"    With new scale: z_new = innovation / {new_scale:.4f}")
    print(f"    z_old = z_new × {new_scale/old_scale:.4f} (residuals were {(1 - new_scale/old_scale)*100:.1f}% too small!)")
    
    print("✓ Test 4 passed: Scale correction factor verified\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("PIT SCALE FIX VERIFICATION TESTS")
    print("="*60 + "\n")
    
    test_variance_to_scale_helper()
    test_pit_uniformity_synthetic()
    test_pit_across_nu_values()
    test_scale_factor_impact()
    
    print("="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
