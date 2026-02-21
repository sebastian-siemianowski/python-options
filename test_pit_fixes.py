"""
Test the PIT calibration fixes implemented in February 2026.

This module tests:
1. Sample-size-aware PIT threshold computation
2. Predictive value reconstruction from filtered values
3. Proper PIT computation for momentum models
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from tuning.pit_calibration import (
    compute_pit_calibration_metrics,
    sample_size_adjusted_pit_threshold,
    is_pit_calibrated,
)


def test_sample_size_adjusted_threshold():
    """Test that threshold scales correctly with sample size."""
    print("Testing sample_size_adjusted_pit_threshold...")
    
    # Small sample: should be 0.05
    assert sample_size_adjusted_pit_threshold(500) == 0.05
    assert sample_size_adjusted_pit_threshold(1000) == 0.05
    
    # Large sample: should be higher
    threshold_5000 = sample_size_adjusted_pit_threshold(5000)
    assert threshold_5000 > 0.05, f"Expected > 0.05, got {threshold_5000}"
    assert threshold_5000 < 0.15, f"Expected < 0.15, got {threshold_5000}"
    
    # Very large sample
    threshold_10000 = sample_size_adjusted_pit_threshold(10000)
    assert threshold_10000 > threshold_5000, f"Should increase with n"
    
    print(f"  n=500: {sample_size_adjusted_pit_threshold(500):.4f}")
    print(f"  n=1000: {sample_size_adjusted_pit_threshold(1000):.4f}")
    print(f"  n=5000: {threshold_5000:.4f}")
    print(f"  n=10000: {threshold_10000:.4f}")
    print("  ✓ sample_size_adjusted_pit_threshold works correctly")


def test_is_pit_calibrated():
    """Test calibration check with sample size adjustment."""
    print("\nTesting is_pit_calibrated...")
    
    # Strict mode (standard 0.05)
    assert is_pit_calibrated(0.06, n_samples=5000, strict=True) == True
    assert is_pit_calibrated(0.04, n_samples=5000, strict=True) == False
    
    # Non-strict mode (adjusted threshold)
    # With 5000 samples, threshold is ~0.0675
    assert is_pit_calibrated(0.07, n_samples=5000, strict=False) == True  # Above threshold
    assert is_pit_calibrated(0.06, n_samples=5000, strict=False) == False  # Below threshold
    assert is_pit_calibrated(0.03, n_samples=5000, strict=False) == False
    
    print("  ✓ is_pit_calibrated works correctly")


def test_pit_calibration_metrics_perfect():
    """Test PIT metrics with perfect uniform distribution."""
    print("\nTesting compute_pit_calibration_metrics with uniform PIT...")
    
    np.random.seed(42)
    uniform_pit = np.random.uniform(0, 1, 1000)
    
    metrics = compute_pit_calibration_metrics(uniform_pit)
    
    print(f"  KS statistic: {metrics['ks_statistic']:.4f}")
    print(f"  KS p-value: {metrics['ks_pvalue']:.4f}")
    print(f"  Mean deviation: {metrics['mean_deviation']:.4f}")
    print(f"  Calibration score: {metrics['calibration_score']:.4f}")
    print(f"  Practical calibration: {metrics['practical_calibration']}")
    
    # Perfect uniform should have high p-value (usually)
    assert metrics['mean_deviation'] < 0.05, f"Expected MAD < 0.05, got {metrics['mean_deviation']}"
    assert metrics['practical_calibration'] == True
    print("  ✓ Perfect uniform passes calibration check")


def test_pit_calibration_metrics_biased():
    """Test PIT metrics with biased (concentrated) PIT values."""
    print("\nTesting compute_pit_calibration_metrics with biased PIT...")
    
    np.random.seed(42)
    # PIT concentrated in middle (classic sign of over-confident model)
    # Use 0.2-0.8 range for clearer miscalibration
    biased_pit = np.random.uniform(0.2, 0.8, 1000)
    
    metrics = compute_pit_calibration_metrics(biased_pit)
    
    print(f"  KS statistic: {metrics['ks_statistic']:.4f}")
    print(f"  KS p-value: {metrics['ks_pvalue']:.4f}")
    print(f"  Mean deviation: {metrics['mean_deviation']:.4f}")
    print(f"  Calibration score: {metrics['calibration_score']:.4f}")
    print(f"  Practical calibration: {metrics['practical_calibration']}")
    
    # Biased should fail calibration
    assert metrics['ks_pvalue'] < 0.05, f"Expected p < 0.05, got {metrics['ks_pvalue']}"
    assert metrics['mean_deviation'] > 0.05, f"Expected MAD > 0.05, got {metrics['mean_deviation']}"
    print("  ✓ Biased PIT correctly detected as miscalibrated")


def test_predictive_reconstruction():
    """Test that we can reconstruct predictive values from filtered values."""
    print("\nTesting predictive reconstruction from filtered values...")
    
    from tuning.tune import (
        reconstruct_predictive_from_filtered_gaussian,
        compute_pit_from_filtered_gaussian,
    )
    from models.gaussian import GaussianDriftModel
    
    np.random.seed(42)
    n = 500
    
    # Generate synthetic data
    returns = np.random.randn(n) * 0.02
    vol = np.abs(np.random.randn(n) * 0.01) + 0.01
    q = 1e-6
    c = 1.0
    phi = 0.95
    
    # Get ground truth predictive values
    mu_filt, P_filt, mu_pred_true, S_pred_true, ll = GaussianDriftModel.filter_phi_with_predictive(
        returns, vol, q, c, phi
    )
    
    # Reconstruct predictive values from filtered
    mu_pred_recon, S_pred_recon = reconstruct_predictive_from_filtered_gaussian(
        returns, mu_filt, P_filt, vol, q, c, phi
    )
    
    # Check that reconstruction is close (won't be exact due to different initial conditions)
    mu_diff = np.abs(mu_pred_recon[10:] - mu_pred_true[10:]).mean()
    S_diff = np.abs(S_pred_recon[10:] - S_pred_true[10:]).mean()
    
    print(f"  Mean |mu_pred_recon - mu_pred_true|: {mu_diff:.8f}")
    print(f"  Mean |S_pred_recon - S_pred_true|: {S_diff:.8f}")
    
    # Should be very close after burn-in
    assert mu_diff < 1e-6, f"Mu reconstruction error too large: {mu_diff}"
    assert S_diff < 1e-6, f"S reconstruction error too large: {S_diff}"
    
    # Test PIT computation
    ks, pit_p = compute_pit_from_filtered_gaussian(
        returns, mu_filt, P_filt, vol, q, c, phi
    )
    print(f"  Reconstructed PIT: KS={ks:.4f}, p={pit_p:.4f}")
    
    # Compare with ground truth PIT
    ks_true, pit_p_true = GaussianDriftModel.pit_ks_predictive(
        returns, mu_pred_true, S_pred_true
    )
    print(f"  Ground truth PIT: KS={ks_true:.4f}, p={pit_p_true:.4f}")
    
    # Should be very close
    assert abs(ks - ks_true) < 0.01, f"KS mismatch: {ks} vs {ks_true}"
    print("  ✓ Predictive reconstruction matches ground truth")


if __name__ == "__main__":
    print("=" * 60)
    print("PIT CALIBRATION FIX TESTS")
    print("=" * 60)
    
    test_sample_size_adjusted_threshold()
    test_is_pit_calibrated()
    test_pit_calibration_metrics_perfect()
    test_pit_calibration_metrics_biased()
    test_predictive_reconstruction()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
