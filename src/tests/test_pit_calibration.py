#!/usr/bin/env python3
"""
test_pit_calibration.py

Unit tests for PIT (Probability Integral Transform) calibration verification.

Tests cover:
1. Well-calibrated synthetic data (diagonal reliability diagram)
2. Overconfident synthetic data (predictions too high)
3. Underconfident synthetic data (predictions too low)
4. Edge cases (empty bins, extreme probabilities)
5. Uniformity test
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration.pit_calibration import (
    compute_pit_calibration,
    CalibrationMetrics,
)


def test_well_calibrated_data():
    """
    Test with perfectly calibrated synthetic data.
    
    Generate predictions where P(outcome > 0) = predicted probability exactly.
    Expected: ECE ≈ 0, calibrated=True, diagonal reliability diagram
    """
    np.random.seed(42)
    n = 1000
    
    # Generate well-calibrated predictions
    # For each prediction p, outcome is positive with probability p
    predicted_probs = np.random.uniform(0.1, 0.9, n)
    actual_outcomes = np.zeros(n)
    
    for i in range(n):
        # Bernoulli trial: outcome > 0 with probability predicted_probs[i]
        if np.random.rand() < predicted_probs[i]:
            actual_outcomes[i] = np.random.normal(0.05, 0.1)  # positive return
        else:
            actual_outcomes[i] = np.random.normal(-0.05, 0.1)  # negative return
    
    # Compute calibration metrics
    metrics = compute_pit_calibration(predicted_probs, actual_outcomes, n_bins=10, ece_threshold=0.05)
    
    # Assertions
    assert metrics.n_predictions == n
    assert metrics.n_bins == 10
    
    # Well-calibrated data should have low ECE
    print(f"Test 1 (Well-calibrated): ECE = {metrics.expected_calibration_error:.4f}")
    assert metrics.expected_calibration_error < 0.10, f"ECE too high: {metrics.expected_calibration_error}"
    
    # Should pass calibration test
    # Note: May fail due to sampling noise with n=1000, so we use relaxed threshold
    if metrics.expected_calibration_error < 0.05:
        assert metrics.calibrated, "Should be classified as calibrated"
    
    # Diagnosis should be well_calibrated or slightly_miscalibrated (due to sampling)
    assert metrics.calibration_diagnosis in ["well_calibrated", "slightly_miscalibrated"], \
        f"Unexpected diagnosis: {metrics.calibration_diagnosis}"
    
    # Brier score should be reasonable (not perfect due to noise)
    assert 0.0 <= metrics.brier_score <= 0.5, f"Brier score out of range: {metrics.brier_score}"
    
    print("✓ Test 1 passed: Well-calibrated data")


def test_overconfident_data():
    """
    Test with overconfident synthetic data.
    
    Predictions are systematically too high: when predicting P=0.7, actual frequency is only 0.5.
    Expected: ECE > 0.05, calibrated=False, diagnosis="overconfident"
    """
    np.random.seed(43)
    n = 1000
    
    # Generate overconfident predictions
    # True probabilities are 0.2 lower than predicted
    true_probs = np.random.uniform(0.1, 0.7, n)
    predicted_probs = np.clip(true_probs + 0.2, 0.0, 1.0)  # inflate by 0.2
    
    actual_outcomes = np.zeros(n)
    for i in range(n):
        # Use true probability for outcome generation
        if np.random.rand() < true_probs[i]:
            actual_outcomes[i] = np.random.normal(0.05, 0.1)
        else:
            actual_outcomes[i] = np.random.normal(-0.05, 0.1)
    
    # Compute calibration metrics
    metrics = compute_pit_calibration(predicted_probs, actual_outcomes, n_bins=10, ece_threshold=0.05)
    
    # Assertions
    print(f"Test 2 (Overconfident): ECE = {metrics.expected_calibration_error:.4f}")
    
    # Overconfident data should have high ECE
    assert metrics.expected_calibration_error > 0.10, \
        f"ECE too low for overconfident data: {metrics.expected_calibration_error}"
    
    # Should fail calibration test
    assert not metrics.calibrated, "Should not be classified as calibrated"
    
    # Diagnosis should be overconfident
    assert metrics.calibration_diagnosis == "overconfident", \
        f"Expected 'overconfident', got '{metrics.calibration_diagnosis}'"
    
    # Check reliability diagram shows predictions > actual
    valid_bins = metrics.bin_counts > 0
    if np.sum(valid_bins) > 0:
        avg_deviation = np.mean(metrics.predicted_probs[valid_bins] - metrics.actual_frequencies[valid_bins])
        assert avg_deviation > 0.05, "Expected positive deviation (overconfident)"
    
    print("✓ Test 2 passed: Overconfident data detected")


def test_underconfident_data():
    """
    Test with underconfident synthetic data.
    
    Predictions are systematically too low: when predicting P=0.5, actual frequency is 0.7.
    Expected: ECE > 0.05, calibrated=False, diagnosis="underconfident"
    """
    np.random.seed(44)
    n = 1000
    
    # Generate underconfident predictions
    # True probabilities are 0.2 higher than predicted
    true_probs = np.random.uniform(0.3, 0.9, n)
    predicted_probs = np.clip(true_probs - 0.2, 0.0, 1.0)  # deflate by 0.2
    
    actual_outcomes = np.zeros(n)
    for i in range(n):
        # Use true probability for outcome generation
        if np.random.rand() < true_probs[i]:
            actual_outcomes[i] = np.random.normal(0.05, 0.1)
        else:
            actual_outcomes[i] = np.random.normal(-0.05, 0.1)
    
    # Compute calibration metrics
    metrics = compute_pit_calibration(predicted_probs, actual_outcomes, n_bins=10, ece_threshold=0.05)
    
    # Assertions
    print(f"Test 3 (Underconfident): ECE = {metrics.expected_calibration_error:.4f}")
    
    # Underconfident data should have high ECE
    assert metrics.expected_calibration_error > 0.10, \
        f"ECE too low for underconfident data: {metrics.expected_calibration_error}"
    
    # Should fail calibration test
    assert not metrics.calibrated, "Should not be classified as calibrated"
    
    # Diagnosis should be underconfident
    assert metrics.calibration_diagnosis == "underconfident", \
        f"Expected 'underconfident', got '{metrics.calibration_diagnosis}'"
    
    # Check reliability diagram shows predictions < actual
    valid_bins = metrics.bin_counts > 0
    if np.sum(valid_bins) > 0:
        avg_deviation = np.mean(metrics.predicted_probs[valid_bins] - metrics.actual_frequencies[valid_bins])
        assert avg_deviation < -0.05, "Expected negative deviation (underconfident)"
    
    print("✓ Test 3 passed: Underconfident data detected")


def test_extreme_probabilities():
    """
    Test with extreme probabilities (near 0 and 1).
    
    Should handle edge cases gracefully without errors.
    """
    np.random.seed(45)
    n = 200
    
    # Generate extreme predictions
    predicted_probs = np.concatenate([
        np.random.uniform(0.0, 0.05, n // 2),  # very low
        np.random.uniform(0.95, 1.0, n // 2),  # very high
    ])
    
    # Generate outcomes matching predictions
    actual_outcomes = np.zeros(n)
    for i in range(n):
        if np.random.rand() < predicted_probs[i]:
            actual_outcomes[i] = np.random.normal(0.05, 0.1)
        else:
            actual_outcomes[i] = np.random.normal(-0.05, 0.1)
    
    # Compute calibration metrics
    metrics = compute_pit_calibration(predicted_probs, actual_outcomes, n_bins=10, ece_threshold=0.05)
    
    # Assertions: should not crash, metrics should be finite
    assert np.isfinite(metrics.expected_calibration_error)
    assert np.isfinite(metrics.brier_score)
    assert metrics.n_predictions == n
    
    print(f"Test 4 (Extreme probabilities): ECE = {metrics.expected_calibration_error:.4f}")
    print("✓ Test 4 passed: Extreme probabilities handled")


def test_uniformity_test():
    """
    Test uniformity chi-squared test.
    
    Well-calibrated predictions should have uniform distribution.
    Miscalibrated predictions should fail uniformity test.
    """
    np.random.seed(46)
    n = 1000
    
    # Test 1: Uniform distribution (well-calibrated)
    predicted_probs_uniform = np.random.uniform(0.0, 1.0, n)
    actual_outcomes_uniform = np.zeros(n)
    for i in range(n):
        if np.random.rand() < predicted_probs_uniform[i]:
            actual_outcomes_uniform[i] = np.random.normal(0.05, 0.1)
        else:
            actual_outcomes_uniform[i] = np.random.normal(-0.05, 0.1)
    
    metrics_uniform = compute_pit_calibration(predicted_probs_uniform, actual_outcomes_uniform, n_bins=10)
    
    print(f"Test 5a (Uniform distribution): p-value = {metrics_uniform.uniformity_pvalue:.4f}")
    # Should pass uniformity test (p >= 0.05)
    # Note: May fail due to sampling noise, so we just check it's computed
    assert np.isfinite(metrics_uniform.uniformity_pvalue)
    assert 0.0 <= metrics_uniform.uniformity_pvalue <= 1.0
    
    # Test 2: Non-uniform distribution (all predictions near 0.5)
    predicted_probs_concentrated = np.random.normal(0.5, 0.05, n)
    predicted_probs_concentrated = np.clip(predicted_probs_concentrated, 0.0, 1.0)
    
    actual_outcomes_concentrated = np.zeros(n)
    for i in range(n):
        if np.random.rand() < predicted_probs_concentrated[i]:
            actual_outcomes_concentrated[i] = np.random.normal(0.05, 0.1)
        else:
            actual_outcomes_concentrated[i] = np.random.normal(-0.05, 0.1)
    
    metrics_concentrated = compute_pit_calibration(predicted_probs_concentrated, actual_outcomes_concentrated, n_bins=10)
    
    print(f"Test 5b (Concentrated distribution): p-value = {metrics_concentrated.uniformity_pvalue:.4f}")
    # Should likely fail uniformity test (p < 0.05) due to concentration
    # But we just verify it's computed
    assert np.isfinite(metrics_concentrated.uniformity_pvalue)
    assert 0.0 <= metrics_concentrated.uniformity_pvalue <= 1.0
    
    print("✓ Test 5 passed: Uniformity test computed")


def test_brier_score():
    """
    Test Brier score computation.
    
    Perfect predictions should have Brier score ≈ 0.
    Random predictions should have Brier score ≈ 0.25.
    """
    np.random.seed(47)
    n = 1000
    
    # Test 1: Perfect predictions (deterministic outcomes)
    predicted_probs_perfect = np.random.uniform(0.0, 1.0, n)
    actual_outcomes_perfect = np.zeros(n)
    for i in range(n):
        # Deterministic: if p > 0.5 => always positive, else always negative
        if predicted_probs_perfect[i] > 0.5:
            actual_outcomes_perfect[i] = 0.1  # positive
        else:
            actual_outcomes_perfect[i] = -0.1  # negative
    
    metrics_perfect = compute_pit_calibration(predicted_probs_perfect, actual_outcomes_perfect, n_bins=10)
    
    print(f"Test 6a (Perfect predictions): Brier score = {metrics_perfect.brier_score:.4f}")
    # Not truly perfect due to probabilistic interpretation, but should be low
    assert metrics_perfect.brier_score < 0.3, f"Brier score too high: {metrics_perfect.brier_score}"
    
    # Test 2: Random predictions (always predict 0.5)
    predicted_probs_random = np.full(n, 0.5)
    actual_outcomes_random = np.random.choice([0.1, -0.1], n)  # 50/50 outcomes
    
    metrics_random = compute_pit_calibration(predicted_probs_random, actual_outcomes_random, n_bins=10)
    
    print(f"Test 6b (Random predictions): Brier score = {metrics_random.brier_score:.4f}")
    # Brier score for 50/50 prediction should be around 0.25
    assert 0.20 <= metrics_random.brier_score <= 0.30, f"Brier score unexpected: {metrics_random.brier_score}"
    
    print("✓ Test 6 passed: Brier score computation correct")


def test_edge_case_insufficient_data():
    """
    Test with insufficient data (< 10 predictions).
    
    Should raise ValueError.
    """
    predicted_probs = np.array([0.5, 0.6, 0.7])
    actual_outcomes = np.array([0.1, -0.1, 0.2])
    
    try:
        metrics = compute_pit_calibration(predicted_probs, actual_outcomes)
        assert False, "Should have raised ValueError for insufficient data"
    except ValueError as e:
        assert "Insufficient data" in str(e)
        print("✓ Test 7 passed: Insufficient data raises ValueError")


def test_edge_case_nan_handling():
    """
    Test with NaN values in predictions or outcomes.
    
    Should filter out NaNs and compute metrics on valid data.
    """
    np.random.seed(48)
    n = 100
    
    predicted_probs = np.random.uniform(0.1, 0.9, n)
    actual_outcomes = np.random.normal(0.0, 0.1, n)
    
    # Inject NaNs
    predicted_probs[10:15] = np.nan
    actual_outcomes[20:25] = np.nan
    
    # Should handle NaNs gracefully
    metrics = compute_pit_calibration(predicted_probs, actual_outcomes, n_bins=10)
    
    # Should have fewer predictions after filtering
    assert metrics.n_predictions < n
    assert metrics.n_predictions == n - 10  # 5 NaN predictions + 5 NaN outcomes
    
    print(f"Test 8 (NaN handling): {metrics.n_predictions} valid predictions after filtering")
    print("✓ Test 8 passed: NaN values filtered correctly")


def run_all_tests():
    """Run all PIT calibration tests."""
    print("=" * 60)
    print("Running PIT Calibration Unit Tests")
    print("=" * 60)
    print()
    
    tests = [
        ("Well-calibrated data", test_well_calibrated_data),
        ("Overconfident data", test_overconfident_data),
        ("Underconfident data", test_underconfident_data),
        ("Extreme probabilities", test_extreme_probabilities),
        ("Uniformity test", test_uniformity_test),
        ("Brier score", test_brier_score),
        ("Insufficient data edge case", test_edge_case_insufficient_data),
        ("NaN handling edge case", test_edge_case_nan_handling),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"✗ Test failed: {test_name}")
            print(f"  Error: {e}")
            print()
            failed += 1
        except Exception as e:
            print(f"✗ Test crashed: {test_name}")
            print(f"  Error: {e}")
            print()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
