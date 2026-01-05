#!/usr/bin/env python3
"""
test_model_comparison.py

Unit tests for structural model comparison framework.
Tests AIC/BIC comparison logic, Akaike weights, and model selection recommendations.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np
import pandas as pd
from model_comparison import (
    ModelSpec, ComparisonResult, compare_models,
    compute_ewma_volatility_with_likelihood, compare_volatility_models,
    compute_gaussian_tail_loglikelihood, compare_tail_models,
    compare_drift_models
)


def test_model_spec_aic_bic():
    """Test ModelSpec AIC/BIC computation."""
    print("Test 1: ModelSpec AIC/BIC computation")
    
    model = ModelSpec(
        name="Test Model",
        n_params=3,
        log_likelihood=100.0,
        n_obs=1000,
        converged=True
    )
    
    # AIC = 2k - 2*ln(L) = 2*3 - 2*100 = 6 - 200 = -194
    expected_aic = 2.0 * 3 - 2.0 * 100.0
    assert abs(model.aic - expected_aic) < 1e-6, f"AIC mismatch: {model.aic} vs {expected_aic}"
    
    # BIC = k*ln(n) - 2*ln(L) = 3*ln(1000) - 2*100 = 3*6.907... - 200 ≈ -179.28
    expected_bic = 3 * np.log(1000) - 2.0 * 100.0
    assert abs(model.bic - expected_bic) < 1e-6, f"BIC mismatch: {model.bic} vs {expected_bic}"
    
    print(f"  AIC: {model.aic:.2f} (expected {expected_aic:.2f})")
    print(f"  BIC: {model.bic:.2f} (expected {expected_bic:.2f})")
    print("  ✓ PASS\n")


def test_compare_models_single_winner():
    """Test model comparison with clear winner."""
    print("Test 2: Model comparison with clear winner")
    
    # Model 1: Simple model with good fit
    model1 = ModelSpec(
        name="Simple",
        n_params=1,
        log_likelihood=-100.0,
        n_obs=500,
        converged=True
    )
    
    # Model 2: Complex model with slightly better fit (not enough to justify complexity)
    model2 = ModelSpec(
        name="Complex",
        n_params=5,
        log_likelihood=-98.0,  # Only 2 log-lik units better
        n_obs=500,
        converged=True
    )
    
    result = compare_models([model1, model2])
    
    print(f"  Winner (AIC): {result.winner_aic}")
    print(f"  Winner (BIC): {result.winner_bic}")
    print(f"  Δ AIC: {result.delta_aic}")
    print(f"  Δ BIC: {result.delta_bic}")
    print(f"  Akaike weights: {result.akaike_weights}")
    
    # BIC should strongly prefer simpler model (more penalty for complexity)
    assert result.winner_bic == "Simple", f"BIC should prefer Simple, got {result.winner_bic}"
    
    # AIC might prefer Simple too (small LL improvement vs 4 extra params)
    # AIC penalty = 2*(5-1) = 8, LL gain = 2*2 = 4, so Simple should still win
    assert result.winner_aic == "Simple", f"AIC should prefer Simple, got {result.winner_aic}"
    
    print("  ✓ PASS\n")


def test_compare_models_identical():
    """Test model comparison with identical models."""
    print("Test 3: Model comparison with identical AIC/BIC")
    
    model1 = ModelSpec(name="Model_A", n_params=2, log_likelihood=-50.0, n_obs=200)
    model2 = ModelSpec(name="Model_B", n_params=2, log_likelihood=-50.0, n_obs=200)
    
    result = compare_models([model1, model2])
    
    # Both should have delta = 0
    assert abs(result.delta_aic["Model_A"]) < 1e-10, "Identical models should have Δ AIC = 0"
    assert abs(result.delta_aic["Model_B"]) < 1e-10, "Identical models should have Δ AIC = 0"
    
    # Akaike weights should be equal (0.5 each)
    assert abs(result.akaike_weights["Model_A"] - 0.5) < 1e-6, "Equal models should have equal weights"
    assert abs(result.akaike_weights["Model_B"] - 0.5) < 1e-6, "Equal models should have equal weights"
    
    print(f"  Akaike weights: {result.akaike_weights}")
    print("  ✓ PASS\n")


def test_akaike_weights_interpretation():
    """Test Akaike weight interpretation with clear differences."""
    print("Test 4: Akaike weight interpretation")
    
    # Model 1: Strong winner (Δ AIC = 0)
    model1 = ModelSpec(name="Best", n_params=2, log_likelihood=-100.0, n_obs=500)
    
    # Model 2: Competitive (Δ AIC ≈ 2)
    model2 = ModelSpec(name="Competitive", n_params=2, log_likelihood=-101.0, n_obs=500)
    
    # Model 3: Weak (Δ AIC ≈ 10)
    model3 = ModelSpec(name="Weak", n_params=2, log_likelihood=-105.0, n_obs=500)
    
    result = compare_models([model1, model2, model3])
    
    print(f"  Δ AIC: {result.delta_aic}")
    print(f"  Akaike weights: {result.akaike_weights}")
    
    # Best should have highest weight
    assert result.akaike_weights["Best"] > result.akaike_weights["Competitive"], "Best should have higher weight"
    assert result.akaike_weights["Competitive"] > result.akaike_weights["Weak"], "Competitive should beat Weak"
    
    # Best should have > 50% weight
    assert result.akaike_weights["Best"] > 0.5, f"Best model should have >50% weight, got {result.akaike_weights['Best']:.2%}"
    
    # Weak should have very small weight (Δ AIC = 10 => weight ≈ exp(-5) ≈ 0.007)
    assert result.akaike_weights["Weak"] < 0.05, f"Weak model should have <5% weight, got {result.akaike_weights['Weak']:.2%}"
    
    print("  ✓ PASS\n")


def test_ewma_volatility_likelihood():
    """Test EWMA volatility with log-likelihood computation."""
    print("Test 5: EWMA volatility log-likelihood")
    
    # Generate synthetic returns
    np.random.seed(42)
    n = 500
    returns = pd.Series(np.random.randn(n) * 0.01, name='ret')
    
    # Compute EWMA volatility with likelihood
    vol, metadata = compute_ewma_volatility_with_likelihood(returns, span=21)
    
    print(f"  EWMA span: {metadata['span']}")
    print(f"  Log-likelihood: {metadata['log_likelihood']:.2f}")
    print(f"  N obs: {metadata['n_obs']}")
    print(f"  N params: {metadata['n_params']}")
    
    # Check metadata
    assert metadata['n_params'] == 1, "EWMA should have 1 parameter"
    assert metadata['converged'] == True, "EWMA should always converge"
    assert np.isfinite(metadata['log_likelihood']), "Log-likelihood should be finite"
    assert metadata['n_obs'] > 0, "Should have positive observations"
    
    # Check volatility series
    assert len(vol) == len(returns), "Volatility series should match returns length"
    assert vol.isna().sum() < len(vol) // 2, "Most volatility values should be non-NaN"
    
    print("  ✓ PASS\n")


def test_gaussian_tail_likelihood():
    """Test Gaussian tail log-likelihood computation."""
    print("Test 6: Gaussian tail log-likelihood")
    
    # Generate synthetic standardized residuals (should be ~ N(0,1))
    np.random.seed(42)
    n = 1000
    z = pd.Series(np.random.randn(n), name='z')
    
    # Compute Gaussian log-likelihood
    gauss_meta = compute_gaussian_tail_loglikelihood(z)
    
    print(f"  Distribution: {gauss_meta['distribution']}")
    print(f"  Log-likelihood: {gauss_meta['log_likelihood']:.2f}")
    print(f"  N obs: {gauss_meta['n_obs']}")
    print(f"  N params: {gauss_meta['n_params']}")
    
    # Check metadata
    assert gauss_meta['n_params'] == 0, "Standard Gaussian has 0 free parameters"
    assert gauss_meta['converged'] == True, "Gaussian should always converge"
    assert np.isfinite(gauss_meta['log_likelihood']), "Log-likelihood should be finite"
    
    # Theoretical log-likelihood for N(0,1): -0.5*n*ln(2π) - 0.5*Σz²
    # For standard normal, Σz² ≈ n (since E[z²] = 1)
    theoretical_ll = -0.5 * n * np.log(2 * np.pi) - 0.5 * np.sum(z.values ** 2)
    
    # Should be close to theoretical (within numerical tolerance)
    assert abs(gauss_meta['log_likelihood'] - theoretical_ll) < 1.0, \
        f"Log-likelihood mismatch: {gauss_meta['log_likelihood']:.2f} vs {theoretical_ll:.2f}"
    
    print(f"  Theoretical LL: {theoretical_ll:.2f}")
    print("  ✓ PASS\n")


def test_compare_volatility_models():
    """Test volatility model comparison."""
    print("Test 7: Volatility model comparison (GARCH vs EWMA)")
    
    # Generate synthetic returns with time-varying volatility
    np.random.seed(42)
    n = 1000
    
    # GARCH-like process
    omega, alpha, beta = 0.0001, 0.1, 0.85
    returns = []
    h = 0.01 ** 2  # initial variance
    
    for t in range(n):
        eps = np.random.randn()
        ret = np.sqrt(h) * eps
        returns.append(ret)
        h = omega + alpha * (ret ** 2) + beta * h
    
    returns = pd.Series(returns, name='ret')
    
    # Mock GARCH params (would come from _garch11_mle in practice)
    garch_params = {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "converged": True,
        "log_likelihood": -1500.0,  # Mock value
        "n_obs": n,
    }
    
    # Compare models
    result = compare_volatility_models(returns, garch_params)
    
    print(f"  Winner (AIC): {result.winner_aic}")
    print(f"  Winner (BIC): {result.winner_bic}")
    print(f"  Models: {[m.name for m in result.models]}")
    print(f"  Δ AIC: {result.delta_aic}")
    
    # Should have GARCH and EWMA models
    model_names = [m.name for m in result.models]
    assert "GARCH(1,1)" in model_names, "Should include GARCH(1,1)"
    assert any("EWMA" in name for name in model_names), "Should include EWMA model(s)"
    
    print("  ✓ PASS\n")


def test_compare_tail_models():
    """Test tail distribution comparison."""
    print("Test 8: Tail distribution comparison (Student-t vs Gaussian)")
    
    # Generate synthetic residuals with heavy tails (Student-t with df=5)
    np.random.seed(42)
    from scipy.stats import t as student_t_dist
    
    n = 1000
    nu_true = 5.0
    z = pd.Series(student_t_dist.rvs(df=nu_true, size=n), name='z')
    
    # Mock Student-t params (would come from _fit_student_nu_mle in practice)
    student_t_params = {
        "nu_hat": 5.5,  # Close to true value
        "ll": -1400.0,  # Mock value
        "n": n,
        "converged": True,
    }
    
    # Compare models
    result = compare_tail_models(z, student_t_params)
    
    print(f"  Winner (AIC): {result.winner_aic}")
    print(f"  Winner (BIC): {result.winner_bic}")
    print(f"  Models: {[m.name for m in result.models]}")
    print(f"  Δ AIC: {result.delta_aic}")
    
    # Should have both models
    model_names = [m.name for m in result.models]
    assert "Student-t" in model_names, "Should include Student-t"
    assert "Gaussian" in model_names, "Should include Gaussian"
    
    # For heavy-tailed data, Student-t should win (but we can't guarantee with mock params)
    print("  ✓ PASS\n")


def test_edge_case_no_converged_models():
    """Test edge case: no models converged."""
    print("Test 9: Edge case - no converged models")
    
    model1 = ModelSpec(name="Failed1", n_params=1, log_likelihood=float('nan'), n_obs=100, converged=False)
    model2 = ModelSpec(name="Failed2", n_params=2, log_likelihood=float('nan'), n_obs=100, converged=False)
    
    result = compare_models([model1, model2])
    
    print(f"  Winner (AIC): {result.winner_aic}")
    print(f"  Winner (BIC): {result.winner_bic}")
    print(f"  Recommendation: {result.recommendation}")
    
    assert result.winner_aic == "none_converged", "Should indicate no convergence"
    assert result.winner_bic == "none_converged", "Should indicate no convergence"
    assert "failed" in result.recommendation.lower() or "fallback" in result.recommendation.lower(), \
        "Recommendation should mention failure"
    
    print("  ✓ PASS\n")


def main():
    """Run all tests."""
    print("=" * 80)
    print("MODEL COMPARISON UNIT TESTS")
    print("Testing AIC/BIC model selection framework")
    print("=" * 80 + "\n")
    
    try:
        test_model_spec_aic_bic()
        test_compare_models_single_winner()
        test_compare_models_identical()
        test_akaike_weights_interpretation()
        test_ewma_volatility_likelihood()
        test_gaussian_tail_likelihood()
        test_compare_volatility_models()
        test_compare_tail_models()
        test_edge_case_no_converged_models()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
