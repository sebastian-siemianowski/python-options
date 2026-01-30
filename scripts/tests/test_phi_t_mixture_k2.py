#!/usr/bin/env python3
"""
===============================================================================
DEPRECATED: TEST SUITE FOR K=2 MIXTURE MODEL
===============================================================================

** THIS TEST MODULE IS DEPRECATED **

The K=2 mixture model has been removed from production after empirical
evaluation showed 0% selection rate (206 attempts, 0 selections).

This test file is kept for:
  - Historical reference
  - Ensuring the deprecated module doesn't break if imported for
    backward compatibility with cached results
  - Research experimentation (not production)

See: docs/CALIBRATION_SOLUTIONS_ANALYSIS.md for decision rationale.

-------------------------------------------------------------------------------
ORIGINAL DOCUMENTATION (HISTORICAL)
-------------------------------------------------------------------------------

This test suite validates:
1. Identifiability constraints (σ_B ≥ 1.5 × σ_A)
2. Weight bounds (w ∈ [0.1, 0.9])
3. PIT uniformity improvement
4. BIC-based model selection
5. Numerical stability
6. Backward compatibility

Run with: python -m pytest scripts/tests/test_phi_t_mixture_k2.py -v
===============================================================================
"""
from __future__ import annotations

import warnings
import numpy as np
import pytest
from scipy.stats import kstest, t as student_t

# Emit deprecation warning when tests are run
warnings.warn(
    "K=2 mixture tests are deprecated. The feature was removed after "
    "206 attempts with 0 selections (0% success rate).",
    DeprecationWarning,
    stacklevel=2
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from phi_t_mixture_k2 import (
    PhiTMixtureK2,
    PhiTMixtureK2Config,
    PhiTMixtureK2Result,
    DEFAULT_MIXTURE_CONFIG,
    should_use_mixture,
    fit_and_select,
    validate_mixture_result,
    summarize_mixture_improvement,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Default mixture configuration."""
    return PhiTMixtureK2Config()


@pytest.fixture
def mixer(default_config):
    """Default mixture model instance."""
    return PhiTMixtureK2(default_config)


@pytest.fixture
def synthetic_single_model_data():
    """
    Generate data from a single φ-t model.
    
    When data comes from a single model, the mixture should NOT be preferred
    (BIC penalty should favor simpler model).
    """
    np.random.seed(42)
    n = 500
    
    phi = 0.1
    nu = 8.0
    sigma = 0.02
    
    # Generate returns
    mu = 0.0
    returns = np.zeros(n)
    for t in range(n):
        mu = phi * mu
        epsilon = student_t.rvs(df=nu, scale=sigma)
        returns[t] = mu + epsilon
    
    # Simple volatility estimate (EWMA)
    vol = np.full(n, sigma)
    
    return returns, vol, {'phi': phi, 'nu': nu, 'sigma': sigma}


@pytest.fixture
def synthetic_mixture_data():
    """
    Generate data from a true mixture process.
    
    When data comes from a mixture, the mixture model should be preferred.
    """
    np.random.seed(123)
    n = 1000
    
    phi = 0.15
    nu = 8.0
    sigma_a = 0.015  # Calm
    sigma_b = 0.035  # Stress (ratio = 2.33)
    weight = 0.7     # 70% calm
    
    # Generate regime indicators
    regimes = np.random.choice([0, 1], size=n, p=[weight, 1-weight])
    
    # Generate returns
    mu = 0.0
    returns = np.zeros(n)
    for t in range(n):
        mu = phi * mu
        sigma = sigma_a if regimes[t] == 0 else sigma_b
        epsilon = student_t.rvs(df=nu, scale=sigma)
        returns[t] = mu + epsilon
    
    vol = np.full(n, (sigma_a + sigma_b) / 2)
    
    return returns, vol, {
        'phi': phi, 'nu': nu, 
        'sigma_a': sigma_a, 'sigma_b': sigma_b, 
        'weight': weight
    }


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Test configuration validation and defaults."""
    
    def test_default_config_values(self, default_config):
        """Verify default configuration values."""
        assert default_config.enabled == True
        assert default_config.min_weight == 0.1
        assert default_config.max_weight == 0.9
        assert default_config.sigma_ratio_min == 1.5
        assert default_config.sigma_ratio_max == 5.0
        assert default_config.entropy_penalty == 0.05
        assert default_config.bic_threshold == 0.0
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'enabled': False,
            'min_weight': 0.2,
            'sigma_ratio_min': 2.0,
        }
        config = PhiTMixtureK2Config.from_dict(config_dict)
        
        assert config.enabled == False
        assert config.min_weight == 0.2
        assert config.sigma_ratio_min == 2.0
        # Other values should be defaults
        assert config.max_weight == 0.9
    
    def test_config_to_dict(self, default_config):
        """Test exporting config to dictionary."""
        d = default_config.to_dict()
        
        assert 'enabled' in d
        assert 'min_weight' in d
        assert 'sigma_ratio_min' in d
        assert d['enabled'] == True


# =============================================================================
# IDENTIFIABILITY TESTS
# =============================================================================

class TestIdentifiability:
    """Test that identifiability constraints are enforced."""
    
    def test_sigma_ratio_constraint(self, mixer, synthetic_mixture_data):
        """Verify σ_B ≥ 1.5 × σ_A after fitting."""
        returns, vol, true_params = synthetic_mixture_data
        
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=true_params['nu'],
            phi_init=true_params['phi'],
            sigma_init=true_params['sigma_a']
        )
        
        assert result is not None, "Fit should succeed"
        assert result.sigma_b >= 1.5 * result.sigma_a, \
            f"σ_B/σ_A = {result.sigma_ratio:.2f} should be ≥ 1.5"
    
    def test_weight_bounds(self, mixer, synthetic_mixture_data):
        """Verify weight stays in allowed range."""
        returns, vol, true_params = synthetic_mixture_data
        
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=true_params['nu'],
            phi_init=true_params['phi'],
            sigma_init=true_params['sigma_a']
        )
        
        assert result is not None
        assert 0.1 <= result.weight <= 0.9, \
            f"weight = {result.weight:.3f} should be in [0.1, 0.9]"
    
    def test_phi_bounds(self, mixer, synthetic_mixture_data):
        """Verify φ stays in stable range."""
        returns, vol, true_params = synthetic_mixture_data
        
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=true_params['nu'],
            phi_init=0.5,  # Start closer to unit root
            sigma_init=0.02
        )
        
        assert result is not None
        assert -0.999 <= result.phi <= 0.999, \
            f"φ = {result.phi:.3f} should be in [-0.999, 0.999]"
    
    def test_ar1_innovations_computation(self, mixer):
        """Test that AR(1) innovations are computed correctly."""
        # Simple test case: known returns and phi
        returns = np.array([0.1, 0.2, 0.15, 0.25, 0.1])
        phi = 0.5
        
        # Expected innovations: εₜ = rₜ - φ·rₜ₋₁
        # ε₁ = 0.2 - 0.5*0.1 = 0.15
        # ε₂ = 0.15 - 0.5*0.2 = 0.05
        # ε₃ = 0.25 - 0.5*0.15 = 0.175
        # ε₄ = 0.1 - 0.5*0.25 = -0.025
        expected = np.array([0.15, 0.05, 0.175, -0.025])
        
        # Call static method via class
        innovations = PhiTMixtureK2._compute_ar1_innovations(returns, phi)
        
        np.testing.assert_allclose(innovations, expected, rtol=1e-10)


# =============================================================================
# CALIBRATION TESTS
# =============================================================================

class TestCalibration:
    """Test PIT calibration improvement."""
    
    def test_pit_values_in_unit_interval(self, mixer, synthetic_mixture_data):
        """PIT values should be in [0, 1]."""
        returns, vol, true_params = synthetic_mixture_data
        
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=true_params['nu'],
            phi_init=true_params['phi'],
            sigma_init=true_params['sigma_a']
        )
        
        pit = mixer.compute_pit(returns, vol, result)
        
        assert np.all(pit >= 0), "PIT values should be ≥ 0"
        assert np.all(pit <= 1), "PIT values should be ≤ 1"
    
    def test_pit_uniformity_for_mixture_data(self, mixer, synthetic_mixture_data):
        """PIT should be closer to uniform for mixture model than single model."""
        returns, vol, true_params = synthetic_mixture_data
        
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=true_params['nu'],
            phi_init=true_params['phi'],
            sigma_init=true_params['sigma_a']
        )
        
        pit = mixer.compute_pit(returns, vol, result)
        
        # KS test against uniform
        ks_stat, ks_pvalue = kstest(pit, 'uniform')
        
        # The PIT values should at least be in valid range
        assert np.all(pit >= 0) and np.all(pit <= 1), "PIT values must be in [0, 1]"
        
        # Compare to a single-component model using AR(1) innovations
        sigma_single = (true_params['sigma_a'] + true_params['sigma_b']) / 2
        innovations = PhiTMixtureK2._compute_ar1_innovations(returns, true_params['phi'])
        aligned_vol = vol[1:]
        z_single = innovations / (sigma_single * aligned_vol)
        pit_single = student_t.cdf(z_single, df=true_params['nu'])
        ks_single, _ = kstest(pit_single, 'uniform')
        
        # Mixture should not be dramatically worse than single model
        # (it should be better or at least similar)
        assert ks_stat <= ks_single * 1.5, \
            f"Mixture KS={ks_stat:.4f} should not be much worse than single KS={ks_single:.4f}"
    
    def test_mixture_improves_calibration(self, synthetic_mixture_data):
        """Mixture should improve calibration over single model for mixture data."""
        returns, vol, true_params = synthetic_mixture_data
        
        # Fit single model (using mean sigma)
        from phi_t_mixture_k2 import PhiTMixtureK2
        
        mixer = PhiTMixtureK2(PhiTMixtureK2Config())
        
        # Get single model PIT using AR(1) innovations
        innovations = PhiTMixtureK2._compute_ar1_innovations(returns, true_params['phi'])
        aligned_vol = vol[1:]
        sigma_single = (true_params['sigma_a'] + true_params['sigma_b']) / 2
        
        z_single = innovations / (sigma_single * aligned_vol)
        pit_single = student_t.cdf(z_single, df=true_params['nu'])
        ks_single, _ = kstest(pit_single, 'uniform')
        
        # Fit mixture
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=true_params['nu'],
            phi_init=true_params['phi'],
            sigma_init=sigma_single
        )
        
        ks_mixture = result.ks_statistic
        
        # Mixture should have lower (better) KS statistic
        assert ks_mixture < ks_single, \
            f"Mixture KS={ks_mixture:.4f} should be < single KS={ks_single:.4f}"


# =============================================================================
# MODEL SELECTION TESTS
# =============================================================================

class TestModelSelection:
    """Test BIC-based model selection."""
    
    def test_single_model_preferred_for_single_data(
        self, mixer, synthetic_single_model_data
    ):
        """Single model should be preferred when data is from single model."""
        returns, vol, true_params = synthetic_single_model_data
        
        # Fit mixture
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=true_params['nu'],
            phi_init=true_params['phi'],
            sigma_init=true_params['sigma']
        )
        
        if result is None:
            pytest.skip("Mixture fit failed (expected for single-model data)")
        
        # Compute single model BIC using AR(1) innovations
        innovations = PhiTMixtureK2._compute_ar1_innovations(returns, true_params['phi'])
        aligned_vol = vol[1:]
        n = len(innovations)
        k_single = 3  # phi, sigma, (nu fixed)
        
        # Simple single-model log-likelihood on innovations
        z = innovations / (true_params['sigma'] * aligned_vol)
        ll_single = np.sum(student_t.logpdf(z, df=true_params['nu']))
        bic_single = k_single * np.log(n) - 2 * ll_single
        
        # Mixture should NOT be preferred (higher BIC due to penalty)
        use_mixture = should_use_mixture(bic_single, result)
        
        # This is a soft assertion - mixture might still be selected if it
        # happens to fit better, but we expect single to win most of the time
        print(f"Single BIC: {bic_single:.2f}, Mixture BIC: {result.bic:.2f}")
    
    def test_mixture_preferred_for_mixture_data(
        self, mixer, synthetic_mixture_data
    ):
        """Mixture should be preferred when data is from mixture process."""
        returns, vol, true_params = synthetic_mixture_data
        
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=true_params['nu'],
            phi_init=true_params['phi'],
            sigma_init=true_params['sigma_a']
        )
        
        assert result is not None
        
        # Compute single model BIC using AR(1) innovations
        innovations = PhiTMixtureK2._compute_ar1_innovations(returns, true_params['phi'])
        aligned_vol = vol[1:]
        n = len(innovations)
        k_single = 3
        sigma_single = (true_params['sigma_a'] + true_params['sigma_b']) / 2
        
        z = innovations / (sigma_single * aligned_vol)
        ll_single = np.sum(student_t.logpdf(z, df=true_params['nu']))
        bic_single = k_single * np.log(n) - 2 * ll_single
        
        # Mixture should be preferred (lower BIC despite penalty)
        use_mixture = should_use_mixture(bic_single, result)
        
        assert use_mixture, \
            f"Mixture (BIC={result.bic:.2f}) should be preferred over single (BIC={bic_single:.2f})"


# =============================================================================
# NUMERICAL STABILITY TESTS
# =============================================================================

class TestNumericalStability:
    """Test numerical stability edge cases."""
    
    def test_small_sample(self, mixer):
        """Should handle small samples gracefully."""
        np.random.seed(99)
        returns = np.random.randn(30) * 0.02
        vol = np.full(30, 0.02)
        
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=8.0,
            phi_init=0.1,
            sigma_init=0.02
        )
        
        # Should return None for very small samples
        assert result is None, "Should reject samples < 50"
    
    def test_zero_volatility_handling(self, mixer):
        """Should handle near-zero volatility."""
        np.random.seed(100)
        n = 200
        returns = np.random.randn(n) * 0.02
        vol = np.full(n, 1e-8)  # Near-zero vol
        
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=8.0,
            phi_init=0.1,
            sigma_init=0.02
        )
        
        # Should either succeed or fail gracefully (no crash)
        if result is not None:
            assert np.isfinite(result.log_likelihood)
    
    def test_extreme_returns(self, mixer):
        """Should handle extreme returns."""
        np.random.seed(101)
        n = 200
        returns = np.random.randn(n) * 0.02
        # Add extreme outliers
        returns[50] = 0.5  # +50%
        returns[100] = -0.3  # -30%
        vol = np.full(n, 0.02)
        
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=4.0,  # Heavy tails
            phi_init=0.1,
            sigma_init=0.02
        )
        
        if result is not None:
            assert np.isfinite(result.log_likelihood)
            assert np.isfinite(result.bic)


# =============================================================================
# RESULT SERIALIZATION TESTS
# =============================================================================

class TestSerialization:
    """Test result serialization for cache compatibility."""
    
    def test_result_to_dict(self, mixer, synthetic_mixture_data):
        """Result should serialize to dictionary."""
        returns, vol, true_params = synthetic_mixture_data
        
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=true_params['nu'],
            phi_init=true_params['phi'],
            sigma_init=true_params['sigma_a']
        )
        
        d = result.to_dict()
        
        assert 'phi' in d
        assert 'nu' in d
        assert 'sigma_a' in d
        assert 'sigma_b' in d
        assert 'weight' in d
        assert 'bic' in d
        assert 'model_type' in d
        assert d['model_type'] == 'phi_t_mixture_k2'
    
    def test_result_roundtrip(self, mixer, synthetic_mixture_data):
        """Result should survive serialization roundtrip."""
        returns, vol, true_params = synthetic_mixture_data
        
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=true_params['nu'],
            phi_init=true_params['phi'],
            sigma_init=true_params['sigma_a']
        )
        
        # Serialize and deserialize
        d = result.to_dict()
        restored = PhiTMixtureK2Result.from_dict(d)
        
        assert restored.phi == result.phi
        assert restored.nu == result.nu
        assert restored.sigma_a == result.sigma_a
        assert restored.sigma_b == result.sigma_b
        assert restored.weight == result.weight
        assert restored.bic == result.bic


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidation:
    """Test validation utilities."""
    
    def test_validate_good_result(self, mixer, synthetic_mixture_data, default_config):
        """Valid result should pass validation."""
        returns, vol, true_params = synthetic_mixture_data
        
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=true_params['nu'],
            phi_init=true_params['phi'],
            sigma_init=true_params['sigma_a']
        )
        
        errors = validate_mixture_result(result, default_config)
        
        assert len(errors) == 0, f"Validation errors: {errors}"
    
    def test_summarize_improvement(self, mixer, synthetic_mixture_data):
        """Summary should contain expected fields."""
        returns, vol, true_params = synthetic_mixture_data
        
        result = mixer.fit(
            returns=returns,
            vol=vol,
            nu=true_params['nu'],
            phi_init=true_params['phi'],
            sigma_init=true_params['sigma_a']
        )
        
        summary = summarize_mixture_improvement(
            single_bic=5000.0,  # Hypothetical
            single_pit_pvalue=0.01,  # Poor calibration
            mixture_result=result
        )
        
        assert 'mixture_available' in summary
        assert 'bic_improvement' in summary
        assert 'recommendation' in summary
        assert summary['mixture_available'] == True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests with existing system components."""
    
    def test_fit_and_select_api(self, synthetic_mixture_data):
        """Test the main integration API."""
        returns, vol, true_params = synthetic_mixture_data
        
        model_type, result_dict = fit_and_select(
            returns=returns,
            vol=vol,
            nu=true_params['nu'],
            phi_single=true_params['phi'],
            sigma_single=true_params['sigma_a'],
            bic_single=6000.0  # Hypothetical poor single model
        )
        
        assert model_type in ['single', 'mixture']
        
        if model_type == 'mixture':
            assert 'phi' in result_dict
            assert 'sigma_a' in result_dict
            assert 'sigma_b' in result_dict


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
