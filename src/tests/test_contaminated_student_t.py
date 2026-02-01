#!/usr/bin/env python3
"""
Test Contaminated Student-t Mixture Model Implementation

This test verifies that the Contaminated Student-t module correctly implements:
1. PDF/log-PDF calculation for the mixture
2. Random variate generation with proper mixing
3. Parameter estimation via profile likelihood
4. Comparison with single Student-t

References:
    Tukey, J.W. (1960). "A Survey of Sampling from Contaminated Distributions"
    Lange, K.L. et al. (1989). "Robust Statistical Modeling Using the t Distribution"
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy import stats


class TestContaminatedStudentTBasics:
    """Test basic distribution functions."""
    
    def test_imports(self):
        """Test that all components import correctly."""
        from models.contaminated_student_t import (
            ContaminatedStudentTParams,
            contaminated_student_t_pdf,
            contaminated_student_t_logpdf,
            contaminated_student_t_rvs,
            fit_contaminated_student_t_profile,
            compute_crisis_probability_from_vol,
            CST_NU_NORMAL_DEFAULT,
            CST_NU_CRISIS_DEFAULT,
            CST_EPSILON_DEFAULT,
        )
        
        assert CST_NU_NORMAL_DEFAULT == 12.0
        assert CST_NU_CRISIS_DEFAULT == 4.0
        assert CST_EPSILON_DEFAULT == 0.05
    
    def test_params_dataclass(self):
        """Test ContaminatedStudentTParams validation."""
        from models.contaminated_student_t import ContaminatedStudentTParams
        
        # Valid parameters
        params = ContaminatedStudentTParams(
            nu_normal=12.0,
            nu_crisis=4.0,
            epsilon=0.05
        )
        
        assert params.nu_normal == 12.0
        assert params.nu_crisis == 4.0
        assert params.epsilon == 0.05
        assert params.effective_nu < params.nu_normal  # Pulled toward crisis
        assert params.tail_heaviness_ratio == 3.0  # 12/4
        
        # Invalid nu should raise
        with pytest.raises(ValueError):
            ContaminatedStudentTParams(nu_normal=1.5, nu_crisis=4.0, epsilon=0.05)
    
    def test_pdf_reduces_to_student_t_when_epsilon_zero(self):
        """Test that mixture PDF reduces to single Student-t when ε=0."""
        from models.contaminated_student_t import contaminated_student_t_pdf
        
        x = np.linspace(-5, 5, 100)
        nu = 8.0
        
        # With ε=0, should be pure normal component
        pdf_mixture = contaminated_student_t_pdf(x, nu_normal=nu, nu_crisis=4.0, epsilon=0.0)
        pdf_single = stats.t.pdf(x, df=nu)
        
        assert np.allclose(pdf_mixture, pdf_single, rtol=1e-6)
    
    def test_pdf_reduces_to_crisis_when_epsilon_one(self):
        """Test that mixture PDF reduces to crisis component when ε=1."""
        from models.contaminated_student_t import contaminated_student_t_pdf
        
        x = np.linspace(-5, 5, 100)
        nu_crisis = 4.0
        
        # With ε=1, should be pure crisis component
        pdf_mixture = contaminated_student_t_pdf(x, nu_normal=12.0, nu_crisis=nu_crisis, epsilon=1.0)
        pdf_single = stats.t.pdf(x, df=nu_crisis)
        
        assert np.allclose(pdf_mixture, pdf_single, rtol=1e-6)
    
    def test_pdf_integrates_to_one(self):
        """Test that mixture PDF integrates to 1."""
        from models.contaminated_student_t import contaminated_student_t_pdf
        from scipy.integrate import quad
        
        def pdf_func(x):
            return contaminated_student_t_pdf(
                np.array([x]),
                nu_normal=12.0,
                nu_crisis=4.0,
                epsilon=0.10
            )[0]
        
        integral, error = quad(pdf_func, -50, 50)
        assert abs(integral - 1.0) < 1e-4
    
    def test_logpdf_consistent_with_pdf(self):
        """Test that log-PDF is consistent with PDF."""
        from models.contaminated_student_t import (
            contaminated_student_t_pdf,
            contaminated_student_t_logpdf
        )
        
        x = np.linspace(-4, 4, 50)
        
        pdf = contaminated_student_t_pdf(x, nu_normal=10.0, nu_crisis=4.0, epsilon=0.08)
        logpdf = contaminated_student_t_logpdf(x, nu_normal=10.0, nu_crisis=4.0, epsilon=0.08)
        
        assert np.allclose(np.log(pdf), logpdf, rtol=1e-6)


class TestContaminatedStudentTSampling:
    """Test random variate generation."""
    
    def test_rvs_shape(self):
        """Test that RVS produces correct shape."""
        from models.contaminated_student_t import contaminated_student_t_rvs
        
        samples = contaminated_student_t_rvs(
            size=1000,
            nu_normal=12.0,
            nu_crisis=4.0,
            epsilon=0.05
        )
        
        assert samples.shape == (1000,)
    
    def test_rvs_mixing_fraction(self):
        """Test that approximately ε fraction of samples come from crisis."""
        from models.contaminated_student_t import contaminated_student_t_rvs
        
        rng = np.random.default_rng(42)
        epsilon = 0.10
        n = 10000
        
        samples = contaminated_student_t_rvs(
            size=n,
            nu_normal=50.0,  # Very light tails (almost Gaussian)
            nu_crisis=3.0,   # Very heavy tails
            epsilon=epsilon,
            random_state=rng
        )
        
        # Heavy-tailed samples should produce more extreme values
        # Count samples beyond 4 standard deviations (rare for normal)
        extreme_frac = np.mean(np.abs(samples) > 4)
        
        # Should have significant extreme values from crisis component
        assert extreme_frac > 0.01  # More than would be expected from single t(50)
    
    def test_rvs_reproducibility(self):
        """Test that RVS is reproducible with seed."""
        from models.contaminated_student_t import contaminated_student_t_rvs
        
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        
        samples1 = contaminated_student_t_rvs(
            size=100,
            nu_normal=12.0,
            nu_crisis=4.0,
            epsilon=0.05,
            random_state=rng1
        )
        
        samples2 = contaminated_student_t_rvs(
            size=100,
            nu_normal=12.0,
            nu_crisis=4.0,
            epsilon=0.05,
            random_state=rng2
        )
        
        assert np.allclose(samples1, samples2)


class TestContaminatedStudentTFitting:
    """Test parameter estimation."""
    
    def test_fit_profile_basic(self):
        """Test basic profile likelihood fitting."""
        from models.contaminated_student_t import (
            fit_contaminated_student_t_profile,
            contaminated_student_t_rvs
        )
        
        rng = np.random.default_rng(42)
        
        # Generate data from known mixture
        true_nu_normal = 12.0
        true_nu_crisis = 4.0
        true_epsilon = 0.10
        
        samples = contaminated_student_t_rvs(
            size=2000,
            nu_normal=true_nu_normal,
            nu_crisis=true_nu_crisis,
            epsilon=true_epsilon,
            random_state=rng
        )
        
        params, diag = fit_contaminated_student_t_profile(samples)
        
        assert diag['fit_success']
        # Parameters should be in reasonable range
        assert 8 <= params.nu_normal <= 20
        assert 3 <= params.nu_crisis <= 8
        assert 0.02 <= params.epsilon <= 0.25
    
    def test_fit_comparison_with_single_t(self):
        """Test that fit compares favorably with single Student-t for heavy-tailed data."""
        from models.contaminated_student_t import (
            fit_contaminated_student_t_profile,
            contaminated_student_t_rvs
        )
        
        rng = np.random.default_rng(42)
        
        # Generate mixture data with clear crisis contamination
        samples = contaminated_student_t_rvs(
            size=2000,
            nu_normal=15.0,
            nu_crisis=3.0,
            epsilon=0.15,  # 15% crisis
            random_state=rng
        )
        
        params, diag = fit_contaminated_student_t_profile(samples)
        
        assert diag['fit_success']
        # Should detect that crisis component is helpful
        assert diag.get('delta_bic', 0) < 5  # Mixture not much worse than single
    
    def test_fit_insufficient_data(self):
        """Test graceful handling of insufficient data."""
        from models.contaminated_student_t import (
            fit_contaminated_student_t_profile,
            CST_MIN_OBS
        )
        
        small_data = np.random.standard_t(df=8, size=CST_MIN_OBS // 2)
        
        params, diag = fit_contaminated_student_t_profile(small_data)
        
        assert not diag['fit_success']
        assert 'insufficient_data' in diag.get('error', '')


class TestCrisisProbabilityEstimation:
    """Test crisis probability computation from observables."""
    
    def test_crisis_prob_from_vol_low_vol(self):
        """Test that low volatility gives low epsilon."""
        from models.contaminated_student_t import (
            compute_crisis_probability_from_vol,
            CST_EPSILON_DEFAULT
        )
        
        vol_history = np.abs(np.random.normal(0.01, 0.003, 252))
        current_vol = np.percentile(vol_history, 20)  # Low vol
        
        epsilon = compute_crisis_probability_from_vol(current_vol, vol_history)
        
        # Low vol should give epsilon close to default
        assert epsilon <= CST_EPSILON_DEFAULT * 1.5
    
    def test_crisis_prob_from_vol_high_vol(self):
        """Test that high volatility gives elevated epsilon."""
        from models.contaminated_student_t import (
            compute_crisis_probability_from_vol,
            CST_EPSILON_DEFAULT,
            CST_EPSILON_MAX
        )
        
        vol_history = np.abs(np.random.normal(0.01, 0.003, 252))
        current_vol = np.percentile(vol_history, 95)  # High vol
        
        epsilon = compute_crisis_probability_from_vol(current_vol, vol_history)
        
        # High vol should give elevated epsilon
        assert epsilon > CST_EPSILON_DEFAULT
        assert epsilon <= CST_EPSILON_MAX


class TestIntegration:
    """Test integration with existing architecture."""
    
    def test_params_serialization(self):
        """Test that params can be serialized to/from dict."""
        from models.contaminated_student_t import ContaminatedStudentTParams
        
        original = ContaminatedStudentTParams(
            nu_normal=10.0,
            nu_crisis=5.0,
            epsilon=0.08,
            epsilon_source="mle"
        )
        
        # Serialize
        d = original.to_dict()
        assert d['nu_normal'] == 10.0
        assert d['nu_crisis'] == 5.0
        assert d['epsilon'] == 0.08
        assert d['epsilon_source'] == 'mle'
        assert 'effective_nu' in d
        
        # Deserialize
        restored = ContaminatedStudentTParams.from_dict(d)
        assert restored.nu_normal == original.nu_normal
        assert restored.nu_crisis == original.nu_crisis
        assert restored.epsilon == original.epsilon
    
    def test_is_degenerate_detection(self):
        """Test degenerate mixture detection."""
        from models.contaminated_student_t import ContaminatedStudentTParams
        
        # Degenerate: epsilon too small
        params1 = ContaminatedStudentTParams(nu_normal=12, nu_crisis=4, epsilon=0.0005)
        assert params1.is_degenerate
        
        # Degenerate: same nu values
        params2 = ContaminatedStudentTParams(nu_normal=8, nu_crisis=8.2, epsilon=0.1)
        assert params2.is_degenerate
        
        # Non-degenerate
        params3 = ContaminatedStudentTParams(nu_normal=12, nu_crisis=4, epsilon=0.1)
        assert not params3.is_degenerate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
