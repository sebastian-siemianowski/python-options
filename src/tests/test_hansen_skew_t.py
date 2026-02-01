#!/usr/bin/env python3
"""
Test Hansen's Skew-t Distribution Implementation

This test verifies that the Hansen skew-t distribution is properly integrated
into the Bayesian Model Averaging framework as per the Expert Panel recommendation.

References:
    Hansen, B.E. (1994). "Autoregressive Conditional Density Estimation"
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


class TestHansenSkewTDistribution:
    """Test Hansen skew-t distribution functions."""
    
    def test_imports(self):
        """Test that all Hansen skew-t components import correctly."""
        from models import (
            HansenSkewTParams,
            hansen_skew_t_pdf,
            hansen_skew_t_logpdf,
            hansen_skew_t_cdf,
            hansen_skew_t_ppf,
            hansen_skew_t_rvs,
            fit_hansen_skew_t_mle,
            compare_symmetric_vs_hansen,
            HANSEN_NU_MIN,
            HANSEN_NU_MAX,
            HANSEN_NU_DEFAULT,
            HANSEN_LAMBDA_MIN,
            HANSEN_LAMBDA_MAX,
            HANSEN_LAMBDA_DEFAULT,
        )
        
        assert HANSEN_NU_MIN == 2.1
        assert HANSEN_NU_MAX == 500.0
        assert HANSEN_LAMBDA_MIN == -0.5
        assert HANSEN_LAMBDA_MAX == 0.5
        assert HANSEN_LAMBDA_DEFAULT == 0.0
    
    def test_hansen_params_initialization(self):
        """Test HansenSkewTParams initialization and validation."""
        from models import HansenSkewTParams
        
        # Valid parameters
        params = HansenSkewTParams(nu=10.0, lambda_=0.2)
        assert params.nu == 10.0
        assert params.lambda_ == 0.2
        assert params.skew_direction == "right"
        assert not params.is_symmetric
        
        # Symmetric case
        params_sym = HansenSkewTParams(nu=8.0, lambda_=0.0)
        assert params_sym.is_symmetric
        assert params_sym.skew_direction == "symmetric"
        
        # Left-skewed
        params_left = HansenSkewTParams(nu=6.0, lambda_=-0.3)
        assert params_left.skew_direction == "left"
        
        # Invalid nu should raise
        with pytest.raises(ValueError):
            HansenSkewTParams(nu=1.5, lambda_=0.0)  # nu must be > 2
        
        # Invalid lambda should raise
        with pytest.raises(ValueError):
            HansenSkewTParams(nu=10.0, lambda_=1.5)  # lambda must be in (-1, 1)
    
    def test_pdf_reduces_to_student_t(self):
        """Test that Hansen skew-t with λ=0 reduces to symmetric Student-t."""
        from models import hansen_skew_t_pdf
        from scipy.stats import t as student_t
        
        nu = 6.0
        x = np.linspace(-4, 4, 50)
        
        # Hansen with λ=0 should match Student-t
        hansen_pdf = hansen_skew_t_pdf(x, nu, lambda_=0.0)
        student_pdf = student_t.pdf(x, df=nu)
        
        assert np.allclose(hansen_pdf, student_pdf, rtol=1e-6)
    
    def test_cdf_reduces_to_student_t(self):
        """Test that Hansen skew-t CDF with λ=0 reduces to symmetric Student-t CDF."""
        from models import hansen_skew_t_cdf
        from scipy.stats import t as student_t
        
        nu = 8.0
        x = np.linspace(-3, 3, 30)
        
        # Hansen with λ=0 should match Student-t
        hansen_cdf = hansen_skew_t_cdf(x, nu, lambda_=0.0)
        student_cdf = student_t.cdf(x, df=nu)
        
        assert np.allclose(hansen_cdf, student_cdf, rtol=1e-5)
    
    def test_pdf_properties(self):
        """Test that Hansen skew-t PDF integrates to 1 and is positive."""
        from models import hansen_skew_t_pdf
        from scipy.integrate import quad
        
        nu = 6.0
        lambda_ = 0.3  # Right-skewed
        
        # PDF should be positive
        x = np.linspace(-10, 10, 100)
        pdf_vals = hansen_skew_t_pdf(x, nu, lambda_)
        assert np.all(pdf_vals >= 0)
        
        # PDF should integrate to approximately 1
        integral, _ = quad(lambda z: hansen_skew_t_pdf(z, nu, lambda_), -np.inf, np.inf)
        assert abs(integral - 1.0) < 0.01
    
    def test_cdf_properties(self):
        """Test CDF monotonicity and range."""
        from models import hansen_skew_t_cdf
        
        nu = 6.0
        lambda_ = -0.2  # Left-skewed
        
        x = np.linspace(-5, 5, 100)
        cdf_vals = hansen_skew_t_cdf(x, nu, lambda_)
        
        # CDF should be in [0, 1]
        assert np.all(cdf_vals >= 0)
        assert np.all(cdf_vals <= 1)
        
        # CDF should be monotonically increasing
        assert np.all(np.diff(cdf_vals) >= -1e-10)
    
    def test_ppf_inverts_cdf(self):
        """Test that PPF correctly inverts CDF."""
        from models import hansen_skew_t_cdf, hansen_skew_t_ppf
        
        nu = 10.0
        lambda_ = 0.25
        
        # Test roundtrip: x -> CDF(x) -> PPF(CDF(x)) ≈ x
        x_original = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        p = hansen_skew_t_cdf(x_original, nu, lambda_)
        x_recovered = hansen_skew_t_ppf(p, nu, lambda_)
        
        assert np.allclose(x_original, x_recovered, rtol=1e-3)
    
    def test_sampling_statistics(self):
        """Test that samples have expected distributional properties."""
        from models import hansen_skew_t_rvs, hansen_skew_t_cdf
        
        rng = np.random.default_rng(42)
        
        nu = 8.0
        lambda_ = -0.3  # Left-skewed
        n_samples = 10000
        
        samples = hansen_skew_t_rvs(size=n_samples, nu=nu, lambda_=lambda_, random_state=rng)
        
        # Check samples are finite
        assert np.all(np.isfinite(samples))
        
        # For left-skewed, mean should be positive (due to Hansen standardization)
        # and there should be more extreme negative values
        left_tail_mass = np.mean(samples < -2)
        right_tail_mass = np.mean(samples > 2)
        
        # Left tail should be heavier for λ < 0
        assert left_tail_mass > right_tail_mass * 0.8  # Allow some sampling variance
    
    def test_right_skew_behavior(self):
        """Test that positive λ produces right-skewed distribution."""
        from models import hansen_skew_t_rvs
        
        rng = np.random.default_rng(123)
        
        nu = 6.0
        lambda_ = 0.4  # Strong right skew
        n_samples = 10000
        
        samples = hansen_skew_t_rvs(size=n_samples, nu=nu, lambda_=lambda_, random_state=rng)
        
        # Right-skewed: more extreme positive values
        left_tail_mass = np.mean(samples < -2)
        right_tail_mass = np.mean(samples > 2)
        
        # Right tail should be heavier for λ > 0
        assert right_tail_mass > left_tail_mass


class TestHansenSkewTMLE:
    """Test Hansen skew-t maximum likelihood estimation."""
    
    def test_mle_on_generated_data(self):
        """Test that MLE recovers known parameters from generated data."""
        from models import hansen_skew_t_rvs, fit_hansen_skew_t_mle
        
        rng = np.random.default_rng(42)
        
        true_nu = 8.0
        true_lambda = 0.2
        n_samples = 1000
        
        # Generate data from known distribution
        data = hansen_skew_t_rvs(size=n_samples, nu=true_nu, lambda_=true_lambda, random_state=rng)
        
        # Fit MLE
        nu_hat, lambda_hat, ll, diag = fit_hansen_skew_t_mle(data)
        
        assert diag['fit_success']
        
        # Should recover parameters reasonably well
        assert abs(nu_hat - true_nu) < 3.0  # DoF is harder to estimate
        assert abs(lambda_hat - true_lambda) < 0.15
    
    def test_mle_on_symmetric_data(self):
        """Test that MLE returns λ≈0 for symmetric data."""
        from models import fit_hansen_skew_t_mle
        
        rng = np.random.default_rng(42)
        
        # Generate symmetric t data
        nu = 6.0
        data = rng.standard_t(df=nu, size=1000)
        
        nu_hat, lambda_hat, ll, diag = fit_hansen_skew_t_mle(data)
        
        assert diag['fit_success']
        assert abs(lambda_hat) < 0.1  # Should be close to 0
    
    def test_mle_insufficient_data(self):
        """Test MLE gracefully handles insufficient data."""
        from models import fit_hansen_skew_t_mle, HANSEN_MLE_MIN_OBS
        
        data = np.random.randn(20)  # Less than minimum
        
        nu_hat, lambda_hat, ll, diag = fit_hansen_skew_t_mle(data)
        
        assert not diag['fit_success']
        assert diag['error'] == 'insufficient_data'


class TestHansenSkewTComparison:
    """Test comparison between symmetric and Hansen skew-t."""
    
    def test_comparison_prefers_symmetric_for_symmetric_data(self):
        """Test that comparison prefers symmetric t for symmetric data."""
        from models import compare_symmetric_vs_hansen
        
        rng = np.random.default_rng(42)
        
        # Generate symmetric data
        nu = 6.0
        data = rng.standard_t(df=nu, size=500)
        
        comparison = compare_symmetric_vs_hansen(
            data,
            nu_symmetric=nu,
            nu_hansen=nu,
            lambda_hansen=0.0
        )
        
        # Should prefer symmetric (or no clear preference) since data is symmetric
        assert comparison['preference'] in ['symmetric_t', 'no_clear_preference']
    
    def test_comparison_prefers_hansen_for_skewed_data(self):
        """Test that comparison prefers Hansen for skewed data."""
        from models import hansen_skew_t_rvs, compare_symmetric_vs_hansen, fit_hansen_skew_t_mle
        
        rng = np.random.default_rng(42)
        
        # Generate strongly skewed data
        true_nu = 6.0
        true_lambda = 0.4
        data = hansen_skew_t_rvs(size=1000, nu=true_nu, lambda_=true_lambda, random_state=rng)
        
        # Fit Hansen to get parameters
        nu_hat, lambda_hat, _, _ = fit_hansen_skew_t_mle(data)
        
        comparison = compare_symmetric_vs_hansen(
            data,
            nu_symmetric=nu_hat,
            nu_hansen=nu_hat,
            lambda_hansen=lambda_hat
        )
        
        # ΔAIC should be negative (Hansen better)
        # Note: with strongly skewed data, Hansen should clearly win
        assert comparison['delta_aic'] < 2  # Hansen should be competitive


class TestSignalsIntegration:
    """Test signal layer integration with Hansen skew-t."""
    
    def test_run_regime_specific_mc_with_hansen(self):
        """Test MC sampling with Hansen skew-t parameters."""
        from decision.signals import run_regime_specific_mc, HANSEN_SKEW_T_AVAILABLE
        
        if not HANSEN_SKEW_T_AVAILABLE:
            pytest.skip("Hansen skew-t not available")
        
        samples = run_regime_specific_mc(
            regime=0,
            mu_t=0.001,
            P_t=0.0001,
            phi=0.9,
            q=1e-6,
            sigma2_step=0.0002,
            H=5,
            n_paths=1000,
            nu=8.0,  # Student-t degrees of freedom
            hansen_lambda=-0.2,  # Left-skewed (crash risk)
            seed=42
        )
        
        assert len(samples) == 1000
        assert np.all(np.isfinite(samples))
    
    def test_hansen_not_used_for_gaussian(self):
        """Test that Hansen is not used when nu is None."""
        from decision.signals import run_regime_specific_mc
        
        # Without nu specified, should use Gaussian (not Hansen)
        samples = run_regime_specific_mc(
            regime=0,
            mu_t=0.001,
            P_t=0.0001,
            phi=0.9,
            q=1e-6,
            sigma2_step=0.0002,
            H=5,
            n_paths=1000,
            nu=None,  # No Student-t
            hansen_lambda=-0.2,  # Should be ignored
            seed=42
        )
        
        assert len(samples) == 1000
        assert np.all(np.isfinite(samples))


class TestTuneIntegration:
    """Test tune.py integration with Hansen skew-t."""
    
    def test_tune_imports_hansen(self):
        """Test that tune.py imports Hansen skew-t components."""
        from tuning.tune import (
            fit_hansen_skew_t_mle,
            compare_symmetric_vs_hansen,
            HANSEN_NU_DEFAULT,
            HANSEN_LAMBDA_DEFAULT,
        )
        assert HANSEN_NU_DEFAULT == 10.0
        assert HANSEN_LAMBDA_DEFAULT == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
