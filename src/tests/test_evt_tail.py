#!/usr/bin/env python3
"""
Test EVT (Extreme Value Theory) POT/GPD Implementation

This test verifies that the EVT tail module correctly implements:
1. GPD parameter estimation (MLE, Hill, PWM)
2. Conditional Tail Expectation calculation
3. Integration with position sizing (EVT-corrected expected loss)

References:
    Pickands, J. (1975). "Statistical Inference Using Extreme Order Statistics"
    McNeil, A.J. & Frey, R. (2000). "Estimation of Tail-Related Risk Measures"
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


class TestGPDBasics:
    """Test GPD distribution functions."""
    
    def test_imports(self):
        """Test that all EVT components import correctly."""
        from calibration.evt_tail import (
            GPDFitResult,
            gpd_pdf,
            gpd_cdf,
            gpd_quantile,
            compute_cte_gpd,
            fit_gpd_mle,
            fit_gpd_hill,
            fit_gpd_pwm,
            fit_gpd_pot,
            compute_evt_expected_loss,
            EVT_THRESHOLD_PERCENTILE_DEFAULT,
            EVT_MIN_EXCEEDANCES,
            EVT_XI_MIN,
            EVT_XI_MAX,
        )
        
        assert EVT_THRESHOLD_PERCENTILE_DEFAULT == 0.90
        assert EVT_MIN_EXCEEDANCES == 30
    
    def test_gpd_pdf_exponential_case(self):
        """Test GPD PDF reduces to exponential when xi=0."""
        from calibration.evt_tail import gpd_pdf
        
        x = np.linspace(0.1, 5, 50)
        sigma = 1.0
        xi = 0.0  # Exponential case
        
        pdf_vals = gpd_pdf(x, xi, sigma)
        expected_exp = (1.0 / sigma) * np.exp(-x / sigma)
        
        assert np.allclose(pdf_vals, expected_exp, rtol=1e-6)
    
    def test_gpd_cdf_exponential_case(self):
        """Test GPD CDF reduces to exponential when xi=0."""
        from calibration.evt_tail import gpd_cdf
        
        x = np.linspace(0.1, 5, 50)
        sigma = 1.0
        xi = 0.0
        
        cdf_vals = gpd_cdf(x, xi, sigma)
        expected_exp_cdf = 1.0 - np.exp(-x / sigma)
        
        assert np.allclose(cdf_vals, expected_exp_cdf, rtol=1e-6)
    
    def test_gpd_cdf_properties(self):
        """Test CDF is monotonic and in [0, 1]."""
        from calibration.evt_tail import gpd_cdf
        
        x = np.linspace(0.01, 10, 100)
        
        # Test for various xi values
        for xi in [-0.3, 0.0, 0.2, 0.5]:
            cdf_vals = gpd_cdf(x, xi, sigma=1.0)
            
            # Should be in [0, 1]
            assert np.all(cdf_vals >= 0)
            assert np.all(cdf_vals <= 1)
            
            # Should be monotonically increasing
            assert np.all(np.diff(cdf_vals) >= -1e-10)
    
    def test_gpd_quantile_inverts_cdf(self):
        """Test quantile correctly inverts CDF."""
        from calibration.evt_tail import gpd_cdf, gpd_quantile
        
        for xi in [0.0, 0.2, 0.4]:
            p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
            sigma = 1.5
            
            x = gpd_quantile(p, xi, sigma)
            p_recovered = gpd_cdf(x, xi, sigma)
            
            assert np.allclose(p, p_recovered, rtol=1e-4)


class TestCTE:
    """Test Conditional Tail Expectation calculation."""
    
    def test_cte_exponential_case(self):
        """Test CTE for exponential (xi=0): CTE = u + sigma."""
        from calibration.evt_tail import compute_cte_gpd
        
        threshold = 1.0
        sigma = 0.5
        xi = 0.0
        
        cte = compute_cte_gpd(threshold, xi, sigma)
        expected = threshold + sigma / (1 - xi)  # = u + sigma for xi=0
        
        assert abs(cte - expected) < 1e-10
    
    def test_cte_increases_with_xi(self):
        """Test CTE increases with heavier tails (larger xi)."""
        from calibration.evt_tail import compute_cte_gpd
        
        threshold = 1.0
        sigma = 0.5
        
        cte_light = compute_cte_gpd(threshold, xi=-0.2, sigma=sigma)
        cte_exp = compute_cte_gpd(threshold, xi=0.0, sigma=sigma)
        cte_heavy = compute_cte_gpd(threshold, xi=0.3, sigma=sigma)
        
        assert cte_light < cte_exp < cte_heavy
    
    def test_cte_with_infinite_mean(self):
        """Test CTE handling when xi >= 1 (infinite mean)."""
        from calibration.evt_tail import compute_cte_gpd, EVT_FALLBACK_MULTIPLIER
        
        threshold = 1.0
        sigma = 0.5
        xi = 1.5  # Infinite mean case
        
        cte = compute_cte_gpd(threshold, xi, sigma)
        
        # Should return conservative fallback
        assert cte > threshold


class TestGPDFitting:
    """Test GPD parameter estimation methods."""
    
    def test_fit_mle_on_exponential_data(self):
        """Test MLE correctly identifies exponential (xi≈0) data."""
        from calibration.evt_tail import fit_gpd_mle
        
        rng = np.random.default_rng(42)
        
        # Generate exponential data (xi=0)
        sigma_true = 1.0
        exceedances = rng.exponential(scale=sigma_true, size=500)
        
        xi_hat, sigma_hat, success, diag = fit_gpd_mle(exceedances)
        
        assert success
        assert abs(xi_hat) < 0.15  # Should be close to 0
        assert abs(sigma_hat - sigma_true) < 0.2
    
    def test_fit_mle_on_pareto_data(self):
        """Test MLE on Pareto (heavy-tailed) data."""
        from calibration.evt_tail import fit_gpd_mle
        
        rng = np.random.default_rng(42)
        
        # Generate Pareto data: GPD with xi > 0
        xi_true = 0.3
        sigma_true = 1.0
        n = 500
        
        # Pareto sample via inverse CDF
        u = rng.uniform(0, 1, n)
        exceedances = (sigma_true / xi_true) * (np.power(1 - u, -xi_true) - 1)
        
        xi_hat, sigma_hat, success, diag = fit_gpd_mle(exceedances)
        
        assert success
        assert abs(xi_hat - xi_true) < 0.15
    
    def test_fit_hill_estimator(self):
        """Test Hill estimator for shape parameter."""
        from calibration.evt_tail import fit_gpd_hill
        
        rng = np.random.default_rng(42)
        
        # Generate heavy-tailed data
        xi_true = 0.25
        n = 500
        
        # Pareto sample
        u = rng.uniform(0, 1, n)
        exceedances = np.power(1 - u, -xi_true) - 1
        
        xi_hat, sigma_hat, success, diag = fit_gpd_hill(exceedances)
        
        assert success
        assert abs(xi_hat - xi_true) < 0.2
    
    def test_fit_pwm(self):
        """Test Probability Weighted Moments estimator."""
        from calibration.evt_tail import fit_gpd_pwm
        
        rng = np.random.default_rng(42)
        
        # Generate exponential data
        exceedances = rng.exponential(scale=1.0, size=500)
        
        xi_hat, sigma_hat, success, diag = fit_gpd_pwm(exceedances)
        
        assert success
        assert abs(xi_hat) < 0.2  # Should be close to 0


class TestPOTFitting:
    """Test full POT/GPD fitting pipeline."""
    
    def test_fit_gpd_pot_basic(self):
        """Test POT fitting on synthetic loss data."""
        from calibration.evt_tail import fit_gpd_pot, GPDFitResult
        
        rng = np.random.default_rng(42)
        
        # Generate losses with moderate tails
        losses = np.abs(rng.standard_t(df=6, size=1000))
        
        result = fit_gpd_pot(losses, threshold_percentile=0.90)
        
        assert isinstance(result, GPDFitResult)
        assert result.fit_success
        assert result.n_exceedances > 50
        assert 0 < result.xi < 0.5  # Moderate tails expected
        assert result.cte > result.threshold
    
    def test_fit_gpd_pot_insufficient_data(self):
        """Test graceful handling of insufficient data."""
        from calibration.evt_tail import fit_gpd_pot, EVT_MIN_EXCEEDANCES
        
        losses = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Too few
        
        result = fit_gpd_pot(losses)
        
        assert not result.fit_success
        assert result.method == 'fallback'
    
    def test_fit_gpd_pot_student_t_consistency(self):
        """Test that GPD xi is consistent with Student-t nu."""
        from calibration.evt_tail import fit_gpd_pot, check_student_t_consistency
        
        rng = np.random.default_rng(42)
        
        # Generate Student-t losses with known nu
        nu_true = 5.0
        xi_expected = 1.0 / nu_true  # = 0.2
        
        losses = np.abs(rng.standard_t(df=nu_true, size=2000))
        
        result = fit_gpd_pot(losses, threshold_percentile=0.90)
        
        if result.fit_success:
            consistency = check_student_t_consistency(nu_true, result.xi)
            # Should be reasonably consistent
            assert consistency.get('relative_difference', 1.0) < 0.5


class TestExpectedLoss:
    """Test EVT-corrected expected loss calculation."""
    
    def test_evt_expected_loss_basic(self):
        """Test EVT expected loss is more conservative than empirical."""
        from calibration.evt_tail import compute_evt_expected_loss
        
        rng = np.random.default_rng(42)
        
        # Generate return samples with heavy tails
        r_samples = rng.standard_t(df=5, size=5000)
        
        evt_loss, emp_loss, gpd_result = compute_evt_expected_loss(r_samples)
        
        # EVT loss should be >= empirical (more conservative)
        assert evt_loss >= emp_loss * 0.99  # Allow small numerical tolerance
    
    def test_evt_expected_loss_light_tails(self):
        """Test that EVT makes minimal adjustment for light-tailed data."""
        from calibration.evt_tail import compute_evt_expected_loss
        
        rng = np.random.default_rng(42)
        
        # Generate Gaussian returns (light tails)
        r_samples = rng.normal(0, 0.01, size=5000)
        
        evt_loss, emp_loss, gpd_result = compute_evt_expected_loss(r_samples)
        
        # For light tails (xi ≈ 0), adjustment should be modest
        if gpd_result.fit_success:
            assert gpd_result.xi < 0.1
            # EVT loss shouldn't be dramatically larger
            assert evt_loss < emp_loss * 2.0
    
    def test_evt_expected_loss_heavy_tails(self):
        """Test that EVT makes significant adjustment for heavy-tailed data."""
        from calibration.evt_tail import compute_evt_expected_loss
        
        rng = np.random.default_rng(42)
        
        # Generate Student-t returns with heavy tails (nu=4)
        r_samples = rng.standard_t(df=4, size=5000) * 0.02
        
        evt_loss, emp_loss, gpd_result = compute_evt_expected_loss(r_samples)
        
        # For heavy tails, EVT should give notably larger loss
        if gpd_result.fit_success and gpd_result.xi > 0.15:
            assert evt_loss > emp_loss * 1.1


class TestIntegration:
    """Test integration with existing architecture."""
    
    def test_gpd_result_serialization(self):
        """Test GPDFitResult can be serialized to/from dict."""
        from calibration.evt_tail import GPDFitResult
        
        result = GPDFitResult(
            xi=0.25,
            sigma=0.5,
            threshold=1.0,
            n_exceedances=100,
            cte=1.75,
            fit_success=True,
            method='mle',
            diagnostics={'log_likelihood': -150.5}
        )
        
        # Serialize
        d = result.to_dict()
        assert d['xi'] == 0.25
        assert d['sigma'] == 0.5
        assert d['fit_success'] == True
        
        # Deserialize
        result2 = GPDFitResult.from_dict(d)
        assert result2.xi == result.xi
        assert result2.sigma == result.sigma
        assert result2.cte == result.cte


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
