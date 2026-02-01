#!/usr/bin/env python3
"""
Test NIG (Normal-Inverse Gaussian) Model Integration with BMA

This test verifies that the NIG model is properly integrated into
the Bayesian Model Averaging framework as Solution 2.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


class TestPhiNIGModel:
    """Test NIG model functionality."""
    
    def test_imports(self):
        """Test that all NIG components import correctly."""
        from models import (
            PhiNIGDriftModel,
            NIG_ALPHA_GRID,
            NIG_BETA_RATIO_GRID,
            NIG_ALPHA_MIN,
            NIG_ALPHA_MAX,
            NIG_ALPHA_DEFAULT,
            NIG_BETA_DEFAULT,
            NIG_DELTA_MIN,
            NIG_DELTA_MAX,
            NIG_DELTA_DEFAULT,
            is_nig_model,
            get_nig_model_name,
            parse_nig_model_name,
        )
        
        assert PhiNIGDriftModel is not None
        assert NIG_ALPHA_GRID == [1.5, 2.5, 4.0, 8.0]
        assert NIG_BETA_RATIO_GRID == [-0.3, 0.0, 0.3]
        assert NIG_ALPHA_MIN == 0.5
        assert NIG_ALPHA_MAX == 50.0
        assert NIG_ALPHA_DEFAULT == 2.0
        assert NIG_BETA_DEFAULT == 0.0
    
    def test_model_naming(self):
        """Test model name generation and parsing."""
        from models import get_nig_model_name, parse_nig_model_name, is_nig_model
        
        # Test name generation
        name = get_nig_model_name(2.5, -0.75)
        assert "phi_nig_alpha_" in name
        assert "beta_" in name
        
        # Test is_nig_model
        assert is_nig_model("phi_nig_alpha_2p5_beta_m0p75") == True
        assert is_nig_model("phi_student_t_nu_6") == False
        assert is_nig_model("kalman_gaussian") == False
    
    def test_nig_sampling(self):
        """Test that NIG sampling produces expected behavior."""
        from models import PhiNIGDriftModel
        
        rng = np.random.default_rng(42)
        
        # Symmetric NIG (beta = 0)
        samples_sym = PhiNIGDriftModel.sample(
            alpha=2.0, beta=0.0, delta=0.01, mu=0, size=10000, rng=rng
        )
        skew_sym = np.mean(((samples_sym - np.mean(samples_sym))/np.std(samples_sym))**3)
        assert abs(skew_sym) < 0.3, f"Symmetric should have ~0 skewness, got {skew_sym}"
        
        # Left-skewed NIG (beta < 0)
        samples_left = PhiNIGDriftModel.sample(
            alpha=2.0, beta=-0.5, delta=0.01, mu=0, size=10000, rng=rng
        )
        skew_left = np.mean(((samples_left - np.mean(samples_left))/np.std(samples_left))**3)
        assert skew_left < 0, f"Left-skewed should have negative skewness, got {skew_left}"
        
        # Right-skewed NIG (beta > 0)
        samples_right = PhiNIGDriftModel.sample(
            alpha=2.0, beta=0.5, delta=0.01, mu=0, size=10000, rng=rng
        )
        skew_right = np.mean(((samples_right - np.mean(samples_right))/np.std(samples_right))**3)
        assert skew_right > 0, f"Right-skewed should have positive skewness, got {skew_right}"
    
    def test_nig_pdf_cdf(self):
        """Test NIG PDF and CDF properties."""
        from models import PhiNIGDriftModel
        
        # CDF should be in [0, 1]
        for x in [-0.1, -0.01, 0, 0.01, 0.1]:
            cdf_val = PhiNIGDriftModel.cdf(x, alpha=2.0, beta=0.0, delta=0.01)
            assert 0 <= cdf_val <= 1, f"CDF({x}) = {cdf_val} not in [0,1]"
        
        # CDF should be monotone increasing
        x_vals = np.linspace(-0.1, 0.1, 50)
        cdf_vals = [PhiNIGDriftModel.cdf(x, alpha=2.0, beta=0.0, delta=0.01) for x in x_vals]
        assert all(cdf_vals[i] <= cdf_vals[i+1] + 1e-10 for i in range(len(cdf_vals)-1)), "CDF not monotone"
        
        # PDF should be positive
        for x in [-0.1, 0, 0.1]:
            pdf_val = PhiNIGDriftModel.pdf(x, alpha=2.0, beta=0.0, delta=0.01)
            assert pdf_val >= 0, f"PDF({x}) = {pdf_val} is negative"
    
    def test_nig_filter(self):
        """Test that NIG filter runs and produces valid output."""
        from models import PhiNIGDriftModel
        
        np.random.seed(42)
        n = 252  # ~1 year of daily data
        returns = np.random.standard_normal(n) * 0.015
        vol = np.ones(n) * 0.015
        
        # Run filter with NIG noise
        mu, P, ll = PhiNIGDriftModel.filter_phi(
            returns, vol, q=1e-6, c=1.0, phi=0.9,
            alpha=2.0, beta=-0.3, delta=0.01
        )
        
        assert len(mu) == n
        assert len(P) == n
        assert np.isfinite(ll)
        assert np.all(P > 0)  # Variance must be positive
    
    def test_nig_pit_calibration(self):
        """Test PIT calibration with NIG."""
        from models import PhiNIGDriftModel
        
        np.random.seed(42)
        n = 500
        returns = np.random.standard_normal(n) * 0.015
        vol = np.ones(n) * 0.015
        
        # Run filter
        mu, P, ll = PhiNIGDriftModel.filter_phi(
            returns, vol, q=1e-6, c=1.0, phi=0.9,
            alpha=2.0, beta=0.0, delta=0.01
        )
        
        # Compute PIT
        ks_stat, ks_p = PhiNIGDriftModel.pit_ks(
            returns, mu, vol, P, c=1.0,
            alpha=2.0, beta=0.0, delta=0.01
        )
        
        assert 0 <= ks_stat <= 1
        assert 0 <= ks_p <= 1
    
    def test_parameter_optimization(self):
        """Test parameter optimization with fixed NIG parameters."""
        from models import PhiNIGDriftModel
        
        np.random.seed(42)
        n = 300
        returns = np.random.standard_normal(n) * 0.015
        vol = np.ones(n) * 0.015
        
        q_opt, c_opt, phi_opt, cv_ll, diag = PhiNIGDriftModel.optimize_params_fixed_nig(
            returns, vol,
            alpha=2.0, beta=0.0, delta=0.01
        )
        
        assert q_opt > 0
        assert c_opt > 0
        assert -1 < phi_opt < 1
        assert np.isfinite(cv_ll)
        assert diag['fit_success'] == True
    
    def test_mle_fit(self):
        """Test NIG MLE fitting."""
        from models import PhiNIGDriftModel
        
        np.random.seed(42)
        # Generate some data
        data = np.random.standard_normal(500) * 0.02
        
        alpha, beta, delta, ll, diag = PhiNIGDriftModel.fit_mle(data)
        
        assert alpha > 0
        assert abs(beta) < alpha
        assert delta > 0
        assert np.isfinite(ll) or ll == float('-inf')


class TestBMAIntegration:
    """Test BMA integration with NIG models."""
    
    def test_tune_imports_nig(self):
        """Test that tune.py imports NIG components."""
        from tuning.tune import (
            PhiNIGDriftModel,
            NIG_ALPHA_GRID,
            NIG_BETA_RATIO_GRID,
            is_nig_model,
            get_nig_model_name,
        )
        assert PhiNIGDriftModel is not None
    
    def test_is_heavy_tailed_model(self):
        """Test that is_heavy_tailed_model recognizes NIG models."""
        from tuning.tune import is_heavy_tailed_model
        
        # Student-t models
        assert is_heavy_tailed_model("phi_student_t_nu_6") == True
        
        # Skew-t models
        assert is_heavy_tailed_model("phi_skew_t_nu_6_gamma_0p85") == True
        
        # NIG models
        assert is_heavy_tailed_model("phi_nig_alpha_2p5_beta_m0p3") == True
        
        # Non-heavy-tailed models
        assert is_heavy_tailed_model("kalman_gaussian") == False
        assert is_heavy_tailed_model("kalman_phi_gaussian") == False


class TestSignalIntegration:
    """Test signal layer integration with NIG models."""
    
    def test_run_regime_specific_mc_with_nig(self):
        """Test that MC sampling works with NIG parameters."""
        from decision.signals import run_regime_specific_mc
        
        samples = run_regime_specific_mc(
            regime=0,
            mu_t=0.001,
            P_t=0.0001,
            phi=0.9,
            q=1e-6,
            sigma2_step=0.0002,
            H=5,
            n_paths=1000,
            nu=None,  # Not using Student-t
            nig_alpha=2.0,
            nig_beta=-0.3,
            nig_delta=0.01,
            seed=42
        )
        
        assert len(samples) == 1000
        assert np.all(np.isfinite(samples))
        
        # Check that samples have reasonable statistics
        assert abs(np.mean(samples)) < 0.1  # Mean should be near 0
        assert 0.001 < np.std(samples) < 0.5  # Std should be reasonable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
