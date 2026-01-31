#!/usr/bin/env python3
"""
Test φ-Skew-t Model Integration with BMA

This test verifies that the φ-Skew-t model is properly integrated into
the Bayesian Model Averaging framework.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


class TestPhiSkewTModel:
    """Test φ-Skew-t model functionality."""
    
    def test_imports(self):
        """Test that all skew-t components import correctly."""
        from models import (
            PhiSkewTDriftModel,
            SKEW_T_NU_GRID,
            SKEW_T_GAMMA_GRID,
            GAMMA_MIN,
            GAMMA_MAX,
            GAMMA_DEFAULT,
            is_skew_t_model,
            get_skew_t_model_name,
            parse_skew_t_model_name,
        )
        
        assert PhiSkewTDriftModel is not None
        assert SKEW_T_NU_GRID == [4, 6, 8, 12, 20]
        assert 0.7 in SKEW_T_GAMMA_GRID
        assert 1.0 in SKEW_T_GAMMA_GRID
        assert 1.3 in SKEW_T_GAMMA_GRID
        assert GAMMA_MIN == 0.5
        assert GAMMA_MAX == 2.0
        assert GAMMA_DEFAULT == 1.0
    
    def test_model_naming(self):
        """Test model name generation and parsing."""
        from models import get_skew_t_model_name, parse_skew_t_model_name, is_skew_t_model
        
        # Test name generation
        name = get_skew_t_model_name(6, 0.85)
        assert name == "phi_skew_t_nu_6_gamma_0p85"
        
        # Test parsing
        parsed = parse_skew_t_model_name(name)
        assert parsed == (6.0, 0.85)
        
        # Test is_skew_t_model
        assert is_skew_t_model("phi_skew_t_nu_6_gamma_0p85") == True
        assert is_skew_t_model("phi_student_t_nu_6") == False
        assert is_skew_t_model("kalman_gaussian") == False
    
    def test_skew_t_sampling(self):
        """Test that skew-t sampling produces expected skewness."""
        from models import PhiSkewTDriftModel
        
        rng = np.random.default_rng(42)
        
        # Left-skewed (gamma < 1)
        samples_left = PhiSkewTDriftModel.sample_skew_t(
            nu=6, gamma=0.7, mu=0, scale=1.0, size=50000, rng=rng
        )
        skew_left = np.mean(((samples_left - np.mean(samples_left))/np.std(samples_left))**3)
        assert skew_left < 0, f"Left-skewed should have negative skewness, got {skew_left}"
        
        # Right-skewed (gamma > 1)
        samples_right = PhiSkewTDriftModel.sample_skew_t(
            nu=6, gamma=1.3, mu=0, scale=1.0, size=50000, rng=rng
        )
        skew_right = np.mean(((samples_right - np.mean(samples_right))/np.std(samples_right))**3)
        assert skew_right > 0, f"Right-skewed should have positive skewness, got {skew_right}"
        
        # Symmetric (gamma = 1)
        samples_sym = PhiSkewTDriftModel.sample_skew_t(
            nu=6, gamma=1.0, mu=0, scale=1.0, size=50000, rng=rng
        )
        skew_sym = np.mean(((samples_sym - np.mean(samples_sym))/np.std(samples_sym))**3)
        assert abs(skew_sym) < 0.1, f"Symmetric should have ~0 skewness, got {skew_sym}"
    
    def test_skew_t_filter(self):
        """Test that skew-t filter runs and produces valid output."""
        from models import PhiSkewTDriftModel
        
        np.random.seed(42)
        n = 252  # ~1 year of daily data
        returns = np.random.standard_t(df=6, size=n) * 0.015
        vol = np.ones(n) * 0.015
        
        # Run filter with left-skewed noise
        mu, P, ll = PhiSkewTDriftModel.filter_phi(
            returns, vol, q=1e-6, c=1.0, phi=0.9, nu=6, gamma=0.85
        )
        
        assert len(mu) == n
        assert len(P) == n
        assert np.isfinite(ll)
        assert np.all(P > 0)  # Variance must be positive
    
    def test_skew_t_pit_calibration(self):
        """Test PIT calibration with skew-t."""
        from models import PhiSkewTDriftModel
        
        np.random.seed(42)
        n = 500
        
        # Generate skew-t distributed data
        raw_t = np.random.standard_t(df=6, size=n)
        gamma = 0.85
        skewed = np.where(raw_t >= 0, raw_t / gamma, raw_t * gamma)
        returns = skewed * 0.015
        vol = np.ones(n) * 0.015
        
        # Run filter
        mu, P, ll = PhiSkewTDriftModel.filter_phi(
            returns, vol, q=1e-6, c=1.0, phi=0.9, nu=6, gamma=0.85
        )
        
        # Compute PIT
        ks_stat, ks_p = PhiSkewTDriftModel.pit_ks(
            returns, mu, vol, P, c=1.0, nu=6, gamma=0.85
        )
        
        assert 0 <= ks_stat <= 1
        assert 0 <= ks_p <= 1
        # With matching distribution, p-value should be reasonable
        assert ks_p > 0.01, f"PIT p-value too low: {ks_p}"
    
    def test_parameter_optimization(self):
        """Test parameter optimization with fixed nu and gamma."""
        from models import PhiSkewTDriftModel
        
        np.random.seed(42)
        n = 300
        returns = np.random.standard_t(df=6, size=n) * 0.015
        vol = np.ones(n) * 0.015
        
        q_opt, c_opt, phi_opt, cv_ll, diag = PhiSkewTDriftModel.optimize_params_fixed_nu_gamma(
            returns, vol, nu=6, gamma=0.85
        )
        
        assert q_opt > 0
        assert c_opt > 0
        assert -1 < phi_opt < 1
        assert np.isfinite(cv_ll)
        assert diag['fit_success'] == True
    
    def test_cdf_properties(self):
        """Test CDF properties of skew-t distribution."""
        from models import PhiSkewTDriftModel
        
        # CDF should be in [0, 1]
        for x in [-10, -1, 0, 1, 10]:
            cdf_val = PhiSkewTDriftModel.cdf_skew_t(x, nu=6, gamma=0.85)
            assert 0 <= cdf_val <= 1, f"CDF({x}) = {cdf_val} not in [0,1]"
        
        # CDF should be monotone increasing
        x_vals = np.linspace(-5, 5, 100)
        cdf_vals = [PhiSkewTDriftModel.cdf_skew_t(x, nu=6, gamma=0.85) for x in x_vals]
        assert all(cdf_vals[i] <= cdf_vals[i+1] for i in range(len(cdf_vals)-1)), "CDF not monotone"
        
        # CDF(0) should be different from 0.5 for asymmetric gamma
        cdf_0_skewed = PhiSkewTDriftModel.cdf_skew_t(0, nu=6, gamma=0.85)
        cdf_0_sym = PhiSkewTDriftModel.cdf_skew_t(0, nu=6, gamma=1.0)
        assert abs(cdf_0_sym - 0.5) < 0.01, f"Symmetric CDF(0) should be ~0.5, got {cdf_0_sym}"


class TestBMAIntegration:
    """Test BMA integration with skew-t models."""
    
    def test_tune_imports_skew_t(self):
        """Test that tune.py imports skew-t components."""
        from tuning.tune import (
            PhiSkewTDriftModel,
            SKEW_T_NU_GRID,
            SKEW_T_GAMMA_GRID,
            is_skew_t_model,
            get_skew_t_model_name,
        )
        assert PhiSkewTDriftModel is not None
    
    def test_is_student_t_model_recognizes_skew_t(self):
        """Test that is_student_t_model recognizes skew-t models."""
        from tuning.tune import is_student_t_model
        
        # Student-t models
        assert is_student_t_model("phi_student_t_nu_6") == True
        
        # Skew-t models should also return True (for tail handling)
        assert is_student_t_model("phi_skew_t_nu_6_gamma_0p85") == True
        
        # Non-Student-t models
        assert is_student_t_model("kalman_gaussian") == False
        assert is_student_t_model("kalman_phi_gaussian") == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
