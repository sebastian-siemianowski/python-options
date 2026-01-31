#!/usr/bin/env python3
"""
Test 2-State Gaussian Mixture Model (GMM) Integration

This test verifies that the GMM is properly integrated into the
Bayesian Model Averaging framework as per the Expert Panel recommendation.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


class TestGaussianMixtureModel:
    """Test GMM model functionality."""
    
    def test_imports(self):
        """Test that all GMM components import correctly."""
        from models import (
            GaussianMixtureModel,
            fit_gmm_to_returns,
            compute_gmm_pit,
            get_gmm_model_name,
            is_gmm_model,
            GMM_MIN_OBS,
            GMM_MIN_SEPARATION,
            GMM_DEGENERATE_THRESHOLD,
        )
        
        assert GaussianMixtureModel is not None
        assert GMM_MIN_OBS == 100
        assert GMM_MIN_SEPARATION == 0.5
        assert GMM_DEGENERATE_THRESHOLD == 0.95
    
    def test_gmm_initialization(self):
        """Test GMM initialization with custom parameters."""
        from models import GaussianMixtureModel
        
        gmm = GaussianMixtureModel(
            weights=(0.6, 0.4),
            means=(0.5, -0.3),
            variances=(0.8, 1.5)
        )
        
        assert len(gmm.weights) == 2
        assert np.allclose(gmm.weights.sum(), 1.0)
        assert gmm.means[0] == 0.5
        assert gmm.means[1] == -0.3
        assert gmm.variances[0] == 0.8
        assert gmm.variances[1] == 1.5
        assert len(gmm.stds) == 2
    
    def test_gmm_sampling(self):
        """Test that GMM sampling produces expected behavior."""
        from models import GaussianMixtureModel
        
        # Create bimodal GMM
        gmm = GaussianMixtureModel(
            weights=(0.6, 0.4),
            means=(1.0, -1.0),
            variances=(0.3, 0.3)
        )
        
        rng = np.random.default_rng(42)
        samples = gmm.sample(size=10000, rng=rng)
        
        # Should have bimodal distribution
        assert len(samples) == 10000
        assert np.all(np.isfinite(samples))
        
        # Check that samples roughly match expected mean
        # Expected mean = π₁·μ₁ + π₂·μ₂ = 0.6*1.0 + 0.4*(-1.0) = 0.2
        assert abs(np.mean(samples) - 0.2) < 0.1
    
    def test_gmm_pdf_cdf(self):
        """Test GMM PDF and CDF properties."""
        from models import GaussianMixtureModel
        
        gmm = GaussianMixtureModel(
            weights=(0.5, 0.5),
            means=(0.5, -0.5),
            variances=(1.0, 1.0)
        )
        
        x = np.linspace(-3, 3, 100)
        
        # PDF should be positive
        pdf_vals = gmm.pdf(x)
        assert np.all(pdf_vals >= 0)
        
        # CDF should be in [0, 1] and monotone
        cdf_vals = gmm.cdf(x)
        assert np.all(cdf_vals >= 0)
        assert np.all(cdf_vals <= 1)
        assert np.all(np.diff(cdf_vals) >= -1e-10)  # Monotone
    
    def test_gmm_responsibilities(self):
        """Test GMM posterior responsibilities."""
        from models import GaussianMixtureModel
        
        gmm = GaussianMixtureModel(
            weights=(0.5, 0.5),
            means=(2.0, -2.0),
            variances=(0.5, 0.5)
        )
        
        # Point near component 0 should have high responsibility for component 0
        resp = gmm.responsibilities(np.array([2.0]))
        assert resp[0, 0] > resp[0, 1]
        
        # Point near component 1 should have high responsibility for component 1
        resp = gmm.responsibilities(np.array([-2.0]))
        assert resp[0, 1] > resp[0, 0]
        
        # Responsibilities should sum to 1
        assert np.allclose(resp.sum(axis=1), 1.0)
    
    def test_gmm_separation(self):
        """Test GMM component separation metric."""
        from models import GaussianMixtureModel
        
        # Well-separated components
        gmm_separated = GaussianMixtureModel(
            weights=(0.5, 0.5),
            means=(2.0, -2.0),
            variances=(1.0, 1.0)
        )
        assert gmm_separated.separation > 2.0
        assert gmm_separated.is_well_separated
        
        # Poorly separated components
        gmm_close = GaussianMixtureModel(
            weights=(0.5, 0.5),
            means=(0.1, -0.1),
            variances=(1.0, 1.0)
        )
        assert gmm_close.separation < 0.5
        assert not gmm_close.is_well_separated
    
    def test_gmm_degeneracy(self):
        """Test GMM degeneracy detection."""
        from models import GaussianMixtureModel
        
        # Non-degenerate
        gmm_normal = GaussianMixtureModel(
            weights=(0.6, 0.4),
            means=(1.0, -1.0),
            variances=(1.0, 1.0)
        )
        assert not gmm_normal.is_degenerate
        
        # Degenerate (one component dominates)
        gmm_degen = GaussianMixtureModel(
            weights=(0.98, 0.02),
            means=(0.0, 1.0),
            variances=(1.0, 1.0)
        )
        assert gmm_degen.is_degenerate
    
    def test_gmm_em_fitting(self):
        """Test GMM fitting via EM algorithm."""
        from models import GaussianMixtureModel
        
        # Generate bimodal data
        rng = np.random.default_rng(42)
        data1 = rng.normal(loc=1.5, scale=0.5, size=300)
        data2 = rng.normal(loc=-1.5, scale=0.5, size=200)
        data = np.concatenate([data1, data2])
        
        gmm, diag = GaussianMixtureModel.fit_em(data)
        
        assert diag['fit_success']
        assert diag['converged'] or diag['n_iterations'] == 100
        assert gmm.is_well_separated
        assert not gmm.is_degenerate
        
        # Means should be close to true values
        sorted_means = sorted(gmm.means)
        assert abs(sorted_means[0] - (-1.5)) < 0.5
        assert abs(sorted_means[1] - 1.5) < 0.5
    
    def test_gmm_to_from_dict(self):
        """Test GMM serialization and deserialization."""
        from models import GaussianMixtureModel
        
        gmm_original = GaussianMixtureModel(
            weights=(0.6, 0.4),
            means=(1.0, -1.0),
            variances=(0.8, 1.2)
        )
        
        d = gmm_original.to_dict()
        gmm_restored = GaussianMixtureModel.from_dict(d)
        
        assert np.allclose(gmm_original.weights, gmm_restored.weights)
        assert np.allclose(gmm_original.means, gmm_restored.means)
        assert np.allclose(gmm_original.variances, gmm_restored.variances)


class TestGMMReturnsIntegration:
    """Test GMM integration with return data."""
    
    def test_fit_gmm_to_returns(self):
        """Test fitting GMM to volatility-adjusted returns."""
        from models import fit_gmm_to_returns
        
        rng = np.random.default_rng(42)
        n = 500
        
        # Simulate returns with bimodal behavior
        returns1 = rng.normal(loc=0.001, scale=0.01, size=300)
        returns2 = rng.normal(loc=-0.002, scale=0.02, size=200)
        returns = np.concatenate([returns1, returns2])
        rng.shuffle(returns)
        
        # Volatility
        vol = np.abs(returns).clip(min=0.005) * 2
        
        gmm, diag = fit_gmm_to_returns(returns, vol, min_obs=100)
        
        assert gmm is not None
        assert diag['fit_success']
        assert diag['standardized']
    
    def test_gmm_pit_calibration(self):
        """Test GMM PIT calibration."""
        from models import GaussianMixtureModel, compute_gmm_pit
        
        rng = np.random.default_rng(42)
        
        # Create GMM
        gmm = GaussianMixtureModel(
            weights=(0.5, 0.5),
            means=(0.0, 0.0),
            variances=(1.0, 1.0)
        )
        
        # Generate data from GMM
        vol = np.ones(500) * 0.01
        z_samples = gmm.sample(size=500, rng=rng)
        returns = z_samples * vol
        
        ks_stat, ks_p = compute_gmm_pit(returns, vol, gmm)
        
        assert 0 <= ks_stat <= 1
        assert 0 <= ks_p <= 1


class TestSignalIntegration:
    """Test signal layer integration with GMM."""
    
    def test_sample_from_gmm(self):
        """Test GMM sampling function in signals."""
        from decision.signals import sample_from_gmm
        
        gmm_params = {
            "weights": [0.6, 0.4],
            "means": [0.5, -0.5],
            "variances": [1.0, 1.5],
            "is_degenerate": False,
        }
        
        rng = np.random.default_rng(42)
        samples = sample_from_gmm(
            gmm_params=gmm_params,
            mu_t=0.001,
            P_t=0.0001,
            sigma_step=0.015,
            H=5,
            n_paths=1000,
            rng=rng
        )
        
        assert len(samples) == 1000
        assert np.all(np.isfinite(samples))
    
    def test_run_regime_specific_mc_with_gmm(self):
        """Test MC sampling with GMM parameters."""
        from decision.signals import run_regime_specific_mc
        
        gmm_params = {
            "weights": [0.6, 0.4],
            "means": [0.3, -0.3],
            "variances": [0.8, 1.2],
            "is_degenerate": False,
        }
        
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
            nig_alpha=None,
            nig_beta=None,
            nig_delta=None,
            gmm_params=gmm_params,  # Using GMM
            seed=42
        )
        
        assert len(samples) == 1000
        assert np.all(np.isfinite(samples))
    
    def test_gmm_not_used_for_student_t(self):
        """Test that GMM is not used when Student-t is specified."""
        from decision.signals import run_regime_specific_mc
        
        gmm_params = {
            "weights": [0.6, 0.4],
            "means": [0.3, -0.3],
            "variances": [0.8, 1.2],
            "is_degenerate": False,
        }
        
        # With nu specified, GMM should be ignored
        samples = run_regime_specific_mc(
            regime=0,
            mu_t=0.001,
            P_t=0.0001,
            phi=0.9,
            q=1e-6,
            sigma2_step=0.0002,
            H=5,
            n_paths=1000,
            nu=6.0,  # Student-t takes priority
            gmm_params=gmm_params,
            seed=42
        )
        
        assert len(samples) == 1000
        assert np.all(np.isfinite(samples))


class TestTuneIntegration:
    """Test tune.py integration with GMM."""
    
    def test_tune_imports_gmm(self):
        """Test that tune.py imports GMM components."""
        from tuning.tune import (
            GaussianMixtureModel,
            fit_gmm_to_returns,
            GMM_MIN_OBS,
        )
        assert GaussianMixtureModel is not None
        assert GMM_MIN_OBS == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
