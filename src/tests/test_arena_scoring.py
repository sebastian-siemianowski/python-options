#!/usr/bin/env python3
"""
test_arena_scoring.py — Unit tests for Arena Scoring Module

Tests CRPS, Hyvarinen, and Combined scoring implementations.
"""

import sys
import os
import unittest
import numpy as np

# Add src to path
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


class TestCRPS(unittest.TestCase):
    """Test CRPS computation."""
    
    def test_crps_gaussian_perfect_forecast(self):
        """Perfect forecast should have low CRPS."""
        from arena.scoring.crps import compute_crps_gaussian
        
        np.random.seed(42)
        n = 100
        mu = np.zeros(n)
        sigma = np.ones(n)
        
        # Observations exactly at mean
        observations = mu.copy()
        
        result = compute_crps_gaussian(observations, mu, sigma)
        
        # CRPS should be relatively low for accurate forecasts
        self.assertLess(result.crps, 0.5)
        self.assertEqual(result.n_observations, n)
        self.assertEqual(result.distribution, "gaussian")
    
    def test_crps_gaussian_poor_forecast(self):
        """Biased forecast should have higher CRPS."""
        from arena.scoring.crps import compute_crps_gaussian
        
        np.random.seed(42)
        n = 100
        mu = np.zeros(n)
        sigma = np.ones(n)
        
        # Observations far from predictions
        observations = np.ones(n) * 3.0  # 3 sigma away
        
        result = compute_crps_gaussian(observations, mu, sigma)
        
        # CRPS should be higher for poor forecasts
        self.assertGreater(result.crps, 1.0)
    
    def test_crps_student_t(self):
        """Test CRPS for Student-t distribution."""
        from arena.scoring.crps import compute_crps_student_t
        
        np.random.seed(42)
        n = 100
        mu = np.zeros(n)
        sigma = np.ones(n)
        nu = 8.0
        
        # Generate Student-t observations
        from scipy.stats import t as student_t
        observations = student_t.rvs(df=nu, loc=0, scale=1, size=n)
        
        result = compute_crps_student_t(observations, mu, sigma, nu)
        
        self.assertGreater(result.crps, 0)
        self.assertIn("student_t", result.distribution)
    
    def test_crps_reliability_decomposition(self):
        """Test CRPS decomposition into reliability and sharpness."""
        from arena.scoring.crps import compute_crps_gaussian
        
        np.random.seed(42)
        n = 200
        mu = np.random.normal(0, 0.1, n)
        sigma = np.ones(n) * 0.5
        observations = mu + np.random.normal(0, 0.5, n)
        
        result = compute_crps_gaussian(observations, mu, sigma)
        
        # Check decomposition components exist
        self.assertIsNotNone(result.reliability)
        self.assertIsNotNone(result.sharpness)
        self.assertGreaterEqual(result.reliability, 0)
        self.assertGreater(result.sharpness, 0)


class TestHyvarinen(unittest.TestCase):
    """Test Hyvarinen score computation."""
    
    def test_hyvarinen_gaussian(self):
        """Test Hyvarinen score for Gaussian."""
        from arena.scoring.hyvarinen import compute_hyvarinen_score_gaussian
        
        np.random.seed(42)
        n = 100
        mu = np.zeros(n)
        sigma = np.ones(n)
        observations = np.random.normal(0, 1, n)
        
        score = compute_hyvarinen_score_gaussian(observations, mu, sigma)
        
        # Hyvarinen score should be finite
        self.assertTrue(np.isfinite(score))
    
    def test_hyvarinen_student_t(self):
        """Test Hyvarinen score for Student-t."""
        from arena.scoring.hyvarinen import compute_hyvarinen_score_student_t
        
        np.random.seed(42)
        n = 100
        mu = np.zeros(n)
        sigma = np.ones(n)
        nu = 8.0
        
        from scipy.stats import t as student_t
        observations = student_t.rvs(df=nu, loc=0, scale=1, size=n)
        
        score = compute_hyvarinen_score_student_t(observations, mu, sigma, nu)
        
        self.assertTrue(np.isfinite(score))


class TestCombinedScoring(unittest.TestCase):
    """Test combined scoring."""
    
    def test_combined_score_calibrated(self):
        """Test combined score for calibrated model."""
        from arena.scoring.combined import compute_combined_score, ScoringConfig
        
        config = ScoringConfig(pit_hard_constraint=False)
        
        result = compute_combined_score(
            bic=-1000.0,
            crps=0.05,
            hyvarinen=0.5,
            pit_pvalue=0.15,
            config=config,
        )
        
        self.assertGreater(result.combined_score, 0)
        self.assertTrue(result.pit_calibrated)
        self.assertIn("bic", result.raw_scores)
        self.assertIn("crps", result.raw_scores)
    
    def test_combined_score_uncalibrated_hard_constraint(self):
        """Test that uncalibrated models get zero score with hard constraint."""
        from arena.scoring.combined import compute_combined_score, ScoringConfig
        
        config = ScoringConfig(pit_hard_constraint=True, pit_threshold=0.05)
        
        result = compute_combined_score(
            bic=-1000.0,
            crps=0.05,
            hyvarinen=0.5,
            pit_pvalue=0.01,  # Below threshold
            config=config,
        )
        
        self.assertEqual(result.combined_score, 0.0)
        self.assertFalse(result.pit_calibrated)
    
    def test_pareto_frontier(self):
        """Test Pareto frontier computation."""
        from arena.scoring.combined import compute_pareto_frontier
        
        models = {
            "model_a": {"score1": 0.9, "score2": 0.3},  # Good at 1, bad at 2
            "model_b": {"score1": 0.3, "score2": 0.9},  # Bad at 1, good at 2
            "model_c": {"score1": 0.5, "score2": 0.5},  # Mediocre at both
            "model_d": {"score1": 0.8, "score2": 0.8},  # Good at both (dominates c)
        }
        
        frontier = compute_pareto_frontier(models)
        
        # model_a, model_b, model_d should be on frontier
        # model_c is dominated by model_d
        self.assertIn("model_a", frontier)
        self.assertIn("model_b", frontier)
        self.assertIn("model_d", frontier)
        self.assertNotIn("model_c", frontier)


class TestSMCParticles(unittest.TestCase):
    """Test SMC particle system."""
    
    def test_particle_creation(self):
        """Test particle creation and weight update."""
        from arena.smc.particle import Particle
        
        particle = Particle(
            particle_id=0,
            model_name="test_model",
            parameters={"q": 1e-6, "c": 1.0, "phi": 0.95},
        )
        
        self.assertEqual(particle.weight, 1.0)
        self.assertEqual(particle.log_weight, 0.0)
        
        # Update weight
        particle.update_weight(0.1)  # CRPS = 0.1
        
        self.assertNotEqual(particle.weight, 1.0)
        self.assertLess(particle.log_weight, 0)  # Negative because CRPS is penalty
    
    def test_particle_cloud(self):
        """Test particle cloud operations."""
        from arena.smc.particle import ParticleCloud, Particle
        
        particles = [
            Particle(i, f"model_{i % 2}", {"q": 1e-6}, weight=1.0)
            for i in range(10)
        ]
        
        cloud = ParticleCloud(particles=particles)
        
        self.assertEqual(cloud.n_particles, 10)
        
        # Check weights normalization
        weights = cloud.weights
        self.assertAlmostEqual(np.sum(weights), 1.0, places=6)
        
        # Check ESS
        ess = cloud.effective_sample_size()
        self.assertAlmostEqual(ess, 10.0, places=5)  # All equal weights -> ESS = N
    
    def test_create_initial_particles(self):
        """Test initial particle creation."""
        from arena.smc.particle import create_initial_particles
        
        model_specs = {
            "model_a": {"default_params": {"q": 1e-6, "c": 1.0}},
            "model_b": {"default_params": {"q": 1e-5, "c": 0.8}},
        }
        
        cloud = create_initial_particles(model_specs, n_particles_per_model=50)
        
        self.assertEqual(cloud.n_particles, 100)  # 50 * 2 models


class TestSMCResampling(unittest.TestCase):
    """Test SMC resampling algorithms."""
    
    def test_systematic_resample(self):
        """Test systematic resampling."""
        from arena.smc.resampling import systematic_resample
        
        np.random.seed(42)
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        
        indices = systematic_resample(weights, n_samples=100)
        
        # Check that higher weight particles are selected more often
        counts = np.bincount(indices, minlength=4)
        
        # Particle 3 (weight 0.4) should have most copies
        self.assertEqual(np.argmax(counts), 3)
        
        # Particle 0 (weight 0.1) should have fewest copies
        self.assertEqual(np.argmin(counts), 0)
    
    def test_effective_sample_size(self):
        """Test ESS computation."""
        from arena.smc.resampling import effective_sample_size
        
        # Equal weights
        weights = np.ones(10) / 10
        ess = effective_sample_size(weights)
        self.assertAlmostEqual(ess, 10.0, places=5)
        
        # One particle has all weight
        weights = np.array([1.0, 0.0, 0.0, 0.0])
        ess = effective_sample_size(weights)
        self.assertAlmostEqual(ess, 1.0, places=5)
    
    def test_should_resample(self):
        """Test resampling trigger."""
        from arena.smc.resampling import should_resample
        
        # ESS = 10, N = 100, threshold = 0.5 -> should resample
        self.assertTrue(should_resample(10, 100, 0.5))
        
        # ESS = 80, N = 100, threshold = 0.5 -> should not resample
        self.assertFalse(should_resample(80, 100, 0.5))


class TestExperimentalModels(unittest.TestCase):
    """Test new experimental models."""
    
    def test_asymmetric_loss_model(self):
        """Test AsymmetricLossModel."""
        from arena.experimental_models import AsymmetricLossModel
        
        np.random.seed(42)
        n = 300
        returns = np.random.normal(0, 0.02, n)
        vol = np.abs(returns) * 0.8 + 0.01
        
        model = AsymmetricLossModel(alpha=2.0)
        result = model.fit(returns, vol)
        
        self.assertIn("q", result)
        self.assertIn("alpha", result)
        self.assertEqual(result["alpha"], 2.0)
        self.assertTrue(result["success"])
    
    def test_multi_horizon_model(self):
        """Test MultiHorizonModel."""
        from arena.experimental_models import MultiHorizonModel
        
        np.random.seed(42)
        n = 500
        returns = np.random.normal(0.0001, 0.02, n)
        vol = np.abs(returns) * 0.8 + 0.01
        
        model = MultiHorizonModel(horizons=[1, 5, 20])
        result = model.fit(returns, vol)
        
        self.assertIn("horizons", result)
        self.assertEqual(result["horizons"], [1, 5, 20])
        self.assertTrue(result["success"])
    
    def test_ensemble_distillation_model(self):
        """Test EnsembleDistillationModel."""
        from arena.experimental_models import EnsembleDistillationModel
        
        np.random.seed(42)
        n = 300
        returns = np.random.normal(0, 0.02, n)
        vol = np.abs(returns) * 0.8 + 0.01
        
        model = EnsembleDistillationModel(lambda_reg=1.0)
        result = model.fit(returns, vol)
        
        self.assertIn("lambda_reg", result)
        self.assertIn("teacher_mean", result)
        self.assertTrue(result["success"])
    
    def test_pit_constrained_model(self):
        """Test PITConstrainedModel."""
        from arena.experimental_models import PITConstrainedModel
        
        np.random.seed(42)
        n = 300
        returns = np.random.normal(0, 0.02, n)
        vol = np.abs(returns) * 0.8 + 0.01
        
        model = PITConstrainedModel(pit_threshold=0.05)
        result = model.fit(returns, vol)
        
        self.assertIn("ks_pvalue", result)
        self.assertIn("pit_calibrated", result)
        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
