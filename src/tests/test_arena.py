#!/usr/bin/env python3
"""
test_arena.py — Unit tests for Arena Model Competition Framework

Tests:
1. Arena configuration validation
2. Experimental model specifications
3. Model fitting (synthetic data)
4. Score computation
5. Ranking determination
"""

import sys
import os
import unittest
import numpy as np

# Add src to path
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


class TestArenaConfig(unittest.TestCase):
    """Test arena configuration."""
    
    def test_default_config(self):
        """Test default configuration is valid."""
        from arena.arena_config import DEFAULT_ARENA_CONFIG, ARENA_BENCHMARK_SYMBOLS
        
        self.assertTrue(DEFAULT_ARENA_CONFIG.validate())
        self.assertEqual(len(DEFAULT_ARENA_CONFIG.symbols), 12)
        self.assertEqual(len(ARENA_BENCHMARK_SYMBOLS), 12)
    
    def test_category_weights(self):
        """Test category-specific scoring weights."""
        from arena.arena_config import get_category_weights, CapCategory
        
        weights = get_category_weights("AAPL")  # Large cap
        self.assertIn("bic", weights)
        self.assertIn("pit", weights)
        self.assertIn("hyvarinen", weights)
        
        # Weights should sum to approximately 1
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=2)


class TestArenaModels(unittest.TestCase):
    """Test experimental models."""
    
    def test_standard_model_specs(self):
        """Test standard model specifications."""
        from arena.arena_models import get_standard_model_specs, STANDARD_MOMENTUM_MODELS
        
        specs = get_standard_model_specs()
        self.assertEqual(len(specs), len(STANDARD_MOMENTUM_MODELS))
        
        # Check Gaussian momentum
        gaussian_spec = next(s for s in specs if "gaussian" in s["name"] and "phi" not in s["name"])
        self.assertEqual(gaussian_spec["n_params"], 2)
    
    def test_experimental_model_specs(self):
        """Test experimental model specifications."""
        from arena.arena_models import get_experimental_model_specs, EXPERIMENTAL_MODELS
        
        specs = get_experimental_model_specs()
        self.assertEqual(len(specs), len(EXPERIMENTAL_MODELS))
        
        # Check v2 model
        v2_spec = next(s for s in specs if "v2" in s.name)
        self.assertIn("nu_base", v2_spec.param_names)
    
    def test_momentum_student_t_v2(self):
        """Test MomentumStudentTV2 model fitting."""
        from arena.arena_models import MomentumStudentTV2
        
        # Generate synthetic data
        np.random.seed(42)
        n = 500
        returns = np.random.normal(0, 0.02, n)
        vol = np.abs(returns) * 0.8 + 0.01  # Simple vol proxy
        
        model = MomentumStudentTV2(nu_base=6.0, alpha=0.5)
        result = model.fit(returns, vol)
        
        self.assertIn("q", result)
        self.assertIn("c", result)
        self.assertIn("phi", result)
        self.assertIn("log_likelihood", result)
        self.assertIn("bic", result)
        self.assertTrue(result["success"])
        
        # Check parameter bounds
        self.assertGreater(result["q"], 0)
        self.assertGreater(result["c"], 0)
        self.assertLess(abs(result["phi"]), 1)
    
    def test_regime_coupled_model(self):
        """Test MomentumStudentTRegimeCoupled model."""
        from arena.experimental_models import MomentumStudentTRegimeCoupled
        
        np.random.seed(42)
        n = 500
        returns = np.random.normal(0, 0.02, n)
        vol = np.abs(returns) * 0.8 + 0.01
        
        model = MomentumStudentTRegimeCoupled()
        result = model.fit(returns, vol)
        
        self.assertIn("regime_nu", result)
        self.assertEqual(len(result["regime_nu"]), 5)  # 5 regimes
        self.assertTrue(result["success"])


class TestArenaScoring(unittest.TestCase):
    """Test scoring and ranking functions."""
    
    def test_model_score_creation(self):
        """Test ModelScore dataclass."""
        from arena.arena_tune import ModelScore
        from arena.arena_config import CapCategory
        
        score = ModelScore(
            model_name="test_model",
            symbol="AAPL",
            category=CapCategory.LARGE_CAP,
            log_likelihood=-1000.0,
            bic=2050.0,
            aic=2020.0,
            pit_pvalue=0.15,
            pit_calibrated=True,
            fit_params={"q": 1e-6, "c": 1.0},
            fit_time_ms=50.0,
        )
        
        self.assertEqual(score.model_name, "test_model")
        self.assertTrue(score.pit_calibrated)
        
        # Test serialization
        d = score.to_dict()
        self.assertIn("model_name", d)
        self.assertIn("bic", d)
    
    def test_ewma_vol_computation(self):
        """Test EWMA volatility computation."""
        from arena.arena_tune import compute_ewma_vol
        
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        vol = compute_ewma_vol(returns)
        
        self.assertEqual(len(vol), len(returns))
        self.assertTrue(np.all(vol >= 0))
        self.assertTrue(np.all(np.isfinite(vol)))


class TestArenaIntegration(unittest.TestCase):
    """Integration tests (require more setup)."""
    
    def test_standard_model_fitting(self):
        """Test fitting standard models on synthetic data."""
        from arena.arena_tune import fit_standard_model
        
        np.random.seed(42)
        n = 300
        returns = np.random.normal(0.0001, 0.02, n)
        vol = np.abs(returns) * 0.8 + 0.01
        regimes = np.zeros(n, dtype=int)
        
        # Test phi_student_t
        result = fit_standard_model(
            "phi_student_t_nu_8_momentum",
            returns, vol, regimes
        )
        
        self.assertIn("log_likelihood", result)
        self.assertIn("bic", result)
        self.assertIn("fit_params", result)
        self.assertEqual(result["fit_params"]["nu"], 8)


if __name__ == "__main__":
    unittest.main()
