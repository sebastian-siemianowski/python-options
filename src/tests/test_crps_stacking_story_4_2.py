"""
Tests for Story 4.2: CRPS Stacking Optimizer.

Acceptance Criteria:
  - crps_stacking_weights(model_crps_matrix, returns) returns w* in simplex
  - Uses scipy.optimize.minimize with method='SLSQP', simplex constraints
  - Warm-started from BIC weights (faster convergence)
  - Stacking weights differ from BIC weights by > 0.05 L1 distance on 60%+ assets
  - Combined CRPS under stacking < combined CRPS under BIC on 70%+ assets
  - Runtime: < 200ms for 14 models x 1000 timesteps
  - Validated on full 50-asset universe
"""
import os
import sys
import time
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, os.pardir))
for p in [SRC_ROOT, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from calibration.crps_stacking import (
    crps_stacking_weights,
    compute_bic_weights,
    combined_crps_score,
    StackingResult,
)


def _make_crps_matrix(T=1000, M=14, seed=42, dominant_model=0):
    """Generate synthetic CRPS matrix where one model is clearly better."""
    rng = np.random.RandomState(seed)
    # Base CRPS for all models
    base = rng.exponential(0.01, size=(T, M))
    # Make dominant_model systematically better
    base[:, dominant_model] *= 0.5
    return base


class TestStackingWeights(unittest.TestCase):
    """Core stacking optimizer tests."""

    def test_returns_stacking_result(self):
        """Returns StackingResult dataclass."""
        crps = _make_crps_matrix(T=100, M=5)
        result = crps_stacking_weights(crps)
        self.assertIsInstance(result, StackingResult)

    def test_weights_on_simplex(self):
        """Weights sum to 1 and are non-negative."""
        crps = _make_crps_matrix(T=500, M=10)
        result = crps_stacking_weights(crps)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=6)
        self.assertTrue(np.all(result.weights >= -1e-8))

    def test_correct_number_of_weights(self):
        """Number of weights equals number of models."""
        for M in [2, 5, 10, 14]:
            crps = _make_crps_matrix(T=200, M=M)
            result = crps_stacking_weights(crps)
            self.assertEqual(len(result.weights), M)

    def test_dominant_model_gets_high_weight(self):
        """Model with best CRPS gets highest weight."""
        crps = _make_crps_matrix(T=500, M=10, dominant_model=3)
        result = crps_stacking_weights(crps)
        self.assertEqual(np.argmax(result.weights), 3)

    def test_converged(self):
        """Optimizer converges on well-posed problem."""
        crps = _make_crps_matrix(T=500, M=5)
        result = crps_stacking_weights(crps)
        self.assertTrue(result.converged)

    def test_combined_crps_computed(self):
        """Combined CRPS is computed and positive."""
        crps = _make_crps_matrix(T=300, M=8)
        result = crps_stacking_weights(crps)
        self.assertGreater(result.combined_crps, 0)
        self.assertTrue(np.isfinite(result.combined_crps))

    def test_l1_distance_computed(self):
        """L1 distance from initial weights is computed."""
        crps = _make_crps_matrix(T=300, M=8)
        result = crps_stacking_weights(crps)
        self.assertGreaterEqual(result.l1_distance_from_init, 0)
        self.assertTrue(np.isfinite(result.l1_distance_from_init))

    def test_stacking_beats_uniform(self):
        """Stacking CRPS <= uniform weighting CRPS."""
        crps = _make_crps_matrix(T=500, M=10, dominant_model=2)
        result = crps_stacking_weights(crps)
        uniform_crps = combined_crps_score(crps, np.ones(10) / 10)
        self.assertLessEqual(result.combined_crps, uniform_crps + 1e-10)


class TestBICWarmStart(unittest.TestCase):
    """Warm-starting from BIC weights."""

    def test_with_bic_weights(self):
        """Optimizer accepts BIC weights as warm start."""
        crps = _make_crps_matrix(T=500, M=8)
        bic_w = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])
        result = crps_stacking_weights(crps, bic_weights=bic_w)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=6)
        self.assertTrue(result.converged)

    def test_without_bic_weights(self):
        """Uses uniform weights when BIC weights not provided."""
        crps = _make_crps_matrix(T=300, M=5)
        result = crps_stacking_weights(crps, bic_weights=None)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=6)

    def test_stacking_differs_from_bic(self):
        """Stacking weights should differ from BIC weights with L1 > 0.05."""
        # Create asymmetric CRPS where BIC weights are suboptimal
        rng = np.random.RandomState(42)
        T, M = 500, 8
        crps = rng.exponential(0.01, size=(T, M))
        crps[:, 0] *= 0.3  # One model much better
        crps[:, 1] *= 0.4
        bic_w = np.ones(M) / M  # Uniform BIC (intentionally suboptimal)
        result = crps_stacking_weights(crps, bic_weights=bic_w)
        self.assertGreater(result.l1_distance_from_init, 0.05)


class TestBICWeights(unittest.TestCase):
    """Test BIC weight computation."""

    def test_bic_weights_sum_to_one(self):
        """BIC weights must sum to 1."""
        bic = np.array([-1000, -990, -980, -970])
        w = compute_bic_weights(bic)
        self.assertAlmostEqual(np.sum(w), 1.0, places=10)

    def test_best_bic_gets_highest_weight(self):
        """Model with lowest BIC gets highest weight."""
        bic = np.array([-1000, -990, -980, -970])
        w = compute_bic_weights(bic)
        self.assertEqual(np.argmax(w), 0)

    def test_bic_weights_non_negative(self):
        """All BIC weights are non-negative."""
        bic = np.array([-500, -450, -400, -350, -300])
        w = compute_bic_weights(bic)
        self.assertTrue(np.all(w >= 0))

    def test_similar_bic_gives_similar_weights(self):
        """Very similar BIC values give approximately equal weights."""
        bic = np.array([-1000.0, -999.9, -1000.1])
        w = compute_bic_weights(bic)
        self.assertAlmostEqual(w[0], w[1], delta=0.05)


class TestCombinedCRPS(unittest.TestCase):
    """Test combined_crps_score utility."""

    def test_combined_crps_matches_manual(self):
        """combined_crps_score matches manual computation."""
        T, M = 100, 5
        rng = np.random.RandomState(42)
        crps = rng.exponential(0.01, size=(T, M))
        w = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
        result = combined_crps_score(crps, w)
        manual = float(np.mean(crps @ w))
        self.assertAlmostEqual(result, manual, places=10)

    def test_single_model_weight_one(self):
        """Single model with weight 1 equals that model's mean CRPS."""
        T = 200
        rng = np.random.RandomState(42)
        crps = rng.exponential(0.01, size=(T, 3))
        w = np.array([0.0, 1.0, 0.0])
        result = combined_crps_score(crps, w)
        expected = float(np.mean(crps[:, 1]))
        self.assertAlmostEqual(result, expected, places=10)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and robustness."""

    def test_single_model(self):
        """Works with M=1."""
        crps = _make_crps_matrix(T=100, M=1)
        result = crps_stacking_weights(crps)
        self.assertEqual(len(result.weights), 1)
        self.assertAlmostEqual(result.weights[0], 1.0)

    def test_two_models(self):
        """Works with M=2."""
        crps = _make_crps_matrix(T=200, M=2, dominant_model=1)
        result = crps_stacking_weights(crps)
        self.assertEqual(len(result.weights), 2)
        self.assertGreater(result.weights[1], result.weights[0])

    def test_many_models(self):
        """Works with M=20."""
        crps = _make_crps_matrix(T=300, M=20, dominant_model=5)
        result = crps_stacking_weights(crps)
        self.assertEqual(len(result.weights), 20)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=6)

    def test_identical_models(self):
        """All identical models get equal weights."""
        T, M = 200, 5
        rng = np.random.RandomState(42)
        single = rng.exponential(0.01, size=(T, 1))
        crps = np.tile(single, (1, M))
        result = crps_stacking_weights(crps)
        # All weights should be approximately equal
        for w in result.weights:
            self.assertAlmostEqual(w, 1.0 / M, delta=0.05)

    def test_empty_models(self):
        """Handles M=0 gracefully."""
        crps = np.empty((100, 0))
        result = crps_stacking_weights(crps)
        self.assertEqual(len(result.weights), 0)
        self.assertFalse(result.converged)

    def test_min_weight_constraint(self):
        """Minimum weight constraint is respected."""
        crps = _make_crps_matrix(T=500, M=5, dominant_model=0)
        result = crps_stacking_weights(crps, min_weight=0.05)
        self.assertTrue(np.all(result.weights >= 0.05 - 1e-8))


class TestPerformance(unittest.TestCase):
    """Runtime tests."""

    def test_runtime_14_models_1000_timesteps(self):
        """Runtime < 200ms for 14 models x 1000 timesteps."""
        crps = _make_crps_matrix(T=1000, M=14)
        bic_w = np.ones(14) / 14

        start = time.perf_counter()
        for _ in range(10):
            _ = crps_stacking_weights(crps, bic_weights=bic_w)
        elapsed = (time.perf_counter() - start) / 10.0

        self.assertLess(elapsed, 0.200, f"Took {elapsed*1000:.1f}ms")


class TestStackingVsBIC(unittest.TestCase):
    """Verify stacking outperforms BIC on synthetic data."""

    def test_stacking_crps_leq_bic_crps(self):
        """Stacking combined CRPS <= BIC combined CRPS on structured data."""
        rng = np.random.RandomState(42)
        T, M = 500, 10
        # Create structured CRPS matrix
        crps = rng.exponential(0.01, size=(T, M))
        crps[:, 0] *= 0.3  # Best model
        crps[:, 1] *= 0.5  # Second best
        # Fake BIC where model 5 is best (misranked)
        bic = np.array([-500 + i * 5 for i in range(M)], dtype=float)
        bic[5] = -600  # BIC wrongly favors model 5
        bic_w = compute_bic_weights(bic)
        result = crps_stacking_weights(crps, bic_weights=bic_w)
        bic_crps = combined_crps_score(crps, bic_w)
        self.assertLessEqual(result.combined_crps, bic_crps + 1e-10)


if __name__ == '__main__':
    unittest.main()
