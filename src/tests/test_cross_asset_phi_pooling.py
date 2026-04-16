"""
Tests for Story 1.3: Adaptive Phi Estimation with Cross-Asset Pooling.

Validates hierarchical Bayesian shrinkage of per-asset phi toward
the cross-asset population median using precision-weighted averaging.

Mathematical specification:
    tau_asset = sqrt(n_samples)
    tau_pop   = 1 / phi_pop_std^2
    phi_shrunk = (tau_asset * phi_mle + tau_pop * phi_pop_median) / (tau_asset + tau_pop)
"""

import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestPhiPoolingConstants(unittest.TestCase):
    """Validate module-level constants for cross-asset phi pooling."""

    def test_constants_exist(self):
        from tuning.tune import DEFAULT_PHI_PRIOR, DEFAULT_PHI_PRIOR_STD, PHI_POOL_MIN_ASSETS
        self.assertIsInstance(DEFAULT_PHI_PRIOR, float)
        self.assertIsInstance(DEFAULT_PHI_PRIOR_STD, float)
        self.assertIsInstance(PHI_POOL_MIN_ASSETS, int)

    def test_default_phi_prior_in_range(self):
        from tuning.tune import DEFAULT_PHI_PRIOR
        self.assertGreater(DEFAULT_PHI_PRIOR, 0.5)
        self.assertLess(DEFAULT_PHI_PRIOR, 1.0)

    def test_default_phi_prior_std_positive(self):
        from tuning.tune import DEFAULT_PHI_PRIOR_STD
        self.assertGreater(DEFAULT_PHI_PRIOR_STD, 0.01)
        self.assertLess(DEFAULT_PHI_PRIOR_STD, 0.5)

    def test_min_assets_reasonable(self):
        from tuning.tune import PHI_POOL_MIN_ASSETS
        self.assertGreaterEqual(PHI_POOL_MIN_ASSETS, 3)
        self.assertLessEqual(PHI_POOL_MIN_ASSETS, 20)


class TestPhiPoolingFunction(unittest.TestCase):
    """Validate apply_cross_asset_phi_pooling function."""

    def _make_cache(self, phi_values, n_samples=None):
        """Build a synthetic cache with given phi values."""
        cache = {}
        for i, phi in enumerate(phi_values):
            ns = n_samples[i] if n_samples else 500
            asset = f"ASSET_{i}"
            cache[asset] = {
                "global": {
                    "phi": phi,
                    "q": 1e-5,
                    "c": 0.01,
                },
                "regime_counts": {"0": ns, "1": ns // 5},
            }
        return cache

    def test_function_exists(self):
        from tuning.tune import apply_cross_asset_phi_pooling
        self.assertTrue(callable(apply_cross_asset_phi_pooling))

    def test_pathological_phi_shrinks_toward_population(self):
        """GOOGL-like negative phi (from short regime window) should be pulled positive."""
        from tuning.tune import apply_cross_asset_phi_pooling
        # 9 assets with phi~0.85 and normal data, 1 with phi=-0.016 and short regime
        phi_values = [0.85, 0.90, 0.80, 0.88, 0.92, 0.82, 0.87, 0.91, 0.83, -0.016]
        n_samples = [500, 500, 500, 500, 500, 500, 500, 500, 500, 80]  # GOOGL: short regime
        cache = self._make_cache(phi_values, n_samples)
        result = apply_cross_asset_phi_pooling(cache)
        googl_phi = result["ASSET_9"]["global"]["phi"]
        # Should be pulled significantly toward population median (~0.85)
        self.assertGreater(googl_phi, 0.3, f"Pathological phi not shrunk enough: {googl_phi}")

    def test_high_confidence_phi_barely_shrinks(self):
        """SPY-like high phi with lots of data should barely change."""
        from tuning.tune import apply_cross_asset_phi_pooling
        phi_values = [0.99, 0.85, 0.90, 0.80, 0.88, 0.92, 0.82, 0.87]
        n_samples = [2500, 500, 500, 500, 500, 500, 500, 500]  # SPY has way more data
        cache = self._make_cache(phi_values, n_samples)
        result = apply_cross_asset_phi_pooling(cache)
        spy_phi = result["ASSET_0"]["global"]["phi"]
        # Should barely move from 0.99
        self.assertGreater(spy_phi, 0.90, f"High-confidence phi shrunk too much: {spy_phi}")

    def test_shrinkage_reduces_variance(self):
        """Cross-asset phi variance should decrease after pooling."""
        from tuning.tune import apply_cross_asset_phi_pooling
        phi_values = [0.99, 0.12, 0.85, -0.02, 0.78, 0.45, 0.91, 0.60]
        cache = self._make_cache(phi_values)
        var_before = np.var(phi_values)
        result = apply_cross_asset_phi_pooling(cache)
        phi_after = [result[f"ASSET_{i}"]["global"]["phi"] for i in range(len(phi_values))]
        var_after = np.var(phi_after)
        self.assertLess(var_after, var_before, "Shrinkage should reduce cross-asset phi variance")

    def test_population_prior_stored_in_cache(self):
        """Pool metadata should be stored under hierarchical_tuning.phi_prior."""
        from tuning.tune import apply_cross_asset_phi_pooling
        phi_values = [0.85, 0.90, 0.80, 0.88, 0.92]
        cache = self._make_cache(phi_values)
        result = apply_cross_asset_phi_pooling(cache)
        # Check first asset for metadata
        ht = result["ASSET_0"].get("hierarchical_tuning", {})
        phi_prior = ht.get("phi_prior", {})
        self.assertIn("phi_population_median", phi_prior)
        self.assertIn("phi_population_std", phi_prior)
        self.assertIn("n_assets_pooled", phi_prior)
        self.assertEqual(phi_prior["n_assets_pooled"], 5)

    def test_too_few_assets_uses_defaults(self):
        """With fewer than PHI_POOL_MIN_ASSETS, should use default prior."""
        from tuning.tune import apply_cross_asset_phi_pooling, DEFAULT_PHI_PRIOR
        phi_values = [0.10, 0.20]  # Only 2 assets
        cache = self._make_cache(phi_values)
        result = apply_cross_asset_phi_pooling(cache)
        # Phi should shrink toward DEFAULT_PHI_PRIOR
        phi0 = result["ASSET_0"]["global"]["phi"]
        phi1 = result["ASSET_1"]["global"]["phi"]
        self.assertGreater(phi0, 0.10, "Should shrink toward default prior")
        self.assertGreater(phi1, 0.20, "Should shrink toward default prior")

    def test_original_phi_preserved(self):
        """Original phi_mle should be stored as phi_mle_original."""
        from tuning.tune import apply_cross_asset_phi_pooling
        phi_values = [0.85, 0.90, 0.80, 0.88, 0.92]
        cache = self._make_cache(phi_values)
        result = apply_cross_asset_phi_pooling(cache)
        for i, phi_orig in enumerate(phi_values):
            stored = result[f"ASSET_{i}"]["global"].get("phi_mle_original")
            self.assertAlmostEqual(stored, phi_orig, places=6)

    def test_phi_clamped_to_valid_range(self):
        """Output phi should be in [-0.999, 0.999]."""
        from tuning.tune import apply_cross_asset_phi_pooling
        phi_values = [0.85, 0.90, 0.80, 0.88, 0.92, 0.87, 0.91]
        cache = self._make_cache(phi_values)
        result = apply_cross_asset_phi_pooling(cache)
        for i in range(len(phi_values)):
            phi = result[f"ASSET_{i}"]["global"]["phi"]
            self.assertGreaterEqual(phi, -0.999)
            self.assertLessEqual(phi, 0.999)

    def test_nan_phi_skipped(self):
        """Assets with NaN phi should be skipped, not crash."""
        from tuning.tune import apply_cross_asset_phi_pooling
        phi_values = [0.85, float("nan"), 0.90, 0.80, 0.88, 0.92]
        cache = self._make_cache(phi_values)
        result = apply_cross_asset_phi_pooling(cache)
        # NaN asset should not have been touched
        nan_phi = result["ASSET_1"]["global"]["phi"]
        self.assertTrue(np.isnan(nan_phi), "NaN phi should remain NaN (skipped)")

    def test_regime_model_phi_also_shrunk(self):
        """Per-regime model phi values should also be shrunk."""
        from tuning.tune import apply_cross_asset_phi_pooling
        phi_values = [0.85, 0.90, 0.80, 0.88, 0.92]
        cache = self._make_cache(phi_values)
        # Add regime model data to first asset
        cache["ASSET_0"]["regime"] = {
            "0": {
                "models": {
                    "kalman_phi_gaussian": {"phi": 0.10, "q": 1e-5},
                    "phi_student_t_nu_8": {"phi": 0.05, "q": 1e-5},
                }
            }
        }
        result = apply_cross_asset_phi_pooling(cache)
        r0_models = result["ASSET_0"]["regime"]["0"]["models"]
        # Regime model phi should have been shrunk toward population
        self.assertGreater(r0_models["kalman_phi_gaussian"]["phi"], 0.10)
        self.assertGreater(r0_models["phi_student_t_nu_8"]["phi"], 0.05)
        # And originals preserved
        self.assertAlmostEqual(r0_models["kalman_phi_gaussian"]["phi_mle_original"], 0.10)


if __name__ == "__main__":
    unittest.main()
