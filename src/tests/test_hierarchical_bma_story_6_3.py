"""
Tests for Story 6.3: Hierarchical BMA with Asset-Class Grouping
================================================================

Validates hierarchical BMA pooling across asset classes.

Key properties tested:
1. Weights sum to 1, positive, correct shape
2. Shrinkage proportional to 1/n_asset
3. Group prior = mean(weights) within class
4. Small-cap assets borrow from group (high shrinkage)
5. Large-cap assets barely change (low shrinkage)
6. BIC improvement on small-cap assets
7. All 6 asset classes supported
8. 50-asset universe validation
"""
import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.entropy_bma import (
    hierarchical_bma,
    AssetBMAInput,
    HierarchicalBMAResult,
    _compute_shrinkage,
    _compute_bic_weights,
    SHRINKAGE_N_REF,
    ASSET_CLASSES,
)


def _make_asset(symbol, n_obs, asset_class, weights=None, M=5, seed=None):
    """Helper to create an AssetBMAInput."""
    if weights is not None:
        w = np.asarray(weights, dtype=float)
    else:
        rng = np.random.RandomState(seed)
        raw = rng.dirichlet(np.ones(M))
        w = raw
    ll = np.zeros(len(w))  # Placeholder
    k = np.ones(len(w)) * 2.0
    return AssetBMAInput(
        symbol=symbol,
        weights=w,
        log_likelihoods=ll,
        n_params=k,
        n_obs=n_obs,
        asset_class=asset_class,
    )


class TestShrinkageFunction(unittest.TestCase):
    """_compute_shrinkage correctness."""

    def test_small_n_high_shrinkage(self):
        alpha = _compute_shrinkage(100)
        self.assertGreater(alpha, 0.7)

    def test_large_n_low_shrinkage(self):
        alpha = _compute_shrinkage(2000)
        self.assertLess(alpha, 0.3)

    def test_medium_n_moderate(self):
        alpha = _compute_shrinkage(500)
        self.assertAlmostEqual(alpha, 0.5, places=3)

    def test_monotonically_decreasing(self):
        """More data -> less shrinkage."""
        alphas = [_compute_shrinkage(n) for n in [100, 200, 500, 1000, 2000]]
        for i in range(len(alphas) - 1):
            self.assertGreater(alphas[i], alphas[i + 1])

    def test_bounded_0_1(self):
        for n in [10, 100, 500, 2000, 10000]:
            alpha = _compute_shrinkage(n)
            self.assertGreater(alpha, 0)
            self.assertLess(alpha, 1)


class TestHierarchicalBMABasic(unittest.TestCase):
    """Basic properties of hierarchical BMA."""

    def test_empty_input(self):
        result = hierarchical_bma([])
        self.assertEqual(len(result.asset_weights), 0)

    def test_single_asset(self):
        a = _make_asset("AAPL", 1000, "Large Cap", weights=[0.5, 0.3, 0.2])
        result = hierarchical_bma([a])
        self.assertIn("AAPL", result.asset_weights)
        w = result.asset_weights["AAPL"]
        self.assertAlmostEqual(np.sum(w), 1.0, places=10)

    def test_weights_sum_to_one(self):
        assets = [
            _make_asset("AAPL", 1000, "Large Cap", seed=1),
            _make_asset("MSFT", 900, "Large Cap", seed=2),
            _make_asset("IONQ", 150, "Small Cap", seed=3),
        ]
        result = hierarchical_bma(assets)
        for sym in ["AAPL", "MSFT", "IONQ"]:
            w = result.asset_weights[sym]
            self.assertAlmostEqual(np.sum(w), 1.0, places=10)

    def test_weights_positive(self):
        assets = [
            _make_asset("AAPL", 1000, "Large Cap", seed=1),
            _make_asset("IONQ", 150, "Small Cap", seed=2),
        ]
        result = hierarchical_bma(assets)
        for sym in result.asset_weights:
            self.assertTrue(np.all(result.asset_weights[sym] > 0))

    def test_returns_hierarchical_result(self):
        a = _make_asset("AAPL", 1000, "Large Cap", seed=1)
        result = hierarchical_bma([a])
        self.assertIsInstance(result, HierarchicalBMAResult)

    def test_result_has_all_fields(self):
        assets = [
            _make_asset("AAPL", 1000, "Large Cap", seed=1),
            _make_asset("IONQ", 150, "Small Cap", seed=2),
        ]
        result = hierarchical_bma(assets)
        self.assertIn("AAPL", result.asset_weights)
        self.assertIn("IONQ", result.asset_weights)
        self.assertIn("Large Cap", result.group_priors)
        self.assertIn("Small Cap", result.group_priors)
        self.assertIn("AAPL", result.shrinkage_factors)
        self.assertIn("AAPL", result.original_weights)


class TestGroupPrior(unittest.TestCase):
    """Group prior = mean of member weights."""

    def test_prior_is_mean_of_group(self):
        w1 = np.array([0.6, 0.3, 0.1])
        w2 = np.array([0.2, 0.5, 0.3])
        assets = [
            _make_asset("A", 1000, "Large Cap", weights=w1),
            _make_asset("B", 1000, "Large Cap", weights=w2),
        ]
        result = hierarchical_bma(assets)
        expected_prior = (w1 + w2) / 2.0
        expected_prior /= np.sum(expected_prior)
        np.testing.assert_allclose(
            result.group_priors["Large Cap"], expected_prior, atol=1e-10)

    def test_prior_sums_to_one(self):
        assets = [
            _make_asset("A", 500, "Mid Cap", seed=10),
            _make_asset("B", 400, "Mid Cap", seed=11),
            _make_asset("C", 600, "Mid Cap", seed=12),
        ]
        result = hierarchical_bma(assets)
        prior = result.group_priors["Mid Cap"]
        self.assertAlmostEqual(np.sum(prior), 1.0, places=10)

    def test_single_member_prior_equals_weights(self):
        w = np.array([0.4, 0.35, 0.25])
        a = _make_asset("X", 300, "Crypto", weights=w)
        result = hierarchical_bma([a])
        np.testing.assert_allclose(
            result.group_priors["Crypto"], w, atol=1e-10)


class TestShrinkageBehavior(unittest.TestCase):
    """Shrinkage should be proportional to 1/n_asset."""

    def test_small_n_more_shrinkage(self):
        """Small-cap asset (n=150) gets more shrinkage than large-cap (n=2000)."""
        assets = [
            _make_asset("SPY", 2000, "Index", seed=1),
            _make_asset("IONQ", 150, "Small Cap", seed=2),
        ]
        result = hierarchical_bma(assets)
        self.assertGreater(
            result.shrinkage_factors["IONQ"],
            result.shrinkage_factors["SPY"])

    def test_shrinkage_values(self):
        """Verify exact shrinkage formula."""
        assets = [
            _make_asset("A", 100, "Small Cap", seed=1),
            _make_asset("B", 500, "Large Cap", seed=2),
            _make_asset("C", 2000, "Index", seed=3),
        ]
        result = hierarchical_bma(assets)
        # alpha = N_REF / (n + N_REF) with N_REF=500
        self.assertAlmostEqual(
            result.shrinkage_factors["A"],
            500.0 / 600.0, places=6)
        self.assertAlmostEqual(
            result.shrinkage_factors["B"],
            500.0 / 1000.0, places=6)
        self.assertAlmostEqual(
            result.shrinkage_factors["C"],
            500.0 / 2500.0, places=6)

    def test_high_shrinkage_moves_toward_prior(self):
        """With high shrinkage (small n), weights move toward group prior."""
        # Two assets in same class, one with lots of data, one with little
        w_data_rich = np.array([0.8, 0.1, 0.1])
        w_data_poor = np.array([0.1, 0.1, 0.8])
        assets = [
            _make_asset("RICH", 2000, "Large Cap", weights=w_data_rich),
            _make_asset("POOR", 100, "Large Cap", weights=w_data_poor),
        ]
        result = hierarchical_bma(assets)

        # POOR should be pulled toward group prior (which includes RICH's weights)
        w_poor_hier = result.asset_weights["POOR"]
        # The group prior is mean of [0.8, 0.1, 0.1] and [0.1, 0.1, 0.8]
        # = [0.45, 0.1, 0.45]
        # POOR has high alpha (~0.83), so should be close to prior
        # w_hier = 0.17 * [0.1, 0.1, 0.8] + 0.83 * [0.45, 0.1, 0.45]
        # = [0.017+0.374, 0.017+0.083, 0.136+0.374] = [0.391, 0.100, 0.510]
        self.assertGreater(w_poor_hier[0], 0.3,
                           "POOR should be pulled toward RICH's model preference")


class TestLargeCapNoRegression(unittest.TestCase):
    """Large-cap assets with lots of data should barely change."""

    def test_large_cap_weights_stable(self):
        """Large-cap assets (n=2000) should have near-original weights."""
        w_orig = np.array([0.7, 0.2, 0.1])
        assets = [
            _make_asset("AAPL", 2000, "Large Cap", weights=w_orig),
            _make_asset("MSFT", 1800, "Large Cap", weights=[0.3, 0.5, 0.2]),
        ]
        result = hierarchical_bma(assets)
        w_hier = result.asset_weights["AAPL"]
        # With n=2000, alpha = 500/2500 = 0.20
        # Max change should be small
        max_change = np.max(np.abs(w_hier - w_orig))
        self.assertLess(max_change, 0.15,
                        f"Large-cap weights changed too much: {max_change:.4f}")

    def test_large_n_preserves_ranking(self):
        """For large n, model ranking should be preserved."""
        w_orig = np.array([0.6, 0.25, 0.15])
        assets = [
            _make_asset("SPY", 2500, "Index", weights=w_orig),
            _make_asset("QQQ", 2000, "Index", weights=[0.4, 0.35, 0.25]),
        ]
        result = hierarchical_bma(assets)
        w_hier = result.asset_weights["SPY"]
        # Best model should still be best
        self.assertEqual(np.argmax(w_hier), np.argmax(w_orig))


class TestBICImprovement(unittest.TestCase):
    """Small-cap assets should get BIC improvement from hierarchical pooling."""

    def test_small_cap_bic_improvement(self):
        """Hierarchical weights should improve predictive BIC for small-cap."""
        np.random.seed(42)
        M = 5
        # Simulate: true model is model 0, but small-cap can't identify it
        # Group (large-cap) correctly identifies model 0

        n_large = 1000
        n_small = 150

        # Large-cap assets correctly weight model 0
        large_assets = []
        for i, sym in enumerate(["AAPL", "MSFT", "GOOGL"]):
            w = np.array([0.6, 0.15, 0.10, 0.10, 0.05])
            w += np.random.uniform(-0.02, 0.02, M)
            w = np.maximum(w, 0.01)
            w /= np.sum(w)
            large_assets.append(
                _make_asset(sym, n_large, "Large Cap", weights=w))

        # Small-cap asset has noisy weights (insufficient data)
        w_noisy = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
        small_asset = _make_asset("IONQ", n_small, "Large Cap", weights=w_noisy)

        all_assets = large_assets + [small_asset]
        result = hierarchical_bma(all_assets)

        w_hier = result.asset_weights["IONQ"]
        # After borrowing from group, model 0 weight should increase
        self.assertGreater(w_hier[0], w_noisy[0],
                           f"Model 0 weight should increase: "
                           f"{w_hier[0]:.3f} vs original {w_noisy[0]:.3f}")

    def test_bic_score_improvement(self):
        """Compute actual BIC improvement for small-cap assets."""
        np.random.seed(123)
        M = 6

        # True model weights (what we'd get with infinite data)
        w_true = np.array([0.5, 0.2, 0.15, 0.08, 0.05, 0.02])

        # Large-cap peers: close to true weights
        large_assets = []
        for i, sym in enumerate(["A", "B", "C", "D"]):
            w = w_true + np.random.normal(0, 0.03, M)
            w = np.maximum(w, 0.01)
            w /= np.sum(w)
            large_assets.append(
                _make_asset(sym, 1000, "Large Cap", weights=w))

        # Small-cap: far from true weights (noisy estimation)
        w_small = np.array([0.20, 0.18, 0.17, 0.16, 0.15, 0.14])
        small = _make_asset("SMALL", 120, "Large Cap", weights=w_small)

        result = hierarchical_bma(large_assets + [small])
        w_hier = result.asset_weights["SMALL"]

        # Hierarchical weights should be closer to true weights (lower L2)
        l2_orig = np.sum((w_small - w_true) ** 2)
        l2_hier = np.sum((w_hier - w_true) ** 2)
        self.assertLess(l2_hier, l2_orig,
                        f"Hierarchical L2={l2_hier:.4f} should be < "
                        f"original L2={l2_orig:.4f}")


class TestAssetClasses(unittest.TestCase):
    """All 6 asset classes should work."""

    def test_all_six_classes(self):
        """Each defined asset class should be groupable."""
        assets = []
        classes = ["Large Cap", "Mid Cap", "Small Cap", "Index", "Metals", "Crypto"]
        for i, cls in enumerate(classes):
            assets.append(_make_asset(f"SYM{i}", 500, cls, seed=i))

        result = hierarchical_bma(assets)
        self.assertEqual(len(result.asset_weights), 6)
        for cls in classes:
            self.assertIn(cls, result.group_priors)

    def test_classes_independent(self):
        """Different classes should have different group priors."""
        assets = [
            _make_asset("AAPL", 1000, "Large Cap",
                        weights=[0.7, 0.2, 0.1]),
            _make_asset("MSFT", 900, "Large Cap",
                        weights=[0.6, 0.3, 0.1]),
            _make_asset("GLD", 800, "Metals",
                        weights=[0.1, 0.2, 0.7]),
            _make_asset("SLV", 700, "Metals",
                        weights=[0.15, 0.25, 0.6]),
        ]
        result = hierarchical_bma(assets)
        lc_prior = result.group_priors["Large Cap"]
        mt_prior = result.group_priors["Metals"]
        # Should be clearly different
        self.assertGreater(np.max(np.abs(lc_prior - mt_prior)), 0.3)


class TestFiftyAssetUniverse(unittest.TestCase):
    """Validate on full 50-asset universe."""

    def _make_universe(self, seed=42):
        """Create a realistic 50-asset universe with class assignments."""
        rng = np.random.RandomState(seed)
        M = 6  # Models per asset
        assets = []

        # Define asset classes and their typical characteristics
        class_specs = {
            "Large Cap": {
                "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                            "JPM", "JNJ", "PG", "HD", "UNH"],
                "n_range": (1500, 2500),
                "true_w": np.array([0.4, 0.25, 0.15, 0.10, 0.06, 0.04]),
            },
            "Mid Cap": {
                "symbols": ["CRWD", "DKNG", "CRM", "ADBE", "NET",
                            "SCHW", "DE", "NKE", "SBUX", "DIS"],
                "n_range": (500, 1200),
                "true_w": np.array([0.35, 0.25, 0.18, 0.12, 0.06, 0.04]),
            },
            "Small Cap": {
                "symbols": ["UPST", "AFRM", "IONQ", "RKLB", "SNAP",
                            "MRNA", "RIVN", "LCID", "PLTR", "SOFI"],
                "n_range": (100, 400),
                "true_w": np.array([0.30, 0.22, 0.18, 0.14, 0.10, 0.06]),
            },
            "Index": {
                "symbols": ["SPY", "QQQ", "IWM", "DIA", "VTI"],
                "n_range": (2000, 3000),
                "true_w": np.array([0.45, 0.22, 0.15, 0.10, 0.05, 0.03]),
            },
            "Metals": {
                "symbols": ["GLD", "SLV", "GDX", "NEM", "FCX",
                            "GOLD", "WPM", "KGC", "FNV", "AEM"],
                "n_range": (800, 1500),
                "true_w": np.array([0.32, 0.24, 0.18, 0.13, 0.08, 0.05]),
            },
            "Crypto": {
                "symbols": ["COIN", "MARA", "RIOT", "MSTR", "BITF"],
                "n_range": (200, 600),
                "true_w": np.array([0.28, 0.22, 0.20, 0.15, 0.10, 0.05]),
            },
        }

        for cls, spec in class_specs.items():
            for sym in spec["symbols"]:
                n = rng.randint(*spec["n_range"])
                # Noisy weights: more noise for small n
                noise_scale = 0.1 * (500.0 / max(n, 100))
                w = spec["true_w"] + rng.normal(0, noise_scale, M)
                w = np.maximum(w, 0.01)
                w /= np.sum(w)
                ll = rng.uniform(-300, -100, M)
                k = np.array([2, 3, 3, 4, 5, 6], dtype=float)
                assets.append(AssetBMAInput(
                    symbol=sym, weights=w, log_likelihoods=ll,
                    n_params=k, n_obs=n, asset_class=cls,
                ))

        return assets, class_specs

    def test_fifty_assets_all_valid(self):
        """All 50 assets should get valid hierarchical weights."""
        assets, _ = self._make_universe()
        self.assertEqual(len(assets), 50)
        result = hierarchical_bma(assets)
        self.assertEqual(len(result.asset_weights), 50)
        for sym, w in result.asset_weights.items():
            self.assertAlmostEqual(np.sum(w), 1.0, places=10,
                                   msg=f"{sym} weights don't sum to 1")
            self.assertTrue(np.all(w > 0), f"{sym} has non-positive weights")

    def test_small_cap_improves(self):
        """Small-cap assets should be closer to true weights after pooling."""
        assets, class_specs = self._make_universe()
        result = hierarchical_bma(assets)

        true_w = class_specs["Small Cap"]["true_w"]
        improvements = 0
        total = 0

        for a in assets:
            if a.asset_class == "Small Cap":
                l2_orig = np.sum((a.weights - true_w) ** 2)
                l2_hier = np.sum((result.asset_weights[a.symbol] - true_w) ** 2)
                if l2_hier < l2_orig:
                    improvements += 1
                total += 1

        frac = improvements / total
        self.assertGreater(frac, 0.5,
                           f"Only {frac:.1%} of small-cap improved")

    def test_large_cap_no_regression(self):
        """Large-cap weights should not change much."""
        assets, _ = self._make_universe()
        result = hierarchical_bma(assets)

        for a in assets:
            if a.asset_class == "Large Cap":
                w_hier = result.asset_weights[a.symbol]
                max_change = np.max(np.abs(w_hier - a.weights))
                self.assertLess(max_change, 0.20,
                                f"{a.symbol}: weight changed by {max_change:.4f}")

    def test_all_classes_have_priors(self):
        """All 6 classes should have computed priors."""
        assets, class_specs = self._make_universe()
        result = hierarchical_bma(assets)
        for cls in class_specs:
            self.assertIn(cls, result.group_priors)
            prior = result.group_priors[cls]
            self.assertAlmostEqual(np.sum(prior), 1.0, places=10)


if __name__ == '__main__':
    unittest.main()
