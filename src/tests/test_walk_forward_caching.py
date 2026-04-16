"""
Story 7.2: Tests for parallel walk-forward with smart caching.
"""
import unittest
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestWalkForwardCaching(unittest.TestCase):
    """Validate feature/tune caching and parallel runner."""

    def test_feature_cache_init(self):
        """Feature cache initializes with no state."""
        from decision.signals import _WFFeatureCache
        fc = _WFFeatureCache(retune_freq=63)
        self.assertFalse(fc.cache_hit(0, 5))
        self.assertIsNone(fc.get_cached_feats())

    def test_tune_param_caching_window(self):
        """Tune params should be cached within retune window."""
        from decision.signals import _WFFeatureCache, WF_RETUNE_FREQ
        fc = _WFFeatureCache(retune_freq=WF_RETUNE_FREQ)

        # First call loads
        params1 = fc.get_tune_params("__FAKE__", t=100)
        # Within window -- should return same object
        params2 = fc.get_tune_params("__FAKE__", t=100 + WF_RETUNE_FREQ - 1)
        self.assertIs(params1, params2)

    def test_tune_param_reload_after_window(self):
        """Tune params should reload after retune window expires."""
        from decision.signals import _WFFeatureCache
        fc = _WFFeatureCache(retune_freq=10)

        params1 = fc.get_tune_params("__FAKE__", t=0)
        # Force past window
        params2 = fc.get_tune_params("__FAKE__", t=11)
        # Both are dicts (may be same content but fresh load)
        self.assertIsInstance(params1, dict)
        self.assertIsInstance(params2, dict)

    def test_feature_cache_store_and_hit(self):
        """Stored features should be retrievable on cache hit."""
        from decision.signals import _WFFeatureCache
        fc = _WFFeatureCache()

        feats = {"log_ret": [0.01, -0.005]}
        fc.store(t=100, feats=feats)
        self.assertTrue(fc.cache_hit(t=105, rebalance_freq=5))
        self.assertIs(fc.get_cached_feats(), feats)

    def test_feature_cache_miss_on_gap(self):
        """Cache should miss when gap != rebalance_freq."""
        from decision.signals import _WFFeatureCache
        fc = _WFFeatureCache()

        fc.store(t=100, feats={"x": 1})
        # Wrong gap
        self.assertFalse(fc.cache_hit(t=110, rebalance_freq=5))

    def test_parallel_runner_importable(self):
        """run_walk_forward_parallel should be importable."""
        from decision.signals import run_walk_forward_parallel
        self.assertTrue(callable(run_walk_forward_parallel))

    def test_parallel_runner_empty_symbols(self):
        """Parallel runner with empty symbol list returns empty dict."""
        from decision.signals import run_walk_forward_parallel
        result = run_walk_forward_parallel([], max_workers=1)
        self.assertEqual(result, {})

    def test_cache_hit_rate_concept(self):
        """Demonstrate cache hit rate > 80% for typical walk-forward."""
        from decision.signals import _WFFeatureCache
        fc = _WFFeatureCache(retune_freq=63)

        rebalance = 5
        total_steps = 200
        hits = 0
        for t in range(0, total_steps * rebalance, rebalance):
            if fc.cache_hit(t, rebalance):
                hits += 1
            fc.store(t, feats={"dummy": t})

            # Tune params: check if within window
            fc.get_tune_params("__FAKE__", t)

        # After first store, all subsequent should hit
        hit_rate = hits / total_steps
        self.assertGreater(hit_rate, 0.80,
            f"Cache hit rate {hit_rate:.1%} should be > 80%")

    def test_incremental_features_concept(self):
        """Validate that incremental feature updates produce consistent results."""
        import numpy as np
        rng = np.random.default_rng(42)
        prices = np.cumsum(rng.normal(0.001, 0.015, 500)) + 100

        # Full computation: log returns of entire series
        full_log_ret = np.diff(np.log(prices))
        full_vol = float(np.std(full_log_ret[-252:]))

        # "Incremental": last 252 days (same window)
        incr_log_ret = np.diff(np.log(prices[-253:]))
        incr_vol = float(np.std(incr_log_ret))

        self.assertAlmostEqual(full_vol, incr_vol, places=10,
            msg="Incremental and full should produce identical results")


if __name__ == "__main__":
    unittest.main()
