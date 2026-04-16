"""
Test Story 7.2: Caching Layer.

Validates:
  1. Cache hit on same data
  2. Cache miss on changed data
  3. Hit rate tracking
  4. Eviction at max size
  5. Clear resets everything
  6. Returns hash deterministic
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from models.computation_cache import (
    ComputationCache,
    compute_returns_hash,
    CacheStats,
)


class TestComputationCache(unittest.TestCase):
    """Tests for computation cache."""

    def test_cache_hit_same_data(self):
        """Same returns -> cache hit, function called once."""
        cache = ComputationCache()
        call_count = [0]
        
        def classify(r):
            call_count[0] += 1
            return "trending"
        
        returns = np.array([0.01, -0.005, 0.008, 0.003, -0.002])
        
        r1 = cache.get_regime("SPY", returns, classify)
        r2 = cache.get_regime("SPY", returns, classify)
        
        self.assertEqual(r1, "trending")
        self.assertEqual(r2, "trending")
        self.assertEqual(call_count[0], 1)  # Only called once

    def test_cache_miss_changed_data(self):
        """Different returns -> cache miss."""
        cache = ComputationCache()
        call_count = [0]
        
        def compute_vol(r):
            call_count[0] += 1
            return float(np.std(r))
        
        r1 = np.array([0.01, -0.005, 0.008])
        r2 = np.array([0.02, -0.01, 0.015])
        
        cache.get_volatility("SPY", r1, compute_vol)
        cache.get_volatility("SPY", r2, compute_vol)
        
        self.assertEqual(call_count[0], 2)

    def test_hit_rate_tracking(self):
        """Hit/miss stats tracked correctly."""
        cache = ComputationCache()
        returns = np.random.normal(0, 0.01, 50)
        
        cache.get_regime("SPY", returns, lambda r: "trending")
        cache.get_regime("SPY", returns, lambda r: "trending")  # Hit
        cache.get_regime("SPY", returns, lambda r: "trending")  # Hit
        
        stats = cache.get_stats_summary()
        self.assertEqual(stats["regime"]["hits"], 2)
        self.assertEqual(stats["regime"]["misses"], 1)
        self.assertAlmostEqual(stats["regime"]["hit_rate"], 2/3, places=3)

    def test_eviction_at_max_size(self):
        """Cache evicts when over max size."""
        cache = ComputationCache(max_size=3)
        
        for i in range(5):
            r = np.random.normal(0, 0.01, 50)
            cache.get_regime(f"SYM_{i}", r, lambda r: "trending")
        
        # Only 3 entries should remain
        self.assertLessEqual(len(cache._regime_cache), 3)

    def test_clear_resets(self):
        """Clear empties caches and stats."""
        cache = ComputationCache()
        returns = np.random.normal(0, 0.01, 50)
        
        cache.get_regime("SPY", returns, lambda r: "trending")
        cache.clear()
        
        stats = cache.get_stats_summary()
        self.assertEqual(stats["regime"]["hits"], 0)
        self.assertEqual(stats["regime"]["misses"], 0)
        self.assertEqual(len(cache._regime_cache), 0)

    def test_hash_deterministic(self):
        """Same data -> same hash."""
        r = np.array([0.01, -0.005, 0.008, 0.003])
        h1 = compute_returns_hash(r)
        h2 = compute_returns_hash(r)
        self.assertEqual(h1, h2)

    def test_hash_different_data(self):
        """Different data -> different hash."""
        r1 = np.array([0.01, -0.005])
        r2 = np.array([0.01, -0.006])
        h1 = compute_returns_hash(r1)
        h2 = compute_returns_hash(r2)
        self.assertNotEqual(h1, h2)

    def test_feature_cache(self):
        """Feature cache works end-to-end."""
        cache = ComputationCache()
        returns = np.random.normal(0, 0.01, 50)
        
        def compute_features(r):
            return {"momentum": float(np.mean(r[-10:])), "skew": 0.5}
        
        f1 = cache.get_features("SPY", returns, compute_features)
        f2 = cache.get_features("SPY", returns, compute_features)
        
        self.assertEqual(f1, f2)
        self.assertEqual(cache.stats["feature"].hits, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
