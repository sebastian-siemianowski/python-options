"""
===============================================================================
TEST: Deterministic Kalman Filter Result Cache
===============================================================================

Tests for the filter cache infrastructure that eliminates redundant filter
executions during tuning and cross-validation.

Coverage:
- Cache key generation and hashing
- LRU eviction behavior
- Fold-aware likelihood slicing
- Warm-start state retrieval
- Numerical equivalence guarantees
- Integration with existing filters
"""

import numpy as np
import pytest
import time
import sys
import os
from typing import Tuple

# Add parent directory to path for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import cache infrastructure
from models.filter_cache import (
    FilterCacheKey,
    FilterCacheValue,
    FilterResultCache,
    get_filter_cache,
    clear_filter_cache,
    get_cache_stats,
    reset_cache_stats,
    set_filter_cache_enabled,
    cached_phi_gaussian_filter,
    cached_phi_student_t_filter,
    compute_cv_likelihood_from_cache,
    _hash_array,
    _round_param,
    FILTER_CACHE_MAX_SIZE,
    PARAM_ROUND_DECIMALS,
)

# Import models for comparison
from models.phi_gaussian import PhiGaussianDriftModel, _kalman_filter_phi_with_trajectory
from models.phi_student_t import PhiStudentTDriftModel


@pytest.fixture
def sample_data():
    """Generate reproducible test data."""
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n) * 0.02
    vol = np.abs(np.random.randn(n)) * 0.01 + 0.01
    return returns, vol


@pytest.fixture
def clean_cache():
    """Ensure clean cache state for each test."""
    clear_filter_cache()
    reset_cache_stats()
    yield
    clear_filter_cache()
    reset_cache_stats()


class TestCacheKeyGeneration:
    """Test cache key creation and hashing."""
    
    def test_array_hash_deterministic(self, sample_data):
        """Same array produces same hash."""
        returns, vol = sample_data
        
        hash1 = _hash_array(returns)
        hash2 = _hash_array(returns)
        
        assert hash1 == hash2
    
    def test_different_arrays_different_hashes(self, sample_data):
        """Different arrays produce different hashes."""
        returns, vol = sample_data
        
        hash1 = _hash_array(returns)
        hash2 = _hash_array(vol)
        
        assert hash1 != hash2
    
    def test_cache_key_immutable(self, sample_data):
        """Cache keys are hashable (immutable)."""
        returns, vol = sample_data
        
        key = FilterCacheKey.create(
            returns=returns,
            vol=vol,
            model_id="phi_gaussian",
            q=1e-6,
            c=1.0,
            phi=0.5,
        )
        
        # Should be usable as dict key
        d = {key: "value"}
        assert d[key] == "value"
    
    def test_parameter_rounding(self):
        """Parameters are rounded to avoid floating point noise."""
        val1 = _round_param(0.12345678901234)
        val2 = _round_param(0.12345678901235)  # Differs at 14th decimal
        
        # Should be equal after rounding to 8 decimals
        assert val1 == val2


class TestCacheOperations:
    """Test basic cache put/get/eviction."""
    
    def test_cache_hit(self, sample_data, clean_cache):
        """Cached result is retrievable."""
        returns, vol = sample_data
        cache = get_filter_cache()
        
        key = FilterCacheKey.create(
            returns=returns,
            vol=vol,
            model_id="test_model",
            q=1e-6,
            c=1.0,
            phi=0.5,
        )
        
        # Put result
        mu = np.zeros(len(returns))
        P = np.ones(len(returns)) * 1e-4
        ll = -500.0
        trajectory = np.random.randn(len(returns))
        
        cache.put(key, mu, P, ll, trajectory)
        
        # Get result
        cached = cache.get(key)
        
        assert cached is not None
        assert np.allclose(cached.mu_filtered, mu)
        assert cached.log_likelihood == ll
    
    def test_cache_miss(self, sample_data, clean_cache):
        """Non-existent key returns None."""
        returns, vol = sample_data
        cache = get_filter_cache()
        
        key = FilterCacheKey.create(
            returns=returns,
            vol=vol,
            model_id="nonexistent",
            q=1e-6,
            c=1.0,
            phi=0.5,
        )
        
        cached = cache.get(key)
        assert cached is None
    
    def test_cache_stores_copies(self, sample_data, clean_cache):
        """Cache stores copies, not references."""
        returns, vol = sample_data
        cache = get_filter_cache()
        
        key = FilterCacheKey.create(
            returns=returns,
            vol=vol,
            model_id="test_model",
            q=1e-6,
            c=1.0,
            phi=0.5,
        )
        
        mu = np.zeros(len(returns))
        P = np.ones(len(returns)) * 1e-4
        ll = -500.0
        trajectory = np.zeros(len(returns))
        
        cache.put(key, mu, P, ll, trajectory)
        
        # Modify original array
        mu[0] = 999.0
        
        # Cached value should be unchanged
        cached = cache.get(key)
        assert cached.mu_filtered[0] == 0.0
    
    def test_lru_eviction(self, sample_data, clean_cache):
        """LRU eviction works correctly."""
        returns, vol = sample_data
        
        # Create small cache
        cache = FilterResultCache(max_size=3)
        
        # Add 3 entries
        for i in range(3):
            key = FilterCacheKey.create(
                returns=returns,
                vol=vol,
                model_id=f"model_{i}",
                q=1e-6,
                c=1.0,
                phi=0.5,
            )
            cache.put(
                key,
                np.zeros(len(returns)),
                np.ones(len(returns)) * 1e-4,
                -500.0,
                np.zeros(len(returns))
            )
        
        # Access first entry to make it recently used
        key_0 = FilterCacheKey.create(
            returns=returns,
            vol=vol,
            model_id="model_0",
            q=1e-6,
            c=1.0,
            phi=0.5,
        )
        cache.get(key_0)
        
        # Add 4th entry - should evict model_1 (least recently used)
        key_3 = FilterCacheKey.create(
            returns=returns,
            vol=vol,
            model_id="model_3",
            q=1e-6,
            c=1.0,
            phi=0.5,
        )
        cache.put(
            key_3,
            np.zeros(len(returns)),
            np.ones(len(returns)) * 1e-4,
            -500.0,
            np.zeros(len(returns))
        )
        
        # model_0 should still be present (accessed recently)
        assert cache.get(key_0) is not None
        
        # model_1 should be evicted
        key_1 = FilterCacheKey.create(
            returns=returns,
            vol=vol,
            model_id="model_1",
            q=1e-6,
            c=1.0,
            phi=0.5,
        )
        assert cache.get(key_1) is None


class TestFoldSlicing:
    """Test fold-aware likelihood computation from cache."""
    
    def test_fold_slicing_equivalence(self, sample_data, clean_cache):
        """Sliced likelihoods equal independently computed fold likelihoods."""
        returns, vol = sample_data
        q, c, phi = 1e-6, 1.0, 0.7
        
        # Get full filter result with trajectory
        mu, P, ll_full, trajectory = _kalman_filter_phi_with_trajectory(
            returns, vol, q, c, phi
        )
        
        # Create cache value
        cache_value = FilterCacheValue(
            mu_filtered=mu,
            P_filtered=P,
            log_likelihood=ll_full,
            loglik_trajectory=trajectory,
            timestamp=time.time(),
        )
        
        # Define folds
        n = len(returns)
        fold_splits = [
            (0, 100, 100, 200),   # train [0:100], test [100:200]
            (0, 200, 200, 300),   # train [0:200], test [200:300]
            (0, 300, 300, 400),   # train [0:300], test [300:400]
        ]
        
        # Compute CV likelihood via slicing
        cv_ll_sliced = compute_cv_likelihood_from_cache(cache_value, fold_splits)
        
        # Compute CV likelihood independently (sum of test fold likelihoods)
        cv_ll_independent = 0.0
        for _, _, test_start, test_end in fold_splits:
            cv_ll_independent += np.sum(trajectory[test_start:test_end])
        
        # Should be numerically equivalent
        np.testing.assert_allclose(cv_ll_sliced, cv_ll_independent, rtol=1e-12)
    
    def test_fold_slicing_additivity(self, sample_data, clean_cache):
        """Full likelihood equals sum of all fold slices."""
        returns, vol = sample_data
        q, c, phi = 1e-6, 1.0, 0.7
        
        mu, P, ll_full, trajectory = _kalman_filter_phi_with_trajectory(
            returns, vol, q, c, phi
        )
        
        cache_value = FilterCacheValue(
            mu_filtered=mu,
            P_filtered=P,
            log_likelihood=ll_full,
            loglik_trajectory=trajectory,
            timestamp=time.time(),
        )
        
        # Slice entire trajectory into non-overlapping parts
        n = len(returns)
        fold_size = 100
        slices = []
        for start in range(0, n, fold_size):
            end = min(start + fold_size, n)
            slices.append(cache_value.slice_likelihood_for_fold(start, end))
        
        # Sum of slices should equal total
        np.testing.assert_allclose(sum(slices), ll_full, rtol=1e-10)


class TestNumericalInvariance:
    """Test that cached results are numerically equivalent to fresh computation."""
    
    def test_phi_gaussian_cache_equivalence(self, sample_data, clean_cache):
        """Cached φ-Gaussian filter matches fresh computation."""
        returns, vol = sample_data
        q, c, phi = 1e-6, 1.0, 0.7
        
        # Fresh computation
        mu_fresh, P_fresh, ll_fresh = PhiGaussianDriftModel.filter(returns, vol, q, c, phi)
        
        # Cached computation (first call - miss)
        mu_cached, P_cached, ll_cached, _ = PhiGaussianDriftModel.filter_with_trajectory(
            returns, vol, q, c, phi
        )
        
        # Results should match
        np.testing.assert_allclose(mu_cached, mu_fresh, rtol=1e-10)
        np.testing.assert_allclose(P_cached, P_fresh, rtol=1e-10)
        np.testing.assert_allclose(ll_cached, ll_fresh, rtol=1e-10)
        
        # Second call - should be cache hit
        stats_before = get_cache_stats()
        hits_before = stats_before.hits
        
        mu_cached2, P_cached2, ll_cached2, _ = PhiGaussianDriftModel.filter_with_trajectory(
            returns, vol, q, c, phi
        )
        
        stats_after = get_cache_stats()
        
        # Should have registered a hit
        assert stats_after.hits > hits_before
        
        # Results still match
        np.testing.assert_allclose(mu_cached2, mu_fresh, rtol=1e-10)
    
    def test_phi_student_t_cache_equivalence(self, sample_data, clean_cache):
        """Cached φ-Student-t filter matches fresh computation."""
        returns, vol = sample_data
        q, c, phi, nu = 1e-6, 1.0, 0.5, 6.0
        
        # Fresh computation
        mu_fresh, P_fresh, ll_fresh = PhiStudentTDriftModel.filter_phi(returns, vol, q, c, phi, nu)
        
        # Cached computation
        mu_cached, P_cached, ll_cached, _ = PhiStudentTDriftModel.filter_with_trajectory(
            returns, vol, q, c, phi, nu
        )
        
        # Results should match (slightly looser tolerance due to gamma computation path)
        np.testing.assert_allclose(mu_cached, mu_fresh, rtol=1e-8)
        np.testing.assert_allclose(P_cached, P_fresh, rtol=1e-8)
        np.testing.assert_allclose(ll_cached, ll_fresh, rtol=1e-6)


class TestCacheStatistics:
    """Test cache statistics tracking."""
    
    def test_hit_miss_counting(self, sample_data, clean_cache):
        """Cache hits and misses are counted correctly."""
        returns, vol = sample_data
        
        reset_cache_stats()
        stats = get_cache_stats()
        assert stats.hits == 0
        assert stats.misses == 0
        
        # First call - miss
        PhiGaussianDriftModel.filter_with_trajectory(returns, vol, 1e-6, 1.0, 0.7)
        
        stats = get_cache_stats()
        assert stats.misses >= 1
        
        # Second call with same params - hit
        PhiGaussianDriftModel.filter_with_trajectory(returns, vol, 1e-6, 1.0, 0.7)
        
        stats = get_cache_stats()
        assert stats.hits >= 1
    
    def test_hit_rate_computation(self, sample_data, clean_cache):
        """Hit rate is computed correctly."""
        returns, vol = sample_data
        
        reset_cache_stats()
        
        # 1 miss, then 3 hits
        for _ in range(4):
            PhiGaussianDriftModel.filter_with_trajectory(returns, vol, 1e-6, 1.0, 0.7)
        
        stats = get_cache_stats()
        
        # 3 hits out of 4 calls = 75% hit rate
        assert stats.hit_rate >= 0.7


class TestCacheDisabling:
    """Test cache enable/disable functionality."""
    
    def test_disabled_cache_always_misses(self, sample_data, clean_cache):
        """Disabled cache always returns None."""
        returns, vol = sample_data
        
        # Disable cache
        set_filter_cache_enabled(False)
        
        try:
            reset_cache_stats()
            
            # Multiple calls
            for _ in range(3):
                # Use internal function to test cache directly
                result = PhiGaussianDriftModel.filter_with_trajectory(
                    returns, vol, 1e-6, 1.0, 0.7, use_cache=True
                )
            
            # Should have no hits (cache disabled)
            stats = get_cache_stats()
            assert stats.hits == 0
            
        finally:
            # Re-enable cache
            set_filter_cache_enabled(True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
