#!/usr/bin/env python3
"""
===============================================================================
COMPREHENSIVE TEST SUITE: Deterministic Kalman Filter Result Cache
===============================================================================

Tests the filter cache infrastructure for eliminating redundant filter executions.
"""

import os
import sys
import time
import numpy as np
import pytest
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.filter_cache import (
    FilterCacheKey,
    FilterCacheValue,
    FilterResultCache,
    get_filter_cache,
    clear_filter_cache,
    get_cache_stats,
    reset_cache_stats,
    _hash_array,
    _round_param,
    FILTER_CACHE_ENABLED,
)

from models.phi_gaussian import (
    PhiGaussianDriftModel,
    _kalman_filter_phi,
    _kalman_filter_phi_with_trajectory,
)

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
    
    def test_cache_key_hashable(self, sample_data):
        """Cache keys are hashable."""
        returns, vol = sample_data
        key = FilterCacheKey.create(
            returns=returns, vol=vol, model_id="test", q=1e-6, c=1.0, phi=0.5
        )
        d = {key: "value"}
        assert d[key] == "value"
    
    def test_parameter_rounding(self):
        """Parameters are rounded correctly."""
        val1 = _round_param(0.12345678901234)
        val2 = _round_param(0.12345678901235)
        assert val1 == val2


class TestCacheOperations:
    """Test basic cache operations."""
    
    def test_cache_hit(self, sample_data, clean_cache):
        """Cached result is retrievable."""
        returns, vol = sample_data
        cache = get_filter_cache()
        
        key = FilterCacheKey.create(
            returns=returns, vol=vol, model_id="test", q=1e-6, c=1.0, phi=0.5
        )
        
        mu = np.zeros(len(returns))
        P = np.ones(len(returns)) * 1e-4
        ll = -500.0
        trajectory = np.random.randn(len(returns))
        
        cache.put(key, mu, P, ll, trajectory)
        cached = cache.get(key)
        
        assert cached is not None
        assert np.allclose(cached.mu_filtered, mu)
        assert cached.log_likelihood == ll
    
    def test_cache_miss(self, sample_data, clean_cache):
        """Non-existent key returns None."""
        returns, vol = sample_data
        cache = get_filter_cache()
        
        key = FilterCacheKey.create(
            returns=returns, vol=vol, model_id="nonexistent", q=1e-6, c=1.0, phi=0.5
        )
        assert cache.get(key) is None
    
    def test_cache_stores_copies(self, sample_data, clean_cache):
        """Cache stores copies, not references."""
        returns, vol = sample_data
        cache = get_filter_cache()
        
        key = FilterCacheKey.create(
            returns=returns, vol=vol, model_id="test", q=1e-6, c=1.0, phi=0.5
        )
        
        mu = np.zeros(len(returns))
        cache.put(key, mu, np.ones(len(returns)) * 1e-4, -500.0, np.zeros(len(returns)))
        
        mu[0] = 999.0
        cached = cache.get(key)
        assert cached.mu_filtered[0] == 0.0


class TestFoldSlicing:
    """Test fold-aware likelihood slicing."""
    
    def test_fold_slicing_sum_equals_total(self, sample_data, clean_cache):
        """Sum of fold slices equals total likelihood."""
        returns, vol = sample_data
        q, c, phi = 1e-6, 1.0, 0.7
        
        mu, P, ll_full, trajectory = _kalman_filter_phi_with_trajectory(returns, vol, q, c, phi)
        
        # Sum trajectory parts
        n = len(returns)
        fold_size = 100
        slice_sum = 0.0
        for start in range(0, n, fold_size):
            end = min(start + fold_size, n)
            slice_sum += np.sum(trajectory[start:end])
        
        np.testing.assert_allclose(slice_sum, ll_full, rtol=1e-12)


class TestNumericalInvariance:
    """Test that cached results match fresh computation."""
    
    def test_phi_gaussian_cached_equals_uncached(self, sample_data, clean_cache):
        """Cached φ-Gaussian matches uncached."""
        returns, vol = sample_data
        q, c, phi = 1e-6, 1.0, 0.7
        
        mu_fresh, P_fresh, ll_fresh = _kalman_filter_phi(returns, vol, q, c, phi)
        mu_cached, P_cached, ll_cached, traj = PhiGaussianDriftModel.filter_with_trajectory(
            returns, vol, q, c, phi, use_cache=True
        )
        
        np.testing.assert_allclose(mu_cached, mu_fresh, rtol=1e-12)
        np.testing.assert_allclose(ll_cached, ll_fresh, rtol=1e-12)
    
    def test_cache_hit_equals_miss(self, sample_data, clean_cache):
        """Cache hit returns identical results to miss."""
        returns, vol = sample_data
        q, c, phi = 1e-6, 1.0, 0.7
        
        mu1, P1, ll1, traj1 = PhiGaussianDriftModel.filter_with_trajectory(returns, vol, q, c, phi)
        mu2, P2, ll2, traj2 = PhiGaussianDriftModel.filter_with_trajectory(returns, vol, q, c, phi)
        
        np.testing.assert_array_equal(mu1, mu2)
        assert ll1 == ll2


class TestCacheStatistics:
    """Test cache statistics tracking."""
    
    def test_hit_miss_counting(self, sample_data, clean_cache):
        """Cache hits and misses are counted."""
        returns, vol = sample_data
        
        reset_cache_stats()
        
        PhiGaussianDriftModel.filter_with_trajectory(returns, vol, 1e-6, 1.0, 0.7)
        stats = get_cache_stats()
        assert stats.misses >= 1
        
        PhiGaussianDriftModel.filter_with_trajectory(returns, vol, 1e-6, 1.0, 0.7)
        stats = get_cache_stats()
        assert stats.hits >= 1


class TestPerformance:
    """Test cache performance."""
    
    def test_cache_provides_speedup(self, sample_data, clean_cache):
        """Cached calls are faster."""
        returns, vol = sample_data
        q, c, phi = 1e-6, 1.0, 0.7
        
        # Warm up
        PhiGaussianDriftModel.filter_with_trajectory(returns, vol, q, c, phi)
        
        n_runs = 20
        
        # Time uncached
        start = time.perf_counter()
        for _ in range(n_runs):
            _kalman_filter_phi_with_trajectory(returns, vol, q, c, phi)
        uncached_time = time.perf_counter() - start
        
        # Time cached
        start = time.perf_counter()
        for _ in range(n_runs):
            PhiGaussianDriftModel.filter_with_trajectory(returns, vol, q, c, phi)
        cached_time = time.perf_counter() - start
        
        speedup = uncached_time / cached_time
        print(f"\nCache speedup: {speedup:.1f}x")
        assert speedup > 2.0


class TestNumericalStability:
    """Test numerical stability."""
    
    def test_extreme_volatility(self, clean_cache):
        """Handle extreme volatility."""
        np.random.seed(42)
        n = 500
        returns = np.random.randn(n) * 0.5
        vol = np.ones(n) * 0.001
        
        mu, P, ll, traj = PhiGaussianDriftModel.filter_with_trajectory(returns, vol, 1e-6, 1.0, 0.7)
        
        assert np.all(np.isfinite(mu))
        assert np.isfinite(ll)


class TestIntegration:
    """Integration tests."""
    
    def test_multiple_models(self, sample_data, clean_cache):
        """Multiple models share cache."""
        returns, vol = sample_data
        
        reset_cache_stats()
        
        PhiGaussianDriftModel.filter_with_trajectory(returns, vol, 1e-6, 1.0, 0.7)
        PhiStudentTDriftModel.filter_with_trajectory(returns, vol, 1e-6, 1.0, 0.5, 6.0)
        
        PhiGaussianDriftModel.filter_with_trajectory(returns, vol, 1e-6, 1.0, 0.7)
        PhiStudentTDriftModel.filter_with_trajectory(returns, vol, 1e-6, 1.0, 0.5, 6.0)
        
        stats = get_cache_stats()
        assert stats.hits == 2
        assert stats.misses == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
