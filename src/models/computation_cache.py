"""
Story 7.2: Caching Layer for Repeated Computations.

LRU cache for regime classification, EWMA volatility, and feature computation.
Content-hash based invalidation with hit/miss counters.

Usage:
    from models.computation_cache import ComputationCache
    cache = ComputationCache()
    regime = cache.get_regime("SPY", returns_array)
"""
import hashlib
import numpy as np
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple


class CacheStats:
    """Track cache hit/miss statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
    
    @property
    def total(self):
        return self.hits + self.misses
    
    @property
    def hit_rate(self):
        return self.hits / self.total if self.total > 0 else 0.0
    
    def record_hit(self):
        self.hits += 1
    
    def record_miss(self):
        self.misses += 1
    
    def reset(self):
        self.hits = 0
        self.misses = 0


def compute_returns_hash(returns: np.ndarray, tail: int = 30) -> str:
    """
    Fast hash of most recent returns for cache invalidation.
    Only hashes the last `tail` values for speed.
    
    Args:
        returns: Array of returns.
        tail: Number of recent values to hash.
    
    Returns:
        16-character hex hash.
    """
    data = returns[-tail:] if len(returns) > tail else returns
    data_bytes = data.tobytes()
    return hashlib.sha256(data_bytes).hexdigest()[:16]


class ComputationCache:
    """
    Cache for expensive intermediate results during signal generation.
    Content-hash based invalidation.
    """
    
    def __init__(self, max_size: int = 500):
        self._regime_cache: Dict[str, Tuple[str, Any]] = {}
        self._vol_cache: Dict[str, Tuple[str, float]] = {}
        self._feature_cache: Dict[str, Tuple[str, Dict]] = {}
        self._max_size = max_size
        
        self.stats = {
            "regime": CacheStats(),
            "vol": CacheStats(),
            "feature": CacheStats(),
        }
    
    def get_regime(
        self,
        symbol: str,
        returns: np.ndarray,
        classify_fn: Callable[[np.ndarray], Any],
    ) -> Any:
        """
        Get cached regime classification.
        
        Args:
            symbol: Asset symbol.
            returns: Returns array.
            classify_fn: Function to compute regime if not cached.
        
        Returns:
            Regime classification result.
        """
        h = compute_returns_hash(returns)
        key = symbol
        
        if key in self._regime_cache:
            cached_hash, cached_result = self._regime_cache[key]
            if cached_hash == h:
                self.stats["regime"].record_hit()
                return cached_result
        
        # Cache miss
        self.stats["regime"].record_miss()
        result = classify_fn(returns)
        self._regime_cache[key] = (h, result)
        self._evict_if_needed(self._regime_cache)
        return result
    
    def get_volatility(
        self,
        symbol: str,
        returns: np.ndarray,
        vol_fn: Callable[[np.ndarray], float],
    ) -> float:
        """Get cached EWMA volatility estimate."""
        h = compute_returns_hash(returns)
        key = symbol
        
        if key in self._vol_cache:
            cached_hash, cached_vol = self._vol_cache[key]
            if cached_hash == h:
                self.stats["vol"].record_hit()
                return cached_vol
        
        self.stats["vol"].record_miss()
        vol = vol_fn(returns)
        self._vol_cache[key] = (h, vol)
        self._evict_if_needed(self._vol_cache)
        return vol
    
    def get_features(
        self,
        symbol: str,
        returns: np.ndarray,
        feature_fn: Callable[[np.ndarray], Dict],
    ) -> Dict:
        """Get cached feature computation."""
        h = compute_returns_hash(returns)
        key = symbol
        
        if key in self._feature_cache:
            cached_hash, cached_features = self._feature_cache[key]
            if cached_hash == h:
                self.stats["feature"].record_hit()
                return cached_features
        
        self.stats["feature"].record_miss()
        features = feature_fn(returns)
        self._feature_cache[key] = (h, features)
        self._evict_if_needed(self._feature_cache)
        return features
    
    def clear(self):
        """Clear all caches."""
        self._regime_cache.clear()
        self._vol_cache.clear()
        self._feature_cache.clear()
        for s in self.stats.values():
            s.reset()
    
    def get_stats_summary(self) -> Dict:
        """Get cache statistics summary."""
        return {
            name: {
                "hits": s.hits,
                "misses": s.misses,
                "hit_rate": s.hit_rate,
            }
            for name, s in self.stats.items()
        }
    
    def _evict_if_needed(self, cache: Dict):
        """Evict oldest entries if over max size."""
        while len(cache) > self._max_size:
            # Remove first (oldest) entry
            oldest_key = next(iter(cache))
            del cache[oldest_key]
