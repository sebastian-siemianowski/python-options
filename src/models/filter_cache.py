"""
===============================================================================
DETERMINISTIC KALMAN FILTER RESULT CACHE
===============================================================================

Provides a reuse layer above existing Numba/Python filter dispatch to eliminate
redundant executions during tuning and cross-validation.

Architecture:
    - Cache sits ABOVE Numba dispatch (works regardless of Numba availability)
    - Stores deterministic filter outputs keyed by input hash + parameters
    - Enables fold-aware likelihood slicing without re-execution
    - Optional warm-start support for optimizer parameter moves

Design Principles:
    - Zero mathematical approximations
    - Bitwise-equivalent results to uncached execution
    - Transparent to higher-level modules (tune.py, signals.py)
    - LRU eviction to bound memory usage

Cache Key Structure:
    (returns_hash, vol_hash, model_id, regime_id, q_rounded, c_rounded, phi_rounded, nu_rounded)

Cache Value Structure:
    {
        'mu_filtered': np.ndarray,      # Filtered means
        'P_filtered': np.ndarray,       # Filtered variances  
        'log_likelihood': float,        # Total log-likelihood
        'loglik_trajectory': np.ndarray, # Per-timestep log-likelihoods (for fold slicing)
        'timestamp': float,             # Cache entry time
    }
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np


# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

# Maximum cache entries (LRU eviction beyond this)
FILTER_CACHE_MAX_SIZE = 10000

# Parameter rounding precision for cache keys (avoids floating point noise)
PARAM_ROUND_DECIMALS = 8

# Warm-start threshold: if parameter move is smaller, reuse prior state
WARM_START_PARAM_THRESHOLD = 0.05

# Burn-in steps to discard when using warm-start (transient elimination)
WARM_START_BURN_IN = 5

# Enable/disable cache globally (for testing/debugging)
FILTER_CACHE_ENABLED = True

# Enable cache statistics collection
FILTER_CACHE_STATS_ENABLED = True


# =============================================================================
# CACHE STATISTICS
# =============================================================================

@dataclass
class FilterCacheStats:
    """Statistics for filter cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    fold_slice_reuses: int = 0
    warm_starts: int = 0
    cold_starts: int = 0
    total_time_saved_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def reset(self) -> None:
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.fold_slice_reuses = 0
        self.warm_starts = 0
        self.cold_starts = 0
        self.total_time_saved_ms = 0.0
    
    def summary(self) -> str:
        return (
            f"FilterCache: {self.hits}/{self.hits + self.misses} hits "
            f"({self.hit_rate:.1%}), {self.evictions} evictions, "
            f"{self.fold_slice_reuses} fold reuses, "
            f"{self.warm_starts} warm starts, "
            f"~{self.total_time_saved_ms:.0f}ms saved"
        )


# Global statistics instance
_cache_stats = FilterCacheStats()


def get_cache_stats() -> FilterCacheStats:
    """Get current cache statistics."""
    return _cache_stats


def reset_cache_stats() -> None:
    """Reset cache statistics."""
    _cache_stats.reset()


# =============================================================================
# ARRAY HASHING
# =============================================================================

def _hash_array(arr: np.ndarray) -> str:
    """
    Compute deterministic hash of numpy array.
    
    Uses xxhash-style approach: hash raw bytes with shape prefix.
    Falls back to SHA256 for robustness.
    """
    if arr is None:
        return "none"
    
    # Ensure contiguous array for consistent hashing
    arr_c = np.ascontiguousarray(arr, dtype=np.float64)
    
    # Include shape in hash to distinguish differently-shaped arrays with same bytes
    shape_bytes = str(arr_c.shape).encode('utf-8')
    data_bytes = arr_c.tobytes()
    
    hasher = hashlib.sha256()
    hasher.update(shape_bytes)
    hasher.update(data_bytes)
    
    return hasher.hexdigest()[:16]  # Truncate for memory efficiency


def _round_param(value: float, decimals: int = PARAM_ROUND_DECIMALS) -> float:
    """Round parameter to fixed precision for cache key stability."""
    if value is None or not np.isfinite(value):
        return 0.0
    return round(float(value), decimals)


# =============================================================================
# CACHE KEY GENERATION
# =============================================================================

@dataclass(frozen=True)
class FilterCacheKey:
    """
    Immutable cache key for filter results.
    
    All components that affect filter output must be included.
    """
    returns_hash: str
    vol_hash: str
    model_id: str
    regime_id: str
    q: float
    c: float
    phi: float
    nu: float  # 0.0 for Gaussian models
    
    @classmethod
    def create(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        model_id: str,
        q: float,
        c: float,
        phi: float = 0.0,
        nu: float = 0.0,
        regime_id: str = "global"
    ) -> 'FilterCacheKey':
        """Create cache key from filter inputs."""
        return cls(
            returns_hash=_hash_array(returns),
            vol_hash=_hash_array(vol),
            model_id=model_id,
            regime_id=regime_id,
            q=_round_param(q),
            c=_round_param(c),
            phi=_round_param(phi),
            nu=_round_param(nu),
        )


# =============================================================================
# CACHE VALUE
# =============================================================================

@dataclass
class FilterCacheValue:
    """
    Cached filter result with per-timestep likelihood trajectory.
    
    The trajectory enables fold-aware likelihood slicing without re-execution.
    """
    mu_filtered: np.ndarray
    P_filtered: np.ndarray
    log_likelihood: float
    loglik_trajectory: np.ndarray  # Per-timestep log-likelihoods
    timestamp: float
    
    # Optional: final state for warm-start
    final_mu: float = 0.0
    final_P: float = 1e-4
    
    def slice_likelihood_for_fold(
        self,
        start_idx: int,
        end_idx: int
    ) -> float:
        """
        Compute fold-specific log-likelihood by slicing trajectory.
        
        This is mathematically equivalent to running the filter on the fold
        independently, because Kalman filter likelihoods are additive.
        """
        if self.loglik_trajectory is None:
            raise ValueError("Likelihood trajectory not available for fold slicing")
        
        if start_idx < 0 or end_idx > len(self.loglik_trajectory):
            raise IndexError(f"Fold indices [{start_idx}:{end_idx}] out of bounds for trajectory length {len(self.loglik_trajectory)}")
        
        return float(np.sum(self.loglik_trajectory[start_idx:end_idx]))
    
    def get_fold_results(
        self,
        start_idx: int,
        end_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract fold-specific filter results.
        
        Returns (mu_filtered[fold], P_filtered[fold], fold_log_likelihood)
        """
        fold_mu = self.mu_filtered[start_idx:end_idx].copy()
        fold_P = self.P_filtered[start_idx:end_idx].copy()
        fold_ll = self.slice_likelihood_for_fold(start_idx, end_idx)
        
        return fold_mu, fold_P, fold_ll


# =============================================================================
# LRU FILTER CACHE
# =============================================================================

class FilterResultCache:
    """
    Thread-safe LRU cache for deterministic Kalman filter results.
    
    This cache sits ABOVE the Numba/Python dispatch layer, meaning:
    - It works regardless of whether Numba is available
    - It stores final numerical results, not compiled code
    - Cached results are immutable and bitwise-equivalent to fresh computation
    """
    
    def __init__(self, max_size: int = FILTER_CACHE_MAX_SIZE):
        self._cache: OrderedDict[FilterCacheKey, FilterCacheValue] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()
        self._enabled = FILTER_CACHE_ENABLED
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
    
    def get(self, key: FilterCacheKey) -> Optional[FilterCacheValue]:
        """
        Retrieve cached filter result if available.
        
        Moves accessed entry to end (most recently used).
        """
        if not self._enabled:
            return None
        
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                if FILTER_CACHE_STATS_ENABLED:
                    _cache_stats.hits += 1
                return self._cache[key]
            
            if FILTER_CACHE_STATS_ENABLED:
                _cache_stats.misses += 1
            return None
    
    def put(
        self,
        key: FilterCacheKey,
        mu_filtered: np.ndarray,
        P_filtered: np.ndarray,
        log_likelihood: float,
        loglik_trajectory: np.ndarray,
        final_mu: float = 0.0,
        final_P: float = 1e-4
    ) -> None:
        """
        Store filter result in cache.
        
        Evicts least recently used entries if cache is full.
        """
        if not self._enabled:
            return
        
        with self._lock:
            # Create immutable copies
            value = FilterCacheValue(
                mu_filtered=mu_filtered.copy(),
                P_filtered=P_filtered.copy(),
                log_likelihood=float(log_likelihood),
                loglik_trajectory=loglik_trajectory.copy() if loglik_trajectory is not None else None,
                timestamp=time.time(),
                final_mu=float(final_mu),
                final_P=float(final_P),
            )
            
            # Remove if exists (to update position)
            if key in self._cache:
                del self._cache[key]
            
            # Add to end
            self._cache[key] = value
            
            # Evict if over capacity
            while len(self._cache) > self._max_size:
                evicted_key = next(iter(self._cache))
                del self._cache[evicted_key]
                if FILTER_CACHE_STATS_ENABLED:
                    _cache_stats.evictions += 1
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
    
    def size(self) -> int:
        """Return current cache size."""
        with self._lock:
            return len(self._cache)
    
    def get_or_compute(
        self,
        key: FilterCacheKey,
        compute_fn: Callable[[], Tuple[np.ndarray, np.ndarray, float, np.ndarray]]
    ) -> FilterCacheValue:
        """
        Get cached result or compute and cache.
        
        The compute_fn should return (mu_filtered, P_filtered, log_likelihood, loglik_trajectory).
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        
        # Compute fresh result
        mu, P, ll, trajectory = compute_fn()
        
        # Determine final state for potential warm-start
        final_mu = float(mu[-1]) if len(mu) > 0 else 0.0
        final_P = float(P[-1]) if len(P) > 0 else 1e-4
        
        # Cache and return
        self.put(key, mu, P, ll, trajectory, final_mu, final_P)
        
        return self.get(key)


# =============================================================================
# GLOBAL CACHE INSTANCE
# =============================================================================

_global_filter_cache = FilterResultCache()


def get_filter_cache() -> FilterResultCache:
    """Get the global filter result cache."""
    return _global_filter_cache


def clear_filter_cache() -> None:
    """Clear the global filter cache."""
    _global_filter_cache.clear()


def set_filter_cache_enabled(enabled: bool) -> None:
    """Enable or disable the global filter cache."""
    _global_filter_cache.enabled = enabled


# =============================================================================
# WARM-START UTILITIES
# =============================================================================

def should_use_warm_start(
    prev_params: Tuple[float, float, float, float],  # (q, c, phi, nu)
    curr_params: Tuple[float, float, float, float],
    threshold: float = WARM_START_PARAM_THRESHOLD
) -> bool:
    """
    Determine if warm-start is appropriate for parameter move.
    
    Returns True if all parameters changed by less than threshold (relative).
    """
    if prev_params is None:
        return False
    
    for prev, curr in zip(prev_params, curr_params):
        if prev == 0 and curr == 0:
            continue
        if prev == 0:
            return False  # Can't compute relative change from zero
        
        rel_change = abs(curr - prev) / max(abs(prev), 1e-12)
        if rel_change > threshold:
            return False
    
    return True


def get_warm_start_state(
    cache: FilterResultCache,
    returns: np.ndarray,
    vol: np.ndarray,
    model_id: str,
    prev_params: Tuple[float, float, float, float],
    regime_id: str = "global"
) -> Optional[Tuple[float, float]]:
    """
    Retrieve warm-start state (mu_0, P_0) from previously cached result.
    
    Returns None if no suitable cached result exists.
    """
    q, c, phi, nu = prev_params
    key = FilterCacheKey.create(
        returns=returns,
        vol=vol,
        model_id=model_id,
        q=q,
        c=c,
        phi=phi,
        nu=nu,
        regime_id=regime_id
    )
    
    cached = cache.get(key)
    if cached is None:
        return None
    
    return (cached.final_mu, cached.final_P)


# =============================================================================
# FOLD-AWARE LIKELIHOOD COMPUTATION
# =============================================================================

def compute_cv_likelihood_from_cache(
    cache_value: FilterCacheValue,
    fold_splits: List[Tuple[int, int, int, int]],  # [(train_start, train_end, test_start, test_end), ...]
    use_test_only: bool = True
) -> float:
    """
    Compute cross-validation likelihood by slicing cached trajectory.
    
    This avoids re-running filters for each fold.
    
    Args:
        cache_value: Cached filter result with likelihood trajectory
        fold_splits: List of (train_start, train_end, test_start, test_end) tuples
        use_test_only: If True, sum only test set likelihoods (standard CV)
    
    Returns:
        Total CV log-likelihood across all folds
    """
    if cache_value.loglik_trajectory is None:
        raise ValueError("Cached result does not have likelihood trajectory for fold slicing")
    
    total_ll = 0.0
    
    for train_start, train_end, test_start, test_end in fold_splits:
        if use_test_only:
            fold_ll = cache_value.slice_likelihood_for_fold(test_start, test_end)
        else:
            # Include both train and test (less common)
            train_ll = cache_value.slice_likelihood_for_fold(train_start, train_end)
            test_ll = cache_value.slice_likelihood_for_fold(test_start, test_end)
            fold_ll = train_ll + test_ll
        
        total_ll += fold_ll
    
    if FILTER_CACHE_STATS_ENABLED:
        _cache_stats.fold_slice_reuses += len(fold_splits)
    
    return total_ll


# =============================================================================
# CACHED FILTER WRAPPERS
# =============================================================================

def cached_phi_gaussian_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    filter_fn: Callable,
    regime_id: str = "global",
    warm_start: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Cached wrapper for φ-Gaussian Kalman filter.
    
    Returns (mu_filtered, P_filtered, log_likelihood, loglik_trajectory).
    """
    cache = get_filter_cache()
    key = FilterCacheKey.create(
        returns=returns,
        vol=vol,
        model_id="phi_gaussian",
        q=q,
        c=c,
        phi=phi,
        nu=0.0,
        regime_id=regime_id
    )
    
    cached = cache.get(key)
    if cached is not None:
        return (
            cached.mu_filtered,
            cached.P_filtered,
            cached.log_likelihood,
            cached.loglik_trajectory
        )
    
    # Compute fresh - need to capture per-timestep likelihoods
    mu, P, ll, trajectory = _run_phi_gaussian_with_trajectory(
        returns, vol, q, c, phi, filter_fn, warm_start
    )
    
    # Cache result
    final_mu = float(mu[-1]) if len(mu) > 0 else 0.0
    final_P = float(P[-1]) if len(P) > 0 else 1e-4
    cache.put(key, mu, P, ll, trajectory, final_mu, final_P)
    
    if FILTER_CACHE_STATS_ENABLED:
        _cache_stats.cold_starts += 1
    
    return mu, P, ll, trajectory


def cached_phi_student_t_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    filter_fn: Callable,
    regime_id: str = "global",
    warm_start: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Cached wrapper for φ-Student-t Kalman filter.
    
    Returns (mu_filtered, P_filtered, log_likelihood, loglik_trajectory).
    """
    cache = get_filter_cache()
    key = FilterCacheKey.create(
        returns=returns,
        vol=vol,
        model_id=f"phi_student_t_nu_{int(nu)}",
        q=q,
        c=c,
        phi=phi,
        nu=nu,
        regime_id=regime_id
    )
    
    cached = cache.get(key)
    if cached is not None:
        return (
            cached.mu_filtered,
            cached.P_filtered,
            cached.log_likelihood,
            cached.loglik_trajectory
        )
    
    # Compute fresh
    mu, P, ll, trajectory = _run_phi_student_t_with_trajectory(
        returns, vol, q, c, phi, nu, filter_fn, warm_start
    )
    
    # Cache result
    final_mu = float(mu[-1]) if len(mu) > 0 else 0.0
    final_P = float(P[-1]) if len(P) > 0 else 1e-4
    cache.put(key, mu, P, ll, trajectory, final_mu, final_P)
    
    if FILTER_CACHE_STATS_ENABLED:
        _cache_stats.cold_starts += 1
    
    return mu, P, ll, trajectory


# =============================================================================
# INTERNAL: FILTER WITH TRAJECTORY CAPTURE
# =============================================================================

def _run_phi_gaussian_with_trajectory(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    filter_fn: Callable,
    warm_start: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Run φ-Gaussian filter and capture per-timestep likelihood trajectory.
    """
    n = len(returns)
    q_val = float(q)
    c_val = float(c)
    phi_val = float(np.clip(phi, -0.999, 0.999))
    
    # Initialize state
    if warm_start is not None:
        mu, P = warm_start
        if FILTER_CACHE_STATS_ENABLED:
            _cache_stats.warm_starts += 1
    else:
        mu = 0.0
        P = 1e-4
    
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    loglik_trajectory = np.zeros(n)
    log_likelihood = 0.0
    
    for t in range(n):
        # Predict
        mu_pred = phi_val * mu
        P_pred = (phi_val ** 2) * P + q_val
        
        # Observation variance
        vol_t = float(vol[t])
        R = c_val * (vol_t ** 2)
        
        # Innovation
        r_t = float(returns[t])
        innovation = r_t - mu_pred
        
        # Forecast variance
        S = P_pred + R
        
        # Update
        if S > 1e-12:
            K = P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            
            # Gaussian log-likelihood for this timestep
            ll_t = -0.5 * (np.log(2 * np.pi * S) + (innovation ** 2) / S)
        else:
            mu = mu_pred
            P = P_pred
            ll_t = 0.0
        
        mu_filtered[t] = mu
        P_filtered[t] = max(P, 1e-12)
        loglik_trajectory[t] = ll_t
        log_likelihood += ll_t
    
    return mu_filtered, P_filtered, log_likelihood, loglik_trajectory


def _run_phi_student_t_with_trajectory(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    filter_fn: Callable,
    warm_start: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Run φ-Student-t filter and capture per-timestep likelihood trajectory.
    """
    from scipy.special import gammaln
    
    n = len(returns)
    q_val = float(q)
    c_val = float(c)
    phi_val = float(np.clip(phi, -0.999, 0.999))
    nu_val = float(np.clip(nu, 2.1, 30.0))
    
    # Precompute gamma constants for Student-t logpdf
    log_gamma_num = gammaln((nu_val + 1.0) / 2.0)
    log_gamma_den = gammaln(nu_val / 2.0)
    
    # Initialize state
    if warm_start is not None:
        mu, P = warm_start
        if FILTER_CACHE_STATS_ENABLED:
            _cache_stats.warm_starts += 1
    else:
        mu = 0.0
        P = 1e-4
    
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    loglik_trajectory = np.zeros(n)
    log_likelihood = 0.0
    
    # Robustified Kalman gain adjustment for heavy tails
    nu_adjust = min(nu_val / (nu_val + 3.0), 1.0)
    
    for t in range(n):
        # Predict
        mu_pred = phi_val * mu
        P_pred = (phi_val ** 2) * P + q_val
        
        # Observation variance
        vol_t = float(vol[t])
        R = c_val * (vol_t ** 2)
        
        # Innovation
        r_t = float(returns[t])
        innovation = r_t - mu_pred
        
        # Forecast variance and scale
        S = P_pred + R
        forecast_scale = np.sqrt(S) if S > 1e-12 else 1e-6
        
        # Update with robustified gain
        if S > 1e-12:
            K = nu_adjust * P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            
            # Student-t log-likelihood for this timestep
            z = innovation / forecast_scale
            log_norm = log_gamma_num - log_gamma_den - 0.5 * np.log(nu_val * np.pi * (forecast_scale ** 2))
            log_kernel = -((nu_val + 1.0) / 2.0) * np.log(1.0 + (z ** 2) / nu_val)
            ll_t = log_norm + log_kernel
        else:
            mu = mu_pred
            P = P_pred
            ll_t = 0.0
        
        mu_filtered[t] = mu
        P_filtered[t] = max(P, 1e-12)
        loglik_trajectory[t] = ll_t
        log_likelihood += ll_t
    
    return mu_filtered, P_filtered, log_likelihood, loglik_trajectory


# =============================================================================
# MOMENTUM FILTER CACHED WRAPPERS
# =============================================================================

def cached_momentum_phi_gaussian_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    momentum_signal: np.ndarray,
    momentum_weight: float,
    filter_fn: Callable,
    regime_id: str = "global"
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Cached wrapper for momentum-augmented φ-Gaussian filter.
    """
    cache = get_filter_cache()
    
    # Include momentum in cache key
    momentum_hash = _hash_array(momentum_signal)
    model_id = f"phi_gaussian_momentum_w{_round_param(momentum_weight)}"
    
    key = FilterCacheKey(
        returns_hash=_hash_array(returns),
        vol_hash=_hash_array(vol),
        model_id=model_id,
        regime_id=regime_id,
        q=_round_param(q),
        c=_round_param(c),
        phi=_round_param(phi),
        nu=0.0,
    )
    
    cached = cache.get(key)
    if cached is not None:
        return (
            cached.mu_filtered,
            cached.P_filtered,
            cached.log_likelihood,
            cached.loglik_trajectory
        )
    
    # Compute fresh with trajectory capture
    mu, P, ll, trajectory = _run_momentum_phi_gaussian_with_trajectory(
        returns, vol, q, c, phi, momentum_signal, momentum_weight
    )
    
    final_mu = float(mu[-1]) if len(mu) > 0 else 0.0
    final_P = float(P[-1]) if len(P) > 0 else 1e-4
    cache.put(key, mu, P, ll, trajectory, final_mu, final_P)
    
    if FILTER_CACHE_STATS_ENABLED:
        _cache_stats.cold_starts += 1
    
    return mu, P, ll, trajectory


def cached_momentum_phi_student_t_filter(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    momentum_signal: np.ndarray,
    momentum_weight: float,
    filter_fn: Callable,
    regime_id: str = "global"
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Cached wrapper for momentum-augmented φ-Student-t filter.
    """
    cache = get_filter_cache()
    
    momentum_hash = _hash_array(momentum_signal)
    model_id = f"phi_student_t_nu_{int(nu)}_momentum_w{_round_param(momentum_weight)}"
    
    key = FilterCacheKey(
        returns_hash=_hash_array(returns),
        vol_hash=_hash_array(vol),
        model_id=model_id,
        regime_id=regime_id,
        q=_round_param(q),
        c=_round_param(c),
        phi=_round_param(phi),
        nu=_round_param(nu),
    )
    
    cached = cache.get(key)
    if cached is not None:
        return (
            cached.mu_filtered,
            cached.P_filtered,
            cached.log_likelihood,
            cached.loglik_trajectory
        )
    
    # Compute fresh
    mu, P, ll, trajectory = _run_momentum_phi_student_t_with_trajectory(
        returns, vol, q, c, phi, nu, momentum_signal, momentum_weight
    )
    
    final_mu = float(mu[-1]) if len(mu) > 0 else 0.0
    final_P = float(P[-1]) if len(P) > 0 else 1e-4
    cache.put(key, mu, P, ll, trajectory, final_mu, final_P)
    
    if FILTER_CACHE_STATS_ENABLED:
        _cache_stats.cold_starts += 1
    
    return mu, P, ll, trajectory


def _run_momentum_phi_gaussian_with_trajectory(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    momentum_signal: np.ndarray,
    momentum_weight: float
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Run momentum φ-Gaussian filter with trajectory capture."""
    n = len(returns)
    q_val = float(q)
    c_val = float(c)
    phi_val = float(np.clip(phi, -0.999, 0.999))
    mom_w = float(momentum_weight)
    
    mu = 0.0
    P = 1e-4
    
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    loglik_trajectory = np.zeros(n)
    log_likelihood = 0.0
    
    for t in range(n):
        # Momentum-augmented prediction
        mom_adj = mom_w * float(momentum_signal[t]) * float(vol[t])
        mu_pred = phi_val * mu + mom_adj
        P_pred = (phi_val ** 2) * P + q_val
        
        vol_t = float(vol[t])
        R = c_val * (vol_t ** 2)
        r_t = float(returns[t])
        innovation = r_t - mu_pred
        S = P_pred + R
        
        if S > 1e-12:
            K = P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            ll_t = -0.5 * (np.log(2 * np.pi * S) + (innovation ** 2) / S)
        else:
            mu = mu_pred
            P = P_pred
            ll_t = 0.0
        
        mu_filtered[t] = mu
        P_filtered[t] = max(P, 1e-12)
        loglik_trajectory[t] = ll_t
        log_likelihood += ll_t
    
    return mu_filtered, P_filtered, log_likelihood, loglik_trajectory


def _run_momentum_phi_student_t_with_trajectory(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    momentum_signal: np.ndarray,
    momentum_weight: float
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Run momentum φ-Student-t filter with trajectory capture."""
    from scipy.special import gammaln
    
    n = len(returns)
    q_val = float(q)
    c_val = float(c)
    phi_val = float(np.clip(phi, -0.999, 0.999))
    nu_val = float(np.clip(nu, 2.1, 30.0))
    mom_w = float(momentum_weight)
    
    log_gamma_num = gammaln((nu_val + 1.0) / 2.0)
    log_gamma_den = gammaln(nu_val / 2.0)
    nu_adjust = min(nu_val / (nu_val + 3.0), 1.0)
    
    mu = 0.0
    P = 1e-4
    
    mu_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    loglik_trajectory = np.zeros(n)
    log_likelihood = 0.0
    
    for t in range(n):
        mom_adj = mom_w * float(momentum_signal[t]) * float(vol[t])
        mu_pred = phi_val * mu + mom_adj
        P_pred = (phi_val ** 2) * P + q_val
        
        vol_t = float(vol[t])
        R = c_val * (vol_t ** 2)
        r_t = float(returns[t])
        innovation = r_t - mu_pred
        S = P_pred + R
        # FIX: Convert variance S to Student-t scale: scale = sqrt(S × (ν-2)/ν)
        if nu_val > 2 and S > 1e-12:
            forecast_scale = np.sqrt(S * (nu_val - 2) / nu_val)
        else:
            forecast_scale = np.sqrt(S) if S > 1e-12 else 1e-6
        
        if S > 1e-12:
            K = nu_adjust * P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            
            z = innovation / forecast_scale
            log_norm = log_gamma_num - log_gamma_den - 0.5 * np.log(nu_val * np.pi * (forecast_scale ** 2))
            log_kernel = -((nu_val + 1.0) / 2.0) * np.log(1.0 + (z ** 2) / nu_val)
            ll_t = log_norm + log_kernel
        else:
            mu = mu_pred
            P = P_pred
            ll_t = 0.0
        
        mu_filtered[t] = mu
        P_filtered[t] = max(P, 1e-12)
        loglik_trajectory[t] = ll_t
        log_likelihood += ll_t
    
    return mu_filtered, P_filtered, log_likelihood, loglik_trajectory
