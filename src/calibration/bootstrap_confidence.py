"""
Story 6.7: Monte Carlo Confidence on Backtest Metrics.

Block bootstrap (Politis & Romano 1994) for confidence intervals on
Sharpe, hit rate, max drawdown, and Sortino. Preserves temporal
dependence via contiguous block resampling.

Key insight: iid bootstrap destroys serial correlation (momentum,
vol clustering) producing artificially tight CIs. Block bootstrap
with b = ceil(n^(1/3)) preserves this structure.

Usage:
    from calibration.bootstrap_confidence import block_bootstrap_ci
    ci = block_bootstrap_ci(daily_returns, n_resamples=1000)
"""
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_N_RESAMPLES = 1000
CI_LOWER_PCT = 2.5
CI_UPPER_PCT = 97.5


@dataclass
class ConfidenceInterval:
    """CI for a single metric."""
    median: float = 0.0
    lower: float = 0.0    # 2.5th percentile
    upper: float = 0.0    # 97.5th percentile
    significant: bool = False  # CI excludes 0


@dataclass
class BootstrapResult:
    """Full bootstrap CI result."""
    sharpe_ci: ConfidenceInterval = field(default_factory=ConfidenceInterval)
    hit_rate_ci: ConfidenceInterval = field(default_factory=ConfidenceInterval)
    max_dd_ci: ConfidenceInterval = field(default_factory=ConfidenceInterval)
    sortino_ci: ConfidenceInterval = field(default_factory=ConfidenceInterval)
    block_size: int = 0
    n_resamples: int = 0
    # iid comparison
    iid_sharpe_ci: Optional[ConfidenceInterval] = None


def block_bootstrap_ci(
    daily_returns: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    block_size: Optional[int] = None,
    include_iid_comparison: bool = True,
    rng_seed: Optional[int] = None,
) -> BootstrapResult:
    """
    Block bootstrap confidence intervals for backtest metrics.
    
    Uses stationary block bootstrap with geometric block length
    distribution (mean = b) where b = ceil(n^(1/3)).
    
    Args:
        daily_returns: Array of daily returns.
        n_resamples: Number of bootstrap samples.
        block_size: Override for block size. Default: ceil(n^(1/3)).
        include_iid_comparison: Also compute iid bootstrap for comparison.
        rng_seed: Random seed for reproducibility.
    
    Returns:
        BootstrapResult with CIs for all metrics.
    """
    n = len(daily_returns)
    if n < 10:
        return BootstrapResult()
    
    rng = np.random.default_rng(rng_seed)
    
    # Block size: b = ceil(n^(1/3))
    if block_size is None:
        block_size = int(math.ceil(n ** (1/3)))
    block_size = max(2, min(block_size, n))
    
    # Block bootstrap
    block_samples = _stationary_block_bootstrap(
        daily_returns, n_resamples, block_size, n, rng
    )
    block_metrics = _compute_bootstrap_metrics(block_samples)
    
    result = BootstrapResult(
        block_size=block_size,
        n_resamples=n_resamples,
    )
    
    result.sharpe_ci = _percentile_ci(block_metrics["sharpe"])
    result.hit_rate_ci = _percentile_ci(block_metrics["hit_rate"])
    result.max_dd_ci = _percentile_ci(block_metrics["max_dd"])
    result.sortino_ci = _percentile_ci(block_metrics["sortino"])
    
    # iid comparison
    if include_iid_comparison:
        iid_samples = _iid_bootstrap(daily_returns, n_resamples, n, rng)
        iid_metrics = _compute_bootstrap_metrics(iid_samples)
        result.iid_sharpe_ci = _percentile_ci(iid_metrics["sharpe"])
    
    return result


def _stationary_block_bootstrap(
    returns: np.ndarray,
    n_resamples: int,
    block_size: int,
    n: int,
    rng: np.random.Generator,
) -> list:
    """
    Stationary block bootstrap with geometric block length.
    
    Each block's actual length is drawn from Geometric(1/b),
    giving mean block length b but with random variation.
    """
    samples = []
    prob = 1.0 / block_size  # Geometric parameter
    
    for _ in range(n_resamples):
        sample = np.empty(n)
        idx = 0
        while idx < n:
            # Random start point
            start = rng.integers(0, n)
            # Geometric block length
            length = rng.geometric(prob)
            length = min(length, n - idx)
            
            for j in range(length):
                sample[idx] = returns[(start + j) % n]
                idx += 1
                if idx >= n:
                    break
        
        samples.append(sample)
    
    return samples


def _iid_bootstrap(
    returns: np.ndarray,
    n_resamples: int,
    n: int,
    rng: np.random.Generator,
) -> list:
    """Standard iid bootstrap (ignores temporal dependence)."""
    samples = []
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        samples.append(returns[indices])
    return samples


def _compute_bootstrap_metrics(samples: list) -> dict:
    """Compute metrics for each bootstrap sample."""
    sharpes = []
    hit_rates = []
    max_dds = []
    sortinos = []
    
    for s in samples:
        mean_r = np.mean(s)
        std_r = np.std(s, ddof=1)
        
        # Sharpe (annualized)
        sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else 0.0
        sharpes.append(sharpe)
        
        # Hit rate
        hit_rates.append(float(np.mean(s > 0)))
        
        # Max drawdown
        cum = np.cumsum(s)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        max_dds.append(float(np.min(dd)))
        
        # Sortino
        downside = s[s < 0]
        if len(downside) > 1:
            ds_std = np.std(downside, ddof=1)
            sortino = (mean_r / ds_std) * math.sqrt(252) if ds_std > 0 else 0.0
        else:
            sortino = 0.0
        sortinos.append(sortino)
    
    return {
        "sharpe": np.array(sharpes),
        "hit_rate": np.array(hit_rates),
        "max_dd": np.array(max_dds),
        "sortino": np.array(sortinos),
    }


def _percentile_ci(values: np.ndarray) -> ConfidenceInterval:
    """Compute 95% CI from percentiles."""
    if len(values) < 10:
        return ConfidenceInterval()
    
    lower = float(np.percentile(values, CI_LOWER_PCT))
    upper = float(np.percentile(values, CI_UPPER_PCT))
    median = float(np.median(values))
    
    # Significant if CI excludes 0
    significant = (lower > 0) or (upper < 0)
    
    return ConfidenceInterval(
        median=median,
        lower=lower,
        upper=upper,
        significant=significant,
    )
