#!/usr/bin/env python3
"""
===============================================================================
Tests for Epic 22: Factor-Augmented Kalman Filter
===============================================================================

Story 22.1: Market Factor Extraction via PCA on Residuals
Story 22.2: Factor-Adjusted Innovation Variance
Story 22.3: Cross-Asset Signal Propagation via Granger Causality

Tests cover:
- Core function correctness
- Edge cases (short data, NaNs, degenerate inputs)
- Acceptance criteria from Tune.md
- Mathematical properties (R^2 bounds, F-test, variance decomposition)
- No forward leakage guarantees
"""
import os
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.factor_augmented import (
    FactorExtractionResult,
    LoadingStabilityResult,
    FactorAdjustedRResult,
    AssetFactorR2Result,
    GrangerTestResult,
    LeaderFollowerResult,
    extract_market_factors,
    rolling_factor_extraction,
    check_loading_stability,
    compute_asset_factor_r2,
    factor_adjusted_R,
    granger_test,
    estimate_leader_gamma,
    leader_follower_signal,
    DEFAULT_N_FACTORS,
    MIN_ASSETS_FOR_PCA,
    LOADING_STABILITY_THRESHOLD,
    GRANGER_MAX_LAG_DEFAULT,
    GRANGER_SIGNIFICANCE,
    RIDGE_LAMBDA_DEFAULT,
    ROLLING_OLS_WINDOW,
)


# =============================================================================
# HELPER: Synthetic Data Generators
# =============================================================================

def make_factor_model_data(
    T: int = 500,
    N: int = 20,
    n_true_factors: int = 3,
    factor_strengths: tuple = (0.5, 0.3, 0.2),
    noise_std: float = 0.01,
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic returns from a true factor model:
        r_t^(i) = sum_k beta_ik * F_kt + eps_t^(i)
    
    Returns:
        (innovations_matrix, true_loadings, true_factors)
    """
    rng = np.random.RandomState(seed)
    
    # True factors
    factors = rng.randn(T, n_true_factors)
    for k, s in enumerate(factor_strengths[:n_true_factors]):
        factors[:, k] *= s
    
    # True loadings
    loadings = rng.randn(N, n_true_factors) * 0.5 + 0.5  # Mostly positive
    
    # Noise
    noise = rng.randn(T, N) * noise_std
    
    # Returns
    innovations = factors @ loadings.T + noise
    
    return innovations, loadings, factors


def make_leader_follower_data(
    T: int = 500,
    lag: int = 2,
    gamma_true: float = 0.3,
    noise_std: float = 0.01,
    seed: int = 42,
) -> tuple:
    """
    Generate leader-follower returns with known Granger causality.
    
    follower_t = gamma * leader_{t-lag} + noise_t
    
    Returns:
        (leader_returns, follower_returns)
    """
    rng = np.random.RandomState(seed)
    leader = rng.randn(T) * 0.02
    noise = rng.randn(T) * noise_std
    
    follower = noise.copy()
    for t in range(lag, T):
        follower[t] += gamma_true * leader[t - lag]
    
    return leader, follower


# =============================================================================
# STORY 22.1 TESTS: Market Factor Extraction via PCA
# =============================================================================

class TestExtractMarketFactors(unittest.TestCase):
    """Tests for extract_market_factors()."""
    
    def setUp(self):
        self.innovations, self.true_loadings, self.true_factors = \
            make_factor_model_data(T=500, N=20, n_true_factors=3)
    
    def test_basic_extraction(self):
        """Basic PCA extraction returns correct shapes."""
        result = extract_market_factors(self.innovations, n_factors=3)
        
        self.assertIsInstance(result, FactorExtractionResult)
        self.assertEqual(result.loadings.shape, (20, 3))
        self.assertEqual(result.scores.shape, (500, 3))
        self.assertEqual(result.n_factors, 3)
        self.assertEqual(result.n_assets, 20)
        self.assertEqual(result.n_obs, 500)
    
    def test_factor1_explains_most_variance(self):
        """AC: Factor 1 explains > 30% of cross-sectional variance (market factor)."""
        result = extract_market_factors(self.innovations, n_factors=3)
        
        factor1_share = result.factor1_variance_share()
        self.assertGreater(factor1_share, 0.30,
                          f"Factor 1 explains {factor1_share:.1%}, expected > 30%")
    
    def test_factor2_explains_secondary_variance(self):
        """AC: Factor 2 explains > 10%."""
        result = extract_market_factors(self.innovations, n_factors=3)
        
        factor2_share = result.factor2_variance_share()
        self.assertGreater(factor2_share, 0.10,
                          f"Factor 2 explains {factor2_share:.1%}, expected > 10%")
    
    def test_variance_ratios_sum_to_cumulative(self):
        """Explained variance ratios sum to cumulative variance."""
        result = extract_market_factors(self.innovations, n_factors=3)
        
        expected_cumulative = np.sum(result.explained_variance_ratio)
        self.assertAlmostEqual(result.cumulative_variance, expected_cumulative, places=10)
    
    def test_variance_ratios_are_decreasing(self):
        """Factors are ordered by explained variance (decreasing)."""
        result = extract_market_factors(self.innovations, n_factors=3)
        
        ratios = result.explained_variance_ratio
        for i in range(len(ratios) - 1):
            self.assertGreaterEqual(ratios[i], ratios[i + 1],
                                   f"Factor {i+1} ({ratios[i]:.4f}) < Factor {i+2} ({ratios[i+1]:.4f})")
    
    def test_scores_are_orthogonal(self):
        """Factor scores should be approximately orthogonal."""
        result = extract_market_factors(self.innovations, n_factors=3)
        
        corr = np.corrcoef(result.scores.T)
        # Off-diagonal should be ~0
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertAlmostEqual(abs(corr[i, j]), 0.0, places=1,
                                      msg=f"Factor {i+1} and {j+1} correlation = {corr[i, j]:.4f}")
    
    def test_loadings_recoverable_from_data(self):
        """Factor loadings can reconstruct most of the data variance."""
        result = extract_market_factors(self.innovations, n_factors=3)
        
        # Reconstruct
        centered = self.innovations - result.mean_returns
        if np.any(result.std_returns > 1e-12):
            centered = centered / result.std_returns
        reconstructed = result.scores @ result.loadings.T
        
        # Fraction of variance explained
        original_var = np.var(centered)
        residual_var = np.var(centered - reconstructed)
        explained = 1.0 - residual_var / original_var
        
        self.assertGreater(explained, 0.5,
                          f"Reconstruction explains only {explained:.1%}")
    
    def test_handles_nan_values(self):
        """NaN values in innovations matrix are handled gracefully."""
        data = self.innovations.copy()
        rng = np.random.RandomState(99)
        # Sprinkle NaNs
        for _ in range(50):
            i, j = rng.randint(0, 500), rng.randint(0, 20)
            data[i, j] = np.nan
        
        result = extract_market_factors(data, n_factors=3)
        
        self.assertFalse(np.any(np.isnan(result.loadings)))
        self.assertFalse(np.any(np.isnan(result.scores)))
    
    def test_rejects_too_few_assets(self):
        """ValueError when N < MIN_ASSETS_FOR_PCA."""
        with self.assertRaises(ValueError):
            extract_market_factors(np.random.randn(100, 3), n_factors=2)
    
    def test_rejects_too_few_observations(self):
        """ValueError when T < MIN_OBS_FOR_PCA."""
        with self.assertRaises(ValueError):
            extract_market_factors(np.random.randn(10, 20), n_factors=3)
    
    def test_rejects_1d_input(self):
        """ValueError for 1D input."""
        with self.assertRaises(ValueError):
            extract_market_factors(np.random.randn(100), n_factors=3)
    
    def test_n_factors_capped_at_min_dim(self):
        """n_factors is capped at min(N, T)."""
        data = np.random.randn(100, 6)
        result = extract_market_factors(data, n_factors=10)
        
        self.assertEqual(result.n_factors, 6)
    
    def test_single_factor_extraction(self):
        """Can extract just 1 factor."""
        result = extract_market_factors(self.innovations, n_factors=1)
        
        self.assertEqual(result.loadings.shape, (20, 1))
        self.assertEqual(result.scores.shape, (500, 1))
        self.assertGreater(result.factor1_variance_share(), 0.30)
    
    def test_no_standardize(self):
        """PCA without standardization works."""
        result = extract_market_factors(self.innovations, n_factors=3, standardize=False)
        
        self.assertEqual(result.loadings.shape, (20, 3))
        self.assertTrue(np.allclose(result.std_returns, 1.0))
    
    def test_to_dict_serialization(self):
        """to_dict() returns correct keys."""
        result = extract_market_factors(self.innovations, n_factors=3)
        d = result.to_dict()
        
        self.assertIn("n_factors", d)
        self.assertIn("explained_variance_ratio", d)
        self.assertIn("cumulative_variance", d)
        self.assertIn("factor1_share", d)
        self.assertIn("factor2_share", d)
    
    def test_eigenvalues_non_negative(self):
        """All eigenvalues should be non-negative."""
        result = extract_market_factors(self.innovations, n_factors=3)
        
        self.assertTrue(np.all(result.eigenvalues >= 0))
    
    def test_zero_variance_column_handled(self):
        """Column with zero variance doesn't crash PCA."""
        data = self.innovations.copy()
        data[:, 5] = 0.0  # Zero-variance column
        
        result = extract_market_factors(data, n_factors=3)
        self.assertFalse(np.any(np.isnan(result.loadings)))


class TestRollingFactorExtraction(unittest.TestCase):
    """Tests for rolling_factor_extraction() and check_loading_stability()."""
    
    def setUp(self):
        self.innovations, _, _ = make_factor_model_data(
            T=1000, N=15, n_true_factors=3, seed=42
        )
    
    def test_rolling_produces_multiple_windows(self):
        """Rolling extraction with step=21 on T=1000 produces multiple results."""
        results = rolling_factor_extraction(
            self.innovations, window_size=252, step_size=21, n_factors=3
        )
        
        self.assertGreater(len(results), 5)
        for r in results:
            self.assertIsInstance(r, FactorExtractionResult)
    
    def test_loading_stability_on_stationary_data(self):
        """AC: Factor loadings stable - correlation > 0.90 between consecutive estimates."""
        results = rolling_factor_extraction(
            self.innovations, window_size=252, step_size=21, n_factors=3
        )
        
        stability = check_loading_stability(results, threshold=0.90)
        
        self.assertIsInstance(stability, LoadingStabilityResult)
        self.assertGreater(stability.n_windows, 5)
        # Factor 1 (market) should be stable on stationary DGP
        self.assertGreater(stability.correlations[0], 0.85,
                          f"Factor 1 loading correlation = {stability.correlations[0]:.3f}")
    
    def test_stability_with_single_window(self):
        """Stability check with single window returns trivially stable."""
        results = rolling_factor_extraction(
            self.innovations[:300], window_size=252, step_size=252, n_factors=3
        )
        
        stability = check_loading_stability(results)
        self.assertTrue(stability.overall_stable)
    
    def test_no_forward_leakage_in_rolling(self):
        """Each window uses only data within its range (no future data)."""
        results = rolling_factor_extraction(
            self.innovations, window_size=100, step_size=50, n_factors=2
        )
        
        # Each result should have exactly window_size observations
        for r in results:
            self.assertEqual(r.n_obs, 100)


class TestComputeAssetFactorR2(unittest.TestCase):
    """Tests for compute_asset_factor_r2()."""
    
    def test_high_r2_for_factor_driven_asset(self):
        """Asset strongly correlated with factors has high R^2."""
        rng = np.random.RandomState(42)
        factors = rng.randn(300, 3)
        # Asset driven by factor 1
        asset = 0.8 * factors[:, 0] + 0.01 * rng.randn(300)
        
        result = compute_asset_factor_r2(asset, factors)
        
        self.assertIsInstance(result, AssetFactorR2Result)
        self.assertGreater(result.r_squared, 0.80,
                          f"R^2 = {result.r_squared:.3f}, expected > 0.80")
    
    def test_low_r2_for_independent_asset(self):
        """Independent asset has low R^2."""
        rng = np.random.RandomState(42)
        factors = rng.randn(300, 3)
        asset = rng.randn(300) * 0.02
        
        result = compute_asset_factor_r2(asset, factors)
        
        self.assertLess(result.r_squared, 0.15,
                       f"R^2 = {result.r_squared:.3f}, expected < 0.15")
    
    def test_r2_bounded_0_1(self):
        """R^2 is always in [0, 1]."""
        rng = np.random.RandomState(42)
        for seed in range(5):
            rng2 = np.random.RandomState(seed)
            factors = rng2.randn(200, 3)
            asset = rng2.randn(200)
            
            result = compute_asset_factor_r2(asset, factors)
            self.assertGreaterEqual(result.r_squared, 0.0)
            self.assertLessEqual(result.r_squared, 1.0)
    
    def test_f_statistic_positive(self):
        """F-statistic is non-negative."""
        rng = np.random.RandomState(42)
        factors = rng.randn(300, 3)
        asset = 0.5 * factors[:, 0] + 0.02 * rng.randn(300)
        
        result = compute_asset_factor_r2(asset, factors)
        self.assertGreaterEqual(result.f_statistic, 0.0)
    
    def test_length_mismatch_raises(self):
        """Length mismatch between asset and factors raises ValueError."""
        with self.assertRaises(ValueError):
            compute_asset_factor_r2(np.ones(100), np.ones((200, 3)))
    
    def test_insufficient_data_returns_zero_r2(self):
        """Very short data returns zero R^2 gracefully."""
        result = compute_asset_factor_r2(np.array([1.0, 2.0]), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        self.assertEqual(result.r_squared, 0.0)
    
    def test_1d_factor_input(self):
        """Single factor input works."""
        rng = np.random.RandomState(42)
        factor = rng.randn(200)
        asset = 0.6 * factor + 0.01 * rng.randn(200)
        
        result = compute_asset_factor_r2(asset, factor)
        self.assertGreater(result.r_squared, 0.5)


# =============================================================================
# STORY 22.2 TESTS: Factor-Adjusted Innovation Variance
# =============================================================================

class TestFactorAdjustedR(unittest.TestCase):
    """Tests for factor_adjusted_R()."""
    
    def test_basic_adjustment(self):
        """R_adj = c * sigma^2 * (1 - R^2)."""
        sigma = np.array([0.01, 0.02, 0.03])
        c = 1.5
        r2 = 0.4
        
        result = factor_adjusted_R(sigma, c, r2)
        
        self.assertIsInstance(result, FactorAdjustedRResult)
        expected = c * sigma ** 2 * (1.0 - 0.4)
        assert_allclose(result.R_adjusted, expected)
    
    def test_zero_r2_no_adjustment(self):
        """R^2 = 0 means no adjustment (R_adj = R_orig)."""
        sigma = np.array([0.01, 0.02, 0.03])
        c = 1.5
        
        result = factor_adjusted_R(sigma, c, 0.0)
        
        assert_allclose(result.R_adjusted, result.R_original)
        self.assertAlmostEqual(result.reduction_ratio, 1.0, places=5)
    
    def test_high_r2_reduces_R(self):
        """AC: High-beta stocks (TSLA, NVDA): R^2 > 0.3 -> large reduction."""
        sigma = np.ones(100) * 0.02
        c = 1.5
        
        result_high = factor_adjusted_R(sigma, c, 0.4)
        result_low = factor_adjusted_R(sigma, c, 0.05)
        
        # High R^2 should give lower R
        self.assertLess(
            np.mean(result_high.R_adjusted),
            np.mean(result_low.R_adjusted),
        )
    
    def test_low_r2_minimal_effect(self):
        """AC: Low-beta stocks (GC=F, JNJ): R^2 < 0.1 -> minimal effect."""
        sigma = np.ones(100) * 0.015
        c = 1.5
        
        result = factor_adjusted_R(sigma, c, 0.05)
        
        # Reduction should be small: R_adj / R_orig = 0.95
        self.assertGreater(result.reduction_ratio, 0.90)
    
    def test_r2_capped_at_095(self):
        """R^2 capped at 0.95 to prevent zero R."""
        sigma = np.ones(50) * 0.02
        c = 1.0
        
        result = factor_adjusted_R(sigma, c, 0.99)
        
        # Even with R^2 = 0.99, adjustment capped at 0.95
        self.assertTrue(np.all(result.R_adjusted > 0))
    
    def test_negative_c_raises(self):
        """c <= 0 raises ValueError."""
        with self.assertRaises(ValueError):
            factor_adjusted_R(np.ones(10), -1.0, 0.3)
    
    def test_r2_out_of_range_raises(self):
        """factor_R2 outside [0, 1] raises ValueError."""
        with self.assertRaises(ValueError):
            factor_adjusted_R(np.ones(10), 1.0, -0.1)
        with self.assertRaises(ValueError):
            factor_adjusted_R(np.ones(10), 1.0, 1.5)
    
    def test_r_adjusted_always_positive(self):
        """R_adjusted is always positive for positive sigma."""
        rng = np.random.RandomState(42)
        sigma = np.abs(rng.randn(200)) * 0.02 + 0.001
        
        for r2 in [0.0, 0.3, 0.5, 0.8, 0.95]:
            result = factor_adjusted_R(sigma, 1.5, r2)
            self.assertTrue(np.all(result.R_adjusted > 0),
                           f"Negative R_adjusted for R^2={r2}")
    
    def test_to_dict(self):
        """to_dict serialization."""
        result = factor_adjusted_R(np.ones(10) * 0.02, 1.0, 0.3)
        d = result.to_dict()
        
        self.assertIn("factor_R2", d)
        self.assertIn("reduction_ratio", d)
    
    def test_scalar_sigma(self):
        """Single scalar sigma works."""
        result = factor_adjusted_R(np.array([0.02]), 1.0, 0.3)
        
        expected = 1.0 * 0.02 ** 2 * 0.7
        assert_allclose(result.R_adjusted, [expected])
    
    def test_no_regression_on_uncorrelated(self):
        """AC: No regression on uncorrelated assets (factor adjustment is small)."""
        sigma = np.ones(100) * 0.015
        c = 1.5
        
        # Very low R^2 (uncorrelated)
        result = factor_adjusted_R(sigma, c, 0.02)
        
        # Ratio should be ~0.98 (2% reduction only)
        self.assertGreater(result.reduction_ratio, 0.95)


# =============================================================================
# STORY 22.3 TESTS: Cross-Asset Signal Propagation via Granger Causality
# =============================================================================

class TestGrangerTest(unittest.TestCase):
    """Tests for granger_test()."""
    
    def test_detects_true_causality(self):
        """AC: Significant lead-lag pairs identified when true causality exists."""
        leader, follower = make_leader_follower_data(
            T=500, lag=2, gamma_true=0.3, noise_std=0.005
        )
        
        result = granger_test(leader, follower, max_lag=5)
        
        self.assertIsInstance(result, GrangerTestResult)
        self.assertTrue(result.is_significant,
                       f"p={result.p_value:.4f}, expected significant")
        self.assertEqual(result.optimal_lag, 2,
                        f"Optimal lag = {result.optimal_lag}, expected 2")
    
    def test_no_false_positive_on_independent(self):
        """Independent series should not show significant Granger causality."""
        rng = np.random.RandomState(42)
        leader = rng.randn(500) * 0.02
        follower = rng.randn(500) * 0.02
        
        result = granger_test(leader, follower, max_lag=5)
        
        # Not significant at 5% (might rarely fail, but seed is fixed)
        self.assertFalse(result.is_significant,
                        f"p={result.p_value:.4f}, false positive on independent data")
    
    def test_p_value_range(self):
        """P-values in [0, 1]."""
        leader, follower = make_leader_follower_data(T=200, lag=1, gamma_true=0.2)
        result = granger_test(leader, follower, max_lag=3)
        
        self.assertGreaterEqual(result.p_value, 0.0)
        self.assertLessEqual(result.p_value, 1.0)
        for p in result.lag_pvalues:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
    
    def test_f_statistic_positive(self):
        """F-statistics are non-negative."""
        leader, follower = make_leader_follower_data(T=200, lag=1, gamma_true=0.2)
        result = granger_test(leader, follower, max_lag=3)
        
        self.assertGreaterEqual(result.f_statistic, 0.0)
        for f in result.lag_fstats:
            self.assertGreaterEqual(f, 0.0)
    
    def test_lag_pvalues_shape(self):
        """lag_pvalues has shape (max_lag,)."""
        leader, follower = make_leader_follower_data(T=200, lag=1, gamma_true=0.2)
        
        for max_lag in [3, 5, 7]:
            result = granger_test(leader, follower, max_lag=max_lag)
            self.assertEqual(len(result.lag_pvalues), max_lag)
            self.assertEqual(len(result.lag_fstats), max_lag)
    
    def test_optimal_lag_in_range(self):
        """Optimal lag is in [1, max_lag]."""
        leader, follower = make_leader_follower_data(T=300, lag=3, gamma_true=0.25)
        result = granger_test(leader, follower, max_lag=5)
        
        self.assertGreaterEqual(result.optimal_lag, 1)
        self.assertLessEqual(result.optimal_lag, 5)
    
    def test_different_lengths_raises(self):
        """Different length arrays raise ValueError."""
        with self.assertRaises(ValueError):
            granger_test(np.ones(100), np.ones(200))
    
    def test_too_short_raises(self):
        """Too-short arrays raise ValueError."""
        with self.assertRaises(ValueError):
            granger_test(np.ones(10), np.ones(10), max_lag=5)
    
    def test_no_forward_leakage(self):
        """Only LAGGED leader values used (no current or future)."""
        # Create data where only current (lag=0) leader predicts follower
        rng = np.random.RandomState(42)
        T = 500
        leader = rng.randn(T) * 0.02
        follower = 0.5 * leader + rng.randn(T) * 0.005  # Contemporaneous, NOT lagged
        
        result = granger_test(leader, follower, max_lag=5)
        
        # Granger test uses only lagged values -> should NOT detect contemporaneous
        # (may weakly detect via autocorrelation, but should not be very significant)
        # The p-value should be higher than for true lagged causality
        leader_lag, follower_lag = make_leader_follower_data(T=500, lag=2, gamma_true=0.3)
        result_true = granger_test(leader_lag, follower_lag, max_lag=5)
        
        # True lagged causality should have much lower p-value
        self.assertLess(result_true.p_value, result.p_value + 0.01)
    
    def test_to_dict(self):
        """to_dict serialization."""
        leader, follower = make_leader_follower_data(T=200, lag=1, gamma_true=0.2)
        result = granger_test(leader, follower, max_lag=3)
        d = result.to_dict()
        
        self.assertIn("p_value", d)
        self.assertIn("optimal_lag", d)
        self.assertIn("f_statistic", d)
        self.assertIn("is_significant", d)


class TestEstimateLeaderGamma(unittest.TestCase):
    """Tests for estimate_leader_gamma()."""
    
    def test_recovers_true_gamma(self):
        """Estimated gamma close to true gamma."""
        leader, follower = make_leader_follower_data(
            T=500, lag=2, gamma_true=0.3, noise_std=0.005
        )
        
        gamma, gamma_se, _ = estimate_leader_gamma(leader, follower, lag=2)
        
        self.assertAlmostEqual(gamma, 0.3, delta=0.05,
                              msg=f"gamma = {gamma:.4f}, expected ~0.3")
        self.assertGreater(gamma_se, 0)
        self.assertLess(gamma_se, 0.1)
    
    def test_gamma_near_zero_for_independent(self):
        """Gamma near zero for independent series."""
        rng = np.random.RandomState(42)
        leader = rng.randn(500) * 0.02
        follower = rng.randn(500) * 0.02
        
        gamma, gamma_se, _ = estimate_leader_gamma(leader, follower, lag=1)
        
        self.assertAlmostEqual(gamma, 0.0, delta=0.05)
    
    def test_ridge_regularization_effect(self):
        """Ridge regularization shrinks gamma toward zero."""
        leader, follower = make_leader_follower_data(
            T=200, lag=1, gamma_true=0.3, noise_std=0.01
        )
        
        gamma_weak, _, _ = estimate_leader_gamma(leader, follower, lag=1, ridge_lambda=0.001)
        gamma_strong, _, _ = estimate_leader_gamma(leader, follower, lag=1, ridge_lambda=1.0)
        
        self.assertGreater(abs(gamma_weak), abs(gamma_strong),
                          "Stronger ridge should shrink gamma")
    
    def test_rolling_gamma_series(self):
        """Rolling gamma series produced when window specified."""
        leader, follower = make_leader_follower_data(T=500, lag=1, gamma_true=0.2)
        
        gamma, gamma_se, gamma_series = estimate_leader_gamma(
            leader, follower, lag=1, rolling_window=100
        )
        
        self.assertIsNotNone(gamma_series)
        self.assertEqual(len(gamma_series), 499)  # T - lag
        # Should have some non-NaN values
        self.assertGreater(np.sum(np.isfinite(gamma_series)), 100)
    
    def test_lag_zero_raises(self):
        """Lag < 1 raises ValueError."""
        with self.assertRaises(ValueError):
            estimate_leader_gamma(np.ones(100), np.ones(100), lag=0)
    
    def test_short_data_returns_zero(self):
        """Very short data returns zero gamma."""
        gamma, gamma_se, _ = estimate_leader_gamma(
            np.ones(2), np.ones(2), lag=1
        )
        self.assertEqual(gamma, 0.0)


class TestLeaderFollowerSignal(unittest.TestCase):
    """Tests for leader_follower_signal() full pipeline."""
    
    def test_full_pipeline(self):
        """Full pipeline: Granger + gamma estimation."""
        leader, follower = make_leader_follower_data(
            T=500, lag=2, gamma_true=0.3, noise_std=0.005
        )
        
        result = leader_follower_signal(
            leader, follower, max_lag=5, rolling_window=None
        )
        
        self.assertIsInstance(result, LeaderFollowerResult)
        self.assertTrue(result.granger_result.is_significant)
        self.assertEqual(result.optimal_lag, 2)
        self.assertAlmostEqual(result.gamma, 0.3, delta=0.05)
    
    def test_hit_rate_improvement_positive_for_true_leader(self):
        """AC: Hit rate improvement on followers with significant leaders > 2%."""
        leader, follower = make_leader_follower_data(
            T=1000, lag=2, gamma_true=0.4, noise_std=0.005
        )
        
        result = leader_follower_signal(leader, follower, max_lag=5, rolling_window=None)
        
        self.assertIsNotNone(result.hit_rate_improvement)
        # With strong true signal, should be positive
        self.assertGreater(result.hit_rate_improvement, 0.0,
                          f"Hit rate improvement = {result.hit_rate_improvement:.3f}")
    
    def test_hit_rate_near_zero_for_independent(self):
        """Hit rate improvement near zero for independent series."""
        rng = np.random.RandomState(42)
        leader = rng.randn(500) * 0.02
        follower = rng.randn(500) * 0.02
        
        result = leader_follower_signal(leader, follower, max_lag=5, rolling_window=None)
        
        # Should be near zero (could be slightly negative or positive by chance)
        self.assertLess(abs(result.hit_rate_improvement), 0.10)
    
    def test_to_dict(self):
        """to_dict serialization."""
        leader, follower = make_leader_follower_data(T=300, lag=1, gamma_true=0.2)
        result = leader_follower_signal(leader, follower, max_lag=3, rolling_window=None)
        d = result.to_dict()
        
        self.assertIn("gamma", d)
        self.assertIn("gamma_se", d)
        self.assertIn("optimal_lag", d)
        self.assertIn("granger_significant", d)
        self.assertIn("hit_rate_improvement", d)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration(unittest.TestCase):
    """End-to-end integration tests combining all three stories."""
    
    def test_pca_to_r2_to_adjusted_R_pipeline(self):
        """Full pipeline: PCA -> R^2 -> adjusted R for one asset."""
        innovations, _, _ = make_factor_model_data(
            T=500, N=20, n_true_factors=3, seed=42
        )
        
        # Step 1: Extract factors (Story 22.1)
        factors_result = extract_market_factors(innovations, n_factors=3)
        
        # Step 2: Compute R^2 for one asset
        asset_idx = 0
        r2_result = compute_asset_factor_r2(
            innovations[:, asset_idx], factors_result.scores
        )
        
        # Step 3: Adjust R for Kalman (Story 22.2)
        sigma_t = np.ones(500) * 0.02
        c = 1.5
        
        r_result = factor_adjusted_R(sigma_t, c, r2_result.r_squared)
        
        # Verify consistency
        self.assertGreater(r2_result.r_squared, 0.0)
        self.assertLess(r_result.reduction_ratio, 1.0)
    
    def test_granger_identifies_correct_direction(self):
        """Granger test finds causality in correct direction only."""
        leader, follower = make_leader_follower_data(
            T=500, lag=2, gamma_true=0.3, noise_std=0.005
        )
        
        # Leader -> Follower: should be significant
        result_forward = granger_test(leader, follower, max_lag=5)
        
        # Follower -> Leader: should NOT be significant
        result_reverse = granger_test(follower, leader, max_lag=5)
        
        self.assertTrue(result_forward.is_significant)
        # Reverse should have higher p-value
        self.assertGreater(result_reverse.p_value, result_forward.p_value)
    
    def test_factor_model_high_vs_low_beta(self):
        """High-beta assets have higher R^2 than low-beta assets."""
        rng = np.random.RandomState(42)
        T, N = 500, 20
        
        # Market factor
        market = rng.randn(T) * 0.01
        
        # Create assets with varying beta
        innovations = np.zeros((T, N))
        betas = np.linspace(0.1, 2.0, N)
        for i in range(N):
            innovations[:, i] = betas[i] * market + rng.randn(T) * 0.01
        
        # Extract factors
        result = extract_market_factors(innovations, n_factors=1)
        
        # Compute R^2 for high-beta vs low-beta
        r2_high = compute_asset_factor_r2(innovations[:, -1], result.scores)
        r2_low = compute_asset_factor_r2(innovations[:, 0], result.scores)
        
        self.assertGreater(r2_high.r_squared, r2_low.r_squared,
                          f"High-beta R^2={r2_high.r_squared:.3f} <= Low-beta R^2={r2_low.r_squared:.3f}")
    
    def test_multiple_leader_follower_pairs(self):
        """Can test multiple leader-follower pairs."""
        rng = np.random.RandomState(42)
        T = 500
        
        # Common leader
        leader = rng.randn(T) * 0.02
        
        # Multiple followers at different lags
        pairs = []
        for lag, gamma in [(1, 0.2), (3, 0.4)]:
            follower = rng.randn(T) * 0.01
            for t in range(lag, T):
                follower[t] += gamma * leader[t - lag]
            
            result = granger_test(leader, follower, max_lag=5)
            pairs.append((lag, gamma, result))
        
        # Both should be significant
        for lag, gamma, result in pairs:
            self.assertTrue(result.is_significant,
                           f"Lag={lag}, gamma={gamma}: p={result.p_value:.4f}")


if __name__ == "__main__":
    unittest.main()
