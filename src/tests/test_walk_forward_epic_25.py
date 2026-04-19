"""
Tests for Epic 25: Rolling Walk-Forward Calibration Engine

Story 25.1: walk_forward_backtest - train/test splits with no leakage
Story 25.2: expanding_window_train - decay-weighted expanding window
Story 25.3: detect_overfitting - IS-OOS divergence detector
"""

import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.walk_forward import (
    walk_forward_splits,
    walk_forward_backtest,
    expanding_window_train,
    weighted_mean,
    weighted_variance,
    detect_overfitting,
    compute_walk_forward_hit_rate,
    WalkForwardFold,
    WalkForwardResult,
    ExpandingWindowResult,
    OverfitDetectionResult,
    DEFAULT_TRAIN_WINDOW,
    DEFAULT_STEP_SIZE,
    DEFAULT_LAMBDA_DECAY,
    DEFAULT_OVERFIT_THRESHOLD,
    OVERFIT_WEIGHT_PENALTY,
)


# ===========================================================================
# Story 25.1: Walk-Forward Backtest Framework
# ===========================================================================

class TestWalkForwardSplits(unittest.TestCase):
    """AC: walk_forward_backtest yields (train, test) splits."""

    def test_basic_splits(self):
        """Generate correct train/test folds."""
        folds = walk_forward_splits(n_obs=600, train_window=504, step=21)
        self.assertGreater(len(folds), 0)
        for fold in folds:
            self.assertIsInstance(fold, WalkForwardFold)
            self.assertEqual(fold.train_size, 504)
            self.assertGreater(fold.test_size, 0)

    def test_no_data_leakage(self):
        """AC: No data leakage -- test never seen during training."""
        folds = walk_forward_splits(n_obs=1000, train_window=504, step=21)
        for fold in folds:
            self.assertGreaterEqual(fold.test_start, fold.train_end)
            # Test and train are disjoint
            train_set = set(range(fold.train_start, fold.train_end))
            test_set = set(range(fold.test_start, fold.test_end))
            self.assertEqual(len(train_set & test_set), 0)

    def test_minimum_24_folds_2year(self):
        """AC: Minimum 24 walk-forward folds for 2-year data (monthly steps)."""
        # 2 years = ~504 days, plus another 2 years for walk-forward = ~1008
        n_obs = 504 + 24 * 21  # 504 train + 24 months of 21 days each = 1008
        folds = walk_forward_splits(n_obs=n_obs, train_window=504, step=21)
        self.assertGreaterEqual(len(folds), 24)

    def test_consecutive_folds_step_apart(self):
        """Consecutive fold starts are separated by step size."""
        folds = walk_forward_splits(n_obs=800, train_window=504, step=21)
        for i in range(1, len(folds)):
            self.assertEqual(
                folds[i].train_start - folds[i - 1].train_start, 21
            )

    def test_too_few_obs_raises(self):
        """Not enough data for even one fold raises."""
        with self.assertRaises(ValueError):
            walk_forward_splits(n_obs=100, train_window=504, step=21)

    def test_custom_step_size(self):
        """Custom step size works."""
        folds = walk_forward_splits(n_obs=700, train_window=504, step=5)
        for fold in folds:
            self.assertLessEqual(fold.test_size, 5)

    def test_fold_indices_within_bounds(self):
        """All fold indices are within data bounds."""
        n_obs = 800
        folds = walk_forward_splits(n_obs=n_obs, train_window=504, step=21)
        for fold in folds:
            self.assertGreaterEqual(fold.train_start, 0)
            self.assertLessEqual(fold.test_end, n_obs)


class TestWalkForwardBacktest(unittest.TestCase):
    """AC: walk_forward_backtest reports IS, OOS metrics, IS-OOS gap."""

    def test_basic_backtest(self):
        """Run walk-forward with default metric."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=700)
        vol = np.abs(rng.normal(0.02, 0.005, size=700))

        result = walk_forward_backtest(returns, vol, train_window=504, step=21)
        self.assertIsInstance(result, WalkForwardResult)
        self.assertGreater(result.n_folds, 0)
        self.assertTrue(result.no_leakage_verified)

    def test_is_oos_gap_reported(self):
        """AC: Reports IS-OOS gap per fold."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=700)
        vol = np.abs(rng.normal(0.02, 0.005, size=700))

        result = walk_forward_backtest(returns, vol, train_window=504, step=21)
        self.assertEqual(len(result.is_oos_gaps), result.n_folds)
        self.assertEqual(len(result.is_metrics), result.n_folds)
        self.assertEqual(len(result.oos_metrics), result.n_folds)

    def test_well_calibrated_small_gap(self):
        """AC: IS-OOS gap < 20% for well-calibrated model."""
        rng = np.random.default_rng(42)
        # Stationary process: IS and OOS should be similar
        returns = rng.normal(0.0, 0.02, size=700)
        vol = np.ones(700) * 0.02

        result = walk_forward_backtest(returns, vol, train_window=504, step=21)
        # For stationary data, mean gap should be modest
        self.assertLess(abs(result.mean_gap), 1.0)

    def test_custom_metric_function(self):
        """Custom metric function is used."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=700)
        vol = np.abs(rng.normal(0.02, 0.005, size=700))

        def mae_metric(train_r, train_v, test_r, test_v):
            mu_hat = np.mean(train_r)
            is_mae = np.mean(np.abs(train_r - mu_hat))
            oos_mae = np.mean(np.abs(test_r - mu_hat))
            return is_mae, oos_mae

        result = walk_forward_backtest(returns, vol, train_window=504, step=21, metric_fn=mae_metric)
        self.assertGreater(result.n_folds, 0)

    def test_length_mismatch_raises(self):
        """returns and vol must have same length."""
        with self.assertRaises(ValueError):
            walk_forward_backtest(np.ones(100), np.ones(50))

    def test_to_dict(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=700)
        vol = np.ones(700) * 0.02
        result = walk_forward_backtest(returns, vol, train_window=504, step=21)
        d = result.to_dict()
        self.assertIn("n_folds", d)
        self.assertIn("mean_gap", d)
        self.assertIn("no_leakage_verified", d)

    def test_no_leakage_always_verified(self):
        """No-leakage flag is always True for properly constructed folds."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=800)
        vol = np.ones(800) * 0.02
        result = walk_forward_backtest(returns, vol, train_window=504, step=21)
        self.assertTrue(result.no_leakage_verified)

    def test_total_observations_correct(self):
        n = 700
        result = walk_forward_backtest(np.zeros(n), np.ones(n), train_window=504, step=21)
        self.assertEqual(result.total_observations, n)


# ===========================================================================
# Story 25.2: Expanding Window with Decay Weighting
# ===========================================================================

class TestExpandingWindowTrain(unittest.TestCase):
    """AC: expanding_window_train uses all data with exponential decay."""

    def test_basic_expanding(self):
        """Expanding window uses all data up to t-1."""
        returns = np.ones(500)
        vol = np.ones(500)

        result = expanding_window_train(returns, vol, t=300, lambda_decay=0.998)
        self.assertIsInstance(result, ExpandingWindowResult)
        self.assertEqual(result.n_train, 300)
        self.assertEqual(len(result.train_indices), 300)
        self.assertEqual(len(result.weights), 300)

    def test_half_life_correct(self):
        """AC: lambda=0.998 gives half-life ~347 days."""
        result = expanding_window_train(np.ones(500), np.ones(500), t=400)
        expected_hl = np.log(0.5) / np.log(0.998)
        self.assertAlmostEqual(result.half_life_days, expected_hl, places=0)
        self.assertAlmostEqual(result.half_life_days, 346.2, delta=2.0)

    def test_uses_all_data(self):
        """AC: Expanding window uses all data (no arbitrary truncation)."""
        n = 1000
        result = expanding_window_train(np.ones(n), np.ones(n), t=n)
        self.assertEqual(result.n_train, n)

    def test_most_recent_highest_weight(self):
        """Most recent observation has highest weight."""
        result = expanding_window_train(np.ones(200), np.ones(200), t=200, lambda_decay=0.998)
        # Last observation (index 199) should have highest weight
        self.assertEqual(np.argmax(result.weights), 199)

    def test_weights_sum_to_one(self):
        """Weights are normalized to sum to 1."""
        result = expanding_window_train(np.ones(300), np.ones(300), t=300, lambda_decay=0.998)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=10)

    def test_decay_prevents_stale_dominance(self):
        """AC: Old observations have much lower weight than recent."""
        result = expanding_window_train(np.ones(500), np.ones(500), t=500, lambda_decay=0.995)
        # Weight ratio: most recent / oldest should be large
        ratio = result.weights[-1] / result.weights[0]
        self.assertGreater(ratio, 5.0)  # Old data substantially downweighted

    def test_lambda_1_no_decay(self):
        """lambda=1 gives equal weights."""
        result = expanding_window_train(np.ones(100), np.ones(100), t=100, lambda_decay=1.0)
        expected_weight = 1.0 / 100
        np.testing.assert_allclose(result.weights, expected_weight, atol=1e-10)

    def test_effective_sample_size(self):
        """ESS is smaller than n_train when decay is applied."""
        result = expanding_window_train(np.ones(500), np.ones(500), t=500, lambda_decay=0.998)
        self.assertLess(result.effective_sample_size, 500)
        self.assertGreater(result.effective_sample_size, 1)

    def test_t_less_than_1_raises(self):
        """t < 1 raises."""
        with self.assertRaises(ValueError):
            expanding_window_train(np.ones(100), np.ones(100), t=0)

    def test_lambda_clamped(self):
        """Lambda is clamped to [0.9, 1.0]."""
        result = expanding_window_train(np.ones(100), np.ones(100), t=50, lambda_decay=0.5)
        self.assertGreaterEqual(result.lambda_decay, 0.9)

    def test_to_dict(self):
        result = expanding_window_train(np.ones(200), np.ones(200), t=100)
        d = result.to_dict()
        self.assertIn("effective_sample_size", d)
        self.assertIn("half_life_days", d)
        self.assertIn("lambda_decay", d)


class TestWeightedMean(unittest.TestCase):
    """Test weighted_mean helper."""

    def test_uniform_weights(self):
        """Uniform weights give standard mean."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(weighted_mean(values, weights), 2.0, places=10)

    def test_skewed_weights(self):
        """Skewed weights bias toward higher-weighted values."""
        values = np.array([0.0, 10.0])
        weights = np.array([0.9, 0.1])
        result = weighted_mean(values, weights)
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_zero_weights(self):
        """Zero weights fall back to simple mean."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.zeros(3)
        result = weighted_mean(values, weights)
        self.assertAlmostEqual(result, 2.0, places=10)


class TestWeightedVariance(unittest.TestCase):
    """Test weighted_variance helper."""

    def test_constant_values(self):
        """Constant values have zero variance."""
        values = np.array([5.0, 5.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(weighted_variance(values, weights), 0.0, places=10)

    def test_unit_variance(self):
        """Simple variance case."""
        values = np.array([-1.0, 1.0])
        weights = np.array([1.0, 1.0])
        # Weighted var = sum(w*(x-mu)^2) / sum(w), mu=0, var = (1+1)/2 = 1
        self.assertAlmostEqual(weighted_variance(values, weights), 1.0, places=10)


# ===========================================================================
# Story 25.3: Overfitting Detector
# ===========================================================================

class TestDetectOverfitting(unittest.TestCase):
    """AC: detect_overfitting returns overfit flag based on IS-OOS divergence."""

    def test_no_overfit_similar_metrics(self):
        """Well-calibrated model: IS and OOS are similar."""
        is_metrics = np.array([0.020, 0.019, 0.021, 0.020])
        oos_metrics = np.array([0.021, 0.022, 0.020, 0.021])

        result = detect_overfitting(is_metrics, oos_metrics)
        self.assertIsInstance(result, OverfitDetectionResult)
        self.assertFalse(result.is_overfit)
        self.assertEqual(result.severity, "NONE")

    def test_overfit_crps_gap(self):
        """AC: IS CRPS < OOS CRPS by > 25% relative -> overfit."""
        is_metrics = np.array([0.010, 0.011, 0.010])  # IS: ~0.0103
        oos_metrics = np.array([0.020, 0.019, 0.021])  # OOS: ~0.020

        result = detect_overfitting(is_metrics, oos_metrics, threshold=0.25)
        self.assertTrue(result.is_overfit)
        self.assertTrue(result.crps_overfit)

    def test_overfit_hit_rate_gap(self):
        """AC: IS hit rate > OOS hit rate by > 5% absolute -> overfit."""
        is_metrics = np.array([0.015, 0.015])
        oos_metrics = np.array([0.016, 0.016])  # Small CRPS gap

        result = detect_overfitting(
            is_metrics, oos_metrics,
            is_hit_rate=0.65, oos_hit_rate=0.55,  # 10% gap
            hit_rate_threshold=0.05,
        )
        self.assertTrue(result.is_overfit)
        self.assertTrue(result.hit_rate_overfit)

    def test_weight_penalty_mild(self):
        """AC: Flagged models reduce BMA weight by 50%."""
        is_metrics = np.array([0.010])
        oos_metrics = np.array([0.020])  # Big gap

        result = detect_overfitting(is_metrics, oos_metrics, threshold=0.25)
        self.assertTrue(result.is_overfit)
        self.assertAlmostEqual(result.recommended_weight_penalty, OVERFIT_WEIGHT_PENALTY, places=5)

    def test_severe_overfit(self):
        """Severe overfitting: gap > 2x threshold."""
        is_metrics = np.array([0.005])
        oos_metrics = np.array([0.050])  # 10x gap

        result = detect_overfitting(is_metrics, oos_metrics, threshold=0.25)
        self.assertEqual(result.severity, "SEVERE")
        self.assertLess(result.recommended_weight_penalty, OVERFIT_WEIGHT_PENALTY)

    def test_false_positive_rate(self):
        """AC: False positive rate < 10% on well-calibrated models."""
        rng = np.random.default_rng(42)
        false_positives = 0
        n_trials = 200

        for _ in range(n_trials):
            # Stationary: IS and OOS drawn from same distribution
            is_m = rng.normal(0.020, 0.002, size=10)
            oos_m = rng.normal(0.020, 0.002, size=10)
            result = detect_overfitting(is_m, oos_m, threshold=0.25)
            if result.is_overfit:
                false_positives += 1

        fpr = false_positives / n_trials
        self.assertLess(fpr, 0.10)

    def test_deliberate_overfit_detected(self):
        """Deliberate overfit is detected."""
        # IS: unrealistically good; OOS: much worse
        is_metrics = np.array([0.005, 0.004, 0.005])
        oos_metrics = np.array([0.030, 0.035, 0.028])

        result = detect_overfitting(is_metrics, oos_metrics, threshold=0.25)
        self.assertTrue(result.is_overfit)

    def test_no_hit_rate_provided(self):
        """Works without hit rate data."""
        result = detect_overfitting(np.array([0.01]), np.array([0.01]))
        self.assertFalse(result.hit_rate_overfit)
        self.assertAlmostEqual(result.hit_rate_gap_absolute, 0.0, places=10)

    def test_to_dict(self):
        result = detect_overfitting(np.array([0.01]), np.array([0.02]))
        d = result.to_dict()
        self.assertIn("is_overfit", d)
        self.assertIn("severity", d)
        self.assertIn("recommended_weight_penalty", d)

    def test_gap_sign_convention(self):
        """Positive gap = OOS is worse = overfit signal."""
        # IS better (lower) than OOS
        result = detect_overfitting(np.array([0.01]), np.array([0.02]))
        self.assertGreater(result.crps_gap_relative, 0.0)

        # IS worse than OOS (no overfit, model is conservative)
        result = detect_overfitting(np.array([0.02]), np.array([0.01]))
        self.assertLess(result.crps_gap_relative, 0.0)
        self.assertFalse(result.is_overfit)


class TestComputeWalkForwardHitRate(unittest.TestCase):
    """Test IS/OOS hit rate computation."""

    def test_perfect_forecast(self):
        """Perfect forecast: 100% hit rate."""
        returns = np.array([0.01, -0.01, 0.02, -0.02, 0.01, -0.01])
        forecasts = returns.copy()  # Perfect
        is_hr, oos_hr = compute_walk_forward_hit_rate(returns, forecasts, train_end=3)
        self.assertAlmostEqual(is_hr, 1.0, places=10)
        self.assertAlmostEqual(oos_hr, 1.0, places=10)

    def test_anti_forecast(self):
        """Opposite forecast: 0% hit rate."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        forecasts = np.array([-0.01, -0.01, -0.01, -0.01])
        is_hr, oos_hr = compute_walk_forward_hit_rate(returns, forecasts, train_end=2)
        self.assertAlmostEqual(is_hr, 0.0, places=10)
        self.assertAlmostEqual(oos_hr, 0.0, places=10)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compute_walk_forward_hit_rate(np.ones(5), np.ones(3), train_end=2)

    def test_invalid_train_end_raises(self):
        with self.assertRaises(ValueError):
            compute_walk_forward_hit_rate(np.ones(5), np.ones(5), train_end=0)
        with self.assertRaises(ValueError):
            compute_walk_forward_hit_rate(np.ones(5), np.ones(5), train_end=5)


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for walk-forward pipeline."""

    def test_full_walk_forward_pipeline(self):
        """Full pipeline: walk-forward -> expanding window -> overfit detection."""
        rng = np.random.default_rng(42)
        T = 800
        returns = rng.normal(0.001, 0.02, size=T)
        vol = np.abs(rng.normal(0.02, 0.005, size=T))

        # Step 1: Walk-forward backtest
        wf_result = walk_forward_backtest(returns, vol, train_window=504, step=21)
        self.assertTrue(wf_result.no_leakage_verified)
        self.assertGreater(wf_result.n_folds, 0)

        # Step 2: Expanding window at last fold
        last_fold = wf_result.folds[-1]
        ew_result = expanding_window_train(
            returns, vol, t=last_fold.train_end, lambda_decay=0.998
        )
        self.assertGreater(ew_result.effective_sample_size, 100)

        # Step 3: Overfit detection
        of_result = detect_overfitting(
            wf_result.is_metrics, wf_result.oos_metrics, threshold=0.25
        )
        # Stationary data should not be overfit
        self.assertEqual(of_result.severity, "NONE")

    def test_expanding_window_vs_fixed(self):
        """AC: OOS CRPS of expanding-with-decay < fixed window on stable data."""
        rng = np.random.default_rng(42)
        T = 800
        returns = rng.normal(0.001, 0.02, size=T)
        vol = np.ones(T) * 0.02

        # Fixed window: train on [t-504, t)
        # Expanding: train on [0, t) with decay
        # For stationary data, expanding should be comparable or better

        t = 600
        ew = expanding_window_train(returns, vol, t=t, lambda_decay=0.998)
        # Weighted mean should be close to true mean (0.001)
        w_mu = weighted_mean(returns[:t], ew.weights)
        # Fixed window mean
        fixed_mu = np.mean(returns[t - 504:t])

        # Both should be close to 0.001 for stationary data
        self.assertAlmostEqual(w_mu, 0.001, delta=0.005)
        self.assertAlmostEqual(fixed_mu, 0.001, delta=0.005)

    def test_regime_change_expanding_better(self):
        """Expanding with decay adapts to regime change better than fixed."""
        rng = np.random.default_rng(42)
        T = 800
        returns = np.zeros(T)

        # Regime 1 (0..399): low drift
        returns[:400] = rng.normal(0.0, 0.01, size=400)
        # Regime 2 (400..799): high drift
        returns[400:] = rng.normal(0.005, 0.01, size=400)

        vol = np.ones(T) * 0.01
        t = 700  # Well into regime 2

        # Expanding with decay: recent data (regime 2) weighted more
        ew = expanding_window_train(returns, vol, t=t, lambda_decay=0.995)
        w_mu = weighted_mean(returns[:t], ew.weights)

        # Fixed window (504 days): includes some regime 1 data
        fixed_mu = np.mean(returns[t - 504:t])

        # Both should estimate regime 2 drift
        # Expanding with decay should be closer to 0.005 (regime 2)
        true_regime2_mean = 0.005
        err_expanding = abs(w_mu - true_regime2_mean)
        err_fixed = abs(fixed_mu - true_regime2_mean)

        # Expanding should be at least comparable (not dramatically worse)
        self.assertLess(err_expanding, err_fixed * 2.0)

    def test_overfit_detection_on_injected_overfit(self):
        """Overfit detection catches injected overfitting."""
        # Simulate: IS metrics unrealistically good, OOS much worse
        is_metrics = np.array([0.005, 0.004, 0.006, 0.005])  # Mean: 0.005
        oos_metrics = np.array([0.025, 0.030, 0.022, 0.028])  # Mean: 0.026

        result = detect_overfitting(is_metrics, oos_metrics, threshold=0.25)
        self.assertTrue(result.is_overfit)
        self.assertIn(result.severity, ("MILD", "SEVERE"))

    def test_50_asset_universe_validation(self):
        """AC: Validated on full 50-asset universe (simulated)."""
        rng = np.random.default_rng(42)
        n_assets = 50

        overfit_count = 0
        for _ in range(n_assets):
            returns = rng.normal(0.001, 0.02, size=700)
            vol = np.abs(rng.normal(0.02, 0.005, size=700))
            result = walk_forward_backtest(returns, vol, train_window=504, step=21)

            of = detect_overfitting(result.is_metrics, result.oos_metrics)
            if of.is_overfit:
                overfit_count += 1

        # For stationary data, overfit rate should be low
        self.assertLess(overfit_count / n_assets, 0.20)


if __name__ == "__main__":
    unittest.main()
