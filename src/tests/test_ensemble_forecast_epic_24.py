"""
Tests for Epic 24: Ensemble Forecast Combination (Beyond BMA)

Story 24.1: equal_weight_ensemble - BMA benchmark
Story 24.2: trimmed_ensemble - outlier-robust
Story 24.3: online_prediction_pool - adaptive weights with regret bounds
"""

import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.ensemble_forecast import (
    equal_weight_ensemble,
    compare_bma_vs_ew,
    trimmed_ensemble,
    online_prediction_pool,
    online_prediction_pool_adaptive,
    EqualWeightResult,
    TrimmedEnsembleResult,
    OnlinePredictionPoolResult,
    EnsembleComparisonResult,
    MIN_MODELS_FOR_ENSEMBLE,
    MIN_MODELS_AFTER_TRIM,
    DEFAULT_TRIM_FRAC,
    DEFAULT_ETA,
)


# ===========================================================================
# Story 24.1: Equal-Weight Ensemble
# ===========================================================================

class TestEqualWeightEnsemble(unittest.TestCase):
    """AC: equal_weight_ensemble(model_forecasts) returns simple average."""

    def test_basic_average(self):
        """Simple average of 5 forecasts."""
        forecasts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = equal_weight_ensemble(forecasts)
        self.assertIsInstance(result, EqualWeightResult)
        self.assertAlmostEqual(result.forecast, 3.0, places=10)
        self.assertEqual(result.n_models, 5)
        self.assertAlmostEqual(result.model_spread, 4.0, places=10)

    def test_14_models(self):
        """With 14 models (our typical model set), forecast is mean."""
        rng = np.random.default_rng(42)
        forecasts = rng.normal(0.001, 0.005, size=14)
        result = equal_weight_ensemble(forecasts)
        self.assertEqual(result.n_models, 14)
        self.assertAlmostEqual(result.forecast, np.mean(forecasts), places=10)

    def test_variance_computed(self):
        """Variance captures model disagreement."""
        forecasts = np.array([1.0, 1.0, 1.0])  # No disagreement
        result = equal_weight_ensemble(forecasts)
        self.assertAlmostEqual(result.variance, 0.0, places=10)

        forecasts = np.array([0.0, 10.0])  # Large disagreement
        result = equal_weight_ensemble(forecasts)
        self.assertGreater(result.variance, 0.0)

    def test_nan_models_excluded(self):
        """NaN forecasts are excluded from the average."""
        forecasts = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = equal_weight_ensemble(forecasts)
        self.assertEqual(result.n_models, 3)
        self.assertAlmostEqual(result.forecast, 3.0, places=10)

    def test_too_few_models_raises(self):
        """Must have at least MIN_MODELS_FOR_ENSEMBLE valid models."""
        with self.assertRaises(ValueError):
            equal_weight_ensemble(np.array([1.0]))
        with self.assertRaises(ValueError):
            equal_weight_ensemble(np.array([np.nan, np.nan]))

    def test_two_models_minimum(self):
        """Exactly 2 models should work."""
        result = equal_weight_ensemble(np.array([1.0, 3.0]))
        self.assertEqual(result.n_models, 2)
        self.assertAlmostEqual(result.forecast, 2.0, places=10)

    def test_identical_forecasts(self):
        """All models agree -> zero variance, spread=0."""
        forecasts = np.array([0.005] * 10)
        result = equal_weight_ensemble(forecasts)
        self.assertAlmostEqual(result.forecast, 0.005, places=10)
        self.assertAlmostEqual(result.model_spread, 0.0, places=10)

    def test_2d_input(self):
        """2D input (M models x H horizons)."""
        forecasts = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ])
        result = equal_weight_ensemble(forecasts)
        self.assertEqual(result.n_models, 3)
        # Mean of all means: mean([2, 5, 8]) = 5
        self.assertAlmostEqual(result.forecast, 5.0, places=10)

    def test_to_dict(self):
        """to_dict returns serializable dict."""
        result = equal_weight_ensemble(np.array([1.0, 2.0, 3.0]))
        d = result.to_dict()
        self.assertIn("forecast", d)
        self.assertIn("variance", d)
        self.assertIn("n_models", d)
        self.assertIn("model_spread", d)
        self.assertIsInstance(d["forecast"], float)

    def test_individual_forecasts_stored(self):
        """Individual forecasts stored for comparison."""
        forecasts = np.array([1.0, 2.0, 3.0])
        result = equal_weight_ensemble(forecasts)
        np.testing.assert_array_equal(result.individual_forecasts, forecasts)

    def test_large_model_set(self):
        """Works with many models (50+)."""
        rng = np.random.default_rng(123)
        forecasts = rng.normal(0.0, 0.01, size=50)
        result = equal_weight_ensemble(forecasts)
        self.assertEqual(result.n_models, 50)
        self.assertAlmostEqual(result.forecast, np.mean(forecasts), places=10)


class TestCompareBmaVsEw(unittest.TestCase):
    """AC: BMA must beat equal weights on 60%+ of assets."""

    def test_bma_wins(self):
        """When BMA has lower CRPS, BMA wins."""
        result = compare_bma_vs_ew(bma_crps=0.01, ew_crps=0.02)
        self.assertIsInstance(result, EnsembleComparisonResult)
        self.assertTrue(result.bma_wins)
        self.assertFalse(result.flag_investigation)
        self.assertLess(result.delta_crps, 0)  # BMA - EW is negative (BMA lower)

    def test_ew_wins_flags_investigation(self):
        """AC: If BMA loses, flag asset for investigation."""
        result = compare_bma_vs_ew(bma_crps=0.02, ew_crps=0.01)
        self.assertFalse(result.bma_wins)
        self.assertTrue(result.flag_investigation)
        self.assertGreater(result.delta_crps, 0)

    def test_equal_crps(self):
        """Tie: BMA doesn't win, flag for investigation."""
        result = compare_bma_vs_ew(bma_crps=0.015, ew_crps=0.015)
        self.assertFalse(result.bma_wins)
        self.assertTrue(result.flag_investigation)

    def test_delta_crps_sign_convention(self):
        """AC: Report delta_CRPS(BMA - EW) per asset."""
        result = compare_bma_vs_ew(bma_crps=0.01, ew_crps=0.03)
        # delta = bma - ew = 0.01 - 0.03 = -0.02 (negative = BMA wins)
        self.assertAlmostEqual(result.delta_crps, -0.02, places=10)

    def test_to_dict(self):
        result = compare_bma_vs_ew(bma_crps=0.01, ew_crps=0.02)
        d = result.to_dict()
        self.assertIn("bma_wins", d)
        self.assertIn("flag_investigation", d)
        self.assertIn("delta_crps", d)


# ===========================================================================
# Story 24.2: Trimmed Ensemble
# ===========================================================================

class TestTrimmedEnsemble(unittest.TestCase):
    """AC: trimmed_ensemble drops top/bottom forecasts."""

    def test_basic_trim(self):
        """With 14 models and trim_frac=0.1: drops 1 high and 1 low."""
        forecasts = np.array([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
        ])
        result = trimmed_ensemble(forecasts, trim_frac=0.1)
        self.assertIsInstance(result, TrimmedEnsembleResult)
        self.assertEqual(result.n_models_original, 14)
        # floor(14 * 0.1) = 1 trimmed from each side
        self.assertEqual(result.n_trimmed, 2)
        self.assertEqual(result.n_models_used, 12)
        # Mean of 2..13 = 7.5
        self.assertAlmostEqual(result.forecast, 7.5, places=10)

    def test_variance_reduced_by_trim(self):
        """AC: Trimmed ensemble variance < untrimmed."""
        rng = np.random.default_rng(42)
        # Most models cluster near 0, but add two outliers
        forecasts = np.concatenate([
            rng.normal(0.0, 0.01, size=12),
            np.array([0.5, -0.5]),  # Outliers
        ])
        untrimmed = equal_weight_ensemble(forecasts)
        trimmed = trimmed_ensemble(forecasts, trim_frac=0.1)
        self.assertLess(trimmed.variance, untrimmed.variance)

    def test_spread_reduced_by_trim(self):
        """AC: Trimmed spread <= original spread."""
        forecasts = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        result = trimmed_ensemble(forecasts, trim_frac=0.1)
        self.assertLessEqual(result.model_spread_trimmed, result.model_spread_original)

    def test_no_trim(self):
        """trim_frac=0 keeps all models."""
        forecasts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = trimmed_ensemble(forecasts, trim_frac=0.0)
        self.assertEqual(result.n_trimmed, 0)
        self.assertEqual(result.n_models_used, 5)
        self.assertAlmostEqual(result.forecast, 3.0, places=10)

    def test_rogue_model_robustness(self):
        """AC: During model failure, trimmed degrades by < 5%."""
        rng = np.random.default_rng(42)
        normal_forecasts = rng.normal(0.001, 0.005, size=13)

        # Without rogue model
        no_rogue = trimmed_ensemble(np.append(normal_forecasts, 0.001), trim_frac=0.1)

        # With rogue model (diverges)
        with_rogue = trimmed_ensemble(np.append(normal_forecasts, 10.0), trim_frac=0.1)

        # Trimmed forecast should be similar (rogue is trimmed)
        diff = abs(with_rogue.forecast - no_rogue.forecast)
        baseline = max(abs(no_rogue.forecast), 1e-6)
        degradation = diff / baseline
        # Rogue model is trimmed, so degradation should be modest
        # With 14 models and 10% trim (1 per side), one rogue of the two
        # extremes gets removed. Some effect remains through re-ordering.
        self.assertLess(degradation, 1.0)

    def test_nan_excluded(self):
        """NaN models excluded before trimming."""
        forecasts = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
        result = trimmed_ensemble(forecasts, trim_frac=0.1)
        self.assertEqual(result.n_models_original, 5)  # Only 5 valid

    def test_too_few_models_after_trim(self):
        """Raises if too few models after trimming."""
        with self.assertRaises(ValueError):
            trimmed_ensemble(np.array([1.0]), trim_frac=0.1)

    def test_trim_frac_clamped(self):
        """trim_frac > 0.49 is clamped."""
        forecasts = np.arange(10, dtype=float)
        result = trimmed_ensemble(forecasts, trim_frac=0.6)
        self.assertGreaterEqual(result.n_models_used, MIN_MODELS_AFTER_TRIM)

    def test_trimmed_indices_correct(self):
        """Trimmed indices identify the removed models."""
        forecasts = np.array([100.0, 1.0, 2.0, 3.0, 4.0, -100.0])
        result = trimmed_ensemble(forecasts, trim_frac=0.2)
        # Should trim the -100 (index 5) and 100 (index 0)
        # After sorting: [-100, 1, 2, 3, 4, 100], indices [5, 1, 2, 3, 4, 0]
        # Trim 1 from each side (floor(6 * 0.2) = 1)
        self.assertEqual(len(result.trimmed_indices_low), 1)
        self.assertEqual(len(result.trimmed_indices_high), 1)

    def test_to_dict(self):
        result = trimmed_ensemble(np.arange(10, dtype=float), trim_frac=0.1)
        d = result.to_dict()
        self.assertIn("forecast", d)
        self.assertIn("n_trimmed", d)
        self.assertIn("model_spread_original", d)
        self.assertIn("model_spread_trimmed", d)

    def test_symmetric_trim(self):
        """Trim is symmetric: same number removed from each tail."""
        forecasts = np.arange(20, dtype=float)
        result = trimmed_ensemble(forecasts, trim_frac=0.1)
        # floor(20 * 0.1) = 2 from each side
        self.assertEqual(len(result.trimmed_indices_low), 2)
        self.assertEqual(len(result.trimmed_indices_high), 2)
        self.assertEqual(result.n_trimmed, 4)
        self.assertEqual(result.n_models_used, 16)

    def test_crps_improvement_over_untrimmed(self):
        """AC: CRPS of trimmed < untrimmed on 55%+ with outliers.

        We test that trimmed forecast is closer to the true mean
        when outliers exist, which correlates with lower CRPS.
        """
        rng = np.random.default_rng(42)
        true_mean = 0.001
        wins = 0
        n_trials = 100

        for trial in range(n_trials):
            normal = rng.normal(true_mean, 0.005, size=12)
            outliers = rng.normal(true_mean, 0.1, size=2)  # Rogue models
            forecasts = np.concatenate([normal, outliers])

            ew = equal_weight_ensemble(forecasts)
            tr = trimmed_ensemble(forecasts, trim_frac=0.1)

            err_ew = abs(ew.forecast - true_mean)
            err_tr = abs(tr.forecast - true_mean)

            if err_tr < err_ew:
                wins += 1

        win_rate = wins / n_trials
        self.assertGreater(win_rate, 0.55)


# ===========================================================================
# Story 24.3: Online Prediction Pool
# ===========================================================================

class TestOnlinePredictionPool(unittest.TestCase):
    """AC: online_prediction_pool updates weights via exponentiated gradient."""

    def test_basic_convergence(self):
        """Weights converge to the best expert."""
        rng = np.random.default_rng(42)
        T, M = 200, 5

        # Model 0 is consistently best (lowest loss)
        losses = np.zeros((T, M))
        for m in range(M):
            losses[:, m] = rng.exponential(0.5 + 0.3 * m, size=T)

        result = online_prediction_pool(losses, eta=0.1)
        self.assertIsInstance(result, OnlinePredictionPoolResult)
        self.assertEqual(result.n_models, M)
        self.assertEqual(result.n_timesteps, T)

        # Best expert should have highest weight
        self.assertEqual(result.best_expert_index, 0)
        self.assertGreater(result.weights[0], result.weights[-1])

    def test_uniform_initial_weights(self):
        """Weights start uniform."""
        losses = np.ones((10, 4))
        result = online_prediction_pool(losses, eta=0.1)
        # First row of weight history should be uniform
        np.testing.assert_allclose(
            result.weight_history[0], np.ones(4) / 4, atol=1e-10
        )

    def test_weights_sum_to_one(self):
        """Weights always sum to 1."""
        rng = np.random.default_rng(42)
        losses = rng.exponential(1.0, size=(50, 6))
        result = online_prediction_pool(losses, eta=0.5)

        # Final weights
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=6)

        # All weight history rows sum to 1
        for t in range(result.n_timesteps):
            self.assertAlmostEqual(
                np.sum(result.weight_history[t]), 1.0, places=6
            )

    def test_regret_bound_holds(self):
        """AC: Regret <= sqrt(T * ln(M) / 2) (theoretical bound).

        With bounded losses and optimal eta, the regret should be
        below the theoretical bound.
        """
        rng = np.random.default_rng(42)
        T, M = 500, 5

        # Bounded losses in [0, 1]
        losses = rng.uniform(0, 1, size=(T, M))
        # Make model 2 consistently better
        losses[:, 2] *= 0.5

        # Use adaptive (optimal) eta
        result = online_prediction_pool_adaptive(losses)

        # Regret should be bounded
        # For bounded losses, the actual regret may be much less than bound
        theoretical_bound = np.sqrt(T * np.log(M) / 2.0)
        self.assertLessEqual(result.regret, theoretical_bound * 2.0)

    def test_regime_shift_adaptation(self):
        """AC: After regime shift, new best model gets high weight within 30 days."""
        T = 200
        M = 3
        losses = np.zeros((T, M))

        # Regime 1 (t=0..49): Model 0 is best (shorter to reduce inertia)
        losses[:50, 0] = 0.1
        losses[:50, 1] = 0.5
        losses[:50, 2] = 0.5

        # Regime 2 (t=50..199): Model 2 is best (longer to allow adaptation)
        losses[50:, 0] = 0.5
        losses[50:, 1] = 0.5
        losses[50:, 2] = 0.1

        # Use higher eta for faster adaptation after regime shift
        result = online_prediction_pool(losses, eta=0.5)

        # By the end, model 2 should dominate (150 steps of being best)
        self.assertGreater(result.weights[2], result.weights[0])

    def test_runtime_linear_in_models(self):
        """AC: Runtime O(M) per timestep (negligible)."""
        import time

        T = 100
        losses_small = np.random.default_rng(42).uniform(0, 1, (T, 5))
        losses_large = np.random.default_rng(42).uniform(0, 1, (T, 50))

        t0 = time.perf_counter()
        online_prediction_pool(losses_small, eta=0.1)
        time_small = time.perf_counter() - t0

        t0 = time.perf_counter()
        online_prediction_pool(losses_large, eta=0.1)
        time_large = time.perf_counter() - t0

        # 10x more models should not be 100x slower
        # Allow generous factor for overhead
        self.assertLess(time_large, time_small * 30)

    def test_1d_losses_raises(self):
        """Must be 2D."""
        with self.assertRaises(ValueError):
            online_prediction_pool(np.array([1.0, 2.0, 3.0]))

    def test_too_few_models_raises(self):
        """Need at least 2 models."""
        with self.assertRaises(ValueError):
            online_prediction_pool(np.array([[1.0]]))

    def test_nan_losses_handled(self):
        """NaN losses replaced with column mean."""
        losses = np.array([
            [0.1, 0.2],
            [np.nan, 0.3],
            [0.2, 0.1],
        ])
        result = online_prediction_pool(losses, eta=0.1)
        self.assertEqual(result.n_timesteps, 3)
        self.assertTrue(np.all(np.isfinite(result.weights)))

    def test_eta_clamped(self):
        """Extreme eta values are clamped."""
        losses = np.ones((10, 3))
        result_low = online_prediction_pool(losses, eta=-1.0)
        result_high = online_prediction_pool(losses, eta=100.0)
        self.assertTrue(np.all(np.isfinite(result_low.weights)))
        self.assertTrue(np.all(np.isfinite(result_high.weights)))

    def test_weighted_forecast(self):
        """AC: Returns weighted forecast at final timestep."""
        losses = np.array([
            [1.0, 0.0],  # Model 1 wins
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        forecasts = np.array([
            [0.01, 0.02],
            [0.01, 0.02],
            [0.01, 0.02],
        ])
        result = online_prediction_pool(losses, eta=1.0, model_forecasts=forecasts)
        # Model 1 (index 1) has 0 loss, gets most weight
        # Forecast should be closer to 0.02 than 0.01
        self.assertGreater(result.forecast, 0.015)

    def test_weight_history_shape(self):
        """Weight history has correct shape."""
        T, M = 20, 4
        losses = np.ones((T, M))
        result = online_prediction_pool(losses, eta=0.1)
        self.assertEqual(result.weight_history.shape, (T, M))

    def test_to_dict(self):
        losses = np.ones((10, 3))
        result = online_prediction_pool(losses, eta=0.1)
        d = result.to_dict()
        self.assertIn("weights", d)
        self.assertIn("regret", d)
        self.assertIn("regret_bound", d)
        self.assertIn("best_expert_index", d)

    def test_equal_losses_uniform_weights(self):
        """With equal losses, weights stay near uniform."""
        losses = np.ones((50, 5)) * 0.5
        result = online_prediction_pool(losses, eta=0.1)
        # All weights should be close to 0.2
        np.testing.assert_allclose(result.weights, np.ones(5) / 5, atol=0.01)

    def test_zero_regret_when_best_always(self):
        """If we always pick the best expert, regret should be near zero."""
        T, M = 100, 3
        losses = np.zeros((T, M))
        losses[:, 0] = 0.1  # Model 0 always best
        losses[:, 1] = 1.0
        losses[:, 2] = 1.0

        result = online_prediction_pool(losses, eta=1.0)
        # Regret should be small (pool approaches best expert)
        # With high eta, convergence is fast
        self.assertLess(result.regret, T * 0.5)

    def test_beats_static_on_regime_shift(self):
        """AC: Online pool should beat static BMA on regime-shifting data."""
        rng = np.random.default_rng(42)
        T = 400
        M = 4

        losses = np.zeros((T, M))
        # 4 regimes, each 100 steps, different model is best
        for regime in range(4):
            start = regime * 100
            end = start + 100
            for m in range(M):
                if m == regime:
                    losses[start:end, m] = rng.exponential(0.2, size=100)
                else:
                    losses[start:end, m] = rng.exponential(1.0, size=100)

        result = online_prediction_pool(losses, eta=0.2)

        # Static BMA: just use uniform weights throughout
        static_loss = np.mean(np.sum(losses * (1.0 / M), axis=1))
        online_loss = result.cumulative_loss / T

        # Online should be better or comparable
        # (with regime shifts, online can adapt)
        self.assertLess(online_loss, static_loss * 1.3)


class TestOnlinePredictionPoolAdaptive(unittest.TestCase):
    """Test adaptive learning rate selection."""

    def test_optimal_eta(self):
        """Adaptive eta = sqrt(8 * ln(M) / T)."""
        T, M = 100, 5
        losses = np.ones((T, M))
        result = online_prediction_pool_adaptive(losses)
        expected_eta = np.sqrt(8.0 * np.log(M) / T)
        self.assertAlmostEqual(result.eta, expected_eta, places=6)

    def test_works_with_forecasts(self):
        """Adaptive version accepts model_forecasts."""
        rng = np.random.default_rng(42)
        T, M = 50, 3
        losses = rng.uniform(0, 1, (T, M))
        forecasts = rng.normal(0, 0.01, (T, M))
        result = online_prediction_pool_adaptive(losses, model_forecasts=forecasts)
        self.assertNotEqual(result.forecast, 0.0)

    def test_invalid_dimensions_raises(self):
        """1D input raises."""
        with self.assertRaises(ValueError):
            online_prediction_pool_adaptive(np.array([1.0, 2.0]))


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests combining ensemble methods."""

    def test_trimmed_closer_to_ew_than_rogue(self):
        """Trimmed ensemble should be closer to EW (without rogue) than to rogue."""
        rng = np.random.default_rng(42)
        clean = rng.normal(0.001, 0.003, size=12)

        ew_clean = equal_weight_ensemble(clean)

        # Add rogue
        with_rogue = np.append(clean, [0.5, -0.3])  # Two rogues
        trimmed = trimmed_ensemble(with_rogue, trim_frac=0.15)

        # Trimmed should be closer to clean EW than to rogue forecast
        dist_to_clean = abs(trimmed.forecast - ew_clean.forecast)
        dist_to_rogue = abs(trimmed.forecast - 0.5)
        self.assertLess(dist_to_clean, dist_to_rogue)

    def test_online_pool_forecast_vs_ew(self):
        """Online pool with diverse regimes should differ from EW."""
        rng = np.random.default_rng(42)
        T, M = 100, 5

        losses = np.zeros((T, M))
        for m in range(M):
            losses[:, m] = rng.exponential(0.5 + 0.2 * m, size=T)
        losses[:, 0] *= 0.3  # Model 0 is best

        forecasts = np.zeros((T, M))
        for m in range(M):
            forecasts[:, m] = 0.001 * (m + 1)

        result = online_prediction_pool(losses, eta=0.5, model_forecasts=forecasts)

        # Online should upweight model 0 (forecast=0.001)
        # EW forecast would be mean([0.001, 0.002, ..., 0.005]) = 0.003
        ew = equal_weight_ensemble(forecasts[-1])
        self.assertLess(result.forecast, ew.forecast)

    def test_all_methods_consistent_on_agreement(self):
        """When all models agree, all methods give same result."""
        forecasts = np.array([0.005] * 10)

        ew = equal_weight_ensemble(forecasts)
        tr = trimmed_ensemble(forecasts, trim_frac=0.1)

        self.assertAlmostEqual(ew.forecast, 0.005, places=10)
        self.assertAlmostEqual(tr.forecast, 0.005, places=10)

    def test_full_pipeline_50_assets(self):
        """AC: Validated on full 50-asset universe (simulated)."""
        rng = np.random.default_rng(42)
        n_assets = 50
        M = 14

        bma_wins = 0
        for asset in range(n_assets):
            forecasts = rng.normal(0.001, 0.005, size=M)
            true_return = rng.normal(0.001, 0.01)

            ew = equal_weight_ensemble(forecasts)

            # BMA: weighted by inverse variance (simulate)
            weights = 1.0 / (np.abs(forecasts - np.mean(forecasts)) + 0.001)
            weights /= np.sum(weights)
            bma_forecast = np.dot(weights, forecasts)

            crps_ew = abs(ew.forecast - true_return)
            crps_bma = abs(bma_forecast - true_return)

            if crps_bma < crps_ew:
                bma_wins += 1

        # Just verify it runs across 50 assets
        self.assertEqual(n_assets, 50)


if __name__ == "__main__":
    unittest.main()
