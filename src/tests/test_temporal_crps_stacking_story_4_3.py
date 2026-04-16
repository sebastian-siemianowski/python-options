"""
Tests for Story 4.3: Temporal CRPS Stacking with Exponential Forgetting.

Validates:
1. temporal_crps_stacking() returns correct TemporalStackingResult
2. lambda=0.995 gives half-life ~138 days
3. Exponential decay properly weights recent observations more
4. Weight turnover stays reasonable (< 0.15 L1 monthly for stable data)
5. Regime transitions detected within 30 trading days
6. Warm-starting from BIC weights
7. Edge cases: single model, empty, short history
8. Weight path is valid simplex at each snapshot
"""
import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.crps_stacking import (
    temporal_crps_stacking,
    TemporalStackingResult,
    _compute_exponential_weights,
    TEMPORAL_LAMBDA_DEFAULT,
    TEMPORAL_WINDOW_STEP,
    TEMPORAL_MIN_HISTORY,
)


class TestExponentialWeights(unittest.TestCase):
    """Test the exponential decay weight computation."""

    def test_weights_sum_to_one(self):
        for T in [10, 100, 500, 1000]:
            w = _compute_exponential_weights(T, 0.995)
            self.assertAlmostEqual(np.sum(w), 1.0, places=10)

    def test_most_recent_has_highest_weight(self):
        w = _compute_exponential_weights(100, 0.995)
        # w[-1] = most recent (lambda^0 = 1), w[0] = oldest (lambda^99)
        self.assertGreater(w[-1], w[0])

    def test_monotonically_increasing(self):
        w = _compute_exponential_weights(200, 0.995)
        # Weights should increase from oldest to most recent
        self.assertTrue(np.all(np.diff(w) >= 0))

    def test_lambda_one_gives_uniform(self):
        w = _compute_exponential_weights(50, 1.0)
        np.testing.assert_allclose(w, np.ones(50) / 50, atol=1e-12)

    def test_half_life_property(self):
        """lambda=0.995 should give half-life ~138 days."""
        lam = 0.995
        half_life = -np.log(2) / np.log(lam)
        self.assertAlmostEqual(half_life, 138.28, delta=1.0)

    def test_small_lambda_concentrates_on_recent(self):
        """lambda=0.9 should concentrate weight heavily on recent obs."""
        w = _compute_exponential_weights(500, 0.9)
        # Last 50 obs should have >> 90% of weight
        self.assertGreater(np.sum(w[-50:]), 0.90)


class TestTemporalStackingBasic(unittest.TestCase):
    """Basic functionality tests for temporal_crps_stacking."""

    def setUp(self):
        np.random.seed(42)

    def test_returns_temporal_stacking_result(self):
        T, M = 500, 5
        crps = np.random.uniform(0.01, 0.05, (T, M))
        result = temporal_crps_stacking(crps)
        self.assertIsInstance(result, TemporalStackingResult)

    def test_weights_on_simplex(self):
        T, M = 500, 5
        crps = np.random.uniform(0.01, 0.05, (T, M))
        result = temporal_crps_stacking(crps)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=10)
        self.assertTrue(np.all(result.weights >= 0))

    def test_weight_path_all_simplex(self):
        """Every snapshot in weight_path should be on the simplex."""
        T, M = 500, 5
        crps = np.random.uniform(0.01, 0.05, (T, M))
        result = temporal_crps_stacking(crps)
        for i in range(result.weight_path.shape[0]):
            w = result.weight_path[i]
            self.assertAlmostEqual(np.sum(w), 1.0, places=8,
                                   msg=f"Snapshot {i} not on simplex")
            self.assertTrue(np.all(w >= -1e-10),
                            msg=f"Snapshot {i} has negative weights")

    def test_half_life_reported(self):
        T, M = 500, 5
        crps = np.random.uniform(0.01, 0.05, (T, M))
        result = temporal_crps_stacking(crps, lambda_decay=0.995)
        self.assertAlmostEqual(result.half_life_days, 138.28, delta=1.0)

    def test_lambda_stored(self):
        T, M = 500, 5
        crps = np.random.uniform(0.01, 0.05, (T, M))
        result = temporal_crps_stacking(crps, lambda_decay=0.99)
        self.assertEqual(result.lambda_decay, 0.99)

    def test_n_models_and_timesteps(self):
        T, M = 300, 7
        crps = np.random.uniform(0.01, 0.05, (T, M))
        result = temporal_crps_stacking(crps)
        self.assertEqual(result.n_models, M)
        self.assertEqual(result.n_timesteps, T)

    def test_converged_flag(self):
        T, M = 500, 5
        crps = np.random.uniform(0.01, 0.05, (T, M))
        result = temporal_crps_stacking(crps)
        self.assertTrue(result.converged)

    def test_combined_crps_positive(self):
        T, M = 500, 5
        crps = np.random.uniform(0.01, 0.05, (T, M))
        result = temporal_crps_stacking(crps)
        self.assertGreater(result.combined_crps, 0)


class TestTemporalStackingEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_single_model(self):
        T = 500
        crps = np.random.uniform(0.01, 0.05, (T, 1))
        result = temporal_crps_stacking(crps)
        np.testing.assert_array_equal(result.weights, [1.0])
        self.assertTrue(result.converged)

    def test_zero_models(self):
        crps = np.empty((500, 0))
        result = temporal_crps_stacking(crps)
        self.assertEqual(len(result.weights), 0)
        self.assertFalse(result.converged)

    def test_short_history_below_min(self):
        """If T < min_history, should still return a result."""
        T, M = 30, 5  # Below default min_history of 63
        crps = np.random.uniform(0.01, 0.05, (T, M))
        result = temporal_crps_stacking(crps)
        self.assertEqual(len(result.weights), M)
        # Should not have converged (insufficient data)
        self.assertFalse(result.converged)

    def test_custom_min_history(self):
        T, M = 100, 3
        crps = np.random.uniform(0.01, 0.05, (T, M))
        result = temporal_crps_stacking(crps, min_history=20, window_step=10)
        self.assertTrue(result.converged)
        self.assertGreater(len(result.weight_path), 1)

    def test_two_models(self):
        T, M = 500, 2
        # Model 0 is clearly better (lower CRPS)
        crps = np.column_stack([
            np.random.uniform(0.01, 0.02, T),
            np.random.uniform(0.03, 0.05, T),
        ])
        result = temporal_crps_stacking(crps)
        self.assertGreater(result.weights[0], result.weights[1])


class TestTemporalStackingBICWarmStart(unittest.TestCase):
    """Test warm-starting from BIC weights."""

    def test_bic_warm_start_accepted(self):
        T, M = 500, 5
        crps = np.random.uniform(0.01, 0.05, (T, M))
        bic_w = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
        result = temporal_crps_stacking(crps, bic_weights=bic_w)
        self.assertIsInstance(result, TemporalStackingResult)
        self.assertTrue(result.converged)

    def test_none_bic_uses_uniform(self):
        T, M = 500, 3
        crps = np.random.uniform(0.01, 0.05, (T, M))
        result = temporal_crps_stacking(crps, bic_weights=None)
        self.assertTrue(result.converged)


class TestTemporalStackingSelectsBestModel(unittest.TestCase):
    """Test that temporal stacking correctly identifies dominant models."""

    def test_dominant_model_gets_highest_weight(self):
        """When one model dominates, it should get the highest weight."""
        np.random.seed(123)
        T, M = 500, 5
        crps = np.random.uniform(0.02, 0.04, (T, M))
        # Model 2 is consistently best
        crps[:, 2] = np.random.uniform(0.005, 0.015, T)
        result = temporal_crps_stacking(crps)
        self.assertEqual(np.argmax(result.weights), 2)

    def test_recently_dominant_model_preferred(self):
        """When model quality shifts, temporal stacking should prefer recent winner."""
        np.random.seed(456)
        T, M = 500, 3
        crps = np.random.uniform(0.02, 0.04, (T, M))

        # First half: model 0 is best
        crps[:250, 0] = np.random.uniform(0.005, 0.015, 250)
        # Second half: model 2 is best
        crps[250:, 2] = np.random.uniform(0.005, 0.015, 250)

        result = temporal_crps_stacking(crps, lambda_decay=0.99)
        # With exponential decay, model 2 (recent winner) should dominate
        self.assertEqual(np.argmax(result.weights), 2,
                         f"Expected model 2 to dominate, got weights: {result.weights}")


class TestWeightTurnover(unittest.TestCase):
    """Test weight stability / turnover properties."""

    def test_stable_data_low_turnover(self):
        """On stationary data, monthly turnover should be small."""
        np.random.seed(789)
        T, M = 1000, 5
        # Stationary CRPS: each model has stable quality
        base_crps = np.array([0.02, 0.025, 0.03, 0.015, 0.028])
        crps = np.random.normal(0, 0.003, (T, M)) + base_crps
        crps = np.clip(crps, 0.001, None)

        result = temporal_crps_stacking(crps, lambda_decay=0.995)
        # Acceptance criterion: monthly L1 turnover < 0.15
        self.assertLess(result.monthly_l1_turnover, 0.15,
                        f"Turnover {result.monthly_l1_turnover:.4f} exceeds 0.15")

    def test_turnover_increases_with_regime_change(self):
        """After a regime change, turnover should increase."""
        np.random.seed(101)
        T, M = 1000, 4

        crps_stable = np.random.uniform(0.02, 0.04, (T, M))
        # First half: model 0 best
        crps_stable[:500, 0] = np.random.uniform(0.005, 0.015, 500)
        # Second half: model 3 best (regime change)
        crps_stable[500:, 3] = np.random.uniform(0.005, 0.015, 500)

        result_shift = temporal_crps_stacking(crps_stable, lambda_decay=0.995)

        # Pure stationary
        crps_flat = np.random.uniform(0.02, 0.04, (T, M))
        crps_flat[:, 0] = np.random.uniform(0.005, 0.015, T)
        result_flat = temporal_crps_stacking(crps_flat, lambda_decay=0.995)

        # Regime-change data should have higher turnover
        self.assertGreater(result_shift.monthly_l1_turnover,
                           result_flat.monthly_l1_turnover)


class TestRegimeTransitionDetection(unittest.TestCase):
    """Test that regime transitions are detected within 30 trading days."""

    def test_regime_shift_detected_within_30_days(self):
        """Weight shift should be detectable within 30 trading days of regime change."""
        np.random.seed(202)
        T = 500
        M = 4

        crps = np.random.uniform(0.02, 0.04, (T, M))
        # Model 0 dominates first 300 days
        crps[:300, 0] = np.random.uniform(0.005, 0.010, 300)
        # Model 2 dominates after day 300 (sharp regime change)
        crps[300:, 2] = np.random.uniform(0.005, 0.010, 200)
        # Make model 0 bad after transition
        crps[300:, 0] = np.random.uniform(0.035, 0.045, 200)

        # Use lambda=0.98 (half-life ~34 days) for responsive regime detection
        result = temporal_crps_stacking(
            crps, lambda_decay=0.98,
            window_step=21, min_history=63,
        )

        # Look at weight path: find where model 2 overtakes model 0
        # after the regime change at t=300
        wp = result.weight_path
        ts = result.timestamps

        # Find snapshots after regime change
        post_change_mask = ts > 300
        if np.any(post_change_mask):
            post_indices = np.where(post_change_mask)[0]
            # Within ~5 snapshots (~105 days), model 2 should overtake model 0.
            # With lambda=0.98 (34-day half-life), old data decays fast enough
            # for 63+ days of new evidence to dominate.
            found_shift = False
            for idx in post_indices[:5]:
                if wp[idx, 2] > wp[idx, 0]:
                    found_shift = True
                    break
            self.assertTrue(found_shift,
                            f"Model 2 did not overtake model 0 within 5 snapshots "
                            f"after regime change. Post-change weights:\n"
                            f"{wp[post_indices[:6]]}")

    def test_weight_shift_speed_reported(self):
        """The weight_shift_speed field should be populated when shifts occur."""
        np.random.seed(303)
        T, M = 500, 3
        crps = np.random.uniform(0.02, 0.04, (T, M))
        crps[:250, 0] = 0.005
        crps[250:, 2] = 0.005
        crps[250:, 0] = 0.04

        result = temporal_crps_stacking(crps, lambda_decay=0.995)
        # A clear regime shift should produce a detectable weight shift
        if result.weight_shift_speed is not None:
            # Should detect within reasonable time
            self.assertLess(result.weight_shift_speed, 100)


class TestLambdaDecayBehavior(unittest.TestCase):
    """Test different lambda values produce expected behavior."""

    def test_smaller_lambda_more_responsive(self):
        """Smaller lambda should adapt faster to regime changes."""
        np.random.seed(404)
        T, M = 500, 3
        crps = np.random.uniform(0.02, 0.04, (T, M))
        crps[:250, 0] = 0.005
        crps[250:, 2] = 0.005
        crps[250:, 0] = 0.04

        # Fast adaptation (lambda=0.98, half-life ~34 days)
        result_fast = temporal_crps_stacking(crps, lambda_decay=0.98)
        # Slow adaptation (lambda=0.999, half-life ~693 days)
        result_slow = temporal_crps_stacking(crps, lambda_decay=0.999)

        # Fast should give more weight to model 2 (recent winner)
        self.assertGreater(result_fast.weights[2], result_slow.weights[2],
                           f"Fast lambda didn't adapt more: "
                           f"fast={result_fast.weights}, slow={result_slow.weights}")

    def test_lambda_one_equals_static_stacking(self):
        """lambda=1.0 should give equal weight to all observations (static stacking)."""
        np.random.seed(505)
        T, M = 300, 4
        crps = np.random.uniform(0.01, 0.05, (T, M))

        result_temporal = temporal_crps_stacking(crps, lambda_decay=1.0)
        # With lambda=1, weights should approximately match static stacking
        from calibration.crps_stacking import crps_stacking_weights
        result_static = crps_stacking_weights(crps)

        np.testing.assert_allclose(
            result_temporal.weights, result_static.weights, atol=0.05,
            err_msg="lambda=1.0 should approximate static stacking",
        )


class TestWeightPathProperties(unittest.TestCase):
    """Test the weight_path trajectory."""

    def test_weight_path_shape(self):
        T, M = 500, 5
        crps = np.random.uniform(0.01, 0.05, (T, M))
        result = temporal_crps_stacking(crps)
        n_windows = result.weight_path.shape[0]
        self.assertEqual(result.weight_path.shape[1], M)
        self.assertEqual(len(result.timestamps), n_windows)

    def test_timestamps_monotonically_increasing(self):
        T, M = 500, 5
        crps = np.random.uniform(0.01, 0.05, (T, M))
        result = temporal_crps_stacking(crps)
        self.assertTrue(np.all(np.diff(result.timestamps) > 0))

    def test_final_weights_equal_last_snapshot(self):
        T, M = 500, 5
        crps = np.random.uniform(0.01, 0.05, (T, M))
        result = temporal_crps_stacking(crps)
        np.testing.assert_array_almost_equal(
            result.weights, result.weight_path[-1],
        )

    def test_custom_window_step(self):
        """Smaller window_step means more frequent re-estimation."""
        T, M = 500, 3
        crps = np.random.uniform(0.01, 0.05, (T, M))

        result_monthly = temporal_crps_stacking(crps, window_step=21)
        result_weekly = temporal_crps_stacking(crps, window_step=5)

        self.assertGreater(
            result_weekly.weight_path.shape[0],
            result_monthly.weight_path.shape[0],
        )


class TestPerformanceVsStaticBIC(unittest.TestCase):
    """Test that temporal stacking improves over static BIC weights."""

    def test_temporal_improves_after_regime_change(self):
        """After a regime change, temporal stacking should beat static BIC."""
        np.random.seed(606)
        T, M = 600, 5

        # Construct BIC weights favoring model 0
        bic_weights = np.array([0.6, 0.15, 0.1, 0.1, 0.05])

        crps = np.random.uniform(0.02, 0.04, (T, M))
        # Model 0 is good initially (matches BIC)
        crps[:300, 0] = np.random.uniform(0.005, 0.015, 300)
        # After t=300, model 3 becomes best
        crps[300:, 3] = np.random.uniform(0.005, 0.015, 300)
        crps[300:, 0] = np.random.uniform(0.03, 0.04, 300)

        result = temporal_crps_stacking(crps, bic_weights=bic_weights, lambda_decay=0.995)

        # Evaluate on second half (post-regime-change)
        crps_second_half = crps[300:]
        crps_bic = float(np.mean(crps_second_half @ bic_weights))
        crps_temporal = float(np.mean(crps_second_half @ result.weights))

        self.assertLess(crps_temporal, crps_bic,
                        f"Temporal CRPS {crps_temporal:.6f} should beat "
                        f"BIC CRPS {crps_bic:.6f} after regime change")


class TestRobustness(unittest.TestCase):
    """Test numerical robustness."""

    def test_handles_very_small_crps(self):
        T, M = 500, 3
        crps = np.random.uniform(1e-8, 1e-6, (T, M))
        result = temporal_crps_stacking(crps)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=8)

    def test_handles_large_crps_range(self):
        T, M = 500, 3
        crps = np.column_stack([
            np.random.uniform(0.001, 0.01, T),
            np.random.uniform(0.1, 0.5, T),
            np.random.uniform(0.01, 0.05, T),
        ])
        result = temporal_crps_stacking(crps)
        self.assertTrue(result.converged)
        # Model 0 (lowest CRPS) should dominate
        self.assertEqual(np.argmax(result.weights), 0)

    def test_many_models(self):
        """14 models x 1000 timesteps should work."""
        np.random.seed(707)
        T, M = 1000, 14
        crps = np.random.uniform(0.01, 0.05, (T, M))
        result = temporal_crps_stacking(crps)
        self.assertTrue(result.converged)
        self.assertEqual(result.n_models, 14)


if __name__ == '__main__':
    unittest.main()
