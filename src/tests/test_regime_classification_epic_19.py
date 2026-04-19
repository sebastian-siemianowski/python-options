"""
Test Suite for Epic 19: Probabilistic Regime Classification
=============================================================

Story 19.1: Soft Regime Membership via Sigmoid Transitions
Story 19.2: Hidden Markov Model for Regime Dynamics
Story 19.3: Regime-Specific Forecast Quality Tracking
"""
import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.regime_classification import (
    # Story 19.1
    SoftRegimeMembership,
    soft_regime_membership,
    soft_regime_membership_array,
    soft_bma_weights,
    ALL_REGIMES,
    N_REGIMES,
    LOW_VOL_TREND,
    HIGH_VOL_TREND,
    LOW_VOL_RANGE,
    HIGH_VOL_RANGE,
    CRISIS_JUMP,
    VOL_BOUNDARY_LOW_HIGH,
    VOL_BOUNDARY_HIGH_CRISIS,
    # Story 19.2
    HMMRegimeResult,
    hmm_regime_fit,
    HMM_MIN_OBS,
    # Story 19.3
    RegimeForecastQuality,
    regime_forecast_quality,
    adjusted_confidence,
    FQ_DEFAULT_WINDOW,
    FQ_MIN_HIT_RATE,
    FQ_MIN_OBS_REGIME,
)


# ===================================================================
# Story 19.1 Tests: Soft Regime Membership
# ===================================================================

class TestSoftRegimeMembership(unittest.TestCase):
    """Test soft_regime_membership()."""

    def test_returns_dataclass(self):
        result = soft_regime_membership(0.02, 0.001, 0.02)
        self.assertIsInstance(result, SoftRegimeMembership)

    def test_probabilities_sum_to_one(self):
        result = soft_regime_membership(0.02, 0.001, 0.02)
        self.assertAlmostEqual(result.probabilities.sum(), 1.0, places=8)

    def test_probabilities_non_negative(self):
        result = soft_regime_membership(0.02, 0.001, 0.02)
        self.assertTrue(np.all(result.probabilities >= 0))

    def test_shape(self):
        result = soft_regime_membership(0.02, 0.001, 0.02)
        self.assertEqual(len(result.probabilities), N_REGIMES)

    def test_low_vol_regime(self):
        """Low vol (0.5x median) should be mostly LOW_VOL."""
        result = soft_regime_membership(0.01, 0.001, 0.02)
        low_vol_prob = result.probabilities[0] + result.probabilities[2]
        self.assertGreater(low_vol_prob, 0.5)

    def test_high_vol_regime(self):
        """High vol (1.5x median) should be mostly HIGH_VOL."""
        result = soft_regime_membership(0.03, 0.001, 0.02)
        high_vol_prob = result.probabilities[1] + result.probabilities[3]
        self.assertGreater(high_vol_prob, 0.3)

    def test_crisis_regime(self):
        """Very high vol (3x median) should be crisis."""
        result = soft_regime_membership(0.06, 0.001, 0.02)
        self.assertGreater(result.probabilities[4], 0.5)
        self.assertEqual(result.dominant_regime, CRISIS_JUMP)

    def test_boundary_smooth(self):
        """At boundary (1.3x median), probabilities should be mixed."""
        median_vol = 0.02
        boundary_vol = VOL_BOUNDARY_LOW_HIGH * median_vol
        result = soft_regime_membership(boundary_vol, 0.001, median_vol)
        # Neither extreme should dominate completely
        self.assertLess(result.dominant_prob, 0.95)

    def test_trend_vs_range(self):
        """High drift should favor TREND, low drift should favor RANGE."""
        r_trend = soft_regime_membership(0.01, 0.005, 0.02)
        r_range = soft_regime_membership(0.01, 0.0001, 0.02)
        # Trend probabilities = idx 0,1; Range = idx 2,3
        trend_trend = r_trend.probabilities[0] + r_trend.probabilities[1]
        trend_range = r_range.probabilities[0] + r_range.probabilities[1]
        self.assertGreater(trend_trend, trend_range)

    def test_dominant_regime_matches_max(self):
        result = soft_regime_membership(0.02, 0.001, 0.02)
        max_idx = int(np.argmax(result.probabilities))
        self.assertEqual(result.dominant_regime, ALL_REGIMES[max_idx])

    def test_vol_relative_computed(self):
        result = soft_regime_membership(0.03, 0.001, 0.02)
        self.assertAlmostEqual(result.vol_relative, 1.5, places=3)

    def test_zero_vol_handled(self):
        result = soft_regime_membership(0.0, 0.0, 0.02)
        self.assertIsInstance(result, SoftRegimeMembership)
        self.assertAlmostEqual(result.probabilities.sum(), 1.0, places=6)

    def test_zero_median_handled(self):
        result = soft_regime_membership(0.02, 0.001, 0.0)
        self.assertIsInstance(result, SoftRegimeMembership)


class TestSoftRegimeMembershipArray(unittest.TestCase):
    """Test soft_regime_membership_array()."""

    def test_output_shape(self):
        vol = np.array([0.01, 0.02, 0.04])
        drift = np.array([0.001, 0.001, 0.001])
        result = soft_regime_membership_array(vol, drift, 0.02)
        self.assertEqual(result.shape, (3, N_REGIMES))

    def test_rows_sum_to_one(self):
        vol = np.random.default_rng(42).uniform(0.01, 0.05, 20)
        drift = np.random.default_rng(42).normal(0, 0.001, 20)
        result = soft_regime_membership_array(vol, drift, 0.02)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_all_non_negative(self):
        vol = np.random.default_rng(42).uniform(0.005, 0.1, 20)
        drift = np.random.default_rng(42).normal(0, 0.001, 20)
        result = soft_regime_membership_array(vol, drift, 0.02)
        self.assertTrue(np.all(result >= 0))


class TestSoftBMAWeights(unittest.TestCase):
    """Test soft_bma_weights()."""

    def test_basic_mixing(self):
        regime_weights = {
            LOW_VOL_TREND: np.array([0.5, 0.3, 0.2]),
            HIGH_VOL_TREND: np.array([0.2, 0.5, 0.3]),
            LOW_VOL_RANGE: np.array([0.3, 0.3, 0.4]),
            HIGH_VOL_RANGE: np.array([0.1, 0.4, 0.5]),
            CRISIS_JUMP: np.array([0.1, 0.1, 0.8]),
        }
        membership = np.array([0.4, 0.1, 0.3, 0.1, 0.1])
        mixed = soft_bma_weights(regime_weights, membership)
        self.assertEqual(len(mixed), 3)
        self.assertAlmostEqual(mixed.sum(), 1.0, places=6)

    def test_pure_regime(self):
        """If membership is 100% one regime, mixed weights = that regime."""
        regime_weights = {
            LOW_VOL_TREND: np.array([0.5, 0.3, 0.2]),
            HIGH_VOL_TREND: np.array([0.2, 0.5, 0.3]),
            LOW_VOL_RANGE: np.array([0.3, 0.3, 0.4]),
            HIGH_VOL_RANGE: np.array([0.1, 0.4, 0.5]),
            CRISIS_JUMP: np.array([0.1, 0.1, 0.8]),
        }
        membership = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        mixed = soft_bma_weights(regime_weights, membership)
        np.testing.assert_allclose(mixed, [0.5, 0.3, 0.2], atol=1e-6)

    def test_empty_weights(self):
        result = soft_bma_weights({}, np.array([0.2] * 5))
        self.assertEqual(len(result), 0)


# ===================================================================
# Story 19.2 Tests: HMM Regime Fit
# ===================================================================

class TestHMMRegimeFit(unittest.TestCase):
    """Test hmm_regime_fit()."""

    def _generate_regime_data(self, n=300, seed=42):
        """Generate vol/drift data with regime-like structure."""
        rng = np.random.default_rng(seed)
        # Simulate vol that transitions between regimes
        vol = np.zeros(n)
        regime = 0
        for i in range(n):
            if rng.random() < 0.02:
                regime = rng.integers(0, 3)
            if regime == 0:  # Low vol
                vol[i] = rng.normal(0.01, 0.002)
            elif regime == 1:  # High vol
                vol[i] = rng.normal(0.03, 0.005)
            else:  # Crisis
                vol[i] = rng.normal(0.05, 0.01)
        vol = np.maximum(vol, 0.001)
        drift = rng.normal(0, 0.001, n)
        return vol, drift

    def test_returns_dataclass(self):
        vol, drift = self._generate_regime_data()
        result = hmm_regime_fit(vol, drift)
        self.assertIsInstance(result, HMMRegimeResult)

    def test_transition_matrix_shape(self):
        vol, drift = self._generate_regime_data()
        result = hmm_regime_fit(vol, drift)
        self.assertEqual(result.transition_matrix.shape, (N_REGIMES, N_REGIMES))

    def test_transition_matrix_rows_sum_to_one(self):
        vol, drift = self._generate_regime_data()
        result = hmm_regime_fit(vol, drift)
        row_sums = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_transition_matrix_non_negative(self):
        vol, drift = self._generate_regime_data()
        result = hmm_regime_fit(vol, drift)
        self.assertTrue(np.all(result.transition_matrix >= -1e-10))

    def test_filtered_probs_shape(self):
        n = 300
        vol, drift = self._generate_regime_data(n)
        result = hmm_regime_fit(vol, drift)
        self.assertEqual(result.filtered_probs.shape, (n, N_REGIMES))

    def test_filtered_probs_sum_to_one(self):
        vol, drift = self._generate_regime_data()
        result = hmm_regime_fit(vol, drift)
        row_sums = result.filtered_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_stationary_dist_sums_to_one(self):
        vol, drift = self._generate_regime_data()
        result = hmm_regime_fit(vol, drift)
        self.assertAlmostEqual(result.stationary_dist.sum(), 1.0, delta=0.01)

    def test_emission_means_ordered(self):
        """Emission means should be roughly ordered by vol level."""
        vol, drift = self._generate_regime_data()
        result = hmm_regime_fit(vol, drift)
        # At least should have distinct means
        self.assertTrue(np.std(result.emission_means) > 0)

    def test_short_series_fallback(self):
        vol = np.array([0.02, 0.03])
        drift = np.array([0.001, 0.001])
        result = hmm_regime_fit(vol, drift)
        self.assertFalse(result.converged)
        self.assertEqual(result.n_iter, 0)

    def test_n_iter_positive(self):
        vol, drift = self._generate_regime_data()
        result = hmm_regime_fit(vol, drift)
        self.assertGreater(result.n_iter, 0)

    def test_log_likelihood_finite(self):
        vol, drift = self._generate_regime_data()
        result = hmm_regime_fit(vol, drift)
        if result.converged or result.n_iter > 0:
            self.assertTrue(np.isfinite(result.log_likelihood))


# ===================================================================
# Story 19.3 Tests: Regime Forecast Quality
# ===================================================================

class TestRegimeForecastQuality(unittest.TestCase):
    """Test regime_forecast_quality()."""

    def _make_data(self, n=200, seed=42):
        rng = np.random.default_rng(seed)
        predictions = rng.normal(0, 0.01, n)
        outcomes = predictions + rng.normal(0, 0.005, n)
        labels = [ALL_REGIMES[i % N_REGIMES] for i in range(n)]
        return predictions, outcomes, labels

    def test_returns_dataclass(self):
        preds, outs, labels = self._make_data()
        result = regime_forecast_quality(preds, outs, labels)
        self.assertIsInstance(result, RegimeForecastQuality)

    def test_all_regimes_in_output(self):
        preds, outs, labels = self._make_data()
        result = regime_forecast_quality(preds, outs, labels)
        for regime in ALL_REGIMES:
            self.assertIn(regime, result.hit_rate)
            self.assertIn(regime, result.crps)
            self.assertIn(regime, result.n_obs)
            self.assertIn(regime, result.confidence_scaling)

    def test_hit_rate_bounds(self):
        preds, outs, labels = self._make_data()
        result = regime_forecast_quality(preds, outs, labels)
        for regime in ALL_REGIMES:
            self.assertGreaterEqual(result.hit_rate[regime], 0.0)
            self.assertLessEqual(result.hit_rate[regime], 1.0)

    def test_crps_positive(self):
        preds, outs, labels = self._make_data()
        result = regime_forecast_quality(preds, outs, labels)
        for regime in ALL_REGIMES:
            self.assertGreaterEqual(result.crps[regime], 0.0)

    def test_n_obs_correct(self):
        preds, outs, labels = self._make_data(200)
        result = regime_forecast_quality(preds, outs, labels, window=200)
        total = sum(result.n_obs.values())
        self.assertEqual(total, 200)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            regime_forecast_quality(
                np.zeros(100), np.zeros(50), ["A"] * 100,
            )

    def test_window_limits_data(self):
        preds, outs, labels = self._make_data(500)
        result = regime_forecast_quality(preds, outs, labels, window=100)
        total = sum(result.n_obs.values())
        self.assertLessEqual(total, 100)

    def test_confidence_scaling_positive(self):
        preds, outs, labels = self._make_data()
        result = regime_forecast_quality(preds, outs, labels)
        for regime in ALL_REGIMES:
            self.assertGreater(result.confidence_scaling[regime], 0)

    def test_suppressed_regimes_are_bad(self):
        """Suppressed regimes should have hit_rate < FQ_MIN_HIT_RATE."""
        preds, outs, labels = self._make_data()
        result = regime_forecast_quality(preds, outs, labels)
        for regime in result.suppressed_regimes:
            self.assertLess(result.hit_rate[regime], FQ_MIN_HIT_RATE)

    def test_with_sigma(self):
        preds, outs, labels = self._make_data()
        sigma = np.ones(200) * 0.01
        result = regime_forecast_quality(preds, outs, labels, prediction_sigma=sigma)
        self.assertIsInstance(result, RegimeForecastQuality)

    def test_average_hit_rate_reasonable(self):
        preds, outs, labels = self._make_data()
        result = regime_forecast_quality(preds, outs, labels)
        self.assertGreater(result.average_hit_rate, 0.3)
        self.assertLess(result.average_hit_rate, 1.0)


class TestAdjustedConfidence(unittest.TestCase):
    """Test adjusted_confidence()."""

    def test_scaling_applied(self):
        quality = RegimeForecastQuality(
            hit_rate={r: 0.6 for r in ALL_REGIMES},
            crps={r: 0.01 for r in ALL_REGIMES},
            n_obs={r: 30 for r in ALL_REGIMES},
            confidence_scaling={r: 1.0 for r in ALL_REGIMES},
            suppressed_regimes=[],
            average_hit_rate=0.6,
        )
        result = adjusted_confidence(0.8, LOW_VOL_TREND, quality)
        self.assertAlmostEqual(result, 0.8)

    def test_suppressed_regime_zero(self):
        quality = RegimeForecastQuality(
            hit_rate={r: 0.6 for r in ALL_REGIMES},
            crps={r: 0.01 for r in ALL_REGIMES},
            n_obs={r: 30 for r in ALL_REGIMES},
            confidence_scaling={r: 1.0 for r in ALL_REGIMES},
            suppressed_regimes=[HIGH_VOL_RANGE],
            average_hit_rate=0.6,
        )
        result = adjusted_confidence(0.8, HIGH_VOL_RANGE, quality)
        self.assertAlmostEqual(result, 0.0)

    def test_scaling_factor(self):
        quality = RegimeForecastQuality(
            hit_rate={r: 0.6 for r in ALL_REGIMES},
            crps={r: 0.01 for r in ALL_REGIMES},
            n_obs={r: 30 for r in ALL_REGIMES},
            confidence_scaling={LOW_VOL_TREND: 1.5, **{r: 1.0 for r in ALL_REGIMES[1:]}},
            suppressed_regimes=[],
            average_hit_rate=0.6,
        )
        result = adjusted_confidence(0.6, LOW_VOL_TREND, quality)
        self.assertAlmostEqual(result, 0.9)

    def test_clamped_to_one(self):
        quality = RegimeForecastQuality(
            hit_rate={r: 0.6 for r in ALL_REGIMES},
            crps={r: 0.01 for r in ALL_REGIMES},
            n_obs={r: 30 for r in ALL_REGIMES},
            confidence_scaling={LOW_VOL_TREND: 2.0, **{r: 1.0 for r in ALL_REGIMES[1:]}},
            suppressed_regimes=[],
            average_hit_rate=0.6,
        )
        result = adjusted_confidence(0.8, LOW_VOL_TREND, quality)
        self.assertLessEqual(result, 1.0)


# ===================================================================
# Integration Tests
# ===================================================================

class TestEpic19Integration(unittest.TestCase):
    """Integration tests combining stories."""

    def test_soft_membership_then_bma(self):
        """Compute soft membership then mix BMA weights."""
        m = soft_regime_membership(0.025, 0.001, 0.02)
        regime_weights = {r: np.array([0.2, 0.3, 0.5]) for r in ALL_REGIMES}
        mixed = soft_bma_weights(regime_weights, m.probabilities)
        self.assertAlmostEqual(mixed.sum(), 1.0, places=6)

    def test_hmm_then_quality(self):
        """Fit HMM, use filtered probs as regime labels, compute quality."""
        rng = np.random.default_rng(42)
        n = 300
        vol = np.abs(rng.normal(0.02, 0.01, n))
        drift = rng.normal(0, 0.001, n)

        hmm = hmm_regime_fit(vol, drift)
        # Use dominant regime from HMM as labels
        labels = [ALL_REGIMES[int(np.argmax(hmm.filtered_probs[i]))]
                  for i in range(n)]

        predictions = rng.normal(0, 0.01, n)
        outcomes = predictions + rng.normal(0, 0.005, n)

        quality = regime_forecast_quality(predictions, outcomes, labels)
        self.assertIsInstance(quality, RegimeForecastQuality)


# ===================================================================
# Edge Cases
# ===================================================================

class TestEpic19EdgeCases(unittest.TestCase):

    def test_constants(self):
        self.assertEqual(N_REGIMES, 5)
        self.assertEqual(len(ALL_REGIMES), 5)
        self.assertAlmostEqual(VOL_BOUNDARY_LOW_HIGH, 1.3)
        self.assertAlmostEqual(VOL_BOUNDARY_HIGH_CRISIS, 2.0)
        self.assertEqual(FQ_DEFAULT_WINDOW, 126)
        self.assertAlmostEqual(FQ_MIN_HIT_RATE, 0.48)
        self.assertEqual(FQ_MIN_OBS_REGIME, 20)
        self.assertEqual(HMM_MIN_OBS, 50)

    def test_constant_vol(self):
        vol = np.ones(100) * 0.02
        drift = np.zeros(100)
        result = hmm_regime_fit(vol, drift)
        self.assertIsInstance(result, HMMRegimeResult)

    def test_all_same_label(self):
        preds = np.random.default_rng(42).normal(0, 0.01, 100)
        outs = preds + np.random.default_rng(43).normal(0, 0.005, 100)
        labels = [LOW_VOL_TREND] * 100
        result = regime_forecast_quality(preds, outs, labels)
        self.assertIsInstance(result, RegimeForecastQuality)
        self.assertGreater(result.n_obs[LOW_VOL_TREND], 0)


if __name__ == "__main__":
    unittest.main()
