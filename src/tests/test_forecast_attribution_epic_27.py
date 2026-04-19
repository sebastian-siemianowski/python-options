"""
Tests for Epic 27: Forecast Performance Attribution

Story 27.1: drift_attribution - direction and magnitude error decomposition
Story 27.2: volatility_attribution - coverage analysis
Story 27.3: bma_attribution - leave-one-model-out CRPS
"""

import os
import sys
import unittest

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.forecast_attribution import (
    drift_attribution,
    volatility_attribution,
    bma_attribution,
    DriftAttributionResult,
    VolatilityAttributionResult,
    BMAAttributionResult,
    DIRECTION_FAILURE_THRESHOLD,
    MAGNITUDE_FAILURE_MULTIPLIER,
    COVERAGE_UNDER_THRESHOLD,
    COVERAGE_OVER_THRESHOLD,
    CRPS_IMPROVEMENT_THRESHOLD,
    MIN_OBSERVATIONS,
)


# ===========================================================================
# Story 27.1: Drift Attribution
# ===========================================================================

class TestDriftAttribution(unittest.TestCase):
    """AC: drift_attribution returns (direction_error, magnitude_error)."""

    def test_basic_result(self):
        """Returns DriftAttributionResult with expected fields."""
        rng = np.random.default_rng(42)
        n = 100
        r = rng.normal(0.001, 0.02, size=n)
        mu = rng.normal(0.001, 0.01, size=n)
        sigma = np.ones(n) * 0.02

        result = drift_attribution(r, mu, sigma)
        self.assertIsInstance(result, DriftAttributionResult)
        self.assertEqual(result.n_observations, n)

    def test_direction_error_fraction(self):
        """Direction error is fraction of wrong-sign predictions."""
        rng = np.random.default_rng(42)
        n = 200
        r = rng.normal(0.001, 0.02, size=n)
        # Perfect direction but noisy magnitude
        mu = np.sign(r) * 0.001 + rng.normal(0, 0.0001, size=n)
        sigma = np.ones(n) * 0.02

        result = drift_attribution(r, mu, sigma)
        # Most predictions should have correct sign (near 0 error)
        self.assertLess(result.direction_error, 0.15)

    def test_directional_failure_flag(self):
        """AC: Direction error > 45% flags DIRECTIONAL_FAILURE."""
        rng = np.random.default_rng(42)
        n = 100
        r = rng.normal(0, 0.02, size=n)
        # Opposite-direction predictions most of the time
        mu = -r + rng.normal(0, 0.001, size=n)
        sigma = np.ones(n) * 0.02

        result = drift_attribution(r, mu, sigma)
        self.assertGreater(result.direction_error, DIRECTION_FAILURE_THRESHOLD)
        self.assertEqual(result.direction_flag, "DIRECTIONAL_FAILURE")

    def test_no_directional_failure_good_model(self):
        """Good model has no directional failure flag."""
        rng = np.random.default_rng(42)
        n = 200
        # Model forecasts sign correctly ~70% of time
        r = rng.normal(0.005, 0.02, size=n)
        mu = r * 0.5 + rng.normal(0, 0.01, size=n)
        sigma = np.ones(n) * 0.02

        result = drift_attribution(r, mu, sigma)
        self.assertLess(result.direction_error, DIRECTION_FAILURE_THRESHOLD)
        self.assertEqual(result.direction_flag, "")

    def test_magnitude_error_defined(self):
        """Magnitude error is MAE when direction is correct."""
        rng = np.random.default_rng(42)
        n = 200
        r = rng.normal(0.005, 0.02, size=n)
        mu = r * 0.3  # Correct direction but under-predicts magnitude
        sigma = np.ones(n) * 0.02

        result = drift_attribution(r, mu, sigma)
        self.assertGreater(result.magnitude_error, 0)
        self.assertGreater(result.correct_direction_count, 0)

    def test_zero_returns_excluded(self):
        """Zero returns are excluded from direction analysis."""
        rng = np.random.default_rng(42)
        n = 50
        r = rng.normal(0.005, 0.02, size=n)
        r[:5] = 0.0  # First 5 are zero
        mu = r * 0.5
        sigma = np.ones(n) * 0.02

        result = drift_attribution(r, mu, sigma)
        self.assertEqual(result.zero_return_count, 5)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            drift_attribution(np.ones(100), np.ones(50), np.ones(100))

    def test_too_few_observations_raises(self):
        with self.assertRaises(ValueError):
            drift_attribution(np.ones(5), np.ones(5), np.ones(5))

    def test_nan_filtered(self):
        """NaN values are filtered."""
        rng = np.random.default_rng(42)
        n = 50
        r = rng.normal(0, 0.02, size=n)
        mu = rng.normal(0, 0.01, size=n)
        sigma = np.ones(n) * 0.02
        r[0] = np.nan
        mu[1] = np.nan

        result = drift_attribution(r, mu, sigma)
        self.assertLess(result.n_observations, n)

    def test_to_dict(self):
        rng = np.random.default_rng(42)
        n = 50
        result = drift_attribution(
            rng.normal(0, 0.02, size=n),
            rng.normal(0, 0.01, size=n),
            np.ones(n) * 0.02,
        )
        d = result.to_dict()
        self.assertIn("direction_error", d)
        self.assertIn("magnitude_error", d)
        self.assertIn("direction_flag", d)
        self.assertIn("magnitude_flag", d)


# ===========================================================================
# Story 27.2: Volatility Attribution
# ===========================================================================

class TestVolatilityAttribution(unittest.TestCase):
    """AC: volatility_attribution returns coverage metrics."""

    def test_basic_result(self):
        """Returns VolatilityAttributionResult."""
        rng = np.random.default_rng(42)
        n = 200
        sigma_true = 0.02
        r = rng.normal(0, sigma_true, size=n)
        mu = np.zeros(n)
        sigma = np.ones(n) * sigma_true

        result = volatility_attribution(r, mu, sigma)
        self.assertIsInstance(result, VolatilityAttributionResult)
        self.assertEqual(result.n_observations, n)

    def test_well_calibrated_coverage(self):
        """Well-calibrated model has ~90% coverage at alpha=0.10."""
        rng = np.random.default_rng(42)
        n = 2000
        sigma_true = 0.02
        r = rng.normal(0, sigma_true, size=n)
        mu = np.zeros(n)
        sigma = np.ones(n) * sigma_true

        result = volatility_attribution(r, mu, sigma, alpha=0.10)
        # Coverage should be near 90%
        self.assertGreater(result.coverage, 0.85)
        self.assertLess(result.coverage, 0.95)
        self.assertEqual(result.volatility_flag, "")

    def test_vol_underestimate_flag(self):
        """AC: Coverage < 85% at 90% PI -> VOL_UNDERESTIMATE."""
        rng = np.random.default_rng(42)
        n = 500
        true_vol = 0.04
        pred_vol = 0.01  # Way too small
        r = rng.normal(0, true_vol, size=n)
        mu = np.zeros(n)
        sigma = np.ones(n) * pred_vol

        result = volatility_attribution(r, mu, sigma, alpha=0.10)
        self.assertLess(result.coverage, COVERAGE_UNDER_THRESHOLD)
        self.assertEqual(result.volatility_flag, "VOL_UNDERESTIMATE")

    def test_vol_overestimate_flag(self):
        """AC: Coverage > 95% at 90% PI -> VOL_OVERESTIMATE."""
        rng = np.random.default_rng(42)
        n = 500
        true_vol = 0.005
        pred_vol = 0.05  # Way too large
        r = rng.normal(0, true_vol, size=n)
        mu = np.zeros(n)
        sigma = np.ones(n) * pred_vol

        result = volatility_attribution(r, mu, sigma, alpha=0.10)
        self.assertGreater(result.coverage, COVERAGE_OVER_THRESHOLD)
        self.assertEqual(result.volatility_flag, "VOL_OVERESTIMATE")

    def test_crps_decomposition(self):
        """CRPS has reliability and sharpness components."""
        rng = np.random.default_rng(42)
        n = 200
        r = rng.normal(0, 0.02, size=n)
        mu = np.zeros(n)
        sigma = np.ones(n) * 0.02

        result = volatility_attribution(r, mu, sigma)
        self.assertGreaterEqual(result.crps_reliability, 0.0)
        self.assertGreater(result.crps_sharpness, 0.0)

    def test_sharpness_vs_reliability(self):
        """Overconfident model: low sharpness but high reliability error."""
        rng = np.random.default_rng(42)
        n = 500
        r = rng.normal(0, 0.03, size=n)
        mu = np.zeros(n)

        # Narrow (sharp) but miscalibrated
        sigma_narrow = np.ones(n) * 0.005
        result_narrow = volatility_attribution(r, mu, sigma_narrow)

        # Wide (not sharp) but well-calibrated
        sigma_wide = np.ones(n) * 0.03
        result_wide = volatility_attribution(r, mu, sigma_wide)

        # Narrow should be sharper (smaller sharpness value)
        self.assertLess(result_narrow.crps_sharpness, result_wide.crps_sharpness)

    def test_rolling_coverage(self):
        """AC: Rolling coverage tracked over 60-day windows."""
        rng = np.random.default_rng(42)
        n = 200
        r = rng.normal(0, 0.02, size=n)
        mu = np.zeros(n)
        sigma = np.ones(n) * 0.02

        result = volatility_attribution(r, mu, sigma, rolling_window=60)
        self.assertGreater(len(result.rolling_coverage), 0)
        self.assertEqual(len(result.rolling_coverage), n - 60 + 1)

    def test_rolling_coverage_short_series(self):
        """Short series returns single coverage value."""
        rng = np.random.default_rng(42)
        n = 30
        r = rng.normal(0, 0.02, size=n)
        mu = np.zeros(n)
        sigma = np.ones(n) * 0.02

        result = volatility_attribution(r, mu, sigma, rolling_window=60)
        self.assertEqual(len(result.rolling_coverage), 1)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            volatility_attribution(np.ones(100), np.ones(50), np.ones(100))

    def test_to_dict(self):
        rng = np.random.default_rng(42)
        n = 50
        result = volatility_attribution(
            rng.normal(0, 0.02, size=n),
            np.zeros(n),
            np.ones(n) * 0.02,
        )
        d = result.to_dict()
        self.assertIn("coverage", d)
        self.assertIn("volatility_flag", d)
        self.assertIn("crps_reliability", d)
        self.assertIn("crps_sharpness", d)


# ===========================================================================
# Story 27.3: BMA Attribution
# ===========================================================================

class TestBMAAttribution(unittest.TestCase):
    """AC: bma_attribution returns per-model contribution to CRPS."""

    def test_basic_result(self):
        """Returns BMAAttributionResult."""
        rng = np.random.default_rng(42)
        n = 200
        r = rng.normal(0, 0.02, size=n)

        forecasts = {
            "model_a": (rng.normal(0, 0.01, size=n), np.ones(n) * 0.02),
            "model_b": (rng.normal(0, 0.005, size=n), np.ones(n) * 0.02),
        }
        weights = {"model_a": 0.6, "model_b": 0.4}

        result = bma_attribution(forecasts, weights, r)
        self.assertIsInstance(result, BMAAttributionResult)
        self.assertEqual(result.n_models, 2)
        self.assertEqual(result.n_observations, n)

    def test_beneficial_model_positive_contribution(self):
        """AC: Model that improves CRPS has positive contribution."""
        rng = np.random.default_rng(42)
        n = 500
        r = rng.normal(0.001, 0.02, size=n)

        # Good model: tracks returns well
        good_mu = r * 0.5 + rng.normal(0, 0.005, size=n)
        # Bad model: random noise
        bad_mu = rng.normal(0, 0.05, size=n)

        forecasts = {
            "good": (good_mu, np.ones(n) * 0.02),
            "bad": (bad_mu, np.ones(n) * 0.05),
        }
        weights = {"good": 0.7, "bad": 0.3}

        result = bma_attribution(forecasts, weights, r)
        # Good model should have positive contribution (removing it worsens CRPS)
        self.assertGreater(result.model_contributions["good"], 0)

    def test_harmful_model_negative_contribution(self):
        """AC: Model that worsens CRPS has negative contribution."""
        rng = np.random.default_rng(42)
        n = 500
        r = rng.normal(0, 0.02, size=n)

        # Good model: unbiased
        good_mu = np.zeros(n)
        # Harmful model: large systematic bias
        bad_mu = np.ones(n) * 0.1  # Huge positive bias

        forecasts = {
            "good": (good_mu, np.ones(n) * 0.02),
            "harmful": (bad_mu, np.ones(n) * 0.02),
        }
        weights = {"good": 0.5, "harmful": 0.5}

        result = bma_attribution(forecasts, weights, r)
        # Harmful model should have negative contribution (removing it helps)
        self.assertLess(result.model_contributions["harmful"], 0)
        self.assertIn("harmful", result.harmful_models)

    def test_best_removal_identified(self):
        """AC: Best model to remove is identified."""
        rng = np.random.default_rng(42)
        n = 200
        r = rng.normal(0, 0.02, size=n)

        forecasts = {
            "a": (np.zeros(n), np.ones(n) * 0.02),
            "b": (np.ones(n) * 0.05, np.ones(n) * 0.02),  # Biased
            "c": (np.zeros(n), np.ones(n) * 0.02),
        }
        weights = {"a": 0.4, "b": 0.3, "c": 0.3}

        result = bma_attribution(forecasts, weights, r)
        # Model b (biased) should be the best to remove
        self.assertEqual(result.best_removal_model, "b")
        self.assertGreater(result.best_removal_improvement, 0)

    def test_crps_improves_on_removal(self):
        """AC: Removing worst model improves CRPS by > 0.001."""
        rng = np.random.default_rng(42)
        n = 500
        r = rng.normal(0, 0.02, size=n)

        forecasts = {
            "good1": (np.zeros(n), np.ones(n) * 0.02),
            "good2": (rng.normal(0, 0.005, size=n), np.ones(n) * 0.02),
            "terrible": (np.ones(n) * 0.2, np.ones(n) * 0.1),  # Very bad
        }
        weights = {"good1": 0.4, "good2": 0.3, "terrible": 0.3}

        result = bma_attribution(forecasts, weights, r)
        self.assertGreater(result.best_removal_improvement, CRPS_IMPROVEMENT_THRESHOLD)

    def test_all_equal_models(self):
        """Equal models have similar contributions."""
        rng = np.random.default_rng(42)
        n = 200
        r = rng.normal(0, 0.02, size=n)

        forecasts = {
            "a": (np.zeros(n), np.ones(n) * 0.02),
            "b": (np.zeros(n), np.ones(n) * 0.02),
        }
        weights = {"a": 0.5, "b": 0.5}

        result = bma_attribution(forecasts, weights, r)
        # Both contributions should be similar (removing either has same effect)
        diff = abs(
            result.model_contributions["a"] - result.model_contributions["b"]
        )
        self.assertLess(diff, 0.01)

    def test_single_model_raises(self):
        """Need >= 2 models."""
        with self.assertRaises(ValueError):
            bma_attribution(
                {"only": (np.zeros(100), np.ones(100) * 0.02)},
                {"only": 1.0},
                np.zeros(100),
            )

    def test_too_few_observations_raises(self):
        with self.assertRaises(ValueError):
            bma_attribution(
                {"a": (np.zeros(5), np.ones(5)), "b": (np.zeros(5), np.ones(5))},
                {"a": 0.5, "b": 0.5},
                np.zeros(5),
            )

    def test_to_dict(self):
        rng = np.random.default_rng(42)
        n = 50
        r = rng.normal(0, 0.02, size=n)
        result = bma_attribution(
            {"a": (np.zeros(n), np.ones(n) * 0.02),
             "b": (np.zeros(n), np.ones(n) * 0.02)},
            {"a": 0.5, "b": 0.5},
            r,
        )
        d = result.to_dict()
        self.assertIn("model_contributions", d)
        self.assertIn("combined_crps", d)
        self.assertIn("harmful_models", d)
        self.assertIn("beneficial_models", d)

    def test_missing_weight_defaulted(self):
        """Models without explicit weight get 0."""
        rng = np.random.default_rng(42)
        n = 50
        r = rng.normal(0, 0.02, size=n)

        forecasts = {
            "a": (np.zeros(n), np.ones(n) * 0.02),
            "b": (np.zeros(n), np.ones(n) * 0.02),
        }
        weights = {"a": 1.0}  # b has no weight

        result = bma_attribution(forecasts, weights, r)
        self.assertEqual(result.n_models, 2)


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for forecast attribution pipeline."""

    def test_full_attribution_pipeline(self):
        """Full pipeline: drift + volatility + BMA attribution."""
        rng = np.random.default_rng(42)
        n = 300
        r = rng.normal(0.001, 0.02, size=n)

        # Model forecasts
        mu = r * 0.3 + rng.normal(0, 0.005, size=n)
        sigma = np.ones(n) * 0.02

        # Step 1: Drift attribution
        drift = drift_attribution(r, mu, sigma)
        self.assertLess(drift.direction_error, 0.5)

        # Step 2: Volatility attribution
        vol = volatility_attribution(r, mu, sigma)
        self.assertGreater(vol.coverage, 0.5)

        # Step 3: BMA attribution (2-model ensemble)
        mu2 = rng.normal(0, 0.01, size=n)
        sigma2 = np.ones(n) * 0.03

        forecasts = {
            "good": (mu, sigma),
            "noisy": (mu2, sigma2),
        }
        weights = {"good": 0.7, "noisy": 0.3}

        bma = bma_attribution(forecasts, weights, r)
        self.assertGreater(bma.combined_crps, 0)
        self.assertEqual(bma.n_models, 2)

    def test_unbiased_model_attribution(self):
        """Unbiased model with slight signal: direction error ~50%, coverage near nominal."""
        rng = np.random.default_rng(42)
        n = 1000
        vol = 0.02
        r = rng.normal(0, vol, size=n)
        # Small noisy forecast (not exactly zero, to get meaningful direction)
        mu = rng.normal(0, 0.001, size=n)
        sigma = np.ones(n) * vol

        drift = drift_attribution(r, mu, sigma)
        vol_attr = volatility_attribution(r, mu, sigma, alpha=0.10)

        # Near-zero forecast: direction ~50% (random)
        self.assertGreater(drift.direction_error, 0.35)
        self.assertLess(drift.direction_error, 0.65)

        # Well-calibrated: coverage near 90%
        self.assertGreater(vol_attr.coverage, 0.85)
        self.assertLess(vol_attr.coverage, 0.95)

    def test_biased_model_attribution(self):
        """Biased model shows both drift and vol issues."""
        rng = np.random.default_rng(42)
        n = 500
        r = rng.normal(0.005, 0.02, size=n)
        mu = np.ones(n) * (-0.005)  # Opposite direction bias
        sigma = np.ones(n) * 0.01   # Underestimates volatility

        drift = drift_attribution(r, mu, sigma)
        vol_attr = volatility_attribution(r, mu, sigma)

        # Should flag directional failure
        self.assertGreater(drift.direction_error, DIRECTION_FAILURE_THRESHOLD)

        # Should flag vol underestimate
        self.assertEqual(vol_attr.volatility_flag, "VOL_UNDERESTIMATE")


if __name__ == "__main__":
    unittest.main()
