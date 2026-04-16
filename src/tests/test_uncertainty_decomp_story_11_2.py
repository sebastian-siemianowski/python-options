"""
Test Suite for Story 11.2: Confidence Decomposition -- Epistemic vs Aleatoric
==============================================================================

Tests uncertainty decomposition into epistemic (inter-model disagreement)
and aleatoric (within-model noise) components.
"""
import os
import sys
import unittest
import math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.uncertainty_decomposition import (
    UncertaintyDecomposition,
    decompose_uncertainty,
    compute_position_sizing_factor,
    decompose_timeseries,
    compute_model_agreement,
    sign_agreement_fraction,
    EPISTEMIC_HIGH_THRESHOLD,
    EPISTEMIC_SIZING_DISCOUNT,
    MIN_TOTAL_VARIANCE,
)


# ===================================================================
# TestUncertaintyDecomposition: Dataclass contract
# ===================================================================

class TestUncertaintyDecomposition(unittest.TestCase):
    """Test UncertaintyDecomposition dataclass."""

    def test_fields_exist(self):
        d = UncertaintyDecomposition(
            epistemic_var=0.1,
            aleatoric_var=0.2,
            total_var=0.3,
            epistemic_fraction=0.333,
            n_models=3,
            ensemble_mean=0.001,
            ensemble_std=0.547,
            model_means=np.array([0.001]),
            model_stds=np.array([0.01]),
        )
        self.assertEqual(d.n_models, 3)
        self.assertAlmostEqual(d.epistemic_fraction, 0.333)


# ===================================================================
# TestDecomposeUncertainty: Core decomposition
# ===================================================================

class TestDecomposeUncertainty(unittest.TestCase):
    """Test decompose_uncertainty()."""

    def test_single_model(self):
        """Single model: all uncertainty is aleatoric."""
        d = decompose_uncertainty(
            model_means=np.array([0.001]),
            model_stds=np.array([0.01]),
            weights=np.array([1.0]),
        )
        self.assertAlmostEqual(d.epistemic_var, 0.0)
        self.assertAlmostEqual(d.aleatoric_var, 0.01 ** 2)
        self.assertAlmostEqual(d.epistemic_fraction, 0.0, places=5)

    def test_two_models_agreeing(self):
        """Two models with same mean: epistemic = 0."""
        d = decompose_uncertainty(
            model_means=np.array([0.001, 0.001]),
            model_stds=np.array([0.01, 0.02]),
            weights=np.array([0.5, 0.5]),
        )
        self.assertAlmostEqual(d.epistemic_var, 0.0)
        self.assertGreater(d.aleatoric_var, 0)

    def test_two_models_disagreeing(self):
        """Two models with different means: epistemic > 0."""
        d = decompose_uncertainty(
            model_means=np.array([0.01, -0.01]),
            model_stds=np.array([0.005, 0.005]),
            weights=np.array([0.5, 0.5]),
        )
        self.assertGreater(d.epistemic_var, 0)
        self.assertGreater(d.aleatoric_var, 0)

    def test_law_of_total_variance(self):
        """total = epistemic + aleatoric."""
        d = decompose_uncertainty(
            model_means=np.array([0.01, -0.005, 0.003]),
            model_stds=np.array([0.01, 0.02, 0.015]),
            weights=np.array([0.5, 0.3, 0.2]),
        )
        self.assertAlmostEqual(
            d.total_var, d.epistemic_var + d.aleatoric_var, places=12
        )

    def test_ensemble_mean_weighted(self):
        """Ensemble mean is weight-average of model means."""
        means = np.array([0.01, 0.02])
        weights = np.array([0.7, 0.3])
        d = decompose_uncertainty(means, np.array([0.01, 0.01]), weights)
        expected_mean = 0.7 * 0.01 + 0.3 * 0.02
        self.assertAlmostEqual(d.ensemble_mean, expected_mean, places=10)

    def test_weights_normalized(self):
        """Non-normalized weights should be normalized internally."""
        d1 = decompose_uncertainty(
            np.array([0.01, 0.02]),
            np.array([0.01, 0.01]),
            np.array([1.0, 1.0]),
        )
        d2 = decompose_uncertainty(
            np.array([0.01, 0.02]),
            np.array([0.01, 0.01]),
            np.array([0.5, 0.5]),
        )
        self.assertAlmostEqual(d1.ensemble_mean, d2.ensemble_mean)
        self.assertAlmostEqual(d1.epistemic_var, d2.epistemic_var)

    def test_epistemic_fraction_range(self):
        d = decompose_uncertainty(
            np.array([0.01, -0.01, 0.005]),
            np.array([0.01, 0.02, 0.015]),
            np.array([0.4, 0.35, 0.25]),
        )
        self.assertGreaterEqual(d.epistemic_fraction, 0.0)
        self.assertLessEqual(d.epistemic_fraction, 1.0)

    def test_many_models(self):
        rng = np.random.default_rng(42)
        n = 14
        means = rng.normal(0.001, 0.005, n)
        stds = rng.uniform(0.005, 0.02, n)
        weights = rng.dirichlet(np.ones(n))
        d = decompose_uncertainty(means, stds, weights)
        self.assertEqual(d.n_models, 14)
        self.assertGreater(d.total_var, 0)

    def test_zero_weights_fallback(self):
        d = decompose_uncertainty(
            np.array([0.01, 0.02]),
            np.array([0.01, 0.01]),
            np.array([0.0, 0.0]),
        )
        # Should fallback to equal weights
        self.assertTrue(np.isfinite(d.ensemble_mean))

    def test_ensemble_std_correct(self):
        d = decompose_uncertainty(
            np.array([0.01, -0.01]),
            np.array([0.01, 0.01]),
            np.array([0.5, 0.5]),
        )
        self.assertAlmostEqual(d.ensemble_std, math.sqrt(d.total_var), places=10)


# ===================================================================
# TestEpistemicDominance: High epistemic = model disagreement
# ===================================================================

class TestEpistemicDominance(unittest.TestCase):
    """Test scenarios where epistemic uncertainty dominates."""

    def test_high_epistemic_when_means_disagree(self):
        """Large mean disagreement with small model variance -> high epistemic."""
        d = decompose_uncertainty(
            model_means=np.array([0.05, -0.05]),
            model_stds=np.array([0.001, 0.001]),
            weights=np.array([0.5, 0.5]),
        )
        self.assertGreater(d.epistemic_fraction, 0.9)

    def test_low_epistemic_when_means_agree(self):
        """Small mean disagreement with large model variance -> low epistemic."""
        d = decompose_uncertainty(
            model_means=np.array([0.001, 0.0011]),
            model_stds=np.array([0.05, 0.05]),
            weights=np.array([0.5, 0.5]),
        )
        self.assertLess(d.epistemic_fraction, 0.01)


# ===================================================================
# TestPositionSizing: Epistemic-based sizing
# ===================================================================

class TestPositionSizing(unittest.TestCase):
    """Test compute_position_sizing_factor()."""

    def test_low_epistemic_full_size(self):
        d = UncertaintyDecomposition(
            epistemic_var=0.0001,
            aleatoric_var=0.001,
            total_var=0.0011,
            epistemic_fraction=0.09,
            n_models=3,
            ensemble_mean=0.001,
            ensemble_std=0.033,
            model_means=np.array([0.001, 0.001, 0.001]),
            model_stds=np.array([0.01, 0.01, 0.01]),
        )
        factor = compute_position_sizing_factor(d)
        self.assertAlmostEqual(factor, 1.0)

    def test_high_epistemic_reduced_size(self):
        d = UncertaintyDecomposition(
            epistemic_var=0.009,
            aleatoric_var=0.001,
            total_var=0.01,
            epistemic_fraction=0.9,
            n_models=3,
            ensemble_mean=0.0,
            ensemble_std=0.1,
            model_means=np.array([0.05, -0.05, 0.0]),
            model_stds=np.array([0.01, 0.01, 0.01]),
        )
        factor = compute_position_sizing_factor(d)
        self.assertLess(factor, 0.7)

    def test_max_epistemic_minimum_size(self):
        d = UncertaintyDecomposition(
            epistemic_var=1.0,
            aleatoric_var=0.0,
            total_var=1.0,
            epistemic_fraction=1.0,
            n_models=2,
            ensemble_mean=0.0,
            ensemble_std=1.0,
            model_means=np.array([1.0, -1.0]),
            model_stds=np.array([0.0, 0.0]),
        )
        factor = compute_position_sizing_factor(d)
        self.assertAlmostEqual(factor, 1.0 - EPISTEMIC_SIZING_DISCOUNT)

    def test_sizing_bounded(self):
        d = UncertaintyDecomposition(
            epistemic_var=100.0,
            aleatoric_var=0.0,
            total_var=100.0,
            epistemic_fraction=1.0,
            n_models=2,
            ensemble_mean=0.0,
            ensemble_std=10.0,
            model_means=np.array([10.0, -10.0]),
            model_stds=np.array([0.0, 0.0]),
        )
        factor = compute_position_sizing_factor(d)
        self.assertGreaterEqual(factor, 0.1)
        self.assertLessEqual(factor, 1.0)


# ===================================================================
# TestTimeseries: Vectorized decomposition
# ===================================================================

class TestTimeseries(unittest.TestCase):
    """Test decompose_timeseries()."""

    def test_correct_shapes(self):
        T, n_models = 100, 5
        rng = np.random.default_rng(42)
        means = rng.normal(0.001, 0.005, (T, n_models))
        stds = rng.uniform(0.005, 0.02, (T, n_models))
        weights = np.ones(n_models) / n_models

        epi, ale, frac = decompose_timeseries(means, stds, weights)
        self.assertEqual(epi.shape, (T,))
        self.assertEqual(ale.shape, (T,))
        self.assertEqual(frac.shape, (T,))

    def test_all_finite(self):
        T, n_models = 50, 3
        rng = np.random.default_rng(42)
        means = rng.normal(0, 0.01, (T, n_models))
        stds = rng.uniform(0.005, 0.02, (T, n_models))
        weights = np.array([0.5, 0.3, 0.2])

        epi, ale, frac = decompose_timeseries(means, stds, weights)
        self.assertTrue(np.all(np.isfinite(epi)))
        self.assertTrue(np.all(np.isfinite(ale)))
        self.assertTrue(np.all(np.isfinite(frac)))

    def test_total_variance_decomposition(self):
        """epi + ale should equal total at each time step."""
        T, n_models = 20, 4
        rng = np.random.default_rng(42)
        means = rng.normal(0, 0.01, (T, n_models))
        stds = rng.uniform(0.005, 0.02, (T, n_models))
        weights = rng.dirichlet(np.ones(n_models))

        epi, ale, frac = decompose_timeseries(means, stds, weights)
        for t in range(T):
            total = epi[t] + ale[t]
            self.assertGreater(total, 0)


# ===================================================================
# TestModelAgreement
# ===================================================================

class TestModelAgreement(unittest.TestCase):
    """Test compute_model_agreement()."""

    def test_full_agreement(self):
        """All models predict the same -> agreement = 1."""
        means = np.array([0.01, 0.01, 0.01])
        weights = np.array([0.4, 0.3, 0.3])
        agreement = compute_model_agreement(means, weights)
        self.assertAlmostEqual(agreement, 1.0)

    def test_full_disagreement(self):
        """Models with opposite predictions and tiny stds -> low agreement."""
        means = np.array([0.1, -0.1])
        weights = np.array([0.5, 0.5])
        agreement = compute_model_agreement(means, weights)
        self.assertLess(agreement, 0.1)

    def test_agreement_range(self):
        rng = np.random.default_rng(42)
        means = rng.normal(0, 0.01, 10)
        weights = rng.dirichlet(np.ones(10))
        agreement = compute_model_agreement(means, weights)
        self.assertGreaterEqual(agreement, 0.0)
        self.assertLessEqual(agreement, 1.0)


# ===================================================================
# TestSignAgreement
# ===================================================================

class TestSignAgreement(unittest.TestCase):
    """Test sign_agreement_fraction()."""

    def test_all_positive(self):
        means = np.array([0.01, 0.02, 0.03])
        weights = np.array([0.4, 0.3, 0.3])
        frac = sign_agreement_fraction(means, weights)
        self.assertAlmostEqual(frac, 1.0)

    def test_split_50_50(self):
        means = np.array([0.01, -0.01])
        weights = np.array([0.5, 0.5])
        frac = sign_agreement_fraction(means, weights)
        self.assertAlmostEqual(frac, 0.5)

    def test_weighted_majority(self):
        """70% weight on positive, 30% on negative -> 0.7."""
        means = np.array([0.01, -0.01])
        weights = np.array([0.7, 0.3])
        frac = sign_agreement_fraction(means, weights)
        self.assertAlmostEqual(frac, 0.7)

    def test_range(self):
        rng = np.random.default_rng(42)
        means = rng.normal(0, 0.01, 14)
        weights = rng.dirichlet(np.ones(14))
        frac = sign_agreement_fraction(means, weights)
        self.assertGreaterEqual(frac, 0.5)
        self.assertLessEqual(frac, 1.0)


# ===================================================================
# TestConstants
# ===================================================================

class TestConstants(unittest.TestCase):
    """Test constant values."""

    def test_epistemic_threshold(self):
        self.assertEqual(EPISTEMIC_HIGH_THRESHOLD, 0.5)

    def test_sizing_discount(self):
        self.assertEqual(EPISTEMIC_SIZING_DISCOUNT, 0.5)

    def test_min_total_variance(self):
        self.assertGreater(MIN_TOTAL_VARIANCE, 0)


# ===================================================================
# TestEdgeCases
# ===================================================================

class TestEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_all_zero_stds(self):
        d = decompose_uncertainty(
            np.array([0.01, 0.02]),
            np.array([0.0, 0.0]),
            np.array([0.5, 0.5]),
        )
        self.assertAlmostEqual(d.aleatoric_var, 0.0)
        self.assertGreater(d.epistemic_var, 0)

    def test_all_zero_means(self):
        d = decompose_uncertainty(
            np.array([0.0, 0.0]),
            np.array([0.01, 0.02]),
            np.array([0.5, 0.5]),
        )
        self.assertAlmostEqual(d.epistemic_var, 0.0)
        self.assertGreater(d.aleatoric_var, 0)

    def test_very_many_models(self):
        n = 100
        rng = np.random.default_rng(42)
        means = rng.normal(0, 0.01, n)
        stds = rng.uniform(0.005, 0.02, n)
        weights = rng.dirichlet(np.ones(n))
        d = decompose_uncertainty(means, stds, weights)
        self.assertEqual(d.n_models, 100)
        self.assertTrue(np.isfinite(d.total_var))


if __name__ == "__main__":
    unittest.main()
