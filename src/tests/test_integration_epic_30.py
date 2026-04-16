"""
Tests for Epic 30: End-to-End Integration Testing

Story 30.1: validate_pipeline_output + test_full_pipeline_smoke
Story 30.2: golden_scores regression testing
Story 30.3: temporal_cv_consistency
"""

import os
import sys
import json
import tempfile
import unittest

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(SCRIPT_DIR, os.pardir)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.integration_testing import (
    validate_pipeline_output,
    aggregate_smoke_results,
    load_golden_scores,
    save_golden_scores,
    check_regression,
    temporal_cv_split,
    temporal_cv_consistency,
    PipelineSmokeResult,
    RegressionTestResult,
    TemporalCVResult,
    BIC_REGRESSION_THRESHOLD,
    CRPS_REGRESSION_THRESHOLD,
    HIT_RATE_REGRESSION_THRESHOLD,
    CV_MAX_COV,
    CV_MIN_HIT_RATE,
)


# ===========================================================================
# Story 30.1: Full Pipeline Smoke Test
# ===========================================================================

class TestValidatePipelineOutput(unittest.TestCase):
    """AC: validate_pipeline_output checks tune + signal output."""

    def _make_tune_result(self, asset="SPY", bic=-5000.0, n_models=5):
        models = {}
        for i in range(n_models):
            models[f"model_{i}"] = {"bic": bic + i * 10, "loglik": -2500.0}
        return {
            "asset": asset,
            "global": {"models": models, "posteriors": {}},
            "meta": {"timestamp": "2024-01-01"},
        }

    def test_valid_output(self):
        """AC: Valid output passes all checks."""
        tune = self._make_tune_result()
        result = validate_pipeline_output("SPY", tune, signal_direction=1, confidence=0.7)
        self.assertTrue(result.success)
        self.assertEqual(result.asset, "SPY")
        self.assertIsNotNone(result.bic)
        self.assertEqual(result.signal_direction, 1)
        self.assertAlmostEqual(result.confidence, 0.7)
        self.assertEqual(result.n_models, 5)

    def test_none_tune_result(self):
        """AC: None tune result -> failure."""
        result = validate_pipeline_output("FAIL", None)
        self.assertFalse(result.success)
        self.assertIn("None", result.error)

    def test_missing_keys(self):
        """AC: Missing required keys -> failure."""
        result = validate_pipeline_output("BAD", {"foo": "bar"})
        self.assertFalse(result.success)
        self.assertIn("Missing key", result.error)

    def test_invalid_signal_direction(self):
        """AC: Signal direction not in {-1, 0, 1} -> failure."""
        tune = self._make_tune_result()
        result = validate_pipeline_output("SPY", tune, signal_direction=2)
        self.assertFalse(result.success)

    def test_confidence_out_of_range(self):
        """AC: Confidence outside [0, 1] -> failure."""
        tune = self._make_tune_result()
        result = validate_pipeline_output("SPY", tune, confidence=1.5)
        self.assertFalse(result.success)

    def test_nan_confidence(self):
        """AC: NaN confidence -> failure."""
        tune = self._make_tune_result()
        result = validate_pipeline_output("SPY", tune, confidence=float('nan'))
        self.assertFalse(result.success)

    def test_zero_direction(self):
        """AC: direction=0 is valid (no signal)."""
        tune = self._make_tune_result()
        result = validate_pipeline_output("SPY", tune, signal_direction=0, confidence=0.0)
        self.assertTrue(result.success)

    def test_to_dict(self):
        tune = self._make_tune_result()
        result = validate_pipeline_output("SPY", tune)
        d = result.to_dict()
        self.assertIn("asset", d)
        self.assertIn("success", d)
        self.assertIn("bic", d)

    def test_best_bic_selected(self):
        """AC: BIC is the best (lowest) across all models."""
        tune = self._make_tune_result(bic=-6000.0, n_models=3)
        result = validate_pipeline_output("SPY", tune)
        self.assertAlmostEqual(result.bic, -6000.0)


class TestFullPipelineSmoke(unittest.TestCase):
    """AC: test_full_pipeline_smoke aggregates results."""

    def test_all_pass(self):
        results = [
            PipelineSmokeResult(asset="SPY", success=True, bic=-5000),
            PipelineSmokeResult(asset="AAPL", success=True, bic=-4500),
        ]
        summary = aggregate_smoke_results(results)
        self.assertTrue(summary["all_passed"])
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["passed"], 2)

    def test_some_fail(self):
        results = [
            PipelineSmokeResult(asset="SPY", success=True, bic=-5000),
            PipelineSmokeResult(asset="BAD", success=False, error="boom"),
        ]
        summary = aggregate_smoke_results(results)
        self.assertFalse(summary["all_passed"])
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(len(summary["failure_details"]), 1)

    def test_empty(self):
        summary = aggregate_smoke_results([])
        self.assertEqual(summary["total"], 0)
        self.assertTrue(summary["all_passed"])


# ===========================================================================
# Story 30.2: Golden Score Regression Registry
# ===========================================================================

class TestGoldenScores(unittest.TestCase):
    """AC: golden_scores.json stores baseline metrics."""

    def test_save_and_load(self):
        """AC: Save and load golden scores round-trips."""
        scores = {
            "SPY": {"bic": -5000.0, "crps": 0.015, "hit_rate": 0.55},
            "AAPL": {"bic": -4500.0, "crps": 0.018, "hit_rate": 0.52},
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_golden_scores(scores, path)
            loaded = load_golden_scores(path)
            self.assertEqual(loaded["SPY"]["bic"], -5000.0)
            self.assertEqual(loaded["AAPL"]["hit_rate"], 0.52)
        finally:
            os.unlink(path)

    def test_load_nonexistent(self):
        """AC: Missing file returns empty dict."""
        loaded = load_golden_scores("/tmp/nonexistent_golden_scores.json")
        self.assertEqual(loaded, {})


class TestCheckRegression(unittest.TestCase):
    """AC: check_regression detects metric regressions."""

    def _golden(self):
        return {
            "SPY": {"bic": -5000.0, "crps": 0.015, "hit_rate": 0.55},
        }

    def test_no_regression(self):
        """AC: Metrics within threshold -> pass."""
        result = check_regression("SPY", -5005.0, 0.014, 0.56, self._golden())
        self.assertTrue(result.passed)
        self.assertEqual(len(result.regressions), 0)

    def test_bic_regression(self):
        """AC: BIC worsens by > 20 nats -> regression."""
        result = check_regression("SPY", -4970.0, 0.015, 0.55, self._golden())
        self.assertFalse(result.passed)
        self.assertTrue(any("BIC" in r for r in result.regressions))

    def test_crps_regression(self):
        """AC: CRPS worsens by > 0.003 -> regression."""
        result = check_regression("SPY", -5000.0, 0.019, 0.55, self._golden())
        self.assertFalse(result.passed)
        self.assertTrue(any("CRPS" in r for r in result.regressions))

    def test_hit_rate_regression(self):
        """AC: Hit rate drops > 2% -> regression."""
        result = check_regression("SPY", -5000.0, 0.015, 0.52, self._golden())
        self.assertFalse(result.passed)
        self.assertTrue(any("Hit rate" in r for r in result.regressions))

    def test_new_asset_passes(self):
        """AC: New asset (no baseline) always passes."""
        result = check_regression("NEWSTOCK", -3000.0, 0.02, 0.50, self._golden())
        self.assertTrue(result.passed)

    def test_multiple_regressions(self):
        """AC: Multiple metric regressions detected."""
        result = check_regression("SPY", -4900.0, 0.020, 0.50, self._golden())
        self.assertFalse(result.passed)
        self.assertGreater(len(result.regressions), 1)

    def test_none_metrics_pass(self):
        """AC: None metrics don't trigger regression."""
        result = check_regression("SPY", None, None, None, self._golden())
        self.assertTrue(result.passed)

    def test_to_dict(self):
        result = check_regression("SPY", -5000.0, 0.015, 0.55, self._golden())
        d = result.to_dict()
        self.assertIn("asset", d)
        self.assertIn("passed", d)
        self.assertIn("regressions", d)


# ===========================================================================
# Story 30.3: Cross-Validation Consistency
# ===========================================================================

class TestTemporalCVSplit(unittest.TestCase):
    """AC: temporal_cv_split preserves temporal ordering."""

    def test_basic_split(self):
        """AC: 5-fold split on 100 returns."""
        returns = np.random.default_rng(42).normal(0, 0.02, 100)
        splits = temporal_cv_split(returns, n_folds=5)
        self.assertEqual(len(splits), 5)

    def test_temporal_ordering(self):
        """AC: Fold k always uses earlier data than fold k+1."""
        returns = np.random.default_rng(42).normal(0, 0.02, 252)
        splits = temporal_cv_split(returns, n_folds=5)

        for k in range(len(splits) - 1):
            train_k, test_k = splits[k]
            train_k1, test_k1 = splits[k + 1]
            # Test set of fold k comes before test set of fold k+1
            self.assertLess(test_k[-1], test_k1[-1])
            # Train of fold k+1 includes all of fold k's train + test
            self.assertGreater(len(train_k1), len(train_k))

    def test_expanding_window(self):
        """AC: Training windows expand (each fold uses more history)."""
        returns = np.random.default_rng(42).normal(0, 0.02, 500)
        splits = temporal_cv_split(returns, n_folds=5)

        train_sizes = [len(train) for train, _ in splits]
        for i in range(len(train_sizes) - 1):
            self.assertGreater(train_sizes[i + 1], train_sizes[i])

    def test_no_overlap_test(self):
        """AC: Test sets are non-overlapping."""
        returns = np.random.default_rng(42).normal(0, 0.02, 252)
        splits = temporal_cv_split(returns, n_folds=5)

        all_test = np.concatenate([test for _, test in splits])
        self.assertEqual(len(all_test), len(set(all_test)))

    def test_too_few_data_raises(self):
        """AC: Too few data points raises ValueError."""
        with self.assertRaises(ValueError):
            temporal_cv_split(np.array([1.0, 2.0]), n_folds=5)


class TestTemporalCVConsistency(unittest.TestCase):
    """AC: temporal_cv_consistency checks fold stability."""

    def _consistent_folds(self):
        """Metrics that are very consistent across folds."""
        rng = np.random.default_rng(42)
        return [
            {"bic": -5000 + rng.normal(0, 50), "crps": 0.015 + rng.normal(0, 0.001), "hit_rate": 0.55 + rng.normal(0, 0.01)}
            for _ in range(5)
        ]

    def test_consistent_folds_pass(self):
        """AC: Consistent metrics across folds -> pass."""
        result = temporal_cv_consistency(self._consistent_folds(), n_folds=5)
        self.assertIsInstance(result, TemporalCVResult)
        self.assertTrue(result.all_folds_consistent)
        self.assertLess(result.cv_bic, CV_MAX_COV)
        self.assertLess(result.cv_crps, CV_MAX_COV)

    def test_outlier_fold_detected(self):
        """AC: Outlier fold (> 2 SD) is detected."""
        # Use very consistent folds with one extreme outlier
        folds = [
            {"bic": -5000, "crps": 0.015, "hit_rate": 0.55},
            {"bic": -5000, "crps": 0.015, "hit_rate": 0.55},
            {"bic": -5000, "crps": 0.015, "hit_rate": 0.10},  # Extreme outlier
            {"bic": -5000, "crps": 0.015, "hit_rate": 0.55},
            {"bic": -5000, "crps": 0.015, "hit_rate": 0.55},
        ]
        result = temporal_cv_consistency(folds, n_folds=5)
        self.assertIn(2, result.outlier_folds)
        self.assertFalse(result.all_folds_consistent)

    def test_worst_hit_rate_tracked(self):
        """AC: Worst fold's hit rate is tracked."""
        folds = [
            {"bic": -5000, "crps": 0.015, "hit_rate": 0.55},
            {"bic": -4900, "crps": 0.016, "hit_rate": 0.49},
            {"bic": -5100, "crps": 0.014, "hit_rate": 0.53},
        ]
        result = temporal_cv_consistency(folds, n_folds=3)
        self.assertAlmostEqual(result.worst_hit_rate, 0.49)

    def test_worst_hit_rate_below_threshold(self):
        """AC: Worst fold hit rate < 48% -> inconsistent."""
        folds = [
            {"bic": -5000, "crps": 0.015, "hit_rate": 0.55},
            {"bic": -5000, "crps": 0.015, "hit_rate": 0.45},  # Below 48%
            {"bic": -5000, "crps": 0.015, "hit_rate": 0.53},
        ]
        result = temporal_cv_consistency(folds, n_folds=3)
        self.assertFalse(result.all_folds_consistent)

    def test_high_variance_fails(self):
        """AC: CoV > 0.15 across folds -> inconsistent."""
        folds = [
            {"bic": -3000, "crps": 0.010, "hit_rate": 0.60},
            {"bic": -7000, "crps": 0.025, "hit_rate": 0.60},
            {"bic": -5000, "crps": 0.015, "hit_rate": 0.60},
            {"bic": -2000, "crps": 0.030, "hit_rate": 0.60},
            {"bic": -8000, "crps": 0.008, "hit_rate": 0.60},
        ]
        result = temporal_cv_consistency(folds, n_folds=5)
        # At least one CoV should exceed threshold
        self.assertTrue(result.cv_bic > CV_MAX_COV or result.cv_crps > CV_MAX_COV)

    def test_single_fold(self):
        """AC: Single fold is trivially consistent."""
        folds = [{"bic": -5000, "crps": 0.015, "hit_rate": 0.55}]
        result = temporal_cv_consistency(folds, n_folds=1)
        self.assertTrue(result.all_folds_consistent)

    def test_to_dict(self):
        result = temporal_cv_consistency(self._consistent_folds(), n_folds=5)
        d = result.to_dict()
        self.assertIn("n_folds", d)
        self.assertIn("all_folds_consistent", d)
        self.assertIn("worst_hit_rate", d)
        self.assertIn("outlier_folds", d)


if __name__ == "__main__":
    unittest.main()
