"""
Test Story 2.10: Benchmark Validation Harness.

Validates:
  1. Synthetic improvement -> GREEN
  2. Synthetic regression -> RED
  3. Within tolerance -> AMBER
  4. Metric computation correctness
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from calibration.ensemble_validator import (
    compute_hit_rate,
    compute_sharpe,
    compute_ece,
    compare_metrics,
    evaluate_metrics,
)


class TestBenchmarkHarness(unittest.TestCase):
    """Tests for benchmark validation harness."""

    def test_hit_rate_perfect(self):
        """Perfect forecasts -> hit rate 1.0."""
        forecasts = [1.0, -1.0, 1.0, -1.0]
        realized = [0.5, -0.3, 0.2, -0.8]
        self.assertAlmostEqual(compute_hit_rate(forecasts, realized), 1.0)

    def test_hit_rate_random(self):
        """Random forecasts -> hit rate near 0.5."""
        np.random.seed(42)
        forecasts = np.random.randn(1000).tolist()
        realized = np.random.randn(1000).tolist()
        hr = compute_hit_rate(forecasts, realized)
        self.assertAlmostEqual(hr, 0.5, delta=0.05)

    def test_improvement_green(self):
        """Higher hit rate + lower CRPS -> GREEN."""
        baseline = {"hit_rate": 0.55, "sharpe": 0.1, "crps": 0.02, "ece": 0.05}
        current = {"hit_rate": 0.58, "sharpe": 0.12, "crps": 0.018, "ece": 0.04}
        result = compare_metrics(current, baseline)
        self.assertEqual(result["verdict"], "GREEN")
        for detail in result["details"].values():
            self.assertEqual(detail, "GREEN")

    def test_regression_red(self):
        """Large degradation -> RED."""
        baseline = {"hit_rate": 0.60, "sharpe": 0.2, "crps": 0.015, "ece": 0.03}
        current = {"hit_rate": 0.50, "sharpe": 0.05, "crps": 0.030, "ece": 0.08}
        result = compare_metrics(current, baseline)
        self.assertEqual(result["verdict"], "RED")

    def test_within_tolerance_amber(self):
        """Small degradation within tolerance -> AMBER."""
        baseline = {"hit_rate": 0.55, "sharpe": 0.10, "crps": 0.020, "ece": 0.050}
        # Small drops within tolerance
        current = {"hit_rate": 0.545, "sharpe": 0.08, "crps": 0.021, "ece": 0.055}
        result = compare_metrics(current, baseline)
        self.assertEqual(result["verdict"], "AMBER")

    def test_evaluate_metrics_structure(self):
        """evaluate_metrics returns correct keys."""
        forecasts = [0.5, -0.3, 0.8, -0.1, 1.2]
        realized = [0.3, -0.5, 0.6, 0.1, 0.9]
        metrics = evaluate_metrics(forecasts, realized)
        for key in ["hit_rate", "sharpe", "crps", "ece"]:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)

    def test_compare_has_deltas(self):
        """compare_metrics returns deltas and details."""
        baseline = {"hit_rate": 0.55, "sharpe": 0.10, "crps": 0.020, "ece": 0.050}
        current = {"hit_rate": 0.58, "sharpe": 0.12, "crps": 0.018, "ece": 0.045}
        result = compare_metrics(current, baseline)
        self.assertIn("deltas", result)
        self.assertIn("verdict", result)
        self.assertIn("details", result)
        self.assertAlmostEqual(result["deltas"]["hit_rate"], 0.03, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
