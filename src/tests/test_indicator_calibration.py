import os
import sys
import unittest

import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from calibration.indicator_calibration import (
    apply_indicator_emos,
    beta_calibration_transform,
    compute_indicator_pit_audit,
    compute_threshold_stability,
    fit_indicator_emos_params,
)


class IndicatorCalibrationTest(unittest.TestCase):
    def test_pit_audit_reports_finite_ad_and_berkowitz_moments(self):
        pit = np.linspace(0.05, 0.95, 25)
        audit = compute_indicator_pit_audit(pit)
        self.assertEqual(audit["n"], 25.0)
        self.assertTrue(np.isfinite(audit["ks_stat"]))
        self.assertTrue(np.isfinite(audit["ad_stat"]))
        self.assertTrue(np.isfinite(audit["berkowitz_moment_error"]))
        self.assertLess(audit["ks_stat"], 0.10)

    def test_indicator_emos_fit_and_apply_are_bounded(self):
        realized = np.array([0.01, -0.02, 0.03, 0.00, 0.02, -0.01])
        mu = np.zeros_like(realized)
        sigma = np.full_like(realized, 0.02)
        indicators = np.column_stack([
            np.linspace(-1.0, 1.0, len(realized)),
            np.array([0.2, -0.1, 0.3, -0.2, 0.1, -0.3]),
        ])
        params = fit_indicator_emos_params(realized, mu, sigma, indicators)
        adj_mu, adj_sigma = apply_indicator_emos(mu, sigma, indicators, params, max_abs_mean_adjustment=0.03)
        self.assertEqual(adj_mu.shape, realized.shape)
        self.assertEqual(adj_sigma.shape, realized.shape)
        self.assertLessEqual(np.max(np.abs(adj_mu - mu)), 0.03)
        self.assertTrue(np.all(adj_sigma >= 0.02 * 0.70))
        self.assertTrue(np.all(adj_sigma <= 0.02 * 1.80))

    def test_beta_calibration_and_threshold_stability_are_finite(self):
        probabilities = np.array([0.45, 0.52, 0.58, 0.62, 0.70, 0.80])
        outcomes = np.array([0, 0, 1, 1, 1, 1], dtype=float)
        calibrated = beta_calibration_transform(probabilities, a=1.1, b=0.9, c=-0.05)
        self.assertTrue(np.all((calibrated > 0.0) & (calibrated < 1.0)))
        audit = compute_threshold_stability(calibrated, outcomes)
        self.assertTrue(np.isfinite(audit["brier"]))
        self.assertTrue(np.isfinite(audit["hit_rate_spread"]))
        self.assertGreaterEqual(audit["mean_coverage"], 0.0)
        self.assertLessEqual(audit["mean_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
