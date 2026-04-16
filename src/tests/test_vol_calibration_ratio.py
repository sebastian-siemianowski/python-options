"""
Tests for Story 2.3: Realized Volatility Feedback for Sigma Calibration.

Validates the per-horizon vol calibration ratio computation:
    vol_ratio_H = std(realized_ret_H) / mean(forecast_sig_H)
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _make_records(n, horizon, forecast_sig, realized_std, rng_seed=42):
    """Create WalkForwardRecord list with controlled vol."""
    from decision.signals import WalkForwardRecord
    rng = np.random.RandomState(rng_seed)
    recs = []
    for i in range(n):
        f_ret = rng.normal(0.0, 0.01)
        f_sig = forecast_sig
        y = rng.normal(0.0, realized_std)
        recs.append(WalkForwardRecord(
            date_idx=i, date=pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            horizon=horizon, forecast_ret=f_ret, forecast_p_up=0.5,
            forecast_sig=f_sig, realized_ret=y,
            hit=(y > 0) == (f_ret > 0), signed_error=y - f_ret,
            calibration_bucket=5,
        ))
    return recs


class TestVolCalibrationConstants(unittest.TestCase):
    def test_clamp_range(self):
        from tuning.tune import VOL_RATIO_CLAMP_LOW, VOL_RATIO_CLAMP_HIGH
        self.assertEqual(VOL_RATIO_CLAMP_LOW, 0.5)
        self.assertEqual(VOL_RATIO_CLAMP_HIGH, 2.0)


class TestVolCalibrationRatios(unittest.TestCase):

    def test_well_calibrated(self):
        """When realized vol matches forecast sig, ratio ~= 1."""
        from tuning.tune import compute_vol_calibration_ratios
        recs = _make_records(500, horizon=7, forecast_sig=0.02, realized_std=0.02)
        ratios = compute_vol_calibration_ratios(recs, horizons=[7])
        self.assertIn(7, ratios)
        self.assertAlmostEqual(ratios[7], 1.0, delta=0.15)

    def test_overconfident(self):
        """When realized vol >> forecast sig, ratio > 1."""
        from tuning.tune import compute_vol_calibration_ratios
        recs = _make_records(500, horizon=7, forecast_sig=0.01, realized_std=0.02)
        ratios = compute_vol_calibration_ratios(recs, horizons=[7])
        self.assertGreater(ratios[7], 1.5)

    def test_too_cautious(self):
        """When realized vol << forecast sig, ratio < 1."""
        from tuning.tune import compute_vol_calibration_ratios
        recs = _make_records(500, horizon=7, forecast_sig=0.03, realized_std=0.01)
        ratios = compute_vol_calibration_ratios(recs, horizons=[7])
        self.assertLess(ratios[7], 0.6)

    def test_clamped_high(self):
        """Extreme overconfidence clamped at 2.0."""
        from tuning.tune import compute_vol_calibration_ratios, VOL_RATIO_CLAMP_HIGH
        recs = _make_records(500, horizon=7, forecast_sig=0.005, realized_std=0.05)
        ratios = compute_vol_calibration_ratios(recs, horizons=[7])
        self.assertLessEqual(ratios[7], VOL_RATIO_CLAMP_HIGH)

    def test_clamped_low(self):
        """Extreme caution clamped at 0.5."""
        from tuning.tune import compute_vol_calibration_ratios, VOL_RATIO_CLAMP_LOW
        recs = _make_records(500, horizon=7, forecast_sig=0.05, realized_std=0.005)
        ratios = compute_vol_calibration_ratios(recs, horizons=[7])
        self.assertGreaterEqual(ratios[7], VOL_RATIO_CLAMP_LOW)

    def test_too_few_records(self):
        """With < 10 records, ratio defaults to 1.0."""
        from tuning.tune import compute_vol_calibration_ratios
        recs = _make_records(5, horizon=7, forecast_sig=0.02, realized_std=0.04)
        ratios = compute_vol_calibration_ratios(recs, horizons=[7])
        self.assertEqual(ratios[7], 1.0)

    def test_multiple_horizons(self):
        """Different horizons can have different ratios."""
        from tuning.tune import compute_vol_calibration_ratios
        recs_h1 = _make_records(200, horizon=1, forecast_sig=0.01, realized_std=0.01)
        recs_h7 = _make_records(200, horizon=7, forecast_sig=0.01, realized_std=0.02)
        ratios = compute_vol_calibration_ratios(recs_h1 + recs_h7, horizons=[1, 7])
        self.assertIn(1, ratios)
        self.assertIn(7, ratios)
        # H=7 should have higher ratio (realized is bigger)
        self.assertGreater(ratios[7], ratios[1])

    def test_in_valid_range(self):
        """All ratios within [0.5, 2.0]."""
        from tuning.tune import compute_vol_calibration_ratios
        recs = _make_records(300, horizon=21, forecast_sig=0.015, realized_std=0.018)
        ratios = compute_vol_calibration_ratios(recs, horizons=[21])
        for h, r in ratios.items():
            self.assertGreaterEqual(r, 0.5)
            self.assertLessEqual(r, 2.0)


if __name__ == "__main__":
    unittest.main()
