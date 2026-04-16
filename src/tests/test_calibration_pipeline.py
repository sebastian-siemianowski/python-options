"""
Tests for Story 2.6: Automated Calibration Pipeline.

Validates run_calibration_pipeline() and save_calibration_report().
Uses mocked walk-forward results to avoid real data fetching.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _make_mock_wf_result(symbol, n=100, horizons=None):
    """Create a mock WalkForwardResult."""
    from decision.signals import WalkForwardRecord, WalkForwardResult
    if horizons is None:
        horizons = [1, 7]
    rng = np.random.RandomState(42)
    recs = []
    for h in horizons:
        for i in range(n):
            f_ret = rng.normal(0.001, 0.01)
            y = rng.normal(0.001, 0.01)
            p_up = 0.55 + rng.normal(0, 0.05)
            recs.append(WalkForwardRecord(
                date_idx=i, date=pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                horizon=h, forecast_ret=f_ret, forecast_p_up=np.clip(p_up, 0, 1),
                forecast_sig=0.02, realized_ret=y,
                hit=(y > 0) == (f_ret > 0), signed_error=y - f_ret,
                calibration_bucket=min(9, int(abs(f_ret) * 500)),
            ))
    wf = WalkForwardResult(symbol=symbol, records=recs)
    # Populate hit_rate
    for h in horizons:
        h_recs = [r for r in recs if r.horizon == h]
        wf.hit_rate[h] = sum(1 for r in h_recs if r.hit) / len(h_recs)
        wf.n_forecasts[h] = len(h_recs)
    return wf


class TestCalibrationConstants(unittest.TestCase):
    def test_constants(self):
        from tuning.tune import CALIBRATION_REPORT_DIR, CALIBRATION_DEFAULT_START
        self.assertEqual(CALIBRATION_REPORT_DIR, "src/data/calibration")
        self.assertEqual(CALIBRATION_DEFAULT_START, "2024-01-01")


class TestCalibrationPipeline(unittest.TestCase):

    @patch("tuning.tune.run_calibration_pipeline.__module__", "tuning.tune")
    def test_pipeline_with_mocked_wf(self):
        """Pipeline runs and produces report with mocked walk-forward."""
        from tuning.tune import run_calibration_pipeline

        mock_wf_spy = _make_mock_wf_result("SPY", n=100, horizons=[1, 7])
        mock_wf_qqq = _make_mock_wf_result("QQQ", n=100, horizons=[1, 7])

        with patch("decision.signals.run_walk_forward_backtest") as mock_wf:
            mock_wf.side_effect = [mock_wf_spy, mock_wf_qqq]
            cache = {}
            report = run_calibration_pipeline(
                ["SPY", "QQQ"], cache, horizons=[1, 7]
            )

        self.assertIn("assets", report)
        self.assertIn("summary", report)
        self.assertEqual(report["summary"]["total_assets"], 2)
        self.assertEqual(report["summary"]["success"], 2)

    @patch("tuning.tune.run_calibration_pipeline.__module__", "tuning.tune")
    def test_cache_updated(self):
        """Pipeline writes calibration data to cache."""
        from tuning.tune import run_calibration_pipeline

        mock_wf = _make_mock_wf_result("SPY", n=100, horizons=[1, 7])

        with patch("decision.signals.run_walk_forward_backtest", return_value=mock_wf):
            cache = {"SPY": {"global": {"q": 1e-5}}}
            run_calibration_pipeline(["SPY"], cache, horizons=[1, 7])

        self.assertIn("calibration", cache["SPY"])
        self.assertIn("emos", cache["SPY"]["calibration"])
        self.assertIn("vol_ratios", cache["SPY"]["calibration"])
        self.assertIn("dig", cache["SPY"]["calibration"])

    @patch("tuning.tune.run_calibration_pipeline.__module__", "tuning.tune")
    def test_pipeline_handles_failure(self):
        """Pipeline gracefully handles asset failure."""
        from tuning.tune import run_calibration_pipeline

        with patch("decision.signals.run_walk_forward_backtest",
                    side_effect=Exception("No data")):
            cache = {}
            report = run_calibration_pipeline(["BAD"], cache, horizons=[1, 7])

        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(report["assets"]["BAD"]["status"], "failed")


class TestCalibrationReport(unittest.TestCase):
    def test_save_report_json(self):
        """Report saves as valid JSON."""
        from tuning.tune import save_calibration_report
        report = {
            "assets": {"SPY": {"status": "success", "hit_rate": {1: 0.55}}},
            "summary": {"total_assets": 1, "success": 1, "failed": 0, "skipped": 0},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_calibration_report(report, output_dir=tmpdir)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["summary"]["success"], 1)

    def test_report_human_readable(self):
        """Report has indentation for readability."""
        from tuning.tune import save_calibration_report
        report = {"assets": {}, "summary": {"total": 0}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_calibration_report(report, output_dir=tmpdir)
            with open(path) as f:
                text = f.read()
            # Indented = human readable
            self.assertIn("  ", text)

    def test_report_numpy_types(self):
        """Report handles numpy types in serialization."""
        from tuning.tune import save_calibration_report
        report = {
            "assets": {"SPY": {"val": np.float64(1.5), "arr": np.array([1, 2])}},
            "summary": {"n": np.int64(3)},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_calibration_report(report, output_dir=tmpdir)
            with open(path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["summary"]["n"], 3)


class TestCLIFlag(unittest.TestCase):
    def test_calibrate_argparse(self):
        """--calibrate flag is recognized by argparse."""
        import argparse
        from tuning.tune import main
        # We just verify the flag exists by importing and inspecting
        # (don't actually run main which requires data)
        import inspect
        src = inspect.getsource(main)
        self.assertIn("--calibrate", src)


if __name__ == "__main__":
    unittest.main()
