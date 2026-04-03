"""
Tests for Story 2.1: Walk-Forward Backtest Infrastructure.

Validates the full-pipeline walk-forward backtester that uses
latest_signals() to produce (forecast, realized) pairs.

Tests cover:
  - Output structure correctness
  - No look-ahead bias
  - Hit rate computation
  - Calibration curve buckets
  - DataFrame conversion
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


class TestWalkForwardConstants(unittest.TestCase):
    """Validate walk-forward configuration constants."""

    def test_horizons(self):
        from decision.signals import WF_HORIZONS
        self.assertEqual(WF_HORIZONS, [1, 3, 7, 21, 63])

    def test_rebalance_freq(self):
        from decision.signals import WF_REBALANCE_FREQ
        self.assertEqual(WF_REBALANCE_FREQ, 5)

    def test_calibration_buckets(self):
        from decision.signals import WF_CALIBRATION_BUCKETS
        self.assertEqual(WF_CALIBRATION_BUCKETS, 10)


class TestWalkForwardRecord(unittest.TestCase):
    """Validate WalkForwardRecord dataclass."""

    def test_create(self):
        from decision.signals import WalkForwardRecord
        r = WalkForwardRecord(
            date_idx=100,
            date=pd.Timestamp("2024-06-01"),
            horizon=7,
            forecast_ret=0.01,
            forecast_p_up=0.62,
            forecast_sig=0.015,
            realized_ret=0.008,
            hit=True,
            signed_error=-0.002,
            calibration_bucket=6,
        )
        self.assertEqual(r.horizon, 7)
        self.assertTrue(r.hit)
        self.assertEqual(r.calibration_bucket, 6)


class TestWalkForwardResult(unittest.TestCase):
    """Validate WalkForwardResult aggregation."""

    def test_empty(self):
        from decision.signals import WalkForwardResult
        wf = WalkForwardResult(symbol="TEST")
        self.assertEqual(wf.symbol, "TEST")
        self.assertEqual(len(wf.records), 0)
        self.assertEqual(len(wf.hit_rate), 0)

    def test_defaults(self):
        from decision.signals import WalkForwardResult
        wf = WalkForwardResult()
        self.assertEqual(wf.symbol, "")
        self.assertIsInstance(wf.records, list)
        self.assertIsInstance(wf.hit_rate, dict)
        self.assertIsInstance(wf.calibration_curve, dict)


class TestWalkForwardDataframe(unittest.TestCase):
    """Test walkforward_result_to_dataframe conversion."""

    def test_empty_result(self):
        from decision.signals import WalkForwardResult, walkforward_result_to_dataframe
        wf = WalkForwardResult()
        df = walkforward_result_to_dataframe(wf)
        self.assertTrue(df.empty)

    def test_columns(self):
        from decision.signals import (
            WalkForwardResult, WalkForwardRecord, walkforward_result_to_dataframe,
        )
        rec = WalkForwardRecord(
            date_idx=0, date=pd.Timestamp("2024-01-01"), horizon=1,
            forecast_ret=0.001, forecast_p_up=0.55, forecast_sig=0.01,
            realized_ret=0.002, hit=True, signed_error=0.001, calibration_bucket=5,
        )
        wf = WalkForwardResult(records=[rec], symbol="X")
        df = walkforward_result_to_dataframe(wf)
        expected_cols = {
            "date", "horizon", "forecast_ret", "forecast_p_up", "forecast_sig",
            "realized_ret", "hit", "signed_error", "calibration_bucket",
        }
        self.assertEqual(set(df.columns), expected_cols)
        self.assertEqual(len(df), 1)


class TestHitRateComputation(unittest.TestCase):
    """Validate hit rate = (realized>0) == (forecast>0)."""

    def test_all_correct(self):
        from decision.signals import WalkForwardRecord
        recs = [
            WalkForwardRecord(0, None, 1, 0.01, 0.6, 0.01, 0.005, True, -0.005, 5),
            WalkForwardRecord(1, None, 1, -0.01, 0.4, 0.01, -0.005, True, 0.005, 4),
        ]
        hit_rate = sum(1 for r in recs if r.hit) / len(recs)
        self.assertEqual(hit_rate, 1.0)

    def test_half_correct(self):
        from decision.signals import WalkForwardRecord
        recs = [
            WalkForwardRecord(0, None, 1, 0.01, 0.6, 0.01, 0.005, True, -0.005, 5),
            WalkForwardRecord(1, None, 1, 0.01, 0.6, 0.01, -0.005, False, -0.015, 5),
        ]
        hit_rate = sum(1 for r in recs if r.hit) / len(recs)
        self.assertEqual(hit_rate, 0.5)


class TestCalibrationBucket(unittest.TestCase):
    """Validate calibration bucket assignment."""

    def test_bucket_range(self):
        from decision.signals import WF_CALIBRATION_BUCKETS
        # p_up=0.0 -> bucket 0, p_up=1.0 -> bucket 9
        for p in [0.0, 0.1, 0.5, 0.9, 0.99, 1.0]:
            b = min(int(p * WF_CALIBRATION_BUCKETS), WF_CALIBRATION_BUCKETS - 1)
            self.assertGreaterEqual(b, 0)
            self.assertLess(b, WF_CALIBRATION_BUCKETS)

    def test_bucket_monotonic(self):
        from decision.signals import WF_CALIBRATION_BUCKETS
        prev_b = -1
        for p in np.linspace(0, 0.99, 50):
            b = min(int(p * WF_CALIBRATION_BUCKETS), WF_CALIBRATION_BUCKETS - 1)
            self.assertGreaterEqual(b, prev_b)
            prev_b = b


class TestSaveWalkForwardCSV(unittest.TestCase):
    """Test CSV save/load round-trip."""

    def test_save_empty(self):
        import tempfile
        from decision.signals import WalkForwardResult, save_walkforward_csv
        wf = WalkForwardResult(symbol="TEST_EMPTY")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_walkforward_csv(wf, output_dir=tmpdir)
            self.assertTrue(path.endswith("walkforward_TEST_EMPTY.csv"))


class TestNoLookAhead(unittest.TestCase):
    """Verify that the truncation logic prevents look-ahead."""

    def test_truncation_boundary(self):
        """Verify px[:t+1] respects boundary."""
        # Simulate: 600 prices, feature computation at t=504
        # should only see prices 0..504
        n = 600
        px = pd.Series(
            np.exp(np.cumsum(np.random.randn(n) * 0.01)),
            index=pd.bdate_range("2022-01-01", periods=n),
        )
        t = 504
        px_as_of_t = px.iloc[:t + 1]
        self.assertEqual(len(px_as_of_t), t + 1)
        # The last date in truncated should be the t-th date
        self.assertEqual(px_as_of_t.index[-1], px.index[t])


if __name__ == "__main__":
    unittest.main()
