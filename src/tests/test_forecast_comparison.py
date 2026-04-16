"""
Test Story 4.7: Forecast Comparison Mode.

Validates:
  1. Save and load snapshot
  2. Compute deltas correctly
  3. Large change detection
  4. Missing previous -> None
  5. Prune old snapshots
  6. Delta arrows
  7. Direction change detected
"""
import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from decision.forecast_comparison import (
    save_signal_snapshot,
    load_previous_snapshot,
    compute_forecast_deltas,
    get_delta_arrow,
    prune_old_snapshots,
    HISTORY_DIR,
    LARGE_CHANGE_RATIO,
)


class TestForecastComparison(unittest.TestCase):
    """Tests for forecast comparison mode."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Monkey-patch HISTORY_DIR for tests
        import decision.forecast_comparison as fc
        self._orig_dir = fc.HISTORY_DIR
        fc.HISTORY_DIR = self.test_dir

    def tearDown(self):
        import decision.forecast_comparison as fc
        fc.HISTORY_DIR = self._orig_dir
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_load(self):
        """Save snapshot, load previous."""
        signals_day1 = {"AAPL": {"horizon_forecasts": {"7": 2.5}}}
        signals_day2 = {"AAPL": {"horizon_forecasts": {"7": 3.0}}}
        
        save_signal_snapshot(signals_day1, "2026-01-01")
        save_signal_snapshot(signals_day2, "2026-01-02")
        
        prev = load_previous_snapshot("2026-01-02")
        self.assertIsNotNone(prev)
        self.assertEqual(prev["AAPL"]["horizon_forecasts"]["7"], 2.5)

    def test_no_previous(self):
        """No prior snapshot -> None."""
        save_signal_snapshot({"AAPL": {}}, "2026-01-01")
        prev = load_previous_snapshot("2026-01-01")
        self.assertIsNone(prev)

    def test_compute_deltas(self):
        """Deltas computed correctly."""
        current = {
            "AAPL": {"horizon_forecasts": {"7": 3.0, "30": -2.0}},
        }
        previous = {
            "AAPL": {"horizon_forecasts": {"7": 1.0, "30": 5.0}},
        }
        deltas = compute_forecast_deltas(current, previous)
        self.assertIn("AAPL", deltas)
        self.assertAlmostEqual(deltas["AAPL"]["7"]["delta"], 2.0)
        self.assertAlmostEqual(deltas["AAPL"]["30"]["delta"], -7.0)

    def test_large_change_detected(self):
        """Large change when |delta| > |previous| * 0.5."""
        current = {"X": {"horizon_forecasts": {"7": 10.0}}}
        previous = {"X": {"horizon_forecasts": {"7": 2.0}}}
        deltas = compute_forecast_deltas(current, previous)
        self.assertTrue(deltas["X"]["7"]["large_change"])

    def test_small_change_not_flagged(self):
        """Small change not flagged as large."""
        current = {"X": {"horizon_forecasts": {"7": 5.1}}}
        previous = {"X": {"horizon_forecasts": {"7": 5.0}}}
        deltas = compute_forecast_deltas(current, previous)
        self.assertFalse(deltas["X"]["7"]["large_change"])

    def test_direction_change(self):
        """Positive -> negative change flagged."""
        current = {"X": {"horizon_forecasts": {"7": -3.0}}}
        previous = {"X": {"horizon_forecasts": {"7": 5.0}}}
        deltas = compute_forecast_deltas(current, previous)
        self.assertAlmostEqual(deltas["X"]["7"]["delta"], -8.0)
        self.assertTrue(deltas["X"]["7"]["large_change"])

    def test_delta_arrows(self):
        """Arrow indicators."""
        self.assertEqual(get_delta_arrow(2.0), "^")
        self.assertEqual(get_delta_arrow(-2.0), "v")
        self.assertEqual(get_delta_arrow(0.001), "=")

    def test_prune_old(self):
        """Prune removes old snapshots."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        save_signal_snapshot({}, "2020-01-01")
        save_signal_snapshot({}, today)
        removed = prune_old_snapshots(retention_days=30)
        self.assertEqual(removed, 1)
        # Recent one still exists
        files = os.listdir(self.test_dir)
        self.assertEqual(len(files), 1)
        self.assertIn(today, files[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
