"""
Test Story 3.1: Incremental Tuning with Change Detection.

Validates:
  1. Content hash computed from price CSV
  2. Changed data detected by hash comparison
  3. Unchanged data skipped
  4. Missing file handled gracefully
  5. Stamp adds hash/date to tune result
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from tuning.tune import (
    compute_price_data_hash,
    get_last_price_date,
    needs_retune,
    stamp_tune_result,
)


class TestIncrementalTuning(unittest.TestCase):
    """Tests for content-based change detection."""

    def setUp(self):
        """Create a temp directory with a mock price CSV."""
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, "TEST_1d.csv")
        lines = ["Date,Open,High,Low,Close,Volume\n"]
        for i in range(30):
            lines.append(f"2024-01-{i+1:02d},100.0,101.0,99.0,100.{i},1000\n")
        with open(self.csv_path, 'w') as f:
            f.writelines(lines)

    def test_hash_computed(self):
        """Hash is computed from price data."""
        h = compute_price_data_hash("TEST", self.tmpdir)
        self.assertIsNotNone(h)
        self.assertEqual(len(h), 64)  # SHA256 hex digest

    def test_hash_stable(self):
        """Same data produces same hash."""
        h1 = compute_price_data_hash("TEST", self.tmpdir)
        h2 = compute_price_data_hash("TEST", self.tmpdir)
        self.assertEqual(h1, h2)

    def test_hash_changes_with_data(self):
        """New data row produces different hash."""
        h1 = compute_price_data_hash("TEST", self.tmpdir)
        with open(self.csv_path, 'a') as f:
            f.write("2024-02-01,100.0,101.0,99.0,101.5,2000\n")
        h2 = compute_price_data_hash("TEST", self.tmpdir)
        self.assertNotEqual(h1, h2)

    def test_missing_file_returns_none(self):
        """Missing price file returns None hash."""
        h = compute_price_data_hash("NONEXISTENT", self.tmpdir)
        self.assertIsNone(h)

    def test_get_last_price_date(self):
        """Last price date extracted from CSV."""
        date = get_last_price_date("TEST", self.tmpdir)
        self.assertIsNotNone(date)
        self.assertTrue(date.startswith("2024-01-"))

    def test_needs_retune_no_cache(self):
        """No cached params -> needs retune."""
        self.assertTrue(needs_retune("TEST", None, self.tmpdir))

    def test_needs_retune_no_hash(self):
        """Cached params without hash -> needs retune."""
        cached = {"global": {"q": 1e-5}}
        self.assertTrue(needs_retune("TEST", cached, self.tmpdir))

    def test_no_retune_same_hash(self):
        """Same hash -> does NOT need retune."""
        h = compute_price_data_hash("TEST", self.tmpdir)
        cached = {"global": {"q": 1e-5, "price_data_hash": h}}
        self.assertFalse(needs_retune("TEST", cached, self.tmpdir))

    def test_retune_after_data_change(self):
        """Hash mismatch -> needs retune."""
        cached = {"global": {"q": 1e-5, "price_data_hash": "old_hash_value"}}
        self.assertTrue(needs_retune("TEST", cached, self.tmpdir))

    def test_stamp_adds_hash(self):
        """stamp_tune_result adds hash and date to result."""
        result = {"global": {"q": 1e-5}}
        stamped = stamp_tune_result(result, "TEST", self.tmpdir)
        self.assertIn("price_data_hash", stamped["global"])
        self.assertIn("last_price_date", stamped["global"])

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
