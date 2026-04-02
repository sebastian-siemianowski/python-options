"""
Test Story 4.2: Data Version Tracking and Consistency Checksums.

Validates:
  1. price_hash stable for same input
  2. price_hash changes when data changes
  3. tune_hash chains correctly
  4. signal_hash chains correctly
  5. Consistent chain -> green
  6. Broken chain -> red
  7. Stale signals -> amber
"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from calibration.data_version import (
    compute_price_hash,
    compute_tune_hash,
    compute_signal_hash,
    verify_hash_chain,
)


class TestDataVersion(unittest.TestCase):
    """Tests for data version tracking."""

    def _write_csv(self, path, lines):
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def test_price_hash_stable(self):
        """Same file = same hash."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Date,Close\n2024-01-01,100\n2024-01-02,101\n")
            path = f.name
        try:
            h1 = compute_price_hash(path)
            h2 = compute_price_hash(path)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 16)
        finally:
            os.unlink(path)

    def test_price_hash_changes(self):
        """Different data = different hash."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Date,Close\n2024-01-01,100\n")
            path = f.name
        try:
            h1 = compute_price_hash(path)
            with open(path, "w") as f:
                f.write("Date,Close\n2024-01-01,100\n2024-01-02,105\n")
            h2 = compute_price_hash(path)
            self.assertNotEqual(h1, h2)
        finally:
            os.unlink(path)

    def test_tune_hash_chains(self):
        """tune_hash depends on price_hash."""
        params = {"q": 1e-5, "c": 1.0}
        h1 = compute_tune_hash("price_aaa", params)
        h2 = compute_tune_hash("price_bbb", params)
        self.assertNotEqual(h1, h2)

    def test_signal_hash_chains(self):
        """signal_hash depends on tune_hash."""
        signal_data = {"forecasts": [1.0, 2.0]}
        h1 = compute_signal_hash("tune_aaa", signal_data)
        h2 = compute_signal_hash("tune_bbb", signal_data)
        self.assertNotEqual(h1, h2)

    def test_consistent_chain_green(self):
        """Consistent hashes -> green status."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Date,Close\n2024-01-01,100\n2024-01-02,101\n")
            path = f.name
        try:
            price_hash = compute_price_hash(path)
            tune_cache = {
                "price_data_hash": price_hash,
                "tune_hash": "tune_abc123",
            }
            result = verify_hash_chain(path, tune_cache)
            self.assertEqual(result["status"], "green")
            self.assertTrue(result["prices_consistent"])
        finally:
            os.unlink(path)

    def test_broken_chain_red(self):
        """Stale price hash -> red status."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Date,Close\n2024-01-01,100\n")
            path = f.name
        try:
            tune_cache = {
                "price_data_hash": "old_stale_hash",
                "tune_hash": "tune_abc",
            }
            result = verify_hash_chain(path, tune_cache)
            self.assertEqual(result["status"], "red")
            self.assertFalse(result["prices_consistent"])
        finally:
            os.unlink(path)

    def test_stale_signals_amber(self):
        """Signals with wrong tune_hash -> amber."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Date,Close\n2024-01-01,100\n")
            path = f.name
        try:
            price_hash = compute_price_hash(path)
            tune_cache = {
                "price_data_hash": price_hash,
                "tune_hash": "tune_new",
            }
            signal_cache = {
                "tune_hash": "tune_old",
            }
            result = verify_hash_chain(path, tune_cache, signal_cache)
            self.assertEqual(result["status"], "amber")
            self.assertFalse(result["signals_consistent"])
        finally:
            os.unlink(path)

    def test_missing_file_red(self):
        """Missing file -> red."""
        result = verify_hash_chain("/tmp/nonexistent_xyz.csv", {})
        self.assertEqual(result["status"], "red")


if __name__ == "__main__":
    unittest.main(verbosity=2)
