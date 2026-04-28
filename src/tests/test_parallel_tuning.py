"""
Test Story 3.2: Parallel Tuning Optimization.

Validates:
  1. Complexity estimation produces reasonable scores
  2. Sorting by complexity puts high-vol first
  3. Optimal worker count is valid
"""
import sys
import os
import subprocess
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from tuning.tune import (
    estimate_tuning_complexity,
    sort_assets_by_complexity,
    get_optimal_worker_count,
)


class TestParallelTuning(unittest.TestCase):
    """Tests for parallel tuning optimization."""

    def setUp(self):
        """Create temp directory with mock price CSVs."""
        self.tmpdir = tempfile.mkdtemp()
        
        # Simple calm asset
        lines = ["Date,Open,High,Low,Close,Volume\n"]
        for i in range(100):
            lines.append(f"2024-01-{i+1:03d},100.0,100.1,99.9,{100.0 + 0.01*i},1000\n")
        with open(os.path.join(self.tmpdir, "CALM_1d.csv"), 'w') as f:
            f.writelines(lines)
        
        # Volatile asset
        np.random.seed(42)
        lines = ["Date,Open,High,Low,Close,Volume\n"]
        price = 100.0
        for i in range(500):
            price *= np.exp(np.random.normal(0, 0.05))
            lines.append(f"2024-01-{i+1:03d},{price},{price*1.05},{price*0.95},{price},5000\n")
        with open(os.path.join(self.tmpdir, "VOLATILE_1d.csv"), 'w') as f:
            f.writelines(lines)

    def test_complexity_score_range(self):
        """Complexity score is positive."""
        score = estimate_tuning_complexity("CALM", self.tmpdir)
        self.assertGreater(score, 0)

    def test_volatile_more_complex(self):
        """Volatile asset has higher complexity score than calm."""
        score_calm = estimate_tuning_complexity("CALM", self.tmpdir)
        score_vol = estimate_tuning_complexity("VOLATILE", self.tmpdir)
        self.assertGreater(score_vol, score_calm)

    def test_missing_asset_default(self):
        """Missing asset gets default complexity."""
        score = estimate_tuning_complexity("NONEXISTENT", self.tmpdir)
        self.assertAlmostEqual(score, 50.0)

    def test_sort_puts_complex_first(self):
        """Sorting puts most complex (volatile) first."""
        sorted_assets = sort_assets_by_complexity(["CALM", "VOLATILE"], self.tmpdir)
        self.assertEqual(sorted_assets[0], "VOLATILE")

    def test_optimal_workers(self):
        """Worker count is positive and reasonable."""
        workers = get_optimal_worker_count()
        physical_cpus = os.cpu_count() or 2
        if sys.platform == "darwin":
            try:
                physical_cpus = int(
                    subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip()
                )
            except Exception:
                pass
        self.assertGreaterEqual(workers, 1)
        self.assertLessEqual(workers, max(1, physical_cpus - 1))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
