import unittest
import sys
from pathlib import Path

import numpy as np
from scipy import stats

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calibration.isotonic_recalibration import _fast_ks_uniform


class IsotonicFastKsTest(unittest.TestCase):
    def test_fast_ks_matches_scipy_statistic_and_decision_scale(self):
        rng = np.random.default_rng(20260429)
        samples = [
            rng.uniform(size=500),
            np.linspace(0.02, 0.98, 500) ** 1.4,
            np.concatenate([rng.beta(0.8, 1.3, size=300), rng.uniform(size=200)]),
        ]

        for pit in samples:
            fast_stat, fast_p = _fast_ks_uniform(pit)
            scipy_res = stats.kstest(pit, "uniform")

            self.assertAlmostEqual(fast_stat, float(scipy_res.statistic), places=12)
            self.assertGreaterEqual(fast_p, 0.0)
            self.assertLessEqual(fast_p, 1.0)
            self.assertLess(abs(fast_p - float(scipy_res.pvalue)), 0.03)


if __name__ == "__main__":
    unittest.main()
