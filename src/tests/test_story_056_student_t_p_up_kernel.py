import os
import sys
import unittest

import numpy as np
from scipy.stats import t as student_t_dist


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class StudentTPUpKernelTest(unittest.TestCase):
    def test_p_up_kernel_matches_scipy(self):
        try:
            from models.numba_wrappers import run_student_t_p_up_array
        except ImportError as exc:
            self.skipTest(f"Numba wrappers unavailable: {exc}")

        mu_pred = np.array([-0.04, -0.01, 0.0, 0.015, 0.05], dtype=np.float64)
        variance = np.array([0.004, 0.001, 0.0005, 0.002, 0.006], dtype=np.float64)

        for nu in (3.0, 4.0, 8.0, 20.0):
            actual = run_student_t_p_up_array(mu_pred, variance, nu)
            scale = np.sqrt(variance * ((nu - 2.0) / nu))
            expected = 1.0 - student_t_dist.cdf((0.0 - mu_pred) / scale, df=nu)
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)

    def test_fast_helper_uses_valid_fallback_policy(self):
        from models.phi_student_t_improved import _fast_t_p_up

        mu_pred = np.array([-0.02, 0.0, 0.02], dtype=np.float64)
        variance = np.array([0.001, 0.001, 0.001], dtype=np.float64)
        p_up = _fast_t_p_up(mu_pred, variance, 8.0)

        self.assertEqual(p_up.shape, mu_pred.shape)
        self.assertTrue(np.all(np.isfinite(p_up)))
        self.assertTrue(np.all((p_up >= 0.0) & (p_up <= 1.0)))
        self.assertLess(p_up[0], 0.5)
        self.assertAlmostEqual(float(p_up[1]), 0.5, places=12)
        self.assertGreater(p_up[2], 0.5)


if __name__ == "__main__":
    unittest.main()
