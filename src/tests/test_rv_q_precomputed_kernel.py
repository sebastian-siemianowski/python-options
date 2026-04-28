import os
import sys
import unittest

import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.numba_wrappers import is_numba_available, precompute_gamma_values
from models.numba_kernels import (
    rv_adaptive_q_gaussian_filter_kernel,
    rv_adaptive_q_gaussian_filter_precomputed_kernel,
    rv_adaptive_q_student_t_filter_kernel,
    rv_adaptive_q_student_t_filter_precomputed_kernel,
)
from models.rv_adaptive_q import _precompute_rv_q_vol_inputs


@unittest.skipUnless(is_numba_available(), "Numba kernels unavailable")
class RVQPrecomputedKernelTest(unittest.TestCase):
    def test_gaussian_precomputed_kernel_matches_baseline(self):
        rng = np.random.default_rng(101)
        returns = np.ascontiguousarray(rng.normal(0.0, 0.012, 260), dtype=np.float64)
        vol = np.ascontiguousarray(np.maximum(0.004, 0.011 + 0.002 * rng.standard_normal(260)), dtype=np.float64)
        vol_sq, delta = _precompute_rv_q_vol_inputs(vol)

        mu_a = np.zeros(len(returns), dtype=np.float64)
        P_a = np.zeros(len(returns), dtype=np.float64)
        q_a = np.zeros(len(returns), dtype=np.float64)
        mu_b = np.zeros(len(returns), dtype=np.float64)
        P_b = np.zeros(len(returns), dtype=np.float64)
        q_b = np.zeros(len(returns), dtype=np.float64)

        ll_a = rv_adaptive_q_gaussian_filter_kernel(
            returns, vol, 1.7e-6, 1.2, 1.35, 0.47, 1e-8, 1e-2, 1e-4, mu_a, P_a, q_a
        )
        ll_b = rv_adaptive_q_gaussian_filter_precomputed_kernel(
            returns, vol_sq, delta, 1.7e-6, 1.2, 1.35, 0.47, 1e-8, 1e-2, 1e-4, mu_b, P_b, q_b
        )

        self.assertAlmostEqual(float(ll_b), float(ll_a), places=10)
        np.testing.assert_allclose(mu_b, mu_a, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(P_b, P_a, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(q_b, q_a, rtol=0.0, atol=1e-14)

    def test_student_t_precomputed_kernel_matches_baseline(self):
        rng = np.random.default_rng(202)
        returns = np.ascontiguousarray(rng.standard_t(5, size=240) * 0.010, dtype=np.float64)
        vol = np.ascontiguousarray(np.maximum(0.004, 0.010 + 0.002 * rng.standard_normal(240)), dtype=np.float64)
        vol_sq, delta = _precompute_rv_q_vol_inputs(vol)
        nu = 6.0
        lg_half_nu, lg_half_nu_plus_half = precompute_gamma_values(nu)

        mu_a = np.zeros(len(returns), dtype=np.float64)
        P_a = np.zeros(len(returns), dtype=np.float64)
        q_a = np.zeros(len(returns), dtype=np.float64)
        mu_b = np.zeros(len(returns), dtype=np.float64)
        P_b = np.zeros(len(returns), dtype=np.float64)
        q_b = np.zeros(len(returns), dtype=np.float64)

        ll_a = rv_adaptive_q_student_t_filter_kernel(
            returns, vol, 2.1e-6, 0.8, 1.2, 0.41, nu,
            lg_half_nu, lg_half_nu_plus_half, 1e-8, 1e-2, 1e-4, mu_a, P_a, q_a
        )
        ll_b = rv_adaptive_q_student_t_filter_precomputed_kernel(
            returns, vol_sq, delta, 2.1e-6, 0.8, 1.2, 0.41, nu,
            lg_half_nu, lg_half_nu_plus_half, 1e-8, 1e-2, 1e-4, mu_b, P_b, q_b
        )

        self.assertAlmostEqual(float(ll_b), float(ll_a), places=10)
        np.testing.assert_allclose(mu_b, mu_a, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(P_b, P_a, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(q_b, q_a, rtol=0.0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
