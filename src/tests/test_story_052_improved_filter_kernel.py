import os
import sys
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.phi_student_t_improved import PhiStudentTDriftModel


class ImprovedStudentTFilterKernelParityTest(unittest.TestCase):
    def test_improved_filter_kernel_matches_python_reference(self):
        rng = np.random.default_rng(52)
        n = 320
        base = rng.standard_t(df=5, size=n) * 0.012
        returns = base + 0.0004 * np.sin(np.linspace(0.0, 9.0, n))
        vol = 0.012 + 0.004 * np.abs(np.sin(np.linspace(0.0, 12.0, n)))
        vov = np.abs(np.gradient(np.log(vol)))
        exo = 0.0002 * np.cos(np.linspace(0.0, 5.0, n))

        old = os.environ.get("PHI_STUDENT_T_IMPROVED_DISABLE_FILTER_KERNEL")
        os.environ["PHI_STUDENT_T_IMPROVED_DISABLE_FILTER_KERNEL"] = "1"
        ref = PhiStudentTDriftModel._filter_phi_core(
            returns, vol, 2.5e-6, 1.35, 0.42, 8.0,
            exogenous_input=exo,
            robust_wt=True,
            online_scale_adapt=True,
            gamma_vov=0.3,
            vov_rolling=vov,
        )
        if old is None:
            os.environ.pop("PHI_STUDENT_T_IMPROVED_DISABLE_FILTER_KERNEL", None)
        else:
            os.environ["PHI_STUDENT_T_IMPROVED_DISABLE_FILTER_KERNEL"] = old

        got = PhiStudentTDriftModel._filter_phi_core(
            returns, vol, 2.5e-6, 1.35, 0.42, 8.0,
            exogenous_input=exo,
            robust_wt=True,
            online_scale_adapt=True,
            gamma_vov=0.3,
            vov_rolling=vov,
        )

        for ref_arr, got_arr in zip(ref[:4], got[:4]):
            np.testing.assert_allclose(got_arr, ref_arr, rtol=2e-12, atol=2e-14)
        self.assertAlmostEqual(got[4], ref[4], places=10)


if __name__ == "__main__":
    unittest.main()
