import os
import sys
import unittest
import copy

import numpy as np

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.numba_wrappers import is_numba_available
from models.phi_student_t_unified import (
    UnifiedPhiStudentTModel,
    UnifiedStudentTConfig,
    compute_ms_process_noise_smooth,
)
from models.phi_student_t_unified_improved import (
    UnifiedPhiStudentTModel as ImprovedUnifiedPhiStudentTModel,
    UnifiedStudentTConfig as ImprovedUnifiedStudentTConfig,
    compute_ms_process_noise_smooth as improved_compute_ms_process_noise_smooth,
    precompute_vov as improved_precompute_vov,
)


@unittest.skipUnless(is_numba_available(), "Numba kernels unavailable")
class UnifiedStage5PrecomputeTest(unittest.TestCase):
    def test_unified_filter_precomputed_structural_arrays_match(self):
        rng = np.random.default_rng(301)
        returns = np.ascontiguousarray(rng.standard_t(6, size=260) * 0.010, dtype=np.float64)
        vol = np.ascontiguousarray(np.maximum(0.004, 0.011 + 0.002 * rng.standard_normal(260)), dtype=np.float64)
        cfg = UnifiedStudentTConfig(
            q=1.8e-6,
            c=1.25,
            phi=0.52,
            nu_base=6.0,
            alpha_asym=-0.08,
            gamma_vov=0.45,
            ms_sensitivity=2.4,
            q_stress_ratio=8.0,
            risk_premium_sensitivity=0.7,
            skew_score_sensitivity=0.02,
        )
        ref = UnifiedPhiStudentTModel.filter_phi_unified(returns, vol, cfg)

        q_t, p_stress = compute_ms_process_noise_smooth(
            vol, cfg.q_calm, cfg.q_stress, cfg.ms_sensitivity, cfg.ms_ewm_lambda
        )
        cfg_fast = copy.copy(cfg)
        cfg_fast._precomputed_q_t = q_t
        cfg_fast._precomputed_p_stress = p_stress
        cfg_fast._precomputed_vov_rolling = UnifiedPhiStudentTModel._precompute_vov(vol, cfg.vov_window)
        cfg_fast._precomputed_momentum = np.zeros(len(returns), dtype=np.float64)
        fast = UnifiedPhiStudentTModel.filter_phi_unified(returns, vol, cfg_fast)

        for got, expected in zip(fast[:4], ref[:4]):
            np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-13)
        self.assertAlmostEqual(float(fast[4]), float(ref[4]), places=10)

    def test_improved_unified_filter_precomputed_structural_arrays_match(self):
        rng = np.random.default_rng(302)
        returns = np.ascontiguousarray(rng.standard_t(7, size=240) * 0.011, dtype=np.float64)
        vol = np.ascontiguousarray(np.maximum(0.004, 0.012 + 0.002 * rng.standard_normal(240)), dtype=np.float64)
        cfg = ImprovedUnifiedStudentTConfig(
            q=2.2e-6,
            c=1.15,
            phi=0.48,
            nu_base=8.0,
            alpha_asym=0.06,
            gamma_vov=0.35,
            ms_sensitivity=2.2,
            q_stress_ratio=7.0,
            risk_premium_sensitivity=0.4,
            skew_score_sensitivity=0.015,
        )
        ref = ImprovedUnifiedPhiStudentTModel.filter_phi_unified(returns, vol, cfg)

        q_t, p_stress = improved_compute_ms_process_noise_smooth(
            vol, cfg.q_calm, cfg.q_stress, cfg.ms_sensitivity, cfg.ms_ewm_lambda
        )
        cfg_fast = copy.copy(cfg)
        cfg_fast._precomputed_q_t = q_t
        cfg_fast._precomputed_p_stress = p_stress
        cfg_fast._precomputed_vov_rolling = improved_precompute_vov(vol, cfg.vov_window)
        cfg_fast._precomputed_momentum = np.zeros(len(returns), dtype=np.float64)
        fast = ImprovedUnifiedPhiStudentTModel.filter_phi_unified(returns, vol, cfg_fast)

        for got, expected in zip(fast[:4], ref[:4]):
            np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-13)
        self.assertAlmostEqual(float(fast[4]), float(ref[4]), places=10)


if __name__ == "__main__":
    unittest.main()
