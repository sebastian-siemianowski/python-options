import unittest
import os
import sys

import numpy as np
from scipy.special import gammaln

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from models.phi_student_t import PhiStudentTDriftModel
from models.phi_student_t_improved import PhiStudentTDriftModel as ImprovedPhiStudentTDriftModel
from models.gaussian import GaussianDriftModel
from models.numba_wrappers import (
    is_numba_available,
    run_phi_gaussian_train_state,
    run_phi_student_t_cv_test_fold,
    run_phi_student_t_cv_test_fold_stats,
    run_phi_student_t_improved_train_state,
    run_phi_student_t_train_state,
)


@unittest.skipUnless(is_numba_available(), "Numba kernels unavailable")
class PhiStudentTTrainStateKernelTest(unittest.TestCase):
    def test_phi_gaussian_train_state_matches_filter_terminal_state(self):
        rng = np.random.default_rng(23)
        returns = np.ascontiguousarray(rng.normal(0.0003, 0.012, size=220), dtype=np.float64)
        vol = np.ascontiguousarray(
            np.maximum(0.004, 0.011 + 0.0015 * rng.standard_normal(220)),
            dtype=np.float64,
        )
        vol_sq = np.ascontiguousarray(vol * vol, dtype=np.float64)

        q = 1.5e-6
        c = 1.35
        phi = 0.62
        start = 0
        end = 175
        mu_f, P_f, ll_ref = GaussianDriftModel.filter_phi(
            returns[start:end], vol[start:end], q, c, phi
        )
        mu_last, P_last, ll_fast = run_phi_gaussian_train_state(
            returns, vol_sq, q, c, phi, start, end
        )

        self.assertAlmostEqual(mu_last, float(mu_f[-1]), places=13)
        self.assertAlmostEqual(P_last, float(P_f[-1]), places=13)
        self.assertAlmostEqual(ll_fast, float(ll_ref), places=10)

    def test_train_state_matches_enhanced_filter_terminal_state(self):
        rng = np.random.default_rng(42)
        returns = np.ascontiguousarray(rng.standard_t(5, size=260) * 0.012, dtype=np.float64)
        vol = np.ascontiguousarray(
            np.maximum(0.006, 0.012 + 0.002 * rng.standard_normal(260)),
            dtype=np.float64,
        )
        vol_sq = np.ascontiguousarray(vol * vol, dtype=np.float64)
        vov = PhiStudentTDriftModel._precompute_vov(vol)

        q = 2.5e-6
        c = 1.4
        phi = 0.58
        nu = 4.0
        log_norm = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi)
        )
        neg_exp = -((nu + 1.0) / 2.0)
        inv_nu = 1.0 / nu

        start = 0
        end = 180
        mu_f, P_f, _, _, ll_ref = PhiStudentTDriftModel._filter_phi_core(
            returns[start:end],
            vol[start:end],
            q,
            c,
            phi,
            nu,
            robust_wt=True,
            online_scale_adapt=True,
            gamma_vov=0.5,
            vov_rolling=vov[start:end],
        )
        mu_last, P_last, ll_fast = run_phi_student_t_train_state(
            returns,
            vol_sq,
            q,
            c,
            phi,
            nu,
            log_norm,
            neg_exp,
            inv_nu,
            start,
            end,
            gamma_vov=0.5,
            vov_rolling=vov,
            robust_wt=True,
            online_scale_adapt=True,
        )

        self.assertAlmostEqual(mu_last, float(mu_f[-1]), places=13)
        self.assertAlmostEqual(P_last, float(P_f[-1]), places=13)
        self.assertAlmostEqual(ll_fast, float(ll_ref), places=10)

    def test_cv_fold_stats_preserve_scalar_ll(self):
        rng = np.random.default_rng(99)
        returns = np.ascontiguousarray(rng.standard_t(7, size=180) * 0.009, dtype=np.float64)
        vol = np.ascontiguousarray(
            np.maximum(0.004, 0.010 + 0.001 * rng.standard_normal(180)),
            dtype=np.float64,
        )
        vol_sq = np.ascontiguousarray(vol * vol, dtype=np.float64)
        vov = PhiStudentTDriftModel._precompute_vov(vol)

        q = 1.2e-6
        c = 1.1
        phi = 0.35
        nu = 8.0
        log_norm = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi)
        )
        neg_exp = -((nu + 1.0) / 2.0)
        inv_nu = 1.0 / nu
        nu_scale = (nu - 2.0) / nu

        mu_last, P_last, _ = run_phi_student_t_train_state(
            returns,
            vol_sq,
            q,
            c,
            phi,
            nu,
            log_norm,
            neg_exp,
            inv_nu,
            0,
            110,
            gamma_vov=0.3,
            vov_rolling=vov,
            robust_wt=True,
            online_scale_adapt=True,
        )
        ll_scalar = run_phi_student_t_cv_test_fold(
            returns, vol_sq, q, c, phi, nu_scale,
            log_norm, neg_exp, inv_nu, mu_last, P_last, 110, 150,
            nu_val=nu, gamma_vov=0.3, vov_rolling=vov,
        )
        ll_stats, obs_count, z2_sum = run_phi_student_t_cv_test_fold_stats(
            returns, vol_sq, q, c, phi, nu_scale,
            log_norm, neg_exp, inv_nu, mu_last, P_last, 110, 150,
            nu_val=nu, gamma_vov=0.3, vov_rolling=vov,
        )

        self.assertAlmostEqual(ll_stats, ll_scalar, places=12)
        self.assertEqual(obs_count, 40)
        self.assertTrue(np.isfinite(z2_sum))
        self.assertGreater(z2_sum, 0.0)

    def test_improved_train_state_matches_improved_filter_terminal_state(self):
        rng = np.random.default_rng(7)
        returns = np.ascontiguousarray(rng.standard_t(6, size=240) * 0.010, dtype=np.float64)
        vol = np.ascontiguousarray(
            np.maximum(0.005, 0.011 + 0.002 * rng.standard_normal(240)),
            dtype=np.float64,
        )
        vol_sq = np.ascontiguousarray(vol * vol, dtype=np.float64)
        vov = ImprovedPhiStudentTDriftModel._precompute_vov(vol)

        q = 1.8e-6
        c = 1.2
        phi = 0.42
        nu = 8.0
        log_norm = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi)
        )
        neg_exp = -((nu + 1.0) / 2.0)
        inv_nu = 1.0 / nu
        scale_factor = (nu - 2.0) / nu

        start = 0
        end = 170
        mu_f, P_f, _, _, ll_ref = ImprovedPhiStudentTDriftModel._filter_phi_core(
            returns[start:end],
            vol[start:end],
            q,
            c,
            phi,
            nu,
            robust_wt=True,
            online_scale_adapt=True,
            gamma_vov=0.4,
            vov_rolling=vov[start:end],
        )
        mu_last, P_last, ll_fast = run_phi_student_t_improved_train_state(
            returns,
            vol_sq,
            q,
            c,
            phi,
            nu,
            log_norm,
            neg_exp,
            inv_nu,
            scale_factor,
            start,
            end,
            gamma_vov=0.4,
            vov_rolling=vov,
            online_scale_adapt=True,
        )

        self.assertAlmostEqual(mu_last, float(mu_f[-1]), places=13)
        self.assertAlmostEqual(P_last, float(P_f[-1]), places=13)
        self.assertAlmostEqual(ll_fast, float(ll_ref), places=10)


if __name__ == "__main__":
    unittest.main()
