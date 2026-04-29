import unittest
import os
import sys

import numpy as np
from scipy.special import gammaln
from scipy.stats import t as student_t_dist

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
    run_phi_student_t_train_state_only,
    run_phi_student_t_improved_cv_test_fold,
    run_phi_student_t_improved_train_state_only,
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

        start = 0
        end = 180
        mu_f, P_f, _, _, _ = PhiStudentTDriftModel._filter_phi_core(
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
        mu_last, P_last = run_phi_student_t_train_state_only(
            returns,
            vol_sq,
            q,
            c,
            phi,
            nu,
            start,
            end,
            gamma_vov=0.5,
            vov_rolling=vov,
            robust_wt=True,
            online_scale_adapt=True,
        )

        self.assertAlmostEqual(mu_last, float(mu_f[-1]), places=13)
        self.assertAlmostEqual(P_last, float(P_f[-1]), places=13)

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
        scale_factor = (nu - 2.0) / nu

        start = 0
        end = 170
        mu_f, P_f, _, _, _ = ImprovedPhiStudentTDriftModel._filter_phi_core(
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
        mu_last, P_last = run_phi_student_t_improved_train_state_only(
            returns,
            vol_sq,
            q,
            c,
            phi,
            nu,
            scale_factor,
            start,
            end,
            gamma_vov=0.4,
            vov_rolling=vov,
            online_scale_adapt=True,
        )

        self.assertAlmostEqual(mu_last, float(mu_f[-1]), places=13)
        self.assertAlmostEqual(P_last, float(P_f[-1]), places=13)

    def test_improved_cv_fold_kernel_matches_python_loop(self):
        rng = np.random.default_rng(13)
        returns = np.ascontiguousarray(rng.standard_t(5, size=210) * 0.011, dtype=np.float64)
        vol = np.ascontiguousarray(
            np.maximum(0.005, 0.012 + 0.0025 * rng.standard_normal(210)),
            dtype=np.float64,
        )
        vol_sq = np.ascontiguousarray(vol * vol, dtype=np.float64)
        vov = ImprovedPhiStudentTDriftModel._precompute_vov(vol)

        q = 2.1e-6
        c = 1.35
        phi = 0.48
        nu = 6.0
        log_norm = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi)
        )
        neg_exp = -0.5 * (nu + 1.0)
        inv_nu = 1.0 / nu
        scale_factor = (nu - 2.0) / nu

        mu_last, P_last = run_phi_student_t_improved_train_state_only(
            returns,
            vol_sq,
            q,
            c,
            phi,
            nu,
            scale_factor,
            0,
            135,
            gamma_vov=0.3,
            vov_rolling=vov,
            online_scale_adapt=True,
        )
        (
            ll_fast,
            n_fast,
            z2_count_fast,
            z2_fast,
            pit_count_fast,
            pit_sum_fast,
            pit2_sum_fast,
            crps_count_fast,
            crps_sum_fast,
            sign_count_fast,
            sign_brier_fast,
        ) = run_phi_student_t_improved_cv_test_fold(
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
            mu_last,
            P_last,
            135,
            190,
            gamma_vov=0.3,
            vov_rolling=vov,
            online_scale_adapt=True,
            p_floor=1e-12,
            p_cap=1.0,
            z2_cap=50.0,
        )

        mu_p = mu_last
        P_p = P_last
        c_adj = 1.0
        chi2_tgt = nu / (nu - 2.0)
        ewm_z2 = chi2_tgt
        osa_strength = min(1.0, (chi2_tgt - 1.0) / 0.5)
        ll_ref = 0.0
        z2_ref = 0.0
        z2_count_ref = 0
        sign_brier_ref = 0.0
        sign_count_ref = 0
        pit_count_ref = 0
        pit_sum_ref = 0.0
        pit2_sum_ref = 0.0
        phi_sq = phi * phi
        chi2_lam = ImprovedPhiStudentTDriftModel._online_scale_lambda(nu)
        for t in range(135, 190):
            mu_pred = phi * mu_p
            P_pred = max(phi_sq * P_p + q, 1e-12)
            R_t = max(c * c_adj * vol_sq[t], 1e-20)
            R_t *= max(0.05, 1.0 + 0.3 * vov[t])
            S = max(P_pred + R_t, 1e-20)
            scale = max(np.sqrt(S * scale_factor), 1e-10)
            inn = returns[t] - mu_pred
            z = inn / scale
            ll_ref += log_norm - np.log(scale) + neg_exp * np.log1p((z * z) * inv_nu)
            p_obs = float(student_t_dist.cdf(z, df=nu))
            p_obs = float(np.clip(p_obs, 1e-10, 1.0 - 1e-10))
            pit_sum_ref += p_obs
            pit2_sum_ref += p_obs * p_obs
            pit_count_ref += 1
            p_up = 1.0 - float(student_t_dist.cdf((0.0 - mu_pred) / scale, df=nu))
            y_up = 1.0 if returns[t] > 0.0 else 0.0
            sign_brier_ref += (p_up - y_up) ** 2
            sign_count_ref += 1
            z_sq_s = (inn * inn) / S
            z2_ref += min(z_sq_s, 50.0)
            z2_count_ref += 1
            w_t = float(np.clip((nu + 1.0) / (nu + z_sq_s), 0.05, 20.0))
            w_t = float(np.clip(1.0 + 0.95 * (w_t - 1.0), 0.08, 8.0))
            R_eff = R_t / max(w_t, 1e-8)
            S_eff = max(P_pred + R_eff, 1e-20)
            K = P_pred / S_eff
            mu_p = mu_pred + K * inn
            P_p = (1.0 - K) * (1.0 - K) * P_pred + K * K * R_eff
            P_p = max(1e-12, min(P_p, 1.0))
            z2w = min(z * z, chi2_tgt * 50.0)
            ewm_z2 = chi2_lam * ewm_z2 + (1.0 - chi2_lam) * z2w
            ratio = float(np.clip(ewm_z2 / chi2_tgt, 0.35, 2.85))
            dev = ratio - 1.0 if ratio >= 1.0 else 1.0 - ratio
            if dev < 0.04:
                c_adj = 1.0
            else:
                c_adj = 1.0 + osa_strength * (np.sqrt(ratio) - 1.0)
                c_adj = float(np.clip(c_adj, 0.4, 2.5))

        self.assertAlmostEqual(ll_fast, ll_ref, places=11)
        self.assertEqual(n_fast, 55)
        self.assertEqual(z2_count_fast, z2_count_ref)
        self.assertAlmostEqual(z2_fast, z2_ref, places=11)
        self.assertEqual(pit_count_fast, pit_count_ref)
        self.assertAlmostEqual(pit_sum_fast, pit_sum_ref, places=11)
        self.assertAlmostEqual(pit2_sum_fast, pit2_sum_ref, places=11)
        self.assertEqual(crps_count_fast, pit_count_ref)
        self.assertTrue(np.isfinite(crps_sum_fast))
        self.assertEqual(sign_count_fast, sign_count_ref)
        self.assertAlmostEqual(sign_brier_fast, sign_brier_ref, places=11)

    def test_canonical_cv_fold_kernel_matches_python_joseph_osa_loop(self):
        rng = np.random.default_rng(29)
        returns = np.ascontiguousarray(rng.standard_t(4, size=220) * 0.013, dtype=np.float64)
        vol = np.ascontiguousarray(
            np.maximum(0.005, 0.012 + 0.002 * rng.standard_normal(220)),
            dtype=np.float64,
        )
        vol_sq = np.ascontiguousarray(vol * vol, dtype=np.float64)
        vov = PhiStudentTDriftModel._precompute_vov(vol)

        q = 2.4e-6
        c = 1.45
        phi = 0.51
        nu = 5.0
        nu_scale = (nu - 2.0) / nu
        log_norm = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi)
        )
        neg_exp = -0.5 * (nu + 1.0)
        inv_nu = 1.0 / nu

        mu_last, P_last = run_phi_student_t_train_state_only(
            returns,
            vol_sq,
            q,
            c,
            phi,
            nu,
            0,
            150,
            gamma_vov=0.25,
            vov_rolling=vov,
            robust_wt=True,
            online_scale_adapt=True,
        )
        ll_fast = run_phi_student_t_cv_test_fold(
            returns,
            vol_sq,
            q,
            c,
            phi,
            nu_scale,
            log_norm,
            neg_exp,
            inv_nu,
            mu_last,
            P_last,
            150,
            205,
            nu_val=nu,
            gamma_vov=0.25,
            vov_rolling=vov,
            online_scale_adapt=True,
        )

        mu_p = mu_last
        P_p = P_last
        c_adj = 1.0
        chi2_tgt = nu / (nu - 2.0)
        ewm_z2 = chi2_tgt
        osa_strength = min(1.0, (chi2_tgt - 1.0) / 0.5)
        ll_ref = 0.0
        phi_sq = phi * phi
        for t in range(150, 205):
            mu_p = phi * mu_p
            P_p = phi_sq * P_p + q
            R_t = c * c_adj * vol_sq[t]
            R_t *= max(0.05, min(20.0, 1.0 + 0.25 * vov[t]))
            S = max(P_p + R_t, 1e-12)
            inn = returns[t] - mu_p
            z_sq_s = (inn * inn) / S
            scale = np.sqrt(S * nu_scale)
            z = inn / scale
            ll_ref += log_norm - np.log(scale) + neg_exp * np.log1p(z * z * inv_nu)
            w_t = float(np.clip((nu + 1.0) / (nu + z_sq_s), 0.05, 20.0))
            R_eff = max(R_t / max(w_t, 1e-8), 1e-20)
            S_eff = max(P_p + R_eff, 1e-20)
            K = P_p / S_eff
            mu_p = mu_p + K * inn
            P_p = (1.0 - K) * (1.0 - K) * P_p + K * K * R_eff
            P_p = max(P_p, 1e-12)
            z2w = min(z * z, chi2_tgt * 50.0)
            ewm_z2 = 0.98 * ewm_z2 + 0.02 * z2w
            ratio = float(np.clip(ewm_z2 / chi2_tgt, 0.3, 3.0))
            dev = ratio - 1.0 if ratio >= 1.0 else 1.0 - ratio
            if ratio >= 1.0:
                dz_lo, dz_rng = 0.25, 0.25
            else:
                dz_lo, dz_rng = 0.05, 0.10
            if dev < dz_lo:
                c_adj = 1.0
            elif dev >= dz_lo + dz_rng:
                c_adj = 1.0 + osa_strength * (np.sqrt(ratio) - 1.0)
            else:
                s_frac = (dev - dz_lo) / dz_rng
                c_adj_raw = 1.0 + s_frac * (np.sqrt(ratio) - 1.0)
                c_adj = 1.0 + osa_strength * (c_adj_raw - 1.0)

        self.assertAlmostEqual(ll_fast, ll_ref, places=11)


if __name__ == "__main__":
    unittest.main()
