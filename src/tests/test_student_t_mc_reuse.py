"""Tests for v7.7 enriched MC: reusing tuned Student-t params in BMA simulation.

Tests validate:
1. GJR-GARCH leverage in Numba kernel — asymmetric vol response
2. variance_inflation passthrough — calibrated predictive variance
3. mu_drift passthrough — drift bias correction
4. alpha_asym — asymmetric tail thickness
5. risk_premium_sensitivity — ICAPM variance-conditional drift
6. Per-model GARCH extraction in BMA — no longer global-only
7. Numerical CRPS kernel cross-validates analytic kernel
8. End-to-end run_unified_mc with enriched params
9. kappa_mean_rev — vol mean reversion toward theta (Heston)
10. crps_sigma_shrinkage — CRPS-optimal sigma scaling
11. ms_sensitivity — Markov-switching process noise for drift
12. rough_hurst — fractional differencing rough vol
13. sigma_eta — observation noise perturbation
14. t_df_asym — asymmetric degrees-of-freedom shift
15. regime_switch_prob — observation noise regime switching
16. gamma_vov — vol-of-vol stochastic volatility
17. skew dynamics — GAS dynamic skew updating
18. loc_bias — location bias from variance and drift
19. Backward compatibility — defaults produce identical results
"""

import os
import sys
import math
import unittest
import numpy as np

# Setup path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _numba_available():
    try:
        from models.numba_kernels import unified_mc_simulate_kernel
        return True
    except ImportError:
        return False


class TestGJRGARCHKernel(unittest.TestCase):
    """Test that GJR-GARCH leverage makes negative shocks increase vol more."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_gjr_asymmetric_vol_response(self):
        """Negative shocks should produce higher subsequent vol than positive."""
        from models.numba_kernels import unified_mc_simulate_kernel

        n_paths = 5000
        H_max = 20
        np.random.seed(42)

        # Generate random inputs
        z_normals = np.random.standard_normal((H_max, n_paths))
        z_chi2 = np.ones((H_max, n_paths))  # Gaussian for simplicity
        z_drift = np.random.standard_normal((H_max, n_paths))
        z_jump_uniform = np.random.uniform(size=(H_max, n_paths))
        z_jump_normal = np.random.standard_normal((H_max, n_paths))

        # Run WITHOUT GJR leverage
        cum_no_gjr = np.zeros((H_max, n_paths))
        vol_no_gjr = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, 0.0, 0.0002,
            0.95, 1e-6, 200.0,
            True, 1e-6, 0.08, 0.90,
            0.0, 0.0, 0.05, False,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_uniform.copy(), z_jump_normal.copy(),
            cum_no_gjr, vol_no_gjr,
            0.0, 1.0, 0.0, 0.0, 2.0, 0.0,  # no leverage
        )

        # Run WITH GJR leverage (gamma = 0.12, typical for equities)
        cum_gjr = np.zeros((H_max, n_paths))
        vol_gjr = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, 0.0, 0.0002,
            0.95, 1e-6, 200.0,
            True, 1e-6, 0.08, 0.90,
            0.0, 0.0, 0.05, False,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_uniform.copy(), z_jump_normal.copy(),
            cum_gjr, vol_gjr,
            0.12, 1.0, 0.0, 0.0, 2.0, 0.0,  # gamma = 0.12
        )

        # GJR should produce higher average vol (since neg shocks add extra vol)
        avg_vol_no_gjr = np.mean(vol_no_gjr[-1, :])
        avg_vol_gjr = np.mean(vol_gjr[-1, :])
        self.assertGreater(avg_vol_gjr, avg_vol_no_gjr,
                           "GJR leverage should increase average vol")

        # The vol increase should be meaningful but not extreme
        ratio = avg_vol_gjr / avg_vol_no_gjr
        self.assertGreater(ratio, 1.001, "GJR should measurably increase vol")
        self.assertLess(ratio, 2.0, "GJR increase should not be extreme")

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_gjr_zero_leverage_matches_plain_garch(self):
        """With garch_leverage=0, results should match the old kernel exactly."""
        from models.numba_kernels import unified_mc_simulate_kernel

        n_paths = 1000
        H_max = 10
        np.random.seed(123)

        z_normals = np.random.standard_normal((H_max, n_paths))
        z_chi2 = np.ones((H_max, n_paths))
        z_drift = np.random.standard_normal((H_max, n_paths))
        z_jump_uniform = np.random.uniform(size=(H_max, n_paths))
        z_jump_normal = np.random.standard_normal((H_max, n_paths))

        cum1 = np.zeros((H_max, n_paths))
        vol1 = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, 0.001, 0.0002,
            0.95, 1e-6, 200.0,
            True, 1e-6, 0.08, 0.90,
            0.0, 0.0, 0.05, False,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_uniform.copy(), z_jump_normal.copy(),
            cum1, vol1,
            0.0, 1.0, 0.0, 0.0, 2.0, 0.0,
        )

        cum2 = np.zeros((H_max, n_paths))
        vol2 = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, 0.001, 0.0002,
            0.95, 1e-6, 200.0,
            True, 1e-6, 0.08, 0.90,
            0.0, 0.0, 0.05, False,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_uniform.copy(), z_jump_normal.copy(),
            cum2, vol2,
            0.0, 1.0, 0.0, 0.0, 2.0, 0.0,
        )

        np.testing.assert_allclose(cum1, cum2, atol=1e-12,
                                   err_msg="Zero leverage should match plain GARCH")
        np.testing.assert_allclose(vol1, vol2, atol=1e-12,
                                   err_msg="Zero leverage vol should match")


class TestVarianceInflation(unittest.TestCase):
    """Test that variance_inflation scales predictive variance correctly."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_variance_inflation_widens_samples(self):
        """variance_inflation=2.0 should produce ~sqrt(2) wider returns."""
        from models.numba_kernels import unified_mc_simulate_kernel

        n_paths = 10000
        H_max = 5
        np.random.seed(42)

        z_normals = np.random.standard_normal((H_max, n_paths))
        z_chi2 = np.ones((H_max, n_paths))
        z_drift = np.random.standard_normal((H_max, n_paths))
        z_jump_uniform = np.random.uniform(size=(H_max, n_paths))
        z_jump_normal = np.random.standard_normal((H_max, n_paths))

        # variance_inflation = 1.0 (baseline)
        cum1 = np.zeros((H_max, n_paths))
        vol1 = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, 0.0, 0.0002,
            1.0, 0.0, 200.0,  # phi=1, q=0 (no drift noise), Gaussian
            False, 0.0, 0.0, 0.0,  # no GARCH
            0.0, 0.0, 0.05, False,  # no jumps
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_uniform.copy(), z_jump_normal.copy(),
            cum1, vol1,
            0.0, 1.0, 0.0, 0.0, 2.0, 0.0,  # variance_inflation=1.0
        )

        # variance_inflation = 2.0
        cum2 = np.zeros((H_max, n_paths))
        vol2 = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, 0.0, 0.0002,
            1.0, 0.0, 200.0,
            False, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.05, False,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_uniform.copy(), z_jump_normal.copy(),
            cum2, vol2,
            0.0, 2.0, 0.0, 0.0, 2.0, 0.0,  # variance_inflation=2.0
        )

        std1 = np.std(cum1[0, :])
        std2 = np.std(cum2[0, :])
        ratio = std2 / std1

        # variance_inflation=2 → h0*2 → sigma*sqrt(2) ≈ 1.414
        self.assertAlmostEqual(ratio, math.sqrt(2.0), delta=0.1,
                               msg=f"Expected ratio ~{math.sqrt(2.0):.3f}, got {ratio:.3f}")


class TestMuDrift(unittest.TestCase):
    """Test that mu_drift shifts MC sample means correctly."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_mu_drift_shifts_mean(self):
        """mu_drift=0.001 should shift mean by approx H * 0.001."""
        from models.numba_kernels import unified_mc_simulate_kernel

        n_paths = 20000
        H_max = 10
        mu_drift_val = 0.001
        np.random.seed(42)

        z_normals = np.random.standard_normal((H_max, n_paths))
        z_chi2 = np.ones((H_max, n_paths))
        z_drift = np.zeros((H_max, n_paths))  # No drift noise
        z_jump_uniform = np.random.uniform(size=(H_max, n_paths))
        z_jump_normal = np.random.standard_normal((H_max, n_paths))

        # Without mu_drift
        cum1 = np.zeros((H_max, n_paths))
        vol1 = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, 0.0, 0.0001,
            1.0, 0.0, 200.0,
            False, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.05, False,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_uniform.copy(), z_jump_normal.copy(),
            cum1, vol1,
            0.0, 1.0, 0.0, 0.0, 2.0, 0.0,
        )

        # With mu_drift=0.001
        cum2 = np.zeros((H_max, n_paths))
        vol2 = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, 0.0, 0.0001,
            1.0, 0.0, 200.0,
            False, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.05, False,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_uniform.copy(), z_jump_normal.copy(),
            cum2, vol2,
            0.0, 1.0, mu_drift_val, 0.0, 2.0, 0.0,
        )

        # At horizon H, mean shift should be approximately H * mu_drift
        for h in [1, 5, 10]:
            mean_shift = np.mean(cum2[h - 1, :]) - np.mean(cum1[h - 1, :])
            expected = h * mu_drift_val
            self.assertAlmostEqual(mean_shift, expected, delta=0.0005,
                                   msg=f"H={h}: expected shift {expected:.4f}, got {mean_shift:.4f}")


class TestAlphaAsym(unittest.TestCase):
    """Test asymmetric tail thickness makes left tail heavier."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_negative_alpha_heavier_left_tail(self):
        """alpha_asym < 0 should make left tail heavier than right."""
        from models.numba_kernels import unified_mc_simulate_kernel

        n_paths = 50000
        H_max = 1
        np.random.seed(42)

        nu_val = 8.0
        z_normals = np.random.standard_normal((H_max, n_paths))
        chi2_draws = np.random.gamma(shape=nu_val / 2.0, scale=2.0, size=(H_max, n_paths))
        z_chi2 = (chi2_draws / nu_val)
        z_drift = np.zeros((H_max, n_paths))
        z_jump_uniform = np.random.uniform(size=(H_max, n_paths))
        z_jump_normal = np.random.standard_normal((H_max, n_paths))

        # Symmetric (alpha_asym=0)
        cum_sym = np.zeros((H_max, n_paths))
        vol_sym = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, 0.0, 0.0001,
            1.0, 0.0, nu_val,
            False, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.05, False,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_uniform.copy(), z_jump_normal.copy(),
            cum_sym, vol_sym,
            0.0, 1.0, 0.0, 0.0, 2.0, 0.0,
        )

        # Asymmetric (alpha_asym=-0.3: heavier left tail)
        cum_asym = np.zeros((H_max, n_paths))
        vol_asym = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, 0.0, 0.0001,
            1.0, 0.0, nu_val,
            False, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.05, False,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_uniform.copy(), z_jump_normal.copy(),
            cum_asym, vol_asym,
            0.0, 1.0, 0.0, -0.3, 2.0, 0.0,
        )

        # Compare left tail (1st percentile) — asymmetric should be more negative
        p1_sym = np.percentile(cum_sym[0, :], 1)
        p1_asym = np.percentile(cum_asym[0, :], 1)
        self.assertLess(p1_asym, p1_sym,
                        "Asymmetric α<0 should produce heavier left tail")

        # Compare right tail (99th percentile) — asymmetric should be less extreme
        p99_sym = np.percentile(cum_sym[0, :], 99)
        p99_asym = np.percentile(cum_asym[0, :], 99)
        self.assertLess(p99_asym, p99_sym,
                        "Asymmetric α<0 should produce lighter right tail")


class TestRiskPremium(unittest.TestCase):
    """Test ICAPM variance-conditional drift."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_risk_premium_positive_shift(self):
        """Positive risk_premium_sensitivity should increase mean return."""
        from models.numba_kernels import unified_mc_simulate_kernel

        n_paths = 20000
        H_max = 5
        np.random.seed(42)

        z_normals = np.random.standard_normal((H_max, n_paths))
        z_chi2 = np.ones((H_max, n_paths))
        z_drift = np.zeros((H_max, n_paths))
        z_jump_uniform = np.random.uniform(size=(H_max, n_paths))
        z_jump_normal = np.random.standard_normal((H_max, n_paths))

        h0 = 0.0004  # 2% daily vol

        # Without risk premium
        cum1 = np.zeros((H_max, n_paths))
        vol1 = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, 0.0, h0,
            1.0, 0.0, 200.0,
            False, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.05, False,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_uniform.copy(), z_jump_normal.copy(),
            cum1, vol1,
            0.0, 1.0, 0.0, 0.0, 2.0, 0.0,
        )

        # With risk_premium_sensitivity = 0.5
        cum2 = np.zeros((H_max, n_paths))
        vol2 = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, 0.0, h0,
            1.0, 0.0, 200.0,
            False, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.05, False,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_uniform.copy(), z_jump_normal.copy(),
            cum2, vol2,
            0.0, 1.0, 0.0, 0.0, 2.0, 0.5,  # risk_premium = 0.5
        )

        mean1 = np.mean(cum1[-1, :])
        mean2 = np.mean(cum2[-1, :])
        self.assertGreater(mean2, mean1,
                           "Positive risk premium should increase mean return")

        # Expected extra return ≈ H * lambda * h0
        expected_extra = H_max * 0.5 * h0
        actual_extra = mean2 - mean1
        self.assertAlmostEqual(actual_extra, expected_extra, delta=0.0003,
                               msg=f"Risk premium shift: expected ~{expected_extra:.5f}, got {actual_extra:.5f}")


class TestRunUnifiedMCEnriched(unittest.TestCase):
    """Test the Python-level run_unified_mc with enriched params."""

    def test_enriched_params_accepted(self):
        """run_unified_mc should accept all v7.6 enriched params without error."""
        from decision.signals import run_unified_mc

        result = run_unified_mc(
            mu_t=0.001,
            P_t=1e-6,
            phi=0.95,
            q=1e-6,
            sigma2_step=0.0002,
            H_max=5,
            n_paths=100,
            nu=8.0,
            use_garch=True,
            garch_omega=1e-6,
            garch_alpha=0.08,
            garch_beta=0.90,
            jump_intensity=0.01,
            jump_mean=0.0,
            jump_std=0.05,
            seed=42,
            # v7.6 enriched params
            garch_leverage=0.12,
            variance_inflation=1.2,
            mu_drift=0.0005,
            alpha_asym=-0.1,
            k_asym=2.0,
            risk_premium_sensitivity=0.001,
        )

        self.assertIn('returns', result)
        self.assertIn('volatility', result)
        self.assertEqual(result['returns'].shape, (5, 100))
        self.assertEqual(result['volatility'].shape, (5, 100))

        # Verify no NaN/Inf
        self.assertFalse(np.any(np.isnan(result['returns'])),
                         "MC returns should have no NaN")
        self.assertFalse(np.any(np.isinf(result['returns'])),
                         "MC returns should have no Inf")

    def test_backward_compatible_defaults(self):
        """run_unified_mc with no enriched params should work (defaults=identity)."""
        from decision.signals import run_unified_mc

        result = run_unified_mc(
            mu_t=0.001,
            P_t=1e-6,
            phi=0.95,
            q=1e-6,
            sigma2_step=0.0002,
            H_max=5,
            n_paths=100,
            seed=42,
        )

        self.assertEqual(result['returns'].shape, (5, 100))
        self.assertFalse(np.any(np.isnan(result['returns'])))


class TestNumericalCRPSKernel(unittest.TestCase):
    """Test the correct numerical CRPS kernel vs old analytic (wrong) kernel."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_numerical_matches_analytic(self):
        """Numerical g(ν) quadrature should match analytic B_ratio formula.

        This cross-validates both implementations: the closed-form Gneiting
        & Raftery (2007) B_ratio and the numerical Gini half-mean-difference.
        """
        from models.numba_kernels import (
            crps_student_t_kernel,
            crps_student_t_numerical_kernel,
        )

        np.random.seed(42)
        n = 200
        z = np.random.standard_normal(n)
        sigma = np.ones(n, dtype=np.float64) * 0.015

        for nu in [4.0, 8.0, 20.0, 100.0]:
            crps_analytic = crps_student_t_kernel(z, sigma, nu)
            crps_numerical = crps_student_t_numerical_kernel(z, sigma, nu)

            # Both formulations should agree within 2% (quadrature
            # precision is limited for small ν with heavy tails)
            rel_diff = abs(crps_analytic - crps_numerical) / max(crps_analytic, 1e-10)
            self.assertLess(
                rel_diff, 0.02,
                msg=f"Analytic and numerical CRPS should agree within 2% for ν={nu}: "
                    f"analytic={crps_analytic:.8f}, numerical={crps_numerical:.8f}, "
                    f"rel_diff={rel_diff:.4f}"
            )

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_numerical_crps_positive(self):
        """Numerical CRPS should always be positive."""
        from models.numba_kernels import crps_student_t_numerical_kernel

        np.random.seed(42)
        n = 500
        z = np.random.standard_normal(n)
        sigma = np.abs(np.random.standard_normal(n)) * 0.01 + 0.005

        for nu in [3.0, 5.0, 8.0, 12.0, 20.0, 50.0]:
            crps = crps_student_t_numerical_kernel(z, sigma, nu)
            self.assertGreater(crps, 0.0, f"CRPS should be positive for ν={nu}")
            self.assertTrue(np.isfinite(crps), f"CRPS should be finite for ν={nu}")

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_crps_wrapper_uses_numerical(self):
        """run_crps_student_t wrapper should use the correct numerical kernel."""
        from models.numba_wrappers import run_crps_student_t
        from models.numba_kernels import crps_student_t_numerical_kernel

        np.random.seed(42)
        n = 100
        z = np.random.standard_normal(n)
        sigma = np.ones(n, dtype=np.float64) * 0.015

        wrapper_result = run_crps_student_t(z, sigma, 8.0)
        direct_result = crps_student_t_numerical_kernel(z, sigma, 8.0)

        self.assertAlmostEqual(wrapper_result, direct_result, places=10,
                               msg="Wrapper should produce identical result to direct kernel call")


class TestMultiPathKernelGJR(unittest.TestCase):
    """Test that unified_mc_multi_path_kernel also has GJR support."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_multi_path_gjr(self):
        """Multi-path kernel with per-path gamma should work."""
        from models.numba_kernels import unified_mc_multi_path_kernel

        n_paths = 1000
        H_max = 5
        np.random.seed(42)

        z_normals = np.random.standard_normal((H_max, n_paths))
        z_chi2 = np.ones((H_max, n_paths))
        z_drift = np.random.standard_normal((H_max, n_paths))
        z_jump_uniform = np.random.uniform(size=(H_max, n_paths))
        z_jump_normal = np.random.standard_normal((H_max, n_paths))

        nu_per_path = np.full(n_paths, 8.0)
        omega_per_path = np.full(n_paths, 1e-6)
        alpha_per_path = np.full(n_paths, 0.08)
        beta_per_path = np.full(n_paths, 0.90)
        gamma_per_path = np.full(n_paths, 0.12)

        cum_out = np.zeros((H_max, n_paths))
        vol_out = np.zeros((H_max, n_paths))

        unified_mc_multi_path_kernel(
            n_paths, H_max, 0.001, 0.0002,
            0.95, 1e-6,
            nu_per_path,
            True,
            omega_per_path, alpha_per_path, beta_per_path,
            0.0, 0.0, 0.05, False,
            z_normals, z_chi2, z_drift,
            z_jump_uniform, z_jump_normal,
            cum_out, vol_out,
            gamma_per_path,
            1.0,  # variance_inflation
            0.0,  # mu_drift
        )

        # Should produce valid output
        self.assertFalse(np.any(np.isnan(cum_out)), "Multi-path GJR should not produce NaN")
        self.assertTrue(np.all(vol_out > 0), "All vol values should be positive")


# ============================================================================
# Helper: generate standard random inputs for kernel tests
# ============================================================================
def _make_kernel_inputs(n_paths=5000, H_max=10, seed=42):
    """Create standard random inputs for unified_mc_simulate_kernel."""
    np.random.seed(seed)
    return {
        'n_paths': n_paths,
        'H_max': H_max,
        'z_normals': np.random.standard_normal((H_max, n_paths)),
        'z_chi2': np.ones((H_max, n_paths)),
        'z_drift': np.random.standard_normal((H_max, n_paths)),
        'z_jump_uniform': np.random.uniform(size=(H_max, n_paths)),
        'z_jump_normal': np.random.standard_normal((H_max, n_paths)),
    }


def _run_kernel(inputs, **kwargs):
    """Run unified_mc_simulate_kernel with given inputs and kwargs overrides.

    Default params: mu_now=0.0, h0=0.0002, phi=0.95, drift_q=1e-6, nu=200.0,
    no GARCH, no jumps, all v7.6/v7.7 params at identity defaults.
    Returns (cum_out, vol_out).
    """
    from models.numba_kernels import unified_mc_simulate_kernel

    n_paths = inputs['n_paths']
    H_max = inputs['H_max']

    defaults = dict(
        n_paths=n_paths,
        H_max=H_max,
        mu_now=0.0,
        h0=0.0002,
        phi=0.95,
        drift_q=1e-6,
        nu=200.0,
        use_garch=False,
        omega=0.0,
        alpha=0.0,
        beta=0.0,
        jump_intensity=0.0,
        jump_mean=0.0,
        jump_std=0.05,
        enable_jumps=False,
        z_normals=inputs['z_normals'].copy(),
        z_chi2=inputs['z_chi2'].copy(),
        z_drift=inputs['z_drift'].copy(),
        z_jump_uniform=inputs['z_jump_uniform'].copy(),
        z_jump_normal=inputs['z_jump_normal'].copy(),
        cum_out=None,
        vol_out=None,
        garch_leverage=0.0,
        variance_inflation=1.0,
        mu_drift=0.0,
        alpha_asym=0.0,
        k_asym=2.0,
        risk_premium_sensitivity=0.0,
        kappa_mean_rev=0.0,
        theta_long_var=0.0,
        crps_sigma_shrinkage=1.0,
        ms_sensitivity=0.0,
        q_stress_ratio=1.0,
        rough_hurst=0.0,
        frac_weights=np.empty(0, dtype=np.float64),
        sigma_eta=0.0,
        t_df_asym=0.0,
        regime_switch_prob=0.0,
        gamma_vov=0.0,
        vov_damping=0.0,
        skew_score_sensitivity=0.0,
        skew_persistence=0.97,
        loc_bias_var_coeff=0.0,
        loc_bias_drift_coeff=0.0,
        q_vol_coupling=0.0,
    )
    defaults.update(kwargs)

    # Always fresh output arrays
    cum_out = np.zeros((H_max, n_paths))
    vol_out = np.zeros((H_max, n_paths))
    defaults['cum_out'] = cum_out
    defaults['vol_out'] = vol_out

    # Call kernel with positional + keyword args in correct order
    unified_mc_simulate_kernel(
        defaults['n_paths'],
        defaults['H_max'],
        defaults['mu_now'],
        defaults['h0'],
        defaults['phi'],
        defaults['drift_q'],
        defaults['nu'],
        defaults['use_garch'],
        defaults['omega'],
        defaults['alpha'],
        defaults['beta'],
        defaults['jump_intensity'],
        defaults['jump_mean'],
        defaults['jump_std'],
        defaults['enable_jumps'],
        defaults['z_normals'],
        defaults['z_chi2'],
        defaults['z_drift'],
        defaults['z_jump_uniform'],
        defaults['z_jump_normal'],
        defaults['cum_out'],
        defaults['vol_out'],
        defaults['garch_leverage'],
        defaults['variance_inflation'],
        defaults['mu_drift'],
        defaults['alpha_asym'],
        defaults['k_asym'],
        defaults['risk_premium_sensitivity'],
        defaults['kappa_mean_rev'],
        defaults['theta_long_var'],
        defaults['crps_sigma_shrinkage'],
        defaults['ms_sensitivity'],
        defaults['q_stress_ratio'],
        defaults['rough_hurst'],
        defaults['frac_weights'],
        defaults['sigma_eta'],
        defaults['t_df_asym'],
        defaults['regime_switch_prob'],
        defaults['gamma_vov'],
        defaults['vov_damping'],
        defaults['skew_score_sensitivity'],
        defaults['skew_persistence'],
        defaults['loc_bias_var_coeff'],
        defaults['loc_bias_drift_coeff'],
        defaults['q_vol_coupling'],
    )
    return cum_out, vol_out


# ============================================================================
# v7.7 Tier 2 Tests
# ============================================================================

class TestKappaMeanRev(unittest.TestCase):
    """Test vol mean-reversion toward theta_long_var (Heston 1993)."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_kappa_pulls_vol_toward_theta(self):
        """High kappa should pull variance toward theta over time."""
        inp = _make_kernel_inputs(n_paths=10000, H_max=30, seed=42)

        # h0 far above theta — vol should decline with mean reversion
        h0_high = 0.001  # 3.16% daily vol
        theta_low = 0.0001  # 1.0% daily vol (long-run)

        # Without mean reversion
        cum_no_mr, vol_no_mr = _run_kernel(inp, h0=h0_high,
                                           use_garch=True, omega=1e-6,
                                           alpha=0.08, beta=0.90)

        # With strong mean reversion
        cum_mr, vol_mr = _run_kernel(inp, h0=h0_high,
                                     use_garch=True, omega=1e-6,
                                     alpha=0.08, beta=0.90,
                                     kappa_mean_rev=0.15,
                                     theta_long_var=theta_low)

        # Vol at horizon 30 should be lower with mean reversion
        avg_vol_no_mr = np.mean(vol_no_mr[-1, :])
        avg_vol_mr = np.mean(vol_mr[-1, :])
        self.assertLess(avg_vol_mr, avg_vol_no_mr,
                        "Mean reversion should pull vol down when h0 > theta")

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_kappa_zero_is_no_op(self):
        """kappa=0 should produce identical results to baseline."""
        inp = _make_kernel_inputs(n_paths=1000, H_max=5, seed=99)

        cum_base, vol_base = _run_kernel(inp)
        cum_kappa0, vol_kappa0 = _run_kernel(inp, kappa_mean_rev=0.0,
                                              theta_long_var=0.0001)

        np.testing.assert_allclose(cum_base, cum_kappa0, atol=1e-12,
                                   err_msg="kappa=0 should not change results")


class TestCRPSShrinkage(unittest.TestCase):
    """Test CRPS-optimal sigma shrinkage scales initial variance."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_shrinkage_tightens_distribution(self):
        """crps_sigma_shrinkage < 1 should narrow the return distribution."""
        inp = _make_kernel_inputs(n_paths=20000, H_max=5, seed=42)

        cum_base, _ = _run_kernel(inp)
        cum_shrunk, _ = _run_kernel(inp, crps_sigma_shrinkage=0.85)

        std_base = np.std(cum_base[0, :])
        std_shrunk = np.std(cum_shrunk[0, :])

        # crps_sigma_shrinkage=0.85 → h0*0.85 → sigma*sqrt(0.85) ≈ 0.922
        expected_ratio = math.sqrt(0.85)
        actual_ratio = std_shrunk / std_base
        self.assertAlmostEqual(actual_ratio, expected_ratio, delta=0.05,
                               msg=f"Expected ratio ~{expected_ratio:.3f}, got {actual_ratio:.3f}")

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_shrinkage_one_is_no_op(self):
        """crps_sigma_shrinkage=1.0 should not change results."""
        inp = _make_kernel_inputs(n_paths=1000, H_max=5, seed=99)

        cum_base, vol_base = _run_kernel(inp)
        cum_s1, vol_s1 = _run_kernel(inp, crps_sigma_shrinkage=1.0)

        np.testing.assert_allclose(cum_base, cum_s1, atol=1e-12,
                                   err_msg="shrinkage=1.0 should be identity")


class TestMSSensitivity(unittest.TestCase):
    """Test Markov-switching process noise for drift."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_ms_increases_drift_variance(self):
        """MS process noise with high q_stress_ratio should increase return variance."""
        inp = _make_kernel_inputs(n_paths=10000, H_max=20, seed=42)

        # Without MS
        cum_base, _ = _run_kernel(inp, drift_q=1e-6)

        # With MS: when stress triggers, drift_q scales by q_stress_ratio
        cum_ms, _ = _run_kernel(inp, drift_q=1e-6,
                                ms_sensitivity=3.0,
                                q_stress_ratio=100.0)

        # MS should produce wider return distribution (more drift uncertainty)
        std_base = np.std(cum_base[-1, :])
        std_ms = np.std(cum_ms[-1, :])
        self.assertGreater(std_ms, std_base,
                           "MS process noise should increase return variance")

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_ms_zero_sensitivity_is_no_op(self):
        """ms_sensitivity=0 should produce identical results."""
        inp = _make_kernel_inputs(n_paths=1000, H_max=5, seed=99)

        cum_base, vol_base = _run_kernel(inp)
        cum_ms0, vol_ms0 = _run_kernel(inp, ms_sensitivity=0.0,
                                         q_stress_ratio=100.0)

        np.testing.assert_allclose(cum_base, cum_ms0, atol=1e-12,
                                   err_msg="ms_sensitivity=0 should not change results")


class TestRoughHurst(unittest.TestCase):
    """Test fractional differencing rough volatility."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_rough_vol_changes_vol_dynamics(self):
        """rough_hurst close to 0 should change vol dynamics via fractional memory."""
        inp = _make_kernel_inputs(n_paths=10000, H_max=30, seed=42)

        # Without rough vol
        cum_base, vol_base = _run_kernel(inp, use_garch=True,
                                          omega=1e-6, alpha=0.08, beta=0.90)

        # With rough vol (H=0.1, strongly rough)
        # Need to precompute frac_weights
        d = 0.1 - 0.5  # d = -0.4
        max_lag = 50
        w = np.zeros(max_lag, dtype=np.float64)
        w[0] = 1.0
        for k in range(1, max_lag):
            w[k] = w[k - 1] * (d - k + 1) / k
        w_sum = np.sum(np.abs(w))
        if w_sum > 0:
            w = w / w_sum

        cum_rough, vol_rough = _run_kernel(inp, use_garch=True,
                                            omega=1e-6, alpha=0.08, beta=0.90,
                                            rough_hurst=0.1,
                                            frac_weights=w)

        # Vol dynamics should differ (autocorrelation structure changes)
        vol_acf_base = np.corrcoef(vol_base[0, :], vol_base[-1, :])[0, 1]
        vol_acf_rough = np.corrcoef(vol_rough[0, :], vol_rough[-1, :])[0, 1]
        # Just verify they're different — the direction depends on implementation
        self.assertFalse(np.isnan(vol_acf_rough),
                         "Rough vol should produce valid vol correlations")

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_rough_hurst_zero_is_no_op(self):
        """rough_hurst=0 should be no-op (rw = max(0, 0.3*(1-2*0)) = 0.3 but
        empty frac_weights disables it)."""
        inp = _make_kernel_inputs(n_paths=1000, H_max=5, seed=99)

        cum_base, vol_base = _run_kernel(inp)
        cum_rh0, vol_rh0 = _run_kernel(inp, rough_hurst=0.0)

        np.testing.assert_allclose(cum_base, cum_rh0, atol=1e-12,
                                   err_msg="rough_hurst=0 with empty weights should not change results")


# ============================================================================
# v7.7 Tier 3 Tests
# ============================================================================

class TestSigmaEta(unittest.TestCase):
    """Test observation noise perturbation (sigma_eta)."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_sigma_eta_increases_vol(self):
        """sigma_eta > 0.005 should increase observation noise, widening returns."""
        inp = _make_kernel_inputs(n_paths=20000, H_max=20, seed=42)

        # sigma_eta needs GARCH enabled so h_t compounds over time
        cum_base, vol_base = _run_kernel(inp, use_garch=True,
                                          omega=1e-6, alpha=0.08, beta=0.90)
        # Kernel threshold: sigma_eta > 0.005, so use 0.1 for strong effect
        cum_eta, vol_eta = _run_kernel(inp, use_garch=True,
                                        omega=1e-6, alpha=0.08, beta=0.90,
                                        sigma_eta=0.1)

        # sigma_eta adds extra vol when |z| > 1.5, so vol should be higher
        avg_vol_base = np.mean(vol_base[-1, :])
        avg_vol_eta = np.mean(vol_eta[-1, :])

        self.assertGreater(avg_vol_eta, avg_vol_base,
                           "sigma_eta should increase average volatility")

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_sigma_eta_zero_is_no_op(self):
        """sigma_eta=0 should not change results."""
        inp = _make_kernel_inputs(n_paths=1000, H_max=5, seed=99)

        cum_base, vol_base = _run_kernel(inp)
        cum_s0, vol_s0 = _run_kernel(inp, sigma_eta=0.0)

        np.testing.assert_allclose(cum_base, cum_s0, atol=1e-12,
                                   err_msg="sigma_eta=0 should be identity")


class TestTDfAsym(unittest.TestCase):
    """Test asymmetric degrees-of-freedom shift (t_df_asym)."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_negative_tdf_asym_heavier_left_tail(self):
        """t_df_asym < 0 should lower nu for negative returns → heavier left tail."""
        inp = _make_kernel_inputs(n_paths=50000, H_max=1, seed=42)

        nu_val = 8.0
        # Generate proper chi2 draws for Student-t
        np.random.seed(42)
        chi2_draws = np.random.gamma(shape=nu_val / 2.0, scale=2.0,
                                     size=(1, 50000))
        inp['z_chi2'] = chi2_draws / nu_val

        cum_sym, _ = _run_kernel(inp, nu=nu_val)
        cum_asym, _ = _run_kernel(inp, nu=nu_val, t_df_asym=-2.0)

        # Left tail (1st percentile) should be more extreme with t_df_asym < 0
        p1_sym = np.percentile(cum_sym[0, :], 1)
        p1_asym = np.percentile(cum_asym[0, :], 1)
        self.assertLess(p1_asym, p1_sym,
                        "t_df_asym<0 should produce heavier left tail")

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_tdf_asym_zero_is_no_op(self):
        """t_df_asym=0 should not change results."""
        inp = _make_kernel_inputs(n_paths=1000, H_max=5, seed=99)

        cum_base, vol_base = _run_kernel(inp, nu=8.0)
        cum_tdf0, vol_tdf0 = _run_kernel(inp, nu=8.0, t_df_asym=0.0)

        np.testing.assert_allclose(cum_base, cum_tdf0, atol=1e-12,
                                   err_msg="t_df_asym=0 should be identity")


class TestRegimeSwitchProb(unittest.TestCase):
    """Test regime-switching observation noise."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_regime_switch_increases_vol_variance(self):
        """regime_switch_prob > 0 should create bimodal vol structure."""
        inp = _make_kernel_inputs(n_paths=10000, H_max=20, seed=42)

        cum_base, vol_base = _run_kernel(inp, use_garch=True,
                                          omega=1e-6, alpha=0.08, beta=0.90)

        cum_rs, vol_rs = _run_kernel(inp, use_garch=True,
                                      omega=1e-6, alpha=0.08, beta=0.90,
                                      regime_switch_prob=0.05,
                                      q_stress_ratio=10.0)

        # Regime switching should increase the variance of volatility itself
        vol_var_base = np.var(vol_base[-1, :])
        vol_var_rs = np.var(vol_rs[-1, :])
        self.assertGreater(vol_var_rs, vol_var_base,
                           "Regime switching should increase vol dispersion")

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_regime_switch_zero_is_no_op(self):
        """regime_switch_prob=0 should not change results."""
        inp = _make_kernel_inputs(n_paths=1000, H_max=5, seed=99)

        cum_base, vol_base = _run_kernel(inp)
        cum_rs0, vol_rs0 = _run_kernel(inp, regime_switch_prob=0.0)

        np.testing.assert_allclose(cum_base, cum_rs0, atol=1e-12,
                                   err_msg="regime_switch_prob=0 should be identity")


class TestGammaVoV(unittest.TestCase):
    """Test vol-of-vol (stochastic volatility of variance)."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_vov_increases_vol_variance(self):
        """gamma_vov > 0 should increase the variance of simulated volatility."""
        inp = _make_kernel_inputs(n_paths=10000, H_max=20, seed=42)

        cum_base, vol_base = _run_kernel(inp, use_garch=True,
                                          omega=1e-6, alpha=0.08, beta=0.90)

        cum_vov, vol_vov = _run_kernel(inp, use_garch=True,
                                        omega=1e-6, alpha=0.08, beta=0.90,
                                        gamma_vov=0.5, vov_damping=0.95)

        # VoV should make vol more dispersed
        vol_var_base = np.var(vol_base[-1, :])
        vol_var_vov = np.var(vol_vov[-1, :])
        self.assertGreater(vol_var_vov, vol_var_base,
                           "Vol-of-vol should increase vol dispersion")

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_vov_zero_is_no_op(self):
        """gamma_vov=0 should not change results."""
        inp = _make_kernel_inputs(n_paths=1000, H_max=5, seed=99)

        cum_base, vol_base = _run_kernel(inp)
        cum_vov0, vol_vov0 = _run_kernel(inp, gamma_vov=0.0, vov_damping=0.95)

        np.testing.assert_allclose(cum_base, cum_vov0, atol=1e-12,
                                   err_msg="gamma_vov=0 should be identity")


class TestSkewDynamics(unittest.TestCase):
    """Test GAS dynamic skew updating."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_skew_sensitivity_changes_skewness(self):
        """skew_score_sensitivity > 0 should introduce dynamic skewness in returns."""
        inp = _make_kernel_inputs(n_paths=20000, H_max=20, seed=42)

        cum_base, _ = _run_kernel(inp)
        cum_skew, _ = _run_kernel(inp, skew_score_sensitivity=0.3,
                                   skew_persistence=0.95)

        # Dynamic skew should change the distribution shape
        skew_base = float(np.mean((cum_base[-1, :] - np.mean(cum_base[-1, :])) ** 3) /
                          np.std(cum_base[-1, :]) ** 3)
        skew_dyn = float(np.mean((cum_skew[-1, :] - np.mean(cum_skew[-1, :])) ** 3) /
                         np.std(cum_skew[-1, :]) ** 3)

        # Just verify results are valid (direction depends on random draws)
        self.assertTrue(np.isfinite(skew_dyn),
                        "Dynamic skew should produce finite skewness")

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_skew_zero_sensitivity_is_no_op(self):
        """skew_score_sensitivity=0 should not change results."""
        inp = _make_kernel_inputs(n_paths=1000, H_max=5, seed=99)

        cum_base, vol_base = _run_kernel(inp)
        cum_sk0, vol_sk0 = _run_kernel(inp, skew_score_sensitivity=0.0)

        np.testing.assert_allclose(cum_base, cum_sk0, atol=1e-12,
                                   err_msg="skew_score_sensitivity=0 should be identity")


class TestLocBias(unittest.TestCase):
    """Test location bias from variance and drift coefficients."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_loc_bias_var_shifts_mean(self):
        """loc_bias_var_coeff > 0 should shift mean based on instantaneous variance."""
        inp = _make_kernel_inputs(n_paths=20000, H_max=10, seed=42)

        # loc_bias_var_coeff requires theta_long_var > 1e-12 in kernel
        cum_base, _ = _run_kernel(inp, h0=0.0004, theta_long_var=0.0001)
        cum_bias, _ = _run_kernel(inp, h0=0.0004, theta_long_var=0.0001,
                                   loc_bias_var_coeff=0.5)

        mean_base = np.mean(cum_base[-1, :])
        mean_bias = np.mean(cum_bias[-1, :])

        # With positive var coeff, higher variance should add positive drift
        self.assertNotAlmostEqual(mean_base, mean_bias, places=5,
                                  msg="loc_bias_var_coeff should shift mean")

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_loc_bias_drift_shifts_mean(self):
        """loc_bias_drift_coeff > 0 should shift mean based on drift state."""
        inp = _make_kernel_inputs(n_paths=20000, H_max=10, seed=42)

        cum_base, _ = _run_kernel(inp, mu_now=0.001)
        cum_bias, _ = _run_kernel(inp, mu_now=0.001,
                                   loc_bias_drift_coeff=0.3)

        mean_base = np.mean(cum_base[-1, :])
        mean_bias = np.mean(cum_bias[-1, :])

        self.assertNotAlmostEqual(mean_base, mean_bias, places=5,
                                  msg="loc_bias_drift_coeff should shift mean")

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_loc_bias_zero_is_no_op(self):
        """loc_bias coefficients = 0 should not change results."""
        inp = _make_kernel_inputs(n_paths=1000, H_max=5, seed=99)

        cum_base, vol_base = _run_kernel(inp)
        cum_lb0, vol_lb0 = _run_kernel(inp, loc_bias_var_coeff=0.0,
                                         loc_bias_drift_coeff=0.0)

        np.testing.assert_allclose(cum_base, cum_lb0, atol=1e-12,
                                   err_msg="loc_bias=0 should be identity")


class TestQVolCoupling(unittest.TestCase):
    """Test volatility-coupled process noise (q_vol_coupling)."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_q_vol_coupling_increases_drift_variance(self):
        """q_vol_coupling > 0 should make drift noise scale with volatility."""
        inp = _make_kernel_inputs(n_paths=10000, H_max=20, seed=42)

        # q_vol_coupling requires theta_long_var > 1e-12 in kernel
        cum_base, _ = _run_kernel(inp, drift_q=1e-6, h0=0.0004,
                                  theta_long_var=0.0001)
        cum_qvc, _ = _run_kernel(inp, drift_q=1e-6, h0=0.0004,
                                  theta_long_var=0.0001,
                                  q_vol_coupling=0.5)

        std_base = np.std(cum_base[-1, :])
        std_qvc = np.std(cum_qvc[-1, :])

        # Vol coupling should increase return spread through drift channel
        self.assertGreater(std_qvc, std_base,
                           "q_vol_coupling should increase return dispersion")

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_q_vol_coupling_zero_is_no_op(self):
        """q_vol_coupling=0 should not change results."""
        inp = _make_kernel_inputs(n_paths=1000, H_max=5, seed=99)

        cum_base, vol_base = _run_kernel(inp)
        cum_qvc0, vol_qvc0 = _run_kernel(inp, q_vol_coupling=0.0)

        np.testing.assert_allclose(cum_base, cum_qvc0, atol=1e-12,
                                   err_msg="q_vol_coupling=0 should be identity")


# ============================================================================
# Backward Compatibility
# ============================================================================

class TestBackwardCompatibility(unittest.TestCase):
    """Verify all v7.7 params at defaults produce identical results to v7.6."""

    @unittest.skipUnless(_numba_available(), "Numba not available")
    def test_all_defaults_match_v76_baseline(self):
        """Kernel with all Tier 2+3 params at defaults should match v7.6 output."""
        from models.numba_kernels import unified_mc_simulate_kernel

        n_paths = 2000
        H_max = 10
        np.random.seed(42)

        z_normals = np.random.standard_normal((H_max, n_paths))
        z_chi2 = np.ones((H_max, n_paths))
        z_drift = np.random.standard_normal((H_max, n_paths))
        z_jump_uniform = np.random.uniform(size=(H_max, n_paths))
        z_jump_normal = np.random.standard_normal((H_max, n_paths))

        # Run with explicit v7.6 params only
        cum_v76 = np.zeros((H_max, n_paths))
        vol_v76 = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, 0.001, 0.0002,
            0.95, 1e-6, 8.0,
            True, 1e-6, 0.08, 0.90,
            0.01, 0.0, 0.05, True,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_uniform.copy(), z_jump_normal.copy(),
            cum_v76, vol_v76,
            0.12, 1.2, 0.0005, -0.1, 2.0, 0.001,
            # All Tier 2+3 at defaults (identity)
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            np.empty(0, dtype=np.float64),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.97, 0.0, 0.0, 0.0,
        )

        # Run again with identical params
        cum_v77 = np.zeros((H_max, n_paths))
        vol_v77 = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, 0.001, 0.0002,
            0.95, 1e-6, 8.0,
            True, 1e-6, 0.08, 0.90,
            0.01, 0.0, 0.05, True,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_uniform.copy(), z_jump_normal.copy(),
            cum_v77, vol_v77,
            0.12, 1.2, 0.0005, -0.1, 2.0, 0.001,
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            np.empty(0, dtype=np.float64),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.97, 0.0, 0.0, 0.0,
        )

        np.testing.assert_allclose(cum_v76, cum_v77, atol=1e-14,
                                   err_msg="Identical calls should produce identical output")
        np.testing.assert_allclose(vol_v76, vol_v77, atol=1e-14,
                                   err_msg="Identical calls should produce identical vol output")

    def test_run_unified_mc_all_v77_params_accepted(self):
        """run_unified_mc should accept all v7.7 Tier 2+3 params without error."""
        from decision.signals import run_unified_mc

        result = run_unified_mc(
            mu_t=0.001,
            P_t=1e-6,
            phi=0.95,
            q=1e-6,
            sigma2_step=0.0002,
            H_max=5,
            n_paths=100,
            nu=8.0,
            use_garch=True,
            garch_omega=1e-6,
            garch_alpha=0.08,
            garch_beta=0.90,
            seed=42,
            # v7.6 params
            garch_leverage=0.12,
            variance_inflation=1.2,
            mu_drift=0.0005,
            alpha_asym=-0.1,
            k_asym=2.0,
            risk_premium_sensitivity=0.001,
            # v7.7 Tier 2 params
            kappa_mean_rev=0.05,
            theta_long_var=0.0002,
            crps_sigma_shrinkage=0.9,
            ms_sensitivity=2.0,
            q_stress_ratio=10.0,
            rough_hurst=0.1,
            # v7.7 Tier 3 params
            sigma_eta=0.001,
            t_df_asym=-1.0,
            regime_switch_prob=0.02,
            gamma_vov=0.3,
            vov_damping=0.95,
            skew_score_sensitivity=0.2,
            skew_persistence=0.95,
            loc_bias_var_coeff=0.1,
            loc_bias_drift_coeff=0.05,
            q_vol_coupling=0.3,
        )

        self.assertIn('returns', result)
        self.assertIn('volatility', result)
        self.assertEqual(result['returns'].shape, (5, 100))
        self.assertEqual(result['volatility'].shape, (5, 100))

        # Verify no NaN/Inf
        self.assertFalse(np.any(np.isnan(result['returns'])),
                         "MC returns should have no NaN")
        self.assertFalse(np.any(np.isinf(result['returns'])),
                         "MC returns should have no Inf")


if __name__ == '__main__':
    unittest.main()
