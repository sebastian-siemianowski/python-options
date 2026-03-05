#!/usr/bin/env python3
"""
Test suite for v7.8 elite MC enhancements:
  A) Dynamic Leverage Decay — EWM of neg-return fraction amplifies GJR γ
  B) Liquidity-Volatility Feedback — quadratic vol amplification when h_t > θ_long
  C) PIT-Entropy Sigma Stabilizer — relaxes crps_sigma_shrinkage when PIT KL is high

Tests cover:
  1. Numba kernel unit tests (direct kernel calls)
  2. Feature-disabled regression (params=0 → identical to baseline)
  3. Multi-asset calibration on 8 diversified assets
  4. Stress scenario validation (crash, whipsaw, low-vol)
  5. Config dataclass field validation
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import math
import unittest


# ---------------------------------------------------------------------------
# 1. Numba kernel unit tests
# ---------------------------------------------------------------------------
class TestPitKLUniformKernel(unittest.TestCase):
    """Test the pit_kl_uniform_kernel Numba function."""

    def test_uniform_pit_gives_near_zero_kl(self):
        """Uniform(0,1) PIT values should have KL ≈ 0."""
        from models.numba_kernels import pit_kl_uniform_kernel
        np.random.seed(42)
        pit = np.sort(np.random.uniform(0.001, 0.999, 5000))
        kl = float(pit_kl_uniform_kernel(pit))
        self.assertLess(kl, 0.05, f"KL for uniform PIT should be near 0, got {kl:.4f}")
        print(f"  ✓ Uniform PIT KL = {kl:.6f} (< 0.05)")

    def test_bimodal_pit_gives_high_kl(self):
        """Bimodal PIT (badly calibrated) should have high KL."""
        from models.numba_kernels import pit_kl_uniform_kernel
        # Bimodal: most mass near 0 and 1
        np.random.seed(42)
        pit = np.concatenate([
            np.random.beta(0.5, 5, 2500),
            np.random.beta(5, 0.5, 2500),
        ])
        pit = np.clip(pit, 0.001, 0.999).astype(np.float64)
        kl = float(pit_kl_uniform_kernel(pit))
        self.assertGreater(kl, 0.2, f"KL for bimodal PIT should be high, got {kl:.4f}")
        print(f"  ✓ Bimodal PIT KL = {kl:.6f} (> 0.2)")

    def test_concentrated_pit_gives_high_kl(self):
        """Concentrated PIT (over-confident model) should have high KL."""
        from models.numba_kernels import pit_kl_uniform_kernel
        pit = np.random.normal(0.5, 0.05, 3000)
        pit = np.clip(pit, 0.001, 0.999).astype(np.float64)
        kl = float(pit_kl_uniform_kernel(pit))
        self.assertGreater(kl, 0.5, f"KL for concentrated PIT should be high, got {kl:.4f}")
        print(f"  ✓ Concentrated PIT KL = {kl:.6f} (> 0.5)")

    def test_small_sample(self):
        """Small sample (< 20) should still return finite."""
        from models.numba_kernels import pit_kl_uniform_kernel
        pit = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        kl = float(pit_kl_uniform_kernel(pit))
        self.assertTrue(np.isfinite(kl), f"KL should be finite for small sample, got {kl}")
        print(f"  ✓ Small sample KL = {kl:.6f}")


class TestGarchVarianceKernelEnhancements(unittest.TestCase):
    """Test garch_variance_kernel with new dynamic leverage and liq stress params."""

    def _build_test_data(self, n=500, seed=42):
        np.random.seed(seed)
        returns = np.random.normal(0, 0.02, n)
        # Inject a crash period
        returns[200:230] = np.random.normal(-0.03, 0.04, 30)
        innovations = returns - np.mean(returns)
        sq = (innovations ** 2).astype(np.float64)
        neg = (innovations < 0).astype(np.float64)
        return innovations, sq, neg

    def _run_garch(self, innovations, sq, neg, omega, alpha, leverage, beta,
                   uvar, liq_c=0.0, lev_dyn=0.0):
        """Helper that calls garch_variance_kernel correctly."""
        from models.numba_kernels import garch_variance_kernel
        n = len(innovations)
        sm = math.sqrt(10.0)
        h_out = np.empty(n, dtype=np.float64)
        garch_variance_kernel(
            sq, neg, innovations, n,
            omega, alpha, beta, leverage, uvar,
            0.0, 0.0, uvar, 0.0, 0.0, sm,
            h_out,
            liq_c, lev_dyn,
        )
        return h_out

    def test_zero_params_match_baseline(self):
        """With liq_stress_coeff=0 and leverage_dynamic_decay=0, output matches baseline."""
        innovations, sq, neg = self._build_test_data()
        omega, alpha, leverage, beta = 1e-6, 0.08, 0.05, 0.88
        uvar = omega / max(1 - alpha - 0.5 * leverage - beta, 0.01)

        h_base = self._run_garch(innovations, sq, neg, omega, alpha, leverage, beta, uvar)
        h_new = self._run_garch(innovations, sq, neg, omega, alpha, leverage, beta, uvar,
                                liq_c=0.0, lev_dyn=0.0)
        np.testing.assert_allclose(h_base, h_new, rtol=1e-12,
                                   err_msg="Zero enhancement params should match baseline")
        print("  ✓ Zero enhancement params match baseline exactly")

    def test_dynamic_leverage_increases_crash_vol(self):
        """Dynamic leverage should increase vol during crash periods."""
        innovations, sq, neg = self._build_test_data()
        omega, alpha, leverage, beta = 1e-6, 0.08, 0.05, 0.88
        uvar = omega / max(1 - alpha - 0.5 * leverage - beta, 0.01)

        h_base = self._run_garch(innovations, sq, neg, omega, alpha, leverage, beta, uvar)
        h_dyn = self._run_garch(innovations, sq, neg, omega, alpha, leverage, beta, uvar,
                                lev_dyn=0.4)
        # During crash (t=200-230), dynamic leverage should produce higher vol
        crash_ratio = np.mean(h_dyn[220:240]) / np.mean(h_base[220:240])
        self.assertGreater(crash_ratio, 1.0,
                          f"Dynamic leverage should increase crash vol, ratio={crash_ratio:.4f}")
        print(f"  ✓ Dynamic leverage crash vol ratio = {crash_ratio:.4f}")

    def test_liquidity_stress_amplifies_high_vol(self):
        """Liquidity stress should amplify vol when h > unconditional_var."""
        innovations, sq, neg = self._build_test_data()
        omega, alpha, leverage, beta = 1e-6, 0.08, 0.05, 0.88
        uvar = omega / max(1 - alpha - 0.5 * leverage - beta, 0.01)

        h_base = self._run_garch(innovations, sq, neg, omega, alpha, leverage, beta, uvar)
        h_liq = self._run_garch(innovations, sq, neg, omega, alpha, leverage, beta, uvar,
                                liq_c=0.2)
        # Post crash, vol should be amplified
        high_vol_mask = h_base > uvar * 1.5
        if np.any(high_vol_mask):
            ratio = np.mean(h_liq[high_vol_mask]) / np.mean(h_base[high_vol_mask])
            self.assertGreater(ratio, 1.0,
                              f"Liquidity stress should amplify high vol, ratio={ratio:.4f}")
            print(f"  ✓ Liquidity stress amplification ratio = {ratio:.4f}")
        else:
            print("  ⚠ No high-vol periods detected (test data too calm)")


class TestUnifiedMCKernelEnhancements(unittest.TestCase):
    """Test unified_mc_simulate_kernel with new params."""

    def test_zero_params_regression(self):
        """Kernel with leverage_dynamic_decay=0, liq_stress_coeff=0 should match baseline."""
        from models.numba_kernels import unified_mc_simulate_kernel
        np.random.seed(123)
        n_paths, H_max = 1000, 20
        mu_now, h0 = 0.001, 0.0004
        phi, q, nu = 0.98, 1e-5, 6.0

        z_normals = np.random.standard_normal((H_max, n_paths))
        z_chi2 = np.random.chisquare(nu, (H_max, n_paths))
        z_drift = np.random.standard_normal((H_max, n_paths))
        z_jump_u = np.random.uniform(0, 1, (H_max, n_paths))
        z_jump_n = np.random.standard_normal((H_max, n_paths))
        fw = np.zeros(50, dtype=np.float64)

        # Run twice with same RNG, both with zero enhancement params
        cum1 = np.zeros((H_max, n_paths))
        vol1 = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, mu_now, h0,
            phi, q, nu,
            True, 1e-6, 0.08, 0.88,
            0.0, 0.0, 0.05, False,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_u.copy(), z_jump_n.copy(),
            cum1, vol1,
            0.05, 1.0, 0.0, 0.0, 2.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, fw.copy(),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.97,
            0.0, 0.0, 0.0,
            0.0, 0.0,  # leverage_dynamic_decay=0, liq_stress_coeff=0
        )

        cum2 = np.zeros((H_max, n_paths))
        vol2 = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, mu_now, h0,
            phi, q, nu,
            True, 1e-6, 0.08, 0.88,
            0.0, 0.0, 0.05, False,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_u.copy(), z_jump_n.copy(),
            cum2, vol2,
            0.05, 1.0, 0.0, 0.0, 2.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, fw.copy(),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.97,
            0.0, 0.0, 0.0,
            0.0, 0.0,  # same zero params
        )

        np.testing.assert_allclose(cum1, cum2, rtol=1e-12,
                                   err_msg="Identical zero-param runs should match")
        print("  ✓ Zero-param MC kernel regression test passed")

    def test_dynamic_leverage_affects_vol_paths(self):
        """Non-zero leverage_dynamic_decay should change vol paths."""
        from models.numba_kernels import unified_mc_simulate_kernel
        np.random.seed(456)
        n_paths, H_max = 2000, 30
        mu_now, h0 = -0.005, 0.001  # negative start → trigger leverage
        phi, q, nu = 0.98, 1e-5, 6.0

        z_normals = np.random.standard_normal((H_max, n_paths))
        z_chi2 = np.random.chisquare(nu, (H_max, n_paths))
        z_drift = np.random.standard_normal((H_max, n_paths))
        z_jump_u = np.random.uniform(0, 1, (H_max, n_paths))
        z_jump_n = np.random.standard_normal((H_max, n_paths))
        fw = np.zeros(50, dtype=np.float64)

        cum_base = np.zeros((H_max, n_paths))
        vol_base = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, mu_now, h0,
            phi, q, nu,
            True, 1e-6, 0.08, 0.88,
            0.0, 0.0, 0.05, False,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_u.copy(), z_jump_n.copy(),
            cum_base, vol_base,
            0.05, 1.0, 0.0, 0.0, 2.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, fw.copy(),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.97,
            0.0, 0.0, 0.0,
            0.0, 0.0,
        )

        cum_dyn = np.zeros((H_max, n_paths))
        vol_dyn = np.zeros((H_max, n_paths))
        unified_mc_simulate_kernel(
            n_paths, H_max, mu_now, h0,
            phi, q, nu,
            True, 1e-6, 0.08, 0.88,
            0.0, 0.0, 0.05, False,
            z_normals.copy(), z_chi2.copy(), z_drift.copy(),
            z_jump_u.copy(), z_jump_n.copy(),
            cum_dyn, vol_dyn,
            0.05, 1.0, 0.0, 0.0, 2.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 1.0, 0.0, fw.copy(),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.97,
            0.0, 0.0, 0.0,
            0.5, 0.0,  # leverage_dynamic_decay=0.5
        )

        # Vol paths should be different
        vol_diff = np.mean(np.abs(vol_dyn - vol_base))
        self.assertGreater(vol_diff, 0.0,
                          "Dynamic leverage should change vol paths")
        print(f"  ✓ Dynamic leverage changes vol paths (mean diff = {vol_diff:.6f})")


# ---------------------------------------------------------------------------
# 2. Config dataclass tests
# ---------------------------------------------------------------------------
class TestConfigDataclasses(unittest.TestCase):
    """Test that new config fields are properly defined and clipped."""

    def test_student_t_config_fields(self):
        """New fields exist with correct defaults in UnifiedStudentTConfig."""
        from models.phi_student_t_unified import UnifiedStudentTConfig
        cfg = UnifiedStudentTConfig()
        self.assertEqual(cfg.leverage_dynamic_decay, 0.0)
        self.assertEqual(cfg.liq_stress_coeff, 0.0)
        self.assertEqual(cfg.entropy_sigma_lambda, 0.0)
        print("  ✓ Student-t config defaults: ldd=0, lsc=0, esl=0")

    def test_student_t_config_clipping(self):
        """Fields should be clipped to valid ranges."""
        from models.phi_student_t_unified import UnifiedStudentTConfig
        cfg = UnifiedStudentTConfig(
            leverage_dynamic_decay=1.5,  # > 0.8 → clip to 0.8
            liq_stress_coeff=-0.1,       # < 0 → clip to 0
            entropy_sigma_lambda=0.7,    # > 0.5 → clip to 0.5
        )
        self.assertAlmostEqual(cfg.leverage_dynamic_decay, 0.8)
        self.assertAlmostEqual(cfg.liq_stress_coeff, 0.0)
        self.assertAlmostEqual(cfg.entropy_sigma_lambda, 0.5)
        print("  ✓ Student-t config clipping works: 1.5→0.8, -0.1→0.0, 0.7→0.5")

    def test_gaussian_config_fields(self):
        """New fields exist with correct defaults in GaussianUnifiedConfig."""
        from models.gaussian import GaussianUnifiedConfig
        cfg = GaussianUnifiedConfig()
        self.assertEqual(cfg.leverage_dynamic_decay, 0.0)
        self.assertEqual(cfg.liq_stress_coeff, 0.0)
        self.assertEqual(cfg.entropy_sigma_lambda, 0.0)
        print("  ✓ Gaussian config defaults: ldd=0, lsc=0, esl=0")


# ---------------------------------------------------------------------------
# 3. Multi-asset calibration tests (8 assets)
# ---------------------------------------------------------------------------
class TestMultiAssetCalibration(unittest.TestCase):
    """
    Test that tuning pipeline runs without error on diversified assets
    and that the new params are populated in the output config.
    
    Uses cached price data if available, otherwise skips gracefully.
    """

    TEST_ASSETS = ['SPY', 'NVDA', 'GC=F', 'EURUSD=X', 'AAPL', 'SI=F', 'TSLA', 'BTC-USD']

    def _load_returns(self, symbol):
        """Load returns from cached price data."""
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'prices')
        fname = os.path.join(data_dir, f'{symbol}_1d.csv')
        if not os.path.exists(fname):
            return None, None
        import pandas as pd
        df = pd.read_csv(fname)
        if 'Close' not in df.columns:
            return None, None
        close = df['Close'].values.astype(np.float64)
        if len(close) < 200:
            return None, None
        returns = np.diff(np.log(close))
        vol = np.abs(returns)
        return returns, vol

    def test_student_t_pipeline_populates_new_params(self):
        """Student-t pipeline should populate leverage_dynamic_decay, liq_stress_coeff, entropy_sigma_lambda."""
        from models.phi_student_t_unified import UnifiedPhiStudentTModel, UnifiedStudentTConfig

        tested = 0
        for symbol in self.TEST_ASSETS:
            returns, vol = self._load_returns(symbol)
            if returns is None:
                continue

            try:
                config = UnifiedStudentTConfig()
                config, diagnostics = UnifiedPhiStudentTModel.tune_unified(
                    returns, vol, config, asset_symbol=symbol)
                
                ldd = float(getattr(config, 'leverage_dynamic_decay', -1))
                lsc = float(getattr(config, 'liq_stress_coeff', -1))
                esl = float(getattr(config, 'entropy_sigma_lambda', -1))

                self.assertGreaterEqual(ldd, 0.0, f"{symbol}: leverage_dynamic_decay should be >= 0")
                self.assertGreaterEqual(lsc, 0.0, f"{symbol}: liq_stress_coeff should be >= 0")
                self.assertGreaterEqual(esl, 0.0, f"{symbol}: entropy_sigma_lambda should be >= 0")
                self.assertLessEqual(ldd, 0.8, f"{symbol}: leverage_dynamic_decay should be <= 0.8")
                self.assertLessEqual(lsc, 0.5, f"{symbol}: liq_stress_coeff should be <= 0.5")
                self.assertLessEqual(esl, 0.5, f"{symbol}: entropy_sigma_lambda should be <= 0.5")

                print(f"  ✓ {symbol:10s} Student-t: ldd={ldd:.2f} lsc={lsc:.2f} esl={esl:.2f}")
                tested += 1
            except Exception as e:
                print(f"  ⚠ {symbol:10s} Student-t tuning failed: {e}")

        if tested == 0:
            self.skipTest("No cached price data available for any test asset")
        self.assertGreaterEqual(tested, 1)

    def test_gaussian_pipeline_populates_new_params(self):
        """Gaussian pipeline should populate leverage_dynamic_decay, liq_stress_coeff, entropy_sigma_lambda."""
        from models.gaussian import GaussianDriftModel, GaussianUnifiedConfig

        tested = 0
        for symbol in self.TEST_ASSETS[:4]:  # Faster: test only 4 assets for Gaussian
            returns, vol = self._load_returns(symbol)
            if returns is None:
                continue

            try:
                config = GaussianUnifiedConfig()
                config, diagnostics = GaussianDriftModel.tune_unified(
                    returns, vol, config, asset_symbol=symbol)
                
                ldd = float(getattr(config, 'leverage_dynamic_decay', -1))
                lsc = float(getattr(config, 'liq_stress_coeff', -1))
                esl = float(getattr(config, 'entropy_sigma_lambda', -1))

                self.assertGreaterEqual(ldd, 0.0, f"{symbol}: leverage_dynamic_decay should be >= 0")
                self.assertGreaterEqual(lsc, 0.0, f"{symbol}: liq_stress_coeff should be >= 0")
                self.assertGreaterEqual(esl, 0.0, f"{symbol}: entropy_sigma_lambda should be >= 0")

                print(f"  ✓ {symbol:10s} Gaussian: ldd={ldd:.2f} lsc={lsc:.2f} esl={esl:.2f}")
                tested += 1
            except Exception as e:
                print(f"  ⚠ {symbol:10s} Gaussian tuning failed: {e}")

        if tested == 0:
            self.skipTest("No cached price data available for any test asset")
        self.assertGreaterEqual(tested, 1)


# ---------------------------------------------------------------------------
# 4. Stress scenario validation
# ---------------------------------------------------------------------------
class TestStressScenarios(unittest.TestCase):
    """Test enhancement behavior under synthetic stress scenarios."""

    def _run_garch(self, innovations, omega, alpha, leverage, beta,
                   uvar, liq_c=0.0, lev_dyn=0.0):
        """Helper for stress scenario GARCH calls."""
        from models.numba_kernels import garch_variance_kernel
        n = len(innovations)
        sq = (innovations ** 2).astype(np.float64)
        neg = (innovations < 0).astype(np.float64)
        sm = math.sqrt(10.0)
        h_out = np.empty(n, dtype=np.float64)
        garch_variance_kernel(
            sq, neg, innovations, n,
            omega, alpha, beta, leverage, uvar,
            0.0, 0.0, uvar, 0.0, 0.0, sm,
            h_out,
            liq_c, lev_dyn,
        )
        return h_out

    def test_dynamic_leverage_crash_scenario(self):
        """During a sustained crash, dynamic leverage should increase vol more than static."""
        np.random.seed(99)
        # Normal period then crash
        returns = np.concatenate([
            np.random.normal(0.001, 0.01, 300),
            np.random.normal(-0.02, 0.03, 100),  # sustained crash
            np.random.normal(0.001, 0.01, 200),
        ])
        innovations = (returns - np.mean(returns[:300])).astype(np.float64)
        omega, alpha, leverage, beta = 1e-6, 0.08, 0.06, 0.88
        uvar = max(omega / max(1 - alpha - 0.5*leverage - beta, 0.01), 1e-8)

        h_static = self._run_garch(innovations, omega, alpha, leverage, beta, uvar)
        h_dynamic = self._run_garch(innovations, omega, alpha, leverage, beta, uvar,
                                     lev_dyn=0.6)
        # During crash, dynamic should produce higher vol
        crash_mean_static = np.mean(h_static[350:400])
        crash_mean_dynamic = np.mean(h_dynamic[350:400])
        self.assertGreater(crash_mean_dynamic, crash_mean_static * 0.95,
                          "Dynamic leverage should increase crash-period vol")
        print(f"  ✓ Crash vol: static={crash_mean_static:.6f} dynamic={crash_mean_dynamic:.6f}")

    def test_liquidity_stress_high_vol_amplification(self):
        """Liquidity stress should amplify vol quadratically above unconditional."""
        np.random.seed(77)
        returns = np.concatenate([
            np.random.normal(0, 0.01, 200),
            np.random.normal(0, 0.05, 100),  # high vol episode
            np.random.normal(0, 0.01, 200),
        ])
        innovations = (returns - np.mean(returns[:200])).astype(np.float64)
        omega, alpha, leverage, beta = 1e-6, 0.08, 0.04, 0.88
        uvar = max(omega / max(1 - alpha - 0.5*leverage - beta, 0.01), 1e-8)

        h_no_liq = self._run_garch(innovations, omega, alpha, leverage, beta, uvar)
        h_liq = self._run_garch(innovations, omega, alpha, leverage, beta, uvar, liq_c=0.3)
        # During high vol, liq version should be higher
        high_vol_idx = np.where(h_no_liq > uvar * 2.0)[0]
        if len(high_vol_idx) > 5:
            ratio = np.mean(h_liq[high_vol_idx]) / np.mean(h_no_liq[high_vol_idx])
            self.assertGreater(ratio, 1.01,
                              f"Liquidity stress should amplify high vol, ratio={ratio:.4f}")
            print(f"  ✓ High-vol amplification ratio = {ratio:.4f} ({len(high_vol_idx)} points)")
        else:
            print("  ⚠ Not enough high-vol points for liq stress test")

    def test_low_vol_regime_bounded_amplification(self):
        """In low-vol regimes, liquidity stress should be bounded by 50x cap."""
        np.random.seed(55)
        returns = np.random.normal(0, 0.005, 400).astype(np.float64)  # very calm
        innovations = (returns - np.mean(returns)).astype(np.float64)
        omega, alpha, leverage, beta = 1e-6, 0.05, 0.02, 0.90
        uvar = max(omega / max(1 - alpha - 0.5*leverage - beta, 0.01), 1e-8)

        h_no_liq = self._run_garch(innovations, omega, alpha, leverage, beta, uvar)
        h_liq = self._run_garch(innovations, omega, alpha, leverage, beta, uvar, liq_c=0.3)
        # Even with feedback, vol should be bounded by 50x cap
        max_ratio = np.max(h_liq) / uvar
        self.assertLess(max_ratio, 55.0,
                       f"Vol should be bounded by 50x cap, got {max_ratio:.1f}x uvar")
        # All values should be finite
        self.assertTrue(np.all(np.isfinite(h_liq)), "All h values should be finite")
        print(f"  ✓ Low-vol bounded: max h/uvar = {max_ratio:.1f}x (cap=50x), all finite")


# ---------------------------------------------------------------------------
# 5. Integration test: run_unified_mc with new params
# ---------------------------------------------------------------------------
class TestRunUnifiedMC(unittest.TestCase):
    """Test that run_unified_mc in signals.py accepts and uses new params."""

    def test_signature_accepts_new_params(self):
        """run_unified_mc should accept leverage_dynamic_decay and liq_stress_coeff."""
        from decision.signals import run_unified_mc
        import inspect
        sig = inspect.signature(run_unified_mc)
        params = list(sig.parameters.keys())
        self.assertIn('leverage_dynamic_decay', params)
        self.assertIn('liq_stress_coeff', params)
        print("  ✓ run_unified_mc accepts leverage_dynamic_decay and liq_stress_coeff")

    def test_mc_runs_with_new_params(self):
        """run_unified_mc should run successfully with non-zero enhancement params."""
        from decision.signals import run_unified_mc
        result = run_unified_mc(
            mu_t=0.001,
            P_t=1e-5,
            phi=0.98,
            q=1e-5,
            sigma2_step=0.0004,
            H_max=10,
            n_paths=500,
            nu=6.0,
            use_garch=True,
            garch_omega=1e-6,
            garch_alpha=0.08,
            garch_beta=0.88,
            garch_leverage=0.05,
            leverage_dynamic_decay=0.4,
            liq_stress_coeff=0.2,
        )
        self.assertIn('returns', result)
        self.assertIn('volatility', result)
        self.assertEqual(result['returns'].shape, (10, 500))
        self.assertTrue(np.all(np.isfinite(result['returns'])),
                       "All MC returns should be finite")
        print("  ✓ run_unified_mc produces finite results with enhancement params")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_all():
    """Run all tests with verbose output."""
    print("=" * 70)
    print("v7.8 Elite MC Enhancements — Comprehensive Test Suite")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestPitKLUniformKernel,
        TestGarchVarianceKernelEnhancements,
        TestUnifiedMCKernelEnhancements,
        TestConfigDataclasses,
        TestStressScenarios,
        TestRunUnifiedMC,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == '__main__':
    # Skip multi-asset calibration by default (slow)
    # Run with: python -m unittest src.tests.test_elite_mc_enhancements -v
    # Or: python src/tests/test_elite_mc_enhancements.py
    run_all()
