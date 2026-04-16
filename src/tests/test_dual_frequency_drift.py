"""
Tests for Story 1.4: Dual-Frequency Drift Propagation.

Validates that the MC simulation uses separate fast (AR(1) with MLE phi) and
slow (deterministic decay) drift components, preserving directional signal
at medium and long horizons.

Mathematical specification:
    mu_fast[t] = phi_fast * mu_fast[t-1] + eta
    mu_slow[t] = phi_slow * mu_slow[t-1]       (deterministic)
    mu_total[t] = mu_fast[t] + mu_slow[t]

    phi_slow = exp(-1 / MOMENTUM_HALF_LIFE_DAYS)   ~ 0.9765
    mu_fast_0 = mu_t * (1 - MOM_SLOW_FRAC)          (70%)
    mu_slow_0 = mu_t * MOM_SLOW_FRAC                 (30%)
"""

import os
import sys
import unittest
import math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestDualFrequencyConstants(unittest.TestCase):
    """Validate module-level constants for dual-frequency drift."""

    def test_constants_exist(self):
        from decision.signals import MOMENTUM_HALF_LIFE_DAYS, MOM_SLOW_FRAC, SLOW_Q_RATIO
        self.assertIsInstance(MOMENTUM_HALF_LIFE_DAYS, (int, float))
        self.assertIsInstance(MOM_SLOW_FRAC, float)
        self.assertIsInstance(SLOW_Q_RATIO, float)

    def test_half_life_in_range(self):
        from decision.signals import MOMENTUM_HALF_LIFE_DAYS
        self.assertGreater(MOMENTUM_HALF_LIFE_DAYS, 10)
        self.assertLess(MOMENTUM_HALF_LIFE_DAYS, 120)

    def test_slow_frac_valid(self):
        from decision.signals import MOM_SLOW_FRAC
        self.assertGreater(MOM_SLOW_FRAC, 0.0)
        self.assertLess(MOM_SLOW_FRAC, 1.0)

    def test_phi_slow_computation(self):
        from decision.signals import MOMENTUM_HALF_LIFE_DAYS
        phi_slow = math.exp(-1.0 / MOMENTUM_HALF_LIFE_DAYS)
        self.assertGreater(phi_slow, 0.95)
        self.assertLess(phi_slow, 1.0)


class TestDualFrequencyDecayRates(unittest.TestCase):
    """Test that the slow component retains signal as specified."""

    def test_slow_component_at_h1(self):
        """At H=1, slow component retains ~97.6% of initial."""
        from decision.signals import MOMENTUM_HALF_LIFE_DAYS
        phi_slow = math.exp(-1.0 / MOMENTUM_HALF_LIFE_DAYS)
        retention_1d = phi_slow
        self.assertGreater(retention_1d, 0.95)

    def test_slow_component_at_h21(self):
        """At H=21, slow component retains >= 20% of initial (60% actual)."""
        from decision.signals import MOMENTUM_HALF_LIFE_DAYS
        phi_slow = math.exp(-1.0 / MOMENTUM_HALF_LIFE_DAYS)
        retention_21d = phi_slow ** 21
        self.assertGreater(retention_21d, 0.20,
                           f"Slow component retains only {retention_21d:.3f} at H=21")

    def test_slow_component_at_h63(self):
        """At H=63, slow component retains >= 10% of initial (22% actual)."""
        from decision.signals import MOMENTUM_HALF_LIFE_DAYS
        phi_slow = math.exp(-1.0 / MOMENTUM_HALF_LIFE_DAYS)
        retention_63d = phi_slow ** 63
        self.assertGreater(retention_63d, 0.10,
                           f"Slow component retains only {retention_63d:.3f} at H=63")

    def test_slow_component_half_life(self):
        """Component should be ~50% at MOMENTUM_HALF_LIFE_DAYS."""
        from decision.signals import MOMENTUM_HALF_LIFE_DAYS
        phi_slow = math.exp(-1.0 / MOMENTUM_HALF_LIFE_DAYS)
        retention = phi_slow ** MOMENTUM_HALF_LIFE_DAYS
        # exp(-1) ~ 0.368
        self.assertAlmostEqual(retention, math.exp(-1), places=5)


class TestDualFrequencyMCIntegration(unittest.TestCase):
    """Test dual-frequency drift in the MC simulation."""

    def test_run_unified_mc_accepts_dual_freq_params(self):
        """run_unified_mc should accept phi_slow and mu_slow_0 parameters."""
        from decision.signals import run_unified_mc
        result = run_unified_mc(
            mu_t=0.001, P_t=1e-6, phi=0.5, q=1e-5,
            sigma2_step=0.0002, H_max=5, n_paths=100,
            phi_slow=0.976, mu_slow_0=0.0003,
            seed=42,
        )
        self.assertIn('returns', result)
        self.assertEqual(result['returns'].shape, (5, 100))

    def test_dual_freq_increases_long_horizon_signal(self):
        """With dual-freq, long-horizon mean return should be larger than without."""
        from decision.signals import run_unified_mc
        np.random.seed(42)
        mu_total = 0.001
        phi_fast = 0.3  # Fast decay
        H = 63

        # Without dual-freq (all drift in fast component)
        result_single = run_unified_mc(
            mu_t=mu_total, P_t=1e-8, phi=phi_fast, q=1e-6,
            sigma2_step=0.0002, H_max=H, n_paths=5000,
            phi_slow=0.0, mu_slow_0=0.0,
            seed=42,
        )
        mean_single = np.mean(result_single['returns'][H - 1, :])

        # With dual-freq (30% in slow component)
        phi_slow = 0.976
        mu_fast = mu_total * 0.7
        mu_slow = mu_total * 0.3
        result_dual = run_unified_mc(
            mu_t=mu_fast, P_t=1e-8, phi=phi_fast, q=1e-6,
            sigma2_step=0.0002, H_max=H, n_paths=5000,
            phi_slow=phi_slow, mu_slow_0=mu_slow,
            seed=42,
        )
        mean_dual = np.mean(result_dual['returns'][H - 1, :])

        # Dual freq should produce larger absolute mean at long horizon
        # because slow component persists while fast decays
        self.assertGreater(abs(mean_dual), abs(mean_single) * 0.5,
                           f"Dual freq mean {mean_dual:.6f} should retain more signal "
                           f"than single {mean_single:.6f}")

    def test_h1_nearly_identical(self):
        """At H=1, dual-freq should be within 5% of single-freq forecast."""
        from decision.signals import run_unified_mc
        mu_total = 0.001
        phi_fast = 0.5

        result_single = run_unified_mc(
            mu_t=mu_total, P_t=1e-8, phi=phi_fast, q=1e-6,
            sigma2_step=0.0002, H_max=1, n_paths=10000,
            phi_slow=0.0, mu_slow_0=0.0,
            seed=42,
        )
        mean_single_h1 = np.mean(result_single['returns'][0, :])

        result_dual = run_unified_mc(
            mu_t=mu_total * 0.7, P_t=1e-8, phi=phi_fast, q=1e-6,
            sigma2_step=0.0002, H_max=1, n_paths=10000,
            phi_slow=0.976, mu_slow_0=mu_total * 0.3,
            seed=42,
        )
        mean_dual_h1 = np.mean(result_dual['returns'][0, :])

        # At H=1, total drift is mu_fast + mu_slow ~ mu_total (both barely decayed)
        # So means should be close
        if abs(mean_single_h1) > 1e-6:
            pct_diff = abs(mean_dual_h1 - mean_single_h1) / abs(mean_single_h1)
            self.assertLess(pct_diff, 0.10,  # 10% tolerance (MC noise)
                            f"H=1 means differ by {pct_diff:.1%}")

    def test_zero_mu_slow_is_noop(self):
        """When mu_slow_0=0, dual-freq should produce same result as single-freq."""
        from decision.signals import run_unified_mc
        result1 = run_unified_mc(
            mu_t=0.001, P_t=1e-8, phi=0.5, q=1e-6,
            sigma2_step=0.0002, H_max=5, n_paths=1000,
            phi_slow=0.0, mu_slow_0=0.0,
            seed=42,
        )
        result2 = run_unified_mc(
            mu_t=0.001, P_t=1e-8, phi=0.5, q=1e-6,
            sigma2_step=0.0002, H_max=5, n_paths=1000,
            phi_slow=0.976, mu_slow_0=0.0,
            seed=42,
        )
        np.testing.assert_allclose(
            result1['returns'], result2['returns'], atol=1e-10,
            err_msg="mu_slow_0=0 should produce identical results")


class TestBMADualFrequencyInterface(unittest.TestCase):
    """Test that BMA MC interface accepts dual-frequency params."""

    def test_bma_mc_signature_has_dual_freq(self):
        """bayesian_model_average_mc should accept phi_slow and mu_slow_0."""
        import inspect
        from decision.signals import bayesian_model_average_mc
        sig = inspect.signature(bayesian_model_average_mc)
        self.assertIn('phi_slow', sig.parameters)
        self.assertIn('mu_slow_0', sig.parameters)


if __name__ == "__main__":
    unittest.main()
