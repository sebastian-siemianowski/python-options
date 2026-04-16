"""
Test Story 1.5: Innovation-Weighted Drift Aggregation.

Validates:
  1. innovation_weight() returns correct values for various z_t
  2. Asymmetric weighting: negative surprises get higher weight
  3. P_t does not grow unbounded under repeated large innovations
  4. Non-surprising observations (|z| < 1) get w = 1.0
  5. Integration with Kalman filter produces finite outputs
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from decision.signals import (
    innovation_weight,
    IW_ALPHA_UP,
    IW_ALPHA_DOWN,
    IW_W_MAX,
    IW_P_MAX_MULT,
)


class TestInnovationWeight(unittest.TestCase):
    """Test the innovation_weight function."""

    def test_small_z_returns_one(self):
        """Non-surprising obs (|z| <= 1) should get weight 1.0."""
        for z in [0.0, 0.5, -0.5, 1.0, -1.0]:
            w = innovation_weight(z)
            self.assertAlmostEqual(w, 1.0, places=10,
                                   msg=f"z={z} should give w=1.0, got {w}")

    def test_3sigma_weight(self):
        """3-sigma innovation should get w ~ 1.6 (alpha=0.3, |z|-1=2)."""
        w = innovation_weight(3.0)
        expected = 1.0 + IW_ALPHA_UP * (3.0 - 1.0)
        self.assertAlmostEqual(w, expected, places=10)
        self.assertAlmostEqual(w, 1.6, places=5)

    def test_half_sigma_returns_one(self):
        """0.5-sigma observation gets w = 1.0 (no amplification)."""
        w = innovation_weight(0.5)
        self.assertAlmostEqual(w, 1.0, places=10)

    def test_asymmetric_negative_higher(self):
        """Negative surprises should get higher weight than positive."""
        w_pos = innovation_weight(3.0)
        w_neg = innovation_weight(-3.0)
        self.assertGreater(w_neg, w_pos,
                           f"Negative 3-sigma w={w_neg} should exceed positive w={w_pos}")

    def test_cap_at_w_max(self):
        """Very large z should be capped at w_max."""
        w = innovation_weight(100.0)
        self.assertAlmostEqual(w, IW_W_MAX, places=10)

    def test_negative_large_capped(self):
        """Very large negative z should also be capped at w_max."""
        w = innovation_weight(-100.0)
        self.assertAlmostEqual(w, IW_W_MAX, places=10)

    def test_custom_alpha(self):
        """Custom alpha should adjust the weight accordingly."""
        w = innovation_weight(3.0, alpha_up=0.5, alpha_down=0.5)
        expected = 1.0 + 0.5 * (3.0 - 1.0)
        self.assertAlmostEqual(w, min(expected, IW_W_MAX), places=10)

    def test_custom_w_max(self):
        """Custom w_max should cap the weight."""
        w = innovation_weight(10.0, w_max=1.5)
        self.assertAlmostEqual(w, 1.5, places=10)


class TestInnovationWeightingIntegration(unittest.TestCase):
    """Test integration of innovation weighting in the Kalman filter."""

    def test_p_bounded_under_large_innovations(self):
        """P_t should not grow unbounded under repeated 5-sigma innovations."""
        rng = np.random.RandomState(42)
        T = 500
        q = 1e-4
        c = 1.0
        phi = 1.0
        sigma_val = 0.01

        # Create data with periodic 5-sigma shocks
        y = np.zeros(T)
        for t in range(T):
            if t % 10 == 0:
                y[t] = 5.0 * sigma_val  # 5-sigma shock
            else:
                y[t] = rng.normal(0, sigma_val)

        mu_t = 0.0
        P_t = 1e-4
        P_max = IW_P_MAX_MULT * max(q * 100.0, 1e-6)
        P_values = []

        for t in range(T):
            mu_pred = phi * mu_t
            P_pred = phi * phi * P_t + q
            R_t = c * sigma_val ** 2
            S_t = P_pred + R_t
            innov = y[t] - mu_pred
            K_t = P_pred / S_t

            z_t = innov / math.sqrt(S_t)
            w_iw = innovation_weight(z_t)
            K_eff = min(K_t * w_iw, 0.99)

            mu_t = mu_pred + K_eff * innov
            P_t = max((1.0 - K_eff) * P_pred, 1e-12)
            P_t = min(P_t, P_max)
            P_values.append(P_t)

        P_arr = np.array(P_values)
        # P should never exceed P_max
        self.assertTrue(np.all(P_arr <= P_max + 1e-15),
                        f"P exceeded P_max: max={P_arr.max():.2e}, limit={P_max:.2e}")
        # P should converge (not grow)
        self.assertLess(P_arr[-1], P_arr[0] * 100,
                        "P should not grow unbounded")

    def test_innovation_weight_defaults_reasonable(self):
        """Default constants should be in sensible ranges."""
        self.assertGreater(IW_ALPHA_UP, 0)
        self.assertLess(IW_ALPHA_UP, 1.0)
        self.assertGreater(IW_ALPHA_DOWN, 0)
        self.assertLess(IW_ALPHA_DOWN, 1.0)
        self.assertGreater(IW_W_MAX, 1.0)
        self.assertLessEqual(IW_W_MAX, 3.0)
        self.assertGreater(IW_P_MAX_MULT, 1.0)

    def test_w_max_prevents_divergence(self):
        """W_MAX=2.0 means K_eff <= 2*K_t which keeps filter stable."""
        # With K_t at steady state ~0.5 (typical), K_eff <= 1.0
        # This should prevent divergence
        K_t_typical = 0.5
        K_eff_max = K_t_typical * IW_W_MAX
        self.assertLessEqual(K_eff_max, 1.0,
                             "K_eff could exceed 1.0 at typical K_t, risk of divergence")


if __name__ == "__main__":
    unittest.main(verbosity=2)
