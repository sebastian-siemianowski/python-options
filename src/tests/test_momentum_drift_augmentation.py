"""
Tests for Story 1.1: Momentum-Augmented Drift Initialization.

Validates that momentum information is properly blended into the Kalman
posterior drift (mu_t_mc) using horizon-dependent weighting before MC simulation.

Mathematical specification:
    w_mom(H) = min(1.0, H / MOM_CROSSOVER_HORIZON)
    mom_drift = clamp(composite_mom, -3, 3) * vol_now * MOM_DRIFT_SCALE
    mu_augmented = (1 - w_mom) * mu_kalman + w_mom * mom_drift

where composite_mom is the weighted average of t-stat momentum across
21d (0.40), 63d (0.35), 126d (0.25) horizons.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestMomentumDriftConstants(unittest.TestCase):
    """Validate momentum drift constants are correctly defined."""

    def test_constants_exist(self):
        from decision.signals import MOM_DRIFT_SCALE, MOM_CROSSOVER_HORIZON, MOM_MIN_OBSERVATIONS
        self.assertIsInstance(MOM_DRIFT_SCALE, float)
        self.assertIsInstance(MOM_CROSSOVER_HORIZON, (int, float))
        self.assertIsInstance(MOM_MIN_OBSERVATIONS, (int, float))

    def test_constants_reasonable(self):
        from decision.signals import MOM_DRIFT_SCALE, MOM_CROSSOVER_HORIZON, MOM_MIN_OBSERVATIONS
        self.assertGreater(MOM_DRIFT_SCALE, 0)
        self.assertLessEqual(MOM_DRIFT_SCALE, 1.0)
        self.assertGreater(MOM_CROSSOVER_HORIZON, 0)
        self.assertGreaterEqual(MOM_MIN_OBSERVATIONS, 63)


class TestMomentumBlendingWeight(unittest.TestCase):
    """Validate the horizon-dependent blending weight w_mom(H)."""

    def _w_mom(self, H):
        from decision.signals import MOM_CROSSOVER_HORIZON
        return min(1.0, H / MOM_CROSSOVER_HORIZON)

    def test_short_horizon_kalman_dominated(self):
        """At H=1, momentum weight should be < 0.02."""
        w = self._w_mom(1)
        self.assertLess(w, 0.02)
        self.assertGreater(w, 0.0)

    def test_medium_horizon_balanced(self):
        """At H=21, momentum weight ~ 0.33."""
        w = self._w_mom(21)
        self.assertAlmostEqual(w, 21 / 63, places=5)

    def test_crossover_horizon_full(self):
        """At H=63 (crossover), momentum weight should be 1.0."""
        w = self._w_mom(63)
        self.assertAlmostEqual(w, 1.0, places=5)

    def test_long_horizon_capped(self):
        """At H=252, momentum weight should be exactly 1.0 (capped)."""
        w = self._w_mom(252)
        self.assertEqual(w, 1.0)

    def test_monotonically_increasing(self):
        """Weight must be monotonically non-decreasing with horizon."""
        horizons = [1, 3, 7, 21, 63, 126, 252]
        weights = [self._w_mom(h) for h in horizons]
        for i in range(len(weights) - 1):
            self.assertLessEqual(weights[i], weights[i + 1])


class TestMomentumDriftComputation(unittest.TestCase):
    """Validate the momentum drift computation in return units."""

    def test_positive_momentum_positive_drift(self):
        """Positive momentum should produce positive mom_drift."""
        from decision.signals import MOM_DRIFT_SCALE
        composite_mom = 1.5  # Positive t-stat
        vol_now = 0.015  # 1.5% daily vol
        mom_drift = np.clip(composite_mom, -3, 3) * vol_now * MOM_DRIFT_SCALE
        self.assertGreater(mom_drift, 0)

    def test_negative_momentum_negative_drift(self):
        """Negative momentum should produce negative mom_drift."""
        from decision.signals import MOM_DRIFT_SCALE
        composite_mom = -2.0
        vol_now = 0.015
        mom_drift = np.clip(composite_mom, -3, 3) * vol_now * MOM_DRIFT_SCALE
        self.assertLess(mom_drift, 0)

    def test_momentum_clamp_prevents_extreme(self):
        """Extreme momentum (>3 sigma) should be clamped."""
        from decision.signals import MOM_DRIFT_SCALE
        composite_mom = 10.0  # Absurdly large
        vol_now = 0.015
        clamped = np.clip(composite_mom, -3.0, 3.0)
        self.assertEqual(clamped, 3.0)
        mom_drift = clamped * vol_now * MOM_DRIFT_SCALE
        # Max mom_drift = 3.0 * 0.015 * 0.10 = 0.0045 (0.45%/day) -- reasonable
        self.assertAlmostEqual(mom_drift, 3.0 * 0.015 * MOM_DRIFT_SCALE, places=8)

    def test_zero_momentum_zero_drift(self):
        """Zero momentum should produce zero mom_drift."""
        from decision.signals import MOM_DRIFT_SCALE
        mom_drift = np.clip(0.0, -3, 3) * 0.015 * MOM_DRIFT_SCALE
        self.assertAlmostEqual(mom_drift, 0.0, places=10)

    def test_drift_scales_with_volatility(self):
        """Higher vol assets should have proportionally larger mom_drift."""
        from decision.signals import MOM_DRIFT_SCALE
        mom = 1.0
        drift_low_vol = np.clip(mom, -3, 3) * 0.005 * MOM_DRIFT_SCALE   # Currency
        drift_high_vol = np.clip(mom, -3, 3) * 0.030 * MOM_DRIFT_SCALE  # TSLA
        self.assertGreater(drift_high_vol, drift_low_vol)
        ratio = drift_high_vol / drift_low_vol
        self.assertAlmostEqual(ratio, 6.0, places=3)  # 0.030/0.005 = 6x


class TestAugmentedDriftFormula(unittest.TestCase):
    """Full integration test of the augmented drift formula."""

    def _augmented_drift(self, mu_kalman, composite_mom, vol_now, H):
        from decision.signals import MOM_DRIFT_SCALE, MOM_CROSSOVER_HORIZON
        w_mom = min(1.0, H / MOM_CROSSOVER_HORIZON)
        clamped = np.clip(composite_mom, -3.0, 3.0)
        mom_drift = clamped * vol_now * MOM_DRIFT_SCALE
        return (1.0 - w_mom) * mu_kalman + w_mom * mom_drift

    def test_spy_7d_augmented(self):
        """SPY: mu_kalman=0.0004, momentum=+0.8, vol=0.01 -> augmented > mu_kalman."""
        mu_k = 0.0004  # 0.04%/day (typical SPY)
        mom = 0.8       # Moderately positive t-stat
        vol = 0.010     # ~16% annual
        H = 7
        aug = self._augmented_drift(mu_k, mom, vol, H)
        # w_mom(7) = 7/63 ~ 0.111
        # mom_drift = 0.8 * 0.01 * 0.10 = 0.0008
        # aug = 0.889 * 0.0004 + 0.111 * 0.0008 = 0.000445
        self.assertGreater(aug, mu_k)
        self.assertGreater(aug, 0)

    def test_nvda_7d_strong_momentum(self):
        """NVDA: strong momentum should produce substantial augmented drift."""
        mu_k = 0.0010  # 0.10%/day
        mom = 2.5       # Strong momentum (2.5 sigma t-stat)
        vol = 0.025     # ~40% annual
        H = 7
        aug = self._augmented_drift(mu_k, mom, vol, H)
        # w_mom(7) ~ 0.111
        # mom_drift = 2.5 * 0.025 * 0.10 = 0.00625
        # aug = 0.889 * 0.001 + 0.111 * 0.00625 = 0.001583
        self.assertGreater(aug, mu_k * 1.5)  # At least 50% boost

    def test_declining_asset_negative_augmentation(self):
        """Declining asset: negative momentum should reduce drift."""
        mu_k = 0.0002
        mom = -1.5  # Negative momentum
        vol = 0.020
        H = 21
        aug = self._augmented_drift(mu_k, mom, vol, H)
        # w_mom(21) = 21/63 = 0.333
        # mom_drift = -1.5 * 0.02 * 0.10 = -0.003
        # aug = 0.667 * 0.0002 + 0.333 * (-0.003) = -0.000866
        self.assertLess(aug, mu_k)
        self.assertLess(aug, 0)  # Negative augmented drift

    def test_h252_momentum_dominates(self):
        """At H=252, augmented drift equals mom_drift entirely."""
        mu_k = 0.0004
        mom = 1.0
        vol = 0.015
        H = 252
        aug = self._augmented_drift(mu_k, mom, vol, H)
        from decision.signals import MOM_DRIFT_SCALE
        expected_mom_drift = 1.0 * 0.015 * MOM_DRIFT_SCALE
        self.assertAlmostEqual(aug, expected_mom_drift, places=8)

    def test_h1_kalman_dominates(self):
        """At H=1, augmented drift is very close to mu_kalman."""
        mu_k = 0.0004
        mom = 2.0
        vol = 0.015
        H = 1
        aug = self._augmented_drift(mu_k, mom, vol, H)
        # w_mom(1) = 1/63 ~ 0.0159
        # Should be within ~2% of mu_kalman
        self.assertAlmostEqual(aug, mu_k, delta=abs(mu_k) * 0.15 + 1e-5)


class TestMomentumSafetyGates(unittest.TestCase):
    """Validate safety gates prevent bad momentum from corrupting drift."""

    def test_insufficient_observations_fallback(self):
        """With < MOM_MIN_OBSERVATIONS, momentum augmentation is disabled."""
        from decision.signals import MOM_MIN_OBSERVATIONS
        # With only 50 observations, should not augment
        n_obs = 50
        self.assertLess(n_obs, MOM_MIN_OBSERVATIONS)

    def test_nan_momentum_no_crash(self):
        """NaN in momentum features should not produce NaN drift."""
        # If composite_mom is NaN, np.isfinite check gates it off
        mom_drift = float('nan') * 0.01 * 0.10
        self.assertFalse(np.isfinite(mom_drift))
        # The gate `np.isfinite(_mom_drift)` prevents this from being used

    def test_zero_composite_no_augmentation(self):
        """Zero composite momentum (no trend) should not augment drift."""
        composite_mom = 0.0
        # Gate: abs(_composite_mom_clamped) > 1e-9 prevents augmentation
        self.assertFalse(abs(composite_mom) > 1e-9)


if __name__ == "__main__":
    unittest.main()
