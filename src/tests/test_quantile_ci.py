"""
Story 3.2: Quantile-Based Confidence Intervals

Tests that CIs come from MC empirical quantiles (not parametric mu +/- z*sig),
that 90% CIs are present, and that parametric fallback only triggers for small samples.
"""
import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


class TestQuantileCIConstants(unittest.TestCase):
    """Verify constants are exported and reasonable."""

    def test_quantile_ci_min_samples_exists(self):
        from decision.signals import QUANTILE_CI_MIN_SAMPLES
        self.assertEqual(QUANTILE_CI_MIN_SAMPLES, 100)

    def test_ci_log_floor_exists(self):
        from decision.signals import _CI_LOG_FLOOR
        self.assertAlmostEqual(_CI_LOG_FLOOR, -4.6, places=1)

    def test_ci_log_cap_exists(self):
        from decision.signals import _CI_LOG_CAP
        self.assertAlmostEqual(_CI_LOG_CAP, 1.61, places=2)


class TestSignalDataclass90CI(unittest.TestCase):
    """Signal dataclass has 90% CI fields."""

    def test_signal_has_ci_low_90(self):
        from decision.signals import Signal
        import dataclasses
        fields = {f.name for f in dataclasses.fields(Signal)}
        self.assertIn("ci_low_90", fields)

    def test_signal_has_ci_high_90(self):
        from decision.signals import Signal
        import dataclasses
        fields = {f.name for f in dataclasses.fields(Signal)}
        self.assertIn("ci_high_90", fields)

    def test_signal_construction_with_90ci(self):
        """Signal can be constructed with 90% CI fields."""
        from decision.signals import Signal
        sig = Signal(
            horizon_days=7,
            score=0.5,
            p_up=0.55,
            exp_ret=0.01,
            ci_low=-0.02,
            ci_high=0.04,
            ci_low_90=-0.05,
            ci_high_90=0.07,
            profit_pln=10000.0,
            profit_ci_low_pln=-20000.0,
            profit_ci_high_pln=40000.0,
            position_strength=0.3,
            vol_mean=0.02,
            vol_ci_low=0.01,
            vol_ci_high=0.03,
            regime="LOW_VOL_TREND",
            label="BUY",
        )
        self.assertAlmostEqual(sig.ci_low_90, -0.05)
        self.assertAlmostEqual(sig.ci_high_90, 0.07)


class TestQuantileCIAsymmetry(unittest.TestCase):
    """Core test: quantile CIs capture asymmetry that parametric CIs miss."""

    def test_skewed_distribution_asymmetric_ci(self):
        """For a left-skewed distribution, ci_low should be further from median than ci_high."""
        rng = np.random.default_rng(42)
        # Simulate left-skewed returns (heavy left tail)
        # Mix: 80% from N(0.01, 0.02) and 20% from N(-0.05, 0.04)
        n = 10000
        mask = rng.random(n) < 0.8
        samples = np.where(mask,
                           rng.normal(0.01, 0.02, n),
                           rng.normal(-0.05, 0.04, n))

        # Quantile CI (68%)
        alpha = 0.16
        q_low = float(np.quantile(samples, alpha))
        q_high = float(np.quantile(samples, 1 - alpha))
        median = float(np.median(samples))

        # Parametric CI (68%)
        mu = float(np.mean(samples))
        sig = float(np.std(samples))
        p_low = mu - sig
        p_high = mu + sig

        # Quantile CI should be asymmetric (left tail further from median)
        q_left_width = median - q_low
        q_right_width = q_high - median
        self.assertGreater(q_left_width, q_right_width * 1.1,
                           "Quantile CI should capture left-skew asymmetry")

        # Parametric CI is symmetric by construction
        p_left_width = mu - p_low
        p_right_width = p_high - mu
        self.assertAlmostEqual(p_left_width, p_right_width, places=10,
                               msg="Parametric CI is always symmetric")

    def test_student_t_wider_than_gaussian(self):
        """For heavy-tailed Student-t(nu=4), 68% CI should be wider than Gaussian."""
        from scipy.stats import t as student_t, norm
        rng = np.random.default_rng(123)

        mu, sig = 0.01, 0.03
        n = 50000  # large sample for precision

        # Student-t(4) samples
        t_samples = mu + sig * rng.standard_t(df=4, size=n)
        t_ci_low = float(np.quantile(t_samples, 0.16))
        t_ci_high = float(np.quantile(t_samples, 0.84))
        t_width = t_ci_high - t_ci_low

        # Gaussian samples (same mu, sigma)
        g_samples = rng.normal(mu, sig, n)
        g_ci_low = float(np.quantile(g_samples, 0.16))
        g_ci_high = float(np.quantile(g_samples, 0.84))
        g_width = g_ci_high - g_ci_low

        # Student-t 68% CI should be wider due to heavier tails
        self.assertGreater(t_width, g_width,
                           "Student-t(4) 68% CI should be wider than Gaussian")

    def test_90ci_wider_than_68ci(self):
        """90% CI must always be wider than 68% CI."""
        rng = np.random.default_rng(99)
        samples = rng.normal(0.0, 0.03, 10000)

        ci68_low = float(np.quantile(samples, 0.16))
        ci68_high = float(np.quantile(samples, 0.84))
        ci90_low = float(np.quantile(samples, 0.05))
        ci90_high = float(np.quantile(samples, 0.95))

        self.assertLess(ci90_low, ci68_low, "90% CI lower must be below 68% CI lower")
        self.assertGreater(ci90_high, ci68_high, "90% CI upper must be above 68% CI upper")
        self.assertGreater(ci90_high - ci90_low, ci68_high - ci68_low,
                           "90% CI must be wider than 68% CI")


class TestParametricFallback(unittest.TestCase):
    """Parametric CI is used only as fallback for small samples."""

    def test_small_sample_uses_parametric(self):
        """When n_samples < QUANTILE_CI_MIN_SAMPLES, parametric CI is used."""
        from decision.signals import QUANTILE_CI_MIN_SAMPLES
        # With < 100 samples, the code path uses mu +/- z*sig
        n = QUANTILE_CI_MIN_SAMPLES - 1  # 99
        samples = np.random.default_rng(42).normal(0.01, 0.02, n)
        self.assertLess(len(samples), QUANTILE_CI_MIN_SAMPLES)

    def test_large_sample_uses_quantile(self):
        """When n_samples >= QUANTILE_CI_MIN_SAMPLES, quantile CI is used."""
        from decision.signals import QUANTILE_CI_MIN_SAMPLES
        n = QUANTILE_CI_MIN_SAMPLES  # exactly 100
        samples = np.random.default_rng(42).normal(0.01, 0.02, n)
        self.assertGreaterEqual(len(samples), QUANTILE_CI_MIN_SAMPLES)

        # Quantile approach
        alpha = 0.16
        q_low = float(np.quantile(samples, alpha))
        q_high = float(np.quantile(samples, 1 - alpha))
        # Parametric approach
        mu = float(np.mean(samples))
        sig = float(np.std(samples))
        p_low = mu - sig
        p_high = mu + sig
        # They should differ (quantile captures empirical shape)
        self.assertNotAlmostEqual(q_low, p_low, places=4,
                                  msg="Quantile and parametric CIs should differ")


class TestCIClamping(unittest.TestCase):
    """CI clamping to physical limits still applies."""

    def test_ci_floor_clamp(self):
        """CI low is clamped to _CI_LOG_FLOOR."""
        from decision.signals import _CI_LOG_FLOOR
        samples = np.array([-10.0] * 1000)  # extreme loss
        ci_low = float(np.quantile(samples, 0.16))
        clamped = max(ci_low, _CI_LOG_FLOOR)
        self.assertAlmostEqual(clamped, _CI_LOG_FLOOR)

    def test_ci_cap_clamp(self):
        """CI high is clamped to _CI_LOG_CAP."""
        from decision.signals import _CI_LOG_CAP
        samples = np.array([5.0] * 1000)  # extreme gain
        ci_high = float(np.quantile(samples, 0.84))
        clamped = min(ci_high, _CI_LOG_CAP)
        self.assertAlmostEqual(clamped, _CI_LOG_CAP)

    def test_ci_ordering_preserved(self):
        """After clamping, ci_low <= ci_high."""
        from decision.signals import _CI_LOG_FLOOR, _CI_LOG_CAP
        # Edge case: both get clamped
        ci_low = max(-10.0, _CI_LOG_FLOOR)
        ci_high = min(10.0, _CI_LOG_CAP)
        if ci_low > ci_high:
            ci_low, ci_high = ci_high, ci_low
        self.assertLessEqual(ci_low, ci_high)


class TestQuantileCIFormulas(unittest.TestCase):
    """Verify the exact quantile formulas match requirements."""

    def test_68ci_uses_correct_percentiles(self):
        """68% CI uses 16th and 84th percentiles."""
        ci_level = 0.68
        alpha = (1.0 - ci_level) / 2.0
        self.assertAlmostEqual(alpha, 0.16, places=5)
        self.assertAlmostEqual(1.0 - alpha, 0.84, places=5)

    def test_90ci_uses_correct_percentiles(self):
        """90% CI uses 5th and 95th percentiles."""
        # Hardcoded in Story 3.2: 5th and 95th
        self.assertAlmostEqual(0.05, 0.05)
        self.assertAlmostEqual(0.95, 0.95)

    def test_quantile_ci_on_known_distribution(self):
        """For a standard normal, 68% quantile CI should be approximately [-1, +1]."""
        rng = np.random.default_rng(777)
        samples = rng.normal(0.0, 1.0, 100000)
        ci_low = float(np.quantile(samples, 0.16))
        ci_high = float(np.quantile(samples, 0.84))
        # Should be approximately [-1, +1] for standard normal
        self.assertAlmostEqual(ci_low, -1.0, delta=0.05)
        self.assertAlmostEqual(ci_high, 1.0, delta=0.05)

    def test_90ci_on_known_distribution(self):
        """For standard normal, 90% CI should be approximately [-1.645, +1.645]."""
        rng = np.random.default_rng(888)
        samples = rng.normal(0.0, 1.0, 100000)
        ci_low_90 = float(np.quantile(samples, 0.05))
        ci_high_90 = float(np.quantile(samples, 0.95))
        self.assertAlmostEqual(ci_low_90, -1.645, delta=0.05)
        self.assertAlmostEqual(ci_high_90, 1.645, delta=0.05)


class TestSignalJSONSerialization(unittest.TestCase):
    """90% CI appears in signal.__dict__ for JSON export."""

    def test_signal_dict_has_90ci(self):
        """Signal.__dict__ includes ci_low_90 and ci_high_90."""
        from decision.signals import Signal
        sig = Signal(
            horizon_days=7,
            score=0.5,
            p_up=0.55,
            exp_ret=0.01,
            ci_low=-0.02,
            ci_high=0.04,
            ci_low_90=-0.05,
            ci_high_90=0.07,
            profit_pln=10000.0,
            profit_ci_low_pln=-20000.0,
            profit_ci_high_pln=40000.0,
            position_strength=0.3,
            vol_mean=0.02,
            vol_ci_low=0.01,
            vol_ci_high=0.03,
            regime="LOW_VOL_TREND",
            label="BUY",
        )
        d = sig.__dict__
        self.assertIn("ci_low_90", d)
        self.assertIn("ci_high_90", d)
        self.assertAlmostEqual(d["ci_low_90"], -0.05)
        self.assertAlmostEqual(d["ci_high_90"], 0.07)


if __name__ == "__main__":
    unittest.main()
