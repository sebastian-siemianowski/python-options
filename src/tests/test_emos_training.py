"""
Tests for Story 2.2: EMOS Calibration from Walk-Forward Errors.

Validates CRPS-minimizing EMOS (a + b*mu, c + d*sig) training
on synthetic (forecast, realized) pairs.
"""

import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _make_synthetic_records(n=200, bias=0.002, scale=1.3, rng_seed=42):
    """Create synthetic WalkForwardRecord-like objects with known bias and scale."""
    from decision.signals import WalkForwardRecord
    import pandas as pd

    rng = np.random.RandomState(rng_seed)
    records = []
    for i in range(n):
        f_ret = rng.normal(0.001, 0.01)
        f_sig = abs(rng.normal(0.015, 0.003))
        # Realized = biased + scaled noise
        realized = (f_ret + bias) + rng.normal(0, f_sig * scale)
        hit = (realized > 0) == (f_ret > 0)
        records.append(WalkForwardRecord(
            date_idx=i,
            date=pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            horizon=7,
            forecast_ret=f_ret,
            forecast_p_up=0.5 + f_ret * 10,
            forecast_sig=f_sig,
            realized_ret=realized,
            hit=hit,
            signed_error=realized - f_ret,
            calibration_bucket=5,
        ))
    return records


class TestEmosConstants(unittest.TestCase):
    def test_train_frac(self):
        from tuning.tune import EMOS_TRAIN_FRAC
        self.assertEqual(EMOS_TRAIN_FRAC, 0.70)

    def test_identity(self):
        from tuning.tune import EMOS_IDENTITY
        self.assertEqual(EMOS_IDENTITY["a"], 0.0)
        self.assertEqual(EMOS_IDENTITY["b"], 1.0)
        self.assertEqual(EMOS_IDENTITY["c"], 0.0)
        self.assertEqual(EMOS_IDENTITY["d"], 1.0)


class TestCrpsNormal(unittest.TestCase):
    """Test the CRPS computation for a Normal distribution."""

    def test_zero_at_mean(self):
        from tuning.tune import _crps_normal_single
        # CRPS at y=mu for Normal(mu, sig) = sig * (sqrt(2)-1) / sqrt(pi)
        sig = 1.0
        expected = sig * (np.sqrt(2) - 1) / np.sqrt(np.pi)
        actual = _crps_normal_single(0.0, 0.0, sig)
        self.assertAlmostEqual(actual, expected, places=4)

    def test_positive(self):
        from tuning.tune import _crps_normal_single
        c = _crps_normal_single(1.0, 0.0, 1.0)
        self.assertGreater(c, 0)

    def test_zero_sigma(self):
        from tuning.tune import _crps_normal_single
        # Zero sigma -> absolute error
        c = _crps_normal_single(1.0, 0.5, 0.0)
        self.assertAlmostEqual(c, 0.5, places=6)


class TestEmosTraining(unittest.TestCase):
    """Test train_emos_parameters on synthetic biased forecasts."""

    @classmethod
    def setUpClass(cls):
        from tuning.tune import train_emos_parameters
        cls.records = _make_synthetic_records(n=300, bias=0.002, scale=1.3)
        cls.results = train_emos_parameters(cls.records, horizons=[7])

    def test_returns_dict(self):
        self.assertIn(7, self.results)

    def test_a_nonzero(self):
        """EMOS 'a' should capture the systematic bias."""
        r = self.results[7]
        # With bias=0.002, 'a' should be positive (capturing drift bias)
        self.assertNotAlmostEqual(r["a"], 0.0, places=5,
                                  msg=f"a={r['a']} should be non-zero (bias)")

    def test_b_near_one(self):
        """EMOS 'b' should be near 1 (forecasts aren't systematically biased in slope)."""
        r = self.results[7]
        self.assertGreater(r["b"], 0.5)
        self.assertLess(r["b"], 2.0)

    def test_d_captures_scale(self):
        """EMOS 'd' should compensate for the sigma miscalibration."""
        r = self.results[7]
        # With scale=1.3, d should be > 1 to widen the sigma
        self.assertGreater(r["d"], 0.5)

    def test_crps_improvement(self):
        """Corrected CRPS should be <= uncorrected."""
        r = self.results[7]
        self.assertLessEqual(r["crps_train"], r["crps_uncorrected"] + 1e-8)

    def test_train_val_counts(self):
        r = self.results[7]
        self.assertGreater(r["n_train"], 0)
        self.assertGreater(r["n_val"], 0)
        self.assertEqual(r["n_train"] + r["n_val"], len(self.records))


class TestEmosIdentity(unittest.TestCase):
    """Test that perfectly calibrated forecasts produce near-identity params."""

    def test_well_calibrated(self):
        from tuning.tune import train_emos_parameters
        from decision.signals import WalkForwardRecord
        import pandas as pd

        rng = np.random.RandomState(99)
        recs = []
        for i in range(200):
            f = rng.normal(0.001, 0.01)
            s = abs(rng.normal(0.015, 0.003))
            y = f + rng.normal(0, s)  # no bias, correct sigma
            recs.append(WalkForwardRecord(
                date_idx=i, date=pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                horizon=1, forecast_ret=f, forecast_p_up=0.5, forecast_sig=s,
                realized_ret=y, hit=(y > 0) == (f > 0), signed_error=y - f,
                calibration_bucket=5,
            ))
        result = train_emos_parameters(recs, horizons=[1])
        r = result[1]
        # Should be close to identity
        self.assertAlmostEqual(r["a"], 0.0, delta=0.005)
        self.assertAlmostEqual(r["b"], 1.0, delta=0.3)


class TestEmosTooFewRecords(unittest.TestCase):
    """With < 20 records, should return identity."""

    def test_identity_fallback(self):
        from tuning.tune import train_emos_parameters
        from decision.signals import WalkForwardRecord
        import pandas as pd

        recs = [WalkForwardRecord(
            date_idx=i, date=pd.Timestamp("2024-01-01"),
            horizon=7, forecast_ret=0.01, forecast_p_up=0.6, forecast_sig=0.01,
            realized_ret=0.005, hit=True, signed_error=-0.005, calibration_bucket=5,
        ) for i in range(5)]
        result = train_emos_parameters(recs, horizons=[7])
        r = result[7]
        self.assertEqual(r["a"], 0.0)
        self.assertEqual(r["b"], 1.0)


class TestEmosNoFlipDirection(unittest.TestCase):
    """EMOS should not flip forecast direction for >95% of records."""

    def test_direction_preservation(self):
        from tuning.tune import train_emos_parameters
        records = _make_synthetic_records(n=300, bias=0.001, scale=1.1)
        results = train_emos_parameters(records, horizons=[7])
        r = results[7]
        a, b = r["a"], r["b"]
        flips = 0
        total = 0
        for rec in records:
            mu_raw = rec.forecast_ret
            mu_cor = a + b * mu_raw
            if abs(mu_raw) > 0.002:  # Exclude near-zero forecasts
                total += 1
                if np.sign(mu_cor) != np.sign(mu_raw):
                    flips += 1
        if total > 0:
            flip_rate = flips / total
            self.assertLess(flip_rate, 0.05,
                            f"EMOS flips direction for {flip_rate:.1%} of forecasts")


if __name__ == "__main__":
    unittest.main()
