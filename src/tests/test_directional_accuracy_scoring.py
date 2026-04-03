"""
Tests for Story 2.4: Directional Information Gain (DIG) for BMA Weight Adjustment.

Validates:
    - DIG computation from walk-forward records
    - Adaptive DIG weight scaling
    - BMA weight adjustment with DIG
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


def _make_records(n, horizon, hit_rate, rng_seed=42):
    """Create WalkForwardRecord list with controlled hit rate."""
    from decision.signals import WalkForwardRecord
    rng = np.random.RandomState(rng_seed)
    recs = []
    for i in range(n):
        is_hit = rng.random() < hit_rate
        f_ret = rng.normal(0.001, 0.01)
        if is_hit:
            r_sign = np.sign(f_ret) if f_ret != 0 else 1.0
        else:
            r_sign = -np.sign(f_ret) if f_ret != 0 else -1.0
        y = r_sign * abs(rng.normal(0.005, 0.01))
        recs.append(WalkForwardRecord(
            date_idx=i, date=pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            horizon=horizon, forecast_ret=f_ret, forecast_p_up=0.5,
            forecast_sig=0.02, realized_ret=y,
            hit=is_hit, signed_error=y - f_ret,
            calibration_bucket=5,
        ))
    return recs


class TestDIGConstants(unittest.TestCase):
    def test_constants(self):
        from tuning.tune import (DIG_BASELINE, DIG_W_START, DIG_W_MAX,
                                 DIG_MIN_RECORDS, DIG_FULL_DATA_THRESHOLD)
        self.assertEqual(DIG_BASELINE, 0.5)
        self.assertEqual(DIG_W_START, 0.10)
        self.assertEqual(DIG_W_MAX, 0.25)
        self.assertEqual(DIG_MIN_RECORDS, 30)
        self.assertEqual(DIG_FULL_DATA_THRESHOLD, 200)


class TestDIGComputation(unittest.TestCase):
    def test_positive_dig_from_good_model(self):
        """80% hit rate -> DIG = +0.30"""
        from tuning.tune import compute_dig_per_model
        recs = _make_records(200, horizon=7, hit_rate=0.80)
        dig = compute_dig_per_model(recs, horizons=[7])
        self.assertIn("ensemble", dig)
        self.assertGreater(dig["ensemble"], 0.20)

    def test_random_dig_near_zero(self):
        """50% hit rate -> DIG near 0"""
        from tuning.tune import compute_dig_per_model
        recs = _make_records(500, horizon=7, hit_rate=0.50)
        dig = compute_dig_per_model(recs, horizons=[7])
        self.assertAlmostEqual(dig["ensemble"], 0.0, delta=0.05)

    def test_negative_dig(self):
        """30% hit rate -> DIG < 0"""
        from tuning.tune import compute_dig_per_model
        recs = _make_records(200, horizon=7, hit_rate=0.30)
        dig = compute_dig_per_model(recs, horizons=[7])
        self.assertLess(dig["ensemble"], -0.10)

    def test_too_few_records(self):
        """< 30 records -> DIG = 0"""
        from tuning.tune import compute_dig_per_model
        recs = _make_records(10, horizon=7, hit_rate=0.80)
        dig = compute_dig_per_model(recs, horizons=[7])
        self.assertEqual(dig["ensemble"], 0.0)

    def test_per_horizon_keys(self):
        """Multiple horizons produce per-horizon DIG entries."""
        from tuning.tune import compute_dig_per_model
        recs = _make_records(100, horizon=1, hit_rate=0.60) + \
               _make_records(100, horizon=7, hit_rate=0.70, rng_seed=99)
        dig = compute_dig_per_model(recs)
        self.assertIn("ensemble", dig)
        self.assertIn("H=1", dig)
        self.assertIn("H=7", dig)
        self.assertGreater(dig["H=7"], dig["H=1"])

    def test_dig_gt_002(self):
        """Acceptance criterion: at least model with DIG > 0.02."""
        from tuning.tune import compute_dig_per_model
        recs = _make_records(200, horizon=7, hit_rate=0.55)
        dig = compute_dig_per_model(recs, horizons=[7])
        self.assertGreater(dig["ensemble"], 0.02)


class TestDIGWeight(unittest.TestCase):
    def test_weight_at_zero(self):
        from tuning.tune import compute_dig_weight, DIG_W_START
        self.assertEqual(compute_dig_weight(0), DIG_W_START)

    def test_weight_at_threshold(self):
        from tuning.tune import compute_dig_weight, DIG_W_MAX, DIG_FULL_DATA_THRESHOLD
        self.assertAlmostEqual(compute_dig_weight(DIG_FULL_DATA_THRESHOLD), DIG_W_MAX)

    def test_weight_beyond_threshold(self):
        from tuning.tune import compute_dig_weight, DIG_W_MAX
        self.assertAlmostEqual(compute_dig_weight(1000), DIG_W_MAX)

    def test_weight_linear_ramp(self):
        """At 50% threshold, weight should be midpoint."""
        from tuning.tune import compute_dig_weight, DIG_W_START, DIG_W_MAX, DIG_FULL_DATA_THRESHOLD
        mid = DIG_FULL_DATA_THRESHOLD // 2
        expected = DIG_W_START + (DIG_W_MAX - DIG_W_START) * (mid / DIG_FULL_DATA_THRESHOLD)
        self.assertAlmostEqual(compute_dig_weight(mid), expected, places=4)


class TestBMAWeightAdjust(unittest.TestCase):
    def test_no_records_identity(self):
        """No data -> raw weights unchanged."""
        from tuning.tune import adjust_bma_weights_with_dig
        raw = {"A": 0.5, "B": 0.5}
        adj = adjust_bma_weights_with_dig(raw, {}, n_records=0)
        self.assertAlmostEqual(adj["A"], 0.5)
        self.assertAlmostEqual(adj["B"], 0.5)

    def test_positive_dig_boosted(self):
        """Model with higher DIG gets larger weight."""
        from tuning.tune import adjust_bma_weights_with_dig
        raw = {"A": 0.5, "B": 0.5}
        dig = {"A": 0.15, "B": -0.05}
        adj = adjust_bma_weights_with_dig(raw, dig, n_records=200)
        self.assertGreater(adj["A"], adj["B"])

    def test_negative_dig_penalized(self):
        """Model with DIG < 0 gets reduced weight vs current."""
        from tuning.tune import adjust_bma_weights_with_dig
        raw = {"A": 0.5, "B": 0.5}
        dig = {"A": 0.10, "B": -0.10}
        adj = adjust_bma_weights_with_dig(raw, dig, n_records=200)
        self.assertLess(adj["B"], 0.5)

    def test_weights_sum_to_one(self):
        """Adjusted weights always renormalize to 1."""
        from tuning.tune import adjust_bma_weights_with_dig
        raw = {"A": 0.3, "B": 0.3, "C": 0.4}
        dig = {"A": 0.12, "B": 0.0, "C": -0.05}
        adj = adjust_bma_weights_with_dig(raw, dig, n_records=100)
        self.assertAlmostEqual(sum(adj.values()), 1.0, places=10)

    def test_no_model_zeroed(self):
        """Floor prevents any model weight from reaching zero."""
        from tuning.tune import adjust_bma_weights_with_dig
        raw = {"A": 0.9, "B": 0.1}
        dig = {"A": 0.20, "B": -0.20}
        adj = adjust_bma_weights_with_dig(raw, dig, n_records=200)
        self.assertGreater(adj["B"], 0.0)

    def test_ensemble_dig_ge_best_model(self):
        """Acceptance: BMA-weighted DIG >= max(individual) * 0.8."""
        from tuning.tune import adjust_bma_weights_with_dig
        raw = {"A": 0.50, "B": 0.30, "C": 0.20}
        dig = {"A": 0.12, "B": 0.08, "C": -0.02}
        adj = adjust_bma_weights_with_dig(raw, dig, n_records=200)
        # Weighted DIG
        ensemble_dig = sum(adj[m] * dig[m] for m in adj)
        max_dig = max(dig.values())
        self.assertGreaterEqual(ensemble_dig, max_dig * 0.8)


if __name__ == "__main__":
    unittest.main()
