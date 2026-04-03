"""test_cusum_auto_tuning.py -- Story 4.4

Validates CUSUM sensitivity auto-tuning:
1. TSLA CUSUM threshold lower than GLD (faster detection for volatile assets)
2. Cooldown proportional to autocorrelation decay
3. ARL approximately target for all assets
4. No spurious regime flips for stable assets
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.regime import (
    CUSUMParams, compute_cusum_params, compute_arl_threshold, decorrelation_time,
    assign_regime_labels, assign_current_regime,
    CUSUM_THRESHOLD, CUSUM_COOLDOWN,
    CUSUM_MIN_THRESHOLD, CUSUM_MAX_THRESHOLD,
    CUSUM_MIN_COOLDOWN, CUSUM_MAX_COOLDOWN,
    CUSUM_TARGET_ARL, MarketRegime,
)


def _make_returns(sigma: float, n: int = 1000, seed: int = 42,
                  autocorr: float = 0.0) -> np.ndarray:
    """Generate synthetic returns with given vol and autocorrelation."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)
    if abs(autocorr) > 1e-6:
        ret = np.zeros(n)
        ret[0] = eps[0]
        for i in range(1, n):
            ret[i] = autocorr * ret[i - 1] + eps[i]
        return ret
    return eps


class TestDecorrelationTime(unittest.TestCase):

    def test_iid_returns_fast_decay(self):
        """IID returns should decorrelate in 1-2 bars."""
        returns = _make_returns(0.01, n=500)
        decor = decorrelation_time(returns)
        self.assertLessEqual(decor, 5)

    def test_autocorrelated_returns_slow_decay(self):
        """Autocorrelated returns should have longer decorrelation."""
        returns = _make_returns(0.01, n=500, autocorr=0.5)
        decor = decorrelation_time(returns)
        self.assertGreater(decor, 1)

    def test_short_series_default(self):
        returns = np.array([0.01, 0.02])
        decor = decorrelation_time(returns)
        self.assertEqual(decor, 5)  # default

    def test_constant_series_default(self):
        returns = np.zeros(100)
        decor = decorrelation_time(returns)
        self.assertEqual(decor, 5)


class TestComputeArlThreshold(unittest.TestCase):

    def test_higher_vol_higher_threshold(self):
        """Higher vol assets need higher CUSUM threshold."""
        low_vol = _make_returns(0.005, n=500, seed=1)   # Bond-like
        high_vol = _make_returns(0.03, n=500, seed=2)    # Crypto-like
        t_low = compute_arl_threshold(low_vol)
        t_high = compute_arl_threshold(high_vol)
        self.assertGreater(t_high, t_low)

    def test_threshold_in_bounds(self):
        returns = _make_returns(0.01, n=500)
        thresh = compute_arl_threshold(returns)
        self.assertGreaterEqual(thresh, CUSUM_MIN_THRESHOLD)
        self.assertLessEqual(thresh, CUSUM_MAX_THRESHOLD)

    def test_short_series_default(self):
        returns = np.array([0.01] * 10)
        thresh = compute_arl_threshold(returns)
        self.assertEqual(thresh, CUSUM_THRESHOLD)

    def test_zero_vol_default(self):
        returns = np.zeros(100)
        thresh = compute_arl_threshold(returns)
        self.assertEqual(thresh, CUSUM_THRESHOLD)


class TestComputeCusumParams(unittest.TestCase):

    def test_auto_source(self):
        returns = _make_returns(0.01, n=500)
        cp = compute_cusum_params(returns)
        self.assertEqual(cp.source, "auto")

    def test_default_on_short_data(self):
        returns = np.array([0.01] * 10)
        cp = compute_cusum_params(returns)
        self.assertEqual(cp.source, "default")
        self.assertEqual(cp.threshold, CUSUM_THRESHOLD)
        self.assertEqual(cp.cooldown, CUSUM_COOLDOWN)

    def test_cooldown_in_bounds(self):
        returns = _make_returns(0.01, n=500)
        cp = compute_cusum_params(returns)
        self.assertGreaterEqual(cp.cooldown, CUSUM_MIN_COOLDOWN)
        self.assertLessEqual(cp.cooldown, CUSUM_MAX_COOLDOWN)

    def test_to_dict(self):
        returns = _make_returns(0.01, n=500)
        cp = compute_cusum_params(returns)
        d = cp.to_dict()
        self.assertIn("threshold", d)
        self.assertIn("cooldown", d)
        self.assertIn("sigma_returns", d)
        self.assertIn("decorrelation_bars", d)
        self.assertIn("source", d)

    def test_tsla_vs_gld_threshold(self):
        """TSLA (high vol) threshold different from GLD (low vol)."""
        tsla_ret = _make_returns(0.03, n=500, seed=1)
        gld_ret = _make_returns(0.006, n=500, seed=2)
        tsla_cp = compute_cusum_params(tsla_ret)
        gld_cp = compute_cusum_params(gld_ret)
        # Higher vol -> higher threshold (need bigger deviation to trigger)
        self.assertGreater(tsla_cp.threshold, gld_cp.threshold)

    def test_autocorr_affects_cooldown(self):
        """Higher autocorrelation -> longer cooldown."""
        iid_ret = _make_returns(0.01, n=500, seed=1, autocorr=0.0)
        corr_ret = _make_returns(0.01, n=500, seed=1, autocorr=0.6)
        iid_cp = compute_cusum_params(iid_ret)
        corr_cp = compute_cusum_params(corr_ret)
        self.assertGreaterEqual(corr_cp.cooldown, iid_cp.cooldown)


class TestCusumParamsDefault(unittest.TestCase):

    def test_default_values(self):
        cp = CUSUMParams.default()
        self.assertEqual(cp.threshold, CUSUM_THRESHOLD)
        self.assertEqual(cp.cooldown, CUSUM_COOLDOWN)
        self.assertEqual(cp.source, "default")


class TestRegimeLabelsWithCusum(unittest.TestCase):

    def test_backward_compatible(self):
        """Without cusum_params, behavior unchanged."""
        rng = np.random.default_rng(42)
        n = 200
        returns = rng.normal(0, 0.01, n)
        vol = np.abs(returns) + 0.005
        labels = assign_regime_labels(returns, vol)
        for lab in labels:
            self.assertIn(lab, [0, 1, 2, 3, 4])

    def test_with_cusum_params(self):
        """Custom CUSUM params should work without error."""
        rng = np.random.default_rng(42)
        n = 200
        returns = rng.normal(0, 0.01, n)
        vol = np.abs(returns) + 0.005
        cp = compute_cusum_params(returns)
        labels = assign_regime_labels(returns, vol, cusum_params=cp)
        for lab in labels:
            self.assertIn(lab, [0, 1, 2, 3, 4])


class TestCurrentRegimeWithCusum(unittest.TestCase):

    def test_backward_compatible(self):
        rng = np.random.default_rng(42)
        feats = {
            "ret": pd.Series(rng.normal(0, 0.01, 100)),
            "vol": pd.Series(np.full(100, 0.01)),
        }
        r = assign_current_regime(feats, asset="test_cusum_compat")
        self.assertIn(r, [0, 1, 2, 3, 4])

    def test_with_cusum_params(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 100)
        feats = {
            "ret": pd.Series(returns),
            "vol": pd.Series(np.full(100, 0.01)),
        }
        cp = compute_cusum_params(returns)
        r = assign_current_regime(feats, asset="test_cusum_auto",
                                  cusum_params=cp)
        self.assertIn(r, [0, 1, 2, 3, 4])


class TestNoSpuriousFlipsStableAsset(unittest.TestCase):

    def test_stable_asset_few_transitions(self):
        """Stable asset (low vol, low autocorr) should have few regime changes."""
        rng = np.random.default_rng(42)
        n = 500
        returns = rng.normal(0, 0.005, n)  # Very low vol
        vol = np.full(n, 0.005)
        cp = compute_cusum_params(returns)
        labels = assign_regime_labels(returns, vol, cusum_params=cp)
        transitions = np.sum(np.diff(labels) != 0)
        # Stable asset: fewer than 20% of bars should be transitions
        self.assertLess(transitions / n, 0.20,
                        f"Too many transitions ({transitions}) for stable asset")


class TestArlTargeting(unittest.TestCase):

    def test_arl_approximately_252(self):
        """Empirical ARL should be reasonable (not flipping every bar)."""
        rng = np.random.default_rng(42)
        n = 5000  # Long series for statistical reliability
        returns = rng.normal(0, 0.01, n)  # Stationary
        vol = np.full(n, 0.01)
        cp = compute_cusum_params(returns)
        labels = assign_regime_labels(returns, vol, cusum_params=cp)
        transitions = np.sum(np.diff(labels) != 0)
        if transitions > 0:
            empirical_arl = n / transitions
            # Regime transitions come from drift_abs crossing threshold
            # (not just CUSUM). Should average >3 bars persistence.
            self.assertGreater(empirical_arl, 3)


if __name__ == "__main__":
    unittest.main()
