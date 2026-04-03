"""
Tests for Story 2.5: Profit-and-Loss Attribution by Horizon.

Validates compute_pnl_attribution() on synthetic walk-forward data.
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


def _make_records(n, horizon, p_up_base, realized_drift, rng_seed=42):
    """Create WalkForwardRecord list with controlled p_up and realized returns."""
    from decision.signals import WalkForwardRecord
    rng = np.random.RandomState(rng_seed)
    recs = []
    for i in range(n):
        p_up = p_up_base + rng.normal(0, 0.02)
        p_up = np.clip(p_up, 0.0, 1.0)
        y = rng.normal(realized_drift, 0.01)
        f_ret = rng.normal(realized_drift, 0.01)
        recs.append(WalkForwardRecord(
            date_idx=i, date=pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            horizon=horizon, forecast_ret=f_ret, forecast_p_up=p_up,
            forecast_sig=0.02, realized_ret=y,
            hit=(y > 0) == (f_ret > 0), signed_error=y - f_ret,
            calibration_bucket=5,
        ))
    return recs


class TestPNLConstants(unittest.TestCase):
    def test_thresholds(self):
        from decision.signals import PNL_BUY_THRESHOLD, PNL_SELL_THRESHOLD, PNL_DEFAULT_NOTIONAL
        self.assertEqual(PNL_BUY_THRESHOLD, 0.55)
        self.assertEqual(PNL_SELL_THRESHOLD, 0.45)
        self.assertEqual(PNL_DEFAULT_NOTIONAL, 1e6)


class TestPNLAttribution(unittest.TestCase):

    def test_bullish_signal_positive_return(self):
        """BUY signal + positive realized -> positive P&L."""
        from decision.signals import compute_pnl_attribution, WalkForwardRecord
        recs = [WalkForwardRecord(
            date_idx=0, date=pd.Timestamp("2024-01-01"),
            horizon=7, forecast_ret=0.01, forecast_p_up=0.70,
            forecast_sig=0.02, realized_ret=0.02,
            hit=True, signed_error=0.01, calibration_bucket=5,
        )]
        result = compute_pnl_attribution(recs, notional=1e6)
        self.assertIn(7, result)
        self.assertGreater(result[7]["cumulative_pnl"], 0)
        self.assertEqual(result[7]["n_trades"], 1)
        self.assertEqual(result[7]["hit_rate"], 1.0)

    def test_bullish_signal_negative_return(self):
        """BUY signal + negative realized -> negative P&L."""
        from decision.signals import compute_pnl_attribution, WalkForwardRecord
        recs = [WalkForwardRecord(
            date_idx=0, date=pd.Timestamp("2024-01-01"),
            horizon=7, forecast_ret=0.01, forecast_p_up=0.70,
            forecast_sig=0.02, realized_ret=-0.02,
            hit=False, signed_error=-0.03, calibration_bucket=5,
        )]
        result = compute_pnl_attribution(recs, notional=1e6)
        self.assertLess(result[7]["cumulative_pnl"], 0)
        self.assertEqual(result[7]["hit_rate"], 0.0)

    def test_sell_signal_negative_return(self):
        """SELL signal + negative realized -> positive P&L."""
        from decision.signals import compute_pnl_attribution, WalkForwardRecord
        recs = [WalkForwardRecord(
            date_idx=0, date=pd.Timestamp("2024-01-01"),
            horizon=7, forecast_ret=-0.01, forecast_p_up=0.30,
            forecast_sig=0.02, realized_ret=-0.02,
            hit=True, signed_error=-0.01, calibration_bucket=5,
        )]
        result = compute_pnl_attribution(recs, notional=1e6)
        self.assertGreater(result[7]["cumulative_pnl"], 0)

    def test_hold_no_trade(self):
        """Neutral signal (p_up ~= 0.50) -> no trade, zero P&L."""
        from decision.signals import compute_pnl_attribution, WalkForwardRecord
        recs = [WalkForwardRecord(
            date_idx=0, date=pd.Timestamp("2024-01-01"),
            horizon=7, forecast_ret=0.001, forecast_p_up=0.50,
            forecast_sig=0.02, realized_ret=0.05,
            hit=True, signed_error=0.049, calibration_bucket=5,
        )]
        result = compute_pnl_attribution(recs, notional=1e6)
        self.assertEqual(result[7]["cumulative_pnl"], 0.0)
        self.assertEqual(result[7]["n_trades"], 0)

    def test_sharpe_computation(self):
        """Sharpe = mean(pnl) / std(pnl) * sqrt(252/H)."""
        from decision.signals import compute_pnl_attribution
        # Many BUY signals with positive drift -> positive Sharpe
        recs = _make_records(200, horizon=7, p_up_base=0.65, realized_drift=0.002)
        result = compute_pnl_attribution(recs)
        self.assertIn(7, result)
        self.assertGreater(result[7]["sharpe"], 0)
        self.assertGreater(result[7]["n_trades"], 50)

    def test_multiple_horizons(self):
        """Each horizon gets independent attribution."""
        from decision.signals import compute_pnl_attribution
        recs_1 = _make_records(100, horizon=1, p_up_base=0.60, realized_drift=0.001)
        recs_7 = _make_records(100, horizon=7, p_up_base=0.60, realized_drift=0.003, rng_seed=99)
        result = compute_pnl_attribution(recs_1 + recs_7)
        self.assertIn(1, result)
        self.assertIn(7, result)

    def test_pnl_formula_exact(self):
        """BUY: pnl = notional * (exp(r) - 1)."""
        from decision.signals import compute_pnl_attribution, WalkForwardRecord
        r = 0.01
        notional = 1e6
        recs = [WalkForwardRecord(
            date_idx=0, date=pd.Timestamp("2024-01-01"),
            horizon=7, forecast_ret=0.01, forecast_p_up=0.70,
            forecast_sig=0.02, realized_ret=r,
            hit=True, signed_error=0.0, calibration_bucket=5,
        )]
        result = compute_pnl_attribution(recs, notional=notional)
        expected = notional * (np.exp(r) - 1.0)
        self.assertAlmostEqual(result[7]["cumulative_pnl"], expected, places=2)

    def test_sell_formula_exact(self):
        """SELL: pnl = notional * (1 - exp(r))."""
        from decision.signals import compute_pnl_attribution, WalkForwardRecord
        r = -0.02
        notional = 1e6
        recs = [WalkForwardRecord(
            date_idx=0, date=pd.Timestamp("2024-01-01"),
            horizon=7, forecast_ret=-0.01, forecast_p_up=0.30,
            forecast_sig=0.02, realized_ret=r,
            hit=True, signed_error=-0.01, calibration_bucket=5,
        )]
        result = compute_pnl_attribution(recs, notional=notional)
        expected = notional * (1.0 - np.exp(r))
        self.assertAlmostEqual(result[7]["cumulative_pnl"], expected, places=2)

    def test_empty_records(self):
        """Empty records -> zero attribution."""
        from decision.signals import compute_pnl_attribution
        result = compute_pnl_attribution([], horizons=[7])
        self.assertEqual(result[7]["cumulative_pnl"], 0.0)
        self.assertEqual(result[7]["n_trades"], 0)

    def test_sharpe_positive_h7(self):
        """Acceptance: H=7d Sharpe > 0.3 with good signals."""
        from decision.signals import compute_pnl_attribution
        # Strong directional edge: p_up=0.70 with positive drift
        recs = _make_records(500, horizon=7, p_up_base=0.70, realized_drift=0.003)
        result = compute_pnl_attribution(recs)
        self.assertGreater(result[7]["sharpe"], 0.3)


if __name__ == "__main__":
    unittest.main()
