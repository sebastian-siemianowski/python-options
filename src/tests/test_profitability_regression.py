"""
Story 7.3: Profitability regression test suite.

Validates core profitability metrics on walk-forward results.
Tests skip gracefully when walk-forward data is unavailable.
"""
import unittest
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CALIBRATION_DIR = os.path.join(REPO_ROOT, "data", "calibration")
BASELINE_PATH = os.path.join(SCRIPT_DIR, "profitability_baseline.json")


def _load_walk_forward_results():
    """Load all walk-forward CSVs from calibration directory into a DataFrame."""
    import pandas as pd
    if not os.path.isdir(CALIBRATION_DIR):
        return None
    csvs = [f for f in os.listdir(CALIBRATION_DIR) if f.startswith("walkforward_") and f.endswith(".csv")]
    if not csvs:
        return None
    frames = []
    for fn in csvs:
        df = pd.read_csv(os.path.join(CALIBRATION_DIR, fn))
        if not df.empty:
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _load_baseline():
    """Load baseline metrics, or return defaults if file doesn't exist."""
    if os.path.isfile(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            return json.load(f)
    return {
        "hit_rate_7d": 0.53,
        "hit_rate_21d": 0.52,
        "non_hold_fraction": 0.15,
        "ece_max": 0.05,
        "crps_max": 0.025,
    }


_wf_df = None
_wf_loaded = False


def _get_wf():
    global _wf_df, _wf_loaded
    if not _wf_loaded:
        _wf_df = _load_walk_forward_results()
        _wf_loaded = True
    return _wf_df


def _skip_if_no_data(test_fn):
    """Decorator: skip test if walk-forward data is unavailable."""
    def wrapper(self):
        wf = _get_wf()
        if wf is None or wf.empty:
            self.skipTest("No walk-forward data in src/data/calibration/")
        return test_fn(self, wf)
    wrapper.__name__ = test_fn.__name__
    wrapper.__doc__ = test_fn.__doc__
    return wrapper


class TestProfitabilityRegression(unittest.TestCase):
    """Validates system-level profitability metrics on cached walk-forward data."""

    @_skip_if_no_data
    def test_directional_accuracy_7d(self, wf):
        """Hit rate at H=7d must exceed 53% on validation universe."""
        baseline = _load_baseline()
        h7 = wf[wf["horizon"] == 7]
        if h7.empty:
            self.skipTest("No H=7 data")
        hit_rate = float(h7["hit"].mean())
        self.assertGreater(hit_rate, baseline.get("hit_rate_7d", 0.53),
            f"7d hit rate {hit_rate:.3f} below minimum")

    @_skip_if_no_data
    def test_directional_accuracy_21d(self, wf):
        """Hit rate at H=21d must exceed 52% on validation universe."""
        baseline = _load_baseline()
        h21 = wf[wf["horizon"] == 21]
        if h21.empty:
            self.skipTest("No H=21 data")
        hit_rate = float(h21["hit"].mean())
        self.assertGreater(hit_rate, baseline.get("hit_rate_21d", 0.52),
            f"21d hit rate {hit_rate:.3f} below minimum")

    @_skip_if_no_data
    def test_signal_differentiation(self, wf):
        """At least 15% of signals must be non-HOLD (forecast_p_up outside [0.45, 0.55])."""
        p_up = wf["forecast_p_up"]
        non_hold = ((p_up < 0.45) | (p_up > 0.55)).mean()
        self.assertGreater(non_hold, 0.15,
            f"Non-HOLD fraction {non_hold:.1%} below 15%")

    @_skip_if_no_data
    def test_crps_regression(self, wf):
        """CRPS-like error should not exceed baseline + 5%."""
        baseline = _load_baseline()
        # CRPS approximation: mean absolute error (realized - forecast)
        crps_approx = float(np.mean(np.abs(wf["signed_error"])))
        max_crps = baseline.get("crps_max", 0.025) * 1.05
        self.assertLess(crps_approx, max_crps,
            f"CRPS approx {crps_approx:.4f} exceeds {max_crps:.4f}")

    @_skip_if_no_data
    def test_calibration_ece(self, wf):
        """Expected Calibration Error must be < 0.05 across all horizons."""
        # ECE: mean |freq(positive) - p_up| across calibration buckets
        baseline = _load_baseline()
        ece_max = baseline.get("ece_max", 0.05)
        for h in wf["horizon"].unique():
            h_df = wf[wf["horizon"] == h]
            if len(h_df) < 20:
                continue
            n_buckets = min(10, len(h_df) // 10)
            if n_buckets < 2:
                continue
            h_df = h_df.copy()
            h_df["bucket"] = (h_df["forecast_p_up"] * n_buckets).clip(0, n_buckets - 1).astype(int)
            ece = 0.0
            for b in range(n_buckets):
                bdf = h_df[h_df["bucket"] == b]
                if len(bdf) == 0:
                    continue
                freq = float(bdf["hit"].mean())
                avg_pup = float(bdf["forecast_p_up"].mean())
                ece += abs(freq - avg_pup) * len(bdf) / len(h_df)
            self.assertLess(ece, ece_max,
                f"ECE at H={h} is {ece:.4f} >= {ece_max}")

    @_skip_if_no_data
    def test_sharpe_positive(self, wf):
        """Signal-following Sharpe ratio must be positive."""
        # Simple: long when forecast_p_up > 0.5, short otherwise
        position = np.where(wf["forecast_p_up"] > 0.5, 1.0, -1.0)
        pnl = position * wf["realized_ret"].values
        if len(pnl) < 10:
            self.skipTest("Insufficient data for Sharpe calculation")
        mean_pnl = float(np.mean(pnl))
        std_pnl = float(np.std(pnl))
        if std_pnl < 1e-10:
            self.skipTest("Zero variance in PnL")
        sharpe = mean_pnl / std_pnl
        self.assertGreater(sharpe, 0.0,
            f"Sharpe ratio {sharpe:.3f} must be positive")

    @_skip_if_no_data
    def test_no_catastrophic_loss(self, wf):
        """No single asset should have cumulative loss > 30%."""
        for sym in wf.get("symbol", wf.get("asset", [])).unique() if "symbol" in wf.columns else []:
            sym_df = wf[wf["symbol"] == sym]
            cum_ret = sym_df["realized_ret"].cumsum()
            max_dd = float(cum_ret.max() - cum_ret.min()) if len(cum_ret) > 0 else 0
            self.assertLess(max_dd, 0.30,
                f"{sym} drawdown {max_dd:.1%} exceeds 30%")

    @_skip_if_no_data
    def test_forecast_monotonicity(self, wf):
        """Forecast variance should increase (or be stable) with horizon."""
        horizons = sorted(wf["horizon"].unique())
        if len(horizons) < 2:
            self.skipTest("Need >= 2 horizons")
        prev_var = 0.0
        for h in horizons:
            h_df = wf[wf["horizon"] == h]
            if "forecast_sig" in h_df.columns and len(h_df) > 5:
                var_h = float(h_df["forecast_sig"].mean())
                if prev_var > 0:
                    self.assertGreaterEqual(var_h, prev_var * 0.8,
                        f"Variance at H={h} ({var_h:.4f}) < 80% of previous ({prev_var:.4f})")
                prev_var = var_h


class TestBaselineMetrics(unittest.TestCase):
    """Tests that don't require walk-forward data -- validate baseline config."""

    def test_baseline_file_schema(self):
        """Baseline JSON should have required keys if it exists."""
        if not os.path.isfile(BASELINE_PATH):
            self.skipTest("Baseline JSON not created yet")
        with open(BASELINE_PATH) as f:
            data = json.load(f)
        required = {"hit_rate_7d", "hit_rate_21d", "non_hold_fraction", "ece_max"}
        for key in required:
            self.assertIn(key, data, f"Missing baseline key: {key}")

    def test_walk_forward_functions_importable(self):
        """Core walk-forward functions should be importable."""
        from decision.signals import (
            run_walk_forward_backtest,
            run_walk_forward_parallel,
            WalkForwardResult,
            WalkForwardRecord,
            walkforward_result_to_dataframe,
            save_walkforward_csv,
        )
        self.assertTrue(callable(run_walk_forward_backtest))
        self.assertTrue(callable(run_walk_forward_parallel))


if __name__ == "__main__":
    unittest.main()
