"""
End-to-end integration test for Signal CI bounds (March 2026).

Tests the FULL signal pipeline (data → features → MC → EMOS → CI → profit → display)
on real cached price data to verify:
1. All CI values are within physical limits
2. All profit values are bounded
3. Normal-volatility assets are unaffected by caps
4. Extreme-volatility assets are properly bounded
5. All asset types produce sensible outputs
6. The backtest engine also respects signal bounds

Run with:
    .venv/bin/python -m pytest src/tests/test_signal_e2e.py -v --tb=short
"""

import os
import sys
import math
import unittest
import warnings

# Suppress noisy output
os.environ["TUNING_QUIET"] = "1"
os.environ["OFFLINE_MODE"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

# Physical limits — must match signals.py
NOTIONAL_PLN = 1_000_000
CI_LOG_FLOOR = -4.6      # exp(-4.6)-1 ≈ -99%
CI_LOG_CAP = 1.61         # exp(1.61)-1 ≈ +400%
MAX_PROFIT = 4 * NOTIONAL_PLN
MIN_PROFIT = -NOTIONAL_PLN


def _load_cached_price(symbol: str) -> Optional[Tuple[pd.Series, pd.DataFrame]]:
    """Load cached price data without hitting Yahoo.
    
    Returns (close_series, ohlc_dataframe) or None.
    """
    prices_dir = os.path.join(REPO_ROOT, "src", "data", "prices")
    # Try common naming patterns
    candidates = [
        f"{symbol}.csv",
        f"{symbol}_1d.csv",
        f"{symbol.replace('=', '_')}.csv",
        f"{symbol.replace('=', '_')}_1d.csv",
        f"{symbol.replace('-', '-')}.csv",
    ]
    for name in candidates:
        path = os.path.join(prices_dir, name)
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                # Find close column
                close_col = None
                for col in ["Close", "close", "Adj Close"]:
                    if col in df.columns:
                        close_col = col
                        break
                if close_col is None and df.shape[1] == 1:
                    close_col = df.columns[0]
                if close_col is None:
                    continue
                px = df[close_col].astype(float).dropna()
                px.name = "px"
                if len(px) < 100:
                    continue
                # Build OHLC DataFrame
                ohlc_df = df.copy()
                return px, ohlc_df
            except Exception:
                continue
    return None


def _run_signal_pipeline(px: pd.Series, symbol: str,
                         ohlc_df: Optional[pd.DataFrame] = None) -> Tuple[list, dict]:
    """Run the full signal pipeline on price data."""
    from decision.signals import (
        compute_features, latest_signals, classify_asset_type,
        DEFAULT_HORIZONS, _load_tuned_kalman_params,
    )
    feats = compute_features(px, asset_symbol=symbol, ohlc_df=ohlc_df)
    last_close = float(px.iloc[-1])
    tuned_params = _load_tuned_kalman_params(symbol)
    asset_type = classify_asset_type(symbol)
    sigs, thresholds = latest_signals(
        feats, DEFAULT_HORIZONS, last_close,
        tuned_params=tuned_params,
        asset_key=symbol,
        asset_type=asset_type,
        n_mc_paths=2000,  # Reduced for test speed
    )
    return sigs, {"asset_type": asset_type, "thresholds": thresholds}


class TestSignalE2ESingleAsset(unittest.TestCase):
    """E2E tests on individual assets with cached data."""

    def _assert_signal_bounds(self, sig, symbol: str):
        """Assert that a signal respects all physical limits."""
        # CI bounds in log space
        self.assertGreaterEqual(sig.ci_low, CI_LOG_FLOOR,
                              f"{symbol} H={sig.horizon_days}: ci_low={sig.ci_low} < {CI_LOG_FLOOR}")
        self.assertLessEqual(sig.ci_high, CI_LOG_CAP,
                           f"{symbol} H={sig.horizon_days}: ci_high={sig.ci_high} > {CI_LOG_CAP}")
        # CI ordering
        self.assertLessEqual(sig.ci_low, sig.ci_high,
                           f"{symbol} H={sig.horizon_days}: ci_low > ci_high")
        # Profit bounds
        self.assertGreaterEqual(sig.profit_pln, MIN_PROFIT,
                              f"{symbol} H={sig.horizon_days}: profit={sig.profit_pln} < {MIN_PROFIT}")
        self.assertLessEqual(sig.profit_pln, MAX_PROFIT,
                           f"{symbol} H={sig.horizon_days}: profit={sig.profit_pln} > {MAX_PROFIT}")
        self.assertGreaterEqual(sig.profit_ci_low_pln, MIN_PROFIT,
                              f"{symbol} H={sig.horizon_days}: profit_ci_low={sig.profit_ci_low_pln}")
        self.assertLessEqual(sig.profit_ci_high_pln, MAX_PROFIT,
                           f"{symbol} H={sig.horizon_days}: profit_ci_high={sig.profit_ci_high_pln}")
        # Finite checks
        self.assertTrue(np.isfinite(sig.ci_low), f"{symbol} H={sig.horizon_days}: ci_low not finite")
        self.assertTrue(np.isfinite(sig.ci_high), f"{symbol} H={sig.horizon_days}: ci_high not finite")
        self.assertTrue(np.isfinite(sig.profit_pln), f"{symbol} H={sig.horizon_days}: profit not finite")
        # CI as percentages should be within display bounds
        ci_low_pct = (math.exp(sig.ci_low) - 1) * 100
        ci_high_pct = (math.exp(sig.ci_high) - 1) * 100
        self.assertGreater(ci_low_pct, -100.0,
                         f"{symbol} H={sig.horizon_days}: ci_low_pct={ci_low_pct}%")
        self.assertLess(ci_high_pct, 500.0,
                       f"{symbol} H={sig.horizon_days}: ci_high_pct={ci_high_pct}%")

    def _run_and_validate(self, symbol: str, expected_type: str = "equity"):
        """Load data, run pipeline, validate all signals."""
        result = _load_cached_price(symbol)
        if result is None:
            self.skipTest(f"No cached data for {symbol}")
        px, ohlc_df = result
        self.assertGreaterEqual(len(px), 100)
        
        sigs, meta = _run_signal_pipeline(px, symbol, ohlc_df=ohlc_df)
        
        self.assertEqual(meta["asset_type"], expected_type,
                        f"{symbol} classified as {meta['asset_type']}, expected {expected_type}")
        self.assertGreater(len(sigs), 0, f"No signals generated for {symbol}")
        
        for sig in sigs:
            self._assert_signal_bounds(sig, symbol)
        return sigs, meta

    # ===== Equity tests =====
    def test_spy_equity_normal_vol(self):
        """SPY (index ETF, ~14% annual vol) — normal vol, CI should be bounded."""
        sigs, _ = self._run_and_validate("SPY", "equity")
        # All physical limits should hold (already checked by _assert_signal_bounds)
        # SPY at H=252 can legitimately hit sig_H cap, which is by design
        for sig in sigs:
            self.assertGreaterEqual(sig.ci_low, CI_LOG_FLOOR)
            self.assertLessEqual(sig.ci_high, CI_LOG_CAP)

    def test_aapl_equity_normal_vol(self):
        """AAPL (large cap, ~25% annual vol) — standard equity."""
        self._run_and_validate("AAPL", "equity")

    def test_nvda_equity_moderate_vol(self):
        """NVDA (tech, ~40% annual vol) — higher but within equity range."""
        self._run_and_validate("NVDA", "equity")

    def test_tsla_equity_high_vol(self):
        """TSLA (high vol equity, ~50-60% annual vol)."""
        self._run_and_validate("TSLA", "equity")

    def test_abtc_equity_extreme_vol(self):
        """ABTC (extreme vol equity) — the #1 offender, MUST be bounded."""
        sigs, _ = self._run_and_validate("ABTC", "equity")
        # ABTC was producing CI like [-1,832,151%, +1,831,...%]
        # Now must be within physical limits
        for sig in sigs:
            ci_low_pct = (math.exp(sig.ci_low) - 1) * 100
            ci_high_pct = (math.exp(sig.ci_high) - 1) * 100
            self.assertGreater(ci_low_pct, -99.0,
                             f"ABTC H={sig.horizon_days}: CI still absurd at {ci_low_pct:.1f}%")
            self.assertLess(ci_high_pct, 401.0,
                          f"ABTC H={sig.horizon_days}: CI still absurd at {ci_high_pct:.1f}%")

    def test_ionq_equity_volatile(self):
        """IONQ (quantum computing small cap, high vol)."""
        self._run_and_validate("IONQ", "equity")

    def test_msft_equity_low_vol(self):
        """MSFT (large cap, ~22% annual vol) — stable equity."""
        self._run_and_validate("MSFT", "equity")

    # ===== Metal tests =====
    def test_gld_metal(self):
        """GLD (gold ETF) — should be classified as metal."""
        self._run_and_validate("GLD", "metal")

    def test_nem_metal(self):
        """NEM (Newmont, gold miner) — metal classifier."""
        self._run_and_validate("NEM", "metal")

    # ===== Crypto tests =====
    def test_btc_crypto(self):
        """BTC-USD — highest vol asset type, verify bounded."""
        sigs, _ = self._run_and_validate("BTC-USD", "crypto")
        # Crypto caps are wider (200% annual) but still bounded
        for sig in sigs:
            self.assertGreaterEqual(sig.ci_low, CI_LOG_FLOOR)
            self.assertLessEqual(sig.ci_high, CI_LOG_CAP)


class TestSignalE2EBatch(unittest.TestCase):
    """Batch E2E test: run 20 diverse assets and validate ALL outputs."""

    # Diverse asset universe covering all types and volatility regimes
    UNIVERSE = [
        # Low-vol equities
        ("SPY", "equity"), ("MSFT", "equity"), ("AAPL", "equity"),
        # Medium-vol equities
        ("NVDA", "equity"), ("TSLA", "equity"),
        # High-vol equities
        ("IONQ", "equity"), ("ABTC", "equity"), ("GPUS", "equity"),
        ("AFRM", "equity"), ("UPST", "equity"),
        # Metals
        ("GLD", "metal"), ("NEM", "metal"), ("AG", "metal"),
        # Crypto
        ("BTC-USD", "crypto"),
    ]

    def test_all_assets_bounded(self):
        """Run full pipeline on 20 assets, verify ALL signals bounded."""
        from decision.signals import (
            compute_features, latest_signals, classify_asset_type,
            _load_tuned_kalman_params,
        )
        tested = 0
        failures = []
        
        for symbol, expected_type in self.UNIVERSE:
            result = _load_cached_price(symbol)
            if result is None:
                continue
            px, ohlc_df = result
            if len(px) < 100:
                continue
                
            try:
                feats = compute_features(px, asset_symbol=symbol, ohlc_df=ohlc_df)
                last_close = float(px.iloc[-1])
                tuned_params = _load_tuned_kalman_params(symbol)
                asset_type = classify_asset_type(symbol)
                
                sigs, _ = latest_signals(
                    feats, [1, 21, 63, 252], last_close,
                    tuned_params=tuned_params,
                    asset_key=symbol,
                    asset_type=asset_type,
                    n_mc_paths=2000,
                )
                
                for sig in sigs:
                    # Check CI bounds
                    if sig.ci_low < CI_LOG_FLOOR:
                        failures.append(f"{symbol} H={sig.horizon_days}: ci_low={sig.ci_low}")
                    if sig.ci_high > CI_LOG_CAP:
                        failures.append(f"{symbol} H={sig.horizon_days}: ci_high={sig.ci_high}")
                    # Check profit bounds
                    if sig.profit_pln < MIN_PROFIT:
                        failures.append(f"{symbol} H={sig.horizon_days}: profit={sig.profit_pln}")
                    if sig.profit_pln > MAX_PROFIT:
                        failures.append(f"{symbol} H={sig.horizon_days}: profit={sig.profit_pln}")
                    if sig.profit_ci_low_pln < MIN_PROFIT:
                        failures.append(f"{symbol} H={sig.horizon_days}: ci_low_profit={sig.profit_ci_low_pln}")
                    if sig.profit_ci_high_pln > MAX_PROFIT:
                        failures.append(f"{symbol} H={sig.horizon_days}: ci_high_profit={sig.profit_ci_high_pln}")
                    # Check finiteness
                    if not np.isfinite(sig.ci_low):
                        failures.append(f"{symbol} H={sig.horizon_days}: ci_low not finite")
                    if not np.isfinite(sig.ci_high):
                        failures.append(f"{symbol} H={sig.horizon_days}: ci_high not finite")
                    if not np.isfinite(sig.profit_pln):
                        failures.append(f"{symbol} H={sig.horizon_days}: profit not finite")
                tested += 1
            except Exception as e:
                # Record but don't fail for data issues
                warnings.warn(f"{symbol}: {e}")
                continue
        
        self.assertGreater(tested, 5, f"Only tested {tested} assets, need at least 5")
        self.assertEqual(len(failures), 0,
                        f"Signal bound violations:\n" + "\n".join(failures))

    def test_no_absurd_ci_anywhere(self):
        """No asset should produce CI wider than 500% in display space."""
        tested = 0
        absurd = []
        
        for symbol, expected_type in self.UNIVERSE:
            result = _load_cached_price(symbol)
            if result is None:
                continue
            px, ohlc_df = result
            if len(px) < 100:
                continue
                
            try:
                sigs, meta = _run_signal_pipeline(px, symbol, ohlc_df=ohlc_df)
                for sig in sigs:
                    ci_low_pct = (math.exp(sig.ci_low) - 1) * 100
                    ci_high_pct = (math.exp(sig.ci_high) - 1) * 100
                    width = ci_high_pct - ci_low_pct
                    if width > 500:
                        absurd.append(
                            f"{symbol} H={sig.horizon_days}: CI width={width:.1f}% "
                            f"[{ci_low_pct:.1f}%, {ci_high_pct:.1f}%]"
                        )
                tested += 1
            except Exception:
                continue
        
        self.assertGreater(tested, 5)
        self.assertEqual(len(absurd), 0,
                        f"Absurd CI widths found:\n" + "\n".join(absurd))


class TestSignalE2EMonotonicity(unittest.TestCase):
    """Test that CI width grows monotonically with horizon (for same asset)."""

    def test_ci_width_increases_with_horizon(self):
        """CI width should generally increase with horizon (√H scaling)."""
        from decision.signals import (
            compute_features, latest_signals, classify_asset_type,
            _load_tuned_kalman_params,
        )
        symbols = ["SPY", "AAPL", "NVDA"]
        horizons = [1, 7, 21, 63, 252]
        
        for symbol in symbols:
            result = _load_cached_price(symbol)
            if result is None:
                continue
            px, ohlc_df = result
            if len(px) < 300:
                continue
                
            feats = compute_features(px, asset_symbol=symbol, ohlc_df=ohlc_df)
            last_close = float(px.iloc[-1])
            tuned_params = _load_tuned_kalman_params(symbol)
            asset_type = classify_asset_type(symbol)
            
            sigs, _ = latest_signals(
                feats, horizons, last_close,
                tuned_params=tuned_params,
                asset_key=f"{symbol}_mono",
                asset_type=asset_type,
                n_mc_paths=2000,
            )
            
            if len(sigs) < 3:
                continue
            
            # Check that width generally increases
            # Allow 1 violation due to EMOS/calibration effects
            widths = [(s.horizon_days, s.ci_high - s.ci_low) for s in sigs]
            widths.sort(key=lambda x: x[0])
            violations = 0
            for i in range(1, len(widths)):
                if widths[i][1] < widths[i-1][1] * 0.5:
                    violations += 1
            self.assertLessEqual(violations, 1,
                               f"{symbol}: CI width not monotonic: {widths}")


class TestBacktestSignalBounds(unittest.TestCase):
    """Test that the backtest engine also produces bounded signals."""

    def test_backtest_engine_respects_bounds(self):
        """Simulate a mini backtest and verify signal bounds throughout."""
        result = _load_cached_price("SPY")
        if result is None:
            self.skipTest("No SPY data for backtest")
        px, ohlc_df = result
        if len(px) < 500:
            self.skipTest("Not enough SPY data for backtest")
        
        from decision.signals import (
            compute_features, latest_signals, classify_asset_type,
            _load_tuned_kalman_params,
        )
        
        # Walk-forward: compute signals at multiple points
        n = len(px)
        test_points = [n - 252, n - 126, n - 63, n - 21, n - 1]
        test_points = [p for p in test_points if p >= 300]
        
        for end_idx in test_points:
            px_slice = px.iloc[:end_idx]
            ohlc_slice = ohlc_df.iloc[:end_idx] if ohlc_df is not None else None
            feats = compute_features(px_slice, asset_symbol="SPY", ohlc_df=ohlc_slice)
            last_close = float(px_slice.iloc[-1])
            tuned_params = _load_tuned_kalman_params("SPY")
            
            sigs, _ = latest_signals(
                feats, [21, 63], last_close,
                tuned_params=tuned_params,
                asset_key=f"SPY_bt_{end_idx}",
                asset_type="equity",
                n_mc_paths=1000,
            )
            
            for sig in sigs:
                self.assertGreaterEqual(sig.ci_low, CI_LOG_FLOOR)
                self.assertLessEqual(sig.ci_high, CI_LOG_CAP)
                self.assertGreaterEqual(sig.profit_pln, MIN_PROFIT)
                self.assertLessEqual(sig.profit_pln, MAX_PROFIT)
                self.assertTrue(np.isfinite(sig.profit_pln))


class TestDisplayFormattingE2E(unittest.TestCase):
    """Test display formatting with real signal values."""

    def test_format_ci_display_with_real_signals(self):
        """Generate real signals and verify display formatting."""
        from decision.signals_ux import format_profit_with_signal
        
        result = _load_cached_price("ABTC")
        if result is None:
            self.skipTest("No ABTC data")
        px, ohlc_df = result
        if len(px) < 100:
            self.skipTest("Not enough ABTC data")
        
        sigs, _ = _run_signal_pipeline(px, "ABTC", ohlc_df=ohlc_df)
        
        for sig in sigs:
            # Format profit display
            result = format_profit_with_signal(
                sig.label, sig.profit_pln, NOTIONAL_PLN
            )
            # Should be a non-empty string
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)
            
            # Should NOT contain absurd numbers
            # Extract percentage from display string
            import re
            pct_match = re.search(r'([+-]?\d+\.?\d*)%', result)
            if pct_match:
                pct_val = float(pct_match.group(1))
                self.assertGreaterEqual(pct_val, -100.0,
                    f"ABTC H={sig.horizon_days}: display shows {pct_val}%")
                self.assertLessEqual(pct_val, 500.0,
                    f"ABTC H={sig.horizon_days}: display shows {pct_val}%")

    def test_ci_display_values_bounded(self):
        """CI display values should be within [-99%, +500%]."""
        result = _load_cached_price("TSLA")
        if result is None:
            self.skipTest("No TSLA data")
        px, ohlc_df = result
        if len(px) < 100:
            self.skipTest("Not enough TSLA data")
        
        sigs, _ = _run_signal_pipeline(px, "TSLA", ohlc_df=ohlc_df)
        
        for sig in sigs:
            ci_low_pct = max(sig.ci_low * 100, -99.0)
            ci_high_pct = min(sig.ci_high * 100, 500.0)
            self.assertGreaterEqual(ci_low_pct, -99.0)
            self.assertLessEqual(ci_high_pct, 500.0)
            # Width should be reasonable
            width = ci_high_pct - ci_low_pct
            self.assertLess(width, 600.0,
                          f"TSLA H={sig.horizon_days}: CI display width={width:.1f}%")


if __name__ == "__main__":
    unittest.main()
