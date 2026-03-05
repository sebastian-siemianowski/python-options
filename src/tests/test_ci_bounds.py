"""
Tests for CI/sig_H bounding system (March 2026).

Validates that:
1. classify_asset_type() correctly classifies all asset types
2. _compute_sig_h_cap() produces horizon-aware, asset-type-specific caps
3. CI bounds are clamped to physical limits [-99%, +400%]
4. Profit values are clamped to [-100%, +400%] of notional
5. Display formatting respects caps
6. Normal-volatility assets are NOT affected by caps
7. Extreme-volatility assets ARE bounded
8. End-to-end signal generation produces bounded outputs
"""

import os
import sys
import math
import unittest
import numpy as np
import pandas as pd

# Setup path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


class TestClassifyAssetType(unittest.TestCase):
    """Test asset type classification for CI bounding."""

    def setUp(self):
        from decision.signals import classify_asset_type
        self.classify = classify_asset_type

    def test_equities(self):
        """Standard US equities should be classified as equity."""
        equities = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "QQQ", "IWM",
                     "ABTC", "GPUS", "UPST", "AFRM", "IONQ", "META", "JPM"]
        for sym in equities:
            self.assertEqual(self.classify(sym), "equity", f"{sym} should be equity")

    def test_crypto_like_equity_not_misclassified(self):
        """Equities containing crypto substrings (ABTC, META) should be equity."""
        # ABTC contains "BTC", META contains no crypto bases but is tested
        # These should NOT be classified as crypto
        self.assertEqual(self.classify("ABTC"), "equity")
        self.assertEqual(self.classify("META"), "equity")
        self.assertEqual(self.classify("MSTR"), "equity")  # MicroStrategy
        self.assertEqual(self.classify("BITO"), "equity")  # Bitcoin ETF

    def test_currencies(self):
        """FX pairs should be classified as currency."""
        currencies = ["USDJPY=X", "EURUSD=X", "GBPUSD=X", "PLNJPY=X",
                      "AUDUSD=X", "USDCHF=X"]
        for sym in currencies:
            self.assertEqual(self.classify(sym), "currency", f"{sym} should be currency")

    def test_metals(self):
        """Metal tickers and futures should be classified as metal."""
        metals = ["GC=F", "SI=F", "GLD", "SLV", "GDX", "NEM", "AEM",
                  "WPM", "PAAS", "AG", "KGC"]
        for sym in metals:
            self.assertEqual(self.classify(sym), "metal", f"{sym} should be metal")

    def test_crypto(self):
        """Crypto tickers should be classified as crypto."""
        cryptos = ["BTC-USD", "ETH-USD", "DOGE-USD", "SOL-USD", "ADA-USD"]
        for sym in cryptos:
            self.assertEqual(self.classify(sym), "crypto", f"{sym} should be crypto")

    def test_empty_and_none(self):
        """Edge cases: empty string and None-like inputs."""
        self.assertEqual(self.classify(""), "equity")
        self.assertEqual(self.classify("   "), "equity")

    def test_case_insensitive(self):
        """Classification should be case-insensitive."""
        self.assertEqual(self.classify("aapl"), "equity")
        self.assertEqual(self.classify("btc-usd"), "crypto")
        self.assertEqual(self.classify("Gld"), "metal")


class TestSigHCap(unittest.TestCase):
    """Test sig_H capping logic."""

    def setUp(self):
        from decision.signals import _compute_sig_h_cap, _SIG_H_ANNUAL_CAP
        self.compute_cap = _compute_sig_h_cap
        self.annual_caps = _SIG_H_ANNUAL_CAP

    def test_annual_caps_exist(self):
        """All asset types should have annual caps defined."""
        for atype in ["equity", "currency", "metal", "crypto"]:
            self.assertIn(atype, self.annual_caps)
            self.assertGreater(self.annual_caps[atype], 0)

    def test_equity_annual_horizon(self):
        """At H=252 (annual), equity cap should equal annual_cap."""
        cap = self.compute_cap(252, "equity")
        self.assertAlmostEqual(cap, self.annual_caps["equity"], places=4)

    def test_sqrt_h_scaling(self):
        """Cap should scale as √H."""
        cap_63 = self.compute_cap(63, "equity")
        cap_252 = self.compute_cap(252, "equity")
        # √(63/252) = √(1/4) = 0.5
        self.assertAlmostEqual(cap_63 / cap_252, 0.5, places=2)

    def test_short_horizon_floor(self):
        """Very short horizons should have a minimum floor."""
        cap_1 = self.compute_cap(1, "equity")
        self.assertGreaterEqual(cap_1, 0.005)

    def test_crypto_wider_than_equity(self):
        """Crypto should have wider bounds than equity."""
        for H in [1, 21, 63, 252]:
            cap_equity = self.compute_cap(H, "equity")
            cap_crypto = self.compute_cap(H, "crypto")
            self.assertGreater(cap_crypto, cap_equity,
                             f"Crypto cap should exceed equity at H={H}")

    def test_currency_tighter_than_equity(self):
        """Currency should have tighter bounds than equity."""
        for H in [1, 21, 63, 252]:
            cap_currency = self.compute_cap(H, "currency")
            cap_equity = self.compute_cap(H, "equity")
            self.assertLess(cap_currency, cap_equity,
                          f"Currency cap should be tighter than equity at H={H}")

    def test_ordering_across_asset_types(self):
        """Caps should follow: currency < metal < equity < crypto."""
        for H in [21, 63, 252]:
            caps = {at: self.compute_cap(H, at)
                    for at in ["currency", "metal", "equity", "crypto"]}
            self.assertLess(caps["currency"], caps["metal"])
            self.assertLess(caps["metal"], caps["equity"])
            self.assertLess(caps["equity"], caps["crypto"])

    def test_unknown_asset_type_defaults_to_equity(self):
        """Unknown asset types should default to equity cap."""
        cap_unknown = self.compute_cap(252, "unknown_type")
        cap_equity = self.compute_cap(252, "equity")
        self.assertAlmostEqual(cap_unknown, cap_equity, places=6)

    def test_specific_values(self):
        """Verify specific cap values for key horizons."""
        # H=1 day, equity: 1.20 × √(1/252) ≈ 0.0756
        cap = self.compute_cap(1, "equity")
        self.assertAlmostEqual(cap, 1.20 * math.sqrt(1/252), places=4)

        # H=21 day (1 month), equity: 1.20 × √(21/252) ≈ 0.346
        cap = self.compute_cap(21, "equity")
        self.assertAlmostEqual(cap, 1.20 * math.sqrt(21/252), places=4)


class TestCIBounds(unittest.TestCase):
    """Test that CI bounds are clamped to physical limits."""

    def setUp(self):
        from decision.signals import _CI_LOG_FLOOR, _CI_LOG_CAP
        self.CI_FLOOR = _CI_LOG_FLOOR
        self.CI_CAP = _CI_LOG_CAP

    def test_ci_floor_value(self):
        """CI floor should be -4.6 (≈-99% loss)."""
        self.assertAlmostEqual(self.CI_FLOOR, -4.6, places=2)
        # Verify the percentage interpretation
        pct = (math.exp(self.CI_FLOOR) - 1) * 100
        self.assertAlmostEqual(pct, -98.99, places=1)

    def test_ci_cap_value(self):
        """CI cap should be 1.61 (≈+400% gain)."""
        self.assertAlmostEqual(self.CI_CAP, 1.61, places=2)
        # Verify the percentage interpretation
        pct = (math.exp(self.CI_CAP) - 1) * 100
        self.assertAlmostEqual(pct, 400.0, places=0)


class TestCIClampingIntegration(unittest.TestCase):
    """Integration tests: verify CI clamping works in the signal pipeline."""

    def _make_synthetic_signals(self, daily_sigma, H, asset_type="equity"):
        """Generate signals with controlled volatility to test clamping.

        Creates minimal feature set and calls latest_signals() for one horizon.
        """
        from decision.signals import latest_signals

        np.random.seed(42)
        n = max(500, H + 252)  # enough history
        dates = pd.bdate_range(start='2023-01-01', periods=n)
        returns = np.random.normal(0.0002, daily_sigma, n)
        px = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=dates, name='price')

        # Minimal feature dict
        ret = px.pct_change().fillna(0)
        mu = ret.ewm(span=63).mean()
        vol = ret.ewm(span=21).std()
        vol_regime = vol / vol.rolling(252, min_periods=63).median()
        vol_regime = vol_regime.fillna(1.0)

        feats = {
            "px": px,
            "ret": ret,
            "mu": mu,
            "mu_post": mu,
            "vol": vol,
            "vol_regime": vol_regime,
            "trend_z": pd.Series(0.0, index=dates),
            "z5": pd.Series(0.0, index=dates),
            "nu": pd.Series(8.0, index=dates),
            "nu_hat": pd.Series(8.0, index=dates),
            "skew": pd.Series(0.0, index=dates),
            "mom21": pd.Series(0.0, index=dates),
            "mom63": pd.Series(0.0, index=dates),
            "mom126": pd.Series(0.0, index=dates),
            "mom252": pd.Series(0.0, index=dates),
        }

        last_close = float(px.iloc[-1])
        sigs, _ = latest_signals(
            feats, [H], last_close,
            tuned_params=None, asset_key="TEST",
            asset_type=asset_type
        )
        return sigs[0] if sigs else None

    def test_normal_equity_not_affected(self):
        """Normal equity (σ_daily ≈ 1.5%) should not hit caps."""
        sig = self._make_synthetic_signals(daily_sigma=0.015, H=21)
        self.assertIsNotNone(sig)
        # CI should be small and finite
        self.assertTrue(np.isfinite(sig.ci_low))
        self.assertTrue(np.isfinite(sig.ci_high))
        # Should be well within bounds (not hitting caps)
        self.assertGreater(sig.ci_low, -0.5)  # Far from -4.6
        self.assertLess(sig.ci_high, 0.5)     # Far from 1.61

    def test_extreme_equity_ci_capped(self):
        """Extreme-vol equity (σ_daily ≈ 50%) should have CI capped."""
        sig = self._make_synthetic_signals(daily_sigma=0.50, H=252)
        self.assertIsNotNone(sig)
        # CI must respect physical limits
        self.assertGreaterEqual(sig.ci_low, -4.6)
        self.assertLessEqual(sig.ci_high, 1.61)
        # Profit must respect limits
        self.assertGreaterEqual(sig.profit_ci_low_pln, -1_000_000)
        self.assertLessEqual(sig.profit_ci_high_pln, 4_000_000)

    def test_currency_tighter_ci(self):
        """Currency should have tighter CI than equity for same volatility."""
        sig_equity = self._make_synthetic_signals(daily_sigma=0.01, H=63,
                                                   asset_type="equity")
        sig_currency = self._make_synthetic_signals(daily_sigma=0.01, H=63,
                                                     asset_type="currency")
        self.assertIsNotNone(sig_equity)
        self.assertIsNotNone(sig_currency)
        # Currency CI width should be <= equity CI width
        width_equity = sig_equity.ci_high - sig_equity.ci_low
        width_currency = sig_currency.ci_high - sig_currency.ci_low
        self.assertLessEqual(width_currency, width_equity + 0.001)

    def test_all_horizons_bounded(self):
        """All standard horizons should produce bounded outputs."""
        from decision.signals import DEFAULT_HORIZONS
        for H in DEFAULT_HORIZONS:
            sig = self._make_synthetic_signals(daily_sigma=0.30, H=H)
            if sig is None:
                continue  # may skip if not enough data
            self.assertGreaterEqual(sig.ci_low, -4.6,
                                  f"CI low out of bounds at H={H}")
            self.assertLessEqual(sig.ci_high, 1.61,
                               f"CI high out of bounds at H={H}")
            self.assertGreaterEqual(sig.profit_ci_low_pln, -1_000_000,
                                  f"Profit CI low out of bounds at H={H}")
            self.assertLessEqual(sig.profit_ci_high_pln, 4_000_000,
                               f"Profit CI high out of bounds at H={H}")

    def test_profit_physical_limits(self):
        """Profit values must never exceed physical limits."""
        sig = self._make_synthetic_signals(daily_sigma=0.50, H=252)
        self.assertIsNotNone(sig)
        # Can't lose more than 100% of notional
        self.assertGreaterEqual(sig.profit_pln, -1_000_000)
        self.assertGreaterEqual(sig.profit_ci_low_pln, -1_000_000)
        # Can't gain more than 400% of notional
        self.assertLessEqual(sig.profit_pln, 4_000_000)
        self.assertLessEqual(sig.profit_ci_high_pln, 4_000_000)

    def test_ci_ordering_preserved(self):
        """ci_low should always be <= ci_high after clamping."""
        for sigma in [0.01, 0.05, 0.20, 0.50]:
            for H in [1, 21, 63, 252]:
                sig = self._make_synthetic_signals(daily_sigma=sigma, H=H)
                if sig is None:
                    continue
                self.assertLessEqual(sig.ci_low, sig.ci_high,
                                   f"CI ordering violated at σ={sigma}, H={H}")


class TestDisplayBounds(unittest.TestCase):
    """Test display formatting respects bounds."""

    def test_format_profit_with_signal_loss_floor(self):
        """format_profit_with_signal should floor at -100%."""
        from decision.signals_ux import format_profit_with_signal
        result = format_profit_with_signal("SELL", -2_000_000, 1_000_000)
        # Should contain -100.0%, not -200%
        self.assertIn("-100.0%", result)

    def test_format_profit_with_signal_gain_cap(self):
        """format_profit_with_signal should cap at +500%."""
        from decision.signals_ux import format_profit_with_signal
        result = format_profit_with_signal("BUY", 10_000_000, 1_000_000)
        # Should contain +500.0%, not +1000%
        self.assertIn("500.0%", result)

    def test_format_profit_normal_values(self):
        """Normal profit values should display unchanged."""
        from decision.signals_ux import format_profit_with_signal
        result = format_profit_with_signal("BUY", 50_000, 1_000_000)
        self.assertIn("+5.0%", result)

    def test_format_profit_nonfinite(self):
        """Non-finite profit should display 0.0%."""
        from decision.signals_ux import format_profit_with_signal
        result = format_profit_with_signal("HOLD", float('nan'), 1_000_000)
        self.assertIn("0.0%", result)
        result_inf = format_profit_with_signal("HOLD", float('inf'), 1_000_000)
        self.assertIn("0.0%", result_inf)


class TestSigHCapEffect(unittest.TestCase):
    """Verify sig_H capping prevents absurd numbers end-to-end."""

    def test_abtc_like_extreme_vol(self):
        """Simulate ABTC-like asset: 50% daily vol, verify bounds."""
        from decision.signals import _compute_sig_h_cap

        # ABTC-like: daily σ ≈ 50%, at H=252
        raw_sig_H = 0.50 * math.sqrt(252)  # ≈ 7.94 (794%!)
        cap = _compute_sig_h_cap(252, "equity")  # Should be ≈ 1.20

        self.assertAlmostEqual(cap, 1.20, places=2)
        self.assertLess(cap, raw_sig_H)  # Cap should be much smaller than raw

        # CI with cap: mu_H ± z_star × cap
        # z_star ≈ 1.0 for Student-t(8)
        z_star = 1.0
        mu_H = -0.5  # modest negative forecast
        ci_low = mu_H - z_star * cap   # -0.5 - 1.2 = -1.7
        ci_high = mu_H + z_star * cap  # -0.5 + 1.2 = 0.7

        # Both within [-4.6, 1.61] physical limits
        self.assertGreater(ci_low, -4.6)
        self.assertLess(ci_high, 1.61)

        # As percentages: [-82%, +101%] — reasonable!
        ci_low_pct = (math.exp(ci_low) - 1) * 100
        ci_high_pct = (math.exp(ci_high) - 1) * 100
        self.assertGreater(ci_low_pct, -99)
        self.assertLess(ci_high_pct, 400)

    def test_normal_spy_unaffected(self):
        """SPY-like asset (σ_daily ≈ 1.2%) should not be capped."""
        from decision.signals import _compute_sig_h_cap

        daily_sigma = 0.012  # SPY typical
        for H in [1, 7, 21, 63, 252]:
            raw_sig_H = daily_sigma * math.sqrt(H)
            cap = _compute_sig_h_cap(H, "equity")
            # Raw should be well below cap
            self.assertLess(raw_sig_H, cap,
                          f"SPY-like sig_H should not be capped at H={H}")


if __name__ == "__main__":
    unittest.main()
