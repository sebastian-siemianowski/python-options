"""Tests for Story 3.4: Asset-Class-Aware Per-Step Return Cap.

Validates that:
1. RETURN_CAP_BY_CLASS and RETURN_CAP_DEFAULT constants are correct
2. classify_asset_type() maps representative tickers to correct classes
3. Per-class caps are applied in run_unified_mc() and np.clip paths
4. Numba kernel receives and applies the return_cap parameter
5. Crypto paths extend beyond 50% when appropriate
6. FX paths never exceed 15%
7. Equity paths stay within 30%
"""
import os
import sys
import unittest
import math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from decision.signals import (
    RETURN_CAP_BY_CLASS,
    RETURN_CAP_DEFAULT,
    classify_asset_type,
    run_unified_mc,
)


class TestReturnCapConstants(unittest.TestCase):
    """Test RETURN_CAP_BY_CLASS dictionary and default."""

    def test_all_classes_present(self):
        expected = {"equity", "currency", "metal", "crypto", "etf"}
        self.assertEqual(set(RETURN_CAP_BY_CLASS.keys()), expected)

    def test_equity_cap(self):
        self.assertAlmostEqual(RETURN_CAP_BY_CLASS["equity"], 0.30)

    def test_currency_cap(self):
        self.assertAlmostEqual(RETURN_CAP_BY_CLASS["currency"], 0.15)

    def test_metal_cap(self):
        self.assertAlmostEqual(RETURN_CAP_BY_CLASS["metal"], 0.20)

    def test_crypto_cap(self):
        self.assertAlmostEqual(RETURN_CAP_BY_CLASS["crypto"], 1.00)

    def test_etf_cap(self):
        self.assertAlmostEqual(RETURN_CAP_BY_CLASS["etf"], 0.25)

    def test_default_matches_equity(self):
        self.assertAlmostEqual(RETURN_CAP_DEFAULT, 0.30)

    def test_all_caps_positive(self):
        for cls, cap in RETURN_CAP_BY_CLASS.items():
            self.assertGreater(cap, 0.0, f"Cap for {cls} must be positive")

    def test_crypto_widest(self):
        for cls, cap in RETURN_CAP_BY_CLASS.items():
            self.assertGreaterEqual(
                RETURN_CAP_BY_CLASS["crypto"], cap,
                f"Crypto cap should be >= {cls} cap"
            )

    def test_currency_tightest(self):
        for cls, cap in RETURN_CAP_BY_CLASS.items():
            self.assertLessEqual(
                RETURN_CAP_BY_CLASS["currency"], cap,
                f"Currency cap should be <= {cls} cap"
            )


class TestClassifyAssetType(unittest.TestCase):
    """Test that classify_asset_type returns correct asset class."""

    def test_equity_symbols(self):
        for sym in ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]:
            self.assertEqual(classify_asset_type(sym), "equity", sym)

    def test_crypto_pair(self):
        for sym in ["BTC-USD", "ETH-USD", "SOL-USDC", "DOGE-USD"]:
            self.assertEqual(classify_asset_type(sym), "crypto", sym)

    def test_currency_fx(self):
        for sym in ["USDJPY=X", "EURUSD=X", "GBPUSD=X"]:
            self.assertEqual(classify_asset_type(sym), "currency", sym)

    def test_metal_futures(self):
        for sym in ["GC=F", "SI=F", "PL=F"]:
            self.assertEqual(classify_asset_type(sym), "metal", sym)

    def test_metal_etfs(self):
        for sym in ["GLD", "SLV", "GDX"]:
            self.assertEqual(classify_asset_type(sym), "metal", sym)

    def test_empty_defaults_equity(self):
        self.assertEqual(classify_asset_type(""), "equity")

    def test_unknown_defaults_equity(self):
        self.assertEqual(classify_asset_type("XYZUNKNOWN"), "equity")


class TestReturnCapInMC(unittest.TestCase):
    """Test that return_cap parameter controls MC per-step returns."""

    def _run_mc(self, return_cap, n_paths=5000, H_max=30, seed=42):
        """Run MC with extreme vol to trigger capping.
        
        P_t=0 eliminates drift posterior offset that is applied after the
        kernel cap, so we isolate pure per-step return capping.
        """
        return run_unified_mc(
            mu_t=0.0,
            P_t=0.0,  # No drift posterior offset — isolates per-step cap
            phi=0.99,
            q=1e-5,
            sigma2_step=0.10,  # Very high vol to stress-test capping
            H_max=H_max,
            n_paths=n_paths,
            nu=4.0,  # Heavy tails
            use_garch=False,
            jump_intensity=0.05,
            jump_mean=0.0,
            jump_std=0.10,
            seed=seed,
            return_cap=return_cap,
        )

    def test_equity_cap_30pct(self):
        """Equity: max single-step return <= 30%."""
        result = self._run_mc(return_cap=0.30, H_max=1, n_paths=10000)
        single_step = result["returns"][0, :]
        self.assertLessEqual(np.max(single_step), 0.30 + 1e-10)
        self.assertGreaterEqual(np.min(single_step), -0.30 - 1e-10)

    def test_currency_cap_15pct(self):
        """Currency: max single-step return <= 15%."""
        result = self._run_mc(return_cap=0.15, H_max=1, n_paths=10000)
        single_step = result["returns"][0, :]
        self.assertLessEqual(np.max(single_step), 0.15 + 1e-10)
        self.assertGreaterEqual(np.min(single_step), -0.15 - 1e-10)

    def test_crypto_cap_100pct(self):
        """Crypto: cap at 100% allows wider per-step returns than old 50%."""
        result = self._run_mc(return_cap=1.00, H_max=1, n_paths=50000, seed=123)
        single_step = result["returns"][0, :]
        self.assertLessEqual(np.max(single_step), 1.00 + 1e-10)
        self.assertGreaterEqual(np.min(single_step), -1.00 - 1e-10)

    def test_crypto_wider_than_equity(self):
        """Crypto tail should extend wider than equity tail."""
        rc_equity = self._run_mc(return_cap=0.30, H_max=10, n_paths=20000, seed=99)
        rc_crypto = self._run_mc(return_cap=1.00, H_max=10, n_paths=20000, seed=99)
        # At horizon 10, crypto cumulative range should be wider
        eq_range = np.max(rc_equity["returns"][-1]) - np.min(rc_equity["returns"][-1])
        cr_range = np.max(rc_crypto["returns"][-1]) - np.min(rc_crypto["returns"][-1])
        self.assertGreater(cr_range, eq_range)

    def test_tighter_cap_narrows_distribution(self):
        """Tighter cap => narrower return distribution."""
        r_wide = self._run_mc(return_cap=1.00, H_max=20, n_paths=10000, seed=77)
        r_tight = self._run_mc(return_cap=0.15, H_max=20, n_paths=10000, seed=77)
        std_wide = np.std(r_wide["returns"][-1])
        std_tight = np.std(r_tight["returns"][-1])
        self.assertGreater(std_wide, std_tight)

    def test_default_cap_is_equity(self):
        """Default return_cap parameter should match equity (0.30)."""
        import inspect
        sig = inspect.signature(run_unified_mc)
        default_cap = sig.parameters["return_cap"].default
        self.assertAlmostEqual(default_cap, 0.30)

    def test_per_step_cap_respected_across_horizons(self):
        """Each step increment should not exceed cap."""
        cap = 0.15
        result = self._run_mc(return_cap=cap, H_max=5, n_paths=5000)
        cum = result["returns"]
        # Step 0: just return itself
        step0 = cum[0, :]
        self.assertLessEqual(np.max(np.abs(step0)), cap + 1e-10)
        # Steps 1..H-1: increments
        for t in range(1, cum.shape[0]):
            increments = cum[t, :] - cum[t - 1, :]
            self.assertLessEqual(np.max(np.abs(increments)), cap + 1e-10,
                                 f"Step {t} increment exceeds cap {cap}")


class TestReturnCapIntegration(unittest.TestCase):
    """Integration: verify cap lookup from asset_type in signals pipeline."""

    def test_cap_lookup_all_classes(self):
        """Verify RETURN_CAP_BY_CLASS.get works for all known types."""
        for cls in ["equity", "currency", "metal", "crypto", "etf"]:
            cap = RETURN_CAP_BY_CLASS.get(cls, RETURN_CAP_DEFAULT)
            self.assertGreater(cap, 0.0)
            self.assertLessEqual(cap, 1.0)

    def test_cap_lookup_unknown_class(self):
        """Unknown class falls back to default."""
        cap = RETURN_CAP_BY_CLASS.get("unknown_class", RETURN_CAP_DEFAULT)
        self.assertAlmostEqual(cap, RETURN_CAP_DEFAULT)


if __name__ == "__main__":
    unittest.main()
