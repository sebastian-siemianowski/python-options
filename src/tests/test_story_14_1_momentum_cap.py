"""
Story 14.1 – Dynamic Momentum Cap Calibration
===============================================
Verify compute_dynamic_momentum_cap() dispatches correct k per asset class,
cap binding monitoring works, and apply_momentum_cap() clips properly.
"""

import os, sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.numba_wrappers import (
    compute_dynamic_momentum_cap,
    monitor_cap_binding,
    apply_momentum_cap,
    _classify_for_momentum,
)


class TestAssetClassDispatch:
    """Asset class detection for momentum caps."""

    @pytest.mark.parametrize("sym,expected", [
        ("SPY", "index"),
        ("AAPL", "large_cap"),
        ("BTC-USD", "crypto"),
        ("MSTR", "high_vol_equity"),
        ("GC=F", "metals_gold"),
        ("SI=F", "metals_silver"),
        ("HG=F", "metals_other"),
        ("CNYJPY=X", "forex"),
        (None, "large_cap"),
        ("UNKNOWN_TICKER", "large_cap"),
    ])
    def test_classify(self, sym, expected):
        assert _classify_for_momentum(sym) == expected


class TestDynamicMomentumCap:
    """compute_dynamic_momentum_cap k values and u_max."""

    def test_standard_k3(self):
        """Standard assets get k=3.0."""
        result = compute_dynamic_momentum_cap(1e-5, "SPY")
        assert result["k"] == 3.0
        assert abs(result["u_max"] - 3.0 * np.sqrt(1e-5)) < 1e-15

    def test_high_vol_k2(self):
        """High-vol assets get k=2.0."""
        result = compute_dynamic_momentum_cap(1e-5, "MSTR")
        assert result["k"] == 2.0

    def test_crypto_k2(self):
        """Crypto assets get k=2.0."""
        result = compute_dynamic_momentum_cap(1e-5, "BTC-USD")
        assert result["k"] == 2.0

    def test_gold_k4(self):
        """Gold/slow assets get k=4.0."""
        result = compute_dynamic_momentum_cap(1e-5, "GC=F")
        assert result["k"] == 4.0

    def test_k_override(self):
        """k_override overrides asset class."""
        result = compute_dynamic_momentum_cap(1e-5, "SPY", k_override=5.0)
        assert result["k"] == 5.0

    def test_u_max_scales_with_q(self):
        """u_max = k * sqrt(q)."""
        r1 = compute_dynamic_momentum_cap(1e-4, "SPY")
        r2 = compute_dynamic_momentum_cap(4e-4, "SPY")
        assert abs(r2["u_max"] / r1["u_max"] - 2.0) < 1e-10

    def test_zero_q(self):
        """q=0 gives u_max=0."""
        result = compute_dynamic_momentum_cap(0.0, "SPY")
        assert result["u_max"] == 0.0


class TestCapBinding:
    """monitor_cap_binding() rate and recommendations."""

    def test_moderate_binding(self):
        """10% binding -> 'ok'."""
        rng = np.random.default_rng(42)
        n = 1000
        mom = rng.normal(0, 0.001, n)
        # Force ~10% to be at cap
        cap_idx = rng.choice(n, size=100, replace=False)
        mom[cap_idx] = 0.005
        result = monitor_cap_binding(mom, 0.005)
        assert result["recommendation"] == "ok"
        assert 0.05 <= result["binding_rate"] <= 0.15

    def test_high_binding_increase_k(self):
        """High binding rate -> recommend increase_k."""
        mom = np.full(100, 0.01)  # all at cap
        result = monitor_cap_binding(mom, 0.01)
        assert result["binding_rate"] > 0.20
        assert result["recommendation"] == "increase_k"

    def test_low_binding_decrease_k(self):
        """Low binding rate -> recommend decrease_k."""
        mom = np.full(100, 0.001)
        result = monitor_cap_binding(mom, 1.0)  # cap never reached
        assert result["binding_rate"] < 0.03
        assert result["recommendation"] == "decrease_k"

    def test_empty_array(self):
        result = monitor_cap_binding(np.array([]), 1.0)
        assert result["binding_rate"] == 0.0
        assert result["recommendation"] == "ok"


class TestApplyMomentumCap:
    """apply_momentum_cap() clips correctly."""

    def test_clips_to_u_max(self):
        """All values clipped to [-u_max, u_max]."""
        mom_raw = np.array([-0.1, -0.01, 0.0, 0.01, 0.1])
        capped, info = apply_momentum_cap(mom_raw, 1e-4, "SPY")
        u_max = info["u_max"]
        assert np.all(capped <= u_max + 1e-15)
        assert np.all(capped >= -u_max - 1e-15)

    def test_small_values_unchanged(self):
        """Values within cap are unchanged."""
        mom_raw = np.array([1e-8, -1e-8, 0.0])
        capped, info = apply_momentum_cap(mom_raw, 1e-4, "SPY")
        np.testing.assert_array_equal(capped, mom_raw)

    def test_zero_q_clips_to_zero(self):
        """q=0 -> all momentum clipped to zero."""
        mom_raw = np.array([0.01, -0.01])
        capped, info = apply_momentum_cap(mom_raw, 0.0, "SPY")
        np.testing.assert_array_equal(capped, np.zeros(2))

    def test_returns_diagnostics(self):
        """Diagnostics contain all expected keys."""
        mom_raw = np.array([0.01, -0.01, 0.0])
        _, info = apply_momentum_cap(mom_raw, 1e-4, "SPY")
        expected_keys = {"k", "u_max", "asset_class", "sqrt_q",
                        "binding_rate", "n_bound", "n_total", "recommendation"}
        assert expected_keys.issubset(set(info.keys()))

    @pytest.mark.parametrize("sym,expected_k", [
        ("SPY", 3.0), ("MSTR", 2.0), ("GC=F", 4.0), ("BTC-USD", 2.0),
    ])
    def test_per_asset_class_cap(self, sym, expected_k):
        """Cap varies correctly by asset class."""
        _, info = apply_momentum_cap(np.zeros(10), 1e-4, sym)
        assert info["k"] == expected_k
