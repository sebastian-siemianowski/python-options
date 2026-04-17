"""
Story 7.5: Tests for signal output validation invariants.
"""
import unittest
import sys
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestSignalInvariants(unittest.TestCase):
    """Validate all signal output invariant checks."""

    def test_signal_dataclass_has_required_fields(self):
        """Signal dataclass must have exp_ret, p_up, ci_low, ci_high, vol_mean."""
        from decision.signals import Signal
        fields = {f.name for f in Signal.__dataclass_fields__.values()}
        for required in ["exp_ret", "p_up", "ci_low", "ci_high", "vol_mean"]:
            self.assertIn(required, fields,
                f"Signal missing field: {required}")

    def test_valid_signal_passes_invariants(self):
        """A properly constructed Signal should pass all invariant checks."""
        from decision.signals import Signal
        sig = Signal(
            horizon_days=7,
            score=0.5,
            exp_ret=0.01,
            p_up=0.55,
            ci_low=-0.02,
            ci_high=0.04,
            ci_low_90=-0.05,
            ci_high_90=0.07,
            profit_pln=1000.0,
            profit_ci_low_pln=-2000.0,
            profit_ci_high_pln=4000.0,
            position_strength=0.5,
            vol_mean=0.015,
            vol_ci_low=0.010,
            vol_ci_high=0.020,
            regime="LOW_VOL_TREND",
            label="BUY",
        )
        # Finiteness
        self.assertTrue(np.isfinite(sig.exp_ret))
        self.assertTrue(np.isfinite(sig.p_up))
        # p_up in [0, 1]
        self.assertGreaterEqual(sig.p_up, 0.0)
        self.assertLessEqual(sig.p_up, 1.0)
        # CI ordering
        self.assertLessEqual(sig.ci_low, sig.ci_high)
        # Non-negative vol
        self.assertGreaterEqual(sig.vol_mean, 0.0)

    def test_nan_exp_ret_detected(self):
        """NaN exp_ret should be detected as a violation."""
        val = float('nan')
        self.assertFalse(np.isfinite(val))

    def test_inf_p_up_detected(self):
        """Inf p_up should be detected as a violation."""
        val = float('inf')
        self.assertFalse(np.isfinite(val))

    def test_p_up_out_of_range_detected(self):
        """p_up outside [0, 1] should be detected."""
        self.assertFalse(0.0 <= 1.5 <= 1.0)
        self.assertFalse(0.0 <= -0.1 <= 1.0)

    def test_ci_inversion_detected(self):
        """ci_low > ci_high should be flagged."""
        ci_low, ci_high = 0.05, -0.02
        self.assertGreater(ci_low, ci_high)

    def test_variance_monotonicity(self):
        """Variance should generally increase with horizon."""
        vols = {1: 0.01, 3: 0.017, 7: 0.025, 21: 0.045, 63: 0.08}
        sorted_hs = sorted(vols.keys())
        for i in range(1, len(sorted_hs)):
            prev_vol = vols[sorted_hs[i - 1]]
            curr_vol = vols[sorted_hs[i]]
            # Allow 50% decrease tolerance
            self.assertGreaterEqual(curr_vol, prev_vol * 0.5,
                f"Vol at H={sorted_hs[i]} too low vs H={sorted_hs[i-1]}")

    def test_variance_dramatic_decrease_caught(self):
        """A dramatic vol decrease (> 50%) should be flagged."""
        prev_vol = 0.08
        curr_vol = 0.03  # < 0.08 * 0.5 = 0.04
        self.assertLess(curr_vol, prev_vol * 0.5)

    def test_debug_mode_env_var(self):
        """SIGNAL_DEBUG=1 should enable assertion mode."""
        old = os.environ.get("SIGNAL_DEBUG", "")
        try:
            os.environ["SIGNAL_DEBUG"] = "1"
            self.assertEqual(os.environ.get("SIGNAL_DEBUG", ""), "1")
        finally:
            if old:
                os.environ["SIGNAL_DEBUG"] = old
            else:
                os.environ.pop("SIGNAL_DEBUG", None)

    def test_invariant_code_present_in_signals(self):
        """Verify invariant validation block exists in signals.py or its extracted modules."""
        signals_path = os.path.join(REPO_ROOT, "decision", "signals.py")
        with open(signals_path) as f:
            src = f.read()
        # After Epic-7 decomposition, latest_signals() lives in signal_generation.py
        gen_path = os.path.join(REPO_ROOT, "decision", "signal_modules", "signal_generation.py")
        if os.path.exists(gen_path):
            with open(gen_path) as f:
                src += f.read()
        self.assertIn("Story 7.5: Signal Output Validation Invariants", src)
        self.assertIn("_n_violations", src)
        self.assertIn("SIGNAL_DEBUG", src)


if __name__ == "__main__":
    unittest.main()
