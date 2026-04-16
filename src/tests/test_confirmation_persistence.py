"""
Story 3.3: Fix Two-Day Confirmation Logic

Tests for signal state persistence (load/save) and proper two-day confirmation
where p_prev comes from the previous run, not from p_now.
"""
import json
import os
import sys
import tempfile
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


class TestSignalStateConstants(unittest.TestCase):
    """Verify constants are exported."""

    def test_signal_state_dir_exists(self):
        from decision.signals import SIGNAL_STATE_DIR
        self.assertIn("signal_state", SIGNAL_STATE_DIR)

    def test_default_p_is_neutral(self):
        from decision.signals import SIGNAL_STATE_DEFAULT_P
        self.assertEqual(SIGNAL_STATE_DEFAULT_P, 0.5)


class TestSignalStatePersistence(unittest.TestCase):
    """Test load/save of signal state files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_nonexistent_returns_empty(self):
        from decision.signals import load_signal_state
        state = load_signal_state("FAKE_SYMBOL_XYZ", state_dir=self.tmpdir)
        self.assertEqual(state, {})

    def test_save_and_load_roundtrip(self):
        from decision.signals import load_signal_state, save_signal_state
        state = {
            "7": {"p_up": 0.62, "label": "BUY"},
            "21": {"p_up": 0.48, "label": "HOLD"},
        }
        path = save_signal_state("TEST_ASSET", state, state_dir=self.tmpdir)
        self.assertTrue(os.path.isfile(path))

        loaded = load_signal_state("TEST_ASSET", state_dir=self.tmpdir)
        self.assertEqual(loaded["7"]["p_up"], 0.62)
        self.assertEqual(loaded["7"]["label"], "BUY")
        self.assertEqual(loaded["21"]["p_up"], 0.48)

    def test_save_creates_directory(self):
        from decision.signals import save_signal_state
        nested = os.path.join(self.tmpdir, "sub", "dir")
        save_signal_state("X", {"1": {"p_up": 0.5, "label": "HOLD"}}, state_dir=nested)
        self.assertTrue(os.path.isdir(nested))

    def test_load_corrupt_file_returns_empty(self):
        from decision.signals import load_signal_state
        path = os.path.join(self.tmpdir, "CORRUPT.json")
        with open(path, "w") as f:
            f.write("not valid json{{{")
        state = load_signal_state("CORRUPT", state_dir=self.tmpdir)
        self.assertEqual(state, {})

    def test_save_overwrites_existing(self):
        from decision.signals import load_signal_state, save_signal_state
        save_signal_state("UPD", {"7": {"p_up": 0.55, "label": "BUY"}}, state_dir=self.tmpdir)
        save_signal_state("UPD", {"7": {"p_up": 0.42, "label": "SELL"}}, state_dir=self.tmpdir)
        loaded = load_signal_state("UPD", state_dir=self.tmpdir)
        self.assertAlmostEqual(loaded["7"]["p_up"], 0.42)
        self.assertEqual(loaded["7"]["label"], "SELL")

    def test_saved_json_is_valid(self):
        from decision.signals import save_signal_state
        save_signal_state("JSON_CHECK", {"1": {"p_up": 0.5, "label": "HOLD"}}, state_dir=self.tmpdir)
        path = os.path.join(self.tmpdir, "JSON_CHECK.json")
        with open(path) as f:
            data = json.load(f)
        self.assertIn("1", data)


class TestConfirmationLogicFix(unittest.TestCase):
    """Test that apply_confirmation_logic uses different prev/now values."""

    def test_same_prev_now_is_buy(self):
        """When prev and now both exceed buy threshold, label is BUY."""
        from decision.signals import apply_confirmation_logic
        label = apply_confirmation_logic(
            p_smoothed_now=0.60,
            p_smoothed_prev=0.60,
            p_raw=0.60,
            pos_strength=0.2,
            buy_thr=0.55,
            sell_thr=0.45,
            edge=0.5,
            edge_floor=0.01,
        )
        self.assertEqual(label, "BUY")

    def test_prev_below_now_above_is_hold(self):
        """When prev is below buy threshold but now is above, confirmation fails -> HOLD."""
        from decision.signals import apply_confirmation_logic
        label = apply_confirmation_logic(
            p_smoothed_now=0.60,
            p_smoothed_prev=0.50,  # below buy_enter (0.56)
            p_raw=0.60,
            pos_strength=0.2,
            buy_thr=0.55,
            sell_thr=0.45,
            edge=0.5,
            edge_floor=0.01,
        )
        self.assertEqual(label, "HOLD")

    def test_first_run_uses_p_now_as_fallback(self):
        """On first run (no state), p_prev = p_now for backward compatibility.
        This allows labels to be generated normally on first run.
        Confirmation only engages on subsequent runs."""
        from decision.signals import apply_confirmation_logic
        # Simulate first run: p_prev = p_now (no state), p_now = 0.58
        p_now = 0.58
        p_prev = p_now  # first run fallback
        alpha_p = 0.7
        p_s_prev = p_prev
        p_s_now = alpha_p * p_now + (1 - alpha_p) * p_prev

        label = apply_confirmation_logic(
            p_smoothed_now=p_s_now,
            p_smoothed_prev=p_s_prev,
            p_raw=p_now,
            pos_strength=0.2,
            buy_thr=0.55,
            sell_thr=0.45,
            edge=0.5,
            edge_floor=0.01,
        )
        # p_s_prev = 0.58 > 0.56 (buy_enter), p_s_now = 0.58 > 0.56, so BUY
        self.assertEqual(label, "BUY")

    def test_second_run_with_state_can_buy(self):
        """Second run: if previous state had p_up=0.58, and now=0.60, both above -> BUY."""
        from decision.signals import apply_confirmation_logic
        p_prev = 0.58  # from loaded state
        p_now = 0.60
        alpha_p = 0.7
        p_s_prev = p_prev
        p_s_now = alpha_p * p_now + (1 - alpha_p) * p_prev

        label = apply_confirmation_logic(
            p_smoothed_now=p_s_now,
            p_smoothed_prev=p_s_prev,
            p_raw=p_now,
            pos_strength=0.2,
            buy_thr=0.55,
            sell_thr=0.45,
            edge=0.5,
            edge_floor=0.01,
        )
        # p_s_prev = 0.58 > 0.56, p_s_now = 0.7*0.60 + 0.3*0.58 = 0.594 > 0.56
        self.assertEqual(label, "BUY")

    def test_sell_confirmation_requires_two_days(self):
        """SELL requires both prev and now below sell threshold."""
        from decision.signals import apply_confirmation_logic
        # Both below sell_enter (0.44)
        label = apply_confirmation_logic(
            p_smoothed_now=0.40,
            p_smoothed_prev=0.42,
            p_raw=0.40,
            pos_strength=0.2,
            buy_thr=0.55,
            sell_thr=0.45,
            edge=0.5,
            edge_floor=0.01,
        )
        self.assertEqual(label, "SELL")

    def test_sell_blocked_if_prev_neutral(self):
        """SELL blocked when prev was neutral (above sell threshold)."""
        from decision.signals import apply_confirmation_logic
        label = apply_confirmation_logic(
            p_smoothed_now=0.40,
            p_smoothed_prev=0.50,  # above sell_enter
            p_raw=0.40,
            pos_strength=0.2,
            buy_thr=0.55,
            sell_thr=0.45,
            edge=0.5,
            edge_floor=0.01,
        )
        self.assertEqual(label, "HOLD")

    def test_strong_buy_overrides_confirmation(self):
        """STRONG BUY ignores two-day confirmation (raw conviction high enough)."""
        from decision.signals import apply_confirmation_logic
        label = apply_confirmation_logic(
            p_smoothed_now=0.70,
            p_smoothed_prev=0.50,  # prev below threshold
            p_raw=0.70,
            pos_strength=0.35,
            buy_thr=0.55,
            sell_thr=0.45,
            edge=0.5,
            edge_floor=0.01,
        )
        self.assertEqual(label, "STRONG BUY")

    def test_edge_floor_forces_hold(self):
        """Even if both days confirm BUY, edge below floor forces HOLD."""
        from decision.signals import apply_confirmation_logic
        label = apply_confirmation_logic(
            p_smoothed_now=0.60,
            p_smoothed_prev=0.60,
            p_raw=0.60,
            pos_strength=0.2,
            buy_thr=0.55,
            sell_thr=0.45,
            edge=0.005,  # below floor
            edge_floor=0.01,
        )
        self.assertEqual(label, "HOLD")


if __name__ == "__main__":
    unittest.main()
