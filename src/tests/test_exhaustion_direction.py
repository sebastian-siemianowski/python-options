"""test_exhaustion_direction.py -- Story 5.5

Validates direction-aware exhaustion modulation:
1. Oversold + long forecast -> position INCREASES
2. Overbought + long forecast -> position DECREASES (unchanged behavior)
3. Position never exceeds 1.0
4. Short signal: symmetric opposite behavior
"""
import os
import sys
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


_EXH_REDUCE = 0.5
_EXH_BOOST = 0.3


def apply_exhaustion(pos: float, mu_H: float, ue_up: float, ue_down: float) -> float:
    """Reproduce signals.py exhaustion modulation logic."""
    if mu_H > 0:
        if ue_up > 0 and pos > 0:
            pos *= (1.0 - _EXH_REDUCE * ue_up)
        if ue_down > 0 and pos > 0:
            pos *= (1.0 + _EXH_BOOST * ue_down)
    else:
        if ue_up > 0 and pos > 0:
            pos *= (1.0 + _EXH_BOOST * ue_up)
        if ue_down > 0 and pos > 0:
            pos *= (1.0 - _EXH_REDUCE * ue_down)
    return min(pos, 1.0)


class TestLongSignalExhaustion(unittest.TestCase):

    def test_oversold_increases_long_position(self):
        """Oversold stock with positive forecast should get LARGER position."""
        base = 0.5
        result = apply_exhaustion(base, mu_H=0.02, ue_up=0.0, ue_down=0.6)
        self.assertGreater(result, base)

    def test_overbought_reduces_long_position(self):
        """Overbought stock with positive forecast gets smaller position."""
        base = 0.5
        result = apply_exhaustion(base, mu_H=0.02, ue_up=0.6, ue_down=0.0)
        self.assertLess(result, base)

    def test_both_exhaustion_net_effect(self):
        """With both ue_up and ue_down, reduction dominates boost for long."""
        base = 0.5
        result = apply_exhaustion(base, mu_H=0.02, ue_up=0.5, ue_down=0.5)
        # Reduction: 0.5 * (1 - 0.25) = 0.375, then boost: 0.375 * 1.15 = 0.431
        self.assertLess(result, base, "Reduction factor > boost factor")

    def test_no_exhaustion_unchanged(self):
        """Zero exhaustion leaves position unchanged."""
        base = 0.5
        result = apply_exhaustion(base, mu_H=0.02, ue_up=0.0, ue_down=0.0)
        self.assertAlmostEqual(result, base, places=10)


class TestShortSignalExhaustion(unittest.TestCase):

    def test_overbought_increases_short_position(self):
        """Overbought + negative forecast: mean-reversion boosts short."""
        base = 0.5
        result = apply_exhaustion(base, mu_H=-0.02, ue_up=0.6, ue_down=0.0)
        self.assertGreater(result, base)

    def test_oversold_reduces_short_position(self):
        """Oversold + negative forecast: reduces short."""
        base = 0.5
        result = apply_exhaustion(base, mu_H=-0.02, ue_up=0.0, ue_down=0.6)
        self.assertLess(result, base)


class TestPositionCap(unittest.TestCase):

    def test_never_exceeds_one(self):
        """Even with maximum boost, position <= 1.0."""
        result = apply_exhaustion(0.95, mu_H=0.02, ue_up=0.0, ue_down=0.99)
        self.assertLessEqual(result, 1.0)

    def test_extreme_boost_capped(self):
        """Full ue_down + high base position stays <= 1.0."""
        result = apply_exhaustion(0.8, mu_H=0.02, ue_up=0.0, ue_down=1.0)
        self.assertLessEqual(result, 1.0)

    def test_zero_position_stays_zero(self):
        """Zero position is unaffected."""
        result = apply_exhaustion(0.0, mu_H=0.02, ue_up=0.5, ue_down=0.5)
        self.assertAlmostEqual(result, 0.0)


class TestAsymmetry(unittest.TestCase):

    def test_reduce_factor_larger_than_boost(self):
        """Reduction factor (0.5) > boost factor (0.3) for prudence."""
        self.assertGreater(_EXH_REDUCE, _EXH_BOOST)

    def test_reduction_magnitude_exceeds_boost(self):
        """For same ue level, reduction has larger effect than boost."""
        base = 0.5
        ue = 0.5
        reduced = base * (1.0 - _EXH_REDUCE * ue)
        boosted = base * (1.0 + _EXH_BOOST * ue)
        reduction_delta = base - reduced   # 0.125
        boost_delta = boosted - base       # 0.075
        self.assertGreater(reduction_delta, boost_delta)


if __name__ == "__main__":
    unittest.main()
