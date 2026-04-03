"""test_regime_module.py -- Story 4.2: Shared Regime Module (DRY)

Validates that:
1. src/models/regime.py is the single source of truth
2. tune.py and signals.py both import from models.regime
3. Regime logic is identical across both consumers
4. MarketRegime IntEnum values match plain integer constants
5. All regime functions work correctly from the shared module
"""
import importlib
import inspect
import math
import sys
import os
import unittest

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class TestRegimeModuleSingleSource(unittest.TestCase):
    """Verify models.regime is the single source of truth."""

    def test_regime_module_exists(self):
        import models.regime
        self.assertTrue(hasattr(models.regime, 'MarketRegime'))
        self.assertTrue(hasattr(models.regime, 'assign_regime_labels'))
        self.assertTrue(hasattr(models.regime, 'assign_current_regime'))

    def test_tune_imports_from_regime(self):
        from tuning import tune
        from models import regime
        self.assertIs(tune.MarketRegime, regime.MarketRegime)
        self.assertIs(tune.REGIME_LABELS, regime.REGIME_LABELS)
        self.assertIs(tune.assign_regime_labels, regime.assign_regime_labels)

    def test_signals_imports_from_regime(self):
        from decision import signals
        from models import regime
        self.assertIs(signals.assign_current_regime, regime.assign_current_regime)
        self.assertIs(signals.compute_regime_probabilities_v2, regime.compute_regime_probabilities_v2)
        self.assertIs(signals.compute_soft_bma_weights, regime.compute_soft_bma_weights)
        self.assertIs(signals.map_regime_label_to_index, regime.map_regime_label_to_index)
        self.assertIs(signals.extract_regime_features, regime.extract_regime_features)

    def test_no_local_assign_current_regime_in_signals(self):
        """signals.py must not define assign_current_regime locally."""
        import decision.signals as sig
        source_file = inspect.getfile(sig.assign_current_regime)
        self.assertIn("regime.py", source_file,
                       "assign_current_regime should be defined in regime.py")

    def test_no_local_assign_regime_labels_in_tune(self):
        """tune.py must not define assign_regime_labels locally."""
        import tuning.tune as t
        source_file = inspect.getfile(t.assign_regime_labels)
        self.assertIn("regime.py", source_file,
                       "assign_regime_labels should be defined in regime.py")


class TestRegimeConstants(unittest.TestCase):
    """Verify regime constant values and IntEnum consistency."""

    def test_intenum_values(self):
        from models.regime import MarketRegime
        self.assertEqual(MarketRegime.LOW_VOL_TREND, 0)
        self.assertEqual(MarketRegime.HIGH_VOL_TREND, 1)
        self.assertEqual(MarketRegime.LOW_VOL_RANGE, 2)
        self.assertEqual(MarketRegime.HIGH_VOL_RANGE, 3)
        self.assertEqual(MarketRegime.CRISIS_JUMP, 4)

    def test_plain_constants_match_intenum(self):
        from models.regime import (
            MarketRegime, REGIME_LOW_VOL_TREND, REGIME_HIGH_VOL_TREND,
            REGIME_LOW_VOL_RANGE, REGIME_HIGH_VOL_RANGE, REGIME_CRISIS_JUMP
        )
        self.assertEqual(REGIME_LOW_VOL_TREND, int(MarketRegime.LOW_VOL_TREND))
        self.assertEqual(REGIME_HIGH_VOL_TREND, int(MarketRegime.HIGH_VOL_TREND))
        self.assertEqual(REGIME_LOW_VOL_RANGE, int(MarketRegime.LOW_VOL_RANGE))
        self.assertEqual(REGIME_HIGH_VOL_RANGE, int(MarketRegime.HIGH_VOL_RANGE))
        self.assertEqual(REGIME_CRISIS_JUMP, int(MarketRegime.CRISIS_JUMP))

    def test_regime_names_all_five(self):
        from models.regime import REGIME_NAMES
        self.assertEqual(len(REGIME_NAMES), 5)
        for i in range(5):
            self.assertIn(i, REGIME_NAMES)

    def test_regime_labels_all_five(self):
        from models.regime import REGIME_LABELS, MarketRegime
        self.assertEqual(len(REGIME_LABELS), 5)
        for r in MarketRegime:
            self.assertIn(r, REGIME_LABELS)

    def test_cusum_constants(self):
        from models.regime import (
            CUSUM_THRESHOLD, CUSUM_COOLDOWN,
            CUSUM_ALPHA_ACCEL, CUSUM_ALPHA_NORMAL
        )
        self.assertEqual(CUSUM_THRESHOLD, 3.0)
        self.assertEqual(CUSUM_COOLDOWN, 5)
        self.assertAlmostEqual(CUSUM_ALPHA_ACCEL, 0.85)
        self.assertAlmostEqual(CUSUM_ALPHA_NORMAL, 0.40)


class TestAssignRegimeLabels(unittest.TestCase):
    """Test the vectorized assign_regime_labels from regime.py."""

    def test_low_vol_trend(self):
        from models.regime import assign_regime_labels, MarketRegime
        rng = np.random.default_rng(42)
        n = 100
        returns = np.full(n, 0.002)  # Strong positive drift
        vol = np.full(n, 0.005)      # Very low vol
        labels = assign_regime_labels(returns, vol, lookback=21)
        # Last label should be LOW_VOL_TREND (low vol, strong drift)
        self.assertIn(labels[-1], [MarketRegime.LOW_VOL_TREND, MarketRegime.HIGH_VOL_TREND])

    def test_crisis_extreme_vol(self):
        from models.regime import assign_regime_labels, MarketRegime
        n = 100
        returns = np.random.default_rng(42).normal(0, 0.01, n)
        vol = np.full(n, 0.01)
        # Create extreme vol spike at end
        vol[-1] = 0.10  # 10x median -> vol_relative > 2.0
        labels = assign_regime_labels(returns, vol, lookback=21)
        self.assertEqual(labels[-1], MarketRegime.CRISIS_JUMP)

    def test_output_shape(self):
        from models.regime import assign_regime_labels
        n = 50
        returns = np.random.default_rng(42).normal(0, 0.01, n)
        vol = np.abs(returns) + 0.005
        labels = assign_regime_labels(returns, vol)
        self.assertEqual(labels.shape, (n,))
        for lab in labels:
            self.assertIn(lab, [0, 1, 2, 3, 4])


class TestAssignCurrentRegime(unittest.TestCase):
    """Test the current-bar assign_current_regime from regime.py."""

    def test_default_on_empty(self):
        from models.regime import assign_current_regime, REGIME_LOW_VOL_RANGE
        feats = {"ret": pd.Series(dtype=float), "vol": pd.Series(dtype=float)}
        r = assign_current_regime(feats)
        self.assertEqual(r, REGIME_LOW_VOL_RANGE)

    def test_crisis_tail(self):
        from models.regime import assign_current_regime, REGIME_CRISIS_JUMP
        vol_data = [0.01] * 50
        ret_data = [0.001] * 49 + [0.08]  # Giant tail event
        feats = {
            "ret": pd.Series(ret_data),
            "vol": pd.Series(vol_data),
        }
        r = assign_current_regime(feats, asset="test_crisis")
        self.assertEqual(r, REGIME_CRISIS_JUMP)

    def test_returns_valid_regime(self):
        from models.regime import assign_current_regime
        rng = np.random.default_rng(42)
        feats = {
            "ret": pd.Series(rng.normal(0, 0.01, 100)),
            "vol": pd.Series(np.full(100, 0.01)),
        }
        r = assign_current_regime(feats, asset="test_valid")
        self.assertIn(r, [0, 1, 2, 3, 4])


class TestMapRegimeLabelToIndex(unittest.TestCase):
    """Test string-to-index mapping."""

    def test_crisis_label(self):
        from models.regime import map_regime_label_to_index, REGIME_CRISIS_JUMP
        self.assertEqual(map_regime_label_to_index("crisis"), REGIME_CRISIS_JUMP)

    def test_high_vol_label(self):
        from models.regime import map_regime_label_to_index, REGIME_HIGH_VOL_RANGE
        self.assertEqual(map_regime_label_to_index("high_vol"), REGIME_HIGH_VOL_RANGE)

    def test_trending_label(self):
        from models.regime import map_regime_label_to_index, REGIME_LOW_VOL_TREND
        self.assertEqual(map_regime_label_to_index("trending"), REGIME_LOW_VOL_TREND)

    def test_default_fallback(self):
        from models.regime import map_regime_label_to_index, REGIME_LOW_VOL_RANGE
        self.assertEqual(map_regime_label_to_index("unknown_xyz"), REGIME_LOW_VOL_RANGE)


class TestExtractRegimeFeatures(unittest.TestCase):
    """Test feature extraction for regime scoring."""

    def test_returns_all_keys(self):
        from models.regime import extract_regime_features
        rng = np.random.default_rng(42)
        feats = {
            "ret": pd.Series(rng.normal(0, 0.01, 100)),
            "vol": pd.Series(np.full(100, 0.01)),
        }
        features = extract_regime_features(feats)
        for key in ["vol_level", "drift_strength", "drift_persistence",
                     "return_autocorr", "tail_indicator"]:
            self.assertIn(key, features)

    def test_bounded_values(self):
        from models.regime import extract_regime_features
        feats = {
            "ret": pd.Series(np.random.default_rng(42).normal(0, 0.01, 100)),
            "vol": pd.Series(np.full(100, 0.01)),
        }
        features = extract_regime_features(feats)
        self.assertGreaterEqual(features["vol_level"], 0.1)
        self.assertLessEqual(features["vol_level"], 10.0)
        self.assertGreaterEqual(features["tail_indicator"], 0.0)
        self.assertLessEqual(features["tail_indicator"], 10.0)


class TestRegimeProbabilities(unittest.TestCase):
    """Test softmax-based regime probabilities."""

    def test_probabilities_sum_to_one(self):
        from models.regime import compute_regime_probabilities
        features = {
            "vol_level": 0.8, "drift_strength": 0.002,
            "drift_persistence": 0.95, "return_autocorr": 0.05,
            "tail_indicator": 0.5,
        }
        probs = compute_regime_probabilities(features)
        self.assertEqual(len(probs), 5)
        self.assertAlmostEqual(probs.sum(), 1.0, places=10)

    def test_v2_probabilities_sum_to_one(self):
        from models.regime import compute_regime_probabilities_v2
        probs = compute_regime_probabilities_v2(1.0, 0.001, 0.5)
        self.assertEqual(len(probs), 5)
        self.assertAlmostEqual(probs.sum(), 1.0, places=10)

    def test_v2_crisis_high_vol(self):
        from models.regime import compute_regime_probabilities_v2, REGIME_CRISIS_JUMP
        probs = compute_regime_probabilities_v2(3.0, 0.001, 5.0)
        self.assertGreater(probs[REGIME_CRISIS_JUMP], 0.3)


class TestSoftBmaWeights(unittest.TestCase):
    """Test soft BMA weight mixing."""

    def test_uniform_regimes_equal_global(self):
        from models.regime import compute_soft_bma_weights
        regime_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        global_posterior = {"model_a": 0.6, "model_b": 0.4}
        weights = compute_soft_bma_weights(regime_probs, {}, global_posterior)
        self.assertAlmostEqual(weights["model_a"], 0.6, places=5)
        self.assertAlmostEqual(weights["model_b"], 0.4, places=5)

    def test_weights_sum_to_one(self):
        from models.regime import compute_soft_bma_weights
        regime_probs = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        global_posterior = {"m1": 0.5, "m2": 0.3, "m3": 0.2}
        weights = compute_soft_bma_weights(regime_probs, {}, global_posterior)
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_regime_specific_posteriors(self):
        from models.regime import compute_soft_bma_weights
        regime_probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        regime_data = {
            "0": {
                "model_posterior": {"m1": 0.8, "m2": 0.2},
                "regime_meta": {},
            },
        }
        global_posterior = {"m1": 0.5, "m2": 0.5}
        weights = compute_soft_bma_weights(regime_probs, regime_data, global_posterior)
        self.assertAlmostEqual(weights["m1"], 0.8, places=5)
        self.assertAlmostEqual(weights["m2"], 0.2, places=5)


class TestTuneSignalsConsistency(unittest.TestCase):
    """Verify tune.py and signals.py use identical regime code."""

    def test_same_market_regime_object(self):
        from tuning.tune import MarketRegime as TuneRegime
        from models.regime import MarketRegime as RegimeRegime
        self.assertIs(TuneRegime, RegimeRegime)

    def test_same_assign_labels_function(self):
        from tuning.tune import assign_regime_labels as tune_fn
        from models.regime import assign_regime_labels as regime_fn
        self.assertIs(tune_fn, regime_fn)

    def test_same_current_regime_function(self):
        from decision.signals import assign_current_regime as sig_fn
        from models.regime import assign_current_regime as regime_fn
        self.assertIs(sig_fn, regime_fn)

    def test_same_v2_probs_function(self):
        from decision.signals import compute_regime_probabilities_v2 as sig_fn
        from models.regime import compute_regime_probabilities_v2 as regime_fn
        self.assertIs(sig_fn, regime_fn)


if __name__ == "__main__":
    unittest.main()
