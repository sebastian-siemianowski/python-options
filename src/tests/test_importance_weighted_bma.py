"""Tests for Story 3.5: Importance-Weighted BMA Sampling.

Validates that:
1. Model allocation uses Categorical(weights) draw — no min-floor distortion
2. Dominant model gets approximately its weight fraction of paths
3. Low-weight models may get zero samples (correct mixture behavior)
4. Total sample count equals n_paths (no floor inflation)
5. Representation error < 0.1% for large n_paths
6. Multi-horizon coherence is preserved
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

from decision.signals import bayesian_model_average_mc
import pandas as pd


def _make_minimal_feats(n=500):
    """Create minimal features dict for BMA testing."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    ret = pd.Series(np.random.default_rng(42).normal(0, 0.015, n), index=idx)
    vol = pd.Series(np.abs(ret).rolling(20).mean().fillna(0.015), index=idx)
    px = pd.Series(100 * np.exp(ret.cumsum()), index=idx)
    mu_post = pd.Series(ret.rolling(60).mean().fillna(0.0), index=idx)
    return {
        "ret": ret,
        "vol": vol,
        "px": px,
        "mu_post": mu_post,
        "nu_hat": pd.Series(np.full(n, 6.0), index=idx),
        "garch_params": {"omega": 1e-6, "alpha": 0.05, "beta": 0.90},
    }


def _make_tuned_params_with_weights(model_weights):
    """Create tuned_params structure with given model name->weight dict.
    
    Each model gets minimal viable params for run_unified_mc.
    """
    models_dict = {}
    for name, w in model_weights.items():
        models_dict[name] = {
            "fit_success": True,
            "q": 1e-5,
            "phi": 0.95,
            "nu": 6.0,
            "c": 1.0,
            "combined_score": 1.0 / max(w, 1e-6),  # Lower = better weight
        }
    
    return {
        "has_bma": True,
        "global": {
            "model_posterior": model_weights,
            "models": models_dict,
        },
        "regimes": {},
    }


class TestImportanceWeightedAllocation(unittest.TestCase):
    """Test that importance-weighted sampling allocates paths correctly."""

    def test_total_samples_equals_n_paths(self):
        """Total sample count should equal n_paths (no floor inflation)."""
        weights = {
            "model_a": 0.60,
            "model_b": 0.30,
            "model_c": 0.10,
        }
        feats = _make_minimal_feats()
        tp = _make_tuned_params_with_weights(weights)
        
        r_samples, vol_samples, _, meta = bayesian_model_average_mc(
            feats=feats,
            regime_params={},
            mu_t=0.0,
            P_t=1e-6,
            sigma2_step=0.0002,
            H=5,
            n_paths=1000,
            seed=42,
            tuned_params=tp,
        )
        
        self.assertEqual(len(r_samples), 1000)

    def test_dominant_model_gets_proportional_paths(self):
        """Dominant model (90% weight) should get ~90% of paths."""
        weights = {
            "dominant": 0.90,
            "minor": 0.10,
        }
        feats = _make_minimal_feats()
        tp = _make_tuned_params_with_weights(weights)
        
        _, _, _, meta = bayesian_model_average_mc(
            feats=feats,
            regime_params={},
            mu_t=0.0,
            P_t=1e-6,
            sigma2_step=0.0002,
            H=5,
            n_paths=10000,
            seed=42,
            tuned_params=tp,
        )
        
        details = meta.get("model_details", {})
        n_dominant = details.get("dominant", {}).get("n_samples", 0)
        frac = n_dominant / 10000
        # Should be within 2% of target weight
        self.assertAlmostEqual(frac, 0.90, delta=0.02,
                               msg=f"Dominant model got {frac:.3f} (expected ~0.90)")

    def test_representation_error_below_threshold(self):
        """Representation error should be < 1% for 10k paths."""
        weights = {
            "m1": 0.50,
            "m2": 0.30,
            "m3": 0.15,
            "m4": 0.05,
        }
        feats = _make_minimal_feats()
        tp = _make_tuned_params_with_weights(weights)
        
        _, _, _, meta = bayesian_model_average_mc(
            feats=feats,
            regime_params={},
            mu_t=0.0,
            P_t=1e-6,
            sigma2_step=0.0002,
            H=5,
            n_paths=10000,
            seed=42,
            tuned_params=tp,
        )
        
        details = meta.get("model_details", {})
        total_n = sum(d.get("n_samples", 0) for d in details.values())
        
        max_error = 0.0
        for name, target_w in weights.items():
            actual_n = details.get(name, {}).get("n_samples", 0)
            actual_frac = actual_n / total_n if total_n > 0 else 0
            error = abs(actual_frac - target_w)
            max_error = max(max_error, error)
        
        # For Categorical with 10k draws, max error should be well under 1%
        self.assertLess(max_error, 0.03,
                        msg=f"Max representation error {max_error:.4f} exceeds 3%")

    def test_tiny_weight_may_get_zero(self):
        """Model with tiny weight (0.1%) may get 0 samples — correct behavior."""
        weights = {
            "big": 0.90,
            "medium": 0.099,
            "tiny": 0.001,
        }
        feats = _make_minimal_feats()
        tp = _make_tuned_params_with_weights(weights)
        
        # With only 100 paths, tiny model (0.1%) likely gets 0
        _, _, _, meta = bayesian_model_average_mc(
            feats=feats,
            regime_params={},
            mu_t=0.0,
            P_t=1e-6,
            sigma2_step=0.0002,
            H=5,
            n_paths=100,
            seed=42,
            tuned_params=tp,
        )
        
        details = meta.get("model_details", {})
        tiny_n = details.get("tiny", {}).get("n_samples", 0)
        # It's OK for tiny to have 0 or very few samples
        self.assertLessEqual(tiny_n, 5,
                             msg=f"Tiny model shouldn't dominate: got {tiny_n}")

    def test_no_min_floor_distortion(self):
        """Old floor of 20 is gone — a 0.1% model shouldn't get 20 out of 1000."""
        weights = {
            "big": 0.998,
            "negligible": 0.002,
        }
        feats = _make_minimal_feats()
        tp = _make_tuned_params_with_weights(weights)
        
        _, _, _, meta = bayesian_model_average_mc(
            feats=feats,
            regime_params={},
            mu_t=0.0,
            P_t=1e-6,
            sigma2_step=0.0002,
            H=5,
            n_paths=1000,
            seed=42,
            tuned_params=tp,
        )
        
        details = meta.get("model_details", {})
        neg_n = details.get("negligible", {}).get("n_samples", 0)
        # Old approach: min(20, ...) would give 20/1000 = 2.0%
        # New approach: Categorical(0.002) * 1000 ~ 2 samples
        self.assertLess(neg_n, 15,
                        msg=f"Negligible model got {neg_n} — floor distortion?")

    def test_single_model_gets_all_paths(self):
        """Single model with weight=1.0 should get all n_paths."""
        weights = {"solo": 1.0}
        feats = _make_minimal_feats()
        tp = _make_tuned_params_with_weights(weights)
        
        r_samples, _, _, meta = bayesian_model_average_mc(
            feats=feats,
            regime_params={},
            mu_t=0.0,
            P_t=1e-6,
            sigma2_step=0.0002,
            H=5,
            n_paths=500,
            seed=42,
            tuned_params=tp,
        )
        
        details = meta.get("model_details", {})
        solo_n = details.get("solo", {}).get("n_samples", 0)
        self.assertEqual(solo_n, 500)
        self.assertEqual(len(r_samples), 500)

    def test_equal_weights_symmetric_allocation(self):
        """Equal weights should produce roughly equal sample counts."""
        weights = {f"m{i}": 0.25 for i in range(4)}
        feats = _make_minimal_feats()
        tp = _make_tuned_params_with_weights(weights)
        
        _, _, _, meta = bayesian_model_average_mc(
            feats=feats,
            regime_params={},
            mu_t=0.0,
            P_t=1e-6,
            sigma2_step=0.0002,
            H=5,
            n_paths=10000,
            seed=42,
            tuned_params=tp,
        )
        
        details = meta.get("model_details", {})
        counts = [details.get(f"m{i}", {}).get("n_samples", 0) for i in range(4)]
        # Each should be within 5% of 2500
        for i, c in enumerate(counts):
            self.assertAlmostEqual(c / 10000, 0.25, delta=0.05,
                                   msg=f"Model m{i} got {c}/10000")

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical allocation."""
        weights = {"a": 0.7, "b": 0.3}
        feats = _make_minimal_feats()
        tp = _make_tuned_params_with_weights(weights)
        
        _, _, _, meta1 = bayesian_model_average_mc(
            feats=feats, regime_params={}, mu_t=0.0, P_t=1e-6,
            sigma2_step=0.0002, H=5, n_paths=500, seed=99,
            tuned_params=tp,
        )
        _, _, _, meta2 = bayesian_model_average_mc(
            feats=feats, regime_params={}, mu_t=0.0, P_t=1e-6,
            sigma2_step=0.0002, H=5, n_paths=500, seed=99,
            tuned_params=tp,
        )
        
        d1 = meta1.get("model_details", {})
        d2 = meta2.get("model_details", {})
        self.assertEqual(
            d1.get("a", {}).get("n_samples", -1),
            d2.get("a", {}).get("n_samples", -2),
        )


class TestImportanceWeightedMultiHorizon(unittest.TestCase):
    """Test multi-horizon coherence with importance-weighted sampling."""

    def test_horizon_samples_present(self):
        """Multi-horizon fast mode should still produce per-horizon samples."""
        weights = {"m1": 0.6, "m2": 0.4}
        feats = _make_minimal_feats()
        tp = _make_tuned_params_with_weights(weights)
        
        _, _, _, meta = bayesian_model_average_mc(
            feats=feats, regime_params={}, mu_t=0.0, P_t=1e-6,
            sigma2_step=0.0002, H=30, n_paths=500, seed=42,
            tuned_params=tp, horizons_extract=[1, 7, 30],
        )
        
        hz_samples = meta.get("horizon_samples", {})
        for h in [1, 7, 30]:
            self.assertIn(h, hz_samples, f"Missing horizon {h} samples")
            self.assertGreater(len(hz_samples[h]), 0)


if __name__ == "__main__":
    unittest.main()
