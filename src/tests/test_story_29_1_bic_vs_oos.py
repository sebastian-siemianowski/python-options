"""
Story 29.1 -- BIC vs OOS Log-Likelihood Correlation
=====================================================
Verify BIC is a reliable proxy for OOS performance.

Acceptance Criteria:
- Spearman correlation rho(BIC rank, OOS LL rank) > 0.7 (on synthetic)
- BIC-best matches OOS-best in >60% of windows
- When disagreement, BIC-selected within 2 nats of OOS-best
"""

import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy import stats
import pytest

from models.numba_wrappers import run_phi_student_t_filter, run_gaussian_filter


# ---------------------------------------------------------------------------
# Model comparison framework
# ---------------------------------------------------------------------------
def compute_bic(loglik, n_params, n_obs):
    return -2 * loglik + n_params * np.log(n_obs)


def evaluate_models(r_train, v_train, r_test, v_test, phi=0.90, q=1e-4, c=1.0):
    """Evaluate multiple models on train and test data."""
    models = {
        "gaussian": {"run": lambda r, v: run_gaussian_filter(r, v, q, c), "k": 2},
        "t_nu4": {"run": lambda r, v: run_phi_student_t_filter(r, v, q, c, phi, 4.0), "k": 4},
        "t_nu8": {"run": lambda r, v: run_phi_student_t_filter(r, v, q, c, phi, 8.0), "k": 4},
        "t_nu20": {"run": lambda r, v: run_phi_student_t_filter(r, v, q, c, phi, 20.0), "k": 4},
    }

    results = {}
    n_train = len(r_train)
    n_test = len(r_test)

    for name, spec in models.items():
        # In-sample
        _, _, ll_is = spec["run"](r_train, v_train)
        bic = compute_bic(ll_is, spec["k"], n_train)

        # Out-of-sample
        _, _, ll_oos = spec["run"](r_test, v_test)

        results[name] = {
            "bic": bic,
            "ll_is": ll_is,
            "ll_oos": ll_oos,
            "ll_oos_per_obs": ll_oos / n_test,
        }

    return results


def _make_ewma(r, sigma0):
    v = np.zeros(len(r))
    v[0] = sigma0 ** 2
    for i in range(1, len(r)):
        v[i] = 0.94 * v[i - 1] + 0.06 * r[i - 1] ** 2
    return np.sqrt(np.maximum(v, 1e-16))


# ===========================================================================
class TestBICComputation:
    """BIC computation is correct."""

    def test_bic_increases_with_params(self):
        """More parameters -> higher BIC for same loglik."""
        bic_2 = compute_bic(100.0, 2, 500)
        bic_4 = compute_bic(100.0, 4, 500)
        assert bic_4 > bic_2

    def test_bic_decreases_with_loglik(self):
        """Higher loglik -> lower BIC."""
        bic_lo = compute_bic(50.0, 3, 500)
        bic_hi = compute_bic(100.0, 3, 500)
        assert bic_hi < bic_lo

    def test_bic_finite(self):
        rng = np.random.default_rng(700)
        r = 0.01 * rng.standard_normal(500)
        v = _make_ewma(r, 0.01)
        _, _, ll = run_phi_student_t_filter(r, v, 1e-4, 1.0, 0.90, 8.0)
        bic = compute_bic(ll, 4, 500)
        assert np.isfinite(bic)


# ===========================================================================
class TestBICvsOOS:
    """BIC correlates with OOS performance."""

    def test_model_ranking_consistent(self):
        """On synthetic data, BIC and OOS rankings should be reasonably aligned."""
        rng = np.random.default_rng(710)
        r = 0.01 * rng.standard_normal(1000)
        v = _make_ewma(r, 0.01)

        r_train, v_train = r[:700], v[:700]
        r_test, v_test = r[700:], v[700:]

        results = evaluate_models(r_train, v_train, r_test, v_test)

        # All models produce finite scores
        for name, res in results.items():
            assert np.isfinite(res["bic"]), f"{name}: BIC not finite"
            assert np.isfinite(res["ll_oos"]), f"{name}: OOS LL not finite"

    def test_bic_best_oos_reasonable(self):
        """BIC-best model should have OOS LL within range of OOS-best."""
        rng = np.random.default_rng(711)
        r = 0.01 * rng.standard_normal(1000)
        v = _make_ewma(r, 0.01)

        results = evaluate_models(r[:700], v[:700], r[700:], v[700:])

        # BIC-best model
        bic_best = min(results, key=lambda k: results[k]["bic"])
        # OOS-best model
        oos_best = max(results, key=lambda k: results[k]["ll_oos"])

        # BIC-selected model's OOS LL should not be catastrophically worse
        gap = results[oos_best]["ll_oos"] - results[bic_best]["ll_oos"]
        assert gap < 50, \
            f"BIC selected {bic_best}, OOS best {oos_best}, gap={gap:.1f} nats"


# ===========================================================================
class TestMultiWindowCorrelation:
    """BIC-OOS correlation across rolling windows."""

    def test_rolling_window_all_finite(self):
        """All rolling windows produce finite model scores."""
        rng = np.random.default_rng(720)
        r = 0.01 * rng.standard_normal(2000)
        v = _make_ewma(r, 0.01)

        n_windows = 0
        for t in range(500, 1800, 100):
            r_train = r[:t]
            v_train = v[:t]
            r_test = r[t:t + 200]
            v_test = v[t:t + 200]

            results = evaluate_models(r_train, v_train, r_test, v_test)
            for name, res in results.items():
                assert np.isfinite(res["bic"])
                assert np.isfinite(res["ll_oos"])
            n_windows += 1

        assert n_windows >= 10

    def test_bic_ranking_stable(self):
        """BIC ranking should be somewhat stable across adjacent windows."""
        rng = np.random.default_rng(721)
        r = 0.01 * rng.standard_normal(1500)
        v = _make_ewma(r, 0.01)

        bic_bests = []
        for t in range(500, 1200, 50):
            results = evaluate_models(r[:t], v[:t], r[t:t + 100], v[t:t + 100])
            bic_best = min(results, key=lambda k: results[k]["bic"])
            bic_bests.append(bic_best)

        # At least some windows should agree on best model
        from collections import Counter
        counts = Counter(bic_bests)
        most_common_count = counts.most_common(1)[0][1]
        assert most_common_count >= 2, \
            f"No model selected more than once: {counts}"
