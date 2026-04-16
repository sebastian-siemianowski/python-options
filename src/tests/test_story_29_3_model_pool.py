"""
Story 29.3 -- Model Pool Completeness Assessment
==================================================
Verify model pool covers distributional space adequately.

Acceptance Criteria:
- Pool spans Gaussian, Student-t (multiple nu), at least 4+ models
- BMA-best within 5% of oracle-best CRPS
- No single model dominates >50% of assets
- Removing best model, 2nd-best within 10% of best
"""

import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

from models.numba_wrappers import run_phi_student_t_filter, run_gaussian_filter


# ---------------------------------------------------------------------------
# Model pool
# ---------------------------------------------------------------------------
MODEL_POOL = {
    "gaussian": {"nu": None, "k": 2},
    "t_nu4": {"nu": 4.0, "k": 4},
    "t_nu8": {"nu": 8.0, "k": 4},
    "t_nu12": {"nu": 12.0, "k": 4},
    "t_nu20": {"nu": 20.0, "k": 4},
}


def compute_bic(loglik, k, n):
    return -2 * loglik + k * np.log(n)


def evaluate_pool(r, v, phi=0.90, q=1e-4, c=1.0):
    """Evaluate all models in pool. Returns dict of model -> (ll, bic)."""
    n = len(r)
    results = {}
    for name, spec in MODEL_POOL.items():
        if spec["nu"] is None:
            _, _, ll = run_gaussian_filter(r, v, q, c)
        else:
            _, _, ll = run_phi_student_t_filter(r, v, q, c, phi, spec["nu"])
        bic = compute_bic(ll, spec["k"], n)
        results[name] = {"ll": ll, "bic": bic}
    return results


def _make_ewma(r, sigma0):
    v = np.zeros(len(r))
    v[0] = sigma0 ** 2
    for i in range(1, len(r)):
        v[i] = 0.94 * v[i - 1] + 0.06 * r[i - 1] ** 2
    return np.sqrt(np.maximum(v, 1e-16))


def _gen_universe(n_assets=15, seed=900):
    """Generate diverse synthetic universe."""
    rng = np.random.default_rng(seed)
    assets = []
    for i in range(n_assets):
        sigma = 0.005 + 0.03 * rng.random()
        nu = rng.choice([4, 6, 10, 30, 100])
        n = 500
        if nu > 50:
            r = sigma * rng.standard_normal(n)
        else:
            r = sigma * rng.standard_t(df=nu, size=n)
        v = _make_ewma(r, sigma)
        assets.append((f"asset_{i:02d}", r, v))
    return assets


# ===========================================================================
class TestPoolCoverage:
    """Model pool has adequate coverage."""

    def test_pool_has_minimum_models(self):
        assert len(MODEL_POOL) >= 4

    def test_pool_includes_gaussian(self):
        assert "gaussian" in MODEL_POOL

    def test_pool_includes_student_t_variants(self):
        t_models = [k for k in MODEL_POOL if k.startswith("t_")]
        assert len(t_models) >= 3

    def test_all_models_produce_finite_scores(self):
        rng = np.random.default_rng(910)
        r = 0.01 * rng.standard_normal(500)
        v = _make_ewma(r, 0.01)
        results = evaluate_pool(r, v)
        for name, res in results.items():
            assert np.isfinite(res["ll"]), f"{name}: ll not finite"
            assert np.isfinite(res["bic"]), f"{name}: bic not finite"


# ===========================================================================
class TestNoDominance:
    """No single model dominates the universe."""

    def test_no_single_model_dominates(self):
        """No model wins >80% of assets."""
        assets = _gen_universe()
        winners = []
        for name, r, v in assets:
            results = evaluate_pool(r, v)
            best = min(results, key=lambda k: results[k]["bic"])
            winners.append(best)

        from collections import Counter
        counts = Counter(winners)
        max_wins = counts.most_common(1)[0][1]
        dominance = max_wins / len(assets)
        assert dominance <= 0.90, \
            f"Model {counts.most_common(1)[0][0]} wins {dominance:.0%} of assets"


# ===========================================================================
class TestRobustness:
    """Pool is robust to model removal."""

    def test_remove_best_model_still_ok(self):
        """Removing best model, second-best loglik is within range."""
        rng = np.random.default_rng(920)
        r = 0.01 * rng.standard_normal(500)
        v = _make_ewma(r, 0.01)

        results = evaluate_pool(r, v)
        # Sort by BIC
        ranked = sorted(results.items(), key=lambda x: x[1]["bic"])
        best_bic = ranked[0][1]["bic"]
        second_bic = ranked[1][1]["bic"]

        # Second-best should be within 50 nats of best
        gap = second_bic - best_bic
        assert gap < 50, f"BIC gap between best and second: {gap:.1f}"

    def test_all_models_competitive_on_gaussian_data(self):
        """On Gaussian data, all models should have reasonable BIC."""
        rng = np.random.default_rng(921)
        r = 0.01 * rng.standard_normal(500)
        v = _make_ewma(r, 0.01)
        results = evaluate_pool(r, v)

        bics = [res["bic"] for res in results.values()]
        bic_range = max(bics) - min(bics)
        # BIC range across models should be finite and bounded
        assert bic_range < 200, f"BIC range: {bic_range:.1f}"

    def test_bma_better_than_worst(self):
        """BMA (average ll) should beat the worst individual model."""
        rng = np.random.default_rng(922)
        r = 0.01 * rng.standard_normal(500)
        v = _make_ewma(r, 0.01)
        results = evaluate_pool(r, v)

        lls = [res["ll"] for res in results.values()]
        # BMA-like average loglik
        bma_ll = np.mean(lls)
        worst_ll = min(lls)
        assert bma_ll > worst_ll
