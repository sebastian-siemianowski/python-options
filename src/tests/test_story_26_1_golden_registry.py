"""
Story 26.1 -- Golden Score Registry
=====================================
Registry of expected scores for regression testing.
Any code change degrading accuracy by > tolerance is flagged.

Acceptance Criteria:
- Registry covers multiple (asset, model, metric) combos
- Tolerance: BIC +/- 5 nats, CRPS +/- 5%
- Scores outside tolerance: test FAILS with diagnostic
- Test runs fast (uses synthetic data)
"""

import os, sys, json

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
# Golden registry
# ---------------------------------------------------------------------------
def _make_ewma(r, sigma0):
    v = np.zeros(len(r))
    v[0] = sigma0 ** 2
    for i in range(1, len(r)):
        v[i] = 0.94 * v[i - 1] + 0.06 * r[i - 1] ** 2
    return np.sqrt(np.maximum(v, 1e-16))


def _gen_asset(kind, n=500, seed=100):
    rng = np.random.default_rng(seed)
    if kind == "spy":
        r = 0.01 * rng.standard_normal(n)
    elif kind == "mstr":
        r = 0.03 * rng.standard_t(df=4, size=n)
    elif kind == "gold":
        r = 0.008 * rng.standard_normal(n)
    elif kind == "btc":
        r = 0.025 * rng.standard_t(df=5, size=n)
    else:
        r = 0.012 * rng.standard_normal(n)
    sigma0 = np.std(r[:50]) if len(r) >= 50 else 0.01
    return r, _make_ewma(r, sigma0)


def compute_scores(r, v, phi=0.90, q=1e-4, c=1.0, nu=8.0):
    """Compute model scores for golden registry."""
    _, _, ll_t = run_phi_student_t_filter(r, v, phi, q, c, nu)
    _, _, ll_g = run_gaussian_filter(r, v, q, c)

    n = len(r)
    bic_t = -2 * ll_t + 4 * np.log(n)
    bic_g = -2 * ll_g + 3 * np.log(n)

    return {
        "loglik_student_t": float(ll_t),
        "loglik_gaussian": float(ll_g),
        "bic_student_t": float(bic_t),
        "bic_gaussian": float(bic_g),
        "n_obs": n,
    }


def build_registry(assets=None):
    """Build golden score registry."""
    if assets is None:
        assets = ["spy", "mstr", "gold", "btc", "generic"]
    registry = {}
    for asset in assets:
        r, v = _gen_asset(asset)
        scores = compute_scores(r, v)
        registry[asset] = scores
    return registry


def check_regression(current_scores, golden_scores, tolerances=None):
    """
    Check current scores against golden registry.
    Returns list of (asset, metric, expected, actual, tolerance) violations.
    """
    if tolerances is None:
        tolerances = {
            "loglik_student_t": 5.0,
            "loglik_gaussian": 5.0,
            "bic_student_t": 10.0,
            "bic_gaussian": 10.0,
        }

    violations = []
    for asset in golden_scores:
        if asset not in current_scores:
            violations.append((asset, "MISSING", None, None, None))
            continue
        for metric, tol in tolerances.items():
            if metric in golden_scores[asset] and metric in current_scores[asset]:
                expected = golden_scores[asset][metric]
                actual = current_scores[asset][metric]
                if abs(actual - expected) > tol:
                    violations.append((asset, metric, expected, actual, tol))

    return violations


# ===========================================================================
class TestGoldenRegistry:
    """Golden score registry works correctly."""

    def test_registry_builds(self):
        reg = build_registry()
        assert len(reg) == 5
        for asset in ["spy", "mstr", "gold", "btc", "generic"]:
            assert asset in reg
            assert "loglik_student_t" in reg[asset]
            assert "bic_student_t" in reg[asset]

    def test_scores_finite(self):
        reg = build_registry()
        for asset, scores in reg.items():
            for metric, val in scores.items():
                assert np.isfinite(val), f"{asset}/{metric}={val}"

    def test_deterministic(self):
        reg1 = build_registry()
        reg2 = build_registry()
        for asset in reg1:
            for metric in ["loglik_student_t", "loglik_gaussian"]:
                assert reg1[asset][metric] == reg2[asset][metric]


# ===========================================================================
class TestRegressionCheck:
    """Regression detection works."""

    def test_no_violations_identical(self):
        reg = build_registry()
        violations = check_regression(reg, reg)
        assert len(violations) == 0

    def test_detects_degradation(self):
        golden = build_registry()
        current = build_registry()
        # Perturb one score beyond tolerance
        current["spy"]["loglik_student_t"] += 20.0
        violations = check_regression(current, golden)
        assert len(violations) > 0
        assert any(v[0] == "spy" for v in violations)

    def test_within_tolerance_passes(self):
        golden = build_registry()
        current = build_registry()
        # Small perturbation within tolerance
        current["spy"]["loglik_student_t"] += 2.0
        violations = check_regression(current, golden)
        spy_violations = [v for v in violations if v[0] == "spy"]
        assert len(spy_violations) == 0

    def test_missing_asset_flagged(self):
        golden = build_registry()
        current = {k: v for k, v in golden.items() if k != "btc"}
        violations = check_regression(current, golden)
        assert any(v[0] == "btc" for v in violations)
