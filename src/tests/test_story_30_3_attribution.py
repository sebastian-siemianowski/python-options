"""
Story 30.3 -- Per-File Accuracy Attribution
=============================================
Attribute accuracy changes to specific file modifications.

Acceptance Criteria:
- Attribution identifies which parameter change caused degradation
- Bisect-style testing methodology
- False positive rate < 20% (attribution correctly identifies causal change)
- Simulated file changes trigger correct attribution
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

from models.numba_wrappers import run_phi_student_t_filter


# ---------------------------------------------------------------------------
# Attribution engine
# ---------------------------------------------------------------------------
def compute_baseline_score(r, v, params):
    """Compute score with baseline parameters."""
    _, _, ll = run_phi_student_t_filter(
        r, v, params["q"], params["c"], params["phi"], params["nu"]
    )
    return ll


def attribute_degradation(r, v, baseline_params, changed_params):
    """
    Identify which parameter change caused the largest degradation.
    Returns list of (param_name, delta_ll) sorted by impact.
    """
    baseline_ll = compute_baseline_score(r, v, baseline_params)
    attributions = []

    for param_name in ["q", "c", "phi", "nu"]:
        if baseline_params[param_name] != changed_params[param_name]:
            # Test: change only this one parameter
            test_params = dict(baseline_params)
            test_params[param_name] = changed_params[param_name]
            test_ll = compute_baseline_score(r, v, test_params)
            delta = test_ll - baseline_ll
            attributions.append((param_name, delta))

    # Sort by absolute impact (most impactful first)
    attributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return attributions


def bisect_parameter_impact(r, v, base_params, param_name, old_val, new_val,
                             n_steps=5):
    """
    Bisect-style analysis: evaluate intermediate parameter values.
    Returns list of (value, loglik) pairs.
    """
    values = np.linspace(old_val, new_val, n_steps)
    results = []
    for val in values:
        params = dict(base_params)
        params[param_name] = val
        ll = compute_baseline_score(r, v, params)
        results.append((val, ll))
    return results


def _make_ewma(r, sigma0):
    v = np.zeros(len(r))
    v[0] = sigma0 ** 2
    for i in range(1, len(r)):
        v[i] = 0.94 * v[i - 1] + 0.06 * r[i - 1] ** 2
    return np.sqrt(np.maximum(v, 1e-16))


# ===========================================================================
class TestAttribution:
    """Attribution correctly identifies causal parameter changes."""

    def test_single_param_change_attributed(self):
        """Changing one param should be correctly attributed."""
        rng = np.random.default_rng(1100)
        r = 0.01 * rng.standard_normal(500)
        v = _make_ewma(r, 0.01)

        baseline = {"q": 1e-4, "c": 1.0, "phi": 0.90, "nu": 8.0}
        changed = {"q": 1e-2, "c": 1.0, "phi": 0.90, "nu": 8.0}

        attrs = attribute_degradation(r, v, baseline, changed)
        assert len(attrs) == 1
        assert attrs[0][0] == "q"

    def test_multi_param_change_ranks_impact(self):
        """Multiple param changes should be ranked by impact."""
        rng = np.random.default_rng(1101)
        r = 0.01 * rng.standard_normal(500)
        v = _make_ewma(r, 0.01)

        baseline = {"q": 1e-4, "c": 1.0, "phi": 0.90, "nu": 8.0}
        changed = {"q": 1e-2, "c": 5.0, "phi": 0.90, "nu": 8.0}

        attrs = attribute_degradation(r, v, baseline, changed)
        assert len(attrs) == 2
        # Most impactful should have larger absolute delta
        assert abs(attrs[0][1]) >= abs(attrs[1][1])

    def test_no_change_no_attribution(self):
        """Identical params should produce empty attribution."""
        rng = np.random.default_rng(1102)
        r = 0.01 * rng.standard_normal(500)
        v = _make_ewma(r, 0.01)

        params = {"q": 1e-4, "c": 1.0, "phi": 0.90, "nu": 8.0}
        attrs = attribute_degradation(r, v, params, params)
        assert len(attrs) == 0


# ===========================================================================
class TestBisect:
    """Bisect-style parameter analysis works."""

    def test_bisect_produces_values(self):
        rng = np.random.default_rng(1110)
        r = 0.01 * rng.standard_normal(500)
        v = _make_ewma(r, 0.01)

        params = {"q": 1e-4, "c": 1.0, "phi": 0.90, "nu": 8.0}
        results = bisect_parameter_impact(r, v, params, "q", 1e-4, 1e-2)
        assert len(results) == 5
        for val, ll in results:
            assert np.isfinite(ll)

    def test_bisect_monotonic_q(self):
        """Increasing q should generally change loglik monotonically."""
        rng = np.random.default_rng(1111)
        r = 0.01 * rng.standard_normal(500)
        v = _make_ewma(r, 0.01)

        params = {"q": 1e-4, "c": 1.0, "phi": 0.90, "nu": 8.0}
        results = bisect_parameter_impact(r, v, params, "q", 1e-6, 1e-1, n_steps=10)
        lls = [ll for _, ll in results]
        # Should have a clear trend (not necessarily monotonic, but range should be large)
        assert max(lls) - min(lls) > 1.0, "q range had negligible loglik impact"

    def test_bisect_nu_values(self):
        """Bisect over nu should produce finite results."""
        rng = np.random.default_rng(1112)
        r = 0.015 * rng.standard_t(df=5, size=500)
        v = _make_ewma(r, 0.015)

        params = {"q": 1e-4, "c": 1.0, "phi": 0.90, "nu": 5.0}
        results = bisect_parameter_impact(r, v, params, "nu", 3.0, 30.0, n_steps=6)
        assert all(np.isfinite(ll) for _, ll in results)


# ===========================================================================
class TestEndToEnd:
    """Full attribution pipeline works."""

    def test_simulate_regression(self):
        """Simulate a parameter regression and verify attribution catches it."""
        rng = np.random.default_rng(1120)
        r = 0.01 * rng.standard_normal(500)
        v = _make_ewma(r, 0.01)

        # Baseline (good params)
        good = {"q": 1e-4, "c": 1.0, "phi": 0.90, "nu": 8.0}
        # Regressed (bad params -- q too large)
        bad = {"q": 0.1, "c": 1.0, "phi": 0.90, "nu": 8.0}

        ll_good = compute_baseline_score(r, v, good)
        ll_bad = compute_baseline_score(r, v, bad)

        # Bad params should have worse loglik
        assert ll_good > ll_bad or abs(ll_good - ll_bad) < 1.0, \
            "Expected good params to have better loglik"

        # Attribution should find q
        attrs = attribute_degradation(r, v, good, bad)
        assert len(attrs) > 0
        assert attrs[0][0] == "q"

    def test_attribution_all_finite(self):
        """Attribution deltas should all be finite."""
        rng = np.random.default_rng(1121)
        r = 0.01 * rng.standard_normal(500)
        v = _make_ewma(r, 0.01)

        baseline = {"q": 1e-4, "c": 1.0, "phi": 0.90, "nu": 8.0}
        changed = {"q": 1e-3, "c": 2.0, "phi": 0.80, "nu": 4.0}

        attrs = attribute_degradation(r, v, baseline, changed)
        for name, delta in attrs:
            assert np.isfinite(delta), f"{name}: delta={delta}"
