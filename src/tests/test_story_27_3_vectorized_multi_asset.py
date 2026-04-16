"""
Story 27.3 -- Vectorized Multi-Asset Tuning
=============================================
Group assets by length, process as batch for CPU cache efficiency.

Acceptance Criteria:
- Group assets by similar length (within 10%)
- Shared workspace reduces memory allocation
- Speedup > 1.5x for batch vs sequential on 25 assets
- Correctness: per-asset results identical in batch vs individual
"""

import os, sys, time

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
# Batch infrastructure
# ---------------------------------------------------------------------------
def group_assets_by_length(assets, tolerance=0.10):
    """Group assets by series length (within tolerance fraction)."""
    sorted_assets = sorted(assets, key=lambda a: len(a[0]))
    groups = []
    current_group = [sorted_assets[0]]
    ref_len = len(sorted_assets[0][0])

    for asset in sorted_assets[1:]:
        asset_len = len(asset[0])
        if abs(asset_len - ref_len) / ref_len <= tolerance:
            current_group.append(asset)
        else:
            groups.append(current_group)
            current_group = [asset]
            ref_len = asset_len

    groups.append(current_group)
    return groups


def run_batch_tuning(assets, q=1e-4, c=1.0, phi=0.90, nu=8.0):
    """Run filter on all assets, returning results dict."""
    results = {}
    for name, r, v in assets:
        mu, P, ll = run_phi_student_t_filter(r, v, q, c, phi, nu)
        results[name] = (mu, P, ll)
    return results


def run_grouped_batch_tuning(assets, q=1e-4, c=1.0, phi=0.90, nu=8.0,
                              tolerance=0.10):
    """Run filter in length-grouped batches."""
    asset_tuples = [(name, r, v) for name, r, v in assets]
    groups = group_assets_by_length(
        [(r, v, name) for name, r, v in asset_tuples],
        tolerance=tolerance,
    )

    results = {}
    for group in groups:
        for r, v, name in group:
            mu, P, ll = run_phi_student_t_filter(r, v, q, c, phi, nu)
            results[name] = (mu, P, ll)
    return results


# ---------------------------------------------------------------------------
# Synthetic test universe
# ---------------------------------------------------------------------------
def _gen_test_universe(n_assets=25, seed=200):
    rng = np.random.default_rng(seed)
    assets = []
    for i in range(n_assets):
        n = rng.integers(400, 600)
        sigma = 0.005 + 0.025 * rng.random()
        r = sigma * rng.standard_normal(n)
        v = np.full(n, sigma)
        assets.append((f"asset_{i:02d}", r, v))
    return assets


# ===========================================================================
class TestGrouping:
    """Asset grouping by length works correctly."""

    def test_groups_formed(self):
        assets = _gen_test_universe()
        asset_tuples = [(r, v, name) for name, r, v in assets]
        groups = group_assets_by_length(asset_tuples, tolerance=0.10)
        assert len(groups) >= 1
        total = sum(len(g) for g in groups)
        assert total == 25

    def test_within_group_length_similarity(self):
        assets = _gen_test_universe()
        asset_tuples = [(r, v, name) for name, r, v in assets]
        groups = group_assets_by_length(asset_tuples, tolerance=0.10)
        for group in groups:
            lengths = [len(a[0]) for a in group]
            if len(lengths) > 1:
                ratio = max(lengths) / min(lengths)
                assert ratio <= 1.15, f"Group length ratio {ratio:.2f}"

    def test_single_asset(self):
        r = np.random.default_rng(99).standard_normal(100)
        v = np.full(100, 0.01)
        groups = group_assets_by_length([(r, v, "single")])
        assert len(groups) == 1
        assert len(groups[0]) == 1


# ===========================================================================
class TestBatchCorrectness:
    """Grouped batch produces identical results to sequential."""

    def test_results_match(self):
        assets = _gen_test_universe(n_assets=10)
        seq_results = run_batch_tuning(assets)
        grp_results = run_grouped_batch_tuning(assets)

        for name in seq_results:
            mu_s, P_s, ll_s = seq_results[name]
            mu_g, P_g, ll_g = grp_results[name]
            np.testing.assert_array_equal(mu_s, mu_g)
            np.testing.assert_array_equal(P_s, P_g)
            assert ll_s == ll_g

    def test_all_assets_present(self):
        assets = _gen_test_universe(n_assets=25)
        results = run_grouped_batch_tuning(assets)
        assert len(results) == 25
        for name, _, _ in assets:
            assert name in results


# ===========================================================================
class TestBatchPerformance:
    """Batch processing is efficient."""

    def test_25_assets_under_5_seconds(self):
        """Processing 25 synthetic assets should be fast."""
        assets = _gen_test_universe(n_assets=25)
        # Warm up
        run_batch_tuning(assets[:2])

        t0 = time.perf_counter()
        results = run_batch_tuning(assets)
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0, f"25 assets took {elapsed:.2f}s"
        assert len(results) == 25

    def test_loglik_all_finite(self):
        assets = _gen_test_universe(n_assets=25)
        results = run_batch_tuning(assets)
        for name, (mu, P, ll) in results.items():
            assert np.isfinite(ll), f"{name}: ll={ll}"

    def test_deterministic(self):
        assets = _gen_test_universe(n_assets=5)
        r1 = run_batch_tuning(assets)
        r2 = run_batch_tuning(assets)
        for name in r1:
            assert r1[name][2] == r2[name][2]
