"""
Story 30.1 -- Accuracy Heatmap (Asset x Metric)
=================================================
Heatmap showing accuracy by asset and metric with color-coded status.

Acceptance Criteria:
- Heatmap: rows (assets) x columns (metrics)
- Color: green (elite), yellow (acceptable), red (failing)
- Thresholds defined per metric
- Generated without errors
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
# Scoring protocol
# ---------------------------------------------------------------------------
METRIC_THRESHOLDS = {
    "loglik_per_obs": {"green": -1.0, "yellow": -2.0},  # Higher = better
    "bic_per_obs": {"green": 3.0, "yellow": 5.0},       # Lower = better
    "pit_ks_stat": {"green": 0.05, "yellow": 0.10},     # Lower = better
    "filter_converged": {"green": 1.0, "yellow": 0.5},  # Boolean-like
}


def classify_score(metric, value):
    """Classify a score as 'green', 'yellow', or 'red'."""
    thresholds = METRIC_THRESHOLDS.get(metric)
    if thresholds is None:
        return "yellow"  # Unknown metric

    if metric in ("loglik_per_obs", "filter_converged"):
        # Higher is better
        if value >= thresholds["green"]:
            return "green"
        elif value >= thresholds["yellow"]:
            return "yellow"
        else:
            return "red"
    else:
        # Lower is better
        if value <= thresholds["green"]:
            return "green"
        elif value <= thresholds["yellow"]:
            return "yellow"
        else:
            return "red"


def compute_heatmap_data(assets, phi=0.90, q=1e-4, c=1.0, nu=8.0):
    """
    Compute heatmap data: dict of asset -> {metric -> (value, color)}.
    """
    heatmap = {}
    for name, r, v in assets:
        n = len(r)
        mu, P, ll = run_phi_student_t_filter(r, v, q, c, phi, nu)

        ll_per_obs = ll / n
        bic = (-2 * ll + 4 * np.log(n)) / n

        # PIT KS stat (simplified)
        from scipy import stats as sp_stats
        innovations = r - mu
        scale = np.sqrt(P + (c * v) ** 2)
        z = innovations / np.maximum(scale, 1e-16)
        pit = sp_stats.t.cdf(z, df=nu)
        ks_stat, _ = sp_stats.kstest(pit[50:], 'uniform')

        # Filter convergence: P should decrease
        P_ratio = P[-1] / P[0] if P[0] > 0 else 1.0
        converged = 1.0 if P_ratio < 1.0 else 0.0

        scores = {
            "loglik_per_obs": ll_per_obs,
            "bic_per_obs": bic,
            "pit_ks_stat": ks_stat,
            "filter_converged": converged,
        }

        heatmap[name] = {
            metric: (val, classify_score(metric, val))
            for metric, val in scores.items()
        }

    return heatmap


def render_text_heatmap(heatmap):
    """Render heatmap as text table."""
    metrics = list(METRIC_THRESHOLDS.keys())
    lines = []
    header = f"{'Asset':<15}" + "".join(f"{m:<18}" for m in metrics)
    lines.append(header)
    lines.append("-" * len(header))

    for asset in sorted(heatmap.keys()):
        row = f"{asset:<15}"
        for metric in metrics:
            val, color = heatmap[asset][metric]
            indicator = {"green": "[OK]", "yellow": "[--]", "red": "[!!]"}[color]
            row += f"{val:>8.4f} {indicator:<8}"
        lines.append(row)

    return "\n".join(lines)


def _make_ewma(r, sigma0):
    v = np.zeros(len(r))
    v[0] = sigma0 ** 2
    for i in range(1, len(r)):
        v[i] = 0.94 * v[i - 1] + 0.06 * r[i - 1] ** 2
    return np.sqrt(np.maximum(v, 1e-16))


def _gen_universe(n_assets=8, seed=1000):
    rng = np.random.default_rng(seed)
    assets = []
    names = ["SPY", "QQQ", "MSTR", "GLD", "BTC", "NFLX", "AAPL", "MSFT"]
    for i in range(n_assets):
        sigma = 0.005 + 0.025 * rng.random()
        r = sigma * rng.standard_normal(500)
        v = _make_ewma(r, sigma)
        assets.append((names[i], r, v))
    return assets


# ===========================================================================
class TestHeatmapComputation:
    """Heatmap data is computed correctly."""

    def test_heatmap_produces_data(self):
        assets = _gen_universe()
        heatmap = compute_heatmap_data(assets)
        assert len(heatmap) == 8

    def test_all_metrics_present(self):
        assets = _gen_universe()
        heatmap = compute_heatmap_data(assets)
        for asset_data in heatmap.values():
            for metric in METRIC_THRESHOLDS:
                assert metric in asset_data

    def test_all_values_finite(self):
        assets = _gen_universe()
        heatmap = compute_heatmap_data(assets)
        for asset, metrics in heatmap.items():
            for metric, (val, color) in metrics.items():
                assert np.isfinite(val), f"{asset}/{metric}={val}"

    def test_colors_valid(self):
        assets = _gen_universe()
        heatmap = compute_heatmap_data(assets)
        valid_colors = {"green", "yellow", "red"}
        for asset, metrics in heatmap.items():
            for metric, (val, color) in metrics.items():
                assert color in valid_colors, f"{asset}/{metric} color={color}"


# ===========================================================================
class TestClassification:
    """Score classification works correctly."""

    def test_green_loglik(self):
        assert classify_score("loglik_per_obs", 0.5) == "green"

    def test_red_loglik(self):
        assert classify_score("loglik_per_obs", -5.0) == "red"

    def test_green_bic(self):
        assert classify_score("bic_per_obs", 2.0) == "green"

    def test_red_bic(self):
        assert classify_score("bic_per_obs", 10.0) == "red"


# ===========================================================================
class TestRendering:
    """Heatmap renders without errors."""

    def test_text_render(self):
        assets = _gen_universe()
        heatmap = compute_heatmap_data(assets)
        text = render_text_heatmap(heatmap)
        assert len(text) > 100
        assert "SPY" in text
        assert "[OK]" in text or "[--]" in text or "[!!]" in text
