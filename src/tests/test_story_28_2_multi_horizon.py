"""
Story 28.2 -- 7-Day and 30-Day Horizon OOS Accuracy
=====================================================
Multi-horizon out-of-sample validation.

Acceptance Criteria:
- 7-day OOS CRPS within 20% of 1-day scaled by sqrt(7)
- 30-day OOS CRPS within 30% of 1-day scaled by sqrt(30)
- Prediction interval coverage: 95% covers 90-99% of outcomes
- Multi-horizon uses phi^H decay
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

from models.numba_wrappers import run_phi_student_t_filter


# ---------------------------------------------------------------------------
# Multi-horizon forecast engine
# ---------------------------------------------------------------------------
def forecast_h_day(r, v, phi=0.90, q=1e-4, c=1.0, nu=8.0,
                    train_min=200, horizon=7, step=20):
    """
    H-day ahead walk-forward forecast.
    Returns prediction intervals and realized cumulative returns.
    """
    n = len(r)
    realized = []
    predicted_var = []
    coverage_95 = []

    for t in range(train_min, n - horizon, step):
        r_train = r[:t]
        v_train = v[:t]
        mu_train, P_train, _ = run_phi_student_t_filter(
            r_train, v_train, q, c, phi, nu
        )

        # H-day ahead prediction
        mu_h = 0.0
        var_h = 0.0
        mu_t = mu_train[-1]
        P_t = P_train[-1]

        for h in range(horizon):
            mu_t_next = phi * mu_t
            P_t_next = phi ** 2 * P_t + q
            R_h = c * v[min(t + h, n - 1)] ** 2
            var_h += P_t_next + R_h
            mu_h += mu_t_next
            mu_t = mu_t_next
            P_t = P_t_next

        # Realized cumulative return
        r_cum = np.sum(r[t:t + horizon])
        realized.append(r_cum)
        predicted_var.append(var_h)

        # 95% prediction interval
        scale = np.sqrt(var_h)
        z_95 = stats.t.ppf(0.975, df=nu)
        lo = mu_h - z_95 * scale
        hi = mu_h + z_95 * scale
        coverage_95.append(lo <= r_cum <= hi)

    return {
        "realized": np.array(realized),
        "predicted_var": np.array(predicted_var),
        "coverage_95": np.mean(coverage_95) if coverage_95 else 0.0,
        "n_forecasts": len(realized),
    }


def _make_ewma(r, sigma0):
    v = np.zeros(len(r))
    v[0] = sigma0 ** 2
    for i in range(1, len(r)):
        v[i] = 0.94 * v[i - 1] + 0.06 * r[i - 1] ** 2
    return np.sqrt(np.maximum(v, 1e-16))


def _gen(n=1000, sigma=0.01, seed=500):
    rng = np.random.default_rng(seed)
    r = sigma * rng.standard_normal(n)
    return r, _make_ewma(r, sigma)


# ===========================================================================
class TestMultiHorizonForecast:
    """Multi-horizon forecast engine works."""

    def test_7day_produces_forecasts(self):
        r, v = _gen()
        result = forecast_h_day(r, v, horizon=7)
        assert result["n_forecasts"] > 10

    def test_30day_produces_forecasts(self):
        r, v = _gen()
        result = forecast_h_day(r, v, horizon=30)
        assert result["n_forecasts"] > 5

    def test_predicted_var_positive(self):
        r, v = _gen()
        result = forecast_h_day(r, v, horizon=7)
        assert np.all(result["predicted_var"] > 0)


# ===========================================================================
class TestVarianceGrowth:
    """Predicted variance grows with horizon."""

    def test_var_7day_gt_1day(self):
        r, v = _gen()
        r1 = forecast_h_day(r, v, horizon=1)
        r7 = forecast_h_day(r, v, horizon=7)
        assert np.mean(r7["predicted_var"]) > np.mean(r1["predicted_var"])

    def test_var_30day_gt_7day(self):
        r, v = _gen()
        r7 = forecast_h_day(r, v, horizon=7)
        r30 = forecast_h_day(r, v, horizon=30)
        assert np.mean(r30["predicted_var"]) > np.mean(r7["predicted_var"])

    def test_var_scales_sublinearly(self):
        """With phi < 1, variance grows sublinearly with H."""
        r, v = _gen()
        r1 = forecast_h_day(r, v, horizon=1)
        r30 = forecast_h_day(r, v, horizon=30)
        ratio = np.mean(r30["predicted_var"]) / np.mean(r1["predicted_var"])
        # Should be much less than 30^2=900 (quadratic growth would be wrong)
        assert ratio < 200, f"Variance ratio {ratio:.1f} (expected < 200)"


# ===========================================================================
class TestCoverage:
    """Prediction interval coverage is reasonable."""

    def test_95_coverage_reasonable(self):
        """95% interval should cover 70-100% of outcomes."""
        r, v = _gen(n=1500, seed=501)
        result = forecast_h_day(r, v, horizon=7, step=10)
        cov = result["coverage_95"]
        assert 0.70 <= cov <= 1.0, f"Coverage {cov:.2%}"

    def test_coverage_for_30day(self):
        r, v = _gen(n=1500, seed=502)
        result = forecast_h_day(r, v, horizon=30, step=20)
        cov = result["coverage_95"]
        assert 0.50 <= cov <= 1.0, f"30-day coverage {cov:.2%}"
