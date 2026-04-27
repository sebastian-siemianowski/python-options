"""Shared numerical helpers for Student-t drift models.

The improved Student-t variants should differ in model structure, not in
duplicated calibration arithmetic.  Keep small distribution/statistic helpers
here so model classes stay focused on filtering and optimization.
"""

from __future__ import annotations

import math

import numpy as np


def clip_nu(nu: float, nu_min: float, nu_max: float) -> float:
    """Clamp Student-t degrees of freedom with finite-input guards."""
    if not np.isfinite(nu):
        return float(nu_min)
    lo = max(2.001, float(nu_min))
    hi = max(lo, float(nu_max))
    return float(np.clip(float(nu), lo, hi))


def variance_to_scale(variance: float, nu: float) -> float:
    """Convert Student-t predictive variance to scale."""
    variance = float(variance) if np.isfinite(variance) else 1e-20
    variance = max(variance, 1e-20)
    nu = float(nu) if np.isfinite(nu) else 8.0
    if nu > 2.0:
        return float(math.sqrt(max(variance * (nu - 2.0) / nu, 1e-20)))
    return float(math.sqrt(variance))


def variance_to_scale_vec(variance: np.ndarray, nu: float) -> np.ndarray:
    """Vectorized Student-t variance-to-scale conversion."""
    variance = np.asarray(variance, dtype=np.float64)
    variance_safe = np.maximum(np.where(np.isfinite(variance), variance, 1e-20), 1e-20)
    nu = float(nu) if np.isfinite(nu) else 8.0
    if nu > 2.0:
        scale = np.sqrt(variance_safe * (nu - 2.0) / nu)
    else:
        scale = np.sqrt(variance_safe)
    return np.maximum(scale, 1e-10)


def precompute_vov(vol: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling std of log-volatility with robust finite guards."""
    vol = np.asarray(vol, dtype=np.float64).ravel()
    n = len(vol)
    if n <= 0:
        return np.empty(0, dtype=np.float64)

    finite = vol[np.isfinite(vol) & (vol > 0)]
    fill = float(np.median(finite)) if finite.size else 1.0
    vol = np.where(np.isfinite(vol) & (vol > 0), vol, fill)
    vol = np.maximum(vol, max(fill * 1e-4, 1e-10))

    window = int(max(2, min(window, max(n, 2))))
    if n <= window:
        return np.zeros(n, dtype=np.float64)

    log_vol = np.log(vol)
    cs1 = np.concatenate(([0.0], np.cumsum(log_vol)))
    cs2 = np.concatenate(([0.0], np.cumsum(log_vol * log_vol)))
    inv_w = 1.0 / float(window)
    idx = np.arange(window, n)
    s1 = cs1[idx] - cs1[idx - window]
    s2 = cs2[idx] - cs2[idx - window]
    var_arr = np.maximum(s2 * inv_w - (s1 * inv_w) ** 2, 0.0)
    vov = np.empty(n, dtype=np.float64)
    vov[window:] = np.sqrt(var_arr)
    vov[:window] = vov[window] if window < n else 0.0
    return np.where(np.isfinite(vov), vov, 0.0)


def compute_cvm_statistic(pit_values: np.ndarray) -> float:
    """Cramer-von Mises W2 for Uniform(0,1) PIT values."""
    pit_values = np.asarray(pit_values, dtype=np.float64).ravel()
    pit_values = pit_values[np.isfinite(pit_values)]
    n = len(pit_values)
    if n < 2:
        return float("inf")
    u = np.sort(np.clip(pit_values, 1e-12, 1.0 - 1e-12))
    i_vals = np.arange(1, n + 1, dtype=np.float64)
    w2 = float(np.sum((u - (2.0 * i_vals - 1.0) / (2.0 * n)) ** 2) + 1.0 / (12.0 * n))
    return w2 if np.isfinite(w2) else float("inf")


def compute_ad_statistic(pit_values: np.ndarray) -> float:
    """Anderson-Darling A2 for Uniform(0,1) PIT values."""
    pit_values = np.asarray(pit_values, dtype=np.float64).ravel()
    pit_values = pit_values[np.isfinite(pit_values)]
    n = len(pit_values)
    if n < 2:
        return float("inf")
    u = np.sort(np.clip(pit_values, 1e-10, 1.0 - 1e-10))
    i_vals = np.arange(1, n + 1, dtype=np.float64)
    a2 = -float(n) - float(np.sum((2.0 * i_vals - 1.0) * (np.log(u) + np.log1p(-u[::-1])))) / float(n)
    return float(max(a2, 0.0)) if np.isfinite(a2) else float("inf")


def ewm_lagged_correction(returns: np.ndarray, mu_pred: np.ndarray, ewm_lambda: float) -> np.ndarray:
    """Causal EWM correction from lagged forecast innovations."""
    returns = np.asarray(returns, dtype=np.float64).ravel()
    mu_pred = np.asarray(mu_pred, dtype=np.float64).ravel()
    n = min(len(returns), len(mu_pred))
    corrections = np.zeros(n, dtype=np.float64)
    if n <= 2 or ewm_lambda < 0.01:
        return corrections

    lam = float(np.clip(ewm_lambda, 0.01, 0.999))
    alpha = 1.0 - lam
    innov_lagged = np.asarray(returns[:n - 1] - mu_pred[:n - 1], dtype=np.float64)
    innov_lagged = np.where(np.isfinite(innov_lagged), innov_lagged, 0.0)
    try:
        from scipy.signal import lfilter

        corrections[1:] = lfilter([alpha], [1.0, -lam], innov_lagged)
    except Exception:
        value = 0.0
        for idx in range(n - 1):
            value = lam * value + alpha * innov_lagged[idx]
            corrections[idx + 1] = value
    return corrections


_ASSET_PHI_CENTER = {
    "index": 0.80,
    "large_cap": 0.65,
    "small_cap": 0.20,
    "high_vol_equity": 0.05,
    "crypto": 0.45,
    "forex": 0.30,
    "metals_gold": 0.75,
    "metals_silver": 0.55,
    "metals_other": 0.45,
}

_ASSET_PHI_TAU = {
    "index": 0.80,
    "large_cap": 0.75,
    "small_cap": 0.60,
    "high_vol_equity": 0.50,
    "crypto": 0.75,
    "forex": 0.65,
    "metals_gold": 0.80,
    "metals_silver": 0.70,
    "metals_other": 0.70,
}

_ASSET_PHI_STRENGTH = {
    "index": 1.00,
    "large_cap": 0.85,
    "small_cap": 0.55,
    "high_vol_equity": 0.35,
    "crypto": 0.65,
    "forex": 0.50,
    "metals_gold": 0.85,
    "metals_silver": 0.70,
    "metals_other": 0.65,
}


def asset_phi_profile(asset_class: str | None) -> tuple[float, float, float]:
    """Return (center, tau, strength) for weak asset-aware phi regularization."""
    return (
        float(_ASSET_PHI_CENTER.get(asset_class, 0.25)),
        float(_ASSET_PHI_TAU.get(asset_class, 0.70)),
        float(_ASSET_PHI_STRENGTH.get(asset_class, 0.60)),
    )
