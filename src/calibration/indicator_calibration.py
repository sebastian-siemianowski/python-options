"""Indicator-aware calibration diagnostics and bounded adjustment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Mapping, Sequence, Tuple

import numpy as np


_NORMAL = NormalDist()


@dataclass(frozen=True)
class IndicatorEmosParams:
    """Linear EMOS-style mean and log-sigma adjustment parameters."""

    mean_intercept: float
    mean_weights: Tuple[float, ...]
    log_sigma_intercept: float
    log_sigma_weights: Tuple[float, ...]


def _as_2d_indicators(indicators: Sequence[Sequence[float]], n_obs: int) -> np.ndarray:
    x = np.asarray(indicators, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(n_obs, 1)
    if x.ndim != 2:
        raise ValueError("indicators must be 2D")
    if x.shape[0] != n_obs:
        raise ValueError("indicator row count must match observations")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def compute_indicator_pit_audit(pit_values: Sequence[float]) -> Mapping[str, float]:
    """Compute PIT KS, Anderson-Darling, and Berkowitz moment diagnostics."""
    pit = np.asarray(pit_values, dtype=np.float64)
    pit = pit[np.isfinite(pit)]
    if pit.size == 0:
        raise ValueError("pit_values must contain at least one finite value")
    u = np.clip(np.sort(pit), 1e-6, 1.0 - 1e-6)
    n = u.size
    empirical = np.arange(1, n + 1, dtype=np.float64) / n
    ks_stat = float(np.max(np.maximum(np.abs(empirical - u), np.abs(u - (np.arange(n, dtype=np.float64) / n)))))
    idx = np.arange(1, n + 1, dtype=np.float64)
    ad_stat = float(-n - np.mean((2.0 * idx - 1.0) * (np.log(u) + np.log(1.0 - u[::-1]))))
    z = np.asarray([_NORMAL.inv_cdf(float(value)) for value in u], dtype=np.float64)
    z_mean = float(np.mean(z))
    z_var = float(np.var(z))
    berkowitz_moment_error = abs(z_mean) + abs(z_var - 1.0)
    return {
        "n": float(n),
        "ks_stat": ks_stat,
        "ad_stat": ad_stat,
        "berkowitz_z_mean": z_mean,
        "berkowitz_z_var": z_var,
        "berkowitz_moment_error": float(berkowitz_moment_error),
    }


def fit_indicator_emos_params(
    realized: Sequence[float],
    mu: Sequence[float],
    sigma: Sequence[float],
    indicators: Sequence[Sequence[float]],
    ridge: float = 1e-3,
) -> IndicatorEmosParams:
    """Fit a small ridge EMOS adjustment from residuals and indicators."""
    y = np.asarray(realized, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    if y.shape != mu_arr.shape or y.shape != sigma_arr.shape:
        raise ValueError("realized, mu, and sigma must have the same shape")
    if y.ndim != 1:
        raise ValueError("realized, mu, and sigma must be 1D")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    x = _as_2d_indicators(indicators, len(y))
    design = np.column_stack([np.ones(len(y), dtype=np.float64), x])
    penalty = ridge * np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    residual = np.nan_to_num(y - mu_arr, nan=0.0, posinf=0.0, neginf=0.0)
    mean_coef = np.linalg.solve(design.T @ design + penalty, design.T @ residual)
    safe_sigma = np.maximum(np.asarray(sigma_arr, dtype=np.float64), 1e-8)
    log_target = np.log(np.maximum(np.abs(residual), 1e-8) / safe_sigma)
    log_target = np.clip(log_target, -2.0, 2.0)
    sigma_coef = np.linalg.solve(design.T @ design + penalty, design.T @ log_target)
    return IndicatorEmosParams(
        mean_intercept=float(mean_coef[0]),
        mean_weights=tuple(float(v) for v in mean_coef[1:]),
        log_sigma_intercept=float(sigma_coef[0]),
        log_sigma_weights=tuple(float(v) for v in sigma_coef[1:]),
    )


def apply_indicator_emos(
    mu: Sequence[float],
    sigma: Sequence[float],
    indicators: Sequence[Sequence[float]],
    params: IndicatorEmosParams,
    max_abs_mean_adjustment: float = 0.05,
    min_sigma_mult: float = 0.70,
    max_sigma_mult: float = 1.80,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply bounded EMOS mean and sigma adjustments."""
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    if mu_arr.shape != sigma_arr.shape or mu_arr.ndim != 1:
        raise ValueError("mu and sigma must be same-shaped 1D arrays")
    x = _as_2d_indicators(indicators, len(mu_arr))
    mean_weights = np.asarray(params.mean_weights, dtype=np.float64)
    sigma_weights = np.asarray(params.log_sigma_weights, dtype=np.float64)
    if mean_weights.shape != (x.shape[1],) or sigma_weights.shape != (x.shape[1],):
        raise ValueError("parameter weight count must match indicator columns")
    mean_adjustment = params.mean_intercept + x @ mean_weights
    mean_adjustment = np.clip(mean_adjustment, -max_abs_mean_adjustment, max_abs_mean_adjustment)
    log_sigma = params.log_sigma_intercept + x @ sigma_weights
    sigma_mult = np.clip(np.exp(np.clip(log_sigma, -2.0, 2.0)), min_sigma_mult, max_sigma_mult)
    return (
        np.ascontiguousarray(mu_arr + mean_adjustment, dtype=np.float64),
        np.ascontiguousarray(np.maximum(sigma_arr, 1e-8) * sigma_mult, dtype=np.float64),
    )


def beta_calibration_transform(
    probabilities: Sequence[float],
    a: float,
    b: float,
    c: float,
) -> np.ndarray:
    """Apply beta calibration in logit space with finite clipping."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    score = a * np.log(p) - b * np.log1p(-p) + c
    out = 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))
    return np.ascontiguousarray(np.clip(out, 1e-6, 1.0 - 1e-6), dtype=np.float64)


def compute_threshold_stability(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    thresholds: Sequence[float] = (0.55, 0.60, 0.65),
) -> Mapping[str, float]:
    """Audit hit-rate stability across probability thresholds."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("probabilities and outcomes must be same-shaped 1D arrays")
    p = np.clip(p, 0.0, 1.0)
    y = (y > 0.0).astype(np.float64)
    hit_rates = []
    coverages = []
    for threshold in thresholds:
        mask = p >= float(threshold)
        coverages.append(float(np.mean(mask)) if len(mask) else 0.0)
        if bool(np.any(mask)):
            hit_rates.append(float(np.mean(y[mask])))
        else:
            hit_rates.append(float("nan"))
    finite_hits = np.asarray([v for v in hit_rates if np.isfinite(v)], dtype=np.float64)
    spread = float(np.max(finite_hits) - np.min(finite_hits)) if finite_hits.size else float("nan")
    brier = float(np.mean((p - y) ** 2)) if len(p) else float("nan")
    return {
        "brier": brier,
        "hit_rate_spread": spread,
        "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "threshold_count": float(len(tuple(thresholds))),
    }
