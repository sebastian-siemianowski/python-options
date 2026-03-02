#!/usr/bin/env python3
"""
===============================================================================
SIGNALS CALIBRATION ENGINE — Pass 2 of Two-Pass Tuning
===============================================================================

After BMA model tuning (Pass 1), this module runs a walk-forward mini-backtest
per asset using the full compute_features() → latest_signals() pipeline.

It measures the actual reliability of p_up, exp_ret, and labels, then stores
per-asset correction factors in the tune JSON (src/data/tune/).

Four correction types per horizon:
  1. p_up isotonic recalibration map  — fixes under-confidence
  2. Magnitude scale factor           — fixes magnitude under-prediction
  3. Additive bias correction          — fixes systematic drift bias
  4. Optimal label thresholds          — per-asset buy/sell thresholds

At inference time, signals.py reads these corrections and applies them inline.

Integration:
  - Called automatically at the end of `make tune` (tune_ux.py)
  - Stores results in src/data/tune/{SYMBOL}.json under "signals_calibration"
  - Backward compatible: missing calibration → raw values used (identity)

Usage:
  # Programmatic (from tune_ux.py):
  from decision.signals_calibration import run_signals_calibration
  cache = run_signals_calibration(cache, workers=8)

  # Standalone:
  python src/decision/signals_calibration.py --assets SPY,QQQ,GLD
  python src/decision/signals_calibration.py --workers 8
===============================================================================
"""
from __future__ import annotations

import argparse
import json
import io
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ.setdefault("TUNING_QUIET", "1")
os.environ.setdefault("OFFLINE_MODE", "1")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rich imports (optional)
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console(force_terminal=True, color_system="truecolor", width=160) if HAS_RICH else None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TUNE_CACHE_DIR = Path(os.path.join(REPO_ROOT, "src", "data", "tune"))
PRICE_CACHE_DIR = Path(os.path.join(REPO_ROOT, "src", "data", "prices"))

CALIBRATION_VERSION = "3.1"

# Horizons to calibrate (days)
CALIBRATE_HORIZONS = [1, 7, 21, 63]
HORIZON_LABELS = {1: "1d", 7: "1w", 21: "1m", 63: "3m"}

# Walk-forward parameters — 500d/5d gives ~100 eval points
# (v2.0 used spacing=3 → 166 points, but 100 is sufficient for
# Beta/EMOS fitting with sqrt(n) shrinkage, and 40% faster)
DEFAULT_EVAL_DAYS = 500
DEFAULT_EVAL_SPACING = 5
MIN_EVAL_POINTS = 10
MIN_PRICE_POINTS = 120
FAST_EVAL_SPACING = 10  # --fast mode: ~50 points, 2x faster again

# Shrinkage: corrections blend toward identity using linear n/SHRINKAGE_FULL_N.
# At n=25 → 50%, n=50 → full weight.  (v3.1: changed from sqrt(n/100)
# to linear(n/50) — sqrt was too conservative for regime partitions
# with 15-30 points where Beta(3) + EMOS(4) params are well-identified.)
SHRINKAGE_FULL_N = 50

# ---------------------------------------------------------------------------
# Beta Calibration (Kull et al. 2017) constants
# ---------------------------------------------------------------------------
# 3-param generalization of Platt:  logit(p_cal) = a*ln(p) - b*ln(1-p) + c
# When a=b reduces to Platt; when a!=b captures asymmetric miscalibration.
# Identity: a=1, b=1, c=0.
BETA_IDENTITY = {"a": 1.0, "b": 1.0, "c": 0.0}
BETA_MIN_POINTS = 8
BETA_CLIP_LO = 0.005  # tighter than Platt 0.01 for ln(p) stability
BETA_CLIP_HI = 0.995

# ---------------------------------------------------------------------------
# EMOS (Gneiting et al. 2005) constants
# ---------------------------------------------------------------------------
# 4-param affine distributional correction optimized via CRPS:
#   mu_cor  = a + b * mu_pred
#   sig_cor = max(eps, c + d * sig_pred)
# Identity: a=0, b=1, c=0, d=1.
# Replaces BOTH mag_scale AND bias — unified distributional correction.
EMOS_IDENTITY = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0}
EMOS_MIN_POINTS = 12
EMOS_SIGMA_FLOOR = 0.01  # prevents degenerate sigma

# ---------------------------------------------------------------------------
# Regime conditioning
# ---------------------------------------------------------------------------
# vol_regime thresholds (matches assign_regime_labels boundaries)
REGIME_BINS = {
    "LOW":    (0.0, 0.85),
    "NORMAL": (0.85, 1.3),
    "HIGH":   (1.3, 99.0),
}
MIN_REGIME_POINTS = 15  # fallback to ALL (pooled) if below this
REGIME_ALL = "ALL"  # key for pooled (all regimes) calibration

# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------
# Exponential decay: w_t = exp(-lambda * (T-t) / spacing)
# v3.1: lambda=0.015 → half-life ~46 eval points (~1 year at spacing=5).
# (v3.0 used 0.005 → half-life 2.8 years, too slow for financial regimes.)
RECENCY_LAMBDA = 0.015

# Legacy constants (kept for backward compat with v2.0 loading)
MAG_SCALE_MIN = 0.3
MAG_SCALE_MAX = 8.0
MAG_SCALE_CAP = {1: 1.5, 7: 3.0, 21: 5.0, 63: 8.0}
BIAS_WINSORIZE_PCT = 5.0
PLATT_MIN_POINTS = 8

# Label threshold grid search
BUY_THR_GRID = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62]
SELL_THR_GRID = [0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48]
MIN_THR_SEPARATION = 0.10

# Label optimization weights
LABEL_HIT_WEIGHT = 0.4
LABEL_BRIER_WEIGHT = 0.3
LABEL_ACC_WEIGHT = 0.3

# All verify horizons (superset — we only calibrate a subset)
ALL_HORIZONS = [1, 3, 7, 21, 63, 126, 252]


# ---------------------------------------------------------------------------
# Numba-accelerated calibration kernels (imported lazily)
# ---------------------------------------------------------------------------
def _get_numba_kernels():
    """Lazy import of Numba-accelerated calibration kernels."""
    try:
        from decision.signals_calibration_numba import (
            isotonic_regression_nb,
            compute_magnitude_scale_nb,
            compute_bias_correction_nb,
            compute_hit_rates_nb,
            compute_brier_score_nb,
            grid_search_thresholds_nb,
            apply_isotonic_map_nb,
        )
        return {
            "isotonic_regression": isotonic_regression_nb,
            "magnitude_scale": compute_magnitude_scale_nb,
            "bias_correction": compute_bias_correction_nb,
            "hit_rates": compute_hit_rates_nb,
            "brier_score": compute_brier_score_nb,
            "grid_search_thresholds": grid_search_thresholds_nb,
            "apply_isotonic_map": apply_isotonic_map_nb,
        }
    except (ImportError, Exception):
        return None


_NUMBA_KERNELS = None


def _nb():
    """Get or initialize Numba kernels (singleton)."""
    global _NUMBA_KERNELS
    if _NUMBA_KERNELS is None:
        _NUMBA_KERNELS = _get_numba_kernels()
    return _NUMBA_KERNELS


# ---------------------------------------------------------------------------
# Helper: load OHLC + Close from disk cache
# ---------------------------------------------------------------------------
def _load_ohlc(symbol: str) -> Optional[pd.DataFrame]:
    """Load full OHLC DataFrame from disk cache CSV."""
    safe = symbol.replace("/", "_").replace("=", "_").replace(":", "_").upper()
    path = PRICE_CACHE_DIR / f"{safe}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, format="ISO8601", errors="coerce")
        df = df[~df.index.isna()]
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if "Close" not in df.columns:
            return None
        return df
    except Exception:
        return None


def _extract_close(ohlc_df: pd.DataFrame) -> Optional[pd.Series]:
    """Extract Close price series from OHLC DataFrame."""
    for col in ["Close", "Adj Close"]:
        if col in ohlc_df.columns:
            px = pd.to_numeric(ohlc_df[col], errors="coerce").dropna()
            if isinstance(px, pd.DataFrame):
                px = px.iloc[:, 0]
            if len(px) > 0:
                return px
    return None


# ===========================================================================
# SIGMA_H DERIVATION (v3.1 — fix for vol_mean mismatch)
# ===========================================================================

def _derive_sigma_H(sig) -> float:
    """Derive the horizon return std dev (sH) from Signal fields.

    v3.0 BUG: Used vol_mean (mean of per-step stochastic vol MC paths)
    as sigma_H proxy.  But at inference time, EMOS corrects sH which is
    sqrt(var(sim_H)) — the std dev of the RETURN distribution, a
    fundamentally different quantity.

    v3.1 FIX: Derive sH from score and exp_ret:
      score = z_stat = mH / sH  (signals.py L7591)
      therefore sH = |mH / z_stat| = |exp_ret / score|

    Fallback chain:
      1. exp_ret / score  (exact, when |score| > 1e-6)
      2. (ci_high - ci_low) / 2  (approximate, from 68% CI width)
      3. vol_mean  (legacy, better than nothing)

    Returns:
        sigma_H in percent units (multiplied by 100)
    """
    score = getattr(sig, "score", 0.0)
    exp_ret = getattr(sig, "exp_ret", 0.0)

    # Primary: exact derivation from score = mu_H / sH
    if abs(score) > 1e-6:
        sH = abs(exp_ret / score)
        if 1e-8 < sH < 10.0:  # sanity: sH in log-return units
            return sH * 100.0

    # Fallback: approximate from CI width (68% CI → ~1 sigma each side)
    ci_low = getattr(sig, "ci_low", None)
    ci_high = getattr(sig, "ci_high", None)
    if ci_low is not None and ci_high is not None:
        ci_width = ci_high - ci_low
        if ci_width > 1e-8:
            return (ci_width / 2.0) * 100.0

    # Last resort: vol_mean (legacy, known to be wrong scale but
    # better than a constant)
    vol_mean = getattr(sig, "vol_mean", 0.0)
    return vol_mean * 100.0


# ===========================================================================
# SHRINKAGE HELPER
# ===========================================================================

def _shrinkage_weight(n_eval: int) -> float:
    """Compute shrinkage weight: 0 at n=0, 1 at n>=SHRINKAGE_FULL_N.

    v3.1: Linear shrinkage min(1.0, n/50).  Replaced sqrt(n/100) which
    was too conservative — at n=25 it discarded 50% of corrections.
    With 7 total params (Beta 3 + EMOS 4), n=25 gives DOF ratio of 3.6,
    sufficient for well-regularized fitting.
    """
    return min(1.0, n_eval / SHRINKAGE_FULL_N)


# ===========================================================================
# BETA CALIBRATION (Kull et al. 2017)
# ===========================================================================
# 3-parameter generalization of Platt scaling:
#
#   logit(p_cal) = a * ln(p) - b * ln(1-p) + c
#
# When a = b → equivalent to Platt scaling (symmetric logit transform)
# When a ≠ b → captures asymmetric miscalibration (the common case):
#     - Model under-confident for p>0.5 but over-confident for p<0.5
#     - Different a, b handle each side of 0.5 independently
#
# Identity (no correction): a=1, b=1, c=0
# Platt special case:        a=b, c=free
#
# Strictly dominates Platt for financial data where asymmetry is the norm.
# ===========================================================================

def _fit_beta_calibration(
    raw_probs: np.ndarray,
    actual_outcomes: np.ndarray,
    n_eval: int,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fit Beta calibration (Kull et al. 2017) — 3-parameter asymmetric
    probability recalibration.

    Model: logit(p_cal) = a * ln(p) - b * ln(1-p) + c
    Identity: a=1, b=1, c=0.
    Corrections are shrunk toward identity proportional to sqrt(n_eval).

    The weighted NLL allows exponential recency weighting.

    Args:
        raw_probs:       Raw p_up values ∈ [0, 1]
        actual_outcomes: Binary outcomes (1 if return > 0)
        n_eval:          Total evaluation points (for shrinkage)
        weights:         Optional per-sample weights (recency)

    Returns:
        {"type": "beta", "a": float, "b": float, "c": float}
    """
    from scipy.optimize import minimize as sp_minimize

    if len(raw_probs) < BETA_MIN_POINTS:
        return {"type": "beta", **BETA_IDENTITY}

    p_clipped = np.clip(raw_probs, BETA_CLIP_LO, BETA_CLIP_HI)
    ln_p = np.log(p_clipped)             # ln(p)
    ln_1mp = np.log(1.0 - p_clipped)     # ln(1-p)
    y = actual_outcomes.astype(np.float64)

    if weights is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = weights / weights.sum() * len(y)  # normalize so sum(w)=N

    def _weighted_nll(params):
        a, b, c = params
        # logit(p_cal) = a*ln(p) - b*ln(1-p) + c
        z = a * ln_p - b * ln_1mp + c
        # Numerically stable sigmoid
        p_cal = np.where(z >= 0,
                         1.0 / (1.0 + np.exp(-z)),
                         np.exp(z) / (1.0 + np.exp(z)))
        p_cal = np.clip(p_cal, 1e-10, 1.0 - 1e-10)
        # Weighted binary cross-entropy
        bce = -(y * np.log(p_cal) + (1.0 - y) * np.log(1.0 - p_cal))
        nll = np.sum(w * bce) / np.sum(w)
        # L2 regularization toward identity (a=1, b=1, c=0)
        reg = 0.01 * ((a - 1.0) ** 2 + (b - 1.0) ** 2 + c ** 2)
        return nll + reg

    try:
        result = sp_minimize(
            _weighted_nll,
            x0=[1.0, 1.0, 0.0],
            method="L-BFGS-B",
            bounds=[(0.01, 10.0), (0.01, 10.0), (-5.0, 5.0)],
        )
        a_fit, b_fit, c_fit = result.x
    except Exception:
        a_fit, b_fit, c_fit = 1.0, 1.0, 0.0

    # v3.1: Linear shrinkage toward identity (was sqrt in v3.0)
    lam = min(1.0, n_eval / SHRINKAGE_FULL_N)
    a_shrunk = lam * a_fit + (1.0 - lam) * 1.0
    b_shrunk = lam * b_fit + (1.0 - lam) * 1.0
    c_shrunk = lam * c_fit  # identity c = 0

    return {
        "type": "beta",
        "a": round(float(a_shrunk), 6),
        "b": round(float(b_shrunk), 6),
        "c": round(float(c_shrunk), 6),
    }


def apply_beta_map(raw_p: float, p_map: Dict) -> float:
    """Apply Beta calibration to a single probability.

    Model: logit(p_cal) = a * ln(p) - b * ln(1-p) + c

    Args:
        raw_p: Raw probability ∈ [0, 1]
        p_map: {"type": "beta", "a": ..., "b": ..., "c": ...}

    Returns:
        Calibrated probability ∈ [0, 1]
    """
    a = p_map.get("a", 1.0)
    b = p_map.get("b", 1.0)
    c = p_map.get("c", 0.0)
    p_clipped = max(BETA_CLIP_LO, min(BETA_CLIP_HI, raw_p))
    z = a * math.log(p_clipped) - b * math.log(1.0 - p_clipped) + c
    if z >= 0:
        result = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        result = ez / (1.0 + ez)
    return max(0.0, min(1.0, result))


def apply_platt_map(raw_p: float, p_map: Dict) -> float:
    """Apply Platt scaling to a single probability (legacy v2.0 compat).

    Args:
        raw_p: Raw probability ∈ [0, 1]
        p_map: {"type": "platt", "a": ..., "b": ...}

    Returns:
        Calibrated probability ∈ [0, 1]
    """
    a = p_map.get("a", 1.0)
    b = p_map.get("b", 0.0)
    p_clipped = max(0.01, min(0.99, raw_p))
    logit = math.log(p_clipped / (1.0 - p_clipped))
    z = a * logit + b
    if z >= 0:
        result = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        result = ez / (1.0 + ez)
    return max(0.0, min(1.0, result))


def apply_p_up_map(raw_p: float, p_map: Dict) -> float:
    """Apply calibration map (auto-detects Beta, Platt, or legacy isotonic).

    Args:
        raw_p: Raw probability ∈ [0, 1]
        p_map: Calibration map dict

    Returns:
        Calibrated probability ∈ [0, 1]
    """
    map_type = p_map.get("type", "isotonic")
    if map_type == "beta":
        return apply_beta_map(raw_p, p_map)
    if map_type == "platt":
        return apply_platt_map(raw_p, p_map)
    # Legacy isotonic fallback
    x = p_map.get("x", [0.0, 1.0])
    y = p_map.get("y", [0.0, 1.0])
    if len(x) < 2:
        return raw_p
    result = float(np.interp(raw_p, x, y))
    return max(0.0, min(1.0, result))


# ===========================================================================
# EMOS — Ensemble Model Output Statistics (Gneiting et al. 2005)
# ===========================================================================
# 4-parameter affine distributional correction OPTIMIZED via CRPS:
#
#   μ_cor  = a + b * μ_pred
#   σ_cor  = max(ε, c + d * σ_pred)
#
# Unified replacement for BOTH mag_scale AND bias:
#   - 'a' absorbs additive bias
#   - 'b' absorbs magnitude scale
#   - 'c','d' calibrate the variance (no equivalent in v2.0!)
#
# Optimized via CRPS (Continuous Ranked Probability Score) — the
# strictly proper scoring rule for distributional forecasts:
#
#   CRPS(N(μ,σ²), y) = σ [ z(2Φ(z)-1) + 2φ(z) - 1/√π ]
#   where z = (y - μ) / σ
#
# Identity: a=0, b=1, c=0, d=1
# ===========================================================================

def _crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Closed-form CRPS for Gaussian predictive distribution (Gneiting 2005).

    CRPS(N(μ,σ²), y) = σ [ z(2Φ(z)-1) + 2φ(z) - 1/√π ]
    where z = (y - μ) / σ

    This is a strictly proper scoring rule rewarding both calibration AND
    sharpness — unlike Brier score (calibration only) or log-likelihood
    (unstable for misspecified variance).

    Args:
        mu:    Predicted means (array)
        sigma: Predicted std devs (array, must be > 0)
        y:     Observed values (array)

    Returns:
        Per-sample CRPS values (array, lower = better)
    """
    from scipy.stats import norm
    z = (y - mu) / np.maximum(sigma, 1e-10)
    crps = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / math.sqrt(math.pi))
    return crps


def _fit_emos(
    predicted: np.ndarray,
    actual: np.ndarray,
    sigma_pred: np.ndarray,
    n_eval: int,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fit EMOS (Gneiting et al. 2005) — CRPS-optimal affine distributional
    correction.

    Model:
        mu_cor  = a + b * mu_pred
        sig_cor = max(eps, c + d * sig_pred)

    Fitting: minimize weighted mean CRPS over walk-forward eval points.
    Falls back to identity if insufficient data.

    Args:
        predicted:  Predicted returns (log %)
        actual:     Actual returns (log %)
        sigma_pred: Predicted standard deviations (log %)
        n_eval:     Total evaluation points (for shrinkage)
        weights:    Optional per-sample weights (recency)

    Returns:
        {"type": "emos", "a": float, "b": float, "c": float, "d": float}
    """
    from scipy.optimize import minimize as sp_minimize

    if len(predicted) < EMOS_MIN_POINTS:
        return {"type": "emos", **EMOS_IDENTITY}

    mu_pred = predicted.astype(np.float64)
    y = actual.astype(np.float64)
    sig_pred = np.maximum(sigma_pred.astype(np.float64), EMOS_SIGMA_FLOOR)

    if weights is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = weights / weights.sum() * len(y)

    def _weighted_crps(params):
        a, b, c, d = params
        mu_cor = a + b * mu_pred
        sig_cor = np.maximum(EMOS_SIGMA_FLOOR, c + d * sig_pred)
        crps_vals = _crps_gaussian(mu_cor, sig_cor, y)
        # Weighted mean CRPS
        loss = np.sum(w * crps_vals) / np.sum(w)
        # L2 regularization toward identity (a=0, b=1, c=0, d=1)
        # v3.1: reg=0.01 (up from 0.005) for stability with wider b bounds
        reg = 0.01 * (a ** 2 + (b - 1.0) ** 2 + c ** 2 + (d - 1.0) ** 2)
        return loss + reg

    try:
        result = sp_minimize(
            _weighted_crps,
            x0=[0.0, 1.0, 0.0, 1.0],
            method="L-BFGS-B",
            # v3.1: b lower bound widened from 0.01 to -1.0.
            # SPY 7d had b=0.01 (hitting bound) — optimizer wants b≈0 or
            # negative, meaning predicted returns have near-zero magnitude
            # signal.  Allowing negative b lets optimizer discover if
            # sign-flip correction is warranted.
            bounds=[(-10.0, 10.0), (-1.0, 5.0), (-5.0, 5.0), (0.01, 5.0)],
        )
        a_fit, b_fit, c_fit, d_fit = result.x
    except Exception:
        a_fit, b_fit, c_fit, d_fit = 0.0, 1.0, 0.0, 1.0

    # v3.1: Linear shrinkage toward identity (was sqrt in v3.0)
    lam = min(1.0, n_eval / SHRINKAGE_FULL_N)
    a_s = lam * a_fit               # identity a = 0
    b_s = lam * b_fit + (1.0 - lam) * 1.0   # identity b = 1
    c_s = lam * c_fit               # identity c = 0
    d_s = lam * d_fit + (1.0 - lam) * 1.0   # identity d = 1

    return {
        "type": "emos",
        "a": round(float(a_s), 6),
        "b": round(float(b_s), 6),
        "c": round(float(c_s), 6),
        "d": round(float(d_s), 6),
    }


def apply_emos_correction(
    mu_H: float,
    sigma_H: float,
    emos_params: Dict,
) -> Tuple[float, float]:
    """
    Apply EMOS distributional correction at inference time.

    Args:
        mu_H:        Raw predicted return (log-return units)
        sigma_H:     Raw predicted std dev (log-return units)
        emos_params: {"type": "emos", "a": ..., "b": ..., "c": ..., "d": ...}

    Returns:
        (mu_corrected, sigma_corrected)
    """
    a = emos_params.get("a", 0.0)
    b = emos_params.get("b", 1.0)
    c = emos_params.get("c", 0.0)
    d = emos_params.get("d", 1.0)
    mu_cor = a + b * mu_H
    sig_cor = max(EMOS_SIGMA_FLOOR, c + d * sigma_H)
    return mu_cor, sig_cor


# ===========================================================================
# Legacy magnitude/bias functions (kept for v2.0 backward compat)
# ===========================================================================

def _compute_mag_scale(
    predicted: np.ndarray,
    actual: np.ndarray,
    horizon: int = 7,
    n_eval: int = 0,
    epsilon: float = 1e-6,
) -> float:
    """Legacy v2.0 magnitude scale (used only when loading v2.0 calibrations)."""
    cap = MAG_SCALE_CAP.get(horizon, MAG_SCALE_MAX)
    abs_pred = np.abs(predicted)
    abs_actual = np.abs(actual)
    valid = abs_pred > epsilon
    if valid.sum() < 3:
        return 1.0
    ratios = abs_actual[valid] / abs_pred[valid]
    scale = float(np.median(ratios))
    scale = max(MAG_SCALE_MIN, min(cap, scale))
    lam = _shrinkage_weight(n_eval)
    return float(lam * scale + (1.0 - lam) * 1.0)


def _compute_bias(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_eval: int = 0,
    winsorize_pct: float = BIAS_WINSORIZE_PCT,
) -> float:
    """Legacy v2.0 bias correction (used only when loading v2.0 calibrations)."""
    errors = actual - predicted
    if len(errors) < 3:
        return 0.0
    lo = np.percentile(errors, winsorize_pct)
    hi = np.percentile(errors, 100.0 - winsorize_pct)
    clipped = np.clip(errors, lo, hi)
    raw_bias = float(np.mean(clipped))
    lam = _shrinkage_weight(n_eval)
    return float(lam * raw_bias)


# ===========================================================================
# RECENCY WEIGHTING
# ===========================================================================

def _compute_recency_weights(n_points: int) -> np.ndarray:
    """
    Compute exponential recency weights for walk-forward eval points.

    w_t = exp(-λ * (N-1-t))  where t=0 is oldest, t=N-1 is newest.
    λ = RECENCY_LAMBDA = 0.005 → half-life ~140 points (~420 trading days).

    Recent calibration errors matter more than ancient ones.

    Args:
        n_points: Number of evaluation points

    Returns:
        Weight array (length n_points), most recent = highest weight
    """
    if n_points <= 1:
        return np.ones(n_points, dtype=np.float64)
    t = np.arange(n_points, dtype=np.float64)
    w = np.exp(-RECENCY_LAMBDA * (n_points - 1 - t))
    return w


# ===========================================================================
# REGIME PARTITIONING
# ===========================================================================

def _partition_by_regime(records: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Partition walk-forward records by vol_regime into LOW/NORMAL/HIGH bins.

    If any regime has fewer than MIN_REGIME_POINTS, it falls back to ALL
    (pooled). The ALL partition is always computed as baseline.

    Args:
        records: List of eval-point dicts (must have 'vol_regime' field)

    Returns:
        {"ALL": [...], "LOW": [...], "NORMAL": [...], "HIGH": [...]}
        Regimes with too few points are omitted (but ALL is always present).
    """
    partitions = {REGIME_ALL: records}
    for regime_name, (lo, hi) in REGIME_BINS.items():
        regime_recs = [
            r for r in records
            if lo <= r.get("vol_regime", 1.0) < hi
        ]
        if len(regime_recs) >= MIN_REGIME_POINTS:
            partitions[regime_name] = regime_recs
    return partitions


# ===========================================================================
# LABEL THRESHOLD OPTIMIZATION
# ===========================================================================

def _optimize_label_thresholds(
    p_ups: np.ndarray,
    actual_ups: np.ndarray,
    exp_rets: np.ndarray,
    actual_rets: np.ndarray,
) -> Dict[str, Any]:
    """
    Grid search for optimal buy/sell thresholds maximizing composite metric.

    Optimizes: hit_rate (0.4) + inv_brier (0.3) + label_accuracy (0.3)

    Args:
        p_ups: Raw p_up values ∈ [0, 1]
        actual_ups: Binary outcomes (1 if return > 0)
        exp_rets: Predicted returns (%)
        actual_rets: Actual returns (%)

    Returns:
        {"buy_thr": float, "sell_thr": float, "best_score": float,
         "label_accuracy_raw": float, "label_accuracy_opt": float}
    """
    nb = _nb()
    if nb is not None and len(p_ups) >= MIN_EVAL_POINTS:
        try:
            buy_grid = np.array(BUY_THR_GRID, dtype=np.float64)
            sell_grid = np.array(SELL_THR_GRID, dtype=np.float64)
            best_buy, best_sell, best_score = nb["grid_search_thresholds"](
                p_ups.astype(np.float64),
                actual_ups.astype(np.float64),
                exp_rets.astype(np.float64),
                actual_rets.astype(np.float64),
                buy_grid,
                sell_grid,
                MIN_THR_SEPARATION,
                LABEL_HIT_WEIGHT,
                LABEL_BRIER_WEIGHT,
                LABEL_ACC_WEIGHT,
            )
            if np.isfinite(best_score):
                # Compute raw accuracy with default thresholds
                raw_acc = _label_accuracy_at_thresholds(
                    p_ups, actual_ups, exp_rets, actual_rets, 0.58, 0.42
                )
                opt_acc = _label_accuracy_at_thresholds(
                    p_ups, actual_ups, exp_rets, actual_rets, best_buy, best_sell
                )
                return {
                    "buy_thr": float(best_buy),
                    "sell_thr": float(best_sell),
                    "best_score": float(best_score),
                    "label_accuracy_raw": float(raw_acc),
                    "label_accuracy_opt": float(opt_acc),
                }
        except Exception:
            pass

    # Fallback: pure Python grid search
    if len(p_ups) < MIN_EVAL_POINTS:
        return {"buy_thr": 0.58, "sell_thr": 0.42, "best_score": 0.0,
                "label_accuracy_raw": 0.0, "label_accuracy_opt": 0.0}

    best_score = -1.0
    best_buy = 0.58
    best_sell = 0.42

    for buy_thr in BUY_THR_GRID:
        for sell_thr in SELL_THR_GRID:
            if buy_thr - sell_thr < MIN_THR_SEPARATION:
                continue

            score = _eval_threshold_pair(
                p_ups, actual_ups, exp_rets, actual_rets, buy_thr, sell_thr
            )
            if score > best_score:
                best_score = score
                best_buy = buy_thr
                best_sell = sell_thr

    raw_acc = _label_accuracy_at_thresholds(
        p_ups, actual_ups, exp_rets, actual_rets, 0.58, 0.42
    )
    opt_acc = _label_accuracy_at_thresholds(
        p_ups, actual_ups, exp_rets, actual_rets, best_buy, best_sell
    )

    return {
        "buy_thr": float(best_buy),
        "sell_thr": float(best_sell),
        "best_score": float(best_score),
        "label_accuracy_raw": float(raw_acc),
        "label_accuracy_opt": float(opt_acc),
    }


def _eval_threshold_pair(
    p_ups: np.ndarray,
    actual_ups: np.ndarray,
    exp_rets: np.ndarray,
    actual_rets: np.ndarray,
    buy_thr: float,
    sell_thr: float,
) -> float:
    """Evaluate a buy/sell threshold pair. Returns composite score."""
    n = len(p_ups)
    if n < 5:
        return 0.0

    # Assign labels
    labels = np.where(p_ups >= buy_thr, 1, np.where(p_ups <= sell_thr, -1, 0))

    # Hit rate: fraction of non-HOLD labels that got direction right
    acted = labels != 0
    if acted.sum() < 3:
        return 0.0

    pred_dirs = np.sign(exp_rets)
    actual_dirs = np.sign(actual_rets)
    nonzero_pred = pred_dirs != 0
    valid_mask = acted & nonzero_pred
    if valid_mask.sum() < 2:
        return 0.0

    hit_rate = float(np.mean(pred_dirs[valid_mask] == actual_dirs[valid_mask]))

    # Brier score (lower is better)
    brier = float(np.mean((p_ups - actual_ups) ** 2))
    inv_brier = max(0.0, 1.0 - brier / 0.25)

    # Label accuracy: BUY signals that went up + SELL signals that went down
    buy_mask = labels == 1
    sell_mask = labels == -1
    correct = 0
    total_labels = 0
    if buy_mask.sum() > 0:
        correct += int(np.sum(actual_rets[buy_mask] > 0))
        total_labels += int(buy_mask.sum())
    if sell_mask.sum() > 0:
        correct += int(np.sum(actual_rets[sell_mask] < 0))
        total_labels += int(sell_mask.sum())
    label_acc = correct / max(total_labels, 1)

    # Composite
    return (
        LABEL_HIT_WEIGHT * hit_rate
        + LABEL_BRIER_WEIGHT * inv_brier
        + LABEL_ACC_WEIGHT * label_acc
    )


def _label_accuracy_at_thresholds(
    p_ups: np.ndarray,
    actual_ups: np.ndarray,
    exp_rets: np.ndarray,
    actual_rets: np.ndarray,
    buy_thr: float,
    sell_thr: float,
) -> float:
    """Compute label accuracy at given thresholds."""
    labels = np.where(p_ups >= buy_thr, 1, np.where(p_ups <= sell_thr, -1, 0))
    buy_mask = labels == 1
    sell_mask = labels == -1
    correct = 0
    total = 0
    if buy_mask.sum() > 0:
        correct += int(np.sum(actual_rets[buy_mask] > 0))
        total += int(buy_mask.sum())
    if sell_mask.sum() > 0:
        correct += int(np.sum(actual_rets[sell_mask] < 0))
        total += int(sell_mask.sum())
    return correct / max(total, 1)


# ===========================================================================
# CORE: Calibrate a single asset (worker function)
# ===========================================================================

def calibrate_single_asset(
    args_tuple: Tuple,
) -> Tuple[str, Optional[Dict], Optional[str]]:
    """
    Walk-forward calibration for one asset.

    Runs compute_features() → latest_signals() at each eval point,
    collects (p_up, exp_ret, label) vs realized returns, then computes
    all four correction types.

    Args:
        args_tuple: (symbol, eval_days, eval_spacing)

    Returns:
        (symbol, calibration_dict_or_None, error_or_None)
    """
    symbol, eval_days, eval_spacing = args_tuple

    warnings.filterwarnings("ignore")

    try:
        _src = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)),
            "src",
        )
        if _src not in sys.path:
            sys.path.insert(0, _src)
        os.environ["TUNING_QUIET"] = "1"
        os.environ["OFFLINE_MODE"] = "1"

        # Mock risk temperature to avoid Yahoo Finance hangs
        import decision.risk_temperature as _rt_mod
        from types import SimpleNamespace

        def _mock_risk_temp(**kwargs):
            return SimpleNamespace(
                risk_temperature=0.5,
                scale_factor=1.0,
                overnight_budget_applied=False,
                overnight_max_position=None,
                metals_stress=0.0,
                fx_stress=0.0,
                equity_stress=0.0,
                commodity_stress=0.0,
                duration_stress=0.0,
            )

        _rt_mod.get_cached_risk_temperature = _mock_risk_temp
        _rt_mod._risk_temp_cache = {}

        # Suppress console output from signals.py
        import contextlib
        from decision.signals import (
            compute_features,
            latest_signals,
            _load_tuned_kalman_params,
        )
        import decision.signals as _sig_mod
        if hasattr(_sig_mod, "console") and _sig_mod.console is not None:
            _sig_mod.console = type(_sig_mod.console)(file=io.StringIO(), force_terminal=False)

        # Load data
        ohlc_df = _load_ohlc(symbol)
        if ohlc_df is None:
            return symbol, None, "No OHLC data"

        px = _extract_close(ohlc_df)
        if px is None or len(px) < MIN_PRICE_POINTS:
            return symbol, None, "Insufficient price data"

        tuned_params = _load_tuned_kalman_params(symbol)

        n = len(px)
        start_idx = max(MIN_PRICE_POINTS, n - eval_days)

        # Collect records per horizon
        records: Dict[int, List[Dict]] = {h: [] for h in CALIBRATE_HORIZONS}

        idx = start_idx
        while idx < n:
            if len(px.iloc[:idx]) < 60:
                idx += eval_spacing
                continue

            px_trunc = px.iloc[:idx]
            ohlc_trunc = ohlc_df.iloc[:idx]
            last_close = float(px_trunc.iloc[-1])

            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    feats = compute_features(px_trunc, asset_symbol=symbol, ohlc_df=ohlc_trunc)
                    sigs, _ = latest_signals(
                        feats,
                        horizons=CALIBRATE_HORIZONS,  # Only compute horizons we calibrate (4 vs 7 = 43% faster)
                        last_close=last_close,
                        t_map=True,
                        ci=0.68,
                        tuned_params=tuned_params,
                        asset_key=symbol,
                    )
            except Exception:
                idx += eval_spacing
                continue

            sig_map = {s.horizon_days: s for s in sigs}

            for H in CALIBRATE_HORIZONS:
                sig = sig_map.get(H)
                if sig is None:
                    continue

                future_idx = idx - 1 + H
                best_future_idx = None
                for offset in range(0, 6):
                    candidate = future_idx + offset
                    if candidate < n:
                        best_future_idx = candidate
                        break

                if best_future_idx is None or best_future_idx >= n:
                    continue

                price_now = float(px.iloc[idx - 1])
                price_future = float(px.iloc[best_future_idx])

                if price_now <= 0 or np.isnan(price_now) or np.isnan(price_future):
                    continue

                # Use log-returns consistently — exp_ret from signals.py is
                # already a log-return.  Using simple-returns here creates a
                # systematic mismatch that biases mag_scale upward at long
                # horizons (>5% moves).
                actual_pct = math.log(price_future / price_now) * 100.0
                pred_pct = sig.exp_ret * 100.0

                # v3.0: Collect 12+ fields per eval point for regime-conditional
                # calibration and EMOS distributional correction.
                # vol_regime, sigma_H are critical for regime partitioning and EMOS.
                vol_regime = feats.get("vol_regime", 1.0) if isinstance(feats, dict) else getattr(feats, "vol_regime", 1.0)
                # Try to get vol_regime from feature dict keys
                if isinstance(feats, dict):
                    vr = feats.get("vol_regime")
                    if vr is None:
                        vr = feats.get("vol_reg")
                    if vr is not None and hasattr(vr, '__len__'):
                        vr = float(vr.iloc[-1]) if hasattr(vr, 'iloc') else float(vr[-1]) if len(vr) > 0 else 1.0
                    vol_regime = float(vr) if vr is not None else 1.0
                else:
                    vol_regime = 1.0

                records[H].append({
                    "predicted": pred_pct,
                    "actual": actual_pct,
                    "p_up": sig.p_up,
                    "label": sig.label,
                    "actual_up": 1 if actual_pct > 0 else 0,
                    # v3.1: sigma_H derived from exp_ret/score to match
                    # inference-time sH = sqrt(var(sim_H)).  v3.0 used
                    # vol_mean (mean of per-step stochastic vol paths) which
                    # is a fundamentally different quantity — EMOS parameter
                    # d was fitted against the wrong scale.
                    "sigma_H": _derive_sigma_H(sig),  # predicted sigma in %
                    "vol_regime": vol_regime,
                    "score": getattr(sig, "score", 0.0),
                    "regime": getattr(sig, "regime", "UNKNOWN"),
                    "eval_idx": idx,  # for recency weighting
                })

            idx += eval_spacing

        # Build calibration dict
        calibration = _build_calibration(records)
        if calibration is None:
            return symbol, None, "Too few eval points"

        return symbol, calibration, None

    except Exception as e:
        return symbol, None, f"{type(e).__name__}: {e}"


# ===========================================================================
# Build calibration dict from walk-forward records
# ===========================================================================

def _build_calibration(
    records: Dict[int, List[Dict]],
) -> Optional[Dict]:
    """
    Compute regime-conditional calibration from walk-forward records.

    v3.1 Elite Calibration:
      1. Beta Calibration (Kull et al. 2017) — 3-param asymmetric p_up
         recalibration. Handles under-confident p>0.5 / over-confident p<0.5.
      2. EMOS (Gneiting et al. 2005) — 4-param distributional correction
         optimized via CRPS. Replaces BOTH mag_scale AND bias.
      3. Regime Conditioning — separate params per vol_regime (LOW/NORMAL/HIGH).
         Fallback to ALL (pooled) if regime has <15 points.
      4. Exponential Recency Weighting — recent errors weighted higher.
         Half-life ~46 eval points (~1 year at spacing=5).
      5. Per-horizon label thresholds (inherited from v2.0).
      6. Temporal Cross-Validation (v3.1) — 70/30 temporal split guards
         against overfitting. Corrections that degrade validation metrics
         are reverted to identity.

    Storage format:
      horizons[H].by_regime[REGIME] = {
          "p_up_map": {type, a, b, c},   # Beta calibration
          "emos": {type, a, b, c, d},     # EMOS params
          "n_eval": int,
      }
      horizons[H].p_up_map = ALL regime map (backward compat)
      horizons[H].emos = ALL regime EMOS (backward compat)

    Returns:
        Calibration dict. None if insufficient data.
    """
    any_valid = False
    horizons_cal = {}
    label_by_horizon = {}
    fallback_label_result = None

    for H in CALIBRATE_HORIZONS:
        recs = records.get(H, [])

        if len(recs) < MIN_EVAL_POINTS:
            continue

        any_valid = True
        n_eval = len(recs)

        # === Recency weights (applied to ALL fitting) ===
        rec_weights = _compute_recency_weights(n_eval)

        # === Regime partitioning ===
        partitions = _partition_by_regime(recs)

        # === Fit per-regime Beta + EMOS with cross-validation guard ===
        by_regime = {}
        cv_diagnostics = {}  # v3.1: track train/val metrics
        for regime_name, regime_recs in partitions.items():
            nr = len(regime_recs)
            if nr < MIN_EVAL_POINTS:
                continue

            predicted = np.array([r["predicted"] for r in regime_recs], dtype=np.float64)
            actual = np.array([r["actual"] for r in regime_recs], dtype=np.float64)
            p_ups = np.array([r["p_up"] for r in regime_recs], dtype=np.float64)
            actual_ups = np.array([r["actual_up"] for r in regime_recs], dtype=np.float64)
            sigma_pred = np.array([r.get("sigma_H", 1.0) for r in regime_recs], dtype=np.float64)

            # Compute recency weights for this partition
            w = _compute_recency_weights(nr)

            # 1) Beta calibration (3-param asymmetric)
            p_up_map = _fit_beta_calibration(p_ups, actual_ups, nr, weights=w)

            # 2) EMOS distributional correction (CRPS-optimal)
            emos_params = _fit_emos(predicted, actual, sigma_pred, nr, weights=w)

            # === v3.1: Temporal cross-validation guard ===
            # Split records 70/30 temporally (records are already in time
            # order from the walk-forward loop).  Fit on first 70%, validate
            # on last 30%.  If calibrated metrics are WORSE on validation
            # set than identity (no-op), revert to identity.
            # Only apply CV guard when we have enough points for meaningful
            # validation (>= 20 total → 14 train, 6 val).
            cv_reverted_beta = False
            cv_reverted_emos = False
            if nr >= 20:
                split_idx = int(nr * 0.7)
                # Validation slice
                val_p_ups = p_ups[split_idx:]
                val_actual_ups = actual_ups[split_idx:]
                val_predicted = predicted[split_idx:]
                val_actual = actual[split_idx:]
                val_sigma = sigma_pred[split_idx:]

                # Train-set-only fit (for CV comparison)
                train_p_ups = p_ups[:split_idx]
                train_actual_ups = actual_ups[:split_idx]
                train_predicted = predicted[:split_idx]
                train_actual = actual[:split_idx]
                train_sigma = sigma_pred[:split_idx]
                n_train = len(train_predicted)
                w_train = _compute_recency_weights(n_train)

                cv_beta = _fit_beta_calibration(train_p_ups, train_actual_ups, n_train, weights=w_train)
                cv_emos = _fit_emos(train_predicted, train_actual, train_sigma, n_train, weights=w_train)

                # Beta CV check: Brier on validation
                brier_raw_val = float(np.mean((val_p_ups - val_actual_ups) ** 2))
                cal_p_val = np.array([apply_p_up_map(p, cv_beta) for p in val_p_ups])
                brier_cal_val = float(np.mean((cal_p_val - val_actual_ups) ** 2))
                if brier_cal_val > brier_raw_val + 0.001:
                    # Calibrated Beta is WORSE on validation → revert
                    p_up_map = {"type": "beta", **BETA_IDENTITY}
                    cv_reverted_beta = True

                # EMOS CV check: CRPS on validation
                crps_raw_val = float(np.mean(_crps_gaussian(
                    val_predicted,
                    np.maximum(val_sigma, EMOS_SIGMA_FLOOR),
                    val_actual,
                )))
                mu_cv = cv_emos.get("a", 0.0) + cv_emos.get("b", 1.0) * val_predicted
                sig_cv = np.maximum(EMOS_SIGMA_FLOOR,
                                    cv_emos.get("c", 0.0) + cv_emos.get("d", 1.0) * val_sigma)
                crps_cal_val = float(np.mean(_crps_gaussian(mu_cv, sig_cv, val_actual)))
                if crps_cal_val > crps_raw_val + 0.001:
                    # Calibrated EMOS is WORSE on validation → revert
                    emos_params = {"type": "emos", **EMOS_IDENTITY}
                    cv_reverted_emos = True

                cv_diagnostics[regime_name] = {
                    "n_train": n_train,
                    "n_val": nr - split_idx,
                    "brier_raw_val": round(brier_raw_val, 6),
                    "brier_cal_val": round(brier_cal_val, 6),
                    "crps_raw_val": round(crps_raw_val, 6),
                    "crps_cal_val": round(crps_cal_val, 6),
                    "beta_reverted": cv_reverted_beta,
                    "emos_reverted": cv_reverted_emos,
                }

            by_regime[regime_name] = {
                "p_up_map": p_up_map,
                "emos": emos_params,
                "n_eval": nr,
            }

        # === Raw metrics for diagnostics (from ALL partition) ===
        all_recs = partitions.get(REGIME_ALL, recs)
        predicted = np.array([r["predicted"] for r in all_recs], dtype=np.float64)
        actual = np.array([r["actual"] for r in all_recs], dtype=np.float64)
        p_ups = np.array([r["p_up"] for r in all_recs], dtype=np.float64)
        actual_ups = np.array([r["actual_up"] for r in all_recs], dtype=np.float64)

        pred_dirs = np.sign(predicted)
        actual_dirs = np.sign(actual)
        nonzero = pred_dirs != 0
        hit_raw = float(np.mean(pred_dirs[nonzero] == actual_dirs[nonzero])) if nonzero.sum() > 0 else 0.5
        brier_raw = float(np.mean((p_ups - actual_ups) ** 2))

        # Calibrated Brier using ALL-regime Beta map
        all_p_map = by_regime.get(REGIME_ALL, {}).get("p_up_map", {"type": "beta", **BETA_IDENTITY})
        calibrated_p = np.array([apply_p_up_map(p, all_p_map) for p in p_ups])
        brier_cal = float(np.mean((calibrated_p - actual_ups) ** 2))

        avg_pred_abs = float(np.mean(np.abs(predicted)))
        avg_actual_abs = float(np.mean(np.abs(actual)))
        mag_ratio_raw = avg_pred_abs / max(avg_actual_abs, 1e-8)

        # EMOS-corrected CRPS (for diagnostics)
        all_emos = by_regime.get(REGIME_ALL, {}).get("emos", {"type": "emos", **EMOS_IDENTITY})
        sigma_all = np.array([r.get("sigma_H", 1.0) for r in all_recs], dtype=np.float64)
        mu_cor_arr = all_emos.get("a", 0.0) + all_emos.get("b", 1.0) * predicted
        sig_cor_arr = np.maximum(EMOS_SIGMA_FLOOR,
                                 all_emos.get("c", 0.0) + all_emos.get("d", 1.0) * sigma_all)
        crps_cal = float(np.mean(_crps_gaussian(mu_cor_arr, sig_cor_arr, actual)))
        crps_raw = float(np.mean(_crps_gaussian(predicted, np.maximum(sigma_all, EMOS_SIGMA_FLOOR), actual)))

        # Backward-compat fields: store ALL-regime calibration at top level
        horizons_cal[str(H)] = {
            # v3.1 regime-conditional storage
            "by_regime": by_regime,
            # Backward-compat: ALL-regime at top level
            "p_up_map": all_p_map,
            "emos": all_emos,
            # Legacy fields (for v2.0 readers)
            "mag_scale": all_emos.get("b", 1.0),  # approximate backward compat
            "bias": all_emos.get("a", 0.0),        # approximate backward compat
            "n_eval": n_eval,
            # Diagnostics
            "hit_rate_raw": float(hit_raw),
            "brier_raw": float(brier_raw),
            "brier_calibrated": float(brier_cal),
            "mag_ratio_raw": float(mag_ratio_raw),
            "crps_raw": float(crps_raw),
            "crps_calibrated": float(crps_cal),
            "regimes_fitted": list(by_regime.keys()),
            # v3.1 diagnostics
            "emos_b_at_bound": abs(all_emos.get("b", 1.0) - (-1.0)) < 0.02,
            "sigma_H_mean": round(float(np.mean(sigma_all)), 4),
            "sigma_H_std": round(float(np.std(sigma_all)), 4),
            "regime_counts": {
                rname: len([r for r in recs if REGIME_BINS.get(rname, (0, 0))[0] <= r.get("vol_regime", 1.0) < REGIME_BINS.get(rname, (0, 0))[1]])
                for rname in REGIME_BINS
            },
            "cv_diagnostics": cv_diagnostics if cv_diagnostics else None,
        }

        # 5) Per-horizon label threshold optimization (from ALL records)
        if n_eval >= MIN_EVAL_POINTS:
            h_label = _optimize_label_thresholds(p_ups, actual_ups, predicted, actual)
            label_by_horizon[str(H)] = h_label
            if H == 7:
                fallback_label_result = h_label

    if not any_valid:
        return None

    # Global label thresholds: prefer 7d, fallback to best available
    if fallback_label_result is None:
        for h_key, h_label in label_by_horizon.items():
            fallback_label_result = h_label
            break
    if fallback_label_result is None:
        fallback_label_result = {
            "buy_thr": 0.58, "sell_thr": 0.42,
            "best_score": 0.0, "label_accuracy_raw": 0.0,
            "label_accuracy_opt": 0.0,
        }

    return {
        "version": CALIBRATION_VERSION,
        "calibrated_at": datetime.now().isoformat(),
        "eval_days": DEFAULT_EVAL_DAYS,
        "eval_spacing": DEFAULT_EVAL_SPACING,
        "horizons": horizons_cal,
        "label_thresholds": fallback_label_result,
        "label_thresholds_by_horizon": label_by_horizon,
    }


# ===========================================================================
# PARALLEL EXECUTION — runs calibration across all assets
# ===========================================================================

def run_signals_calibration(
    cache: Dict[str, Dict],
    workers: int = 8,
    eval_days: int = DEFAULT_EVAL_DAYS,
    eval_spacing: int = DEFAULT_EVAL_SPACING,
    assets: Optional[List[str]] = None,
    quiet: bool = False,
) -> Dict[str, Dict]:
    """
    Run Pass 2 signal calibration for all assets in cache.

    Called from tune_ux.py after Pass 1 (BMA tuning).

    Args:
        cache: Full tune cache dict {symbol: params}
        workers: Number of parallel workers
        eval_days: Walk-forward window (trading days)
        eval_spacing: Days between eval points
        assets: Optional subset of assets to calibrate
        quiet: Suppress output

    Returns:
        Updated cache with "signals_calibration" added to each asset
    """
    # Determine which assets to calibrate
    if assets:
        symbols = [s for s in assets if s in cache]
    else:
        symbols = list(cache.keys())

    if not symbols:
        return cache

    total = len(symbols)
    t0 = time.time()

    if not quiet and console:
        console.print()
        console.print(Panel(
            f"[bold]Pass 2: Signal Calibration[/bold]\n"
            f"Assets: {total} | Eval: {eval_days}d every {eval_spacing}d | "
            f"Horizons: {', '.join(HORIZON_LABELS[h] for h in CALIBRATE_HORIZONS)} | "
            f"Workers: {workers}",
            style="cyan",
            expand=False,
        ))

    calibrated = 0
    failed = 0
    results_summary = []

    if workers <= 1:
        # Sequential
        for i, sym in enumerate(symbols):
            sym_result, cal_dict, error = calibrate_single_asset(
                (sym, eval_days, eval_spacing)
            )
            if cal_dict is not None:
                cache[sym]["signals_calibration"] = cal_dict
                calibrated += 1
                _print_progress(i + 1, total, sym, cal_dict, quiet)
                results_summary.append((sym, cal_dict))
            else:
                failed += 1
                if not quiet and console:
                    console.print(f"  [{i+1}/{total}] {sym}: {error}", style="dim")
    else:
        # Parallel
        ctx = mp.get_context("spawn")
        work_items = [(sym, eval_days, eval_spacing) for sym in symbols]

        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {pool.submit(calibrate_single_asset, w): w[0] for w in work_items}
            done_count = 0

            for fut in as_completed(futures):
                done_count += 1
                try:
                    sym_result, cal_dict, error = fut.result(timeout=300)
                    if cal_dict is not None:
                        cache[sym_result]["signals_calibration"] = cal_dict
                        calibrated += 1
                        _print_progress(done_count, total, sym_result, cal_dict, quiet)
                        results_summary.append((sym_result, cal_dict))
                    else:
                        failed += 1
                        if not quiet and console:
                            console.print(
                                f"  [{done_count}/{total}] {sym_result}: {error}",
                                style="dim",
                            )
                except Exception as e:
                    failed += 1
                    sym_name = futures[fut]
                    if not quiet and console:
                        console.print(
                            f"  [{done_count}/{total}] {sym_name}: {e}",
                            style="dim red",
                        )

    elapsed = time.time() - t0

    # Print summary
    if not quiet and console:
        _print_summary(calibrated, failed, total, elapsed, results_summary)

    return cache


def _print_progress(idx: int, total: int, symbol: str, cal: Dict, quiet: bool):
    """Print single-asset progress line."""
    if quiet or not console:
        return
    h7 = cal.get("horizons", {}).get("7", {})
    hit_raw = h7.get("hit_rate_raw", 0)
    brier_r = h7.get("brier_raw", 0)
    brier_c = h7.get("brier_calibrated", 0)
    crps_r = h7.get("crps_raw", 0)
    crps_c = h7.get("crps_calibrated", 0)
    regimes = h7.get("regimes_fitted", [])
    label_thr = cal.get("label_thresholds", {})
    buy_t = label_thr.get("buy_thr", 0.58)
    sell_t = label_thr.get("sell_thr", 0.42)

    brier_delta = brier_r - brier_c
    brier_sign = "+" if brier_delta > 0 else ""
    crps_delta = crps_r - crps_c
    crps_sign = "+" if crps_delta > 0 else ""

    console.print(
        f"  [{idx}/{total}] {symbol:<12s} "
        f"1w_hit={hit_raw:.0%} "
        f"brier={brier_r:.3f}→{brier_c:.3f}({brier_sign}{brier_delta:.3f}) "
        f"crps={crps_r:.3f}→{crps_c:.3f}({crps_sign}{crps_delta:.3f}) "
        f"regimes={len(regimes)} thr=({buy_t:.2f}/{sell_t:.2f})"
    )


def _print_summary(
    calibrated: int,
    failed: int,
    total: int,
    elapsed: float,
    results: List[Tuple[str, Dict]],
):
    """Print final calibration summary."""
    if not console:
        return

    # Aggregate improvements
    brier_improvements = []
    crps_improvements = []
    for sym, cal in results:
        h7 = cal.get("horizons", {}).get("7", {})
        if "brier_raw" in h7 and "brier_calibrated" in h7:
            brier_improvements.append(h7["brier_raw"] - h7["brier_calibrated"])
        if "crps_raw" in h7 and "crps_calibrated" in h7:
            crps_improvements.append(h7["crps_raw"] - h7["crps_calibrated"])

    avg_brier_imp = float(np.mean(brier_improvements)) if brier_improvements else 0.0
    avg_crps_imp = float(np.mean(crps_improvements)) if crps_improvements else 0.0

    console.print()
    console.print(Panel(
        f"[bold green]Signal Calibration v3.1 Complete[/bold green]\n"
        f"Calibrated: {calibrated}/{total} | Failed: {failed} | "
        f"Elapsed: {elapsed:.1f}s\n"
        f"Avg 1w Brier improvement: {avg_brier_imp:+.4f} | "
        f"Avg 1w CRPS improvement: {avg_crps_imp:+.4f}\n"
        f"Methods: Beta Calibration + EMOS + Soft Regime Blending + CV Guard + Recency Weighting",
        style="green",
        expand=False,
    ))
    console.print()


# ===========================================================================
# STANDALONE CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Signal Calibration Engine — Pass 2 of Two-Pass Tuning",
    )
    parser.add_argument("--assets", type=str, help="Comma-separated asset symbols")
    parser.add_argument("--eval-days", type=int, default=DEFAULT_EVAL_DAYS)
    parser.add_argument("--eval-spacing", type=int, default=DEFAULT_EVAL_SPACING)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: spacing=10 (~50 eval points, 3x faster)")
    args = parser.parse_args()

    if args.fast:
        args.eval_spacing = max(args.eval_spacing, FAST_EVAL_SPACING)

    # Load existing cache
    from tuning.kalman_cache import load_full_cache, save_tuned_params

    cache = load_full_cache()
    if not cache:
        if console:
            console.print("[red]No tuned parameters found. Run 'make tune' first.[/red]")
        sys.exit(1)

    asset_list = None
    if args.assets:
        asset_list = [s.strip() for s in args.assets.split(",")]

    w = 1 if args.no_parallel else args.workers

    cache = run_signals_calibration(
        cache,
        workers=w,
        eval_days=args.eval_days,
        eval_spacing=args.eval_spacing,
        assets=asset_list,
    )

    # Save updated cache
    for sym, params in cache.items():
        if "signals_calibration" in params:
            save_tuned_params(sym, params)

    if console:
        console.print("[green]Calibration saved to tune cache.[/green]")


if __name__ == "__main__":
    main()
