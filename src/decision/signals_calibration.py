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
  cache = run_signals_calibration(cache, workers=0)  # 0=auto, uses all CPUs

  # Standalone:
  python src/decision/signals_calibration.py --assets SPY,QQQ,GLD
  python src/decision/signals_calibration.py --workers 0  # auto-detect CPUs
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
from dataclasses import dataclass, field
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

CALIBRATION_VERSION = "7.5"

# v7.5: Records version — separate from calibration version.
# Only bump when the PREDICTION pipeline changes (compute_features, latest_signals).
# Fitting-only changes (Beta, EMOS, pipeline steps) do NOT require re-collection.
RECORDS_VERSION = "1.0"

# v7.5: Reduced MC paths for calibration context.
# p_up/exp_ret/sigma_H converge fast — 2000 paths is sufficient.
# Production signals use 10000.
CAL_MC_PATHS = 2000

# Horizons to calibrate (days)
CALIBRATE_HORIZONS = [1, 7, 21, 63]
HORIZON_LABELS = {1: "1d", 7: "1w", 21: "1m", 63: "3m"}

# Walk-forward parameters — 500d/3d gives ~166 eval points
# v5.0: spacing 5→3 for 66% more eval points.  With regime
# partitioning this ensures ~40-55 points per regime (was 15-25)
# giving DOF ratio > 5 for the 7-param Beta+EMOS model.
DEFAULT_EVAL_DAYS = 500
DEFAULT_EVAL_SPACING = 3
MIN_EVAL_POINTS = 10
MIN_PRICE_POINTS = 120
FAST_EVAL_SPACING = 5  # --fast mode: ~100 points, 40% faster

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
EMOS_SIGMA_FLOOR = 0.01  # prevents degenerate sigma (in percent-space during calibration)

# v7.1: Realized-vol sigma floor
SIGMA_FLOOR_FRAC = 0.5  # sigma_pred must be >= 50% of realized vol (MAD-based)

# ---------------------------------------------------------------------------
# Regime conditioning
# ---------------------------------------------------------------------------
# vol_regime thresholds (matches assign_regime_labels boundaries)
REGIME_BINS = {
    "LOW":    (0.0, 0.85),
    "NORMAL": (0.85, 1.3),
    "HIGH":   (1.3, 99.0),
}
MIN_REGIME_POINTS = 30  # v5.0: 15→30.  7 params (Beta 3 + EMOS 4) need DOF>4 for robust fitting
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
# Records caching (v4.0 — skip-if-fresh)
# ---------------------------------------------------------------------------
RECORDS_CACHE_DIR = Path(os.path.join(REPO_ROOT, "src", "data", "calibration_records"))
MAX_RECORDS_AGE_DAYS = 7  # v7.5: 3→7 days (records depend on prices, not fitting algorithm)


# ===========================================================================
# PIPELINE DATACLASSES (v4.0)
# ===========================================================================
# Functional pipeline: each step is step(data, prior) → PipelineResult.
# No mutation — every step returns a new result. Metrics are evaluated
# before/after each step so marginal lift is measured independently.
# ===========================================================================

@dataclass
class HorizonData:
    """Extracted numpy arrays for one horizon, ready for fitting.

    This is the immutable input to all pipeline steps.  Constructed once
    from the walk-forward records list and never modified.

    v6.0: Added nu_hat (degrees of freedom) for Student-t CRPS.
    """
    H: int
    n_eval: int
    predicted: np.ndarray      # predicted returns (%)
    actual: np.ndarray         # actual returns (%)
    p_ups: np.ndarray          # raw p_up ∈ [0,1]
    actual_ups: np.ndarray     # binary actual outcome
    sigma_pred: np.ndarray     # predicted sigma_H (%)
    vol_regime: np.ndarray     # per-record vol_regime scalar
    weights: np.ndarray        # recency weights
    nu_hat: np.ndarray = field(default_factory=lambda: np.array([]))  # v6.0: BMA-averaged ν


@dataclass
class StepResult:
    """Outcome of a single pipeline step."""
    step_name: str
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    was_reverted: bool = False
    detail: Optional[Dict[str, Any]] = None


@dataclass
class PipelineResult:
    """Accumulated state flowing through the pipeline.

    Each step reads the current p_up_map/emos and returns a NEW
    PipelineResult with updated values + one more StepResult appended.
    """
    # Current calibration params (evolve step by step)
    by_regime: Dict[str, Dict] = field(default_factory=dict)
    label_thresholds: Optional[Dict[str, Any]] = None
    # Pipeline trace (append-only)
    steps: List[StepResult] = field(default_factory=list)


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
            # v4.0 pipeline kernels
            apply_beta_map_batch_nb,
            crps_gaussian_nb,
            crps_gaussian_mean_nb,
            beta_nll_objective_nb,
            emos_crps_objective_nb,
            evaluate_metrics_nb,
            # v6.0 pipeline kernels
            crps_student_t_nb,
            crps_student_t_mean_nb,
            emos_crps_student_t_objective_nb,
            beta_focal_nll_objective_nb,
            temperature_scaling_nll_nb,
            apply_isotonic_beta_blend_nb,
            brier_decomposition_nb,
            expanding_cv_fold_indices_nb,
            # v7.0 two-stage EMOS kernels
            emos_crps_mean_only_nb,
            emos_crps_scale_only_nb,
            # v7.1 CRPS fix kernels
            compute_realized_vol_floor_nb,
            emos_crps_mean_with_grad_nb,
            # v7.2 deep CRPS kernels
            emos_crps_scale_v72_nb,
            emos_crps_joint_v72_nb,
            # v7.3 multi-diagnostic calibration kernels
            compute_pit_values_nb,
            pit_ks_test_nb,
            pit_ad_test_nb,
            hyvarinen_score_nb,
            berkowitz_test_nb,
            mad_score_nb,
            log_score_nb,
            dss_score_nb,
            emos_crps_pit_v73_nb,
            # v7.4 Numba-native optimizers
            _emos_stage1_optimize_nb,
            _beta_cal_optimize_nb,
        )
        return {
            "isotonic_regression": isotonic_regression_nb,
            "magnitude_scale": compute_magnitude_scale_nb,
            "bias_correction": compute_bias_correction_nb,
            "hit_rates": compute_hit_rates_nb,
            "brier_score": compute_brier_score_nb,
            "grid_search_thresholds": grid_search_thresholds_nb,
            "apply_isotonic_map": apply_isotonic_map_nb,
            # v4.0 pipeline kernels
            "beta_batch": apply_beta_map_batch_nb,
            "crps_gaussian": crps_gaussian_nb,
            "crps_gaussian_mean": crps_gaussian_mean_nb,
            "beta_nll": beta_nll_objective_nb,
            "emos_crps": emos_crps_objective_nb,
            "evaluate_metrics": evaluate_metrics_nb,
            # v6.0 pipeline kernels
            "crps_student_t": crps_student_t_nb,
            "crps_student_t_mean": crps_student_t_mean_nb,
            "emos_crps_student_t": emos_crps_student_t_objective_nb,
            "beta_focal_nll": beta_focal_nll_objective_nb,
            "temp_scaling_nll": temperature_scaling_nll_nb,
            "isotonic_beta_blend": apply_isotonic_beta_blend_nb,
            "brier_decomposition": brier_decomposition_nb,
            "cv_fold_indices": expanding_cv_fold_indices_nb,
            # v7.0 two-stage EMOS kernels
            "emos_mean_only": emos_crps_mean_only_nb,
            "emos_scale_only": emos_crps_scale_only_nb,
            # v7.1 CRPS fix kernels
            "realized_vol_floor": compute_realized_vol_floor_nb,
            "emos_mean_grad": emos_crps_mean_with_grad_nb,
            # v7.2 deep CRPS kernels
            "emos_scale_v72": emos_crps_scale_v72_nb,
            "emos_joint_v72": emos_crps_joint_v72_nb,
            # v7.3 multi-diagnostic kernels
            "pit_values": compute_pit_values_nb,
            "pit_ks": pit_ks_test_nb,
            "pit_ad": pit_ad_test_nb,
            "hyvarinen": hyvarinen_score_nb,
            "berkowitz": berkowitz_test_nb,
            "mad": mad_score_nb,
            "log_score": log_score_nb,
            "dss_score": dss_score_nb,
            "emos_pit_v73": emos_crps_pit_v73_nb,
            # v7.4 Numba-native optimizers
            "emos_stage1_opt": _emos_stage1_optimize_nb,
            "beta_cal_opt": _beta_cal_optimize_nb,
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

    v5.0 FIX: CI-width is now PRIMARY source (always stable).
    Score-based derivation requires |score| > 0.5 (was 1e-6)
    because when score ≈ 0.007, sigma_H = |exp_ret / 0.007|
    explodes to 100+, making EMOS 'd' parameter unlearnable.

    v3.1 had score-based as primary with |score| > 1e-6 threshold,
    producing sigma_H range of 0.065 to 102.8 — a 1500x variance
    that destroyed EMOS fitting stability.

    Fallback chain:
      1. (ci_high - ci_low) / 2  (stable, always available)
      2. exp_ret / score  (exact, when |score| > 0.5 only)
      3. vol_mean  (legacy, better than nothing)

    Returns:
        sigma_H in percent units (multiplied by 100)
    """
    # Primary: CI width (always stable, no division-by-near-zero risk)
    ci_low = getattr(sig, "ci_low", None)
    ci_high = getattr(sig, "ci_high", None)
    if ci_low is not None and ci_high is not None:
        ci_width = ci_high - ci_low
        if ci_width > 1e-8:
            sH = ci_width / 2.0
            if 1e-8 < sH < 1.0:  # sanity: sH in log-return units
                return sH * 100.0

    # Secondary: exact derivation from score = mu_H / sH
    # v5.0: require |score| > 0.5 (was 1e-6 — blows up near zero)
    score = getattr(sig, "score", 0.0)
    exp_ret = getattr(sig, "exp_ret", 0.0)
    if abs(score) > 0.5:
        sH = abs(exp_ret / score)
        if 1e-8 < sH < 1.0:  # tightened from 10.0
            return sH * 100.0

    # Last resort: vol_mean (legacy, known to be wrong scale but
    # better than a constant)
    vol_mean = getattr(sig, "vol_mean", 0.0)
    if vol_mean > 1e-8:
        return vol_mean * 100.0

    # Absolute fallback: 1% sigma (conservative)
    return 1.0


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

    FOCAL_GAMMA = 2.0
    nb = _nb()

    # v7.4: Use Numba-native optimizer (eliminates Python↔Numba boundary)
    if nb is not None and "beta_cal_opt" in nb:
        try:
            a_fit, b_fit, c_fit = nb["beta_cal_opt"](
                ln_p.astype(np.float64),
                ln_1mp.astype(np.float64),
                y,
                w.astype(np.float64),
                FOCAL_GAMMA,
                0.01,   # reg_strength
                100,     # max_iter
            )
        except Exception:
            a_fit, b_fit, c_fit = 1.0, 1.0, 0.0
    elif nb is not None and "beta_focal_nll" in nb:
        _beta_focal_nb = nb["beta_focal_nll"]

        def _weighted_nll(params):
            a, b, c = params
            return _beta_focal_nb(a, b, c, ln_p, ln_1mp, y, w, 0.01, FOCAL_GAMMA)

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
    else:
        def _weighted_nll(params):
            a, b, c = params
            z = a * ln_p - b * ln_1mp + c
            p_cal = np.where(z >= 0,
                             1.0 / (1.0 + np.exp(-z)),
                             np.exp(z) / (1.0 + np.exp(z)))
            p_cal = np.clip(p_cal, 1e-10, 1.0 - 1e-10)
            p_t = np.where(y == 1, p_cal, 1.0 - p_cal)
            focal_w = (1.0 - p_t) ** FOCAL_GAMMA
            bce = -(y * np.log(p_cal) + (1.0 - y) * np.log(1.0 - p_cal))
            nll = np.sum(w * focal_w * bce) / np.sum(w)
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

    v4.0: Uses Numba kernel when available (no scipy dependency, ~20x faster).
    Falls back to scipy.stats.norm for compatibility.

    Args:
        mu:    Predicted means (array)
        sigma: Predicted std devs (array, must be > 0)
        y:     Observed values (array)

    Returns:
        Per-sample CRPS values (array, lower = better)
    """
    nb = _nb()
    if nb is not None and "crps_gaussian" in nb:
        return nb["crps_gaussian"](
            mu.astype(np.float64),
            np.maximum(sigma.astype(np.float64), 1e-10),
            y.astype(np.float64),
        )
    # Fallback: scipy
    from scipy.stats import norm
    z = (y - mu) / np.maximum(sigma, 1e-10)
    crps = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / math.sqrt(math.pi))
    return crps


def _crps_student_t(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    nu: np.ndarray,
) -> np.ndarray:
    """CRPS for Student-t predictive distribution (v6.0).

    Uses Numba kernel (numerical Gini half-mean-difference).
    Falls back to Gaussian CRPS if kernel unavailable.

    Args:
        mu:    Predicted means (array)
        sigma: Predicted std devs (array, must be > 0)
        y:     Observed values (array)
        nu:    Degrees of freedom (array)

    Returns:
        Per-sample CRPS values (array, lower = better)
    """
    nb = _nb()
    if nb is not None and "crps_student_t" in nb:
        return nb["crps_student_t"](
            mu.astype(np.float64),
            np.maximum(sigma.astype(np.float64), 1e-10),
            y.astype(np.float64),
            np.maximum(nu.astype(np.float64), 2.5),
        )
    # Fallback: Gaussian CRPS (ν→∞ limit)
    return _crps_gaussian(mu, sigma, y)


def _fit_emos_student_t(
    predicted: np.ndarray,
    actual: np.ndarray,
    sigma_pred: np.ndarray,
    nu_hat: np.ndarray,
    n_eval: int,
    weights: Optional[np.ndarray] = None,
    skip_stage3: bool = False,
) -> Dict[str, Any]:
    """Fit 5-parameter Student-t EMOS via two-stage optimization (v7.0).

    v7.4: Added ``skip_stage3`` flag. Stage 3 (5-param joint polish with
    PIT/CvM penalty) is the most expensive stage (~8-12ms) and provides
    diminishing returns on sub-regimes with < 80 data points.

    v7.0: Two-stage optimization fixes the root cause of EMOS σ-inflation.
    The old joint optimizer preferred inflating σ (via c) over scaling
    the mean (via b) because wider σ reduces CRPS more cheaply when
    predictions are systematically too small (mag_ratio ≈ 0.15).

    Stage 1 — Mean Correction (a, b only):
        Fix c=0, d=1, ν=ν_prior (identity for scale).
        Optimize a, b to minimize CRPS with NO regularization on b.
        This FORCES the optimizer to rescale predictions before touching σ.

    Stage 2 — Scale Correction (c, d, ν only):
        Fix a, b from Stage 1.
        Optimize c, d, ν with standard regularization.

    Args:
        predicted:  Predicted returns (log %)
        actual:     Actual returns (log %)
        sigma_pred: Predicted standard deviations (log %)
        nu_hat:     BMA-averaged ν per eval point (prior for regularization)
        n_eval:     Total evaluation points (for shrinkage)
        weights:    Optional per-sample weights (recency)

    Returns:
        {"type": "emos", "a": float, "b": float, "c": float, "d": float, "nu": float}
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

    avg_actual_abs = float(np.mean(np.abs(y)))
    # Prior ν from BMA: regularize toward median of nu_hat
    nu_prior = float(np.median(nu_hat)) if len(nu_hat) > 0 else 30.0
    nu_prior = max(3.0, min(50.0, nu_prior))

    nb = _nb()

    # ========================================================================
    # STAGE 1: Mean correction (a, b) — fix scale at identity
    # ========================================================================
    # v7.4: Use Numba-native optimizer (eliminates Python↔Numba boundary)
    # Fallback: scipy L-BFGS-B with analytical gradient
    # ========================================================================

    # Initial guess: mag_ratio as starting point for b
    med_pred = float(np.median(np.abs(mu_pred))) if len(mu_pred) > 0 else 1e-8
    med_actual = float(np.median(np.abs(y))) if len(y) > 0 else 1e-8
    b_init = min(15.0, max(0.5, med_actual / max(med_pred, 1e-8)))

    if nb is not None and "emos_stage1_opt" in nb:
        # v7.4: Fully Numba-native Stage 1 — no scipy overhead
        try:
            a_fit, b_fit = nb["emos_stage1_opt"](
                mu_pred, sig_pred, y, w,
                EMOS_SIGMA_FLOOR, nu_prior,
                0.0, b_init,     # a_init, b_init
                0.1, 15.0,       # b_lo, b_hi
                100,             # max_iter
            )
        except Exception:
            a_fit, b_fit = 0.0, 1.0
    else:
        # Fallback: scipy L-BFGS-B
        _use_grad = (nb is not None and "emos_mean_grad" in nb)

        if _use_grad:
            _mean_grad_nb = nb["emos_mean_grad"]

            def _stage1_obj_grad(params):
                a, b = params
                loss, ga, gb = _mean_grad_nb(
                    a, b, mu_pred, sig_pred, y, w,
                    EMOS_SIGMA_FLOOR, nu_prior,
                )
                return loss, np.array([ga, gb])
        elif nb is not None and "emos_mean_only" in nb:
            _mean_only_nb = nb["emos_mean_only"]

            def _stage1_obj_grad(params):
                a, b = params
                return _mean_only_nb(
                    a, b, mu_pred, sig_pred, y, w,
                    EMOS_SIGMA_FLOOR, nu_prior,
                )
            _use_grad = False
        else:
            def _stage1_obj_grad(params):
                a, b = params
                mu_cor = a + b * mu_pred
                sig_cor = np.maximum(EMOS_SIGMA_FLOOR, sig_pred)
                crps_vals = _crps_gaussian(mu_cor, sig_cor, y)
                loss = float(np.sum(w * crps_vals) / np.sum(w))
                reg = 0.001 * (a ** 2)
                return loss + reg
            _use_grad = False

        try:
            result1 = sp_minimize(
                _stage1_obj_grad,
                x0=[0.0, b_init],
                method="L-BFGS-B",
                jac=_use_grad,
                bounds=[(-10.0, 10.0), (0.1, 15.0)],
            )
            a_fit, b_fit = result1.x
        except Exception:
            a_fit, b_fit = 0.0, 1.0

    # ========================================================================
    # STAGE 2: Scale correction (c, d, ν) — v7.2 improvements
    # ========================================================================
    # v7.2 key improvements:
    # 1. DSS-optimal d_init (warm-start from variance matching)
    # 2. Adaptive regularization (weaker for large n — data speaks louder)
    # 3. DSS variance penalty (forces E[z²] ≈ ν/(ν-2) for calibration)
    # 4. Wider bounds: d ∈ [0.01, 10], ν ∈ [2.2, 200]
    # ========================================================================

    # v7.2: DSS-optimal d initialization
    # σ_DSS = RMSE(residuals after mean correction) → d_DSS = σ_DSS / mean(σ_pred)
    residuals_s1 = y - (a_fit + b_fit * mu_pred)
    rmse_residual = float(np.sqrt(np.sum(w * residuals_s1**2) / np.sum(w)))
    mean_sigma = float(np.mean(sig_pred))
    d_init = max(0.1, min(10.0, rmse_residual / max(mean_sigma, 1e-8)))

    # v7.2: Adaptive regularization — scale with min(50,n)/n
    # At n=167: reg_weight ≈ 0.003 (3x weaker, data speaks)
    # At n=50:  reg_weight = 0.01 (standard)
    # At n=20:  reg_weight = 0.01 (conservative)
    reg_weight = max(0.001, 0.01 * min(50.0, float(n_eval)) / max(float(n_eval), 1.0))

    if nb is not None and "emos_scale_v72" in nb:
        _scale_v72_nb = nb["emos_scale_v72"]

        def _stage2_obj(params):
            c, d, log_nu = params
            nu = math.exp(log_nu)
            return _scale_v72_nb(
                c, d, nu, a_fit, b_fit,
                mu_pred, sig_pred, y, w,
                EMOS_SIGMA_FLOOR, nu_prior, reg_weight,
            )
    elif nb is not None and "emos_scale_only" in nb:
        _scale_only_nb = nb["emos_scale_only"]

        def _stage2_obj(params):
            c, d, log_nu = params
            nu = math.exp(log_nu)
            return _scale_only_nb(
                c, d, nu, a_fit, b_fit,
                mu_pred, sig_pred, y, w,
                EMOS_SIGMA_FLOOR, nu_prior,
            )
    else:
        # Fallback: Python CRPS with fixed mean
        def _stage2_obj(params):
            c, d, _log_nu = params
            mu_cor = a_fit + b_fit * mu_pred
            sig_cor = np.maximum(EMOS_SIGMA_FLOOR, c + d * sig_pred)
            crps_vals = _crps_gaussian(mu_cor, sig_cor, y)
            loss = float(np.sum(w * crps_vals) / np.sum(w))
            reg = reg_weight * (c ** 2 + (d - 1.0) ** 2)
            return loss + reg

    log_nu_init = math.log(max(3.0, nu_prior))

    try:
        result2 = sp_minimize(
            _stage2_obj,
            x0=[0.0, d_init, log_nu_init],  # v7.2: DSS d_init
            method="L-BFGS-B",
            bounds=[
                (-5.0, 5.0),                       # c
                (0.01, 10.0),                        # d — v7.2: wider
                (math.log(2.2), math.log(200.0)),    # log(ν) — v7.2: wider
            ],
        )
        c_fit, d_fit, log_nu_fit = result2.x
        nu_fit = math.exp(log_nu_fit)
    except Exception:
        c_fit, d_fit, nu_fit = 0.0, d_init, nu_prior

    # ========================================================================
    # STAGE 3: Joint polish (v7.3) — PIT-aware composite objective
    # ========================================================================
    # v7.2 polished using CRPS + DSS variance penalty.
    # v7.3 adds Cramér-von Mises (CvM) penalty on PIT uniformity.
    # v7.4: Skip for sub-regimes with few data points (diminishing returns).
    # ========================================================================
    _PIT_PENALTY_WEIGHT = 0.10  # CvM weight in composite objective

    if skip_stage3:
        # v7.4: Skip Stage 3 for sub-regimes — Stages 1+2 are sufficient
        pass
    elif n_eval >= 30 and nb is not None and "emos_pit_v73" in nb:
        _pit_v73_nb = nb["emos_pit_v73"]

        def _stage3_obj(params):
            a, b, c, d, log_nu = params
            nu = math.exp(log_nu)
            return _pit_v73_nb(
                a, b, c, d, nu,
                mu_pred, sig_pred, y, w,
                EMOS_SIGMA_FLOOR, nu_prior, reg_weight * 0.5,
                _PIT_PENALTY_WEIGHT,
            )

        # Tight bounds around two-stage solution (±exploration range)
        b_lo = max(0.1, b_fit * 0.5)
        b_hi = min(15.0, b_fit * 2.0)
        d_lo = max(0.01, d_fit * 0.5)
        d_hi = min(10.0, d_fit * 2.0)
        nu_lo = math.log(max(2.2, nu_fit * 0.5))
        nu_hi = math.log(min(200.0, nu_fit * 2.0))

        try:
            result3 = sp_minimize(
                _stage3_obj,
                x0=[a_fit, b_fit, c_fit, d_fit, math.log(nu_fit)],
                method="L-BFGS-B",
                bounds=[
                    (a_fit - 2.0, a_fit + 2.0),   # a: ±2 around Stage 1
                    (b_lo, b_hi),                   # b: ×[0.5, 2] around Stage 1
                    (c_fit - 2.0, c_fit + 2.0),   # c: ±2 around Stage 2
                    (d_lo, d_hi),                   # d: ×[0.5, 2] around Stage 2
                    (nu_lo, nu_hi),                 # ν: ×[0.5, 2] around Stage 2
                ],
                options={"maxiter": 75},  # v7.3: more iterations for PIT convergence
            )
            a_fit, b_fit = result3.x[0], result3.x[1]
            c_fit, d_fit = result3.x[2], result3.x[3]
            nu_fit = math.exp(result3.x[4])
        except Exception:
            pass  # Keep two-stage solution if polish fails

    elif n_eval >= 30 and nb is not None and "emos_joint_v72" in nb:
        # Fallback to v7.2 joint polish (without PIT penalty)
        _joint_v72_nb = nb["emos_joint_v72"]

        def _stage3_obj_v72(params):
            a, b, c, d, log_nu = params
            nu = math.exp(log_nu)
            return _joint_v72_nb(
                a, b, c, d, nu,
                mu_pred, sig_pred, y, w,
                EMOS_SIGMA_FLOOR, nu_prior, reg_weight * 0.5,
            )

        b_lo = max(0.1, b_fit * 0.5)
        b_hi = min(15.0, b_fit * 2.0)
        d_lo = max(0.01, d_fit * 0.5)
        d_hi = min(10.0, d_fit * 2.0)
        nu_lo = math.log(max(2.2, nu_fit * 0.5))
        nu_hi = math.log(min(200.0, nu_fit * 2.0))

        try:
            result3 = sp_minimize(
                _stage3_obj_v72,
                x0=[a_fit, b_fit, c_fit, d_fit, math.log(nu_fit)],
                method="L-BFGS-B",
                bounds=[
                    (a_fit - 2.0, a_fit + 2.0),
                    (b_lo, b_hi),
                    (c_fit - 2.0, c_fit + 2.0),
                    (d_lo, d_hi),
                    (nu_lo, nu_hi),
                ],
                options={"maxiter": 50},
            )
            a_fit, b_fit = result3.x[0], result3.x[1]
            c_fit, d_fit = result3.x[2], result3.x[3]
            nu_fit = math.exp(result3.x[4])
        except Exception:
            pass

    # Shrinkage toward identity
    lam = min(1.0, n_eval / SHRINKAGE_FULL_N)
    a_s = lam * a_fit
    b_s = lam * b_fit + (1.0 - lam) * 1.0
    c_s = lam * c_fit
    d_s = lam * d_fit + (1.0 - lam) * 1.0
    # ν shrinkage toward prior
    nu_s = lam * nu_fit + (1.0 - lam) * nu_prior

    return {
        "type": "emos",
        "a": round(float(a_s), 6),
        "b": round(float(b_s), 6),
        "c": round(float(c_s), 6),
        "d": round(float(d_s), 6),
        "nu": round(float(nu_s), 4),
    }


def _fit_temperature_scaling(
    p_ups: np.ndarray,
    actual_ups: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Fit single-parameter temperature scaling (Guo et al. 2017).

    Model: p_cal = sigmoid(logit(p_raw) / T)
    T > 1 → soften (reduce overconfidence)
    T < 1 → sharpen

    Most robust post-hoc calibration method. Single parameter = minimal
    overfitting risk.

    Args:
        p_ups:       Raw p_up values ∈ [0, 1]
        actual_ups:  Binary outcomes
        weights:     Optional recency weights

    Returns:
        Temperature T (float). T=1.0 means no change.
    """
    from scipy.optimize import minimize_scalar

    if len(p_ups) < BETA_MIN_POINTS:
        return 1.0

    p_clipped = np.clip(p_ups, 0.001, 0.999)
    logits = np.log(p_clipped / (1.0 - p_clipped))
    y = actual_ups.astype(np.float64)

    if weights is None:
        w = np.ones(len(y), dtype=np.float64)
    else:
        w = weights / weights.sum() * len(y)

    nb = _nb()
    if nb is not None and "temp_scaling_nll" in nb:
        _temp_nll_nb = nb["temp_scaling_nll"]

        def _nll(T):
            return _temp_nll_nb(T, logits, y, w)
    else:
        def _nll(T):
            scaled = logits / max(T, 0.01)
            p_cal = np.where(scaled >= 0,
                             1.0 / (1.0 + np.exp(-scaled)),
                             np.exp(scaled) / (1.0 + np.exp(scaled)))
            p_cal = np.clip(p_cal, 1e-10, 1.0 - 1e-10)
            bce = -(y * np.log(p_cal) + (1.0 - y) * np.log(1.0 - p_cal))
            return float(np.sum(w * bce) / np.sum(w))

    try:
        result = minimize_scalar(_nll, bounds=(0.1, 5.0), method="bounded")
        T = float(result.x)
    except Exception:
        T = 1.0

    return max(0.1, min(5.0, T))


def _apply_temperature(p_raw: float, T: float) -> float:
    """Apply temperature scaling to a single probability.

    Args:
        p_raw: Raw probability ∈ [0, 1]
        T:     Temperature (T=1 → identity)

    Returns:
        Temperature-scaled probability
    """
    p_clipped = max(0.001, min(0.999, p_raw))
    logit = math.log(p_clipped / (1.0 - p_clipped))
    z = logit / max(T, 0.01)
    if z >= 0:
        result = 1.0 / (1.0 + math.exp(-z))
    else:
        ez = math.exp(z)
        result = ez / (1.0 + ez)
    return max(0.0, min(1.0, result))


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

    # v6.0: Use Numba-accelerated inner loop when available (~5x faster)
    # v6.0: Stronger magnitude penalty (0.15 vs 0.05) pushes b properly
    avg_actual_abs = float(np.mean(np.abs(y)))
    MAG_PENALTY_WEIGHT = 0.15  # v6.0: was 0.05 in v5.0
    nb = _nb()
    if nb is not None and "emos_crps" in nb:
        _emos_crps_nb = nb["emos_crps"]

        def _weighted_crps(params):
            a, b, c, d = params
            return _emos_crps_nb(a, b, c, d, mu_pred, sig_pred, y, w,
                                 EMOS_SIGMA_FLOOR, 0.01, avg_actual_abs)
    else:
        def _weighted_crps(params):
            a, b, c, d = params
            mu_cor = a + b * mu_pred
            sig_cor = np.maximum(EMOS_SIGMA_FLOOR, c + d * sig_pred)
            crps_vals = _crps_gaussian(mu_cor, sig_cor, y)
            loss = np.sum(w * crps_vals) / np.sum(w)
            reg = 0.01 * (a ** 2 + (b - 1.0) ** 2 + c ** 2 + (d - 1.0) ** 2)
            # v6.0: Stronger magnitude penalty (0.15)
            avg_pred_abs = float(np.mean(np.abs(mu_cor)))
            mag_ratio = avg_pred_abs / max(avg_actual_abs, 1e-8)
            mag_penalty = MAG_PENALTY_WEIGHT * (mag_ratio - 1.0) ** 2
            return loss + reg + mag_penalty

    try:
        result = sp_minimize(
            _weighted_crps,
            x0=[0.0, 1.0, 0.0, 1.0],
            method="L-BFGS-B",
            # v6.0: b upper bound 5.0→15.0 to allow full magnitude
            # correction.  v5.0's cap at 5.0 prevented proper scaling
            # when predictions are 10x too small.
            bounds=[(-10.0, 10.0), (-1.0, 15.0), (-5.0, 5.0), (0.01, 5.0)],
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
    p_up_map: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Grid search for optimal buy/sell thresholds maximizing composite metric.

    Optimizes: hit_rate (0.4) + inv_brier (0.3) + label_accuracy (0.3)

    v6.0: Accepts optional p_up_map so Brier is computed on calibrated
    probabilities instead of raw p_ups.

    Args:
        p_ups: Raw p_up values ∈ [0, 1]
        actual_ups: Binary outcomes (1 if return > 0)
        exp_rets: Predicted returns (%)
        actual_rets: Actual returns (%)
        p_up_map: Optional Beta calibration map for Brier computation

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
                p_ups, actual_ups, exp_rets, actual_rets, buy_thr, sell_thr,
                p_up_map=p_up_map,
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
    p_up_map: Optional[Dict] = None,
) -> float:
    """Evaluate a buy/sell threshold pair. Returns composite score.

    v6.0: Brier computed on CALIBRATED p_ups (via p_up_map) instead of raw.
    The raw Brier was measuring model miscalibration, not threshold quality.
    """
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

    # v6.0: Brier on calibrated p_ups (not raw)
    if p_up_map is not None:
        nb = _nb()
        if (nb is not None and "beta_batch" in nb
                and p_up_map.get("type") == "beta"):
            cal_p = nb["beta_batch"](
                p_ups,
                p_up_map.get("a", 1.0), p_up_map.get("b", 1.0),
                p_up_map.get("c", 0.0),
                p_up_map.get("clip_lo", 0.01), p_up_map.get("clip_hi", 0.99),
            )
        else:
            cal_p = np.array([apply_p_up_map(p, p_up_map) for p in p_ups])
    else:
        cal_p = p_ups
    brier = float(np.mean((cal_p - actual_ups) ** 2))
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
# RECORDS CACHING (v4.0 — skip-if-fresh)
# ===========================================================================
# Walk-forward signal generation is 99% of calibration runtime (~40s/asset).
# Fitting is <1% (<50ms).  By caching records, re-calibration with
# algorithm changes completes in <1s/asset instead of ~40s.
# ===========================================================================

def _records_cache_path(symbol: str) -> Path:
    """Return cache file path for a symbol's walk-forward records."""
    safe = symbol.replace("/", "_").replace("=", "_").replace(":", "_").upper()
    return RECORDS_CACHE_DIR / f"{safe}_records.json"


def _save_records(symbol: str, records: Dict[int, List[Dict]]) -> None:
    """Save walk-forward records to disk cache."""
    RECORDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _records_cache_path(symbol)
    # Convert int keys to strings for JSON
    serializable = {
        "saved_at": datetime.now().isoformat(),
        "eval_spacing": DEFAULT_EVAL_SPACING,  # v5.0: track spacing for invalidation
        "records_version": RECORDS_VERSION,  # v7.5: separate from calibration_version
        "horizons": {str(k): v for k, v in records.items()},
    }
    try:
        with open(path, "w") as f:
            json.dump(serializable, f)
    except Exception:
        pass  # Non-critical — silently skip if write fails


def _load_records(symbol: str, max_age_days: int = MAX_RECORDS_AGE_DAYS) -> Optional[Dict[int, List[Dict]]]:
    """Load cached records if they exist and are fresh enough.

    v5.0: Also invalidates cache if eval_spacing has changed.
    v7.5: Uses records_version (not calibration_version) so fitting-only
    changes don't force expensive re-collection.

    Args:
        symbol:       Asset symbol
        max_age_days: Maximum age before records are considered stale

    Returns:
        Records dict {horizon_int: [record_dicts]} or None if stale/missing
    """
    path = _records_cache_path(symbol)
    if not path.exists():
        return None

    try:
        # Check freshness
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age_days = (datetime.now() - mtime).total_seconds() / 86400
        if age_days > max_age_days:
            return None

        with open(path, "r") as f:
            data = json.load(f)

        # v5.0: Invalidate if eval_spacing changed (e.g., 5→3)
        cached_spacing = data.get("eval_spacing")
        if cached_spacing is not None and cached_spacing != DEFAULT_EVAL_SPACING:
            return None

        # v7.5: Invalidate if records_version changed (separate from calibration_version).
        # Old records without records_version key are accepted (backward compat).
        cached_records_ver = data.get("records_version")
        if cached_records_ver is not None and cached_records_ver != RECORDS_VERSION:
            return None

        # Convert string keys back to ints
        horizons = data.get("horizons", {})
        return {int(k): v for k, v in horizons.items()}
    except Exception:
        return None


# ===========================================================================
# v7.5: Feature precomputation for walk-forward speedup
# ===========================================================================

def _slice_features_at(feats_full: Dict, idx: int) -> Dict:
    """Create a feature dict truncated at index ``idx``.

    Used by the precomputation path: ``compute_features`` runs ONCE on the
    full price series, then this function slices the result for each eval
    point so that ``latest_signals`` sees only data up to ``idx``.

    * pd.Series / pd.DataFrame values are sliced to ``.iloc[:idx]``.
    * The ``hmm_result`` dict is shallow-copied with internal time-indexed
      data (regime_series, posteriors) sliced to ``[:idx]``.
    * ``nu_hat`` (single-element Series at the end of the series) is passed
      through unchanged to avoid empty-Series issues.
    * Scalar / dict values are passed through unchanged.
    """
    sliced = {}
    for k, v in feats_full.items():
        if k == "hmm_result" and isinstance(v, dict):
            hmm_sliced = dict(v)
            for hk, hv in v.items():
                if isinstance(hv, (pd.Series, pd.DataFrame)):
                    hmm_sliced[hk] = hv.iloc[:idx]
            sliced[k] = hmm_sliced
        elif k == "nu_hat":
            sliced[k] = v  # 1-element Series at end — pass through
        elif isinstance(v, pd.Series):
            sliced[k] = v.iloc[:idx]
        elif isinstance(v, pd.DataFrame):
            sliced[k] = v.iloc[:idx]
        else:
            sliced[k] = v
    return sliced


# ===========================================================================
# CORE: Calibrate a single asset (worker function)
# ===========================================================================

def calibrate_single_asset(
    args_tuple: Tuple,
) -> Tuple[str, Optional[Dict], Optional[str]]:
    """
    Walk-forward calibration for one asset.

    v4.0: Supports records caching for skip-if-fresh semantics.
    Walk-forward signal generation is 99% of runtime (~40s/asset).
    Cached records allow re-calibration in <1s/asset.

    Runs compute_features() → latest_signals() at each eval point,
    collects (p_up, exp_ret, label) vs realized returns, then computes
    all correction types via the pipeline.

    Args:
        args_tuple: (symbol, eval_days, eval_spacing) or
                    (symbol, eval_days, eval_spacing, force_collect) or
                    (symbol, eval_days, eval_spacing, force_collect, use_precompute)

    Returns:
        (symbol, calibration_dict_or_None, error_or_None)
    """
    if len(args_tuple) >= 5:
        symbol, eval_days, eval_spacing, force_collect, use_precompute = args_tuple[:5]
    elif len(args_tuple) >= 4:
        symbol, eval_days, eval_spacing, force_collect = args_tuple[:4]
        use_precompute = True
    else:
        symbol, eval_days, eval_spacing = args_tuple
        force_collect = False
        use_precompute = True

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

        # v4.0: Check records cache (skip-if-fresh)
        cached_records = None if force_collect else _load_records(symbol)

        if cached_records is not None:
            # Cache hit — skip the expensive walk-forward loop
            records = cached_records
        else:
            # Cache miss or forced — run walk-forward collection
            # Collect records per horizon
            records: Dict[int, List[Dict]] = {h: [] for h in CALIBRATE_HORIZONS}

            # v7.5: Feature precomputation — compute once on full series,
            # slice per eval point.  Eliminates ~166 redundant calls to
            # compute_features (Kalman filter, HMM, EWMA, HAR-GK, etc.).
            feats_full = None
            if use_precompute:
                try:
                    with contextlib.redirect_stdout(io.StringIO()), \
                         contextlib.redirect_stderr(io.StringIO()):
                        feats_full = compute_features(px, asset_symbol=symbol, ohlc_df=ohlc_df)
                except Exception:
                    feats_full = None  # Fall back to per-truncation

            idx = start_idx
            while idx < n:
                if len(px.iloc[:idx]) < 60:
                    idx += eval_spacing
                    continue

                last_close = float(px.iloc[idx - 1])

                try:
                    with contextlib.redirect_stdout(io.StringIO()), \
                         contextlib.redirect_stderr(io.StringIO()):
                        if feats_full is not None:
                            # v7.5: Slice precomputed features at eval point
                            feats = _slice_features_at(feats_full, idx)
                        else:
                            # Per-truncation fallback (strict walk-forward)
                            px_trunc = px.iloc[:idx]
                            ohlc_trunc = ohlc_df.iloc[:idx]
                            feats = compute_features(px_trunc, asset_symbol=symbol, ohlc_df=ohlc_trunc)
                        sigs, _ = latest_signals(
                            feats,
                            horizons=CALIBRATE_HORIZONS,  # Only compute horizons we calibrate (4 vs 7 = 43% faster)
                            last_close=last_close,
                            t_map=True,
                            ci=0.68,
                            tuned_params=tuned_params,
                            asset_key=symbol,
                            _calibration_fast_mode=True,  # v7.5: multi-horizon MC + caching
                            n_mc_paths=CAL_MC_PATHS,       # v7.5: 2000 vs 10000 paths
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

                    # v6.0: extract BMA-averaged ν for Student-t CRPS
                    _nu_hat_val = 30.0  # default near-Gaussian
                    if isinstance(feats, dict):
                        _nhs = feats.get("nu_hat")
                        if _nhs is not None:
                            try:
                                if hasattr(_nhs, 'iloc') and len(_nhs) > 0:
                                    _nu_hat_val = float(_nhs.iloc[-1])
                                elif hasattr(_nhs, '__float__'):
                                    _nu_hat_val = float(_nhs)
                            except (ValueError, TypeError, IndexError):
                                pass

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
                        "nu_hat": _nu_hat_val,  # v6.0: BMA-averaged ν
                    })

                idx += eval_spacing

            # v4.0: Save records to cache for future skip-if-fresh
            _save_records(symbol, records)

        # Build calibration dict
        calibration = _build_calibration(records)
        if calibration is None:
            return symbol, None, "Too few eval points"

        return symbol, calibration, None

    except Exception as e:
        return symbol, None, f"{type(e).__name__}: {e}"


# ===========================================================================
# PIPELINE v4.0 — Metrics evaluator + 5 functional steps + runner
# ===========================================================================
# Each step is a pure function: step(data, prior) → PipelineResult.
# The pipeline runner chains them and records marginal lift per step.
# ===========================================================================

def _make_horizon_data(recs: List[Dict], H: int) -> HorizonData:
    """Convert raw record dicts into HorizonData arrays.

    v6.0: Also extracts nu_hat (BMA-averaged degrees of freedom).
    Default 30.0 = near-Gaussian for records without nu_hat.
    """
    n = len(recs)
    return HorizonData(
        H=H,
        n_eval=n,
        predicted=np.array([r["predicted"] for r in recs], dtype=np.float64),
        actual=np.array([r["actual"] for r in recs], dtype=np.float64),
        p_ups=np.array([r["p_up"] for r in recs], dtype=np.float64),
        actual_ups=np.array([r["actual_up"] for r in recs], dtype=np.float64),
        sigma_pred=np.array([r.get("sigma_H", 1.0) for r in recs], dtype=np.float64),
        vol_regime=np.array([r.get("vol_regime", 1.0) for r in recs], dtype=np.float64),
        weights=_compute_recency_weights(n),
        nu_hat=np.array([r.get("nu_hat", 30.0) for r in recs], dtype=np.float64),
    )


def _evaluate_metrics(
    data: HorizonData,
    p_up_map: Dict,
    emos_params: Dict,
    full: bool = False,
) -> Dict[str, float]:
    """Evaluate calibration metrics for a given calibration state.

    v7.4 performance: Added ``full`` flag.  Intermediate pipeline steps
    only need brier/crps/hit_rate/mag_ratio for decision-making.
    The expensive v7.3 diagnostics (PIT, AD, Berk, Hyv, MAD, LogS, DSS)
    are only computed when ``full=True`` (final pipeline evaluation).

    Args:
        data:        HorizonData (immutable arrays)
        p_up_map:    Beta calibration map (applied to p_ups)
        emos_params: EMOS distributional params (applied to predicted/sigma)
        full:        If True, compute all 11 metrics including v7.3
                     diagnostics.  If False, only compute core 4 metrics
                     (brier, crps, hit_rate, mag_ratio) for speed.

    Returns:
        {"brier": float, "crps": float, "hit_rate": float, "mag_ratio": float,
         "pit_p": float, "ad_p": float, "hyv": float, "berk_p": float,
         "mad": float, "log_s": float, "dss": float}
    """
    # v4.0: Fast path — use batch Numba kernel if Beta calibration
    nb = _nb()
    if (nb is not None and "beta_batch" in nb
            and p_up_map.get("type") == "beta"):
        beta_a = p_up_map.get("a", 1.0)
        beta_b = p_up_map.get("b", 1.0)
        beta_c = p_up_map.get("c", 0.0)
        clip_lo = p_up_map.get("clip_lo", 0.01)
        clip_hi = p_up_map.get("clip_hi", 0.99)
        cal_p = nb["beta_batch"](data.p_ups, beta_a, beta_b, beta_c,
                                 clip_lo, clip_hi)
    else:
        cal_p = np.array([apply_p_up_map(p, p_up_map) for p in data.p_ups])
    brier = float(np.mean((cal_p - data.actual_ups) ** 2))

    # CRPS: EMOS-corrected distributional score
    # v6.0: Use Student-t CRPS when median ν < 25 (genuinely heavy tails)
    a = emos_params.get("a", 0.0)
    b = emos_params.get("b", 1.0)
    c = emos_params.get("c", 0.0)
    d = emos_params.get("d", 1.0)
    mu_cor = a + b * data.predicted
    sig_cor = np.maximum(EMOS_SIGMA_FLOOR, c + d * data.sigma_pred)

    nu_emos = emos_params.get("nu", None)
    median_nu = float(np.median(data.nu_hat)) if len(data.nu_hat) > 0 else 30.0
    use_student_t = (median_nu < 25.0) and (nb is not None) and ("crps_student_t_mean" in nb)

    if use_student_t:
        # Use BMA-averaged ν or EMOS-fitted ν
        nu_for_crps = nu_emos if nu_emos is not None else max(3.0, median_nu)
        nu_arr = np.full(len(mu_cor), nu_for_crps, dtype=np.float64)
        crps = float(nb["crps_student_t_mean"](mu_cor, sig_cor, data.actual, nu_arr))
    else:
        crps = float(np.mean(_crps_gaussian(mu_cor, sig_cor, data.actual)))

    # Hit rate: directional accuracy
    pred_dirs = np.sign(data.predicted)
    actual_dirs = np.sign(data.actual)
    nonzero = pred_dirs != 0
    hit_rate = float(np.mean(pred_dirs[nonzero] == actual_dirs[nonzero])) if nonzero.sum() > 0 else 0.5

    # Magnitude ratio
    avg_pred_abs = float(np.mean(np.abs(data.predicted)))
    avg_actual_abs = float(np.mean(np.abs(data.actual)))
    mag_ratio = avg_pred_abs / max(avg_actual_abs, 1e-8)

    # v7.4: Fast path — skip expensive diagnostics for intermediate steps
    _core = {
        "brier": round(brier, 6),
        "crps": round(crps, 6),
        "hit_rate": round(hit_rate, 6),
        "mag_ratio": round(mag_ratio, 6),
    }
    if not full:
        _core.update({
            "pit_p": 1.0, "ad_p": 1.0, "hyv": 0.0, "berk_p": 1.0,
            "mad": 0.0, "log_s": 0.0, "dss": 0.0,
        })
        return _core

    # ===================================================================
    # v7.3: Multi-diagnostic metrics (PIT, Hyvärinen, AD, Berkowitz, MAD,
    #        LogS, DSS) — proper scoring rule diversity for honest calibration
    # Only computed when full=True (final pipeline evaluation).
    # ===================================================================
    pit_p = 1.0
    ad_p = 1.0
    hyv = 0.0
    berk_p = 1.0
    mad_val = 0.0
    log_s = 0.0
    dss_val = 0.0

    # Build ν array for diagnostic kernels
    nu_for_diag = nu_emos if nu_emos is not None else max(3.0, median_nu)
    nu_diag = np.full(len(mu_cor), nu_for_diag, dtype=np.float64)
    w_diag = data.weights.astype(np.float64) if data.weights is not None else np.ones(len(mu_cor), dtype=np.float64)

    if nb is not None and len(mu_cor) >= 10:
        mu_c64 = mu_cor.astype(np.float64)
        sig_c64 = sig_cor.astype(np.float64)
        y_64 = data.actual.astype(np.float64)

        # PIT values → uniformity tests
        if "pit_values" in nb:
            pit_vals = nb["pit_values"](mu_c64, sig_c64, y_64, nu_diag)

            if "pit_ks" in nb:
                _, pit_p = nb["pit_ks"](pit_vals)
            if "pit_ad" in nb:
                _, ad_p = nb["pit_ad"](pit_vals)
            if "berkowitz" in nb:
                _, berk_p = nb["berkowitz"](pit_vals)

        # Hyvärinen score
        if "hyvarinen" in nb:
            hyv = float(nb["hyvarinen"](mu_c64, sig_c64, y_64, nu_diag, w_diag))

        # MAD
        if "mad" in nb:
            mad_val = float(nb["mad"](mu_c64, y_64, w_diag))

        # Log score
        if "log_score" in nb:
            log_s = float(nb["log_score"](mu_c64, sig_c64, y_64, nu_diag, w_diag))

        # DSS
        if "dss_score" in nb:
            dss_val = float(nb["dss_score"](mu_c64, sig_c64, y_64, w_diag))

    _core.update({
        # v7.3 multi-diagnostic
        "pit_p": round(float(pit_p), 4),
        "ad_p": round(float(ad_p), 4),
        "hyv": round(float(hyv), 2),
        "berk_p": round(float(berk_p), 4),
        "mad": round(float(mad_val), 4),
        "log_s": round(float(log_s), 4),
        "dss": round(float(dss_val), 4),
    })
    return _core


# ---------------------------------------------------------------------------
# Step 1: Partition — split records by vol regime + compute recency weights
# ---------------------------------------------------------------------------

def _step_partition(
    recs: List[Dict],
    H: int,
) -> Tuple[Dict[str, HorizonData], HorizonData]:
    """Partition records by volatility regime.

    Returns per-regime HorizonData dicts + the ALL (pooled) data.
    This is not a PipelineResult step — it's a pre-processing step that
    produces the regime-partitioned data the other steps consume.

    Args:
        recs:  Raw walk-forward records for one horizon
        H:     Horizon in days

    Returns:
        (regime_data: {regime_name: HorizonData}, all_data: HorizonData)
    """
    partitions = _partition_by_regime(recs)
    regime_data = {}
    for regime_name, regime_recs in partitions.items():
        if len(regime_recs) >= MIN_EVAL_POINTS:
            regime_data[regime_name] = _make_horizon_data(regime_recs, H)
    # Ensure ALL always present
    if REGIME_ALL not in regime_data:
        regime_data[REGIME_ALL] = _make_horizon_data(recs, H)
    all_data = regime_data[REGIME_ALL]
    return regime_data, all_data


# ---------------------------------------------------------------------------
# Step 2: Beta — fit per-regime Beta calibration
# ---------------------------------------------------------------------------

def _step_beta(
    regime_data: Dict[str, HorizonData],
    prior: PipelineResult,
) -> PipelineResult:
    """Fit Beta calibration for each regime.

    Pure function: returns a new PipelineResult with beta p_up_maps.

    Args:
        regime_data: Per-regime HorizonData
        prior:       Previous pipeline state

    Returns:
        New PipelineResult with beta calibration fitted
    """
    new_by_regime = {}
    detail = {}

    for regime_name, data in regime_data.items():
        identity_p = {"type": "beta", **BETA_IDENTITY}
        identity_e = {"type": "emos", **EMOS_IDENTITY}

        # Metrics BEFORE beta fitting (identity state)
        prior_p = prior.by_regime.get(regime_name, {}).get("p_up_map", identity_p)
        prior_e = prior.by_regime.get(regime_name, {}).get("emos", identity_e)
        metrics_before = _evaluate_metrics(data, prior_p, prior_e)

        # Fit beta
        p_up_map = _fit_beta_calibration(
            data.p_ups, data.actual_ups, data.n_eval, weights=data.weights,
        )

        # Metrics AFTER beta fitting
        metrics_after = _evaluate_metrics(data, p_up_map, prior_e)

        new_by_regime[regime_name] = {
            "p_up_map": p_up_map,
            "emos": prior_e,
            "n_eval": data.n_eval,
        }
        detail[regime_name] = {
            "brier_before": metrics_before["brier"],
            "brier_after": metrics_after["brier"],
        }

    step = StepResult(
        step_name="beta",
        metrics_before=_evaluate_metrics(
            regime_data[REGIME_ALL],
            prior.by_regime.get(REGIME_ALL, {}).get("p_up_map", {"type": "beta", **BETA_IDENTITY}),
            prior.by_regime.get(REGIME_ALL, {}).get("emos", {"type": "emos", **EMOS_IDENTITY}),
        ),
        metrics_after=_evaluate_metrics(
            regime_data[REGIME_ALL],
            new_by_regime.get(REGIME_ALL, {}).get("p_up_map", {"type": "beta", **BETA_IDENTITY}),
            new_by_regime.get(REGIME_ALL, {}).get("emos", {"type": "emos", **EMOS_IDENTITY}),
        ),
        detail=detail,
    )

    return PipelineResult(
        by_regime=new_by_regime,
        label_thresholds=prior.label_thresholds,
        steps=prior.steps + [step],
    )


# ---------------------------------------------------------------------------
# Step 2b: Isotonic — post-Beta isotonic refinement (v6.0)
# ---------------------------------------------------------------------------

def _step_isotonic(
    regime_data: Dict[str, HorizonData],
    prior: PipelineResult,
) -> PipelineResult:
    """Post-Beta isotonic regression refinement (v6.0).

    After Beta calibration (parametric), apply isotonic regression
    (nonparametric) to capture any remaining non-linear miscalibration.

    The isotonic map is stored per-regime and applied as a blend with
    the parametric Beta output. This is the "best of both worlds":
    Beta handles the smooth shape, isotonic handles the kinks.

    Args:
        regime_data: Per-regime HorizonData
        prior:       Previous pipeline state (with Beta fitted)

    Returns:
        New PipelineResult with isotonic maps added to by_regime
    """
    new_by_regime = {}
    detail = {}

    nb = _nb()

    for regime_name, data in regime_data.items():
        cur_entry = prior.by_regime.get(regime_name, {})
        cur_p_map = cur_entry.get("p_up_map", {"type": "beta", **BETA_IDENTITY})
        cur_emos = cur_entry.get("emos", {"type": "emos", **EMOS_IDENTITY})
        n_eval = cur_entry.get("n_eval", data.n_eval)

        # Metrics BEFORE isotonic
        metrics_before = _evaluate_metrics(data, cur_p_map, cur_emos)

        # Apply Beta calibration to get calibrated p_ups
        if (nb is not None and "beta_batch" in nb
                and cur_p_map.get("type") == "beta"):
            cal_p = nb["beta_batch"](
                data.p_ups,
                cur_p_map.get("a", 1.0), cur_p_map.get("b", 1.0),
                cur_p_map.get("c", 0.0),
                cur_p_map.get("clip_lo", BETA_CLIP_LO),
                cur_p_map.get("clip_hi", BETA_CLIP_HI),
            )
        else:
            cal_p = np.array([apply_p_up_map(p, cur_p_map) for p in data.p_ups])

        # Fit isotonic regression on Beta-calibrated probabilities
        isotonic_map = None
        if data.n_eval >= MIN_EVAL_POINTS and nb is not None:
            try:
                # isotonic_regression_nb returns (x_breakpoints, y_breakpoints)
                # directly from binned PAV algorithm.
                # Use 50 bins for fine resolution (data may occupy narrow range)
                iso_x_arr, iso_y_arr = nb["isotonic_regression"](
                    cal_p.astype(np.float64),
                    data.actual_ups.astype(np.float64),
                    50,  # n_bins (50 = 0.02 width per bin)
                )

                if len(iso_x_arr) >= 3:
                    iso_x = [float(v) for v in iso_x_arr]
                    iso_y = [max(0.0, min(1.0, float(v))) for v in iso_y_arr]
                    isotonic_map = {"x": iso_x, "y": iso_y}
            except Exception:
                pass  # isotonic fitting failed; skip silently

        # Store isotonic map in regime entry
        new_entry = {
            "p_up_map": cur_p_map,
            "emos": cur_emos,
            "n_eval": n_eval,
        }
        if isotonic_map is not None:
            new_entry["isotonic_map"] = isotonic_map

        new_by_regime[regime_name] = new_entry
        detail[regime_name] = {
            "isotonic_fitted": isotonic_map is not None,
            "n_breakpoints": len(isotonic_map["x"]) if isotonic_map else 0,
        }

    # Overall step metrics
    all_data = regime_data[REGIME_ALL]
    metrics_before = _evaluate_metrics(
        all_data,
        prior.by_regime.get(REGIME_ALL, {}).get("p_up_map", {"type": "beta", **BETA_IDENTITY}),
        prior.by_regime.get(REGIME_ALL, {}).get("emos", {"type": "emos", **EMOS_IDENTITY}),
    )

    # metrics_after: if isotonic was fitted for ALL regime, compute Brier
    # using the isotonic-blended probabilities
    all_entry = new_by_regime.get(REGIME_ALL, {})
    all_iso = all_entry.get("isotonic_map", None)
    if all_iso is not None and nb is not None and "isotonic_beta_blend" in nb:
        all_p_map = all_entry.get("p_up_map", {"type": "beta", **BETA_IDENTITY})
        cal_p_blended = nb["isotonic_beta_blend"](
            all_data.p_ups,
            all_p_map.get("a", 1.0), all_p_map.get("b", 1.0), all_p_map.get("c", 0.0),
            np.array(all_iso["x"], dtype=np.float64),
            np.array(all_iso["y"], dtype=np.float64),
            0.3,  # blend_w for Beta (30% Beta, 70% isotonic)
            all_p_map.get("clip_lo", BETA_CLIP_LO),
            all_p_map.get("clip_hi", BETA_CLIP_HI),
        )
        brier_after = float(np.mean((cal_p_blended - all_data.actual_ups) ** 2))
        metrics_after = dict(metrics_before)
        metrics_after["brier"] = brier_after
    else:
        metrics_after = _evaluate_metrics(
            all_data,
            new_by_regime.get(REGIME_ALL, {}).get("p_up_map", {"type": "beta", **BETA_IDENTITY}),
            new_by_regime.get(REGIME_ALL, {}).get("emos", {"type": "emos", **EMOS_IDENTITY}),
        )

    step = StepResult(
        step_name="isotonic",
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        detail=detail,
    )

    return PipelineResult(
        by_regime=new_by_regime,
        label_thresholds=prior.label_thresholds,
        steps=prior.steps + [step],
    )


# ---------------------------------------------------------------------------
# Step 2: Sigma Floor — realized-vol floor on sigma_pred (v7.1)
# ---------------------------------------------------------------------------

def _step_sigma_floor(
    regime_data: Dict[str, HorizonData],
    prior: PipelineResult,
) -> PipelineResult:
    """Apply realized-vol floor to sigma_pred (v7.1).

    Prevents catastrophic under-dispersion where sigma_pred << actual vol.
    Uses MAD × 1.4826 as robust sigma estimator.  Applied BEFORE EMOS
    so that Stage 2 (scale correction) operates on reasonable sigmas.

    This modifies sigma_pred in the HorizonData objects (they are mutable
    numpy arrays shared across regime_data views).

    Args:
        regime_data: Per-regime HorizonData
        prior:       Previous pipeline state

    Returns:
        Same PipelineResult (sigma_pred modified in-place on ALL partition)
    """
    nb = _nb()
    all_data = regime_data[REGIME_ALL]

    metrics_before = _evaluate_metrics(
        all_data,
        prior.by_regime.get(REGIME_ALL, {}).get("p_up_map", {"type": "beta", **BETA_IDENTITY}),
        prior.by_regime.get(REGIME_ALL, {}).get("emos", {"type": "emos", **EMOS_IDENTITY}),
    )

    n_floored = 0
    if len(all_data.actual) >= 10:
        if nb is not None and "realized_vol_floor" in nb:
            new_sigma = nb["realized_vol_floor"](
                all_data.actual.astype(np.float64),
                all_data.sigma_pred.astype(np.float64),
                SIGMA_FLOOR_FRAC,
            )
        else:
            # Python fallback
            med_actual = float(np.median(all_data.actual))
            mad = float(np.median(np.abs(all_data.actual - med_actual)))
            sigma_realized = mad * 1.4826
            floor_val = SIGMA_FLOOR_FRAC * sigma_realized
            new_sigma = np.maximum(all_data.sigma_pred, floor_val)

        n_floored = int(np.sum(new_sigma > all_data.sigma_pred))
        # Update sigma_pred in-place for ALL regimes (they share the same array)
        all_data.sigma_pred[:] = new_sigma

        # Also update per-regime views if they're separate arrays
        for rname, rdata in regime_data.items():
            if rname == REGIME_ALL:
                continue
            if len(rdata.actual) >= 10:
                if nb is not None and "realized_vol_floor" in nb:
                    rdata.sigma_pred[:] = nb["realized_vol_floor"](
                        rdata.actual.astype(np.float64),
                        rdata.sigma_pred.astype(np.float64),
                        SIGMA_FLOOR_FRAC,
                    )
                else:
                    med_r = float(np.median(rdata.actual))
                    mad_r = float(np.median(np.abs(rdata.actual - med_r)))
                    sigma_r = mad_r * 1.4826
                    rdata.sigma_pred[:] = np.maximum(rdata.sigma_pred, SIGMA_FLOOR_FRAC * sigma_r)

    metrics_after = _evaluate_metrics(
        all_data,
        prior.by_regime.get(REGIME_ALL, {}).get("p_up_map", {"type": "beta", **BETA_IDENTITY}),
        prior.by_regime.get(REGIME_ALL, {}).get("emos", {"type": "emos", **EMOS_IDENTITY}),
    )

    step = StepResult(
        step_name="sigma_floor",
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        detail={"n_floored": n_floored, "floor_frac": SIGMA_FLOOR_FRAC},
    )

    return PipelineResult(
        by_regime=prior.by_regime,
        label_thresholds=prior.label_thresholds,
        steps=prior.steps + [step],
    )


# ---------------------------------------------------------------------------
# Step 3: EMOS — fit per-regime EMOS distributional correction
# ---------------------------------------------------------------------------

def _step_emos(
    regime_data: Dict[str, HorizonData],
    prior: PipelineResult,
) -> PipelineResult:
    """Fit EMOS for each regime.

    Pure function: returns a new PipelineResult with EMOS params.

    Args:
        regime_data: Per-regime HorizonData
        prior:       Previous pipeline state (with beta already fitted)

    Returns:
        New PipelineResult with EMOS fitted
    """
    new_by_regime = {}
    detail = {}

    for regime_name, data in regime_data.items():
        identity_e = {"type": "emos", **EMOS_IDENTITY}
        cur_p_map = prior.by_regime.get(regime_name, {}).get(
            "p_up_map", {"type": "beta", **BETA_IDENTITY}
        )
        prior_e = prior.by_regime.get(regime_name, {}).get("emos", identity_e)

        # Metrics BEFORE EMOS fitting
        metrics_before = _evaluate_metrics(data, cur_p_map, prior_e)

        # v6.0: Use Student-t EMOS when median ν < 25 (heavy tails)
        median_nu = float(np.median(data.nu_hat)) if len(data.nu_hat) > 0 else 30.0
        use_student_t = median_nu < 25.0

        # v7.4: Skip Stage 3 for sub-regimes (< 80 points) — diminishing returns
        _skip_s3 = (regime_name != REGIME_ALL and data.n_eval < 80)

        if use_student_t:
            emos_params = _fit_emos_student_t(
                data.predicted, data.actual, data.sigma_pred,
                data.nu_hat, data.n_eval, weights=data.weights,
                skip_stage3=_skip_s3,
            )
        else:
            emos_params = _fit_emos(
                data.predicted, data.actual, data.sigma_pred,
                data.n_eval, weights=data.weights,
            )

        # Metrics AFTER EMOS fitting
        metrics_after = _evaluate_metrics(data, cur_p_map, emos_params)

        # Preserve isotonic_map from prior step if present
        prior_entry = prior.by_regime.get(regime_name, {})
        new_entry = {
            "p_up_map": cur_p_map,
            "emos": emos_params,
            "n_eval": data.n_eval,
        }
        if "isotonic_map" in prior_entry:
            new_entry["isotonic_map"] = prior_entry["isotonic_map"]

        new_by_regime[regime_name] = new_entry
        detail[regime_name] = {
            "crps_before": metrics_before["crps"],
            "crps_after": metrics_after["crps"],
            "student_t": use_student_t,
            "median_nu": round(median_nu, 2),
        }

    step = StepResult(
        step_name="emos",
        metrics_before=_evaluate_metrics(
            regime_data[REGIME_ALL],
            prior.by_regime.get(REGIME_ALL, {}).get("p_up_map", {"type": "beta", **BETA_IDENTITY}),
            prior.by_regime.get(REGIME_ALL, {}).get("emos", {"type": "emos", **EMOS_IDENTITY}),
        ),
        metrics_after=_evaluate_metrics(
            regime_data[REGIME_ALL],
            new_by_regime.get(REGIME_ALL, {}).get("p_up_map", {"type": "beta", **BETA_IDENTITY}),
            new_by_regime.get(REGIME_ALL, {}).get("emos", {"type": "emos", **EMOS_IDENTITY}),
        ),
        detail=detail,
    )

    return PipelineResult(
        by_regime=new_by_regime,
        label_thresholds=prior.label_thresholds,
        steps=prior.steps + [step],
    )


# ---------------------------------------------------------------------------
# Step 3b: Magnitude — explicit magnitude scale correction (v5.0)
# ---------------------------------------------------------------------------

def _step_magnitude(
    regime_data: Dict[str, HorizonData],
    prior: PipelineResult,
) -> PipelineResult:
    """Explicit magnitude calibration step (v5.0).

    The core problem: predictions are systematically ~6.7x too small
    (mag_ratio ≈ 0.15). EMOS's b parameter *should* capture this via
    CRPS optimization, but the optimizer prefers inflating sigma (c)
    over scaling mean (b) because wider sigma more efficiently reduces
    CRPS when the mean is already small.

    This step directly computes the magnitude correction and applies it
    by multiplying the EMOS 'b' parameter. Only applied if it improves
    CRPS (no-harm guard).

    Mathematical formulation:
      scale = median(|actual|) / median(|predicted|)
      b_new = b_current * clip(scale, 0.3, 8.0)
      a_new = a_current * clip(scale, 0.3, 8.0)  # bias also needs rescaling

    We use medians (not means) for robustness to outliers.

    Args:
        regime_data: Per-regime HorizonData
        prior:       Previous pipeline state (with EMOS already fitted)

    Returns:
        New PipelineResult with magnitude-corrected EMOS params
    """
    new_by_regime = {}
    detail = {}

    for regime_name, data in regime_data.items():
        cur_entry = prior.by_regime.get(regime_name, {})
        cur_p_map = cur_entry.get("p_up_map", {"type": "beta", **BETA_IDENTITY})
        cur_emos = dict(cur_entry.get("emos", {"type": "emos", **EMOS_IDENTITY}))
        n_eval = cur_entry.get("n_eval", data.n_eval)

        # Current EMOS params
        a_cur = cur_emos.get("a", 0.0)
        b_cur = cur_emos.get("b", 1.0)

        # v7.1: Compute RESIDUAL magnitude correction on EMOS-corrected predictions
        # Old (v5.0): raw_scale = med(|actual|) / med(|predicted|) → double-counts EMOS b
        # New (v7.1): residual_scale = med(|actual|) / med(|corrected|) → only fixes residual
        pred_corrected = a_cur + b_cur * data.predicted
        pred_abs = np.abs(pred_corrected)
        actual_abs = np.abs(data.actual)
        med_pred = float(np.median(pred_abs)) if len(pred_abs) > 0 else 1e-8
        med_actual = float(np.median(actual_abs)) if len(actual_abs) > 0 else 1e-8

        if med_pred > 1e-8:
            raw_scale = med_actual / med_pred
        else:
            raw_scale = 1.0

        # Clip to reasonable range and apply shrinkage
        scale = max(0.3, min(8.0, raw_scale))
        # Shrinkage toward 1.0 (no correction) based on data quantity
        lam = min(1.0, n_eval / SHRINKAGE_FULL_N)
        scale_shrunk = lam * scale + (1.0 - lam) * 1.0

        # Apply magnitude correction to EMOS b and a
        b_new = b_cur * scale_shrunk
        a_new = a_cur * scale_shrunk  # bias magnitude also needs rescaling

        new_emos = dict(cur_emos)
        new_emos["b"] = round(float(b_new), 6)
        new_emos["a"] = round(float(a_new), 6)

        # No-harm guard: only apply if CRPS improves
        crps_before = _evaluate_metrics(data, cur_p_map, cur_emos)["crps"]
        crps_after = _evaluate_metrics(data, cur_p_map, new_emos)["crps"]

        if crps_after <= crps_before:
            applied_emos = new_emos
            was_applied = True
        else:
            applied_emos = cur_emos
            was_applied = False

        new_entry = {
            "p_up_map": cur_p_map,
            "emos": applied_emos,
            "n_eval": n_eval,
        }
        # v6.0: preserve isotonic_map from prior steps
        if "isotonic_map" in cur_entry:
            new_entry["isotonic_map"] = cur_entry["isotonic_map"]
        new_by_regime[regime_name] = new_entry
        detail[regime_name] = {
            "raw_scale": round(raw_scale, 4),
            "scale_shrunk": round(scale_shrunk, 4),
            "b_before": round(b_cur, 6),
            "b_after": round(float(applied_emos.get("b", 1.0)), 6),
            "crps_before": crps_before,
            "crps_after": crps_after if was_applied else crps_before,
            "applied": was_applied,
        }

    # Overall step metrics (on ALL partition)
    all_data = regime_data[REGIME_ALL]
    step = StepResult(
        step_name="magnitude",
        metrics_before=_evaluate_metrics(
            all_data,
            prior.by_regime.get(REGIME_ALL, {}).get("p_up_map", {"type": "beta", **BETA_IDENTITY}),
            prior.by_regime.get(REGIME_ALL, {}).get("emos", {"type": "emos", **EMOS_IDENTITY}),
        ),
        metrics_after=_evaluate_metrics(
            all_data,
            new_by_regime.get(REGIME_ALL, {}).get("p_up_map", {"type": "beta", **BETA_IDENTITY}),
            new_by_regime.get(REGIME_ALL, {}).get("emos", {"type": "emos", **EMOS_IDENTITY}),
        ),
        detail=detail,
    )

    return PipelineResult(
        by_regime=new_by_regime,
        label_thresholds=prior.label_thresholds,
        steps=prior.steps + [step],
    )


# ---------------------------------------------------------------------------
# Step 4: CV Guard — temporal cross-validation revert guard
# ---------------------------------------------------------------------------

def _step_cv_guard(
    regime_data: Dict[str, HorizonData],
    recs_by_regime: Dict[str, List[Dict]],
    prior: PipelineResult,
) -> PipelineResult:
    """Cross-validation guard with adaptive folds (v7.0).

    v7.0: Adaptive fold count based on data quantity.
    - n_eval < 30:  Skip CV entirely (insufficient data for reliable CV)
    - n_eval < 100: 2-fold (train 50%, val 50%) — less noisy than 3-fold
    - n_eval < 200: 3-fold (original v6.0)
    - n_eval >= 200: 5-fold (more stable estimate)

    Also: Since EMOS is now two-stage (2+3 params instead of 5 joint),
    each stage has lower effective dimensionality, reducing minimum
    data requirements.

    Expanding folds for 3-fold:
      Fold 1: train [0, 33%), val [33%, 67%)
      Fold 2: train [0, 50%), val [50%, 83%)
      Fold 3: train [0, 67%), val [67%, 100%)

    Args:
        regime_data:    Per-regime HorizonData (full data)
        recs_by_regime: Raw record lists per regime (for re-fitting train split)
        prior:          Previous pipeline state (with beta + EMOS fitted)

    Returns:
        New PipelineResult (may revert some regime params to identity)
    """
    new_by_regime = {}
    detail = {}
    any_reverted = False

    for regime_name, data in regime_data.items():
        cur_entry = prior.by_regime.get(regime_name, {})
        cur_p_map = cur_entry.get("p_up_map", {"type": "beta", **BETA_IDENTITY})
        cur_emos = cur_entry.get("emos", {"type": "emos", **EMOS_IDENTITY})
        n_eval = cur_entry.get("n_eval", data.n_eval)

        cv_reverted_beta = False
        cv_reverted_emos = False

        # v7.0: Adaptive fold count
        if data.n_eval < 30:
            # Too few points — skip CV, trust the fitting
            N_FOLDS = 0
        elif data.n_eval < 100:
            N_FOLDS = 2
        elif data.n_eval < 200:
            N_FOLDS = 3
        else:
            N_FOLDS = 5

        if N_FOLDS > 0 and data.n_eval >= 20:
            n = data.n_eval
            fold_brier_deltas = []
            fold_crps_deltas = []

            for fold_idx in range(N_FOLDS):
                # Expanding window: train = [0, train_end), val = [train_end, val_end)
                train_frac = (fold_idx + 1) / (N_FOLDS + 1)
                val_frac = (fold_idx + 2) / (N_FOLDS + 1)
                train_end = int(n * train_frac)
                val_end = min(n, int(n * val_frac))

                if train_end < BETA_MIN_POINTS or val_end - train_end < 5:
                    continue

                # Validation arrays
                val_p_ups = data.p_ups[train_end:val_end]
                val_actual_ups = data.actual_ups[train_end:val_end]
                val_predicted = data.predicted[train_end:val_end]
                val_actual = data.actual[train_end:val_end]
                val_sigma = data.sigma_pred[train_end:val_end]

                # Train arrays
                train_p_ups = data.p_ups[:train_end]
                train_actual_ups = data.actual_ups[:train_end]
                train_predicted = data.predicted[:train_end]
                train_actual = data.actual[:train_end]
                train_sigma = data.sigma_pred[:train_end]
                n_train = len(train_predicted)
                w_train = _compute_recency_weights(n_train)

                # Re-fit Beta on train set only (3 params, stable on small n)
                cv_beta = _fit_beta_calibration(
                    train_p_ups, train_actual_ups, n_train, weights=w_train
                )
                # v7.1: Use globally-fitted EMOS, don't re-fit per fold.
                # Root cause of 0.000 CRPS: re-fitting EMOS on 40 train points
                # produces noisy b → fails validation → reverts to identity.
                # Instead, validate whether the GLOBAL EMOS generalizes OOS.
                cv_emos = cur_emos
                train_nu = data.nu_hat[:train_end] if len(data.nu_hat) > 0 else np.array([])
                median_nu = float(np.median(train_nu)) if len(train_nu) > 0 else 30.0

                # Beta CV: Brier delta on validation
                brier_raw_val = float(np.mean((val_p_ups - val_actual_ups) ** 2))
                cal_p_val = np.array([apply_p_up_map(p, cv_beta) for p in val_p_ups])
                brier_cal_val = float(np.mean((cal_p_val - val_actual_ups) ** 2))
                fold_brier_deltas.append(brier_cal_val - brier_raw_val)

                # EMOS CV: CRPS delta on validation
                # v6.0 fix: Use Student-t CRPS when median_nu < 25 (consistent
                # with main evaluation). Previously used Gaussian CRPS always,
                # causing false reverts when EMOS fitted for Student-t metric.
                nb = _nb()
                use_t = (median_nu < 25.0) and (nb is not None) and ("crps_student_t_mean" in nb)

                if use_t:
                    nu_raw = max(3.0, median_nu)
                    nu_arr_raw = np.full(len(val_predicted), nu_raw, dtype=np.float64)
                    crps_raw_val = float(nb["crps_student_t_mean"](
                        val_predicted,
                        np.maximum(val_sigma, EMOS_SIGMA_FLOOR),
                        val_actual,
                        nu_arr_raw,
                    ))
                else:
                    crps_raw_val = float(np.mean(_crps_gaussian(
                        val_predicted,
                        np.maximum(val_sigma, EMOS_SIGMA_FLOOR),
                        val_actual,
                    )))

                mu_cv = cv_emos.get("a", 0.0) + cv_emos.get("b", 1.0) * val_predicted
                sig_cv = np.maximum(EMOS_SIGMA_FLOOR,
                                    cv_emos.get("c", 0.0) + cv_emos.get("d", 1.0) * val_sigma)

                if use_t:
                    nu_emos = cv_emos.get("nu", None)
                    nu_for_cv = nu_emos if nu_emos is not None else max(3.0, median_nu)
                    nu_arr_cv = np.full(len(mu_cv), nu_for_cv, dtype=np.float64)
                    crps_cal_val = float(nb["crps_student_t_mean"](mu_cv, sig_cv, val_actual, nu_arr_cv))
                else:
                    crps_cal_val = float(np.mean(_crps_gaussian(mu_cv, sig_cv, val_actual)))

                fold_crps_deltas.append(crps_cal_val - crps_raw_val)

            # v7.0: Adaptive tolerance scales with 1/sqrt(total val points per fold)
            avg_val_pts = max(1, data.n_eval // (N_FOLDS + 1))
            cv_tol = max(0.005, 0.08 / math.sqrt(avg_val_pts))

            # Average delta across folds
            if fold_brier_deltas:
                avg_brier_delta = float(np.mean(fold_brier_deltas))
                if avg_brier_delta > cv_tol:
                    cur_p_map = {"type": "beta", **BETA_IDENTITY}
                    cv_reverted_beta = True
                    any_reverted = True

            # v7.1: Decouple mean/scale reversion — don't kill good mean
            # correction because scale correction was unstable.
            # Evaluate three configurations on each fold:
            #   identity: (0, 1, 0, 1, nu_prior)
            #   mean-only: (a_fit, b_fit, 0, 1, nu_prior)
            #   full: (a_fit, b_fit, c_fit, d_fit, nu_fit)
            cv_reverted_mean = False
            cv_reverted_scale = False

            if fold_crps_deltas:
                avg_crps_delta = float(np.mean(fold_crps_deltas))

                if avg_crps_delta > cv_tol:
                    # Full EMOS worsens OOS — check if mean-only is still beneficial
                    a_g = cur_emos.get("a", 0.0)
                    b_g = cur_emos.get("b", 1.0)
                    c_g = cur_emos.get("c", 0.0)
                    d_g = cur_emos.get("d", 1.0)
                    nu_g = cur_emos.get("nu", None)

                    # Build mean-only EMOS (keep a,b, revert c,d,nu)
                    mean_only_emos = dict(cur_emos)
                    mean_only_emos["c"] = 0.0
                    mean_only_emos["d"] = 1.0
                    if nu_g is not None:
                        mean_only_emos["nu"] = nu_g  # keep nu from global fit

                    # Test mean-only vs identity on full data
                    crps_identity = _evaluate_metrics(
                        data, cur_p_map,
                        {"type": "emos", **EMOS_IDENTITY},
                    )["crps"]
                    crps_mean_only = _evaluate_metrics(
                        data, cur_p_map, mean_only_emos,
                    )["crps"]

                    if crps_mean_only <= crps_identity:
                        # Mean correction helps — keep it, revert only scale
                        cur_emos = mean_only_emos
                        cv_reverted_scale = True
                        any_reverted = True
                    else:
                        # Mean correction also hurts — revert everything
                        cur_emos = {"type": "emos", **EMOS_IDENTITY}
                        cv_reverted_mean = True
                        cv_reverted_scale = True
                        cv_reverted_emos = True
                        any_reverted = True

            detail[regime_name] = {
                "n_folds": N_FOLDS,
                "avg_brier_delta": round(float(np.mean(fold_brier_deltas)), 6) if fold_brier_deltas else 0.0,
                "avg_crps_delta": round(float(np.mean(fold_crps_deltas)), 6) if fold_crps_deltas else 0.0,
                "cv_tol": round(cv_tol, 4),
                "beta_reverted": cv_reverted_beta,
                "emos_reverted": cv_reverted_emos,
                "mean_reverted": cv_reverted_mean,
                "scale_reverted": cv_reverted_scale,
            }
        else:
            # v7.0: Skip CV for very small datasets — trust the EMOS fit
            detail[regime_name] = {
                "n_folds": 0,
                "avg_brier_delta": 0.0,
                "avg_crps_delta": 0.0,
                "cv_tol": 0.0,
                "beta_reverted": False,
                "emos_reverted": False,
                "skipped": True,
            }

        new_entry = {
            "p_up_map": cur_p_map,
            "emos": cur_emos,
            "n_eval": n_eval,
        }
        # v6.0: preserve isotonic_map
        if "isotonic_map" in cur_entry:
            new_entry["isotonic_map"] = cur_entry["isotonic_map"]
        new_by_regime[regime_name] = new_entry

    # Overall metrics (on ALL partition)
    all_data = regime_data[REGIME_ALL]
    step = StepResult(
        step_name="cv_guard",
        metrics_before=_evaluate_metrics(
            all_data,
            prior.by_regime.get(REGIME_ALL, {}).get("p_up_map", {"type": "beta", **BETA_IDENTITY}),
            prior.by_regime.get(REGIME_ALL, {}).get("emos", {"type": "emos", **EMOS_IDENTITY}),
        ),
        metrics_after=_evaluate_metrics(
            all_data,
            new_by_regime.get(REGIME_ALL, {}).get("p_up_map", {"type": "beta", **BETA_IDENTITY}),
            new_by_regime.get(REGIME_ALL, {}).get("emos", {"type": "emos", **EMOS_IDENTITY}),
        ),
        was_reverted=any_reverted,
        detail=detail,
    )

    return PipelineResult(
        by_regime=new_by_regime,
        label_thresholds=prior.label_thresholds,
        steps=prior.steps + [step],
    )


# ---------------------------------------------------------------------------
# Step 4b: Temperature Ensemble — blend Beta+isotonic with temperature (v6.0)
# ---------------------------------------------------------------------------

def _step_temp_ensemble(
    regime_data: Dict[str, HorizonData],
    prior: PipelineResult,
) -> PipelineResult:
    """Temperature scaling ensemble (v6.0).

    Fits temperature scaling on raw p_ups, then stores the T parameter
    alongside the Beta+isotonic calibration. At inference time, the
    final calibrated probability is:

        p_final = α * p_beta_isotonic + (1 - α) * p_temperature

    where α = 0.7 (Beta+isotonic gets more weight — it's more flexible).

    Temperature scaling (Guo et al. 2017) is the most robust single-
    parameter post-hoc calibration. It provides a regularized baseline
    that prevents the more flexible Beta+isotonic from overfitting.

    Args:
        regime_data: Per-regime HorizonData
        prior:       Previous pipeline state

    Returns:
        New PipelineResult with temp_scale added to by_regime
    """
    TEMP_BLEND_ALPHA = 0.7  # Weight for Beta+isotonic vs temperature

    new_by_regime = {}
    detail = {}

    for regime_name, data in regime_data.items():
        cur_entry = prior.by_regime.get(regime_name, {})
        cur_p_map = cur_entry.get("p_up_map", {"type": "beta", **BETA_IDENTITY})
        cur_emos = cur_entry.get("emos", {"type": "emos", **EMOS_IDENTITY})
        n_eval = cur_entry.get("n_eval", data.n_eval)

        # Fit temperature scaling on raw p_ups
        T = _fit_temperature_scaling(data.p_ups, data.actual_ups, data.weights)

        new_entry = dict(cur_entry)
        new_entry["temp_scale"] = round(T, 4)
        new_entry["temp_blend_alpha"] = TEMP_BLEND_ALPHA

        new_by_regime[regime_name] = new_entry
        detail[regime_name] = {
            "temperature": round(T, 4),
            "blend_alpha": TEMP_BLEND_ALPHA,
        }

    # Overall step metrics — compute actual temperature-blended Brier
    all_data = regime_data[REGIME_ALL]
    all_p_map = prior.by_regime.get(REGIME_ALL, {}).get("p_up_map", {"type": "beta", **BETA_IDENTITY})
    all_emos = prior.by_regime.get(REGIME_ALL, {}).get("emos", {"type": "emos", **EMOS_IDENTITY})
    metrics_before = _evaluate_metrics(all_data, all_p_map, all_emos)

    # Compute after-metrics: blend Beta(+isotonic) with temperature scaling
    all_new = new_by_regime.get(REGIME_ALL, {})
    T = all_new.get("temp_scale", 1.0)
    alpha = all_new.get("temp_blend_alpha", TEMP_BLEND_ALPHA)
    nb = _nb()

    # Get Beta(+isotonic) calibrated probs
    all_iso = all_new.get("isotonic_map", None)
    if all_iso is not None and nb is not None and "isotonic_beta_blend" in nb:
        cal_p_beta_iso = nb["isotonic_beta_blend"](
            all_data.p_ups,
            all_p_map.get("a", 1.0), all_p_map.get("b", 1.0), all_p_map.get("c", 0.0),
            np.array(all_iso["x"], dtype=np.float64),
            np.array(all_iso["y"], dtype=np.float64),
            0.3,  # blend_w for Beta
            all_p_map.get("clip_lo", BETA_CLIP_LO),
            all_p_map.get("clip_hi", BETA_CLIP_HI),
        )
    elif nb is not None and "beta_batch" in nb and all_p_map.get("type") == "beta":
        cal_p_beta_iso = nb["beta_batch"](
            all_data.p_ups,
            all_p_map.get("a", 1.0), all_p_map.get("b", 1.0),
            all_p_map.get("c", 0.0),
            all_p_map.get("clip_lo", BETA_CLIP_LO),
            all_p_map.get("clip_hi", BETA_CLIP_HI),
        )
    else:
        cal_p_beta_iso = np.array([apply_p_up_map(p, all_p_map) for p in all_data.p_ups])

    # Temperature-scaled probs
    logits = np.log(np.clip(all_data.p_ups, 1e-8, 1 - 1e-8) / (1 - np.clip(all_data.p_ups, 1e-8, 1 - 1e-8)))
    cal_p_temp = 1.0 / (1.0 + np.exp(-logits / max(T, 0.01)))

    # Blend
    cal_p_ensemble = alpha * cal_p_beta_iso + (1.0 - alpha) * cal_p_temp
    brier_after = float(np.mean((cal_p_ensemble - all_data.actual_ups) ** 2))
    metrics_after = dict(metrics_before)
    metrics_after["brier"] = brier_after

    step = StepResult(
        step_name="temp_ensemble",
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        detail=detail,
    )

    return PipelineResult(
        by_regime=new_by_regime,
        label_thresholds=prior.label_thresholds,
        steps=prior.steps + [step],
    )


# ---------------------------------------------------------------------------
# Step 5: Threshold — optimize label buy/sell thresholds
# ---------------------------------------------------------------------------

def _step_threshold(
    all_data: HorizonData,
    prior: PipelineResult,
) -> PipelineResult:
    """Optimize buy/sell label thresholds via grid search.

    Pure function: returns PipelineResult with label_thresholds set.

    v6.0: Passes p_up_map so Brier is computed on calibrated probabilities.

    Args:
        all_data: HorizonData for ALL (pooled) regime
        prior:    Previous pipeline state

    Returns:
        New PipelineResult with label_thresholds
    """
    # v6.0: Get the calibrated p_up_map for Brier computation
    all_p_map = prior.by_regime.get(REGIME_ALL, {}).get("p_up_map", {"type": "beta", **BETA_IDENTITY})

    label_result = _optimize_label_thresholds(
        all_data.p_ups, all_data.actual_ups,
        all_data.predicted, all_data.actual,
        p_up_map=all_p_map,
    )

    # Thresholds don't change Brier/CRPS — record identity metrics
    all_emos = prior.by_regime.get(REGIME_ALL, {}).get("emos", {"type": "emos", **EMOS_IDENTITY})
    metrics = _evaluate_metrics(all_data, all_p_map, all_emos)

    step = StepResult(
        step_name="threshold",
        metrics_before=metrics,
        metrics_after=metrics,
        detail=label_result,
    )

    return PipelineResult(
        by_regime=prior.by_regime,
        label_thresholds=label_result,
        steps=prior.steps + [step],
    )


# ---------------------------------------------------------------------------
# Pipeline runner — chains all steps for one horizon
# ---------------------------------------------------------------------------

def _run_pipeline(
    recs: List[Dict],
    H: int,
) -> Tuple[Dict, Dict, PipelineResult]:
    """Run the full calibration pipeline for one horizon.

    v6.0 Functional chain (8 steps):
      Partition → Beta(focal) → Isotonic → EMOS(Student-t) → Magnitude →
      CVGuard(3-fold) → TempEnsemble → Threshold

    Args:
        recs: Walk-forward records for horizon H
        H:    Horizon in days

    Returns:
        (horizon_cal_dict, label_result, pipeline_result)
    """
    # Pre-step: partition by regime
    partitions = _partition_by_regime(recs)
    regime_data = {}
    for regime_name, regime_recs in partitions.items():
        if len(regime_recs) >= MIN_EVAL_POINTS:
            regime_data[regime_name] = _make_horizon_data(regime_recs, H)
    if REGIME_ALL not in regime_data:
        regime_data[REGIME_ALL] = _make_horizon_data(recs, H)

    all_data = regime_data[REGIME_ALL]

    # Identity initial state
    initial = PipelineResult(
        by_regime={
            rn: {
                "p_up_map": {"type": "beta", **BETA_IDENTITY},
                "emos": {"type": "emos", **EMOS_IDENTITY},
                "n_eval": rd.n_eval,
            }
            for rn, rd in regime_data.items()
        },
    )

    # Step 1: Beta calibration (focal loss, gamma=2.0)
    result = _step_beta(regime_data, initial)

    # Step 2: Sigma floor (v7.1 — realized-vol floor prevents under-dispersion)
    result = _step_sigma_floor(regime_data, result)

    # Step 3: EMOS distributional correction (two-stage: mean-first, then scale)
    result = _step_emos(regime_data, result)

    # Step 4: Magnitude scale correction (v7.1 — residual ratio, not raw)
    result = _step_magnitude(regime_data, result)

    # Step 5: CV guard (expanding-window cross-validation revert)
    result = _step_cv_guard(regime_data, partitions, result)

    # v7.0: Isotonic refinement and temperature ensemble REMOVED
    # These 2 layers each shrank p_up toward 0.5, compounding to ~10% signal loss.
    # The single Beta layer in _apply_p_up_calibration is now the only p_up transform.

    # Step 6: Threshold optimization (Brier on calibrated p_ups)
    result = _step_threshold(all_data, result)

    # === Assemble horizon calibration dict ===
    by_regime = result.by_regime
    all_entry = by_regime.get(REGIME_ALL, {})
    all_p_map = all_entry.get("p_up_map", {"type": "beta", **BETA_IDENTITY})
    all_emos = all_entry.get("emos", {"type": "emos", **EMOS_IDENTITY})

    # Raw metrics from identity (full=True for v7.3 diagnostics)
    identity_metrics = _evaluate_metrics(
        all_data,
        {"type": "beta", **BETA_IDENTITY},
        {"type": "emos", **EMOS_IDENTITY},
        full=True,
    )
    # Calibrated metrics (full=True for v7.3 diagnostics)
    cal_metrics = _evaluate_metrics(all_data, all_p_map, all_emos, full=True)

    # Build pipeline log (per-step marginal lift)
    pipeline_log = []
    for s in result.steps:
        entry = {
            "step": s.step_name,
            "was_reverted": s.was_reverted,
        }
        # Marginal lift
        for metric_key in ("brier", "crps", "hit_rate", "mag_ratio"):
            before = s.metrics_before.get(metric_key, 0.0)
            after = s.metrics_after.get(metric_key, 0.0)
            entry[f"{metric_key}_before"] = before
            entry[f"{metric_key}_after"] = after
            # For brier/crps: lower is better.  For hit_rate: higher.
            if metric_key in ("brier", "crps"):
                entry[f"{metric_key}_lift"] = round(before - after, 6)
            else:
                entry[f"{metric_key}_lift"] = round(after - before, 6)
        if s.detail:
            entry["detail"] = s.detail
        pipeline_log.append(entry)

    # Sigma_H diagnostics
    sigma_all = all_data.sigma_pred

    # Regime counts
    regime_counts = {}
    for rname, bounds in REGIME_BINS.items():
        lo, hi = bounds
        regime_counts[rname] = int(np.sum(
            (all_data.vol_regime >= lo) & (all_data.vol_regime < hi)
        ))

    horizon_cal = {
        # v4.0 regime-conditional storage
        "by_regime": by_regime,
        # Backward-compat: ALL-regime at top level
        "p_up_map": all_p_map,
        "emos": all_emos,
        # Legacy fields (for v2.0 readers)
        "mag_scale": all_emos.get("b", 1.0),
        "bias": all_emos.get("a", 0.0),
        "n_eval": all_data.n_eval,
        # Diagnostics
        "hit_rate_raw": identity_metrics["hit_rate"],
        "brier_raw": identity_metrics["brier"],
        "brier_calibrated": cal_metrics["brier"],
        "mag_ratio_raw": identity_metrics["mag_ratio"],
        "crps_raw": identity_metrics["crps"],
        "crps_calibrated": cal_metrics["crps"],
        "regimes_fitted": list(by_regime.keys()),
        # v7.1: CRPS Skill Score (normalized across assets)
        # CRPSS = 1 - CRPS_cal / CRPS_ref, where CRPS_ref = sigma_actual / sqrt(pi)
        "crpss": round(
            1.0 - cal_metrics["crps"] / max(
                float(np.std(all_data.actual)) / math.sqrt(math.pi),
                1e-8,
            ), 4
        ) if all_data.n_eval > 5 else None,
        # v7.3: Multi-diagnostic metrics (raw = before calibration, cal = after)
        "pit_p_raw": identity_metrics.get("pit_p", None),
        "pit_p_cal": cal_metrics.get("pit_p", None),
        "ad_p_raw": identity_metrics.get("ad_p", None),
        "ad_p_cal": cal_metrics.get("ad_p", None),
        "hyv_raw": identity_metrics.get("hyv", None),
        "hyv_cal": cal_metrics.get("hyv", None),
        "berk_p_raw": identity_metrics.get("berk_p", None),
        "berk_p_cal": cal_metrics.get("berk_p", None),
        "mad_raw": identity_metrics.get("mad", None),
        "mad_cal": cal_metrics.get("mad", None),
        "log_s_raw": identity_metrics.get("log_s", None),
        "log_s_cal": cal_metrics.get("log_s", None),
        "dss_raw": identity_metrics.get("dss", None),
        "dss_cal": cal_metrics.get("dss", None),
        # v3.1 diagnostics
        "emos_b_at_bound": abs(all_emos.get("b", 1.0) - (-1.0)) < 0.02,
        "sigma_H_mean": round(float(np.mean(sigma_all)), 4),
        "sigma_H_std": round(float(np.std(sigma_all)), 4),
        "regime_counts": regime_counts,
        "cv_diagnostics": {
            s.step_name: s.detail
            for s in result.steps
            if s.step_name == "cv_guard" and s.detail
        }.get("cv_guard", None),
        # v4.0: pipeline trace
        "pipeline_log": pipeline_log,
        # v6.0: top-level isotonic, temperature, and ν for inference
        "isotonic_map": all_entry.get("isotonic_map", None),
        "temp_scale": all_entry.get("temp_scale", None),
        "temp_blend_alpha": all_entry.get("temp_blend_alpha", 0.7),
        "emos_nu": all_emos.get("nu", None),
    }

    return horizon_cal, result.label_thresholds, result


# ===========================================================================
# Build calibration dict from walk-forward records (v4.0 — delegates to pipeline)
# ===========================================================================

def _build_calibration(
    records: Dict[int, List[Dict]],
) -> Optional[Dict]:
    """
    Compute regime-conditional calibration from walk-forward records.

    v4.0 Pipeline Architecture:
      Delegates to _run_pipeline() per horizon.  Each pipeline step is a
      pure function: step(data, prior) → PipelineResult.  The pipeline
      runner chains them and records marginal lift per step.

      Steps: Partition → Beta → EMOS → CVGuard → Threshold

    Backward-compatible output format (same JSON schema as v3.1).
    New: "pipeline_log" per horizon with per-step metric deltas.

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

        # Run the full pipeline for this horizon
        horizon_cal, label_result, pipeline_result = _run_pipeline(recs, H)

        horizons_cal[str(H)] = horizon_cal

        if label_result and len(recs) >= MIN_EVAL_POINTS:
            label_by_horizon[str(H)] = label_result
            if H == 7:
                fallback_label_result = label_result

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
    workers: int = 0,
    eval_days: int = DEFAULT_EVAL_DAYS,
    eval_spacing: int = DEFAULT_EVAL_SPACING,
    assets: Optional[List[str]] = None,
    quiet: bool = False,
    force_collect: bool = False,
    use_precompute: bool = True,
) -> Dict[str, Dict]:
    """
    Run Pass 2 signal calibration for all assets in cache.

    v4.0: Supports records caching.  Walk-forward records are cached
    per asset.  Re-runs use cached records and only re-fit the pipeline
    (<1s/asset vs ~40s/asset).

    Called from tune_ux.py after Pass 1 (BMA tuning).

    Args:
        cache: Full tune cache dict {symbol: params}
        workers: Number of parallel workers (0=auto, uses all CPUs)
        eval_days: Walk-forward window (trading days)
        eval_spacing: Days between eval points
        assets: Optional subset of assets to calibrate
        quiet: Suppress output
        force_collect: Force re-collection even if cached records exist
        use_precompute: v7.5: precompute features once per asset (default True)

    Returns:
        Updated cache with "signals_calibration" added to each asset
    """
    # v4.0: Auto-detect CPU count when workers=0
    if workers <= 0:
        workers = os.cpu_count() or 8
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
                (sym, eval_days, eval_spacing, force_collect, use_precompute)
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
        work_items = [(sym, eval_days, eval_spacing, force_collect, use_precompute) for sym in symbols]

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

    # v7.1: CRPS Skill Score — normalized metric across assets
    crpss = h7.get("crpss", None)
    crpss_str = f" ss={crpss:+.2f}" if crpss is not None else ""

    # v7.3: Multi-diagnostic summary
    pit_p = h7.get("pit_p_cal", None)
    ad_p = h7.get("ad_p_cal", None)
    hyv = h7.get("hyv_cal", None)
    berk_p = h7.get("berk_p_cal", None)

    diag_parts = []
    if pit_p is not None:
        pit_flag = "✓" if pit_p > 0.05 else "✗"
        diag_parts.append(f"PIT={pit_p:.2f}{pit_flag}")
    if hyv is not None:
        hyv_flag = "✓" if hyv < 0 else "✗"
        diag_parts.append(f"Hyv={hyv:.1f}{hyv_flag}")
    if ad_p is not None:
        ad_flag = "✓" if ad_p > 0.05 else "✗"
        diag_parts.append(f"AD={ad_p:.2f}{ad_flag}")
    if berk_p is not None:
        bk_flag = "✓" if berk_p > 0.05 else "✗"
        diag_parts.append(f"Bk={berk_p:.2f}{bk_flag}")
    diag_str = " " + " ".join(diag_parts) if diag_parts else ""

    console.print(
        f"  [{idx}/{total}] {symbol:<12s} "
        f"1w_hit={hit_raw:.0%} "
        f"brier={brier_r:.3f}→{brier_c:.3f}({brier_sign}{brier_delta:.3f}) "
        f"crps={crps_r:.3f}→{crps_c:.3f}({crps_sign}{crps_delta:.3f}){crpss_str}"
        f"{diag_str} "
        f"rgm={len(regimes)} thr=({buy_t:.2f}/{sell_t:.2f})"
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
    pit_pass_count = 0
    ad_pass_count = 0
    hyv_values = []
    berk_pass_count = 0
    total_diag = 0

    for sym, cal in results:
        h7 = cal.get("horizons", {}).get("7", {})
        if "brier_raw" in h7 and "brier_calibrated" in h7:
            brier_improvements.append(h7["brier_raw"] - h7["brier_calibrated"])
        if "crps_raw" in h7 and "crps_calibrated" in h7:
            crps_improvements.append(h7["crps_raw"] - h7["crps_calibrated"])
        # v7.3 diagnostics
        pit_p = h7.get("pit_p_cal")
        ad_p = h7.get("ad_p_cal")
        hyv = h7.get("hyv_cal")
        berk_p = h7.get("berk_p_cal")
        if pit_p is not None:
            total_diag += 1
            if pit_p > 0.05:
                pit_pass_count += 1
            if ad_p is not None and ad_p > 0.05:
                ad_pass_count += 1
            if hyv is not None:
                hyv_values.append(hyv)
            if berk_p is not None and berk_p > 0.05:
                berk_pass_count += 1

    avg_brier_imp = float(np.mean(brier_improvements)) if brier_improvements else 0.0
    avg_crps_imp = float(np.mean(crps_improvements)) if crps_improvements else 0.0
    avg_hyv = float(np.mean(hyv_values)) if hyv_values else 0.0

    # v7.3: Diagnostic summary line
    diag_line = ""
    if total_diag > 0:
        diag_line = (
            f"\nPIT pass: {pit_pass_count}/{total_diag} | "
            f"AD pass: {ad_pass_count}/{total_diag} | "
            f"Berk pass: {berk_pass_count}/{total_diag} | "
            f"Avg Hyv: {avg_hyv:.0f}"
        )

    console.print()
    console.print(Panel(
        f"[bold green]Signal Calibration v{CALIBRATION_VERSION} Complete[/bold green]\n"
        f"Calibrated: {calibrated}/{total} | Failed: {failed} | "
        f"Elapsed: {elapsed:.1f}s\n"
        f"Avg 1w Brier improvement: {avg_brier_imp:+.4f} | "
        f"Avg 1w CRPS improvement: {avg_crps_imp:+.4f}"
        f"{diag_line}\n"
        f"Pipeline: Beta \u2192 SigmaFloor \u2192 EMOS(PIT) \u2192 Magnitude \u2192 CVGuard \u2192 Threshold",
        style="green",
        expand=False,
    ))
    console.print()


# ===========================================================================
# STANDALONE CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Signal Calibration Engine v4.0 — Pipeline Architecture",
    )
    parser.add_argument("--assets", type=str, help="Comma-separated asset symbols")
    parser.add_argument("--eval-days", type=int, default=DEFAULT_EVAL_DAYS)
    parser.add_argument("--eval-spacing", type=int, default=DEFAULT_EVAL_SPACING)
    parser.add_argument("--workers", type=int, default=0,
                        help="Worker count (0=auto, uses all CPUs)")
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: spacing=10 (~50 eval points, 3x faster)")
    parser.add_argument("--force-collect", action="store_true",
                        help="Force re-collection of walk-forward records (ignore cache)")
    parser.add_argument("--no-precompute", action="store_true",
                        help="Disable feature precomputation (strict walk-forward, slower)")
    parser.add_argument("--records-only", action="store_true",
                        help="Only collect and cache records, skip fitting")
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

    w = 1 if args.no_parallel else (args.workers if args.workers > 0 else (os.cpu_count() or 8))

    cache = run_signals_calibration(
        cache,
        workers=w,
        eval_days=args.eval_days,
        eval_spacing=args.eval_spacing,
        assets=asset_list,
        force_collect=args.force_collect,
        use_precompute=not getattr(args, 'no_precompute', False),
    )

    # Save updated cache
    for sym, params in cache.items():
        if "signals_calibration" in params:
            save_tuned_params(sym, params)

    if console:
        console.print("[green]Calibration saved to tune cache.[/green]")


if __name__ == "__main__":
    main()
