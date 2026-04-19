"""
Calibration pipeline: EMOS, vol ratios, DIG weighting, regime q-floors.

Extracted from tune.py (Story 2.1).
"""
import math
import os
import json

import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from tuning.tune_modules.config import (
    GaussianDriftModel,
    PhiGaussianDriftModel,
    PhiStudentTDriftModel,
)
from tuning.tune_modules.utilities import (
    Q_FLOOR_BY_REGIME,
    Q_FLOOR_BIC_LAMBDA,
    compute_vol_proportional_q_floor,
    _log,
)

__all__ = [
    # Constants
    "EMOS_TRAIN_FRAC", "EMOS_IDENTITY",
    "VOL_RATIO_CLAMP_LOW", "VOL_RATIO_CLAMP_HIGH",
    "DIG_BASELINE", "DIG_W_START", "DIG_W_MAX", "DIG_MIN_RECORDS",
    "DIG_FULL_DATA_THRESHOLD",
    "CALIBRATION_REPORT_DIR", "CALIBRATION_REPORT_FILE",
    "CALIBRATION_DEFAULT_START",
    # Private helpers (needed by existing tests)
    "_crps_normal_single", "_emos_crps_objective",
    # Public functions
    "train_emos_parameters",
    "compute_vol_calibration_ratios",
    "compute_dig_per_model", "compute_dig_weight",
    "adjust_bma_weights_with_dig",
    "run_calibration_pipeline", "save_calibration_report",
    "apply_regime_q_floor",
    # Re-exported import
    "assign_regime_labels",
]


# Story 2.2: EMOS Calibration from Walk-Forward Errors
# =========================================================================
# Train per-horizon EMOS parameters via CRPS minimization on walk-forward
# (forecast, realized) pairs.
#
# EMOS (Gneiting 2005) applies:
#   mu_corrected  = a + b * mu_raw
#   sig_corrected = c + d * sig_raw   (c,d >= 0 enforced)
#
# The CRPS for a Normal(mu, sig) distribution is:
#   CRPS(y; mu, sig) = sig * [ z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ]
# where z = (y - mu) / sig.
# =========================================================================

EMOS_TRAIN_FRAC = 0.70   # Use first 70% for training, last 30% for validation
EMOS_IDENTITY = {"a": 0.0, "b": 1.0, "c": 0.0, "d": 1.0}


def _crps_normal_single(y: float, mu: float, sig: float) -> float:
    """CRPS for a single observation against Normal(mu, sig)."""
    if sig <= 0:
        return abs(y - mu)
    from scipy.stats import norm
    z = (y - mu) / sig
    return sig * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / math.sqrt(math.pi))


def _emos_crps_objective(params: np.ndarray, forecasts: np.ndarray,
                         sigmas: np.ndarray, realized: np.ndarray) -> float:
    """Mean CRPS for EMOS-corrected Normal distribution."""
    a, b, c, d = params
    mu_cor = a + b * forecasts
    sig_cor = np.abs(c + d * sigmas)
    sig_cor = np.maximum(sig_cor, 1e-8)
    n = len(realized)
    total = 0.0
    for i in range(n):
        total += _crps_normal_single(realized[i], mu_cor[i], sig_cor[i])
    return total / n


def train_emos_parameters(
    wf_records: list,
    horizons: Optional[List[int]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Train per-horizon EMOS parameters via CRPS minimization.

    Using walk-forward (forecast, realized) pairs, fits the affine EMOS
    correction (a + b*mu, |c + d*sig|) that minimises the CRPS of the
    corrected Normal predictive distribution.

    Training uses the first EMOS_TRAIN_FRAC of records, validation uses
    the remainder.

    Args:
        wf_records: List of WalkForwardRecord (from Story 2.1).
        horizons: Horizons to calibrate (default [1,3,7,21,63]).

    Returns:
        {horizon: {'a', 'b', 'c', 'd', 'crps_train', 'crps_val',
                   'crps_uncorrected', 'crps_improvement_pct', 'n_train', 'n_val'}}
    """
    from scipy.optimize import minimize as sp_minimize

    if horizons is None:
        horizons = [1, 3, 7, 21, 63]

    results: Dict[int, Dict[str, float]] = {}

    for H in horizons:
        h_recs = [r for r in wf_records if r.horizon == H]
        if len(h_recs) < 20:
            results[H] = dict(EMOS_IDENTITY)
            results[H].update({"n_train": 0, "n_val": 0,
                               "crps_train": float("nan"),
                               "crps_val": float("nan"),
                               "crps_uncorrected": float("nan"),
                               "crps_improvement_pct": 0.0})
            continue

        # Sort by date_idx to respect temporal ordering
        h_recs.sort(key=lambda r: r.date_idx)

        # Train/validation split
        n_train = max(10, int(len(h_recs) * EMOS_TRAIN_FRAC))
        train = h_recs[:n_train]
        val = h_recs[n_train:]

        f_train = np.array([r.forecast_ret for r in train])
        s_train = np.array([r.forecast_sig for r in train])
        y_train = np.array([r.realized_ret for r in train])

        # Optimize EMOS params on training set
        x0 = np.array([0.0, 1.0, 0.0, 1.0])
        try:
            opt = sp_minimize(
                _emos_crps_objective, x0, args=(f_train, s_train, y_train),
                method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8},
            )
            a, b, c, d = opt.x
        except Exception:
            a, b, c, d = 0.0, 1.0, 0.0, 1.0

        # Ensure sigma correction is non-negative
        if c + d * np.mean(s_train) < 0:
            c, d = 0.0, 1.0

        crps_train = _emos_crps_objective(np.array([a, b, c, d]), f_train, s_train, y_train)
        crps_uncorrected = _emos_crps_objective(x0, f_train, s_train, y_train)

        # Validation
        crps_val = float("nan")
        n_val = len(val)
        if n_val > 0:
            f_val = np.array([r.forecast_ret for r in val])
            s_val = np.array([r.forecast_sig for r in val])
            y_val = np.array([r.realized_ret for r in val])
            crps_val = _emos_crps_objective(np.array([a, b, c, d]), f_val, s_val, y_val)

        improvement = 0.0
        if crps_uncorrected > 0:
            improvement = (crps_uncorrected - crps_train) / crps_uncorrected * 100.0

        results[H] = {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "d": float(d),
            "crps_train": float(crps_train),
            "crps_val": float(crps_val),
            "crps_uncorrected": float(crps_uncorrected),
            "crps_improvement_pct": float(improvement),
            "n_train": n_train,
            "n_val": n_val,
        }

    return results


# =========================================================================
# Story 2.3: Realized Volatility Feedback for Sigma Calibration
# =========================================================================
# Compute per-horizon vol calibration ratio from walk-forward data.
#
# vol_ratio_H = realized_std(returns_H) / mean(forecast_sig_H)
#
# If vol_ratio > 1: system is overconfident (underestimates uncertainty)
# If vol_ratio < 1: system is too cautious (overestimates uncertainty)
#
# The ratio is clamped to [0.5, 2.0] for stability.
# =========================================================================

VOL_RATIO_CLAMP_LOW = 0.5
VOL_RATIO_CLAMP_HIGH = 2.0


def compute_vol_calibration_ratios(
    wf_records: list,
    horizons: Optional[List[int]] = None,
) -> Dict[int, float]:
    """
    Compute per-horizon volatility calibration ratio from walk-forward data.

    vol_ratio_H = std(realized_ret_H) / mean(forecast_sig_H)

    If vol_ratio > 1.0 the system is overconfident.
    If vol_ratio < 1.0 the system is too cautious.

    Args:
        wf_records: List of WalkForwardRecord from Story 2.1.
        horizons: Horizons (default [1,3,7,21,63]).

    Returns:
        {horizon: vol_ratio} clamped to [0.5, 2.0].
    """
    if horizons is None:
        horizons = [1, 3, 7, 21, 63]

    ratios: Dict[int, float] = {}
    for H in horizons:
        h_recs = [r for r in wf_records if r.horizon == H]
        if len(h_recs) < 10:
            ratios[H] = 1.0
            continue
        realized = np.array([r.realized_ret for r in h_recs])
        forecast_sig = np.array([r.forecast_sig for r in h_recs])
        realized_vol = float(np.std(realized, ddof=1))
        mean_sig = float(np.mean(forecast_sig))
        if mean_sig <= 0:
            ratios[H] = 1.0
            continue
        ratio = realized_vol / mean_sig
        ratios[H] = float(np.clip(ratio, VOL_RATIO_CLAMP_LOW, VOL_RATIO_CLAMP_HIGH))
    return ratios


# ─── Story 2.4: Directional Information Gain (DIG) for BMA Weights ──────
DIG_BASELINE = 0.5          # random walk hit rate
DIG_W_START = 0.10          # DIG weight when data is sparse
DIG_W_MAX = 0.25            # DIG weight at full data accumulation
DIG_MIN_RECORDS = 30        # minimum records per model for meaningful DIG
DIG_FULL_DATA_THRESHOLD = 200  # records at which w_dig reaches DIG_W_MAX


def compute_dig_per_model(
    wf_records: list,
    horizons: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute Directional Information Gain (DIG) per model from walk-forward data.

    DIG_m = hit_rate_m - 0.5

    Where hit_rate_m = P(sign(forecast_ret) == sign(realized_ret)).
    Positive DIG means directional edge over random. Negative DIG means
    worse than random (penalized in BMA weighting).

    Args:
        wf_records: List of WalkForwardRecord from Story 2.1.
        horizons: Horizons to include (default all).

    Returns:
        {model_or_horizon_key: DIG}. Uses "ensemble" key for the
        overall ensemble, and "H={h}" for per-horizon.
    """
    if horizons is not None:
        recs = [r for r in wf_records if r.horizon in horizons]
    else:
        recs = list(wf_records)

    if len(recs) < DIG_MIN_RECORDS:
        return {"ensemble": 0.0}

    hit_count = sum(1 for r in recs if r.hit)
    hit_rate = hit_count / len(recs)
    result: Dict[str, float] = {"ensemble": hit_rate - DIG_BASELINE}

    # Per-horizon DIG
    horizon_set = set(r.horizon for r in recs)
    for h in sorted(horizon_set):
        h_recs = [r for r in recs if r.horizon == h]
        if len(h_recs) >= DIG_MIN_RECORDS:
            h_hits = sum(1 for r in h_recs if r.hit)
            result[f"H={h}"] = h_hits / len(h_recs) - DIG_BASELINE
    return result


def compute_dig_weight(n_records: int) -> float:
    """
    Compute adaptive DIG weight that grows from DIG_W_START to DIG_W_MAX
    as walk-forward data accumulates.

    Linear ramp: w_dig = DIG_W_START + (DIG_W_MAX - DIG_W_START) * frac
    where frac = min(1, n_records / DIG_FULL_DATA_THRESHOLD).

    Args:
        n_records: Total walk-forward records available.

    Returns:
        DIG weight in [DIG_W_START, DIG_W_MAX].
    """
    if n_records <= 0:
        return DIG_W_START
    frac = min(1.0, n_records / DIG_FULL_DATA_THRESHOLD)
    return DIG_W_START + (DIG_W_MAX - DIG_W_START) * frac


def adjust_bma_weights_with_dig(
    raw_weights: Dict[str, float],
    dig_values: Dict[str, float],
    n_records: int,
) -> Dict[str, float]:
    """
    Adjust BMA weights using Directional Information Gain (DIG).

    Models with DIG > 0 get a proportional boost. Models with DIG < 0
    get penalized. The adjustment is multiplicative:

    adjusted_w_m = raw_w_m * (1 + w_dig * DIG_m_standardized)

    Where DIG_m_standardized is robust z-score across the ensemble.

    Args:
        raw_weights: BMA weights from the 6-component scoring system.
        dig_values: Per-model DIG values (model_name -> DIG).
        n_records: Walk-forward record count (for adaptive weighting).

    Returns:
        Adjusted and renormalized BMA weights.
    """
    if not dig_values or n_records < DIG_MIN_RECORDS:
        return dict(raw_weights)

    w_dig = compute_dig_weight(n_records)

    # Robust standardize DIG across models present in raw_weights
    model_digs = {}
    for m in raw_weights:
        if m in dig_values:
            model_digs[m] = dig_values[m]
    if not model_digs:
        return dict(raw_weights)

    vals = np.array(list(model_digs.values()))
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    if mad < 1e-10:
        mad = float(np.std(vals)) if np.std(vals) > 1e-10 else 1.0

    adjusted = {}
    for m, w in raw_weights.items():
        if m in model_digs:
            dig_std = (model_digs[m] - med) / mad
            dig_std = float(np.clip(dig_std, -3.0, 3.0))  # winsorize
            multiplier = 1.0 + w_dig * dig_std
            multiplier = max(0.1, multiplier)  # floor to prevent zeroing
            adjusted[m] = w * multiplier
        else:
            adjusted[m] = w

    # Renormalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {m: v / total for m, v in adjusted.items()}
    return adjusted


# ─── Story 2.6: Automated Calibration Pipeline ──────────────────────────
CALIBRATION_REPORT_DIR = "src/data/calibration"
CALIBRATION_REPORT_FILE = "calibration_report.json"
CALIBRATION_DEFAULT_START = "2024-01-01"


def run_calibration_pipeline(
    assets: List[str],
    cache: Dict[str, Dict],
    cache_json: str = "src/data/tune",
    start_date: str = CALIBRATION_DEFAULT_START,
    horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Run the full calibration pipeline for a list of assets.

    For each asset:
        1. Run walk-forward backtest (Story 2.1)
        2. Train EMOS parameters (Story 2.2)
        3. Compute volatility calibration ratios (Story 2.3)
        4. Compute DIG (Story 2.4)
        5. Compute P&L attribution (Story 2.5)
        6. Update tune cache with calibration data

    Args:
        assets: List of asset symbols.
        cache: Existing tune cache (modified in-place).
        cache_json: Path to cache directory.
        start_date: Walk-forward start date.
        horizons: Horizons for calibration (default [1,3,7,21,63]).

    Returns:
        Calibration report dict with per-asset results and summary.
    """
    from decision.signals import (
        run_walk_forward_backtest, compute_pnl_attribution,
        WF_HORIZONS,
    )

    if horizons is None:
        horizons = WF_HORIZONS

    report: Dict[str, Any] = {
        "assets": {},
        "summary": {},
    }

    n_nontrivial_emos = 0
    total_assets = len(assets)

    for i, asset in enumerate(assets, 1):
        print(f"\n[Calibrate {i}/{total_assets}] {asset}")
        asset_report: Dict[str, Any] = {"asset": asset, "status": "pending"}

        try:
            # Step 1: Walk-forward
            wf = run_walk_forward_backtest(
                asset, start_date=start_date, horizons=horizons,
            )
            n_records = len(wf.records)
            asset_report["n_records"] = n_records
            asset_report["hit_rate"] = dict(wf.hit_rate)

            if n_records < 10:
                asset_report["status"] = "skipped_insufficient_data"
                report["assets"][asset] = asset_report
                continue

            # Step 2: EMOS
            emos_params = train_emos_parameters(wf.records, horizons=horizons)
            asset_report["emos_params"] = {}
            for h, ep in emos_params.items():
                is_identity = (abs(ep.get("a", 0)) < 0.01
                               and abs(ep.get("b", 1) - 1.0) < 0.01
                               and abs(ep.get("c", 0)) < 0.01
                               and abs(ep.get("d", 1) - 1.0) < 0.01)
                if not is_identity:
                    n_nontrivial_emos += 1
                asset_report["emos_params"][h] = {
                    "a": ep.get("a", 0), "b": ep.get("b", 1),
                    "c": ep.get("c", 0), "d": ep.get("d", 1),
                    "crps_train": ep.get("crps_train", None),
                    "crps_val": ep.get("crps_val", None),
                    "is_identity": is_identity,
                }

            # Step 3: Vol ratios
            vol_ratios = compute_vol_calibration_ratios(wf.records, horizons=horizons)
            asset_report["vol_ratios"] = {str(k): v for k, v in vol_ratios.items()}

            # Step 4: DIG
            dig = compute_dig_per_model(wf.records, horizons=horizons)
            asset_report["dig"] = dig

            # Step 5: P&L attribution
            pnl = compute_pnl_attribution(wf.records, horizons=horizons)
            asset_report["pnl_attribution"] = {
                str(k): v for k, v in pnl.items()
            }

            # Step 6: Update cache
            if asset not in cache:
                cache[asset] = {}
            if "calibration" not in cache[asset]:
                cache[asset]["calibration"] = {}
            cache[asset]["calibration"]["emos"] = {
                str(h): {k: v for k, v in ep.items()
                         if k in ("a", "b", "c", "d")}
                for h, ep in emos_params.items()
            }
            cache[asset]["calibration"]["vol_ratios"] = {
                str(k): v for k, v in vol_ratios.items()
            }
            cache[asset]["calibration"]["dig"] = dig

            asset_report["status"] = "success"

        except Exception as e:
            asset_report["status"] = "failed"
            asset_report["error"] = str(e)

        report["assets"][asset] = asset_report

    # Summary
    n_success = sum(1 for r in report["assets"].values() if r["status"] == "success")
    n_failed = sum(1 for r in report["assets"].values() if r["status"] == "failed")
    n_skipped = sum(1 for r in report["assets"].values() if r["status"].startswith("skipped"))
    report["summary"] = {
        "total_assets": total_assets,
        "success": n_success,
        "failed": n_failed,
        "skipped": n_skipped,
        "n_nontrivial_emos": n_nontrivial_emos,
    }

    return report


def save_calibration_report(report: Dict[str, Any],
                            output_dir: str = CALIBRATION_REPORT_DIR) -> str:
    """Save calibration report to JSON. Returns path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, CALIBRATION_REPORT_FILE)

    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(path, "w") as f:
        json.dump(report, f, indent=2, cls=_Encoder)
    return path


def apply_regime_q_floor(
    models: Dict[str, Dict],
    regime: int,
    returns: np.ndarray,
    vol: np.ndarray,
    asset_vol: float = 0.0,
) -> Tuple[int, int]:
    """
    Apply regime-conditional q floor to all fitted models in-place.

    For each model where q_mle < q_floor:
      1. Store original q as q_mle_original
      2. Set q = q_floor
      3. Re-run filter to get updated log-likelihood
      4. Recompute BIC with penalty term
      5. Set q_floor_applied = True

    This preserves the optimiser's choice of c and phi while only lifting q.
    The BIC adjustment ensures model selection is not distorted by the floor.

    Story 1.7: When asset_vol > 0, uses vol-proportional q floor via
    compute_vol_proportional_q_floor().  Falls back to Q_FLOOR_BY_REGIME
    when asset_vol is not provided.

    Args:
        models: Dict of model_name -> model_dict (modified in-place)
        regime: Regime index (0-4) for floor lookup
        returns: Regime-specific returns (for re-filtering)
        vol: Regime-specific volatility (for re-filtering)
        asset_vol: Annualized asset volatility (for vol-proportional floor)

    Returns:
        (n_floored, n_total): count of models where floor was applied
    """
    if asset_vol > 0:
        q_floor = compute_vol_proportional_q_floor(regime, asset_vol)
    else:
        q_floor = Q_FLOOR_BY_REGIME.get(regime, 0.0)
    if q_floor <= 0:
        return 0, len(models)

    n_floored = 0
    n_total = 0
    n_obs = len(returns)

    for model_name, info in models.items():
        if not info.get("fit_success", False):
            continue
        n_total += 1

        q_mle = info.get("q")
        if q_mle is None or q_mle >= q_floor:
            info["q_floor_applied"] = False
            continue

        # Floor binds: override q
        n_floored += 1
        info["q_mle_original"] = float(q_mle)
        info["q"] = float(q_floor)
        info["q_floor_applied"] = True
        info["q_floor_regime"] = int(regime)
        info["q_floor_value"] = float(q_floor)

        # Re-run filter with floored q to get updated log-likelihood
        c_val = info.get("c", 1.0)
        phi_val = info.get("phi")
        nu_val = info.get("nu")

        try:
            if nu_val is not None and phi_val is not None:
                # Student-t model
                _, _, ll_new = PhiStudentTDriftModel.filter_phi(
                    returns, vol, q_floor, c_val, phi_val, nu_val
                )
            elif phi_val is not None:
                # Phi-Gaussian model
                _, _, ll_new = PhiGaussianDriftModel.filter_phi(
                    returns, vol, q_floor, c_val, phi_val
                )
            else:
                # Gaussian model
                _, _, ll_new = GaussianDriftModel.filter(
                    returns, vol, q_floor, c_val
                )

            # Update log-likelihood
            old_ll = info.get("log_likelihood", ll_new)
            info["log_likelihood"] = float(ll_new)
            info["mean_log_likelihood"] = float(ll_new / max(n_obs, 1))

            # Recompute BIC with penalty for floor deviation
            n_params = info.get("n_params", 3)
            from calibration.model_selection import compute_bic as _compute_bic
            base_bic = _compute_bic(ll_new, n_params, n_obs)

            # BIC penalty: penalise for deviating from MLE-optimal q
            # log(q_floor / q_mle) is always positive when floor binds
            bic_penalty = Q_FLOOR_BIC_LAMBDA * math.log(q_floor / max(q_mle, 1e-15))
            info["bic"] = float(base_bic + bic_penalty)
            info["bic_floor_penalty"] = float(bic_penalty)

        except Exception as e:
            # If re-filtering fails, keep original BIC but still apply floor
            info["q_floor_refilter_error"] = str(e)

    return n_floored, n_total


# =============================================================================
# REGIME CLASSIFICATION FUNCTION
# Story 4.2: Imported from shared module models.regime
# =============================================================================
from models.regime import assign_regime_labels
