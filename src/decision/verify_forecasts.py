#!/usr/bin/env python3
"""
===============================================================================
FORECAST VERIFICATION ENGINE (Enhanced for signals.py debugging)
===============================================================================

Walk-forward verification of ensemble_forecast() predictions against realized
returns. Tests whether tuned parameters produce accurate signals.

ENHANCED: Collects full diagnostic data needed for debugging signals.py:
  - Per-eval-point regime detection + realized volatility
  - Confidence level tracking from ensemble_forecast
  - Temporal accuracy (early vs late — staleness detection)
  - Per-regime and per-model accuracy breakdowns
  - Worst misses investigation with dates for root-cause analysis
  - Error distribution stats (skew, kurtosis, tail behavior)
  - Deep tune cache extraction (phi, nu, BMA weight concentration)
  - JSON export for scripting and next-step analysis

Usage:
  make verify                             # Full universe, 252 days
  make verify-quick                       # 8 key assets, 90 days
  python src/decision/verify_forecasts.py --assets SPY,AAPL --eval-days 120
  python src/decision/verify_forecasts.py --export-json /tmp/verify.json
  python src/decision/verify_forecasts.py --worst 20

===============================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup – follow project convention
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
# Rich imports
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    from rich.rule import Rule
    from rich.align import Align
    from rich.columns import Columns

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console(force_terminal=True, color_system="truecolor", width=200) if HAS_RICH else None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TUNE_CACHE_DIR = Path(os.path.join(REPO_ROOT, "src", "data", "tune"))
PRICE_CACHE_DIR = Path(os.path.join(REPO_ROOT, "src", "data", "prices"))
HC_BUY_DIR = Path(os.path.join(REPO_ROOT, "src", "data", "high_conviction", "buy"))
HC_SELL_DIR = Path(os.path.join(REPO_ROOT, "src", "data", "high_conviction", "sell"))

VERIFY_HORIZONS = [7, 30]
DEFAULT_EVAL_DAYS = 252
DEFAULT_EVAL_SPACING = 5
MIN_EVAL_POINTS = 5
HC_THRESHOLD = 2.0
MIN_PRICE_POINTS = 120

GRADE_WEIGHTS = {
    "hit_7d": 0.35,
    "hit_30d": 0.25,
    "mae_7d": 0.15,
    "corr_30d": 0.10,
    "pit": 0.15,
}


# ---------------------------------------------------------------------------
# Helper: detect asset type from symbol
# ---------------------------------------------------------------------------
def _detect_asset_type(symbol: str) -> str:
    """Infer asset_type for ensemble_forecast from ticker pattern."""
    s = symbol.upper()
    if any(x in s for x in ["BTC", "ETH", "DOGE", "SOL", "ADA"]):
        return "crypto"
    if s.endswith("=X") or "JPY" in s or "USD" in s.split("=")[0][-3:]:
        if not any(s.startswith(p) for p in ["BTC", "ETH"]):
            return "currency"
    if s.endswith("=F"):
        return "metal"
    metal_tickers = {
        "GLD", "SLV", "GDX", "GDXJ", "SIL", "SLVR", "GOLD", "NEM",
        "AEM", "WPM", "PAAS", "AG", "KGC", "FNV", "RGLD", "MAG",
        "EXK", "CDE", "HL", "FSM", "SILV",
    }
    if s in metal_tickers:
        return "metal"
    return "equity"


# ---------------------------------------------------------------------------
# Helper: regime detection (mirror of market_temperature._regime_detect)
# ---------------------------------------------------------------------------
def _regime_detect(returns: np.ndarray) -> str:
    """Detect market regime from return characteristics."""
    try:
        if len(returns) < 20:
            return "calm"
        vol = np.std(returns[-20:]) * np.sqrt(252)
        mom = np.sum(returns[-20:])
        autocorr = (
            np.corrcoef(returns[-21:-1], returns[-20:])[0, 1]
            if len(returns) >= 21
            else 0
        )
        if vol > 0.30:
            return "volatile"
        elif abs(mom) > 0.10:
            return "trending"
        elif autocorr < -0.2:
            return "mean_reverting"
        else:
            return "calm"
    except Exception:
        return "calm"


# ---------------------------------------------------------------------------
# Helper: load tune quality from cache (deep extraction)
# ---------------------------------------------------------------------------
def _load_tune_quality(symbol: str) -> Dict[str, Any]:
    """Load model quality metrics from the tune cache JSON.

    Enhanced: extracts phi, BMA weight concentration, model count, regime info.
    """
    safe = symbol.replace("/", "_").replace("=", "_").replace(":", "_").upper()
    path = TUNE_CACHE_DIR / f"{safe}.json"
    result = {
        "best_model": "—",
        "pit": None,
        "crps": None,
        "bic": None,
        "nu": None,
        "phi": None,
        "grade": "—",
        "n_models": 0,
        "bma_concentration": None,  # Herfindahl index of model weights
        "top_weight": None,         # Weight of best model
        "has_regime": False,
        "n_regimes": 0,
        "gamma_vov": None,
    }
    if not path.exists():
        return result
    try:
        with open(path) as f:
            d = json.load(f)
        g = d.get("global", {})
        best_model = g.get("best_model", "—")
        result["best_model"] = best_model
        result["pit"] = g.get("pit_ks_pvalue")
        result["crps"] = g.get("crps")
        result["bic"] = g.get("bic")
        result["nu"] = g.get("nu")
        result["phi"] = g.get("phi")
        result["grade"] = g.get("pit_calibration_grade", "—")
        result["gamma_vov"] = g.get("gamma_vov")

        # CRPS often None at global level; grab from best model
        models = g.get("models", {})
        if isinstance(models, dict):
            result["n_models"] = sum(
                1 for m in models.values()
                if isinstance(m, dict) and m.get("fit_success", False)
            )
            if result["crps"] is None and best_model and best_model in models:
                result["crps"] = models[best_model].get("crps")

        # BMA weight concentration (Herfindahl index)
        weights = g.get("model_posterior") or g.get("model_weights")
        if isinstance(weights, dict) and weights:
            w_vals = [float(v) for v in weights.values() if v and float(v) > 0]
            if w_vals:
                total_w = sum(w_vals)
                if total_w > 0:
                    normed = [w / total_w for w in w_vals]
                    result["bma_concentration"] = sum(w ** 2 for w in normed)
                    result["top_weight"] = max(normed)

        # Regime info
        regime_data = d.get("regime", {})
        if isinstance(regime_data, dict) and regime_data:
            result["has_regime"] = True
            result["n_regimes"] = len(regime_data)

    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Helper: load price series
# ---------------------------------------------------------------------------
def _load_prices(symbol: str) -> Optional[pd.Series]:
    """Load close prices from disk cache CSV."""
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
        for col in ["Close", "Adj Close"]:
            if col in df.columns:
                px = pd.to_numeric(df[col], errors="coerce").dropna()
                if isinstance(px, pd.DataFrame):
                    px = px.iloc[:, 0]
                if len(px) > 0:
                    return px
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core: verify a single asset (worker function)
# ---------------------------------------------------------------------------
def verify_single_asset(
    args_tuple: Tuple,
) -> Tuple[str, Optional[Dict], Optional[Dict], Optional[str]]:
    """
    Walk-forward evaluation of ensemble_forecast for one asset.

    Enhanced: collects per-eval-point regime, vol, confidence, and all
    individual predictions for worst-miss analysis.

    Returns: (symbol, metrics_dict, tune_quality_dict, error_or_None)
    """
    symbol, eval_days, eval_spacing = args_tuple

    import warnings
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

        from decision.market_temperature import (
            ensemble_forecast,
            get_forecast_by_horizon,
            get_forecast_confidence,
        )

        prices = _load_prices(symbol)
        if prices is None or len(prices) < MIN_PRICE_POINTS:
            return symbol, None, _load_tune_quality(symbol), "No price data"

        tune_q = _load_tune_quality(symbol)
        asset_type = _detect_asset_type(symbol)

        n = len(prices)
        start_idx = max(MIN_PRICE_POINTS, n - eval_days)

        eval_records_7d: List[Dict] = []
        eval_records_30d: List[Dict] = []

        idx = start_idx
        while idx < n:
            prices_trunc = prices.iloc[:idx]
            eval_date = prices.index[idx - 1]

            if len(prices_trunc) < 60:
                idx += eval_spacing
                continue

            # Compute regime and realized vol at this eval point
            log_rets = np.log(prices_trunc / prices_trunc.shift(1)).dropna().values
            regime = _regime_detect(log_rets)
            realized_vol = float(np.std(log_rets[-20:]) * np.sqrt(252)) if len(log_rets) >= 20 else 0.0

            try:
                result = ensemble_forecast(
                    prices_trunc,
                    asset_type=asset_type,
                    asset_name=symbol,
                )
                confidence = get_forecast_confidence(result)
            except Exception:
                idx += eval_spacing
                continue

            pred_7d = get_forecast_by_horizon(result, 7)
            pred_30d = get_forecast_by_horizon(result, 30)
            # Also grab 1d and 3d for cross-horizon consistency check
            pred_1d = get_forecast_by_horizon(result, 1)
            pred_3d = get_forecast_by_horizon(result, 3)

            # Progress fraction through eval window (0.0 = earliest, 1.0 = latest)
            progress_frac = (idx - start_idx) / max(1, n - 1 - start_idx)

            for horizon, pred, records in [
                (7, pred_7d, eval_records_7d),
                (30, pred_30d, eval_records_30d),
            ]:
                future_idx = idx - 1 + horizon
                best_future_idx = None
                for offset in range(0, 6):
                    candidate = future_idx + offset
                    if candidate < n:
                        best_future_idx = candidate
                        break
                if best_future_idx is not None and best_future_idx < n:
                    price_now = float(prices.iloc[idx - 1])
                    price_future = float(prices.iloc[best_future_idx])
                    if price_now > 0 and not np.isnan(price_now) and not np.isnan(price_future):
                        actual_pct = (price_future / price_now - 1.0) * 100.0
                        records.append({
                            "date": str(eval_date.date()),
                            "predicted": pred,
                            "actual": actual_pct,
                            "pred_dir": 1 if pred > 0 else (-1 if pred < 0 else 0),
                            "actual_dir": 1 if actual_pct > 0 else (-1 if actual_pct < 0 else 0),
                            "is_hc": abs(pred) >= HC_THRESHOLD,
                            "regime": regime,
                            "realized_vol": realized_vol,
                            "confidence": confidence,
                            "progress": progress_frac,
                            "price": price_now,
                            "error": pred - actual_pct,
                            # Cross-horizon data for consistency
                            "pred_1d": pred_1d,
                            "pred_3d": pred_3d,
                        })

            idx += eval_spacing

        metrics = _compute_metrics(eval_records_7d, eval_records_30d)
        if metrics is None:
            return symbol, None, tune_q, f"Too few eval points (<{MIN_EVAL_POINTS})"

        return symbol, metrics, tune_q, None

    except Exception as e:
        return symbol, None, _load_tune_quality(symbol), str(e)


def _compute_metrics(
    records_7d: List[Dict], records_30d: List[Dict]
) -> Optional[Dict]:
    """Compute forecast accuracy metrics from evaluation records.

    Enhanced: adds regime breakdown, temporal accuracy, error distribution,
    worst misses, magnitude calibration, and raw records for JSON export.
    """

    def _horizon_metrics(records: List[Dict], horizon_label: str) -> Dict:
        if len(records) < MIN_EVAL_POINTS:
            return {}

        preds = np.array([r["predicted"] for r in records])
        actuals = np.array([r["actual"] for r in records])
        errors = np.array([r["error"] for r in records])
        pred_dirs = np.array([r["pred_dir"] for r in records])
        actual_dirs = np.array([r["actual_dir"] for r in records])

        nonzero = pred_dirs != 0
        hit_rate = float(np.mean(pred_dirs[nonzero] == actual_dirs[nonzero])) if nonzero.sum() > 0 else 0.5

        mae = float(np.mean(np.abs(errors)))
        bias = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        # Correlation
        if np.std(preds) > 1e-12 and np.std(actuals) > 1e-12:
            corr = float(np.corrcoef(preds, actuals)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        # Brier score
        p_up_pred = np.clip((preds / 10.0) + 0.5, 0.0, 1.0)
        actual_up = (actuals > 0).astype(float)
        brier = float(np.mean((p_up_pred - actual_up) ** 2))

        # High-conviction
        hc_mask = np.array([r["is_hc"] for r in records])
        if hc_mask.sum() >= 3:
            hc_nonzero = hc_mask & nonzero
            hc_hit = float(np.mean(pred_dirs[hc_nonzero] == actual_dirs[hc_nonzero])) if hc_nonzero.sum() > 0 else None
            hc_count = int(hc_mask.sum())
        else:
            hc_hit = None
            hc_count = 0

        # Profit factor
        correct_mask = nonzero & (pred_dirs == actual_dirs)
        wrong_mask = nonzero & (pred_dirs != actual_dirs)
        sum_correct = float(np.abs(actuals[correct_mask]).sum()) if correct_mask.sum() > 0 else 0.0
        sum_wrong = float(np.abs(actuals[wrong_mask]).sum()) if wrong_mask.sum() > 0 else 0.001
        profit_factor = sum_correct / sum_wrong

        # ---- ENHANCED DIAGNOSTICS ----

        # Magnitude calibration: avg(|predicted|) / avg(|actual|)
        avg_pred_mag = float(np.mean(np.abs(preds)))
        avg_actual_mag = float(np.mean(np.abs(actuals)))
        mag_ratio = avg_pred_mag / avg_actual_mag if avg_actual_mag > 1e-6 else None

        # Error distribution stats
        error_std = float(np.std(errors))
        n_pts = len(errors)
        if n_pts >= 8:
            # Skewness and kurtosis
            m3 = float(np.mean((errors - bias) ** 3))
            m4 = float(np.mean((errors - bias) ** 4))
            error_skew = m3 / (error_std ** 3) if error_std > 1e-12 else 0.0
            error_kurt = (m4 / (error_std ** 4)) - 3.0 if error_std > 1e-12 else 0.0
        else:
            error_skew = 0.0
            error_kurt = 0.0

        # Temporal accuracy: first half vs second half
        progress_vals = np.array([r["progress"] for r in records])
        early_mask = progress_vals < 0.5
        late_mask = progress_vals >= 0.5
        early_hit = None
        late_hit = None
        if early_mask.sum() >= 3:
            e_nz = early_mask & nonzero
            if e_nz.sum() > 0:
                early_hit = float(np.mean(pred_dirs[e_nz] == actual_dirs[e_nz]))
        if late_mask.sum() >= 3:
            l_nz = late_mask & nonzero
            if l_nz.sum() > 0:
                late_hit = float(np.mean(pred_dirs[l_nz] == actual_dirs[l_nz]))

        # Per-regime accuracy
        regime_acc = {}
        regimes = [r["regime"] for r in records]
        for regime in set(regimes):
            r_mask = np.array([r["regime"] == regime for r in records])
            r_nz = r_mask & nonzero
            if r_nz.sum() >= 3:
                r_hit = float(np.mean(pred_dirs[r_nz] == actual_dirs[r_nz]))
                r_mae = float(np.mean(np.abs(errors[r_mask])))
                regime_acc[regime] = {
                    "hit_rate": r_hit,
                    "mae": r_mae,
                    "n": int(r_mask.sum()),
                }

        # Per-confidence accuracy
        conf_acc = {}
        confs = [r["confidence"] for r in records]
        for conf in set(confs):
            c_mask = np.array([r["confidence"] == conf for r in records])
            c_nz = c_mask & nonzero
            if c_nz.sum() >= 3:
                c_hit = float(np.mean(pred_dirs[c_nz] == actual_dirs[c_nz]))
                conf_acc[conf] = {"hit_rate": c_hit, "n": int(c_mask.sum())}

        # Vol-bucket accuracy (low/med/high vol)
        vols = np.array([r["realized_vol"] for r in records])
        vol_33, vol_66 = np.percentile(vols, [33, 66]) if len(vols) >= 6 else (0, 0)
        vol_acc = {}
        for label, mask_fn in [
            ("low_vol", lambda v: v <= vol_33),
            ("mid_vol", lambda v: (v > vol_33) & (v <= vol_66)),
            ("high_vol", lambda v: v > vol_66),
        ]:
            v_mask = np.array([mask_fn(r["realized_vol"]) for r in records])
            v_nz = v_mask & nonzero
            if v_nz.sum() >= 3:
                v_hit = float(np.mean(pred_dirs[v_nz] == actual_dirs[v_nz]))
                v_mae = float(np.mean(np.abs(errors[v_mask])))
                vol_acc[label] = {"hit_rate": v_hit, "mae": v_mae, "n": int(v_mask.sum())}

        # Worst misses (sorted by absolute error, descending)
        sorted_by_error = sorted(records, key=lambda r: abs(r["error"]), reverse=True)
        worst_misses = []
        for r in sorted_by_error[:10]:
            worst_misses.append({
                "date": r["date"],
                "predicted": r["predicted"],
                "actual": r["actual"],
                "error": r["error"],
                "regime": r["regime"],
                "realized_vol": r["realized_vol"],
                "confidence": r["confidence"],
            })

        # Cross-horizon consistency: correlation between 1d/3d/7d predictions
        preds_1d = np.array([r.get("pred_1d", 0) for r in records])
        preds_3d = np.array([r.get("pred_3d", 0) for r in records])
        if np.std(preds) > 1e-12 and np.std(preds_3d) > 1e-12:
            cross_corr_3d = float(np.corrcoef(preds_3d, preds)[0, 1])
            if np.isnan(cross_corr_3d):
                cross_corr_3d = 0.0
        else:
            cross_corr_3d = None

        return {
            "n": len(records),
            "hit_rate": hit_rate,
            "mae": mae,
            "bias": bias,
            "rmse": rmse,
            "corr": corr,
            "brier": brier,
            "hc_hit": hc_hit,
            "hc_count": hc_count,
            "profit_factor": profit_factor,
            # Enhanced
            "mag_ratio": mag_ratio,
            "error_std": error_std,
            "error_skew": error_skew,
            "error_kurt": error_kurt,
            "early_hit": early_hit,
            "late_hit": late_hit,
            "regime_acc": regime_acc,
            "conf_acc": conf_acc,
            "vol_acc": vol_acc,
            "worst_misses": worst_misses,
            "cross_corr_3d": cross_corr_3d,
            # Raw records for JSON export
            "records": records,
        }

    m7 = _horizon_metrics(records_7d, "7d")
    m30 = _horizon_metrics(records_30d, "30d")

    if not m7 and not m30:
        return None

    return {"7d": m7, "30d": m30}


# ---------------------------------------------------------------------------
# Composite grade
# ---------------------------------------------------------------------------
def _composite_score(metrics: Dict, tune_q: Dict) -> Tuple[float, str]:
    """Compute composite score [0, 1] and letter grade."""
    score = 0.0
    m7 = metrics.get("7d", {})
    m30 = metrics.get("30d", {})

    if "hit_rate" in m7:
        s = max(0.0, min(1.0, (m7["hit_rate"] - 0.40) / 0.30))
        score += s * GRADE_WEIGHTS["hit_7d"]
    if "hit_rate" in m30:
        s = max(0.0, min(1.0, (m30["hit_rate"] - 0.40) / 0.30))
        score += s * GRADE_WEIGHTS["hit_30d"]
    if "mae" in m7:
        s = max(0.0, min(1.0, 1.0 - (m7["mae"] - 2.0) / 6.0))
        score += s * GRADE_WEIGHTS["mae_7d"]
    if "corr" in m30:
        s = max(0.0, min(1.0, m30["corr"] / 0.30))
        score += s * GRADE_WEIGHTS["corr_30d"]
    pit = tune_q.get("pit")
    if pit is not None:
        s = max(0.0, min(1.0, pit))
        score += s * GRADE_WEIGHTS["pit"]

    total_weight = sum(
        v
        for k, v in GRADE_WEIGHTS.items()
        if (k == "hit_7d" and "hit_rate" in m7)
        or (k == "hit_30d" and "hit_rate" in m30)
        or (k == "mae_7d" and "mae" in m7)
        or (k == "corr_30d" and "corr" in m30)
        or (k == "pit" and pit is not None)
    )
    if total_weight > 0:
        score = score / total_weight

    if score >= 0.80:
        grade = "A"
    elif score >= 0.65:
        grade = "B"
    elif score >= 0.50:
        grade = "C"
    elif score >= 0.35:
        grade = "D"
    else:
        grade = "F"
    return score, grade


# ---------------------------------------------------------------------------
# High-conviction signal verification
# ---------------------------------------------------------------------------
def _verify_high_conviction_signals() -> List[Dict]:
    """Verify saved high-conviction signals against actual outcomes."""
    results = []
    for directory, signal_type in [(HC_BUY_DIR, "BUY"), (HC_SELL_DIR, "SELL")]:
        if not directory.exists():
            continue
        for path in directory.glob("*.json"):
            if path.name == "manifest.json":
                continue
            try:
                with open(path) as f:
                    sig = json.load(f)
                ticker = sig.get("ticker")
                gen_at = sig.get("generated_at")
                horizon = sig.get("horizon_days")
                prob_up = sig.get("probability_up")
                exp_ret = sig.get("expected_return_pct")
                if not all([ticker, gen_at, horizon]):
                    continue
                prices = _load_prices(ticker)
                if prices is None:
                    continue
                sig_date = pd.Timestamp(gen_at).normalize()
                idx = prices.index.searchsorted(sig_date)
                if idx >= len(prices):
                    idx = len(prices) - 1
                future_idx = idx + horizon
                if future_idx >= len(prices):
                    results.append({
                        "ticker": ticker, "type": signal_type, "horizon": horizon,
                        "date": str(sig_date.date()), "prob_up": prob_up,
                        "exp_ret": exp_ret, "actual_ret": None, "hit": None,
                        "status": "PENDING",
                    })
                    continue
                price_at_signal = float(prices.iloc[idx])
                price_at_horizon = float(prices.iloc[future_idx])
                if price_at_signal <= 0:
                    continue
                actual_ret = (price_at_horizon / price_at_signal - 1.0) * 100.0
                hit = actual_ret > 0 if signal_type == "BUY" else actual_ret < 0
                results.append({
                    "ticker": ticker, "type": signal_type, "horizon": horizon,
                    "date": str(sig_date.date()), "prob_up": prob_up,
                    "exp_ret": exp_ret, "actual_ret": actual_ret, "hit": hit,
                    "status": "HIT" if hit else "MISS",
                })
            except Exception:
                continue
    return results


# ===========================================================================
# Color helpers
# ===========================================================================

def _color_hit(val, threshold_good=0.60):
    if val is None:
        return "[dim]---[/]"
    pct = val * 100
    if pct >= threshold_good * 100:
        return f"[bright_green]{pct:.1f}%[/]"
    elif pct >= 50.0:
        return f"[yellow]{pct:.1f}%[/]"
    else:
        return f"[indian_red1]{pct:.1f}%[/]"


def _color_mae(val):
    if val is None:
        return "[dim]---[/]"
    if val < 3.0:
        return f"[bright_green]{val:.2f}%[/]"
    elif val < 6.0:
        return f"[yellow]{val:.2f}%[/]"
    else:
        return f"[indian_red1]{val:.2f}%[/]"


def _color_corr(val):
    if val is None:
        return "[dim]---[/]"
    if val > 0.15:
        return f"[bright_green]{val:+.3f}[/]"
    elif val > 0.0:
        return f"[yellow]{val:+.3f}[/]"
    else:
        return f"[indian_red1]{val:+.3f}[/]"


def _color_bias(val):
    if val is None:
        return "[dim]---[/]"
    if abs(val) < 1.0:
        return f"[bright_green]{val:+.2f}[/]"
    elif abs(val) < 3.0:
        return f"[yellow]{val:+.2f}[/]"
    else:
        return f"[indian_red1]{val:+.2f}[/]"


def _color_pit(val):
    if val is None:
        return "[dim]---[/]"
    if val >= 0.10:
        return f"[bright_green]{val:.3f}[/]"
    elif val >= 0.05:
        return f"[yellow]{val:.3f}[/]"
    else:
        return f"[indian_red1]{val:.3f}[/]"


def _color_grade(grade):
    colors = {"A": "bold bright_green", "B": "bright_green", "C": "yellow", "D": "orange1"}
    return f"[{colors.get(grade, 'indian_red1')}]{grade}[/]"


def _color_pf(val):
    if val is None:
        return "[dim]---[/]"
    if val >= 1.5:
        return f"[bright_green]{val:.2f}[/]"
    elif val >= 1.0:
        return f"[yellow]{val:.2f}[/]"
    else:
        return f"[indian_red1]{val:.2f}[/]"


def _color_val(val, good_thresh, bad_thresh, fmt=".2f", higher_better=True):
    """Generic color helper."""
    if val is None:
        return "[dim]---[/]"
    if higher_better:
        if val >= good_thresh:
            return f"[bright_green]{val:{fmt}}[/]"
        elif val >= bad_thresh:
            return f"[yellow]{val:{fmt}}[/]"
        else:
            return f"[indian_red1]{val:{fmt}}[/]"
    else:
        if val <= good_thresh:
            return f"[bright_green]{val:{fmt}}[/]"
        elif val <= bad_thresh:
            return f"[yellow]{val:{fmt}}[/]"
        else:
            return f"[indian_red1]{val:{fmt}}[/]"


# ===========================================================================
# Rendering: Summary Panel
# ===========================================================================
def render_summary_panel(results, elapsed, n_total, n_failed, eval_days, eval_spacing):
    if not HAS_RICH:
        return
    n_verified = len(results)
    total_eval_pts = sum(
        max(r[1].get("7d", {}).get("n", 0), r[1].get("30d", {}).get("n", 0))
        for r in results
    )
    hit_7d_vals = [r[1]["7d"]["hit_rate"] for r in results if r[1].get("7d", {}).get("hit_rate") is not None]
    hit_30d_vals = [r[1]["30d"]["hit_rate"] for r in results if r[1].get("30d", {}).get("hit_rate") is not None]
    mae_7d_vals = [r[1]["7d"]["mae"] for r in results if r[1].get("7d", {}).get("mae") is not None]

    avg_hit_7d = np.mean(hit_7d_vals) * 100 if hit_7d_vals else 0
    avg_hit_30d = np.mean(hit_30d_vals) * 100 if hit_30d_vals else 0
    avg_mae_7d = np.mean(mae_7d_vals) if mae_7d_vals else 0
    pct_above_50 = sum(1 for h in hit_7d_vals if h > 0.50) / max(len(hit_7d_vals), 1) * 100

    grade_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for r in results:
        grade_dist[r[4]] = grade_dist.get(r[4], 0) + 1

    grid = Table.grid(padding=(0, 4))
    grid.add_column(style="bold white", justify="right")
    grid.add_column(style="bright_cyan")
    grid.add_column(style="bold white", justify="right")
    grid.add_column(style="bright_cyan")

    grid.add_row("Assets verified:", f"{n_verified}", "Failed/skipped:", f"{n_failed}")
    grid.add_row("Total eval points:", f"{total_eval_pts:,}", "Eval window:", f"{eval_days}d (every {eval_spacing}d)")
    grid.add_row("Avg 7D hit rate:", _color_hit(avg_hit_7d / 100), "Avg 30D hit rate:", _color_hit(avg_hit_30d / 100))
    grid.add_row("Avg 7D MAE:", _color_mae(avg_mae_7d), "Hit rate > 50%:", f"[bright_cyan]{pct_above_50:.0f}%[/] of assets")
    grid.add_row(
        "Elapsed:", f"{elapsed:.1f}s",
        "Grades:",
        f"[bright_green]A:{grade_dist['A']}[/]  [bright_green]B:{grade_dist['B']}[/]  "
        f"[yellow]C:{grade_dist['C']}[/]  [orange1]D:{grade_dist['D']}[/]  "
        f"[indian_red1]F:{grade_dist['F']}[/]",
    )
    panel = Panel(grid, title="[bold bright_cyan]>>> FORECAST VERIFICATION SUMMARY[/]",
                  border_style="bright_cyan", box=box.DOUBLE, padding=(1, 2))
    console.print()
    console.print(Align.center(panel))
    console.print()


# ===========================================================================
# Rendering: Asset type breakdown
# ===========================================================================
def render_sector_summary(results):
    if not HAS_RICH or not results:
        return
    by_type = defaultdict(list)
    for symbol, metrics, tune_q, comp_score, grade in results:
        by_type[_detect_asset_type(symbol)].append((metrics, comp_score, grade))

    table = Table(title="[bold]Accuracy by Asset Type[/]", box=box.SIMPLE_HEAVY, header_style="bold bright_cyan")
    table.add_column("Asset Type", style="bold white", min_width=10)
    table.add_column("Count", justify="right")
    table.add_column("Avg 7D Hit%", justify="right")
    table.add_column("Avg 30D Hit%", justify="right")
    table.add_column("Avg 7D MAE", justify="right")
    table.add_column("Avg Score", justify="right")
    table.add_column("A+B %", justify="right")

    for at in ["equity", "metal", "currency", "crypto"]:
        items = by_type.get(at, [])
        if not items:
            continue
        h7 = [m["7d"]["hit_rate"] for m, _, _ in items if m.get("7d", {}).get("hit_rate") is not None]
        h30 = [m["30d"]["hit_rate"] for m, _, _ in items if m.get("30d", {}).get("hit_rate") is not None]
        mae7 = [m["7d"]["mae"] for m, _, _ in items if m.get("7d", {}).get("mae") is not None]
        scores = [s for _, s, _ in items]
        ab_pct = sum(1 for _, _, g in items if g in ("A", "B")) / max(len(items), 1) * 100
        table.add_row(
            at.capitalize(), f"{len(items)}",
            _color_hit(np.mean(h7)) if h7 else "[dim]---[/]",
            _color_hit(np.mean(h30)) if h30 else "[dim]---[/]",
            _color_mae(np.mean(mae7)) if mae7 else "[dim]---[/]",
            f"[bright_cyan]{np.mean(scores):.2f}[/]" if scores else "[dim]---[/]",
            f"[bright_green]{ab_pct:.0f}%[/]" if ab_pct >= 50 else f"[yellow]{ab_pct:.0f}%[/]",
        )
    console.print(table)
    console.print()


# ===========================================================================
# Rendering: Per-regime accuracy breakdown
# ===========================================================================
def render_regime_breakdown(results):
    """Show accuracy broken down by detected market regime across all assets."""
    if not HAS_RICH or not results:
        return

    # Aggregate regime stats from 7D horizon
    regime_agg = defaultdict(lambda: {"hits": 0, "total": 0, "mae_sum": 0.0})
    for _, metrics, _, _, _ in results:
        m7 = metrics.get("7d", {})
        for regime, acc in m7.get("regime_acc", {}).items():
            n = acc["n"]
            regime_agg[regime]["total"] += n
            regime_agg[regime]["hits"] += int(acc["hit_rate"] * n)
            regime_agg[regime]["mae_sum"] += acc["mae"] * n

    if not regime_agg:
        return

    table = Table(title="[bold]7D Accuracy by Market Regime[/]", box=box.SIMPLE_HEAVY,
                  header_style="bold bright_cyan")
    table.add_column("Regime", style="bold white", min_width=16)
    table.add_column("N Eval Points", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("Avg MAE", justify="right")

    for regime in ["trending", "calm", "mean_reverting", "volatile"]:
        data = regime_agg.get(regime)
        if not data or data["total"] == 0:
            continue
        hit = data["hits"] / data["total"]
        mae = data["mae_sum"] / data["total"]
        table.add_row(regime.replace("_", " ").title(), f"{data['total']:,}",
                      _color_hit(hit), _color_mae(mae))

    console.print(table)
    console.print()


# ===========================================================================
# Rendering: Per-model accuracy breakdown
# ===========================================================================
def render_model_breakdown(results):
    """Show accuracy broken down by tuned model type."""
    if not HAS_RICH or not results:
        return

    model_agg = defaultdict(lambda: {"hits_7d": 0, "total_7d": 0, "mae_sum": 0.0, "count": 0, "scores": []})

    for _, metrics, tune_q, comp_score, _ in results:
        model = tune_q.get("best_model", "---") or "---"
        # Simplify model name to category
        if "gaussian" in model:
            category = "gaussian"
        elif "nu_3" in model or "nu_4" in model:
            category = "student_t_heavy (nu<=4)"
        elif "nu_6" in model or "nu_8" in model:
            category = "student_t_medium (nu 6-8)"
        elif "nu_12" in model or "nu_20" in model:
            category = "student_t_light (nu>=12)"
        elif model == "---":
            category = "no_tune_cache"
        else:
            category = "other"

        m7 = metrics.get("7d", {})
        if "hit_rate" in m7:
            n = m7["n"]
            model_agg[category]["total_7d"] += n
            model_agg[category]["hits_7d"] += int(m7["hit_rate"] * n)
            model_agg[category]["mae_sum"] += m7["mae"] * n
            model_agg[category]["count"] += 1
            model_agg[category]["scores"].append(comp_score)

    if not model_agg:
        return

    table = Table(title="[bold]7D Accuracy by Model Type[/]", box=box.SIMPLE_HEAVY,
                  header_style="bold bright_cyan")
    table.add_column("Model Category", style="bold white", min_width=24)
    table.add_column("Assets", justify="right")
    table.add_column("Eval Pts", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("Avg MAE", justify="right")
    table.add_column("Avg Score", justify="right")

    for cat in sorted(model_agg.keys()):
        data = model_agg[cat]
        if data["total_7d"] == 0:
            continue
        hit = data["hits_7d"] / data["total_7d"]
        mae = data["mae_sum"] / data["total_7d"]
        avg_score = np.mean(data["scores"]) if data["scores"] else 0
        table.add_row(
            cat, f"{data['count']}", f"{data['total_7d']:,}",
            _color_hit(hit), _color_mae(mae),
            f"[bright_cyan]{avg_score:.2f}[/]",
        )
    console.print(table)
    console.print()


# ===========================================================================
# Rendering: Temporal stability (early vs late accuracy)
# ===========================================================================
def render_temporal_stability(results):
    """Show whether forecasts are getting better or worse over time."""
    if not HAS_RICH or not results:
        return

    improving = 0
    degrading = 0
    stable = 0
    details = []

    for symbol, metrics, _, _, grade in results:
        m7 = metrics.get("7d", {})
        early = m7.get("early_hit")
        late = m7.get("late_hit")
        if early is not None and late is not None:
            diff = late - early
            if diff > 0.05:
                improving += 1
            elif diff < -0.05:
                degrading += 1
            else:
                stable += 1
            if abs(diff) > 0.15:
                details.append((symbol, early, late, diff, grade))

    total = improving + degrading + stable
    if total == 0:
        return

    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="bold white", justify="right")
    grid.add_column(style="bright_cyan")
    grid.add_row("Improving (late > early + 5%):", f"[bright_green]{improving}[/] ({improving/total*100:.0f}%)")
    grid.add_row("Stable:", f"[yellow]{stable}[/] ({stable/total*100:.0f}%)")
    grid.add_row("Degrading (late < early - 5%):", f"[indian_red1]{degrading}[/] ({degrading/total*100:.0f}%)")

    panel = Panel(grid, title="[bold]Temporal Stability (7D: Early vs Late)[/]",
                  border_style="dim", box=box.ROUNDED, padding=(0, 2))
    console.print(panel)

    # Show extreme cases
    if details:
        details.sort(key=lambda x: x[3])
        table = Table(title="[dim]Largest Temporal Shifts[/]", box=box.SIMPLE, header_style="dim")
        table.add_column("Symbol", style="bold white")
        table.add_column("Early Hit%", justify="right")
        table.add_column("Late Hit%", justify="right")
        table.add_column("Shift", justify="right")
        table.add_column("Grade", justify="center")
        for sym, early, late, diff, grade in details[:10]:
            shift_color = "bright_green" if diff > 0 else "indian_red1"
            table.add_row(
                sym, f"{early*100:.0f}%", f"{late*100:.0f}%",
                f"[{shift_color}]{diff*100:+.0f}pp[/]", _color_grade(grade),
            )
        console.print(table)
    console.print()


# ===========================================================================
# Rendering: Vol-bucket accuracy
# ===========================================================================
def render_vol_breakdown(results):
    """Show accuracy broken down by volatility regime."""
    if not HAS_RICH or not results:
        return

    vol_agg = defaultdict(lambda: {"hits": 0, "total": 0, "mae_sum": 0.0})
    for _, metrics, _, _, _ in results:
        m7 = metrics.get("7d", {})
        for bucket, acc in m7.get("vol_acc", {}).items():
            n = acc["n"]
            vol_agg[bucket]["total"] += n
            vol_agg[bucket]["hits"] += int(acc["hit_rate"] * n)
            vol_agg[bucket]["mae_sum"] += acc["mae"] * n

    if not vol_agg:
        return

    table = Table(title="[bold]7D Accuracy by Volatility Bucket[/]", box=box.SIMPLE_HEAVY,
                  header_style="bold bright_cyan")
    table.add_column("Vol Bucket", style="bold white", min_width=12)
    table.add_column("N Eval Points", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("Avg MAE", justify="right")

    for bucket in ["low_vol", "mid_vol", "high_vol"]:
        data = vol_agg.get(bucket)
        if not data or data["total"] == 0:
            continue
        hit = data["hits"] / data["total"]
        mae = data["mae_sum"] / data["total"]
        table.add_row(bucket.replace("_", " ").title(), f"{data['total']:,}",
                      _color_hit(hit), _color_mae(mae))
    console.print(table)
    console.print()


# ===========================================================================
# Rendering: Main per-asset table (enhanced)
# ===========================================================================
def render_main_table(results):
    if not HAS_RICH or not results:
        return

    table = Table(title="[bold]Per-Asset Forecast Accuracy[/]", box=box.SIMPLE_HEAVY,
                  show_lines=False, pad_edge=True, header_style="bold bright_cyan")

    table.add_column("Symbol", style="bold white", min_width=10)
    table.add_column("Type", style="dim", min_width=5)
    table.add_column("Model", style="dim", min_width=14, max_width=26)
    table.add_column("PIT", justify="right", min_width=6)
    table.add_column("CRPS", justify="right", min_width=7)
    table.add_column("nu", justify="right", min_width=4)
    table.add_column("phi", justify="right", min_width=5)
    table.add_column("7D Hit", justify="right", min_width=7)
    table.add_column("30D Hit", justify="right", min_width=7)
    table.add_column("7D MAE", justify="right", min_width=7)
    table.add_column("Bias", justify="right", min_width=7)
    table.add_column("30D r", justify="right", min_width=7)
    table.add_column("PF", justify="right", min_width=5)
    table.add_column("MagR", justify="right", min_width=5)
    table.add_column("E/L", justify="right", min_width=7)
    table.add_column("BMA", justify="right", min_width=5)
    table.add_column("N", justify="right", min_width=4)
    table.add_column("Grd", justify="center", min_width=3)

    for symbol, metrics, tune_q, comp_score, grade in results:
        m7 = metrics.get("7d", {})
        m30 = metrics.get("30d", {})

        model_name = tune_q.get("best_model", "---") or "---"
        if len(model_name) > 26:
            model_name = model_name[:23] + "..."

        crps = tune_q.get("crps")
        crps_str = f"[bright_cyan]{crps:.4f}[/]" if crps is not None else "[dim]---[/]"

        nu = tune_q.get("nu")
        nu_str = f"{nu:.0f}" if nu is not None else "---"

        phi = tune_q.get("phi")
        phi_str = f"{phi:.2f}" if phi is not None else "---"

        # Magnitude ratio
        mag_r = m7.get("mag_ratio")
        if mag_r is not None:
            if 0.5 <= mag_r <= 2.0:
                mag_str = f"[bright_green]{mag_r:.1f}[/]"
            elif 0.3 <= mag_r <= 3.0:
                mag_str = f"[yellow]{mag_r:.1f}[/]"
            else:
                mag_str = f"[indian_red1]{mag_r:.1f}[/]"
        else:
            mag_str = "[dim]---[/]"

        # Early/Late hit rate shift
        early = m7.get("early_hit")
        late = m7.get("late_hit")
        if early is not None and late is not None:
            e_pct = int(early * 100)
            l_pct = int(late * 100)
            diff = late - early
            d_color = "bright_green" if diff > 0.05 else ("indian_red1" if diff < -0.05 else "yellow")
            el_str = f"[{d_color}]{e_pct}/{l_pct}[/]"
        else:
            el_str = "[dim]---[/]"

        # BMA concentration
        bma_c = tune_q.get("bma_concentration")
        bma_str = f"{bma_c:.2f}" if bma_c is not None else "---"

        n_pts = max(m7.get("n", 0), m30.get("n", 0))

        table.add_row(
            symbol, _detect_asset_type(symbol)[:5], model_name,
            _color_pit(tune_q.get("pit")), crps_str,
            nu_str, phi_str,
            _color_hit(m7.get("hit_rate")),
            _color_hit(m30.get("hit_rate")),
            _color_mae(m7.get("mae")),
            _color_bias(m7.get("bias")),
            _color_corr(m30.get("corr")),
            _color_pf(m7.get("profit_factor")),
            mag_str, el_str, bma_str,
            f"[bright_cyan]{n_pts}[/]",
            _color_grade(grade),
        )

    console.print(table)
    console.print()


# ===========================================================================
# Rendering: Problem assets
# ===========================================================================
def render_problem_assets(results):
    if not HAS_RICH:
        return

    problems = []
    for symbol, metrics, tune_q, comp_score, grade in results:
        m7 = metrics.get("7d", {})
        m30 = metrics.get("30d", {})
        pit = tune_q.get("pit")
        issues = []

        hit_7d = m7.get("hit_rate")
        hit_30d = m30.get("hit_rate")
        if hit_7d is not None and hit_7d < 0.50:
            issues.append(f"7D hit {hit_7d*100:.0f}%<50%")
        if hit_30d is not None and hit_30d < 0.50:
            issues.append(f"30D hit {hit_30d*100:.0f}%<50%")
        if pit is not None and pit < 0.05:
            issues.append(f"PIT={pit:.3f}")
        corr_30d = m30.get("corr")
        if corr_30d is not None and corr_30d < -0.10:
            issues.append(f"30D r={corr_30d:+.2f}")
        mag_r = m7.get("mag_ratio")
        if mag_r is not None and (mag_r > 3.0 or mag_r < 0.3):
            issues.append(f"MagR={mag_r:.1f}")
        # Temporal degradation
        early = m7.get("early_hit")
        late = m7.get("late_hit")
        if early is not None and late is not None and (late - early) < -0.15:
            issues.append(f"stale({int(early*100)}->{int(late*100)}%)")

        if issues:
            problems.append((symbol, grade, tune_q.get("best_model", ""), issues))

    if not problems:
        console.print("[bright_green]No problem assets detected.[/]")
        console.print()
        return

    table = Table(title=f"[bold indian_red1]Problem Assets ({len(problems)})[/]",
                  box=box.SIMPLE_HEAVY, header_style="bold indian_red1")
    table.add_column("Symbol", style="bold white", min_width=12)
    table.add_column("Grade", justify="center", min_width=3)
    table.add_column("Model", style="dim", max_width=26)
    table.add_column("Issues", style="indian_red1")

    for symbol, grade, model, issues in sorted(problems, key=lambda x: x[1], reverse=True):
        m = model[:23] + "..." if len(model) > 26 else model
        table.add_row(symbol, _color_grade(grade), m, " | ".join(issues))

    console.print(table)
    console.print()


# ===========================================================================
# Rendering: Worst misses investigation
# ===========================================================================
def render_worst_misses(results, n_worst=15):
    """Show the worst individual forecast misses across all assets."""
    if not HAS_RICH or not results:
        return

    all_misses = []
    for symbol, metrics, tune_q, _, grade in results:
        m7 = metrics.get("7d", {})
        for miss in m7.get("worst_misses", []):
            all_misses.append({
                "symbol": symbol,
                "grade": grade,
                "model": tune_q.get("best_model", "---"),
                **miss,
            })

    if not all_misses:
        return

    all_misses.sort(key=lambda x: abs(x["error"]), reverse=True)

    table = Table(title=f"[bold]Top {n_worst} Worst 7D Forecast Misses[/]",
                  box=box.SIMPLE_HEAVY, header_style="bold bright_cyan")
    table.add_column("Symbol", style="bold white")
    table.add_column("Date")
    table.add_column("Predicted", justify="right")
    table.add_column("Actual", justify="right")
    table.add_column("Error", justify="right")
    table.add_column("Regime")
    table.add_column("Vol", justify="right")
    table.add_column("Conf")
    table.add_column("Model", style="dim", max_width=24)

    for miss in all_misses[:n_worst]:
        err = miss["error"]
        err_color = "indian_red1" if abs(err) > 10 else "yellow"
        actual_color = "bright_green" if miss["actual"] > 0 else "indian_red1"
        table.add_row(
            miss["symbol"], miss["date"],
            f"{miss['predicted']:+.2f}%",
            f"[{actual_color}]{miss['actual']:+.2f}%[/]",
            f"[{err_color}]{err:+.2f}%[/]",
            miss["regime"],
            f"{miss['realized_vol']:.0%}",
            miss["confidence"],
            (miss["model"][:21] + "...") if len(miss["model"]) > 24 else miss["model"],
        )

    console.print(table)
    console.print()


# ===========================================================================
# Rendering: Error distribution overview
# ===========================================================================
def render_error_distribution(results):
    """Show error distribution statistics across the universe."""
    if not HAS_RICH or not results:
        return

    skews = []
    kurts = []
    stds = []
    biases = []
    for _, metrics, _, _, _ in results:
        m7 = metrics.get("7d", {})
        if "error_skew" in m7:
            skews.append(m7["error_skew"])
            kurts.append(m7["error_kurt"])
            stds.append(m7["error_std"])
            biases.append(m7["bias"])

    if not skews:
        return

    grid = Table.grid(padding=(0, 4))
    grid.add_column(style="bold white", justify="right")
    grid.add_column(style="bright_cyan")
    grid.add_column(style="bold white", justify="right")
    grid.add_column(style="bright_cyan")

    avg_skew = np.mean(skews)
    avg_kurt = np.mean(kurts)
    avg_std = np.mean(stds)
    avg_bias = np.mean(biases)

    skew_color = "bright_green" if abs(avg_skew) < 0.5 else ("yellow" if abs(avg_skew) < 1.0 else "indian_red1")
    kurt_color = "bright_green" if avg_kurt < 2.0 else ("yellow" if avg_kurt < 5.0 else "indian_red1")

    grid.add_row(
        "Avg error std:", f"{avg_std:.2f}%",
        "Avg error bias:", _color_bias(avg_bias),
    )
    grid.add_row(
        "Avg error skewness:", f"[{skew_color}]{avg_skew:+.2f}[/]  (0=symmetric)",
        "Avg error kurtosis:", f"[{kurt_color}]{avg_kurt:+.2f}[/]  (0=normal tails)",
    )
    grid.add_row(
        "Negative bias assets:", f"{sum(1 for b in biases if b < -1.0)} ({sum(1 for b in biases if b < -1.0)/len(biases)*100:.0f}%)",
        "Heavy-tail assets:", f"{sum(1 for k in kurts if k > 3.0)} ({sum(1 for k in kurts if k > 3.0)/len(kurts)*100:.0f}%)",
    )

    panel = Panel(grid, title="[bold]7D Prediction Error Distribution[/]",
                  border_style="dim", box=box.ROUNDED, padding=(0, 2))
    console.print(panel)
    console.print()


# ===========================================================================
# Rendering: High-conviction verification
# ===========================================================================
def render_hc_verification(hc_results):
    if not HAS_RICH or not hc_results:
        return
    evaluated = [r for r in hc_results if r["status"] != "PENDING"]
    pending = [r for r in hc_results if r["status"] == "PENDING"]

    if evaluated:
        hits = sum(1 for r in evaluated if r["hit"])
        total = len(evaluated)
        hit_rate = hits / total if total > 0 else 0

        table = Table(
            title=f"[bold]High-Conviction Signal Verification ({hits}/{total} = {hit_rate*100:.0f}% hit rate)[/]",
            box=box.SIMPLE_HEAVY, header_style="bold bright_cyan")
        table.add_column("Ticker", style="bold white")
        table.add_column("Type", min_width=6)
        table.add_column("Date")
        table.add_column("Horizon", justify="right")
        table.add_column("P(up)", justify="right")
        table.add_column("Exp Ret%", justify="right")
        table.add_column("Actual%", justify="right")
        table.add_column("Result", justify="center")

        for r in sorted(evaluated, key=lambda x: x["date"], reverse=True):
            type_style = "[bright_green]" if r["type"] == "BUY" else "[indian_red1]"
            result_str = "[bright_green]HIT[/]" if r["hit"] else "[indian_red1]MISS[/]"
            actual_color = "bright_green" if r["actual_ret"] > 0 else "indian_red1"
            table.add_row(
                r["ticker"], f"{type_style}{r['type']}[/]", r["date"],
                f"{r['horizon']}d",
                f"{r['prob_up']:.3f}" if r["prob_up"] else "---",
                f"{r['exp_ret']:+.2f}%" if r["exp_ret"] else "---",
                f"[{actual_color}]{r['actual_ret']:+.2f}%[/]",
                result_str,
            )
        console.print(table)

    if pending:
        console.print(
            f"[dim]  {len(pending)} signal(s) still pending: "
            f"{', '.join(r['ticker'] + '_' + str(r['horizon']) + 'd' for r in pending[:10])}"
            f"{'...' if len(pending) > 10 else ''}[/]"
        )
    console.print()


# ===========================================================================
# JSON export (for scripting / next-step debugging)
# ===========================================================================
def export_json(results, failed, path, eval_days, eval_spacing):
    """Export all verification data to JSON for debugging signals.py."""

    def _clean_for_json(obj):
        """Make numpy types JSON-serializable, strip raw records to save space."""
        if isinstance(obj, dict):
            return {k: _clean_for_json(v) for k, v in obj.items() if k != "records"}
        elif isinstance(obj, list):
            return [_clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        "generated_at": datetime.now().isoformat(),
        "eval_days": eval_days,
        "eval_spacing": eval_spacing,
        "n_verified": len(results),
        "n_failed": len(failed),
        "summary": {},
        "assets": {},
        "failed_assets": {sym: err for sym, err in failed},
    }

    # Summary stats
    hit_7d = [r[1]["7d"]["hit_rate"] for r in results if r[1].get("7d", {}).get("hit_rate") is not None]
    hit_30d = [r[1]["30d"]["hit_rate"] for r in results if r[1].get("30d", {}).get("hit_rate") is not None]
    output["summary"] = {
        "avg_7d_hit_rate": float(np.mean(hit_7d)) if hit_7d else None,
        "avg_30d_hit_rate": float(np.mean(hit_30d)) if hit_30d else None,
        "pct_above_50": float(sum(1 for h in hit_7d if h > 0.5) / max(len(hit_7d), 1)),
    }

    # Per-asset with all diagnostics (minus raw records)
    for symbol, metrics, tune_q, comp_score, grade in results:
        output["assets"][symbol] = {
            "grade": grade,
            "composite_score": float(comp_score),
            "asset_type": _detect_asset_type(symbol),
            "tune_quality": _clean_for_json(tune_q),
            "metrics_7d": _clean_for_json(metrics.get("7d", {})),
            "metrics_30d": _clean_for_json(metrics.get("30d", {})),
        }

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    if HAS_RICH:
        console.print(f"[bright_green]Exported verification data to {path}[/]")
        console.print()


# ===========================================================================
# JSON export with raw records (full detail for scripting)
# ===========================================================================
def export_json_full(results, failed, path, eval_days, eval_spacing):
    """Export ALL data including raw per-eval-point records."""

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_clean(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = {
        "generated_at": datetime.now().isoformat(),
        "eval_days": eval_days,
        "eval_spacing": eval_spacing,
        "n_verified": len(results),
        "assets": {},
    }
    for symbol, metrics, tune_q, comp_score, grade in results:
        output["assets"][symbol] = {
            "grade": grade,
            "score": float(comp_score),
            "tune": _clean(tune_q),
            "m7": _clean(metrics.get("7d", {})),
            "m30": _clean(metrics.get("30d", {})),
        }

    with open(path, "w") as f:
        json.dump(output, f, indent=1, default=str)

    if HAS_RICH:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        console.print(f"[bright_green]Exported FULL data to {path} ({size_mb:.1f} MB)[/]")
        console.print()


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Walk-forward forecast verification engine (enhanced)")
    parser.add_argument("--assets", type=str, default=None,
                        help="Comma-separated list of assets (default: full universe)")
    parser.add_argument("--eval-days", type=int, default=DEFAULT_EVAL_DAYS,
                        help=f"Trading days to evaluate (default: {DEFAULT_EVAL_DAYS})")
    parser.add_argument("--eval-spacing", type=int, default=DEFAULT_EVAL_SPACING,
                        help=f"Evaluate every Nth day (default: {DEFAULT_EVAL_SPACING})")
    parser.add_argument("--min-eval-points", type=int, default=MIN_EVAL_POINTS,
                        help=f"Min eval points to include (default: {MIN_EVAL_POINTS})")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count - 1)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--sort", type=str, default="grade",
                        choices=["grade", "hit7d", "hit30d", "mae7d", "symbol"],
                        help="Sort results by (default: grade)")
    parser.add_argument("--no-hc", action="store_true", help="Skip high-conviction verification")
    parser.add_argument("--worst", type=int, default=15,
                        help="Number of worst misses to show (default: 15)")
    parser.add_argument("--export-json", type=str, default=None,
                        help="Export results to JSON file (summary, no raw records)")
    parser.add_argument("--export-json-full", type=str, default=None,
                        help="Export FULL results including raw records to JSON")
    args = parser.parse_args()

    if args.assets:
        assets = [s.strip() for s in args.assets.split(",") if s.strip()]
    else:
        from ingestion.data_utils import get_default_asset_universe
        assets = get_default_asset_universe()

    n_total = len(assets)
    if HAS_RICH:
        console.print(Rule(style="bright_cyan"))
        console.print(Panel(
            f"[bold]Walk-Forward Forecast Verification (Enhanced)[/]\n"
            f"[dim]Assets: {n_total} | Eval: {args.eval_days}d every {args.eval_spacing}d | "
            f"Horizons: 7D, 30D | Worst: {args.worst}[/]",
            border_style="bright_cyan", box=box.DOUBLE,
        ))
        console.print()

    work_items = [(sym, args.eval_days, args.eval_spacing) for sym in assets]

    t0 = time.time()
    all_results = []
    failed_assets = []
    completed = 0
    n_workers = args.workers or max(1, (mp.cpu_count() or 4) - 1)

    if args.no_parallel or n_total == 1:
        for item in work_items:
            sym, metrics, tune_q, error = verify_single_asset(item)
            completed += 1
            if error:
                if HAS_RICH:
                    console.print(f"  [dim][{completed}/{n_total}][/] [indian_red1]{sym}: {error}[/]")
                failed_assets.append((sym, error))
            else:
                m7h = metrics.get("7d", {}).get("hit_rate")
                hit_str = f"{m7h*100:.0f}%" if m7h else "---"
                if HAS_RICH:
                    console.print(f"  [dim][{completed}/{n_total}][/] [bright_green]{sym}[/] 7D={hit_str}")
                all_results.append((sym, metrics, tune_q, None))
    else:
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
            futures_map = {}
            for item in work_items:
                future = executor.submit(verify_single_asset, item)
                futures_map[future] = item[0]

            for future in as_completed(futures_map):
                symbol = futures_map[future]
                completed += 1
                try:
                    sym, metrics, tune_q, error = future.result()
                    if error:
                        if HAS_RICH:
                            console.print(f"  [dim][{completed}/{n_total}][/] [indian_red1]{sym}: {error}[/]")
                        failed_assets.append((sym, error))
                    else:
                        m7h = metrics.get("7d", {}).get("hit_rate")
                        hit_str = f"{m7h*100:.0f}%" if m7h else "---"
                        if HAS_RICH:
                            console.print(f"  [dim][{completed}/{n_total}][/] [bright_green]{sym}[/] 7D={hit_str}")
                        all_results.append((sym, metrics, tune_q, None))
                except Exception as e:
                    if HAS_RICH:
                        console.print(f"  [dim][{completed}/{n_total}][/] [indian_red1]{symbol}: {e}[/]")
                    failed_assets.append((symbol, str(e)))

    elapsed = time.time() - t0

    # Score and grade
    scored_results = []
    for sym, metrics, tune_q, _ in all_results:
        comp_score, grade = _composite_score(metrics, tune_q)
        scored_results.append((sym, metrics, tune_q, comp_score, grade))

    # Sort
    sort_keys = {
        "grade": lambda x: (-x[3], x[0]),
        "hit7d": lambda x: -(x[1].get("7d", {}).get("hit_rate", 0)),
        "hit30d": lambda x: -(x[1].get("30d", {}).get("hit_rate", 0)),
        "mae7d": lambda x: x[1].get("7d", {}).get("mae", 999),
        "symbol": lambda x: x[0],
    }
    scored_results.sort(key=sort_keys.get(args.sort, sort_keys["grade"]))

    if HAS_RICH:
        console.print()
        console.print(Rule(style="bright_cyan"))

    # ---- Render all sections ----
    render_summary_panel(scored_results, elapsed, n_total, len(failed_assets), args.eval_days, args.eval_spacing)
    render_sector_summary(scored_results)
    render_regime_breakdown(scored_results)
    render_model_breakdown(scored_results)
    render_vol_breakdown(scored_results)
    render_temporal_stability(scored_results)
    render_error_distribution(scored_results)
    render_main_table(scored_results)
    render_problem_assets(scored_results)
    render_worst_misses(scored_results, n_worst=args.worst)

    if not args.no_hc:
        hc_results = _verify_high_conviction_signals()
        if hc_results:
            render_hc_verification(hc_results)

    # JSON export
    if args.export_json:
        export_json(scored_results, failed_assets, args.export_json, args.eval_days, args.eval_spacing)
    if args.export_json_full:
        export_json_full(scored_results, failed_assets, args.export_json_full, args.eval_days, args.eval_spacing)

    if HAS_RICH:
        console.print(Rule(style="dim"))
        console.print(
            f"[dim]Completed {len(scored_results)} assets in {elapsed:.1f}s "
            f"({len(failed_assets)} failed/skipped)[/]"
        )
        console.print()


if __name__ == "__main__":
    main()
