#!/usr/bin/env python3
"""
===============================================================================
SIGNALS TABLE VERIFICATION ENGINE
===============================================================================

Walk-forward verification of the Active Signals table produced by
compute_features() → latest_signals() in signals.py.

Measures whether the 1d/3d/7d/21d/63d/126d/252d forecast columns
(p_up, exp_ret, label) are accurate against realized returns.

This is SEPARATE from verify_forecasts.py which only verifies
ensemble_forecast() from market_temperature.py.

Metrics:
  - Hit rate:      Directional accuracy (sign(exp_ret) vs sign(actual))
  - Brier score:   Calibration of p_up probability
  - MAE:           Mean absolute error of exp_ret vs actual (%)
  - Bias:          Systematic over/under prediction
  - Correlation:   Pearson between predicted and actual
  - Magnitude ratio:  avg(|predicted|) / avg(|actual|) — size calibration
  - Label accuracy:    BUY hit rate, SELL hit rate

Usage:
  make verify-signals                                # Full universe, 252 days
  make verify-signals-quick                          # 8 key assets, 90 days
  python src/decision/verify_signals.py --assets SPY,AAPL --eval-days 120
  python src/decision/verify_signals.py --export-json /tmp/verify_signals.json

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
import traceback
import warnings
from collections import defaultdict
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

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console(force_terminal=True, color_system="truecolor", width=200) if HAS_RICH else None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TUNE_CACHE_DIR = Path(os.path.join(REPO_ROOT, "src", "data", "tune"))
PRICE_CACHE_DIR = Path(os.path.join(REPO_ROOT, "src", "data", "prices"))

VERIFY_HORIZONS = [1, 3, 7, 21, 63, 126, 252]
HORIZON_LABELS = {1: "1d", 3: "3d", 7: "1w", 21: "1m", 63: "3m", 126: "6m", 252: "12m"}
DEFAULT_EVAL_DAYS = 252
DEFAULT_EVAL_SPACING = 5
MIN_EVAL_POINTS = 5
MIN_PRICE_POINTS = 120

# Grading weights — emphasize 7d/21d hit + calibration
GRADE_WEIGHTS = {
    "hit_7d": 0.25,
    "hit_21d": 0.20,
    "brier_7d": 0.20,
    "mae_7d": 0.15,
    "corr_21d": 0.10,
    "pit": 0.10,
}


# ---------------------------------------------------------------------------
# Helper: detect asset type
# ---------------------------------------------------------------------------
def _detect_asset_type(symbol: str) -> str:
    s = symbol.upper()
    if any(x in s for x in ["BTC", "ETH", "DOGE", "SOL", "ADA"]):
        return "crypto"
    if s.endswith("=X") or "JPY" in s or "USD" in s.split("=")[0][-3:]:
        if not any(s.startswith(p) for p in ["BTC", "ETH"]):
            return "currency"
    if s.endswith("=F"):
        return "metal"
    metal_tickers = {
        "GLD", "SLV", "GDX", "GDXJ", "SIL", "GOLD", "NEM",
        "AEM", "WPM", "PAAS", "AG", "KGC", "FNV",
    }
    if s in metal_tickers:
        return "metal"
    return "equity"


# ---------------------------------------------------------------------------
# Helper: regime detection
# ---------------------------------------------------------------------------
def _regime_detect(returns: np.ndarray) -> str:
    try:
        if len(returns) < 20:
            return "calm"
        vol = np.std(returns[-20:]) * np.sqrt(252)
        mom = np.sum(returns[-20:])
        if vol > 0.30:
            return "volatile"
        elif abs(mom) > 0.10:
            return "trending"
        else:
            return "calm"
    except Exception:
        return "calm"


# ---------------------------------------------------------------------------
# Helper: load OHLC DataFrame + Close Series from cache
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


# ---------------------------------------------------------------------------
# Helper: load tune quality from cache
# ---------------------------------------------------------------------------
def _load_tune_quality(symbol: str) -> Dict[str, Any]:
    safe = symbol.replace("/", "_").replace("=", "_").replace(":", "_").upper()
    path = TUNE_CACHE_DIR / f"{safe}.json"
    result = {"best_model": "—", "pit": None, "crps": None}
    if not path.exists():
        return result
    try:
        with open(path) as f:
            d = json.load(f)
        g = d.get("global", {})
        result["best_model"] = g.get("best_model", "—")
        result["pit"] = g.get("pit_ks_pvalue")
        result["crps"] = g.get("crps")
        if result["crps"] is None:
            models = g.get("models", {})
            bm = result["best_model"]
            if isinstance(models, dict) and bm and bm in models:
                result["crps"] = models[bm].get("crps")
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Core: verify a single asset (worker function)
# ---------------------------------------------------------------------------
def verify_single_asset(
    args_tuple: Tuple,
) -> Tuple[str, Optional[Dict], Dict, Optional[str]]:
    """
    Walk-forward evaluation of compute_features() → latest_signals()
    for one asset.

    Returns: (symbol, records_by_horizon, tune_quality, error_or_None)
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

        # Mock risk temperature to avoid Yahoo Finance calls during verification.
        # Risk temperature only affects position sizing (pos_strength), NOT
        # the forecast fields we're verifying (p_up, exp_ret, label).
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

        # Suppress Rich console output from compute_features/latest_signals
        import io
        import contextlib
        from decision.signals import (
            compute_features,
            latest_signals,
            _load_tuned_kalman_params,
            DEFAULT_HORIZONS,
        )
        # Redirect signals.py's Rich console to /dev/null
        import decision.signals as _sig_mod
        if hasattr(_sig_mod, 'console') and _sig_mod.console is not None:
            _sig_mod.console = type(_sig_mod.console)(file=io.StringIO(), force_terminal=False)

        # Load data
        ohlc_df = _load_ohlc(symbol)
        if ohlc_df is None:
            return symbol, None, _load_tune_quality(symbol), "No OHLC data"

        px = _extract_close(ohlc_df)
        if px is None or len(px) < MIN_PRICE_POINTS:
            return symbol, None, _load_tune_quality(symbol), "Insufficient price data"

        tune_q = _load_tune_quality(symbol)

        # Load tuned params once (shared across all eval points)
        tuned_params = _load_tuned_kalman_params(symbol)

        n = len(px)
        start_idx = max(MIN_PRICE_POINTS, n - eval_days)

        # Collect records per horizon
        records_by_horizon: Dict[int, List[Dict]] = {h: [] for h in VERIFY_HORIZONS}

        idx = start_idx
        while idx < n:
            if len(px.iloc[:idx]) < 60:
                idx += eval_spacing
                continue

            # Truncate data to simulate "present"
            px_trunc = px.iloc[:idx]
            ohlc_trunc = ohlc_df.iloc[:idx]
            last_close = float(px_trunc.iloc[-1])
            eval_date = px.index[idx - 1]

            # Detect regime
            log_rets = np.log(px_trunc / px_trunc.shift(1)).dropna().values
            regime = _regime_detect(log_rets)

            try:
                # Suppress console output from compute_features/latest_signals
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    feats = compute_features(px_trunc, asset_symbol=symbol, ohlc_df=ohlc_trunc)
                    sigs, _ = latest_signals(
                        feats,
                        horizons=VERIFY_HORIZONS,
                        last_close=last_close,
                        t_map=True,
                        ci=0.68,
                        tuned_params=tuned_params,
                        asset_key=symbol,
                    )
            except Exception:
                idx += eval_spacing
                continue

            # Map signals by horizon
            sig_map = {s.horizon_days: s for s in sigs}

            for H in VERIFY_HORIZONS:
                sig = sig_map.get(H)
                if sig is None:
                    continue

                future_idx = idx - 1 + H
                # Allow small offset for missing trading days
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

                actual_pct = (price_future / price_now - 1.0) * 100.0
                # exp_ret is in log-return units; convert to pct for comparison
                pred_pct = sig.exp_ret * 100.0

                records_by_horizon[H].append({
                    "date": str(eval_date.date()),
                    "predicted": pred_pct,
                    "actual": actual_pct,
                    "p_up": sig.p_up,
                    "label": sig.label,
                    "actual_up": 1 if actual_pct > 0 else 0,
                    "pred_dir": 1 if pred_pct > 0 else (-1 if pred_pct < 0 else 0),
                    "actual_dir": 1 if actual_pct > 0 else (-1 if actual_pct < 0 else 0),
                    "regime": regime,
                    "error": pred_pct - actual_pct,
                })

            idx += eval_spacing

        # Compute metrics
        metrics = _compute_metrics(records_by_horizon)
        if metrics is None:
            return symbol, None, tune_q, f"Too few eval points (<{MIN_EVAL_POINTS})"

        return symbol, metrics, tune_q, None

    except Exception as e:
        return symbol, None, _load_tune_quality(symbol), f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def _compute_metrics(
    records_by_horizon: Dict[int, List[Dict]],
) -> Optional[Dict]:
    """Compute forecast accuracy metrics from evaluation records per horizon."""

    any_valid = False
    metrics: Dict[str, Any] = {}

    for H in VERIFY_HORIZONS:
        records = records_by_horizon.get(H, [])
        label = HORIZON_LABELS[H]

        if len(records) < MIN_EVAL_POINTS:
            metrics[label] = {"n": len(records)}
            continue

        any_valid = True

        preds = np.array([r["predicted"] for r in records])
        actuals = np.array([r["actual"] for r in records])
        errors = np.array([r["error"] for r in records])
        p_ups = np.array([r["p_up"] for r in records])
        actual_ups = np.array([r["actual_up"] for r in records])
        pred_dirs = np.array([r["pred_dir"] for r in records])
        actual_dirs = np.array([r["actual_dir"] for r in records])
        labels = [r["label"] for r in records]

        # Hit rate (directional)
        nonzero = pred_dirs != 0
        hit_rate = float(np.mean(pred_dirs[nonzero] == actual_dirs[nonzero])) if nonzero.sum() > 0 else 0.5

        # MAE
        mae = float(np.mean(np.abs(errors)))

        # Bias
        bias = float(np.mean(errors))

        # RMSE
        rmse = float(np.sqrt(np.mean(errors ** 2)))

        # Correlation
        if np.std(preds) > 1e-12 and np.std(actuals) > 1e-12:
            corr = float(np.corrcoef(preds, actuals)[0, 1])
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0

        # Brier score: p_up vs actual_up
        brier = float(np.mean((p_ups - actual_ups) ** 2))

        # Magnitude ratio: avg(|pred|) / avg(|actual|)
        avg_pred_abs = float(np.mean(np.abs(preds)))
        avg_actual_abs = float(np.mean(np.abs(actuals)))
        mag_ratio = avg_pred_abs / max(avg_actual_abs, 1e-8)

        # Label accuracy
        buy_records = [(r["actual"] > 0) for r in records if r["label"] in ("BUY", "STRONG BUY")]
        sell_records = [(r["actual"] < 0) for r in records if r["label"] in ("SELL", "STRONG SELL")]
        buy_hit = float(np.mean(buy_records)) if buy_records else None
        sell_hit = float(np.mean(sell_records)) if sell_records else None

        # p_up calibration: bin by decile
        calibration_bins = {}
        for b_lo in np.arange(0.0, 1.0, 0.1):
            b_hi = b_lo + 0.1
            mask = (p_ups >= b_lo) & (p_ups < b_hi)
            if mask.sum() >= 3:
                calibration_bins[f"{b_lo:.1f}-{b_hi:.1f}"] = {
                    "n": int(mask.sum()),
                    "predicted_p_up": float(np.mean(p_ups[mask])),
                    "actual_freq_up": float(np.mean(actual_ups[mask])),
                }

        # Regime breakdown
        regime_acc = {}
        for regime in set(r["regime"] for r in records):
            rmask = [i for i, r in enumerate(records) if r["regime"] == regime]
            if len(rmask) >= 3:
                r_preds = preds[rmask]
                r_actuals = actuals[rmask]
                r_pred_dirs = pred_dirs[rmask]
                r_actual_dirs = actual_dirs[rmask]
                r_nonzero = r_pred_dirs != 0
                r_hit = float(np.mean(r_pred_dirs[r_nonzero] == r_actual_dirs[r_nonzero])) if r_nonzero.sum() > 0 else 0.5
                regime_acc[regime] = {"n": len(rmask), "hit_rate": r_hit}

        # Worst misses
        abs_errors = np.abs(errors)
        worst_idx = np.argsort(abs_errors)[-10:][::-1]
        worst_misses = [
            {
                "date": records[i]["date"],
                "predicted": float(preds[i]),
                "actual": float(actuals[i]),
                "error": float(errors[i]),
                "p_up": float(p_ups[i]),
                "label": records[i]["label"],
            }
            for i in worst_idx
        ]

        metrics[label] = {
            "n": len(records),
            "hit_rate": hit_rate,
            "mae": mae,
            "bias": bias,
            "rmse": rmse,
            "corr": corr,
            "brier": brier,
            "mag_ratio": mag_ratio,
            "buy_hit": buy_hit,
            "sell_hit": sell_hit,
            "calibration": calibration_bins,
            "regime_acc": regime_acc,
            "worst_misses": worst_misses,
        }

    if not any_valid:
        return None

    return metrics


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def _composite_score(metrics: Dict, tune_q: Dict) -> Tuple[float, str]:
    """Composite score [0,1] and letter grade."""
    score = 0.0
    m1w = metrics.get("1w", {})  # 7d
    m1m = metrics.get("1m", {})  # 21d

    if "hit_rate" in m1w:
        s = max(0.0, min(1.0, (m1w["hit_rate"] - 0.40) / 0.30))
        score += s * GRADE_WEIGHTS["hit_7d"]
    if "hit_rate" in m1m:
        s = max(0.0, min(1.0, (m1m["hit_rate"] - 0.40) / 0.30))
        score += s * GRADE_WEIGHTS["hit_21d"]
    if "brier" in m1w:
        s = max(0.0, min(1.0, 1.0 - (m1w["brier"] - 0.15) / 0.20))
        score += s * GRADE_WEIGHTS["brier_7d"]
    if "mae" in m1w:
        s = max(0.0, min(1.0, 1.0 - (m1w["mae"] - 1.0) / 5.0))
        score += s * GRADE_WEIGHTS["mae_7d"]
    if "corr" in m1m:
        s = max(0.0, min(1.0, m1m["corr"] / 0.30))
        score += s * GRADE_WEIGHTS["corr_21d"]
    pit = tune_q.get("pit")
    if pit is not None:
        s = max(0.0, min(1.0, pit))
        score += s * GRADE_WEIGHTS["pit"]

    total_weight = sum(
        v for k, v in GRADE_WEIGHTS.items()
        if (k == "hit_7d" and "hit_rate" in m1w)
        or (k == "hit_21d" and "hit_rate" in m1m)
        or (k == "brier_7d" and "brier" in m1w)
        or (k == "mae_7d" and "mae" in m1w)
        or (k == "corr_21d" and "corr" in m1m)
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


# ===========================================================================
# Rendering helpers
# ===========================================================================
def _color_val(val, fmt, good, bad, higher_is_better=True):
    if val is None:
        return "[dim]---[/]"
    if higher_is_better:
        if val >= good:
            return f"[bright_green]{val:{fmt}}[/]"
        elif val >= bad:
            return f"[yellow]{val:{fmt}}[/]"
        else:
            return f"[indian_red1]{val:{fmt}}[/]"
    else:
        if val <= good:
            return f"[bright_green]{val:{fmt}}[/]"
        elif val <= bad:
            return f"[yellow]{val:{fmt}}[/]"
        else:
            return f"[indian_red1]{val:{fmt}}[/]"


def _color_hit(v):
    return _color_val(v, ".1%", 0.55, 0.48, True)


def _color_brier(v):
    return _color_val(v, ".3f", 0.22, 0.27, False)


def _color_mae(v):
    return _color_val(v, ".2f", 3.0, 6.0, False)


def _color_mag(v):
    if v is None:
        return "[dim]---[/]"
    if 0.5 <= v <= 2.0:
        return f"[bright_green]{v:.2f}x[/]"
    elif 0.3 <= v <= 3.0:
        return f"[yellow]{v:.2f}x[/]"
    else:
        return f"[indian_red1]{v:.2f}x[/]"


def _grade_color(grade):
    colors = {"A": "bright_green", "B": "bright_green", "C": "yellow", "D": "orange1", "F": "indian_red1"}
    return f"[{colors.get(grade, 'white')}]{grade}[/]"


# ===========================================================================
# Rendering: Summary Panel
# ===========================================================================
def render_summary_panel(results, elapsed, n_total, n_failed, eval_days, eval_spacing):
    if not HAS_RICH:
        return
    n_verified = len(results)

    # Aggregate per-horizon hit rates
    horizon_stats = {}
    for lbl in HORIZON_LABELS.values():
        hits = [r[1].get(lbl, {}).get("hit_rate") for r in results if r[1].get(lbl, {}).get("hit_rate") is not None]
        briers = [r[1].get(lbl, {}).get("brier") for r in results if r[1].get(lbl, {}).get("brier") is not None]
        maes = [r[1].get(lbl, {}).get("mae") for r in results if r[1].get(lbl, {}).get("mae") is not None]
        if hits:
            horizon_stats[lbl] = {
                "avg_hit": np.mean(hits),
                "avg_brier": np.mean(briers) if briers else None,
                "avg_mae": np.mean(maes) if maes else None,
            }

    grade_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
    for r in results:
        grade_dist[r[4]] = grade_dist.get(r[4], 0) + 1

    pct_above_50_7d = 0
    h7_vals = [r[1].get("1w", {}).get("hit_rate") for r in results if r[1].get("1w", {}).get("hit_rate") is not None]
    if h7_vals:
        pct_above_50_7d = sum(1 for h in h7_vals if h > 0.50) / len(h7_vals) * 100

    grid = Table.grid(padding=(0, 4))
    grid.add_column(style="bold white", justify="right")
    grid.add_column(style="bright_cyan")
    grid.add_column(style="bold white", justify="right")
    grid.add_column(style="bright_cyan")

    grid.add_row("Assets verified:", f"{n_verified}", "Failed/skipped:", f"{n_failed}")
    grid.add_row("Eval window:", f"{eval_days}d (every {eval_spacing}d)", "Horizons:", "1d 3d 1w 1m 3m 6m 12m")
    grid.add_row(
        "7D hit rate > 50%:", f"[bright_cyan]{pct_above_50_7d:.0f}%[/] of assets",
        "Elapsed:", f"{elapsed:.1f}s",
    )
    grid.add_row(
        "Grades:",
        f"[bright_green]A:{grade_dist['A']}[/]  [bright_green]B:{grade_dist['B']}[/]  "
        f"[yellow]C:{grade_dist['C']}[/]  [orange1]D:{grade_dist['D']}[/]  "
        f"[indian_red1]F:{grade_dist['F']}[/]",
        "", "",
    )

    panel = Panel(grid, title="[bold bright_cyan]>>> SIGNALS TABLE VERIFICATION SUMMARY[/]",
                  border_style="bright_cyan", box=box.DOUBLE, padding=(1, 2))
    console.print()
    console.print(Align.center(panel))
    console.print()


# ===========================================================================
# Rendering: Per-horizon aggregate table
# ===========================================================================
def render_horizon_summary(results):
    if not HAS_RICH or not results:
        return

    table = Table(title="[bold]Per-Horizon Aggregate Metrics[/]", box=box.SIMPLE_HEAVY,
                  header_style="bold bright_cyan")
    table.add_column("Horizon", style="bold white", min_width=8)
    table.add_column("N Assets", justify="right")
    table.add_column("Eval Pts", justify="right")
    table.add_column("Avg Hit%", justify="right")
    table.add_column("Avg Brier", justify="right")
    table.add_column("Avg MAE", justify="right")
    table.add_column("Avg Bias", justify="right")
    table.add_column("Avg Corr", justify="right")
    table.add_column("Avg Mag Ratio", justify="right")

    for H in VERIFY_HORIZONS:
        lbl = HORIZON_LABELS[H]
        hits = []
        briers = []
        maes = []
        biases = []
        corrs = []
        mags = []
        total_pts = 0

        for _, metrics, _, _, _ in results:
            m = metrics.get(lbl, {})
            if "hit_rate" in m:
                hits.append(m["hit_rate"])
                briers.append(m.get("brier", 0))
                maes.append(m.get("mae", 0))
                biases.append(m.get("bias", 0))
                corrs.append(m.get("corr", 0))
                mags.append(m.get("mag_ratio", 1.0))
                total_pts += m.get("n", 0)

        if not hits:
            continue

        table.add_row(
            lbl,
            f"{len(hits)}",
            f"{total_pts:,}",
            _color_hit(np.mean(hits)),
            _color_brier(np.mean(briers)),
            _color_mae(np.mean(maes)),
            f"[{'bright_green' if abs(np.mean(biases)) < 1.0 else 'yellow'}]{np.mean(biases):+.2f}%[/]",
            _color_val(np.mean(corrs), ".3f", 0.10, 0.0, True),
            _color_mag(np.mean(mags)),
        )

    console.print(table)
    console.print()


# ===========================================================================
# Rendering: Calibration table (p_up reliability)
# ===========================================================================
def render_calibration_table(results):
    if not HAS_RICH or not results:
        return

    # Aggregate calibration bins for 1w horizon
    agg_bins: Dict[str, Dict] = {}
    for _, metrics, _, _, _ in results:
        cal = metrics.get("1w", {}).get("calibration", {})
        for bin_label, data in cal.items():
            if bin_label not in agg_bins:
                agg_bins[bin_label] = {"n": 0, "pred_sum": 0.0, "actual_sum": 0.0}
            agg_bins[bin_label]["n"] += data["n"]
            agg_bins[bin_label]["pred_sum"] += data["predicted_p_up"] * data["n"]
            agg_bins[bin_label]["actual_sum"] += data["actual_freq_up"] * data["n"]

    if not agg_bins:
        return

    table = Table(title="[bold]p_up Calibration (7D horizon — reliability diagram)[/]",
                  box=box.SIMPLE_HEAVY, header_style="bold bright_cyan")
    table.add_column("p_up Bin", style="bold white")
    table.add_column("N Points", justify="right")
    table.add_column("Avg Predicted p_up", justify="right")
    table.add_column("Actual Freq Up", justify="right")
    table.add_column("Gap", justify="right")
    table.add_column("Quality", justify="center")

    for bin_label in sorted(agg_bins.keys()):
        data = agg_bins[bin_label]
        n = data["n"]
        if n < 5:
            continue
        pred_avg = data["pred_sum"] / n
        actual_avg = data["actual_sum"] / n
        gap = abs(pred_avg - actual_avg)

        if gap < 0.05:
            quality = "[bright_green]●[/] Excellent"
        elif gap < 0.10:
            quality = "[yellow]●[/] Good"
        elif gap < 0.15:
            quality = "[orange1]●[/] Fair"
        else:
            quality = "[indian_red1]●[/] Poor"

        table.add_row(
            bin_label,
            f"{n:,}",
            f"{pred_avg:.3f}",
            f"{actual_avg:.3f}",
            f"[{'bright_green' if gap < 0.05 else 'yellow' if gap < 0.10 else 'indian_red1'}]{gap:.3f}[/]",
            quality,
        )

    console.print(table)
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

    table = Table(title="[bold]Accuracy by Asset Type[/]", box=box.SIMPLE_HEAVY,
                  header_style="bold bright_cyan")
    table.add_column("Asset Type", style="bold white", min_width=10)
    table.add_column("Count", justify="right")
    table.add_column("1w Hit%", justify="right")
    table.add_column("1m Hit%", justify="right")
    table.add_column("1w Brier", justify="right")
    table.add_column("Avg Score", justify="right")
    table.add_column("A+B %", justify="right")

    for at in ["equity", "metal", "currency", "crypto"]:
        items = by_type.get(at, [])
        if not items:
            continue
        h7 = [m["1w"]["hit_rate"] for m, _, _ in items if m.get("1w", {}).get("hit_rate") is not None]
        h21 = [m["1m"]["hit_rate"] for m, _, _ in items if m.get("1m", {}).get("hit_rate") is not None]
        b7 = [m["1w"]["brier"] for m, _, _ in items if m.get("1w", {}).get("brier") is not None]
        scores = [s for _, s, _ in items]
        ab_pct = sum(1 for _, _, g in items if g in ("A", "B")) / max(len(items), 1) * 100
        table.add_row(
            at.capitalize(), f"{len(items)}",
            _color_hit(np.mean(h7)) if h7 else "[dim]---[/]",
            _color_hit(np.mean(h21)) if h21 else "[dim]---[/]",
            _color_brier(np.mean(b7)) if b7 else "[dim]---[/]",
            f"[bright_cyan]{np.mean(scores):.2f}[/]" if scores else "[dim]---[/]",
            f"[bright_green]{ab_pct:.0f}%[/]" if ab_pct >= 50 else f"[yellow]{ab_pct:.0f}%[/]",
        )
    console.print(table)
    console.print()


# ===========================================================================
# Rendering: Main per-asset table
# ===========================================================================
def render_main_table(results):
    if not HAS_RICH or not results:
        return

    table = Table(title="[bold]Per-Asset Signals Verification[/]", box=box.SIMPLE_HEAVY,
                  header_style="bold bright_cyan", show_lines=False)
    table.add_column("Asset", style="bold white", min_width=10)
    table.add_column("Type", style="dim")
    table.add_column("Grade", justify="center")
    table.add_column("Score", justify="right")

    # Hit rate per horizon
    for lbl in ["1d", "3d", "1w", "1m", "3m", "6m", "12m"]:
        table.add_column(f"Hit {lbl}", justify="right")

    table.add_column("Brier 1w", justify="right")
    table.add_column("MAE 1w", justify="right")
    table.add_column("Mag 1w", justify="right")
    table.add_column("Model", style="dim", max_width=20)

    for symbol, metrics, tune_q, comp_score, grade in results:
        row = [
            symbol,
            _detect_asset_type(symbol)[:4],
            _grade_color(grade),
            f"{comp_score:.2f}",
        ]

        for lbl in ["1d", "3d", "1w", "1m", "3m", "6m", "12m"]:
            hr = metrics.get(lbl, {}).get("hit_rate")
            row.append(_color_hit(hr) if hr is not None else "[dim]---[/]")

        brier_1w = metrics.get("1w", {}).get("brier")
        mae_1w = metrics.get("1w", {}).get("mae")
        mag_1w = metrics.get("1w", {}).get("mag_ratio")
        row.append(_color_brier(brier_1w))
        row.append(_color_mae(mae_1w))
        row.append(_color_mag(mag_1w))
        row.append(str(tune_q.get("best_model", "—"))[:20])

        table.add_row(*row)

    console.print(table)
    console.print()


# ===========================================================================
# Rendering: Magnitude outliers
# ===========================================================================
def render_magnitude_outliers(results):
    if not HAS_RICH or not results:
        return

    outliers = []
    for symbol, metrics, _, _, grade in results:
        for lbl in ["1w", "1m", "3m"]:
            m = metrics.get(lbl, {})
            mag = m.get("mag_ratio")
            if mag is not None and (mag > 3.0 or mag < 0.3):
                outliers.append((symbol, lbl, mag, m.get("bias", 0), m.get("mae", 0)))

    if not outliers:
        return

    table = Table(title="[bold indian_red1]⚠ Magnitude Outliers (mag ratio > 3x or < 0.3x)[/]",
                  box=box.SIMPLE_HEAVY, header_style="bold indian_red1")
    table.add_column("Asset", style="bold white")
    table.add_column("Horizon", justify="center")
    table.add_column("Mag Ratio", justify="right")
    table.add_column("Avg Bias", justify="right")
    table.add_column("MAE", justify="right")

    for sym, lbl, mag, bias, mae in sorted(outliers, key=lambda x: abs(x[2] - 1.0), reverse=True)[:25]:
        table.add_row(
            sym, lbl, _color_mag(mag),
            f"{bias:+.2f}%",
            f"{mae:.2f}%",
        )

    console.print(table)
    console.print()


# ===========================================================================
# Rendering: Problem assets
# ===========================================================================
def render_problem_assets(results):
    if not HAS_RICH or not results:
        return

    problems = []
    for symbol, metrics, _, comp_score, grade in results:
        issues = []
        m1w = metrics.get("1w", {})
        if m1w.get("hit_rate") is not None and m1w["hit_rate"] < 0.45:
            issues.append(f"Hit 1w: {m1w['hit_rate']:.0%}")
        if m1w.get("brier") is not None and m1w["brier"] > 0.30:
            issues.append(f"Brier: {m1w['brier']:.3f}")
        if m1w.get("mag_ratio") is not None and (m1w["mag_ratio"] > 5.0 or m1w["mag_ratio"] < 0.1):
            issues.append(f"Mag: {m1w['mag_ratio']:.1f}x")
        m1m = metrics.get("1m", {})
        if m1m.get("hit_rate") is not None and m1m["hit_rate"] < 0.45:
            issues.append(f"Hit 1m: {m1m['hit_rate']:.0%}")
        if issues:
            problems.append((symbol, grade, comp_score, "; ".join(issues)))

    if not problems:
        return

    table = Table(title="[bold indian_red1]Problem Assets[/]", box=box.SIMPLE_HEAVY,
                  header_style="bold indian_red1")
    table.add_column("Asset", style="bold white")
    table.add_column("Grade", justify="center")
    table.add_column("Issues", style="indian_red1")

    for sym, grade, _, issues in sorted(problems, key=lambda x: x[2])[:30]:
        table.add_row(sym, _grade_color(grade), issues)

    console.print(table)
    console.print()


# ===========================================================================
# Rendering: Worst misses
# ===========================================================================
def render_worst_misses(results, n_worst=15):
    if not HAS_RICH or not results:
        return

    all_misses = []
    for symbol, metrics, _, _, _ in results:
        for lbl in ["1w", "1m"]:
            for miss in metrics.get(lbl, {}).get("worst_misses", []):
                all_misses.append({**miss, "symbol": symbol, "horizon": lbl})

    if not all_misses:
        return

    all_misses.sort(key=lambda x: abs(x["error"]), reverse=True)

    table = Table(title=f"[bold]Top {n_worst} Worst Misses (1w + 1m)[/]",
                  box=box.SIMPLE_HEAVY, header_style="bold bright_cyan")
    table.add_column("Asset", style="bold white")
    table.add_column("Hz", justify="center")
    table.add_column("Date")
    table.add_column("Predicted", justify="right")
    table.add_column("Actual", justify="right")
    table.add_column("Error", justify="right")
    table.add_column("p_up", justify="right")
    table.add_column("Label", justify="center")

    for miss in all_misses[:n_worst]:
        err = miss["error"]
        table.add_row(
            miss["symbol"],
            miss["horizon"],
            miss["date"],
            f"{miss['predicted']:+.2f}%",
            f"{miss['actual']:+.2f}%",
            f"[indian_red1]{err:+.2f}%[/]",
            f"{miss['p_up']:.2f}",
            miss["label"],
        )

    console.print(table)
    console.print()


# ===========================================================================
# JSON export
# ===========================================================================
def export_json(results, failed, path, eval_days, eval_spacing):
    out = {
        "generated_at": datetime.now().isoformat(),
        "eval_days": eval_days,
        "eval_spacing": eval_spacing,
        "n_verified": len(results),
        "n_failed": len(failed),
        "assets": {},
        "failed": [{"symbol": s, "error": e} for s, e in failed],
    }
    for symbol, metrics, tune_q, comp_score, grade in results:
        asset_entry = {"grade": grade, "score": comp_score}
        for lbl in HORIZON_LABELS.values():
            m = metrics.get(lbl, {})
            asset_entry[lbl] = {
                k: v for k, v in m.items()
                if k not in ("calibration", "regime_acc", "worst_misses")
            }
        out["assets"][symbol] = asset_entry

    # Aggregate
    agg = {}
    for lbl in HORIZON_LABELS.values():
        hits = [r[1].get(lbl, {}).get("hit_rate") for r in results if r[1].get(lbl, {}).get("hit_rate") is not None]
        briers = [r[1].get(lbl, {}).get("brier") for r in results if r[1].get(lbl, {}).get("brier") is not None]
        if hits:
            agg[lbl] = {
                "avg_hit_rate": float(np.mean(hits)),
                "avg_brier": float(np.mean(briers)) if briers else None,
                "n_assets": len(hits),
            }
    out["aggregate"] = agg

    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    if HAS_RICH:
        console.print(f"[dim]Exported JSON → {path}[/]")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward verification of signals table (compute_features → latest_signals)"
    )
    parser.add_argument("--assets", type=str, default=None,
                        help="Comma-separated list of assets (default: all with tune cache + price data)")
    parser.add_argument("--eval-days", type=int, default=DEFAULT_EVAL_DAYS,
                        help=f"Trading days to evaluate (default: {DEFAULT_EVAL_DAYS})")
    parser.add_argument("--eval-spacing", type=int, default=DEFAULT_EVAL_SPACING,
                        help=f"Evaluate every Nth day (default: {DEFAULT_EVAL_SPACING})")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count - 1)")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel processing")
    parser.add_argument("--sort", type=str, default="grade",
                        choices=["grade", "hit1w", "hit1m", "brier1w", "mae1w", "symbol"],
                        help="Sort results by (default: grade)")
    parser.add_argument("--worst", type=int, default=15,
                        help="Number of worst misses to show (default: 15)")
    parser.add_argument("--export-json", type=str, default=None,
                        help="Export results to JSON file")
    args = parser.parse_args()

    # Resolve asset list
    if args.assets:
        assets = [s.strip() for s in args.assets.split(",") if s.strip()]
    else:
        # Find assets that have BOTH tune cache and price data
        tune_assets = set()
        if TUNE_CACHE_DIR.exists():
            for p in TUNE_CACHE_DIR.glob("*.json"):
                tune_assets.add(p.stem)
        price_assets = set()
        if PRICE_CACHE_DIR.exists():
            for p in PRICE_CACHE_DIR.glob("*.csv"):
                price_assets.add(p.stem)
        # Intersect and convert back to original symbols
        valid = tune_assets & price_assets
        # Try to use the default universe ordering if available
        try:
            from ingestion.data_utils import get_default_asset_universe
            universe = get_default_asset_universe()
            safe_map = {}
            for sym in universe:
                safe = sym.replace("/", "_").replace("=", "_").replace(":", "_").upper()
                safe_map[safe] = sym
            assets = [safe_map[s] for s in valid if s in safe_map]
            # Add remaining
            remaining = [s for s in valid if s not in safe_map]
            assets.extend(remaining)
        except Exception:
            assets = sorted(valid)

    n_total = len(assets)
    if n_total == 0:
        print("No assets found with both tune cache and price data.")
        return

    if HAS_RICH:
        console.print(Rule(style="bright_cyan"))
        console.print(Panel(
            f"[bold]Signals Table Walk-Forward Verification[/]\n"
            f"[dim]Assets: {n_total} | Eval: {args.eval_days}d every {args.eval_spacing}d | "
            f"Horizons: {', '.join(HORIZON_LABELS.values())}[/]",
            border_style="bright_cyan", box=box.DOUBLE,
        ))
        console.print()

    work_items = [(sym, args.eval_days, args.eval_spacing) for sym in assets]

    t0 = time.time()
    all_results = []
    failed_assets = []
    completed = 0
    n_workers = args.workers or max(1, (mp.cpu_count() or 4) - 1)

    if args.no_parallel or n_total <= 2:
        for item in work_items:
            sym, metrics, tune_q, error = verify_single_asset(item)
            completed += 1
            if error:
                if HAS_RICH:
                    console.print(f"  [dim][{completed}/{n_total}][/] [indian_red1]{sym}: {error}[/]")
                failed_assets.append((sym, error))
            else:
                m1w = metrics.get("1w", {}).get("hit_rate") if metrics else None
                hit_str = f"{m1w*100:.0f}%" if m1w else "---"
                if HAS_RICH:
                    console.print(f"  [dim][{completed}/{n_total}][/] [bright_green]{sym}[/] 1w={hit_str}")
                all_results.append((sym, metrics, tune_q, None))
    else:
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
            futures_map = {}
            for item in work_items:
                future = executor.submit(verify_single_asset, item)
                futures_map[future] = item[0]

            for future in as_completed(futures_map):
                symbol_key = futures_map[future]
                completed += 1
                try:
                    sym, metrics, tune_q, error = future.result()
                    if error:
                        if HAS_RICH:
                            console.print(f"  [dim][{completed}/{n_total}][/] [indian_red1]{sym}: {error}[/]")
                        failed_assets.append((sym, error))
                    else:
                        m1w = metrics.get("1w", {}).get("hit_rate") if metrics else None
                        hit_str = f"{m1w*100:.0f}%" if m1w else "---"
                        if HAS_RICH:
                            console.print(f"  [dim][{completed}/{n_total}][/] [bright_green]{sym}[/] 1w={hit_str}")
                        all_results.append((sym, metrics, tune_q, None))
                except Exception as e:
                    if HAS_RICH:
                        console.print(f"  [dim][{completed}/{n_total}][/] [indian_red1]{symbol_key}: {e}[/]")
                    failed_assets.append((symbol_key, str(e)))

    elapsed = time.time() - t0

    if not all_results:
        if HAS_RICH:
            console.print("[indian_red1]No assets produced valid results.[/]")
        return

    # Score and grade
    scored_results = []
    for sym, metrics, tune_q, _ in all_results:
        comp_score, grade = _composite_score(metrics, tune_q)
        scored_results.append((sym, metrics, tune_q, comp_score, grade))

    # Sort
    sort_keys = {
        "grade": lambda x: (-x[3], x[0]),
        "hit1w": lambda x: -(x[1].get("1w", {}).get("hit_rate", 0)),
        "hit1m": lambda x: -(x[1].get("1m", {}).get("hit_rate", 0)),
        "brier1w": lambda x: x[1].get("1w", {}).get("brier", 999),
        "mae1w": lambda x: x[1].get("1w", {}).get("mae", 999),
        "symbol": lambda x: x[0],
    }
    scored_results.sort(key=sort_keys.get(args.sort, sort_keys["grade"]))

    if HAS_RICH:
        console.print()
        console.print(Rule(style="bright_cyan"))

    # ---- Render all sections ----
    render_summary_panel(scored_results, elapsed, n_total, len(failed_assets), args.eval_days, args.eval_spacing)
    render_horizon_summary(scored_results)
    render_calibration_table(scored_results)
    render_sector_summary(scored_results)
    render_main_table(scored_results)
    render_magnitude_outliers(scored_results)
    render_problem_assets(scored_results)
    render_worst_misses(scored_results, n_worst=args.worst)

    # JSON export
    if args.export_json:
        export_json(scored_results, failed_assets, args.export_json, args.eval_days, args.eval_spacing)

    if HAS_RICH:
        console.print(Rule(style="dim"))
        console.print(
            f"[dim]Completed {len(scored_results)} assets in {elapsed:.1f}s "
            f"({len(failed_assets)} failed/skipped)[/]"
        )
        console.print()


if __name__ == "__main__":
    main()
