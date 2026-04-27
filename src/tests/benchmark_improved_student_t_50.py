#!/usr/bin/env python3
"""Benchmark improved Student-t models on the canonical 50-equity universe.

This is intentionally a script, not a unittest: it measures predictive scoring,
runtime, and a simple long-only signal proxy before/after model changes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import sys
import time
from pathlib import Path

# Keep benchmark parallelism process-based. Avoid nested BLAS/OpenMP thread pools
# multiplying CPU usage inside each worker process.
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena.backtest_config import BACKTEST_UNIVERSE, Sector
from ingestion.data_utils import DEFAULT_ASSET_UNIVERSE
from models.asset_classification import detect_asset_class
from models.phi_student_t_improved import PhiStudentTDriftModel as ImprovedPhiStudentTDriftModel
from models.phi_student_t_unified_improved import UnifiedPhiStudentTModel as ImprovedUnifiedPhiStudentTModel
from tuning.diagnostics import compute_crps_student_t_inline


_NON_STOCK_ETFS = {
    "SPY", "VOO", "VTI", "QQQ", "IWM", "OEF", "DIA", "RSP", "SMH", "SOXX",
    "GLD", "GLDM", "IAU", "SGOL", "SGLP", "SGLP.L", "GLDE", "GLDW",
    "SLV", "SIVR", "SLVI", "GDX", "GDXJ", "COPX", "PPLT",
}


def _equity_symbols() -> list[str]:
    symbols = []
    for symbol, info in BACKTEST_UNIVERSE.items():
        path = REPO_ROOT / "src" / "data" / "prices" / f"{symbol}.csv"
        if info.get("sector") != Sector.INDEX and path.exists():
            symbols.append(symbol)
    for symbol in DEFAULT_ASSET_UNIVERSE:
        symbol = str(symbol).strip().upper()
        if symbol in symbols:
            continue
        if symbol in _NON_STOCK_ETFS:
            continue
        if "=" in symbol or symbol.endswith("-USD"):
            continue
        asset_class = detect_asset_class(symbol)
        if asset_class in {"index", "forex", "crypto"}:
            continue
        path = REPO_ROOT / "src" / "data" / "prices" / f"{symbol}.csv"
        if path.exists():
            symbols.append(symbol)
        if len(symbols) >= 50:
            break
    return symbols[:50]


def _load_returns(symbol: str, max_obs: int) -> tuple[np.ndarray, np.ndarray]:
    path = REPO_ROOT / "src" / "data" / "prices" / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    prices = pd.to_numeric(df[price_col], errors="coerce").dropna()
    prices = prices[prices > 0]
    log_returns = np.log(prices).diff().dropna()
    if max_obs and len(log_returns) > max_obs:
        log_returns = log_returns.iloc[-max_obs:]
    returns = log_returns.to_numpy(dtype=np.float64)
    vol = log_returns.ewm(span=21, adjust=False, min_periods=10).std().bfill().ffill()
    vol_arr = vol.to_numpy(dtype=np.float64)
    finite = np.isfinite(vol_arr) & (vol_arr > 0)
    fallback = float(np.std(returns[np.isfinite(returns)])) if np.any(np.isfinite(returns)) else 0.01
    fallback = max(fallback, 1e-4)
    vol_arr = np.where(finite, vol_arr, fallback)
    return returns, np.maximum(vol_arr, fallback * 1e-4)


def _predictive_scores(returns: np.ndarray, mu_pred: np.ndarray, s_pred: np.ndarray,
                       nu: float, split: int) -> dict:
    r = returns[split:]
    mu = mu_pred[split:]
    s_var = np.maximum(s_pred[split:], 1e-12)
    scale_factor = (float(nu) - 2.0) / float(nu) if float(nu) > 2.0 else 1.0
    sigma = np.sqrt(s_var * scale_factor)
    z = np.clip((r - mu) / np.maximum(sigma, 1e-12), -1e6, 1e6)
    pit = np.asarray(student_t.cdf(z, df=float(nu)), dtype=np.float64)
    pit = np.clip(pit[np.isfinite(pit)], 1e-12, 1.0 - 1e-12)
    p_up = 1.0 - student_t.cdf((0.0 - mu) / np.maximum(sigma, 1e-12), df=float(nu))
    positions = (p_up > 0.55).astype(np.float64)
    strategy = positions * r
    turnover = np.abs(np.diff(np.r_[0.0, positions]))
    strategy_net = strategy - turnover * 0.0002
    equity = np.cumsum(strategy_net)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.min(equity - peak)) if len(equity) else 0.0
    mean = float(np.mean(strategy_net)) if len(strategy_net) else 0.0
    std = float(np.std(strategy_net)) if len(strategy_net) else 0.0
    sharpe = mean / std * math.sqrt(252.0) if std > 1e-12 else 0.0
    direction = np.sign(mu)
    realized = np.sign(r)
    active = direction != 0
    hit_rate = float(np.mean(direction[active] == realized[active])) if np.any(active) else float("nan")
    crps = compute_crps_student_t_inline(r, mu, sigma, float(nu))
    if len(pit) >= 2:
        from models.phi_student_t_unified_improved import _fast_ks_uniform
        _, pit_p = _fast_ks_uniform(pit)
    else:
        pit_p = float("nan")
    return {
        "n_test": int(len(r)),
        "crps": float(crps),
        "pit_pvalue": float(pit_p),
        "pit_pass": bool(np.isfinite(pit_p) and pit_p >= 0.05),
        "direction_hit_rate": hit_rate,
        "signal_coverage": float(np.mean(positions)) if len(positions) else 0.0,
        "strategy_return": float(np.sum(strategy_net)),
        "strategy_sharpe": float(sharpe),
        "max_drawdown": max_dd,
    }


def _bench_base(symbol: str, returns: np.ndarray, vol: np.ndarray,
                split: int, nu: float) -> dict:
    train_r = returns[:split]
    train_v = vol[:split]
    start = time.perf_counter()
    q, c, phi, ll, diag = ImprovedPhiStudentTDriftModel.optimize_params_fixed_nu(
        train_r, train_v, nu=nu, train_frac=0.75, asset_symbol=symbol)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    _, _, mu_pred, s_pred, full_ll = ImprovedPhiStudentTDriftModel.filter_phi_with_predictive(
        returns, vol, q, c, phi, nu)
    filter_seconds = time.perf_counter() - start
    scores = _predictive_scores(returns, mu_pred, s_pred, nu, split)
    scores.update({
        "model": "phi_student_t_improved",
        "symbol": symbol,
        "fit_seconds": float(fit_seconds),
        "filter_seconds": float(filter_seconds),
        "log_likelihood": float(full_ll),
        "q": float(q),
        "c": float(c),
        "phi": float(phi),
        "nu": float(nu),
        "fit_success": bool(diag.get("fit_success", np.isfinite(ll))),
    })
    return scores


def _bench_unified(symbol: str, returns: np.ndarray, vol: np.ndarray,
                   split: int, nu: float) -> dict:
    train_r = returns[:split]
    train_v = vol[:split]
    start = time.perf_counter()
    config, diag = ImprovedUnifiedPhiStudentTModel.optimize_params_unified(
        train_r, train_v, nu_base=nu, train_frac=0.75, asset_symbol=symbol)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    _, _, mu_pred, s_pred, full_ll = ImprovedUnifiedPhiStudentTModel.filter_phi_unified(
        returns, vol, config)
    filter_seconds = time.perf_counter() - start
    scores = _predictive_scores(returns, mu_pred, s_pred, float(config.nu_base), split)
    scores.update({
        "model": "phi_student_t_unified_improved",
        "symbol": symbol,
        "fit_seconds": float(fit_seconds),
        "filter_seconds": float(filter_seconds),
        "log_likelihood": float(full_ll),
        "q": float(config.q),
        "c": float(config.c),
        "phi": float(config.phi),
        "nu": float(config.nu_base),
        "fit_success": bool(diag.get("success", True)),
    })
    return scores


def _bench_symbol(symbol: str, max_obs: int, nu: float, models: tuple[str, ...]) -> dict:
    """Benchmark all requested models for one symbol inside a worker process."""
    try:
        returns, vol = _load_returns(symbol, max_obs)
        if len(returns) < 400:
            raise ValueError(f"insufficient returns: {len(returns)}")
        split = int(len(returns) * 0.70)
        rows = []
        model_set = set(models)
        if "base" in model_set:
            rows.append(_bench_base(symbol, returns, vol, split, nu))
        if "unified" in model_set:
            rows.append(_bench_unified(symbol, returns, vol, split, nu))
        return {"symbol": symbol, "rows": rows, "failure": None}
    except Exception as exc:
        return {"symbol": symbol, "rows": [], "failure": {"symbol": symbol, "error": str(exc)}}


def _aggregate(rows: list[dict]) -> dict:
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    out = {}
    fields = [
        "crps", "pit_pvalue", "direction_hit_rate", "signal_coverage",
        "strategy_return", "strategy_sharpe", "max_drawdown",
        "fit_seconds", "filter_seconds",
    ]
    for model, model_rows in by_model.items():
        agg = {"n_symbols": len(model_rows)}
        for field in fields:
            vals = np.array([r[field] for r in model_rows], dtype=float)
            vals = vals[np.isfinite(vals)]
            agg[f"mean_{field}"] = float(np.mean(vals)) if vals.size else float("nan")
            agg[f"median_{field}"] = float(np.median(vals)) if vals.size else float("nan")
        agg["pit_pass_rate"] = float(np.mean([r["pit_pass"] for r in model_rows]))
        out[model] = agg
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="", help="Comma-separated override list")
    parser.add_argument("--max-symbols", type=int, default=50)
    parser.add_argument("--max-obs", type=int, default=1000)
    parser.add_argument("--nu", type=float, default=8.0)
    parser.add_argument("--models", default="base,unified",
                        help="Comma-separated: base,unified")
    parser.add_argument("--output", default="")
    parser.add_argument("--workers", type=int, default=0,
                        help="Process workers (0=all available CPUs)")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Run sequentially for debugging")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else _equity_symbols()
    symbols = symbols[:args.max_symbols]
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    symbol_order = {symbol: idx for idx, symbol in enumerate(symbols)}
    model_order = {"base": 0, "phi_student_t_improved": 0,
                   "unified": 1, "phi_student_t_unified_improved": 1}

    rows = []
    failures = []
    wall_start = time.perf_counter()
    cpu_count = os.cpu_count() or 1
    workers = cpu_count if args.workers <= 0 else max(1, args.workers)
    workers = min(workers, max(len(symbols), 1))

    if args.no_parallel or workers <= 1 or len(symbols) <= 1:
        for symbol in symbols:
            result = _bench_symbol(symbol, args.max_obs, args.nu, models)
            rows.extend(result["rows"])
            if result["failure"]:
                failures.append(result["failure"])
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_bench_symbol, symbol, args.max_obs, args.nu, models): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                result = future.result()
                rows.extend(result["rows"])
                if result["failure"]:
                    failures.append(result["failure"])

    rows.sort(key=lambda row: (
        symbol_order.get(row["symbol"], 10**9),
        model_order.get(row["model"], 10**9),
    ))
    failures.sort(key=lambda item: symbol_order.get(item["symbol"], 10**9))

    payload = {
        "symbols": symbols,
        "n_rows": len(rows),
        "workers": int(workers if not args.no_parallel else 1),
        "parallel": bool(not args.no_parallel and workers > 1 and len(symbols) > 1),
        "wall_seconds": float(time.perf_counter() - wall_start),
        "failures": failures,
        "aggregate": _aggregate(rows),
        "rows": rows,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
