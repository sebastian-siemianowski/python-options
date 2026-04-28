#!/usr/bin/env python3
"""Focused A/B benchmark for the canonical phi_student_t fixed-nu optimizer."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import kstest, t as student_t_dist


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("OFFLINE_MODE", "1")
os.environ.setdefault("TUNING_QUIET", "1")

from ingestion.data_utils import _download_prices  # noqa: E402
from calibration.realized_volatility import compute_hybrid_volatility_har  # noqa: E402
from models.phi_student_t import PhiStudentTDriftModel  # noqa: E402
from models.phi_student_t_improved import PhiStudentTDriftModel as ImprovedPhiStudentTDriftModel  # noqa: E402
from tuning.benchmark_retune_50 import BENCHMARK_50  # noqa: E402


def _physical_workers() -> int:
    count = os.cpu_count() or 2
    if sys.platform == "darwin":
        try:
            count = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip())
        except Exception:
            pass
    return max(1, count - 1)


def _prepare_asset(symbol: str) -> tuple[np.ndarray, np.ndarray]:
    df = _download_prices(symbol, "2015-01-01", None)
    if df is None or df.empty:
        raise ValueError("empty price data")
    cols = {c.lower(): c for c in df.columns}
    close = df[cols["close"]].astype(float)
    returns = np.log(close / close.shift(1)).dropna().to_numpy(dtype=np.float64)
    df_aligned = df.iloc[1:].copy()
    vol, _ = compute_hybrid_volatility_har(
        open_=df_aligned[cols["open"]].to_numpy(dtype=np.float64),
        high=df_aligned[cols["high"]].to_numpy(dtype=np.float64),
        low=df_aligned[cols["low"]].to_numpy(dtype=np.float64),
        close=df_aligned[cols["close"]].to_numpy(dtype=np.float64),
        span=21,
        annualize=False,
        use_har=True,
    )
    n = min(len(returns), len(vol))
    returns = returns[:n]
    vol = vol[:n]
    valid = np.isfinite(returns) & np.isfinite(vol) & (vol > 0) & (np.abs(returns) > 1e-10)
    return (
        np.ascontiguousarray(returns[valid], dtype=np.float64),
        np.ascontiguousarray(vol[valid], dtype=np.float64),
    )


def _finite_mean(values: List[float]) -> Optional[float]:
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else None


def _profit_metrics(actual: np.ndarray, p_up: np.ndarray) -> Dict[str, float]:
    position = np.zeros_like(actual)
    position[p_up >= 0.55] = 1.0
    position[p_up <= 0.45] = -1.0
    active = position != 0.0
    pnl = position[active] * actual[active]
    if pnl.size == 0:
        return {
            "trade_count": 0,
            "strategy_mean_return": 0.0,
            "strategy_sharpe": 0.0,
            "profit_factor": 0.0,
            "trade_hit_rate": 0.0,
        }
    std = float(np.std(pnl))
    gains = pnl[pnl > 0.0]
    losses = pnl[pnl < 0.0]
    return {
        "trade_count": int(pnl.size),
        "strategy_mean_return": float(np.mean(pnl)),
        "strategy_sharpe": float(np.mean(pnl) / std * np.sqrt(252.0)) if std > 1e-12 else 0.0,
        "profit_factor": float(np.sum(gains) / max(abs(np.sum(losses)), 1e-12)),
        "trade_hit_rate": float(np.mean(pnl > 0.0)),
    }


def _oos_quality(
    model_cls,
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    split: int,
    vov: np.ndarray,
) -> Dict[str, float]:
    _, _, mu_pred, s_pred, _ = model_cls.filter_phi_with_predictive(
        returns,
        vol,
        q,
        c,
        phi,
        nu,
        robust_wt=True,
        online_scale_adapt=True,
        gamma_vov=model_cls._GAMMA_VOV_DEFAULT,
        vov_rolling=vov,
    )
    actual = returns[split:]
    pred = mu_pred[split:]
    variance = np.maximum(s_pred[split:], 1e-20)
    scale_factor = (nu - 2.0) / nu if nu > 2.0 else 1.0
    scale = np.sqrt(np.maximum(variance * scale_factor, 1e-20))
    z = (actual - pred) / scale
    ll = student_t_dist.logpdf(z, df=nu) - np.log(scale)
    p_up = 1.0 - student_t_dist.cdf((0.0 - pred) / scale, df=nu)
    pit = np.clip(student_t_dist.cdf(z, df=nu), 1e-8, 1.0 - 1e-8)
    pred_dirs = np.sign(p_up - 0.5)
    actual_dirs = np.sign(actual)
    nonzero = pred_dirs != 0.0
    hit = float(np.mean(pred_dirs[nonzero] == actual_dirs[nonzero])) if np.any(nonzero) else 0.5
    brier = float(np.mean((p_up - (actual > 0.0).astype(np.float64)) ** 2))
    try:
        pit_ks_p = float(kstest(pit, "uniform").pvalue)
    except Exception:
        pit_ks_p = 0.0
    centered_z2 = (actual - pred) * (actual - pred) / variance
    quality = {
        "oos_count": int(len(actual)),
        "oos_log_score_mean": float(np.mean(ll)),
        "oos_brier": brier,
        "oos_hit_rate": hit,
        "oos_pit_ks_p": pit_ks_p,
        "oos_variance_ratio": float(np.mean(np.clip(centered_z2, 0.0, 50.0))),
    }
    quality.update(_profit_metrics(actual, p_up))
    return quality


def _worker(args: tuple[str, float, bool, str]) -> Dict[str, Any]:
    symbol, nu, disable_train_state, model = args
    env_key = (
        "PHI_STUDENT_T_IMPROVED_DISABLE_TRAIN_STATE_KERNEL"
        if model == "improved" else "PHI_STUDENT_T_DISABLE_TRAIN_STATE_KERNEL"
    )
    cv_env_key = "PHI_STUDENT_T_IMPROVED_DISABLE_CV_TEST_KERNEL" if model == "improved" else None
    fused_env_key = "PHI_STUDENT_T_IMPROVED_DISABLE_FUSED_CV_OBJECTIVE" if model == "improved" else None
    state_only_env_key = "PHI_STUDENT_T_IMPROVED_DISABLE_STATE_ONLY_KERNEL" if model == "improved" else None
    if disable_train_state:
        os.environ[env_key] = "1"
        if cv_env_key is not None:
            os.environ[cv_env_key] = "1"
        if fused_env_key is not None:
            os.environ[fused_env_key] = "1"
        if state_only_env_key is not None:
            os.environ[state_only_env_key] = "1"
    else:
        os.environ.pop(env_key, None)
        if cv_env_key is not None:
            os.environ.pop(cv_env_key, None)
        if fused_env_key is not None:
            os.environ.pop(fused_env_key, None)
        if state_only_env_key is not None:
            os.environ.pop(state_only_env_key, None)
    returns, vol = _prepare_asset(symbol)
    model_cls = ImprovedPhiStudentTDriftModel if model == "improved" else PhiStudentTDriftModel
    split = max(80, min(int(len(returns) * 0.70), len(returns) - 30))
    vov = model_cls._precompute_vov(vol)
    t0 = time.perf_counter()
    q, c, phi, ll, diag = model_cls.optimize_params_fixed_nu(
        returns[:split],
        vol[:split],
        nu=nu,
        asset_symbol=symbol,
        gamma_vov=model_cls._GAMMA_VOV_DEFAULT,
        vov_rolling=vov[:split],
    )
    elapsed = time.perf_counter() - t0
    quality = _oos_quality(model_cls, returns, vol, q, c, phi, nu, split, vov)
    return {
        "symbol": symbol,
        "seconds": elapsed,
        "q": float(q),
        "c": float(c),
        "phi": float(phi),
        "ll": float(ll),
        "converged": bool(diag.get("optimizer_converged", False)),
        "optimizer_message": str(diag.get("optimizer_message", "")),
        "fused_cv_objective_enabled": bool(diag.get("fused_cv_objective_enabled", False)),
        "state_only_kernel_enabled": bool(diag.get("state_only_kernel_enabled", False)),
        "cv_test_kernel_enabled": bool(diag.get("cv_test_kernel_enabled", False)),
        "cv_var_calibration_enabled": bool(diag.get("cv_var_calibration_enabled", False)),
        "optimizer_maxiter": int(diag.get("optimizer_maxiter", 0) or 0),
        "fun": float(diag.get("objective", float("nan"))),
        **quality,
    }


def _run(label: str, assets: List[str], nu: float, workers: int, disable: bool, model: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    results: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_worker, (sym, nu, disable, model)): sym for sym in assets}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                results[sym] = fut.result()
            except Exception as exc:
                failures[sym] = str(exc)
    seconds = time.perf_counter() - t0
    per_asset = [r["seconds"] for r in results.values()]
    return {
        "label": label,
        "disable_train_state_kernel": disable,
        "workers": workers,
        "nu": nu,
        "asset_count": len(assets),
        "success_count": len(results),
        "failed_assets": failures,
        "total_seconds": seconds,
        "asset_seconds_mean": float(np.mean(per_asset)) if per_asset else None,
        "asset_seconds_max": float(np.max(per_asset)) if per_asset else None,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--nu", type=float, default=8.0)
    parser.add_argument("--model", choices=["canonical", "improved"], default="canonical")
    parser.add_argument("--metrics-json", default="src/data/benchmarks/phi_student_t_fixed_nu_metrics.json")
    parser.add_argument("--after-only", action="store_true")
    args = parser.parse_args()

    assets = list(BENCHMARK_50)
    workers = args.workers if args.workers > 0 else _physical_workers()
    workers = max(1, min(workers, len(assets)))
    before = None if args.after_only else _run("before_train_state_kernel", assets, args.nu, workers, True, args.model)
    after = _run("after_train_state_kernel", assets, args.nu, workers, False, args.model)

    paired = []
    if before is not None:
        for sym, after_result in after["results"].items():
            before_result = before["results"].get(sym)
            if not before_result:
                continue
            paired.append({
                "symbol": sym,
                "seconds_delta": before_result["seconds"] - after_result["seconds"],
                "q_delta": before_result["q"] - after_result["q"],
                "c_delta": before_result["c"] - after_result["c"],
                "phi_delta": before_result["phi"] - after_result["phi"],
                "ll_delta": before_result["ll"] - after_result["ll"],
            })

    speedup = (
        before["total_seconds"] / after["total_seconds"]
        if before is not None and after["total_seconds"] > 0 else None
    )
    summary = {
        "workers": workers,
        "model": args.model,
        "nu": args.nu,
        "speedup_total": speedup,
        "before_total_seconds": before["total_seconds"] if before is not None else None,
        "after_total_seconds": after["total_seconds"],
        "paired_count": len(paired),
        "max_abs_q_delta": max((abs(p["q_delta"]) for p in paired), default=None),
        "max_abs_c_delta": max((abs(p["c_delta"]) for p in paired), default=None),
        "max_abs_phi_delta": max((abs(p["phi_delta"]) for p in paired), default=None),
        "max_abs_ll_delta": max((abs(p["ll_delta"]) for p in paired), default=None),
        "after_oos_log_score_mean": _finite_mean([r.get("oos_log_score_mean") for r in after["results"].values()]),
        "after_oos_brier_mean": _finite_mean([r.get("oos_brier") for r in after["results"].values()]),
        "after_oos_hit_rate_mean": _finite_mean([r.get("oos_hit_rate") for r in after["results"].values()]),
        "after_oos_pit_ks_p_mean": _finite_mean([r.get("oos_pit_ks_p") for r in after["results"].values()]),
        "after_oos_variance_ratio_mean": _finite_mean([r.get("oos_variance_ratio") for r in after["results"].values()]),
        "after_strategy_sharpe_mean": _finite_mean([r.get("strategy_sharpe") for r in after["results"].values()]),
        "after_profit_factor_mean": _finite_mean([r.get("profit_factor") for r in after["results"].values()]),
        "after_trade_hit_rate_mean": _finite_mean([r.get("trade_hit_rate") for r in after["results"].values()]),
    }
    payload = {"summary": summary, "before": before, "after": after, "paired": paired}
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
