#!/usr/bin/env python3
"""Focused A/B benchmark for unified Student-t Stage 5 structural precompute."""

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
from typing import Any, Dict, List

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("OFFLINE_MODE", "1")
os.environ.setdefault("TUNING_QUIET", "1")

from ingestion.data_utils import _download_prices  # noqa: E402
from calibration.realized_volatility import compute_hybrid_volatility_har  # noqa: E402
from models.phi_student_t_unified import UnifiedPhiStudentTModel  # noqa: E402
from models.phi_student_t_unified_improved import (  # noqa: E402
    UnifiedPhiStudentTModel as ImprovedUnifiedPhiStudentTModel,
)
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


def _worker(args: tuple[str, bool, str, float]) -> Dict[str, Any]:
    symbol, disable_precompute, model, nu_base = args
    if disable_precompute:
        os.environ["UNIFIED_STAGE5_DISABLE_PRECOMPUTE"] = "1"
        os.environ.pop("UNIFIED_STAGE5_ENABLE_PRECOMPUTE", None)
    else:
        os.environ.pop("UNIFIED_STAGE5_DISABLE_PRECOMPUTE", None)
        os.environ["UNIFIED_STAGE5_ENABLE_PRECOMPUTE"] = "1"
    returns, vol = _prepare_asset(symbol)
    model_cls = ImprovedUnifiedPhiStudentTModel if model == "improved" else UnifiedPhiStudentTModel
    t0 = time.perf_counter()
    config, diagnostics = model_cls.optimize_params_unified(
        returns,
        vol,
        nu_base=nu_base,
        asset_symbol=symbol,
    )
    elapsed = time.perf_counter() - t0
    return {
        "symbol": symbol,
        "seconds": elapsed,
        "q": float(config.q),
        "c": float(config.c),
        "phi": float(config.phi),
        "nu_base": float(config.nu_base),
        "variance_inflation": float(config.variance_inflation),
        "mu_drift": float(config.mu_drift),
        "success": bool(diagnostics.get("success", True)),
        "stage": diagnostics.get("stage"),
    }


def _run(label: str, assets: List[str], workers: int, disable: bool, model: str, nu_base: float) -> Dict[str, Any]:
    t0 = time.perf_counter()
    results: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_worker, (sym, disable, model, nu_base)): sym for sym in assets}
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
        "disable_precompute": disable,
        "workers": workers,
        "model": model,
        "nu_base": nu_base,
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
    parser.add_argument("--model", choices=["canonical", "improved"], default="canonical")
    parser.add_argument("--nu-base", type=float, default=8.0)
    parser.add_argument("--metrics-json", default="src/data/benchmarks/unified_stage5_precompute_metrics.json")
    args = parser.parse_args()

    assets = list(BENCHMARK_50)
    workers = args.workers if args.workers > 0 else _physical_workers()
    workers = max(1, min(workers, len(assets)))
    before = _run("before_stage5_precompute", assets, workers, True, args.model, args.nu_base)
    after = _run("after_stage5_precompute", assets, workers, False, args.model, args.nu_base)

    paired = []
    fields = ("q", "c", "phi", "nu_base", "variance_inflation", "mu_drift")
    for sym, after_result in after["results"].items():
        before_result = before["results"].get(sym)
        if not before_result:
            continue
        item = {"symbol": sym, "seconds_delta": before_result["seconds"] - after_result["seconds"]}
        for field in fields:
            item[f"{field}_delta"] = before_result[field] - after_result[field]
        paired.append(item)

    speedup = before["total_seconds"] / after["total_seconds"] if after["total_seconds"] > 0 else math.nan
    summary = {
        "workers": workers,
        "model": args.model,
        "nu_base": args.nu_base,
        "speedup_total": speedup,
        "before_total_seconds": before["total_seconds"],
        "after_total_seconds": after["total_seconds"],
        "paired_count": len(paired),
    }
    for field in fields:
        summary[f"max_abs_{field}_delta"] = max((abs(p[f"{field}_delta"]) for p in paired), default=None)
    payload = {"summary": summary, "before": before, "after": after, "paired": paired}
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
