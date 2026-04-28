#!/usr/bin/env python3
"""Focused A/B benchmark for Gaussian unified Stage 1 CV fitting."""

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
from models.gaussian import GaussianDriftModel, GaussianUnifiedConfig  # noqa: E402
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


def _worker(args: tuple[str, bool, bool]) -> Dict[str, Any]:
    symbol, phi_mode, disable_train_state = args
    if disable_train_state:
        os.environ["PHI_GAUSSIAN_DISABLE_TRAIN_STATE_KERNEL"] = "1"
        os.environ.pop("PHI_GAUSSIAN_ENABLE_TRAIN_STATE_KERNEL", None)
    else:
        os.environ.pop("PHI_GAUSSIAN_DISABLE_TRAIN_STATE_KERNEL", None)
        os.environ["PHI_GAUSSIAN_ENABLE_TRAIN_STATE_KERNEL"] = "1"
    returns, vol = _prepare_asset(symbol)
    n_train = int(len(returns) * 0.7)
    returns_train = returns[:n_train]
    vol_train = vol[:n_train]
    config = GaussianUnifiedConfig.auto_configure(returns_train, vol_train)
    t0 = time.perf_counter()
    result = GaussianDriftModel._gaussian_stage_1(
        returns_train,
        vol_train,
        len(returns_train),
        config,
        phi_mode=phi_mode,
    )
    elapsed = time.perf_counter() - t0
    return {
        "symbol": symbol,
        "seconds": elapsed,
        "q": float(result["q"]),
        "c": float(result["c"]),
        "phi": float(result["phi"]),
        "log_q": float(result["log_q"]),
        "success": bool(result.get("success", False)),
    }


def _run(label: str, assets: List[str], workers: int, phi_mode: bool, disable: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    results: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_worker, (sym, phi_mode, disable)): sym for sym in assets}
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
        "phi_mode": phi_mode,
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
    parser.add_argument("--phi-mode", action="store_true")
    parser.add_argument("--metrics-json", default="src/data/benchmarks/gaussian_stage1_metrics.json")
    args = parser.parse_args()

    assets = list(BENCHMARK_50)
    workers = args.workers if args.workers > 0 else _physical_workers()
    workers = max(1, min(workers, len(assets)))
    before = _run("before_train_state_kernel", assets, workers, args.phi_mode, True)
    after = _run("after_train_state_kernel", assets, workers, args.phi_mode, False)

    paired = []
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
            "log_q_delta": before_result["log_q"] - after_result["log_q"],
        })

    speedup = before["total_seconds"] / after["total_seconds"] if after["total_seconds"] > 0 else math.nan
    summary = {
        "workers": workers,
        "phi_mode": args.phi_mode,
        "speedup_total": speedup,
        "before_total_seconds": before["total_seconds"],
        "after_total_seconds": after["total_seconds"],
        "paired_count": len(paired),
        "max_abs_q_delta": max((abs(p["q_delta"]) for p in paired), default=None),
        "max_abs_c_delta": max((abs(p["c_delta"]) for p in paired), default=None),
        "max_abs_phi_delta": max((abs(p["phi_delta"]) for p in paired), default=None),
        "max_abs_log_q_delta": max((abs(p["log_q_delta"]) for p in paired), default=None),
    }
    payload = {"summary": summary, "before": before, "after": after, "paired": paired}
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
