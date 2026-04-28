#!/usr/bin/env python3
"""Benchmark the retune stack on a fixed 50-stock real-data universe.

Runs the same per-asset tuning worker used by ``make tune`` plus the same
Pass-2 signal calibration step.  The harness writes compact JSON metrics so
speed and calibration quality can be compared before/after code changes.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("TUNING_QUIET", "1")
os.environ.setdefault("OFFLINE_MODE", "1")

from decision.signals_calibration import run_signals_calibration  # noqa: E402
from tuning.tune_modules.asset_tuning import _tune_worker  # noqa: E402
from tuning.tune_modules.cli import _extract_previous_posteriors  # noqa: E402
from tuning.tune_modules.process_noise import sort_assets_by_complexity  # noqa: E402
from tuning.tune_modules.volatility_fitting import load_cache, save_cache_json  # noqa: E402


BENCHMARK_50 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "CRM", "ADBE", "CRWD", "NET",
    "JPM", "BAC", "GS", "MS", "SCHW", "AFRM",
    "LMT", "RTX", "NOC", "GD",
    "JNJ", "UNH", "PFE", "ABBV", "MRNA",
    "CAT", "DE", "BA", "UPS", "GE",
    "XOM", "CVX", "COP", "SLB", "OXY",
    "LIN", "FCX", "NEM", "NUE",
    "AMZN", "TSLA", "HD", "NKE", "SBUX", "PG", "KO", "PEP", "COST",
    "META", "NFLX", "DIS", "SNAP",
]


def _physical_cpu_default() -> int:
    count = os.cpu_count() or 2
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip()
            count = int(out)
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            packages = set()
            physical_id = None
            core_id = None
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        if physical_id is not None and core_id is not None:
                            packages.add((physical_id, core_id))
                        physical_id = None
                        core_id = None
                        continue
                    if line.startswith("physical id"):
                        physical_id = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":", 1)[1].strip()
            if packages:
                count = len(packages)
        except Exception:
            pass
    return max(1, count - 1)


def _quiet_tune_worker(
    args_tuple: Tuple[str, str, Optional[str], float, float, float, Optional[Dict[int, Dict[str, float]]]]
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str], Optional[str], float]:
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        asset, result, error, traceback_str = _tune_worker(args_tuple)
    return asset, result, error, traceback_str, time.perf_counter() - t0


def _finite(values: List[float]) -> List[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]


def _summarize_cache(cache: Dict[str, Dict[str, Any]], assets: List[str]) -> Dict[str, Any]:
    globals_: List[Dict[str, Any]] = []
    for sym in assets:
        entry = cache.get(sym)
        if isinstance(entry, dict):
            g = entry.get("global", entry)
            if isinstance(g, dict):
                globals_.append(g)

    pit = _finite([g.get("pit_ks_pvalue") for g in globals_])
    crps = _finite([g.get("crps") for g in globals_])
    bic = _finite([g.get("bic") for g in globals_])
    hyv = _finite([g.get("hyvarinen_score") for g in globals_])
    warnings = sum(1 for g in globals_ if g.get("calibration_warning"))

    best_models: Dict[str, int] = {}
    model_counts: List[int] = []
    for g in globals_:
        best = str(g.get("best_model") or g.get("noise_model") or "unknown")
        best_models[best] = best_models.get(best, 0) + 1
        mp = g.get("model_posterior")
        if isinstance(mp, dict):
            model_counts.append(len(mp))

    return {
        "assets_with_results": len(globals_),
        "calibration_warnings": warnings,
        "pit_mean": statistics.fmean(pit) if pit else None,
        "pit_min": min(pit) if pit else None,
        "crps_mean": statistics.fmean(crps) if crps else None,
        "bic_mean": statistics.fmean(bic) if bic else None,
        "hyvarinen_mean": statistics.fmean(hyv) if hyv else None,
        "models_per_asset_mean": statistics.fmean(model_counts) if model_counts else None,
        "best_model_counts": dict(sorted(best_models.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def _summarize_signal_calibration(cache: Dict[str, Dict[str, Any]], assets: List[str]) -> Dict[str, Any]:
    brier_delta: List[float] = []
    crps_delta: List[float] = []
    pit_cal: List[float] = []
    calibrated_assets = 0
    for sym in assets:
        cal = cache.get(sym, {}).get("signals_calibration", {})
        if not isinstance(cal, dict):
            continue
        horizons = cal.get("horizons", {})
        if not isinstance(horizons, dict) or not horizons:
            continue
        calibrated_assets += 1
        for h in horizons.values():
            if not isinstance(h, dict):
                continue
            br = h.get("brier_raw")
            bc = h.get("brier_calibrated")
            if isinstance(br, (int, float)) and isinstance(bc, (int, float)):
                brier_delta.append(float(br) - float(bc))
            cr = h.get("crps_raw")
            cc = h.get("crps_calibrated")
            if isinstance(cr, (int, float)) and isinstance(cc, (int, float)):
                crps_delta.append(float(cr) - float(cc))
            pp = h.get("pit_p_cal")
            if isinstance(pp, (int, float)) and math.isfinite(float(pp)):
                pit_cal.append(float(pp))

    return {
        "signals_calibrated_assets": calibrated_assets,
        "signals_brier_improvement_mean": statistics.fmean(brier_delta) if brier_delta else None,
        "signals_crps_improvement_mean": statistics.fmean(crps_delta) if crps_delta else None,
        "signals_pit_cal_mean": statistics.fmean(pit_cal) if pit_cal else None,
        "signals_pit_cal_min": min(pit_cal) if pit_cal else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retune on 50 stocks.")
    parser.add_argument("--cache-json", default="src/data/benchmarks/retune_50_cache")
    parser.add_argument("--metrics-json", default="src/data/benchmarks/retune_50_metrics.json")
    parser.add_argument("--label", default="benchmark")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--prior-mean", type=float, default=-6.0)
    parser.add_argument("--prior-lambda", type=float, default=1.0)
    parser.add_argument("--lambda-regime", type=float, default=0.05)
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args()

    assets = sort_assets_by_complexity(list(BENCHMARK_50))
    workers = args.workers if args.workers > 0 else _physical_cpu_default()
    workers = max(1, min(workers, len(assets)))

    Path(args.cache_json).mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)

    cache = load_cache(args.cache_json)
    worker_args = [
        (
            sym,
            args.start,
            args.end,
            args.prior_mean,
            args.prior_lambda,
            args.lambda_regime,
            _extract_previous_posteriors(cache.get(sym)),
        )
        for sym in assets
    ]

    t0 = time.perf_counter()
    failures: Dict[str, str] = {}
    durations: Dict[str, float] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_quiet_tune_worker, arg): arg[0] for arg in worker_args}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                asset, result, error, _traceback_str, elapsed = fut.result()
                durations[asset] = elapsed
                if result:
                    cache[asset] = result
                else:
                    failures[asset] = error or "tuning returned no result"
            except Exception as exc:
                failures[sym] = str(exc)

    tune_seconds = time.perf_counter() - t0
    save_cache_json(cache, args.cache_json)

    calibration_seconds = 0.0
    if not args.skip_calibration:
        c0 = time.perf_counter()
        cache = run_signals_calibration(cache, workers=workers, assets=assets, quiet=True)
        calibration_seconds = time.perf_counter() - c0
        save_cache_json(cache, args.cache_json)

    metrics = {
        "label": args.label,
        "assets": assets,
        "asset_count": len(assets),
        "workers": workers,
        "tune_seconds": tune_seconds,
        "calibration_seconds": calibration_seconds,
        "total_seconds": tune_seconds + calibration_seconds,
        "failed_assets": failures,
        "failed_count": len(failures),
        "duration_mean": statistics.fmean(durations.values()) if durations else None,
        "duration_p50": statistics.median(durations.values()) if durations else None,
        "duration_max": max(durations.values()) if durations else None,
        "cache_summary": _summarize_cache(cache, assets),
        "signals_calibration_summary": _summarize_signal_calibration(cache, assets),
    }

    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
