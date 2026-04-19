"""
CLI runner for the indicators backtesting engine.
Runs all 500 strategies across the 120-asset universe,
produces ranked JSON results, and prints a summary.

Usage:
    python -m indicators.cli                  # full run
    python -m indicators.cli --top 10         # show top N
    python -m indicators.cli --family "Trend"  # filter family
    python -m indicators.cli --ids 1,2,500    # specific IDs
    python -m indicators.cli --quick          # 10 assets only
"""

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure src/ on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from indicators.base import get_indicators, load_ohlcv, RESULTS_DIR, clear_caches
from indicators.backtest import backtest_strategy
from indicators.scoring import aggregate_results, rank_strategies
from indicators.registry import get_all_strategies, get_strategy
from indicators.universe import UNIVERSE


QUICK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "JPM", "XOM",
    "SPY", "QQQ", "GLD", "BTC-USD", "MSTR",
]


def _run_strategy_on_asset(sid, fn, symbol):
    """Run a single strategy on a single asset. Returns result dict or None."""
    try:
        ind = get_indicators(symbol)
        if ind is None or len(ind["close"]) < 60:
            return None
        signal = fn(ind)
        if signal is None or signal.isna().all():
            return None
        result = backtest_strategy(signal, ind["close"])
        if result is not None:
            result["symbol"] = symbol
        return result
    except Exception:
        return None


def _run_strategy_batch(sid, name, family, fn, symbols):
    """Run one strategy across all symbols. Returns (sid, result_dict)."""
    per_asset = []
    for sym in symbols:
        r = _run_strategy_on_asset(sid, fn, sym)
        if r is not None:
            per_asset.append(r)

    agg = aggregate_results(per_asset) if per_asset else {}

    return sid, {
        "name": name,
        "family": family,
        "aggregate": agg,
        "per_asset": per_asset,
    }


def run_backtest(
    strategy_ids=None,
    family_filter=None,
    symbols=None,
    workers=1,
    verbose=True,
):
    """
    Run backtest for selected strategies on selected universe.

    Returns:
        dict of {sid: {"name", "family", "aggregate", "per_asset"}}
    """
    all_strats = get_all_strategies()

    # Filter
    if strategy_ids:
        ids = sorted(strategy_ids)
    elif family_filter:
        ids = sorted(
            sid for sid, info in all_strats.items()
            if family_filter.lower() in info["family"].lower()
        )
    else:
        ids = sorted(all_strats.keys())

    if symbols is None:
        symbols = UNIVERSE

    # Check which symbols have data
    valid_symbols = []
    for sym in symbols:
        try:
            df = load_ohlcv(sym)
            if df is not None and len(df) >= 60:
                valid_symbols.append(sym)
        except Exception:
            pass

    if verbose:
        print(f"Backtesting {len(ids)} strategies x {len(valid_symbols)} assets")
        print(f"Workers: {workers}")
        print()

    results = {}
    t0 = time.time()

    for i, sid in enumerate(ids, 1):
        info = all_strats.get(sid)
        if not info:
            continue

        name = info["name"]
        family = info["family"]
        fn = info["fn"]

        st = time.time()
        _, result = _run_strategy_batch(sid, name, family, fn, valid_symbols)
        elapsed = time.time() - st
        results[sid] = result

        if verbose:
            comp = result["aggregate"].get("composite", 0)
            n = len(result["per_asset"])
            sharpe = result["aggregate"].get("med_sharpe", "n/a")
            print(
                f"  [{i:3d}/{len(ids)}] S{sid:03d} {name:40s} | "
                f"comp={comp:5.1f}  sharpe={sharpe}  assets={n:3d}  {elapsed:.1f}s"
            )

        # Clear caches periodically to manage memory
        if i % 50 == 0:
            clear_caches()

    elapsed_total = time.time() - t0
    if verbose:
        print(f"\nCompleted in {elapsed_total:.0f}s")

    return results


def save_results(results, filepath=None):
    """Save results to JSON. Strips per-asset detail for smaller file."""
    if filepath is None:
        filepath = os.path.join(RESULTS_DIR, "backtest_results.json")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Full results
    with open(filepath, "w") as f:
        json.dump(results, f, indent=1, default=str)

    # Summary (no per-asset)
    summary_path = filepath.replace(".json", "_summary.json")
    summary = {}
    for sid, data in results.items():
        summary[sid] = {
            "name": data["name"],
            "family": data["family"],
            "aggregate": data["aggregate"],
            "n_assets": len(data.get("per_asset", [])),
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1, default=str)

    return filepath, summary_path


def print_leaderboard(results, top_n=20):
    """Print ranked leaderboard."""
    ranked = rank_strategies(results)
    if not ranked:
        print("No results to rank.")
        return ranked

    print(f"\n{'='*110}")
    print(f"{'Rank':>4}  {'ID':>4}  {'Strategy':<38} {'Family':<22} {'Comp':>5} {'Sharpe':>6} {'Sort':>6} {'CAGR':>6} {'MaxDD':>7} {'Hit%':>5}")
    print(f"{'-'*110}")

    for row in ranked[:top_n]:
        print(
            f"{row['rank']:4d}  "
            f"S{row['id']:03d}  "
            f"{row['name']:<38.38s} "
            f"{row['family']:<22.22s} "
            f"{row['composite']:5.1f} "
            f"{row['sharpe'] or 0:6.3f} "
            f"{row['sortino'] or 0:6.3f} "
            f"{row['cagr'] or 0:6.1f} "
            f"{row['max_dd'] or 0:7.1f} "
            f"{row['buy_hit'] or 0:5.1f}"
        )

    print(f"{'='*110}")
    print(f"Showing top {min(top_n, len(ranked))} of {len(ranked)} strategies\n")
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Indicators Backtesting Engine")
    parser.add_argument("--top", type=int, default=20, help="Show top N strategies")
    parser.add_argument("--family", type=str, default=None, help="Filter by family name")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated strategy IDs")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 10 assets only")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--no-save", action="store_true", help="Skip saving results")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    strategy_ids = None
    if args.ids:
        strategy_ids = [int(x.strip()) for x in args.ids.split(",")]

    symbols = QUICK_UNIVERSE if args.quick else UNIVERSE

    results = run_backtest(
        strategy_ids=strategy_ids,
        family_filter=args.family,
        symbols=symbols,
        workers=args.workers,
    )

    ranked = print_leaderboard(results, top_n=args.top)

    if not args.no_save:
        full_path, summary_path = save_results(results, filepath=args.output)
        print(f"Full results:    {full_path}")
        print(f"Summary:         {summary_path}")

    return results, ranked


if __name__ == "__main__":
    main()
