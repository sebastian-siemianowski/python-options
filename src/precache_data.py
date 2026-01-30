#!/usr/bin/env python3
"""
precache_data.py

Pre-download and cache price data for all assets in the default universe.
This speeds up subsequent screening and backtesting by avoiding repeated API calls.

Usage:
    python src/precache_data.py
    python src/precache_data.py --workers 2 --batch-size 16
    python src/precache_data.py --years 5  # fetch 5 years of history
    python src/precache_data.py --assets AAPL,MSFT,GOOGL  # specific assets only
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import (
    DEFAULT_ASSET_UNIVERSE,
    download_prices_bulk,
    PRICE_CACHE_DIR_PATH,
)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download and cache price data for assets"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of symbols to download per batch (default: 10)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="Number of years of history to fetch (default: 10)",
    )
    parser.add_argument(
        "--assets",
        type=str,
        default=None,
        help="Comma-separated list of assets to cache (default: full universe)",
    )
    parser.add_argument(
        "--fx-only",
        action="store_true",
        help="Only cache FX rates (currency pairs)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Determine asset list
    if args.assets:
        assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()]
    else:
        assets = list(DEFAULT_ASSET_UNIVERSE)

    # Filter FX only if requested
    if args.fx_only:
        assets = [a for a in assets if "=" in a or a.endswith("-USD")]

    if not assets:
        print("No assets to cache.")
        return

    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=args.years * 365)
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    if not args.quiet:
        print(f"=" * 60)
        print(f"Price Data Pre-Caching")
        print(f"=" * 60)
        print(f"Cache directory: {PRICE_CACHE_DIR_PATH}")
        print(f"Assets to cache: {len(assets)}")
        print(f"Date range: {start_str} to {end_str} ({args.years} years)")
        print(f"Batch size: {args.batch_size}")
        print(f"Workers: {args.workers}")
        print(f"=" * 60)
        print()

    # Add common FX pairs that are often needed for currency conversion
    fx_pairs = [
        "USDPLN=X",
        "EURPLN=X",
        "GBPPLN=X",
        "PLNJPY=X",
        "EURUSD=X",
        "USDJPY=X",
        "GBPUSD=X",
        "USDCHF=X",
        "AUDUSD=X",
        "USDCAD=X",
        "CHFPLN=X",
        "CADPLN=X",
        "AUDPLN=X",
        "SEKPLN=X",
    ]

    # Combine assets with FX pairs (deduplicated)
    all_symbols = list(dict.fromkeys(assets + fx_pairs))

    if not args.quiet:
        print(f"Total symbols (including FX): {len(all_symbols)}")
        print()

    # Download in bulk
    log_fn = None if args.quiet else print
    try:
        results = download_prices_bulk(
            symbols=all_symbols,
            start=start_str,
            end=end_str,
            chunk_size=args.batch_size,
            progress=not args.quiet,
            log_fn=log_fn,
        )

        # Summary
        successful = sum(1 for s in results.values() if s is not None and len(s) > 0)
        failed = len(all_symbols) - successful

        if not args.quiet:
            print()
            print(f"=" * 60)
            print(f"Summary")
            print(f"=" * 60)
            print(f"Successfully cached: {successful}/{len(all_symbols)}")
            if failed > 0:
                print(f"Failed: {failed}")
                failed_syms = [
                    sym
                    for sym in all_symbols
                    if sym not in results or results.get(sym) is None or len(results.get(sym, [])) == 0
                ]
                if failed_syms:
                    print(f"Failed symbols: {', '.join(failed_syms[:20])}")
                    if len(failed_syms) > 20:
                        print(f"  ... and {len(failed_syms) - 20} more")
            print(f"Cache location: {PRICE_CACHE_DIR_PATH}")
            print(f"=" * 60)

    except Exception as e:
        print(f"Error during bulk download: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
