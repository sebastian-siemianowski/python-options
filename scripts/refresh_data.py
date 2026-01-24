#!/usr/bin/env python3
"""
refresh_data.py

Refresh price data by:
1. Deleting the last N days from all cached price files
2. Bulk downloading ALL symbols N times (always runs all passes)

This ensures fresh data. The download is unreliable so we always run multiple passes.

Usage:
    python scripts/refresh_data.py
    python scripts/refresh_data.py --days 5 --retries 5
    python scripts/refresh_data.py --days 3 --retries 3 --workers 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from typing import List

import pandas as pd

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fx_data_utils import (
    DEFAULT_ASSET_UNIVERSE,
    download_prices_bulk,
    PRICE_CACHE_DIR_PATH,
)


def trim_last_n_days(days: int = 5, quiet: bool = False) -> int:
    """Remove the last N days of data from all cached CSV files."""
    if not PRICE_CACHE_DIR_PATH.exists():
        if not quiet:
            print(f"Cache directory does not exist: {PRICE_CACHE_DIR_PATH}")
        return 0
    
    cutoff_date = datetime.now().date() - timedelta(days=days)
    files_modified = 0
    
    csv_files = list(PRICE_CACHE_DIR_PATH.glob("*.csv"))
    if not quiet:
        print(f"Trimming last {days} days (before {cutoff_date}) from {len(csv_files)} cache files...")
    
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            if df.empty:
                continue
            
            original_len = len(df)
            df_trimmed = df[df.index.date < cutoff_date]
            
            if len(df_trimmed) < original_len:
                df_trimmed.to_csv(csv_path)
                files_modified += 1
                if not quiet:
                    removed = original_len - len(df_trimmed)
                    print(f"  {csv_path.stem}: removed {removed} rows")
                    
        except Exception as e:
            if not quiet:
                print(f"  Warning: Could not process {csv_path.name}: {e}")
    
    if not quiet:
        print(f"Modified {files_modified} files")
    
    return files_modified


def get_all_symbols() -> List[str]:
    """Get all symbols that should be downloaded (universe + FX pairs)."""
    all_symbols = list(DEFAULT_ASSET_UNIVERSE)
    
    fx_pairs = [
        "USDPLN=X", "EURPLN=X", "GBPPLN=X", "PLNJPY=X",
        "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X",
        "AUDUSD=X", "USDCAD=X", "CHFPLN=X", "CADPLN=X",
        "AUDPLN=X", "SEKPLN=X",
    ]
    
    return list(dict.fromkeys(all_symbols + fx_pairs))


def bulk_download_n_times(
    symbols: List[str],
    num_passes: int = 5,
    batch_size: int = 16,
    workers: int = 2,
    years: int = 10,
    quiet: bool = False,
) -> int:
    """
    Download data for symbols, ALWAYS running num_passes times with BULK ONLY.
    Individual fallback only happens on the final pass for any remaining symbols.
    Each pass downloads ALL symbols to ensure reliability.
    """
    if not symbols:
        if not quiet:
            print("No symbols to download.")
        return 0
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=years * 365)
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()
    
    all_symbols = list(symbols)
    last_failed_count = len(all_symbols)
    last_failed_symbols = all_symbols.copy()
    
    # Run num_passes bulk-only passes (no individual fallback)
    for pass_num in range(1, num_passes + 1):
        is_final_pass = (pass_num == num_passes)
        
        if not quiet:
            print(f"\n{'='*60}")
            print(f"Bulk download pass {pass_num}/{num_passes}" + (" (final - with individual fallback)" if is_final_pass else " (bulk only)"))
            print(f"Downloading {len(all_symbols)} symbols...")
            print(f"{'='*60}")
        
        log_fn = None if quiet else print
        
        try:
            # Skip individual fallback on all passes except the last one
            results = download_prices_bulk(
                symbols=all_symbols,
                start=start_str,
                end=end_str,
                chunk_size=batch_size,
                progress=not quiet,
                log_fn=log_fn,
                skip_individual_fallback=not is_final_pass,  # Only allow individual fallback on final pass
            )
            
            failed_symbols = []
            for sym in all_symbols:
                if sym not in results or results.get(sym) is None or len(results.get(sym, [])) == 0:
                    failed_symbols.append(sym)
            
            successful = len(all_symbols) - len(failed_symbols)
            last_failed_count = len(failed_symbols)
            last_failed_symbols = failed_symbols
            
            if not quiet:
                print(f"\nPass {pass_num} results: {successful}/{len(all_symbols)} successful, {last_failed_count} failed")
            
            # Wait 5 seconds between passes to avoid rate limiting
            if pass_num < num_passes:
                if not quiet:
                    print(f"Waiting 5 seconds before next pass...")
                time.sleep(5)
                
        except Exception as e:
            if not quiet:
                print(f"Error during bulk download pass {pass_num}: {e}")
            # Still wait before next pass even on error
            if pass_num < num_passes:
                if not quiet:
                    print(f"Waiting 5 seconds before next pass...")
                time.sleep(5)
    
    if not quiet:
        print(f"\n{'='*60}")
        print(f"Completed {num_passes} download passes")
        print(f"{'='*60}")
        if last_failed_count == 0:
            print(f"All {len(all_symbols)} symbols have data!")
        else:
            print(f"{last_failed_count} symbols may still have issues")
            if last_failed_symbols:
                print(f"Potentially incomplete: {', '.join(last_failed_symbols[:20])}")
                if len(last_failed_symbols) > 20:
                    print(f"  ... and {len(last_failed_symbols) - 20} more")
    
    return last_failed_count


def main():
    parser = argparse.ArgumentParser(
        description="Refresh price data by trimming recent days and re-downloading"
    )
    parser.add_argument("--days", type=int, default=5, help="Days to trim from cache (default: 5)")
    parser.add_argument("--retries", type=int, default=5, help="Download passes - ALWAYS runs this many (default: 5)")
    parser.add_argument("--workers", type=int, default=2, help="Parallel download workers (default: 2)")
    parser.add_argument("--batch-size", type=int, default=16, help="Symbols per batch (default: 16)")
    parser.add_argument("--years", type=int, default=10, help="Years of history (default: 10)")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--skip-trim", action="store_true", help="Skip trimming, only download")

    args = parser.parse_args()

    if not args.quiet:
        print(f"{'='*60}")
        print(f"Price Data Refresh")
        print(f"{'='*60}")
        print(f"Cache directory: {PRICE_CACHE_DIR_PATH}")
        print(f"Days to trim: {args.days}")
        print(f"Download passes: {args.retries} (always runs all)")
        print(f"Batch size: {args.batch_size}")
        print(f"Workers: {args.workers}")
        print(f"{'='*60}")

    # Step 1: Trim last N days from cache
    if not args.skip_trim:
        if not args.quiet:
            print("\nStep 1: Trimming recent data from cache...")
        trim_last_n_days(days=args.days, quiet=args.quiet)
    else:
        if not args.quiet:
            print("\nStep 1: Skipping trim (--skip-trim)")

    # Step 2: Get all symbols
    all_symbols = get_all_symbols()
    if not args.quiet:
        print(f"\nStep 2: Found {len(all_symbols)} symbols to download")

    # Step 3: Bulk download N times (always runs all passes)
    if not args.quiet:
        print(f"\nStep 3: Running {args.retries} bulk download passes...")
    
    failed_count = bulk_download_n_times(
        symbols=all_symbols,
        num_passes=args.retries,
        batch_size=args.batch_size,
        workers=args.workers,
        years=args.years,
        quiet=args.quiet,
    )

    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"Refresh Complete - ran {args.retries} passes for {len(all_symbols)} symbols")
        print(f"{'='*60}")

    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
