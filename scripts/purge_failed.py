#!/usr/bin/env python3
"""
purge_failed.py

Purge cached price data for assets that have consistently failed to process.
This helps clean up potentially corrupted or problematic cache files.

Usage:
    python scripts/purge_failed.py          # Purge cache files for failed assets
    python scripts/purge_failed.py --list   # List failed assets without purging
    python scripts/purge_failed.py --clear  # Clear the failed assets list (keep cache)
    python scripts/purge_failed.py --all    # Purge cache AND clear the list
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.fx_data_utils import (
    load_failed_assets,
    get_failed_asset_symbols,
    purge_failed_assets_from_cache,
    clear_failed_assets_list,
    FAILED_ASSETS_FILE,
    PRICE_CACHE_DIR_PATH,
)


def main():
    parser = argparse.ArgumentParser(
        description="Purge cached price data for failed assets"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List failed assets without purging",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the failed assets list (but keep cache files)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Purge cache files AND clear the failed assets list",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    if args.list:
        # Just list failed assets
        failures = load_failed_assets()
        if not failures:
            print("No failed assets recorded.")
            return
        
        print(f"Failed Assets ({len(failures)} total):")
        print(f"{'='*60}")
        print(f"{'Asset':<12} {'Display Name':<30} {'Error':<40}")
        print(f"{'-'*12} {'-'*30} {'-'*40}")
        for asset, info in sorted(failures.items()):
            disp = info.get("display_name", asset)[:28]
            err = str(info.get("last_error", ""))[:38]
            print(f"{asset:<12} {disp:<30} {err:<40}")
        print(f"{'='*60}")
        print(f"\nFailed assets file: {FAILED_ASSETS_FILE}")
        print(f"Price cache dir: {PRICE_CACHE_DIR_PATH}")
        return

    if args.clear:
        # Just clear the list
        print("Clearing failed assets list...")
        clear_failed_assets_list(verbose=verbose)
        return

    # Purge cache files
    print("Purging cached data for failed assets...")
    print(f"{'='*60}")
    results = purge_failed_assets_from_cache(verbose=verbose)
    print(f"{'='*60}")

    if args.all:
        # Also clear the list
        print("\nClearing failed assets list...")
        clear_failed_assets_list(verbose=verbose)
        print("\nCache purged and failed assets list cleared.")
    else:
        print(f"\nNote: Failed assets list retained at: {FAILED_ASSETS_FILE}")
        print("Run with --all to also clear the list, or --clear to clear without purging.")


if __name__ == "__main__":
    main()
