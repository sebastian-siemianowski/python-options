#!/usr/bin/env python3
"""
clean_cache.py

Clean cached price data by removing empty rows (dates before company existed).

Usage:
    python src/ingestion/clean_cache.py          # Clean all cached price files
    python src/ingestion/clean_cache.py --quiet  # Clean without verbose output
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.data_utils import clean_price_cache, PRICE_CACHE_DIR_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Clean cached price data by removing empty rows"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    if verbose:
        print(f"=" * 60)
        print(f"Price Cache Cleanup")
        print(f"=" * 60)
        print(f"Cache directory: {PRICE_CACHE_DIR_PATH}")
        print()

    results = clean_price_cache(verbose=verbose)

    if verbose:
        print()
        print(f"=" * 60)
        print(f"Cleanup complete!")
        print(f"=" * 60)


if __name__ == "__main__":
    main()
