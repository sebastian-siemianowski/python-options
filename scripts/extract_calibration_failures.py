#!/usr/bin/env python3
"""
extract_calibration_failures.py

Extract assets with calibration failures (PIT p-value < 0.05) from the
calibration_failures.json file for targeted re-tuning.

Usage:
    python scripts/extract_calibration_failures.py
    python scripts/extract_calibration_failures.py --severity critical
    python scripts/extract_calibration_failures.py --min-pit 0.01
    
Output:
    Comma-separated list of asset symbols suitable for --assets argument
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

# Default path to calibration failures file
DEFAULT_CALIBRATION_FILE = os.path.join(
    os.path.dirname(__file__),
    'quant', 'cache', 'calibration', 'calibration_failures.json'
)


def load_calibration_failures(filepath: str) -> dict:
    """Load calibration failures from JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: Calibration file not found: {filepath}", file=sys.stderr)
        print("Run 'make tune' first to generate calibration data.", file=sys.stderr)
        sys.exit(1)
    
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_failed_assets(
    data: dict,
    severity: Optional[str] = None,
    min_pit: Optional[float] = None,
    max_pit: float = 0.05,
    exclude_high_kurtosis: bool = False,
    include_failed: bool = True,
) -> List[str]:
    """
    Extract asset symbols with calibration issues.
    
    Args:
        data: Calibration failures JSON data
        severity: Filter by severity ('critical', 'warning', or None for all)
        min_pit: Minimum PIT p-value (exclude assets with p-value below this)
        max_pit: Maximum PIT p-value threshold (default 0.05)
        exclude_high_kurtosis: Exclude assets flagged only for high kurtosis
        include_failed: Include completely failed assets
        
    Returns:
        List of asset symbols
    """
    assets = []
    
    for issue in data.get('issues', []):
        asset = issue.get('asset')
        if not asset:
            continue
        
        issue_type = issue.get('issue_type', '')
        issue_severity = issue.get('severity', '')
        pit_p = issue.get('pit_ks_pvalue')
        
        # Filter by severity
        if severity and issue_severity != severity:
            continue
        
        # Skip completely failed assets unless requested
        if 'FAILED' in issue_type and not include_failed:
            continue
        
        # Check PIT p-value filters
        if pit_p is not None:
            if min_pit is not None and pit_p < min_pit:
                continue
            if pit_p >= max_pit:
                continue
        
        # Skip if only high kurtosis (no PIT issue)
        if exclude_high_kurtosis:
            if 'High Kurt' in issue_type and 'PIT' not in issue_type:
                continue
        
        assets.append(asset)
    
    return assets


def main():
    parser = argparse.ArgumentParser(
        description="Extract assets with calibration failures for re-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # All calibration failures
  %(prog)s --severity critical          # Only critical failures
  %(prog)s --min-pit 0.001              # Exclude extreme failures (p < 0.001)
  %(prog)s --pit-only                   # Only PIT failures (exclude kurtosis-only)
  %(prog)s --count                      # Just show count, don't list assets
  %(prog)s --summary                    # Show summary statistics
        """
    )
    parser.add_argument('--file', type=str, default=DEFAULT_CALIBRATION_FILE,
                       help='Path to calibration_failures.json')
    parser.add_argument('--severity', type=str, choices=['critical', 'warning'],
                       help='Filter by severity level')
    parser.add_argument('--min-pit', type=float, default=None,
                       help='Minimum PIT p-value (exclude extreme failures)')
    parser.add_argument('--max-pit', type=float, default=0.05,
                       help='Maximum PIT p-value threshold (default: 0.05)')
    parser.add_argument('--pit-only', action='store_true',
                       help='Only include PIT failures (exclude kurtosis-only issues)')
    parser.add_argument('--include-failed', action='store_true',
                       help='Include completely failed assets')
    parser.add_argument('--count', action='store_true',
                       help='Only show count of assets')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary statistics')
    parser.add_argument('--newline', action='store_true',
                       help='Output one asset per line instead of comma-separated')

    args = parser.parse_args()

    # Load calibration data
    data = load_calibration_failures(args.file)

    # Show summary if requested
    if args.summary:
        summary = data.get('summary', {})
        print(f"Calibration Summary:")
        print(f"  Total assets:     {summary.get('total_assets', 'N/A')}")
        print(f"  Total issues:     {summary.get('total_issues', 'N/A')}")
        print(f"  Critical:         {summary.get('critical', 'N/A')}")
        print(f"  Warnings:         {summary.get('warnings', 'N/A')}")
        print(f"  Failed:           {summary.get('failed', 'N/A')}")
        print(f"  Passed:           {summary.get('passed', 'N/A')}")
        print()

    # Extract failed assets
    assets = extract_failed_assets(
        data,
        severity=args.severity,
        min_pit=args.min_pit,
        max_pit=args.max_pit,
        exclude_high_kurtosis=args.pit_only,
        include_failed=args.include_failed,
    )

    # Remove duplicates while preserving order
    seen = set()
    unique_assets = []
    for a in assets:
        if a not in seen:
            seen.add(a)
            unique_assets.append(a)
    assets = unique_assets

    if args.count:
        print(len(assets))
    elif args.summary:
        print(f"Assets matching filters: {len(assets)}")
        if assets:
            print(f"First 10: {', '.join(assets[:10])}")
    elif args.newline:
        for asset in assets:
            print(asset)
    else:
        # Comma-separated output for --assets argument
        print(','.join(assets))


if __name__ == '__main__':
    main()
