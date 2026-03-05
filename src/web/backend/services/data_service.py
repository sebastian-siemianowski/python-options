"""
Data service — manages price data, cache status, and data refresh operations.
"""

import os
import glob
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
DATA_DIR = os.path.join(SRC_DIR, "data")
PRICES_DIR = os.path.join(DATA_DIR, "prices")


def list_price_files() -> List[Dict[str, Any]]:
    """
    List all price data files with metadata.
    
    Returns list of dicts with symbol, filename, last_modified, size_kb, rows.
    """
    if not os.path.isdir(PRICES_DIR):
        return []

    results = []
    for filepath in sorted(glob.glob(os.path.join(PRICES_DIR, "*.csv"))):
        fname = os.path.basename(filepath)
        # Extract symbol: AAPL_1d.csv -> AAPL
        symbol = fname.replace("_1d.csv", "").replace(".csv", "")
        stat = os.stat(filepath)
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()

        # Count rows (fast line count)
        try:
            with open(filepath, "r") as f:
                row_count = sum(1 for _ in f) - 1  # subtract header
        except IOError:
            row_count = 0

        results.append({
            "symbol": symbol,
            "filename": fname,
            "last_modified": mtime,
            "age_hours": round((time.time() - stat.st_mtime) / 3600, 1),
            "size_kb": round(stat.st_size / 1024, 1),
            "rows": max(0, row_count),
        })

    return results


def get_data_summary() -> Dict[str, Any]:
    """High-level summary of data status."""
    files = list_price_files()
    if not files:
        return {"total_files": 0, "stale_files": 0, "freshest": None, "oldest": None}

    ages = [f["age_hours"] for f in files]
    stale = sum(1 for a in ages if a > 24)

    return {
        "total_files": len(files),
        "stale_files": stale,
        "fresh_files": len(files) - stale,
        "freshest_hours": round(min(ages), 1) if ages else None,
        "oldest_hours": round(max(ages), 1) if ages else None,
        "total_size_mb": round(sum(f["size_kb"] for f in files) / 1024, 1),
    }


def get_price_data(symbol: str, tail: int = 500) -> Optional[List[Dict[str, Any]]]:
    """
    Load OHLCV price data for a symbol as a list of dicts.
    
    Args:
        symbol: Asset symbol (e.g., 'AAPL')
        tail: Number of most recent rows to return
        
    Returns:
        List of {date, open, high, low, close, volume} dicts
    """
    # Try common filename patterns
    for pattern in [f"{symbol}_1d.csv", f"{symbol.upper()}_1d.csv",
                    f"{symbol}.csv", f"{symbol.upper()}.csv",
                    f"{symbol.replace('-', '_')}_1d.csv",
                    f"{symbol.replace('-', '_')}.csv"]:
        filepath = os.path.join(PRICES_DIR, pattern)
        if os.path.isfile(filepath):
            break
    else:
        return None

    try:
        import pandas as pd
        df = pd.read_csv(filepath, parse_dates=["Date"])
        df = df.tail(tail)
        
        records = []
        for _, row in df.iterrows():
            records.append({
                "date": row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"]),
                "open": round(float(row.get("Open", 0)), 4),
                "high": round(float(row.get("High", 0)), 4),
                "low": round(float(row.get("Low", 0)), 4),
                "close": round(float(row.get("Close", 0)), 4),
                "volume": int(row.get("Volume", 0)) if not _is_nan(row.get("Volume")) else 0,
            })
        return records
    except Exception:
        return None


def _is_nan(val) -> bool:
    """Check if a value is NaN."""
    try:
        import math
        return math.isnan(float(val))
    except (ValueError, TypeError):
        return False


def get_directories_summary() -> Dict[str, Dict[str, Any]]:
    """Summarize key data directories."""
    dirs = {
        "prices": PRICES_DIR,
        "tune": os.path.join(DATA_DIR, "tune"),
        "high_conviction_buy": os.path.join(DATA_DIR, "high_conviction", "buy"),
        "high_conviction_sell": os.path.join(DATA_DIR, "high_conviction", "sell"),
        "calibration": os.path.join(DATA_DIR, "calibration"),
        "currencies": os.path.join(DATA_DIR, "currencies"),
    }

    result = {}
    for name, path in dirs.items():
        if os.path.isdir(path):
            files = os.listdir(path)
            result[name] = {
                "path": path,
                "file_count": len([f for f in files if os.path.isfile(os.path.join(path, f))]),
                "exists": True,
            }
        else:
            result[name] = {"path": path, "file_count": 0, "exists": False}

    return result
