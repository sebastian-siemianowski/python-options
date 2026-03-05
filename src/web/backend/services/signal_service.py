"""
Signal service — reads cached signal data and high conviction signals.
"""

import json
import os
import glob
from typing import Any, Dict, List, Optional

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
DATA_DIR = os.path.join(SRC_DIR, "data")

DEFAULT_CACHE_PATH = os.path.join(DATA_DIR, "currencies", "fx_plnjpy.json")
HIGH_CONVICTION_DIR = os.path.join(DATA_DIR, "high_conviction")


def get_cached_signals(cache_path: str = DEFAULT_CACHE_PATH) -> Optional[Dict[str, Any]]:
    """Load signals from the JSON cache written by signals.py main()."""
    if not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def get_summary_rows(cache_path: str = DEFAULT_CACHE_PATH) -> List[Dict[str, Any]]:
    """Return the summary_rows array from the signal cache."""
    data = get_cached_signals(cache_path)
    if data is None:
        return []
    return data.get("summary_rows", [])


def get_asset_blocks(cache_path: str = DEFAULT_CACHE_PATH) -> List[Dict[str, Any]]:
    """Return the full asset blocks with all signal fields."""
    data = get_cached_signals(cache_path)
    if data is None:
        return []
    return data.get("assets", [])


def get_failed_assets(cache_path: str = DEFAULT_CACHE_PATH) -> List[str]:
    """Return list of assets that failed during signal computation."""
    data = get_cached_signals(cache_path)
    if data is None:
        return []
    return data.get("failed_assets", [])


def get_horizons(cache_path: str = DEFAULT_CACHE_PATH) -> List[int]:
    """Return the horizons used in the signal computation."""
    data = get_cached_signals(cache_path)
    if data is None:
        return []
    return data.get("horizons", [])


def get_high_conviction_signals(signal_type: str = "buy") -> List[Dict[str, Any]]:
    """
    Load high conviction signals from the buy/ or sell/ directory.
    
    Args:
        signal_type: 'buy' or 'sell'
    
    Returns:
        List of signal dictionaries
    """
    dir_path = os.path.join(HIGH_CONVICTION_DIR, signal_type)
    if not os.path.isdir(dir_path):
        return []

    signals = []
    for filepath in sorted(glob.glob(os.path.join(dir_path, "*.json"))):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    signals.extend(data)
                else:
                    signals.append(data)
        except (json.JSONDecodeError, IOError):
            continue
    return signals


def get_cache_age_seconds(cache_path: str = DEFAULT_CACHE_PATH) -> Optional[float]:
    """Return the age of the signal cache in seconds, or None if missing."""
    if not os.path.isfile(cache_path):
        return None
    import time
    return time.time() - os.path.getmtime(cache_path)


def get_signal_stats(cache_path: str = DEFAULT_CACHE_PATH) -> Dict[str, Any]:
    """Return summary statistics about the current signal cache."""
    data = get_cached_signals(cache_path)
    if data is None:
        return {"cached": False, "total_assets": 0, "failed": 0}

    summary_rows = data.get("summary_rows", [])
    failed = data.get("failed_assets", [])
    buy_count = 0
    sell_count = 0
    hold_count = 0

    for row in summary_rows:
        for _hz, sig in row.get("horizon_signals", {}).items():
            label = (sig.get("label") or "HOLD").upper()
            if label == "BUY":
                buy_count += 1
            elif label == "SELL":
                sell_count += 1
            else:
                hold_count += 1

    age = get_cache_age_seconds(cache_path)
    return {
        "cached": True,
        "total_assets": len(summary_rows),
        "failed": len(failed),
        "buy_signals": buy_count,
        "sell_signals": sell_count,
        "hold_signals": hold_count,
        "cache_age_seconds": round(age, 1) if age is not None else None,
    }
