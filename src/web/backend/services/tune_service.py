"""
Tune service — reads per-asset tuning cache and provides summaries.
"""

import json
import os
import glob
from typing import Any, Dict, List, Optional

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
TUNE_DIR = os.path.join(SRC_DIR, "data", "tune")


def list_tuned_assets() -> List[Dict[str, Any]]:
    """
    List all tuned assets with summary information.
    
    Returns list of dicts with symbol, best_model, pit_grade, last_tuned, etc.
    """
    if not os.path.isdir(TUNE_DIR):
        return []

    results = []
    for filepath in sorted(glob.glob(os.path.join(TUNE_DIR, "*.json"))):
        symbol = os.path.splitext(os.path.basename(filepath))[0]
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            results.append({"symbol": symbol, "error": "corrupt cache file"})
            continue

        g = data.get("global", {})
        meta = data.get("meta", {})

        results.append({
            "symbol": symbol,
            "best_model": g.get("best_model", "unknown"),
            "pit_calibration_grade": g.get("pit_calibration_grade", "N/A"),
            "ad_stat": g.get("ad_stat"),
            "ad_critical": g.get("ad_critical_5pct"),
            "ad_pass": g.get("ad_pass"),
            "num_models": len(g.get("model_weights", {})),
            "cache_version": meta.get("cache_version", "unknown"),
            "last_tuned": meta.get("timestamp", "unknown"),
            "file_size_kb": round(os.path.getsize(filepath) / 1024, 1),
        })

    return results


def get_tune_detail(symbol: str) -> Optional[Dict[str, Any]]:
    """Load full tuning detail for a single asset."""
    filepath = os.path.join(TUNE_DIR, f"{symbol}.json")
    if not os.path.isfile(filepath):
        # Try uppercase
        filepath = os.path.join(TUNE_DIR, f"{symbol.upper()}.json")
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def get_pit_failures() -> List[Dict[str, Any]]:
    """Return list of assets failing PIT calibration (AD test)."""
    assets = list_tuned_assets()
    failures = []
    for asset in assets:
        if asset.get("ad_pass") is False:
            failures.append(asset)
    return failures


def get_tune_stats() -> Dict[str, Any]:
    """Summary statistics for the tuning cache."""
    assets = list_tuned_assets()
    if not assets:
        return {"total": 0, "pit_pass": 0, "pit_fail": 0}

    pit_pass = sum(1 for a in assets if a.get("ad_pass") is True)
    pit_fail = sum(1 for a in assets if a.get("ad_pass") is False)
    pit_unknown = len(assets) - pit_pass - pit_fail

    models_used = {}
    for a in assets:
        m = a.get("best_model", "unknown")
        models_used[m] = models_used.get(m, 0) + 1

    return {
        "total": len(assets),
        "pit_pass": pit_pass,
        "pit_fail": pit_fail,
        "pit_unknown": pit_unknown,
        "models_distribution": models_used,
    }
