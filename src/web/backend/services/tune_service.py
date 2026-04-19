"""
Tune service — reads per-asset tuning cache and provides summaries.

Includes in-memory caching to avoid re-reading all JSON files on every request.
"""

import json
import os
import glob
import time
from typing import Any, Dict, List, Optional

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
TUNE_DIR = os.path.join(SRC_DIR, "data", "tune")

# ── In-memory cache ─────────────────────────────────────────────────
_tune_cache: List[Dict[str, Any]] = []
_tune_cache_time: float = 0.0
_TUNE_TTL = 300.0  # 5 minutes


def _invalidate_tune_cache() -> None:
    """Force reload on next access."""
    global _tune_cache, _tune_cache_time
    _tune_cache = []
    _tune_cache_time = 0.0


def list_tuned_assets() -> List[Dict[str, Any]]:
    """
    List all tuned assets with summary information (cached for performance).
    
    Returns list of dicts with symbol, best_model, pit_grade, last_tuned, etc.
    """
    global _tune_cache, _tune_cache_time

    now = time.time()
    if _tune_cache and (now - _tune_cache_time) < _TUNE_TTL:
        return _tune_cache

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

        # Extract key parameters for the asset list
        model_weights = g.get("model_weights", {})
        top_weight = max(model_weights.values()) if model_weights else 0.0

        results.append({
            "symbol": symbol,
            "best_model": g.get("best_model", "unknown"),
            "pit_calibration_grade": g.get("pit_calibration_grade", "N/A"),
            "ad_stat": g.get("ad_stat"),
            "ad_critical": g.get("ad_critical_5pct"),
            "ad_pass": g.get("ad_pass"),
            "num_models": len(model_weights),
            "bic": g.get("bic"),
            "phi": g.get("phi"),
            "nu": g.get("nu"),
            "n_obs": g.get("n_obs"),
            "top_weight": round(top_weight, 4),
            "cache_version": meta.get("cache_version", "unknown"),
            "last_tuned": meta.get("timestamp", "unknown"),
            "file_size_kb": round(os.path.getsize(filepath) / 1024, 1),
        })

    _tune_cache = results
    _tune_cache_time = time.time()
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
