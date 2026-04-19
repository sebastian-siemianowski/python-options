"""
Tune service — reads per-asset tuning cache and provides summaries.

Includes in-memory caching to avoid re-reading all JSON files on every request.
"""

import json
import math
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


def _safe_float(v: Any) -> Optional[float]:
    """Return a JSON-safe float, converting NaN/Inf to None."""
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None so json.dumps won't crash."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


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

        # Derive PIT pass/fail from KS p-value (the actual field in tune files)
        ks_pvalue = _safe_float(g.get("pit_ks_pvalue"))
        ks_stat = _safe_float(g.get("ks_statistic"))
        pit_grade = g.get("pit_calibration_grade")
        if ks_pvalue is not None:
            pit_pass = ks_pvalue >= 0.05
        else:
            pit_pass = None

        results.append({
            "symbol": symbol,
            "best_model": g.get("best_model", "unknown"),
            "pit_calibration_grade": pit_grade,
            "ad_pass": pit_pass,
            "ks_pvalue": round(ks_pvalue, 6) if ks_pvalue is not None else None,
            "ks_stat": round(ks_stat, 6) if ks_stat is not None else None,
            "num_models": len(model_weights),
            "bic": _safe_float(g.get("bic")),
            "phi": _safe_float(g.get("phi")),
            "nu": _safe_float(g.get("nu")),
            "n_obs": g.get("n_obs"),
            "top_weight": round(_safe_float(top_weight) or 0, 4),
            "cache_version": meta.get("cache_version", "unknown"),
            "last_tuned": meta.get("timestamp", "unknown"),
            "file_size_kb": round(os.path.getsize(filepath) / 1024, 1),
        })

    _tune_cache = results
    _tune_cache_time = time.time()
    return results


def get_tune_detail(symbol: str) -> Optional[Dict[str, Any]]:
    """Load full tuning detail for a single asset (sanitized for JSON safety)."""
    filepath = os.path.join(TUNE_DIR, f"{symbol}.json")
    if not os.path.isfile(filepath):
        # Try uppercase
        filepath = os.path.join(TUNE_DIR, f"{symbol.upper()}.json")
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return _sanitize_for_json(data)
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
    """Summary statistics for the tuning cache — includes rich per-model analytics."""
    assets = list_tuned_assets()
    if not assets:
        return {"total": 0, "pit_pass": 0, "pit_fail": 0}

    pit_pass = sum(1 for a in assets if a.get("ad_pass") is True)
    pit_fail = sum(1 for a in assets if a.get("ad_pass") is False)
    pit_unknown = len(assets) - pit_pass - pit_fail

    # Build rich per-model stats
    model_detail: Dict[str, Dict[str, Any]] = {}
    for a in assets:
        m = a.get("best_model", "unknown")
        if m not in model_detail:
            model_detail[m] = {
                "count": 0,
                "bics": [],
                "phis": [],
                "nus": [],
                "weights": [],
                "ks_pvalues": [],
                "pit_pass": 0,
                "pit_fail": 0,
                "n_obs_list": [],
                "symbols": [],
            }
        md = model_detail[m]
        md["count"] += 1
        md["symbols"].append(a["symbol"])
        if a.get("bic") is not None:
            md["bics"].append(a["bic"])
        if a.get("phi") is not None:
            md["phis"].append(a["phi"])
        if a.get("nu") is not None:
            md["nus"].append(a["nu"])
        if a.get("top_weight") is not None:
            md["weights"].append(a["top_weight"])
        if a.get("ks_pvalue") is not None:
            md["ks_pvalues"].append(a["ks_pvalue"])
        if a.get("n_obs") is not None:
            md["n_obs_list"].append(a["n_obs"])
        if a.get("ad_pass") is True:
            md["pit_pass"] += 1
        elif a.get("ad_pass") is False:
            md["pit_fail"] += 1

    def _avg(lst: list) -> Optional[float]:
        return round(sum(lst) / len(lst), 4) if lst else None

    def _median(lst: list) -> Optional[float]:
        if not lst:
            return None
        s = sorted(lst)
        n = len(s)
        mid = n // 2
        return round((s[mid] + s[mid - 1]) / 2, 4) if n % 2 == 0 else round(s[mid], 4)

    # Flatten to JSON-friendly structure
    models_distribution: Dict[str, int] = {}
    models_analytics: Dict[str, Dict[str, Any]] = {}
    for m, md in model_detail.items():
        models_distribution[m] = md["count"]
        total_pit = md["pit_pass"] + md["pit_fail"]
        models_analytics[m] = {
            "count": md["count"],
            "avg_bic": _avg(md["bics"]),
            "median_bic": _median(md["bics"]),
            "best_bic": round(min(md["bics"]), 1) if md["bics"] else None,
            "worst_bic": round(max(md["bics"]), 1) if md["bics"] else None,
            "avg_phi": _avg(md["phis"]),
            "avg_nu": _avg(md["nus"]),
            "avg_weight": _avg(md["weights"]),
            "avg_ks_pvalue": _avg(md["ks_pvalues"]),
            "median_ks_pvalue": _median(md["ks_pvalues"]),
            "pit_pass": md["pit_pass"],
            "pit_fail": md["pit_fail"],
            "pit_pass_rate": round(md["pit_pass"] / total_pit, 4) if total_pit > 0 else None,
            "avg_n_obs": _avg(md["n_obs_list"]),
            "top_symbols": md["symbols"][:5],
        }

    return {
        "total": len(assets),
        "pit_pass": pit_pass,
        "pit_fail": pit_fail,
        "pit_unknown": pit_unknown,
        "models_distribution": models_distribution,
        "models_analytics": models_analytics,
    }
