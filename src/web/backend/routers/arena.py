"""
Arena router — arena results, safe storage, model competition.
"""

import json
import os
import glob
from typing import Any, Dict, List

from fastapi import APIRouter

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
ARENA_DIR = os.path.join(SRC_DIR, "arena")
SAFE_STORAGE_DIR = os.path.join(ARENA_DIR, "safe_storage")
SAFE_STORAGE_DATA_DIR = os.path.join(SAFE_STORAGE_DIR, "data")
ARENA_DATA_DIR = os.path.join(ARENA_DIR, "data")

router = APIRouter()


def _load_safe_storage_scores() -> Dict[str, Dict[str, Any]]:
    """Load scoring data from safe_storage_results.json."""
    results_path = os.path.join(SAFE_STORAGE_DATA_DIR, "safe_storage_results.json")
    if not os.path.isfile(results_path):
        return {}
    try:
        with open(results_path, "r") as f:
            data = json.load(f)
        return data.get("models", {})
    except (json.JSONDecodeError, IOError):
        return {}


def _list_safe_storage_models() -> List[Dict[str, Any]]:
    """List models in safe storage with scores from results file."""
    if not os.path.isdir(SAFE_STORAGE_DIR):
        return []

    scores = _load_safe_storage_scores()

    models = []
    for f in sorted(os.listdir(SAFE_STORAGE_DIR)):
        if f.endswith(".py") and not f.startswith("__"):
            name = f.replace(".py", "")
            filepath = os.path.join(SAFE_STORAGE_DIR, f)
            size_kb = round(os.path.getsize(filepath) / 1024, 1)
            entry: Dict[str, Any] = {
                "name": name,
                "filename": f,
                "size_kb": size_kb,
            }
            # Merge scoring data if available
            if name in scores:
                s = scores[name]
                entry.update({
                    "final": s.get("final"),
                    "bic": s.get("bic"),
                    "crps": s.get("crps"),
                    "hyv": s.get("hyv"),
                    "pit": s.get("pit"),
                    "pit_rate": s.get("pit_rate"),
                    "css": s.get("css"),
                    "fec": s.get("fec"),
                    "time_ms": s.get("time_ms"),
                    "n_tests": s.get("n_tests"),
                    "has_scores": True,
                })
            else:
                entry["has_scores"] = False
            models.append(entry)

    # Sort by final score descending if available
    models.sort(key=lambda m: m.get("final") or 0, reverse=True)
    return models


def _get_latest_results() -> Dict[str, Any]:
    """Load the latest arena competition results."""
    results_dir = os.path.join(ARENA_DATA_DIR, "results")
    if not os.path.isdir(results_dir):
        return {"error": "No results directory found"}

    # Find the most recent results JSON
    result_files = sorted(glob.glob(os.path.join(results_dir, "*.json")), reverse=True)
    if not result_files:
        return {"error": "No result files found"}

    try:
        with open(result_files[0], "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"error": "Could not parse results file"}


@router.get("/safe-storage")
async def arena_safe_storage():
    """List models in safe storage."""
    models = _list_safe_storage_models()
    return {"models": models, "count": len(models)}


@router.get("/results")
async def arena_results():
    """Latest arena competition results."""
    return _get_latest_results()


@router.get("/status")
async def arena_status():
    """Arena system status — experimental models, benchmark universe."""
    experimental_dir = os.path.join(ARENA_DIR, "experimental_models")
    exp_count = 0
    if os.path.isdir(experimental_dir):
        exp_count = len([f for f in os.listdir(experimental_dir)
                         if f.endswith(".py") and not f.startswith("__")])

    safe_models = _list_safe_storage_models()

    return {
        "safe_storage_count": len(safe_models),
        "experimental_count": exp_count,
        "benchmark_symbols": ["UPST", "AFRM", "IONQ", "CRWD", "DKNG", "SNAP",
                              "AAPL", "NVDA", "TSLA", "SPY", "QQQ", "IWM"],
    }
