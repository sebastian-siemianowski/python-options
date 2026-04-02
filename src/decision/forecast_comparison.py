"""
Story 4.7: Forecast Comparison Mode (Previous vs Current).

Stores daily signal snapshots and computes deltas between current
and previous forecasts.

Usage:
    from decision.forecast_comparison import (
        save_signal_snapshot, load_previous_snapshot,
        compute_forecast_deltas, prune_old_snapshots,
    )
"""
import json
import os
import glob
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
HISTORY_DIR = os.path.join(REPO_ROOT, "data", "signal_history")

# History retention
HISTORY_RETENTION_DAYS = 30

# Large change threshold: |delta| > |previous| * 0.5
LARGE_CHANGE_RATIO = 0.5


def save_signal_snapshot(signals: dict, date_str: Optional[str] = None):
    """
    Save current signal output as dated snapshot.
    
    File: signal_history/signals_{YYYY-MM-DD}.json
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    path = os.path.join(HISTORY_DIR, f"signals_{date_str}.json")
    with open(path, "w") as f:
        json.dump(signals, f, indent=2, default=str)
    
    return path


def load_previous_snapshot(current_date: Optional[str] = None) -> Optional[dict]:
    """
    Find and load the most recent snapshot BEFORE current_date.
    """
    if current_date is None:
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    pattern = os.path.join(HISTORY_DIR, "signals_*.json")
    files = sorted(glob.glob(pattern))
    
    # Find most recent file before current_date
    for filepath in reversed(files):
        basename = os.path.basename(filepath)
        # Extract date: signals_YYYY-MM-DD.json
        file_date = basename.replace("signals_", "").replace(".json", "")
        if file_date < current_date:
            with open(filepath, "r") as f:
                return json.load(f)
    
    return None


def compute_forecast_deltas(
    current: dict,
    previous: dict,
) -> Dict[str, Dict[str, dict]]:
    """
    Compute per-asset per-horizon deltas between current and previous forecasts.
    
    Returns:
        {
            "AAPL": {
                "7": {"current": 2.5, "previous": 1.0, "delta": 1.5, "large_change": False},
                "30": {"current": -3.0, "previous": 5.0, "delta": -8.0, "large_change": True},
            },
            ...
        }
    """
    deltas = {}
    
    for symbol, curr_data in current.items():
        if symbol not in previous:
            continue
        
        prev_data = previous[symbol]
        asset_deltas = {}
        
        # Extract horizon forecasts from both
        curr_hf = curr_data if isinstance(curr_data, dict) else {}
        prev_hf = prev_data if isinstance(prev_data, dict) else {}
        
        # Get horizon forecasts
        curr_forecasts = curr_hf.get("horizon_forecasts", curr_hf)
        prev_forecasts = prev_hf.get("horizon_forecasts", prev_hf)
        
        for horizon_key, curr_val in curr_forecasts.items():
            if horizon_key not in prev_forecasts:
                continue
            
            prev_val = prev_forecasts[horizon_key]
            
            # Extract point forecast
            c = curr_val if isinstance(curr_val, (int, float)) else curr_val.get("point_forecast_pct", 0)
            p = prev_val if isinstance(prev_val, (int, float)) else prev_val.get("point_forecast_pct", 0)
            
            delta = c - p
            large = abs(delta) > abs(p) * LARGE_CHANGE_RATIO if p != 0 else abs(delta) > 1.0
            
            asset_deltas[str(horizon_key)] = {
                "current": c,
                "previous": p,
                "delta": delta,
                "large_change": large,
            }
        
        if asset_deltas:
            deltas[symbol] = asset_deltas
    
    return deltas


def get_delta_arrow(delta: float) -> str:
    """Return text arrow indicator for delta."""
    if abs(delta) < 0.01:
        return "="
    return "^" if delta > 0 else "v"


def prune_old_snapshots(retention_days: int = HISTORY_RETENTION_DAYS) -> int:
    """
    Remove snapshots older than retention_days.
    Returns count of removed files.
    """
    if not os.path.exists(HISTORY_DIR):
        return 0
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    
    pattern = os.path.join(HISTORY_DIR, "signals_*.json")
    removed = 0
    
    for filepath in glob.glob(pattern):
        basename = os.path.basename(filepath)
        file_date = basename.replace("signals_", "").replace(".json", "")
        if file_date < cutoff_str:
            os.remove(filepath)
            removed += 1
    
    return removed
