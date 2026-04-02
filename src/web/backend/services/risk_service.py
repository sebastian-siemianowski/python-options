"""
Risk service — wraps risk_dashboard for JSON output with caching.
"""

import os
import sys
import json
import time
from typing import Any, Dict, Optional

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
CACHE_DIR = os.path.join(SRC_DIR, "data", "cache")
RISK_CACHE_FILE = os.path.join(CACHE_DIR, "risk_dashboard.json")
RISK_CACHE_MAX_AGE = 3600  # 1 hour

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _load_cached_risk() -> Optional[Dict[str, Any]]:
    """Load cached risk data if fresh enough."""
    if not os.path.isfile(RISK_CACHE_FILE):
        return None
    try:
        age = time.time() - os.path.getmtime(RISK_CACHE_FILE)
        if age > RISK_CACHE_MAX_AGE:
            return None
        with open(RISK_CACHE_FILE, "r") as f:
            data = json.load(f)
        data["_cached"] = True
        data["_cache_age_seconds"] = round(age)
        return data
    except (json.JSONDecodeError, IOError, OSError):
        return None


def _save_risk_cache(data: Dict[str, Any]) -> None:
    """Save risk data to cache file."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(RISK_CACHE_FILE, "w") as f:
            json.dump(data, f, default=str)
    except (IOError, OSError):
        pass


def compute_risk_json(start_date: str = "2020-01-01", force: bool = False) -> Dict[str, Any]:
    """
    Run the full unified risk dashboard and return JSON.
    
    Uses cached data if available and fresh (< 1 hour), unless force=True.
    This calls the existing compute_and_render_unified_risk with output_json=True.
    """
    if not force:
        cached = _load_cached_risk()
        if cached:
            return cached

    try:
        from decision.risk_dashboard import compute_and_render_unified_risk
        result = compute_and_render_unified_risk(
            start_date=start_date,
            suppress_output=True,
            output_json=True,
            use_parallel=True,
        )
        if result and "error" not in result:
            _save_risk_cache(result)
        return result or {"error": "No result returned"}
    except Exception as e:
        return {"error": str(e)}


def get_risk_temperature_summary(risk_json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract high-level temperature summary from risk JSON.
    
    Returns combined temperature, status, and per-module temperatures.
    """
    if risk_json is None:
        risk_json = compute_risk_json()

    if "error" in risk_json:
        return risk_json

    risk = risk_json.get("risk_temperature", {})
    metals = risk_json.get("metals_risk_temperature", {})
    market = risk_json.get("market_temperature", {})

    risk_temp = risk.get("temperature", 0)
    metals_temp = metals.get("temperature", 0)
    market_temp = market.get("temperature", 0)

    combined_temp = 0.4 * risk_temp + 0.3 * metals_temp + 0.3 * market_temp

    if combined_temp < 0.3:
        status = "Calm"
    elif combined_temp < 0.7:
        status = "Elevated"
    elif combined_temp < 1.2:
        status = "Stressed"
    else:
        status = "Crisis"

    return {
        "combined_temperature": round(combined_temp, 3),
        "status": status,
        "risk_temperature": round(risk_temp, 3),
        "metals_temperature": round(metals_temp, 3),
        "market_temperature": round(market_temp, 3),
        "computed_at": risk_json.get("computed_at"),
    }
