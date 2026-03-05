"""
Overview router — system health, summary stats, recent activity.

Hardened: each service call is wrapped in try/except so one failure
does not bring down the entire overview endpoint.
"""

import traceback
from fastapi import APIRouter

from web.backend.services.signal_service import get_signal_stats, get_cache_age_seconds
from web.backend.services.tune_service import get_tune_stats
from web.backend.services.data_service import get_data_summary

router = APIRouter()

_SIGNAL_DEFAULTS = {"cached": False, "total_assets": 0, "failed": 0,
                    "buy_signals": 0, "sell_signals": 0, "hold_signals": 0,
                    "strong_buy_signals": 0, "strong_sell_signals": 0, "exit_signals": 0,
                    "cache_age_seconds": None}
_TUNE_DEFAULTS = {"total": 0, "pit_pass": 0, "pit_fail": 0, "pit_unknown": 0,
                  "models_distribution": {}}
_DATA_DEFAULTS = {"total_files": 0, "stale_files": 0, "fresh_files": 0,
                  "freshest_hours": None, "oldest_hours": None, "total_size_mb": 0}


@router.get("/overview")
async def overview():
    """Full system overview for the dashboard home page."""
    errors = []

    try:
        signal_stats = get_signal_stats()
    except Exception as e:
        signal_stats = dict(_SIGNAL_DEFAULTS)
        errors.append(f"signals: {e}")

    try:
        tune_stats = get_tune_stats()
    except Exception as e:
        tune_stats = dict(_TUNE_DEFAULTS)
        errors.append(f"tuning: {e}")

    try:
        data_summary = get_data_summary()
    except Exception as e:
        data_summary = dict(_DATA_DEFAULTS)
        errors.append(f"data: {e}")

    result = {
        "signals": signal_stats,
        "tuning": tune_stats,
        "data": data_summary,
    }
    if errors:
        result["errors"] = errors
    return result
