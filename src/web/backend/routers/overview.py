"""
Overview router — system health, summary stats, recent activity.
"""

from fastapi import APIRouter

from web.backend.services.signal_service import get_signal_stats, get_cache_age_seconds
from web.backend.services.tune_service import get_tune_stats
from web.backend.services.data_service import get_data_summary

router = APIRouter()


@router.get("/overview")
async def overview():
    """Full system overview for the dashboard home page."""
    signal_stats = get_signal_stats()
    tune_stats = get_tune_stats()
    data_summary = get_data_summary()

    return {
        "signals": signal_stats,
        "tuning": tune_stats,
        "data": data_summary,
    }
