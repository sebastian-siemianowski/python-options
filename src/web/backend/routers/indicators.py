"""
Indicators router — leaderboard, strategy detail, families, heatmaps, backtest.
"""

from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/leaderboard")
async def leaderboard(top: int = Query(0, ge=0, le=500), family: str = Query(None)):
    """Ranked leaderboard of all 500 indicator strategies. top=0 means all."""
    from web.backend.services.indicators_service import get_leaderboard
    return get_leaderboard(top_n=top, family=family)


@router.get("/top10")
async def top_10():
    """The elite top 10 strategies."""
    from web.backend.services.indicators_service import get_top_10
    return get_top_10()


@router.get("/families")
async def families():
    """Strategy families with counts and average scores."""
    from web.backend.services.indicators_service import get_families
    return get_families()


@router.get("/strategy/{strategy_id}")
async def strategy_detail(strategy_id: int):
    """Full detail for a single strategy including per-asset results."""
    from web.backend.services.indicators_service import get_strategy_detail
    result = get_strategy_detail(strategy_id)
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Strategy not found")
    return result


@router.get("/strategy/{strategy_id}/heatmap")
async def strategy_heatmap(strategy_id: int):
    """Per-asset performance heatmap for a strategy."""
    from web.backend.services.indicators_service import get_asset_heatmap
    result = get_asset_heatmap(strategy_id)
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Strategy not found")
    return result


@router.post("/refresh")
async def refresh_cache():
    """Clear cached results to reload from disk."""
    from web.backend.services.indicators_service import clear_cache
    clear_cache()
    return {"status": "ok"}


@router.post("/backtest")
async def run_backtest(mode: str = Query("full", pattern="^(quick|full)$")):
    """Start a background backtest run. mode=quick (10 assets) or full (all)."""
    from web.backend.services.indicators_service import start_backtest
    return start_backtest(mode=mode)


@router.get("/backtest/status")
async def backtest_status():
    """Get the current backtest run status."""
    from web.backend.services.indicators_service import get_backtest_status
    return get_backtest_status()
