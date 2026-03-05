"""
Tuning router — tuning cache, PIT calibration status, model weights.
"""

from fastapi import APIRouter

from web.backend.services.tune_service import (
    list_tuned_assets,
    get_tune_detail,
    get_pit_failures,
    get_tune_stats,
)

router = APIRouter()


@router.get("/list")
async def tune_list():
    """List all tuned assets with summary info."""
    assets = list_tuned_assets()
    return {"assets": assets, "total": len(assets)}


@router.get("/stats")
async def tune_stats():
    """Tuning cache statistics."""
    return get_tune_stats()


@router.get("/pit-failures")
async def pit_failures():
    """Assets failing PIT calibration (AD test)."""
    failures = get_pit_failures()
    return {"failures": failures, "count": len(failures)}


@router.get("/detail/{symbol}")
async def tune_detail(symbol: str):
    """Full tuning detail for a single asset."""
    detail = get_tune_detail(symbol)
    if detail is None:
        return {"error": f"No tuning data for {symbol}"}
    return {"symbol": symbol, "data": detail}
