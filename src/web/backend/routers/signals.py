"""
Signals router — signal cache data, high conviction signals, computation trigger.
"""

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Query

from web.backend.services.signal_service import (
    get_summary_rows,
    get_asset_blocks,
    get_failed_assets,
    get_high_conviction_signals,
    get_signal_stats,
    get_horizons,
)

router = APIRouter()


@router.get("/summary")
async def signal_summary():
    """Summary rows — the universal signal table data."""
    rows = get_summary_rows()
    horizons = get_horizons()
    return {"summary_rows": rows, "horizons": horizons, "total": len(rows)}


@router.get("/assets")
async def signal_assets():
    """Full asset blocks with all signal fields."""
    blocks = get_asset_blocks()
    return {"assets": blocks, "total": len(blocks)}


@router.get("/stats")
async def signal_stats():
    """Signal cache statistics."""
    return get_signal_stats()


@router.get("/failed")
async def signal_failed():
    """Assets that failed during signal computation."""
    failed = get_failed_assets()
    return {"failed_assets": failed, "count": len(failed)}


@router.get("/high-conviction/{signal_type}")
async def high_conviction(signal_type: str):
    """
    High conviction signals.
    
    Args:
        signal_type: 'buy' or 'sell'
    """
    if signal_type not in ("buy", "sell"):
        return {"error": "signal_type must be 'buy' or 'sell'"}
    signals = get_high_conviction_signals(signal_type)
    return {"signal_type": signal_type, "signals": signals, "count": len(signals)}


@router.get("/asset/{symbol}")
async def signal_for_asset(symbol: str):
    """Get signals for a specific asset."""
    blocks = get_asset_blocks()
    for block in blocks:
        block_sym = block.get("symbol", "")
        if block_sym.upper() == symbol.upper():
            return block
    return {"error": f"Asset {symbol} not found in cache"}
