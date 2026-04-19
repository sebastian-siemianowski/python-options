"""
Signals router — signal cache data, high conviction signals, computation trigger.
"""

import asyncio
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Query

from web.backend.services.signal_service import (
    get_summary_rows,
    get_asset_blocks,
    get_failed_assets,
    get_high_conviction_signals,
    get_signal_stats,
    get_horizons,
    get_signals_by_sector,
    get_strong_signal_symbols,
    _invalidate_signal_cache,
)
from web.backend.ws import manager as ws_manager

router = APIRouter()


@router.get("/summary")
async def signal_summary():
    """Summary rows — the universal signal table data."""
    rows = get_summary_rows()
    horizons = get_horizons()
    return {"summary_rows": rows, "horizons": horizons, "total": len(rows)}


@router.get("/by-sector")
async def signals_by_sector():
    """Summary rows grouped by consolidated sector with aggregate stats."""
    sectors = get_signals_by_sector()
    return {"sectors": sectors, "total_sectors": len(sectors)}


@router.get("/strong-signals")
async def strong_signals():
    """Symbols with STRONG BUY or STRONG SELL labels."""
    data = get_strong_signal_symbols()
    return data


@router.get("/assets")
async def signal_assets():
    """Full asset blocks with all signal fields."""
    blocks = get_asset_blocks()
    return {"assets": blocks, "total": len(blocks)}


@router.get("/stats")
async def signal_stats():
    """Signal cache statistics."""
    return get_signal_stats()


@router.post("/refresh-cache")
async def refresh_signal_cache():
    """Invalidate the in-memory signal cache, forcing a reload on next request."""
    _invalidate_signal_cache()
    # Story 6.4: Broadcast signal updates to WebSocket clients
    rows = get_summary_rows()
    for row in rows:
        await ws_manager.broadcast({
            "type": "signal_update",
            "symbol": row.get("asset_label", ""),
            "timestamp": datetime.now().isoformat(),
            "summary": row,
        })
    return {"status": "ok", "message": "Signal cache invalidated"}


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


@router.get("/quality-scores")
async def quality_scores():
    """AI Business Quality Scores for all assets (0-100)."""
    from web.backend.services.quality_scores import get_all_quality_scores
    return get_all_quality_scores()


@router.get("/intrinsic-values")
async def intrinsic_values():
    """Buffett/Munger intrinsic value estimates with current prices and valuation gaps."""
    from web.backend.services.intrinsic_values import get_all_intrinsic_data
    return get_all_intrinsic_data()


@router.get("/asset/{symbol}")
async def signal_for_asset(symbol: str):
    """Get signals for a specific asset."""
    blocks = get_asset_blocks()
    for block in blocks:
        block_sym = block.get("symbol", "")
        if block_sym.upper() == symbol.upper():
            return block
    return {"error": f"Asset {symbol} not found in cache"}
