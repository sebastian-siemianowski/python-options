"""
Data router — price data status, data refresh, directory summaries.
"""

from fastapi import APIRouter

from web.backend.services.data_service import (
    list_price_files,
    get_data_summary,
    get_directories_summary,
)

router = APIRouter()


@router.get("/status")
async def data_status():
    """Summary of data freshness and availability."""
    return get_data_summary()


@router.get("/prices")
async def data_prices():
    """List all price data files with metadata."""
    files = list_price_files()
    return {"files": files, "total": len(files)}


@router.get("/directories")
async def data_directories():
    """Summary of key data directories."""
    return get_directories_summary()
