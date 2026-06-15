"""
Politicians router -- public disclosure monitoring endpoints.
"""

import asyncio
from datetime import date

from fastapi import APIRouter, HTTPException, Query, Request

from ingestion.politicians.daily import run_daily_politicians_sync
from web.backend.services.politicians_service import (
    get_politicians_disabled_response,
    get_politicians_filer_response,
    get_politicians_notice_response,
    get_politicians_asset_response,
    get_politicians_source_health_response,
    get_politicians_summary_response,
    get_politicians_trades_response,
    invalidate_politicians_cache,
)
from ingestion.politicians.compliance import is_politicians_enabled

router = APIRouter()

VALID_TRADE_FLAGS = {"late_disclosure", "large_trade_bucket", "ticker_ambiguous", "amended"}
VALID_TRANSACTION_TYPES = {"purchase", "sale", "sale_partial", "exchange", "received", "other", "unknown"}
VALID_TRANSACTION_SIDES = {"purchase", "sale"}
VALID_CHAMBERS = {"house", "senate"}
VALID_OWNERS = {"self", "spouse", "dependent_child", "joint", "unknown"}


@router.get("/notice")
async def politicians_notice():
    """Canonical data-use notice for politician disclosure data."""
    return get_politicians_notice_response()


@router.get("/source-health")
async def politicians_source_health():
    """Source health including parser-confidence buckets."""
    return get_politicians_source_health_response()


@router.get("/summary")
async def politicians_summary(as_of: str | None = Query(default=None)):
    """Headline aggregate metrics for politician disclosure monitoring."""
    return get_politicians_summary_response(as_of_date=as_of)


@router.post("/refresh-cache")
async def politicians_refresh_cache():
    """Invalidate politician disclosure API caches."""
    return invalidate_politicians_cache()


@router.post("/sync")
async def politicians_sync(
    date_from: str | None = Query(default=None, alias="from"),
    date_to: str | None = Query(default=None, alias="to"),
):
    """Run source sync, parser, validation, and API-cache refresh now."""
    if not is_politicians_enabled():
        return get_politicians_disabled_response(endpoint="POST /sync")
    return await asyncio.to_thread(
        run_daily_politicians_sync,
        date_from=date_from,
        date_to=date_to,
    )


@router.get("/trades")
async def politicians_trades(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    symbol: str | None = Query(default=None),
    filer: str | None = Query(default=None),
    chamber: str | None = Query(default=None),
    party: str | None = Query(default=None),
    state: str | None = Query(default=None),
    transaction_type: str | None = Query(default=None),
    transaction_side: str | None = Query(default=None),
    owner: str | None = Query(default=None),
    flag: str | None = Query(default=None),
    tracked_only: bool = Query(default=False),
    watchlist_only: bool = Query(default=False),
    top_traders_only: bool = Query(default=False),
    stock_linked_only: bool = Query(default=False),
    from_date: str | None = Query(default=None, alias="from"),
    to_date: str | None = Query(default=None, alias="to"),
):
    """Paginated normalized trades with disclosure, filer, asset, and flag filters."""
    _validate_trade_feed_filters(
        chamber=chamber,
        transaction_type=transaction_type,
        transaction_side=transaction_side,
        owner=owner,
        flag=flag,
        from_date=from_date,
        to_date=to_date,
    )
    return get_politicians_trades_response(
        flag=flag,
        limit=limit,
        offset=offset,
        symbol=symbol,
        filer=filer,
        chamber=chamber,
        party=party,
        state=state,
        transaction_type=transaction_type,
        transaction_side=transaction_side,
        owner=owner,
        from_date=from_date,
        to_date=to_date,
        tracked_only=tracked_only,
        watchlist_only=watchlist_only,
        top_traders_only=top_traders_only,
        stock_linked_only=stock_linked_only,
    )


@router.get("/assets/{symbol}")
async def politicians_asset(symbol: str, window_days: int = Query(default=180, ge=1, le=3650)):
    """Politician activity for chart and signal context surfaces."""
    return get_politicians_asset_response(symbol, window_days=window_days)


@router.get("/filers/{filer_id}")
async def politicians_filer(filer_id: str, window_days: int = Query(default=180, ge=1, le=3650)):
    """Sanitized member-level disclosure activity."""
    return get_politicians_filer_response(filer_id, window_days=window_days)


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def politicians_fallback(path: str, request: Request):
    """Return structured disabled payloads for all politician endpoints."""
    if not is_politicians_enabled():
        return get_politicians_disabled_response(endpoint=f"{request.method} /{path}")
    raise HTTPException(status_code=404, detail=f"Unknown politicians endpoint: /{path}")


def _validate_trade_feed_filters(
    *,
    chamber: str | None,
    transaction_type: str | None,
    transaction_side: str | None,
    owner: str | None,
    flag: str | None,
    from_date: str | None,
    to_date: str | None,
) -> None:
    errors: list[dict[str, str]] = []
    _validate_choice(errors, "chamber", chamber, VALID_CHAMBERS)
    _validate_choice(errors, "transaction_type", transaction_type, VALID_TRANSACTION_TYPES)
    _validate_choice(errors, "transaction_side", transaction_side, VALID_TRANSACTION_SIDES)
    _validate_choice(errors, "owner", owner, VALID_OWNERS)
    _validate_choice(errors, "flag", flag, VALID_TRADE_FLAGS)
    parsed_from = _parse_query_date(errors, "from", from_date)
    parsed_to = _parse_query_date(errors, "to", to_date)
    if parsed_from and parsed_to and parsed_from > parsed_to:
        errors.append({"field": "from", "message": "from must be on or before to"})
    if errors:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Invalid politician trade filters",
                "errors": errors,
            },
        )


def _validate_choice(
    errors: list[dict[str, str]],
    field: str,
    value: str | None,
    allowed: set[str],
) -> None:
    if value is None:
        return
    if value.lower() not in allowed:
        errors.append({
            "field": field,
            "message": f"{field} must be one of: {', '.join(sorted(allowed))}",
        })


def _parse_query_date(errors: list[dict[str, str]], field: str, value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        errors.append({"field": field, "message": f"{field} must be an ISO date like 2026-05-29"})
        return None
