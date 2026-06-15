"""
Service helpers for politician disclosure monitoring.

The full ingestion and query service arrives in later stories. Story 0.1 starts
with the shared data-use notice so every future response can include the same
guardrail text.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta
from typing import Any, Dict

from decision.politician_context import compute_politician_activity_score
from decision.politician_trader_success import (
    build_successful_trader_leaderboard,
    canonical_filer_key,
    selected_filer_keys,
)
from ingestion.politicians.compliance import get_data_use_notice, get_feature_availability
from ingestion.politicians.flags import filter_rows_by_flag
from ingestion.politicians.paths import get_politicians_data_dir
from ingestion.politicians.quality import confidence_bucket, summarize_confidence_buckets
from ingestion.politicians.source_health import read_source_health
from ingestion.politicians.sync import read_sync_state

BUY_TRANSACTION_TYPES = {"purchase", "received"}
SELL_TRANSACTION_TYPES = {"sale", "sale_partial"}
OWNER_BUCKETS = ("self", "spouse", "dependent_child", "joint", "unknown")
POLITICIANS_RESPONSE_SCHEMA_VERSION = "politicians-api-v1"
POLITICIANS_CACHE_SCHEMA_VERSION = f"{POLITICIANS_RESPONSE_SCHEMA_VERSION}:jsonl-v2-successful-traders"
_TRADE_ROWS_CACHE: dict[str, tuple[str, int, int, list[dict[str, Any]]]] = {}
_SUMMARY_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def invalidate_politicians_cache() -> Dict[str, Any]:
    """Clear in-memory politician disclosure caches."""
    cleared = len(_TRADE_ROWS_CACHE) + len(_SUMMARY_CACHE)
    _TRADE_ROWS_CACHE.clear()
    _SUMMARY_CACHE.clear()
    return {
        "status": "ok",
        "cache": "politicians",
        "cleared_entries": cleared,
    }


def get_politicians_notice_response() -> Dict[str, Any]:
    """Return the canonical data-use notice response for the web UI."""
    availability = get_feature_availability()
    if not availability["enabled"]:
        return get_politicians_disabled_response()
    return {
        "feature": "politicians",
        **availability,
        "status": "notice_only",
        **_response_meta(),
        "data_use_notice": get_data_use_notice(),
    }


def get_politicians_disabled_response(endpoint: str | None = None) -> Dict[str, Any]:
    """Return a structured disabled response for politician API surfaces."""
    availability = get_feature_availability()
    return {
        "feature": "politicians",
        **availability,
        "status": "disabled",
        **_response_meta(),
        "endpoint": endpoint,
        "message": (
            "Politician disclosure monitoring is disabled. Set "
            "POLITICIANS_ENABLED=1 after compliance review to enable it."
        ),
        "data_use_notice": get_data_use_notice(),
    }


def get_politicians_summary_response(as_of_date: str | None = None) -> Dict[str, Any]:
    """Return headline aggregate metrics for the politician dashboard."""
    availability = get_feature_availability()
    if not availability["enabled"]:
        return get_politicians_disabled_response(endpoint="GET /summary")
    data_root = get_politicians_data_dir()
    rows = _read_trade_rows() or []
    tracked_symbols = _load_tracked_symbols()
    watchlist_symbols = _load_watchlist_symbols()
    source_health = read_source_health(data_root)
    cache_key = _summary_cache_key(
        as_of_date=as_of_date,
        tracked_symbols=tracked_symbols,
        watchlist_symbols=watchlist_symbols,
        data_root=data_root,
    )
    summary = _SUMMARY_CACHE.get(cache_key)
    if summary is None:
        summary = _compute_summary_metrics(
            rows,
            tracked_symbols=tracked_symbols,
            watchlist_symbols=watchlist_symbols,
            source_health=source_health,
            as_of_date=as_of_date,
        )
        _SUMMARY_CACHE[cache_key] = summary
    return {
        "feature": "politicians",
        **availability,
        "status": "ok",
        **_response_meta(),
        **summary,
        "summary": summary,
        "data_use_notice": get_data_use_notice(),
    }


def get_politicians_source_health_response() -> Dict[str, Any]:
    """Return source health plus parser-confidence bucket counts."""
    availability = get_feature_availability()
    if not availability["enabled"]:
        return get_politicians_disabled_response(endpoint="GET /source-health")
    data_root = get_politicians_data_dir()
    rows = _read_trade_rows() or []
    filings = _read_jsonl(data_root / "filings.jsonl")
    parse_errors = _read_jsonl(data_root / "parse_errors.jsonl")
    raw_source_health = read_source_health(data_root)
    sync_state = read_sync_state(data_root)
    normalized_sources = _build_source_health_summary(
        rows=rows,
        filings=filings,
        parse_errors=parse_errors,
        raw_source_health=raw_source_health,
        sync_state=sync_state,
    )
    return {
        "feature": "politicians",
        **availability,
        "status": "ok",
        **_response_meta(),
        "overall_status": _overall_source_status(normalized_sources),
        "sources": normalized_sources,
        "source_health": raw_source_health,
        "confidence_buckets": summarize_confidence_buckets(rows),
        "parse_error_count": len(parse_errors),
        "data_use_notice": get_data_use_notice(),
    }


def get_politicians_trades_response(
    flag: str | None = None,
    *,
    limit: int | None = None,
    offset: int = 0,
    symbol: str | None = None,
    filer: str | None = None,
    chamber: str | None = None,
    party: str | None = None,
    state: str | None = None,
    transaction_type: str | None = None,
    transaction_side: str | None = None,
    owner: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    tracked_only: bool = False,
    watchlist_only: bool = False,
    top_traders_only: bool = False,
    stock_linked_only: bool = False,
) -> Dict[str, Any]:
    """Return normalized politician trades, optionally filtered by flag."""
    availability = get_feature_availability()
    if not availability["enabled"]:
        return get_politicians_disabled_response(endpoint="GET /trades")
    raw_rows = _read_trade_rows()
    if raw_rows is None:
        return _missing_data_response(endpoint="GET /trades")
    rows = enrich_asset_linkage(raw_rows)
    successful_traders = build_successful_trader_leaderboard(
        rows,
        prices_dir=get_politicians_data_dir().parent / "prices",
        limit=10,
    )
    rows = _attach_successful_trader_context(rows, successful_traders)
    filtered = filter_rows_by_flag(rows, flag)
    filtered = _filter_trade_rows(
        filtered,
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
    filtered.sort(key=_trade_sort_key, reverse=True)
    total_count = len(filtered)
    safe_offset = max(0, offset or 0)
    page_rows = filtered[safe_offset:]
    if limit is not None:
        page_rows = page_rows[:max(0, limit)]
    page_rows = [_attach_official_source_url(row) for row in page_rows]
    return {
        "feature": "politicians",
        **availability,
        "status": "ok",
        **_response_meta(),
        "filter": {
            "flag": flag,
            "symbol": symbol,
            "filer": filer,
            "chamber": chamber,
            "party": party,
            "state": state,
            "transaction_type": transaction_type,
            "transaction_side": transaction_side,
            "owner": owner,
            "from": from_date,
            "to": to_date,
            "tracked_only": tracked_only,
            "watchlist_only": watchlist_only,
            "top_traders_only": top_traders_only,
            "stock_linked_only": stock_linked_only,
        },
        "page": {
            "limit": limit,
            "offset": safe_offset,
            "returned": len(page_rows),
            "total": total_count,
            "has_next": safe_offset + len(page_rows) < total_count,
        },
        "total": total_count,
        "trades": page_rows,
        "successful_traders": successful_traders,
        "data_use_notice": get_data_use_notice(),
    }


def get_politicians_asset_response(
    symbol: str,
    *,
    window_days: int = 180,
    as_of_date: str | None = None,
) -> Dict[str, Any]:
    """Return politician activity for one normalized ticker symbol."""
    availability = get_feature_availability()
    if not availability["enabled"]:
        return get_politicians_disabled_response(endpoint=f"GET /assets/{symbol}")
    normalized = symbol.upper()
    raw_rows = _read_trade_rows()
    if raw_rows is None:
        return _missing_data_response(endpoint=f"GET /assets/{symbol}")
    rows = [
        row
        for row in enrich_asset_linkage(raw_rows)
        if str(row.get("ticker", "")).upper() == normalized
    ]
    as_of = _parse_date(as_of_date) or date.today()
    recent_rows = _recent_asset_rows(rows, window_days=window_days, as_of=as_of)
    recent_rows.sort(key=_trade_sort_key, reverse=True)
    recent_rows = [_attach_official_source_url(row) for row in recent_rows]
    return {
        "feature": "politicians",
        **availability,
        "status": "ok",
        **_response_meta(),
        "symbol": normalized,
        "window_days": window_days,
        "total": len(recent_rows),
        "total_symbol_trades": len(rows),
        "recent_trades": recent_rows,
        "trades": recent_rows,
        "unique_filers": _unique_filers(recent_rows),
        "unique_filer_count": len(_unique_filers(recent_rows)),
        "buy_sell_imbalance": _compute_buy_sell_imbalance(recent_rows),
        "amount_estimates": _compute_amount_estimates(recent_rows),
        "activity": compute_politician_activity_score(recent_rows, as_of_date=as_of.isoformat()),
        "disclosure_timeline": _build_disclosure_timeline(recent_rows),
        "known_limitations": _known_asset_limitations(rows),
        "data_use_notice": get_data_use_notice(),
    }


def get_politicians_filer_response(
    filer_id: str,
    *,
    window_days: int = 180,
    as_of_date: str | None = None,
) -> Dict[str, Any]:
    """Return a sanitized member-level politician disclosure view."""
    availability = get_feature_availability()
    if not availability["enabled"]:
        return get_politicians_disabled_response(endpoint=f"GET /filers/{filer_id}")
    raw_rows = _read_trade_rows()
    if raw_rows is None:
        return _missing_data_response(endpoint=f"GET /filers/{filer_id}")
    rows = [
        row
        for row in enrich_asset_linkage(raw_rows)
        if _matches_filer_identifier(row, filer_id)
    ]
    as_of = _parse_date(as_of_date) or date.today()
    recent_rows = _recent_asset_rows(rows, window_days=window_days, as_of=as_of)
    recent_rows.sort(key=_trade_sort_key, reverse=True)
    enriched_recent = [_attach_official_source_url(row) for row in recent_rows]
    sanitized_recent = [_sanitize_filer_trade(row) for row in enriched_recent]
    metadata = _build_filer_metadata(filer_id, rows)
    return {
        "feature": "politicians",
        **availability,
        "status": "ok",
        **_response_meta(),
        "filer_id": metadata["filer_id"],
        "window_days": window_days,
        "metadata": metadata,
        "total": len(sanitized_recent),
        "total_filer_trades": len(rows),
        "recent_trades": sanitized_recent,
        "top_tickers": _top_tickers(recent_rows),
        "top_sectors": _top_sectors(recent_rows),
        "delay_stats": _delay_stats(recent_rows),
        "ownership_breakdown": _ownership_breakdown(recent_rows),
        "source_documents": _source_documents(enriched_recent),
        "data_use_notice": get_data_use_notice(),
    }


def enrich_asset_linkage(
    rows: list[dict[str, Any]],
    *,
    tracked_symbols: set[str] | None = None,
    watchlist_symbols: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Attach tracked/watchlist linkage booleans to politician rows."""
    tracked = tracked_symbols if tracked_symbols is not None else _load_tracked_symbols()
    watchlist = watchlist_symbols if watchlist_symbols is not None else _load_watchlist_symbols()
    enriched = []
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        enriched.append({
            **row,
            "is_tracked_asset": bool(ticker and ticker in tracked),
            "is_watchlist_asset": bool(ticker and ticker in watchlist),
        })
    return enriched


def _read_trade_rows() -> list[dict[str, Any]] | None:
    data_root = get_politicians_data_dir()
    trades_path = data_root / "trades.jsonl"
    if not trades_path.exists():
        return None
    stat = trades_path.stat()
    cache_key = str(trades_path)
    cached = _TRADE_ROWS_CACHE.get(cache_key)
    if (
        cached
        and cached[0] == POLITICIANS_CACHE_SCHEMA_VERSION
        and cached[1] == stat.st_mtime_ns
        and cached[2] == stat.st_size
    ):
        return cached[3]
    rows = []
    with trades_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    _TRADE_ROWS_CACHE[cache_key] = (POLITICIANS_CACHE_SCHEMA_VERSION, stat.st_mtime_ns, stat.st_size, rows)
    return rows


def _read_jsonl(path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"source": "unknown", "errors": ["invalid_jsonl"], "raw": line.strip()})
    return rows


def _build_source_health_summary(
    *,
    rows: list[dict[str, Any]],
    filings: list[dict[str, Any]],
    parse_errors: list[dict[str, Any]],
    raw_source_health: dict[str, Any],
    sync_state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    sources = {
        "house",
        "senate",
        *raw_source_health.get("sources", {}).keys(),
        *sync_state.get("sources", {}).keys(),
        *(str(row.get("source")) for row in rows if row.get("source")),
        *(str(filing.get("source")) for filing in filings if filing.get("source")),
        *(str(error.get("source")) for error in parse_errors if error.get("source")),
    }
    return {
        source: _source_health_entry(
            source=source,
            rows=[row for row in rows if str(row.get("source") or "") == source],
            filings=[filing for filing in filings if str(filing.get("source") or "") == source],
            parse_errors=[error for error in parse_errors if str(error.get("source") or "") == source],
            raw_entry=raw_source_health.get("sources", {}).get(source, {}),
            sync_entry=sync_state.get("sources", {}).get(source, {}),
        )
        for source in sorted(sources)
    }


def _source_health_entry(
    *,
    source: str,
    rows: list[dict[str, Any]],
    filings: list[dict[str, Any]],
    parse_errors: list[dict[str, Any]],
    raw_entry: dict[str, Any],
    sync_entry: dict[str, Any],
) -> dict[str, Any]:
    success_count = len(rows)
    error_count = len(parse_errors)
    attempts = success_count + error_count
    parse_success_rate = round(success_count / attempts, 4) if attempts else None
    status = _normalize_source_status(raw_entry.get("status"), parse_success_rate, error_count)
    return {
        "status": status,
        "last_sync_time": (
            sync_entry.get("last_successful_sync_at")
            or sync_entry.get("last_failed_sync_at")
            or raw_entry.get("updated_at")
        ),
        "newest_filing": _newest_source_filing([*filings, *rows]),
        "parse_success_rate": parse_success_rate,
        "trade_count": success_count,
        "parse_error_count": error_count,
        "low_confidence_rows": sum(
            1 for row in rows if confidence_bucket(row.get("parser_confidence")) != "high"
        ),
        "recent_errors": _recent_source_errors(raw_entry, parse_errors),
        "remediation": _remediation_for_status(status, source),
    }


def _normalize_source_status(
    raw_status: Any,
    parse_success_rate: float | None,
    error_count: int,
) -> str:
    normalized = str(raw_status or "").strip().lower()
    if normalized in {"disabled", "offline"}:
        return normalized
    if normalized in {"error", "failed", "failure", "timeout"}:
        return "offline"
    if normalized == "degraded":
        return "degraded"
    if error_count or (parse_success_rate is not None and parse_success_rate < 0.95):
        return "degraded"
    if normalized == "ok" or parse_success_rate is not None:
        return "ok"
    return "offline"


def _newest_source_filing(rows: list[dict[str, Any]]) -> str | None:
    dates = [
        date_value
        for date_value in (
            _normalized_date_string(row.get("disclosure_date") or row.get("filed_date"))
            for row in rows
        )
        if date_value
    ]
    return max(dates) if dates else None


def _recent_source_errors(raw_entry: dict[str, Any], parse_errors: list[dict[str, Any]]) -> list[Any]:
    recent = []
    for error in raw_entry.get("errors") or []:
        recent.append(error)
    for error in parse_errors[-5:]:
        recent.append({
            "report_id": error.get("report_id"),
            "errors": error.get("errors") or [error.get("error")],
            "created_at": error.get("created_at"),
        })
    return recent[-5:]


def _remediation_for_status(status: str, source: str) -> str:
    if status == "ok":
        return f"{source} source healthy; continue normal monitoring."
    if status == "degraded":
        return f"Review recent {source} parser errors and low-confidence rows; source layout may have changed."
    if status == "offline":
        return f"Check {source} availability, credentials, rate limits, and source URL configuration."
    return f"{source} source disabled; enable only after compliance review and operator approval."


def _overall_source_status(sources: dict[str, dict[str, Any]]) -> str:
    statuses = {entry.get("status") for entry in sources.values()}
    if not statuses:
        return "offline"
    if statuses == {"disabled"}:
        return "disabled"
    if "offline" in statuses:
        return "offline" if statuses <= {"offline", "disabled"} else "degraded"
    if "degraded" in statuses:
        return "degraded"
    return "ok"


def _filter_trade_rows(
    rows: list[dict[str, Any]],
    *,
    symbol: str | None,
    filer: str | None,
    chamber: str | None,
    party: str | None,
    state: str | None,
    transaction_type: str | None,
    transaction_side: str | None,
    owner: str | None,
    from_date: str | None,
    to_date: str | None,
    tracked_only: bool,
    watchlist_only: bool,
    top_traders_only: bool,
    stock_linked_only: bool,
) -> list[dict[str, Any]]:
    from_iso = _normalized_date_string(from_date)
    to_iso = _normalized_date_string(to_date)
    return [
        row
        for row in rows
        if _matches_exact(row.get("ticker"), symbol, uppercase=True)
        and _matches_contains(row.get("filer_name"), filer)
        and _matches_exact(row.get("chamber"), chamber)
        and _matches_exact(row.get("party"), party, uppercase=True)
        and _matches_exact(row.get("state"), state, uppercase=True)
        and _matches_exact(row.get("transaction_type"), transaction_type)
        and _matches_transaction_side(row.get("transaction_type"), transaction_side)
        and _matches_exact(row.get("owner"), owner)
        and (not tracked_only or row.get("is_tracked_asset") is True)
        and (not watchlist_only or row.get("is_watchlist_asset") is True)
        and (not top_traders_only or row.get("successful_trader_rank") is not None)
        and (not stock_linked_only or _is_stock_linked_trade(row))
        and _within_disclosure_window(row, from_iso=from_iso, to_iso=to_iso)
    ]


def _is_stock_linked_trade(row: dict[str, Any]) -> bool:
    ticker = str(row.get("ticker") or "").strip()
    asset_type = str(row.get("asset_type") or "").strip().lower()
    return bool(ticker) and asset_type in {"stock", "etf", "option"}


def _matches_transaction_side(value: Any, expected: str | None) -> bool:
    if not expected:
        return True
    normalized = str(value or "").strip().lower()
    side = expected.strip().lower()
    if side == "purchase":
        return normalized in BUY_TRANSACTION_TYPES
    if side == "sale":
        return normalized in SELL_TRANSACTION_TYPES
    return True


def _attach_successful_trader_context(
    rows: list[dict[str, Any]],
    successful_traders: dict[str, Any],
) -> list[dict[str, Any]]:
    leaderboard = successful_traders.get("leaderboard", [])
    by_key = {
        str(entry.get("filer_key")): entry
        for entry in leaderboard
        if entry.get("filer_key")
    }
    selected_keys = selected_filer_keys(successful_traders)
    enriched = []
    for row in rows:
        key = canonical_filer_key(row.get("filer_name"))
        entry = by_key.get(key)
        if key in selected_keys and entry:
            enriched.append({
                **row,
                "successful_trader_rank": entry.get("rank"),
                "successful_trader_overall_rank": entry.get("overall_rank"),
                "successful_trader_score": entry.get("success_score"),
                "successful_trader_return_pct": entry.get("average_signed_return_pct"),
                "successful_trader_win_rate": entry.get("win_rate"),
                "successful_trader_required_profile": entry.get("included_by_requested_profile", False),
            })
        else:
            enriched.append(row)
    return enriched


def _matches_exact(value: Any, expected: str | None, *, uppercase: bool = False) -> bool:
    if not expected:
        return True
    left = str(value or "").strip()
    right = expected.strip()
    if uppercase:
        return left.upper() == right.upper()
    return left.lower() == right.lower()


def _matches_contains(value: Any, expected: str | None) -> bool:
    if not expected:
        return True
    return expected.strip().lower() in str(value or "").lower()


def _within_disclosure_window(row: dict[str, Any], *, from_iso: str | None, to_iso: str | None) -> bool:
    disclosure_date = _normalized_date_string(row.get("disclosure_date"))
    if from_iso and (not disclosure_date or disclosure_date < from_iso):
        return False
    if to_iso and (not disclosure_date or disclosure_date > to_iso):
        return False
    return True


def _trade_sort_key(row: dict[str, Any]) -> tuple[str, str]:
    disclosure_date = _normalized_date_string(row.get("disclosure_date")) or ""
    trade_id = str(row.get("trade_id") or row.get("row_hash") or "")
    return disclosure_date, trade_id


def _attach_official_source_url(row: dict[str, Any]) -> dict[str, Any]:
    official_source_url = (
        row.get("official_source_url")
        or row.get("document_url")
        or row.get("source_url")
        or row.get("filing_url")
    )
    return {
        **row,
        "official_source_url": official_source_url,
        "parser_confidence": row.get("parser_confidence"),
    }


def _recent_asset_rows(
    rows: list[dict[str, Any]],
    *,
    window_days: int,
    as_of: date,
) -> list[dict[str, Any]]:
    cutoff_iso = (as_of - timedelta(days=max(0, window_days))).isoformat()
    as_of_iso = as_of.isoformat()
    recent = []
    for row in rows:
        disclosure_date = _normalized_date_string(row.get("disclosure_date"))
        if disclosure_date is None or cutoff_iso <= disclosure_date <= as_of_iso:
            recent.append(row)
    return recent


def _unique_filers(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({
        str(row.get("filer_name")).strip()
        for row in rows
        if str(row.get("filer_name") or "").strip()
    })


def _compute_buy_sell_imbalance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    imbalance = {
        "buy_count": 0,
        "sell_count": 0,
        "net_count": 0,
        "buy_amount_mid_usd": 0.0,
        "sell_amount_mid_usd": 0.0,
        "net_amount_mid_usd": 0.0,
    }
    for row in rows:
        transaction_type = str(row.get("transaction_type") or "unknown").lower()
        amount_mid = _number_or_zero(row.get("amount_mid_usd"))
        if transaction_type in BUY_TRANSACTION_TYPES:
            imbalance["buy_count"] += 1
            imbalance["buy_amount_mid_usd"] += amount_mid
        elif transaction_type in SELL_TRANSACTION_TYPES:
            imbalance["sell_count"] += 1
            imbalance["sell_amount_mid_usd"] += amount_mid
    imbalance["net_count"] = imbalance["buy_count"] - imbalance["sell_count"]
    imbalance["net_amount_mid_usd"] = (
        imbalance["buy_amount_mid_usd"] - imbalance["sell_amount_mid_usd"]
    )
    return imbalance


def _compute_amount_estimates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_mid = 0.0
    total_min = 0.0
    total_max = 0.0
    estimated_rows = 0
    unknown_rows = 0
    for row in rows:
        amount_mid = row.get("amount_mid_usd")
        amount_min = row.get("amount_min_usd")
        amount_max = row.get("amount_max_usd")
        if amount_mid is None and amount_min is None and amount_max is None:
            unknown_rows += 1
            continue
        total_mid += _number_or_zero(amount_mid)
        total_min += _number_or_zero(amount_min)
        total_max += _number_or_zero(amount_max)
        if row.get("amount_mid_usd_is_estimate") is True:
            estimated_rows += 1
    return {
        "amount_mid_usd": total_mid,
        "amount_min_usd": total_min,
        "amount_max_usd": total_max,
        "estimated_row_count": estimated_rows,
        "unknown_amount_row_count": unknown_rows,
    }


def _build_disclosure_timeline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: dict[str, dict[str, Any]] = {}
    for row in rows:
        disclosure_date = _normalized_date_string(row.get("disclosure_date"))
        if disclosure_date is None:
            continue
        bucket = timeline.setdefault(disclosure_date, {
            "date": disclosure_date,
            "trade_count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "net_amount_mid_usd": 0.0,
        })
        bucket["trade_count"] += 1
        transaction_type = str(row.get("transaction_type") or "unknown").lower()
        amount_mid = _number_or_zero(row.get("amount_mid_usd"))
        if transaction_type in BUY_TRANSACTION_TYPES:
            bucket["buy_count"] += 1
            bucket["net_amount_mid_usd"] += amount_mid
        elif transaction_type in SELL_TRANSACTION_TYPES:
            bucket["sell_count"] += 1
            bucket["net_amount_mid_usd"] -= amount_mid
    return [timeline[key] for key in sorted(timeline)]


def _known_asset_limitations(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    if any(
        row.get("ticker_resolution_status") == "ambiguous"
        or row.get("ticker_ambiguous") is True
        or "ticker_ambiguous" in (row.get("flags") or [])
        for row in rows
    ):
        return [{
            "code": "ticker_resolution_ambiguous",
            "message": (
                "One or more rows were matched from ambiguous disclosure text; "
                "verify the official source before drawing conclusions."
            ),
        }]
    return []


def _matches_filer_identifier(row: dict[str, Any], filer_id: str) -> bool:
    target = str(filer_id or "").strip().lower()
    if not target:
        return False
    candidates = {
        str(row.get("filer_id") or "").strip().lower(),
        str(row.get("filer_name") or "").strip().lower(),
        _slugify_filer(row.get("filer_name")),
    }
    return target in candidates


def _slugify_filer(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return normalized


def _build_filer_metadata(filer_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0] if rows else {}
    name = first.get("filer_name")
    normalized_id = first.get("filer_id") or _slugify_filer(name) or str(filer_id)
    metadata = {
        "filer_id": str(normalized_id),
        "filer_name": str(name) if name else None,
        "chamber": first.get("chamber"),
        "party": first.get("party"),
        "state": first.get("state"),
        "source": first.get("source"),
        "committee_enrichment": _committee_enrichment(rows),
        "committee_data_source": "enrichment_not_source_disclosure",
        "metadata_complete": bool(name and first.get("chamber") and first.get("state")),
    }
    return metadata


FILER_TRADE_PUBLIC_FIELDS = {
    "trade_id",
    "ticker",
    "asset_name",
    "asset_type",
    "sector",
    "asset_sector",
    "transaction_type",
    "owner",
    "amount_min_usd",
    "amount_max_usd",
    "amount_mid_usd",
    "amount_mid_usd_is_estimate",
    "transaction_date",
    "disclosure_date",
    "delay_days",
    "chamber",
    "party",
    "state",
    "source",
    "report_id",
    "document_url",
    "official_source_url",
    "flags",
    "parser_confidence",
    "confidence_status",
    "is_tracked_asset",
    "is_watchlist_asset",
    "committee",
    "committees",
}


def _sanitize_filer_trade(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key in FILER_TRADE_PUBLIC_FIELDS
    }


def _committee_enrichment(rows: list[dict[str, Any]]) -> list[str]:
    committees: set[str] = set()
    for row in rows:
        raw = row.get("committees")
        if isinstance(raw, list):
            committees.update(str(item).strip() for item in raw if str(item).strip())
        if row.get("committee"):
            committees.add(str(row["committee"]).strip())
    return sorted(committees)


def _top_tickers(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("ticker") or "UNKNOWN").upper()
        bucket = buckets.setdefault(ticker, {
            "ticker": ticker,
            "trade_count": 0,
            "amount_mid_usd": 0.0,
        })
        bucket["trade_count"] += 1
        bucket["amount_mid_usd"] += _number_or_zero(row.get("amount_mid_usd"))
    return sorted(
        buckets.values(),
        key=lambda item: (item["amount_mid_usd"], item["trade_count"], item["ticker"]),
        reverse=True,
    )[:limit]


def _top_sectors(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        sector = str(row.get("sector") or row.get("asset_sector") or "Unknown")
        bucket = buckets.setdefault(sector, {
            "sector": sector,
            "trade_count": 0,
            "amount_mid_usd": 0.0,
        })
        bucket["trade_count"] += 1
        bucket["amount_mid_usd"] += _number_or_zero(row.get("amount_mid_usd"))
    return sorted(
        buckets.values(),
        key=lambda item: (item["amount_mid_usd"], item["trade_count"], item["sector"]),
        reverse=True,
    )[:limit]


def _delay_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    delays = sorted(
        float(row["delay_days"])
        for row in rows
        if isinstance(row.get("delay_days"), (int, float))
    )
    if not delays:
        return {
            "count": 0,
            "average_days": None,
            "median_days": None,
            "max_days": None,
            "late_filing_count": 0,
        }
    midpoint = len(delays) // 2
    median = delays[midpoint] if len(delays) % 2 else (delays[midpoint - 1] + delays[midpoint]) / 2
    return {
        "count": len(delays),
        "average_days": round(sum(delays) / len(delays), 2),
        "median_days": round(median, 2),
        "max_days": max(delays),
        "late_filing_count": sum(1 for delay in delays if delay > 45),
    }


def _ownership_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    breakdown = {owner: 0 for owner in OWNER_BUCKETS}
    for row in rows:
        owner = str(row.get("owner") or "unknown").lower().replace("-", "_").replace(" ", "_")
        if owner not in breakdown:
            owner = "unknown"
        breakdown[owner] += 1
    return breakdown


def _source_documents(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for row in rows:
        url = row.get("official_source_url")
        if not url:
            continue
        key = str(url)
        documents.setdefault(key, {
            "official_source_url": key,
            "source": row.get("source"),
            "report_id": row.get("report_id"),
            "disclosure_date": row.get("disclosure_date"),
        })
    return sorted(
        documents.values(),
        key=lambda item: str(item.get("disclosure_date") or ""),
        reverse=True,
    )


def _compute_summary_metrics(
    rows: list[dict[str, Any]],
    *,
    tracked_symbols: set[str],
    watchlist_symbols: set[str],
    source_health: dict[str, Any],
    as_of_date: str | None,
) -> dict[str, Any]:
    as_of = _parse_date(as_of_date) or date.today()
    as_of_iso = as_of.isoformat()
    cutoff_iso = (as_of - timedelta(days=7)).isoformat()
    total_trades = len(rows)
    new_disclosures = 0
    new_tracked_asset_disclosures = 0
    new_watchlist_disclosures = 0
    tracked_asset_trades = 0
    watchlist_trades = 0
    late_filings = 0
    newest_disclosure_date: str | None = None
    by_chamber: dict[str, dict[str, Any]] = {}

    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if ticker and ticker in tracked_symbols:
            tracked_asset_trades += 1
        if ticker and ticker in watchlist_symbols:
            watchlist_trades += 1

        disclosure_date = _normalized_date_string(row.get("disclosure_date"))
        if disclosure_date:
            if cutoff_iso <= disclosure_date <= as_of_iso:
                new_disclosures += 1
                if ticker and ticker in tracked_symbols:
                    new_tracked_asset_disclosures += 1
                if ticker and ticker in watchlist_symbols:
                    new_watchlist_disclosures += 1
            if newest_disclosure_date is None or disclosure_date > newest_disclosure_date:
                newest_disclosure_date = disclosure_date

        flags = row.get("flags") if isinstance(row.get("flags"), list) else []
        delay_days = row.get("delay_days")
        if "late_disclosure" in flags or (
            isinstance(delay_days, (int, float)) and delay_days > 45
        ):
            late_filings += 1

        chamber = str(row.get("chamber") or "unknown").strip().lower() or "unknown"
        bucket = by_chamber.setdefault(chamber, _empty_chamber_summary())
        bucket["trade_count"] += 1
        transaction_type = str(row.get("transaction_type") or "unknown").lower()
        amount_mid = _number_or_zero(row.get("amount_mid_usd"))
        if transaction_type in BUY_TRANSACTION_TYPES:
            bucket["buy_count"] += 1
            bucket["buy_amount_mid_usd"] += amount_mid
        elif transaction_type in SELL_TRANSACTION_TYPES:
            bucket["sell_count"] += 1
            bucket["sell_amount_mid_usd"] += amount_mid

    for bucket in by_chamber.values():
        bucket["net_buy_amount_mid_usd"] = (
            bucket["buy_amount_mid_usd"] - bucket["sell_amount_mid_usd"]
        )

    return {
        "total_trades": total_trades,
        "new_disclosures_7d": new_disclosures,
        "new_disclosures_last_7_days": new_disclosures,
        "new_tracked_asset_disclosures_7d": new_tracked_asset_disclosures,
        "new_watchlist_disclosures_7d": new_watchlist_disclosures,
        "tracked_asset_trades": tracked_asset_trades,
        "watchlist_trades": watchlist_trades,
        "late_filings": late_filings,
        "newest_disclosure_date": newest_disclosure_date,
        "source_health": source_health,
        "by_chamber": by_chamber,
    }


def _empty_chamber_summary() -> dict[str, Any]:
    return {
        "trade_count": 0,
        "buy_count": 0,
        "sell_count": 0,
        "buy_amount_mid_usd": 0.0,
        "sell_amount_mid_usd": 0.0,
        "net_buy_amount_mid_usd": 0.0,
    }


def _normalized_date_string(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    candidate = value[:10]
    if _parse_date(candidate) is None:
        return None
    return candidate


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _number_or_zero(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _missing_data_response(endpoint: str) -> Dict[str, Any]:
    availability = get_feature_availability()
    return {
        "feature": "politicians",
        **availability,
        "status": "missing_data",
        **_response_meta(),
        "endpoint": endpoint,
        "message": "Politician normalized trade data is missing. Run make politicians-parse after source sync.",
        "data_use_notice": get_data_use_notice(),
    }


def _response_meta() -> Dict[str, Any]:
    data_root = get_politicians_data_dir()
    candidates = (
        data_root / "trades.jsonl",
        data_root / "filings.jsonl",
        data_root / "source_health.json",
        data_root / "sync_state.json",
    )
    mtimes = [path.stat().st_mtime for path in candidates if path.exists()]
    newest = max(mtimes) if mtimes else None
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_age_seconds": round(time.time() - newest, 1) if newest else None,
        "schema_version": POLITICIANS_RESPONSE_SCHEMA_VERSION,
        "cache_schema_version": POLITICIANS_CACHE_SCHEMA_VERSION,
    }


def _summary_cache_key(
    *,
    as_of_date: str | None,
    tracked_symbols: set[str],
    watchlist_symbols: set[str],
    data_root,
) -> tuple[Any, ...]:
    root = data_root
    return (
        POLITICIANS_RESPONSE_SCHEMA_VERSION,
        as_of_date,
        _file_signature(root / "trades.jsonl"),
        _file_signature(root / "source_health.json"),
        _file_signature(root.parent / "watchlist.json"),
        tuple(sorted(tracked_symbols)),
        tuple(sorted(watchlist_symbols)),
    )


def _file_signature(path) -> tuple[int | None, int | None]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (None, None)
    return (stat.st_mtime_ns, stat.st_size)


def _load_tracked_symbols() -> set[str]:
    data_root = get_politicians_data_dir()
    prices_dir = data_root.parent / "prices"
    if not prices_dir.exists():
        return set()
    return {
        path.name[:-7].upper()
        for path in prices_dir.glob("*_1d.csv")
        if path.name.endswith("_1d.csv")
    }


def _load_watchlist_symbols() -> set[str]:
    data_root = get_politicians_data_dir()
    path = data_root.parent / "watchlist.json"
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    raw = payload.get("symbols", []) if isinstance(payload, dict) else payload
    return {str(symbol).upper() for symbol in raw if isinstance(symbol, str)}
