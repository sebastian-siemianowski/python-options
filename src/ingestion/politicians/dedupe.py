"""Deterministic identity, deduplication, and amendment helpers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any


VOLATILE_FIELDS = {
    "created_at",
    "updated_at",
    "downloaded_at",
    "parsed_at",
    "trade_id",
    "row_hash",
}


def compute_row_hash(row: dict[str, Any]) -> str:
    """Return a deterministic hash for a normalized trade row."""
    canonical = _canonicalize(row)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def compute_trade_id(row: dict[str, Any]) -> str:
    """Return a deterministic trade_id using source/year/report/hash context."""
    source = _slug(row.get("source", "unknown"))
    year = str(row.get("filing_year") or row.get("year") or "unknown")
    report_id = _slug(row.get("report_id", "unknown"))
    row_hash = row.get("row_hash") or compute_row_hash(row)
    return f"{source}:{year}:{report_id}:{str(row_hash).split(':')[-1][:16]}"


def attach_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of row with row_hash and trade_id attached."""
    enriched = dict(row)
    enriched["row_hash"] = compute_row_hash(enriched)
    enriched["trade_id"] = compute_trade_id(enriched)
    return enriched


def deduplicate_trades(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach identity and remove exact duplicate normalized rows."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        enriched = attach_identity(row)
        if enriched["row_hash"] in seen:
            continue
        seen.add(enriched["row_hash"])
        unique.append(enriched)
    return unique


def build_amendment_history(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group original and amendment rows by amended report ID for detail views."""
    identified = [attach_identity(row) if "row_hash" not in row else dict(row) for row in rows]
    by_report: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in identified:
        report_id = str(row.get("report_id", ""))
        if report_id:
            by_report[report_id].append(row)
    history: dict[str, list[dict[str, Any]]] = {}
    for row in identified:
        amended = row.get("amends_report_id")
        if row.get("is_amendment") and amended:
            key = str(amended)
            history.setdefault(key, [])
            history[key].extend(by_report.get(key, []))
            history[key].append(row)
    return history


def effective_trades_for_aggregation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deduplicated rows with amended original reports superseded."""
    unique = deduplicate_trades(rows)
    amended_report_ids = {
        str(row["amends_report_id"])
        for row in unique
        if row.get("is_amendment") and row.get("amends_report_id")
    }
    return [
        row
        for row in unique
        if str(row.get("report_id")) not in amended_report_ids or row.get("is_amendment")
    ]


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _canonicalize(val)
            for key, val in sorted(value.items())
            if key not in VOLATILE_FIELDS
        }
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def _slug(value: Any) -> str:
    text = str(value).strip().lower()
    safe = "".join(ch if ch.isalnum() else "-" for ch in text).strip("-")
    return safe or "unknown"
