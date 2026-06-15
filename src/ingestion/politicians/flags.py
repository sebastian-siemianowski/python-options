"""Anomaly and review flags for politician disclosure rows."""

from __future__ import annotations

from typing import Any


LATE_DISCLOSURE_DAYS = 45
LARGE_TRADE_MIN_USD = 1_000_001


def compute_trade_flags(row: dict[str, Any]) -> list[str]:
    """Compute review flags for a normalized politician trade row."""
    flags: list[str] = []
    delay = row.get("delay_days")
    if isinstance(delay, (int, float)) and delay > LATE_DISCLOSURE_DAYS:
        flags.append("late_disclosure")
    amount_min = row.get("amount_min_usd")
    amount_max = row.get("amount_max_usd")
    if (
        isinstance(amount_min, (int, float)) and amount_min >= LARGE_TRADE_MIN_USD
    ) or (
        amount_max is None and isinstance(amount_min, (int, float)) and amount_min > 0
    ):
        flags.append("large_trade_bucket")
    if row.get("ticker_resolution_status") == "ambiguous" or row.get("ticker_ambiguous") is True:
        flags.append("ticker_ambiguous")
    if row.get("is_amendment") is True or row.get("amends_report_id"):
        flags.append("amended")
    return flags


def apply_trade_flags(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of row with computed flags merged into existing flags."""
    existing = list(row.get("flags", []))
    merged = sorted(set([*existing, *compute_trade_flags(row)]))
    return {**row, "flags": merged}


def filter_rows_by_flag(rows: list[dict[str, Any]], flag: str | None) -> list[dict[str, Any]]:
    """Filter rows by computed or stored flag."""
    flagged = [apply_trade_flags(row) for row in rows]
    if not flag:
        return flagged
    return [row for row in flagged if flag in row.get("flags", [])]
