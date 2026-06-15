"""Normalization helpers for politician disclosure records."""

from __future__ import annotations

from datetime import date
import re
from typing import Any


AMOUNT_ESTIMATE_LABEL = "Estimated midpoint from public disclosure amount bucket"


def normalize_amount_bucket(raw_bucket: str | None) -> dict[str, Any]:
    """Normalize an official disclosure amount bucket into numeric bounds."""
    raw = (raw_bucket or "").strip()
    result: dict[str, Any] = {
        "amount_bucket_raw": raw or None,
        "amount_min_usd": None,
        "amount_max_usd": None,
        "amount_mid_usd": None,
        "amount_mid_usd_is_estimate": True,
        "amount_estimate_label": AMOUNT_ESTIMATE_LABEL,
        "amount_bucket_parse_status": "missing" if not raw else "unknown",
    }
    if not raw:
        return result

    values = [_parse_money(match) for match in re.findall(r"\$?\s*[\d,]+", raw)]
    lowered = raw.lower()
    if "over" in lowered and values:
        minimum = values[0] + 1
        result.update({
            "amount_min_usd": minimum,
            "amount_max_usd": None,
            "amount_mid_usd": None,
            "amount_bucket_parse_status": "open_ended",
        })
        return result

    if len(values) >= 2:
        minimum, maximum = values[0], values[1]
        result.update({
            "amount_min_usd": minimum,
            "amount_max_usd": maximum,
            "amount_mid_usd": (minimum + maximum) / 2,
            "amount_bucket_parse_status": "range",
        })
        return result

    if len(values) == 1 and any(marker in lowered for marker in ("less", "under", "or less")):
        maximum = values[0]
        result.update({
            "amount_min_usd": 0,
            "amount_max_usd": maximum,
            "amount_mid_usd": maximum / 2,
            "amount_bucket_parse_status": "range",
        })
        return result

    return result


def normalize_date_fields(
    *,
    transaction_date: str | None = None,
    notification_date: str | None = None,
    filed_date: str | None = None,
    disclosure_date: str | None = None,
) -> dict[str, Any]:
    """Normalize politician disclosure date fields without silent inference."""
    tx = _normalize_date(transaction_date)
    notification = _normalize_date(notification_date)
    filed = _normalize_date(filed_date)
    disclosure = _normalize_date(disclosure_date)
    warnings: list[str] = []
    delay_days = None
    if tx and disclosure:
        delay_days = (date.fromisoformat(disclosure) - date.fromisoformat(tx)).days
        if delay_days < 0:
            warnings.append("disclosure_before_transaction")
    if notification and tx and date.fromisoformat(notification) < date.fromisoformat(tx):
        warnings.append("notification_before_transaction")
    return {
        "transaction_date": tx,
        "notification_date": notification,
        "filed_date": filed,
        "disclosure_date": disclosure,
        "delay_days": delay_days,
        "date_warnings": warnings,
        "validation_status": "valid_with_warnings" if warnings else "valid",
    }


def get_event_study_anchor_date(record: dict[str, Any], anchor: str = "disclosure_date") -> str | None:
    """Return the event-study anchor date, defaulting to disclosure_date."""
    if anchor != "disclosure_date":
        raise ValueError("Politician event studies must default to disclosure_date to avoid lookahead leakage.")
    value = record.get(anchor)
    return str(value) if value else None


def _parse_money(value: str) -> int:
    return int(re.sub(r"[^\d]", "", value))


def _normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"unknown", "n/a", "none", "missing"}:
        return None
    match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if match:
        y, m, d = match.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    match = re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", text)
    if match:
        m, d, y = match.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return None
