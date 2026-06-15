"""Parser-confidence scoring for politician disclosure rows."""

from __future__ import annotations

from collections import Counter
from typing import Any


HIGH_CONFIDENCE_THRESHOLD = 0.95
WARNING_CONFIDENCE_THRESHOLD = 0.80


def score_parser_confidence(
    *,
    field_completeness: float,
    table_alignment: float,
    date_validity: float,
    amount_bucket_recognition: float,
    ticker_resolution_confidence: float,
) -> float:
    """Combine parser quality signals into one bounded confidence score."""
    weighted = (
        0.35 * _clip(field_completeness)
        + 0.20 * _clip(table_alignment)
        + 0.15 * _clip(date_validity)
        + 0.15 * _clip(amount_bucket_recognition)
        + 0.15 * _clip(ticker_resolution_confidence)
    )
    return round(weighted, 2)


def confidence_bucket(confidence: float | None) -> str:
    """Return high, warning, or quarantined for a confidence value."""
    if confidence is None:
        return "quarantined"
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        return "high"
    if confidence >= WARNING_CONFIDENCE_THRESHOLD:
        return "warning"
    return "quarantined"


def validation_status_for_confidence(confidence: float | None) -> str:
    """Map confidence to row validation status."""
    bucket = confidence_bucket(confidence)
    if bucket == "high":
        return "valid"
    if bucket == "warning":
        return "valid_with_warnings"
    return "quarantined"


def summarize_confidence_buckets(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count rows by confidence bucket."""
    counts = Counter(confidence_bucket(row.get("parser_confidence")) for row in rows)
    return {
        "high": counts.get("high", 0),
        "warning": counts.get("warning", 0),
        "quarantined": counts.get("quarantined", 0),
    }


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
