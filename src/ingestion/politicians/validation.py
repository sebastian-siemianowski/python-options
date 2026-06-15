"""Validation contract for normalized politician trade rows."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ingestion.politicians.dedupe import attach_identity, deduplicate_trades
from ingestion.politicians.paths import ensure_politicians_data_dirs
from ingestion.politicians.source_health import utc_now_iso


REQUIRED_FIELDS = (
    "source",
    "report_id",
    "filer_name",
    "asset_name_raw",
    "asset_type",
    "transaction_type",
    "amount_bucket_raw",
    "source_hash",
    "raw_artifact_path",
    "parser_version",
)

VALID_SOURCES = {"house", "senate", "manual_fixture", "third_party_validation"}
VALID_ASSET_TYPES = {"stock", "etf", "option", "bond", "fund", "crypto", "commodity", "private_asset", "unknown"}
VALID_TRANSACTION_TYPES = {"purchase", "sale", "sale_partial", "exchange", "received", "other", "unknown"}
PUBLIC_TICKER_REQUIRED_ASSET_TYPES = {"stock", "etf"}


def validate_trade_record(row: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate one normalized trade row."""
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if row.get(field) in (None, ""):
            errors.append(f"missing_required:{field}")
    if row.get("source") not in VALID_SOURCES:
        errors.append(f"invalid_enum:source:{row.get('source')}")
    if row.get("asset_type") not in VALID_ASSET_TYPES:
        errors.append(f"invalid_enum:asset_type:{row.get('asset_type')}")
    if row.get("transaction_type") not in VALID_TRANSACTION_TYPES:
        errors.append(f"invalid_enum:transaction_type:{row.get('transaction_type')}")
    if row.get("source") != "manual_fixture" and not row.get("document_url"):
        errors.append("missing_required:document_url")
    if row.get("asset_type") in PUBLIC_TICKER_REQUIRED_ASSET_TYPES and not row.get("ticker"):
        errors.append("missing_required:ticker_for_public_asset")
    return not errors, errors


def write_validated_trades(
    rows: list[dict[str, Any]],
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate, dedupe, and write normalized trade rows plus parse errors."""
    root = ensure_politicians_data_dirs(data_root)
    trades_path = root / "trades.jsonl"
    errors_path = root / "parse_errors.jsonl"
    valid_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        ok, errors = validate_trade_record(row)
        if ok:
            valid_rows.append(attach_identity(row))
        else:
            error_rows.append({
                "source": row.get("source"),
                "report_id": row.get("report_id"),
                "row_index": idx,
                "errors": errors,
                "row_context": row,
                "created_at": utc_now_iso(),
            })

    unique_rows = deduplicate_trades(valid_rows)
    with trades_path.open("w", encoding="utf-8") as handle:
        for row in unique_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with errors_path.open("w", encoding="utf-8") as handle:
        for row in error_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "status": "ok" if not error_rows else "valid_with_errors",
        "input_count": len(rows),
        "valid_count": len(unique_rows),
        "error_count": len(error_rows),
        "trades_path": str(trades_path),
        "parse_errors_path": str(errors_path),
        "error_summary": summarize_validation_errors(error_rows),
    }


def summarize_validation_errors(error_rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count validation errors by code."""
    counts: Counter[str] = Counter()
    for row in error_rows:
        for error in row.get("errors", []):
            counts[str(error)] += 1
    return dict(sorted(counts.items()))
