"""Tests for politician trade deduplication and amendments."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.dedupe import (
    attach_identity,
    build_amendment_history,
    compute_row_hash,
    compute_trade_id,
    deduplicate_trades,
    effective_trades_for_aggregation,
)


def _trade(report_id="R1", asset="NVDA", row_number=1, **extra):
    row = {
        "source": "house",
        "filing_year": 2026,
        "report_id": report_id,
        "row_number": row_number,
        "filer_name": "Jane Doe",
        "ticker": asset,
        "transaction_type": "purchase",
        "amount_bucket_raw": "$1,001 - $15,000",
        "transaction_date": "2026-04-12",
        "is_amendment": False,
        "amends_report_id": None,
    }
    row.update(extra)
    return row


def test_every_row_gets_deterministic_trade_id_and_row_hash():
    row = _trade()
    first = attach_identity(row)
    second = attach_identity(dict(row))

    assert first["row_hash"].startswith("sha256:")
    assert first["trade_id"] == second["trade_id"]
    assert first["row_hash"] == second["row_hash"]
    assert compute_trade_id(first) == first["trade_id"]


def test_row_hash_ignores_volatile_metadata():
    row = _trade()
    changed_metadata = {**row, "updated_at": "later", "created_at": "now"}

    assert compute_row_hash(row) == compute_row_hash(changed_metadata)


def test_exact_duplicate_rows_from_duplicate_pdfs_are_stored_once():
    rows = [_trade(report_id="R1"), _trade(report_id="R1")]

    unique = deduplicate_trades(rows)

    assert len(unique) == 1


def test_repeated_rows_within_single_filing_are_deduplicated():
    rows = [_trade(report_id="R1", row_number=1), _trade(report_id="R1", row_number=1)]

    unique = deduplicate_trades(rows)

    assert len(unique) == 1


def test_amendments_are_linked_and_effective_aggregations_use_latest_rows():
    original = _trade(report_id="R1", asset="NVDA")
    amendment = _trade(
        report_id="R2",
        asset="AAPL",
        is_amendment=True,
        amends_report_id="R1",
    )

    history = build_amendment_history([original, amendment])
    effective = effective_trades_for_aggregation([original, amendment])

    assert "R1" in history
    assert [row["report_id"] for row in history["R1"]] == ["R1", "R2"]
    assert [row["report_id"] for row in effective] == ["R2"]
