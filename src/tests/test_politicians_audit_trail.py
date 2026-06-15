"""Tests for immutable politician source audit trail."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.audit import attach_audit_fields, compute_source_hash, write_filing_audit_record
from ingestion.politicians.parsers.house_pdf import PARSER_VERSION, parse_house_ptr_pdf
from ingestion.politicians.validation import validate_trade_record


def _valid_row():
    return {
        "source": "house",
        "report_id": "R1",
        "filing_year": 2026,
        "filer_name": "Jane Doe",
        "asset_name_raw": "NVIDIA Corporation",
        "ticker": "NVDA",
        "asset_type": "stock",
        "transaction_type": "purchase",
        "amount_bucket_raw": "$1,001 - $15,000",
    }


def test_audit_fields_attach_source_hash_row_hash_parser_version_and_paths(tmp_path):
    raw = tmp_path / "R1.pdf"
    raw.write_text("source bytes", encoding="utf-8")

    row = attach_audit_fields(
        _valid_row(),
        raw_artifact_path=raw,
        document_url="https://example.test/R1.pdf",
        parser_version="parser-v1",
    )

    assert row["source_hash"] == compute_source_hash(raw)
    assert row["row_hash"].startswith("sha256:")
    assert row["trade_id"].startswith("house:2026:r1:")
    assert row["parser_version"] == "parser-v1"
    assert row["document_url"] == "https://example.test/R1.pdf"
    assert row["raw_artifact_path"] == str(raw)
    assert validate_trade_record(row)[0] is True


def test_reparsing_same_raw_artifact_same_parser_version_has_identical_hash(tmp_path):
    raw = tmp_path / "R1.pdf"
    raw.write_text(
        "Owner | Asset | Transaction Type | Transaction Date | Notification Date | Amount\n"
        "Self | NVIDIA Corporation | P | 04/12/2026 | 04/18/2026 | $1,001 - $15,000\n",
        encoding="utf-8",
    )

    parsed_a = parse_house_ptr_pdf(raw, report_id="R1", filing_year=2026, filer_name="Jane Doe")
    parsed_b = parse_house_ptr_pdf(raw, report_id="R1", filing_year=2026, filer_name="Jane Doe")
    row_a = attach_audit_fields(
        {**_valid_row(), **parsed_a["rows"][0], "asset_type": "stock", "ticker": "NVDA", "source": "house"},
        raw_artifact_path=raw,
        document_url="https://example.test/R1.pdf",
        parser_version=PARSER_VERSION,
    )
    row_b = attach_audit_fields(
        {**_valid_row(), **parsed_b["rows"][0], "asset_type": "stock", "ticker": "NVDA", "source": "house"},
        raw_artifact_path=raw,
        document_url="https://example.test/R1.pdf",
        parser_version=PARSER_VERSION,
    )

    assert row_a["row_hash"] == row_b["row_hash"]


def test_parser_version_changes_are_recorded_in_filings_jsonl(tmp_path):
    raw = tmp_path / "R1.pdf"
    raw.write_text("source bytes", encoding="utf-8")
    data_root = tmp_path / "politicians"

    write_filing_audit_record(
        report_id="R1",
        source="house",
        parser_version="parser-v1",
        raw_artifact_path=raw,
        document_url="https://example.test/R1.pdf",
        data_root=data_root,
    )
    write_filing_audit_record(
        report_id="R1",
        source="house",
        parser_version="parser-v2",
        raw_artifact_path=raw,
        document_url="https://example.test/R1.pdf",
        data_root=data_root,
    )
    rows = [json.loads(line) for line in (data_root / "filings.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 1
    assert rows[0]["parser_version"] == "parser-v2"
    assert rows[0]["parser_version_history"] == ["parser-v1", "parser-v2"]
