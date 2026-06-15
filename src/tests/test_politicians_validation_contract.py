"""Tests for politician trade validation contract."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.status import get_politicians_status
from ingestion.politicians.validation import validate_trade_record, write_validated_trades


def _valid_trade(**extra):
    row = {
        "source": "house",
        "report_id": "R1",
        "filer_name": "Jane Doe",
        "asset_name_raw": "NVIDIA Corporation",
        "ticker": "NVDA",
        "asset_type": "stock",
        "transaction_type": "purchase",
        "amount_bucket_raw": "$1,001 - $15,000",
        "document_url": "https://example.test/R1.pdf",
        "source_hash": "sha256:abc",
        "raw_artifact_path": "raw/house/2026/R1.pdf",
        "parser_version": "parser-v1",
    }
    row.update(extra)
    return row


def test_required_fields_are_validated_before_writing():
    ok, errors = validate_trade_record(_valid_trade(asset_name_raw=None))

    assert ok is False
    assert "missing_required:asset_name_raw" in errors


def test_invalid_enum_values_fail_validation():
    ok, errors = validate_trade_record(_valid_trade(asset_type="spaceship", transaction_type="teleport"))

    assert ok is False
    assert "invalid_enum:asset_type:spaceship" in errors
    assert "invalid_enum:transaction_type:teleport" in errors


def test_missing_document_url_fails_except_manual_fixture():
    ok, errors = validate_trade_record(_valid_trade(document_url=None))
    fixture_ok, fixture_errors = validate_trade_record(_valid_trade(source="manual_fixture", document_url=None))

    assert ok is False
    assert "missing_required:document_url" in errors
    assert fixture_ok is True
    assert fixture_errors == []


def test_missing_ticker_allowed_only_for_non_public_equity_assets():
    stock_ok, stock_errors = validate_trade_record(_valid_trade(ticker=None, asset_type="stock"))
    bond_ok, bond_errors = validate_trade_record(_valid_trade(ticker=None, asset_type="bond"))

    assert stock_ok is False
    assert "missing_required:ticker_for_public_asset" in stock_errors
    assert bond_ok is True
    assert bond_errors == []


def test_validation_failures_written_to_parse_errors_with_context(tmp_path):
    result = write_validated_trades([
        _valid_trade(),
        _valid_trade(report_id="R2", ticker=None, asset_type="stock"),
    ], data_root=tmp_path / "politicians")
    errors_path = tmp_path / "politicians" / "parse_errors.jsonl"
    errors = [json.loads(line) for line in errors_path.read_text(encoding="utf-8").splitlines()]

    assert result["status"] == "valid_with_errors"
    assert result["valid_count"] == 1
    assert result["error_count"] == 1
    assert errors[0]["report_id"] == "R2"
    assert "row_context" in errors[0]
    assert "missing_required:ticker_for_public_asset" in errors[0]["errors"]


def test_status_includes_validation_summary(tmp_path):
    data_root = tmp_path / "politicians"
    write_validated_trades([
        _valid_trade(report_id="R1"),
        _valid_trade(report_id="R2", ticker=None, asset_type="stock"),
    ], data_root=data_root)

    status = get_politicians_status(data_root)

    assert status["trade_count"] == 1
    assert status["parse_error_count"] == 1
    assert status["validation_summary"]["missing_required:ticker_for_public_asset"] == 1
