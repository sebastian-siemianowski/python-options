"""Tests for politician parser confidence scoring."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.quality import (
    confidence_bucket,
    score_parser_confidence,
    summarize_confidence_buckets,
    validation_status_for_confidence,
)
from web.backend.services.politicians_service import get_politicians_source_health_response


def test_parser_confidence_combines_all_required_components():
    full = score_parser_confidence(
        field_completeness=1.0,
        table_alignment=1.0,
        date_validity=1.0,
        amount_bucket_recognition=1.0,
        ticker_resolution_confidence=1.0,
    )
    weak_ticker = score_parser_confidence(
        field_completeness=1.0,
        table_alignment=1.0,
        date_validity=1.0,
        amount_bucket_recognition=1.0,
        ticker_resolution_confidence=0.5,
    )

    assert full == 1.0
    assert weak_ticker == 0.93


def test_confidence_bucket_and_validation_status_thresholds():
    assert confidence_bucket(0.96) == "high"
    assert validation_status_for_confidence(0.96) == "valid"
    assert confidence_bucket(0.80) == "warning"
    assert validation_status_for_confidence(0.80) == "valid_with_warnings"
    assert confidence_bucket(0.79) == "quarantined"
    assert validation_status_for_confidence(0.79) == "quarantined"


def test_source_health_response_reports_confidence_buckets(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    (data_root / "trades.jsonl").write_text(
        json.dumps({"parser_confidence": 0.97}) + "\n" +
        json.dumps({"parser_confidence": 0.85}) + "\n" +
        json.dumps({"parser_confidence": 0.40}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_source_health_response()

    assert response["status"] == "ok"
    assert response["confidence_buckets"] == {"high": 1, "warning": 1, "quarantined": 1}


def test_summarize_confidence_buckets_handles_missing_values():
    summary = summarize_confidence_buckets([
        {"parser_confidence": 0.95},
        {"parser_confidence": 0.80},
        {},
    ])

    assert summary == {"high": 1, "warning": 1, "quarantined": 1}
