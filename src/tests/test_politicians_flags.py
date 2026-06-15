"""Tests for politician trade anomaly flags."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.flags import apply_trade_flags, compute_trade_flags, filter_rows_by_flag
from web.backend.services.politicians_service import get_politicians_trades_response


def test_late_disclosure_flag():
    assert "late_disclosure" in compute_trade_flags({"delay_days": 46})
    assert "late_disclosure" not in compute_trade_flags({"delay_days": 45})


def test_large_trade_bucket_flag():
    assert "large_trade_bucket" in compute_trade_flags({"amount_min_usd": 1_000_001, "amount_max_usd": 5_000_000})
    assert "large_trade_bucket" in compute_trade_flags({"amount_min_usd": 50_000_001, "amount_max_usd": None})
    assert "large_trade_bucket" not in compute_trade_flags({"amount_min_usd": 1_001, "amount_max_usd": 15_000})


def test_ticker_ambiguous_and_amended_flags():
    row = {
        "ticker_resolution_status": "ambiguous",
        "is_amendment": True,
        "amends_report_id": "R1",
    }

    flags = compute_trade_flags(row)

    assert "ticker_ambiguous" in flags
    assert "amended" in flags


def test_flags_merge_with_existing_flags_and_filter_rows():
    rows = [
        {"trade_id": "1", "delay_days": 46, "flags": ["manual_review"]},
        {"trade_id": "2", "delay_days": 2},
    ]

    flagged = apply_trade_flags(rows[0])
    filtered = filter_rows_by_flag(rows, "late_disclosure")

    assert flagged["flags"] == ["late_disclosure", "manual_review"]
    assert [row["trade_id"] for row in filtered] == ["1"]


def test_backend_trade_response_filters_by_flag(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    (data_root / "trades.jsonl").write_text(
        json.dumps({"trade_id": "1", "delay_days": 46}) + "\n" +
        json.dumps({"trade_id": "2", "delay_days": 2}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_trades_response(flag="late_disclosure")

    assert response["status"] == "ok"
    assert response["filter"]["flag"] == "late_disclosure"
    assert response["total"] == 1
    assert response["trades"][0]["trade_id"] == "1"
    assert "late_disclosure" in response["trades"][0]["flags"]
