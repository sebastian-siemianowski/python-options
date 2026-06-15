"""Filer detail endpoint tests for politician disclosure monitoring."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.backend.services.politicians_service import get_politicians_filer_response


def test_filer_detail_returns_metadata_rollups_delay_stats_and_documents(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    _write_jsonl(
        data_root / "trades.jsonl",
        [
            _trade("jane-doe", "Jane Doe", "NVDA", "Technology", "self", "purchase", 10_000, 10, "2026-05-28"),
            _trade("jane-doe", "Jane Doe", "MSFT", "Technology", "spouse", "sale", 5_000, 60, "2026-05-20"),
            _trade("jane-doe", "Jane Doe", "XOM", "Energy", "dependent_child", "purchase", 3_000, 20, "2026-05-10"),
            _trade("john-smith", "John Smith", "AAPL", "Technology", "self", "purchase", 99_000, 5, "2026-05-28"),
        ],
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_filer_response("jane-doe", window_days=60, as_of_date="2026-05-29")

    assert response["status"] == "ok"
    assert response["filer_id"] == "jane-doe"
    assert response["metadata"]["filer_id"] == "jane-doe"
    assert response["metadata"]["filer_name"] == "Jane Doe"
    assert response["metadata"]["chamber"] == "house"
    assert response["metadata"]["party"] == "D"
    assert response["metadata"]["state"] == "CA"
    assert response["metadata"]["source"] == "house"
    assert response["metadata"]["committee_data_source"] == "enrichment_not_source_disclosure"
    assert response["metadata"]["metadata_complete"] is True
    assert response["total"] == 3
    assert response["top_tickers"][0] == {"ticker": "NVDA", "trade_count": 1, "amount_mid_usd": 10_000.0}
    assert response["top_sectors"][0] == {"sector": "Technology", "trade_count": 2, "amount_mid_usd": 15_000.0}
    assert response["delay_stats"] == {
        "count": 3,
        "average_days": 30.0,
        "median_days": 20.0,
        "max_days": 60.0,
        "late_filing_count": 1,
    }
    assert len(response["source_documents"]) == 3
    assert response["source_documents"][0]["official_source_url"].startswith("https://example.test/")


def test_filer_detail_does_not_expose_non_public_personal_data(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    row = _trade("jane-doe", "Jane Doe", "NVDA", "Technology", "self", "purchase", 10_000, 10, "2026-05-28")
    row.update({
        "email": "jane@example.test",
        "phone": "555-0100",
        "home_address": "123 Private Street",
        "spouse_name": "Private Name",
    })
    _write_jsonl(data_root / "trades.jsonl", [row])
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_filer_response("jane-doe", as_of_date="2026-05-29")
    serialized = json.dumps(response)

    assert "jane@example.test" not in serialized
    assert "555-0100" not in serialized
    assert "Private Street" not in serialized
    assert "Private Name" not in serialized


def test_filer_detail_distinguishes_all_ownership_categories(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    owners = ["self", "spouse", "dependent_child", "joint", "unexpected"]
    _write_jsonl(
        data_root / "trades.jsonl",
        [
            _trade("jane-doe", "Jane Doe", f"T{idx}", "Unknown", owner, "purchase", 1_000, 1, "2026-05-28")
            for idx, owner in enumerate(owners)
        ],
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_filer_response("jane-doe", as_of_date="2026-05-29")

    assert response["ownership_breakdown"] == {
        "self": 1,
        "spouse": 1,
        "dependent_child": 1,
        "joint": 1,
        "unknown": 1,
    }


def test_filer_detail_works_when_member_metadata_is_incomplete(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    row = _trade(None, None, "NVDA", "Technology", "unknown", "purchase", 1_000, 1, "2026-05-28")
    row["filer_id"] = "missing-meta"
    _write_jsonl(data_root / "trades.jsonl", [row])
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_filer_response("missing-meta", as_of_date="2026-05-29")

    assert response["metadata"]["filer_id"] == "missing-meta"
    assert response["metadata"]["filer_name"] is None
    assert response["metadata"]["metadata_complete"] is False
    assert response["total"] == 1


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _trade(
    filer_id,
    filer_name,
    ticker,
    sector,
    owner,
    transaction_type,
    amount_mid_usd,
    delay_days,
    disclosure_date,
):
    return {
        "trade_id": f"{filer_id}-{ticker}-{owner}",
        "filer_id": filer_id,
        "filer_name": filer_name,
        "ticker": ticker,
        "sector": sector,
        "owner": owner,
        "transaction_type": transaction_type,
        "amount_mid_usd": amount_mid_usd,
        "delay_days": delay_days,
        "disclosure_date": disclosure_date,
        "document_url": f"https://example.test/{filer_id}-{ticker}.pdf",
        "source": "house",
        "report_id": f"R-{ticker}",
        "chamber": "house",
        "party": "D",
        "state": "CA",
    }
