"""Source-health endpoint tests for politician disclosure monitoring."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.backend.services.politicians_service import get_politicians_source_health_response


def test_source_health_returns_operator_ready_source_summaries(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    _write_jsonl(
        data_root / "trades.jsonl",
        [
            {"source": "house", "disclosure_date": "2026-05-28", "parser_confidence": 0.97},
            {"source": "house", "disclosure_date": "2026-05-20", "parser_confidence": 0.60},
        ],
    )
    _write_jsonl(data_root / "filings.jsonl", [{"source": "house", "filed_date": "2026-05-29"}])
    _write_jsonl(
        data_root / "parse_errors.jsonl",
        [{"source": "senate", "report_id": "S1", "errors": ["missing_required:ticker"], "created_at": "2026-05-29T10:00:00Z"}],
    )
    (data_root / "source_health.json").write_text(
        json.dumps({
            "updated_at": "2026-05-29T10:00:00Z",
            "sources": {
                "house": {"status": "ok", "updated_at": "2026-05-29T10:00:00Z", "errors": []},
                "senate": {"status": "error", "updated_at": "2026-05-29T09:00:00Z", "errors": ["403"]},
            },
        }),
        encoding="utf-8",
    )
    (data_root / "sync_state.json").write_text(
        json.dumps({
            "updated_at": "2026-05-29T10:05:00Z",
            "sources": {
                "house": {"last_successful_sync_at": "2026-05-29T10:00:00Z"},
                "senate": {"last_failed_sync_at": "2026-05-29T09:00:00Z"},
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_source_health_response()

    assert response["status"] == "ok"
    assert response["overall_status"] == "degraded"
    assert response["parse_error_count"] == 1
    assert set(response["sources"]) == {"house", "senate"}
    house = response["sources"]["house"]
    assert house["status"] == "ok"
    assert house["last_sync_time"] == "2026-05-29T10:00:00Z"
    assert house["newest_filing"] == "2026-05-29"
    assert house["parse_success_rate"] == 1.0
    assert house["low_confidence_rows"] == 1
    assert house["recent_errors"] == []
    assert "healthy" in house["remediation"]
    senate = response["sources"]["senate"]
    assert senate["status"] == "offline"
    assert senate["last_sync_time"] == "2026-05-29T09:00:00Z"
    assert senate["parse_success_rate"] == 0.0
    assert senate["recent_errors"]
    assert "availability" in senate["remediation"]


def test_source_health_status_values_are_constrained_and_services_page_ready(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_source_health_response()

    allowed = {"ok", "degraded", "offline", "disabled"}
    assert response["overall_status"] in allowed
    for source in response["sources"].values():
        assert source["status"] in allowed
        assert {
            "status",
            "last_sync_time",
            "newest_filing",
            "parse_success_rate",
            "low_confidence_rows",
            "recent_errors",
            "remediation",
        } <= set(source)


def test_source_health_disabled_response_uses_disabled_status(monkeypatch):
    monkeypatch.setenv("POLITICIANS_ENABLED", "0")

    response = get_politicians_source_health_response()

    assert response["status"] == "disabled"
    assert response["endpoint"] == "GET /source-health"


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
