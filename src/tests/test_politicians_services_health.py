"""Services-health integration tests for politician ingestion."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.backend.services import health_service
from web.backend.services.health_service import get_full_health, get_recent_errors, log_error


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_services_health_includes_politician_source_status_and_data_age(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    _write_jsonl(data_root / "trades.jsonl", [
        {"source": "house", "disclosure_date": "2026-05-28", "parser_confidence": 0.98},
    ])
    _write_jsonl(data_root / "filings.jsonl", [
        {"source": "house", "filed_date": "2026-05-28"},
        {"source": "senate", "filed_date": "2026-05-27"},
    ])
    _write_jsonl(data_root / "parse_errors.jsonl", [
        {"source": "senate", "report_id": "S1", "errors": ["missing_required:ticker"], "created_at": "2026-05-29T10:00:00Z"},
    ])
    (data_root / "source_health.json").write_text(json.dumps({
        "sources": {
            "house": {"status": "ok", "updated_at": "2026-05-29T10:00:00Z", "errors": []},
            "senate": {"status": "degraded", "updated_at": "2026-05-29T09:00:00Z", "errors": ["layout changed"]},
        },
    }), encoding="utf-8")
    (data_root / "sync_state.json").write_text(json.dumps({
        "sources": {
            "house": {"last_successful_sync_at": "2026-05-29T10:00:00Z"},
            "senate": {"last_failed_sync_at": "2026-05-29T09:00:00Z"},
        },
    }), encoding="utf-8")
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    health = get_full_health()

    assert health["api"]["status"] == "ok"
    assert health["politicians"]["status"] == "degraded"
    assert health["politicians"]["degraded_blocks_app"] is False
    assert health["politicians"]["data_age_seconds"] is not None
    assert health["politicians"]["details_url"] == "/politicians"
    assert health["politicians"]["source_health_url"] == "/api/politicians/source-health"
    assert health["politicians"]["sources"]["house"]["status"] == "ok"
    assert health["politicians"]["sources"]["senate"]["status"] == "degraded"


def test_services_error_log_keeps_politician_parser_context():
    health_service._error_log.clear()

    log_error(
        "politicians.senate",
        "parse failed",
        filing_id="S1",
        parser_version="senate-parser-v1",
        exception_class="ValueError",
    )
    error = get_recent_errors(1)[0]

    assert error["source"] == "politicians.senate"
    assert error["filing_id"] == "S1"
    assert error["parser_version"] == "senate-parser-v1"
    assert error["exception_class"] == "ValueError"


def test_services_page_links_to_politicians_source_health_details():
    page = REPO_ROOT / "src" / "web" / "frontend" / "src" / "pages" / "ServicesPage.tsx"
    text = page.read_text(encoding="utf-8")

    assert "Politician Ingestion" in text
    assert "Source Health Details" in text
    assert "to={data?.details_url || '/politicians'}" in text


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
