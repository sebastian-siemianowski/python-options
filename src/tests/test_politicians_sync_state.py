"""Tests for politician disclosure sync state and orchestration."""

import io
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console

from ingestion.politicians.cli import main as politicians_cli_main
from ingestion.politicians.sync import (
    default_incremental_window,
    read_sync_state,
    render_sync_summary,
    run_house_backfill,
    run_incremental_sync,
    write_sync_state,
)


def _ok_manifest(source: str, **counts):
    return {"source": source, "status": "ok", "errors": [], **counts}


def test_default_incremental_window_is_last_14_calendar_days():
    date_from, date_to = default_incremental_window(date(2026, 5, 29))

    assert date_from == "2026-05-15"
    assert date_to == "2026-05-29"


def test_incremental_sync_state_records_success_window_counts_and_errors(tmp_path):
    result = run_incremental_sync(
        date_to="2026-05-29",
        data_root=tmp_path / "politicians",
        house_sync_fn=lambda **kwargs: _ok_manifest("house", artifact_count=1, filing_count=2),
        senate_sync_fn=lambda **kwargs: _ok_manifest("senate", filing_count=3),
        console=Console(file=io.StringIO(), force_terminal=False),
    )
    state = read_sync_state(tmp_path / "politicians")

    assert result["date_window"] == {"from": "2026-05-15", "to": "2026-05-29"}
    assert state["sources"]["house"]["last_successful_sync_at"]
    assert state["sources"]["house"]["date_window"] == {"from": "2026-05-15", "to": "2026-05-29"}
    assert state["sources"]["house"]["counts"]["artifact_count"] == 1
    assert state["sources"]["senate"]["counts"]["filing_count"] == 3
    assert state["sources"]["senate"]["error_summary"] == []


def test_house_backfill_preserves_unrelated_years(tmp_path):
    data_root = tmp_path / "politicians"
    existing = {
        "updated_at": "before",
        "sources": {},
        "backfills": {
            "house": {
                "2025": {
                    "source": "house",
                    "year": 2025,
                    "status": "ok",
                    "counts": {"artifact_count": 9},
                }
            }
        },
    }
    write_sync_state(existing, data_root)

    result = run_house_backfill(
        2026,
        data_root=data_root,
        backfill_fn=lambda year, data_root: _ok_manifest("house", artifact_count=1),
        fetch_pdfs_fn=lambda year, data_root: _ok_manifest("house", filing_count=2, pdf_count=2),
        console=Console(file=io.StringIO(), force_terminal=False),
    )
    state = result["state"]

    assert state["backfills"]["house"]["2025"]["counts"]["artifact_count"] == 9
    assert state["backfills"]["house"]["2026"]["status"] == "ok"
    assert state["backfills"]["house"]["2026"]["counts"]["pdf_count"] == 2


def test_rich_sync_summary_emits_table():
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=120)

    render_sync_summary([
        {
            "source": "house",
            "status": "ok",
            "date_window": {"from": "2026-05-15", "to": "2026-05-29"},
            "counts": {"filing_count": 2},
            "error_summary": [],
        }
    ], console=console)

    output = buffer.getvalue()
    assert "Politician Sync" in output
    assert "house" in output
    assert "filing_count=2" in output


def test_failed_source_does_not_block_other_source_or_post_processing(tmp_path):
    post_processed = []

    def failing_house(**kwargs):
        raise RuntimeError("house broke")

    def ok_senate(**kwargs):
        return _ok_manifest("senate", filing_count=4)

    result = run_incremental_sync(
        date_from="2026-05-01",
        date_to="2026-05-29",
        data_root=tmp_path / "politicians",
        house_sync_fn=failing_house,
        senate_sync_fn=ok_senate,
        post_source_fn=lambda source, row: post_processed.append((source, row["status"])),
        console=Console(file=io.StringIO(), force_terminal=False),
    )
    state = read_sync_state(tmp_path / "politicians")

    assert [row["source"] for row in result["results"]] == ["house", "senate"]
    assert result["results"][0]["status"] == "degraded"
    assert result["results"][1]["status"] == "ok"
    assert post_processed == [("house", "degraded"), ("senate", "ok")]
    assert state["sources"]["senate"]["last_successful_sync_at"]
    assert state["sources"]["house"]["last_failed_sync_at"]


def test_sync_incremental_cli_shape(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(tmp_path / "politicians"))
    monkeypatch.setattr(
        "ingestion.politicians.cli.run_incremental_sync",
        lambda date_from=None, date_to=None: {
            "date_window": {"from": "2026-05-15", "to": "2026-05-29"},
            "results": [
                {"source": "house", "status": "ok", "date_window": {}, "counts": {}, "error_summary": []},
                {"source": "senate", "status": "ok", "date_window": {}, "counts": {}, "error_summary": []},
            ],
        },
    )

    exit_code = politicians_cli_main(["sync", "incremental", "--to", "2026-05-29"])
    out = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert out["status"] == "ok"
    assert out["date_window"]["to"] == "2026-05-29"
