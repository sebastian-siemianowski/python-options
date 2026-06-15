"""Daily sync task tests for politician disclosure operations."""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.cli import main as politicians_cli_main
from ingestion.politicians.daily import run_daily_politicians_sync
from web.backend.routers.tasks import PoliticiansDailySyncRequest, trigger_politicians_daily_sync


def test_daily_sync_runs_sync_parse_validate_cache_and_writes_summary(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    monkeypatch.delenv("OFFLINE_MODE", raising=False)
    calls = []

    def sync_fn(**kwargs):
        calls.append(("sync", kwargs["date_to"]))
        return {
            "status": "ok",
            "date_window": {"from": kwargs["date_from"], "to": kwargs["date_to"]},
            "results": [
                {"source": "house", "status": "ok", "counts": {"filing_count": 1}, "error_summary": []},
                {"source": "senate", "status": "ok", "counts": {"filing_count": 1}, "error_summary": []},
            ],
        }

    def parse_fn(data_root):
        calls.append(("parse", str(data_root)))
        root = data_root
        (root / "trades.jsonl").write_text(json.dumps(_valid_trade()) + "\n", encoding="utf-8")
        (root / "parse_errors.jsonl").write_text("", encoding="utf-8")
        return {"status": "ok", "filing_count": 2, "trade_count": 1, "parse_error_count": 0}

    def cache_refresh_fn():
        calls.append(("cache", "refresh"))
        return {"status": "ok", "cleared_entries": 1}

    result = run_daily_politicians_sync(
        date_from="2026-05-01",
        date_to="2026-05-29",
        data_root=data_root,
        sync_fn=sync_fn,
        parse_fn=parse_fn,
        cache_refresh_fn=cache_refresh_fn,
    )

    assert result["status"] == "ok"
    assert result["safe_to_rerun"] is True
    assert result["steps"]["sync_sources"]["status"] == "ok"
    assert result["steps"]["parse_artifacts"]["trade_count"] == 1
    assert result["steps"]["validate_output"]["valid_count"] == 1
    assert result["steps"]["refresh_api_cache"]["status"] == "ok"
    assert result["counts"]["filing_count"] == 4
    assert result["counts"]["trade_count"] == 2
    assert calls == [("sync", "2026-05-29"), ("parse", str(data_root)), ("cache", "refresh")]
    latest = data_root / "runs" / "daily_sync_latest.json"
    assert latest.exists()
    assert json.loads(latest.read_text(encoding="utf-8"))["task"] == "politicians_daily_sync"


def test_daily_sync_offline_mode_skips_source_network_sync(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    monkeypatch.setenv("OFFLINE_MODE", "1")

    def sync_fn(**kwargs):
        raise AssertionError("offline mode must not call source sync")

    def parse_fn(data_root):
        (data_root / "trades.jsonl").write_text(json.dumps(_valid_trade()) + "\n", encoding="utf-8")
        return {"status": "ok", "filing_count": 0, "trade_count": 1, "parse_error_count": 0}

    result = run_daily_politicians_sync(
        date_to="2026-05-29",
        data_root=data_root,
        sync_fn=sync_fn,
        parse_fn=parse_fn,
        cache_refresh_fn=lambda: {"status": "ok"},
    )

    assert result["status"] == "ok"
    assert result["offline_mode"] is True
    assert result["steps"]["sync_sources"] == {
        "status": "skipped",
        "reason": "OFFLINE_MODE=1",
        "date_window": {"from": "2026-05-15", "to": "2026-05-29"},
    }


def test_daily_sync_cli_shape(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(tmp_path / "politicians"))
    monkeypatch.setattr(
        "ingestion.politicians.cli.run_daily_politicians_sync",
        lambda date_from=None, date_to=None: {
            "task": "politicians_daily_sync",
            "status": "ok",
            "date_window": {"from": date_from, "to": date_to},
            "counts": {"trade_count": 1},
            "errors": [],
        },
    )

    exit_code = politicians_cli_main(["sync", "daily", "--from", "2026-05-01", "--to", "2026-05-29"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["task"] == "politicians_daily_sync"
    assert output["date_window"] == {"from": "2026-05-01", "to": "2026-05-29"}


def test_backend_task_router_queues_politician_daily_sync(monkeypatch):
    captured = {}

    class FakeTaskLauncher:
        def delay(self, *, date_from=None, date_to=None):
            captured["date_from"] = date_from
            captured["date_to"] = date_to
            return type("QueuedTask", (), {"id": "task-politicians-1"})()

    monkeypatch.setattr("web.backend.tasks.politicians_daily_sync_task", FakeTaskLauncher())

    response = asyncio.run(trigger_politicians_daily_sync(
        PoliticiansDailySyncRequest(date_from="2026-05-01", date_to="2026-05-29")
    ))

    assert response == {
        "task_id": "task-politicians-1",
        "task_type": "Politician Daily Sync",
        "status": "queued",
    }
    assert captured == {"date_from": "2026-05-01", "date_to": "2026-05-29"}


def _valid_trade():
    return {
        "source": "house",
        "report_id": "20031020",
        "filer_name": "Jane Doe",
        "asset_name_raw": "NVIDIA Corporation",
        "asset_type": "stock",
        "ticker": "NVDA",
        "transaction_type": "purchase",
        "amount_bucket_raw": "$1,001 - $15,000",
        "source_hash": "sha256:test",
        "raw_artifact_path": "raw/house/2026/20031020.pdf",
        "document_url": "https://example.test/20031020.pdf",
        "parser_version": "test-v1",
    }
