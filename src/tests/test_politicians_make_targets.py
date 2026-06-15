"""Tests for politician disclosure ops commands and Make targets."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.cli import main as politicians_cli_main
from ingestion.politicians.parse import parse_local_raw_artifacts
from ingestion.politicians.source_health import write_source_health
from ingestion.politicians.status import get_politicians_status


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_parse_local_artifacts_writes_normalized_jsonl_scaffolding(tmp_path):
    data_root = tmp_path / "politicians"
    manifest_dir = data_root / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "house_ptr_2026.json").write_text(json.dumps({
        "filings": [
            {
                "report_id": "20031020",
                "filing_year": 2026,
                "document_type": "PTR",
                "filer_name": "Jane Doe",
                "source_url": "https://example.test/20031020.pdf",
                "path": "raw/house/2026/20031020.pdf",
            }
        ]
    }), encoding="utf-8")
    (manifest_dir / "senate_2026.json").write_text(json.dumps({
        "response": {"raw_artifact_path": "raw/senate/2026/search.html"},
        "filings": [
            {
                "report_id": "999999",
                "filing_year": 2026,
                "report_type": "Periodic Transaction Report",
                "filer_name": "John Senator",
                "document_url": "https://efdsearch.senate.gov/search/view/paper/999999/",
                "filed_date": "2026-05-15",
            }
        ],
    }), encoding="utf-8")

    result = parse_local_raw_artifacts(data_root)
    filings = (data_root / "filings.jsonl").read_text(encoding="utf-8").strip().splitlines()

    assert result["status"] == "ok"
    assert result["filing_count"] == 2
    assert result["trade_count"] == 0
    assert (data_root / "trades.jsonl").exists()
    assert (data_root / "parse_errors.jsonl").exists()
    assert len(filings) == 2
    assert json.loads(filings[0])["parse_status"] == "pending_transaction_parser"


def test_parse_local_artifacts_writes_real_house_ptr_trades(tmp_path):
    data_root = tmp_path / "politicians"
    manifest_dir = data_root / "manifests"
    raw_dir = data_root / "raw" / "house" / "2026"
    manifest_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    pdf = raw_dir / "20034622.pdf"
    pdf.write_text(
        "P T R\n"
        "ID Owner Asset Transaction\n"
        "Type\n"
        "Date Notification\n"
        "Date\n"
        "Amount Cap.\n"
        "$200?\n"
        "SP State Street Corporation Common\n"
        "Stock (STT) [ST]\n"
        "S (partial) 05/18/202605/18/2026$15,001 -\n"
        "$50,000\n",
        encoding="utf-8",
    )
    (manifest_dir / "house_ptr_2026.json").write_text(json.dumps({
        "source": "house",
        "year": 2026,
        "filings": [
            {
                "status": "ok",
                "report_id": "20034622",
                "filing_year": 2026,
                "document_type": "PTR",
                "filer_name": "Hon. Jane Doe",
                "source_url": "https://example.test/20034622.pdf",
                "path": str(pdf),
                "filed_date": "2026-05-22",
                "state_district": "CA12",
                "filing_type_code": "P",
            }
        ],
    }), encoding="utf-8")

    result = parse_local_raw_artifacts(data_root)
    trades = [json.loads(line) for line in (data_root / "trades.jsonl").read_text(encoding="utf-8").splitlines()]

    assert result["status"] == "ok"
    assert result["trade_count"] == 1
    assert trades[0]["filer_name"] == "Hon. Jane Doe"
    assert trades[0]["ticker"] == "STT"
    assert trades[0]["owner"] == "spouse"
    assert trades[0]["transaction_type"] == "sale_partial"
    assert trades[0]["disclosure_date"] == "2026-05-22"


def test_status_reports_health_counts_parse_errors_and_newest_disclosure(tmp_path):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    (data_root / "filings.jsonl").write_text(
        json.dumps({"filed_date": "2026-05-10"}) + "\n" +
        json.dumps({"filed_date": "2026-05-15"}) + "\n",
        encoding="utf-8",
    )
    (data_root / "trades.jsonl").write_text("", encoding="utf-8")
    (data_root / "parse_errors.jsonl").write_text(json.dumps({"error": "bad row"}) + "\n", encoding="utf-8")
    write_source_health("house", status="ok", message="healthy", data_root=data_root)

    status = get_politicians_status(data_root)

    assert status["filing_count"] == 2
    assert status["trade_count"] == 0
    assert status["parse_error_count"] == 1
    assert status["newest_disclosure"] == "2026-05-15"
    assert status["source_health"]["sources"]["house"]["status"] == "ok"


def test_parse_and_status_cli_shapes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(tmp_path / "politicians"))

    parse_exit = politicians_cli_main(["parse", "local"])
    parse_out = json.loads(capsys.readouterr().out)
    status_exit = politicians_cli_main(["status"])
    status_output = capsys.readouterr().out

    assert parse_exit == 0
    assert parse_out["status"] == "ok"
    assert status_exit == 0
    assert '"filing_count"' in status_output
    assert "Politician Disclosure Status" in status_output


def test_makefile_contains_politician_ops_targets():
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    for target in (
        "politicians-sync:",
        "politicians-backfill:",
        "politicians-parse:",
        "politicians-status:",
        "politicians-test:",
    ):
        assert target in makefile

    assert "ingestion.politicians.cli sync daily" in makefile
    assert "ingestion.politicians.cli house backfill --year $(YEAR)" in makefile
    assert "ingestion.politicians.cli parse local" in makefile
    assert "ingestion.politicians.cli status" in makefile
