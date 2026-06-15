"""End-to-end golden fixture pipeline tests."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.fixtures import load_golden_fixtures, run_golden_fixture_pipeline
from ingestion.politicians.parse import parse_local_raw_artifacts
from ingestion.politicians.validation import validate_trade_record


def test_golden_fixture_pipeline_runs_raw_artifacts_to_trades_jsonl_under_30s(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    monkeypatch.setenv("OFFLINE_MODE", "1")
    started = time.perf_counter()

    result = run_golden_fixture_pipeline(data_root=data_root)
    elapsed = time.perf_counter() - started
    trades = _read_jsonl(data_root / "trades.jsonl")

    assert elapsed < 30
    assert result["duration_seconds"] < 30
    assert result["status"] == "ok"
    assert result["offline_mode"] is True
    assert result["network_calls_attempted"] == 0
    assert result["artifact_count"] == len(load_golden_fixtures())
    assert result["trade_count"] == len(load_golden_fixtures())
    assert result["parse_error_count"] == 0
    assert len(result["row_hashes"]) == len(load_golden_fixtures())
    assert len(set(result["row_hashes"])) == len(result["row_hashes"])
    assert all(str(row["source_hash"]).startswith("sha256:") for row in trades)
    assert all(str(row["row_hash"]).startswith("sha256:") for row in trades)
    assert all(row.get("trade_id") for row in trades)
    assert (data_root / "parse_errors.jsonl").read_text(encoding="utf-8") == ""


def test_golden_fixture_pipeline_validates_schema_and_is_idempotent(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    monkeypatch.setenv("OFFLINE_MODE", "1")

    first = run_golden_fixture_pipeline(data_root=data_root)
    first_trades = (data_root / "trades.jsonl").read_text(encoding="utf-8")
    second = run_golden_fixture_pipeline(data_root=data_root)
    second_trades = (data_root / "trades.jsonl").read_text(encoding="utf-8")
    trades = _read_jsonl(data_root / "trades.jsonl")

    assert first["row_hashes"] == second["row_hashes"]
    assert first_trades == second_trades
    assert all(validate_trade_record(row)[0] for row in trades)


def test_parse_local_skips_house_image_only_ptr_without_error(tmp_path):
    data_root = tmp_path / "politicians"
    manifest_dir = data_root / "manifests"
    raw_dir = data_root / "raw" / "house" / "2025"
    manifest_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    pdf = raw_dir / "8220747.pdf"
    pdf.write_bytes(b"")
    manifest = {
        "source": "house",
        "year": 2025,
        "status": "ok",
        "filing_count": 1,
        "pdf_count": 1,
        "missing_artifact_count": 0,
        "filings": [
            {
                "report_id": "8220747",
                "filing_year": 2025,
                "document_type": "PTR",
                "filer_name": "Hon. Image Only",
                "source_url": "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/2025/8220747.pdf",
                "path": str(pdf),
                "filed_date": "2025-01-15",
                "state_district": "FL12",
                "status": "ok",
            }
        ],
        "errors": [],
    }
    (manifest_dir / "house_ptr_2025.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = parse_local_raw_artifacts(data_root=data_root)

    assert result["status"] == "ok"
    assert result["trade_count"] == 0
    assert result["parse_error_count"] == 0
    assert (data_root / "parse_errors.jsonl").read_text(encoding="utf-8") == ""


def _read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
