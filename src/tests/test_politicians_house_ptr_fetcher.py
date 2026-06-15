"""Tests for House PTR PDF fetcher."""

import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.cli import main as politicians_cli_main
from ingestion.politicians.source_health import read_source_health
from ingestion.politicians.sources.house import (
    HouseFiling,
    download_house_ptr_pdfs,
    load_house_filings_from_archives,
)


def test_house_ptr_fetcher_records_pdf_metadata(tmp_path):
    pdf = tmp_path / "source" / "20031020.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"%PDF-1.4 sample")
    filing = HouseFiling(
        report_id="20031020",
        filing_year=2026,
        document_type="PTR",
        filer_name="Jane Doe",
        document_url=pdf.as_uri(),
    )

    manifest = download_house_ptr_pdfs(
        2026,
        filings=[filing],
        data_root=tmp_path / "politicians",
        rate_limit_seconds=0,
    )
    row = manifest["filings"][0]
    raw_pdf = tmp_path / "politicians" / "raw" / "house" / "2026" / "20031020.pdf"

    assert manifest["status"] == "ok"
    assert raw_pdf.exists()
    assert row["report_id"] == "20031020"
    assert row["filing_year"] == 2026
    assert row["document_type"] == "PTR"
    assert row["filer_name"] == "Jane Doe"
    assert row["source_url"] == pdf.as_uri()
    assert row["content_type"] == "application/pdf"
    assert row["content_length"] == len(b"%PDF-1.4 sample")
    assert row["sha256"].startswith("sha256:")
    assert row["downloaded_at"]


def test_house_ptr_fetcher_reuses_pdf_when_remote_hash_unchanged(tmp_path):
    pdf = tmp_path / "source" / "20031020.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"same pdf")
    filing = HouseFiling("20031020", 2026, "Periodic Transaction Report", "Jane Doe", pdf.as_uri())
    data_root = tmp_path / "politicians"

    first = download_house_ptr_pdfs(2026, filings=[filing], data_root=data_root, rate_limit_seconds=0)
    second = download_house_ptr_pdfs(2026, filings=[filing], data_root=data_root, rate_limit_seconds=0)

    assert first["filings"][0]["sha256"] == second["filings"][0]["sha256"]
    assert second["filings"][0]["reused_existing"] is True
    assert second["filings"][0]["remote_changed"] is False


def test_house_ptr_fetcher_rewrites_pdf_when_remote_hash_changes(tmp_path):
    pdf = tmp_path / "source" / "20031020.pdf"
    pdf.parent.mkdir()
    pdf.write_bytes(b"old pdf")
    filing = HouseFiling("20031020", 2026, "PTR", "Jane Doe", pdf.as_uri())
    data_root = tmp_path / "politicians"

    first = download_house_ptr_pdfs(2026, filings=[filing], data_root=data_root, rate_limit_seconds=0)
    pdf.write_bytes(b"new pdf")
    second = download_house_ptr_pdfs(2026, filings=[filing], data_root=data_root, rate_limit_seconds=0)

    assert first["filings"][0]["sha256"] != second["filings"][0]["sha256"]
    assert second["filings"][0]["remote_changed"] is True
    assert (data_root / "raw" / "house" / "2026" / "20031020.pdf").read_bytes() == b"new pdf"


def test_house_ptr_fetcher_marks_missing_artifact_in_manifest_and_health(tmp_path):
    missing = tmp_path / "source" / "missing.pdf"
    filing = HouseFiling("20031020", 2026, "PTR", "Jane Doe", missing.as_uri())
    data_root = tmp_path / "politicians"

    manifest = download_house_ptr_pdfs(2026, filings=[filing], data_root=data_root, rate_limit_seconds=0, backoff_seconds=0)
    health = read_source_health(data_root)

    assert manifest["status"] == "degraded"
    assert manifest["missing_artifact_count"] == 1
    assert manifest["filings"][0]["status"] == "missing_artifact"
    assert health["sources"]["house"]["status"] == "degraded"
    assert health["sources"]["house"]["missing_artifact_count"] == 1


def test_house_ptr_fetcher_retries_with_exponential_backoff(tmp_path):
    calls = []
    sleeps = []
    filing = HouseFiling("20031020", 2026, "PTR", "Jane Doe", "https://example.test/20031020.pdf")

    def flaky_reader(url):
        calls.append(url)
        if len(calls) == 1:
            raise OSError("temporary")
        return b"pdf bytes", "application/pdf"

    manifest = download_house_ptr_pdfs(
        2026,
        filings=[filing],
        data_root=tmp_path / "politicians",
        retries=3,
        backoff_seconds=0.25,
        rate_limit_seconds=0.5,
        read_bytes_fn=flaky_reader,
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    assert manifest["status"] == "ok"
    assert len(calls) == 2
    assert sleeps == [0.25, 0.5]


def test_house_ptr_fetcher_loads_filing_rows_from_archive_manifest(tmp_path):
    data_root = tmp_path / "politicians"
    raw_dir = data_root / "raw" / "house" / "2026"
    manifest_dir = data_root / "manifests"
    raw_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    archive = raw_dir / "2026FD.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("2026FD.txt", "Doe|Jane|PTR|2026|20031020\nSmith|John|Annual|2026|10000001\n")
    (manifest_dir / "house_2026.json").write_text(json.dumps({
        "source": "house",
        "year": 2026,
        "artifacts": [{"path": str(archive)}],
    }), encoding="utf-8")

    filings = load_house_filings_from_archives(2026, data_root=data_root)

    assert len(filings) == 2
    assert filings[0].report_id == "20031020"
    assert filings[0].document_type == "PTR"
    assert filings[0].filer_name == "Doe Jane"


def test_house_fetch_pdfs_cli_shape_with_empty_manifest(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(tmp_path / "politicians"))

    exit_code = politicians_cli_main(["house", "fetch-pdfs", "--year", "2026"])
    out = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert out["source"] == "house"
    assert out["year"] == 2026
    assert out["status"] == "degraded"
    assert out["filing_count"] == 0
