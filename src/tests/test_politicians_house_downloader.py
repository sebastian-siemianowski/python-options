"""Tests for House politician disclosure archive downloader."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.cli import main as politicians_cli_main
from ingestion.politicians.source_health import read_source_health
from ingestion.politicians.sources.house import (
    discover_house_yearly_archive_urls,
    download_house_yearly_archive,
)


def test_house_archive_download_writes_raw_file_manifest_and_hash(tmp_path, monkeypatch):
    monkeypatch.delenv("OFFLINE_MODE", raising=False)
    source = tmp_path / "source" / "2026FD.zip"
    source.parent.mkdir()
    source.write_bytes(b"house archive bytes")

    manifest = download_house_yearly_archive(
        2026,
        archive_urls=[source.as_uri()],
        data_root=tmp_path / "politicians",
    )

    artifact = manifest["artifacts"][0]
    raw_path = tmp_path / "politicians" / "raw" / "house" / "2026" / "2026FD.zip"
    manifest_path = tmp_path / "politicians" / "manifests" / "house_2026.json"

    assert manifest["status"] == "ok"
    assert raw_path.exists()
    assert manifest_path.exists()
    assert artifact["filename"] == "2026FD.zip"
    assert artifact["sha256"].startswith("sha256:")
    assert artifact["size_bytes"] == len(b"house archive bytes")


def test_house_archive_download_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.delenv("OFFLINE_MODE", raising=False)
    source = tmp_path / "source" / "2026FD.zip"
    source.parent.mkdir()
    source.write_bytes(b"same bytes")
    data_root = tmp_path / "politicians"

    first = download_house_yearly_archive(2026, archive_urls=[source.as_uri()], data_root=data_root)
    second = download_house_yearly_archive(2026, archive_urls=[source.as_uri()], data_root=data_root)

    assert first["artifact_count"] == 1
    assert second["artifact_count"] == 1
    assert first["artifacts"][0]["sha256"] == second["artifacts"][0]["sha256"]
    assert second["artifacts"][0]["reused_existing"] is True


def test_house_archive_offline_mode_reuses_existing_raw_only(tmp_path, monkeypatch):
    monkeypatch.delenv("OFFLINE_MODE", raising=False)
    source = tmp_path / "source" / "2026FD.zip"
    source.parent.mkdir()
    source.write_bytes(b"offline bytes")
    data_root = tmp_path / "politicians"
    download_house_yearly_archive(2026, archive_urls=[source.as_uri()], data_root=data_root)

    monkeypatch.setenv("OFFLINE_MODE", "1")
    manifest = download_house_yearly_archive(2026, data_root=data_root)

    assert manifest["status"] == "ok"
    assert manifest["offline_mode"] is True
    assert manifest["artifact_count"] == 1
    assert manifest["artifacts"][0]["url"].startswith("offline://")


def test_house_archive_source_error_writes_degraded_health(tmp_path, monkeypatch):
    monkeypatch.delenv("OFFLINE_MODE", raising=False)
    missing = tmp_path / "missing" / "2026FD.zip"
    data_root = tmp_path / "politicians"

    manifest = download_house_yearly_archive(2026, archive_urls=[missing.as_uri()], data_root=data_root)
    health = read_source_health(data_root)

    assert manifest["status"] == "degraded"
    assert manifest["errors"]
    assert health["sources"]["house"]["status"] == "degraded"
    assert health["sources"]["house"]["errors"]


def test_house_archive_discovery_from_index_page(tmp_path, monkeypatch):
    monkeypatch.delenv("OFFLINE_MODE", raising=False)
    archive = tmp_path / "source" / "2026FD.zip"
    archive.parent.mkdir()
    archive.write_bytes(b"discovered bytes")
    index = tmp_path / "index.html"
    index.write_text(f'<html><a href="{archive.as_uri()}">2026</a></html>', encoding="utf-8")

    urls = discover_house_yearly_archive_urls(2026, index_url=index.as_uri())
    manifest = download_house_yearly_archive(
        2026,
        index_url=index.as_uri(),
        data_root=tmp_path / "politicians",
    )

    assert urls == [archive.as_uri()]
    assert manifest["status"] == "ok"
    assert manifest["artifact_count"] == 1


def test_house_backfill_cli_shape_and_output(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OFFLINE_MODE", raising=False)
    source = tmp_path / "source" / "2026FD.zip"
    source.parent.mkdir()
    source.write_bytes(b"cli bytes")
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(tmp_path / "politicians"))

    exit_code = politicians_cli_main(["house", "backfill", "--year", "2026", "--archive-url", source.as_uri()])
    out = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert out["source"] == "house"
    assert out["year"] == 2026
    assert out["status"] == "ok"
    assert out["artifact_count"] == 1
