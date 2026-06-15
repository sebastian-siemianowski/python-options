"""Immutable source audit trail helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ingestion.politicians.dedupe import attach_identity
from ingestion.politicians.paths import ensure_politicians_data_dirs
from ingestion.politicians.source_health import utc_now_iso


def compute_source_hash(path: str | Path) -> str:
    """Return SHA-256 hash for a raw source artifact."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def attach_audit_fields(
    row: dict[str, Any],
    *,
    raw_artifact_path: str | Path,
    document_url: str,
    parser_version: str,
) -> dict[str, Any]:
    """Attach immutable source audit fields and deterministic identity."""
    audited = dict(row)
    audited.update({
        "source_hash": compute_source_hash(raw_artifact_path),
        "raw_artifact_path": str(raw_artifact_path),
        "document_url": document_url,
        "parser_version": parser_version,
    })
    return attach_identity(audited)


def write_filing_audit_record(
    *,
    report_id: str,
    source: str,
    parser_version: str,
    raw_artifact_path: str | Path,
    document_url: str,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Upsert filing-level parse metadata and parser-version history."""
    root = ensure_politicians_data_dirs(data_root)
    filings_path = root / "filings.jsonl"
    existing = _read_jsonl(filings_path)
    now = utc_now_iso()
    source_hash = compute_source_hash(raw_artifact_path)
    updated = False
    for row in existing:
        if row.get("report_id") == report_id and row.get("source") == source:
            history = row.setdefault("parser_version_history", [])
            if parser_version not in history:
                history.append(parser_version)
            row.update({
                "parser_version": parser_version,
                "raw_artifact_path": str(raw_artifact_path),
                "document_url": document_url,
                "source_hash": source_hash,
                "updated_at": now,
            })
            updated = True
            break
    if not updated:
        existing.append({
            "source": source,
            "report_id": report_id,
            "parser_version": parser_version,
            "parser_version_history": [parser_version],
            "raw_artifact_path": str(raw_artifact_path),
            "document_url": document_url,
            "source_hash": source_hash,
            "created_at": now,
            "updated_at": now,
        })
    with filings_path.open("w", encoding="utf-8") as handle:
        for row in existing:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return existing[-1] if not updated else row


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
