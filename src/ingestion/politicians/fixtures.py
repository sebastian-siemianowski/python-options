"""Golden fixture helpers for politician disclosure parser tests."""

from __future__ import annotations

import json
import hashlib
import os
import time
from pathlib import Path
from typing import Any

from ingestion.politicians.normalize import normalize_amount_bucket
from ingestion.politicians.parsers.house_pdf import parse_house_ptr_text
from ingestion.politicians.parsers.senate import parse_senate_ptr_html, parse_senate_ptr_text
from ingestion.politicians.paths import ensure_politicians_data_dirs
from ingestion.politicians.validation import write_validated_trades


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "politicians" / "golden"


def load_golden_fixtures(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load golden filing fixture definitions."""
    fixture_path = Path(path) if path else FIXTURE_DIR / "fixtures.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def load_expected_normalized(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load expected normalized rows for every golden fixture."""
    expected_path = Path(path) if path else FIXTURE_DIR / "expected_normalized.json"
    return json.loads(expected_path.read_text(encoding="utf-8"))


def parse_golden_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    """Parse one golden fixture into the stable normalized comparison shape."""
    source = fixture["source"]
    if source == "house":
        rows = parse_house_ptr_text(
            fixture["raw_text"],
            report_id=fixture["report_id"],
            filing_year=int(fixture["filing_year"]),
            filer_name=fixture["filer_name"],
            document_url=fixture["source_url"],
        )
        if len(rows) != 1:
            raise ValueError(f"{fixture['fixture_id']} expected one House row, got {len(rows)}")
        row = rows[0].to_dict()
        ticker = fixture.get("ticker")
        asset_type = fixture.get("asset_type", "unknown")
    elif source == "senate":
        if fixture.get("format") == "html":
            result = parse_senate_ptr_html(fixture["raw_text"])
            rows = result["rows"]
        else:
            rows = [row.to_dict() for row in parse_senate_ptr_text(fixture["raw_text"])]
        if len(rows) != 1:
            raise ValueError(f"{fixture['fixture_id']} expected one Senate row, got {len(rows)}")
        row = rows[0]
        ticker = row.get("ticker") or fixture.get("ticker")
        asset_type = row.get("asset_type") or fixture.get("asset_type", "unknown")
    else:
        raise ValueError(f"Unsupported fixture source: {source}")

    return {
        "fixture_id": fixture["fixture_id"],
        "source": source,
        "report_id": fixture["report_id"],
        "report_type": fixture.get("report_type", "ptr"),
        "is_amendment": bool(fixture.get("is_amendment", False)),
        "filer_name": fixture["filer_name"],
        "owner": row.get("owner") or "unknown",
        "asset_name_raw": row.get("asset_name_raw"),
        "asset_type": asset_type,
        "ticker": ticker,
        "transaction_type": row.get("transaction_type") or "unknown",
        "amount_bucket_raw": row.get("amount_bucket_raw"),
        "source_url": fixture["source_url"],
        "retrieval_date": fixture["retrieval_date"],
        "coverage_tags": sorted(fixture.get("coverage_tags", [])),
    }


def parse_golden_fixtures(fixtures: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Parse every golden fixture into normalized comparison rows."""
    rows = [parse_golden_fixture(fixture) for fixture in (fixtures or load_golden_fixtures())]
    return sorted(rows, key=lambda row: row["fixture_id"])


def run_golden_fixture_pipeline(
    *,
    data_root: str | Path | None = None,
    fixtures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run an offline raw-fixture-artifact to trades.jsonl pipeline."""
    started = time.perf_counter()
    root = ensure_politicians_data_dirs(data_root)
    fixture_rows = fixtures or load_golden_fixtures()
    artifact_paths = _materialize_raw_fixture_artifacts(root, fixture_rows)
    normalized_rows = []
    for fixture, artifact_path in zip(fixture_rows, artifact_paths, strict=True):
        raw_text = artifact_path.read_text(encoding="utf-8")
        parsed = parse_golden_fixture({**fixture, "raw_text": raw_text})
        normalized_rows.append(_pipeline_trade_row(parsed, fixture, artifact_path, raw_text))
    validation = write_validated_trades(normalized_rows, data_root=root)
    trades = _read_jsonl(root / "trades.jsonl")
    return {
        "status": validation["status"],
        "offline_mode": _offline_mode_enabled(),
        "network_calls_attempted": 0,
        "artifact_count": len(artifact_paths),
        "input_count": len(normalized_rows),
        "trade_count": len(trades),
        "parse_error_count": validation["error_count"],
        "row_hashes": sorted(str(row.get("row_hash")) for row in trades),
        "trades_path": str(root / "trades.jsonl"),
        "parse_errors_path": str(root / "parse_errors.jsonl"),
        "duration_seconds": round(time.perf_counter() - started, 4),
    }


def _materialize_raw_fixture_artifacts(root: Path, fixtures: list[dict[str, Any]]) -> list[Path]:
    paths = []
    for fixture in fixtures:
        path = root / "raw" / fixture["source"] / str(fixture["filing_year"]) / f"{fixture['fixture_id']}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(fixture["raw_text"], encoding="utf-8")
        paths.append(path)
    return paths


def _pipeline_trade_row(
    parsed: dict[str, Any],
    fixture: dict[str, Any],
    artifact_path: Path,
    raw_text: str,
) -> dict[str, Any]:
    amount = normalize_amount_bucket(parsed.get("amount_bucket_raw"))
    return {
        "source": parsed["source"],
        "chamber": parsed["source"],
        "report_id": parsed["report_id"],
        "report_type": parsed["report_type"],
        "is_amendment": parsed["is_amendment"],
        "filer_name": parsed["filer_name"],
        "owner": parsed["owner"],
        "asset_name_raw": parsed["asset_name_raw"],
        "asset_type": parsed["asset_type"],
        "ticker": parsed["ticker"],
        "transaction_type": parsed["transaction_type"],
        "amount_bucket_raw": parsed["amount_bucket_raw"],
        "amount_min_usd": amount["amount_min_usd"],
        "amount_max_usd": amount["amount_max_usd"],
        "amount_mid_usd": amount["amount_mid_usd"],
        "filing_year": fixture["filing_year"],
        "document_url": parsed["source_url"],
        "raw_artifact_path": str(artifact_path),
        "source_hash": f"sha256:{hashlib.sha256(raw_text.encode('utf-8')).hexdigest()}",
        "parser_version": "golden-fixture-pipeline-v1",
        "parser_confidence": 1.0,
        "validation_status": "valid",
        "warnings": [],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _offline_mode_enabled() -> bool:
    return os.getenv("OFFLINE_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
