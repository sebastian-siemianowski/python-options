"""Local raw-artifact inventory parser for politician disclosures."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ingestion.politicians.audit import compute_source_hash
from ingestion.politicians.normalize import normalize_amount_bucket, normalize_date_fields
from ingestion.politicians.parsers.house_pdf import PARSER_VERSION as HOUSE_PARSER_VERSION
from ingestion.politicians.parsers.house_pdf import extract_pdf_text
from ingestion.politicians.parsers.house_pdf import parse_house_ptr_pdf
from ingestion.politicians.parsers.senate import PARSER_VERSION as SENATE_PARSER_VERSION
from ingestion.politicians.parsers.senate import parse_senate_ptr_document, parse_senate_ptr_html
from ingestion.politicians.paths import ensure_politicians_data_dirs
from ingestion.politicians.source_health import utc_now_iso
from ingestion.politicians.ticker_resolution import resolve_ticker
from ingestion.politicians.validation import write_validated_trades


def parse_local_raw_artifacts(data_root: str | Path | None = None) -> dict[str, Any]:
    """
    Convert current source manifests and raw artifacts into normalized JSONL.
    """
    root = ensure_politicians_data_dirs(data_root)
    filings = _collect_filing_rows(root)
    filings_path = root / "filings.jsonl"
    house_rows, house_errors = _parse_house_ptr_artifacts(root)
    senate_rows, senate_errors = _parse_senate_ptr_artifacts(root)
    normalized_rows = [*house_rows, *senate_rows]
    extraction_errors = [*house_errors, *senate_errors]

    with filings_path.open("w", encoding="utf-8") as handle:
        for row in filings:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    validation = write_validated_trades(normalized_rows, data_root=root)
    _append_extraction_errors(root / "parse_errors.jsonl", extraction_errors)

    return {
        "status": _parse_status(validation, extraction_errors),
        "generated_at": utc_now_iso(),
        "filing_count": len(filings),
        "trade_count": validation["valid_count"],
        "parse_error_count": validation["error_count"] + len(extraction_errors),
        "filings_path": str(filings_path),
        "trades_path": str(root / "trades.jsonl"),
        "parse_errors_path": str(root / "parse_errors.jsonl"),
        "error_summary": validation.get("error_summary", {}),
    }


def _collect_filing_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted((root / "manifests").glob("house_ptr_*.json")):
        manifest = _read_json(manifest_path)
        for row in manifest.get("filings", []):
            rows.append({
                "source": "house",
                "report_id": row.get("report_id"),
                "filing_year": row.get("filing_year"),
                "report_type": row.get("document_type"),
                "filer_name": row.get("filer_name"),
                "document_url": row.get("source_url"),
                "raw_artifact_path": row.get("path"),
                "filed_date": row.get("filed_date"),
                "state_district": row.get("state_district"),
                "parse_status": row.get("status") or "pending_transaction_parser",
                "parser_version": HOUSE_PARSER_VERSION,
            })
    for manifest_path in sorted((root / "manifests").glob("senate_*.json")):
        manifest = _read_json(manifest_path)
        for row in manifest.get("filings", []):
            rows.append({
                "source": "senate",
                "report_id": row.get("report_id"),
                "filing_year": row.get("filing_year"),
                "report_type": row.get("report_type"),
                "filer_name": row.get("filer_name"),
                "document_url": row.get("document_url"),
                "raw_artifact_path": row.get("raw_artifact_path") or manifest.get("response", {}).get("raw_artifact_path"),
                "filed_date": row.get("filed_date"),
                "parse_status": row.get("document_status") or "pending_transaction_parser",
                "parser_version": SENATE_PARSER_VERSION,
            })
    return rows


def _parse_house_ptr_artifacts(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for manifest_path in sorted((root / "manifests").glob("house_ptr_*.json")):
        manifest = _read_json(manifest_path)
        for filing in manifest.get("filings", []):
            if filing.get("status") != "ok":
                continue
            artifact_path = Path(str(filing.get("path") or ""))
            if not artifact_path.exists():
                errors.append(_extraction_error("house", filing, "missing_raw_artifact"))
                continue
            try:
                parsed = parse_house_ptr_pdf(
                    artifact_path,
                    report_id=str(filing.get("report_id") or ""),
                    filing_year=int(filing.get("filing_year") or manifest.get("year") or 0),
                    filer_name=str(filing.get("filer_name") or "Unknown"),
                    document_url=str(filing.get("source_url") or filing.get("document_url") or ""),
                )
            except Exception as exc:
                errors.append(_extraction_error("house", filing, f"{type(exc).__name__}: {exc}"))
                continue
            if not parsed.get("rows"):
                if _is_house_image_only_paper_filing(artifact_path):
                    continue
                errors.append(_extraction_error("house", filing, "no_transactions_parsed"))
                continue
            for parsed_row in parsed["rows"]:
                rows.append(_normalize_house_trade_row(parsed_row, filing, artifact_path))
    return rows, errors


def _parse_senate_ptr_artifacts(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for manifest_path in sorted((root / "manifests").glob("senate_*.json")):
        manifest = _read_json(manifest_path)
        for filing in manifest.get("filings", []):
            document_status = filing.get("document_status")
            if document_status and document_status != "ok":
                if filing.get("document_url"):
                    errors.append(_extraction_error("senate", filing, "missing_or_unfetched_raw_artifact"))
                continue
            if not filing.get("raw_artifact_path"):
                if document_status == "ok":
                    errors.append(_extraction_error("senate", filing, "missing_raw_artifact"))
                continue
            artifact_path = Path(str(filing.get("raw_artifact_path") or ""))
            if not artifact_path.exists():
                errors.append(_extraction_error("senate", filing, "missing_raw_artifact"))
                continue
            try:
                if artifact_path.suffix.lower() == ".pdf":
                    parsed = parse_senate_ptr_document(artifact_path)
                else:
                    parsed = parse_senate_ptr_html(artifact_path.read_text(encoding="utf-8", errors="ignore"))
            except Exception as exc:
                errors.append(_extraction_error("senate", filing, f"{type(exc).__name__}: {exc}"))
                continue
            if not parsed.get("rows"):
                if _is_senate_image_only_paper_filing(filing, artifact_path):
                    continue
                errors.append(_extraction_error("senate", filing, "no_transactions_parsed"))
                continue
            for parsed_row in parsed["rows"]:
                rows.append(_normalize_senate_trade_row(parsed_row, filing, artifact_path))
    return rows, errors


def _is_house_image_only_paper_filing(artifact_path: Path) -> bool:
    try:
        return not extract_pdf_text(artifact_path).strip()
    except Exception:
        return False


def _normalize_house_trade_row(
    row: dict[str, Any],
    filing: dict[str, Any],
    artifact_path: Path,
) -> dict[str, Any]:
    asset_name = row.get("asset_name_raw")
    asset_meta = _house_asset_metadata(asset_name)
    ticker_resolution = resolve_ticker(
        asset_name=asset_name,
        explicit_ticker=asset_meta.get("ticker"),
        asset_type=asset_meta.get("asset_type"),
    )
    asset_type = asset_meta["asset_type"]
    if asset_type in {"stock", "etf"} and not ticker_resolution.get("ticker"):
        asset_type = "unknown"
    amount = normalize_amount_bucket(row.get("amount_bucket_raw"))
    dates = normalize_date_fields(
        transaction_date=row.get("transaction_date"),
        notification_date=row.get("notification_date"),
        filed_date=filing.get("filed_date"),
        disclosure_date=filing.get("filed_date"),
    )
    state_district = str(filing.get("state_district") or "")
    warnings = list(row.get("warnings") or [])
    warnings.extend(dates.get("date_warnings") or [])
    return {
        "source": "house",
        "chamber": "house",
        "report_id": filing.get("report_id") or row.get("report_id"),
        "report_type": "ptr",
        "is_amendment": False,
        "filer_name": filing.get("filer_name") or row.get("filer_name"),
        "owner": row.get("owner") or "unknown",
        "asset_name_raw": asset_name,
        "asset_type": asset_type,
        "ticker": ticker_resolution.get("ticker"),
        "ticker_resolution_status": ticker_resolution.get("ticker_resolution_status"),
        "ticker_resolution_reason": ticker_resolution.get("ticker_resolution_reason"),
        "ticker_resolution_confidence": ticker_resolution.get("ticker_resolution_confidence"),
        "transaction_type": row.get("transaction_type") or "unknown",
        "transaction_type_raw": row.get("transaction_type_raw"),
        "transaction_date": dates["transaction_date"],
        "notification_date": dates["notification_date"],
        "filed_date": dates["filed_date"],
        "disclosure_date": dates["disclosure_date"],
        "delay_days": dates["delay_days"],
        "amount_bucket_raw": amount["amount_bucket_raw"],
        "amount_min_usd": amount["amount_min_usd"],
        "amount_max_usd": amount["amount_max_usd"],
        "amount_mid_usd": amount["amount_mid_usd"],
        "amount_mid_usd_is_estimate": amount["amount_mid_usd_is_estimate"],
        "amount_estimate_label": amount["amount_estimate_label"],
        "filing_year": filing.get("filing_year") or row.get("filing_year"),
        "state": state_district[:2] or None,
        "district": state_district[2:] or None,
        "party": None,
        "document_url": filing.get("source_url") or row.get("document_url"),
        "source_url": filing.get("source_url") or row.get("document_url"),
        "raw_artifact_path": str(artifact_path),
        "source_hash": compute_source_hash(artifact_path),
        "parser_version": row.get("parser_version") or HOUSE_PARSER_VERSION,
        "parser_confidence": row.get("parser_confidence"),
        "validation_status": row.get("validation_status"),
        "warnings": warnings,
        "source_extra": {
            "house_asset_type_code": asset_meta.get("house_asset_type_code"),
            "house_filing_type_code": filing.get("filing_type_code"),
        },
    }


def _normalize_senate_trade_row(
    row: dict[str, Any],
    filing: dict[str, Any],
    artifact_path: Path,
) -> dict[str, Any]:
    asset_name = row.get("asset_name_raw")
    explicit_ticker = _normalize_senate_ticker(row.get("ticker"))
    asset_type = row.get("asset_type") or "unknown"
    ticker_resolution = resolve_ticker(
        asset_name=asset_name,
        explicit_ticker=explicit_ticker,
        asset_type=asset_type,
    )
    amount = normalize_amount_bucket(row.get("amount_bucket_raw"))
    dates = normalize_date_fields(
        transaction_date=row.get("transaction_date"),
        notification_date=None,
        filed_date=filing.get("filed_date"),
        disclosure_date=filing.get("filed_date"),
    )
    warnings = list(row.get("warnings") or [])
    warnings.extend(dates.get("date_warnings") or [])
    report_type = str(filing.get("report_type") or "Periodic Transaction Report")
    return {
        "source": "senate",
        "chamber": "senate",
        "report_id": filing.get("report_id"),
        "report_type": "ptr",
        "is_amendment": "amendment" in report_type.lower(),
        "filer_name": filing.get("filer_name") or "Unknown",
        "owner": row.get("owner") or "unknown",
        "asset_name_raw": asset_name,
        "asset_type": asset_type,
        "ticker": ticker_resolution.get("ticker"),
        "ticker_resolution_status": ticker_resolution.get("ticker_resolution_status"),
        "ticker_resolution_reason": ticker_resolution.get("ticker_resolution_reason"),
        "ticker_resolution_confidence": ticker_resolution.get("ticker_resolution_confidence"),
        "transaction_type": row.get("transaction_type") or "unknown",
        "transaction_type_raw": row.get("transaction_type_raw"),
        "transaction_date": dates["transaction_date"],
        "notification_date": dates["notification_date"],
        "filed_date": dates["filed_date"],
        "disclosure_date": dates["disclosure_date"],
        "delay_days": dates["delay_days"],
        "amount_bucket_raw": amount["amount_bucket_raw"],
        "amount_min_usd": amount["amount_min_usd"],
        "amount_max_usd": amount["amount_max_usd"],
        "amount_mid_usd": amount["amount_mid_usd"],
        "amount_mid_usd_is_estimate": amount["amount_mid_usd_is_estimate"],
        "amount_estimate_label": amount["amount_estimate_label"],
        "filing_year": filing.get("filing_year"),
        "state": None,
        "district": None,
        "party": None,
        "document_url": filing.get("document_url"),
        "source_url": filing.get("document_url"),
        "raw_artifact_path": str(artifact_path),
        "source_hash": compute_source_hash(artifact_path),
        "parser_version": row.get("parser_version") or SENATE_PARSER_VERSION,
        "parser_confidence": row.get("parser_confidence"),
        "validation_status": row.get("validation_status"),
        "warnings": warnings,
        "source_extra": {
            **(row.get("source_extra") or {}),
            "senate_document_content_type": filing.get("document_content_type"),
        },
    }


def _house_asset_metadata(asset_name: Any) -> dict[str, Any]:
    text = str(asset_name or "")
    code_match = re.search(r"\[([A-Z]{1,4})\]\s*$", text)
    code = code_match.group(1) if code_match else None
    ticker = None
    ticker_match = re.search(r"\(([A-Z][A-Z0-9.\-]{0,7})\)\s*(?:\[[A-Z]{1,4}\])?\s*$", text)
    if ticker_match and code in {"ST", "EF", "ETF", "OP"}:
        ticker = ticker_match.group(1)
    asset_type = {
        "ST": "stock",
        "EF": "etf",
        "ETF": "etf",
        "OP": "option",
        "MF": "fund",
        "GS": "bond",
        "CB": "bond",
        "CS": "stock",
    }.get(code or "", "unknown")
    return {
        "ticker": ticker,
        "asset_type": asset_type,
        "house_asset_type_code": code,
    }


def _extraction_error(source: str, filing: dict[str, Any], message: str) -> dict[str, Any]:
    return {
        "source": source,
        "report_id": filing.get("report_id"),
        "errors": [message],
        "row_context": {
            "filer_name": filing.get("filer_name"),
            "raw_artifact_path": filing.get("path") or filing.get("raw_artifact_path"),
            "document_url": filing.get("source_url") or filing.get("document_url"),
        },
        "created_at": utc_now_iso(),
    }


def _append_extraction_errors(path: Path, errors: list[dict[str, Any]]) -> None:
    if not errors:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in errors:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _parse_status(validation: dict[str, Any], extraction_errors: list[dict[str, Any]]) -> str:
    if validation.get("status") == "ok" and not extraction_errors:
        return "ok"
    if validation.get("valid_count", 0) > 0:
        return "valid_with_errors"
    return "degraded"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _normalize_senate_ticker(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text or text in {"--", "-", "N/A", "n/a", "None"}:
        return None
    return text


def _is_senate_image_only_paper_filing(filing: dict[str, Any], artifact_path: Path) -> bool:
    document_url = str(filing.get("document_url") or "").lower()
    content_type = str(filing.get("document_content_type") or "").lower()
    if "/paper/" not in document_url or "html" not in content_type:
        return False
    try:
        html = artifact_path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    return "efd-media-public.senate.gov" in html and ".gif" in html
