"""Senate PTR parser for HTML, printable text, and PDF fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Any

from ingestion.politicians.normalize import normalize_amount_bucket
from ingestion.politicians.parsers.house_pdf import extract_pdf_text
from ingestion.politicians.quality import score_parser_confidence, validation_status_for_confidence


PARSER_VERSION = "senate-ptr-parser-v1"

OWNER_MAP = {
    "self": "self",
    "spouse": "spouse",
    "dependent child": "dependent_child",
    "dependent": "dependent_child",
    "child": "dependent_child",
    "joint": "joint",
    "jointly held": "joint",
}

TRANSACTION_MAP = {
    "purchase": "purchase",
    "buy": "purchase",
    "sale": "sale",
    "sell": "sale",
    "exchange": "exchange",
    "received": "received",
}


@dataclass(frozen=True)
class SenatePtrRow:
    """Parsed Senate PTR transaction row."""

    owner: str
    transaction_date: str | None
    ticker: str | None
    asset_name_raw: str | None
    asset_type: str
    transaction_type_raw: str | None
    transaction_type: str
    amount_bucket_raw: str | None
    comments: str | None
    source_extra: dict[str, Any]
    parser_confidence: float
    validation_status: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "transaction_date": self.transaction_date,
            "ticker": self.ticker,
            "asset_name_raw": self.asset_name_raw,
            "asset_type": self.asset_type,
            "transaction_type_raw": self.transaction_type_raw,
            "transaction_type": self.transaction_type,
            "amount_bucket_raw": self.amount_bucket_raw,
            "comments": self.comments,
            "source_extra": dict(self.source_extra),
            "parser_version": PARSER_VERSION,
            "parser_confidence": self.parser_confidence,
            "validation_status": self.validation_status,
            "warnings": list(self.warnings),
        }


def parse_senate_ptr_document(path: str | Path) -> dict[str, Any]:
    """Parse a Senate printable/PDF artifact."""
    text = extract_pdf_text(path)
    rows = parse_senate_ptr_text(text)
    return _result(rows)


def parse_senate_ptr_html(html: str) -> dict[str, Any]:
    """Parse Senate PTR rows from an HTML table layout."""
    rows = _parse_html_tables(html)
    if not rows:
        rows = parse_senate_ptr_text(_strip_html(html))
    return _result(rows)


def parse_senate_ptr_text(text: str) -> list[SenatePtrRow]:
    """Parse Senate PTR rows from printable text blocks."""
    blocks = re.split(r"\n\s*(?=Owner\s*:)", text, flags=re.IGNORECASE)
    rows: list[SenatePtrRow] = []
    for block in blocks:
        if "asset" not in block.lower() or "transaction" not in block.lower():
            continue
        fields = _extract_labelled_fields(block)
        rows.append(_build_row(
            owner=fields.get("owner"),
            transaction_date=fields.get("transaction date") or fields.get("date"),
            ticker=fields.get("ticker"),
            asset_name_raw=fields.get("asset") or fields.get("asset name"),
            asset_type=fields.get("asset type"),
            transaction_type_raw=fields.get("transaction") or fields.get("transaction type"),
            amount_bucket_raw=fields.get("amount"),
            comments=fields.get("comments"),
            source_extra={key: value for key, value in fields.items() if key not in {
                "owner", "ticker", "asset", "asset name", "asset type", "transaction",
                "transaction type", "transaction date", "date", "amount", "comments",
            }},
        ))
    return rows


def normalize_senate_owner(value: str | None) -> str:
    """Map Senate owner labels to the canonical owner enum."""
    if not value:
        return "unknown"
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    return OWNER_MAP.get(normalized, "unknown")


def normalize_senate_transaction_type(value: str | None) -> str:
    """Map Senate transaction text to canonical transaction type."""
    if not value:
        return "unknown"
    normalized = value.strip().lower()
    for key, mapped in TRANSACTION_MAP.items():
        if key in normalized:
            return mapped
    return "unknown"


def infer_asset_type(asset_type_raw: str | None, asset_name: str | None, ticker: str | None) -> str:
    """Infer a canonical asset type from Senate fields."""
    combined = f"{asset_type_raw or ''} {asset_name or ''}".lower()
    if "option" in combined or "call" in combined or "put" in combined:
        return "option"
    if "etf" in combined or "exchange traded fund" in combined:
        return "etf"
    bond_markers = (
        "bond", "treasury", "municipal", "matures:", "rate/coupon:", "rev bds",
        "revenue bds", "gen oblig", "obligation", "notes", " note",
    )
    if any(marker in combined for marker in bond_markers):
        return "bond"
    if "fund" in combined or "mutual" in combined:
        return "fund"
    if "stock" in combined:
        return "stock" if ticker else "unknown"
    if ticker:
        return "stock"
    return "unknown"


def _parse_html_tables(html: str) -> list[SenatePtrRow]:
    rows: list[SenatePtrRow] = []
    for table in re.findall(r"<table\b.*?</table>", html, flags=re.IGNORECASE | re.DOTALL):
        raw_rows = re.findall(r"<tr\b.*?</tr>", table, flags=re.IGNORECASE | re.DOTALL)
        if not raw_rows:
            continue
        headers = [_normalize_header(cell) for cell in _extract_cells(raw_rows[0])]
        if "owner" not in headers or not any(header in headers for header in ("asset", "asset_name")):
            continue
        for raw_row in raw_rows[1:]:
            cells = _extract_cells(raw_row)
            if len(cells) < 4:
                continue
            row_map = {headers[idx]: cells[idx] for idx in range(min(len(headers), len(cells)))}
            known_keys = {
                "owner", "transaction_date", "date", "ticker", "asset", "asset_name",
                "asset_type", "transaction", "transaction_type", "amount", "comment", "comments", "",
            }
            rows.append(_build_row(
                owner=row_map.get("owner"),
                transaction_date=row_map.get("transaction_date") or row_map.get("date"),
                ticker=row_map.get("ticker"),
                asset_name_raw=row_map.get("asset_name") or row_map.get("asset"),
                asset_type=row_map.get("asset_type"),
                transaction_type_raw=row_map.get("transaction_type") or row_map.get("transaction"),
                amount_bucket_raw=row_map.get("amount"),
                comments=row_map.get("comments"),
                source_extra={key: value for key, value in row_map.items() if key not in known_keys},
            ))
    return rows


def _build_row(
    *,
    owner: str | None,
    transaction_date: str | None,
    ticker: str | None,
    asset_name_raw: str | None,
    asset_type: str | None,
    transaction_type_raw: str | None,
    amount_bucket_raw: str | None,
    comments: str | None,
    source_extra: dict[str, Any],
) -> SenatePtrRow:
    canonical_owner = normalize_senate_owner(owner)
    canonical_transaction = normalize_senate_transaction_type(transaction_type_raw)
    clean_ticker = _normalize_ticker(_clean(ticker))
    clean_asset = _clean(asset_name_raw)
    canonical_asset_type = infer_asset_type(asset_type, clean_asset, clean_ticker)
    confidence, warnings = _score_confidence(
        owner=canonical_owner,
        asset_name_raw=clean_asset,
        asset_type=canonical_asset_type,
        transaction_type=canonical_transaction,
        amount_bucket_raw=amount_bucket_raw,
        ticker=clean_ticker,
    )
    return SenatePtrRow(
        owner=canonical_owner,
        transaction_date=_clean(transaction_date),
        ticker=clean_ticker,
        asset_name_raw=clean_asset,
        asset_type=canonical_asset_type,
        transaction_type_raw=_clean(transaction_type_raw),
        transaction_type=canonical_transaction,
        amount_bucket_raw=_clean(amount_bucket_raw),
        comments=_clean(comments),
        source_extra=source_extra,
        parser_confidence=confidence,
        validation_status=validation_status_for_confidence(confidence),
        warnings=warnings,
    )


def _score_confidence(**fields: str | None) -> tuple[float, list[str]]:
    weights = {
        "owner": 0.15,
        "asset_name_raw": 0.25,
        "asset_type": 0.15,
        "transaction_type": 0.20,
        "amount_bucket_raw": 0.20,
    }
    field_score = 0.0
    warnings: list[str] = []
    for key, weight in weights.items():
        value = fields.get(key)
        if value and value != "unknown":
            field_score += weight
        else:
            warnings.append(f"missing_or_unknown_{key}")
    ticker_confidence = 1.0
    ticker_optional_asset_types = {"bond", "fund", "private_asset", "unknown"}
    if not fields.get("ticker") and fields.get("asset_type") not in ticker_optional_asset_types:
        warnings.append("missing_ticker")
        ticker_confidence = 0.5
    if fields.get("ticker"):
        field_score += 0.05
    amount_status = normalize_amount_bucket(fields.get("amount_bucket_raw")).get("amount_bucket_parse_status")
    amount_recognition = 1.0 if amount_status in {"range", "open_ended"} else 0.0
    return score_parser_confidence(
        field_completeness=min(field_score, 1.0),
        table_alignment=1.0,
        date_validity=1.0,
        amount_bucket_recognition=amount_recognition,
        ticker_resolution_confidence=ticker_confidence,
    ), warnings


def _extract_cells(row: str) -> list[str]:
    cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row, flags=re.IGNORECASE | re.DOTALL)
    return [_strip_html(cell) for cell in cells]


def _normalize_header(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if normalized in {"asset_name", "asset"}:
        return "asset_name"
    if normalized in {"transaction_date", "date"}:
        return "transaction_date"
    if normalized in {"transaction", "transaction_type", "type"}:
        return "transaction_type"
    if normalized in {"amount", "amount_range"}:
        return "amount"
    if normalized in {"comment", "comments"}:
        return "comments"
    return normalized


def _extract_labelled_fields(block: str) -> dict[str, str]:
    matches = list(re.finditer(r"^\s*([A-Za-z][A-Za-z ]{1,40})\s*:", block, flags=re.IGNORECASE | re.MULTILINE))
    fields: dict[str, str] = {}
    for idx, match in enumerate(matches):
        key = match.group(1).strip().lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(block)
        fields[key] = re.sub(r"\s+", " ", block[start:end]).strip()
    return fields


def _result(rows: list[SenatePtrRow]) -> dict[str, Any]:
    return {
        "parser_version": PARSER_VERSION,
        "row_count": len(rows),
        "rows": [row.to_dict() for row in rows],
    }


def _strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", unescape(value))
    return re.sub(r"\s+", " ", text).strip()


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned or None


def _normalize_ticker(value: str | None) -> str | None:
    if not value:
        return None
    if value in {"--", "-", "N/A", "n/a", "None"}:
        return None
    return value
