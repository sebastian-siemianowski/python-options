"""House PTR PDF parser.

The parser is built in two layers:
1. PDF text extraction, using optional local PDF libraries when installed.
2. Deterministic text-to-row parsing, used by tests and extraction fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ingestion.politicians.normalize import normalize_amount_bucket
from ingestion.politicians.quality import score_parser_confidence, validation_status_for_confidence


PARSER_VERSION = "house-ptr-parser-v1"
QUARANTINE_CONFIDENCE_THRESHOLD = 0.80

TRANSACTION_TYPE_MAP = {
    "P": "purchase",
    "PURCHASE": "purchase",
    "S": "sale",
    "SALE": "sale",
    "S-P": "sale_partial",
    "SP": "sale_partial",
    "E": "exchange",
    "EXCHANGE": "exchange",
}


@dataclass(frozen=True)
class HousePtrRow:
    """Parsed House PTR transaction row."""

    report_id: str
    filing_year: int
    filer_name: str
    owner: str | None
    asset_name_raw: str | None
    transaction_type_raw: str | None
    transaction_type: str
    transaction_date: str | None
    notification_date: str | None
    amount_bucket_raw: str | None
    document_url: str | None
    parser_confidence: float
    validation_status: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "filing_year": self.filing_year,
            "filer_name": self.filer_name,
            "owner": self.owner,
            "asset_name_raw": self.asset_name_raw,
            "transaction_type_raw": self.transaction_type_raw,
            "transaction_type": self.transaction_type,
            "transaction_date": self.transaction_date,
            "notification_date": self.notification_date,
            "amount_bucket_raw": self.amount_bucket_raw,
            "document_url": self.document_url,
            "parser_version": PARSER_VERSION,
            "parser_confidence": self.parser_confidence,
            "validation_status": self.validation_status,
            "warnings": list(self.warnings),
        }


def parse_house_ptr_pdf(
    path: str | Path,
    *,
    report_id: str,
    filing_year: int,
    filer_name: str,
    document_url: str | None = None,
) -> dict[str, Any]:
    """Extract and parse House PTR rows from a PDF artifact."""
    text = extract_pdf_text(path)
    rows = parse_house_ptr_text(
        text,
        report_id=report_id,
        filing_year=filing_year,
        filer_name=filer_name,
        document_url=document_url,
    )
    return {
        "parser_version": PARSER_VERSION,
        "report_id": report_id,
        "filing_year": filing_year,
        "filer_name": filer_name,
        "row_count": len(rows),
        "valid_row_count": sum(1 for row in rows if row.validation_status != "quarantined"),
        "quarantined_row_count": sum(1 for row in rows if row.validation_status == "quarantined"),
        "rows": [row.to_dict() for row in rows],
    }


def extract_pdf_text(path: str | Path) -> str:
    """Extract text from a PDF, with optional libraries and plain-text fallback."""
    pdf_path = Path(path)
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(str(pdf_path)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        pass
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(pdf_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        pass
    return pdf_path.read_bytes().decode("utf-8", errors="ignore")


def parse_house_ptr_text(
    text: str,
    *,
    report_id: str,
    filing_year: int,
    filer_name: str,
    document_url: str | None = None,
) -> list[HousePtrRow]:
    """Parse House PTR rows from extracted text."""
    modern = _parse_modern_table_text(text, report_id, filing_year, filer_name, document_url)
    extracted = _parse_extracted_pdf_table_text(text, report_id, filing_year, filer_name, document_url)
    legacy = _parse_legacy_blocks(text, report_id, filing_year, filer_name, document_url)
    rows = modern or extracted or legacy
    return rows


def normalize_transaction_type(value: str | None) -> str:
    """Map House transaction codes to canonical transaction types."""
    if not value:
        return "unknown"
    if "partial" in value.lower():
        return "sale_partial"
    normalized = value.strip().upper().replace(" ", "")
    return TRANSACTION_TYPE_MAP.get(normalized, TRANSACTION_TYPE_MAP.get(value.strip().upper(), "unknown"))


def _parse_modern_table_text(
    text: str,
    report_id: str,
    filing_year: int,
    filer_name: str,
    document_url: str | None,
) -> list[HousePtrRow]:
    rows: list[HousePtrRow] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        if "asset" in line.lower() and "amount" in line.lower():
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 6:
            continue
        owner, asset, tx_raw, tx_date, notify_date, amount = parts[:6]
        rows.append(_build_row(
            report_id=report_id,
            filing_year=filing_year,
            filer_name=filer_name,
            owner=owner,
            asset_name_raw=asset,
            transaction_type_raw=tx_raw,
            transaction_date=_normalize_date(tx_date),
            notification_date=_normalize_date(notify_date),
            amount_bucket_raw=amount,
            document_url=document_url,
        ))
    return rows


def _parse_extracted_pdf_table_text(
    text: str,
    report_id: str,
    filing_year: int,
    filer_name: str,
    document_url: str | None,
) -> list[HousePtrRow]:
    lines = [_normalize_extracted_line(line) for line in text.replace("\x00", "").splitlines()]
    rows: list[HousePtrRow] = []
    asset_parts: list[str] = []
    idx = 0
    table_seen = False
    while idx < len(lines):
        line = lines[idx]
        idx += 1
        if not line or _is_extracted_noise_line(line):
            if line.lower() == "id owner asset transaction":
                table_seen = True
            continue
        if not table_seen:
            if line.lower() == "id owner asset transaction":
                table_seen = True
            continue
        match = _find_transaction_segment(line)
        if not match:
            asset_parts.append(line)
            continue
        prefix = line[:match.start()].strip()
        if prefix:
            asset_parts.append(prefix)
        amount = match.group("amount").strip()
        while _amount_needs_continuation(amount) and idx < len(lines):
            continuation = lines[idx]
            if not continuation or _is_extracted_noise_line(continuation):
                idx += 1
                continue
            if _find_transaction_segment(continuation):
                break
            amount = f"{amount} {continuation}".strip()
            idx += 1
        asset_text = _clean_asset_text(" ".join(asset_parts))
        owner, asset_name = _extract_house_owner(asset_text)
        rows.append(_build_row(
            report_id=report_id,
            filing_year=filing_year,
            filer_name=filer_name,
            owner=owner,
            asset_name_raw=asset_name,
            transaction_type_raw=match.group("type"),
            transaction_date=_normalize_date(match.group("transaction_date")),
            notification_date=_normalize_date(match.group("notification_date")),
            amount_bucket_raw=amount,
            document_url=document_url,
        ))
        asset_parts = []
    return rows


def _find_transaction_segment(line: str) -> re.Match[str] | None:
    return re.search(
        r"(?P<type>S\s*\(partial\)|S-P|SP|P|S|E)\s+"
        r"(?P<transaction_date>\d{1,2}/\d{1,2}/20\d{2})\s*"
        r"(?P<notification_date>\d{1,2}/\d{1,2}/20\d{2})\s*"
        r"(?P<amount>.*)$",
        line,
        flags=re.IGNORECASE,
    )


def _normalize_extracted_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.replace("\x00", " ")).strip()


def _is_extracted_noise_line(line: str) -> bool:
    lowered = line.lower()
    exact_noise = {
        "id owner asset transaction",
        "type",
        "date notification",
        "date",
        "amount cap.",
        "gains >",
        "$200?",
        "yes no",
    }
    if lowered in exact_noise:
        return True
    prefixes = (
        "clerk of the house",
        "filing status:",
        "f s:",
        "s o:",
        "d:",
        "* for the complete list",
        "i certify",
        "digitally signed:",
        "filing id #",
        "initial public offering",
        "certification and signature",
    )
    return any(lowered.startswith(prefix) for prefix in prefixes)


def _amount_needs_continuation(amount: str) -> bool:
    if not amount:
        return False
    if amount.strip().endswith("-"):
        return True
    return len(re.findall(r"\$?\s*[\d,]+", amount)) < 2 and "over" not in amount.lower()


def _clean_asset_text(value: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", value).strip(" -")
    return cleaned or None


def _extract_house_owner(asset_text: str | None) -> tuple[str, str | None]:
    if not asset_text:
        return "unknown", None
    match = re.match(r"^(SP|DC|JT|SELF)\s+(.+)$", asset_text, flags=re.IGNORECASE)
    if not match:
        return "self", asset_text
    owner_code = match.group(1).upper()
    owner = {
        "SP": "spouse",
        "DC": "dependent_child",
        "JT": "joint",
        "SELF": "self",
    }.get(owner_code, "unknown")
    return owner, match.group(2).strip()


def _parse_legacy_blocks(
    text: str,
    report_id: str,
    filing_year: int,
    filer_name: str,
    document_url: str | None,
) -> list[HousePtrRow]:
    blocks = re.split(r"\n\s*(?=Owner\s*:)", text, flags=re.IGNORECASE)
    rows: list[HousePtrRow] = []
    for block in blocks:
        if "transaction" not in block.lower() or "asset" not in block.lower():
            continue
        fields = _extract_legacy_fields(block)
        rows.append(_build_row(
            report_id=report_id,
            filing_year=filing_year,
            filer_name=filer_name,
            owner=fields.get("owner"),
            asset_name_raw=fields.get("asset"),
            transaction_type_raw=fields.get("transaction"),
            transaction_date=_normalize_date(fields.get("transaction_date")),
            notification_date=_normalize_date(fields.get("notification_date")),
            amount_bucket_raw=fields.get("amount"),
            document_url=document_url,
        ))
    return rows


def _extract_legacy_fields(block: str) -> dict[str, str]:
    labels = {
        "owner": r"Owner\s*:",
        "asset": r"Asset\s*:",
        "transaction": r"Transaction\s*:",
        "transaction_date": r"Transaction Date\s*:",
        "notification_date": r"Notification Date\s*:",
        "amount": r"Amount\s*:",
    }
    spans: list[tuple[str, int, int]] = []
    for key, pattern in labels.items():
        match = re.search(pattern, block, flags=re.IGNORECASE)
        if match:
            spans.append((key, match.start(), match.end()))
    spans.sort(key=lambda item: item[1])
    fields: dict[str, str] = {}
    for idx, (key, _start, value_start) in enumerate(spans):
        value_end = spans[idx + 1][1] if idx + 1 < len(spans) else len(block)
        value = block[value_start:value_end].strip()
        fields[key] = re.sub(r"\s+", " ", value).strip()
    return fields


def _build_row(
    *,
    report_id: str,
    filing_year: int,
    filer_name: str,
    owner: str | None,
    asset_name_raw: str | None,
    transaction_type_raw: str | None,
    transaction_date: str | None,
    notification_date: str | None,
    amount_bucket_raw: str | None,
    document_url: str | None,
) -> HousePtrRow:
    transaction_type = normalize_transaction_type(transaction_type_raw)
    confidence, warnings = _score_confidence(
        owner=owner,
        asset_name_raw=asset_name_raw,
        transaction_type=transaction_type,
        transaction_date=transaction_date,
        notification_date=notification_date,
        amount_bucket_raw=amount_bucket_raw,
    )
    validation_status = validation_status_for_confidence(confidence)
    return HousePtrRow(
        report_id=report_id,
        filing_year=filing_year,
        filer_name=filer_name,
        owner=_clean(owner),
        asset_name_raw=_clean(asset_name_raw),
        transaction_type_raw=_clean(transaction_type_raw),
        transaction_type=transaction_type,
        transaction_date=transaction_date,
        notification_date=notification_date,
        amount_bucket_raw=_clean(amount_bucket_raw),
        document_url=document_url,
        parser_confidence=confidence,
        validation_status=validation_status,
        warnings=warnings,
    )


def _score_confidence(**fields: str | None) -> tuple[float, list[str]]:
    weights = {
        "owner": 0.10,
        "asset_name_raw": 0.25,
        "transaction_type": 0.20,
        "transaction_date": 0.15,
        "notification_date": 0.10,
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
    date_validity = 0.0 if "missing_or_unknown_transaction_date" in warnings else 1.0
    amount_status = normalize_amount_bucket(fields.get("amount_bucket_raw")).get("amount_bucket_parse_status")
    amount_recognition = 1.0 if amount_status in {"range", "open_ended"} else 0.0
    return score_parser_confidence(
        field_completeness=field_score,
        table_alignment=1.0,
        date_validity=date_validity,
        amount_bucket_recognition=amount_recognition,
        ticker_resolution_confidence=1.0,
    ), warnings


def _normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if match:
        y, m, d = match.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    match = re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", text)
    if match:
        m, d, y = match.groups()
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return None


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned or None
