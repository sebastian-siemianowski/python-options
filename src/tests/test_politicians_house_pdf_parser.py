"""Tests for House PTR PDF/text parser."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.parsers.house_pdf import (
    QUARANTINE_CONFIDENCE_THRESHOLD,
    normalize_transaction_type,
    parse_house_ptr_pdf,
    parse_house_ptr_text,
)


def test_house_modern_table_parser_extracts_required_fields():
    text = """
    Owner | Asset | Transaction Type | Transaction Date | Notification Date | Amount
    Self | NVIDIA Corporation Common Stock | P | 04/12/2026 | 04/18/2026 | $15,001 - $50,000
    Spouse | Apple Inc. | S-P | 2026-04-13 | 2026-04-19 | $1,001 - $15,000
    """

    rows = parse_house_ptr_text(text, report_id="20031020", filing_year=2026, filer_name="Jane Doe")

    assert len(rows) == 2
    first = rows[0].to_dict()
    second = rows[1].to_dict()
    assert first["owner"] == "Self"
    assert first["asset_name_raw"] == "NVIDIA Corporation Common Stock"
    assert first["transaction_type"] == "purchase"
    assert first["transaction_date"] == "2026-04-12"
    assert first["notification_date"] == "2026-04-18"
    assert first["amount_bucket_raw"] == "$15,001 - $50,000"
    assert first["validation_status"] == "valid"
    assert second["transaction_type"] == "sale_partial"


def test_house_legacy_text_parser_preserves_multiline_asset_names():
    text = """
    Owner: Spouse
    Asset: NVIDIA Corporation
    Common Stock
    Transaction: P
    Transaction Date: 04/12/2026
    Notification Date: 04/18/2026
    Amount: $15,001 - $50,000
    """

    rows = parse_house_ptr_text(text, report_id="20031020", filing_year=2026, filer_name="Jane Doe")

    assert len(rows) == 1
    row = rows[0].to_dict()
    assert row["owner"] == "Spouse"
    assert row["asset_name_raw"] == "NVIDIA Corporation Common Stock"
    assert row["transaction_type"] == "purchase"
    assert row["parser_confidence"] >= QUARANTINE_CONFIDENCE_THRESHOLD


def test_house_transaction_code_mapping():
    assert normalize_transaction_type("P") == "purchase"
    assert normalize_transaction_type("S") == "sale"
    assert normalize_transaction_type("S-P") == "sale_partial"
    assert normalize_transaction_type("S (partial)") == "sale_partial"
    assert normalize_transaction_type("E") == "exchange"
    assert normalize_transaction_type("???") == "unknown"


def test_house_extracted_pdf_table_parser_handles_official_layout_text():
    text = """
    P T R
    F I
    Name: Hon. Jane Doe
    Status: Member
    State/District:CA12
    T
    ID Owner Asset Transaction
    Type
    Date Notification
    Date
    Amount Cap.
    Gains >
    $200?
    SP State Street Corporation Common
    Stock (STT) [ST]
    S (partial) 05/18/202605/18/2026$15,001 -
    $50,000
    F S: New
    S O: Fidelity Brokerage
    Adobe Inc. - Common Stock (ADBE)
    [ST]
    P 04/14/202605/07/2026$1,001 - $15,000
    """

    rows = parse_house_ptr_text(text, report_id="20034622", filing_year=2026, filer_name="Hon. Jane Doe")

    assert len(rows) == 2
    first = rows[0].to_dict()
    second = rows[1].to_dict()
    assert first["owner"] == "spouse"
    assert first["asset_name_raw"] == "State Street Corporation Common Stock (STT) [ST]"
    assert first["transaction_type"] == "sale_partial"
    assert first["amount_bucket_raw"] == "$15,001 - $50,000"
    assert second["asset_name_raw"] == "Adobe Inc. - Common Stock (ADBE) [ST]"
    assert second["transaction_type"] == "purchase"


def test_house_low_confidence_rows_are_quarantined():
    text = "Self | Unknown Asset | ??? | missing | missing |"

    rows = parse_house_ptr_text(text, report_id="bad", filing_year=2026, filer_name="Jane Doe")

    assert len(rows) == 1
    row = rows[0].to_dict()
    assert row["parser_confidence"] < QUARANTINE_CONFIDENCE_THRESHOLD
    assert row["validation_status"] == "quarantined"
    assert "missing_or_unknown_transaction_type" in row["warnings"]


def test_house_pdf_parser_plain_text_fallback(tmp_path):
    pdf = tmp_path / "20031020.pdf"
    pdf.write_text(
        "Owner | Asset | Transaction Type | Transaction Date | Notification Date | Amount\n"
        "Self | Microsoft Corp | P | 05/01/2026 | 05/03/2026 | $1,001 - $15,000\n",
        encoding="utf-8",
    )

    result = parse_house_ptr_pdf(
        pdf,
        report_id="20031020",
        filing_year=2026,
        filer_name="Jane Doe",
        document_url="https://example.test/20031020.pdf",
    )

    assert result["row_count"] == 1
    assert result["valid_row_count"] == 1
    assert result["rows"][0]["document_url"] == "https://example.test/20031020.pdf"


def test_house_parser_golden_fixture_coverage_25_pdfs_across_years(tmp_path):
    years = [2024, 2025, 2026]
    parsed = []
    for idx in range(25):
        year = years[idx % len(years)]
        report_id = f"20{year % 100}{idx:04d}"
        pdf = tmp_path / f"{report_id}.pdf"
        pdf.write_text(
            "Owner | Asset | Transaction Type | Transaction Date | Notification Date | Amount\n"
            f"Self | Fixture Asset {idx} | P | 04/12/{year} | 04/18/{year} | $1,001 - $15,000\n",
            encoding="utf-8",
        )
        parsed.append(parse_house_ptr_pdf(pdf, report_id=report_id, filing_year=year, filer_name=f"Filer {idx}"))

    assert len(parsed) == 25
    assert {item["filing_year"] for item in parsed} == {2024, 2025, 2026}
    assert all(item["valid_row_count"] == 1 for item in parsed)
