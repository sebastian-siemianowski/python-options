"""Tests for Senate PTR parser."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.parsers.senate import (
    infer_asset_type,
    normalize_senate_owner,
    parse_senate_ptr_document,
    parse_senate_ptr_html,
    parse_senate_ptr_text,
)


def test_senate_html_table_parser_extracts_core_fields_and_source_extra():
    html = """
    <table>
      <tr>
        <th>Owner</th><th>Ticker</th><th>Asset Name</th><th>Asset Type</th>
        <th>Transaction Type</th><th>Amount</th><th>Comments</th><th>Rate</th>
      </tr>
      <tr>
        <td>Spouse</td><td>NVDA</td><td>NVIDIA Corporation</td><td>Stock</td>
        <td>Purchase</td><td>$15,001 - $50,000</td><td>Filed electronically</td><td>N/A</td>
      </tr>
    </table>
    """

    result = parse_senate_ptr_html(html)
    row = result["rows"][0]

    assert result["row_count"] == 1
    assert row["owner"] == "spouse"
    assert row["ticker"] == "NVDA"
    assert row["asset_name_raw"] == "NVIDIA Corporation"
    assert row["asset_type"] == "stock"
    assert row["transaction_type"] == "purchase"
    assert row["amount_bucket_raw"] == "$15,001 - $50,000"
    assert row["comments"] == "Filed electronically"
    assert row["source_extra"]["rate"] == "N/A"


def test_senate_printable_text_parser_supports_missing_ticker_without_failure():
    text = """
    Owner: Self
    Asset: Vanguard Total Stock Market ETF
    Asset Type: Exchange Traded Fund
    Transaction: Sale
    Amount: $1,001 - $15,000
    Comments: No ticker supplied by source
    """

    rows = parse_senate_ptr_text(text)
    row = rows[0].to_dict()

    assert row["owner"] == "self"
    assert row["ticker"] is None
    assert row["asset_type"] == "etf"
    assert row["transaction_type"] == "sale"
    assert row["validation_status"] == "valid_with_warnings"
    assert "missing_ticker" in row["warnings"]


def test_senate_owner_mapping():
    assert normalize_senate_owner("Self") == "self"
    assert normalize_senate_owner("Spouse") == "spouse"
    assert normalize_senate_owner("Dependent Child") == "dependent_child"
    assert normalize_senate_owner("Jointly Held") == "joint"
    assert normalize_senate_owner("Other") == "unknown"


def test_senate_asset_type_inference():
    assert infer_asset_type("Stock", "Apple Inc.", "AAPL") == "stock"
    assert infer_asset_type("ETF", "SPDR S&P 500 ETF", "SPY") == "etf"
    assert infer_asset_type("Option", "NVDA Call Option", None) == "option"
    assert infer_asset_type("Bond", "US Treasury Bond", None) == "bond"
    assert infer_asset_type(None, "Ambiguous private holding", None) == "unknown"


def test_senate_pdf_printable_fallback(tmp_path):
    pdf = tmp_path / "senate_ptr.pdf"
    pdf.write_text(
        "Owner: Joint\n"
        "Ticker: MSFT\n"
        "Asset: Microsoft Corporation\n"
        "Asset Type: Stock\n"
        "Transaction: Purchase\n"
        "Amount: $15,001 - $50,000\n"
        "Comments: Printable report\n",
        encoding="utf-8",
    )

    result = parse_senate_ptr_document(pdf)

    assert result["row_count"] == 1
    assert result["rows"][0]["owner"] == "joint"
    assert result["rows"][0]["ticker"] == "MSFT"


def test_senate_parser_golden_fixture_coverage_15_filings():
    fixtures = [
        ("Self", "AAPL", "Apple Inc.", "Stock"),
        ("Spouse", "SPY", "SPDR S&P 500 ETF", "ETF"),
        ("Dependent Child", None, "NVDA Call Option", "Option"),
        ("Joint", None, "US Treasury Bond", "Bond"),
        ("Self", None, "Ambiguous private holding", "Other"),
    ]
    parsed = []
    for idx in range(15):
        owner, ticker, asset, asset_type = fixtures[idx % len(fixtures)]
        text = (
            f"Owner: {owner}\n"
            f"{'Ticker: ' + ticker + chr(10) if ticker else ''}"
            f"Asset: {asset}\n"
            f"Asset Type: {asset_type}\n"
            "Transaction: Purchase\n"
            "$1,001 - $15,000\n"
            "Amount: $1,001 - $15,000\n"
            "Comments: Fixture\n"
        )
        parsed.append(parse_senate_ptr_text(text)[0].to_dict())

    assert len(parsed) == 15
    assert {row["asset_type"] for row in parsed} >= {"stock", "etf", "option", "bond", "unknown"}
    assert any(row["ticker"] is None and "missing_ticker" in row["warnings"] for row in parsed)
