"""Trade feed endpoint tests for politician disclosure monitoring."""

import json
import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.backend.routers.politicians import _validate_trade_feed_filters
from web.backend.services.politicians_service import get_politicians_trades_response


def test_trade_feed_filters_by_asset_filer_source_dates_and_linkage(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    (prices_dir / "NVDA_1d.csv").write_text("Date,Close\n2026-05-29,100\n", encoding="utf-8")
    _write_jsonl(
        data_root / "trades.jsonl",
        [
            _trade(
                "NVDA",
                "Jane Doe",
                "house",
                "D",
                "CA",
                "purchase",
                "self",
                "2026-05-28",
                "https://example.test/nvda.pdf",
            ),
            _trade(
                "TSLA",
                "John Smith",
                "senate",
                "R",
                "TX",
                "sale",
                "spouse",
                "2026-05-20",
                "https://example.test/tsla.pdf",
            ),
        ],
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_trades_response(
        limit=10,
        offset=0,
        symbol="nvda",
        filer="jane",
        chamber="HOUSE",
        party="d",
        state="ca",
        transaction_type="purchase",
        owner="self",
        from_date="2026-05-01",
        to_date="2026-05-29",
        tracked_only=True,
    )

    assert response["status"] == "ok"
    assert response["total"] == 1
    assert response["page"] == {"limit": 10, "offset": 0, "returned": 1, "total": 1, "has_next": False}
    assert response["filter"]["symbol"] == "nvda"
    assert response["filter"]["from"] == "2026-05-01"
    assert response["trades"][0]["ticker"] == "NVDA"
    assert response["trades"][0]["is_tracked_asset"] is True
    assert response["trades"][0]["official_source_url"] == "https://example.test/nvda.pdf"


def test_trade_feed_filters_watchlist_and_flags(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    (tmp_path / "watchlist.json").write_text(json.dumps({"symbols": ["MSFT"]}), encoding="utf-8")
    _write_jsonl(
        data_root / "trades.jsonl",
        [
            _trade(
                "MSFT",
                "Jane Doe",
                "house",
                "D",
                "CA",
                "purchase",
                "self",
                "2026-05-28",
                "https://example.test/msft.pdf",
                delay_days=60,
            ),
            _trade(
                "NVDA",
                "Jane Doe",
                "house",
                "D",
                "CA",
                "purchase",
                "self",
                "2026-05-28",
                "https://example.test/nvda.pdf",
            ),
        ],
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_trades_response(flag="late_disclosure", watchlist_only=True)

    assert response["total"] == 1
    assert response["trades"][0]["ticker"] == "MSFT"
    assert response["trades"][0]["is_watchlist_asset"] is True
    assert "late_disclosure" in response["trades"][0]["flags"]


def test_trade_feed_stock_linked_filter_hides_bonds_notes_funds_and_unmapped_rows(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    _write_jsonl(
        data_root / "trades.jsonl",
        [
            _trade("AAPL", "Jane Doe", "house", "D", "CA", "purchase", "self", "2026-05-29", "https://example.test/aapl.pdf", asset_type="stock"),
            _trade("SPY", "Jane Doe", "house", "D", "CA", "purchase", "self", "2026-05-28", "https://example.test/spy.pdf", asset_type="etf"),
            _trade("NVDA", "Jane Doe", "house", "D", "CA", "purchase", "self", "2026-05-27", "https://example.test/nvda.pdf", asset_type="option"),
            _trade(None, "Jane Doe", "house", "D", "CA", "purchase", "self", "2026-05-26", "https://example.test/muni.pdf", asset_type="bond", asset_name_raw="PENNSYLVANIA ST TPK COMMN REV Rate/Coupon: 5.25% Matures: 2030-07-15"),
            _trade(None, "Jane Doe", "house", "D", "CA", "purchase", "self", "2026-05-25", "https://example.test/note.pdf", asset_type="unknown", asset_name_raw="GS Managed Structured Note Strategy MSCI EAFE Linked Note"),
            _trade(None, "Jane Doe", "house", "D", "CA", "sale", "self", "2026-05-24", "https://example.test/fund.pdf", asset_type="fund", asset_name_raw="Matthews International Mutual Fund"),
        ],
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_trades_response(limit=20, stock_linked_only=True)

    assert response["filter"]["stock_linked_only"] is True
    assert response["total"] == 3
    assert {row["ticker"] for row in response["trades"]} == {"AAPL", "SPY", "NVDA"}
    assert all(row["asset_type"] in {"stock", "etf", "option"} for row in response["trades"])


def test_trade_feed_purchase_and_sale_side_filters(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    rows = [
        _trade("BUY", "Jane Doe", "house", "D", "CA", "purchase", "self", "2026-05-29", "https://example.test/buy.pdf"),
        _trade("GIFT", "Jane Doe", "house", "D", "CA", "received", "self", "2026-05-28", "https://example.test/gift.pdf"),
        _trade("SELL", "Jane Doe", "house", "D", "CA", "sale", "self", "2026-05-27", "https://example.test/sell.pdf"),
        _trade("PART", "Jane Doe", "house", "D", "CA", "sale_partial", "self", "2026-05-26", "https://example.test/part.pdf"),
        _trade("EXCH", "Jane Doe", "house", "D", "CA", "exchange", "self", "2026-05-25", "https://example.test/exch.pdf"),
    ]
    _write_jsonl(data_root / "trades.jsonl", rows)
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    purchases = get_politicians_trades_response(limit=20, transaction_side="purchase")
    sales = get_politicians_trades_response(limit=20, transaction_side="sale")

    assert purchases["filter"]["transaction_side"] == "purchase"
    assert {row["transaction_type"] for row in purchases["trades"]} == {"purchase", "received"}
    assert sales["filter"]["transaction_side"] == "sale"
    assert {row["transaction_type"] for row in sales["trades"]} == {"sale", "sale_partial"}


def test_trade_feed_top_traders_filter_uses_success_leaderboard_and_includes_pelosi(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    rows = []
    for idx in range(10):
        ticker = f"TOP{idx}"
        _write_price(prices_dir, ticker, [("2026-01-02", 100), ("2026-01-09", 115), ("2026-04-02", 150 + idx)])
        rows.append(_trade(ticker, f"Rep. Top Trader {idx}", "house", "D", "CA", "purchase", "self", "2026-01-02", f"https://example.test/top-{idx}.pdf"))
    _write_price(prices_dir, "AAPL", [("2026-01-02", 100), ("2026-01-09", 101), ("2026-04-02", 102)])
    rows.append(_trade("AAPL", "Hon. Nancy Pelosi", "house", "D", "CA", "purchase", "self", "2026-01-02", "https://example.test/pelosi.pdf"))
    _write_price(prices_dir, "LOW", [("2026-01-02", 100), ("2026-01-09", 95), ("2026-04-02", 80)])
    rows.append(_trade("LOW", "Rep. Low Scorer", "house", "D", "CA", "purchase", "self", "2026-01-02", "https://example.test/low.pdf"))
    _write_jsonl(data_root / "trades.jsonl", rows)
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_trades_response(limit=20, top_traders_only=True)

    assert response["filter"]["top_traders_only"] is True
    assert response["total"] == 10
    assert response["successful_traders"]["limit"] == 10
    assert "Hon. Nancy Pelosi" in {row["filer_name"] for row in response["trades"]}
    assert "Rep. Low Scorer" not in {row["filer_name"] for row in response["trades"]}
    assert all(row["successful_trader_rank"] for row in response["trades"])
    pelosi = [row for row in response["trades"] if row["filer_name"] == "Hon. Nancy Pelosi"][0]
    assert pelosi["successful_trader_required_profile"] is True


def test_trade_feed_sorts_newest_first_and_returns_page_metadata(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    _write_jsonl(
        data_root / "trades.jsonl",
        [
            _trade("OLD", "Jane Doe", "house", "D", "CA", "purchase", "self", "2026-05-01", "https://example.test/old.pdf"),
            _trade("NEW", "Jane Doe", "house", "D", "CA", "purchase", "self", "2026-05-29", "https://example.test/new.pdf"),
            _trade("MID", "Jane Doe", "house", "D", "CA", "purchase", "self", "2026-05-15", "https://example.test/mid.pdf"),
        ],
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    first_page = get_politicians_trades_response(limit=1, offset=0)
    second_page = get_politicians_trades_response(limit=1, offset=1)

    assert first_page["trades"][0]["ticker"] == "NEW"
    assert first_page["page"] == {"limit": 1, "offset": 0, "returned": 1, "total": 3, "has_next": True}
    assert second_page["trades"][0]["ticker"] == "MID"
    assert second_page["page"] == {"limit": 1, "offset": 1, "returned": 1, "total": 3, "has_next": True}


def test_trade_feed_includes_official_source_url_key_for_every_row(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    _write_jsonl(
        data_root / "trades.jsonl",
        [
            _trade("DOC", "Jane Doe", "house", "D", "CA", "purchase", "self", "2026-05-29", "https://example.test/doc.pdf"),
            {
                **_trade("SRC", "Jane Doe", "house", "D", "CA", "purchase", "self", "2026-05-28", None),
                "source_url": "https://example.test/source.pdf",
            },
        ],
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_trades_response()

    assert all("official_source_url" in row for row in response["trades"])
    assert all("parser_confidence" in row for row in response["trades"])
    assert all(row["parser_confidence"] == 0.94 for row in response["trades"])
    assert {row["official_source_url"] for row in response["trades"]} == {
        "https://example.test/doc.pdf",
        "https://example.test/source.pdf",
    }


def test_trade_feed_invalid_filters_raise_http_422_with_helpful_detail():
    with pytest.raises(HTTPException) as excinfo:
        _validate_trade_feed_filters(
            chamber="parliament",
            transaction_type="teleport",
            transaction_side="mystery-side",
            owner="neighbor",
            flag="mystery",
            from_date="2026-06-01",
            to_date="2026-05-01",
        )

    assert excinfo.value.status_code == 422
    detail = excinfo.value.detail
    assert detail["message"] == "Invalid politician trade filters"
    fields = {error["field"] for error in detail["errors"]}
    assert fields == {"chamber", "transaction_type", "transaction_side", "owner", "flag", "from"}
    assert all(error["message"] for error in detail["errors"])


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_price(prices_dir, ticker, points):
    body = "Date,Close\n" + "".join(f"{day},{close}\n" for day, close in points)
    (prices_dir / f"{ticker}.csv").write_text(body, encoding="utf-8")


def _trade(
    ticker,
    filer_name,
    chamber,
    party,
    state,
    transaction_type,
    owner,
    disclosure_date,
    document_url,
    *,
    delay_days=0,
    asset_type="stock",
    asset_name_raw=None,
):
    return {
        "trade_id": f"{ticker}-{disclosure_date}",
        "ticker": ticker,
        "filer_name": filer_name,
        "chamber": chamber,
        "party": party,
        "state": state,
        "transaction_type": transaction_type,
        "owner": owner,
        "disclosure_date": disclosure_date,
        "document_url": document_url,
        "delay_days": delay_days,
        "asset_type": asset_type,
        "asset_name_raw": asset_name_raw or f"{ticker or 'Unmapped'} asset",
        "parser_confidence": 0.94,
    }
