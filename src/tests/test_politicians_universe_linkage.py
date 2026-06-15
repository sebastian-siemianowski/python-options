"""Tests for linking politician trades to tracked and watchlist assets."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.backend.services.politicians_service import (
    enrich_asset_linkage,
    get_politicians_asset_response,
    get_politicians_trades_response,
)


def test_asset_linkage_marks_tracked_and_watchlist_symbols():
    rows = [{"ticker": "NVDA"}, {"ticker": "AAPL"}, {"ticker": "MSFT"}]

    enriched = enrich_asset_linkage(
        rows,
        tracked_symbols={"NVDA", "AAPL"},
        watchlist_symbols={"AAPL"},
    )

    assert enriched[0]["is_tracked_asset"] is True
    assert enriched[0]["is_watchlist_asset"] is False
    assert enriched[1]["is_tracked_asset"] is True
    assert enriched[1]["is_watchlist_asset"] is True
    assert enriched[2]["is_tracked_asset"] is False


def test_trades_api_can_filter_tracked_and_watchlist_assets(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    politicians = data_dir / "politicians"
    prices = data_dir / "prices"
    politicians.mkdir(parents=True)
    prices.mkdir()
    (prices / "NVDA_1d.csv").write_text("Date,Close\n2026-01-01,1\n", encoding="utf-8")
    (data_dir / "watchlist.json").write_text(json.dumps({"symbols": ["AAPL"]}), encoding="utf-8")
    (politicians / "trades.jsonl").write_text(
        json.dumps({"trade_id": "1", "ticker": "NVDA"}) + "\n" +
        json.dumps({"trade_id": "2", "ticker": "AAPL"}) + "\n" +
        json.dumps({"trade_id": "3", "ticker": "MSFT"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(politicians))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    all_rows = get_politicians_trades_response()["trades"]
    tracked = [row for row in all_rows if row["is_tracked_asset"]]
    watchlist = [row for row in all_rows if row["is_watchlist_asset"]]

    assert [row["trade_id"] for row in tracked] == ["1"]
    assert [row["trade_id"] for row in watchlist] == ["2"]


def test_asset_endpoint_returns_single_symbol_activity(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    politicians = data_dir / "politicians"
    politicians.mkdir(parents=True)
    (politicians / "trades.jsonl").write_text(
        json.dumps({"trade_id": "1", "ticker": "NVDA"}) + "\n" +
        json.dumps({"trade_id": "2", "ticker": "AAPL"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(politicians))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_asset_response("nvda")

    assert response["status"] == "ok"
    assert response["symbol"] == "NVDA"
    assert response["total"] == 1
    assert response["trades"][0]["trade_id"] == "1"


def test_missing_or_disabled_context_does_not_touch_signal_outputs():
    rows = enrich_asset_linkage([], tracked_symbols=set(), watchlist_symbols=set())

    assert rows == []
