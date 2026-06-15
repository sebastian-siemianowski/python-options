"""Asset detail endpoint tests for politician disclosure monitoring."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.backend.services.politicians_service import get_politicians_asset_response


def test_asset_detail_returns_recent_trades_rollups_activity_and_timeline(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    _write_jsonl(
        data_root / "trades.jsonl",
        [
            _trade("NVDA", "Jane Doe", "purchase", 10_000, "2026-05-28", amount_min=1_001, amount_max=15_000),
            _trade("NVDA", "John Smith", "sale", 5_000, "2026-05-20", amount_min=1_001, amount_max=15_000),
            _trade("NVDA", "Old Filer", "purchase", 20_000, "2025-01-01"),
            _trade("AAPL", "Jane Doe", "purchase", 99_000, "2026-05-28"),
        ],
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_asset_response("nvda", window_days=30, as_of_date="2026-05-29")

    assert response["status"] == "ok"
    assert response["symbol"] == "NVDA"
    assert response["window_days"] == 30
    assert response["total"] == 2
    assert response["total_symbol_trades"] == 3
    assert [row["trade_id"] for row in response["recent_trades"]] == ["NVDA-2026-05-28-purchase", "NVDA-2026-05-20-sale"]
    assert response["trades"] == response["recent_trades"]
    assert response["unique_filers"] == ["Jane Doe", "John Smith"]
    assert response["unique_filer_count"] == 2
    assert response["buy_sell_imbalance"]["buy_count"] == 1
    assert response["buy_sell_imbalance"]["sell_count"] == 1
    assert response["buy_sell_imbalance"]["net_amount_mid_usd"] == 5_000
    assert response["amount_estimates"]["amount_mid_usd"] == 15_000
    assert response["amount_estimates"]["amount_min_usd"] == 2_002
    assert response["amount_estimates"]["amount_max_usd"] == 30_000
    assert response["activity"]["explanation"]["model_usage"] == "contextual_research_only_not_bma_or_kelly"
    assert response["disclosure_timeline"] == [
        {"date": "2026-05-20", "trade_count": 1, "buy_count": 0, "sell_count": 1, "net_amount_mid_usd": -5_000.0},
        {"date": "2026-05-28", "trade_count": 1, "buy_count": 1, "sell_count": 0, "net_amount_mid_usd": 10_000.0},
    ]
    assert all(row["official_source_url"] for row in response["recent_trades"])


def test_asset_detail_includes_known_limitations_for_ambiguous_ticker_resolution(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    row = _trade("NVDA", "Jane Doe", "purchase", 10_000, "2026-05-28")
    row["ticker_resolution_status"] = "ambiguous"
    _write_jsonl(data_root / "trades.jsonl", [row])
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_asset_response("NVDA", window_days=180, as_of_date="2026-05-29")

    assert response["known_limitations"][0]["code"] == "ticker_resolution_ambiguous"
    assert "official source" in response["known_limitations"][0]["message"]


def test_asset_detail_supports_dots_dashes_and_equals_in_symbols(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    _write_jsonl(
        data_root / "trades.jsonl",
        [
            _trade("BRK.B", "Jane Doe", "purchase", 10_000, "2026-05-28"),
            _trade("BRK-B", "Jane Doe", "purchase", 10_000, "2026-05-28"),
            _trade("EURUSD=X", "Jane Doe", "purchase", 10_000, "2026-05-28"),
        ],
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    assert get_politicians_asset_response("brk.b", as_of_date="2026-05-29")["total"] == 1
    assert get_politicians_asset_response("brk-b", as_of_date="2026-05-29")["total"] == 1
    assert get_politicians_asset_response("eurusd=x", as_of_date="2026-05-29")["total"] == 1


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _trade(
    ticker,
    filer_name,
    transaction_type,
    amount_mid_usd,
    disclosure_date,
    *,
    amount_min=None,
    amount_max=None,
):
    return {
        "trade_id": f"{ticker}-{disclosure_date}-{transaction_type}",
        "ticker": ticker,
        "filer_name": filer_name,
        "transaction_type": transaction_type,
        "amount_mid_usd": amount_mid_usd,
        "amount_min_usd": amount_min,
        "amount_max_usd": amount_max,
        "amount_mid_usd_is_estimate": True,
        "disclosure_date": disclosure_date,
        "document_url": f"https://example.test/{ticker}.pdf",
        "parser_confidence": 0.95,
    }
