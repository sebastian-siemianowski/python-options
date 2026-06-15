"""Summary endpoint tests for politician disclosure monitoring."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.backend.main import app
from web.backend.services.politicians_service import get_politicians_summary_response


def test_politicians_summary_route_registered_in_main_app():
    assert any(getattr(route, "path", "") == "/api/politicians/summary" for route in app.routes)


def test_summary_counts_health_and_new_disclosures(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    (prices_dir / "NVDA_1d.csv").write_text("Date,Close\n2026-05-29,100\n", encoding="utf-8")
    (prices_dir / "AAPL_1d.csv").write_text("Date,Close\n2026-05-29,200\n", encoding="utf-8")
    (tmp_path / "watchlist.json").write_text(json.dumps({"symbols": ["MSFT", "AAPL"]}), encoding="utf-8")
    (data_root / "source_health.json").write_text(
        json.dumps({"updated_at": "2026-05-29T10:00:00Z", "sources": {"house": {"status": "ok"}}}),
        encoding="utf-8",
    )
    _write_jsonl(
        data_root / "trades.jsonl",
        [
            _trade("NVDA", "house", "purchase", 1000, "2026-05-28", delay_days=10),
            _trade("TSLA", "senate", "sale", 2000, "2026-05-20", delay_days=10),
            _trade("MSFT", "house", "sale_partial", 5000, "2026-05-29", delay_days=46),
            _trade("AAPL", "senate", "received", 3000, "2026-05-23", flags=["late_disclosure"]),
        ],
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_summary_response(as_of_date="2026-05-29")

    assert response["status"] == "ok"
    assert response["total_trades"] == 4
    assert response["new_disclosures_7d"] == 3
    assert response["new_disclosures_last_7_days"] == 3
    assert response["new_tracked_asset_disclosures_7d"] == 2
    assert response["new_watchlist_disclosures_7d"] == 2
    assert response["tracked_asset_trades"] == 2
    assert response["watchlist_trades"] == 2
    assert response["late_filings"] == 2
    assert response["newest_disclosure_date"] == "2026-05-29"
    assert response["source_health"]["sources"]["house"]["status"] == "ok"
    assert response["summary"]["total_trades"] == 4


def test_summary_computes_buy_sell_midpoint_totals_by_chamber(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    _write_jsonl(
        data_root / "trades.jsonl",
        [
            _trade("NVDA", "house", "purchase", 1000, "2026-05-28"),
            _trade("MSFT", "house", "sale_partial", 5000, "2026-05-29"),
            _trade("TSLA", "senate", "sale", 2000, "2026-05-20"),
            _trade("AAPL", "senate", "received", 3000, "2026-05-23"),
        ],
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_summary_response(as_of_date="2026-05-29")

    assert response["by_chamber"]["house"]["trade_count"] == 2
    assert response["by_chamber"]["house"]["buy_count"] == 1
    assert response["by_chamber"]["house"]["sell_count"] == 1
    assert response["by_chamber"]["house"]["buy_amount_mid_usd"] == 1000
    assert response["by_chamber"]["house"]["sell_amount_mid_usd"] == 5000
    assert response["by_chamber"]["house"]["net_buy_amount_mid_usd"] == -4000
    assert response["by_chamber"]["senate"]["buy_count"] == 1
    assert response["by_chamber"]["senate"]["sell_count"] == 1
    assert response["by_chamber"]["senate"]["buy_amount_mid_usd"] == 3000
    assert response["by_chamber"]["senate"]["sell_amount_mid_usd"] == 2000


def test_summary_handles_missing_dataset_with_zero_counts(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_summary_response(as_of_date="2026-05-29")

    assert response["status"] == "ok"
    assert response["total_trades"] == 0
    assert response["new_disclosures_7d"] == 0
    assert response["tracked_asset_trades"] == 0
    assert response["watchlist_trades"] == 0
    assert response["late_filings"] == 0
    assert response["newest_disclosure_date"] is None
    assert response["by_chamber"] == {}


def test_summary_warm_process_handles_100k_rows_under_250ms(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    row = _trade("NVDA", "house", "purchase", 1000, "2026-05-28")
    with (data_root / "trades.jsonl").open("w", encoding="utf-8") as handle:
        line = json.dumps(row) + "\n"
        for _ in range(100_000):
            handle.write(line)
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")
    get_politicians_summary_response(as_of_date="2026-05-29")

    started = time.perf_counter()
    response = get_politicians_summary_response(as_of_date="2026-05-29")
    elapsed = time.perf_counter() - started

    assert response["total_trades"] == 100_000
    assert elapsed < 0.25


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _trade(
    ticker,
    chamber,
    transaction_type,
    amount_mid_usd,
    disclosure_date,
    *,
    delay_days=0,
    flags=None,
):
    return {
        "trade_id": f"{ticker}-{transaction_type}-{disclosure_date}",
        "ticker": ticker,
        "chamber": chamber,
        "transaction_type": transaction_type,
        "amount_mid_usd": amount_mid_usd,
        "disclosure_date": disclosure_date,
        "delay_days": delay_days,
        "flags": flags or [],
    }
