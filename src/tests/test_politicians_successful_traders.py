"""Successful congressional trader ranking tests."""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.politician_trader_success import (
    build_successful_trader_leaderboard,
    evaluate_trade_success,
    selected_filer_keys,
)


def test_leaderboard_scores_public_post_disclosure_returns_and_sales(tmp_path):
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    _write_price(prices_dir, "NVDA", [("2026-01-02", 100), ("2026-01-09", 112), ("2026-04-02", 130)])
    _write_price(prices_dir, "TSLA", [("2026-01-02", 100), ("2026-01-09", 92), ("2026-04-02", 80)])
    _write_price(prices_dir, "BAD", [("2026-01-02", 100), ("2026-01-09", 91), ("2026-04-02", 70)])
    rows = [
        _trade("NVDA", "Hon. Nancy Pelosi", "purchase", amount_mid_usd=100_000),
        _trade("TSLA", "Sen. John Seller", "sale", amount_mid_usd=75_000),
        _trade("BAD", "Rep. Jane Buyer", "purchase", amount_mid_usd=50_000),
        _trade("", "Hon. Nancy Pelosi", "exchange", amount_mid_usd=10_000),
    ]

    leaderboard = build_successful_trader_leaderboard(
        rows,
        prices_dir=prices_dir,
        as_of_date="2026-04-15",
        limit=10,
    )

    assert leaderboard["scored_trade_count"] == 3
    assert leaderboard["unscored_trade_count"] == 1
    assert "nancy pelosi" in selected_filer_keys(leaderboard)
    pelosi = _entry(leaderboard, "nancy pelosi")
    seller = _entry(leaderboard, "john seller")
    assert pelosi["average_signed_return_pct"] == 30.0
    assert seller["average_signed_return_pct"] == 20.0
    assert pelosi["scored_trades"] == 1

    sale_eval = evaluate_trade_success(
        rows[1],
        prices_dir=prices_dir,
        price_cache={},
        horizon_days=90,
        min_holding_days=5,
        as_of=date(2026, 4, 15),
    )
    assert sale_eval is not None
    assert round(sale_eval.raw_return, 4) == -0.2
    assert round(sale_eval.signed_return, 4) == 0.2


def test_pelosi_is_included_in_top_ten_only_from_real_scored_records(tmp_path):
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    rows = []
    for idx in range(11):
        ticker = f"WIN{idx}"
        _write_price(prices_dir, ticker, [("2026-01-02", 100), ("2026-01-09", 115), ("2026-04-02", 150 + idx)])
        rows.append(_trade(ticker, f"Rep. High Scorer {idx}", "purchase", amount_mid_usd=100_000))
    _write_price(prices_dir, "AAPL", [("2026-01-02", 100), ("2026-01-09", 101), ("2026-04-02", 102)])
    rows.append(_trade("AAPL", "Hon. Nancy Pelosi", "purchase", amount_mid_usd=100_000))

    leaderboard = build_successful_trader_leaderboard(
        rows,
        prices_dir=prices_dir,
        as_of_date="2026-04-15",
        limit=10,
    )

    assert len(leaderboard["leaderboard"]) == 10
    pelosi = _entry(leaderboard, "nancy pelosi")
    assert pelosi["rank"] == 10
    assert pelosi["overall_rank"] > pelosi["rank"]
    assert pelosi["included_by_requested_profile"] is True

    without_pelosi = build_successful_trader_leaderboard(
        rows[:-1],
        prices_dir=prices_dir,
        as_of_date="2026-04-15",
        limit=10,
    )
    assert "nancy pelosi" not in selected_filer_keys(without_pelosi)


def _entry(leaderboard, filer_key):
    matches = [row for row in leaderboard["leaderboard"] if row["filer_key"] == filer_key]
    assert matches
    return matches[0]


def _trade(ticker, filer_name, transaction_type, *, amount_mid_usd):
    return {
        "ticker": ticker,
        "filer_name": filer_name,
        "chamber": "house",
        "party": "D",
        "state": "CA",
        "transaction_type": transaction_type,
        "disclosure_date": "2026-01-02",
        "amount_mid_usd": amount_mid_usd,
    }


def _write_price(prices_dir, ticker, points):
    body = "Date,Close\n" + "".join(f"{day},{close}\n" for day, close in points)
    (prices_dir / f"{ticker}.csv").write_text(body, encoding="utf-8")
