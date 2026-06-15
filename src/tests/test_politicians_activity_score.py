"""Tests for contextual politician activity score."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.politician_context import compute_politician_activity_score
from web.backend.services.politicians_service import get_politicians_asset_response


def _trade(transaction_type, amount=10_000, disclosure_date="2026-05-20", filer="Jane Doe"):
    return {
        "transaction_type": transaction_type,
        "amount_mid_usd": amount,
        "disclosure_date": disclosure_date,
        "parser_confidence": 0.95,
        "filer_name": filer,
    }


def test_activity_score_positive_for_net_purchases_and_negative_for_net_sales():
    buys = compute_politician_activity_score([_trade("purchase"), _trade("purchase")], as_of_date="2026-05-29")
    sells = compute_politician_activity_score([_trade("sale"), _trade("sale_partial")], as_of_date="2026-05-29")

    assert buys["politician_activity_score"] > 0
    assert sells["politician_activity_score"] < 0


def test_activity_score_is_bounded_and_confidence_is_separate():
    score = compute_politician_activity_score([_trade("purchase", amount=50_000_000) for _ in range(5)], as_of_date="2026-05-29")

    assert -1 <= score["politician_activity_score"] <= 1
    assert 0 <= score["confidence"] <= 1
    assert score["explanation"]["bounded_range"] == "[-1, 1]"


def test_activity_score_uses_disclosure_date_recency_decay():
    recent = compute_politician_activity_score([_trade("purchase", disclosure_date="2026-05-28")], as_of_date="2026-05-29")
    old = compute_politician_activity_score([_trade("purchase", disclosure_date="2025-05-28")], as_of_date="2026-05-29")

    assert recent["explanation"]["components"]["average_recency_decay"] > old["explanation"]["components"]["average_recency_decay"]


def test_asset_response_returns_score_explanation_for_tooltips(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    politicians = data_dir / "politicians"
    politicians.mkdir(parents=True)
    (politicians / "trades.jsonl").write_text(
        json.dumps({
            "trade_id": "1",
            "ticker": "NVDA",
            "transaction_type": "purchase",
            "amount_mid_usd": 10_000,
            "disclosure_date": "2026-05-20",
            "parser_confidence": 0.95,
            "filer_name": "Jane Doe",
        }) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(politicians))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_asset_response("NVDA")

    assert "activity" in response
    assert response["activity"]["politician_activity_score"] > 0
    assert response["activity"]["explanation"]["model_usage"] == "contextual_research_only_not_bma_or_kelly"
