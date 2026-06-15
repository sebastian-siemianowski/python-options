"""API contract tests for politician router registration and response states."""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.backend.main import app
from web.backend.routers.politicians import politicians_refresh_cache
from web.backend.services.politicians_service import (
    POLITICIANS_CACHE_SCHEMA_VERSION,
    POLITICIANS_RESPONSE_SCHEMA_VERSION,
    get_politicians_disabled_response,
    get_politicians_notice_response,
    get_politicians_trades_response,
)


def test_politicians_router_registered_in_main_app():
    assert any(getattr(route, "path", "") == "/api/politicians/notice" for route in app.routes)
    assert any(getattr(route, "path", "") == "/api/politicians/trades" for route in app.routes)
    assert any(getattr(route, "path", "") == "/api/politicians/refresh-cache" for route in app.routes)
    assert any(getattr(route, "path", "") == "/api/politicians/sync" for route in app.routes)


def test_notice_response_has_metadata_and_data_use_notice(monkeypatch):
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_notice_response()

    assert response["status"] == "notice_only"
    assert response["generated_at"]
    assert "data_age_seconds" in response
    assert response["schema_version"] == POLITICIANS_RESPONSE_SCHEMA_VERSION
    assert response["cache_schema_version"] == POLITICIANS_CACHE_SCHEMA_VERSION
    assert response["data_use_notice"]["summary"]


def test_missing_data_response_is_structured(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_trades_response()

    assert response["status"] == "missing_data"
    assert response["endpoint"] == "GET /trades"
    assert response["message"]
    assert response["generated_at"]
    assert "data_age_seconds" in response


def test_healthy_trades_response_has_metadata(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    (data_root / "trades.jsonl").write_text(json.dumps({"trade_id": "1", "ticker": "NVDA"}) + "\n", encoding="utf-8")
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    response = get_politicians_trades_response()

    assert response["status"] == "ok"
    assert response["total"] == 1
    assert response["generated_at"]
    assert response["data_age_seconds"] is not None
    assert response["schema_version"] == POLITICIANS_RESPONSE_SCHEMA_VERSION


def test_trade_cache_invalidates_when_jsonl_file_updates(tmp_path, monkeypatch):
    data_root = tmp_path / "politicians"
    data_root.mkdir()
    trades_path = data_root / "trades.jsonl"
    trades_path.write_text(json.dumps({"trade_id": "1", "ticker": "NVDA"}) + "\n", encoding="utf-8")
    monkeypatch.setenv("POLITICIANS_DATA_DIR", str(data_root))
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")

    first = get_politicians_trades_response()
    trades_path.write_text(
        json.dumps({"trade_id": "1", "ticker": "NVDA"}) + "\n" +
        json.dumps({"trade_id": "2", "ticker": "MSFT"}) + "\n",
        encoding="utf-8",
    )
    second = get_politicians_trades_response()

    assert first["total"] == 1
    assert second["total"] == 2
    assert {row["ticker"] for row in second["trades"]} == {"NVDA", "MSFT"}


def test_manual_refresh_endpoint_invalidates_politician_cache():
    response = asyncio.run(politicians_refresh_cache())

    assert response["status"] == "ok"
    assert response["cache"] == "politicians"
    assert isinstance(response["cleared_entries"], int)


def test_disabled_response_has_metadata_and_no_stacktrace(monkeypatch):
    monkeypatch.setenv("POLITICIANS_ENABLED", "0")

    response = get_politicians_disabled_response(endpoint="GET /trades")

    assert response["status"] == "disabled"
    assert response["message"]
    assert response["generated_at"]
    assert "traceback" not in json.dumps(response).lower()
