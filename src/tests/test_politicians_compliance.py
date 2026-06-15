"""Tests for politician disclosure data-use guardrails."""

import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.compliance import (
    DATA_USE_NOTICE,
    DATA_USE_POLICY,
    get_compliance_mode,
    get_data_use_notice,
    get_feature_availability,
    is_politicians_enabled,
)
from web.backend.services.politicians_service import (
    get_politicians_disabled_response,
    get_politicians_notice_response,
)


def test_data_use_policy_is_explicit_about_delayed_public_records():
    text = " ".join([DATA_USE_NOTICE, *DATA_USE_POLICY["bullets"]]).lower()

    assert "delayed" in text
    assert "public" in text
    assert "not real-time" in text
    assert "official source" in text


def test_data_use_policy_prohibits_misleading_or_restricted_use():
    text = " ".join([DATA_USE_NOTICE, *DATA_USE_POLICY["bullets"]]).lower()

    assert "credit rating" in text
    assert "unlawful" in text
    assert "solicitation" in text
    assert "insider-trading signal" in text
    assert "copy" in text


def test_data_use_notice_is_json_serializable_shape():
    notice = get_data_use_notice()

    assert notice["title"] == DATA_USE_POLICY["title"]
    assert notice["summary"] == DATA_USE_NOTICE
    assert len(notice["bullets"]) >= 5
    assert "official_sources" in notice
    assert "reviewed_at" in notice


def test_backend_notice_response_links_policy_under_data_use_notice(monkeypatch):
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")
    monkeypatch.setenv("POLITICIANS_COMPLIANCE_MODE", "research_only")

    response = get_politicians_notice_response()

    assert response["feature"] == "politicians"
    assert response["status"] == "notice_only"
    assert response["enabled"] is True
    assert response["compliance_mode"] == "research_only"
    assert response["data_use_notice"]["summary"] == DATA_USE_NOTICE


def test_enabled_flag_controls_backend_availability(monkeypatch):
    monkeypatch.setenv("POLITICIANS_ENABLED", "0")

    assert is_politicians_enabled() is False
    availability = get_feature_availability()
    response = get_politicians_notice_response()

    assert availability["enabled"] is False
    assert availability["disabled_reason"] == "POLITICIANS_ENABLED=0"
    assert response["status"] == "disabled"
    assert response["enabled"] is False
    assert response["data_use_notice"]["summary"] == DATA_USE_NOTICE


def test_enabled_mode_returns_notice_response(monkeypatch):
    monkeypatch.setenv("POLITICIANS_ENABLED", "1")
    monkeypatch.setenv("POLITICIANS_COMPLIANCE_MODE", "internal")

    response = get_politicians_notice_response()

    assert response["status"] == "notice_only"
    assert response["enabled"] is True
    assert response["compliance_mode"] == "internal"
    assert response["compliance_mode_valid"] is True


def test_unknown_compliance_mode_falls_back_to_research_only(monkeypatch):
    monkeypatch.setenv("POLITICIANS_COMPLIANCE_MODE", "launch_party")

    mode = get_compliance_mode()

    assert mode["compliance_mode"] == "research_only"
    assert mode["requested_compliance_mode"] == "launch_party"
    assert mode["compliance_mode_valid"] is False
    assert "warning" in mode


def test_disabled_response_is_structured(monkeypatch):
    monkeypatch.setenv("POLITICIANS_ENABLED", "0")

    response = get_politicians_disabled_response(endpoint="GET /trades")

    assert response["feature"] == "politicians"
    assert response["status"] == "disabled"
    assert response["endpoint"] == "GET /trades"
    assert response["message"]
    assert response["data_use_notice"]["summary"] == DATA_USE_NOTICE


def test_disabled_api_wildcard_returns_structured_response(monkeypatch):
    monkeypatch.setenv("POLITICIANS_ENABLED", "0")

    from starlette.requests import Request
    from web.backend.routers.politicians import politicians_fallback

    request = Request({"type": "http", "method": "GET", "path": "/api/politicians/trades", "headers": []})
    body = asyncio.run(politicians_fallback("trades", request))

    assert body["feature"] == "politicians"
    assert body["status"] == "disabled"
    assert body["enabled"] is False
    assert body["endpoint"] == "GET /trades"
