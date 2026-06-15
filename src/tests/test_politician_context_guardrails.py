"""Tests for politician context production-safety guardrails."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision.politician_context import (
    DEFAULT_EVENT_TIME_ANCHOR,
    POLITICIAN_ACTIVITY_CLASSIFICATION,
    PROHIBITED_CLASSIFICATION,
    assert_contextual_research_only,
    assert_not_model_integration_target,
    get_politician_context_policy,
    validate_no_leakage_research_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_politician_activity_is_contextual_research_not_model_feature():
    policy = get_politician_context_policy()

    assert POLITICIAN_ACTIVITY_CLASSIFICATION == "contextual_research"
    assert policy.classification == "contextual_research"
    assert policy.classification != PROHIBITED_CLASSIFICATION


def test_contextual_research_assertion_rejects_model_feature():
    assert_contextual_research_only("contextual_research")

    with pytest.raises(ValueError, match="contextual_research only"):
        assert_contextual_research_only("model_feature")


def test_prohibited_model_integration_targets_are_blocked():
    for target in (
        "bma_weights",
        "pit_calibration",
        "kelly_sizing",
        "high_conviction_labels",
        "latest_signals_output",
    ):
        with pytest.raises(ValueError, match="cannot feed"):
            assert_not_model_integration_target(target)


def test_no_leakage_research_gate_requires_disclosure_date():
    result = validate_no_leakage_research_gate(event_time_anchor=DEFAULT_EVENT_TIME_ANCHOR)

    assert result["approved_for_research"] is True
    assert result["approved_for_model_feature"] is False

    with pytest.raises(ValueError, match="disclosure_date"):
        validate_no_leakage_research_gate(event_time_anchor="transaction_date")


def test_no_leakage_research_gate_cannot_directly_approve_production_targets():
    with pytest.raises(ValueError, match="production integration"):
        validate_no_leakage_research_gate(
            event_time_anchor="disclosure_date",
            requested_targets=["bma_weights"],
        )


def test_latest_signals_production_path_does_not_import_politician_context():
    signals_path = REPO_ROOT / "src" / "decision" / "signals.py"
    signals_text = signals_path.read_text(encoding="utf-8").lower()

    assert "politician_context" not in signals_text
    assert "politician_activity_score" not in signals_text


def test_tuning_production_path_does_not_import_politician_context():
    tuning_dir = REPO_ROOT / "src" / "tuning"
    touched = []
    for path in tuning_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        if "politician_context" in text or "politician_activity_score" in text:
            touched.append(str(path.relative_to(REPO_ROOT)))

    assert touched == []
